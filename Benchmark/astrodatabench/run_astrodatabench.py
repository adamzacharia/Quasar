"""
AstroDataBench Data-Discovery runner (Harbor-free, Windows-friendly)
=====================================================================
Runs the AstroDataBench *data-discovery* suite (Harbor-format tasks under
``AstroDataBench-main/AstroDataBench-main/data-discovery/<category>/<task>/``)
against three kinds of candidate:

  * ``quasar``   -- the Quasar chat backend (POST /api/chat, SSE). Quasar cannot
                    write files, so the instruction's "write /app/output.json"
                    sentence is rewritten to "return the JSON in a ```json block"
                    and the harness writes the file. The exact prompt sent is saved.
  * ``shell:<provider>/<model>`` -- a model-neutral tool-loop agent (Terminus-2
                    style): the model gets the instruction verbatim (with
                    /app/output.json mapped to ./output.json), plus two tools --
                    ``python`` (runs code with an interpreter carrying the task
                    Dockerfile pins) and ``bash`` (Git Bash) -- in a fresh work dir.
                    Providers: ``openai/gpt-5.6-terra`` (Responses API) and
                    ``anthropic/claude-sonnet-5`` (Messages API).
  * ``oracle``    -- the task's reference ``solution/solve.sh`` (sanity check that
                    the verifier and the live archive agree, as the README advises).

Grading is the task's own ``tests/test_outputs.py`` run with pytest in an
environment that the host ``uvx`` resolves from the same ``--with`` pins the
task's ``tests/test.sh`` uses (``test.sh`` itself installs a specific uv inside
the container; unpinned entries such as HH212's ``astroquery`` float in both
places), and reward = passed/total exactly as ``test.sh`` computes it. This is a
LOCAL APPROXIMATION of the Harbor verifier, not a byte-identical environment.

Results are written in Harbor's job layout under
``data-discovery/jobs/<run-id>-<candidate>/<task>__<candidate>__trialN/result.json``
(+ ``verifier/report.json``, ``verifier/reward.txt``, ``output.json``,
``agent/``). The upstream ``benchmark.py report --suite data-discovery
--data-discovery-run-id <run-id>-<candidate>`` reads ONE candidate's job
unchanged; for the cross-candidate table use ``--report <run-id>`` here (the
upstream loader numbers trials per task only and would merge candidates).

This is a LOCAL, SYSTEM-LEVEL comparison, not a Harbor-equivalent run. Deviations
(recorded in every result.json under ``harness.disclosure``):
  - no container: agent commands run on this host (network is unrestricted;
    the container allow-lists archive hosts + PyPI);
  - Quasar returns JSON in text instead of writing a file;
  - agent wall-clock limit defaults to task.toml's 600 s (``--agent-timeout``)
    and is enforced as a hard stop (no API call or tool call starts after it);
  - with several ``--agents`` in one process the default ``--order interleaved``
    runs each (task, trial) for every candidate before the next one, so archive
    load and time-of-day effects are shared across candidates.

Usage (from the repo root, with the backend up for the quasar arm):
    python Benchmark/astrodatabench/run_astrodatabench.py --list
    python Benchmark/astrodatabench/run_astrodatabench.py --agents oracle
    python Benchmark/astrodatabench/run_astrodatabench.py --agents quasar \
        --quasar-model gpt-oss-120b --trials 3 --run-id 20260914
    python Benchmark/astrodatabench/run_astrodatabench.py \
        --agents shell:openai/gpt-5.6-terra shell:anthropic/claude-sonnet-5 --trials 3
    python Benchmark/astrodatabench/run_astrodatabench.py --verify-only <trial_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_BENCH_ROOT = REPO_ROOT / "AstroDataBench-main" / "AstroDataBench-main"
DEFAULT_AGENT_PYTHON = REPO_ROOT / "tmp" / "astrodatabench" / "agent-venv" / "Scripts" / "python.exe"
DEFAULT_UV_CACHE = REPO_ROOT / "tmp" / "astrodatabench" / "uv-cache"
GIT_BASH_CANDIDATES = [
    Path(r"C:\Program Files\Git\bin\bash.exe"),
    Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
]
HARNESS_VERSION = "0.1.0"
APP_OUTPUT = "/app/output.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def write_json(p: Path, obj: Any) -> None:
    write_text(p, json.dumps(obj, indent=2, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@dataclass
class Task:
    task_id: str
    category: str
    path: Path
    instruction: str
    name: str
    agent_timeout_s: float
    verifier_timeout_s: float
    verifier_pins: List[str]

    @property
    def test_file(self) -> Path:
        return self.path / "tests" / "test_outputs.py"

    @property
    def solve_sh(self) -> Path:
        return self.path / "solution" / "solve.sh"


_TOML_NAME_RE = re.compile(r'^\s*name\s*=\s*"([^"]*)"', re.M)
_TOML_CATEGORY_RE = re.compile(r'^\s*category\s*=\s*"([^"]*)"', re.M)
_WITH_PIN_RE = re.compile(r"--with\s+([A-Za-z0-9_.\-\[\]]+(?:==[A-Za-z0-9_.\-]+)?)")


def _toml_section_float(text: str, section: str, key: str, default: float) -> float:
    """Tiny TOML reader: value of ``key`` inside ``[section]`` (no tomllib dependency
    on the *agent* python; the harness python is 3.11+ so tomllib exists, but the
    task files are simple enough that a regex is more robust to comments)."""
    m = re.search(rf"^\[{re.escape(section)}\][^\[]*", text, re.M | re.S)
    if not m:
        return default
    k = re.search(rf"^\s*{re.escape(key)}\s*=\s*([0-9.]+)", m.group(0), re.M)
    return float(k.group(1)) if k else default


def load_tasks(bench_root: Path) -> List[Task]:
    suite = bench_root / "data-discovery"
    if not suite.exists():
        raise SystemExit(f"data-discovery suite not found under {bench_root}")
    tasks: List[Task] = []
    for cat_dir in sorted(p for p in suite.iterdir() if p.is_dir() and p.name != "jobs" and not p.name.startswith(".")):
        for toml_path in sorted(cat_dir.glob("*/task.toml")):
            tdir = toml_path.parent
            toml_text = read_text(toml_path)
            name_m = _TOML_NAME_RE.search(toml_text)
            cat_m = _TOML_CATEGORY_RE.search(toml_text)
            test_sh = tdir / "tests" / "test.sh"
            pins = _WITH_PIN_RE.findall(read_text(test_sh)) if test_sh.exists() else []
            tasks.append(Task(
                task_id=tdir.name,
                category=cat_m.group(1) if cat_m else cat_dir.name,
                path=tdir,
                instruction=read_text(tdir / "instruction.md").strip(),
                name=name_m.group(1) if name_m else f"STABLE/{tdir.name}",
                agent_timeout_s=_toml_section_float(toml_text, "agent", "timeout_sec", 600.0),
                verifier_timeout_s=_toml_section_float(toml_text, "verifier", "timeout_sec", 600.0),
                verifier_pins=pins,
            ))
    return tasks


# ---------------------------------------------------------------------------
# JSON extraction (for candidates that answer in text)
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n(.*?)```", re.S)


def extract_json(text: str) -> Tuple[Optional[Any], str]:
    """Return (value, source). The reply was asked for exactly one fenced ```json block at
    the end, so: if any fenced block exists, ONLY the last one counts (source
    'last_fence'; an invalid last fence is a parse failure, earlier example blocks are
    never graded -- CX-01). With no fence at all, fall back to the largest balanced JSON
    object/array in the text (source 'largest_balanced')."""
    fences = _FENCE_RE.findall(text)
    if fences:
        try:
            return json.loads(fences[-1].strip()), "last_fence"
        except json.JSONDecodeError:
            return None, "last_fence_invalid"
    dec = json.JSONDecoder()
    best: Tuple[int, Any] = (0, None)
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            val, end = dec.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(val, (dict, list)) and end > best[0]:
            best = (end, val)
    return best[1], ("largest_balanced" if best[1] is not None else "none")


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    output: Optional[Any]            # parsed JSON the agent produced (None = nothing usable)
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    prompt: str = ""
    n_tool_calls: int = 0
    elapsed_s: float = 0.0
    wrote_output_file: bool = False  # shell agents write output.json themselves


class Candidate:
    name = "base"
    model_name = ""

    def slug(self) -> str:
        return slug(self.model_name or self.name)

    def run(self, task: Task, workdir: Path, args) -> AgentResult:  # pragma: no cover
        raise NotImplementedError


# -- oracle -----------------------------------------------------------------

_HEREDOC_RE = re.compile(r"python3?\s+-\s+<<\s*'?EOF'?\s*\n(.*?)\nEOF", re.S)


class OracleCandidate(Candidate):
    name = "oracle"
    model_name = "oracle"

    def run(self, task: Task, workdir: Path, args) -> AgentResult:
        t0 = time.time()
        sh = read_text(task.solve_sh)
        m = _HEREDOC_RE.search(sh)
        if not m:
            return AgentResult(None, error="solve.sh has no python heredoc to extract", elapsed_s=time.time() - t0)
        out_path = (workdir / "output.json").resolve()
        code = m.group(1).replace(APP_OUTPUT, out_path.as_posix())
        script = workdir / "solve.py"
        write_text(script, code)
        proc = _run([str(args.agent_python), str(script)], cwd=workdir, timeout=task.agent_timeout_s)
        res = AgentResult(None, transcript=[{"role": "oracle", "stdout": proc["stdout"], "stderr": proc["stderr"], "rc": proc["rc"]}],
                          prompt="(reference solution)", elapsed_s=time.time() - t0)
        if out_path.exists():
            res.wrote_output_file = True
            try:
                res.output = json.loads(read_text(out_path))
            except json.JSONDecodeError as e:
                res.error = f"solution wrote invalid JSON: {e}"
        else:
            res.error = f"solution did not write output.json (rc={proc['rc']}): {proc['stderr'][-800:]}"
        return res


# -- quasar -----------------------------------------------------------------

def rewrite_for_chat(instruction: str) -> str:
    """Map the file-writing instruction onto a chat reply, touching nothing else."""
    text = re.sub(r"Write your (report|findings) to `/app/output\.json`",
                  r"Provide your \1 as a single fenced ```json code block at the end of your reply", instruction)
    text = text.replace("`/app/output.json`", "the fenced ```json block")
    text += ("\n\nOutput requirement: end your reply with exactly one fenced ```json code block containing "
             "only the JSON described above (no comments inside the JSON).")
    return text


_NON_TOOL_STATUS_PREFIXES = ("Connecting", "Calling", "Generating", "Composing", "Searching the web", "Web search")


def _is_tool_status_step(step: str) -> bool:
    """Quasar streams one `status` event per tool invocation ("Running X", "Querying ALMA ...", ...);
    the pipeline's own phases (connecting, calling the model, composing) are excluded."""
    return bool(step) and not step.startswith(_NON_TOOL_STATUS_PREFIXES)


def quasar_login(api_url: str, timeout: int = 30) -> Optional[str]:
    import httpx
    user = os.getenv("QUASAR_BENCH_USER", "1@1")
    pw = os.getenv("QUASAR_BENCH_PASS", "1")
    try:
        r = httpx.post(f"{api_url.rstrip('/')}/api/auth/login", json={"username": user, "password": pw}, timeout=timeout)
        if r.status_code == 200 and r.json().get("token"):
            return r.json()["token"]
        print(f"  [auth] login as {user} failed ({r.status_code}); continuing anonymously")
    except Exception as e:  # noqa: BLE001
        print(f"  [auth] login skipped ({e}); continuing anonymously")
    return None


class QuasarCandidate(Candidate):
    name = "quasar"

    def __init__(self, api_url: str, model: str, token: Optional[str], web_search: bool):
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.model_name = f"quasar/{model}"
        self.token = token
        self.web_search = web_search

    def run(self, task: Task, workdir: Path, args) -> AgentResult:
        import httpx
        prompt = rewrite_for_chat(task.instruction)
        payload = {"message": prompt, "model": self.model, "web_search": self.web_search}
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        events: List[dict] = []
        text_parts: List[str] = []
        usage: Dict[str, Any] = {}
        errors: List[str] = []
        t0 = time.time()
        deadline = t0 + task.agent_timeout_s if args.agent_timeout is None else t0 + args.agent_timeout
        timed_out = False
        import threading
        lock = threading.Lock()
        stop = threading.Event()
        state: Dict[str, Any] = {"tool_calls": 0, "omitted_calls": 0, "status_tool_steps": 0, "client": None, "exc": None}

        def record(line: str) -> None:
            """Apply one SSE line to the captured state -- under the lock and only while not stopped,
            so nothing can be mutated after the main thread has frozen the trial (CX-02)."""
            data = line[6:]
            try:
                ev = json.loads(data)
            except json.JSONDecodeError:
                ev = None
            with lock:
                if stop.is_set():
                    raise TimeoutError("agent deadline")
                if ev is None:
                    text_parts.append(data)
                    return
                events.append(ev)
                et = ev.get("type")
                if et == "token":
                    text_parts.append(ev.get("content", ""))
                elif et == "tool_trace":
                    # the backend may truncate the trace and report omitted_calls (CX-09)
                    state["omitted_calls"] = int(ev.get("omitted_calls") or 0)
                    state["tool_calls"] = len(ev.get("calls", []) or []) + state["omitted_calls"]
                elif et == "tool_call":
                    state["tool_calls"] += 1
                elif et == "status" and ev.get("state") == "running" and _is_tool_status_step(str(ev.get("step", ""))):
                    # streamed per-tool status steps: the only tool-count evidence when the turn is cut
                    # before the final tool_trace arrives (lower bound; see n_tool_calls_source)
                    state["status_tool_steps"] += 1
                elif et == "usage":
                    usage.update({"input_tokens": ev.get("inputTokens"), "output_tokens": ev.get("outputTokens"),
                                  "total_tokens": ev.get("totalTokens"), "cost_usd": ev.get("costUsd")})
                elif et == "error":
                    errors.append(str(ev.get("content", ""))[:500])

        def consume() -> None:
            # Runs in a DAEMON thread. The main thread enforces the HARD deadline: at expiry it
            # sets `stop` and snapshots the captured state under the lock (so the worker can no
            # longer change anything the trial will grade), closes the client to abort a blocked
            # read, and moves on WITHOUT waiting for this thread (CX-02).
            client = httpx.Client(timeout=httpx.Timeout(connect=30, read=max(1.0, deadline - time.time()), write=60, pool=30))
            state["client"] = client
            with client:
                with client.stream("POST", f"{self.api_url}/api/chat", json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if time.time() >= deadline or stop.is_set():
                            raise TimeoutError("agent deadline")
                        if not line.startswith("data: "):
                            continue
                        if line[6:] == "[DONE]":
                            break
                        record(line)

        def worker() -> None:
            try:
                consume()
            except Exception as e:  # noqa: BLE001
                state["exc"] = e

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        th.join(timeout=max(0.0, deadline - time.time()))
        with lock:
            stop.set()  # from here on the worker cannot touch text_parts/events/usage/errors/state counters
            text_parts = list(text_parts)
            events = list(events)
            usage = dict(usage)
            errors = list(errors)
            tool_calls, omitted_calls, status_steps = state["tool_calls"], state["omitted_calls"], state["status_tool_steps"]
        if th.is_alive():
            timed_out = True
            client = state.get("client")
            if client is not None:
                try:
                    client.close()  # aborts a blocked read; the daemon thread dies on its own, nobody waits for it
                except Exception:  # noqa: BLE001
                    pass
        elif state["exc"] is not None:
            exc = state["exc"]
            if isinstance(exc, (TimeoutError, httpx.ReadTimeout)):
                timed_out = True
            else:
                errors.append(f"{type(exc).__name__}: {exc}")
        elapsed = time.time() - t0
        response_text = "".join(text_parts)
        write_text(workdir / "agent" / "response.md", response_text)
        with (workdir / "agent" / "events.jsonl").open("w", encoding="utf-8") as fh:
            for ev in events:
                fh.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
        parsed, json_source = extract_json(response_text) if response_text else (None, "none")
        usage["json_source"] = json_source
        if omitted_calls:
            usage["tool_trace_omitted_calls"] = omitted_calls
        usage["status_tool_steps"] = status_steps
        if tool_calls == 0 and status_steps > 0:
            tool_calls = status_steps  # trace never arrived (timeout/cut stream): status steps are a lower bound
            usage["n_tool_calls_source"] = "status_steps_lower_bound"
        else:
            usage["n_tool_calls_source"] = "tool_trace" if tool_calls else "none"
        err = None
        if timed_out:
            err = f"agent timeout after {elapsed:.0f}s"
        elif errors:
            err = "; ".join(errors)[:1000]
        if parsed is None and not err:
            err = f"no gradable JSON in Quasar's reply (json_source={json_source})"
        return AgentResult(parsed, transcript=[{"role": "user", "content": prompt}, {"role": "assistant", "content": response_text}],
                           usage=usage, error=err, prompt=prompt, n_tool_calls=tool_calls, elapsed_s=elapsed)


# -- shell agent (model-neutral tool loop) ------------------------------------

SHELL_SYSTEM_PROMPT = """You are an autonomous agent completing an astronomy data-discovery task.

You are working in a fresh working directory that plays the role of `/app`: whenever the task says
`/app/output.json`, write `output.json` in the current working directory instead (the `python` and `bash`
tools already run there). Available tools:
  - `python`: run a Python program (the interpreter has astroquery, astropy, pyvo, astro-datalab, pandas,
    numpy installed; `pip install` works for anything else).
  - `bash`: run a bash command line (Git Bash on Windows; `python` on PATH is the same interpreter).
You have internet access to the archives named in the task and to PyPI.

Work step by step: query the archive, inspect real results, and only then write `output.json`.
Never fabricate identifiers, URLs, or numbers -- every value in `output.json` must come from a query you ran.
Finish by confirming `output.json` exists and is valid JSON, then reply with a brief summary.
"""

TOOLS_SPEC = [
    {"name": "python", "description": "Execute a Python program in the working directory and return stdout/stderr.",
     "schema": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source code to run."}},
                "required": ["code"]}},
    {"name": "bash", "description": "Run a bash command line in the working directory and return stdout/stderr.",
     "schema": {"type": "object", "properties": {"command": {"type": "string", "description": "Command line to run."}},
                "required": ["command"]}},
]


def _run(cmd: List[str], cwd: Path, timeout: float, env: Optional[dict] = None, input_text: Optional[str] = None) -> Dict[str, Any]:
    e = dict(os.environ)
    e["PYTHONIOENCODING"] = "utf-8"
    e["PYTHONUTF8"] = "1"
    if env:
        e.update(env)
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, timeout=timeout, env=e,
                           input=input_text, text=True, encoding="utf-8", errors="replace")
        return {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except subprocess.TimeoutExpired as ex:
        return {"rc": -1, "stdout": (ex.stdout or "") if isinstance(ex.stdout, str) else "",
                "stderr": f"[timeout after {timeout:.0f}s]"}


def _truncate(s: str, limit: int = 12000) -> str:
    if len(s) <= limit:
        return s
    head, tail = s[: limit // 2], s[-limit // 2:]
    return f"{head}\n...[{len(s) - limit} chars truncated]...\n{tail}"


class ShellToolbox:
    def __init__(self, workdir: Path, agent_python: Path, cmd_timeout: float, deadline: float):
        self.workdir = workdir
        self.agent_python = agent_python
        self.cmd_timeout = cmd_timeout
        self.deadline = deadline
        self.bash = next((p for p in GIT_BASH_CANDIDATES if p.exists()), None)
        scripts_dir = str(agent_python.parent)
        self.env = {"PATH": scripts_dir + os.pathsep + os.environ.get("PATH", ""), "VIRTUAL_ENV": str(agent_python.parent.parent)}

    def remaining(self) -> float:
        return self.deadline - time.time()

    def _timeout(self) -> float:
        # a tool call may never outlive the agent deadline (CX-02)
        return min(self.cmd_timeout, self.remaining())

    def call(self, name: str, arguments: Dict[str, Any]) -> str:
        if self.remaining() <= 0:
            return "ERROR: the agent's time budget is exhausted; no further commands will run."
        if name == "python":
            code = str(arguments.get("code", ""))
            script = self.workdir / ".agent_cell.py"
            write_text(script, code)
            r = _run([str(self.agent_python), str(script)], cwd=self.workdir, timeout=self._timeout(), env=self.env)
        elif name == "bash":
            if self.bash is None:
                return "ERROR: bash is not available on this host; use the python tool."
            r = _run([str(self.bash), "-lc", str(arguments.get("command", ""))], cwd=self.workdir, timeout=self._timeout(), env=self.env)
        else:
            return f"ERROR: unknown tool {name}"
        out = f"exit_code: {r['rc']}\n"
        if r["stdout"]:
            out += f"stdout:\n{_truncate(r['stdout'])}\n"
        if r["stderr"]:
            out += f"stderr:\n{_truncate(r['stderr'], 6000)}\n"
        return out


class ShellAgentCandidate(Candidate):
    name = "shell-agent"

    def __init__(self, provider_model: str, openai_effort: str, anthropic_thinking_budget: int, max_turns: int, cmd_timeout: float):
        if "/" not in provider_model:
            raise SystemExit(f"shell agent model must be provider/model, got {provider_model!r}")
        self.provider, self.model = provider_model.split("/", 1)
        if self.provider not in ("openai", "anthropic"):
            raise SystemExit(f"unsupported provider {self.provider!r} (openai|anthropic)")
        self.model_name = provider_model
        self.openai_effort = openai_effort
        self.anthropic_thinking_budget = anthropic_thinking_budget
        self.max_turns = max_turns
        self.cmd_timeout = cmd_timeout

    def _instruction(self, task: Task) -> str:
        return task.instruction.replace(APP_OUTPUT, "output.json (in the current working directory, which stands in for /app)")

    def run(self, task: Task, workdir: Path, args) -> AgentResult:
        prompt = self._instruction(task)
        t0 = time.time()
        deadline = t0 + (task.agent_timeout_s if args.agent_timeout is None else args.agent_timeout)
        toolbox = ShellToolbox(workdir, Path(args.agent_python), self.cmd_timeout, deadline)
        try:
            if self.provider == "openai":
                res = self._run_openai(prompt, toolbox, deadline)
            else:
                res = self._run_anthropic(prompt, toolbox, deadline)
        except Exception as e:  # noqa: BLE001
            res = AgentResult(None, error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}")
        res.prompt = prompt
        res.elapsed_s = time.time() - t0
        out_path = workdir / "output.json"
        if out_path.exists():
            res.wrote_output_file = True
            try:
                res.output = json.loads(read_text(out_path))
            except json.JSONDecodeError as e:
                res.error = (res.error + "; " if res.error else "") + f"agent wrote invalid JSON: {e}"
        elif not res.error:
            res.error = "agent finished without writing output.json"
        return res

    # OpenAI Responses API ---------------------------------------------------
    def _run_openai(self, prompt: str, toolbox: ShellToolbox, deadline: float) -> AgentResult:
        from openai import OpenAI
        client = OpenAI()
        tools = [{"type": "function", "name": t["name"], "description": t["description"], "parameters": t["schema"]} for t in TOOLS_SPEC]
        transcript: List[Dict[str, Any]] = [{"role": "system", "content": SHELL_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "calls": 0}
        inputs: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
        prev_id = None
        n_calls = 0
        error = None
        for turn in range(self.max_turns):
            remaining = deadline - time.time()
            if remaining <= 0:
                error = f"agent timeout ({turn} turns)"
                break
            kwargs: Dict[str, Any] = dict(model=self.model, input=inputs, tools=tools, instructions=SHELL_SYSTEM_PROMPT,
                                          reasoning={"effort": self.openai_effort}, store=True,
                                          timeout=remaining)  # CX-02: API call bounded by the remaining budget, no floor
            if prev_id:
                kwargs["previous_response_id"] = prev_id
            try:
                resp = client.responses.create(**kwargs)
            except Exception as e:  # noqa: BLE001
                if time.time() >= deadline:
                    error = f"agent timeout during API call ({turn} turns)"
                    break
                raise e
            prev_id = resp.id
            usage["calls"] += 1
            if resp.usage:
                usage["input_tokens"] += resp.usage.input_tokens or 0
                usage["output_tokens"] += resp.usage.output_tokens or 0
                det = getattr(resp.usage, "output_tokens_details", None)
                usage["reasoning_tokens"] += getattr(det, "reasoning_tokens", 0) or 0
            calls = [item for item in resp.output if getattr(item, "type", "") == "function_call"]
            text = resp.output_text or ""
            if text:
                transcript.append({"role": "assistant", "content": text})
            if time.time() >= deadline:  # CX-02: the deadline passed during the API call -> nothing more runs
                error = (f"agent timeout: final response arrived after the deadline ({turn + 1} turns)" if not calls
                         else f"agent timeout before executing tool calls ({turn + 1} turns)")
                break
            if not calls:
                break
            inputs = []
            for c in calls:
                try:
                    arguments = json.loads(c.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                n_calls += 1
                transcript.append({"role": "assistant", "tool_call": c.name, "arguments": arguments})
                result = toolbox.call(c.name, arguments)
                transcript.append({"role": "tool", "name": c.name, "output": result})
                inputs.append({"type": "function_call_output", "call_id": c.call_id, "output": result})
        else:
            error = f"max turns ({self.max_turns}) reached"
        return AgentResult(None, transcript=transcript, usage=usage, error=error, n_tool_calls=n_calls)

    # Anthropic Messages API ---------------------------------------------------
    def _run_anthropic(self, prompt: str, toolbox: ShellToolbox, deadline: float) -> AgentResult:
        import anthropic
        client = anthropic.Anthropic()
        tools = [{"name": t["name"], "description": t["description"], "input_schema": t["schema"]} for t in TOOLS_SPEC]
        messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
        transcript: List[Dict[str, Any]] = [{"role": "system", "content": SHELL_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        n_calls = 0
        error = None
        for turn in range(self.max_turns):
            remaining = deadline - time.time()
            if remaining <= 0:
                error = f"agent timeout ({turn} turns)"
                break
            kwargs: Dict[str, Any] = dict(model=self.model, max_tokens=16000, system=SHELL_SYSTEM_PROMPT, tools=tools, messages=messages,
                                          timeout=remaining)  # CX-02: no floor
            if self.anthropic_thinking_budget > 0:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.anthropic_thinking_budget}
            try:
                msg = client.messages.create(**kwargs)
            except Exception as e:  # noqa: BLE001
                if time.time() >= deadline:
                    error = f"agent timeout during API call ({turn} turns)"
                    break
                raise e
            usage["calls"] += 1
            usage["input_tokens"] += msg.usage.input_tokens
            usage["output_tokens"] += msg.usage.output_tokens
            content_blocks = [b.model_dump() for b in msg.content]
            messages.append({"role": "assistant", "content": content_blocks})
            for b in msg.content:
                if b.type == "text" and b.text:
                    transcript.append({"role": "assistant", "content": b.text})
            tool_uses = [b for b in msg.content if b.type == "tool_use"]
            wants_tools = msg.stop_reason == "tool_use" and bool(tool_uses)
            if time.time() >= deadline:  # CX-02: nothing more runs after the deadline
                error = (f"agent timeout before executing tool calls ({turn + 1} turns)" if wants_tools
                         else f"agent timeout: final response arrived after the deadline ({turn + 1} turns)")
                break
            if not wants_tools:
                break
            results = []
            for tu in tool_uses:
                n_calls += 1
                arguments = tu.input if isinstance(tu.input, dict) else {}
                transcript.append({"role": "assistant", "tool_call": tu.name, "arguments": arguments})
                result = toolbox.call(tu.name, arguments)
                transcript.append({"role": "tool", "name": tu.name, "output": result})
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": result})
            messages.append({"role": "user", "content": results})
        else:
            error = f"max turns ({self.max_turns}) reached"
        return AgentResult(None, transcript=transcript, usage=usage, error=error, n_tool_calls=n_calls)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

def run_verifier(task: Task, trial_dir: Path, args) -> Dict[str, Any]:
    """Run tests/test_outputs.py against <trial_dir>/output.json the way test.sh does."""
    vdir = trial_dir / "verifier"
    vdir.mkdir(parents=True, exist_ok=True)
    out_path = (trial_dir / "output.json").resolve()
    test_src = read_text(task.test_file)
    if APP_OUTPUT not in test_src:
        print(f"  [verify] WARNING: {task.test_file} does not reference {APP_OUTPUT}; running unmodified")
    write_text(vdir / "test_outputs.py", test_src.replace(f'"{APP_OUTPUT}"', json.dumps(out_path.as_posix())).replace(f"'{APP_OUTPUT}'", json.dumps(out_path.as_posix())))
    report = vdir / "report.json"
    if report.exists():
        report.unlink()
    pins = list(task.verifier_pins)
    if not any(p.startswith("pytest-json-report") for p in pins):
        pins.append("pytest-json-report")
    if args.verifier_mode == "uvx":
        cmd = ["uvx", "--python", args.verifier_python_version]
        for p in pins:
            cmd += ["--with", p]
        cmd += ["pytest", "test_outputs.py", "-p", "no:cacheprovider", "--json-report", f"--json-report-file={report}"]
        env = {"UV_CACHE_DIR": str(args.uv_cache)}
    else:
        cmd = [str(args.verifier_python), "-m", "pytest", "test_outputs.py", "-p", "no:cacheprovider", "--json-report", f"--json-report-file={report}"]
        env = None
    t0 = time.time()
    r = _run(cmd, cwd=vdir, timeout=task.verifier_timeout_s, env=env)
    write_text(vdir / "pytest_stdout.txt", r["stdout"] + ("\n[stderr]\n" + r["stderr"] if r["stderr"] else ""))
    summary: Dict[str, Any] = {"reward": 0.0, "passed": 0, "total": 0, "tests": [], "rc": r["rc"], "elapsed_s": round(time.time() - t0, 1),
                               "command": " ".join(cmd), "error": None}
    if report.exists():
        rep = json.loads(read_text(report))
        s = rep.get("summary", {})
        summary["passed"] = int(s.get("passed", 0))
        summary["total"] = int(s.get("total", 0))
        summary["reward"] = (summary["passed"] / summary["total"]) if summary["total"] else 0.0
        summary["tests"] = [{"id": t.get("nodeid", "").split("::")[-1], "outcome": t.get("outcome")} for t in rep.get("tests", [])]
    else:
        summary["error"] = f"verifier produced no report.json (rc={r['rc']}): {r['stderr'][-600:] or r['stdout'][-600:]}"
    write_text(vdir / "reward.txt", f"{summary['reward']}\n")
    return summary


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_candidates(args) -> List[Candidate]:
    cands: List[Candidate] = []
    quasar_token: Optional[str] = None
    for spec in args.agents:
        if spec == "oracle":
            cands.append(OracleCandidate())
        elif spec == "quasar":
            if quasar_token is None:
                quasar_token = args.quasar_token or os.getenv("QUASAR_API_TOKEN") or quasar_login(args.quasar_url)
            cands.append(QuasarCandidate(args.quasar_url, args.quasar_model, quasar_token, web_search=not args.quasar_no_web_search))
        elif spec.startswith("shell:"):
            cands.append(ShellAgentCandidate(spec[len("shell:"):], args.openai_effort, args.anthropic_thinking_budget, args.max_turns, args.cmd_timeout))
        else:
            raise SystemExit(f"unknown agent spec {spec!r} (oracle | quasar | shell:<provider>/<model>)")
    return cands


def _cmd_version(cmd: List[str]) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace")
        return (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr).strip() else None
    except Exception:  # noqa: BLE001
        return None


def _git_rev(path: Path) -> Optional[str]:
    try:
        r = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=30)
        rev = r.stdout.strip() or None
        if rev:
            dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=60).stdout.strip()
            rev += "-dirty" if dirty else ""
        return rev
    except Exception:  # noqa: BLE001
        return None


def build_disclosure(task: Task, cand: Candidate, args) -> Dict[str, Any]:
    """Everything a reader needs to know about how this trial differed from a Docker/Harbor run (CX-03, CX-04).
    This is a LOCAL, SYSTEM-LEVEL comparison, not a Harbor-equivalent run."""
    agent_py = Path(args.agent_python)
    d: Dict[str, Any] = {
        "comparison_type": "local system-level comparison (no Docker/Harbor); scores are a local approximation of the Harbor verifier",
        "quasar_backend_git_rev": _git_rev(REPO_ROOT),
        "verifier": {"mode": args.verifier_mode, "pins_from_test_sh": task.verifier_pins,
                     "uvx_version": _cmd_version(["uvx", "--version"]) if args.verifier_mode == "uvx" else None,
                     "python_version_requested": args.verifier_python_version if args.verifier_mode == "uvx" else _cmd_version([str(args.verifier_python), "--version"]),
                     "note": "test.sh installs its own pinned uv inside the container; here the host uvx resolves the same --with pins"},
        "network": "unrestricted host network (task.toml allow-lists archive hosts + PyPI inside the container)",
        "agent_timeout_s": task.agent_timeout_s if args.agent_timeout is None else args.agent_timeout,
        "graded_after_timeout": "yes: whatever output.json exists when the agent stops is graded, as Harbor does",
    }
    if isinstance(cand, QuasarCandidate):
        d["candidate"] = {"kind": "quasar", "scaffold": "Quasar production agent (its own system prompt, tool set, routing); web_search=" + str(cand.web_search),
                          "output_path": "instruction's /app/output.json sentence rewritten to a fenced ```json block; harness writes the file",
                          "json_extraction": "last fenced block only; largest balanced JSON value if no fence"}
    elif isinstance(cand, ShellAgentCandidate):
        d["candidate"] = {"kind": "shell-agent", "scaffold": "harness tool loop (Terminus-2 style), identical for every provider",
                          "system_prompt": SHELL_SYSTEM_PROMPT, "tools": [t["name"] for t in TOOLS_SPEC],
                          "output_path": "/app/output.json -> ./output.json in a fresh work dir",
                          "agent_python": str(agent_py), "agent_python_version": _cmd_version([str(agent_py), "--version"]),
                          "agent_packages": _cmd_version([str(agent_py), "-c", "import astroquery,astropy,pyvo,numpy,pandas;print('astroquery',astroquery.__version__,'astropy',astropy.__version__,'pyvo',pyvo.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__)"]),
                          "max_turns": args.max_turns, "cmd_timeout_s": args.cmd_timeout,
                          "openai_reasoning_effort": getattr(cand, "openai_effort", None), "anthropic_thinking_budget": getattr(cand, "anthropic_thinking_budget", None)}
    else:
        d["candidate"] = {"kind": "oracle", "scaffold": "task solution/solve.sh python heredoc run with the agent interpreter"}
    return d


def run_trial(task: Task, cand: Candidate, trial: int, job_dir: Path, args) -> Dict[str, Any]:
    trial_dir = job_dir / f"{task.task_id}__{cand.slug()}__trial{trial}"
    if (trial_dir / "result.json").exists() and not args.overwrite:
        # CX-07: never silently destroy graded evidence
        raise SystemExit(f"{trial_dir} already holds a result.json; pass --overwrite or a new --run-id")
    if trial_dir.exists():
        shutil.rmtree(trial_dir)  # an interrupted (result-less) trial, or an explicit --overwrite
    (trial_dir / "agent").mkdir(parents=True)
    started = now_iso()
    print(f"\n=== {task.task_id} | {cand.model_name} | trial {trial} ===", flush=True)
    result: Optional[AgentResult] = None
    crash: Optional[str] = None
    try:
        result = cand.run(task, trial_dir, args)
    except KeyboardInterrupt:
        raise
    except Exception:  # noqa: BLE001  (CX-08: a crash still yields a zero-reward row)
        crash = traceback.format_exc()
        print(f"  !! candidate crashed:\n{crash}", flush=True)
        result = AgentResult(None, error=f"harness/candidate crash: {crash.strip().splitlines()[-1]}")
    if result.output is not None and not result.wrote_output_file:
        write_json(trial_dir / "output.json", result.output)
    write_text(trial_dir / "agent" / "prompt.md", result.prompt)
    write_json(trial_dir / "agent" / "transcript.json", result.transcript)
    write_json(trial_dir / "agent" / "usage.json", result.usage)
    if crash:
        write_text(trial_dir / "agent" / "crash.txt", crash)
    status = "ok" if result.error is None else result.error.splitlines()[0][:200]
    print(f"  agent: {result.elapsed_s:.0f}s, {result.n_tool_calls} tool call(s), output.json={'yes' if (trial_dir / 'output.json').exists() else 'NO'}, status={status}", flush=True)

    verifier: Dict[str, Any] = {"reward": None}
    if not args.skip_verify:
        try:
            verifier = run_verifier(task, trial_dir, args)
        except Exception:  # noqa: BLE001
            verifier = {"reward": 0.0, "passed": 0, "total": 0, "tests": [], "elapsed_s": 0, "error": f"verifier crashed: {traceback.format_exc().strip().splitlines()[-1]}"}
        print(f"  verifier: reward={verifier['reward']:.3f} ({verifier['passed']}/{verifier['total']} tests, {verifier['elapsed_s']}s)"
              + (f" ERROR: {verifier['error']}" if verifier.get("error") else ""), flush=True)
    finished = now_iso()

    record = {
        "harness": {"name": "run_astrodatabench.py", "version": HARNESS_VERSION, "mode": "local-host (no Docker/Harbor)",
                    "deviations": ["no container isolation or host allow-list",
                                   "Quasar answers in text; harness writes output.json" if isinstance(cand, QuasarCandidate) else "agent writes output.json itself",
                                   f"verifier env: {args.verifier_mode} with pins {task.verifier_pins}"],
                    "disclosure": build_disclosure(task, cand, args),
                    "overwrote_previous_trial": bool(args.overwrite)},
        "task_name": task.name,
        # CX-06: forward-slash, repo-relative path so Path(...).name works on POSIX too
        "task_id": {"path": _portable_task_path(task, args)},
        "candidate": cand.model_name,
        "trial_name": trial_dir.name,
        "trial": trial,
        "agent_info": {"name": cand.name, "version": HARNESS_VERSION},
        "config": {"agent": {"name": cand.name, "model_name": cand.model_name,
                             "openai_effort": getattr(cand, "openai_effort", None),
                             "anthropic_thinking_budget": getattr(cand, "anthropic_thinking_budget", None),
                             "quasar_url": getattr(cand, "api_url", None), "web_search": getattr(cand, "web_search", None)}},
        "agent_result": {"elapsed_s": round(result.elapsed_s, 1), "n_tool_calls": result.n_tool_calls, "usage": result.usage,
                         "error": result.error, "produced_output": (trial_dir / "output.json").exists()},
        "verifier_result": {"rewards": {"reward": verifier.get("reward")}, "passed": verifier.get("passed"), "total": verifier.get("total"),
                            "tests": verifier.get("tests"), "error": verifier.get("error")},
        "exception_info": result.error,
        "started_at": started,
        "finished_at": finished,
    }
    write_json(trial_dir / "result.json", record)
    return record


def _portable_task_path(task: Task, args) -> str:
    try:
        return task.path.resolve().relative_to(Path(args.bench_root).resolve()).as_posix()
    except ValueError:
        return task.path.as_posix()


def write_job_summary(job_dir: Path, records: List[Dict[str, Any]]) -> None:
    lines = [f"# {job_dir.name}", "", "| Task | Candidate | Trial | Reward | Tests | Agent time | Tool calls | Status |", "|---|---|---|---|---|---|---|---|"]
    for r in records:
        rw = r["verifier_result"]["rewards"]["reward"]
        lines.append(f"| {Path(r['task_id']['path']).name} | {r['config']['agent']['model_name']} | {r['trial_name'].rsplit('trial', 1)[-1]} | "
                     f"{'-' if rw is None else f'{rw:.3f}'} | {r['verifier_result'].get('passed')}/{r['verifier_result'].get('total')} | "
                     f"{r['agent_result']['elapsed_s']}s | {r['agent_result']['n_tool_calls']} | {(r['exception_info'] or 'ok').splitlines()[0][:80]} |")
    rewards = [r["verifier_result"]["rewards"]["reward"] for r in records if r["verifier_result"]["rewards"]["reward"] is not None]
    if rewards:
        lines += ["", f"**mean reward: {sum(rewards) / len(rewards):.3f}** over {len(rewards)} trial(s)"]
    write_text(job_dir / "summary.md", "\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    ap.add_argument("--tasks", nargs="*", default=None, help="task ids (default: all data-discovery tasks)")
    ap.add_argument("--agents", nargs="+", default=["oracle"], help="oracle | quasar | shell:<provider>/<model> (repeatable)")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--run-id", default=None, help="base run id; jobs are named <run-id>-<candidate> (default: UTC timestamp)")
    ap.add_argument("--list", action="store_true", help="list tasks and exit")
    ap.add_argument("--dry-run", action="store_true", help="show the prompts that would be sent and exit")
    ap.add_argument("--overwrite", action="store_true", help="allow re-running a trial that already has a result.json (default: refuse)")
    ap.add_argument("--order", default="interleaved", choices=["interleaved", "candidate-major"],
                    help="with several --agents: interleaved (default) runs each (task, trial) for every candidate before moving on; "
                         "candidate-major finishes one candidate first (CX-10)")
    ap.add_argument("--report", metavar="RUN_ID", default=None,
                    help="compile every jobs/<RUN_ID>-*/ job into one per-(task, candidate) table and exit (CX-05: the upstream "
                         "benchmark.py report numbers trials per task only, so point it at ONE job, e.g. --data-discovery-run-id <RUN_ID>-<candidate>)")
    # quasar
    ap.add_argument("--quasar-url", default=os.getenv("QUASAR_API_URL", "http://localhost:8000"))
    ap.add_argument("--quasar-model", default="gpt-oss-120b", help="Quasar chat model id (Quasar's production default is gpt-oss-120b)")
    ap.add_argument("--quasar-token", default=None)
    ap.add_argument("--quasar-no-web-search", action="store_true")
    # shell agent
    ap.add_argument("--agent-python", type=Path, default=DEFAULT_AGENT_PYTHON, help="interpreter with the task Dockerfile pins")
    ap.add_argument("--openai-effort", default="medium", choices=["low", "medium", "high", "xhigh"])
    ap.add_argument("--anthropic-thinking-budget", type=int, default=0, help="0 = extended thinking off (API default)")
    ap.add_argument("--max-turns", type=int, default=200, help="safety cap on model turns; the 600 s deadline is the intended limit (run 20260914 used 40)")
    ap.add_argument("--cmd-timeout", type=float, default=300.0, help="per tool call, seconds")
    ap.add_argument("--agent-timeout", type=float, default=None, help="override task.toml [agent] timeout_sec")
    # verifier
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--verify-only", type=Path, default=None, help="re-run the verifier on an existing trial dir and exit")
    ap.add_argument("--verifier-mode", default="uvx", choices=["uvx", "python"])
    ap.add_argument("--verifier-python", type=Path, default=Path(sys.executable))
    ap.add_argument("--verifier-python-version", default="3.13")
    ap.add_argument("--uv-cache", type=Path, default=DEFAULT_UV_CACHE)
    args = ap.parse_args()

    tasks = load_tasks(args.bench_root)
    if args.tasks:
        unknown = [t for t in args.tasks if t not in {x.task_id for x in tasks}]
        if unknown:
            raise SystemExit(f"unknown task(s) {unknown}; available: {[t.task_id for t in tasks]}")
        tasks = [t for t in tasks if t.task_id in set(args.tasks)]
    if args.list:
        for t in tasks:
            print(f"{t.task_id:28s} {t.category:16s} agent_timeout={t.agent_timeout_s:.0f}s pins={t.verifier_pins}")
        return 0

    if args.report:
        return write_run_report(args.bench_root, args.report, tasks)

    if args.verify_only:
        tdir = args.verify_only.resolve()
        task_id = tdir.name.split("__", 1)[0]
        task = next(t for t in tasks if t.task_id == task_id)
        v = run_verifier(task, tdir, args)
        print(json.dumps(v, indent=2))
        rp = tdir / "result.json"
        if rp.exists():
            rec = json.loads(read_text(rp))
            rec["verifier_result"] = {"rewards": {"reward": v["reward"]}, "passed": v["passed"], "total": v["total"], "tests": v["tests"], "error": v["error"]}
            write_json(rp, rec)
        return 0

    if any(a.startswith("shell:") or a == "oracle" for a in args.agents) and not Path(args.agent_python).exists():
        raise SystemExit(f"agent python not found: {args.agent_python}\n"
                         f"create it with: uv venv --python 3.13 {DEFAULT_AGENT_PYTHON.parent.parent} && uv pip install --python {DEFAULT_AGENT_PYTHON} "
                         f"astroquery==0.4.11 astropy==8.0.0 pyvo==1.9.1 astro-datalab==2.22.1 pandas==3.0.3 numpy==2.5.0")
    if args.verifier_mode == "uvx" and shutil.which("uvx") is None:
        raise SystemExit("uvx not on PATH; install uv or use --verifier-mode python")

    cands = build_candidates(args)
    if args.dry_run:
        for t in tasks:
            for c in cands:
                p = rewrite_for_chat(t.instruction) if isinstance(c, QuasarCandidate) else (c._instruction(t) if isinstance(c, ShellAgentCandidate) else "(reference solution)")
                print(f"\n##### {t.task_id} -> {c.model_name}\n{p}")
        return 0

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    jobs_root = args.bench_root / "data-discovery" / "jobs"
    job_dirs = {cand.slug(): jobs_root / f"{run_id}-{cand.slug()}" for cand in cands}
    for jd in job_dirs.values():
        jd.mkdir(parents=True, exist_ok=True)
    records_by_cand: Dict[str, List[Dict[str, Any]]] = {cand.slug(): [] for cand in cands}
    # CX-10: default interleaved order -- every candidate sees the same (task, trial) at about the same
    # time, so archive load / time-of-day effects are shared instead of confounded with candidate identity.
    if args.order == "interleaved":
        schedule = [(task, trial, cand) for task in tasks for trial in range(1, args.trials + 1) for cand in cands]
    else:
        schedule = [(task, trial, cand) for cand in cands for task in tasks for trial in range(1, args.trials + 1)]
    for task, trial, cand in schedule:
        records_by_cand[cand.slug()].append(run_trial(task, cand, trial, job_dirs[cand.slug()], args))
    all_records: List[Dict[str, Any]] = []
    for cand in cands:
        write_job_summary(job_dirs[cand.slug()], records_by_cand[cand.slug()])
        all_records += records_by_cand[cand.slug()]
    print(f"\nJobs written under {jobs_root} (run id {run_id}).")
    print(f"Combined per-candidate table: python {Path(__file__).name} --report {run_id}")
    print(f"Upstream report (ONE job at a time): python {args.bench_root / 'benchmark.py'} report --suite data-discovery --data-discovery-run-id {run_id}-<candidate>")
    return 0


def write_run_report(bench_root: Path, run_id: str, tasks: List[Task]) -> int:
    """Combine every jobs/<run_id>-*/ job into one table keyed by (task, candidate): per-trial rewards,
    mean, tool calls, agent time, failure count. Written to jobs/<run_id>-REPORT.md/.csv."""
    import csv
    import statistics
    jobs_root = bench_root / "data-discovery" / "jobs"
    job_dirs = sorted(p for p in jobs_root.iterdir() if p.is_dir() and p.name.startswith(f"{run_id}-") and not p.name.endswith("-REPORT"))
    rows: List[Dict[str, Any]] = []
    for jd in job_dirs:
        for rp in sorted(jd.glob("*/result.json")):
            r = json.loads(read_text(rp))
            rows.append({"task": Path(r["task_id"]["path"]).name, "candidate": r.get("candidate") or r["config"]["agent"]["model_name"],
                         "trial": r.get("trial") or int(r["trial_name"].rsplit("trial", 1)[-1]),
                         "reward": r["verifier_result"]["rewards"]["reward"], "passed": r["verifier_result"].get("passed"),
                         "total": r["verifier_result"].get("total"), "tool_calls": r["agent_result"]["n_tool_calls"],
                         "agent_s": r["agent_result"]["elapsed_s"], "error": (r.get("exception_info") or "").splitlines()[0][:100] if r.get("exception_info") else "",
                         "started_at": r["started_at"], "job": jd.name})
    if not rows:
        print(f"no results under {jobs_root}/{run_id}-*", file=sys.stderr)
        return 1
    task_order = [t.task_id for t in tasks]
    cands = sorted({r["candidate"] for r in rows})
    lines = [f"# AstroDataBench data-discovery — run {run_id}", "",
             "Reward = fraction of the task's pytest checks passed (1.0 = all). Local system-level comparison, no Docker/Harbor; see each result.json `harness.disclosure`.", "",
             "## Mean reward per task × candidate", "", "| Task | " + " | ".join(cands) + " |", "|---|" + "---|" * len(cands)]
    for task in task_order + sorted({r["task"] for r in rows} - set(task_order)):
        cells = []
        for c in cands:
            rs = [r["reward"] for r in rows if r["task"] == task and r["candidate"] == c and r["reward"] is not None]
            cells.append(f"{statistics.fmean(rs):.3f} (n={len(rs)})" if rs else "—")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    cells = []
    for c in cands:
        rs = [r["reward"] for r in rows if r["candidate"] == c and r["reward"] is not None]
        cells.append(f"**{statistics.fmean(rs):.3f}** (n={len(rs)})" if rs else "—")
    lines.append("| **All tasks** | " + " | ".join(cells) + " |")
    lines += ["", "## Every trial", "", "| Task | Candidate | Trial | Reward | Tests | Tool calls | Agent time | Status |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (task_order.index(r["task"]) if r["task"] in task_order else 99, r["candidate"], r["trial"])):
        lines.append(f"| {r['task']} | {r['candidate']} | {r['trial']} | {'-' if r['reward'] is None else f'{r[chr(114)+chr(101)+chr(119)+chr(97)+chr(114)+chr(100)]:.3f}'} | "
                     f"{r['passed']}/{r['total']} | {r['tool_calls']} | {r['agent_s']}s | {r['error'] or 'ok'} |")
    out_md = jobs_root / f"{run_id}-REPORT.md"
    out_csv = jobs_root / f"{run_id}-REPORT.csv"
    write_text(out_md, "\n".join(lines) + "\n")
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n".join(lines))
    print(f"\n(wrote {out_md} and {out_csv})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
