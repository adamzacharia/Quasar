"""Run-once check for archive code in a final answer (ALMAbench round 5, 2026-10-16).

When an answer contains a Python script that queries an astronomy archive (pyvo, astroquery, ALminer or a TAP
endpoint), the script is run ONCE, unchanged, in a child interpreter before the answer is shown:

* ``python -I`` in a fresh temporary directory, with a minimal environment (no API keys, tokens or other secrets
  from the server), HOME and the astropy/XDG cache directories pointed into the temporary directory, and a
  wall-clock timeout;
* an audit hook (PEP 578) installed before the script starts vetoes process creation, native library loading,
  object-graph introspection, writes outside the temporary directory, reads outside the temporary directory and the
  Python installation, and network access to any host that is not on the archive allowlist
  (``QUASAR_CODE_CHECK_HOSTS`` overrides it);
* data downloads in astroquery's ALMA client and ALminer are replaced by stubs, so a check never pulls data files.

If the script fails (an exception, a non-zero exit or a timeout), the SAME model that wrote the answer gets one
repair call with the error, and the repaired script is run once more. The answer then keeps the version that ran
(or the original, if neither did) and ends with a short, honest note on what happened, with the first lines of the
output in a text block. Nothing in this module knows about any particular question: it only runs code the model
wrote and reports the result.

``QUASAR_CODE_CHECK=1`` turns the check on. It is OFF by default (2026-10-17): it runs model-written code on the
server, and the sandbox has only been tested on Windows (the ALMAbench round-5 runs had it on).
``QUASAR_CODE_CHECK_TIMEOUT`` (default 90 s) is the limit per run and ``QUASAR_CODE_CHECK_BUDGET`` (default 240 s)
the limit for the whole check, run, repair and re-run.
"""
from __future__ import annotations

import ast
import datetime as _dt
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Hosts (and their subdomains) a checked script may reach: the ALMA archive and its mirrors, and the public
# astronomy services such scripts commonly use for name resolution and cross matching.
DEFAULT_ALLOWED_HOSTS: Tuple[str, ...] = (
    "almascience.org", "almascience.eso.org", "almascience.nrao.edu", "almascience.nao.ac.jp", "alma.cl",
    "eso.org", "nrao.edu", "nao.ac.jp",
    "cds.unistra.fr", "u-strasbg.fr",  # Sesame, SIMBAD, VizieR
    "ipac.caltech.edu",  # NED, IRSA
    "stsci.edu",  # MAST
    "esac.esa.int",  # ESA archives
    "noirlab.edu",  # Astro Data Lab
    "heasarc.gsfc.nasa.gov",
    "g-vo.org", "ivoa.net",  # registry
)

ARCHIVE_MARKERS = re.compile(
    r"\bimport\s+(?:pyvo|alminer)\b|\bfrom\s+(?:pyvo|alminer|astroquery)\b|\bimport\s+astroquery\b"
    r"|\bTAPService\s*\(|/tap\b|\bivoa\.obscore\b",
    re.IGNORECASE,
)
# Code that cannot run unattended here: CASA, credentials or prompts, notebook magics, generic file downloads.
SKIP_MARKERS = re.compile(
    r"\b(?:casatasks|casatools|casashell|casaviewer)\b|^\s*from\s+casa|\bimport\s+casa"
    r"|\.login\s*\(|\bgetpass\b|\binput\s*\("
    r"|^\s*[!%]"
    r"|\burlretrieve\s*\(|\bgetdataset\s*\(|\bdownload_file\s*\(|\bwget\b|\bcurl\b"
    r"|\bYOUR_[A-Z_]+\b|<(?i:your|path|insert|replace|enter)[^>\n]*>",
    re.MULTILINE,
)
_FENCE_RE = re.compile(r"(^|\n)(?P<fence>```+|~~~+)[ \t]*(?P<lang>[A-Za-z0-9_+-]*)[^\n]*\n(?P<body>.*?)\n(?P=fence)[ \t]*(?=\n|$)",
                       re.DOTALL)
_PY_LANGS = {"python", "py", "python3", "ipython", ""}
_EMPTY_RE = re.compile(
    r"\b(?:0|zero|no)\s+(?:rows?|results?|records?|observations?|matches?|datasets?)\b"
    r"|\bLength\s*=\s*0\b|<Table\s+length=0>|\bNo\s+(?:results|data|observations|matches)\s+found\b"
    r"|^\s*Empty\s+DataFrame\b",
    re.IGNORECASE | re.MULTILINE,
)
OUTPUT_LINES = 12
OUTPUT_LINE_CHARS = 160
STREAM_CAP_BYTES = 200_000
REPAIR_MAX_TOKENS = 6000


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _seconds(name: str, default: float) -> float:
    try:
        val = float(os.getenv(name, "") or default)
    except ValueError:
        return default
    return val if val > 0 else default


def code_check_enabled() -> bool:
    return _flag("QUASAR_CODE_CHECK", False)


def run_timeout_seconds() -> float:
    return _seconds("QUASAR_CODE_CHECK_TIMEOUT", 90.0)


def budget_seconds() -> float:
    return _seconds("QUASAR_CODE_CHECK_BUDGET", 240.0)


def allowed_hosts() -> Tuple[str, ...]:
    raw = os.getenv("QUASAR_CODE_CHECK_HOSTS", "")
    hosts = tuple(h.strip().lower().lstrip(".") for h in raw.split(",") if h.strip())
    return hosts or DEFAULT_ALLOWED_HOSTS


# --------------------------------------------------------------------------------------------- finding the script

@dataclass
class CodeBlock:
    start: int  # span of the body inside the answer text
    end: int
    code: str


def _top_level_modules(code: str) -> Optional[List[str]]:
    """Imported top-level module names, or None when the code does not parse (the run reports the error)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    mods: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.extend(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            mods.append(node.module.split(".")[0])
    return mods


def _missing_modules(code: str) -> List[str]:
    mods = _top_level_modules(code) or []
    missing = []
    for m in dict.fromkeys(mods):
        try:
            if importlib.util.find_spec(m) is None:
                missing.append(m)
        except (ImportError, ValueError):
            missing.append(m)
    return missing


def code_only(code: str) -> str:
    """The script without comments and docstrings, so a placeholder such as "Author: <your name>" in a header
    does not stop a check of code that runs fine."""
    import io
    import tokenize

    lines = code.splitlines(keepends=True)
    drop: List[Tuple[int, int]] = []  # 1-based inclusive line ranges of docstrings
    try:
        for node in ast.walk(ast.parse(code)):
            body = getattr(node, "body", None)
            if isinstance(body, list):
                for stmt in body:
                    if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                            and isinstance(stmt.value.value, str)):
                        drop.append((stmt.lineno, getattr(stmt, "end_lineno", stmt.lineno)))
    except SyntaxError:
        return code
    kept = [l for i, l in enumerate(lines, 1) if not any(a <= i <= b for a, b in drop)]
    try:
        toks = [t._replace(string="") if t.type == tokenize.COMMENT else t
                for t in tokenize.generate_tokens(io.StringIO("".join(kept)).readline)]
        return tokenize.untokenize(toks)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return "".join(kept)


def _looks_like_python(code: str) -> bool:
    """An untagged block counts as a script only if it parses as Python and imports something."""
    mods = _top_level_modules(code)
    return bool(mods)


def find_archive_script(text: str) -> Optional[CodeBlock]:
    """The longest fenced Python block that queries an archive and can run unattended, or None."""
    best: Optional[CodeBlock] = None
    for m in _FENCE_RE.finditer(text or ""):
        if m.group("lang").lower() not in _PY_LANGS:
            continue
        body = m.group("body")
        if not m.group("lang") and not _looks_like_python(body):
            continue  # an untagged block may be an ADQL query or shell text: never "repair" it into Python
        if not ARCHIVE_MARKERS.search(body) or SKIP_MARKERS.search(code_only(body)):
            continue
        if len(body.strip().splitlines()) < 3:
            continue
        if _missing_modules(body):
            continue
        if best is None or len(body) > len(best.code):
            best = CodeBlock(m.start("body"), m.end("body"), body)
    return best


# ------------------------------------------------------------------------------------------------ running it once

# The bootstrap runs first in the child: it installs the audit hook, stubs data downloads, caps the output, then
# runs the script as __main__. Hook state lives only in closures; the names are deleted before the script starts.
_BOOTSTRAP = r'''
import sys, os, socket
def _install(script, tmp, hosts):
    import runpy
    tmp = os.path.normcase(os.path.realpath(tmp))
    roots = {tmp}
    import sysconfig
    for p in [sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix] + list(sysconfig.get_paths().values()):
        if p:
            roots.add(os.path.normcase(os.path.realpath(p)))
    roots = tuple(sorted(roots))
    hosts = tuple(hosts)
    real_gai = socket.getaddrinfo
    resolved = set()
    busy = []
    def host_ok(h):
        h = (h.decode() if isinstance(h, bytes) else str(h)).lower().rstrip(".")
        return any(h == s or h.endswith("." + s) for s in hosts)
    def under(path, bases):
        try:
            p = os.path.normcase(os.path.realpath(os.fsdecode(path)))
        except Exception:
            return False
        return any(p == b or p.startswith(b.rstrip("\\/") + os.sep) for b in bases)
    def deny(msg):
        raise PermissionError("code check sandbox: " + msg)
    blocked = {"subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork", "os.forkpty",
               "os.startfile", "pty.spawn", "sys._current_frames",
               "gc.get_objects", "gc.get_referrers", "gc.get_referents", "socket.bind", "winreg.SetValue",
               "winreg.CreateKey", "winreg.DeleteKey", "winreg.SetValueEx", "winreg.DeleteValue"}
    bad_libs = {"ws2_32", "wsock32", "mswsock", "wininet", "winhttp", "urlmon", "shell32", "msvcrt", "ucrtbase",
                "libc", "advapi32", "ntdll"}
    bad_syms = ("CreateProcess", "WinExec", "ShellExecute", "LoadLibrary", "GetProcAddress", "VirtualAlloc",
                "VirtualProtect", "WriteProcessMemory", "CreateRemoteThread", "CreateThread", "OpenProcess",
                "CreateFile", "DeleteFile", "CopyFile", "ReplaceFile", "CreateHardLink",
                "CreateSymbolicLink", "RegSet", "RegCreate", "RegDelete", "system", "popen", "_wsystem", "_popen",
                "exec", "spawn", "fork", "dlopen", "socket", "connect", "WSA", "Internet", "URLDownload")
    def hook(event, args):
        if busy:
            return
        if event in blocked:
            deny(event + " is not allowed")
        # Scientific libraries load a few Windows system DLLs through ctypes (numpy: kernel32), so native code
        # is not blocked outright; the libraries and entry points that would get around this hook are.
        if event == "ctypes.dlopen":
            name = os.path.basename(str(args[0] or "")).lower()
            if name.split(".")[0] in bad_libs:
                deny("loading %r is not allowed" % (name,))
            return
        if event in ("ctypes.dlsym", "ctypes.dlsym/handle"):
            sym = args[1] if len(args) > 1 else ""
            sym = sym.decode() if isinstance(sym, bytes) else str(sym)
            if any(sym.startswith(p) for p in bad_syms):
                deny("native function %r is not allowed" % (sym,))
            return
        if event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex"):
            host = args[0]
            if host is None:
                return
            if not host_ok(host):
                deny("network access to %r is not allowed" % (host,))
            busy.append(1)
            try:
                for info in real_gai(host, None):
                    resolved.add(info[4][0])
            except OSError:
                pass
            finally:
                busy.pop()
        elif event in ("socket.connect", "socket.sendto", "socket.sendmsg"):
            addr = args[1] if len(args) > 1 else None
            ip = addr[0] if isinstance(addr, tuple) and addr else addr
            if ip not in resolved:
                deny("connection to %r is not allowed" % (ip,))
        elif event == "open":
            path, mode, flags = (list(args) + [None, None, None])[:3]
            if isinstance(path, int) or path is None:
                return
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
            if writing and not under(path, (tmp,)):
                deny("writing %r is not allowed" % (os.fsdecode(path),))
            if not writing and not under(path, roots):
                deny("reading %r is not allowed" % (os.fsdecode(path),))
        elif event in ("os.remove", "os.rename", "os.rmdir", "shutil.rmtree", "os.mkdir", "os.chmod", "os.truncate",
                       "shutil.move", "shutil.copyfile", "os.link", "os.symlink"):
            for p in args[:2]:
                if isinstance(p, (str, bytes, os.PathLike)) and not under(p, (tmp,)):
                    deny("%s on %r is not allowed" % (event, os.fsdecode(p)))
    sys.addaudithook(hook)

    class Capped:
        def __init__(self, inner, cap):
            self.inner, self.left = inner, cap
        def __getattr__(self, name):
            return getattr(self.inner, name)
        def write(self, s):
            if self.left > 0:
                piece = s[: self.left]
                self.left -= len(piece)
                self.inner.write(piece)
                if self.left <= 0:
                    self.inner.write("\n[output truncated by the code check]\n")
            return len(s)
        def flush(self):
            self.inner.flush()
    sys.stdout = Capped(sys.stdout, CAP)
    sys.stderr = Capped(sys.stderr, CAP)

    src = open(script, encoding="utf-8").read()
    def stub(name):
        def _skipped(*a, **k):
            print("[code check] %s skipped: no data files are downloaded during the check" % name)
            return []
        return _skipped
    if "astroquery" in src and "alma" in src.lower() and any(
            w in src for w in ("download_files", "retrieve_data_from_uid", "download_and_extract_files")):
        try:
            from astroquery.alma import core as _aq
            for n in ("download_files", "retrieve_data_from_uid", "download_and_extract_files"):
                if hasattr(_aq.AlmaClass, n):
                    setattr(_aq.AlmaClass, n, stub(n))
        except Exception:
            pass
    if "alminer" in src and "download_data" in src:
        try:
            import alminer as _am
            _am.download_data = stub("download_data")
        except Exception:
            pass
    sys.argv = [script]
    import traceback
    try:
        runpy.run_path(script, run_name="__main__")
    except (AttributeError, ImportError) as exc:
        # After the traceback, name what DOES exist (the object's attributes, or the module's public names), so the
        # one repair call can pick the real method instead of guessing again.
        traceback.print_exc()
        hint = ""
        try:
            if isinstance(exc, AttributeError) and getattr(exc, "obj", None) is not None:
                names = [n for n in dir(exc.obj) if not n.startswith("_")][:120]
                hint = "public attributes of %s: %s" % (type(exc.obj).__name__, ", ".join(names))
            elif isinstance(exc, ImportError) and sys.modules.get(getattr(exc, "name", None) or ""):
                mod = sys.modules[exc.name]
                names = [n for n in dir(mod) if not n.startswith("_")][:120]
                hint = "public names in module %s: %s" % (mod.__name__, ", ".join(names))
        except Exception:
            hint = ""
        if hint:
            sys.stderr.write("[code check] " + hint + "\n")
        sys.stderr.flush()
        raise SystemExit(1)
CAP = int(sys.argv[4])
_args = (sys.argv[1], sys.argv[2], sys.argv[3].split(","))
_install(*_args)
'''


@dataclass
class RunResult:
    ok: bool
    returncode: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool = False
    elapsed_s: float = 0.0
    skipped: str = ""

    @property
    def empty(self) -> bool:
        return self.ok and bool(_EMPTY_RE.search(self.stdout or ""))

    def error_line(self) -> str:
        """The last informative stderr line (the exception), shortened."""
        if self.timed_out:
            return "did not finish within the time limit"
        lines = [l.strip() for l in (self.stderr or "").splitlines()
                 if l.strip() and not l.strip().startswith("[code check] public")]
        for line in reversed(lines):
            if re.match(r"^[A-Za-z_][\w.]*(?:Error|Exception|Fault|Warning)?:", line) or "Error" in line:
                return line[:240]
        return (lines[-1][:240] if lines else f"exit code {self.returncode}")


def _child_env(tmp: str) -> Dict[str, str]:
    env = {
        "HOME": tmp, "USERPROFILE": tmp, "TMP": tmp, "TEMP": tmp, "TMPDIR": tmp,
        "APPDATA": tmp, "LOCALAPPDATA": tmp,
        "XDG_CONFIG_HOME": os.path.join(tmp, "config"), "XDG_CACHE_HOME": os.path.join(tmp, "cache"),
        "MPLBACKEND": "Agg", "MPLCONFIGDIR": os.path.join(tmp, "mpl"),
        "NO_PROXY": "", "PYTHONDONTWRITEBYTECODE": "1",
    }
    # Only what the interpreter and the OS socket layer need; never keys, tokens or service settings.
    for key in ("SYSTEMROOT", "SystemRoot", "WINDIR", "PATH", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def run_script(code: str, timeout: Optional[float] = None, python: Optional[str] = None) -> RunResult:
    """Run ``code`` once in the sandboxed child interpreter; never raises."""
    timeout = timeout or run_timeout_seconds()
    tmp = tempfile.mkdtemp(prefix="quasar-codecheck-")
    t0 = time.monotonic()
    try:
        for sub in ("config/astropy", "cache/astropy", "mpl"):
            os.makedirs(os.path.join(tmp, *sub.split("/")), exist_ok=True)
        script = os.path.join(tmp, "answer_script.py")
        boot = os.path.join(tmp, "_bootstrap.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(code)
        with open(boot, "w", encoding="utf-8") as fh:
            fh.write(_BOOTSTRAP)
        cmd = [python or sys.executable, "-I", "-B", "-X", "utf8", boot, script, tmp, ",".join(allowed_hosts()),
               str(STREAM_CAP_BYTES)]
        kwargs: Dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.run(cmd, cwd=tmp, env=_child_env(tmp), stdin=subprocess.DEVNULL,
                                  capture_output=True, timeout=timeout, **kwargs)
        except subprocess.TimeoutExpired as exc:
            out = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            err = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return RunResult(False, None, out, err, timed_out=True, elapsed_s=round(time.monotonic() - t0, 1))
        out = proc.stdout.decode("utf-8", "replace")
        err = proc.stderr.decode("utf-8", "replace")
        ok = proc.returncode == 0 and "Traceback (most recent call last)" not in err
        return RunResult(ok, proc.returncode, out, err, elapsed_s=round(time.monotonic() - t0, 1))
    except Exception as exc:  # noqa: BLE001 - a broken check must never break the answer
        return RunResult(False, None, "", "", skipped=f"{type(exc).__name__}: {exc}"[:200],
                         elapsed_s=round(time.monotonic() - t0, 1))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------- repairing

_UID_RE = re.compile(r"uid://A\d{3}/X[0-9a-f]+/X[0-9a-f]+", re.IGNORECASE)
_TRUNCATED_RE = re.compile(r"^\s*\.\.\.\s*$|\bLength\s*=\s*\d+\s+rows\b|\[\d+\s+rows\s+x\s+\d+\s+columns\]",
                           re.MULTILINE)


def _uid_map(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for u in _UID_RE.findall(text or ""):
        out.setdefault(u.lower(), u)
    return out


def _uids(text: str) -> set:
    return set(_uid_map(text))


def _similarity(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def uid_mismatch(stdout: str, reference: str) -> Optional[Dict[str, Any]]:
    """When the script printed archive UIDs and the answer's own archive tool results hold UIDs too, a description
    of how far apart the two sets are, or None when they agree (or one side has none). A script that prints only
    the first rows of a long table (a truncated table print) is not a mismatch for the rows it left out."""
    pmap, rmap = _uid_map(stdout), _uid_map(reference)
    printed, ref = set(pmap), set(rmap)
    if not printed or not ref:
        return None
    sim = _similarity(printed, ref)
    if sim >= 0.5:
        return None
    if printed <= ref and _TRUNCATED_RE.search(stdout or ""):
        return None
    return {
        "similarity": round(sim, 3), "printed": len(printed), "reference": len(ref),
        "missing": [rmap[u] for u in sorted(ref - printed)[:20]],
        "extra": [pmap[u] for u in sorted(printed - ref)[:20]],
    }


def _library_versions() -> str:
    """Versions of the archive libraries the check runs with, so the repair targets the installed API."""
    from importlib import metadata

    out = []
    for name in ("astroquery", "pyvo", "alminer", "astropy", "pandas", "numpy"):
        try:
            out.append(f"{name} {metadata.version(name)}")
        except Exception:
            continue
    return ", ".join(out) or "unknown"


def build_repair_prompt(question: str, code: str, result: RunResult, today: str,
                        mismatch: Optional[Dict[str, Any]] = None) -> str:
    if result.timed_out:
        what = (f"It did not finish within {run_timeout_seconds():.0f} seconds (a query that scans too much, or a "
                "loop of many small queries, is the usual cause).")
    elif mismatch:
        what = (
            "It ran without errors, but the archive UIDs it printed differ from the ones the archive tools returned "
            "for this same answer. "
            f"UIDs the tools returned that the script did not print (up to 20): {', '.join(mismatch['missing']) or 'none'}. "
            f"UIDs the script printed that the tools did not return (up to 20): {', '.join(mismatch['extra']) or 'none'}. "
            "If the script is meant to return the same selection, find the cause (for example a name match that "
            "misses other spellings of the target name, a different search radius, band, date or data-rights filter, "
            "or printing a different kind of UID) and fix it."
        )
    elif result.empty:
        what = "It ran without errors, but printed an empty result."
    else:
        what = f"It failed (exit code {result.returncode})."
    return (
        "You wrote the Python script below as part of your answer to a user's question. Before the answer is shown, "
        f"the script was run once, unchanged, on {today}, in a sandbox that can reach the astronomy archive services "
        "(the ALMA archive and its mirrors, CDS, NED, MAST, IRSA) but nothing else, with no credentials and with "
        f"data downloads disabled. Installed: {_library_versions()}.\n"
        f"{what}\n\n"
        f"User question (for context):\n{(question or '').strip()[:2000]}\n\n"
        f"Script:\n```python\n{code}\n```\n\n"
        f"Standard error (last part):\n```text\n{(result.stderr or '')[-3500:]}\n```\n\n"
        f"Standard output (last part):\n```text\n{(result.stdout or '')[-1500:]}\n```\n\n"
        "Fix the script so that it runs and does what the question asks. Keep the same approach, libraries and "
        "output where possible and change only what is wrong. Read the error message carefully: for a TAP or ADQL "
        "error, the server's message names the problem (a function the service does not support, a misspelled "
        "column or table, a quoting error). Do not guess column names that the error does not support. Do not add "
        "logins, credentials, data downloads or interactive input. If the result is genuinely correct as it is, "
        "return the script unchanged.\n"
        "Reply with ONLY the complete corrected script in one ```python block."
    )


def _extract_python(raw: str) -> Optional[str]:
    try:
        from core.harmony_filter import strip_harmony_markup

        raw = strip_harmony_markup(raw)
    except Exception:
        pass
    blocks = [m.group("body") for m in _FENCE_RE.finditer("\n" + (raw or ""))
              if m.group("lang").lower() in _PY_LANGS]
    if not blocks:
        return None
    code = max(blocks, key=len)
    return code if code.strip() else None


def _default_repair_call(prompt: str, model: str, max_tokens: int) -> str:
    from core.llm_client import LLMClient

    client = LLMClient(model=model)
    resp = client.responses.create(model=model, input=prompt, max_output_tokens=max_tokens)
    return str(getattr(resp, "output_text", "") or "")


def _call_with_timeout(fn: Callable[[], str], timeout: float) -> Tuple[Optional[str], str]:
    from core.llm_client import get_llm_request_context, reinstall_llm_request_context

    ctx = get_llm_request_context()
    box: Dict[str, Any] = {}

    def body() -> None:
        try:
            with reinstall_llm_request_context(ctx):
                box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = f"{type(exc).__name__}: {exc}"

    t = threading.Thread(target=body, name="code-check-repair", daemon=True)
    t.start()
    t.join(timeout=max(1.0, timeout))
    if t.is_alive():
        return None, f"repair call timed out after {timeout:.0f}s"
    if "error" in box:
        return None, str(box["error"])[:200]
    return box.get("value"), ""


# --------------------------------------------------------------------------------------------------- the whole step

@dataclass
class CheckOutcome:
    text: str
    status: str  # skipped | passed | repaired | failed | timed_out | empty
    info: Dict[str, Any] = field(default_factory=dict)
    stdout: str = ""
    note: str = ""


def _output_excerpt(stdout: str) -> str:
    lines = [l.rstrip()[:OUTPUT_LINE_CHARS] for l in (stdout or "").splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ""
    more = len(lines) > OUTPUT_LINES
    body = "\n".join(lines[:OUTPUT_LINES]) + ("\n..." if more else "")
    return f"\n\nFirst lines of the output when run here:\n\n```text\n{body}\n```"


def _note(status: str, today: str, first: RunResult, final: Optional[RunResult], repaired: bool = False) -> str:
    head = f"\n\n**Code check:** I ran the script above once against the live archive services ({today})"
    err = first.error_line().replace("`", "'")
    what_first = ("printed archive UIDs that differ from the archive results in this answer" if first.ok
                  else f"failed (`{err}`)")
    if status == "passed":
        return head + ", and it completed without errors." + _output_excerpt(first.stdout)
    if status == "repaired" and final is not None:
        return (head + f". The first version {what_first}, so the script above is a corrected version, which then "
                "completed without errors." + _output_excerpt(final.stdout))
    if status == "repaired_empty":
        return (head + f". The first version failed (`{err}`), so the script above is a corrected version. It "
                "completed without errors but printed an empty result, so check the query constraints.")
    if status == "empty":
        return head + ". It completed without errors but printed an empty result, so check the query constraints."
    if status == "mismatch":
        return (head + ". It completed without errors, but the archive UIDs it printed differ from the archive "
                "results in this answer, so compare the two before relying on either." + _output_excerpt(first.stdout))
    if status == "timed_out":
        tail = " A corrected attempt did not finish either." if repaired else ""
        return (head + ", but it did not finish within the time limit here (archive queries can be slow), so it "
                f"is not verified.{tail}")
    tail = " A corrected attempt did not run either." if repaired else ""
    return (head + f", and it failed (`{err}`).{tail} Treat the script as a starting point and check it before "
            "use.")


def check_answer_code(
    text: str,
    *,
    question: str = "",
    model: str = "",
    reference_text: str = "",
    repair_call: Optional[Callable[[str, str, int], str]] = None,
    runner: Optional[Callable[[str, float], RunResult]] = None,
    deadline: Optional[float] = None,
    today: Optional[str] = None,
) -> CheckOutcome:
    """Run the answer's archive script once; repair once on failure, a timeout, an empty result or archive UIDs
    that differ from ``reference_text`` (this turn's tool results); return the answer with the note appended."""
    info: Dict[str, Any] = {"status": "skipped"}
    if not code_check_enabled():
        info["skipped"] = "QUASAR_CODE_CHECK off"
        return CheckOutcome(text, "skipped", info)
    block = find_archive_script(text)
    if block is None:
        info["skipped"] = "no runnable archive script"
        return CheckOutcome(text, "skipped", info)
    t0 = time.monotonic()
    end_by = t0 + budget_seconds()
    if deadline is not None:
        end_by = min(end_by, deadline - 20.0)
    per_run = run_timeout_seconds()
    if end_by - t0 < min(per_run, 30.0):
        info["skipped"] = "not enough time left in the turn"
        return CheckOutcome(text, "skipped", info)
    run = runner or (lambda code, timeout: run_script(code, timeout))
    today = today or _dt.date.today().isoformat()

    first = run(block.code, min(per_run, end_by - time.monotonic()))
    if first.skipped:
        info["skipped"] = first.skipped
        return CheckOutcome(text, "skipped", info)
    mism = uid_mismatch(first.stdout, reference_text) if first.ok else None
    info.update(first_ok=first.ok, first_rc=first.returncode, first_timed_out=first.timed_out,
                first_elapsed_s=first.elapsed_s, first_error=None if first.ok else first.error_line(),
                first_empty=first.empty, mismatch=mism)
    _err = "" if first.ok else f" error={first.error_line()[:120]!r}"
    print(f"[CODE CHECK] first run ok={first.ok} empty={first.empty} timed_out={first.timed_out} "
          f"mismatch={bool(mism)} elapsed={first.elapsed_s}s{_err}")

    if first.ok and not first.empty and not mism:
        status, final, new_code = "passed", first, None
    else:
        status = ("timed_out" if first.timed_out else "mismatch" if mism else "empty" if first.ok else "failed")
        final, new_code = None, None
        left = end_by - time.monotonic()
        if left > 25.0 and model:
            prompt = build_repair_prompt(question, block.code, first, today, mismatch=mism)
            call = repair_call or _default_repair_call
            raw, err = _call_with_timeout(lambda: call(prompt, model, REPAIR_MAX_TOKENS), min(90.0, left - 15.0))
            info["repair_error"] = err or None
            fixed = _extract_python(raw or "") if raw else None
            if fixed and fixed.strip() != block.code.strip() and not SKIP_MARKERS.search(code_only(fixed)):
                left = end_by - time.monotonic()
                if left > 10.0:
                    second = run(fixed, min(per_run, left))
                    info["repair_attempted"] = True
                    info.update(second_ok=second.ok, second_timed_out=second.timed_out,
                                second_elapsed_s=second.elapsed_s, second_empty=second.empty,
                                second_error=None if second.ok else second.error_line())
                    _err2 = "" if second.ok else f" error={second.error_line()[:160]!r}"
                    print(f"[CODE CHECK] repaired run ok={second.ok} empty={second.empty} "
                          f"elapsed={second.elapsed_s}s{_err2}")
                    if mism:
                        sim2 = _similarity(_uids(second.stdout), _uids(reference_text))
                        info["second_similarity"] = round(sim2, 3)
                        if second.ok and not second.empty and sim2 > mism["similarity"]:
                            status, final, new_code = "repaired", second, fixed
                    elif second.ok and not second.empty:
                        status, final, new_code = "repaired", second, fixed
                    elif second.ok and not first.ok:
                        status, final, new_code = "repaired_empty", second, fixed
            elif fixed:
                info["repair"] = "unchanged"

    out = text
    if new_code is not None:
        out = text[:block.start] + new_code.rstrip("\n") + text[block.end:]
    note = _note(status, today, first, final, repaired=bool(info.get("repair_attempted")))
    try:
        from services.content_safety import looks_explicit_text

        if looks_explicit_text(note) and not looks_explicit_text(text):
            # archive output can hold anything (a "XXX" placeholder in a table): the final-answer guard would then
            # replace the whole answer, so keep the note without the output excerpt
            note = note.split("\n\nFirst lines of the output", 1)[0]
            if looks_explicit_text(note):
                note = ""
    except Exception:
        pass
    out = out.rstrip() + note
    info["status"] = status
    info["elapsed_s"] = round(time.monotonic() - t0, 1)
    return CheckOutcome(out, status, info, stdout=(final.stdout if final else first.stdout) or "", note=note)
