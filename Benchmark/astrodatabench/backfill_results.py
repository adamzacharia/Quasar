"""One-off, additive back-fill for result.json files written by run_astrodatabench.py v0.1.0
before the Codex-review fields existed (CX-03/05/06): adds `candidate`, `trial`, a portable
`task_id.path`, `harness.disclosure` and `usage.json_source` (re-derived from the saved reply).
Never touches rewards or verifier output. Idempotent.

    python Benchmark/astrodatabench/backfill_results.py --run-id 20260914
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_astrodatabench as h  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--bench-root", type=Path, default=h.DEFAULT_BENCH_ROOT)
    ap.add_argument("--agent-python", type=Path, default=h.DEFAULT_AGENT_PYTHON)
    args = ap.parse_args()
    # mirror the defaults the runs used
    args.verifier_mode, args.verifier_python_version, args.verifier_python = "uvx", "3.13", Path(sys.executable)
    args.agent_timeout, args.max_turns, args.cmd_timeout = None, 40, 300.0

    tasks = {t.task_id: t for t in h.load_tasks(args.bench_root)}
    jobs_root = args.bench_root / "data-discovery" / "jobs"
    n = 0
    for rp in sorted(jobs_root.glob(f"{args.run_id}-*/*/result.json")):
        r = json.loads(h.read_text(rp))
        task_id = Path(r["task_id"]["path"]).name
        if task_id not in tasks:  # Windows absolute path from v0.1.0
            task_id = rp.parent.name.split("__", 1)[0]
        task = tasks[task_id]
        model_name = r["config"]["agent"]["model_name"]
        name = r["agent_info"]["name"]
        if name == "quasar":
            cand = h.QuasarCandidate("http://localhost:8000", model_name.split("/", 1)[1], None, bool(r["config"]["agent"].get("web_search", True)))
        elif name == "shell-agent":
            cand = h.ShellAgentCandidate(model_name, r["config"]["agent"].get("openai_effort") or "medium",
                                         int(r["config"]["agent"].get("anthropic_thinking_budget") or 0), args.max_turns, args.cmd_timeout)
        else:
            cand = h.OracleCandidate()
        changed = False
        if "candidate" not in r:
            r["candidate"] = model_name; changed = True
        if "trial" not in r:
            r["trial"] = int(r["trial_name"].rsplit("trial", 1)[-1]); changed = True
        portable = h._portable_task_path(task, args)
        if r["task_id"]["path"] != portable:
            r["task_id"]["path"] = portable; changed = True
        if "disclosure" not in r["harness"]:
            r["harness"]["disclosure"] = h.build_disclosure(task, cand, args)
            r["harness"]["disclosure"]["backfilled"] = True
            r["harness"]["overwrote_previous_trial"] = False
            changed = True
        usage = r["agent_result"].get("usage") or {}
        if name == "quasar" and "json_source" not in usage:
            resp = rp.parent / "agent" / "response.md"
            if resp.exists():
                _, src = h.extract_json(h.read_text(resp))
                usage["json_source"] = src
                r["agent_result"]["usage"] = usage
                changed = True
        if name == "quasar" and "n_tool_calls_source" not in usage:
            # v0.1.0 recorded 0 tool calls whenever the final tool_trace never arrived (timeouts);
            # recount from the saved SSE events: trace if present, else status steps as a lower bound
            ev_path = rp.parent / "agent" / "events.jsonl"
            trace_calls = status_steps = 0
            if ev_path.exists():
                for line in h.read_text(ev_path).splitlines():
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("type") == "tool_trace":
                        trace_calls = len(ev.get("calls") or []) + int(ev.get("omitted_calls") or 0)
                    elif ev.get("type") == "status" and ev.get("state") == "running" and h._is_tool_status_step(str(ev.get("step", ""))):
                        status_steps += 1
            usage["status_tool_steps"] = status_steps
            if trace_calls:
                usage["n_tool_calls_source"] = "tool_trace"
                r["agent_result"]["n_tool_calls"] = trace_calls
            elif status_steps:
                usage["n_tool_calls_source"] = "status_steps_lower_bound"
                r["agent_result"]["n_tool_calls"] = status_steps
            else:
                usage["n_tool_calls_source"] = "none"
            r["agent_result"]["usage"] = usage
            changed = True
        if changed:
            h.write_json(rp, r)
            n += 1
    print(f"back-filled {n} result.json file(s) under {jobs_root}/{args.run_id}-*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
