"""
ALMABench v0 — runner (reuses the DataLabBench scorer / judge / reporter)
==========================================================================
Benchmarks Quasar's ALMA archive layer against the 12 held-out ALMABench
questions (almabench_dataset_v0.py) with step-level partial credit.

Usage:
    python Benchmark/almabench/run_almabench.py                       # full run, local backend
    python Benchmark/almabench/run_almabench.py --questions ALMA-01 ALMA-07
    python Benchmark/almabench/run_almabench.py --skip-judge          # deterministic checks only
    python Benchmark/almabench/run_almabench.py --dry-run
    python Benchmark/almabench/run_almabench.py --self-test           # offline scorer integrity check
    python Benchmark/almabench/run_almabench.py --runs 3              # n=3 repeats per question

Before/after protocol (report Section 3.3, wave 3): run once against the
pre-wave-1 backend and once after, same model, same judge, n=3; the paired
numbers (results.json rollup.full_pct / auto_pct per run) are the publishable
delta. The judge provider/model is recorded in results.json.

Harness patches this bench relies on (in run_datalabbench.py): the uniform
`request` provenance of every traced call (kind 'adql' + literal text) is
captured on ToolCall; the SQL-capable tool set is a module variable widened
here to the ALMA ADQL tools; ADQL CIRCLE(...) cones feed position_near /
radius_near. DataLabBench scoring is unaffected (it keeps the datalab_*
default).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DLB_DIR = REPO_ROOT / "Benchmark" / "datalabbench"
for _p in (REPO_ROOT, DLB_DIR, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_datalabbench as harness  # noqa: E402  (shared scorer/judge/reporter)
import almabench_dataset_v0 as dataset  # noqa: E402


def install_alma_harness() -> None:
    """Point the shared scorer at the ALMA dataset and ADQL-executing tools."""
    harness.SQL_CAPABLE_TOOL_RE = re.compile(dataset.ALMA_SQL_CAPABLE_TOOL_PATTERN, re.IGNORECASE)
    harness.QUESTIONS = dataset.QUESTIONS
    harness.GLOBAL_PENALTIES = dataset.GLOBAL_PENALTIES
    harness.BENCH_NAME = dataset.BENCH_NAME
    harness.BENCH_VERSION = dataset.BENCH_VERSION
    harness.TIER_WEIGHTS = dataset.TIER_WEIGHTS
    harness.TIER_LABELS = dataset.TIER_LABELS
    # Axis mapping: reuse the DataLabBench rule (auto -> faithfulness, judge ->
    # correctness) with the ALMABench overrides.
    _orig_axis = harness.axis_for_checkpoint

    def _axis(question_id, checkpoint):
        override = dataset.AXIS_OVERRIDES.get(question_id, {}).get(checkpoint.get("id"))
        if override:
            return override
        return _orig_axis(question_id, checkpoint)

    harness.axis_for_checkpoint = _axis


def _self_test() -> int:
    """Offline integrity checks specific to the ALMA harness patches."""
    install_alma_harness()
    failures = 0

    def check(name, cond, detail=""):
        nonlocal failures
        print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
        if not cond:
            failures += 1

    problems = dataset.validate_dataset()
    check("dataset validates", not problems, "; ".join(problems))

    adql = ("SELECT member_ous_uid FROM ivoa.obscore WHERE INTERSECTS(CIRCLE('ICRS', 83.81, -5.37, 0.1), "
            "s_region) = 1 AND em_min <= 0.000868 AND em_max >= 0.000857")
    call = harness.ToolCall(name="advanced_search", arguments={"query": adql}, ok=True,
                            request_kind="adql", request_text=adql)
    ev = harness.Evidence(calls=[call], response_text="12 rows across 4 MOUS and 9 execution blocks")
    ev.trace_source = "tool_trace"
    ev.sql_texts = harness.extract_executed_sql(ev.calls)
    check("ALMA ADQL request enters executed SQL", any("INTERSECTS" in s for s in ev.sql_texts))
    check("CIRCLE cone feeds position_near",
          harness.eval_check({"kind": "position_near", "ra": 83.81, "dec": -5.37, "tol_deg": 0.05}, ev).passed)
    check("CIRCLE radius feeds radius_near",
          harness.eval_check({"kind": "radius_near", "value": 0.1, "tol": 0.02}, ev).passed)
    check("INTERSECTS trace_regex passes",
          harness.eval_check({"kind": "trace_regex", "pattern": r"INTERSECTS\s*\(.*s_region"}, ev).passed)

    # Point-only cone with no INTERSECTS trips GP-A3; grain words suppress GP-A1.
    point_adql = "SELECT * FROM ivoa.obscore WHERE CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', 1, 2, 0.1)) = 1"
    ev2 = harness.Evidence(calls=[harness.ToolCall(name="advanced_search", arguments={"query": point_adql}, ok=True,
                                                    request_kind="adql", request_text=point_adql)],
                           response_text="There are 143 observations of M87.")
    ev2.trace_source = "tool_trace"
    ev2.sql_texts = harness.extract_executed_sql(ev2.calls)
    hits = {p.id for p in harness.apply_penalties(dataset.QUESTIONS[0], ev2)}
    check("GP-A3 fires on point-only cone", "GP-A3" in hits, str(hits))
    check("GP-A1 fires on 'N observations' without grain words", "GP-A1" in hits, str(hits))
    hits_ok = {p.id for p in harness.apply_penalties(dataset.QUESTIONS[0], ev)}
    check("GP-A1/GP-A3 silent with grain words + INTERSECTS", not ({"GP-A1", "GP-A3"} & hits_ok), str(hits_ok))

    # A Data Lab tool's query must still NOT count as ALMA executed SQL.
    ev3 = harness.Evidence(calls=[harness.ToolCall(name="datalab_sql_query", arguments={"sql": "SELECT 1"}, ok=True)])
    ev3.sql_texts = harness.extract_executed_sql(ev3.calls)
    check("datalab tool SQL excluded under the ALMA tool set", ev3.sql_texts == [])

    # Bandwidth/2 idiom in executed ADQL trips GP-A2.
    bw_adql = "SELECT * FROM ivoa.obscore WHERE (frequency - 0.5*bandwidth/1e9) < 230"
    ev4 = harness.Evidence(calls=[harness.ToolCall(name="advanced_search", arguments={"query": bw_adql}, ok=True,
                                                    request_kind="adql", request_text=bw_adql)],
                           response_text="found 3 MOUS")
    ev4.trace_source = "tool_trace"
    ev4.sql_texts = harness.extract_executed_sql(ev4.calls)
    check("GP-A2 fires on bandwidth/2 idiom", "GP-A2" in {p.id for p in harness.apply_penalties(dataset.QUESTIONS[0], ev4)})

    print(f"\n{'ALL OK' if not failures else f'{failures} FAILURE(S)'}")
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{dataset.BENCH_NAME} v{dataset.BENCH_VERSION} runner")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--model", default=os.getenv("ALMABENCH_MODEL", "gpt-4.1"), help="Quasar model under test")
    parser.add_argument("--judge-model", default=os.getenv("DLB_JUDGE_MODEL", "gpt-4o"))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--questions", nargs="*", default=None, help="Question ids, e.g. ALMA-01 ALMA-07")
    parser.add_argument("--runs", type=int, default=1, help="Repeat each question n times (n=3 for publication)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        sys.exit(_self_test())

    install_alma_harness()
    problems = dataset.validate_dataset()
    if problems:
        print("DATASET INVALID:")
        for p in problems:
            print(" -", p)
        sys.exit(2)

    selected = dataset.QUESTIONS
    if args.questions:
        wanted = {w.upper() for w in args.questions}
        selected = [q for q in dataset.QUESTIONS if q["id"].upper() in wanted]
        if not selected:
            print(f"No questions matched: {args.questions}")
            sys.exit(1)

    if args.dry_run:
        print(f"\n{dataset.BENCH_NAME} v{dataset.BENCH_VERSION} — dry run ({len(selected)} questions x {args.runs})\n")
        for q in selected:
            n_auto = sum(1 for c in q["checkpoints"] if c["type"] == "auto")
            n_judge = len(q["checkpoints"]) - n_auto
            print(f"  [{q['id']}] T{q['tier']} (w={dataset.TIER_WEIGHTS[q['tier']]}) {q['title']}")
            print(f"      {n_auto} auto + {n_judge} judge checkpoints; prompt: {q['prompt'][:90]}…\n")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else \
        HERE / "results" / f"{timestamp}_{args.model.replace('/', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    auth_token = args.auth_token or os.getenv("QUASAR_API_TOKEN") or harness.login_for_token(args.api_url)
    judge = None if args.skip_judge else harness.LLMJudge(args.judge_model)

    print(f"\n{'=' * 64}")
    print(f"  {dataset.BENCH_NAME} v{dataset.BENCH_VERSION}")
    print(f"  API: {args.api_url} | model: {args.model} | judge: {'(skipped)' if args.skip_judge else args.judge_model}")
    print(f"  questions: {len(selected)} x {args.runs} run(s) | output: {out_dir}")
    print(f"{'=' * 64}\n")

    per_run = []
    for run_idx in range(1, args.runs + 1):
        run_dir = out_dir / f"run{run_idx}" if args.runs > 1 else out_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        results: List[harness.QuestionResult] = []
        for i, q in enumerate(selected, 1):
            print(f"  [run {run_idx}] [{i}/{len(selected)}] {q['id']} (T{q['tier']}): {q['title']}")
            results.append(harness.run_question(q, args, auth_token, judge, run_dir))
        rollup = harness.overall_rollup(results)
        (run_dir / "results.json").write_text(json.dumps({
            "bench": dataset.BENCH_NAME, "version": dataset.BENCH_VERSION,
            "date": timestamp, "run": run_idx, "api_url": args.api_url, "model": args.model,
            "judge_model": None if args.skip_judge else args.judge_model,
            "rollup": rollup,
            "questions": [{
                **{k: getattr(r, k) for k in ("id", "tier", "title", "response_time_s",
                                              "error", "judge_error", "trace_source",
                                              "n_tool_calls", "n_images", "usage")},
                "percentage": r.percentage,
                "auto_percentage": r.auto_percentage,
                "grade": r.grade,
                "checkpoints": [asdict(c) for c in r.checkpoints],
                "penalties": [asdict(p) for p in r.penalties],
            } for r in results],
        }, indent=2, default=str), encoding="utf-8")
        try:
            harness.generate_charts(results, run_dir)
        except Exception as exc:  # noqa: BLE001 - charts are optional
            print(f"    [charts skipped: {exc}]")
        harness.generate_report(results, run_dir, args, rollup)
        per_run.append(rollup)
        full = rollup["full_pct"]
        print(f"\n  RUN {run_idx} OVERALL: {f'{full:.1f} / 100' if full is not None else 'N/A (unjudged)'}"
              f"   (auto-only: {rollup['auto_pct']:.1f} / 100)")
        for r in results:
            pct = f"{r.percentage:5.1f}%" if r.percentage is not None else f"a{r.auto_percentage:5.1f}%"
            pens = f"  penalties: {', '.join(p.id for p in r.penalties)}" if r.penalties else ""
            print(f"    {r.id}  T{r.tier}  {pct}  [{r.grade}]{pens}")

    if args.runs > 1:
        autos = [r["auto_pct"] for r in per_run]
        fulls = [r["full_pct"] for r in per_run if r["full_pct"] is not None]
        summary = {
            "runs": args.runs,
            "auto_pct_mean": sum(autos) / len(autos),
            "auto_pct_min": min(autos), "auto_pct_max": max(autos),
            "full_pct_mean": (sum(fulls) / len(fulls)) if fulls else None,
            "full_pct_min": min(fulls) if fulls else None, "full_pct_max": max(fulls) if fulls else None,
            "judge_model": None if args.skip_judge else args.judge_model,
            "model": args.model,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n  n={args.runs} auto mean {summary['auto_pct_mean']:.1f} "
              f"(min {summary['auto_pct_min']:.1f}, max {summary['auto_pct_max']:.1f})")
    print(f"{'=' * 64}")
    print(f"  Results: {out_dir}")


if __name__ == "__main__":
    main()
