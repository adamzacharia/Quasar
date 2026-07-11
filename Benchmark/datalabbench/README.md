# DataLabBench v1

A 15-question benchmark for Quasar's **NOIRLab Astro Data Lab** subsystem
(TAP/ADQL catalog science, q3c spatial queries, SIA image cutouts, SVO filter
service, analysis plots). The questions are taken verbatim from *"Suggested
science-user prompts for an LLM + Astro Data Lab MCP server"* and span 7 tiers
of increasing complexity — from schema discovery to a fully open-ended
Milky Way satellite search.

The goal of the loop: **run the bench → read the "Top improvement targets"
table → improve Quasar's tools/routing → re-run → repeat until 100/100.**

## Files

| File | Purpose |
|---|---|
| `dlb_dataset_v1.py` | The 15 questions + machine-readable rubrics (single source of truth) |
| `run_datalabbench.py` | Runner (SSE client), deterministic scorer, LLM judge, report + charts |
| `RUBRIC.md` | Human-readable rubric — **generated**, regenerate with `--emit-rubric` |
| `results/<timestamp>_<model>/` | Per-run artifacts (gitignored) |

## Quick start

```bash
# 1. start the backend (PowerShell):  ./scripts/local/start_quasar.ps1 -Restart
# 2. full run against local backend:
python Benchmark/datalabbench/run_datalabbench.py

# subset / smoke:
python Benchmark/datalabbench/run_datalabbench.py --questions DLB-01 DLB-02 DLB-03

# deterministic checks only (no judge LLM, cheaper):
python Benchmark/datalabbench/run_datalabbench.py --skip-judge

# remote deployment:
python Benchmark/datalabbench/run_datalabbench.py --api-url https://... --auth-token <jwt>

# harness integrity check (offline, no API/LLM):
python Benchmark/datalabbench/run_datalabbench.py --self-test

# regenerate RUBRIC.md after editing the dataset:
python Benchmark/datalabbench/run_datalabbench.py --emit-rubric
```

Auth: `--auth-token` or `QUASAR_API_TOKEN`; otherwise the runner attempts the
local test login (`QUASAR_BENCH_USER`/`QUASAR_BENCH_PASS`, default `1@1`/`1`).

## Scoring model (v1)

Each question = **100 points** of weighted **checkpoints** distilled from the
source PDF's "Expected MCP actions" / "Expected query":

- **auto** checkpoints — scored deterministically against the captured
  evidence. Execution credit only counts **executed** evidence: arguments +
  `query_summary` SQL of *successful* tool calls (structured cut args are
  also flattened to SQL-like strings so reference patterns match). SQL merely
  pasted into the answer text is captured separately and earns nothing — a
  model cannot score by quoting the reference query. A checkpoint holds N
  checks; credit = `points × passed/N` → **partial credit for partially
  completed steps**.
- **judge** checkpoints — an LLM judge awards fractional credit `0.0–1.0`
  per the rubric's guidance, grounded in the tool trace (**claims not backed
  by a tool call earn 0** — fabrication cannot score).
- **penalties** — the PDF's query-size guardrails, enforced bench-wide:
  fabricated results (GP-00), unbounded row scans (GP-01), the flat
  `q3c_join` anti-pattern (GP-02), `ra/dec BETWEEN` boxes as the only spatial
  bound of row-level queries (GP-03), plus question-specific ones (e.g.
  joining a separate WISE catalog in DLB-10). Questions whose *reference
  solution* legitimately uses a pattern disable that penalty locally.

**Overall score** = tier-weighted mean of question percentages:
T1×1.0, T2×1.2, T3×1.4, T4×1.6, T5×1.8, T6×2.0, T7×2.4.
Grades: A+ ≥97, A ≥90, B ≥75, C ≥60, D ≥40, F <40.

Full per-question rubrics: see [RUBRIC.md](RUBRIC.md).

## Evidence capture

The runner consumes Quasar's SSE stream and records **every** event
(`events.jsonl`). Primary evidence is the `tool_trace` event (full tool
names/arguments/outputs) emitted at the end of a run; if the backend
predates it the runner falls back to `tool_call`/`run_progress` events
(names only — auto checks that need arguments will then under-score, and
the report flags `trace_source: fallback`).

Per-question artifacts under `results/<run>/<DLB-XX>/`:
`response.md`, `events.jsonl`, `trace.json` (calls + extracted SQL + images),
`score.json`.

Known limitations (documented, non-blocking — Codex-reviewed):
- Tool calls made inside *conductor subtask threads* are not traced into the
  parent request yet; the main streaming loop (which handles the Data Lab
  flows) is fully traced.
- DLB-13 carries an explicit PDF-fidelity exception: GP-01/GP-03 are waived
  there because the source PDF's reference query is itself a row-level
  RA/Dec BETWEEN box over the compact `sdss_dr17.specobj` table.

## Versioning

This is **DataLabBench v1.0**. The question set, checkpoint weights, and
penalties are frozen for a version: cross-run scores are only comparable
within the same version. To change the rubric, bump `BENCH_VERSION` in
`dlb_dataset_v1.py`, regenerate `RUBRIC.md`, and note the change here.

Planned for v1.x (not in v1.0):
- calibrated numeric goldens (e.g. the exact DLB-02 cone count) obtained by
  running each `reference_sql` live against Data Lab
- vision-judge scoring of the emitted plots (today plots are judged from the
  tool parameters + text description)
