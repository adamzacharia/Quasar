# ALMABench v0

Twelve held-out ALMA archive questions on the DataLabBench checkpoint DSL, each
expressing guardrails of the reviewed *Working with ALMA data* skill
(`third_party/alma-data-skill`, reviewed 2026-07-18): row grain (rows vs MOUS vs
execution blocks), ObsCore units, band-to-band `band_list` tokens, footprint
cones (`INTERSECTS` on `s_region`), wavelength-overlap prefilters plus
`frequency_support` verification, cycle project-code periods, the three-state
QA2 disposition, typed DataLink inventories, the scriptForPI restore, and the
Technical Handbook sensitivity equation.

```
Benchmark/almabench/
├── almabench_dataset_v0.py   # questions, checkpoints, penalties (GP-00, GP-A1..A3)
├── run_almabench.py          # runner: reuses the DataLabBench scorer/judge/reporter
└── results/                  # <timestamp>_<model>/[runN/]results.json, report, traces
```

## Run

```bash
# backend on :8000 (scripts/local/start_quasar.ps1 -Restart), then:
.venv/Scripts/python Benchmark/almabench/run_almabench.py --self-test
.venv/Scripts/python Benchmark/almabench/run_almabench.py --dry-run
.venv/Scripts/python Benchmark/almabench/run_almabench.py --model <model> --runs 3
.venv/Scripts/python Benchmark/almabench/run_almabench.py --questions ALMA-03 ALMA-04 --skip-judge
```

## Before / after protocol

Wave 3 of the ALMA integration plan runs the bench once against the pre-wave-1
backend and once after, same model, same judge model, `--runs 3`. The pair of
`summary.json` means (auto-only and judged) is the publishable delta. The judge
provider and model are recorded in every `results.json`.

## Harness dependencies

The bench needs three small patches in `Benchmark/datalabbench/run_datalabbench.py`
(all present): the uniform `request` provenance of each traced call is captured
on `ToolCall` (`request_kind`, `request_text`), so the EXACT executed ADQL of
the ALMA tools is scored; the SQL-capable tool set is a module variable
(`SQL_CAPABLE_TOOL_RE`) that this runner widens to the ALMA ADQL tools; and
ADQL `CIRCLE('ICRS', ra, dec, r)` cones feed `position_near` / `radius_near`.
DataLabBench keeps the `datalab_*` default and its scoring is unchanged.

## Penalties

| id | trigger | points |
|---|---|---|
| GP-00 | concrete archive numbers in the answer with no successful ALMA tool call | 20 |
| GP-A1 | "N observations" stated without any row / MOUS / dataset / EB grain word | 10 |
| GP-A2 | executed ADQL uses `frequency ± bandwidth/2` (aggregate bandwidth as one window) | 10 |
| GP-A3 | executed cone tests only `CONTAINS(POINT(s_ra, s_dec))` with no `INTERSECTS` | 5 |

## Held-out design

The prompts were written from the skill's guardrails, not from the unit tests
or the fix list that implemented them, and no router rule, golden example or
test fixture quotes them. Public identifiers only (NGC 253, Sgr A*, M87, Orion
KL, MOUS `uid://A001/X133d/X1d1`); if one goes stale the rubric still scores
process (tool choice, ADQL shape, disclosures), not a single archive number.
