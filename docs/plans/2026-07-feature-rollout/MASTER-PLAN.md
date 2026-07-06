# MASTER PLAN — Quasar feature rollout, July 2026

Nine features approved from the Claude⇄Codex duel ranking
(`tmp/codex/tasks/rank-13-features/final.md`). Excluded by user: #7 GW
ranking, #12 acknowledgments, #13 AstroBench. #5 executable CASA stays gated
on Linux worker infra (not in this rollout).

Any capable model can execute a task: each spec file is self-contained,
assumes zero conversation context, and points to `CONVENTIONS.md` for the
shared contract. The intended driver is Codex via the bridge; Claude (or the
user) runs the review gates.

## Build order and why

| Phase | Task | Spec file | New deps | Rationale for position |
|---|---|---|---|---|
| 0 | Prerequisites | (below) | lightkurve, psrqpy | everything else assumes them |
| 1a | F02 MOC coverage | `F02-moc-coverage.md` | — | smallest; validates the pipeline end-to-end |
| 1b | F09 dust/extinction | `F09-dust-extinction.md` | — | quick win |
| 1c | F10 distances | `F10-distances.md` | — | quick win |
| 1d | F04 ATNF pulsars | `F04-atnf-pulsars.md` | psrqpy | quick win |
| 1e | F11 solar system | `F11-solar-system.md` | — | quick win |
| 2 | F06 light curves | `F06-lightcurves.md` | lightkurve | mid-size; reuses AlerceClient |
| 3 | F03 radio SED | `F03-radio-sed.md` | — | highest science value; needs care |
| 4 | F08 sky monitors | `F08-sky-monitors.md` | — | touches DB; reuses AlerceClient |
| 5 | F01 VO registry | `F01-vo-registry.md` | — | biggest surface; do last with practice |
| 6 | Integration + guard review | (below) | — | cumulative check |

Tasks are SEQUENTIAL (each build edits `core/agent.py`; parallel builds would
conflict). Within a phase the order is a suggestion; across phases it is not.

## Phase 0 — prerequisites (run once, ~5 min)

```powershell
# 1. install new deps
.venv/Scripts/python.exe -m pip install lightkurve psrqpy

# 2. pin them (append two lines to requirements.txt)
#    lightkurve>=2.4
#    psrqpy>=1.2

# 3. record the unit-test baseline (pre-existing failures) into PROGRESS.md
.venv/Scripts/python.exe -m pytest tests/unit -q 2>&1 | Select-Object -Last 5
```

Known pytest quirks (from prior sessions): use the `.venv` python explicitly;
if temp-dir errors appear, add `--basetemp=pytest_tmp_root`; `issue_report`
tests can be isolation-flaky — rerun alone before blaming a change.

## Per-task execution loop (repeat for every spec, in order)

```bash
# 1. BUILD — Codex implements hands-off (consult -> build -> self-review -> test)
bash codex-bridge/codex_loop.sh --mode build --task <task-id> \
  --test '.venv/Scripts/python.exe -m pytest tests/unit/<test-file> -q' \
  < docs/plans/2026-07-feature-rollout/<spec-file>
# task ids: f02-moc, f09-dust, f10-dist, f04-psr, f11-sso, f06-lc, f03-rsed,
#           f08-mon, f01-vor
```

Then the REVIEW GATE (Claude or user; nothing merges without it):

```powershell
# 2. inspect the diff — check against the spec's acceptance list
git diff --stat; git diff core/agent.py services/ tests/ scripts/

# 3. offline tests (feature + full suite; compare to Phase 0 baseline)
.venv/Scripts/python.exe -m pytest tests/unit/<test-file> -q
.venv/Scripts/python.exe -m pytest tests/unit -q

# 4. LIVE smoke (expected values are in each spec's "Smoke expectations")
.venv/Scripts/python.exe scripts/smoke/smoke_<feature>.py

# 5. lazy-import check
.venv/Scripts/python.exe -c "import time,importlib; t=time.time(); importlib.import_module('services.<module>'); print(round(time.time()-t,2),'s')"
```

Review checklist (fix or explicitly waive each finding — never skip silently):
- All Codex findings from `tmp/codex/tasks/<task-id>/review.md` addressed.
- agent.py: insertions only at the 4 anchors; no reformatting; tool
  descriptions accurate; status labels + prompt bullet present.
- Service returns `success/warnings/provenance`; no raw exceptions escape;
  no live network in unit tests; lazy imports verified.
- Tick the task in `PROGRESS.md` with date + test/smoke results.

If a build fails or smoke exposes a wrong endpoint/column: re-run the SAME
task id with a short corrective spec on stdin (the bridge threads the session
so Codex remembers its previous work):
`bash codex-bridge/codex_loop.sh --mode build --task <same-id> --test '...' <<< "fix: <finding>"`

## Phase 6 — integration + cumulative review

```bash
# full suite green vs baseline
.venv/Scripts/python.exe -m pytest tests/unit -q

# have Codex hard-review the whole rollout diff over threaded rounds
bash codex-bridge/codex_loop.sh --mode guard --base beta

# token accounting across all tasks (report Codex vs Claude split)
bash codex-bridge/usage_report.sh
```

Also start the app once (`./scripts/local/start_quasar.ps1 -Restart`,
login 1@1 / 1) and ask the chat agent one question per feature (each spec's
"Chat acceptance question") to confirm the tools actually fire end-to-end.

## Execution note (2026-07-04, budget): user reports only ~9% Codex quota
left. Allocation: let F03's in-flight build finish (sunk cost); Claude builds
F08 + F01 SOLO; the remaining Codex budget is RESERVED for the final
cumulative `--mode guard` review (cheapest, highest-value: cross-model review
of Claude's solo code — it caught 3 real bugs in F11). If F03's build dies at
the limit, Claude finishes it solo from whatever landed.

## Execution note (2026-07-03)

Codex hit its usage limit mid-rollout (resets Jul 4 01:13). Per the
codex-limit-fallback rule, Claude implements the REMAINING features solo
(F11, F06, F03, F08, F01), following each spec exactly. F02/F09/F10/F04 were
Codex-built + Claude-gated. Same review discipline applies to solo builds:
unit tests + live smoke + targeted regression before ticking PROGRESS.

## Hard rules

- No commits/pushes as part of this plan; the user decides when.
- Each feature is independent: a failed task blocks nothing else — skip and
  continue, record the failure in PROGRESS.md.
- Specs > improvisation: if reality contradicts a spec (endpoint moved,
  column renamed), fix the SPEC file too so it stays the source of truth.
