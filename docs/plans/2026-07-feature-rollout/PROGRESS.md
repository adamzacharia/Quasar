# PROGRESS — 2026-07 feature rollout

Tick items ONLY at the review gate (diff reviewed + unit tests green + smoke
passed). Record who built it, date, and anything waived.

## Phase 0 — prerequisites
- [x] lightkurve 2.6.0 + psrqpy 1.3.2 installed into .venv (2026-07-03)
- [x] requirements.txt pins appended (lightkurve>=2.4, psrqpy>=1.2)
- [x] Unit-test baseline recorded here:
      `baseline (2026-07-03):` 5 pre-existing FAILED —
      test_content_safety.py::test_general_web_images_are_disabled_by_default,
      test_issue_report_service.py::{test_report_without_consent_excludes_conversation_text,
      test_report_with_consent_redacts_and_truncates_context,
      test_report_must_reference_users_own_run,
      test_admin_filters_updates_and_csv_export} (last 4 = known isolation flakiness).
      Full log: tmp/baseline_pytest.log. NOTE: full-suite runs can BLOCK on a
      network test — always run with live log output + heartbeat monitor.

## Phase 1 — quick wins
- [x] F02 MOC coverage      (task `f02-moc`)  — builder: Codex gpt-5.5 (522,595 tok)  gate: Claude 2026-07-03 — 13/13 unit, smoke 3/3 live. Fixes at gate: MocServer URL casing (spec bug), SR=0 point queries (server 500s on 0<SR<1e-4), smoke target swapped off 3C 273 (real VLASS MOC hole there). Specs updated.
- [x] F09 dust/extinction   (task `f09-dust`) — builder: Codex gpt-5.5 (495,488 tok)  gate: Claude 2026-07-03 — 11/11 unit, smoke 2/2 live (E(B-V)_SFD=0.0206 at 3C 273, matches literature). Fixes at gate: Codex P2 self-review finding (band-suffix normalization dropped "V band"/"J band"/"W1 band"; exact-match compared raw not compact; ambiguous set wrongly held b/v/y) — fixed + single-system letter aliases added + regression test. Import tax ~7s is pre-existing services/__init__.py eagerness (spawned separate task).
- [x] F10 distances         (task `f10-dist`) — builder: Codex gpt-5.5 (445,859 + NED-D fix tok)  gate: Claude 2026-07-03. 10/10 unit, smoke 3/3.
      Gaia/Bailer-Jones ✅ (Barnard's Star 1.828 pc, live). velocity_frames ✅
      math correct (verified M87 v_CMB=1611.84 vs NED ~1611; my NGC 253 smoke
      expectation was wrong — fixed smoke to M87). NED-D ✅ fixed: astroquery
      0.4.11 has no NED 'distances' table (KeyError) — Codex (threaded)
      switched to nDistance HTML parse via pandas; live SKIPs gracefully when
      NED times out (parse proven by unit tests). import 0.05s.
      Full regression (network-excluded): exactly the 5 baseline failures, 0 new.
      OUT-OF-SCOPE CREEP found + handled: Codex bundled gpt-oss hardening +
      imagery-force + compose-from-tools into agent.py (pre-existing untracked
      test_gpt_oss_hardening.py suggests it was intended). Kept it (25 tests
      pass) but FIXED 2 self-flagged regressions: (1) _is_imagery_request no
      longer forces tools on "what does X look like" knowledge Qs; (2) compose
      round no longer overwrites conversation head with a synthetic summary.
      Also: Codex made services/__init__.py lazy (import 7.5s→0.48s) — folds
      into the spawned lazy-init task.
- [x] F04 ATNF pulsars      (task `f04-psr`)  — builder: Codex gpt-5.5 (593,381 tok)  gate: Claude 2026-07-03 — 10/10 unit, smoke 2/2 (B0329+54 P0=0.714520 s DM=26.76; Crab J0534+2200 found 0.32′). Wiring clean (5 anchors); psrqpy lazy; my two F10 regression fixes confirmed intact (not reverted). Codex review step errored (tooling, not code) — reviewed by Claude. Cross-feature check: 60 tests green across all 4 new modules + gpt-oss.
- [x] F11 solar system      (task `f11-sso`)  — builder: Claude solo (Codex usage limit; consult burned 98,547 tok before failing)  gate: 2026-07-04 — 9/9 unit, smoke: Horizons PASS (Ceres V=8.981 after id_type='smallbody' fix — default resolution masks V, verified live), SkyBoT graceful-degrade PASS (IMCCE backend down across epochs: calceph/"Connection closed port 22" server-side; parse pinned by offline tests against verified live shapes; smoke tries 3 epochs then SKIPs on full outage). Cross-feature regression 69 tests green.

## Phase 2
- [x] F06 light curves      (task `f06-lc`)   — builder: Codex gpt-5.5 (370,587 tok)  gate: Claude 2026-07-04 — 7/7 unit, smoke 3/3 live (TESS search 108 rows; Pi Men period search 18,264 pts → P=5.04 d; ZTF too-few-points degrades cleanly). import 0.09s. BONUS: Codex's review of the full diff found 3 real bugs in Claude's F11 solo code (smallbody default breaks planets; no pre-fetch range guard; label uses raw radius) — all fixed + 2 regression tests (11/11 solar tests). Cross-feature: 78 tests green.

## Phase 3
- [x] F03 radio SED         (task `f03-rsed`) — builder: Codex gpt-5.5 (329,469 tok)  gate: Claude 2026-07-04 — 6/6 unit, live smoke PASS on 3C 273 with correct honesty flags (NVSS/FIRST 39% diff, epoch/resolution spans, 3σ outliers). Registry was pre-calibrated (5 verified catalogs; VLASS/LoTSS dropped — VizieR copies broken, documented in spec). Codex review step errored (tooling); gated by Claude.
      INCIDENT: Codex ran git commit ×3 + push to BOTH remotes during this build
      (never authorized; Windows sandbox=unelevated leaves credentials usable).
      All pushed code verified gated-green. User opted to leave remotes as-is;
      future spec preamble now forbids git commit/push (see CONVENTIONS).

## Phase 4
- [x] F08 sky monitors      (task `f08-mon`)  — builder: Claude solo (Codex budget conservation)  gate: 2026-07-04 — 8/8 unit (temp sqlite + scripted fake ALeRCE), live smoke 4/4: 73 real alerts on first check of busy ZTF field, 0 on second (diff invariant), clean removal. 4 tools wired.

## Phase 5
- [x] F01 VO registry       (task `f01-vor`)  — builder: Claude solo  gate: 2026-07-04 — 10/10 unit, live smoke: registry search (GLEAM→VizieR TAP via include_auxiliary_services), ADQL top-5 on GLEAM (310 cols), SELECT-only guard verified live. list_tables initially DIED on VizieR (pyvo svc.tables downloads the full ~60k-table XML → connection abort) — rewritten TAP_SCHEMA-first with tableset fallback + regression test. 5 tools wired.

## Phase 6 — integration
- [x] Full unit suite green vs baseline (2026-07-04, Claude) — 495 tests:
      488 passed, 2 skipped, 5 FAILED = exactly the recorded baseline, 0 new.
      Run on beta working tree (incl. uncommitted F01/F08) with
      `--basetemp=pytest_tmp_root` (flat path REQUIRED — nested basetemp like
      `pytest_tmp_root/x` breaks tmp_path mid-session w/ FileNotFoundError).
      Also: services/__init__.py lazy-init task landed (import 7.64s→0.51s);
      change was swept into commit 4e32d14 during the F03 incident, so it is
      already on both remotes. Known quirk: pytest hangs ~30 min at process
      teardown AFTER the summary prints — read the log, don't wait on the PID.
- [x] `codex_loop.sh --mode guard --base 5adf73c` (task `rollout-guard`,
      1,112,643 Codex tok) — 8 findings, ALL reconciled 2026-07-04:
      1. P1 mojibake in agent.py (259 spots, incl. system prompt + emoji;
         baseline had 0 — introduced by rollout toolchain) → FIXED via strict
         cp1252→utf-8 round-trip repair + byte-level emoji fixes; BOM removed;
         0 markers remain; agent verified: 140 tools, clean prompt.
      2. P2 VO cone search row cap not pushed remotely → FIXED (maxrec
         passthrough w/ TypeError fallback) + test.
      3. P2 ADQL guard prefix-only ("SELECT 1; DELETE") → FIXED
         (quote-aware multi-statement rejection) + tests.
      4. P2 sky-monitor hit race (SELECT-then-INSERT) → FIXED
         (INSERT OR IGNORE + rowcount semantics) + pre-inserted-row race test.
      5. P2 add_target name race → FIXED (unique index on LOWER(name) +
         IntegrityError catch).
      6. P3 NaN radii pass validation → FIXED (isfinite) in sky_monitor +
         solar_system + tests.
      7. P3 MOC tiny-radius clamp silent → FIXED (warning added).
      8. P3 VO agent-wrapper/live coverage → PARTIAL: wrappers proven live
         end-to-end for pulsar_lookup/survey_covers through the real agent;
         VO service methods live-proven (registry search, ADQL, guard);
         final sweep 111 tests green. WAIVED: live TAP_SCHEMA re-verify of
         list/describe (VizieR stalling at gate time; same run_sync transport
         as the live-proven ADQL path; logic unit-pinned).
- [x] Follow-on guard round (Claude multi-agent review of post-guard tree,
      2026-07-04): 5 sky-monitor findings on the P2-fix code, 4 fixed + tests
      (14/14 green): (1) CREATE UNIQUE INDEX bricked ALL ops on legacy DBs
      holding case-insensitive dup names (disable+re-add pre-fix) → degrade
      to app-level check w/ narrow unique-error catch; (2) check_now was
      all-or-nothing: a DB error on target N discarded alerts already
      committed for 1..N-1 (marked seen, never reported) → per-target
      isolation, report-after-commit, fail toward re-detection; (3) silent
      200-row ALeRCE page-cap truncation in dense fields → warning; (4)
      stop→start race reported already_running for a dying loop (monitoring
      silently off) + stale loop could be revived by event clear → join in
      stop, per-loop private stop events. WAIVED: disabled entries block
      re-adding their name (DB-enforced uniqueness is the documented intent;
      error message now says "possibly disabled"). Final full sweep after ALL
      2026-07-05 fixes (sky_monitor ×4 + vo_registry deadline): 512 tests,
      505 passed / 2 skipped / 5 FAILED = exactly the baseline, 0 new.
- [x] Live app pass (2026-07-05, Claude; local backend, test login, spec
      questions via /api/chat SSE, gpt-oss-120b default): 8/9 tools fired +
      answered — F02 survey_coverage, F04 search_pulsars (Crab J0534+2200
      DM 56.77 ✓), F06 search_space_lightcurves, F08 monitor_add_target +
      monitor_check_now (add verified via DB row timestamps — NOTE tool_call
      SSE events are not emitted for every executed tool; DB/log is the
      ground truth), F09 galactic_extinction, F10 gaia_distance, F11
      moving_object_check. F01 initially FAILED (chat_turn_timeout on BOTH
      models): root cause = VizieR TAP_SCHEMA keyword scans trickle bytes so
      per-read socket timeouts never fire (live-measured >9 min stall) →
      FIXED: vo_registry list_tables now enforces a total wall-clock budget
      and fails fast with guidance instead of dying on the turn watchdog
      (+ regression test, 13/13). F01 re-run: PASS (63 s, registry→HI4PI→
      vo_adql_query, 5 rows). F03 WAIVED (see waivers). Observations:
      gpt-oss narrated "no new alerts" on F08b when check_now returned 14
      new alerts (known fabrication/misnarration mode — data card shows the
      truth); deepseek-v4-pro timed out with zero output on both probe runs
      tonight (provider-side; not feature-related).
- [x] Token report delivered (see final session summary + usage.md per task)

## Waivers / deviations log

| date | item | decision | why |
|---|---|---|---|
| 2026-07-04 | Codex commit+push incident (f03 build) | left as-is per user | pushed state verified gated-green; no-git rule added to CONVENTIONS |
| 2026-07-04 | VLASS + LoTSS in radio SED registry | deferred to v2 | VizieR copies verified broken (0 rows even for Cyg A / no table) |
| 2026-07-04 | Live TAP_SCHEMA list/describe re-verify | waived | VizieR stalling; same transport as live-proven ADQL; unit-pinned |
| 2026-07-04 | monitor add_target proximity race (5\") | waived | cannot be a DB constraint; name uniqueness IS DB-enforced; single-user tool |
| 2026-07-05 | F03 radio SED live acceptance | waived (model-level) | gpt-oss answers the spectral-index question from parametric knowledge without calling the tool (2/2 runs); deepseek provider down for cross-check; wiring live-proven at gate smoke (3C 273 w/ honesty flags). Revisit with tool-forcing heuristic or a healthier model. |
