# Quasar — Improvements Backlog

> Living tracker of outstanding fixes/improvements for Quasar. Compiled 2026-07-06 by
> reconciling `docs/plans/2026-07-feature-rollout/PROGRESS.md`, the 2026-07-05 live-test
> handoff (`tmp/HANDOFF-2026-07-05-live-test-continuation.md`), the newest fix log
> (`tmp/live-retest-2026-07-05-fixbatch2.md`), and the current working tree.
>
> **Context that shifted the landscape:** the deepseek mid-turn truncation that was the #1
> blocker in the handoff is **fixed and verified** — root cause was a hardcoded
> `AgentConfig.max_tokens=2000` + a cp1252 stdout crash on non-Latin-1 chars, NOT a provider
> clamp (see memory `deepseek-truncation-root-cause`). Fixes: `LLM_MAX_OUTPUT_TOKENS` (16384) +
> `DEEPSEEK_MAX_OUTPUT_TOKENS` clamp + UTF-8 stdio + finish_reason logging + ≤2 auto-continuations.

## Legend
- `[ ]` open · `[x]` done (with date + note) · `[~]` partial / needs live verification
- **Evidence rule (user directive):** for the live DataLabBench cells, only *live UI runs with
  visually verified artifacts* count — unit tests are not accepted as evidence there. For
  library-level fixes (this file's Tier 2/6 code items), offline unit tests + code review are the
  appropriate evidence.

---

## Tier 1 — Live DataLabBench residuals (blocking a clean 15/15 sweep + the publication run)

- [ ] **T1.1 — deepseek P6: WD HR diagram misidentifies the main sequence as the WD cooling track.**
  The `M_G∈[8,16]` + PM sample has no color/locus cut, so it's dominated by the lower main
  sequence / M-dwarfs; the true WD track (BP−RP 0–1.2) is sparse. Needs a WD-locus color cut
  applied and honest prose. `core/agent.py`. *(verified science flaw, fixbatch2)*
- [ ] **T1.2 — gpt-oss P9: on-sky map plots the row-capped (limit=5000) crossmatch**, whose q3c
  ordering slices the sample to a SW corner that doesn't contain Pal 5. Prompt-rule fix applied
  ("plot the full-cone PM+CMD select, not a row-capped crossmatch; only claim the cluster is shown
  if its center is inside the plotted range") but **not yet live-verified** (session-restart
  instability during fixbatch2).
- [ ] **T1.3 — gpt-oss P8 and deepseek P15 not live-re-verified.** Code paths verified on sibling
  runs (deepseek P8 tiling PASSed; P15 job-poll-stop is logic-verified only), but the actual cells
  were killed by backend restarts. Need clean live re-runs.

## Tier 2 — Product gaps that cause the remaining PARTIALs

- [x] **T2.1 — `sky_density_map` log-scale option.** *(Already implemented — verified in code
  2026-07-06.)* `services/datalab_analysis.py::sky_density_map` has `log_scale: bool = True` with a
  `_log_norm` LogNorm path and a "(log scale)" colorbar label; exposed through the agent tool
  (`core/agent.py` schema + `_datalab_sky_density_map` wrapper). Remaining: confirm live that the
  P8-class answer's "log(source count)" claim now matches the rendered colorbar.
- [x] **T2.2 — `datalab_period_fold` folds mixed bands.** *(Fixed 2026-07-06 — this session.)*
  Added optional `band` + `band_col` (default `filter`) params to
  `services/datalab_analysis.py::period_fold`, the `core/agent.py` wrapper, and the tool schema;
  folding now restricts to a single band when requested and rejects ±99/99.99 sentinel magnitudes
  before the Lomb-Scargle search. Per adversarial review, a requested band against a frame with no
  band column now **raises** (instead of silently folding all bands and looking successful). Unit
  tests added in `tests/unit/test_datalab_p1.py` (single-band, case-insensitivity, empty-band,
  absent-column, sentinel rejection).
- [x] **T2.3 — HTML-escaped tool args in plot titles.** *(Fixed 2026-07-06 — this session.)*
  deepseek emitted `g&lt;18` in a plot title (rendered literally by matplotlib). Added a
  `_clean_label` helper (`html.unescape` + strip) applied to the `title` of every Data Lab plotting
  function in `services/datalab_analysis.py`. Unit test added.
- [~] **T2.4 — Tiled density maps have tile-seam gaps.** Partial coverage from skipped/failed tiles
  renders as circular footprints with holes. Honest warnings surface it, but a gap-free mesh render
  is a deeper change (logged as a known limitation, not yet built). `services/datalab_orchestration.py`.
- [ ] **T2.5 — Dual-source scaffold leak.** The "📚 ALMA Documentation" disclaimer + forced "🌐 Web"
  section still bleed onto pure Data Lab / imagery answers (gpt-oss P4/P5/P6/P8/P9). Gate the
  DUAL-SOURCE response structure (in `_build_system_prompt`) and `_synthesize_web_summary` appending
  on whether rag/web results actually exist for *this* query type. `core/agent.py`.
- [x] **T2.6 — `value_cuts` reject `==` and string values.** *(Fixed 2026-07-06 — this session.)*
  ⚠️ Memory (`live-mcp-prompt-test-2026-07` fixbatch2) claimed this was already done, but an audit
  found it was **not in the `beta` tree** (may have lived only in an unmerged worktree). Now
  `services/datalab_query_builders.py` normalizes operator aliases (`==`→`=`, `<>`→`!=`) and renders
  string values as quote-escaped SQL literals for `=`/`!=` (the P13 `type = 'GALAXY'` case), while
  inequality ops still require numbers. Numeric strings like `'169'` stay numeric; single quotes are
  escaped (injection guard). Unit tests in `tests/unit/test_datalab_p2.py`.
- [x] **T2.7 — `datalab_density_vetting` null-coercion (full).** *(Fixed 2026-07-06 — this session.)*
  ⚠️ Memory claimed done; audit found only `ra`/`dec` were null-safe. Now `radius_deg`/`step_deg`/
  `top_n`/`fov_deg` also fall back to their documented defaults on explicit `null` (was crashing on
  `float(None)`/`int(None)`), matching the color-magnitude/color-color handlers. `core/agent.py`.

## Tier 3 — Infra / config (unblocks Tier 1/2)

- [ ] **T3.1 — Set a real `DATALAB_TOKEN` in `.env`** *(user action — free NOIRLab Data Lab
  account).* The anonymous token 401s on `/status`, so the async-job path can't run; a real token
  converts heavy P8-class aggregates from tile/subsample to real results on both models. Code
  already handles it (`integrations/datalab_client.py`).
- [~] **T3.2 — Aladin viewer hang — durable fix deferred.** Mitigated with CDN preconnect +
  35s timeout + one cache-busting retry in `ui-pro/src/lib/aladin-loader.ts`; the durable fix is
  self-hosting the Aladin bundle instead of loading from `aladin.cds.unistra.fr`.

## Tier 4 — Feature-rollout waivers (F01–F11 shipped; these flags stay open)

- [ ] **T4.1 — F03 radio SED live acceptance waived.** gpt-oss answers the spectral-index question
  from parametric knowledge without calling the tool (2/2 runs). Needs a tool-forcing heuristic or
  a healthier model to close. Wiring is live-proven at gate smoke (3C 273 with honesty flags).
- [ ] **T4.2 — `vo_find_services` breaks if pyvo is too old.** `services/vo_registry.py:175` uses
  `registry.Freetext`; surfaced live during P6 as `module 'pyvo.registry' has no attribute
  'Freetext'`. Verify/pin the pyvo version in the conda `quasar` env (the env the live app runs in).
- [ ] **T4.3 — VizieR TAP_SCHEMA `list_tables`/`describe_table` live re-verify waived** (VizieR was
  stalling at gate time; same transport as the live-proven ADQL path; logic unit-pinned).

## Tier 5 — Spectral Line Explorer / Splatalogue deeper science
> ⚠️ Sourced from the `splatalogue-feature-state` memory (was ~13 days old at compile time).
> Re-verify each against current code before acting — some may already be addressed.

- [ ] **T5.1 — Confusion score is catalog-proximity only** (no excitation / abundance / Einstein-A /
  optical-depth weighting). `services/spectral_line_explorer.py`.
- [ ] **T5.2 — Job durability:** jobs use an in-process `ThreadPoolExecutor` + DiskCache — a restart
  orphans running jobs. Needs a durable job store or restart-recovery.
- [ ] **T5.3 — FITS Workbench cube-freq-range → Splatalogue integration missing** — `line_overlays`
  does a ±0.01 GHz pinprick instead of calling `search_spectral_lines`.
- [ ] **T5.4 — "CASA export" is a plain-Python snippet, not a real CASA table/script.**

## Tier 6 — Enhancements / deferred

- [ ] **T6.1 — Extend interactive Plotly to the remaining plot tools:** `sed_plot`, `lss_wedge`
  (polar wedge), `sky_density_map` (heatmap trace), `catalog_scatter`, `period_fold` — currently
  static PNGs (they do get the lightbox). The SSE/frontend path (`PlotlyCard.tsx`, `type:"plotly"`)
  is already generic; only the services need to emit specs.
- [ ] **T6.2 — Strip inline `![](/plots/...)` markdown image URLs** the models sometimes embed in
  answer tables (violates the NO IMAGE URLS prompt rule; relative URLs may not resolve from the
  frontend origin). Either strengthen the prompt rule or post-strip inline image markdown.

---

## Terminal goal
Once Tier 1–2 clear, run `Benchmark/datalabbench/run_datalabbench.py` for the publication numbers
(the improve-until-100 loop). Auto-credit needs the SSE `tool_trace` event (works). Never edit
rubrics silently — bump `BENCH_VERSION`.

## Suggested sequencing
1. **T3.1** (a real DATALAB_TOKEN unblocks the most cells for free).
2. **T1.1 / T1.2** + the completed T2.2 (fixes that directly convert PARTIAL→PASS).
3. Live re-sweep (**T1.3**), then the DataLabBench run.
4. Tiers 4–6 are lower-stakes cleanup.

## Change log
- **2026-07-06** — file created. Completed **T2.2** (period_fold band filter + sentinel guard +
  absent-column guard), **T2.3** (HTML-unescape plot titles), **T2.6** (`value_cuts` `==`/string
  support), and **T2.7** (density_vetting null-coercion); confirmed **T2.1** (sky_density_map
  log-scale) already implemented in-tree. All changes reviewed by a 3-lens adversarial workflow
  (correctness/science/wiring) + a done-claims audit, which surfaced T2.6/T2.7 as stale "done"
  claims in memory that were never actually on `beta`. New/changed unit tests all green; the only
  failing tests in `test_datalab_p1.py` are the pre-existing `color_image`/`cutout` image-service
  failures (confirmed identical on committed `beta` with these changes stashed).
