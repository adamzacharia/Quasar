# FILE_AND_CODE_MAP.md

Map of every relevant file/folder for V2 work. **Modify?** column: KEEP (don't touch), REFACTOR (change in place), EXTRACT (move logic to `capabilities/`), SPLIT (break up), DELETE, NEW (create). Line numbers marked **(v)** are verified 2026-07-08; others are **(approx — verify with `grep -n`)**.

## A. The monolith targets (highest priority)

| Path | Lines | What it does | Modify? | Notes |
|---|---|---|---|---|
| `core/agent.py` | **12,714 (v)** | God object: 145 tool regs + impls + streaming loop + conductor executor + sandbox bridge + routing + RAG gate + prompts + thread-local state + legacy CLI | **SPLIT/EXTRACT** | The #1 target. Tool impls → `capabilities/`; `stream_response_api` → `core/runner.py`; regexes → `core/router.py`. Duplicate `_filter_results` at **8397 (v)** and **10147 (v)**. MCP bridge `new_event_loop` at **522–523 (v)** & **11671–11674 (v)**. `inputSchema→Tool.parameters` map ~497–536 (approx). Thread-local `last_search_results` ~602–608 (approx). |
| `ui-pro/api/main.py` | **4,368 (v)** | FastAPI: routing + SSE state machine + UI column maps + CADC XML + matplotlib FITS + quotas + jobs + channels | **SPLIT** | Into `api/routers/*` + `api/sse.py` + `api/serializers/`. `/api/mcp-servers` routes at **2683–2702 (v)**. Column maps ~924–1027 (approx). |

## B. Core orchestration (`core/`) — 28 files

| Path | Modify? | Notes |
|---|---|---|
| `core/task_dag.py` | **KEEP** | Clean DAG: Kahn cycle detection, adaptive SLAs. Carry forward as-is. |
| `core/complexity.py` | **KEEP (retune)** | Two-stage gate. Keep concept; retune keyword lists/threshold. |
| `core/tools.py` | **REFACTOR (extend)** | Tool dataclass + ToolRegistry. Add namespacing, validation, collision protection. Becomes `core/tool_registry.py`. |
| `core/conductor.py` | **REFACTOR** | Shared singleton → request-scoped `OrchestrationRun`. Planner whitelist (~72–97) → generate from live registry. `_trace_id` NameError (~721, 813). |
| `core/recovery.py` | **REFACTOR** | Fix executor-bypass (route through full node executor + SLA). `_is_empty_result` ~351–364 treats 0 rows as failure. |
| `core/sandbox.py` | **REFACTOR → compute-mcp** | In-process `exec()` + regex denylist — not a boundary, and unreachable in prod. Move to isolated worker. |
| `core/llm_client.py` | **KEEP** | Multi-provider Responses shim (OpenAI/Anthropic/Google/DeepSeek/TACC/local). |
| `core/model_router.py` | **KEEP** | agent_type→model routing. |
| `core/retry.py` | **KEEP** | Backoff/jitter decorators. |
| `core/token_budget.py`, `tool_budget.py` | **KEEP** | Budget + truncation. Consolidate the 4 magic truncation numbers. |
| `core/workflow_memory.py` | **KEEP (fix)** | Thread-safe blackboard; swap char-truncation for typed handles. |
| `core/dag_cache.py` | **REFACTOR** | Wrong-target replay bug (normalizes target, reuses subtasks verbatim). |
| `core/observability.py`, `langfuse_integration.py`, `logger.py` | **KEEP** | Tracing/logging. |
| `core/memory.py` | **REFACTOR** | ConversationMemory sliding window (keep short-term); fold long-term into one memory service. |
| `core/agent_pool.py` | **DELETE** | Instantiated, never used. |
| `core/workflow_store.py` | **DELETE** | Checkpoint/resume; imported nowhere. |
| `core/result_cache.py` | **DELETE** | Instantiated, never consulted (get/put never called). |
| `core/context_manager.py` | **DELETE** | `compact_if_needed` never called. |
| `core/health_monitor.py` | **DELETE or WIRE** | `record_success/failure` never called → fallback can't trigger. |
| `core/model_council.py` | **DELETE or WIRE** | Multi-model consensus; imported nowhere. Candidate to revive for consensus planning (later). |
| `core/session_memory.py` | **DELETE** | Runs LLM calls per turn; `get_context()` has zero call sites. |
| `core/cli.py`, `doc_updater.py` | **REFACTOR/KEEP** | Legacy CLI → route through Runner. |

## C. Business logic (`services/`) — 74 files

| Path | Modify? | Notes |
|---|---|---|
| `services/datalab_sql_policy.py` | **KEEP → into capability + MANNA** | Best safety code in the repo. Body of `capabilities/datalab.py`; also runs server-side in MANNA. |
| `services/datalab_query_builders.py` | **KEEP → capability** | Pure SQL builders. |
| `services/datalab_registry.py` | **KEEP → capability** | Hand-verified catalog registry. |
| `services/splatalogue.py` | **KEEP → capability** | astroquery→SLAP fallback; cache-only-non-degraded. |
| `services/vo_registry.py` | **KEEP → capability** | Timeout-bound TAP, ADQL guards. |
| `services/citation_verifier.py`, `provenance.py`, `evidence_quality.py` | **KEEP (native)** | Hallucination controls — stay on Quasar's trust boundary. |
| `services/search.py` | **REFACTOR** | Swallow→empty-DataFrame (~74–108) → typed errors. `check_line_coverage` is a `pass` stub. |
| `services/analysis.py` | **REFACTOR** | Mock UV results (~23–39) → typed error. |
| `services/user_tools_service.py` | **DELETE** | `exec()` RCE (~63–65, 238–244). Superseded by MCP servers. |
| `services/mcp_server_service.py` | **KEEP** | Per-user MCP registry — correct persistence shape; the seam V2 promotes. |
| `services/admin_access.py` | **REFACTOR (SEC)** | `'1@1'` in `DEFAULT_ADMIN_EMAILS` (~9–21) → remove; DB role column. |
| `services/db.py` | **REFACTOR (SCALE)** | New client per call; `commit()` no-op (no transactions). Pool + transactions + migrations. |
| `services/auth.py` | **REFACTOR (SEC)** | `==` password compare (~275) → `hmac.compare_digest`; rate limit. |
| `services/rag_service.py` | **REFACTOR** | Score-scale gate bug; `rank_bm25` missing; CYCLE_YEAR_MAP dup (~176). |
| `services/vector_db.py` | **KEEP** | Clean Qdrant abstraction. Fix `scroll_all` pagination (~206–229). |
| `services/web_search_service.py` | **REFACTOR → MCP** | 1,142-line Brave/Tavily/Exa router; quota in flat JSON. Candidate for official search MCP. |
| `services/plotting.py`, `visualization_service.py`, `sed_plotter.py`, `radio_sed.py`, `datalab_analysis.py` | **CONSOLIDATE** | 5 reimplementations → one plotting capability. |
| `services/*_client.py` (alerce, gcn, pulsar, dust, moc, hips, distance, lightcurve…) | **KEEP → capabilities** | Self-contained archive adapters; each a clean capability + MCP candidate. |
| (remaining ~40 modules) | **CASE-BY-CASE** | Regroup flat dir into `platform/` `compute/` `archive-adapters/`. See `.research/data/402`. |

## D. External adapters (`integrations/`) — 15 files

| Path | Modify? | Notes |
|---|---|---|
| `integrations/datalab_client.py` | **KEEP (template)** | Reference-quality REST adapter; provenance + async-fallback. |
| `integrations/datalab_sia_client.py` | **KEEP (template)** | Endpoint fallback chain; coverage_gap-vs-error; cos(dec) box. |
| `integrations/svo_fps_client.py` | **KEEP** | Memory+disk TTL cache pattern. |
| `integrations/tap.py` | **CONSOLIDATE** | NRAO TAP; fold into one TAP engine. `_TimeoutHTTPSession` (~20–34) — keep the technique. |
| `integrations/alminer_client.py` | **REFACTOR** | ALMA; thread-race (~98–154) leaks + non-deterministic columns. |
| `integrations/eso_tap_client.py` | **REFACTOR (SEC)** | Raw ADQL interpolation; plain `http://` endpoint (~64). |
| `integrations/ads_client.py` | **REFACTOR (COR)** | `_get_example_papers` fabricates (~414–461). Merge into one ADS client. |
| `integrations/openalex_client.py` | **KEEP** | Real retry/backoff. Verify funding field names. |
| `integrations/datalink.py`, `mast/`, `irsa`, `skyview`, `casa.py`, `carta.py` | **CASE-BY-CASE** | `carta.py` is instructions-only (stub); `casa.py` has simulated paths. See `.research/data/403`. |

## E. Frontend (`ui-pro/src/`) — Next.js, ~13.2k LOC

| Area | Modify? | Notes |
|---|---|---|
| `ui-pro/src/lib/api.ts` | **REFACTOR** | Hand-rolled SSE parsing ×2; 8 duplicated `API_BASE` fallbacks; no reconnect/resume. |
| `SettingsModal.tsx` | **REFACTOR (bug)** | 10+ escaped `${` in className template literals (active-tab highlight broken). Also hosts the MCP-servers panel. |
| `DataTableCard.tsx` | **REFACTOR** | Magic keys (`_link`, `_preview`, `Proposal ID`); no virtualization. Typed table descriptor from backend. |
| workbench page, spectral-lines page | **REFACTOR** | Fixed-interval polling → job-events channel. |
| `AladinSkyView`, `hips-imagery.js`, `sky-geometry.js` | **KEEP** | Client-side CDS calls; keep out of MCP. |

## F. Standalone MCP / bridge

| Path | Modify? | Notes |
|---|---|---|
| `C:/Users/adama/Desktop/ALMA_MCP/server.py` | **ABSORB into MANNA** | 16-intent FastMCP proof. Reuse tool taxonomy/docstrings; NOT the buggy internals (Hz/GHz, `public_only` no-op, ADQL injection, stdout `print()`). See `.research/data/409`. |
| `codex-bridge/` | **KEEP (dev-only)** | Git-ignored dev tooling; do not entangle in product V2. |

## G. Config / CI / deployment

| Path | Modify? | Notes |
|---|---|---|
| `config/settings.py`, `config/__init__.py` | **DELETE → NEW** | Dead (imported nowhere). Replace with one typed pydantic-settings schema (the only `os.getenv` consumer). **179** `os.getenv/environ` calls (v) to migrate. |
| `.github/workflows/ci.yml` | **REFACTOR** | Lints only 5 files; triggers on `beta` only (README claims `main`). |
| `keep_alive.py` + Actions cron | **REPLACE** | Render free-tier keep-alive hack; ephemeral disk data loss. |
| `requirements.txt` | **REFACTOR** | Add `rank_bm25` (missing → hybrid RAG silently no-ops). |
| `CONVENTIONS.md` | **KEEP → codify** | The `{success,warnings,provenance}` contract → base class. |
| `docs/QUASAR_V2_DESIGN.md`, `docs/QUASAR_V2_EXECUTION_PLAN.md` | **KEEP** | The full design + 7-phase plan behind this handoff. |
| `docs/ARCHITECTURE.md`, `FEATURES.md`, `PROGRESS.md` | **REFACTOR** | Drift (tool count 27+/35+/75+ vs 142). Generate reference from code. |

## H. Backing analysis (read-only evidence — outside the repo)

| Path | What |
|---|---|
| `C:/Users/adama/Desktop/France/.research/data/400_quasar_api-backend.md` | FastAPI deep analysis |
| `…/401_quasar_core*.md` | agent/conductor/registry/sandbox — **most important for the monolith** |
| `…/402_quasar_services*.md` | 74-module services analysis |
| `…/403_quasar_integrations*.md` | adapters analysis |
| `…/404_quasar_rag*.md` | RAG + memory |
| `…/405_quasar_frontend*.md` | Next.js |
| `…/406_quasar_agents*.md` | sub-agents + notebook (dead-layer findings) |
| `…/407_quasar_benchmark*.md` | eval harness |
| `…/408_quasar_docs*.md` | config/CI/deploy |
| `…/409_quasar_alma*.md` | ALMA_MCP |
| `C:/Users/adama/Desktop/France/.research/` (281 files) | Astro Data Lab KB (MANNA/archive reference) |
