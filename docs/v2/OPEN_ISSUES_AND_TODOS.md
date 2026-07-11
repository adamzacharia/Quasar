# OPEN_ISSUES_AND_TODOS.md

## P0 resolution status (2026-07-09)

Landed this session (see `STATUS.md` change log; regression tests in `tests/unit/test_p0_stabilization.py`):

| ID | Status | Note |
|---|---|---|
| C1 | ✅ FIXED | `analysis.py` → typed `casa_unavailable` error; agent wrapper reflects `success=False`. |
| C2 | ✅ FIXED | `_get_example_papers` deleted; `search_papers` raises `ADSServiceError`. |
| C3 | 🟡 MOSTLY | `search.py` tags failed searches via `.attrs["quasar_error"]`; `search_by_target` (single/multi/per-band) + `_search_by_position` surface it. Frequency/keyword/source handlers get full typed errors with the P1 `ToolResult` migration. |
| C5 | ✅ FIXED | RAG gate reads `_semantic_score` (0–1), not RRF `_score` (~0.016). |
| C6 | ✅ FIXED | `rank_bm25` added to `requirements.txt`. |
| C7 | ✅ FIXED | Recovery routes through `_execute_node`; SLA enforced. ⚠️ Conductor path not benchmark-covered — needs a complex-query smoke test (P4). |
| C9 | ✅ FIXED | DAGCache re-substitutes the target (safety net: skip cache if unclean). |
| C11 | ✅ FIXED | Dead shadowed `_filter_results` (agent.py:8397) deleted. |
| S1 | ✅ GATED | `exec()` user tools behind `QUASAR_ENABLE_USER_TOOL_EXEC` (default OFF). Deletion is P3. |
| S2 | ✅ GATED | stdio MCP spawn behind `QUASAR_ENABLE_MCP_STDIO` (default OFF). http/SSE still allowed. |
| S3 | ✅ FIXED | Hard-coded `1@1` admin removed (`DEFAULT_ADMIN_EMAILS` now `set()`); admin via `ADMIN_EMAILS` env **or** `users.role='admin'` column (`is_admin_email` DB-aware, TTL-cached). Local seed carries `role='admin'`. Tests: `test_s3_s6_security.py`. |
| S6 | ✅ FIXED | Password `==` → `hmac.compare_digest`; login rate limiting (per-user + per-IP) with **429 + Retry-After** (CORS-exposed). httpOnly-cookie migration **DONE + browser-verified**: backend sets/reads/clears the cookie (Bearer still works); frontend no longer persists the JWT (server-authoritative `/me` bootstrap + auth-generation guard), 54 fetches `credentials:'include'`, ~20 `!token` guards → `isAuthenticated`. **Cookie auth's CSRF surface closed in the same change** by `CookieCsrfMiddleware` (Origin check on cookie-authed writes; negative-control proved a cross-site `POST /api/chat/upload` returned 200 before it). `tsc` 0, `test_s3_s6_security.py` 36✓ (61✓ across suites). |

Still open from the tables below: C4, C8, C10, C12–C17, and all dead-code / testing / design-question items. (S3–S6 all resolved; S5 by the api-split track, S6 frontend flip by the Fable 5 session.)

**Newly discovered (during the P0 adversarial review) — not yet fixed:**
- 🟡 `_search_by_target` positional fallback is dead: gate reads `resolved.get("ra")` but `_resolve_target` returns `ra_deg`/`dec_deg` (`core/agent.py` ~5480). Enabling it is a behavior change → defer (not P0).
- 🔵 `services/ads_service.py` is a **dead duplicate** of `integrations/ads_client.py` (zero imports); fabrication neutralized here too, but the whole module should be deleted in P5 (one ADS client).
- 🟡 C7 SLA `asyncio.wait_for` cannot cancel the blocking `run_in_executor` tool thread (thread runs to completion, result discarded). Pre-existing on the non-recovery path; inherent until compute moves to a cancellable worker (P2/P3). Reviewed and accepted as a known limitation.

Unresolved issues from the analysis. **Severity:** 🔴 blocker/safety, 🟠 correctness, 🟡 quality, 🔵 design question. Line numbers **(v)** verified 2026-07-08; others **(approx)** — verify with `grep -n` before acting. Source column points to the analysis file with full detail.

## Safety / security (do in P0)

| # | Sev | Issue | Where | Verify / Fix | Source |
|---|---|---|---|---|---|
| S1 | 🔴 | `exec()` of user Python with full `__builtins__`+`os` in the shared process (RCE) | `services/user_tools_service.py` ~63–65, 238–244 | `grep -n "exec(" services/user_tools_service.py`; flag OFF in prod, delete in P3 | 402, 406 |
| S2 | 🔴 | `/api/mcp-servers` persists `{command,args}` and spawns it via stdio (arbitrary command exec) | `ui-pro/api/main.py:2689 (v)` | allowlist commands or restrict to HTTP/SSE URLs | 400, 406 |
| S3 | ✅ | ~~`'1@1'` compiled into `DEFAULT_ADMIN_EMAILS`, always admin~~ **FIXED** — env allowlist + DB `role` column | `services/admin_access.py`, `services/auth.py` | done (TTL-cached DB lookup; local seed `role='admin'`) | 402 |
| S4 | 🟠 | CORS wildcard `*.vercel.app`/`*.onrender.com` with `allow_credentials=True` | `main.py:181 (approx)` | explicit origin allowlist | 400 |
| S5 | 🟠 | Unauthenticated `/api/conductor/traces*`, `/api/proposals/review` | `main.py` ~4315–4368 | add auth deps | 400 |
| S6 | ✅ | ~~JWT in `localStorage` (XSS-exfiltratable); password `==` compare~~ **FIXED** | frontend store; `services/auth.py` | `hmac.compare_digest` ✅ + rate limit (429) ✅; httpOnly cookie **DONE + browser-verified** ✅ (JWT out of localStorage; 54 fetches credentialed; guards→isAuthenticated) | 400, 402, 405 |

## Correctness — fabrication (do in P0)

| # | Sev | Issue | Where | Fix | Source |
|---|---|---|---|---|---|
| C1 | 🔴 | `analyze_uv_coverage` returns hardcoded mock results when CASA absent | `services/analysis.py` ~23–39 | typed "CASA not installed" error | 402 |
| C2 | 🔴 | ADS `_get_example_papers` returns invented bibcodes/DOIs on missing key OR any exception | `integrations/ads_client.py` ~414–461 | typed ADS-unavailable error | 403 |
| C3 | 🔴 | `search.py` swallows all exceptions → empty DataFrame → "archive down" indistinguishable from "no data" | `services/search.py` ~74–108 | `ToolResult(success=False, error/degraded)` | 402 |
| C4 | 🟠 | Other canned-data paths: `data_processor.fetch_source_data`, `casa.py` simulated calibration/imaging | services | audit + typed errors or delete | 402, 403 |

## Correctness — logic bugs

| # | Sev | Issue | Where | Fix | Source |
|---|---|---|---|---|---|
| C5 | 🟠 | RAG score-scale gate drops valid docs: filter `>= 0.15` vs RRF `_score` capped ~0.017 | `agent.py` ~11311–11318 | normalize to one scale; add gate unit test | 404 |
| C6 | 🟠 | `rank_bm25` not in `requirements.txt` → hybrid search silently no-ops | `requirements.txt`, `rag_service.py:1066` | add dep or drop claim | 404 |
| C7 | 🟠 | Recovery path bypasses routing/sandbox/SLA (bare `run_in_executor` lambda) | `core/recovery.py`, `conductor.py` | route through full node executor + `wait_for(sla)` | 401, 406 |
| C8 | 🟠 | `_is_empty_result` treats 0 rows as failure → REPLAN rewrites valid "none found" answers | `core/recovery.py` ~351–364 | gate REPLAN on meaningfulness | 401, 406 |
| C9 | 🟠 | DAGCache wrong-target replay (normalizes `{TARGET}`, reuses subtasks verbatim) | `core/dag_cache.py` ~96–101; `conductor.py` ~407–414 | key on raw target or re-substitute | 406 |
| C10 | 🟠 | MCP bridge cross-loop asyncio (new loop per call vs loop-bound session) | `agent.py:522–523 (v)`, `11671–11674 (v)` | one long-lived loop + `run_coroutine_threadsafe` | 401, 406, 409 |
| C11 | 🟡 | `_filter_results` defined twice | `agent.py:8397 (v)`, `10147 (v)` | delete one during extraction | 401 |
| C12 | 🟡 | Conductor shared singleton → concurrent complex queries corrupt each other | `core/conductor.py` ~401 | request-scoped `OrchestrationRun` | 401, 406 |
| C13 | 🟡 | `_trace_id` NameError swallows all DAG telemetry | `conductor.py` ~721, 813 | fix scope | 401, 406 |
| C14 | 🟡 | ALminer thread-race leaks daemon thread; non-deterministic columns | `alminer_client.py` ~98–154 | primary + explicit fallback + cancel | 403 |
| C15 | 🟡 | `vector_db.scroll_all` ignores `next_offset` → silent truncation | `services/vector_db.py` ~206–229 | loop on offset | 404 |
| C16 | 🟡 | SIMBAD `lru_cache` caches `None` forever (transient outage poisons target) | `alminer_client.py` ~30–49 | TTL cache | 403 |
| C17 | 🟡 | `db.py` `commit()` is a no-op → no transactions (auth/conversation races) | `services/db.py` | real transactions + pooling | 400, 402 |
| C18 | 🟡 | **`all_sky` opt-in gate is a bare truthiness check**, so a model emitting the *string* `"false"` / `"no"` / `"0"` triggers an unbounded whole-catalog aggregate (`elif all_sky:` — any non-empty string is truthy). Preserved verbatim in the P1 migration (`DensityAggregateInput.all_sky: Any`, parity), so the bug is unchanged, not introduced. | `services/datalab_query_builders.py:168`; `capabilities/datalab.py` `DensityAggregateInput` | require `all_sky is True` (strict identity, like `expert_ack`) — **behaviour change, benchmark-gate it** | P1 stateful-migration review (2026-07-09) |

## Dead code to remove (P5)

| Where | Evidence |
|---|---|
| `agents/` (7 files) | imported nowhere outside itself **(v)** |
| `core/agent_pool.py` | instantiated, never used |
| `core/workflow_store.py`, `result_cache.py`, `context_manager.py` | zero live call sites |
| `core/health_monitor.py`, `model_council.py`, `session_memory.py` | never wired / dead memory stack |
| `config/settings.py`, `config/__init__.py` | imported nowhere **(v)** |

## Missing tests

- No per-tool/per-capability unit tests (blocked by the monolith — the capabilities layer unblocks this).
- No adapter-parity tests (native vs MCP) — will exist only after the shared core.
- No **full** benchmark run persisted through the harness — headline numbers live in hand-written `tmp/live-ui-test-*.md`. **Capture a real baseline in P0.**
- CI lints only 5 files; benchmark scorer self-tests not in CI.
- `tests/smoke/` is empty (only `__init__.py`).

## Design questions to resolve (🔵 — decide before/at the relevant phase)

| # | Question | Why it matters | How to decide |
|---|---|---|---|
| D1 | **FastMCP version target** (MANNA says 3.x; ALMA_MCP pins ≥2.0) | API differences | confirm with the MANNA/Dan Gause repo before coding the adapter |
| D2 | **Who owns the result store** when a capability runs server-side in MANNA | `result_id` follow-ups | if MANNA executes, MANNA owns the store + exposes `get_rows/to_csv/stats` |
| D3 | **One MCP server or several** (Quasar producer vs mesh) | deployment topology | design doc §5 leans "mesh"; start with MANNA + Quasar-producer, add servers later |
| D4 | **Token passthrough vs OAuth proxy** for per-user archive auth | security/isolation | per-session MCP init options first; OAuth 2.1 for remote/public |
| D5 | **How much schema context** the model needs to write correct ADQL | tool ergonomics vs prompt cost | resolved by schema-discovery tools + the Data Lab KB registry |
| D6 | **Is the MANNA repo public yet** | can we consume it directly | the 4 ecosystem-research agents were interrupted; verify `github.com/nrao` / ask Dan Gause |
| D7 | **Local-LLM (dlai1) tool-calling reliability** | drives model tier choice | benchmark open-weights vs commercial once P1–P3 land |

## Uncertain assumptions — verify before relying on

- **All "(approx)" line numbers** come from the subagent analysis reading the real code, but may have drifted. Verify with `grep -n` before editing.
- The exact tool count ("142") vs the 145 register/`Tool(` call sites (v) — some are helper `Tool(` constructions, not registrations. Get the real number from `registry.list_tools()` at runtime.
- Whether `services/` is 74 or 76 modules (analysis said 76; `ls` now shows **74 (v)**) — the dir changes; recount when it matters.
- The frontend LOC (~13.2k) and several `main.py` sub-line-ranges are approximate.
