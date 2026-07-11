# HANDOFF_NEXT_SESSION.md — continue the Quasar V2 main line

**Written 2026-07-09 by the main-line (agent.py / capabilities) session.**
Assume the reader remembers nothing. This file + `STATUS.md` are enough to continue.

> **Read order:** this file → `STATUS.md` (living log, has every track's entries) →
> `IMPLEMENTATION_PRIORITIES.md` → `SHARED_CORE_ARCHITECTURE.md`.
> Keep `OPEN_ISSUES_AND_TODOS.md` + `FILE_AND_CODE_MAP.md` open as reference.

---

## 1. Where the project actually is

**P0 (stabilize): ✅ DONE.** Fabrication → typed errors (C1/C2/C3), RCE surfaces gated
off (S1/S2), cheap wins (C5/C6/C7/C9/C11), benchmark baseline captured. Two adversarial
review rounds found 6 + 3 real defects; all fixed. Regression suite:
`tests/unit/test_p0_stabilization.py` (19✓).

**P1 (monolith / shared core): IN PROGRESS — this is THE priority.**

| Piece | State |
|---|---|
| `capabilities/base.py` contracts | ✅ done |
| `core/tools.py` registry (namespacing/validation/collision/subsetting) | ✅ done |
| `adapters/native/` (capability→Tool bridge) | ✅ done |
| `capabilities/datalab.py` | ✅ **20 of ~25 tools migrated + inline deleted** |
| `ui-pro/api/main.py` split | ✅ done by the api-split track (4,368 → ~250 lines) |
| `core/agent.py` shrink | 🔄 **12,714 → 12,345** (gate: < ~1,500) |
| Other families (alma/ads/vo/viz/resolve/files/spectra) | ❌ not started |
| `core/runner.py` / `core/router.py` extraction | ❌ not started |

**P2–P6:** not started (except P5 partial dead-code, done by another track).

---

## 2. The four parallel sessions (what they did / what they left you)

All four were run concurrently in separate sessions on disjoint file sets. They
communicate **only** through `docs/v2/STATUS.md` + `OPEN_ISSUES_AND_TODOS.md` —
a manual "blackboard", not a message bus. **Re-read STATUS.md before starting.**

| Track | Scope | Status | What it left for the main line |
|---|---|---|---|
| **A — api split** | `ui-pro/api/main.py` | ✅ **LANDED.** 4,368 → ~250 lines; `api/routers/*` (17), `api/sse.py`, `api/serializers/*`, `deps.py`, `models.py`, `bootstrap.py`. 68-route parity, boot-verified, 5-agent review → 0 divergences. **S4** (CORS allowlist) + **S5** (auth on traces/proposals) fixed. | Nothing. The `main.py < 1,500` half of the P1 gate is **already met**. |
| **B — dead code (P5)** | dead modules | ✅ **PARTIAL.** Deleted `agents/`, `core/workflow_store.py`, `core/model_council.py`, `core/result_cache.py` (also unwired from `conductor.py:36,306`). | ⚠️ **Handoff to you:** `agent_pool`, `context_manager`, `session_memory`, `health_monitor` are wired in `core/agent.py` (~:354–363) and can only be removed by the agent.py owner. **Blocked:** `config/settings.py` has a live CI importer (`.github/workflows/ci.yml:75`). |
| **C — CI + docs** | `.github/`, non-v2 docs | (check STATUS for its entry) | — |
| **D — security** | `services/admin_access.py`, `auth.py`, frontend | ✅ **S3 fixed** (no hard-coded admin; env allowlist + DB `role` column). **S6:** `hmac.compare_digest` + login rate limiting + **httpOnly cookie backend WIRED & verified**. | ⚠️ **Frontend cookie flip deferred** (→ a future session): ~20 `!token` guards must migrate to `isAuthenticated` + `credentials:'include'`. Full recipe in `STATUS.md` §"S6 httpOnly-cookie migration". |

### ⚠️ Important finding on Track B's handoff — DO NOT blind-delete
I investigated it. It is **not** a clean wiring removal:
- **`session_memory` is actively called at 3 live sites** — `extract_if_needed` (~agent.py:734), `record_tool_calls` (~:12007), `clear` (~:12321). Removing it removes per-turn LLM extraction → a **behaviour change**.
- **`agent_pool`** is passed into the Conductor (~agent.py:390).
- **`health_monitor`** is injected into `model_router` (~:359) and read hasattr-guarded in the API.
- **`context_manager`** looks genuinely dead (instantiated, no method calls) → likely the only safe one.

**Recommendation:** do this only when you can run DataLabBench to gate it, or land it as an explicit, documented behaviour change. Deferred deliberately.

---

## 3. The architecture you must keep (patterns already established)

### 3a. Capability contract (`capabilities/base.py`)
A capability is **transport-pure**: `run(inp: InputModel, ctx: CallContext) -> ToolResult`.
It must NOT touch `self`, thread-locals, `os.environ` tokens, `print()`, or SSE.
Everything it needs is injected via `CallContext.services` / `.result_store`.

`ToolResult` has canonical fields (`success/error/warnings/degraded/result_id/provenance/meta`)
plus a **migration escape hatch**: `native` = the exact legacy output dict.
`to_native()` returns `native` verbatim when set → **byte-parity guaranteed**.
`to_mcp()` drops `native` (for the future MCP adapter).

### 3b. Two registration wrappers in `core/agent.py`
Legacy tool **schemas/descriptions are reused verbatim** at registration (only the
`function=` is swapped), so the LLM-facing surface never drifts.

```python
# non-image tools
function=self._datalab_tool_fn("datalab_cone_count")

# image-producing tools (plots, SIA cutouts, color diagrams)
function=self._datalab_image_tool_fn("datalab_catalog_scatter")
```

- `_datalab_tool_fn(name)` → `adapters.native.build_tool(cap, self._datalab_ctx_provider).function`
- `_datalab_image_tool_fn(name)` → same, then applies `self._datalab_attach_image_result(raw, caption)`
  which sets the `last_run_result` image card and strips base64 from the LLM dict.
  A capability can pass a computed caption via a **private `_caption` key**, which the
  wrapper **pops** (so it never leaks to the model).
- `_datalab_ctx_provider()` clears `last_run_result` (the one transport concern) and injects:
  `datalab_client`, `svo_fps_client`, `datalab_image_service`, `datalab_job_service`,
  `resolve_coordinates` (= the bound `_datalab_coordinates`), plus `result_store`.

### 3c. Migration recipe (proven 4×, follow it exactly)
1. Read the inline `_datalab_X` method. **Relocate its logic verbatim** into
   `capabilities/datalab.py` as a `BaseCapability` + a pydantic `InputModel`
   mirroring the method signature. Mirror any **null-coercion** the legacy does
   (models send `null`; use `Optional[...]` + coerce in `run()`).
2. Set `ToolResult(native=<the exact legacy dict>)`.
3. Append the capability to `SQL_CAPABILITIES` / `PLOT_CAPABILITIES` / `MISC_CAPABILITIES`.
4. In `_register_tools`, change only `function=` → `self._datalab_tool_fn("...")` (or the image variant).
5. **Delete the inline method.**
6. Add unit tests (mock the injected service) in `tests/unit/test_datalab_capability.py`.
7. Verify: `py_compile`, unit tests, **boot the backend**.

---

## 4. What's left, in order

### Step 1 — the 5 remaining Data Lab tools (all still inline in `core/agent.py`)

| Tool | Line (approx — grep!) | Why it's deferred | Suggested approach |
|---|---|---|---|
| `_datalab_density_aggregate` | ~6295 | auto-tiling + **per-turn state** `self._datalab_agg_timeout_tables` (a set of tables whose sync aggregate timed out this turn) | Inject the set via `CallContext.services["datalab_agg_timeout_tables"]` (the agent owns/creates it lazily, so it persists across calls in a turn). Then relocate the tiling+merge logic. |
| `_datalab_job_status` | ~7877 | **per-turn state** `self._job_poll_counts` (stop-polling nudge after 3 polls) | Same pattern: inject the dict via `CallContext`. |
| `_datalab_density_vetting` | ~7784 | image is **nested** in `out["cutout_grid"]`, not the top-level result → the image wrapper doesn't fit | Either extend `_datalab_image_tool_fn` to accept a nested key, or give this tool its own small wrapper. |
| `_datalab_tiled_search` | ~7836 | starts a background job with a closure via `default_job_service().start(...)` | Job service is already injected; the closure captures orchestration args — relocate as-is. |
| `_datalab_export_notebook` | ~7908 | sets its own notebook card and **deliberately does NOT clear `last_run_result`** | Needs a third wrapper (or a `clears_card=False` flag on the ctx provider). |

**Do the two "request-scoped state" ones together** (`density_aggregate` + `job_status`) —
they share the exact same fix (inject mutable per-turn state via `CallContext`), which is
also the pattern every later family will need.

### Step 2 — retire the shared helpers
Once **no inline datalab tool remains**, delete from `core/agent.py`:
`_execute_datalab_sql`, `_datalab_error`, `_datalab_fit_rows`, `_datalab_query_summary`,
and (if no other tool uses it) `_datalab_coordinates`.
⚠️ `_datalab_coordinates` is currently ALSO used by non-datalab tools — grep before deleting.
This is a big shrink (~250 lines).

### Step 3 — next family: **ALMA**
Same recipe. Create `capabilities/alma.py`. Note the C11 follow-up: the richer
`_filter_results` (aliasing + `pd.to_numeric` coercion) was deliberately NOT promoted —
revisit during this migration (benchmark-gate it).

### Step 4 — remaining families
`ads` → `vo` → `viz` → `resolve` → `files` → `spectra`.

### Step 5 — extract `core/runner.py` + `core/router.py`
Only after the tools have left `agent.py`. `stream_response_api` → `Runner`;
the routing regex forest → a small-model structured router.

### Step 6 — the P1 exit gate
`agent.py` < ~1,500 lines (main.py ✅ already), benchmark ≥ baseline, and adding a
capability requires **no** edit to the runner or god object.

---

## 5. Verification recipe (use every time)

```bash
# compile
.venv/Scripts/python.exe -m py_compile core/agent.py capabilities/datalab.py

# the suites that matter for this work
QUASAR_ENV=development .venv/Scripts/python.exe -m pytest \
  tests/unit/test_capabilities_base.py tests/unit/test_datalab_capability.py \
  tests/unit/test_datalab_p0.py tests/unit/test_p0_stabilization.py -q

# boot the backend (proves registration works in the REAL __init__)
QUASAR_ENV=development PYTHONPATH=. .venv/Scripts/python.exe -m uvicorn api.main:app \
  --app-dir ui-pro --port 8000
# then POST /api/auth/login {"username":"1@1","password":"1"} → expect a token
```

**Known pre-existing failures (NOT yours):** 3 in `test_datalab_p1` (color-image /
cutout reprojection — confirmed failing on clean HEAD), plus `test_content_safety`
and `test_issue_report_service` flakes.

**Gotcha:** `QUASAR_ENV=development` is REQUIRED locally — `.env` pins
`ENVIRONMENT=production`, and the backend refuses to boot without
`USER_API_KEY_FERNET_KEY` in prod mode.

---

## 6. The benchmark gate (currently PARKED by the user)

- **Baseline:** `65.5/100` overall, `72.3` auto — SHA `650c354`, gpt-4.1, judge gpt-4o →
  `Benchmark/datalabbench/results/baseline_650c354_gpt-4.1/BASELINE.md`.
- **Parity proven once:** capability path scored `65.6 / 74.2` ≥ baseline →
  `Benchmark/datalabbench/results/p1_datalab_capability_gpt-4.1/PARITY.md`.
- **Currently blocked:** the local test account's weekly OpenAI token quota is exhausted
  (two full benchmark runs). **The user said: skip benchmarks for now, they'll ask later.**
- Migrations since then are verbatim relocations verified by unit tests + boot. Run
  **one** DataLabBench pass before anything is considered merge-ready:

```bash
.venv/Scripts/python.exe Benchmark/datalabbench/run_datalabbench.py \
  --api-url http://localhost:8000 --model gpt-4.1 \
  --output-dir Benchmark/datalabbench/results/<name>
```

---

## 7. Non-negotiable rules (carried from the original brief)

1. **Never fabricate scientific data** — failures are typed errors, never mocks.
2. **No business logic in adapters** — logic lives in `capabilities/`.
3. **One implementation per capability** — inline + capability duplication is a bug; delete the inline once migrated.
4. **Benchmark-gate every change** (currently parked by user instruction — resume when told).
5. **Keep quasarassistant.com working** — native in-process path stays production.
6. **Capabilities are transport-pure** — no thread-locals / `os.environ` tokens / `print()` / SSE.
7. **Verify "approx" line numbers with grep before editing.**
8. **Production stays NATIVE.** Do NOT deploy an MCP server to Render (2 GB RAM). MANNA is *consumed*, not self-hosted.
9. **Never commit or push unless explicitly told.** Do not even mention it.

---

## 8. Discovered issues logged but NOT fixed

- `_search_by_target` positional fallback is **dead**: gate reads `resolved.get("ra")` but `_resolve_target` returns `ra_deg`/`dec_deg` (`core/agent.py` ~5480). Enabling it is a behaviour change.
- `services/ads_service.py` is a **dead duplicate** of `integrations/ads_client.py` (fabrication neutralized in both) → delete in P5.
- C7's SLA `asyncio.wait_for` cannot cancel the blocking `run_in_executor` tool thread (inherent; accepted).
- C3 not yet applied to the frequency/keyword/source search handlers (lands with the ToolResult migration).
- C11: the richer `_filter_results` variant was deleted rather than promoted — revisit in the ALMA migration.

---

## 9. Files this session created/owns

```
capabilities/__init__.py        capabilities/base.py        capabilities/datalab.py
adapters/__init__.py            adapters/native/__init__.py
core/tools.py                   (extended registry)
core/agent.py                   (wrappers + ctx provider; 20 tools repointed, inline deleted)
tests/unit/test_capabilities_base.py     (15✓)
tests/unit/test_datalab_capability.py    (~23✓)
tests/unit/test_p0_stabilization.py      (19✓)
Benchmark/datalabbench/results/baseline_650c354_gpt-4.1/BASELINE.md
Benchmark/datalabbench/results/p1_datalab_capability_gpt-4.1/PARITY.md
```

**Lane discipline:** the main line owns `core/agent.py`, `capabilities/`, `adapters/`,
`core/tools.py`, and its own tests. If other sessions run concurrently, keep to that
boundary and log every change in `STATUS.md`.
