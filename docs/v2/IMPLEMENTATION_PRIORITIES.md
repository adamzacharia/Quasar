# IMPLEMENTATION_PRIORITIES.md — Ordered priority list

Priorities are strictly ordered. Do not start priority N+1 before priority N's **Exit gate** is green, except where "runs concurrently" is noted. Each priority names concrete work; the deep how-to is in the referenced doc.

---

## Priority 0 (PREREQUISITE, small, runs concurrently with P1) — Make the base safe to refactor

**Why it exists:** refactoring on top of fabricated-data paths and in-process RCE is dangerous, and you cannot measure the refactor without a baseline. Keep this minimal — it is *not* the focus, just the floor. This is the honest addition to your requested order (you asked for suggestions): a handful of fixes must land first or alongside, or the monolith work sits on sand.

**Do (only these):**
1. Turn fabrication into typed errors: `services/analysis.py` mock UV results; `integrations/ads_client.py` `_get_example_papers` fake bibcodes; `services/search.py` swallow→empty-DataFrame. (approx lines in `OPEN_ISSUES_AND_TODOS.md`.)
2. Disable the two RCE surfaces behind a prod feature flag: `services/user_tools_service.py` `exec()`; the `/api/mcp-servers` stdio command spawner (`ui-pro/api/main.py:2689`). (Proper replacement comes in P2/P3.)
3. Capture a real benchmark baseline through the harness (`Benchmark/datalabbench/` + ALMA bench) with git SHA + model recorded. Today the headline numbers only live in `tmp/*.md`.

**Exit gate:** no code path returns synthetic science data; no user can trigger arbitrary code/commands in the server; a provenance-stamped benchmark baseline exists in the repo.

---

## Priority 1 (THE priority) — Solve the monolith bottleneck via the shared core

**This is the heart of V2.** Extract the logic buried in `core/agent.py` (12,714 lines, 145 tool registrations) and `ui-pro/api/main.py` (4,368 lines) into a clean **capabilities layer** that both MCP and native tools will call. Doing this *is* the de-monolith AND the foundation for dual-path. Full design: `MONOLITH_BOTTLENECK_ANALYSIS.md` + `SHARED_CORE_ARCHITECTURE.md`.

**Do (in this order):**
1. Define the shared contracts: `ToolResult` envelope, `CallContext`, and the `Capability` protocol (`SHARED_CORE_ARCHITECTURE.md` §3).
2. Extend the tool registry (`core/tools.py`) with namespacing + validation + collision protection.
3. Extract tool implementations out of `agent.py` into `capabilities/<domain>.py` modules (alma, datalab, ads, vo, viz, resolve, files, spectra), each a pure function returning `ToolResult`. Start with the **datalab** family (cleanest, already result-handle-based) as the reference migration, then alma, then the rest.
4. Extract the streaming loop (`stream_response_api`, approx `agent.py:10817–12467`) into a `Runner` class and the routing regexes into a small-model router.
5. Fix the duplicate `_filter_results` (`agent.py:8397` and `10147`).
6. Split `ui-pro/api/main.py` into `APIRouter` modules; extract the SSE state machine into a testable class.

**Exit gate:** `core/agent.py` and `ui-pro/api/main.py` each < ~1,500 lines; adding a capability is a new file in `capabilities/` + registration, with **no** edit to the runner or the god object; the datalab + alma families run through capability functions; benchmark ≥ baseline.

---

## Priority 2 — Build the MCP path (on top of the shared core) — BUILD + TEST LOCALLY, DEFER DEPLOY

> **HOSTING CONSTRAINT (2026-07-08, decisive).** There is currently **no production host for MCP servers.** Quasar's public deployment is on **Render with 2 GB RAM**, and co-locating an MCP server there would **double** the memory footprint (a second Python process loading astropy/pyvo/pandas) — which 2 GB cannot absorb. **Therefore:**
> - **Production keeps using native in-process tools** (the capabilities layer via the native adapter). This is correct, not a compromise — MCP only relieves RAM when the server is on a *separate* host.
> - The MCP path is **built and tested LOCALLY** (stdio subprocess on the dev PC; point Claude Desktop at it for a zero-cost demo), kept behind a feature flag, and **not deployed to Render**.
> - The realistic first *remote* MCP is **MANNA, which NRAO/STABLE hosts — Quasar consumes it (needs only a URL), does not host it.** Test against a local MANNA until theirs is up.
> - Production MCP deploy is **deferred** until hosting exists (Cloud Run / Modal scale-to-zero, a small VPS, or dlai1). Nothing in P0/P1 waits on this.

Expose the capability layer through a FastMCP server and fix the client bridge so Quasar can consume MANNA. Full plan: `MCP_IMPLEMENTATION_PLAN.md`.

**Because of the constraint, P2 reorders relative to P3:** the native adapter (P3) is the *production* path and should reach parity first; the MCP adapter here is validated locally and shelved-ready. The monolith win (P1) is fully independent of MCP hosting.

**Do:**
1. Build the MCP **adapter**: a FastMCP server that registers each `Capability` as an `@mcp.tool()` (schema auto-derived from the capability's pydantic input model). Zero business logic in the adapter.
2. Fix the **client bridge**: replace the per-call `asyncio.new_event_loop()` (`agent.py:522–523`, `11671–11674`) with one long-lived loop + `run_coroutine_threadsafe`; mount servers over streamable HTTP (one mount per process), namespaced `{server}__{tool}`.
3. Wire MANNA in behind a per-tool-family feature flag; migrate `datalab_*` first, then ALMA/NRAO TAP.
4. Move per-user tokens to per-session MCP auth (not process-global `os.environ`).

**Exit gate:** the same capability is callable through both the native registry and the MCP server; Quasar consumes MANNA for Data Lab at ≥ baseline scores with the native path still available behind a flag; the concurrent-call bridge smoke test passes.

---

## Priority 3 — Preserve & improve the native tool path (compatibility)

Ensure the direct-tool path keeps working and improves, sharing the capability layer (no duplication). Full plan: `TOOLS_IMPLEMENTATION_PLAN.md`.

**Do:**
1. Rebuild the native **adapter** as a thin wrapper that registers capabilities as `Tool`s (schema from the same pydantic model) — the tool path and MCP path now differ only in transport.
2. Replace "LAST results" thread-local coupling with `result_id` handles registry-wide (generalize the datalab pattern).
3. Add intent-based tool subsetting (stop sending all ~142 schemas every round).
4. Delete `exec()` user tools; point users to "bring your own MCP server."

**Exit gate:** every capability has a native adapter and an MCP adapter over one implementation; no capability logic is duplicated between paths; wrong-tool-selection benchmark failures drop.

---

## Priority 4 — Testing

**Do:**
1. Unit tests per capability (offline, mocked transport) — the capability layer makes this finally possible.
2. Adapter parity tests: native adapter and MCP adapter over the same capability produce equivalent `ToolResult`s.
3. Extend the benchmark: fabrication penalty for "claimed-vs-emitted artifact"; trace conductor-subtask tool calls; add repeats/variance; salt prompts to defeat `result_cache` replay.
4. Wire scorer self-tests + capability unit tests into CI (`.github/workflows/ci.yml` — today lints only 5 files).

**Exit gate:** each capability + both adapters have offline unit tests; CI runs them and the benchmark self-tests; parity tests pass.

---

## Priority 5 — Cleanup

**Do:**
1. Delete confirmed dead modules (`agents/`, `core/agent_pool.py`, `core/workflow_store.py`, `core/result_cache.py`, `core/context_manager.py`, `core/health_monitor.py`, `core/model_council.py`, the dead memory stack, `config/settings.py`).
2. Consolidate duplication: one IVOA-TAP engine, one ADS client, one plotting service, one SIMBAD resolver.
3. Config-as-contract: replace the 179 ad-hoc `os.getenv` calls with one typed pydantic-settings schema (fail-fast on missing prod secrets).
4. Repo slimming: move `docs/pdfs`, the 36 MB GIF, PDFs to object storage.

**Exit gate:** no dead modules remain; `grep os.getenv` returns only the settings module; duplicated clients collapsed.

---

## Priority 6 — Documentation

**Do:**
1. Keep this `docs/v2/` set updated (esp. `STATUS.md`) as implementation proceeds — **every major change updates the relevant file.**
2. Generate reference docs from code in CI (tool-registry doc, mermaid diagrams); delete the stale `ARCHITECTURE.md`; fix the tool-count drift.
3. Write a human `CONTRIBUTING.md` separating LLM-workflow docs from onboarding.

**Exit gate:** the tool-registry doc is generated (not hand-written); `docs/v2/STATUS.md` reflects reality; no doc claims a tool count that disagrees with the code.

---

## Priority map at a glance

```
P0 Stabilize (small) ─┐
                      ├─► P1 MONOLITH / SHARED CORE ─► P2 MCP path ─► P3 Tool path ─► P4 Testing ─► P5 Cleanup ─► P6 Docs
(runs concurrently) ──┘                                   (both paths share the capability layer)
```

Later horizons (request-scoped orchestration, durable runs, control-plane split, local-LLM tier, the astronomy MCP mesh, new science tools) are in `../QUASAR_V2_EXECUTION_PLAN.md` Phases 4–6 and are **out of scope until P1–P3 land**.
