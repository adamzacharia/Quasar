# NEXT_SESSION_START_HERE.md

**You are continuing Quasar V2 implementation. You remember nothing from the prior session. This file tells you exactly what to do.**

---
> ## ⚠️ 2026-07-09 — READ `HANDOFF_NEXT_SESSION.md` FIRST
>
> P0 is **done**. P1 is **well underway**: the `capabilities/` layer + `adapters/native/`
> exist, **20 of ~25 Data Lab tools are migrated and their inline copies deleted**
> (`core/agent.py` 12,714 → 12,345), and `ui-pro/api/main.py` is already split (4,368 → ~250).
> Four parallel sessions (api-split / dead-code / CI-docs / security) also landed work.
>
> **`docs/v2/HANDOFF_NEXT_SESSION.md` supersedes Step 3 below.** It has the current state,
> every track's handoff, the exact migration recipe, and the ordered next steps.
> The sections below remain accurate for background/context and the golden rules.
---

## Step 1 — Read, in this order (≈20 min)

1. `INDEX.md` — the map + the one governing decision + verified facts.
2. `SESSION_CONTEXT.md` — what Quasar is, what MANNA is, why MCP-first-but-keep-tools.
3. `IMPLEMENTATION_PRIORITIES.md` — the ordered plan. **Monolith is #1.**
4. `MONOLITH_BOTTLENECK_ANALYSIS.md` + `SHARED_CORE_ARCHITECTURE.md` — the two that define the actual work.
5. Skim `OPEN_ISSUES_AND_TODOS.md` and `FILE_AND_CODE_MAP.md` — reference while coding.

For deeper evidence on any claim: `C:/Users/adama/Desktop/France/.research/data/401_quasar_core*.md` (the core/agent analysis — most relevant to the monolith) and its siblings 400–409.

## Step 2 — Inspect the code before changing anything (≈30 min)

Run these to ground yourself (and confirm the doc line numbers haven't drifted):

```bash
cd C:/Users/adama/Desktop/Quasar-main
wc -l core/agent.py ui-pro/api/main.py                 # expect 12714 / 4368
grep -n "def _filter_results" core/agent.py            # expect 8397 and 10147
grep -n "new_event_loop\|run_until_complete" core/agent.py   # the broken bridge
grep -c "register_tool\|Tool(" core/agent.py           # ~145
sed -n '900,960p' core/agent.py                        # see how _register_tools starts
sed -n '10817,10900p' core/agent.py                    # see stream_response_api
cat core/tools.py                                      # the registry you'll extend (~65 lines)
sed -n '1,120p' services/datalab_registry.py           # the reference "keep" code
ls services/ | grep datalab                            # the family you migrate first
cat tests/unit/test_datalab_p0*.py 2>/dev/null | head  # the contract to preserve
```

Confirm the benchmark harness runs:
```bash
python Benchmark/datalabbench/run_datalabbench.py --self-test   # scorer sanity
```

## Step 3 — Do the work, in this order

### First: P0 stabilization (small, ~1 week) — see `IMPLEMENTATION_PRIORITIES.md` P0
Do the minimum that makes refactoring safe:
1. **C1/C2/C3** (fabrication → typed errors): `services/analysis.py`, `integrations/ads_client.py`, `services/search.py`.
2. **S1/S2** (disable RCE surfaces behind a prod flag): `user_tools_service.py`, `/api/mcp-servers` spawner.
3. **Capture a benchmark baseline** through the harness with git SHA recorded. This is your regression gate for everything after.

Also fix the cheap high-value bugs while you're here: **C5** (RAG score-gate), **C6** (`rank_bm25`), **C7** (recovery bypass), **C9** (DAGCache wrong-target), **C11** (duplicate `_filter_results`).

### Then: P1 monolith / shared core (THE priority) — see `SHARED_CORE_ARCHITECTURE.md`
1. Create `capabilities/base.py` with `ToolResult`, `CallContext`, `Capability`, `ServiceClient` base, `ResultStore`.
2. Extend `core/tools.py` → registry with namespacing + validation.
3. **Migrate the `datalab` family first** (reference migration): move `services/datalab_*` logic into `capabilities/datalab.py` as pure functions returning `ToolResult`; register via a new `adapters/native/`. Prove DataLabBench ≥ baseline. This is the template.
4. Repeat per family (alma → ads → vo → viz → resolve → files → spectra), each behind a flag, each benchmark-gated. `agent.py` shrinks each time.
5. Extract `stream_response_api` → `core/runner.py`; regexes → `core/router.py`.
6. In parallel, split `ui-pro/api/main.py` into `api/routers/*` + `api/sse.py`.

### Then: P2 MCP path — see `MCP_IMPLEMENTATION_PLAN.md`
1. Fix the client bridge (long-lived loop + `run_coroutine_threadsafe`).
2. Build `adapters/mcp/server.py` (FastMCP over the capabilities).
3. Mount MANNA behind a per-family flag; migrate `datalab_*` calls to it; prove parity + benchmark.

### Then: P3 native path improvements, P4 tests, P5 cleanup, P6 docs — see `IMPLEMENTATION_PRIORITIES.md`.

## Step 4 — Keep the docs alive

After **every** meaningful change:
1. Update `STATUS.md` (what changed, what's next, benchmark delta).
2. If the change alters architecture/scope, update the relevant doc (`SHARED_CORE_ARCHITECTURE.md`, `FILE_AND_CODE_MAP.md`, etc.).
3. Close the corresponding item in `OPEN_ISSUES_AND_TODOS.md`.

## The golden rules (do not violate)

1. **Never fabricate scientific data** — failures are typed errors, never mock/invented values.
2. **No business logic in adapters** — logic lives in `capabilities/`; adapters only translate transport. If you're writing ALMA/DataLab logic inside `adapters/mcp` or `adapters/native`, stop and move it.
3. **One implementation per capability** — duplication between the MCP and tool paths is a bug, not a stage.
4. **Benchmark-gate every merge** — no refactor lands if it regresses the P0 baseline.
5. **Keep quasarassistant.com working** — migrate behind per-family feature flags; native path stays selectable.
6. **Capabilities are transport-pure** — no thread-locals, no `os.environ` token reads, no `print()`, no SSE emission inside a capability. Use `CallContext`.
7. **Verify before editing** — line numbers marked "(approx)" may have drifted; `grep -n` first.

## If you're unsure what to do next

The single most valuable first coding action is: **create `capabilities/base.py` and migrate the `datalab` family as the reference.** Everything else in V2 follows the pattern that migration establishes. If that's already done, check `STATUS.md` for the current family in flight.
