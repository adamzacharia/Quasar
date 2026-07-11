# MONOLITH_BOTTLENECK_ANALYSIS.md

**The #1 problem to solve.** This file explains exactly what the monolith bottleneck is, why it blocks everything, which code is affected, and the target structure to split into.

## 1. What the monolith bottleneck is

Two god objects hold most of the system's logic, mixing many responsibilities in single files:

| File | Size (verified 2026-07-08) | What's crammed in |
|---|---|---|
| `core/agent.py` | **12,714 lines** | 145 tool registrations + ~200 private tool implementations + the streaming loop + the Conductor tool executor + the sandbox bridge + routing regexes + web-search orchestration + RAG gating + system-prompt building + thread-local per-request state + a legacy CLI intent pipeline |
| `ui-pro/api/main.py` | **4,368 lines** | HTTP routing + the SSE state machine + UI presentation (hardcoded per-archive column maps) + CADC VOTable XML parsing + matplotlib FITS rendering + quota checks + job dicts + channels |

Supporting evidence of the tangle:
- **145** `Tool(`/register call sites inside `agent.py` (verified). Tools are registered *inline* in `_register_tools()` (approx `agent.py:905–3790` — verify).
- `_filter_results` is **defined twice** (`agent.py:8397` and `agent.py:10147`, verified) — the file is large enough that a duplicate method survives.
- The project's own `CONVENTIONS.md` documents the extension procedure as **"insert at these 4 anchors in a ~10,850-line file, never reformat."** A manual, collision-prone protocol is the definition of a bottleneck.
- `services/` is a **flat directory of 74 modules**; `integrations/` is **15** modules with ~5 near-duplicate IVOA-TAP clients.

## 2. Why it matters (what it blocks)

- **Maintainability:** every new capability edits the god object; merge conflicts and duplicate methods (`_filter_results`) are inevitable.
- **Testing:** logic entangled with the streaming loop and thread-local state cannot be unit-tested in isolation. There is essentially no per-tool unit coverage.
- **Extension:** you cannot add MCP or a second tool path cleanly when the "tool" and its "transport" (the streaming loop) live in the same 12k-line class.
- **Parallel development:** two people cannot work on two tool families without colliding in `agent.py`.
- **Correctness at scale:** per-request state lives in **thread-locals** (`self.last_search_results`, response-ID maps — approx `agent.py:602–608`, verify) and module-level dicts, so concurrent requests and multiple workers corrupt each other.
- **The dual-path requirement is impossible without this split:** MCP and native tools can only "share logic instead of duplicating" if the logic is extracted into shared functions first.

## 3. Which responsibilities are mixed (and where each should go)

| Responsibility (currently in `agent.py`) | Should live in |
|---|---|
| Tool *implementations* (what an ALMA/DataLab/ADS query actually does) | **`capabilities/<domain>.py`** (the shared core) |
| Tool *registration* / schema exposure | thin **adapters** (`adapters/native`, `adapters/mcp`) |
| The streaming tool loop (`stream_response_api`) | **`core/runner.py`** (a Runner class) |
| Routing regexes (live-data / RAG-gate / researcher detection) | **`core/router.py`** (one small-model structured router) |
| Conductor tool executor (`_conductor_tool_executor`) | **`core/conductor.py`** (request-scoped) |
| Sandbox bridge (`_sandbox_tool_bridge`) | **`core/compute.py`** / later `compute-mcp` |
| System-prompt building | **`core/prompts/`** |
| Per-request state (results, tokens, trace) | **`CallContext`** object passed explicitly (kills thread-locals) |
| Legacy CLI intent pipeline (`process_query`/`determine_intent`) | **delete** (route CLI through the Runner) |
| Data-card shaping / column maps (in `main.py`) | **serializer module** or frontend |
| SSE state machine (in `main.py`) | **`api/sse.py`** (testable class) |

## 4. Target module structure

```
quasar/
  core/                      # the orchestration BRAIN (keep, refactor)
    task_dag.py              # KEEP as-is (clean, tested-by-use)
    complexity.py            # KEEP concept, retune thresholds
    conductor.py             # REFACTOR → request-scoped OrchestrationRun
    runner.py                # NEW — extracted streaming loop
    router.py                # NEW — small-model structured router (replaces regex forest)
    tool_registry.py         # EXTENDED core/tools.py (namespacing, validation)
    recovery.py              # KEEP, fix the executor-bypass
    llm_client.py            # KEEP (multi-provider shim)
    prompts/                 # NEW — system/decomposition/synthesis prompts

  capabilities/              # ★ THE SHARED CORE — extracted from agent.py ★
    base.py                  # ToolResult, CallContext, Capability protocol, ServiceClient base
    alma.py                  # search_alma, line_coverage, ... (pure functions → ToolResult)
    datalab.py               # datalab_query, sia_search, crossmatch, ... (reference migration)
    ads.py                   # literature search / fulltext / citations
    vo.py                    # generic IVOA TAP/SIA/SCS
    resolve.py               # SIMBAD/NED/Sesame name resolution
    viz.py                   # plotting / sky maps / SED
    files.py                 # DataLink / VOSpace / MyDB / downloads
    spectra.py               # SPARCL

  adapters/                  # ★ thin transport shims over capabilities ★
    native/__init__.py       # registers capabilities as core Tools
    mcp/server.py            # FastMCP server exposing the same capabilities

  services/                  # PLATFORM (auth/db/quota/keys/conversations) + heavy compute
    (regrouped from the flat 74-module dir into platform/ compute/ archive-adapters/)

  integrations/              # PROTOCOL ENGINES (one TAP engine + shared transport core)

  api/                       # split FastAPI: routers/ + sse.py + serializers/
```

## 5. Extraction strategy (safe, incremental — this is the P1 work)

1. **Stand up the contracts first** (`capabilities/base.py`) — `ToolResult`, `CallContext`, `Capability`. Nothing moves yet.
2. **Extend the registry** (`core/tools.py`) so it can register a `Capability` and expose it to both adapters.
3. **Migrate ONE family end-to-end as the template:** the `datalab_*` family (it already uses `result_id` handles and has tests `tests/unit/test_datalab_p0–p2.py`). Move its implementations into `capabilities/datalab.py`; register via the native adapter; prove the benchmark is unchanged.
4. **Repeat per family** (alma → ads → vo → viz → resolve → files → spectra), each behind a feature flag, each benchmark-gated. `agent.py` shrinks family by family.
5. **Extract the runner + router** once tools no longer live in `agent.py`.
6. **Split `main.py`** into routers in parallel (independent of the agent work).

**Golden rule during extraction:** a capability function must be **pure w.r.t. transport** — it takes a typed input model + `CallContext` and returns a `ToolResult`. It must not import the agent, touch thread-locals, read `os.environ` for tokens, or emit SSE. That purity is what makes it shareable by both paths and unit-testable. See `SHARED_CORE_ARCHITECTURE.md`.

## 6. What NOT to touch during the split

- `core/task_dag.py`, `core/complexity.py` — keep.
- The Data Lab triad (`services/datalab_sql_policy.py`, `datalab_query_builders.py`, `datalab_registry.py`) — these become the body of `capabilities/datalab.py`; keep the logic verbatim, just relocate/wrap.
- The SSE event *vocabulary* (`token/thought/status/data/papers/plotly/image/task_*/plan_review/usage`) — keep; only move where it's emitted.
- BYOK/Fernet key handling — keep; it's the correct isolation model.
