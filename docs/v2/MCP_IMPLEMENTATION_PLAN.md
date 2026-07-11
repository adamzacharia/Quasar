# MCP_IMPLEMENTATION_PLAN.md

**How to build the MCP path on top of the shared core.** Prereq: `SHARED_CORE_ARCHITECTURE.md` contracts exist (P1 step 1) and at least the `datalab` family is a capability.

## 0. Deployment reality (read first)

**There is no production MCP host yet, and the public app runs on Render / 2 GB RAM.** This dictates *how* MCP is built:

- **Do NOT run an MCP server on the Render box.** A co-located stdio/HTTP MCP server means two Python processes each loading astropy/pyvo/pandas → the RAM footprint doubles and OOMs the 2 GB instance. Production stays on **native in-process tools** (the capabilities layer). MCP relieves memory only when the server is on a *separate* host.
- **Build the MCP adapter and test it LOCALLY** — run it as a stdio subprocess on the dev PC, or `localhost` streamable-HTTP; point **Claude Desktop** at it as a zero-cost validation + demo. Keep it behind a feature flag; do not deploy it to Render.
- **The first remote MCP you consume is MANNA — hosted by NRAO/STABLE, not you.** Quasar needs only its URL. Until that URL exists, test the client bridge against a **local** MANNA/`ALMA_MCP` instance.
- **Defer production MCP deploy** until hosting exists: Google **Cloud Run** or **Modal** (scale-to-zero, pay-per-use, more RAM than Render for heavy deps), a small **VPS** (e.g. Hetzner), or **dlai1** (the on-prem GPU box in the MANNA plan). None of this blocks P0/P1.
- **Net effect on sequencing:** the *native* adapter is the production path and reaches parity first (P3); the *MCP* adapter (this doc) is validated locally and shelved-ready. The monolith fix (P1) is fully independent of all of the above.

## 1. Two MCP roles — don't conflate them

Quasar sits on **both sides** of MCP:

1. **MCP server (producer):** `adapters/mcp/server.py` — a FastMCP server exposing Quasar's capabilities so *other* hosts (Claude Desktop, the STABLE team, jupyter-ai on gp13) can use them.
2. **MCP client (consumer):** Quasar's agent consumes external MCP servers — first and foremost **MANNA** (Data Lab + NRAO/ALMA over IVOA TAP/SIA/SCS). The client bridge already half-exists but is **broken**.

Build the client-consumer path first (it's what unblocks MANNA and the monolith relief); the producer server is a straightforward second adapter.

## 2. Server structure (producer) — `adapters/mcp/server.py`

- Framework: **FastMCP** (matches MANNA/ALMA_MCP and the `dl`/`sparclclient`/`pyvo` stack). Confirm version — MANNA one-pager says "FastMCP 3.x"; ALMA_MCP pins `fastmcp>=2.0.0`. **(verify the target version before coding.)**
- Transport: **streamable HTTP** for hosted use (mount once per process); **stdio** for local dev / Claude Desktop.
- Registration: iterate the capability list; for each, `@mcp.tool(name=cap.name, description=cap.description)`; input schema derived from `cap.InputModel`; attach `cap.annotations` (SLA/read-only/cost) as MCP tool annotations.
- Resources: expose the ALMA handbook/Data Lab schema as MCP **resources** where useful (read-only reference the model can pull).
- The server calls `cap.run(inp, ctx)` and returns `result.model_dump()` as content blocks. **No logic here.**

## 3. Client bridge (consumer) — the critical fix

**The bug (verified):** the current bridge spins a **new event loop per tool call** (`core/agent.py:522–523` and `11671–11674`) to drive a `ClientSession` whose `anyio` streams are bound to a *different* loop → cross-loop failures / races. It also spawns per-user stdio subprocesses.

**The fix:**
1. One **long-lived asyncio event loop** (or anyio task group) in a dedicated thread owns **all** `ClientSession`s.
2. Synchronous callers submit coroutines via `asyncio.run_coroutine_threadsafe(coro, the_loop)` and block on the returned future. Never create a loop per call.
3. Sessions are **long-lived and pooled**, not created per call.
4. Mount external servers over **streamable HTTP, once per process** with shared rate limiting — **not** per-user stdio subprocesses (V1's registry spawns one server per user per agent → unbounded processes).
5. Namespacing: bridged tools register as `{server}__{tool}` (e.g. `manna__cone_search`). Ensure the planner reads tool names from the **live registry**, not hardcoded lists.

**Acceptance:** a smoke test mounts a local FastMCP server and issues 100 concurrent tool calls with zero cross-loop errors.

## 4. Tool registration & schema design

- **Schema comes from one pydantic model** per capability (`InputModel`). Both the MCP `@mcp.tool` and the native `Tool` derive their JSON schema from it → schemas can never drift between paths.
- Keep schemas **strict and small**: explicit field types, enums for choices, sane defaults, docstrings the LLM sees. (Lesson from ALMA_MCP: raw f-string ADQL interpolation broke on inputs like `Barnard's Star` — validate + quote in the capability.)
- Two-tier tool design (lesson from ALMA_MCP): a set of **curated intent tools** (cone search, line coverage, proposal lookup) with strict schemas + honest pagination, plus **one raw-ADQL escape hatch per service** guarded SELECT-only. MANNA should own the guard (port `services/datalab_sql_policy.py` server-side).

## 5. Request/response flow (consumer path)

```
User query
  → core/router.py (needs which tools? intent categories)
  → registry.list_tools() [native + bridged MCP tools, namespaced]
  → Runner / Conductor selects tool
  → if native:  cap.run(inp, ctx)
    if MCP:      bridge.call(server, tool, args)  # via the long-lived loop
  → ToolResult (native) OR MCP content blocks → normalized to ToolResult
  → serializer → SSE data/image/plotly event  (frontend unchanged)
  → provenance harvested, citations verified (Quasar side)
```

**Normalization is mandatory:** MCP returns JSON content blocks; a normalization layer reconstructs a `ToolResult` (records + columns + provenance) so the rest of Quasar (data-card serializer, provenance harvester, Conductor soft-failure detection) sees one shape regardless of transport.

## 6. Error handling

- Capabilities **never raise** to the caller and **never fabricate** — they return `ToolResult(success=False, error=...)` (this is the `CONVENTIONS.md` contract). Adapters translate that to the transport's error convention.
- Map HTTP semantics (from the Data Lab analysis): 403 = missing/invalid token, 503 = backend unavailable, 429 = rate limit, 504 = gateway timeout (SIA is prone to this — retry with backoff).
- **Data Lab client gotcha (from the KB):** the `dl` client returns **error strings** instead of raising (e.g. `"Error: The provided security token is invalid."`). MANNA/capabilities must sniff leading `"Error"` and convert to typed errors.
- Timeouts/limits (from the KB): sync query max 600 s → auto-promote to async job; async max 86400 s; SPARCL `SparclClient` clamps `connect_timeout` to 3.1 s (patch `MAX_CONNECT_TIMEOUT` before constructing); SPARCL retrieve cap 24,000 records; upload chunk 4 MiB.

## 7. Logging & observability

- **Never `print()` to stdout on a stdio MCP transport** — it corrupts the JSON-RPC stream (this is a real ALMA_MCP bug: module-level `print()` around `mcp.run()`). Route all diagnostics to **stderr / structured logging**.
- Structured log per call: capability name, transport, endpoint, executed query, duration, rowcount, backend used, degraded flag.
- Map MCP **progress notifications** onto Quasar's existing `__tool_heartbeat__ → run_progress` SSE path; map cancellation (`DELETE /api/datalab/jobs/{id}`) onto MCP request cancellation.
- Feed per-call cost/latency into the existing langfuse plumbing.

## 8. Auth model

- **Per-session, not process-global.** Today per-user Data Lab/ADS tokens go through `os.environ` injection (`agent.py:418` approx — verify) → cross-user leak risk. Behind MCP, tokens flow via **MCP session initialization options / OAuth 2.1** for remote servers, held in `CallContext.tokens`.
- Anonymous is the default tier (the Data Lab KB verified anonymous covers all reads + shared scratch writes); login upgrades to private MyDB/VOSpace.

## 9. Testing strategy

1. **Capability unit tests** (offline, mocked integration clients) — the capability layer makes these possible for the first time.
2. **Bridge concurrency smoke test** (§3 acceptance).
3. **Adapter parity test:** native adapter vs MCP adapter over the same capability → equivalent `ToolResult`.
4. **MANNA integration test** behind a flag: a live (or recorded-VCR) `datalab_query` through MANNA matches the native `datalab_client` result contract (`tests/unit/test_datalab_p0–p2.py` defines the contract).
5. **Benchmark gate:** DataLabBench with MANNA ON ≥ the P0 native baseline.

## 10. Rollout sequence

1. Fix the client bridge (long-lived loop). 
2. Stand up MANNA mount behind a feature flag; namespace its tools.
3. Migrate `datalab_*` capabilities to call MANNA (flag per tool); keep native `datalab_client` path selectable.
4. Prove parity + benchmark ≥ baseline; then ALMA/NRAO TAP.
5. Build the producer server (`adapters/mcp/server.py`) exposing Quasar capabilities to external hosts.
6. Replace `exec()` user tools with "connect your own MCP server."
