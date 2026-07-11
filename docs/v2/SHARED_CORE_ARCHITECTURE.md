# SHARED_CORE_ARCHITECTURE.md

**The design that makes dual-path possible without duplication.** Read `MONOLITH_BOTTLENECK_ANALYSIS.md` first.

## 1. The core idea: ports and adapters (hexagonal)

One capability is defined **once** as a pure function and exposed through **two thin adapters**:

```
                       ┌─────────────────────────────┐
   native tool  ──────►│                             │
   (in-process)        │   capabilities/<domain>.py  │──────► integrations / services
                       │   (SHARED CORE — the logic) │        (TAP engine, ADS client, …)
   MCP tool     ──────►│                             │
   (FastMCP)           └─────────────────────────────┘
```

- **Capability = the logic.** "Search ALMA," "run a Data Lab ADQL query," "resolve a name." Knows nothing about MCP or the agent. Input: a typed model + `CallContext`. Output: a `ToolResult`.
- **Native adapter** registers the capability as a `core.tools.Tool` (JSON schema from the capability's pydantic input model).
- **MCP adapter** registers the *same* capability as a FastMCP `@mcp.tool()`.
- Both adapters are ~10 lines: validate input → call capability → serialize output. **No business logic in adapters.** This is literally how "share logic instead of duplicating" is enforced.

**Consequence:** extracting logic into capabilities *is* the monolith fix; adding the MCP path is then just a second adapter. The two headline goals collapse into one body of work.

## 2. What lives where

| Concern | Shared core (`capabilities/`) | MCP-specific (`adapters/mcp/`) | Tool-specific (`adapters/native/`) |
|---|---|---|---|
| The actual query/computation logic | ✅ | — | — |
| Input validation model (pydantic) | ✅ (defined once) | reuses it → JSON schema | reuses it → JSON schema |
| `ToolResult` envelope construction | ✅ | — | — |
| Provenance, warnings, degraded flags | ✅ | — | — |
| Transport (JSON-RPC / streamable HTTP) | — | ✅ | — |
| MCP progress notifications, resources, annotations | — | ✅ | — |
| Per-session MCP auth / OAuth | — | ✅ | — |
| In-process registration + intent-subsetting | — | — | ✅ |
| SSE `data`/`image`/`plotly` event emission | — | — | ✅ (via serializer) |
| Result-handle store (`result_id`) | ✅ interface; impl co-located with executor | server owns it if server executes | agent owns it if native executes |
| Orchestration (DAG/complexity/recovery/synthesis) | stays in `core/` (neither adapter) | — | — |
| Safety governor / SQL policy | ✅ in core; also runs server-side in MANNA (defense in depth) | mirrors it | uses it |

**Rule of thumb:** if it talks about *what the science operation is*, it's a capability. If it talks about *how bytes move or how a UI renders*, it's an adapter.

## 3. The three contracts (build these first — P1 step 1)

### 3a. `ToolResult` (`capabilities/base.py`)
The canonical envelope every capability returns. Replaces bare pandas DataFrames and the `__eager_data__` thread-local hack.

```python
class Provenance(BaseModel):
    service: str            # "datalab" | "alma" | "ads" | ...
    endpoint: str | None    # actual URL/TAP endpoint hit
    query: str | None       # the exact ADQL/SQL/param string executed
    retrieved_at: str       # ISO timestamp (passed in via CallContext, not Date.now)
    rowcount: int | None

class ToolResult(BaseModel):
    success: bool
    data: Any | None = None            # records (list[dict]) or scalar/artifact ref
    columns: list[dict] | None = None  # [{name, dtype, role}] role∈{link,preview,target,project,...}
    warnings: list[str] = []
    error: str | None = None           # typed error message; NEVER fabricated data
    degraded: bool = False             # served a fallback / partial result
    provenance: Provenance
    pagination_cursor: str | None = None
    reproducible_snippet: str | None = None  # e.g. the astroquery/pyvo line that reproduces this
```

### 3b. `CallContext` (`capabilities/base.py`)
Per-request state passed **explicitly** (kills thread-locals and `os.environ` token injection).

```python
class CallContext(BaseModel, arbitrary_types_allowed=True):
    user_id: str | None
    tokens: dict[str, str]     # per-session archive tokens (datalab, ads, alma) — NOT process env
    result_store: ResultStore  # result_id → prior ToolResult (replaces "LAST results")
    emit: Callable | None      # optional progress/status emitter (SSE or MCP progress)
    budget: Budget             # token/time/row budgets
    trace_id: str
    now: str                   # injected timestamp for reproducibility
```

### 3c. `Capability` protocol (`capabilities/base.py`)

```python
class Capability(Protocol):
    name: str                        # canonical tool name (no transport prefix)
    description: str                 # LLM-facing description
    category: str                    # for intent-subsetting + planner grouping
    InputModel: type[BaseModel]      # one pydantic model → JSON schema for both adapters
    annotations: dict                # SLA hint, read-only flag, cost class (→ MCP annotations)
    def run(self, inp: BaseModel, ctx: CallContext) -> ToolResult: ...
```

## 4. The adapters (thin)

**Native adapter** (`adapters/native/__init__.py`):
```python
def register_all(registry, capabilities):
    for cap in capabilities:
        registry.register(Tool(
            name=cap.name,
            description=cap.description,
            parameters=cap.InputModel.model_json_schema(),
            category=cap.category,
            fn=lambda args, ctx, cap=cap: cap.run(cap.InputModel(**args), ctx),
        ))
```

**MCP adapter** (`adapters/mcp/server.py`, FastMCP):
```python
mcp = FastMCP("quasar")
for cap in capabilities:
    @mcp.tool(name=cap.name, description=cap.description)
    def _tool(cap=cap, **args):
        ctx = build_call_context_from_mcp_session()   # per-session tokens/auth
        result = cap.run(cap.InputModel(**args), ctx)
        return result.model_dump()                     # MCP content blocks
```

Both call `cap.run(...)`. That's the whole point.

## 5. Result handles (`result_id`) — the state fix

V1 chains tools via `self.last_search_results` (a thread-local), which breaks across Conductor threads and is impossible over stateless MCP. The `datalab_*` family already does it right: a search returns a `result_id`; follow-up tools (`filter`, `check_line_coverage`, `plot`) take that `result_id` and look it up in a `ResultStore`.

**V2 rule:** every capability that produces a table returns a `result_id` in its `ToolResult`; every follow-up capability takes a `result_id` input. The `ResultStore` lives in `CallContext`. When a capability executes **server-side in MANNA**, MANNA owns the store and exposes `get_rows/to_csv/stats` follow-ups; Quasar just passes the handle through.

## 6. Where orchestration sits (neither adapter)

The Conductor/DAG/complexity/recovery/synthesis brain stays in `core/` and calls capabilities through the registry. It is **transport-agnostic**: it sees `Tool`s in the registry whether they're backed by a native capability or a bridged MCP tool. This is why the planner must generate its tool list from the **live registry** (not a hardcoded whitelist) — with both paths active, the tool set is dynamic.

## 7. Migration invariant

At every step, **exactly one implementation** of each capability exists. A family is "migrated" when: (1) its logic is in `capabilities/<domain>.py`, (2) a native adapter registers it, (3) an MCP adapter exposes it, (4) both produce equivalent `ToolResult`s (parity test), (5) the benchmark is ≥ baseline. Duplication between the MCP and tool paths is a bug, not a stage.
