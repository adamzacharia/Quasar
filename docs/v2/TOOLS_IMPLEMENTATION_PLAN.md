# TOOLS_IMPLEMENTATION_PLAN.md

**How the direct/native tool path is preserved and improved — sharing logic with MCP, never duplicating.** Prereq: `SHARED_CORE_ARCHITECTURE.md`.

## 1. Principle

The native tool path is **not** a legacy branch to be tolerated; it is a **first-class transport** over the same capability layer. After the shared core exists, a "native tool" is just `adapters/native` registering a capability as a `core.tools.Tool`. The tool path and the MCP path differ **only** in transport and in a few native-only concerns (in-process latency, SSE emission, intent-subsetting).

**Non-duplication is enforced structurally:** both adapters call `cap.run(inp, ctx)`. If you ever find ALMA/DataLab query logic in `adapters/native` OR in the MCP server, that's the bug — move it to `capabilities/`.

## 2. Why keep it (do not delete the native path)

- **Latency:** no network hop — correct for sub-second/chatty/UI-coupled calls (hips2fits thumbnails, deep-link builders, data-card shaping, client-side Aladin cutouts).
- **Trust boundary:** safety governor, provenance, citation verification, quotas, auth run on Quasar's side.
- **Continuity:** quasarassistant.com is live and must keep working through the entire MCP migration.
- **Offline / non-MCP providers:** local dev and LLM providers without hosted-MCP support still need a native path.

## 3. What changes for the native path (improvements)

### 3a. Thin adapter, no logic
Rebuild native registration as `adapters/native/register_all(registry, capabilities)` (see `SHARED_CORE_ARCHITECTURE.md` §4). Delete the inline `_register_tools()` bodies from `agent.py` as each family migrates.

### 3b. Result handles replace thread-locals (correctness)
Generalize the `datalab_*` `result_id` pattern registry-wide. Kill `self.last_search_results` (thread-local, approx `agent.py:602–608`). The 15+ "operate on LAST results" tools (`filter_results`, `check_co_lines`, `check_line_coverage`, `get_mast_products`, `download_mast_data`, `plot_alma_results`, `plot_sky_map`) take a `result_id` input and read from `CallContext.result_store`. This simultaneously fixes cross-thread bugs and makes native tools MCP-portable.

### 3c. Intent-based tool subsetting (cost + accuracy)
Native-only optimization: stop sending all ~145 tool schemas every round. `core/router.py` classifies the query into 2–3 capability categories; the registry returns only those schemas + a `search_tools` escape hatch. (MCP hosts do their own selection; this is a Quasar-agent concern.)

### 3d. SSE serialization from `ToolResult`
Native tools no longer stash pandas DataFrames in thread-locals for the SSE loop to find. The serializer (`api/serializers/`) turns a `ToolResult` into the existing `data`/`image`/`plotly`/`papers` SSE events. The frontend contract is unchanged.

### 3e. Delete `exec()` user tools
The per-user `exec()` tool feature (`services/user_tools_service.py`) is RCE and is superseded by "connect your own MCP server." Remove it (disabled in P0, deleted here). Users who want custom tools register an MCP server via the existing `/api/mcp-servers` UI.

## 4. Shared-logic checklist (per capability)

For each migrated family, verify:
- [ ] Logic lives only in `capabilities/<domain>.py`.
- [ ] One pydantic `InputModel` → schema for both adapters.
- [ ] Returns `ToolResult` with provenance + `reproducible_snippet`.
- [ ] Native adapter registers it; MCP adapter exposes it; **parity test passes**.
- [ ] Uses `result_id` handles for any table it produces/consumes.
- [ ] No `os.environ` token reads (uses `CallContext.tokens`); no thread-locals; no `print()`.
- [ ] No fabricated-data fallback (typed error instead).

## 5. Integrations consolidation (supports the native path)

The native tools call `integrations/` clients. During/after the split, collapse duplication (see `services`/`integrations` analyses):
- **One IVOA-TAP engine** (pyvo, timeout-injected, retrying) parameterized by endpoint + ObsCore dialect — replaces the ~5 near-duplicate TAP clients (`tap.py` NRAO, `alminer_client.py` ALMA, `eso_tap_client.py`, the inline CADC path in `agent.py`, and the Data Lab REST client).
- **One SIMBAD resolver** with a TTL cache (fixes the `lru_cache` permanent-`None` bug where a transient SIMBAD outage poisons a target until restart).
- **One ADS client** (merge `ads_service`, `paper_search`, `citation_verifier`'s lookups).
- **One shared transport core**: `requests.Session` pooling, default timeouts, exponential backoff + `Retry-After`, structured logging, and the `_TimeoutHTTPSession` pyvo-timeout injection (required regardless).

These are capabilities' dependencies, so consolidating them benefits both paths.

## 6. Selection policy (which path runs a given tool)

A per-tool-family **feature flag** decides native vs MCP at runtime:
- Default during migration: native ON, MCP OFF, flip per family as parity is proven.
- Long-term default: archive/literature/compute families → MCP (MANNA/mesh); latency/UI-coupled families → native.
- The registry presents a unified tool list to the planner regardless of which path backs each tool. The planner never needs to know the transport.

## 7. Deprecation policy

Nothing is deleted until its capability has both adapters and passes parity + benchmark. The native path is **improved and retained**, not sunset. Only the genuinely broken/dangerous pieces (`exec()` tools, thread-local chaining, duplicate methods, fabrication) are removed.
