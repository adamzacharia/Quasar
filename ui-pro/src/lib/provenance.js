/**
 * Query-provenance selection logic (Feature 1).
 *
 * Kept as plain JS beside eval-mode.js so `node --test tests/*.test.mjs` can
 * exercise it without a React harness (CX-24): the card-request-vs-trace
 * narrowing is exactly the SSE/reload consumption contract the UI relies on,
 * and it is pure — QueryProvenance.tsx re-imports it for rendering.
 */

/**
 * The request entries a provenance block should render.
 *
 * Rules (mirrored by QueryProvenance.tsx and asserted by
 * tests/query-provenance.test.mjs):
 *  - A card-level `request` (with copyable text) wins outright — it is
 *    unambiguous, so the trace is ignored.
 *  - Otherwise the turn trace is used, narrowed to `toolName` when given.
 *  - Entries without `request.text` are dropped: text IS the copy payload,
 *    a request without it renders an empty block.
 *
 * @param {object|undefined} request  A single card's own request.
 * @param {Array|undefined}  calls    The turn's tool_trace calls.
 * @param {string|undefined} toolName Narrow the trace to one tool.
 * @returns {{name: string, request: object}[]}
 */
export function requestsFrom(request, calls, toolName) {
    if (request && request.text && isDataQuery(request)) return [{ name: toolName || "", request }];
    if (!Array.isArray(calls) || calls.length === 0) return [];
    return calls
        .filter((c) => c && c.request && c.request.text && isDataQuery(c.request)
            && (!toolName || c.name === toolName))
        .map((c) => ({ name: c.name, request: c.request }));
}

/**
 * "Show query" exists to show the query that pulled DATA from an archive or
 * Data Lab: ADQL/SQL, or the archive HTTP request (cutout, SIA). An ADS
 * literature search, a parameterized service call or plain tool arguments are
 * not that, and a "Show query" under every paper card was noise (user,
 * 2026-10-03). A request without a kind (older persisted traces) still shows.
 */
const DATA_QUERY_KINDS = new Set(["adql", "http"]);

export function isDataQuery(request) {
    if (!request) return false;
    return !request.kind || DATA_QUERY_KINDS.has(request.kind);
}

/** The literal text a copy click puts on the clipboard for one entry. */
export function copyPayload(entry) {
    if (!entry || !entry.request) return "";
    return String(entry.request.text || "");
}
