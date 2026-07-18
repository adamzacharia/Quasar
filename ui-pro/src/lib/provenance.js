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
    if (request && request.text) return [{ name: toolName || "", request }];
    if (!Array.isArray(calls) || calls.length === 0) return [];
    return calls
        .filter((c) => c && c.request && c.request.text && (!toolName || c.name === toolName))
        .map((c) => ({ name: c.name, request: c.request }));
}

/** The literal text a copy click puts on the clipboard for one entry. */
export function copyPayload(entry) {
    if (!entry || !entry.request) return "";
    return String(entry.request.text || "");
}
