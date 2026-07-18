/**
 * Eval-mode gate logic (Feature 4).
 *
 * Kept as plain JS beside feedback-report.js so `node --test tests/*.test.mjs`
 * can exercise it without a React or zustand harness — the gate is the part
 * with the sharp edges, so it is the part that needs tests.
 */

/** Card kinds that carry a rating. Mirrors types.ts::RATEABLE_BLOCK_KINDS and
 *  services/block_identity.py::RATEABLE_BLOCK_KINDS. `web_sources` is absent on
 *  purpose: it is a provider-attribution strip, not a science block. */
export const RATEABLE_KINDS = ["text", "data", "plotly", "image", "papers", "notebook"];

/**
 * The blocks belonging to the most recent assistant turn.
 *
 * A "turn" is the contiguous run of assistant messages after the last user
 * message — the flat message model has no turn object, so the boundary has to
 * be recovered positionally.
 */
export function lastAssistantTurn(messages) {
    if (!Array.isArray(messages) || messages.length === 0) return [];
    let lastUserIndex = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i] && messages[i].role === "user") {
            lastUserIndex = i;
            break;
        }
    }
    return messages.slice(lastUserIndex + 1).filter((m) => m && m.role === "assistant");
}

/**
 * True when this turn ended in an infrastructure error.
 *
 * Such turns are EXEMPT from the gate: the user cannot meaningfully rate an
 * answer the backend never produced, and gating on one would wedge the
 * composer with no way out. `runMeta.status` is persisted from chat_runs.status
 * (sse.py sets it to "failed" on the error path).
 */
export function isFailedTurn(turnBlocks) {
    return (turnBlocks || []).some((m) => m && m.runMeta && m.runMeta.status === "failed");
}

/**
 * Could this block carry a rating at all?
 *
 * Structural checks only. Whether it is actually ON SCREEN is a separate
 * question that only the rendered widget can answer — see `unratedBlocks`.
 */
export function isRateableBlock(message) {
    if (!message || message.role !== "assistant") return false;
    // Pre-Feature-4 turns have no stable id, so a rating could not be keyed to
    // them; they are unrateable rather than blocking.
    if (!message.blockId) return false;
    const kind = message.blockKind || message.type;
    return RATEABLE_KINDS.includes(kind);
}

/**
 * Blocks of the last turn still awaiting a rating.
 *
 * `ratings` is a blockId -> rating (1-5) map. `visibleBlocks` is a
 * blockId -> mount-count map maintained by the BlockRating widgets themselves.
 *
 * THE INVISIBLE-BLOCK TRAP: a block can hold a perfectly good stable id and
 * still render nothing — an empty prose bubble on a table-only answer, a 0-row
 * data card (ChatMessage suppresses it), a papers grid swallowed by the
 * ObservationPaperGraph. Demanding a star on one of those wedges the composer
 * forever: unrated, unrateable, unsendable. Gating on what actually mounted a
 * widget makes that impossible by construction, and keeps working when someone
 * adds the next suppression rule to ChatMessage.
 *
 * `visibleBlocks` is omitted by callers that only want the structural view
 * (and by the pre-mount tick), in which case no block is treated as visible.
 */
export function unratedBlocks(messages, ratings, visibleBlocks) {
    const turn = lastAssistantTurn(messages);
    if (turn.length === 0) return [];
    if (isFailedTurn(turn)) return [];
    const scores = ratings || {};
    const visible = visibleBlocks || {};
    return turn.filter(
        (m) => isRateableBlock(m) && visible[m.blockId] && !scores[m.blockId],
    );
}

/**
 * Should handleSend refuse this prompt?
 *
 * Only when eval mode is on AND the last turn left visible, rateable blocks
 * unrated.
 */
export function shouldBlockSend(evalMode, messages, ratings, visibleBlocks) {
    if (!evalMode) return false;
    return unratedBlocks(messages, ratings, visibleBlocks).length > 0;
}

/** Human-readable nudge naming what is still missing. */
export function nudgeText(pending) {
    const n = (pending || []).length;
    if (n === 0) return "";
    const kinds = [...new Set(pending.map((m) => m.blockKind || m.type))];
    const what = kinds.length === 1 ? `${kinds[0]} block` : "blocks";
    return n === 1
        ? `Eval mode: rate the ${what} above before sending your next prompt.`
        : `Eval mode: ${n} unrated ${kinds.length === 1 ? `${kinds[0]} blocks` : "blocks"} above — rate them before sending your next prompt.`;
}
