import test from "node:test";
import assert from "node:assert/strict";

import {
    isFailedTurn,
    isRateableBlock,
    lastAssistantTurn,
    nudgeText,
    shouldBlockSend,
    unratedBlocks,
} from "../src/lib/eval-mode.js";

const user = (content) => ({ role: "user", content, type: "text" });

/** Blocks whose BlockRating widget mounted, i.e. that are actually on screen.
 *  Anything absent here rendered nothing and therefore cannot gate. */
const seen = (...ids) => Object.fromEntries(ids.map((id) => [id, 1]));

const block = (kind, blockId, extra = {}) => ({
    role: "assistant",
    type: kind,
    blockKind: kind,
    blockId,
    content: kind === "text" ? "an answer" : "",
    ...extra,
});

test("a turn is the assistant messages after the last user message", () => {
    const messages = [
        user("first"), block("text", "t1"),
        user("second"), block("text", "t2"), block("data", "d2"),
    ];
    assert.deepEqual(lastAssistantTurn(messages).map((m) => m.blockId), ["t2", "d2"]);
});

test("no messages means no turn and no gate", () => {
    assert.deepEqual(lastAssistantTurn([]), []);
    assert.equal(shouldBlockSend(true, [], {}, {}), false);
});

test("gate is off entirely when eval mode is off", () => {
    const messages = [user("q"), block("data", "d1")];
    assert.equal(shouldBlockSend(false, messages, {}, seen("d1")), false);
    assert.equal(shouldBlockSend(true, messages, {}, seen("d1")), true);
});

test("gate blocks until every visible block of the last turn is rated", () => {
    const messages = [user("q"), block("text", "t1"), block("data", "d1"), block("plotly", "p1")];
    const visible = seen("t1", "d1", "p1");

    assert.equal(shouldBlockSend(true, messages, {}, visible), true);
    assert.equal(shouldBlockSend(true, messages, { t1: 5 }, visible), true);
    assert.equal(shouldBlockSend(true, messages, { t1: 5, d1: 4 }, visible), true);
    assert.equal(shouldBlockSend(true, messages, { t1: 5, d1: 4, p1: 2 }, visible), false);
});

test("only the LAST turn gates — earlier unrated turns don't wedge the composer", () => {
    const messages = [
        user("old"), block("data", "old-d"),
        user("new"), block("data", "new-d"),
    ];
    assert.equal(shouldBlockSend(true, messages, { "new-d": 3 }, seen("old-d", "new-d")), false);
});

test("an infrastructure-error turn is exempt", () => {
    // The backend never produced an answer, so there is nothing to judge —
    // gating here would wedge the composer with no way out.
    const messages = [
        user("q"),
        block("text", "t1", { runMeta: { run_id: "r1", status: "failed" } }),
    ];
    assert.equal(isFailedTurn(lastAssistantTurn(messages)), true);
    assert.equal(shouldBlockSend(true, messages, {}, seen("t1")), false);
});

test("a completed turn is not exempt", () => {
    const messages = [user("q"), block("text", "t1", { runMeta: { run_id: "r1" } })];
    assert.equal(isFailedTurn(lastAssistantTurn(messages)), false);
    assert.equal(shouldBlockSend(true, messages, {}, seen("t1")), true);
});

// ── The invisible-block trap ─────────────────────────────────────
// A block can hold a perfectly good stable id and still render nothing. Each of
// these would wedge the composer forever if the gate demanded a star on it:
// unrated, unrateable, unsendable. The widget registering itself is what makes
// that impossible — no widget, no gate.

test("an empty text block never gates — it renders no widget", () => {
    // A table-only answer still carries a text block with no prose;
    // ChatMessage skips it, so its BlockRating never mounts.
    const messages = [user("q"), block("text", "t1", { content: "" }), block("data", "d1")];
    const visible = seen("d1");
    assert.deepEqual(unratedBlocks(messages, {}, visible).map((m) => m.blockId), ["d1"]);
    assert.equal(shouldBlockSend(true, messages, { d1: 4 }, visible), false);
});

test("a 0-row data card never gates — ChatMessage suppresses it", () => {
    // Real case, caught in local verification: an ALMA search persisted an
    // empty table. The card has a stable blockId but renders nothing.
    const messages = [user("q"), block("text", "t1"), block("data", "empty-d")];
    const visible = seen("t1");
    assert.deepEqual(unratedBlocks(messages, {}, visible).map((m) => m.blockId), ["t1"]);
    assert.equal(shouldBlockSend(true, messages, { t1: 3 }, visible), false);
});

test("a papers grid absorbed by the observation graph never gates", () => {
    // ChatMessage returns null for the papers branch when a graph replaced it.
    const messages = [user("q"), block("text", "t1"), block("papers", "absorbed-p")];
    assert.equal(shouldBlockSend(true, messages, { t1: 4 }, seen("t1")), false);
});

test("a block that DID mount a widget gates", () => {
    const messages = [user("q"), block("data", "d1")];
    assert.equal(shouldBlockSend(true, messages, {}, seen("d1")), true);
});

test("a block with no id never gates — a rating could not be keyed to it", () => {
    // Turns that predate stable ids are unrateable, not blocking.
    const legacy = { role: "assistant", type: "data", content: "" };
    assert.equal(isRateableBlock(legacy), false);
    assert.equal(shouldBlockSend(true, [user("q"), legacy], {}, {}), false);
});

test("web_sources is not a rateable block", () => {
    const messages = [user("q"), block("web_sources", "w1"), block("text", "t1")];
    assert.equal(isRateableBlock(messages[1]), false);
    assert.deepEqual(unratedBlocks(messages, {}, seen("w1", "t1")).map((m) => m.blockId), ["t1"]);
});

test("falls back to message.type when blockKind is absent", () => {
    assert.equal(isRateableBlock({ role: "assistant", type: "data", blockId: "d1", content: "" }), true);
});

test("user messages are never rateable", () => {
    assert.equal(isRateableBlock({ role: "user", type: "text", blockId: "u1", content: "hi" }), false);
});

test("an in-flight rating does not open the gate (CX-02)", () => {
    // The star lights optimistically from pendingRatings, but the gate reads only
    // the CONFIRMED map. If in-flight counted, a user could star a block and send
    // before the POST landed — and a later failure would leave the label set with
    // a hole exactly where the gate promised there could not be one.
    const messages = [user("q"), block("data", "d1")];
    const confirmedNothing = {};          // POST still in flight
    assert.equal(shouldBlockSend(true, messages, confirmedNothing, seen("d1")), true);
    assert.equal(shouldBlockSend(true, messages, { d1: 4 }, seen("d1")), false);
});

test("a turn marked failed by a terminal run_meta is exempt (CX-01)", () => {
    // The agent-is-None path streams its traceback as `token` text and ends with
    // an ordinary [DONE] — no error event, so onError never fires. sse.py now
    // re-emits run_meta with the settled status, which lands here.
    const messages = [
        user("q"),
        block("text", "t1", { runMeta: { run_id: "r1", status: "failed", errorCode: "agent_unavailable" } }),
    ];
    assert.equal(isFailedTurn(lastAssistantTurn(messages)), true);
    assert.equal(shouldBlockSend(true, messages, {}, seen("t1")), false);
});

test("a zero rating does not count as rated", () => {
    // Ratings are 1-5; a 0 means the server rejected it.
    const messages = [user("q"), block("data", "d1")];
    assert.equal(shouldBlockSend(true, messages, { d1: 0 }, seen("d1")), true);
});

test("nudge names what is still missing", () => {
    const one = unratedBlocks([user("q"), block("data", "d1")], {}, seen("d1"));
    assert.match(nudgeText(one), /rate the data block/);

    const many = unratedBlocks(
        [user("q"), block("data", "d1"), block("text", "t1")], {}, seen("d1", "t1"),
    );
    assert.match(nudgeText(many), /2 unrated blocks/);

    assert.equal(nudgeText([]), "");
});
