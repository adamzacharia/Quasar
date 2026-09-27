/**
 * Reloaded messages must carry the same `type` as live ones.
 *
 * The backend persists every row with message_type "general" (its default),
 * while the live stream creates plain answers as "text". Everything that finds
 * "the answer of this turn" keys on "text", so a reloaded grounded answer lost
 * its [W#] chips and sources strip and its web_sources card fell back to the
 * full view. serverMessageToLocal (store.ts) normalizes through the helper
 * tested here.
 */
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

import {
    findLastAssistantTextIndex,
    normalizeServerMessageType,
} from "../src/lib/chat-message-updaters.js";
import {
    hasEvidenceIds,
    turnHasTextMessage,
    webSourcesForTurn,
    webSourcesMessageView,
} from "../src/lib/web-citations.js";

const store = fs.readFileSync(new URL("../src/lib/store.ts", import.meta.url), "utf8");

test("plain-answer server types become text, block types pass through", () => {
    assert.equal(normalizeServerMessageType("general"), "text");
    assert.equal(normalizeServerMessageType(""), "text");
    assert.equal(normalizeServerMessageType(undefined), "text");
    assert.equal(normalizeServerMessageType(null), "text");
    assert.equal(normalizeServerMessageType("something_new"), "text");
    for (const t of ["text", "data", "papers", "tool_call", "image", "plotly", "critique", "notebook", "web_sources"]) {
        assert.equal(normalizeServerMessageType(t), t);
    }
});

test("serverMessageToLocal types the base message through the normalizer", () => {
    const fn = store.slice(store.indexOf("function serverMessageToLocal("));
    const base = fn.slice(0, fn.indexOf("const messages: Message[] = [base];"));
    assert.match(base, /type:\s*normalizeServerMessageType\(msg\.type\)/);
    assert.doesNotMatch(base, /msg\.type\s*\|\|\s*"text"/);
});

// Shape of a reloaded grounded turn (conversation a27f1c66...): the server rows
// are typed "general", and serverMessageToLocal appends the web_sources block
// rebuilt from metadata.
function reloadedGroundedTurn(normalize) {
    const type = (t) => (normalize ? normalizeServerMessageType(t) : t);
    return [
        { role: "user", type: type("general"), content: "How does the ALMA pipeline choose cell size?" },
        { role: "assistant", type: type("general"), content: "Five pixels per beam [W1], robust 0.5 [W7]." },
        {
            role: "assistant",
            type: "web_sources",
            webSources: [
                { id: "W1", url: "https://almascience.nrao.edu/a", cited: true },
                { id: "W7", url: "https://almascience.nrao.edu/b", cited: true },
            ],
        },
    ];
}

test("regression: an un-normalized reload loses the grounded answer", () => {
    const messages = reloadedGroundedTurn(false);
    assert.equal(turnHasTextMessage(messages, 2), false);
    assert.equal(webSourcesMessageView(messages, 2), "full");
    assert.equal(findLastAssistantTextIndex(messages), -1);
});

test("a normalized reload renders like the live turn", () => {
    const messages = reloadedGroundedTurn(true);
    // ChatArea: the answer is the turn's text message and gets the W# sources
    assert.equal(messages[1].type, "text");
    assert.equal(findLastAssistantTextIndex(messages), 1);
    assert.ok(hasEvidenceIds(webSourcesForTurn(messages, 1)));
    assert.deepEqual(webSourcesForTurn(messages, 1).map((s) => s.id), ["W1", "W7"]);
    // the separate sources card is hidden (sources live inside the answer)
    assert.equal(turnHasTextMessage(messages, 2), true);
    assert.equal(webSourcesMessageView(messages, 2), "hide");
    const withImages = messages.map((m, i) => (i === 2 ? { ...m, webImages: [{ url: "https://x.org/i.jpg" }] } : m));
    assert.equal(webSourcesMessageView(withImages, 2), "images-only");
});
