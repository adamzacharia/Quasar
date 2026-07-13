import test from "node:test";
import assert from "node:assert/strict";

import { buildCrossMatchPrompt, dispatchPrefillPrompt, PREFILL_PROMPT_EVENT } from "../src/lib/prompt-dispatch.js";

test("builds an xmatch_user_list SIMBAD prompt with coordinates and label", () => {
    const prompt = buildCrossMatchPrompt({ ra: 10.6847, dec: 41.269, label: "M31" });
    assert.match(prompt, /xmatch_user_list/);
    assert.match(prompt, /catalog="simbad"/);
    assert.match(prompt, /"ra": 10\.6847/);
    assert.match(prompt, /"dec": 41\.269/);
    assert.match(prompt, /\(M31\)/);
});

test("omits the label parenthetical when absent and keeps clean coordinates", () => {
    const prompt = buildCrossMatchPrompt({ ra: 1, dec: 2 });
    assert.doesNotMatch(prompt, /\(\)/);
    assert.match(prompt, /"ra": 1, "dec": 2/);
});

test("returns empty string for invalid coordinates", () => {
    assert.equal(buildCrossMatchPrompt({ ra: NaN, dec: 2 }), "");
    assert.equal(buildCrossMatchPrompt({}), "");
});

test("dispatchPrefillPrompt no-ops without a window (SSR-safe)", () => {
    const saved = globalThis.window;
    delete globalThis.window;
    assert.doesNotThrow(() => dispatchPrefillPrompt("anything"));
    if (saved !== undefined) globalThis.window = saved;
});

test("dispatchPrefillPrompt emits one CustomEvent with the trimmed prompt", () => {
    const events = [];
    const savedWin = globalThis.window;
    const savedCE = globalThis.CustomEvent;
    if (typeof globalThis.CustomEvent === "undefined") {
        globalThis.CustomEvent = class {
            constructor(type, init) { this.type = type; this.detail = init?.detail; }
        };
    }
    globalThis.window = { dispatchEvent: (e) => events.push(e) };

    dispatchPrefillPrompt("  do it  ");
    dispatchPrefillPrompt("   "); // blank → ignored

    globalThis.window = savedWin;
    globalThis.CustomEvent = savedCE;

    assert.equal(events.length, 1);
    assert.equal(events[0].type, PREFILL_PROMPT_EVENT);
    assert.equal(events[0].detail.prompt, "do it");
});
