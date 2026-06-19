import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

const globals = fs.readFileSync(new URL("../src/app/globals.css", import.meta.url), "utf8");
const chatArea = fs.readFileSync(new URL("../src/components/ChatArea.tsx", import.meta.url), "utf8");
const chatInput = fs.readFileSync(new URL("../src/components/ChatInput.tsx", import.meta.url), "utf8");

test("chat content and composer use fluid viewport-aware widths", () => {
    assert.match(
        globals,
        /--q-chat-content-width:\s*calc\(100%\s*-\s*clamp\(/,
        "chat content should expand with the available main pane",
    );
    assert.match(
        globals,
        /--q-chat-input-width:\s*calc\(100%\s*-\s*clamp\(/,
        "composer should expand with the available main pane",
    );
    assert.match(chatArea, /w-full max-w-\[var\(--q-chat-content-width\)\]/);
    assert.match(chatInput, /w-full max-w-\[var\(--q-chat-input-width\)\]/);
});

test("large-screen breakpoints do not replace fluid chat widths with fixed rem caps", () => {
    assert.equal((globals.match(/--q-chat-content-width:/g) ?? []).length, 1);
    assert.equal((globals.match(/--q-chat-input-width:/g) ?? []).length, 1);
});
