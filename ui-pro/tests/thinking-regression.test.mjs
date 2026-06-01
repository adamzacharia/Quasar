import test from "node:test";
import assert from "node:assert/strict";

import {
    attachThinkingStepsToLastAssistant,
    sanitizeAssistantContent,
    updateLastAssistantContent,
    updateLastAssistantThinking,
} from "../src/lib/chat-message-updaters.js";

function assistantText(overrides = {}) {
    return {
        id: "assistant-1",
        role: "assistant",
        type: "text",
        content: "",
        timestamp: new Date("2026-01-01T00:00:00Z"),
        ...overrides,
    };
}

test("streamed thinking does not replace the final assistant answer", () => {
    let messages = [
        { id: "user-1", role: "user", type: "text", content: "Question", timestamp: new Date("2026-01-01T00:00:00Z") },
        assistantText(),
    ];

    messages = updateLastAssistantThinking(messages, "Checking sources...");
    messages = updateLastAssistantContent(messages, "Final answer text.");

    assert.equal(messages[1].thinking, "Checking sources...");
    assert.equal(messages[1].content, "Final answer text.");
});

test("late thinking deltas do not hide already streamed final text", () => {
    let messages = [assistantText({ content: "Visible final answer." })];

    messages = updateLastAssistantThinking(messages, "Late thought summary");

    assert.equal(messages[0].content, "Visible final answer.");
    assert.equal(messages[0].thinking, "Late thought summary");
});

test("finalizing thinking attaches to the assistant text message even after source cards", () => {
    const messages = [
        assistantText({ content: "Answer with citations.", thinking: "I checked the evidence." }),
        {
            id: "web-1",
            role: "assistant",
            type: "web_sources",
            content: "",
            timestamp: new Date("2026-01-01T00:00:01Z"),
            webSources: [{ title: "Source", url: "https://example.com", snippet: "" }],
        },
    ];

    const result = attachThinkingStepsToLastAssistant(messages, [
        { text: "Calling web search", status: "running" },
    ]);

    assert.equal(result.messages[0].content, "Answer with citations.");
    assert.equal(result.messages[0].thinking, "I checked the evidence.");
    assert.deepEqual(result.messages[0].thinkingSteps, [
        { text: "Calling web search", status: "completed" },
    ]);
    assert.equal(result.messages[1].type, "web_sources");
    assert.equal(result.messages[1].webSources.length, 1);
});

test("assistant text strips lab-science emoji from streamed content", () => {
    let messages = [assistantText()];

    messages = updateLastAssistantContent(messages, "\u{1F52C} ALMA Band 6 results");

    assert.equal(messages[0].content, "ALMA Band 6 results");
    assert.equal(sanitizeAssistantContent("\u{1F9D1}\u200D\u{1F52C} User"), "User");
});
