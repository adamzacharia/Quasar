export function findLastAssistantTextIndex(messages) {
    for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i]?.role === "assistant" && messages[i]?.type === "text") {
            return i;
        }
    }
    return -1;
}

// Block types a server row may legitimately carry. Anything else ("general",
// the backend's default message_type, or a missing value) is a plain answer
// and must become "text": the [W#] chips, the sources strip and the
// last-answer lookups above all key on type "text", so a reloaded turn typed
// "general" rendered raw [W#] tags and the full web_sources card.
const SERVER_MESSAGE_TYPES = new Set([
    "text", "data", "papers", "tool_call", "image", "plotly", "critique", "notebook", "web_sources",
]);

export function normalizeServerMessageType(type) {
    return SERVER_MESSAGE_TYPES.has(type) ? type : "text";
}

export function sanitizeAssistantContent(content) {
    const text = String(content ?? "");
    return text.replace(/(?:\u{1F9D1}\u200D)?\u{1F52C}\s*/gu, "");
}

export function updateLastAssistantContent(messages, content) {
    const index = findLastAssistantTextIndex(messages);
    if (index < 0) return messages;

    const next = [...messages];
    next[index] = { ...next[index], content: sanitizeAssistantContent(content) };
    return next;
}

export function updateLastAssistantThinking(messages, thinking) {
    const index = findLastAssistantTextIndex(messages);
    if (index < 0) return messages;

    const next = [...messages];
    next[index] = { ...next[index], thinking: sanitizeAssistantContent(thinking) };
    return next;
}

export function attachThinkingStepsToLastAssistant(messages, thinkingSteps) {
    const index = findLastAssistantTextIndex(messages);
    if (index < 0) {
        return { messages, didAttach: false };
    }

    const assistantMsg = messages[index];
    const hasThinkingText = Boolean(assistantMsg.thinking);
    if (thinkingSteps.length === 0 && !hasThinkingText) {
        return { messages, didAttach: false };
    }

    const finalSteps = thinkingSteps.map((step) =>
        step.status === "running" ? { ...step, status: "completed" } : step
    );

    const next = [...messages];
    const durationMs = Date.now() - new Date(assistantMsg.timestamp).getTime();
    const durationSeconds = Math.max(1, Math.round(durationMs / 1000));

    next[index] = {
        ...assistantMsg,
        thinkingSteps: finalSteps.length > 0 ? finalSteps : assistantMsg.thinkingSteps,
        // A backend-reported duration (usage event durationMs) is authoritative;
        // the wall-clock fallback measures stream LIFETIME, which drip
        // throttling inflates in background tabs (live P15: 1,149s shown for a
        // 253s turn).
        thinkingDuration: assistantMsg.thinkingDuration || durationSeconds,
    };
    return { messages: next, didAttach: true };
}
