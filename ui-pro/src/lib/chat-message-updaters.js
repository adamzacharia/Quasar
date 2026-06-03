export function findLastAssistantTextIndex(messages) {
    for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i]?.role === "assistant" && messages[i]?.type === "text") {
            return i;
        }
    }
    return -1;
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
        thinkingDuration: durationSeconds,
    };
    return { messages: next, didAttach: true };
}
