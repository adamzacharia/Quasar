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
    const hasThinkingText = index >= 0 && Boolean(messages[index].thinking);
    if (thinkingSteps.length === 0 && !hasThinkingText) {
        return { messages, didAttach: false };
    }

    const finalSteps = thinkingSteps.map((step) =>
        step.status === "running" ? { ...step, status: "completed" } : step
    );
    if (index < 0 || finalSteps.length === 0) {
        return { messages, didAttach: hasThinkingText };
    }

    const next = [...messages];
    next[index] = { ...next[index], thinkingSteps: finalSteps };
    return { messages: next, didAttach: true };
}
