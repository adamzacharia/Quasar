export const ISSUE_REPORT_CATEGORIES = Object.freeze([
    "stuck_slow",
    "wrong_answer",
    "incorrect_data",
    "ui_problem",
    "other",
]);

export function shouldOpenIssueReport(feedbackType) {
    return feedbackType === "dislike";
}

export function canSubmitIssueReport(description, runId) {
    return Boolean(String(description || "").trim() && String(runId || "").trim());
}

export function splitProviderChunk(provider, content, chunkSize = 32) {
    const text = String(content || "");
    if (provider !== "tacc" || text.length <= 80) return [text];
    const pieces = [];
    for (let index = 0; index < text.length; index += chunkSize) {
        pieces.push(text.slice(index, index + chunkSize));
    }
    return pieces;
}
