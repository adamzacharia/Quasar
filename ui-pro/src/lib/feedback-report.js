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

// The report form defaults to sharing the question, the answer and the
// surrounding turns. With the box unticked by default, most reports arrived as
// "this answer is wrong" with no question, no answer and no context, which the
// maintainer could not act on. The reporter can still untick it; the server
// stores conversation text only when this flag is true.
export function defaultIncludeContext() {
    return true;
}

export const CONTEXT_CONSENT_LABEL =
    "Share the entire conversation: every question, answer, available tool call/result and diagnostics (recommended). " +
    "A separate copy is kept even if you delete the chat. Limit: 400 messages / 4 MiB; omissions are marked. " +
    "Untick to send feedback without conversation context.";

// Thumbs votes that never become a report are reviewed from these previews,
// so they carry the whole question and a readable slice of the answer.
export const FEEDBACK_PREVIEW_LIMITS = Object.freeze({ prompt: 600, response: 2000 });

// The server keeps at most this much of the client excerpts (MAX_CONTEXT_CHARS
// / MAX_RESPONSE_CHARS in services/issue_report_service.py); sending more is
// wasted upload since consent now defaults on. The server-side excerpt carries
// the full context regardless.
export const REPORT_EXCERPT_LIMITS = Object.freeze({ prompt: 2000, response: 6000 });

export function clipReportExcerpt(text, limit) {
    const max = Math.max(0, limit | 0);
    const s = String(text || "");
    if (s.length <= max) return s;
    let end = max;
    // Never split a UTF-16 surrogate pair (an emoji straddling the boundary):
    // a lone surrogate survives JSON encoding, but SQLite cannot UTF-8 encode
    // it and the whole report insert would fail.
    const code = s.charCodeAt(end - 1);
    if (end > 0 && code >= 0xd800 && code <= 0xdbff) end -= 1;
    return s.slice(0, end);
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
