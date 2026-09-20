import test from "node:test";
import assert from "node:assert/strict";

import {
    CONTEXT_CONSENT_LABEL,
    FEEDBACK_PREVIEW_LIMITS,
    REPORT_EXCERPT_LIMITS,
    canSubmitIssueReport,
    clipReportExcerpt,
    defaultIncludeContext,
    shouldOpenIssueReport,
    splitProviderChunk,
} from "../src/lib/feedback-report.js";

test("report form shares question, answer and context by default (opt-out)", () => {
    // A report without the question and answer is unactionable for the
    // maintainer; the default is on, the label says what is shared, and the
    // reporter can still untick it.
    assert.equal(defaultIncludeContext(), true);
    assert.match(CONTEXT_CONSENT_LABEL, /question/i);
    assert.match(CONTEXT_CONSENT_LABEL, /answer/i);
    assert.match(CONTEXT_CONSENT_LABEL, /diagnostics/i);
});

test("thumbs previews carry the whole question and a readable answer slice", () => {
    assert.equal(FEEDBACK_PREVIEW_LIMITS.prompt, 600);
    assert.equal(FEEDBACK_PREVIEW_LIMITS.response, 2000);
    assert.ok(Object.isFrozen(FEEDBACK_PREVIEW_LIMITS));
});

test("report excerpts are clipped to what the server keeps before upload", () => {
    // Mirrors MAX_CONTEXT_CHARS / MAX_RESPONSE_CHARS in services/issue_report_service.py.
    assert.equal(REPORT_EXCERPT_LIMITS.prompt, 2000);
    assert.equal(REPORT_EXCERPT_LIMITS.response, 6000);
    assert.equal(clipReportExcerpt("x".repeat(7000), REPORT_EXCERPT_LIMITS.response).length, 6000);
    assert.equal(clipReportExcerpt("short", REPORT_EXCERPT_LIMITS.prompt), "short");
    assert.equal(clipReportExcerpt(null, 10), "");
    assert.equal(clipReportExcerpt("abc", -5), "");
});

test("clipping never splits a surrogate pair at the boundary", () => {
    // An emoji is two UTF-16 code units; cutting between them yields a lone
    // surrogate that SQLite cannot store. Cut before the pair instead.
    const text = "a".repeat(1999) + "😀" + "tail";
    const clipped = clipReportExcerpt(text, 2000);
    assert.equal(clipped.length, 1999);
    assert.ok(!/[\uD800-\uDBFF]$/.test(clipped));
    // A boundary that does not land inside a pair is untouched.
    assert.equal(clipReportExcerpt("a".repeat(2000) + "😀", 2000).length, 2000);
    assert.equal(clipReportExcerpt("😀", 2), "😀");
});

test("issue form opens only for thumbs-down feedback", () => {
    assert.equal(shouldOpenIssueReport("dislike"), true);
    assert.equal(shouldOpenIssueReport("like"), false);
    assert.equal(shouldOpenIssueReport(null), false);
});

test("report submission requires a description and linked run", () => {
    assert.equal(canSubmitIssueReport("The run was stuck", "run-1"), true);
    assert.equal(canSubmitIssueReport("   ", "run-1"), false);
    assert.equal(canSubmitIssueReport("The run was stuck", ""), false);
});

test("large TACC chunks are smoothed without changing text", () => {
    const content = "x".repeat(160);
    const pieces = splitProviderChunk("tacc", content);
    assert.ok(pieces.length > 1);
    assert.equal(pieces.join(""), content);
    assert.deepEqual(splitProviderChunk("openai", content), [content]);
});
