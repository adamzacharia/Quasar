import test from "node:test";
import assert from "node:assert/strict";

import {
    canSubmitIssueReport,
    shouldOpenIssueReport,
    splitProviderChunk,
} from "../src/lib/feedback-report.js";

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
