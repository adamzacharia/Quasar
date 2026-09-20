import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createRequire, Module } from "node:module";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

// Render the actual TSX component using the project's installed compiler and
// React runtime. No new test framework or browser dependency is required.
const filename = fileURLToPath(new URL("../src/components/FeedbackTranscriptView.tsx", import.meta.url));
const compiled = ts.transpileModule(readFileSync(filename, "utf8"), {
    compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;
const component = new Module(filename);
component.require = createRequire(filename);
component._compile(compiled, filename);
const { FeedbackTranscriptView } = component.exports;
function sample() {
    return { id: "s1", captured_at: "2026-09-05", conversation_id: "c1", run_id: "r1", message_id: "m2",
        model: "test-model", provider: "test-provider", vote: "dislike", feedback_text: "wrong answer",
        limits: { max_messages: 400, max_bytes: 4194304 },
        truncation: { truncated: false, captured_messages: 2, total_messages: 2, reasons: [] },
        messages: [{ id: "1", role: "user", content: "First question", created_at: "time1" },
            { id: "2", role: "assistant", content: "answer ".repeat(2000), is_reported_answer: true,
              metadata: { feedbackToolTrace: { calls: [{ name: "lookup", arguments: { target: "M87" }, output: { count: 12 } }], truncated: false, total_calls: 1 } } }],
    };
}
const render = snapshot => renderToStaticMarkup(React.createElement(FeedbackTranscriptView, { snapshot }));
test("transcript renders every full turn, feedback, identifiers and metadata", () => {
    const s = sample();
    const html = render(s);
    for (const expected of ["Full conversation", "First question", s.messages[1].content, "Reported answer", "wrong answer", "c1", "r1", "test-provider", "400 messages / 4 MiB"]) assert.ok(html.includes(expected), expected);
});
test("tools are collapsed by default and preserve call/results for expansion", () => {
    const html = render(sample());
    assert.match(html, /<details[^>]*><summary[^>]*>Tool 1:/);
    assert.doesNotMatch(html, /<details[^>]*\bopen/);
    assert.ok(html.includes("M87") && html.includes("count") && html.includes("12"));
});
test("truncation and older trace limitations are visible", () => {
    const s = sample();
    s.truncation = { truncated: true, marker: "Truncated at 2 of 401 messages", total_messages: 401, captured_messages: 2, reasons: ["message_limit"] };
    s.messages[1].metadata = { toolTrace: [{ name: "legacy" }] };
    const html = render(s);
    assert.ok(html.includes("Truncated at 2 of 401 messages") && html.includes("result unavailable"));
});
test("transcript escapes untrusted HTML", () => {
    const s = sample();
    s.messages[0].content = '<img src=x onerror="alert(1)">';
    const html = render(s);
    assert.ok(html.includes("&lt;img"));
    assert.ok(!html.includes("<img src=x"));
});

test("interrupted turns explain partial tool evidence", () => {
    const s = sample();
    s.messages[1].metadata.feedbackToolTrace.partial = true;
    s.messages[1].metadata.feedbackToolTrace.run_status = "cancelled";
    const html = render(s);
    assert.ok(html.includes("Partial tool evidence: run cancelled"));
    assert.ok(html.includes("calls still running then have no captured result"));
});
