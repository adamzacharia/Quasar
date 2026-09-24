import test from "node:test";
import assert from "node:assert/strict";

import { answerLinkAttributes } from "../src/lib/answer-markdown.js";

test("external answer links open in a new tab with rel=noopener noreferrer (D8)", () => {
    assert.deepEqual(answerLinkAttributes("https://almascience.org/proposing"), {
        target: "_blank",
        rel: "noopener noreferrer",
    });
    assert.deepEqual(answerLinkAttributes("HTTP://example.com"), {
        target: "_blank",
        rel: "noopener noreferrer",
    });
});

test("footnote anchors and relative links stay in the same tab", () => {
    assert.deepEqual(answerLinkAttributes("#user-content-fn-1"), {});
    assert.deepEqual(answerLinkAttributes("/plots/figure.png"), {});
    assert.deepEqual(answerLinkAttributes(""), {});
    assert.deepEqual(answerLinkAttributes(undefined), {});
    assert.deepEqual(answerLinkAttributes("javascript:alert(1)"), {});
});

test("protocol-relative links are external too (guard CX-05)", () => {
    assert.deepEqual(answerLinkAttributes("//example.org/path"), { target: "_blank", rel: "noopener noreferrer" });
    assert.deepEqual(answerLinkAttributes("///not-a-host"), {});
});
