import test from "node:test";
import assert from "node:assert/strict";

import {
    normalizeWebDecision,
    normalizeWebSearchMode,
    webDecisionLabel,
    webDecisionTitle,
} from "../src/lib/web-citations.js";

// Web search redesign, Phase 2 (PLAN 2.4): the off|auto|always mode and the
// web_decision badge helpers are pure so node tests cover them.

test("web search mode normalises to off | auto | always, default auto", () => {
    assert.equal(normalizeWebSearchMode("always"), "always");
    assert.equal(normalizeWebSearchMode(" OFF "), "off");
    assert.equal(normalizeWebSearchMode("auto"), "auto");
    assert.equal(normalizeWebSearchMode(undefined), "auto");
    assert.equal(normalizeWebSearchMode("bogus"), "auto");
    assert.equal(normalizeWebSearchMode(null, "off"), "off");
});

test("a searched decision reads 'Searched the web (N queries)'", () => {
    const d = { need_web: true, queries: ["ALMA Cycle 13 proprietary period", "ALMA Cycle 13 proposer's guide"], mode: "auto", source: "planner", planner_ms: 1240, domain_pack: "alma_policy" };
    assert.equal(webDecisionLabel(d), "Searched the web (2 queries)");
    assert.equal(webDecisionLabel({ need_web: true, queries: ["x"] }), "Searched the web (1 query)");
    assert.equal(webDecisionLabel({ need_web: true }), "Searched the web");
    const title = webDecisionTitle(d);
    assert.match(title, /"ALMA Cycle 13 proprietary period"/);
    assert.match(title, /official sites: alma_policy/);
    assert.match(title, /planner 1\.2 s/);
});

test("a skipped decision reads 'Web skipped: <reason>'", () => {
    assert.equal(webDecisionLabel({ need_web: false, reason: "archive or paper tools answer this" }), "Web skipped: archive or paper tools answer this");
    assert.equal(webDecisionLabel({ need_web: false }), "Web skipped");
});

test("malformed events are ignored and long fields are bounded", () => {
    assert.equal(normalizeWebDecision(null), null);
    assert.equal(normalizeWebDecision({ reason: "no need_web" }), null);
    assert.equal(webDecisionLabel("nope"), "");
    const d = normalizeWebDecision({ need_web: true, queries: ["a", "", "b", "c", "d"], reason: "r".repeat(500), follow_up: 1 });
    assert.deepEqual(d.queries, ["a", "b", "c"]);
    assert.equal(d.reason.length, 160);
    assert.equal(d.follow_up, true);
});
