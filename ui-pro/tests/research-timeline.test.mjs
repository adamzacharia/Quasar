import test from "node:test";
import assert from "node:assert/strict";

import { buildResearchTimeline, REASONING_VISIBLE, stepIcon, turnPaperCount } from "../src/lib/research-timeline.js";

const kinds = (t) => t.sections.map((s) => s.kind);

test("plumbing steps are hidden", () => {
    const t = buildResearchTimeline({
        running: true,
        steps: [
            { text: "Connecting to QUASAR engine", status: "completed" },
            { text: "Calling gpt-oss-120b", status: "completed" },
            { text: "__data_ready__{}", status: "ready" },
            { text: "🧠 Reasoning", status: "running" },
        ],
    });
    assert.deepEqual(t.sections, []);
    assert.equal(t.current, undefined);
});

test("archive tools become Querying chips with the running one live", () => {
    const t = buildResearchTimeline({
        running: true,
        steps: [
            { text: "Querying ALMA by target", status: "completed" },
            { text: "Searching astronomy literature", status: "running", elapsedSeconds: 20 },
        ],
    });
    assert.deepEqual(kinds(t), ["querying"]);
    const q = t.sections[0];
    assert.equal(q.status, "running");
    assert.equal(q.items[0].icon, "database");
    assert.equal(q.items[1].icon, "book");
    assert.equal(q.items[1].elapsedSeconds, 20);
    assert.equal(t.current, "Searching astronomy literature");
});

test("web query labels become query chips; Reading sits right after searching", () => {
    const t = buildResearchTimeline({
        running: true,
        steps: [
            { text: 'Searching the web: "ALMA cycle 13 deadline"', status: "completed" },
            { text: "Querying ALMA by target", status: "running" },
            { text: "Reading 3 pages", status: "completed" },
        ],
        webSources: [
            { url: "https://almascience.org/news", title: "News" },
            { url: "https://www.almascience.org/other", title: "dup domain" },
            { url: "https://www.eso.org/x", title: "ESO" },
        ],
    });
    assert.deepEqual(kinds(t), ["searching", "reading", "querying"]);
    assert.equal(t.sections[0].items[0].text, "ALMA cycle 13 deadline");
    const reading = t.sections[1];
    assert.deepEqual(reading.items.map((i) => i.text), ["almascience.org", "eso.org"]);
    assert.equal(t.sourceCount, 3); // three pages, two sites
    assert.deepEqual(t.favicons, ["almascience.org", "eso.org"]);
});

test("a query run on official sites and the open web is one chip; any success wins over an error", () => {
    const q = "ALMA Cycle 13 Large Program threshold";
    const t = buildResearchTimeline({
        running: true,
        steps: [
            { text: `Searching the web: "${q}" (official sites)`, status: "error" },
            { text: `Searching the web: "${q}"`, status: "completed" },
            { text: "Searching ALMA Manuals & Documentation", status: "completed" },
        ],
    });
    const s = t.sections.find((x) => x.kind === "searching");
    assert.equal(s.items.length, 1);
    assert.equal(s.items[0].status, "completed");
    assert.equal(t.sections.find((x) => x.kind === "querying").items[0].icon, "book");
});

test("generic web status is replaced by the planner's queries", () => {
    const t = buildResearchTimeline({
        running: true,
        steps: [{ text: "🌐 Query needs current information: searching the web in parallel", status: "running" }],
        webDecision: { need_web: true, queries: ["jwst cycle 5 call", "jwst proposal deadline"] },
    });
    const s = t.sections[0];
    assert.equal(s.kind, "searching");
    assert.deepEqual(s.items.map((i) => i.text), ["jwst cycle 5 call", "jwst proposal deadline"]);
    assert.equal(s.status, "running");
});

test("generic web status is kept when there are no planner queries", () => {
    const t = buildResearchTimeline({
        running: false,
        steps: [{ text: "Calling web search agent", status: "completed" }],
    });
    assert.equal(t.sections[0].kind, "searching");
    assert.equal(t.sections[0].items[0].text, "Calling web search agent");
});

test("reloaded history: duplicates collapse to the last state and nothing stays running", () => {
    const t = buildResearchTimeline({
        running: false,
        steps: [
            { text: "Querying ALMA by target", status: "running" },
            { text: "Querying ALMA by target", status: "completed" },
            { text: "Composing final answer from tool results", status: "running" },
        ],
    });
    assert.deepEqual(kinds(t), ["querying", "wrapping"]);
    assert.equal(t.sections[0].items.length, 1);
    assert.ok(t.sections.every((s) => s.status === "completed"));
    assert.equal(t.stepCount, 2);
});

test("Wrapping up is always last and errors are kept", () => {
    const t = buildResearchTimeline({
        running: true,
        steps: [
            { text: "Composing final answer from tool results", status: "running" },
            { text: "Searching CADC archive", status: "error" },
        ],
    });
    assert.deepEqual(kinds(t), ["querying", "wrapping"]);
    assert.equal(t.sections[0].status, "error");
});

test("post-hoc 'Calling tool:' duplicates are dropped when named labels exist", () => {
    const t = buildResearchTimeline({
        running: false,
        steps: [
            { text: "Querying ALMA by target", status: "completed" },
            { text: "Calling tool: Search By Target", status: "completed" },
        ],
    });
    assert.equal(t.sections[0].items.length, 1);
    const only = buildResearchTimeline({ steps: [{ text: "Calling tool: Search By Target", status: "completed" }] });
    assert.equal(only.sections[0].items[0].text, "Search By Target");
});

test("reasoning keeps every line and marks how many are folded", () => {
    const steps = ["a", "b", "c", "d", "e"].map((x) => ({ text: `💭 thought ${x}`, status: "completed" }));
    const t = buildResearchTimeline({ steps });
    const r = t.sections[0];
    assert.equal(r.kind, "reasoning");
    assert.equal(r.items.length, 5, "no reasoning line is lost");
    assert.equal(r.hiddenCount, 2);
    assert.equal(REASONING_VISIBLE, 3);
});

test("a timed-out web pre-pass is shown as a failure, not hidden", () => {
    const t = buildResearchTimeline({
        running: false,
        steps: [
            { text: "🌐 Query needs current information: searching the web in parallel", status: "error" },
            { text: 'Searching the web: "A"', status: "completed" },
            { text: "Web search timed out after 45s: answered without web results", status: "completed" },
        ],
    });
    const s = t.sections.find((x) => x.kind === "searching");
    assert.ok(s.items.some((i) => i.text.startsWith("Web search timed out") && i.status === "error"));
    assert.ok(s.items.some((i) => i.status === "error" && /searching the web in parallel/.test(i.text)));
});

test("planner queries inherit the umbrella step's failure", () => {
    const t = buildResearchTimeline({
        running: false,
        steps: [{ text: "🌐 Web search always on: searching the web in parallel", status: "error" }],
        webDecision: { need_web: true, queries: ["q1", "q1", "q2"] },
    });
    const s = t.sections[0];
    assert.deepEqual(s.items.map((i) => i.text), ["q1", "q2"], "duplicate planner queries are one chip");
    assert.ok(s.items.every((i) => i.status === "error"));
    assert.equal(s.status, "error");
});

test("in-loop web tools are their own chips, named by their hint", () => {
    const t = buildResearchTimeline({
        running: true,
        steps: [
            { text: 'Searching the web: "A"', status: "completed" },
            { text: 'Calling web search agent ("B follow-up")', status: "running" },
            { text: 'Extracting web source ("https://example.org/p")', status: "error" },
            { text: "Calling deep web research agent", status: "completed" },
        ],
    });
    const s = t.sections.find((x) => x.kind === "searching");
    assert.deepEqual(s.items.map((i) => [i.text, i.status]), [
        ["A", "completed"],
        ["B follow-up", "running"],
        ["https://example.org/p", "error"],
        ["Calling deep web research agent", "completed"],
    ]);
    assert.equal(t.current, 'Calling web search agent ("B follow-up")');
});

test("an umbrella step still open keeps the section live without faking a chip", () => {
    const t = buildResearchTimeline({
        running: true,
        steps: [
            { text: "🌐 Query needs current information: searching the web in parallel", status: "running" },
            { text: 'Searching the web: "A"', status: "completed" },
        ],
    });
    const s = t.sections[0];
    assert.equal(s.status, "running");
    assert.deepEqual(s.items.map((i) => i.status), ["completed"]);
});

test("unsafe sources are withheld and not counted", () => {
    const t = buildResearchTimeline({
        steps: [],
        webSources: [{ url: "https://almascience.org/a", title: "ok" }, { url: "https://4chan.org/x", title: "no" }],
    });
    const r = t.sections.find((x) => x.kind === "reading");
    assert.deepEqual(r.items.map((i) => i.text), ["almascience.org"]);
    assert.equal(t.sourceCount, 1);
});

test("workforce detection is hidden; a degraded-model failover is a notice", () => {
    const t = buildResearchTimeline({
        running: false,
        steps: [
            { text: "Complex query detected (score=0.81, tier=deep) — activating multi-agent workforce", status: "completed" },
            { text: "Model deepseek-v4-flash is degraded — using gpt-oss-120b for this turn", status: "completed" },
        ],
    });
    assert.deepEqual(kinds(t), ["notice"]);
    assert.equal(t.sections[0].items[0].icon, "alert");
});

test("more than six sources overflow; papers join the Reading row", () => {
    const webSources = Array.from({ length: 8 }, (_, i) => ({ url: `https://site${i}.org/p` }));
    const t = buildResearchTimeline({ steps: [], webSources, paperCount: 4 });
    const r = t.sections.find((s) => s.kind === "reading");
    assert.equal(r.items.length, 7);
    assert.equal(r.overflow, 2);
    assert.equal(r.items.at(-1).text, "4 papers from ADS");
    assert.equal(t.sourceCount, 12);
});

test("stepIcon keyword mapping", () => {
    assert.equal(stepIcon("Downloading ALMA data"), "download");
    assert.equal(stepIcon("Fetching DSS2 cutout"), "image");
    assert.equal(stepIcon("Plotting space light curve"), "chart");
    assert.equal(stepIcon("Cross-matching ALMA + JWST"), "merge");
    assert.equal(stepIcon("Running SQL on Astro Data Lab"), "database");
    assert.equal(stepIcon("Something else"), "spark");
});

test("turnPaperCount counts only the turn's papers", () => {
    const messages = [
        { role: "user", type: "text" },
        { role: "assistant", type: "papers", papers: [{}, {}] },
        { role: "assistant", type: "text" },
        { role: "assistant", type: "papers", papers: [{}] },
        { role: "user", type: "text" },
        { role: "assistant", type: "papers", papers: [{}, {}, {}] },
        { role: "assistant", type: "text" },
    ];
    assert.equal(turnPaperCount(messages, 2), 3);
    assert.equal(turnPaperCount(messages, 6), 3);
    assert.equal(turnPaperCount(messages, 99), 0);
});
