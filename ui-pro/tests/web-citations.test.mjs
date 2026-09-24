import test from "node:test";
import assert from "node:assert/strict";

import {
    citationIdFromHref,
    citationLabel,
    hasEvidenceIds,
    mergeTurnWebSources,
    normalizeWebSource,
    sourceCardDomId,
    splitCitedSources,
    tokenizeWebCitations,
    webSourcesForTurn,
} from "../src/lib/web-citations.js";

test("single, grouped and adjacent [W#] tags become chip links", () => {
    const md = "Twelve months [W2]. Both agree [W1, W3]. Adjacent [W4][W2]. Semicolon [W5; W6].";
    assert.equal(
        tokenizeWebCitations(md),
        "Twelve months [2](#web-cite-W2). Both agree [1](#web-cite-W1)[3](#web-cite-W3). " +
        "Adjacent [4](#web-cite-W4)[2](#web-cite-W2). Semicolon [5](#web-cite-W5)[6](#web-cite-W6).",
    );
});

test("tags inside fenced code, inline code and math are untouched", () => {
    const md = [
        "Real [W1].",
        "```python",
        "x = refs[W2]",
        "```",
        "Inline `arr[W3]` and $a[W4]$ and $$b[W5]$$ stay.",
        "~~~",
        "[W6]",
        "~~~",
    ].join("\n");
    const out = tokenizeWebCitations(md);
    assert.match(out, /Real \[1\]\(#web-cite-W1\)\./);
    assert.match(out, /x = refs\[W2\]/);
    assert.match(out, /`arr\[W3\]`/);
    assert.match(out, /\$a\[W4\]\$/);
    assert.match(out, /\$\$b\[W5\]\$\$/);
    assert.match(out, /\n\[W6\]\n/);
});

test("paper references and unknown ids are not chips", () => {
    assert.equal(tokenizeWebCitations("Smith et al. [1] and [12]."), "Smith et al. [1] and [12].");
    const out = tokenizeWebCitations("Known [W1], stale [W9], mixed [W1, W9].", ["W1"]);
    assert.equal(out, "Known [1](#web-cite-W1), stale [W9], mixed [1](#web-cite-W1).");
});

test("chip hrefs round-trip to citation ids and labels", () => {
    assert.equal(citationIdFromHref("#web-cite-W7"), "W7");
    assert.equal(citationIdFromHref("#user-content-fn-1"), null);
    assert.equal(citationIdFromHref("https://example.com"), null);
    assert.equal(citationLabel("W12"), "12");
    assert.equal(sourceCardDomId("msg:1/2", "W3"), "web-src-msg12-W3");
});

test("normalizeWebSource keeps id, cited, date, domain and provider", () => {
    const s = normalizeWebSource({
        id: "W2", cited: true, title: "Guide", url: "https://www.almascience.org/x】",
        snippet: "12 months", published_date: "2026-03-01", provider: "Tavily",
    });
    assert.deepEqual(
        { id: s.id, cited: s.cited, url: s.url, date: s.publishedDate, domain: s.domain, provider: s.provider },
        { id: "W2", cited: true, url: "https://www.almascience.org/x", date: "2026-03-01", domain: "almascience.org", provider: "Tavily" },
    );
    assert.equal(normalizeWebSource({ title: "no url" }), null);
});

test("mid-stream events merge by URL; the final replace event wins", () => {
    const first = mergeTurnWebSources([], [
        { id: "W1", url: "https://a.org", title: "A" },
        { id: "W2", url: "https://b.org", title: "B" },
    ]);
    const second = mergeTurnWebSources(first, [
        { id: "W2", url: "https://b.org/", title: "B" },
        { id: "W3", url: "https://c.org", title: "C" },
    ]);
    assert.deepEqual(second.map((s) => s.id), ["W1", "W2", "W3"]);
    const final = mergeTurnWebSources(second, [
        { id: "W3", url: "https://c.org", cited: true },
        { id: "W1", url: "https://a.org", cited: true },
        { id: "W2", url: "https://b.org", cited: false },
    ], { replace: true });
    assert.deepEqual(final.map((s) => [s.id, s.cited]), [["W3", true], ["W1", true], ["W2", false]]);
    const { cited, uncited } = splitCitedSources(final);
    assert.deepEqual(cited.map((s) => s.id), ["W3", "W1"]);
    assert.deepEqual(uncited.map((s) => s.id), ["W2"]);
});

test("legacy sources without ids keep their order and are not evidence", () => {
    const merged = mergeTurnWebSources([], [{ url: "https://b.org" }, { url: "https://a.org" }]);
    assert.deepEqual(merged.map((s) => s.url), ["https://b.org", "https://a.org"]);
    assert.equal(hasEvidenceIds(merged), false);
    assert.equal(hasEvidenceIds([{ id: "W1", url: "https://a.org" }]), true);
});

test("webSourcesForTurn collects the turn's sources (rehydrated message order)", () => {
    const messages = [
        { role: "user", content: "q1" },
        { role: "assistant", type: "text", content: "old [W1]" },
        { role: "assistant", type: "web_sources", webSources: [{ id: "W1", url: "https://old.org", cited: true }] },
        { role: "user", content: "q2" },
        { role: "assistant", type: "text", content: "new [W2]" },
        { role: "assistant", type: "data", content: "" },
        { role: "assistant", type: "web_sources", webSources: [
            { id: "W2", url: "https://new.org", cited: true },
            { id: "W1", url: "https://other.org", cited: false },
        ] },
    ];
    assert.deepEqual(webSourcesForTurn(messages, 4).map((s) => s.id), ["W2", "W1"]);
    assert.deepEqual(webSourcesForTurn(messages, 1).map((s) => s.url), ["https://old.org"]);
    assert.deepEqual(webSourcesForTurn(messages, 99), []);
});

// ── guard task-25bee13-20119 ──────────────────────────────────────────────

import { urlKey, webSourcesMessageView } from "../src/lib/web-citations.js";

test("CX-20: a grounded turn hides its sources message but keeps web image tiles", () => {
    const base = [
        { role: "user", content: "q" },
        { role: "assistant", type: "text", content: "a [W1]" },
    ];
    const grounded = { role: "assistant", type: "web_sources", webSources: [{ id: "W1", url: "https://a.org" }] };
    assert.equal(webSourcesMessageView([...base, grounded], 2), "hide");
    assert.equal(webSourcesMessageView([...base, { ...grounded, webImages: [{ url: "https://a.org/i.jpg" }] }], 2), "images-only");
    // legacy (no ids) and a turn without an answer message keep the full card
    assert.equal(webSourcesMessageView([...base, { role: "assistant", type: "web_sources", webSources: [{ url: "https://a.org" }] }], 2), "full");
    assert.equal(webSourcesMessageView([{ role: "user", content: "q" }, grounded], 1), "full");
});

test("CX-21: URL keys keep meaningful queries and path case", () => {
    assert.notEqual(urlKey("https://x.org/article?id=1"), urlKey("https://x.org/article?id=2"));
    assert.notEqual(urlKey("https://x.org/Case"), urlKey("https://x.org/case"));
    assert.equal(urlKey("http://www.X.org/a/?utm_source=z#frag"), urlKey("https://x.org/a"));
    assert.equal(urlKey("https://x.org:443/a"), urlKey("https://x.org/a"));
    const merged = mergeTurnWebSources([], [
        { id: "W1", url: "https://x.org/article?id=1" },
        { id: "W2", url: "https://x.org/article?id=2" },
    ]);
    assert.deepEqual(merged.map((s) => s.id), ["W1", "W2"]);
});

test("CX-22: indented code, existing links and escaped tags are not tokenized", () => {
    const md = "Intro [W1].\n\n    const ref = arr[W1]\n\nSee [W1](https://example.com) and \\[W2\\] and [W2].\n- item\n    continued [W1]";
    const out = tokenizeWebCitations(md);
    assert.match(out, /Intro \[1\]\(#web-cite-W1\)\./);
    assert.match(out, /\n {4}const ref = arr\[W1\]\n/);
    assert.match(out, /\[W1\]\(https:\/\/example\.com\)/);
    assert.ok(out.includes("\\[W2\\]"), "escaped tag stays as typed");
    assert.match(out, /and \[2\]\(#web-cite-W2\)\./);
    assert.match(out, /continued \[1\]\(#web-cite-W1\)$/);
});
