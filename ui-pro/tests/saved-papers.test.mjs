import test from "node:test";
import assert from "node:assert/strict";

import { mergeSavedPapers, savedPaperKey } from "../src/lib/saved-papers.js";

test("key prefers bibcode, then DOI, then arXiv id, then title+year", () => {
    assert.equal(savedPaperKey({ id: "paper-0", bibcode: "2024ApJ...1A", doi: "10.1/x" }), "2024ApJ...1A");
    assert.equal(savedPaperKey({ id: "paper-0", doi: "https://doi.org/10.3847/ABC" }), "doi:10.3847/abc");
    assert.equal(savedPaperKey({ id: "paper-0", arxivId: "arXiv:2401.01234" }), "arxiv:2401.01234");
    assert.equal(savedPaperKey({ id: "paper-0", title: "  A   Radio Survey ", year: 2024 }), "title:a radio survey|2024");
});

test("positional card ids never become bookmark keys", () => {
    // Two different papers that both rendered as result #1 of their search.
    const a = savedPaperKey({ id: "paper-0", title: "Paper A", year: 2020 });
    const b = savedPaperKey({ id: "paper-0", title: "Paper B", year: 2021 });
    assert.notEqual(a, b);
    assert.equal(savedPaperKey({ id: "paper-0", title: "Untitled" }), "");
    assert.equal(savedPaperKey(null), "");
});

test("merge keeps server order and in-flight local bookmarks", () => {
    const server = [{ id: "b", title: "B" }, { id: "a", title: "A" }];
    const local = [{ id: "a", title: "A (stale)" }, { id: "c", title: "C, save in flight" }];
    const merged = mergeSavedPapers(server, local);
    assert.deepEqual(merged.map((p) => p.id), ["b", "a", "c"]);
    assert.equal(merged[1].title, "A");
});

test("merge tolerates empty and duplicate input", () => {
    assert.deepEqual(mergeSavedPapers([], []), []);
    assert.deepEqual(mergeSavedPapers(null, undefined), []);
    assert.equal(mergeSavedPapers([{ id: "x" }, { id: "x" }], []).length, 1);
});
