import test from "node:test";
import assert from "node:assert/strict";

import {
    mergeEvidenceQuality,
    rankWebSources,
} from "../src/lib/evidence-quality.js";

test("web sources are ranked by evidence quality score", () => {
    const ranked = rankWebSources([
        {
            title: "General",
            url: "https://example.com/post",
            snippet: "",
            evidenceQuality: { score: 42, tier: "general", label: "General web" },
        },
        {
            title: "Archive",
            url: "https://almascience.eso.org/documents",
            snippet: "",
            evidenceQuality: { score: 96, tier: "primary", label: "Primary archive" },
        },
        {
            title: "Preprint",
            url: "https://arxiv.org/abs/2401.1",
            snippet: "",
            evidenceQuality: { score: 78, tier: "preprint", label: "Preprint" },
        },
    ]);

    assert.equal(ranked[0].title, "Archive");
    assert.equal(ranked[1].title, "Preprint");
    assert.equal(ranked[2].title, "General");
});

test("duplicate source merge keeps stronger evidence quality", () => {
    const merged = mergeEvidenceQuality(
        { score: 42, tier: "general", label: "General web" },
        { score: 90, tier: "peer_reviewed", label: "Peer reviewed" },
    );

    assert.equal(merged.score, 90);
    assert.equal(merged.label, "Peer reviewed");
});
