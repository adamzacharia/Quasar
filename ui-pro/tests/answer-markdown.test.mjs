import test from "node:test";
import assert from "node:assert/strict";

import {
    humanizeToolIdentifiers,
    humanizeToolName,
    normalizeMathDelimiters,
    prepareAnswerMarkdown,
    stripBrTags,
    stripImagePlaceholders,
} from "../src/lib/answer-markdown.js";

test("<br> in a table row becomes a separator, elsewhere a hard break (D02)", () => {
    const md = "| Item | Detail |\n|---|---|\n| BLC | 64 antennas<br>4 quadrants |\n\nLine one<br>Line two";
    const out = stripBrTags(md);
    assert.match(out, /64 antennas · 4 quadrants/);
    assert.match(out, /Line one  \nLine two/);
    assert.doesNotMatch(out, /<br/i);
});

test("orphan image placeholders are dropped, real images already handled upstream (L15)", () => {
    const md = "| 1 | 179.70 | ![CMD 1] | ![Cutout 1] |\nText ![alt](http://x/y.png) end";
    const out = prepareAnswerMarkdown(md, { humanizeTools: false });
    assert.doesNotMatch(out, /!\[/);
    assert.match(out, /\| 1 \| 179\.70 \|\s+\|\s+\|/);
});

test("bracket LaTeX lines become $$ math, citations are untouched (D01, D05)", () => {
    const md = "The rms is\n[ \\sigma = \\frac{2 k T_{sys}}{A \\sqrt{N(N-1)\\,\\Delta\\nu\\,t}} ]\nSee [1] and [Source: handbook].\nInline \\( \\theta \\approx 1.2\\lambda/B \\) here.";
    const out = normalizeMathDelimiters(md);
    assert.match(out, /\$\$ ?\\sigma = \\frac/);
    assert.match(out, /See \[1\] and \[Source: handbook\]\./);
    assert.match(out, /Inline \$ \\theta \\approx 1\.2\\lambda\/B \$ here\./);
});

test("internal tool identifiers in prose are humanised, code is left alone", () => {
    const md = "We ran datalab_density_vetting(ra=180, dec=30) and then query_alma_science_archive.\n\n```python\nAlma.query_object('M83')\nresult = datalab_cone_count(...)\n```\nUse `search_by_target` in code.";
    const out = humanizeToolIdentifiers(md);
    assert.match(out, /We ran the Data Lab density vetting and then the ALMA science-archive query\./);
    assert.match(out, /result = datalab_cone_count\(\.\.\.\)/, "fenced code untouched");
    assert.doesNotMatch(out, /`search_by_target`/, "a code span holding only a tool name is humanised");
    assert.match(out, /Use the ALMA target search in code\./);
    assert.equal(humanizeToolIdentifiers("Run `np.log10(x)` first"), "Run `np.log10(x)` first", "real inline code untouched");
});

test("generic snake_case tools get a readable label", () => {
    assert.equal(humanizeToolName("datalab_tiled_search"), "the Data Lab tiled search step");
    assert.equal(humanizeToolName("sparcl_find_spectra"), "the SPARCL find spectra step");
    assert.equal(humanizeToolName("match_cross_archive_sources"), "the cross-archive source match");
});

test("prepareAnswerMarkdown is idempotent", () => {
    const md = "Ran datalab_cone_count.<br>[ \\alpha ]\n| a | b<br>c |";
    const once = prepareAnswerMarkdown(md);
    assert.equal(prepareAnswerMarkdown(once), once);
});

test("stripImagePlaceholders keeps normal brackets", () => {
    assert.equal(stripImagePlaceholders("see [1] and ![x]"), "see [1] and ");
});
