import test from "node:test";
import assert from "node:assert/strict";

import { humanizeToolIdentifiers, prepareAnswerMarkdown, splitCode } from "../src/lib/answer-markdown.js";

// Guard CX-26: the full preparation used to strip images, rewrite <br> and
// normalise math BEFORE splitting out code, corrupting fenced HTML/Markdown
// samples and code-recipe answers.
test("prepareAnswerMarkdown never changes fenced or inline code", () => {
    const md = [
        "Here is an HTML sample:",
        "~~~html",
        "<p>a<br>b</p>",
        "![logo](http://x/y.png) and ![placeholder]",
        "[ \\alpha ]",
        "~~~",
        "````markdown",
        "```python",
        "Alma.query_object('M83')",
        "```",
        "````",
        "Inline `a<br>b` and ``x ` y``.",
        "Prose line<br>next ![orphan]",
        "[ \\beta = 1 ]",
    ].join("\n");
    const out = prepareAnswerMarkdown(md, { humanizeTools: false });
    assert.match(out, /~~~html\n<p>a<br>b<\/p>\n!\[logo\]\(http:\/\/x\/y\.png\) and !\[placeholder\]\n\[ \\alpha \]\n~~~/);
    assert.match(out, /````markdown\n```python\nAlma\.query_object\('M83'\)\n```\n````/);
    assert.match(out, /Inline `a<br>b` and ``x ` y``\./);
    assert.match(out, /Prose line {2}\nnext /);
    assert.doesNotMatch(out.split("````")[2], /!\[orphan\]/);
    assert.match(out, /\$\$ ?\\beta = 1 ?\$\$/);
});

test("splitCode keeps every character and follows CommonMark fences", () => {
    const md = "a `b` c\n````\n```\ninside\n```\n````\n~~~\nopen to end";
    const parts = splitCode(md);
    assert.equal(parts.map(([, s]) => s).join(""), md);
    assert.deepEqual(parts.filter(([c]) => c).map(([, s]) => s), ["`b`", "````\n```\ninside\n```\n````\n", "~~~\nopen to end"]);
});

// Guard CX-27: the generic shape rewrote scientific identifiers.
test("column and catalogue names are not humanised as tools", () => {
    const md = "Columns image_id, survey_id and velocity_frame from gaia_dr3 and ls_dr9; flux_g too.";
    assert.equal(humanizeToolIdentifiers(md), md);
    const tools = "We ran datalab_cone_count and search_by_target then get_mast_products.";
    assert.equal(humanizeToolIdentifiers(tools),
        "We ran the Data Lab cone count and the ALMA target search then the MAST product listing.");
});
