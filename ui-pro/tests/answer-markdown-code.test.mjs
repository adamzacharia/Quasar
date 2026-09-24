import test from "node:test";
import assert from "node:assert/strict";

import { convertHtmlLists, convertSubSup, humanizeToolIdentifiers, prepareAnswerMarkdown, splitCode } from "../src/lib/answer-markdown.js";

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

test("<sub>/<sup> literals become sub/superscripts outside code (D01)", () => {
    assert.equal(convertSubSup("H<sub>2</sub>O, 10<sup>5</sup>, cm<sup>-2</sup>"), "H₂O, 10⁵, cm⁻²");
    assert.equal(convertSubSup("T<sub>sys</sub>"), "T_sys");
    const out = prepareAnswerMarkdown("| T<sub>sys</sub> | 120 K |\n`x<sub>1</sub>`", { humanizeTools: false });
    assert.match(out, /\| T_sys \| 120 K \|/);
    assert.match(out, /`x<sub>1<\/sub>`/, "code untouched");
});

test("<ul><li> in a table cell becomes an inline bullet list; elsewhere markdown bullets (D03)", () => {
    const row = "| Exceptions | <ul><li>DDT: 6 months</li><li>Commissioning data</li></ul> |";
    const out = convertHtmlLists(row);
    assert.doesNotMatch(out, /<\/?(ul|li)>/);
    assert.match(out, /^\| Exceptions \| • DDT: 6 months • Commissioning data \|$/);
    const prose = convertHtmlLists("Notes:<ul><li>one</li><li>two</li></ul>");
    assert.match(prose, /\n- one\n- two/);
    assert.equal(convertHtmlLists("no lists here"), "no lists here");
});

// Guard task-25bee13-9557 CX-06: a CommonMark code span may cross ONE line
// break; a <br> inside it must survive the hygiene pass.
test("inline code spanning a single line break is never rewritten", () => {
    const md = "Use `a<br>\nb` in the cell.";
    assert.ok(prepareAnswerMarkdown(md).includes("`a<br>\nb`"));
    assert.deepEqual(splitCode("x `a\nb` y"), [[false, "x "], [true, "`a\nb`"], [false, " y"]]);
    // a blank line ends a paragraph, so it is not a code span
    assert.deepEqual(splitCode("x `a\n\nb` y"), [[false, "x `a\n\nb` y"]]);
});

test("a determiner absorbs the humanised label's article (UI re-run D19/D16)", () => {
    const names = ["archive_overlay", "search_by_position"];
    assert.equal(humanizeToolIdentifiers("The **archive_overlay** call was able.", names), "The **archive overlay step** was able.");
    assert.equal(humanizeToolIdentifiers("in a new search_by_position call.", names), "in a new ALMA position search call.");
    assert.equal(humanizeToolIdentifiers("search_by_position found it.", names), "The ALMA position search found it.");
});
