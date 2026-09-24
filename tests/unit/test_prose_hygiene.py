"""core/prose_hygiene.py — internal tool identifiers, JSON argument dumps,
orphan image placeholders and <br> literals out of user-facing prose.

Regression corpus: the 37 saved UI answers of the 2026-09-22 benchmark
(tmp/ui-bench-2026-09-22/{domain,datalab}/*/answer.md, plus the re-run) —
after processing, zero internal identifiers remain in the prose.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.prose_hygiene import (
    find_internal_identifiers,
    humanize_prose,
    humanize_tool_name,
    strip_br_tags,
    strip_image_placeholders,
)
from tests.unit.test_tool_budget_hierarchy import _registered_tool_names

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "tmp" / "ui-bench-2026-09-22"
ANSWERS = sorted(BENCH.glob("domain/*/answer.md")) + sorted(BENCH.glob("datalab/*/answer.md")) + sorted(BENCH.glob("datalab-rerun/*/answer.md"))


@pytest.fixture(scope="module")
def tool_names():
    return sorted(_registered_tool_names())


def test_humanize_tool_name_examples():
    assert humanize_tool_name("datalab_density_vetting") == "the Data Lab density vetting"
    assert humanize_tool_name("query_alma_science_archive") == "the ALMA science-archive query"
    assert humanize_tool_name("datalab_tiled_search") == "the Data Lab tiled search"
    assert humanize_tool_name("datalab_export_notebook") == "the Data Lab export notebook step"
    assert humanize_tool_name("sparcl_find_spectra") == "the SPARCL find spectra step"
    assert humanize_tool_name("vo_adql_query") == "the VO ADQL query step"


def test_prose_is_rewritten_but_code_is_untouched(tool_names):
    text = (
        "I ran datalab_density_vetting(ra=180, dec=30, radius=2) and then query_alma_science_archive with "
        '{"query_type": "cycle_solar_projects", "cycle": 10}.\n\n'
        "```python\nfrom astroquery.alma import Alma\nresult = search_by_target('M83')\n```\n"
        "Call `datalab_cone_count` in code.\n"
        "| RA | CMD |\n|---|---|\n| 179.7 | ![CMD 1] |\nrow one<br>row two"
    )
    out = humanize_prose(text, tool_names)
    assert "the Data Lab density vetting" in out and "the ALMA science-archive query" in out
    assert "datalab_density_vetting(" not in out
    assert '{"query_type"' not in out and "Show-query panel" in out
    assert "result = search_by_target('M83')" in out, "fenced code untouched"
    assert "`datalab_cone_count`" not in out and "the Data Lab cone count" in out, "a code span holding only a tool name is prose"
    assert humanize_prose("Use `np.log10(x)` here", tool_names) == "Use `np.log10(x)` here", "real inline code untouched"
    assert "![CMD 1]" not in out
    assert "<br>" not in out and "row one  \nrow two" in out
    assert find_internal_identifiers(out, tool_names) == []


def test_idempotent(tool_names):
    text = "We used datalab_color_magnitude_diagram and match_cross_archive_sources."
    once = humanize_prose(text, tool_names)
    assert humanize_prose(once, tool_names) == once


def test_strip_br_in_tables_uses_a_separator():
    assert strip_br_tags("| a<br>b | c |") == "| a · b | c |"
    assert strip_br_tags("a<br/>b") == "a  \nb"


def test_strip_image_placeholders_keeps_real_links():
    assert strip_image_placeholders("![x] and ![y](http://u) and [z]") == " and ![y](http://u) and [z]"


@pytest.mark.skipif(not ANSWERS, reason="benchmark answer corpus not present")
def test_every_saved_benchmark_answer_is_free_of_internal_identifiers(tool_names):
    assert len(ANSWERS) >= 37, f"expected the 37+ saved answers, found {len(ANSWERS)}"
    before_total = 0
    leftovers = {}
    for path in ANSWERS:
        raw = path.read_text(encoding="utf-8", errors="replace")
        before_total += len(find_internal_identifiers(raw, tool_names))
        cleaned = humanize_prose(raw, tool_names)
        left = find_internal_identifiers(cleaned, tool_names)
        if left:
            leftovers[path.parent.name] = left
    assert before_total > 0, "the corpus is supposed to contain the identifiers we clean"
    assert leftovers == {}, f"internal identifiers left in prose: {leftovers}"


def test_code_is_protected_from_global_cleanups():
    """Guard CX-25: <br> conversion and empty-parenthesis removal ran on the
    joined text and rewrote code (`Client()`, <br> in an HTML sample)."""
    from core.prose_hygiene import humanize_prose

    text = ("Create `Client()` first.\n```html\n<p>a<br>b</p>\nq = Alma()\n```\n"
            "~~~~python\nx = Tap()\n~~~\nstill = code()\n~~~~\n"
            "Line one<br>line two via ``tricky `inline` ()`` and `datalab_sql_query`.")
    out = humanize_prose(text, ["datalab_sql_query"])
    assert "`Client()`" in out and "<p>a<br>b</p>" in out and "q = Alma()" in out
    assert "x = Tap()\n~~~\nstill = code()" in out
    assert "``tricky `inline` ()``" in out
    assert "Line one  \nline two" in out and "`datalab_sql_query`" not in out



def test_article_doubling_across_a_humanised_code_span_is_collapsed():
    """UI 2026-09-23 D07: "Use the `list_alma_files` tool" read "Use the the
    ALMA file listing tool" -- the doubling only exists after the join."""
    from core.prose_hygiene import humanize_prose

    out = humanize_prose("Use the `list_alma_files` tool, then the `inspect_fits_header` tool and `x = f()`.",
                         ["list_alma_files", "inspect_fits_header"])
    assert "the the" not in out.lower()
    assert "Use the ALMA file listing, then the inspect FITS header step and `x = f()`." == out


def test_a_determiner_absorbs_the_labels_article_across_emphasis_and_adjectives():
    """UI re-run 2026-09-23: "The **the archive image overlay** call" (D19) and
    "in a new the ALMA position search" (D16)."""
    from core.prose_hygiene import humanize_prose

    names = ["archive_overlay", "search_by_position"]
    assert humanize_prose("The **archive_overlay** call was able to locate it.", names) == "The **archive image overlay** was able to locate it."
    assert humanize_prose("Set it in a new search_by_position call.", names) == "Set it in a new ALMA position search call."
    assert humanize_prose("archive_overlay found it. search_by_position too.", names) == "The archive image overlay found it. The ALMA position search too."
    assert "\x01" not in humanize_prose("I used search_by_position.", names)
