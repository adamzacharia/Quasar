import pytest

from services.provenance import (
    ProvenanceLedger,
    SourceRecord,
    build_sources_appendix,
)

pytestmark = pytest.mark.unit


def test_extract_harvests_all_kinds_and_dedupes():
    results = {
        "task1": {
            "summary": "Found in project 2021.1.00123.S and also 2021.1.00123.S again.",
            "chunks": [
                {"source_file": "docs/alma_handbook.pdf", "doc_year": 2024, "text": "x"},
            ],
        },
        "task2": {
            "lines": [
                {"species": "CO", "transition": "2-1", "frequency_ghz": 230.538},
            ],
            "web": [
                {"url": "https://almascience.org/x", "title": "ALMA page"},
                {"url": "https://almascience.org/x", "title": "dup"},  # dedupe by url
            ],
        },
    }
    ledger = ProvenanceLedger()
    ledger.extract_from_results(results)
    items = ledger.to_list()
    kinds = {item["kind"] for item in items}
    assert {"project_code", "rag_chunk", "splatalogue", "web"} <= kinds

    project_codes = [i for i in items if i["kind"] == "project_code"]
    assert len(project_codes) == 1  # deduped
    assert project_codes[0]["id"] == "2021.1.00123.S"

    web = [i for i in items if i["kind"] == "web"]
    assert len(web) == 1  # deduped by url

    rag = [i for i in items if i["kind"] == "rag_chunk"]
    assert "alma_handbook.pdf" in rag[0]["label"]
    assert "2024" in rag[0]["label"]


def test_render_markdown_empty_and_grouped():
    assert ProvenanceLedger().render_markdown() == ""

    ledger = ProvenanceLedger()
    ledger.add(SourceRecord("project_code", "2021.1.00123.S", "ALMA 2021.1.00123.S"))
    ledger.add(SourceRecord("web", "https://x", "Example", ref="https://x"))
    md = ledger.render_markdown()
    assert md.startswith("## Sources")
    assert "ALMA projects" in md
    assert "Web sources" in md
    assert "2021.1.00123.S" in md


def test_extract_is_defensive_against_malformed_input():
    # None, lists of scalars, deeply nested junk — must not raise.
    for junk in (None, 123, [1, 2, 3], {"a": [{"b": [None, "no codes here"]}]}, "plain"):
        ledger = ProvenanceLedger()
        ledger.extract_from_results(junk)
        assert isinstance(ledger.to_list(), list)


def test_bibcode_extraction_optional_dependency(monkeypatch):
    # Item 5 must work even if the citation_verifier (Item 2) is absent.
    import services.provenance as prov

    monkeypatch.setattr(prov, "_extract_bibcodes", lambda text: [])
    md = build_sources_appendix({"t": "see 2018ApJ...869L..41A"})
    assert "ADS" not in md  # bibcode harvesting skipped, no error


def test_build_sources_appendix_convenience():
    md = build_sources_appendix(
        {"t": {"summary": "project 2019.1.00011.S", "url": "https://x", "title": "T"}}
    )
    assert "## Sources" in md
    assert "2019.1.00011.S" in md
