"""services/alma_bibliography.py + the alma_bibliography tool: topic filter, ObsCore paper-project map, archival class."""
from __future__ import annotations

import pandas as pd

from capabilities import alma_tools
from services import alma_bibliography as ab
from tests.unit.test_new_oneshot_tools import _alma_ctx, _run

MAP_ROWS = [
    {"proposal_id": "2013.1.00486.S", "pi_name": "Boley, Aaron", "proposal_authors": "Ford, Eric; Wilner, David",
     "bib_reference": "2025ApJ...990L..40C 2016ApJ...820..41B"},
    {"proposal_id": "2019.1.00001.S", "pi_name": "Doe, Jane", "proposal_authors": "Roe, Richard",
     "bib_reference": "2025ApJ...990L..40C 2025A&A...700A...1D"},
    {"proposal_id": "2011.0.00010.SV", "pi_name": "ALMA", "proposal_authors": "",
     "bib_reference": "2024MNRAS.500....1S"},
]


def test_name_keys_and_map():
    assert ab.name_key("Müller, Jörg K.") == "muller j" and ab.name_key("Jane Doe") == "doe j" and ab.name_key("") == ""
    m = ab.parse_bib_map(MAP_ROWS)
    assert [p["project"] for p in m["2025ApJ...990L..40C"]] == ["2013.1.00486.S", "2019.1.00001.S"]
    assert m["2016ApJ...820..41B"][0]["team"] == {"boley a", "ford e", "wilner d"}


def test_archival_classes_follow_the_author_overlap_definition():
    m = ab.parse_bib_map(MAP_ROWS)
    assert ab.classify(["Chen, X.", "Lee, Y."], m["2025ApJ...990L..40C"])[0] == ab.ARCHIVAL_ONLY
    assert ab.classify(["Chen, X.", "Doe, J."], m["2025ApJ...990L..40C"])[0] == ab.ARCHIVAL_PLUS_PI
    cls, ev = ab.classify(["Boley, A.", "Roe, R."], m["2025ApJ...990L..40C"])
    assert cls == ab.PI_DATA and all(e["author_on_proposal"] for e in ev)
    assert ab.classify(["Anyone, A."], m["2024MNRAS.500....1S"])[0] == ab.ARCHIVAL_ONLY, "SV data are archival use"
    assert ab.classify(["Anyone, A."], [])[0] == ab.UNKNOWN
    assert ab.CONFIDENCE[ab.PI_DATA] == "high" and ab.CONFIDENCE[ab.ARCHIVAL_PLUS_PI] == "low"


def test_topic_evidence_needs_title_keyword_or_two_abstract_mentions():
    pats = ab.topic_patterns(["debris disk", "exoKuiper belt"])
    assert ab.topic_evidence({"title": ["Resolving debris disks with ALMA"]}, pats).startswith("title")
    assert ab.topic_evidence({"title": "X", "keyword": ["exoKuiper belts"]}, pats).startswith("keyword")
    assert ab.topic_evidence({"title": "X", "abstract": "A debris disk. Another debris-disk result."}, pats).startswith("abstract")
    assert ab.topic_evidence({"title": "X", "abstract": "One debris disk mention."}, pats) is None
    gal = ab.topic_patterns(["submillimeter galaxy"])
    assert ab.topic_evidence({"title": "Two submillimeter galaxies at z = 4"}, gal) and ab.topic_evidence({"title": "A submillimeter galaxy"}, gal)
    assert ab.ads_query(["debris disk", "exoKuiper belt"]) == 'bibgroup:ALMA (abs:"debris disk" OR abs:"exoKuiper belt")'


def test_bib_map_is_cached():
    calls = []

    def run(q):
        calls.append(q)
        return pd.DataFrame(MAP_ROWS)
    ab._MAP_CACHE.update(at=0.0, map=None)
    m1, _, ok1 = ab.bib_map(run, now=1000.0)
    m2, _, ok2 = ab.bib_map(run, now=1000.0 + 3600)
    assert ok1 and ok2
    assert m1 is m2 and calls == [ab.BIB_MAP_ADQL]
    ab.bib_map(run, now=1000.0 + 25 * 3600)
    assert len(calls) == 2
    ab._MAP_CACHE.update(at=0.0, map=None)


class _Ads:
    def __init__(self, papers):
        self.papers, self.calls = papers, []

    def search_papers(self, query, max_results=10, sort="date desc", fields=None, filters=None, start=0):
        self.calls.append((query, max_results, sort, filters, start))
        return self.papers if start == 0 else []


def _paper(bib, title, authors, abstract="Text."):
    return {"bibcode": bib, "title": [title], "author": authors, "pubdate": "2025-08-00", "pub": "ApJ", "abstract": abstract}


def test_tool_keeps_on_topic_archival_papers_and_closes_the_list():
    ab._MAP_CACHE.update(at=0.0, map=None)
    papers = [
        _paper("2025ApJ...990L..40C", "Debris disk around a nearby star", ["Chen, X.", "Lee, Y."],
               "We use archival ALMA data. Second sentence. Third."),
        _paper("2016ApJ...820..41B", "A debris disk seen by ALMA", ["Boley, A.", "Ford, E."]),
        _paper("2025A&A...700A...1D", "Protoplanetary disk gas masses", ["Chen, X."]),
        _paper("2099ApJ...999..999X", "Debris disks without a linked project", ["Nobody, N."]),
    ]
    ads = _Ads(papers)
    ctx, state, _ = _alma_ctx(lambda q: pd.DataFrame(MAP_ROWS), ads_client=ads)
    out = _run(alma_tools.AlmaBibliography(), ctx, topic="debris disk", n=10, since=2015)
    assert out["success"] and out["status"] == "ok" and out["n_returned"] == 1
    row = out["results"][0]
    assert row["bibcode"] == "2025ApJ...990L..40C" and row["archival_class"] == "archival only" and row["confidence"] == "medium"
    assert row["alma_project_codes"] == "2013.1.00486.S, 2019.1.00001.S" and row["summary"] == "We use archival ALMA data. Second sentence. Third."
    assert row["first_author"] == "Chen, X. et al." and row["pubdate"] == "2025-08"
    q, _, sort, filters, _ = ads.calls[0]
    assert q.startswith("bibgroup:ALMA (") and sort == "date desc" and "property:refereed" in filters and "year:2015-" in filters
    assert out["search"]["off_topic_dropped"] == 1 and out["search"]["dropped_by_archival_class"] == 2
    assert "do not add" in out["note"] and out["as_of"] and "Stoehr" not in out["note"]
    assert state.last_run_result["type"] == "papers"
    every = _run(alma_tools.AlmaBibliography(), ctx, topic="debris disk", archival="ignore")
    assert every["n_returned"] == 3
    ab._MAP_CACHE.update(at=0.0, map=None)


def test_tool_without_the_project_map_is_partial_and_rejects_bad_modes():
    ab._MAP_CACHE.update(at=0.0, map=None)

    def boom(q):
        raise RuntimeError("TAP down")
    ctx, _, _ = _alma_ctx(boom, ads_client=_Ads([_paper("2025X", "debris disk", ["A, B."])]))
    out = _run(alma_tools.AlmaBibliography(), ctx, topic="debris disk")
    assert out["status"] == "partial" and out["n_returned"] == 0 and "could not be classified" in out["method_note"]
    assert _run(alma_tools.AlmaBibliography(), ctx, topic="x", archival="maybe")["status"] == "invalid_input"


def test_cx01_full_author_lists_come_from_raw_ads_records_and_formatted_ones_are_unknown():
    ab._MAP_CACHE.update(at=0.0, map=None)

    class _RawAds(_Ads):
        def search_raw(self, query, rows=200, sort="date desc", fields=None, filters=None, start=0):
            self.calls.append((query, rows, sort, filters, start))
            return self.papers if start == 0 else []
    pi_fourth = _paper("2025ApJ...990L..40C", "A debris disk", ["Chen, X.", "Lee, Y.", "Kim, Z.", "Doe, Jane"])
    pi_fourth.update(keyword=["debris disks"], title=["Unrelated title"])
    ctx, _, _ = _alma_ctx(lambda q: pd.DataFrame(MAP_ROWS), ads_client=_RawAds([pi_fourth]))
    out = _run(alma_tools.AlmaBibliography(), ctx, topic="debris disk", archival="ignore")
    row = out["results"][0]
    assert row["archival_class"] == "archival + PI", "Doe (PI of 2019.1.00001.S) is the 4th author"
    assert row["topic_evidence"].startswith("keyword") and row["pubdate"] == "2025-08"
    formatted = {"bibcode": "2025ApJ...990L..40C", "title": "A debris disk", "authors": "Chen, X. et al.", "year": "2025"}
    ctx, _, _ = _alma_ctx(lambda q: pd.DataFrame(MAP_ROWS), ads_client=_Ads([formatted]))
    out = _run(alma_tools.AlmaBibliography(), ctx, topic="debris disk", archival="ignore")
    assert out["results"][0]["archival_class"] == "unknown"
    ab._MAP_CACHE.update(at=0.0, map=None)


def test_cx05_truncated_project_map_is_not_cached_and_marks_the_result_partial():
    ab._MAP_CACHE.update(at=0.0, map=None)

    def capped(q):
        df = pd.DataFrame(MAP_ROWS)
        df.attrs["truncated"] = True
        return df
    m, _, complete = ab.bib_map(capped, now=5.0)
    assert m and complete is False and ab._MAP_CACHE["map"] is None
    import services.alma_bibliography as abm

    ctx, _, _ = _alma_ctx(lambda q: pd.DataFrame(MAP_ROWS), ads_client=_Ads([_paper("2025ApJ...990L..40C", "Debris disk", ["Chen, X."])]))
    orig = abm.bib_map
    abm.bib_map = lambda run_query, now=None: (ab.parse_bib_map(MAP_ROWS), "now", False)
    try:
        out = _run(alma_tools.AlmaBibliography(), ctx, topic="debris disk")
    finally:
        abm.bib_map = orig
    assert out["status"] == "partial" and "provisional" in out["method_note"]
    ab._MAP_CACHE.update(at=0.0, map=None)
