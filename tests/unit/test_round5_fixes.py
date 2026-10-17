"""ALMAbench round 5 generic fixes: before-cuts funnel counts, fuller paper abstracts, the completeness pass."""
from __future__ import annotations

import pandas as pd

from core.answer_completeness import HEADING, clean_points, complete_answer
from services.alma_bibliography import summarize
from services.alma_science_queries import alma_cone_funnel_adql


def test_funnel_query_drops_only_quality_cuts():
    q = alma_cone_funnel_adql(10.0, -5.0, 0.01, public=False, band=[7], max_resolution_arcsec=0.5,
                              science_only=True, released_before="2026-01-01", footprint=True)
    assert q.startswith("SELECT COUNT(DISTINCT member_ous_uid) AS n_mous, COUNT(DISTINCT proposal_id) AS n_projects")
    assert "spatial_resolution" not in q and "science_observation = 'T'" in q
    assert "obs_release_date < '2026-01-01'" in q and "band_list" in q


def test_funnel_query_none_without_cuts():
    assert alma_cone_funnel_adql(10.0, -5.0, 0.01, public=True, band=[6], max_resolution_arcsec=None,
                                 science_only=False, footprint=True) is None


class _Res:
    def __init__(self, df):
        self.df = df

    def to_table(self):
        df = self.df

        class _T:
            def to_pandas(self_inner):
                return df

        return _T()


class _Service:
    def __init__(self):
        self.queries = []

    def search(self, q, maxrec=None):
        self.queries.append(q)
        return _Res(pd.DataFrame([{"n_mous": 17, "n_projects": 10}]))


def test_before_quality_cuts_reports_counts_and_provenance():
    from capabilities.alma_tools import AlmaSourceSummary

    svc, prov = _Service(), {"query": "SELECT main"}
    kw = dict(public=False, band=[7], max_resolution_arcsec=1.0, science_only=True, released_before=None, footprint=True)
    out = AlmaSourceSummary._before_quality_cuts(svc, 85.9, -1.0, 0.0167, kw, prov, "SELECT main")
    assert out["n_mous"] == 17 and out["n_projects"] == 10
    assert out["cuts_removed"] == ["max_resolution_arcsec"]
    assert svc.queries and "spatial_resolution" not in svc.queries[0]
    assert prov["query"].startswith("SELECT main") and svc.queries[0] in prov["query"]


def test_before_quality_cuts_absent_without_cuts():
    from capabilities.alma_tools import AlmaSourceSummary

    kw = dict(public=True, band=[6], max_resolution_arcsec=None, science_only=False, released_before=None, footprint=True)
    assert AlmaSourceSummary._before_quality_cuts(_Service(), 1.0, 1.0, 0.01, kw, {}, "q") is None


def test_summarize_keeps_whole_abstract_or_whole_sentences():
    short = "We observed X with ALMA.  We find Y."
    assert summarize(short) == "We observed X with ALMA. We find Y."
    long = " ".join(f"Sentence {i} ends here." for i in range(300))
    out = summarize(long)
    assert len(out) <= 1510 and out.endswith("here. [...]")
    assert summarize("") == "(no abstract in the ADS record)"


DRAFT = "The answer explains the main idea in detail. " * 10


def test_completeness_appends_points():
    calls = []

    def call(prompt, model, max_tokens):
        calls.append((prompt, model))
        return "- First missing caveat.\n- Second missing check — with detail."

    out, info = complete_answer(DRAFT, question="How do I do X?", model="arm-model", call=call)
    assert info["status"] == "added" and info["points"] == 2
    assert out.startswith(DRAFT.rstrip()) and HEADING in out
    assert "—" not in out
    assert calls[0][1] == "arm-model" and "How do I do X?" in calls[0][0] and DRAFT.strip()[:40] in calls[0][0]


def test_completeness_none_and_errors_leave_answer():
    out, info = complete_answer(DRAFT, question="q", model="m", call=lambda *a: "NONE")
    assert out == DRAFT and info["status"] == "none"

    def boom(*a):
        raise RuntimeError("down")

    out, info = complete_answer(DRAFT, question="q", model="m", call=boom)
    assert out == DRAFT and "down" in info["skipped"]
    out, info = complete_answer("short", question="q", model="m", call=lambda *a: "- x")
    assert out == "short"


def test_completeness_flag_off(monkeypatch):
    monkeypatch.setenv("QUASAR_COMPLETENESS_PASS", "0")
    out, info = complete_answer(DRAFT, question="q", model="m", call=lambda *a: "- x")
    assert out == DRAFT


def test_clean_points_drops_preamble_and_caps_six():
    raw = "Here are the points:\n" + "\n".join(f"- point {i}" for i in range(9))
    pts = clean_points(raw)
    assert pts.startswith("- point 0") and "point 5" in pts and "point 6" not in pts
    assert clean_points("Nothing to add here.") is None


def test_before_quality_cuts_ignores_default_science_only_and_bad_replies():
    from capabilities.alma_tools import AlmaSourceSummary

    kw = dict(public=False, band=[6], max_resolution_arcsec=None, science_only=True, released_before=None, footprint=True)
    assert AlmaSourceSummary._before_quality_cuts(_Service(), 1.0, 1.0, 0.01, kw, {}, "q") is None

    class _Odd(_Service):
        def search(self, q, maxrec=None):
            return _Res(pd.DataFrame([{"unexpected": 1}]))

    kw["max_resolution_arcsec"] = 0.5
    assert AlmaSourceSummary._before_quality_cuts(_Odd(), 1.0, 1.0, 0.01, kw, {}, "q") is None


def test_completeness_addition_that_trips_the_safety_guard_is_dropped(monkeypatch):
    from services import content_safety

    monkeypatch.setattr(content_safety, "strict_web_filter_enabled", lambda: True)
    out, info = complete_answer(DRAFT, question="q", model="m", call=lambda *a: "- Projects named 2019.1.XXX. are placeholders.")
    assert out == DRAFT and info["status"].startswith("dropped")
