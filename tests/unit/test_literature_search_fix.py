"""Literature-search fix (2026-10-03, tmp/literature-search-fix-2026-10-03/PLAN.md).

Offline: ADS and the LLM are faked. Covers the query builder (model choice,
parsing, validation, timeout, cache), identifier lookups with wrong-year
bibcode recovery, the deterministic fallback, the zero-hit relaxation ladder,
find_citing_papers grouping, the capability payloads, and the per-turn guards
in core/turn_guards.py.
"""
import json
import re
import time

import pytest

from integrations import ads_client as ac
from integrations.ads_client import (
    ADSQueryBuilder,
    ADSQueryBuilderError,
    ADSService,
    ADSServiceError,
    bibcode_fuzzy_query,
    drop_query_clause,
    extract_literature_identifiers,
    heuristic_ads_query,
    pick_fuzzy_bibcode_match,
    relaxation_steps,
    validate_ads_query,
    widen_year_clause,
)
from core import turn_guards as tg


# ─────────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────────
def _paper(bib, title="T", abstract="", citations=0):
    return {"bibcode": bib, "title": title, "abstract": abstract, "citations": citations,
            "authors": "A et al.", "year": bib[:4], "journal": "J", "link": f"https://ui.adsabs.harvard.edu/abs/{bib}"}


class _FakeADS(ADSService):
    """ADSService whose search_papers answers from a query -> papers table."""

    def __init__(self, table=None, default=None, raise_on=()):
        super().__init__(api_key="test-key")
        self.table = table or {}
        self.default = default if default is not None else []
        self.raise_on = set(raise_on)
        self.calls = []

    def search_papers(self, query, max_results=10, sort="date desc", fields=None, filters=None, start=0):
        self.calls.append({"query": query, "sort": sort, "filters": filters, "rows": max_results})
        if query in self.raise_on:
            raise ADSServiceError("ADS API error 400: syntax")
        for key, papers in self.table.items():
            if (key(query) if callable(key) else key == query):
                return [dict(p) for p in papers]
        return [dict(p) for p in self.default]

    def count_matches(self, query, filters=None):
        return None  # offline: no numFound (subclasses override)


@pytest.fixture(autouse=True)
def _clear_builder_cache(monkeypatch):
    # Fresh in-flight slots per test: the timeout test leaves its abandoned
    # call holding one for a few seconds, by design (guard CX-02).
    import threading

    monkeypatch.setattr(ac, "_BUILDER_SLOTS", threading.BoundedSemaphore(2))
    ac._BUILDER_CACHE.clear()
    yield
    ac._BUILDER_CACHE.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Query builder
# ─────────────────────────────────────────────────────────────────────────────
def test_builder_model_ignores_default_llm_model(monkeypatch):
    monkeypatch.delenv("ADS_QUERY_MODEL", raising=False)
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "deepseek-v4-pro")
    assert ac.ads_query_builder_model() == "gpt-oss-120b"
    assert ADSQueryBuilder().model == "gpt-oss-120b"
    monkeypatch.setenv("ADS_QUERY_MODEL", "deepseek-v4-flash")
    assert ADSQueryBuilder().model == "deepseek-v4-flash"


def test_builder_parses_harmony_and_fenced_json():
    seen = {}

    def llm(instructions, prompt, model, max_tokens):
        seen["model"] = model
        seen["instructions"] = instructions
        return ('<|channel|>final<|message|>```json\n{"query": "title:(phosphine Venus)", '
                '"sort": "score desc", "rows": 7, "filters": "property:refereed"}\n```')

    out = ADSQueryBuilder(model="gpt-oss-120b", llm_call=llm).build_query("phosphine Venus", 10, None, None)
    assert out["query"] == "title:(phosphine Venus)"
    assert out["filters"] == ["property:refereed"] and out["rows"] == 7 and out["sort"] == "score desc"
    assert seen["model"] == "gpt-oss-120b" and "NEVER put author names inside abstract" in seen["instructions"]


def test_builder_explicit_caller_sort_wins():
    llm = lambda *a: '{"query": "keyword:\\"disks\\"", "sort": "citation_count desc"}'
    out = ADSQueryBuilder(llm_call=llm).build_query("recent disks", 10, "date desc", None)
    assert out["sort"] == "date desc"


@pytest.mark.parametrize("bad", [
    '{"query": "object:\\"HL Tau\\""}',
    '{"query": "title:\\"unclosed"}',
    '{"query": "a AND (b"}',
    '{"query": ""}',
    'no json at all',
    '',
])
def test_builder_rejects_unusable_output(bad):
    with pytest.raises(ADSQueryBuilderError):
        ADSQueryBuilder(llm_call=lambda *a: bad).build_query("q", 10, None, None)


def test_builder_timeout_raises(monkeypatch):
    monkeypatch.setenv("ADS_QUERY_BUILDER_TIMEOUT_SECONDS", "1")

    def slow(*a):
        time.sleep(3)
        return '{"query": "x"}'

    t0 = time.monotonic()
    with pytest.raises(ADSQueryBuilderError, match="timed out"):
        ADSQueryBuilder(llm_call=slow).build_query("q", 10, None, None)
    assert time.monotonic() - t0 < 2.5


def test_builder_cache_avoids_second_call():
    n = {"calls": 0}

    def llm(*a):
        n["calls"] += 1
        return '{"query": "title:(x)"}'

    b = ADSQueryBuilder(llm_call=llm)
    assert b.build_query("x paper", 10, None, None)["query"] == "title:(x)"
    assert b.build_query("x paper", 10, None, None)["query"] == "title:(x)"
    assert n["calls"] == 1


def test_validate_ads_query():
    assert validate_ads_query('title:"x" AND (a OR b)') is None
    assert "object" in validate_ads_query('simbad:"M31"')
    assert validate_ads_query('"a') == "unbalanced quotes"
    assert validate_ads_query("(a") == "unbalanced parentheses"


# ─────────────────────────────────────────────────────────────────────────────
# Identifiers
# ─────────────────────────────────────────────────────────────────────────────
def test_extract_identifiers():
    x = extract_literature_identifiers("2020NatAs...5..655G")
    assert x["bibcodes"] == ["2020NatAs...5..655G"] and x["only"]
    x = extract_literature_identifiers("Find the paper 2020NatAs...5..655G and tell me its title and authors.")
    assert x["only"] and x["rest"] == ""
    x = extract_literature_identifiers("What paper has the DOI 10.1038/s41550-020-1174-4? Give its title.")
    assert x["dois"] == ["10.1038/s41550-020-1174-4"] and x["only"]
    assert extract_literature_identifiers("What is arXiv:1812.04040?")["arxiv"] == ["1812.04040"]
    assert extract_literature_identifiers("1812.04040v2")["arxiv"] == ["1812.04040"]
    assert extract_literature_identifiers("see astro-ph/0001001")["arxiv"] == ["astro-ph/0001001"]
    x = extract_literature_identifiers("papers citing 2018ApJ...869L..41A on disk substructure")
    assert x["bibcodes"] == ["2018ApJ...869L..41A"] and not x["only"] and "substructure" in x["rest"]
    # a frequency is not an arXiv id, a plain question has no identifiers
    assert not extract_literature_identifiers("the 1420.4058 MHz line")["any"]
    assert not extract_literature_identifiers("first paper reporting phosphine on Venus")["any"]


def test_bibcode_fuzzy_query_and_pick():
    assert bibcode_fuzzy_query("2020NatAs...5..655G") == 'bibstem:"NatAs" AND volume:"5" AND page:"655"'
    assert bibcode_fuzzy_query("2018ApJ...869L..41A") == 'bibstem:"ApJ" AND volume:"869" AND page:"L41"'
    assert bibcode_fuzzy_query("2016A&A...586A.133P") == 'bibstem:"A&A" AND volume:"586" AND page:"A133"'
    assert bibcode_fuzzy_query("short") is None
    cands = [_paper("2021NatAs...5..655X"), _paper("2021NatAs...5..655G"), _paper("2030NatAs...5..655G")]
    assert pick_fuzzy_bibcode_match("2020NatAs...5..655G", cands)["bibcode"] == "2021NatAs...5..655G"
    assert pick_fuzzy_bibcode_match("2020NatAs...5..655G", [_paper("2030NatAs...5..655G")]) is None


def test_wrong_year_bibcode_resolves():
    real = _paper("2021NatAs...5..655G", "Phosphine gas in the cloud decks of Venus")
    ads = _FakeADS({
        'bibcode:"2020NatAs...5..655G"': [],
        'bibstem:"NatAs" AND volume:"5" AND page:"655"': [real],
    })
    out = ads.search_natural_language("2020NatAs...5..655G", max_results=5)
    assert out["query_source"] == "identifier"
    assert out["resolved_from"] == {"2020NatAs...5..655G": "2021NatAs...5..655G"}
    assert [p["bibcode"] for p in out["papers"]] == ["2021NatAs...5..655G"]
    assert all(c["filters"] is None for c in ads.calls), "identifier lookups must not be refereed-only"


def test_doi_and_arxiv_queries_use_one_clause_each():
    ads = _FakeADS(default=[_paper("2018ApJ...869L..41A")])
    ads.search_natural_language("arXiv:1812.04040", max_results=5)
    ads.search_natural_language("doi 10.1038/s41550-020-1174-4", max_results=5)
    qs = [c["query"] for c in ads.calls]
    assert 'identifier:"arXiv:1812.04040"' in qs and 'doi:"10.1038/s41550-020-1174-4"' in qs
    out = ads.search_natural_language("arXiv:1812.04040", max_results=5)
    # a valid arXiv id is a match, not a correction (CX-17)
    assert out["resolved_from"] == {} and out["matched_ids"] == {"arXiv:1812.04040": "2018ApJ...869L..41A"}
    assert not any("doi:(" in q for q in qs), "doi:(...) is a Solr 400"


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic fallback
# ─────────────────────────────────────────────────────────────────────────────
def test_heuristic_no_false_author_and_unfielded_terms():
    q = heuristic_ads_query("Phosphine gas in the cloud decks of Venus Greaves Nature Astronomy")
    assert "author:" not in q and 'bibstem:"NatAs"' in q and "Greaves" in q and "abs:" not in q
    assert heuristic_ads_query("papers by Sean Andrews on disk surveys").startswith('author:"Andrews, Sean"')
    assert 'author:"Greaves"' in heuristic_ads_query("Greaves et al. 2020 phosphine")
    assert "author:" not in heuristic_ads_query("images taken by Hubble Space Telescope of K2-18 b")
    assert '"K2-18"' in heuristic_ads_query("K2-18 b DMS")


def test_heuristic_never_puts_identifiers_in_a_field():
    assert heuristic_ads_query("2020NatAs...5..655G") == "*:*"
    assert ".." not in heuristic_ads_query("phosphine 2020NatAs...5..655G Venus")


# ─────────────────────────────────────────────────────────────────────────────
# Relaxation ladder
# ─────────────────────────────────────────────────────────────────────────────
def test_relaxation_helpers():
    assert widen_year_clause("a AND year:2020")[0] == "a AND year:[2019 TO 2021]"
    assert widen_year_clause("year:[2011 TO 2013] AND a")[0] == "year:[2010 TO 2014] AND a"
    assert widen_year_clause("year:[2015 TO *]")[0] == "year:[2014 TO *]"
    assert widen_year_clause("no year") is None
    assert drop_query_clause('year:2020 AND bibstem:"NatAs" AND phosphine', "year") == 'bibstem:"NatAs" AND phosphine'
    assert drop_query_clause('phosphine AND bibstem:"NatAs"', "bibstem") == "phosphine"
    assert drop_query_clause("phosphine", "year") is None
    labels = [s[0] for s in relaxation_steps('year:2020 AND bibstem:"NatAs" AND x', ["property:refereed"],
                                             rebuild="x AND y")]
    # four steps at most, so the cap never hides the filter drop or the rebuild (CX-07)
    assert labels == ["widened year 2020 to 2019-2021", "dropped the year and the journal",
                      "rebuilt the query without the LLM", "included non-refereed records"]
    assert len(labels) <= ADSService._MAX_RELAXATIONS
    # counting questions keep the year exactly (CX-03)
    kept = relaxation_steps("year:2020 AND x", ["property:refereed"], keep_year=True)
    assert [s[0] for s in kept] == ["included non-refereed records"] and "year:2020" in kept[0][1]


def test_zero_hits_relax_until_found(monkeypatch):
    monkeypatch.setenv("ADS_QUERY_BUILDER", "off")
    hit = _paper("2021NatAs...5..655G")
    ads = _FakeADS({lambda q: "year:[2019 TO 2021]" in q: [hit]})
    out = ads.search_natural_language("Greaves phosphine Venus Nature Astronomy 2020", max_results=5)
    assert out["query_source"] == "fallback"
    assert out["relaxed"] == ["widened year 2020 to 2019-2021"]
    assert out["papers"][0]["bibcode"] == "2021NatAs...5..655G"
    assert [a["hits"] for a in out["attempts"]] == [0, 1]
    assert "disabled" in out["builder_error"]


def test_relaxation_is_capped(monkeypatch):
    monkeypatch.setenv("ADS_QUERY_BUILDER", "off")
    ads = _FakeADS()
    out = ads.search_natural_language("Greaves phosphine Venus Nature Astronomy 2020", max_results=5)
    assert out["papers"] == [] and len(ads.calls) <= 1 + ADSService._MAX_RELAXATIONS
    assert ads.calls[-1]["filters"] is None, "the non-refereed step is reached"


def test_builder_query_rejected_by_ads_falls_back(monkeypatch):
    monkeypatch.setattr(ac, "_default_builder_llm_call", lambda *a: '{"query": "weird:(x)"}')
    ads = _FakeADS({lambda q: q.startswith("phosphine"): [_paper("2021NatAs...5..655G")]},
                   raise_on={"weird:(x)"})
    out = ads.search_natural_language("phosphine Venus", max_results=5)
    assert out["query_source"] == "fallback" and out["papers"]
    assert "rejected" in out["builder_error"]


# ─────────────────────────────────────────────────────────────────────────────
# find_citing_papers
# ─────────────────────────────────────────────────────────────────────────────
def test_find_citing_papers_groups_and_ranks():
    src = _paper("2021NatAs...5..655G", "Phosphine gas in the cloud decks of Venus")
    citing = [
        _paper("2021NatAs...5..631V", "No evidence of phosphine in the atmosphere of Venus", citations=100),
        _paper("2021A&A...649L...1O", "Upper limits for phosphine in the atmosphere of Mars", citations=300),
        _paper("2021NatAs...5..636G", "Reply to: No evidence of phosphine in the atmosphere of Venus"),
        _paper("2025AJ....170..257S", "K2-18b Does Not Meet the Standards of Evidence for Life", citations=50),
    ]
    ads = _FakeADS({
        'bibcode:"2021NatAs...5..655G"': [src],
        lambda q: q.startswith('citations(bibcode:"2021NatAs...5..655G") AND ('): citing,
    })
    out = ads.find_citing_papers("2021NatAs...5..655G")
    reb = [p["bibcode"] for p in out["rebuttals"]]
    assert reb[0] == "2021NatAs...5..631V", "title topic overlap outranks a more-cited Mars paper"
    assert "2025AJ....170..257S" not in reb, "rebuttal words without the paper's topic are not a rebuttal"
    assert "2021A&A...649L...1O" not in reb, "one shared title word is not enough topic overlap (CX-08)"
    assert [p["bibcode"] for p in out["replies"]] == ["2021NatAs...5..636G"]
    assert out["source_title"].startswith("Phosphine") and out["success"]
    assert "not" not in ac._REBUTTAL_TERMS


def test_find_citing_papers_fallback_and_missing_source():
    src = _paper("2020ApJ...900L...1X", "Some paper")
    ads = _FakeADS({'bibcode:"2020ApJ...900L...1X"': [src],
                    'citations(bibcode:"2020ApJ...900L...1X")': [_paper("2022MNRAS.500....1Z", "A follow-up")]})
    out = ads.find_citing_papers("2020ApJ...900L...1X")
    assert out["count"] == 0 and out["note"] and out["other_citing"][0]["bibcode"] == "2022MNRAS.500....1Z"
    missing = _FakeADS().find_citing_papers("2020ApJ...999L...1X")
    assert missing["success"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Capability payloads
# ─────────────────────────────────────────────────────────────────────────────
def test_capability_payloads():
    from capabilities.base import CallContext
    from capabilities.papers import FindCitingPapers, SearchPapers

    state = {}
    ctx = CallContext(services={"console_log": lambda *_: None,
                                "set_last_run_result": lambda v: state.update(lrr=v)})

    class _Ads:
        def search_natural_language(self, question, max_results, sort):
            return {"papers": [], "query": "x", "query_source": "fallback", "relaxed": ["dropped the year"],
                    "attempts": [{"query": "x AND year:2020", "hits": 0}, {"query": "x", "hits": 0}],
                    "resolved_from": {}}

        def find_citing_papers(self, bibcode, focus, max_results):
            return {"success": True, "bibcode": bibcode, "rebuttals": [{"bibcode": "B"}], "replies": [],
                    "papers": [_paper("2021NatAs...5..631V")], "query": "citations(...)"}

    ctx.services["ads_client"] = _Ads()
    out = SearchPapers().run(SearchPapers.InputModel(query="x"), ctx).to_native()
    assert out["count"] == 0 and out["query_source"] == "fallback" and out["relaxed"] == ["dropped the year"]
    assert "Do not re-run this search with small rewordings" in out["hint"] and len(out["attempts"]) == 2
    out = FindCitingPapers().run(FindCitingPapers.InputModel(bibcode="2021NatAs...5..655G"), ctx).to_native()
    assert out["success"] and "papers" not in out and state["lrr"]["type"] == "papers"


def test_literature_pack_offered_for_pushback_questions():
    from core import tool_packs

    pat = tool_packs.PACKS["literature"]["re"]
    assert "find_citing_papers" in tool_packs.PACKS["literature"]["tools"]
    assert pat.search("which studies pushed back on that detection")
    assert pat.search("who disputed the BICEP2 claim")


# ─────────────────────────────────────────────────────────────────────────────
# core/turn_guards.py
# ─────────────────────────────────────────────────────────────────────────────
def test_repair_tool_name():
    known = {"search_papers", "find_tools"}
    assert tg.repair_tool_name("search_papers<|channel|>commentary", known) == "search_papers"
    assert tg.repair_tool_name("functions.search_papers", known) == "search_papers"
    assert tg.repair_tool_name("search_papers <|constrain|>json", known) == "search_papers"
    assert tg.repair_tool_name("datlab_typo<|channel|>x", known) == "datlab_typo<|channel|>x"
    assert tg.repair_tool_name("search_papers", known) == "search_papers"


def test_budget_status_text_keeps_ui_prefix():
    assert tg.budget_status_text("max tool rounds reached (8)") == \
        "Tool budget reached (all 8 tool rounds used), composing the final answer from collected results"
    assert "time limit reached after 140 s" in tg.budget_status_text("tool time budget reached (140s)")
    assert tg.budget_status_text(None).startswith("Tool budget reached, composing")
    assert "—" not in tg.budget_status_text("token budget / tool budget reached")


def test_literature_guard_blocks_rewordings_of_empty_searches():
    g = tg.LiteratureSearchGuard()
    empty = json.dumps({"success": True, "count": 0, "papers": []})
    qs = ["Greaves phosphine Venus Nature Astronomy 2020", "phosphine Venus Greaves Nature Astronomy",
          "Greaves Venus phosphine 2020"]
    for q in qs:
        assert g.check("search_papers", {"query": q}) is None
        g.record("search_papers", {"query": q}, empty)
    hint = g.check("search_papers", {"query": "phosphine Venus Nature Astronomy Greaves"})
    assert hint and "Stop rephrasing" in hint
    # something genuinely different still runs; other tools are untouched
    assert g.check("search_papers", {"query": "M87 event horizon shadow"}) is None
    assert g.check("web_search", {"query": qs[0]}) is None


def test_literature_guard_resets_after_a_hit():
    g = tg.LiteratureSearchGuard()
    empty = json.dumps({"success": True, "count": 0})
    for q in ["a b c", "a b d", "a b e"]:
        g.record("search_papers", {"query": q}, empty)
    g.record("search_papers", {"query": "z"}, json.dumps({"success": True, "count": 3}))
    assert g.check("search_papers", {"query": "a b c"}) is None


SCREENSHOT_LEAK = (
    'The user asks: "Which paper first reported phosphine in the atmosphere of Venus, and which papers pushed '
    'back on that detection? Give me bibcodes."\n\nWe need to retrieve papers from ADS. The first detection claim '
    'was the 2020 Nature paper. Bibcode: 2020Natur.586..37G (I think). Actually the Nature paper: "Phosphine gas '
    'in the cloud decks of Venus" maybe?'
)


def test_leak_detector():
    assert tg.looks_like_leaked_reasoning(SCREENSHOT_LEAK)
    assert tg.looks_like_leaked_reasoning("We need to retrieve the papers first.")
    assert tg.looks_like_leaked_reasoning("Okay, so the user wants the bibcodes.")
    for good in (
        "**Detection paper**: Greaves et al. 2021, 2021NatAs...5..655G.",
        "The first report was Greaves et al. (2021NatAs...5..655G). Several groups re-analysed the data.",
        "## Results\n\nWe find three papers. To confirm this we need to check the ALMA data, which I did.",
        "",
    ):
        assert not tg.looks_like_leaked_reasoning(good), good


def test_strip_leaked_reasoning():
    assert tg.strip_leaked_reasoning(SCREENSHOT_LEAK) == ""
    kept = tg.strip_leaked_reasoning(SCREENSHOT_LEAK + "\n\n**Detection**: 2021NatAs...5..655G.")
    assert kept == "**Detection**: 2021NatAs...5..655G."


def test_leak_guard_only_for_gpt_oss():
    assert tg.model_leaks_reasoning("gpt-oss-120b")
    assert not tg.model_leaks_reasoning("deepseek-v4-pro")


def test_final_note_leads_with_the_answer():
    from core import runner

    note = runner.TOOL_BUDGET_FINAL_NOTE.format(reason="x")
    assert "Lead with the" in note and "do not restate the question" in note


def test_citing_followup_hint():
    ok = json.dumps({"success": True, "count": 1, "papers": [{"bibcode": "2021NatAs...5..655G"}]})
    q = "Which paper first reported phosphine on Venus, and which papers pushed back on that detection?"
    out = tg.citing_followup_hint(q, "search_papers", ok, set())
    assert out and "find_citing_papers" in json.loads(out)["next_step"]
    assert "2021NatAs...5..655G" in json.loads(out)["next_step"]
    # not for other questions, other tools, empty results, or once the tool was called
    assert tg.citing_followup_hint("recent papers on disks", "search_papers", ok, set()) is None
    assert tg.citing_followup_hint(q, "ads_search", ok, set()) is None
    assert tg.citing_followup_hint(q, "search_papers", json.dumps({"success": True, "papers": []}), set()) is None
    assert tg.citing_followup_hint(q, "search_papers", ok, {"find_citing_papers"}) is None
    # the earliest results are offered as the original-paper candidates (CX-20)
    many = json.dumps({"success": True, "papers": [
        {"bibcode": "2021NatAs...5..631V", "year": "2021", "title": "No evidence of phosphine"},
        {"bibcode": "2020A&A...644L...2S", "year": "2020", "title": "Re-analysis"},
        {"bibcode": "2019XYZ....1....1A", "year": "2019", "title": "Earliest"}]})
    hint = json.loads(tg.citing_followup_hint(q, "search_papers", many, set()))["next_step"]
    assert hint.index("2019XYZ....1....1A") < hint.index("2021NatAs...5..631V") and "ORIGINAL" in hint
    assert tg.asks_for_responses("which papers proposed alternative explanations for it?")
    assert tg.asks_for_responses("Which papers challenged the BICEP2 claim?")


# ─────────────────────────────────────────────────────────────────────────────
# Guard task-efccf37-126 follow-ups
# ─────────────────────────────────────────────────────────────────────────────
def test_builder_carries_request_context_into_its_thread():
    from core.llm_client import get_llm_request_context, llm_request_context

    seen = {}

    def llm(*a):
        ctx = get_llm_request_context()
        seen["user"] = getattr(ctx, "user_id", None)
        return '{"query": "x"}'

    with llm_request_context(user_id="u-42"):
        ADSQueryBuilder(llm_call=llm).build_query("q ctx", 10, None, None)
    assert seen["user"] == "u-42"  # CX-01


def test_builder_is_bounded_when_busy():
    assert ac._BUILDER_SLOTS.acquire(blocking=False) and ac._BUILDER_SLOTS.acquire(blocking=False)
    try:
        with pytest.raises(ADSQueryBuilderError, match="busy"):
            ADSQueryBuilder(llm_call=lambda *a: '{"query": "x"}').build_query("q busy", 10, None, None)
    finally:
        ac._BUILDER_SLOTS.release()
        ac._BUILDER_SLOTS.release()  # CX-02


def test_builder_rows_never_exceed_the_request(monkeypatch):
    out = ADSQueryBuilder(llm_call=lambda *a: '{"query": "x", "rows": 200}').build_query("q rows", 15, None, None)
    assert out["rows"] == 15  # CX-12
    monkeypatch.setattr(ac, "_default_builder_llm_call", lambda *a: '{"query": "x", "rows": 200}')
    ads = _FakeADS(default=[_paper("2021NatAs...5..655G")])
    ads.search_natural_language("x papers", max_results=15)
    assert ads.calls[-1]["rows"] == 15


def test_identifier_lookup_errors_are_not_no_record():
    ads = _FakeADS(raise_on={'bibcode:"2020NatAs...5..655G"'})
    with pytest.raises(ADSServiceError, match="lookup failed"):
        ads.search_natural_language("2020NatAs...5..655G", max_results=5)  # CX-04


def test_counting_question_keeps_its_year(monkeypatch):
    monkeypatch.setenv("ADS_QUERY_BUILDER", "off")
    ads = _FakeADS({lambda q: "2021" in q or "TO" in q: [_paper("2021NatAs...5..655G")]})
    out = ads.search_natural_language("how many papers on phosphine Venus in 2020", max_results=5)
    assert out["papers"] == [] and all("year:2020" in c["query"] for c in ads.calls)  # CX-03


def test_relaxed_results_are_flagged_for_the_model(monkeypatch):
    monkeypatch.setenv("ADS_QUERY_BUILDER", "off")
    ads = _FakeADS({lambda q: "year:[2019 TO 2021]" in q: [_paper("2021NatAs...5..655G")]})
    out = ads.search_natural_language("Greaves phosphine Venus 2020", max_results=5)
    assert out["relaxed_results"] is True
    from capabilities.papers import _search_provenance
    prov = _search_provenance(out, out["papers"])
    assert "do not count them as matches" in prov["relaxed_note"]
    assert prov["count"] == 0 and prov["relaxed_count"] == 1  # CX-03 reopen: no contradiction


def test_only_syntax_errors_trigger_the_heuristic_retry(monkeypatch):
    monkeypatch.setattr(ac, "_default_builder_llm_call", lambda *a: '{"query": "fine:(x)"}')

    class _Down(_FakeADS):
        def search_papers(self, query, **kw):
            self.calls.append({"query": query})
            raise ADSServiceError("ADS API error 503: unavailable")

    ads = _Down()
    with pytest.raises(ADSServiceError, match="503"):
        ads.search_natural_language("phosphine Venus", max_results=5)
    assert len(ads.calls) == 1  # CX-13: no second call for an outage


def test_cited_question_returns_the_citing_papers():
    src = _paper("2018ApJ...869L..41A", "DSHARP I")
    ads = _FakeADS({'bibcode:"2018ApJ...869L..41A"': [src],
                    'citations(bibcode:"2018ApJ...869L..41A")': [_paper("2019ApJ...880L..10Z", "Follow-up")]})
    out = ads.search_natural_language("Which papers cited 2018ApJ...869L..41A?", max_results=5)
    # CX-11 (reopen): the citing papers themselves, the cited paper kept aside
    assert out["query_source"] == "citations" and out["cited_papers"][0]["bibcode"] == "2018ApJ...869L..41A"
    assert [p["bibcode"] for p in out["papers"]] == ["2019ApJ...880L..10Z"]
    assert ads.calls[-1]["sort"] == "citation_count desc"


def test_streak_guard_ignores_failures_and_allows_identifiers():
    g = tg.LiteratureSearchGuard()
    fail = json.dumps({"success": False, "error": "ADS API error 503"})
    for q in ["phosphine Venus a", "phosphine Venus b", "phosphine Venus c"]:
        g.record("search_papers", {"query": q}, fail)
    assert g.check("search_papers", {"query": "phosphine Venus d"}) is None  # CX-05
    empty = json.dumps({"success": True, "count": 0})
    for q in ["phosphine Venus a", "phosphine Venus b", "phosphine Venus c"]:
        g.record("search_papers", {"query": q}, empty)
    assert g.check("search_papers", {"query": "phosphine Venus 2021NatAs...5..655G"}) is None  # CX-06
    assert g.check("search_papers", {"query": "phosphine Venus e"})


def test_leak_detector_labelled_and_direct_answers():
    assert tg.looks_like_leaked_reasoning("Analysis:\nThe user asks for bibcodes. We need to search.")
    assert not tg.looks_like_leaked_reasoning("We need to use independent dust maps to separate the foreground.")
    assert tg.strip_leaked_reasoning("Analysis:\n\nThe user asks X.\n\n**Answer**: Y") == "**Answer**: Y"  # CX-09


def test_find_citing_reports_bounded_scan():
    src = _paper("2021NatAs...5..655G", "Phosphine gas in the cloud decks of Venus")
    many = [_paper(f"2021XXX....{i:04d}A"[:19].ljust(19, "A"), "Re-analysis of phosphine on Venus") for i in range(100)]
    ads = _FakeADS({'bibcode:"2021NatAs...5..655G"': [src],
                    lambda q: q.startswith('citations(bibcode:"2021NatAs...5..655G") AND ('): many})
    out = ads.find_citing_papers("2021NatAs...5..655G")
    assert out["scanned"] == 100 and "100 most-cited" in out["scan_note"]  # CX-15


def test_counting_question_is_never_relaxed(monkeypatch):
    monkeypatch.setenv("ADS_QUERY_BUILDER", "off")
    ads = _FakeADS({lambda q: "bibstem" not in q: [_paper("2020MNRAS.500....1Z")]})
    out = ads.search_natural_language("How many refereed AJ papers on stellar streams in 2020?", max_results=5)
    assert out["papers"] == [] and out["relaxed"] == [] and len(ads.calls) == 1  # CX-03 reopen


def test_partial_identifier_failure_is_reported():
    a = _paper("2021NatAs...5..655G")

    class _Ads(_FakeADS):
        def search_papers(self, query, **kw):
            self.calls.append({"query": query, "filters": kw.get("filters")})
            if query.startswith("bibstem:"):
                raise ADSServiceError("ADS API error 503")
            return [dict(a)] if "655G" in query else []

    out = _Ads().search_natural_language("2021NatAs...5..655G 2020ApJ...999L..12Q", max_results=5)
    assert [p["bibcode"] for p in out["papers"]] == ["2021NatAs...5..655G"]
    assert out["identifier_errors"] and "503" in out["identifier_errors"][0]  # CX-04 reopen
    from capabilities.papers import _search_provenance
    assert _search_provenance(out, out["papers"])["partial"] is True


def test_one_topic_word_in_title_and_abstract_is_still_one():
    src = _paper("2021NatAs...5..655G", "Phosphine gas in the cloud decks of Venus")
    mars = _paper("2021A&A...649L...1O", "Upper limits for phosphine in the atmosphere of Mars",
                  abstract="We searched for phosphine on Mars and report upper limits.")
    ads = _FakeADS({'bibcode:"2021NatAs...5..655G"': [src],
                    lambda q: q.startswith('citations(bibcode:"2021NatAs...5..655G") AND ('): [mars]})
    out = ads.find_citing_papers("2021NatAs...5..655G")
    assert out["rebuttals"] == []  # CX-08 reopen


def test_server_500_is_not_a_syntax_error():
    assert not ac._is_query_syntax_error(ADSServiceError("ADS API error 500: Internal Server Error"))
    assert ac._is_query_syntax_error(ADSServiceError('ADS API error 500: {"responseHeader":{"status":500}}'))
    assert ac._is_query_syntax_error(ADSServiceError("ADS API error 400: bad"))  # CX-13 reopen


def test_late_doi_and_arxiv_lookups_are_skipped(monkeypatch):
    monkeypatch.setattr(ADSService, "_EXTRA_REQUESTS_WALL_S", -1.0)
    ads = _FakeADS(default=[_paper("2021NatAs...5..655G")])
    out = ads.search_natural_language("2021NatAs...5..655G 10.1038/s41550-020-1174-4 arXiv:1812.04040",
                                      max_results=5)
    qs = [c["query"] for c in ads.calls]
    assert qs == ['bibcode:"2021NatAs...5..655G"'], qs  # CX-14 reopen: only the first request
    assert any("skipped" in e for e in out["identifier_errors"])


def test_builder_releases_the_semaphore_it_acquired(monkeypatch):
    import threading

    monkeypatch.setenv("ADS_QUERY_BUILDER_TIMEOUT_SECONDS", "1")
    original = threading.BoundedSemaphore(2)
    monkeypatch.setattr(ac, "_BUILDER_SLOTS", original)
    done = threading.Event()

    def slow(*a):
        done.wait(5)
        return '{"query": "x"}'

    with pytest.raises(ADSQueryBuilderError, match="timed out"):
        ADSQueryBuilder(llm_call=slow).build_query("q swap", 10, None, None)
    swapped = threading.BoundedSemaphore(2)
    monkeypatch.setattr(ac, "_BUILDER_SLOTS", swapped)
    done.set()
    for t in threading.enumerate():
        if t.name == "ads-query-builder":
            t.join(5)
    # the worker released the semaphore it acquired, not the swapped global (CX-22)
    assert original.acquire(blocking=False) and original.acquire(blocking=False)
    assert swapped.acquire(blocking=False) and swapped.acquire(blocking=False)


# ── verify round 2 (CX-08, CX-11, CX-19, CX-23, CX-24) ──────────────────────
def test_abstract_only_overlap_is_a_possible_rebuttal_not_a_rebuttal():
    src = _paper("2021NatAs...5..655G", "Phosphine gas in the cloud decks of Venus")
    mars = _paper("2021A&A...649L...1O", "Upper limits for phosphine in the atmosphere of Mars",
                  abstract="Our limits are compared with the phosphine reported on Venus.")
    ven = _paper("2021NatAs...5..631V", "No evidence of phosphine in the atmosphere of Venus")
    ads = _FakeADS({'bibcode:"2021NatAs...5..655G"': [src],
                    lambda q: q.startswith('citations(bibcode:"2021NatAs...5..655G") AND ('): [mars, ven]})
    out = ads.find_citing_papers("2021NatAs...5..655G")
    assert [p["bibcode"] for p in out["rebuttals"]] == ["2021NatAs...5..631V"]
    assert [p["bibcode"] for p in out["possible_rebuttals"]] == ["2021A&A...649L...1O"]
    assert "Check each title" in out["possible_note"]  # CX-08 round 2


def test_cited_question_with_topic_words_and_two_papers():
    a, b = _paper("2018ApJ...869L..41A", "DSHARP I"), _paper("2021NatAs...5..655G", "Phosphine")
    ads = _FakeADS({'bibcode:"2018ApJ...869L..41A" OR bibcode:"2021NatAs...5..655G"': [a, b],
                    'bibcode:"2018ApJ...869L..41A"': [a]},
                   default=[_paper("2019ApJ...880L..10Z", "Follow-up")])
    out = ads.search_natural_language("Which papers cited 2018ApJ...869L..41A on disk substructure?", max_results=5)
    last = ads.calls[-1]["query"]
    assert out["query_source"] == "citations" and last.startswith('citations(bibcode:"2018ApJ...869L..41A")')
    assert "disk" in last and "substructure" in last and "cited" not in last  # CX-11 round 2
    ads.calls.clear()
    out = ads.search_natural_language("Which papers cited 2018ApJ...869L..41A and 2021NatAs...5..655G?", max_results=5)
    assert ads.calls[-1]["query"] == 'citations(bibcode:"2018ApJ...869L..41A" OR bibcode:"2021NatAs...5..655G")'
    assert len(out["cited_papers"]) == 2  # CX-23


def test_fuzzy_recovery_cap_is_reported():
    ads = _FakeADS()  # nothing exists
    with pytest.raises(ADSServiceError) as e:
        ads.search_natural_language("2020NatAs...5..655G 2020NatAs...5..656G 2020NatAs...5..657G", max_results=5)
    assert "recovery skipped" in str(e.value) or "2020NatAs...5..657G" in str(e.value)  # CX-24
    real = _paper("2021NatAs...5..655G")
    ads = _FakeADS({'bibcode:"2020NatAs...5..655G" OR bibcode:"2020NatAs...5..656G" OR bibcode:"2020NatAs...5..657G"': [],
                    'bibstem:"NatAs" AND volume:"5" AND page:"655"': [real]})
    out = ads.search_natural_language("2020NatAs...5..655G 2020NatAs...5..656G 2020NatAs...5..657G", max_results=5)
    assert any("2020NatAs...5..657G" in e and "skipped" in e for e in out["identifier_errors"])


def test_sort_shorthands_are_normalized():
    assert ac.normalize_ads_sort("date") == "date desc"
    assert ac.normalize_ads_sort("Citation_Count desc") == "citation_count desc"
    assert ac.normalize_ads_sort("relevance") == "score desc"
    assert ac.normalize_ads_sort("bogus order") is None and ac.normalize_ads_sort(None) is None
    ads = _FakeADS(default=[_paper("2021NatAs...5..655G")])
    import os
    os.environ["ADS_QUERY_BUILDER"] = "off"
    try:
        ads.search_natural_language("phosphine Venus", max_results=5, sort="date")
    finally:
        del os.environ["ADS_QUERY_BUILDER"]
    assert ads.calls[-1]["sort"] == "date desc"


# ── verify round 3 (CX-11, CX-23, CX-25) ─────────────────────────────────────
class _CountingADS(_FakeADS):
    def count_matches(self, query, filters=None):
        self.calls.append({"query": "COUNT " + query, "filters": filters})
        return 1234


def test_present_tense_cite_and_both_and_totals():
    a, b = _paper("2018ApJ...869L..41A", "DSHARP I"), _paper("2021NatAs...5..655G", "Phosphine")
    ads = _CountingADS({'bibcode:"2018ApJ...869L..41A" OR bibcode:"2021NatAs...5..655G"': [a, b],
                        'bibcode:"2018ApJ...869L..41A"': [a]},
                       default=[_paper("2019ApJ...880L..10Z", "Follow-up")])
    out = ads.search_natural_language("Which papers cite 2018ApJ...869L..41A?", max_results=15)
    assert out["query_source"] == "citations"  # CX-11 round 3: present tense
    assert out["total"] == 1234  # CX-25
    ads.calls.clear()
    out = ads.search_natural_language("Which papers cited both 2018ApJ...869L..41A and 2021NatAs...5..655G?",
                                      max_results=15)
    q = [c["query"] for c in ads.calls if not c["query"].startswith("COUNT")][-1]
    assert q == '(citations(bibcode:"2018ApJ...869L..41A") AND citations(bibcode:"2021NatAs...5..655G"))'
    assert out["combine"] == "all" and "both" not in q  # CX-23 round 3
    from capabilities.papers import _search_provenance
    prov = _search_provenance(out, out["papers"])
    assert prov["count"] == 1234 and prov["returned"] == len(out["papers"])  # CX-25


def test_counting_topic_question_reports_the_ads_total(monkeypatch):
    monkeypatch.setenv("ADS_QUERY_BUILDER", "off")
    ads = _CountingADS(default=[_paper("2020MNRAS.500....1Z")])
    out = ads.search_natural_language("How many papers on stellar streams in 2020?", max_results=5)
    assert out["total"] == 1234 and any(c["query"].startswith("COUNT year:2020") for c in ads.calls)


def test_unknown_total_is_reported_not_implied():
    from capabilities.papers import _search_provenance

    page = [_paper(f"2021NatAs...5..{i:03d}G") for i in range(15)]
    result = {"papers": page, "query": "citations(...)", "query_source": "citations",
              "total": None, "total_requested": True}
    prov = _search_provenance(result, page)
    assert prov["total_unknown"] is True and prov["partial"] is True and prov["returned"] == 15
    assert "not the total" in prov["total_note"]  # CX-25 round 4

    class _NoTotal(_FakeADS):
        def count_matches(self, query, filters=None):
            raise ADSServiceError("ADS API error 503")

    a = _paper("2018ApJ...869L..41A")
    ads = _NoTotal({'bibcode:"2018ApJ...869L..41A"': [a]}, default=page)
    out = ads.search_natural_language("How many papers cite 2018ApJ...869L..41A?", max_results=15)
    assert out["total"] is None and out["total_requested"] and len(out["papers"]) == 15
    assert any("503" in e for e in out["identifier_errors"])
