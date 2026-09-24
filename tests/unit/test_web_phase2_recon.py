"""Regression tests for the Phase 2 independent review (P2-01 .. P2-19), one or
more per finding that was FIXED. Negative controls where a timing or safety
bound is involved."""
from __future__ import annotations

import inspect
import json
import re
import threading
import time
from types import SimpleNamespace

import pytest

from core import web_planner as wp
from core import web_revise as wr
from services import web_domain_packs as packs
from services.web_evidence import EvidenceRegistry, unsupported_citation_claims


def _payload(*pages):
    return {"success": True, "provider": "Tavily", "results": [
        {"url": u, "title": t, "snippet": s} for u, t, s in pages]}


# P2-01: the planner never reads the process-wide SessionMemory


def test_p2_01_planner_context_has_no_session_memory():
    import core.runner as runner
    src = inspect.getsource(runner._stream_response_api_impl)
    assert "agent.session_memory.get_context()" not in src
    assert '_session_ctx = ""' in src


# P2-02: a revision that drops the tag and keeps the value is rejected


def _reg_dates():
    reg = EvidenceRegistry(query="q")
    reg.add_from_payload(_payload(("https://alma.org/pg", "Guide", "Proposals close 15:00 UTC. No date given here.")))
    reg.render_prompt_block()
    return reg


def test_p2_02_revise_rejects_dropping_the_tag_and_keeping_the_value():
    reg = _reg_dates()
    text = "Proposals are due 23 April 2026 [W1]."

    def llm(prompt, model, max_tokens):
        return json.dumps({"revisions": [{"original": text, "revised": "Proposals are due 23 April 2026."}]})
    out, info = wr.revise_unsupported(text, reg, llm_call=llm, budget_s=2)
    assert out == text and info["revised"] == 0


def test_p2_02_revise_accepts_removing_the_unsupported_value_with_the_tag_kept():
    reg = _reg_dates()
    text = "Proposals are due 23 April 2026 [W1]."

    def llm(prompt, model, max_tokens):
        return json.dumps({"revisions": [{"original": text, "revised": "Proposals are due on a date not stated [W1]."}]})
    out, info = wr.revise_unsupported(text, reg, llm_call=llm, budget_s=2)
    assert "not stated [W1]" in out and info["revised"] == 1 and unsupported_citation_claims(out, reg) == []


# P2-03: catalog ids and tool data in the same row are not web claims; revise may not delete supported values


def test_p2_03_catalog_identifiers_are_not_checkable_numbers():
    reg = EvidenceRegistry(query="q")
    reg.add_from_payload(_payload(("https://a.org/x", "T", "The proprietary period is 12 months.")))
    reg.render_prompt_block()
    assert unsupported_citation_claims("For NGC 1068 the proprietary period is 12 months [W1].", reg) == []
    assert unsupported_citation_claims("M 87 and 3C 273 data: 12 months [W1].", reg) == []
    # a new clause after the tag about tool data is not attributed to the web source
    assert unsupported_citation_claims("The period is 12 months [W1]; the archive lists 37 projects for it.", reg) == []
    # but a short continuation still is
    assert unsupported_citation_claims("The period is 12 months [W1], starting 3 March 2027.", reg)


def test_p2_03_revise_never_removes_a_supported_or_tool_value():
    reg = EvidenceRegistry(query="q")
    reg.add_from_payload(_payload(("https://a.org/x", "T", "The proprietary period is 12 months.")))
    reg.render_prompt_block()
    text = "| Cycle 13 | 37 projects | 12 months, from 3 March 2027 [W1] |"
    assert unsupported_citation_claims(text, reg)          # the date is unsupported, the rest is fine

    def llm(prompt, model, max_tokens):
        return json.dumps({"revisions": [{"original": text, "revised": "| Cycle 13 | not stated | 12 months [W1] |"}]})   # removed 37
    out, info = wr.revise_unsupported(text, reg, llm_call=llm, budget_s=2)
    assert out == text and info["revised"] == 0

    def llm2(prompt, model, max_tokens):
        return json.dumps({"revisions": [{"original": text, "revised": "| Cycle 13 | 37 projects | 12 months, start not stated [W1] |"}]})
    out, info = wr.revise_unsupported(text, reg, llm_call=llm2, budget_s=2)
    assert "37 projects" in out and "3 March" not in out and info["revised"] == 1


# P2-04 / P2-14: the RAG freshness supplement is gated by the planner decision and the turn mode


def test_p2_04_rag_supplement_is_gated_by_the_planner_decision_and_the_turn_switch():
    import core.runner as runner
    src = inspect.getsource(runner._stream_response_api_impl)
    i = src.index("_needs_web_supplement = False")
    block = src[i: i + 1400]
    assert "and not (_web_planner.planner_enabled() and _web_decision_sent)" in block
    assert "if not _web_allowed_turn:" in block
    assert "or not web_search" not in block


# P2-05: the legacy classifier is bounded


def test_p2_05_legacy_classifier_times_out_to_no_web(monkeypatch):
    from core.agent import QuasarAgent

    class Slow:
        class responses:
            @staticmethod
            def create(**kw):
                time.sleep(1.0)
                return SimpleNamespace(output_text="YES")

    monkeypatch.setenv("QUASAR_WEB_INTENT_TIMEOUT", "0.2")
    import core.llm_client as llm
    monkeypatch.setattr(llm, "LLMClient", lambda model=None, **kw: Slow())
    agent = SimpleNamespace(memory=SimpleNamespace(get_last_n_turns=lambda n: []))
    t0 = time.monotonic()
    assert QuasarAgent._detect_web_search_needed_via_llm(agent, "Who won the Nobel prize in physics this year?") is False
    assert time.monotonic() - t0 < 0.8


def test_p2_05_legacy_classifier_negative_control_without_the_bound_waits(monkeypatch):
    from core.agent import QuasarAgent

    class Slow:
        class responses:
            @staticmethod
            def create(**kw):
                time.sleep(0.6)
                return SimpleNamespace(output_text="YES")

    monkeypatch.setenv("QUASAR_WEB_INTENT_TIMEOUT", "5")
    import core.llm_client as llm
    monkeypatch.setattr(llm, "LLMClient", lambda model=None, **kw: Slow())
    agent = SimpleNamespace(memory=SimpleNamespace(get_last_n_turns=lambda n: []))
    t0 = time.monotonic()
    assert QuasarAgent._detect_web_search_needed_via_llm(agent, "Who won the Nobel prize in physics this year?") is True
    assert time.monotonic() - t0 >= 0.55


# P2-07: carried pages keep their ids; new pages continue after them


def test_p2_07_carried_pages_keep_their_ids_through_reorder():
    reg = EvidenceRegistry(query="q")
    carried = [
        {"id": "W2", "url": "https://science.nrao.edu/config", "title": "Config plans", "snippet": "D until Oct 19"},
        {"id": "W5", "url": "https://en.wikipedia.org/wiki/VLA", "title": "VLA", "snippet": "x"},
    ]
    added = reg.add_carried(carried)
    assert [ev.id for ev in added] == ["W2", "W5"]
    reg.add_from_payload(_payload(("https://stackexchange.com/q", "SE", "BnA"), ("https://public.nrao.edu/x", "NRAO", "y")))
    assert reg.ids() == ["W2", "W5", "W6", "W7"]           # new ids continue past the carried ones
    ordered = packs.order_evidence(reg.items(), pack="nrao", query="q")
    assert reg.reorder(ordered)
    by_url = {ev.url: ev.id for ev in reg.items()}
    assert by_url["https://science.nrao.edu/config"] == "W2" and by_url["https://en.wikipedia.org/wiki/VLA"] == "W5"
    others = [ev.id for ev in reg.items() if ev.origin != "carried"]
    assert set(others) == {"W1", "W3"} and len(set(reg.ids())) == 4
    # a carried id that collides with an existing one gets a fresh number instead
    reg2 = EvidenceRegistry(query="q")
    reg2.add_from_payload(_payload(("https://a.org/1", "A", "a")))
    ev = reg2.add_carried([{"id": "W1", "url": "https://b.org/1", "title": "B", "snippet": "b"}])
    assert ev[0].id == "W2"


# P2-08 / P2-09: the carry store is keyed by user and conversation and holds the previous turn only


def test_p2_08_carry_memory_key_includes_the_user():
    import core.runner as runner
    src = inspect.getsource(runner._stream_response_api_impl)
    assert src.count("f\"{user_id or 'anonymous'}|{conversation_id}\"") >= 2
    mem = wp.ConversationWebMemory()
    mem.remember("alice|c1", evidence=[{"url": "https://a.org/x"}])
    assert mem.recall("bob|c1") == {"evidence": [], "entities": []}


def test_p2_09_a_turn_without_citable_pages_clears_the_carry():
    mem = wp.ConversationWebMemory()
    mem.remember("u|c", evidence=[{"url": "https://alma.org/x"}], entities=["ALMA"])
    mem.remember("u|c", evidence=(), entities=["JWST"])
    assert mem.recall("u|c")["evidence"] == [] and mem.recall("u|c")["entities"] == ["JWST", "ALMA"]
    assert not wp.looks_like_follow_up("Is it true that JWST Cycle 5 closed in October?")   # 9 words, not a follow-up


# P2-10: trace bookkeeping records are not entities


def test_p2_10_pre_pass_and_check_records_are_not_entities():
    trace = [
        {"tool": "web_search", "args": {"query": "ALMA Cycle 13 proprietary period", "source": "pre-pass"}},
        {"tool": "web_citation_check", "args": {"source": "answer verifier", "check": "web_citation_support"}},
        {"tool": "search_by_target", "args": {"target": "NGC 1068"}},
    ]
    assert wp.conversation_entities(trace) == ["NGC 1068"]


# P2-11: D15 gaps


@pytest.mark.parametrize("query", [
    "What does the Event Horizon Telescope study?",
    "Tell me about the ALMA publication policy",
    "Tell me about Green Bank",
    "Tell me about Kitt Peak",
])
def test_p2_11_facilities_and_places_are_not_researchers(query):
    assert not wp.looks_like_researcher_query(query)


def test_p2_11_people_still_are():
    assert wp.looks_like_researcher_query("What does Alyssa Goodman study?")
    assert wp.looks_like_researcher_query("What is Paola Caselli's h-index?")


# P2-15: a semester is not a cycle


def test_p2_15_semester_is_not_a_cycle_number():
    plan = wp.parse_plan('{"need_web": true, "queries": ["q"], "entities": {"cycle": "2026A"}}')
    assert plan.entities["cycle"] is None
    plan = wp.parse_plan('{"need_web": true, "queries": ["q"], "entities": {"cycle": "Cycle 13"}}')
    assert plan.entities["cycle"] == "13"


# P2-16: the status labels stay truthful


def test_p2_16_reading_label_counts_only_readable_pages_and_late_stays_late():
    import core.runner as runner
    src = inspect.getsource(runner._stream_response_api_impl)
    assert "not _web_evidence._skip_deep_read(ev.url)][:3]" in src
    assert "if label in _late_labels:" in src


# P2-17: the log decorator is back on the Phase 1 search


def test_p2_17_tavily_web_search_keeps_its_log_decorator():
    src = open("core/agent.py", encoding="utf-8").read()
    assert "    @log_tool\n    def _tavily_web_search(" in src
    assert "    @log_tool\n    def _web_search_plan(" not in src


# P2-18: cite_as is assigned under the registry lock


def test_p2_18_cite_as_assigned_under_the_lock():
    src = open("services/web_evidence.py", encoding="utf-8").read()
    i = src.index("def _tag_list(value: Any) -> Any:")
    block = src[i: i + 1200]
    assert block.index("with self._lock:") < block.index('item["cite_as"] = f"[{ev.id}]"')


# P2-19: a urls list becomes a readable query string


def test_p2_19_badge_correction_joins_url_lists():
    import core.runner as runner
    src = inspect.getsource(runner._stream_response_api_impl)
    assert '", ".join(str(u) for u in _q_or_urls[:3])' in src


# P2-13: a revision may not add a link


def test_p2_13_revise_rejects_added_links():
    reg = _reg_dates()
    text = "Proposals are due 23 April 2026 [W1]."

    def llm(prompt, model, max_tokens):
        return json.dumps({"revisions": [{"original": text, "revised": "Proposals are due on a date not stated [W1], see [guide](https://phish.example/g)."}]})
    out, info = wr.revise_unsupported(text, reg, llm_call=llm, budget_s=2)
    assert out == text and info["revised"] == 0


# live TRN-01 (final run): a revise answer cut by the token budget is salvaged object by object


def test_revise_salvages_complete_revisions_from_a_cut_answer():
    reg = _reg_dates()
    text = "Proposals are due 23 April 2026 [W1].\nReviews close 3 June 2026 [W1]."
    cut = ('{"revisions": [{"original": "Proposals are due 23 April 2026 [W1].", "revised": "Proposals are due on a date not stated [W1]."}, '
           '{"original": "Reviews close 3 June 2026 [W1].", "revised": "Reviews close on a d')
    out, info = wr.revise_unsupported(text, reg, llm_call=lambda p, m, t: cut, budget_s=2)
    assert "not stated [W1]" in out and "3 June 2026 [W1]" in out and info["revised"] == 1
    assert wr._salvage_revisions("no json here") == []
