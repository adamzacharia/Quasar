"""Web search redesign, Phase 2: planner (core/web_planner.py), domain packs and
ordering (services/web_domain_packs.py), the revise pass (core/web_revise.py),
the support-checker normalisation fixes, and the runner wiring
(planner decision, mode toggle, web_decision event, carried evidence, E)."""
from __future__ import annotations

import json
import re
import threading
import time
from types import SimpleNamespace

import pytest

from core import web_planner as wp
from core import web_revise as wr
from services import web_domain_packs as packs
from services.web_evidence import EvidenceRegistry, PrepassEvidence, unsupported_citation_claims


# ── recorded model outputs ─────────────────────────────────────────────────

PLAN_JSON = (
    '{"need_web": true, "reason": "policy that changes", "follow_up": false, '
    '"queries": ["ALMA Cycle 13 proprietary period", "ALMA Cycle 13 proposer\'s guide data release"], '
    '"freshness": "year", "domain_pack": "alma_policy", '
    '"entities": {"objects": [], "facility": "ALMA", "cycle": "13", "person": null}, "want_images": false}'
)
FENCED = "Here is the plan:\n```json\n" + PLAN_JSON + "\n```\nDone."
HARMONY = (
    "<|channel|>analysis<|message|>The user asks about ALMA policy.<|end|>"
    "<|start|>assistant<|channel|>final<|message|>" + PLAN_JSON
)
FOLLOW_UP = (
    '{"need_web": true, "reason": "recent event", "follow_up": true, '
    '"queries": ["Vera C. Rubin Observatory LSST survey official start date"], "freshness": "month", '
    '"domain_pack": "noirlab", "entities": {"objects": [], "facility": "Rubin Observatory", "cycle": null, "person": null}, '
    '"want_images": false}'
)
NO_WEB = '{"need_web": false, "reason": "textbook derivation", "queries": [], "freshness": "any", "domain_pack": "general"}'


def _llm(text, delay=0.0):
    def _call(prompt, model, max_tokens):
        if delay:
            time.sleep(delay)
        return text
    return _call


# ── parse ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw", [PLAN_JSON, FENCED, HARMONY])
def test_parse_plan_tolerates_fences_and_harmony_leakage(raw):
    plan = wp.parse_plan(raw)
    assert plan is not None and plan.need_web
    assert plan.queries == ["ALMA Cycle 13 proprietary period", "ALMA Cycle 13 proposer's guide data release"]
    assert plan.domain_pack == "alma_policy" and plan.freshness == "year"
    assert plan.entities["cycle"] == "13" and plan.entities["facility"] == "ALMA"
    assert plan.follow_up is False and plan.want_images is False


def test_parse_plan_normalises_bad_fields_and_caps_queries():
    raw = ('{"need_web": "yes", "queries": ["a", "a", "b", "c", "d"], "freshness": "soon", '
           '"domain_pack": "ALMA-Policy", "entities": {"cycle": "Cycle 14", "objects": "NGC 1068"}, "want_images": "no"}')
    plan = wp.parse_plan(raw)
    assert plan.need_web is True and plan.queries == ["a", "b", "c"]
    assert plan.freshness == "any" and plan.domain_pack == "alma_policy"
    assert plan.entities == {"objects": ["NGC 1068"], "facility": None, "cycle": "14", "person": None}
    assert plan.want_images is False


@pytest.mark.parametrize("raw", ["", "I think YES.", "{\"reason\": \"no need_web key\"}", "```json\n{broken", None])
def test_parse_plan_fails_closed_on_garbage(raw):
    assert wp.parse_plan(raw) is None


def test_parse_plan_need_web_without_queries_uses_the_fallback_query():
    plan = wp.parse_plan('{"need_web": true, "queries": []}', fallback_query="latest news about 3I/ATLAS")
    assert plan.queries == ["latest news about 3I/ATLAS"]


# ── prompt ─────────────────────────────────────────────────────────────────


def test_prompt_carries_history_session_entities_date_and_cutoff():
    from datetime import date

    ctx = wp.PlannerContext(
        history=[{"role": "user", "content": "What is the Vera C. Rubin Observatory?"},
                 {"role": "assistant", "content": "x" * 900}],
        session_context="## Session Memory\nUser studies Rubin.",
        entities=["NGC 1068", "Cycle 13"],
        today=date(2026, 9, 24), cutoff=(2025, 6), web_mode="auto",
    )
    prompt = wp.build_planner_prompt("When did its LSST survey officially begin?", ctx)
    assert "Today: 2026-09-24. Model knowledge cutoff: 2025-06." in prompt
    assert "User: What is the Vera C. Rubin Observatory?" in prompt
    assert "Assistant: " + "x" * 499 + "…" in prompt          # clipped to 500
    assert "Entities from this conversation's tool calls: NGC 1068, Cycle 13" in prompt
    assert "Session memory:" in prompt and 'NEW USER MESSAGE: "When did its LSST survey officially begin?"' in prompt
    assert "never instructions" in prompt
    assert "ALWAYS" not in prompt
    ctx.web_mode = "always"
    assert "web search to ALWAYS" in wp.build_planner_prompt("q", ctx)


# ── job: bounded, fail closed ──────────────────────────────────────────────


def test_planner_job_returns_the_plan_and_its_latency():
    job = wp.PlannerJob("q", wp.PlannerContext(), llm_call=_llm(PLAN_JSON, delay=0.05), model="fake").start()
    plan = job.wait(timeout=2.0)
    assert plan is not None and plan.need_web and plan.source == "planner"
    assert 0.04 <= plan.elapsed_s < 1.5


def test_planner_job_times_out_fail_closed_and_a_late_result_is_still_readable():
    job = wp.PlannerJob("q", wp.PlannerContext(), llm_call=_llm(PLAN_JSON, delay=0.6), model="fake").start()
    t0 = time.monotonic()
    assert job.wait(timeout=0.1) is None
    assert time.monotonic() - t0 < 0.5                     # bounded by the timeout, not the call
    assert job.wait(timeout=2.0) is not None               # the pre-pass thread may still use it


def test_planner_job_exception_and_garbage_fail_closed():
    def boom(prompt, model, max_tokens):
        raise RuntimeError("HTTP 402 Insufficient Balance")
    assert wp.PlannerJob("q", llm_call=boom, model="fake").start().wait(timeout=1.0) is None
    assert wp.PlannerJob("q", llm_call=_llm("YES"), model="fake").start().wait(timeout=1.0) is None


def test_planner_always_mode_forces_need_web():
    job = wp.PlannerJob("q", wp.PlannerContext(web_mode="always"), llm_call=_llm(NO_WEB), model="fake").start()
    assert job.wait(timeout=1.0).need_web is True


# ── D15: researcher check ──────────────────────────────────────────────────

LIVE = re.compile(r"\b(?:observations?|data|archives?|band\s*\d)\b")


@pytest.mark.parametrize("query", [
    "Who is Paola Caselli?", "Who is Crystal Brogan?", "What does Alyssa Goodman research?",
    "Where does Crystal Brogan work?", "What is Paola Caselli's h-index?", "Look up Dr. Jane van der Berg",
    "Profile of Ewine van Dishoeck", "Which institution is Anthony Remijan at?",
])
def test_d15_person_questions_take_the_researcher_path(query):
    assert wp.looks_like_researcher_query(query, LIVE)


@pytest.mark.parametrize("query", [
    "Tell me about the Square Kilometre Array.", "Tell me about black holes", "Tell me about M87",
    "Who is the director of ALMA?", "Tell me about ALMA Band 7 observations of M87",
    "Tell me about the Vera C. Rubin Observatory", "Look up NGC 1068", "Tell me about the ALMA correlator",
])
def test_d15_non_person_questions_do_not(query):
    assert not wp.looks_like_researcher_query(query, LIVE)


def test_researcher_name_strips_titles_and_trigger_phrases():
    assert wp.researcher_name("Who is Paola Caselli?") == "Paola Caselli"
    assert wp.researcher_name("Look up Dr. Jane van der Berg") == "Jane van der Berg"


# ── entities + follow-up memory ────────────────────────────────────────────


def test_conversation_entities_come_from_tool_arguments_only():
    trace = [
        {"tool": "search_by_target", "args": {"target": "NGC 1068", "band": 7}},
        {"tool": "alma_project_census", "args": json.dumps({"cycle": "13"})},
        {"tool": "web_search", "args": {"query": "a very long free text query about many things indeed"}},
        {"tool": "x", "args": {"target": "y" * 80}},
        "garbage",
    ]
    assert wp.conversation_entities(trace) == ["NGC 1068", "Cycle 13"]


def test_conversation_web_memory_is_bounded_lru_and_keeps_evidence_over_a_no_web_turn():
    mem = wp.ConversationWebMemory(max_conversations=2, max_items=3)
    ev = [{"url": f"https://a.org/{i}", "title": f"T{i}", "snippet": "s"} for i in range(5)]
    mem.remember("c1", evidence=ev, entities=["NGC 1068"])
    assert len(mem.recall("c1")["evidence"]) == 3
    mem.remember("c1", evidence=(), entities=["Cycle 13"])          # a turn without web: only the previous turn's pages carry (P2-09)
    state = mem.recall("c1")
    assert state["evidence"] == [] and state["entities"] == ["Cycle 13", "NGC 1068"]
    mem.remember("c1", evidence=ev, entities=[])
    assert len(mem.recall("c1")["evidence"]) == 3
    mem.remember("c2", evidence=ev[:1]); mem.remember("c3", evidence=ev[:1])
    assert len(mem) == 2 and mem.recall("c1") == {"evidence": [], "entities": []}   # evicted (c1 was recalled last before c2/c3)


def test_conversation_web_memory_is_thread_safe():
    mem = wp.ConversationWebMemory(max_conversations=50)
    errors = []

    def work(i):
        try:
            for j in range(50):
                mem.remember(f"c{i % 7}", evidence=[{"url": f"https://a.org/{i}/{j}"}], entities=[f"e{j}"])
                mem.recall(f"c{(i + 1) % 7}")
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert not errors and len(mem) == 7


def test_registry_add_carried_gives_new_ids_and_skips_deep_read():
    reg = EvidenceRegistry(query="q")
    carried = [{"url": "https://rubinobservatory.org/news/start", "title": "Survey start", "snippet": "began 29 June 2026", "carried": True}]
    added = reg.add_carried(carried)
    assert [ev.id for ev in added] == ["W1"] and added[0].origin == "carried" and added[0].deep_read
    reg.add_from_payload({"success": True, "results": [{"url": "https://b.org/x", "title": "B", "snippet": "b"}]})
    assert reg.ids() == ["W1", "W2"]


def test_looks_like_follow_up():
    assert wp.looks_like_follow_up("When did its LSST survey officially begin?")
    assert wp.looks_like_follow_up("What about Cycle 13?")
    assert not wp.looks_like_follow_up("Derive the Jeans mass for an isothermal, uniform-density gas cloud.")


# ── domain packs + ordering ────────────────────────────────────────────────


def test_pack_list_is_data_and_covers_the_planned_packs():
    assert set(packs.known_packs()) == {"alma_policy", "nrao", "stsci", "noirlab", "esa", "transients", "researcher", "general"}
    assert "almascience.org" in packs.pack_domains("alma_policy") and "science.nrao.edu" in packs.pack_domains("nrao")
    assert packs.pack_domains("researcher") == [] and packs.pack_domains("general") == []
    assert packs.pack_domains("bogus") == []
    assert packs.pack_topic("transients") == "news"
    assert "| alma_policy |" in packs.review_table()


def test_freshness_maps_to_tavily_and_brave():
    assert packs.tavily_time_range("month") == "month" and packs.tavily_time_range("any") is None
    assert packs.brave_freshness("week") == "pw" and packs.brave_freshness("any") is None
    assert packs.normalize_freshness("recent") == "week" and packs.normalize_freshness("bogus") == "any"


@pytest.mark.parametrize("query,pack", [
    ("What is the proprietary period for ALMA Cycle 13 data?", "alma_policy"),
    ("Which configuration is the VLA in right now?", "nrao"),
    ("When is the JWST Cycle 5 deadline?", "stsci"),
    ("When did the Rubin LSST survey begin?", "noirlab"),
    ("What is known about GRB 250702B?", "transients"),
    ("What is the Jeans mass?", "general"),
])
def test_infer_pack_deterministic_fallback(query, pack):
    assert packs.infer_pack(query) == pack
    assert packs.infer_pack(query, person=True) == "researcher"


def _ev(url, title="", excerpt="", relevance=0.5, tier=None):
    return SimpleNamespace(url=url, domain=packs._host(url), title=title, excerpt=excerpt, relevance=relevance,
                           quality={"tier": tier} if tier else {}, rank=0, origin="prepass")


def test_official_pack_domain_outranks_stack_exchange_and_general_pages():
    items = [
        _ev("https://astronomy.stackexchange.com/q/1", "VLA config?", "BnA ...", relevance=1.0),
        _ev("https://en.wikipedia.org/wiki/VLA", "VLA", "...", relevance=0.5, tier="reference"),
        _ev("https://science.nrao.edu/facilities/vla/proposing/configpropdeadlines", "Configuration plans", "D from 2026 Jul 10", relevance=0.33, tier="primary"),
    ]
    ordered = packs.order_evidence(items, pack="nrao", query="Which configuration is the VLA in right now?")
    assert [packs._host(e.url) for e in ordered] == ["science.nrao.edu", "en.wikipedia.org", "astronomy.stackexchange.com"]


def test_current_cycle_page_outranks_an_older_cycle_page_of_the_same_site():
    items = [
        _ev("https://almascience.org/documents-and-tools/cycle7/alma-proposers-guide", "ALMA Cycle 7 Proposer's Guide", "proprietary period 12 months", relevance=1.0),
        _ev("https://almascience.org/documents-and-tools/cycle13/alma-proposers-guide", "ALMA Cycle 13 Proposer's Guide", "proprietary period 12 months", relevance=0.5),
        _ev("https://almascience.org/alma-data/archive/proprietary-period", "Proprietary period", "no cycle named", relevance=0.33),
    ]
    ordered = packs.order_evidence(items, pack="alma_policy", query="What is the proprietary period for ALMA Cycle 13 data?")
    assert "cycle13" in ordered[0].url and "cycle7" in ordered[-1].url
    # the entity hint alone also pins the cycle
    ordered2 = packs.order_evidence(items, pack="alma_policy", query="proprietary period?", entities={"cycle": "13"})
    assert "cycle7" in ordered2[-1].url
    # no cycle asked: relevance order is kept
    ordered3 = packs.order_evidence(items, pack="alma_policy", query="ALMA proprietary period")
    assert "cycle7" in ordered3[0].url


def test_order_evidence_is_deterministic_and_stable_for_ties():
    items = [_ev(f"https://x{i}.org/p", relevance=0.5) for i in range(6)]
    assert [e.url for e in packs.order_evidence(items, pack="general", query="q")] == [e.url for e in items]
    assert packs.order_evidence([], pack="general") == []


def test_registry_reorder_renumbers_only_before_anything_is_shown():
    reg = EvidenceRegistry(query="q")
    reg.add_from_payload({"success": True, "results": [
        {"url": "https://stackexchange.com/q", "title": "SE", "snippet": "a"},
        {"url": "https://science.nrao.edu/x", "title": "NRAO", "snippet": "b"},
    ]})
    ordered = packs.order_evidence(reg.items(), pack="nrao", query="q")
    assert reg.reorder(ordered) is True
    assert [(ev.id, ev.domain) for ev in reg.items()] == [("W1", "science.nrao.edu"), ("W2", "stackexchange.com")]
    reg.render_prompt_block()
    assert reg.reorder(list(reversed(reg.items()))) is False       # frozen once shown
    assert reg.items()[0].id == "W1" and reg.items()[0].domain == "science.nrao.edu"


# ── support checker normalisation fixes (part C audit) ────────────────────


def _reg(src):
    reg = EvidenceRegistry()
    reg.add_from_payload({"success": True, "results": [{"title": "T", "url": "https://a.org/x", "snippet": src}]})
    reg.render_prompt_block()
    return reg


@pytest.mark.parametrize("claim,src", [
    ("The D/H ratio is 9.8 ± 0.6 × 10⁻³ [W1].", "the ratio is (9.8 ± 0.6) × 10^-3"),
    ("The D/H ratio is 9.8 ± 0.6 × 10⁻³ [W1].", "ratio of 0.0098 with error 0.6"),
    ("The proprietary period is 12 months [W1].", "proprietary period of twelve months"),
    ("Moves to C on 19 October 2026 [W1].", "D configuration until Oct 19; the 2026 schedule"),
    ("Construction ends 2028-2029 [W1].", "completion expected 2028-29"),
    ("It has 30 times more heavy water [W1].", "contains 30x more heavy water"),
    ("Data Release 2 has 1.5 million rows [W1].", "DR2 contains 1,500,000 rows"),
    ("Value 10.0 K [W1].", "a value of 10 K"),
    ("It weighs 2e6 solar masses [W1].", "The mass is 2 x 10^6 solar masses."),
    ("Cycle 13 starts in October 2026 [W1].", "Observing starts 2026-10-01 for Cycle 13."),
])
def test_checker_normalisation_no_false_positive(claim, src):
    assert unsupported_citation_claims(claim, _reg(src)) == []


@pytest.mark.parametrize("claim,src,missing", [
    ("The D/H ratio is 9.8 ± 0.6 × 10⁻³ [W1].", "30 times more heavy water", "9.8e-3"),
    ("Moves to C on 19 October 2026 [W1].", "D configuration until Oct 19", "19 October 2026"),
    ("Started in July 2025 [W1].", "observations in July of that year", "July 2025"),
    ("Cycle 14 rule [W1].", "Cycle 13 guide; 14 pages", "cycle 14"),
])
def test_checker_still_flags_real_unsupported_values(claim, src, missing):
    findings = unsupported_citation_claims(claim, _reg(src))
    assert findings and missing in findings[0]["missing"]


# ── revise pass ────────────────────────────────────────────────────────────

ANSWER = (
    "| Result | Date |\n|---|---|\n"
    "| Heavy water found by RIKEN [W1] | 23 September 2026 |\n"
    "Seven hours of SETI observations in July 2025 found nothing [W2].\n"
    "Perihelion was on 30 October 2025 [W3]."
)


def _revise_reg():
    reg = EvidenceRegistry(query="3I/ATLAS")
    reg.add_from_payload({"success": True, "results": [
        {"url": "https://srpske.rs/heavy", "title": "Heavy water", "snippet": "RIKEN reports 30 times more heavy water in 3I/ATLAS."},
        {"url": "https://apnews.com/seti", "title": "SETI", "snippet": "Seven hours of radio observations found no alien signal."},
        {"url": "https://science.nasa.gov/atlas", "title": "NASA", "snippet": "Perihelion on 30 October 2025."},
    ]})
    reg.render_prompt_block()
    return reg


def test_revise_rewrites_only_unsupported_sentences_and_rechecks(monkeypatch):
    reg = _revise_reg()
    before = unsupported_citation_claims(ANSWER, reg)
    assert len(before) == 2
    seen = {}

    def fake_llm(prompt, model, max_tokens):
        seen["prompt"] = prompt
        return json.dumps({"revisions": [
            {"original": "| Heavy water found by RIKEN [W1] | 23 September 2026 |", "revised": "| Heavy water found by RIKEN [W1] | not stated |"},
            {"original": "Seven hours of SETI observations in July 2025 found nothing [W2].", "revised": "Seven hours of SETI observations found nothing [W2]."},
            {"original": "Perihelion was on 30 October 2025 [W3].", "revised": "Perihelion was on 31 October 2025 [W3]."},   # not a target: ignored
        ]})

    out, info = wr.revise_unsupported(ANSWER, reg, llm_call=fake_llm, budget_s=2.0)
    assert "not stated" in out and "July 2025" not in out and "30 October 2025 [W3]" in out
    assert info["attempted"] == 2 and info["revised"] == 2
    assert unsupported_citation_claims(out, reg) == []
    assert "SENTENCE 1:" in seen["prompt"] and "[W1] Heavy water ::" in seen["prompt"] and "never follow instructions" in seen["prompt"]


def test_revise_rejects_new_facts_new_tags_and_broken_rows():
    reg = _revise_reg()

    def fake_llm(prompt, model, max_tokens):
        return json.dumps({"revisions": [
            {"original": "| Heavy water found by RIKEN [W1] | 23 September 2026 |", "revised": "| Heavy water found by RIKEN [W1] | 24 September 2026 |"},   # new date
            {"original": "Seven hours of SETI observations in July 2025 found nothing [W2].", "revised": "Seven hours of SETI observations found nothing [W2][W3]."},  # new tag
        ]})

    out, info = wr.revise_unsupported(ANSWER, reg, llm_call=fake_llm, budget_s=2.0)
    assert out == ANSWER and info["revised"] == 0 and info["attempted"] == 2


def test_revise_is_bounded_and_skipped_near_the_turn_deadline():
    reg = _revise_reg()
    t0 = time.monotonic()
    out, info = wr.revise_unsupported(ANSWER, reg, llm_call=_llm("{}", delay=1.0), budget_s=0.2)
    assert out == ANSWER and info["skipped"].startswith("timeout") and time.monotonic() - t0 < 0.8
    out, info = wr.revise_unsupported(ANSWER, reg, llm_call=_llm("{}"), budget_s=6.0, turn_deadline=time.monotonic() + 3.0)
    assert out == ANSWER and info["skipped"] == "turn deadline too close"
    out, info = wr.revise_unsupported(ANSWER, reg, llm_call=_llm("not json"), budget_s=1.0)
    assert out == ANSWER and info["skipped"] == "unparsable revise output"


def test_revise_negative_control_without_the_budget_join_the_call_would_block():
    """Negative control for the timing bound: the same slow call, joined
    without a timeout, takes the full call time."""
    t0 = time.monotonic()
    th = threading.Thread(target=_llm("{}", delay=0.6), args=("p", "m", 1), daemon=True)
    th.start(); th.join()
    assert time.monotonic() - t0 >= 0.55


def test_revise_flag_off_is_honoured(monkeypatch):
    monkeypatch.setenv("QUASAR_WEB_REVISE", "0")
    assert wr.revise_enabled() is False
    monkeypatch.delenv("QUASAR_WEB_REVISE")
    assert wr.revise_enabled() is True


# ── runner wiring ──────────────────────────────────────────────────────────

import core.runner as runner
from core import web_policy


class _Stop(Exception):
    pass


def _drive(monkeypatch, query, *, web_search=True, mode=None, plan_text=None, plan_delay=0.0, live=False,
           has_key=True, carried=None, conv="c1", memory_turns=None, planner_env="1"):
    """Run the runner up to the RAG step. Returns a dict with the pre-pass
    calls, the events emitted and the status labels."""
    monkeypatch.setenv("QUASAR_WEB_PLANNER", planner_env)
    calls, plan_calls, events, statuses = [], [], [], []

    class FakeAgent:
        _LIVE_DATA_KEYWORDS_RE = re.compile(r"\b(?:data|archive|observations?)\b")

        def __init__(self):
            self._tls = threading.local()
            self.config = SimpleNamespace(model="m")
            self.memory = SimpleNamespace(get_last_n_turns=lambda n: list(memory_turns or []))
            self.session_memory = SimpleNamespace(get_context=lambda: "")
            self._accumulated_tool_trace = []

        def _begin_response_run(self, *a, **k): pass
        def _cleanup_conv_states(self): pass
        def _prune_session_if_needed(self, *a, **k): pass
        def _is_live_data_query(self, q): return live
        def _detect_beyond_cutoff(self, q): return None
        def _has_web_provider_key(self): return has_key
        def _detect_web_search_needed_via_llm(self, q):
            calls.append({"legacy_classifier": q}); return False
        def _tavily_web_search(self, query, max_results=10, search_depth="basic", want_images=None, **kw):
            calls.append({"query": query, "want_images": want_images, **kw})
            return {"success": True, "provider": "Tavily", "query": query, "results": [
                {"url": "https://science.nrao.edu/x", "title": "Official", "snippet": "D configuration until Oct 19 2026"}]}
        def _web_search_plan(self, queries, **kw):
            plan_calls.append({"queries": list(queries), **{k: v for k, v in kw.items() if k != "on_search"}})
            on_search = kw.get("on_search")
            for q in queries:
                if on_search:
                    on_search("start", q, bool(kw.get("include_domains")), None)
                    on_search("done", q, bool(kw.get("include_domains")), {"success": True})
            return {"success": True, "provider": "Tavily", "query": queries[0], "queries": list(queries), "results": [
                {"url": "https://astronomy.stackexchange.com/q/1", "title": "SE", "snippet": "BnA"},
                {"url": "https://science.nrao.edu/x", "title": "Official", "snippet": "D configuration until Oct 19 2026"},
            ], "searches": []}

    agent = FakeAgent()
    if carried:
        # the runner keys the carry store by user AND conversation (P2-08); the
        # default user_id of the runner is "user"
        wp.conversation_web_memory(agent).remember(f"user|{conv}", evidence=carried)
    if plan_text is not None:
        monkeypatch.setattr(wp, "_default_llm_call", _llm(plan_text, delay=plan_delay))
    else:
        def boom(prompt, model, max_tokens):
            raise RuntimeError("HTTP 402")
        monkeypatch.setattr(wp, "_default_llm_call", boom)
    monkeypatch.setattr(runner, "is_explicit_query", lambda q: False)
    monkeypatch.setattr(runner, "_maybe_failover_model", lambda a, m, *x, **k: m)
    import services.rag_service as rag

    def _boom(*a, **k):
        raise _Stop()
    monkeypatch.setattr(rag, "is_domain_relevant", _boom, raising=False)

    def on_status(step, state):
        if step.startswith("__event__"):
            events.append(json.loads(step[len("__event__"):]))
        else:
            statuses.append((step, state))
    try:
        runner._stream_response_api_impl(agent, query, web_search=web_search, conversation_id=conv,
                                         on_status=on_status, web_search_mode=mode)
    except _Stop:
        pass
    for t in threading.enumerate():
        if t is not threading.current_thread() and t.daemon and t.name.startswith("Thread"):
            t.join(timeout=3)
    return {"calls": calls, "plan_calls": plan_calls, "events": events, "statuses": statuses, "agent": agent,
            "registry": getattr(agent._tls, "web_registry", None)}


def _decisions(r):
    return [e for e in r["events"] if e.get("type") == "web_decision"]


def test_runner_planner_decides_the_undecided_case_and_emits_one_decision(monkeypatch):
    plan = ('{"need_web": true, "reason": "current schedule", "queries": ["VLA current configuration", "VLA configuration schedule 2026"], '
            '"freshness": "month", "domain_pack": "nrao", "entities": {}, "want_images": false}')
    r = _drive(monkeypatch, "Which configuration is the VLA in right now?", plan_text=plan)
    assert r["plan_calls"] and r["plan_calls"][0]["queries"] == ["VLA current configuration", "VLA configuration schedule 2026"]
    assert r["plan_calls"][0]["include_domains"] == packs.pack_domains("nrao") and r["plan_calls"][0]["freshness"] == "month"
    assert not any("legacy_classifier" in c for c in r["calls"])
    d = _decisions(r)
    assert len(d) == 1 and d[0]["need_web"] and d[0]["source"] == "planner" and len(d[0]["queries"]) == 2
    assert d[0]["mode"] == "auto" and d[0]["domain_pack"] == "nrao"
    labels = [s for s, _ in r["statuses"]]
    assert 'Searching the web: "VLA current configuration" (official sites)' in labels
    assert any(l.startswith("Reading ") and l.endswith("pages") or l.endswith("page") for l in labels)
    # official page ordered first -> W1
    assert r["registry"].items()[0].domain == "science.nrao.edu"


def test_runner_planner_says_no_web_emits_skipped_decision(monkeypatch):
    r = _drive(monkeypatch, "Derive the Jeans mass.", plan_text=NO_WEB)
    assert not r["plan_calls"] and not r["calls"]
    d = _decisions(r)
    assert len(d) == 1 and d[0]["need_web"] is False and d[0]["reason"] == "textbook derivation"


def test_runner_planner_failure_fails_closed_without_the_legacy_classifier(monkeypatch):
    r = _drive(monkeypatch, "Some undecided question about stars")
    assert not r["plan_calls"] and not r["calls"]
    d = _decisions(r)
    assert len(d) == 1 and d[0]["need_web"] is False and d[0]["reason"] == "web planner unavailable"


def test_runner_planner_timeout_is_bounded(monkeypatch):
    monkeypatch.setenv("QUASAR_WEB_PLANNER_TIMEOUT", "0.2")
    t0 = time.monotonic()
    # the planner call itself takes 4 s; the turn must not wait for it (a wide
    # margin so a loaded machine, e.g. the full suite next to a UI benchmark,
    # does not make this flaky: it failed once at 1.4 s under that load)
    r = _drive(monkeypatch, "Some undecided question about stars", plan_text=NO_WEB, plan_delay=4.0)
    assert time.monotonic() - t0 < 3.0 and not r["calls"] and not r["plan_calls"]


def test_runner_planner_timeout_negative_control(monkeypatch):
    """Without the bound the same turn waits for the whole planner call."""
    monkeypatch.setenv("QUASAR_WEB_PLANNER_TIMEOUT", "5")
    t0 = time.monotonic()
    _drive(monkeypatch, "Some undecided question about stars", plan_text=NO_WEB, plan_delay=0.7)
    assert time.monotonic() - t0 >= 0.65


def test_runner_deterministic_policy_path_starts_search_and_uses_plan_hints(monkeypatch):
    r = _drive(monkeypatch, "What is the proprietary period for ALMA Cycle 13 data?", plan_text=PLAN_JSON, live=True)
    assert r["plan_calls"] and r["plan_calls"][0]["include_domains"] == packs.pack_domains("alma_policy")
    d = _decisions(r)
    assert len(d) == 1 and d[0]["need_web"] and d[0]["reason"] == "policy"


def test_runner_mode_off_and_always(monkeypatch):
    r = _drive(monkeypatch, "Which configuration is the VLA in right now?", plan_text=PLAN_JSON, mode="off", web_search=True)
    assert not r["plan_calls"] and not r["calls"] and _decisions(r)[0]["reason"] == "web search is off" and not web_policy.web_allowed()
    r = _drive(monkeypatch, "Derive the Jeans mass.", plan_text=NO_WEB, mode="always")
    assert r["plan_calls"] and _decisions(r)[0]["need_web"] and _decisions(r)[0]["mode"] == "always"
    # the boolean switch still works (False = off)
    r = _drive(monkeypatch, "Which configuration is the VLA in right now?", plan_text=PLAN_JSON, web_search=False)
    assert not r["plan_calls"] and not r["calls"]


def test_runner_planner_off_restores_phase1_path(monkeypatch):
    r = _drive(monkeypatch, "Which configuration is the VLA in right now?", plan_text=PLAN_JSON, planner_env="0")
    assert any("legacy_classifier" in c for c in r["calls"]) and not r["plan_calls"] and not _decisions(r)
    r = _drive(monkeypatch, "What is the proprietary period for ALMA Cycle 13 data?", plan_text=PLAN_JSON, live=True, planner_env="0")
    assert r["calls"] and r["calls"][0]["query"] == "What is the proprietary period for ALMA Cycle 13 data?" and not r["plan_calls"]


def test_runner_d15_ska_is_not_a_researcher_search_but_a_person_is(monkeypatch):
    r = _drive(monkeypatch, "Tell me about the Square Kilometre Array.", plan_text=NO_WEB)
    assert not r["plan_calls"] and not r["calls"]
    r = _drive(monkeypatch, "Who is Paola Caselli?", plan_text=NO_WEB)
    assert r["plan_calls"] and r["plan_calls"][0]["include_domains"] == []   # researcher pack: unrestricted
    assert any("Paola Caselli email contact" in c.get("query", "") for c in r["calls"])
    assert _decisions(r)[0]["reason"] == "researcher_supplement"


def test_runner_follow_up_carries_the_previous_turns_pages(monkeypatch):
    carried = [{"url": "https://rubinobservatory.org/news/first-look", "title": "Rubin first look", "snippet": "released 23 June 2025", "carried": True}]
    r = _drive(monkeypatch, "When did its LSST survey officially begin?", plan_text=FOLLOW_UP, carried=carried, conv="c9")
    reg = r["registry"]
    assert reg is not None and any(ev.origin == "carried" for ev in reg.items())
    assert _decisions(r)[0]["follow_up"] is True
    monkeypatch.setenv("QUASAR_WEB_CARRY_EVIDENCE", "0")
    r = _drive(monkeypatch, "When did its LSST survey officially begin?", plan_text=FOLLOW_UP, carried=carried, conv="c9")
    assert not any(ev.origin == "carried" for ev in r["registry"].items())


def test_runner_history_and_entities_reach_the_planner_prompt(monkeypatch):
    seen = {}
    real_build = wp.build_planner_prompt

    def spy(query, ctx=None):
        seen["ctx"] = ctx
        return real_build(query, ctx)
    monkeypatch.setattr(wp, "build_planner_prompt", spy)
    turns = [{"role": "user", "content": "What is the Vera C. Rubin Observatory?"}, {"role": "assistant", "content": "A survey telescope."}]
    carried = [{"url": "https://rubinobservatory.org/x", "title": "Rubin", "snippet": "s"}]
    r = _drive(monkeypatch, "When did its LSST survey officially begin?", plan_text=NO_WEB, memory_turns=turns, carried=carried, conv="c7")
    wp.conversation_web_memory(r["agent"]).remember("c7", entities=["Rubin Observatory"])
    ctx = seen["ctx"]
    assert ctx.history == turns and ctx.web_mode == "auto" and ctx.cutoff == (2025, 6)
    # entities remembered from an earlier turn's tool calls reach the next planner call
    r2 = _drive(monkeypatch, "And its first light?", plan_text=NO_WEB, memory_turns=turns, conv="c8")
    wp.conversation_web_memory(r2["agent"]).remember("c8", entities=["NGC 1068"])
    assert wp.conversation_web_memory(r2["agent"]).recall("c8")["entities"] == ["NGC 1068"]


def test_e_policy_question_is_not_an_alma_census_query():
    from core.router import policy_web_override
    q = "What is the proprietary period for ALMA Cycle 13 data?"
    assert policy_web_override(q)
    src = open("core/runner.py", encoding="utf-8").read()
    i = src.index("_is_alma_science_archive_query = (")
    assert "and not _policy_web_override(_user_query)" in src[i: src.index("if _is_alma_science_archive_query:", i)]


@pytest.mark.parametrize("query", [
    "What is the default exclusive access period for HST GO data in Cycle 34?",
    "When does ALMA Cycle 13 science observing start, and when does the cycle end?",
    "What is the Large Program threshold in the ALMA Cycle 13 Call for Proposals?",
    "What are the HST data rights for Large programs?",
    "Which VLA configuration will be in use in March 2027? Show the configuration schedule.",
])
def test_policy_phrases_added_in_phase2_take_the_policy_path(query):
    from core.router import policy_web_override
    assert policy_web_override(query)


@pytest.mark.parametrize("query", [
    "Find ALMA observations of NGC 1068 in Band 7.",
    "Show me the Gaia data for M67.",
    "What is the Jeans mass of a cloud?",
])
def test_policy_phrases_do_not_fire_on_data_requests(query):
    from core.router import policy_web_override
    assert not policy_web_override(query)


def test_e_census_fallback_requires_an_alma_count_or_list_intent():
    src = open("core/runner.py", encoding="utf-8").read()
    i = src.index("_is_alma_science_archive_query = (")
    block = src[i: src.index("if _is_alma_science_archive_query:", i)]
    assert 'and re.search(r"\\balma\\b", _query_lower)' in block
    assert "how\\s+many|number\\s+of|count|list|find|search|show|which|what" in block
    assert "and not _policy_web_override(_user_query)" in block


def test_normalize_web_mode():
    assert runner._normalize_web_mode(None, True) == "auto" and runner._normalize_web_mode(None, False) == "off"
    assert runner._normalize_web_mode("ALWAYS", False) == "always" and runner._normalize_web_mode("bogus", True) == "auto"


# ── evidence wait honours a planner credit made after the wait began ──────


def test_prepass_wait_extends_when_the_planner_time_is_credited_during_the_wait():
    reg = EvidenceRegistry(query="q")
    pre = PrepassEvidence(reg)

    def worker():
        time.sleep(0.3)                # the planner call
        pre.credit(0.3)
        time.sleep(0.35)               # the search
        pre.record_search({"success": True, "results": [{"url": "https://a.org/x", "title": "A", "snippet": "s"}]})
        pre.finish()

    threading.Thread(target=worker, daemon=True).start()
    t0 = time.monotonic()
    ready = pre.wait(0.5)              # 0.5 s budget from the CREDITED start (0.3 s in) covers the search at 0.65 s
    assert ready is True and 0.6 <= time.monotonic() - t0 < 1.2


def test_prepass_wait_negative_control_without_the_credit_the_search_is_missed():
    reg = EvidenceRegistry(query="q")
    pre = PrepassEvidence(reg)

    def worker():
        time.sleep(0.65)
        pre.record_search({"success": True, "results": [{"url": "https://a.org/x", "title": "A", "snippet": "s"}]})
        pre.finish()

    threading.Thread(target=worker, daemon=True).start()
    t0 = time.monotonic()
    assert pre.wait(0.5) is False and time.monotonic() - t0 < 0.65


# ── live findings 2026-09-24 (bench/phase2 pass 1) ────────────────────────


def test_currency_penalty_weighs_url_and_title_over_a_sidebar_mention():
    asked = packs.asked_versions("What is the proprietary period for ALMA Cycle 13 data?")
    # a Cycle 12 documents page whose excerpt mentions the Cycle 13 call in a sidebar
    assert packs.currency_penalty("Cycle 12 documents. Sidebar: Cycle 13 Call for Proposals is now open", asked,
                                  strong_text="https://almascience.eso.org/documents-and-tools/cycle-12 Documents and tools Cycle 12") == 1
    assert packs.currency_penalty("proprietary period 12 months", asked,
                                  strong_text="https://almascience.eso.org/documents-and-tools/cycle13/alma-proposers-guide ALMA Cycle 13 Proposer's Guide") == 0
    # no version in URL / title: the excerpt decides, as before
    assert packs.currency_penalty("This Cycle 7 guide ...", asked, strong_text="https://almascience.org/x Guide") == 1
    assert packs.currency_penalty("no version here", asked, strong_text="https://almascience.org/x Guide") == 0
    items = [
        _ev("https://almascience.eso.org/documents-and-tools/cycle-12", "Documents and tools", "Cycle 13 Call for Proposals is now open", relevance=1.0),
        _ev("https://almascience.eso.org/documents-and-tools/cycle13/alma-proposers-guide", "ALMA Cycle 13 Proposer's Guide", "proprietary period", relevance=0.1),
    ]
    assert "cycle13" in packs.order_evidence(items, pack="alma_policy", query="ALMA Cycle 13 proprietary period")[0].url


def test_search_plan_applies_freshness_to_the_unrestricted_search_only(monkeypatch):
    from services.web_search_service import WebSearchService

    svc = WebSearchService.__new__(WebSearchService)
    seen = []

    def fake_route(query, **kw):
        seen.append((query, kw.get("include_domains"), kw.get("freshness"), kw.get("topic")))
        return {"success": True, "provider": "Tavily", "results": [{"url": f"https://x.org/{len(seen)}", "title": "t", "snippet": "s"}]}
    svc.route_and_search = fake_route
    out = svc.search_plan(["JWST Cycle 5 GO deadline"], include_domains=["stsci.edu"], freshness="month", topic="news", budget_s=2)
    assert out["success"]
    by_restricted = {bool(d): (f, t) for q, d, f, t in seen}
    assert by_restricted[True] == (None, None) and by_restricted[False] == ("month", "news")
