"""Runner integration of grounded web evidence (PLAN 1.3, 1.5, 1.8): the bounded
join + injection, the legacy fallback, [W#] cleanup, the cited-first listing
and the verifier hook in _finalize_answer_text."""
from __future__ import annotations

import inspect
import threading
import time
from types import SimpleNamespace as NS

import core.runner as runner
from core.agent import QuasarAgent
from services.web_evidence import EvidenceRegistry, PrepassEvidence


def _payload(*pages):
    return {
        "success": True,
        "provider": "Tavily",
        "query": "q",
        "results": [
            {"title": t, "url": u, "snippet": s, "published_date": "2025-08-01"} for t, u, s in pages
        ],
    }


JWST = ("JWST Cycle 5 Call for Proposals", "https://www.stsci.edu/jwst/cycle-5",
        "The Cycle 5 General Observer proposal deadline is October 15, 2025, 8:00 pm EDT.")
NASA = ("Cycle 5 deadline extension", "https://science.nasa.gov/jwst-cycle-5-deadline-extension",
        "An extension was offered to proposers affected by the government shutdown.")


# ── 4d: bounded join + injection ────────────────────────────────────────


def test_ready_evidence_is_injected_before_the_user_message():
    reg = EvidenceRegistry(query="When is the JWST Cycle 5 deadline?")
    pre = PrepassEvidence(reg)
    pre.record_search(_payload(JWST, NASA))
    pre.finish()
    full = "DOCS...\n\nUser: When is the JWST Cycle 5 deadline?\n\n[MANDATORY INSTRUCTION: x]"
    out, block = runner._inject_web_evidence(full, "When is the JWST Cycle 5 deadline?", reg, pre,
                                             evidence_query="When is the JWST Cycle 5 deadline?", wait_s=5)
    assert block.startswith("WEB EVIDENCE (retrieved ")
    assert out.index("WEB EVIDENCE") < out.index("User: When is the JWST")
    assert "[W1] JWST Cycle 5 Call for Proposals | stsci.edu | 2025-08-01" in out
    assert "October 15, 2025" in out and out.rstrip().endswith("for this same question.]")
    assert "answering from your knowledge" not in out.lower()
    assert reg.injected and reg.injected_ids == ["W1", "W2"]


def test_not_ready_in_time_keeps_the_input_unchanged_for_the_legacy_path():
    reg = EvidenceRegistry()
    pre = PrepassEvidence(reg)               # search never comes back
    full = "User: q"
    t0 = time.monotonic()
    out, block = runner._inject_web_evidence(full, "q", reg, pre, evidence_query="q", wait_s=0.3)
    assert block == "" and out == full and not reg.injected
    assert time.monotonic() - t0 < 1.0     # bounded by the budget


def test_budget_counts_from_prepass_start_not_from_the_join():
    reg = EvidenceRegistry()
    pre = PrepassEvidence(reg)
    pre.started -= 30                        # the pre-pass started 30 s ago
    t0 = time.monotonic()
    out, block = runner._inject_web_evidence("User: q", "q", reg, pre, evidence_query="q", wait_s=8)
    assert block == "" and time.monotonic() - t0 < 0.5


def test_search_back_but_deep_read_slow_injects_the_snippets():
    reg = EvidenceRegistry()
    pre = PrepassEvidence(reg)

    def worker():
        pre.record_search(_payload(JWST))
        time.sleep(1.5)                      # a slow deep read
        pre.finish()

    threading.Thread(target=worker, daemon=True).start()
    t0 = time.monotonic()
    out, block = runner._inject_web_evidence("User: q", "q", reg, pre, evidence_query="q", wait_s=0.5)
    assert "October 15, 2025" in block and time.monotonic() - t0 < 1.2


def test_failed_search_falls_back_without_waiting_the_whole_budget():
    reg = EvidenceRegistry()
    pre = PrepassEvidence(reg)
    pre.record_search({"success": False, "error": "quota"})
    pre.finish()
    t0 = time.monotonic()
    assert runner._inject_web_evidence("User: q", "q", reg, pre, evidence_query="q", wait_s=8) == ("User: q", "")
    assert time.monotonic() - t0 < 0.5


# ── _finalize_answer_text: citations, markers, verifier ────────────────


def _agent():
    agent = QuasarAgent.__new__(QuasarAgent)
    agent._tls = threading.local()
    agent.tool_registry = NS(list_tools=lambda: [])
    agent._accumulated_tool_trace = []
    agent._accumulated_run_results = []
    agent.last_run_result = None
    return agent


def _registry():
    """A registry whose evidence block was rendered into the model input
    (only shown ids are citable, guard CX-08)."""
    reg = EvidenceRegistry()
    reg.add_from_payload(_payload(JWST, NASA))
    reg.render_prompt_block()
    return reg


def test_unknown_ids_are_stripped_and_known_ids_kept():
    out = runner._finalize_answer_text(
        _agent(), "The deadline was October 15, 2025 [W1, W7]. An extension was offered [W2][W9].",
        on_token=None, user_query="q", url_sources=[], all_tool_results=[], had_tool_calls=False,
        web_registry=_registry(),
    )
    assert "[W1]" in out and "[W2]" in out and "W7" not in out and "W9" not in out
    assert "Verification" not in out


def test_native_markers_are_normalized_and_doc_citations_kept():
    text = ("The deadline was October 15, 2025【W1】. Twelve months."
            "【Source: users-policies.pdf, Page 4, Date: March 2026】 Extra【the web search:0】.")
    out = runner._finalize_answer_text(_agent(), text, on_token=None, user_query="q", url_sources=[],
                                       all_tool_results=[], had_tool_calls=False, web_registry=_registry())
    assert "October 15, 2025 [W1]." in out
    assert "【Source: users-policies.pdf, Page 4, Date: March 2026】" in out
    assert "web search:0" not in out


def test_unsupported_cited_number_gets_a_verification_block_without_tool_calls(monkeypatch):
    # Phase 1 behaviour: findings are only listed. The Phase 2 revise pass
    # (QUASAR_WEB_REVISE) is off here so no model call can rewrite the sentence.
    monkeypatch.setenv("QUASAR_WEB_REVISE", "0")
    streamed = []
    out = runner._finalize_answer_text(
        _agent(), "The Cycle 5 deadline is June 7, 2025 [W1].", on_token=streamed.append, user_query="q",
        url_sources=[], all_tool_results=[], had_tool_calls=False, web_registry=_registry(),
    )
    assert "cited web source does not show" in out and "June" in out.split("Verification")[1]
    assert streamed and "Verification" in streamed[-1]


def test_no_registry_means_legacy_behaviour():
    text = "Answer [W3] stays as typed."
    out = runner._finalize_answer_text(_agent(), text, on_token=None, user_query="q", url_sources=[],
                                       all_tool_results=[], had_tool_calls=False, web_registry=None)
    assert out == text


# ── wiring (source contract) ────────────────────────────────────────────


def test_runner_wiring_contract():
    src = inspect.getsource(runner._stream_response_api_impl)
    # one registry per turn, behind the flag
    assert "_web_evidence.EvidenceRegistry(query=_user_query) if _web_evidence.grounded_enabled() else None" in src
    # every pre-pass thread registers its results and deep-reads
    assert src.count("_prepass_search(") >= 4
    # grounded: no post-hoc appendix; legacy appendix only on the fallback path
    assert "if _web_thread is not None and _web_grounded:" in src
    assert "_synthesize_web_summary" in src
    # tool path: cite_as ids before the result is serialized for the model
    i_annotate = src.index("annotate_tool_result(result")
    i_confirm = src.index("_web_registry.confirm_tool_shown(result_str)")
    assert i_annotate < src.index("result_str = serialize_tool_result(result)", i_annotate) < i_confirm
    # final cited-first listing replaces the turn's sources
    assert '_emit_registry_sources(_cited_ids, replace=True, phase="final")' in src
    # the registry reaches the finalizer (citation cleanup + verifier)
    assert "web_registry=_web_registry," in src


def test_scaffold_and_rag_rule_ask_for_inline_tags():
    from core.agent import DUAL_SOURCE_SCAFFOLD

    assert "[W#]" in DUAL_SOURCE_SCAFFOLD and "Updated Information from the Web" not in DUAL_SOURCE_SCAFFOLD
    assert "3. WEB EVIDENCE:" in inspect.getsource(runner._stream_response_api_impl)


# ── guard task-25bee13-20119: execution-level paths (CX-24) ─────────────


def _conductor_agent(dispatch_result):
    """A bare agent whose sub-agent makes one web_search call, then answers."""
    agent = QuasarAgent.__new__(QuasarAgent)
    agent._tls = threading.local()
    agent.config = NS(model="m")
    agent.conductor = NS(conductor_model="m")
    agent.last_run_result = None
    agent._build_tools_for_responses_api = lambda: [{"name": "web_search"}, {"name": "search_by_target"}]
    agent._tool_status_label = lambda name, args: f"Calling {name}"

    call = NS(type="function_call", name="web_search", arguments='{"query": "q"}', call_id="c1")
    responses = iter([NS(output=[call], output_text="", id="r1"), NS(output=[], output_text="done", id="r2")])
    agent.client = NS(responses=NS(create=lambda **kw: next(responses)))

    def dispatch(name, args):
        agent._tls.last_dispatch_result = dispatch_result
        return "{}"

    agent._dispatch_tool_call = dispatch
    return agent


def test_cx01_conductor_web_card_goes_to_the_turn_sink_not_the_shared_callback():
    from core import web_policy

    payload = {"success": True, "provider": "Tavily", "query": "q",
               "results": [{"title": "T", "url": "https://a.org/x", "snippet": "s"}]}
    agent = _conductor_agent(payload)
    mine, other = [], []
    agent._last_on_status = lambda step, state: other.append((step, state))  # a concurrent turn's callback
    with web_policy.web_scope(True, lambda step, state: mine.append((step, state))):
        agent._conductor_tool_executor("find the page")
    assert any(state == "web_sources" and step.startswith("__event__") for step, state in mine)
    assert not any(state == "web_sources" for _s, state in other)


def test_cx01_conductor_without_a_turn_sink_emits_no_card():
    payload = {"success": True, "results": [{"url": "https://a.org/x", "snippet": "s"}]}
    agent = _conductor_agent(payload)
    seen = []
    agent._last_on_status = lambda step, state: seen.append(state)
    agent._conductor_tool_executor("find the page")
    assert "web_sources" not in seen


def test_cx03_conductor_answers_lose_every_web_tag():
    out = runner._finalize_answer_text(_agent(), "The deadline was October 15, 2025 [W1]. Extra [W2, W3].",
                                       on_token=None, user_query="q", url_sources=[], all_tool_results=[],
                                       had_tool_calls=False, web_registry=EvidenceRegistry())
    assert "[W" not in out and "Verification" not in out


def test_cx08_finalizer_keeps_only_ids_the_model_was_shown():
    reg = EvidenceRegistry()
    reg.add_from_payload({"success": True, "results": [
        {"title": f"T{i}", "url": f"https://a{i}.org", "snippet": "October 15, 2025"} for i in range(12)
    ]})
    reg.render_prompt_block(max_items=10)
    out = runner._finalize_answer_text(_agent(), "Due October 15, 2025 [W2]. Guessed [W11].", on_token=None,
                                       user_query="q", url_sources=[], all_tool_results=[], had_tool_calls=False,
                                       web_registry=reg)
    assert "[W2]" in out and "W11" not in out


def test_cx05_legacy_join_needs_only_the_search_not_the_deep_read():
    reg = EvidenceRegistry()
    pre = PrepassEvidence(reg)

    def worker():
        pre.record_search(_payload(JWST))   # search back
        time.sleep(2.0)                     # slow deep read
        pre.finish()

    threading.Thread(target=worker, daemon=True).start()
    t0 = time.monotonic()
    assert pre.search_done.wait(5) is True and time.monotonic() - t0 < 1.0
    src = inspect.getsource(runner._stream_response_api_impl)
    assert src.count("_web_prepass.search_done.wait(") >= 2   # legacy standard path + Conductor path


def test_cx04_deep_read_after_injection_leaves_the_verifier_on_the_shown_text():
    reg = EvidenceRegistry()
    pre = PrepassEvidence(reg)
    pre.record_search(_payload(JWST))
    out, block = runner._inject_web_evidence("User: q", "q", reg, pre, evidence_query="q", wait_s=0.2)
    assert block
    reg.update_excerpt(JWST[1], "Unrelated replacement text with no dates.")   # late deep read
    final = runner._finalize_answer_text(_agent(), "The deadline is October 15, 2025 [W1].", on_token=None,
                                         user_query="q", url_sources=[], all_tool_results=[],
                                         had_tool_calls=False, web_registry=reg)
    assert "Verification" not in final
