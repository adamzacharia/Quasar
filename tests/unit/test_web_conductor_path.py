"""Execution-level test of a grounded turn that the Conductor answers
(guard task-25bee13-20119 CX-02 / CX-03 / CX-05 / CX-07 / CX-24): the real
_stream_response_api_impl runs the pre-pass, the evidence join and the
Conductor branch against a mocked agent."""
from __future__ import annotations

import json
import re
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

import core.runner as runner


PAYLOAD = {
    "success": True,
    "provider": "Tavily",
    "query": "JWST Cycle 5 deadline",
    "results": [
        {"title": "JWST Cycle 5 CfP", "url": "https://www.stsci.edu/jwst/cycle-5", "snippet": "Proposals due 15 October 2025."},
        {"title": "Deadline extension", "url": "https://science.nasa.gov/jwst-extension", "snippet": "An extension was offered."},
    ],
}


def _agent(*, deep_read_s: float, conductor_answer: str):
    agent = MagicMock()
    agent._tls = threading.local()
    agent.config = NS(model="m", temperature=0.2, max_tokens=100)
    agent._LIVE_DATA_KEYWORDS_RE = re.compile(r"\b(?:archive)\b")
    agent._is_live_data_query = lambda q: False
    agent._detect_beyond_cutoff = lambda q: None
    agent._has_web_provider_key = lambda: True
    agent._detect_web_search_needed_via_llm = lambda q: False
    agent.long_term_memory = None
    agent._build_tools_for_responses_api = lambda: [{"name": "web_search"}]
    agent.system_prompt = "sys"
    agent.memory = NS(get_last_n_turns=lambda n: [])
    agent._conductor_images = []

    def search(query, max_results=10, search_depth="basic", want_images=None):
        return json.loads(json.dumps(PAYLOAD))

    agent._tavily_web_search = search
    agent._synthesize_web_summary = lambda q, data, reason: "The deadline was 15 October 2025."
    agent._build_web_sources_event = lambda payload: {
        "type": "web_sources", "sources": [dict(r) for r in payload.get("results", [])], "images": [],
        "query": payload.get("query", ""), "provider": "Tavily", "image_provider": "", "search_type": "",
    }
    trace = []
    agent._record_tool_trace = lambda name, args, result_str, result_obj=None, provenance=None: trace.append((name, args))
    agent._accumulated_tool_trace = []
    agent.complexity_detector = NS(assess=lambda q: NS(score=0.99))
    agent.query_tracer = MagicMock()

    async def orchestrate(query, **kw):
        return conductor_answer, NS(dag=MagicMock())

    agent.conductor = NS(orchestrate=orchestrate)
    agent._strip_unverified_urls = lambda text, sources, on_token=None, user_query="": text
    agent.tool_registry = NS(list_tools=lambda: [])
    agent.last_run_result = None
    return agent, trace


@pytest.fixture
def slow_deep_read(monkeypatch):
    from services import web_evidence

    calls = {}

    def fake_deep_read(registry, query, **kw):
        calls["started"] = time.monotonic()
        time.sleep(calls.get("seconds", 0))
        return 0

    monkeypatch.setattr(web_evidence, "deep_read", fake_deep_read)
    monkeypatch.setenv("QUASAR_WEB_EVIDENCE_WAIT", "0.3")
    monkeypatch.setattr(runner, "is_explicit_query", lambda q: False)
    monkeypatch.setattr(runner, "_maybe_failover_model", lambda a, m, *x, **k: m)
    import services.rag_service as rag

    monkeypatch.setattr(rag, "is_domain_relevant", lambda *a, **k: False, raising=False)
    monkeypatch.setattr(runner.Conductor, "COMPLEXITY_THRESHOLD", 0.5, raising=False)
    monkeypatch.setattr(runner.Conductor, "classify_tier", staticmethod(lambda c: ("high", 3)), raising=False)
    return calls


def _events(statuses):
    out = []
    for step, state in statuses:
        if isinstance(step, str) and step.startswith("__event__"):
            out.append(json.loads(step[len("__event__"):]))
    return out


def _run(agent):
    statuses, tokens = [], []
    answer = runner._stream_response_api_impl(
        agent, "When is the JWST Cycle 5 deadline?", web_search=True, conversation_id="c1",
        on_status=lambda step, state: statuses.append((step, state)), on_token=tokens.append,
    )
    return answer, statuses


def test_conductor_turn_never_shows_numbered_sources_and_keeps_the_trace(slow_deep_read):
    slow_deep_read["seconds"] = 0.0
    agent, trace = _agent(deep_read_s=0.0, conductor_answer="Conductor answer [W1].")
    answer, statuses = _run(agent)
    web_events = [e for e in _events(statuses) if e.get("type") == "web_sources"]
    assert web_events, "the Conductor turn still shows its web sources"
    assert all(not s.get("id") for e in web_events for s in e.get("sources", [])), "no numbered source is ever shown"
    assert "[W1]" not in answer                      # CX-03: tags stripped, the Conductor saw no evidence
    assert "From the Web" in answer                  # legacy appendix for the Conductor answer
    assert any(name == "web_search" and args.get("source") == "pre-pass" for name, args in trace)   # CX-07


def test_conductor_legacy_join_waits_for_the_search_not_the_deep_read(slow_deep_read):
    slow_deep_read["seconds"] = 3.0                  # the deep read runs long
    agent, _trace = _agent(deep_read_s=3.0, conductor_answer="Conductor answer.")
    t0 = time.monotonic()
    answer, _statuses = _run(agent)
    assert "From the Web" in answer
    assert time.monotonic() - t0 < 2.5               # did not wait for the 3 s deep read (CX-05)
