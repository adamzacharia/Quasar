"""Image tiles are dropped on non-imagery answers on EVERY web path
(guard CX-30): the parallel supplement already did it; LLM-initiated web
tools and the Conductor path did not."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace as NS

import core.runner as runner
from core.llm_client import LLMClient
from core.runner import _web_event_payload
from tests.unit.test_discovery_recovery import _tool_agent

WEB_RESULT = {
    "success": True,
    "results": [{"url": "https://almascience.org/proposing/proprietary", "title": "ALMA proprietary period",
                 "content": "The proprietary period is 12 months."}],
    "images": ["https://images.example/unrelated-1.png", "https://images.example/unrelated-2.png"],
}


def test_payload_helper_keeps_images_only_for_imagery():
    assert _web_event_payload(WEB_RESULT, keep_images=False)["images"] == []
    assert _web_event_payload(WEB_RESULT, keep_images=True)["images"] == WEB_RESULT["images"]
    assert WEB_RESULT["images"], "the helper copies, never mutates the tool result"


def _web_events(query):
    calls = []

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if len(calls) == 1:
            delta = NS(content=None, tool_calls=[NS(index=0, id="call-1", function=NS(name="web_search", arguments=json.dumps({"query": query})))])
        else:
            delta = NS(content="Answer from the web result.", tool_calls=None)
        return iter([NS(choices=[NS(delta=delta, finish_reason="stop")])])

    client = LLMClient(model="gpt-oss-120b")
    client._get_tacc_client = lambda: NS(chat=NS(completions=NS(create=create)))
    agent, executed = _tool_agent(client.responses, tool_name="web_search", result=copy.deepcopy(WEB_RESULT))
    statuses = []
    agent.stream_response_api(query, conversation_id=f"cx30-{len(query)}", on_status=lambda m, s: statuses.append(m))
    events = [json.loads(m[len("__event__"):]) for m in statuses if isinstance(m, str) and m.startswith("__event__")]
    return executed, [e for e in events if e.get("type") == "web_sources"]


def test_llm_initiated_web_search_drops_images_on_a_knowledge_answer():
    executed, events = _web_events("Search the web: what is the ALMA proprietary period for Cycle 12 data?")
    assert executed, "the web tool ran"
    assert events, "a web_sources event was emitted"
    assert all(not e.get("images") for e in events), [e.get("images") for e in events]


def test_every_web_event_site_goes_through_the_helper():
    src = open(runner.__file__, encoding="utf-8").read()
    assert src.count("_web_event_payload(") >= 4  # definition + parallel + LLM tools + Conductor
    assert "agent._build_web_sources_event(result)" not in src
    assert "web_event_payload = dict(web_data)" not in src
