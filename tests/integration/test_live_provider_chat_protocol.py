"""Opt-in provider smoke tests for two-turn streaming tool calls."""

import json
import os

import pytest

from core.llm_client import LLMClient


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_PROVIDER_TESTS") != "1",
        reason="Set RUN_LIVE_PROVIDER_TESTS=1 to call configured model providers.",
    ),
]

SEARCH_TOOL = {
    "type": "function",
    "name": "search_papers",
    "description": "Search astronomy papers.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}
INSTRUCTIONS = (
    "For a paper request, call search_papers exactly once. "
    "After receiving tool output, answer with the word done and no tool call."
)


def _configured_models():
    candidates = [
        ("openai", os.getenv("OPENAI_PROVIDER_SMOKE_MODEL", "gpt-4o-mini"), "OPENAI_API_KEY"),
        ("deepseek", os.getenv("DEEPSEEK_PROVIDER_SMOKE_MODEL", "deepseek-v4-flash"), "DEEPSEEK_API_KEY"),
        ("tacc", os.getenv("TACC_PROVIDER_SMOKE_MODEL", "gpt-oss-120b"), "TACC_API_KEY"),
    ]
    return [
        pytest.param(model, id=provider)
        for provider, model, key_name in candidates
        if os.getenv(key_name)
    ]


def _complete_turn(client, model, question, previous_response_id=None):
    kwargs = {
        "model": model,
        "instructions": INSTRUCTIONS,
        "input": question,
        "tools": [SEARCH_TOOL],
        "max_output_tokens": 200,
        "stream": True,
    }
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id
    events = list(client.responses.create(**kwargs))
    response_id = next(
        event.response.id for event in events if event.type == "response.created"
    )
    completed = next(
        event.response for event in events if event.type == "response.completed"
    )
    calls = [
        item for item in completed.output if getattr(item, "type", "") == "function_call"
    ]
    assert len(calls) == 1

    follow_up = list(
        client.responses.create(
            model=model,
            instructions=INSTRUCTIONS,
            previous_response_id=response_id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": calls[0].call_id,
                    "output": json.dumps({"papers": [{"title": "Smoke test"}]}),
                }
            ],
            tools=[SEARCH_TOOL],
            max_output_tokens=200,
            stream=True,
        )
    )
    final_id = next(
        event.response.id for event in follow_up if event.type == "response.created"
    )
    final_text = "".join(
        event.delta for event in follow_up if event.type == "response.output_text.delta"
    )
    assert final_text.strip()
    return final_id


@pytest.mark.parametrize("model", _configured_models())
def test_two_consecutive_streaming_literature_turns(model):
    client = LLMClient(model=model)
    response_id = _complete_turn(
        client,
        model,
        "Find recent papers on protostellar outflows with ALMA.",
    )
    _complete_turn(
        client,
        model,
        "Find recent papers on proper motions of protostellar outflows with ALMA.",
        previous_response_id=response_id,
    )
