"""A provider stream that dies mid-round must not cost the whole turn.

Live 2026-09-17 (AstroDataBench data-discovery reruns): 10 of 27 quasar/gpt-oss
trials scored 0.000 because the reasoning-only RE-SAMPLE — the only request
Quasar sends with server-enforced ``tool_choice="required"`` — died with an
in-band ``{"error": ...}`` event on an already-200 SSE stream
(``litellm.MidStreamFallbackError ... An error occurred during streaming``).
Three independent gaps let it through:

1. ``core.retry._is_retryable`` classified on type name / HTTP status only; the
   openai SDK raises the bare ``APIError`` (status None) for an in-band error,
   so the runner's stream-round retry never fired (tests in
   test_retry_transport_errors.py).
2. The ``required``→``auto`` fallback existed only for a request-time 400 on the
   non-streaming path; ``_stream_tacc`` had none.
3. A turn that lost its stream returned a bare error string even when tool
   rounds had already produced evidence.

Replaying the captured body non-streaming exposed vLLM's real error —
``Specified tool_choice: "required", but the model output contains ...`` — i.e.
this server validates ``required`` AFTER generation instead of enforcing it, so
a reasoning-only stop under ``required`` becomes a dead stream rather than a
recoverable empty round.
"""
import json
from types import SimpleNamespace as NS

import httpx
import openai
import pytest

from core.llm_client import LLMClient, ResponsesShim
from tests.unit.test_discovery_recovery import _Responses, _events, _tool_agent


def _midstream_error(message="litellm.MidStreamFallbackError: litellm.APIConnectionError: "
                             "APIConnectionError: OpenAIException - An error occurred during streaming."):
    return openai.APIError(
        message=message,
        request=httpx.Request("POST", "http://tacc.test/v1/chat/completions"),
        body={"message": message, "type": None, "param": None, "code": "500"},
    )


def _dying_round(prefix_events, exc):
    """A provider stream that emits ``prefix_events`` and then raises ``exc``."""
    def gen():
        yield from prefix_events
        raise exc
    return gen()


_TOOLS = [{"type": "function", "name": "query_archive", "description": "d",
           "parameters": {"type": "object", "properties": {}}}]


# ── shim: streaming required→auto fallback ───────────────────────────────────

class _TaccEngine:
    """Fake chat.completions engine: ``required`` streams reasoning then dies
    mid-stream; ``auto`` streams a real tool call."""

    def __init__(self, fail_required_times=99):
        self.calls = []
        self.fail_required_times = fail_required_times

    def create(self, **kw):
        self.calls.append(kw)
        if kw.get("tool_choice") == "required" and self.fail_required_times > 0:
            self.fail_required_times -= 1
            return _dying_round(
                [NS(choices=[NS(delta=NS(content=None, tool_calls=None, reasoning_content="thinking..."), finish_reason=None)])],
                _midstream_error(),
            )
        return iter([
            NS(choices=[NS(delta=NS(content=None, tool_calls=None, reasoning_content="ok, calling"), finish_reason=None)]),
            NS(choices=[NS(delta=NS(content=None, tool_calls=[
                NS(index=0, id="call-1", function=NS(name="query_archive", arguments='{"x": 1}'))]), finish_reason="tool_calls")]),
        ])


@pytest.fixture
def fresh_required_state(monkeypatch):
    monkeypatch.setattr(ResponsesShim, "_tacc_required_supported", True)
    monkeypatch.setattr(ResponsesShim, "_tacc_required_cooldown_until", 0.0)


def _tacc_client(engine):
    client = LLMClient(model="gpt-oss-120b")
    client._get_tacc_client = lambda: NS(chat=NS(completions=engine))
    return client


def test_stream_death_under_required_falls_back_to_auto_and_completes(fresh_required_state, capsys):
    engine = _TaccEngine()
    client = _tacc_client(engine)
    events = list(client.responses.create(model="gpt-oss-120b", instructions="sys", input="find data",
                                          tools=_TOOLS, tool_choice="required", tool_choice_strict=True, stream=True))
    assert [c["tool_choice"] for c in engine.calls] == ["required", "auto"]
    assert all("Tool use is REQUIRED" in c["messages"][0]["content"] for c in engine.calls)  # nudge kept on the fallback
    completed = next(e.response for e in events if e.type == "response.completed")
    assert [fc.name for fc in completed.output] == ["query_archive"] and completed.finish_reason == "tool_calls"
    # reasoning from BOTH attempts was forwarded (commentary, harmless); exactly one 'done'
    assert [e.delta for e in events if e.type == "response.reasoning_summary_text.delta"] == ["thinking...", "ok, calling"]
    assert sum(e.type == "response.reasoning_summary_text.done" for e in events) == 1
    assert "died under tool_choice=required" in capsys.readouterr().out
    # the history cache holds the completed attempt only
    cached = client.responses._history_cache[completed.id]
    assert cached[-1]["role"] == "assistant" and cached[-1]["tool_calls"][0]["function"]["name"] == "query_archive"


def test_required_is_cooled_down_after_a_mid_stream_death_then_restored(fresh_required_state, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("core.llm_client.time.monotonic", lambda: clock[0])
    monkeypatch.setenv("TACC_REQUIRED_TOOL_CHOICE_COOLDOWN_SECONDS", "300")
    engine = _TaccEngine(fail_required_times=1)
    client = _tacc_client(engine)
    list(client.responses.create(model="gpt-oss-120b", input="q1", tools=_TOOLS, tool_choice="required",
                                 tool_choice_strict=True, stream=True))
    assert [c["tool_choice"] for c in engine.calls] == ["required", "auto"]
    # within the cooldown a strict escalation is sent as auto (+nudge), no wasted death
    list(client.responses.create(model="gpt-oss-120b", input="q2", tools=_TOOLS, tool_choice="required",
                                 tool_choice_strict=True, stream=True))
    assert engine.calls[-1]["tool_choice"] == "auto" and "Tool use is REQUIRED" in engine.calls[-1]["messages"][0]["content"]
    clock[0] = 1000.0 + 301
    list(client.responses.create(model="gpt-oss-120b", input="q3", tools=_TOOLS, tool_choice="required",
                                 tool_choice_strict=True, stream=True))
    assert engine.calls[-1]["tool_choice"] == "required"  # cooldown over: server enforcement is tried again


def test_stream_death_under_auto_is_not_swallowed_by_the_shim(fresh_required_state):
    class _Engine:
        def create(self, **kw):
            return _dying_round([], _midstream_error())
    client = _tacc_client(_Engine())
    with pytest.raises(openai.APIError):
        list(client.responses.create(model="gpt-oss-120b", input="q", tools=_TOOLS, tool_choice="auto", stream=True))


def test_stream_death_after_a_tool_call_started_is_not_retried_inside_the_shim(fresh_required_state):
    """Once the failed attempt already produced a call fragment the shim must
    not silently restart (the runner decides); the error propagates."""
    class _Engine:
        def __init__(self):
            self.calls = []
        def create(self, **kw):
            self.calls.append(kw)
            return _dying_round([NS(choices=[NS(delta=NS(content=None, tool_calls=[
                NS(index=0, id="c1", function=NS(name="query_archive", arguments='{"x"'))]), finish_reason=None)])],
                _midstream_error())
    engine = _Engine()
    client = _tacc_client(engine)
    with pytest.raises(openai.APIError):
        list(client.responses.create(model="gpt-oss-120b", input="q", tools=_TOOLS, tool_choice="required",
                                     tool_choice_strict=True, stream=True))
    assert len(engine.calls) == 1


def test_non_transport_error_under_required_is_not_retried_inside_the_shim(fresh_required_state):
    class _Engine:
        def __init__(self):
            self.calls = []
        def create(self, **kw):
            self.calls.append(kw)
            return _dying_round([], ValueError("bad payload"))
    engine = _Engine()
    client = _tacc_client(engine)
    with pytest.raises(ValueError):
        list(client.responses.create(model="gpt-oss-120b", input="q", tools=_TOOLS, tool_choice="required",
                                     tool_choice_strict=True, stream=True))
    assert len(engine.calls) == 1


# ── runner: stream-round retry + strict downgrade + partial evidence ─────────

def test_runner_retries_a_dead_resample_and_downgrades_strict_required(capsys):
    # round 0: reasoning-only stop -> re-sample (strict) dies mid-stream -> retry (emulated) -> tool -> answer
    responses = _Responses([
        _events(1, reasoning="Let me call the archive tool."),
        _dying_round(_events(2, reasoning="thinking")[:2], _midstream_error()),
        _events(3, tool="query_archive"),
        _events(4, text="Done."),
    ])
    agent, executed = _tool_agent(responses)
    result = agent.stream_response_api("Show me an image of Synthetic 4", conversation_id="dead-resample")
    assert "Done." in result and len(executed) == 1 and len(responses.calls) == 4
    assert responses.calls[1]["tool_choice_strict"] is True          # the escalated re-sample
    assert responses.calls[2]["tool_choice_strict"] is False         # retried in the emulated form
    assert responses.calls[2]["input"] == responses.calls[1]["input"]  # same logical round
    out = capsys.readouterr().out
    assert "[STREAM] Round 1 stream died (APIError" in out and "downgraded to emulated required" in out
    assert "The language-model provider returned an error" not in result


def test_runner_gives_up_after_the_round_retry_budget_and_reports_the_failure(monkeypatch):
    monkeypatch.setenv("LLM_STREAM_ROUND_RETRIES", "1")
    monkeypatch.setattr("core.runner.time.sleep", lambda s: None)
    responses = _Responses([
        _dying_round([], _midstream_error()),
        _dying_round([], _midstream_error()),
    ])
    agent, executed = _tool_agent(responses)
    result = agent.stream_response_api("hello there", conversation_id="dead-twice")
    assert len(responses.calls) == 2 and not executed
    assert "provider returned an error" in result
    assert agent._tls.last_provider_failure["error_class"] == "APIError"


def test_turn_that_loses_the_stream_after_tool_results_returns_the_partial_evidence(monkeypatch, capsys):
    monkeypatch.setenv("LLM_STREAM_ROUND_RETRIES", "0")
    tokens = []
    responses = _Responses([
        _events(1, tool="query_archive"),
        _dying_round([], _midstream_error()),   # composing round dies, retries exhausted
    ])
    agent, executed = _tool_agent(responses, result={"success": True, "results": [{"value": 42}]})
    result = agent.stream_response_api("hello there", conversation_id="partial", on_token=tokens.append)
    assert len(executed) == 1
    assert "provider returned an error" in result
    partial = json.loads(result[result.index("{"):])
    assert partial["partial"] is True and partial["tool_results"][0]["results"] == [{"value": 42}]
    assert "Provider stream failed" in partial["reason"]
    assert "".join(tokens).endswith(result)  # the notice + evidence were streamed, not only returned
    assert "[RECOVERY] provider failure after 1 tool result(s)" in capsys.readouterr().out
    assert agent._tls.last_provider_failure["error_class"] == "APIError"  # the run is still recorded as failed
