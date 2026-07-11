"""stream_response_api error handling — malformed tool-call recovery.

When gpt-oss emits unparseable tool-call JSON (e.g. a JSON comment inside the
arguments), the litellm proxy rejects the request with "Failed to parse tool
call from GPT OSS output ... Invalid function calling output".  The agent must
clear the poisoned response state, retry once, and never surface the raw
provider error dump as the assistant message.
"""

import threading
from types import SimpleNamespace

from core.agent import QuasarAgent


PARSE_ERROR = (
    "litellm.APIError: APIError: OpenAIException - Error code: 400 - "
    "{'error': {'message': 'litellm.BadRequestError: OpenAIException - "
    "Failed to parse tool call from GPT OSS output: Expecting value: "
    "line 1 column 58 (char 57). Invalid function calling output'}}"
)


class _FakeResponses:
    """Mimics LLMClient.responses — fails N times, then streams a text answer."""

    def __init__(self, failures, error_text=PARSE_ERROR, answer="recovered answer"):
        self.failures = failures
        self.error_text = error_text
        self.answer = answer
        self.create_calls = []
        self.cleared = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if len(self.create_calls) <= self.failures:
            raise RuntimeError(self.error_text)
        return iter(
            [
                SimpleNamespace(
                    type="response.created",
                    response=SimpleNamespace(id=f"resp_{len(self.create_calls)}"),
                ),
                SimpleNamespace(type="response.output_text.delta", delta=self.answer),
            ]
        )

    def clear_history(self, response_id=None):
        self.cleared.append(response_id)


def _make_agent(fake_responses):
    agent = object.__new__(QuasarAgent)
    agent._tls = threading.local()
    agent._conv_response_ids = {}
    agent._conv_run_tokens = {}
    agent._conv_ids_lock = threading.Lock()
    agent.config = SimpleNamespace(
        model="gpt-oss-120b", temperature=0.2, max_tokens=2048, verbose=False
    )
    agent.system_prompt = "system prompt"
    agent.ads_client = None
    agent.long_term_memory = None
    agent.client = SimpleNamespace(responses=fake_responses)
    # Stub preamble services not under test
    agent._prune_session_if_needed = lambda *a, **k: None
    agent._build_tools_for_responses_api = lambda: []
    return agent


def test_parse_error_clears_state_and_retry_succeeds():
    fake = _FakeResponses(failures=1)
    agent = _make_agent(fake)
    agent._set_response_id("conv-1", "poisoned-resp", "gpt-oss-120b")

    result = agent.stream_response_api("hello there", conversation_id="conv-1")

    assert result == "recovered answer"
    assert len(fake.create_calls) == 2
    # The poisoned chain was cleared before the retry...
    assert fake.cleared == ["poisoned-resp"]
    assert fake.create_calls[0]["previous_response_id"] == "poisoned-resp"
    # ...so the retry starts a fresh conversation chain.
    assert fake.create_calls[1]["previous_response_id"] is None


def test_parse_error_twice_returns_friendly_message_without_raw_dump():
    fake = _FakeResponses(failures=2)
    agent = _make_agent(fake)

    result = agent.stream_response_api("hello there", conversation_id="conv-2")

    # Retried exactly once — no recovery loop.
    assert len(fake.create_calls) == 2
    assert "malformed tool call" in result
    assert "could not be completed" in result
    # The raw litellm/provider dump never reaches the user.
    assert "litellm" not in result
    assert "Failed to parse tool call" not in result
    assert "OpenAIException" not in result


def test_generic_error_returns_friendly_message_without_retry():
    fake = _FakeResponses(failures=5, error_text="kaboom: connection reset by peer")
    agent = _make_agent(fake)

    result = agent.stream_response_api("hello there", conversation_id="conv-3")

    # Not a parse/hanging-tool error — no retry.
    assert len(fake.create_calls) == 1
    assert "could not be completed" in result
    assert "kaboom" not in result
