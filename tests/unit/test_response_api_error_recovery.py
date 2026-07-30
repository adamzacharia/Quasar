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


def test_provider_failure_is_recorded_for_run_status():
    """The friendly message rides back as normal text, so the SSE layer needs
    the thread-local failure record to mark the run failed in chat_runs."""
    fake = _FakeResponses(failures=5, error_text="kaboom: connection reset by peer")
    agent = _make_agent(fake)

    agent.stream_response_api("hello there", conversation_id="conv-3b")

    failure = agent._tls.last_provider_failure
    assert failure["error_class"] == "RuntimeError"
    assert "kaboom" in failure["message"]
    assert len(failure["message"]) <= 500


class _FakeReadTimeout(Exception):
    """Name contains 'timeout' so core.retry classifies it as retryable."""


class _MidStreamFailResponses:
    """Mimics LLMClient.responses — the stream itself dies mid-iteration for
    the first N create() calls, then a later call streams a clean answer."""

    def __init__(self, failures, answer="resumed answer", pre_text=""):
        self.failures = failures
        self.answer = answer
        self.pre_text = pre_text
        self.create_calls = []
        self.cleared = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if len(self.create_calls) <= self.failures:
            return self._dying_stream(len(self.create_calls))
        return iter(
            [
                SimpleNamespace(
                    type="response.created",
                    response=SimpleNamespace(id=f"resp_{len(self.create_calls)}"),
                ),
                SimpleNamespace(type="response.output_text.delta", delta=self.answer),
            ]
        )

    def _dying_stream(self, n):
        yield SimpleNamespace(
            type="response.created", response=SimpleNamespace(id=f"resp_{n}")
        )
        if self.pre_text:
            yield SimpleNamespace(type="response.output_text.delta", delta=self.pre_text)
        raise _FakeReadTimeout("read timed out waiting for the provider")

    def clear_history(self, response_id=None):
        self.cleared.append(response_id)


def _no_backoff(monkeypatch):
    monkeypatch.setattr("core.runner._retry_compute_delay", lambda *a, **k: 0.0)


def test_mid_stream_timeout_retries_only_that_round(monkeypatch):
    _no_backoff(monkeypatch)
    fake = _MidStreamFailResponses(failures=1)
    agent = _make_agent(fake)

    result = agent.stream_response_api("hello there", conversation_id="conv-4")

    # The round was re-issued and the turn completed normally.
    assert result == "resumed answer"
    assert len(fake.create_calls) == 2
    assert agent._tls.last_provider_failure is None
    # The retry re-sent the SAME round (same input), not a fresh turn.
    assert fake.create_calls[0]["input"] == fake.create_calls[1]["input"]


def test_mid_stream_timeout_after_visible_text_does_not_retry(monkeypatch):
    _no_backoff(monkeypatch)
    fake = _MidStreamFailResponses(failures=1, pre_text="partial answer text ")
    agent = _make_agent(fake)
    tokens = []

    result = agent.stream_response_api(
        "hello there", conversation_id="conv-5", on_token=tokens.append
    )

    # Text already reached the user — a retry would duplicate it.
    assert len(fake.create_calls) == 1
    assert "could not be completed" in result
    assert "(the model timed out)" in result
    assert agent._tls.last_provider_failure["error_class"] == "_FakeReadTimeout"


def test_mid_stream_timeout_retries_exhausted_returns_friendly_message(monkeypatch):
    _no_backoff(monkeypatch)
    fake = _MidStreamFailResponses(failures=10)
    agent = _make_agent(fake)

    result = agent.stream_response_api("hello there", conversation_id="conv-6")

    # Initial attempt + 2 retries, then the turn-level handler takes over.
    assert len(fake.create_calls) == 3
    assert "could not be completed" in result
    assert "(the model timed out)" in result
    assert agent._tls.last_provider_failure["error_class"] == "_FakeReadTimeout"


class _FakeEmptyReadTimeout(Exception):
    """Mimics httpx.ReadTimeout('') — str(e) is EMPTY, only the class signals."""


class _EmptyMessageFailResponses(_MidStreamFailResponses):
    def _dying_stream(self, n):
        yield SimpleNamespace(
            type="response.created", response=SimpleNamespace(id=f"resp_{n}")
        )
        raise _FakeEmptyReadTimeout("")


def test_provider_failure_with_empty_message_is_still_recorded(monkeypatch):
    """httpx.ReadTimeout('') stringifies empty — the failure record must still
    carry the error_class so the SSE layer can mark the run failed."""
    _no_backoff(monkeypatch)
    fake = _EmptyMessageFailResponses(failures=10)
    agent = _make_agent(fake)

    result = agent.stream_response_api("hello there", conversation_id="conv-7")

    assert "could not be completed" in result
    failure = agent._tls.last_provider_failure
    assert failure["error_class"] == "_FakeEmptyReadTimeout"
    assert failure["message"] == ""


def test_empty_message_failure_maps_to_failed_run_status():
    """The SSE mapping gates on error_class, not message — an empty-str
    exception must not fall through to run_status='completed'."""
    import sys
    from pathlib import Path

    ui_pro = str(Path(__file__).resolve().parents[2] / "ui-pro")
    if ui_pro not in sys.path:
        sys.path.insert(0, ui_pro)
    from api.sse import _provider_failure_error_message  # noqa: E402

    # error_class alone marks the run failed; the class name stands in for
    # the empty message so error_message is never blank.
    assert (
        _provider_failure_error_message(
            {"error_class": "_FakeEmptyReadTimeout", "message": ""}
        )
        == "_FakeEmptyReadTimeout"
    )
    assert (
        _provider_failure_error_message(
            {"error_class": "ReadTimeout", "message": "read timed out"}
        )
        == "ReadTimeout: read timed out"
    )
    # No recorded failure → the done path leaves the run completed.
    assert _provider_failure_error_message(None) is None
    assert _provider_failure_error_message({"error_class": "", "message": ""}) is None
