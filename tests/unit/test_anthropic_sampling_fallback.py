"""Anthropic sampling-parameter handling.

Regression cover for the BYOK catalog exposing Claude models that are not in
the static no-sampling prefix list (e.g. claude-opus-5), which returned
400 "`temperature` is deprecated for this model." on every turn.
"""
from __future__ import annotations

import pytest

from core.llm_client import ResponsesShim


class _Rejection(Exception):
    """Mimics anthropic.BadRequestError: has .status_code and a body in str()."""

    status_code = 400

    def __init__(self, message: str = "`temperature` is deprecated for this model."):
        super().__init__(
            "Error code: 400 - {'type': 'error', 'error': {'type': "
            f"'invalid_request_error', 'message': '{message}'}}"
        )


@pytest.fixture(autouse=True)
def _clear_learned():
    ResponsesShim._ANTHROPIC_NO_SAMPLING_LEARNED.clear()
    yield
    ResponsesShim._ANTHROPIC_NO_SAMPLING_LEARNED.clear()


@pytest.mark.parametrize("model", [
    "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "claude-fable-5", "claude-mythos-5",
])
def test_known_no_sampling_models_never_send_temperature(model):
    assert ResponsesShim._anthropic_accepts_sampling(model) is False


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-opus-4-6", "claude-sonnet-4-6"])
def test_older_models_still_accept_sampling(model):
    assert ResponsesShim._anthropic_accepts_sampling(model) is True


def test_rejection_strips_sampling_and_is_remembered():
    call_kwargs = {"model": "claude-neptune-9", "temperature": 0.7, "max_tokens": 256}
    assert ResponsesShim._anthropic_accepts_sampling("claude-neptune-9") is True

    assert ResponsesShim._strip_sampling_after_rejection(call_kwargs, _Rejection()) is True
    assert "temperature" not in call_kwargs, "the retry must not resend the rejected param"
    assert call_kwargs["max_tokens"] == 256, "unrelated params are preserved"

    # An unknown future model is learned from the provider's own answer, so the
    # next call skips the wasted round-trip.
    assert ResponsesShim._anthropic_accepts_sampling("claude-neptune-9") is False


def test_learning_is_case_insensitive():
    call_kwargs = {"model": "Claude-Neptune-9", "temperature": 0.7}
    assert ResponsesShim._strip_sampling_after_rejection(call_kwargs, _Rejection()) is True
    assert ResponsesShim._anthropic_accepts_sampling("claude-neptune-9") is False


def test_top_p_and_top_k_are_stripped_together():
    call_kwargs = {"model": "claude-neptune-9", "temperature": 0.7, "top_p": 0.9, "top_k": 40}
    assert ResponsesShim._strip_sampling_after_rejection(call_kwargs, _Rejection()) is True
    assert not {"temperature", "top_p", "top_k"} & set(call_kwargs)


def test_unrelated_400_is_not_retried():
    call_kwargs = {"model": "claude-opus-5", "temperature": 0.7}
    unrelated = _Rejection("max_tokens: must be greater than 0")
    assert ResponsesShim._strip_sampling_after_rejection(call_kwargs, unrelated) is False
    assert call_kwargs["temperature"] == 0.7, "an unrelated failure must not mutate the call"
    assert not ResponsesShim._ANTHROPIC_NO_SAMPLING_LEARNED


def test_non_400_errors_are_not_retried():
    class _ServerError(Exception):
        status_code = 500

    call_kwargs = {"model": "claude-neptune-9", "temperature": 0.7}
    assert ResponsesShim._strip_sampling_after_rejection(call_kwargs, _ServerError("boom")) is False
    assert call_kwargs["temperature"] == 0.7


def test_rejection_without_sampling_params_present_is_not_retried():
    # Already stripped: retrying the identical call would just fail again.
    call_kwargs = {"model": "claude-opus-5", "max_tokens": 256}
    assert ResponsesShim._strip_sampling_after_rejection(call_kwargs, _Rejection()) is False


class _FakeStreamManager:
    def __init__(self, owner, kwargs):
        self._owner = owner
        self._kwargs = kwargs

    def __enter__(self):
        self._owner.calls.append(dict(self._kwargs))
        if "temperature" in self._kwargs:
            raise _Rejection()
        return "stream-object"

    def __exit__(self, *exc_info):
        return False


class _FakeMessages:
    def __init__(self, owner):
        self._owner = owner

    def stream(self, **kwargs):
        return _FakeStreamManager(self._owner, kwargs)


class _FakeClient:
    def __init__(self):
        self.calls: list[dict] = []
        self.messages = _FakeMessages(self)


def test_stream_retries_once_without_sampling():
    shim = ResponsesShim.__new__(ResponsesShim)
    client = _FakeClient()
    call_kwargs = {"model": "claude-neptune-9", "temperature": 0.7, "max_tokens": 128}

    with shim._anthropic_stream_cm(client, call_kwargs) as stream:
        assert stream == "stream-object"

    assert len(client.calls) == 2, "one rejected attempt, then one clean retry"
    assert "temperature" in client.calls[0]
    assert "temperature" not in client.calls[1]
    assert client.calls[1]["max_tokens"] == 128


def test_stream_failure_after_opening_is_not_replayed():
    """A mid-stream error must propagate, never re-run the request."""
    shim = ResponsesShim.__new__(ResponsesShim)
    client = _FakeClient()
    call_kwargs = {"model": "claude-haiku-4-5"}  # opens cleanly, no temperature

    with pytest.raises(_Rejection):
        with shim._anthropic_stream_cm(client, call_kwargs):
            raise _Rejection()

    assert len(client.calls) == 1, "the stream had already opened; no retry is safe"


class _Msg:
    def __init__(self, model):
        self.model = model


class _Event:
    def __init__(self, model):
        self.message = _Msg(model)


def test_served_model_alias_to_snapshot_is_not_a_mismatch(caplog):
    with caplog.at_level("INFO", logger="core.llm_client"):
        ResponsesShim._log_served_model("claude-haiku-4-5", _Event("claude-haiku-4-5-20251001"))
    assert "served model=claude-haiku-4-5-20251001" in caplog.text
    assert "but" not in caplog.text, "a dated snapshot of the requested alias is expected"


def test_served_model_substitution_warns(caplog):
    with caplog.at_level("WARNING", logger="core.llm_client"):
        ResponsesShim._log_served_model("claude-opus-5", _Event("claude-haiku-4-5"))
    assert "but claude-opus-5 was requested" in caplog.text


def test_served_model_missing_is_ignored(caplog):
    with caplog.at_level("INFO", logger="core.llm_client"):
        ResponsesShim._log_served_model("claude-opus-5", _Event(None))
    assert caplog.text == ""
