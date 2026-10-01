"""GPT-6 (gpt-6-luna / -sol / -astra) request-parameter handling.

The GPT-6 family rejects temperature and top_p with a 400 "Unsupported
parameter" (live 2026-09-28), which made every Quasar turn on gpt-6-luna fail.
QUASAR_OPENAI_REASONING_EFFORT sets reasoning.effort for the family (default high).
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from core.llm_client import ResponsesShim
from core.observability import MODEL_PRICING
from services.model_pricing import get_model_pricing


def _strip(kwargs):
    # _strip_unsupported_params does not touch instance state.
    return ResponsesShim._strip_unsupported_params(object.__new__(ResponsesShim), kwargs)


@pytest.fixture(autouse=True)
def _no_effort_env(monkeypatch):
    monkeypatch.delenv("QUASAR_OPENAI_REASONING_EFFORT", raising=False)


@pytest.mark.parametrize("model", ["gpt-6-luna", "gpt-6-sol", "gpt-6-astra"])
def test_gpt6_drops_temperature_and_top_p(model):
    out = _strip({"model": model, "input": "hi", "temperature": 0.2, "top_p": 0.9, "stream": True})
    assert "temperature" not in out and "top_p" not in out
    assert out["stream"] is True and out["input"] == "hi"
    assert out["reasoning"] == {"effort": "high"}  # no env -> Quasar default (high)


def test_effort_env_overrides_default(monkeypatch):
    monkeypatch.setenv("QUASAR_OPENAI_REASONING_EFFORT", "medium")
    assert _strip({"model": "gpt-6-luna"})["reasoning"] == {"effort": "medium"}


def test_other_models_keep_sampling_params():
    out = _strip({"model": "gpt-4.1", "temperature": 0.2, "top_p": 0.9})
    assert out["temperature"] == 0.2 and out["top_p"] == 0.9
    assert "temperature" not in _strip({"model": "gpt-5.4-mini", "temperature": 0.2})


def test_effort_env_merges_with_existing_summary(monkeypatch):
    monkeypatch.setenv("QUASAR_OPENAI_REASONING_EFFORT", "High")
    kwargs = {"model": "gpt-6-luna", "reasoning": {"summary": "auto"}}
    out = _strip(kwargs)
    assert out["reasoning"] == {"summary": "auto", "effort": "high"}
    assert kwargs["reasoning"] == {"summary": "auto"}  # caller's dict not mutated


def test_effort_env_not_applied_to_other_models(monkeypatch):
    monkeypatch.setenv("QUASAR_OPENAI_REASONING_EFFORT", "high")
    assert "reasoning" not in _strip({"model": "gpt-4.1"})


def test_unknown_effort_falls_back_to_default_and_logs(monkeypatch, caplog):
    monkeypatch.setenv("QUASAR_OPENAI_REASONING_EFFORT", "ultra")
    with caplog.at_level(logging.WARNING):
        assert _strip({"model": "gpt-6-luna"})["reasoning"] == {"effort": "high"}
    assert any("QUASAR_OPENAI_REASONING_EFFORT='ultra'" in r.getMessage() for r in caplog.records)


class _FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return object()


@pytest.mark.parametrize("stream", [False, True])
def test_call_openai_sends_sanitized_kwargs_to_sdk(monkeypatch, stream):
    """The SDK boundary (streaming and not) receives the stripped kwargs."""
    monkeypatch.setenv("QUASAR_OPENAI_REASONING_EFFORT", "high")
    fake = _FakeResponses()
    shim = object.__new__(ResponsesShim)
    shim._llm = SimpleNamespace(_get_openai_client=lambda: SimpleNamespace(responses=fake))
    shim._call_openai({"model": "gpt-6-luna", "input": "hi", "temperature": 0.2, "top_p": 0.9,
                       "stream": stream, "reasoning": {"summary": "auto"}})
    (sent,) = fake.calls
    assert "temperature" not in sent and "top_p" not in sent
    assert sent["stream"] is stream
    assert sent["reasoning"] == {"summary": "auto", "effort": "high"}


@pytest.mark.parametrize("model,rates", [
    ("gpt-6-luna", (0.10, 0.50)),
    ("gpt-6-sol", (2.00, 10.00)),
    ("gpt-6-astra", (10.00, 50.00)),
])
def test_gpt6_pricing_present(model, rates):
    assert get_model_pricing("openai", model) == {"input_per_mtok": rates[0], "output_per_mtok": rates[1]}


@pytest.mark.parametrize("model,expected", [
    ("gpt-6-luna", True), ("gpt-6-sol", True), ("gpt-6-astra", True),
    ("gpt-5.4-mini", True), ("o3", True), ("deepseek-v4-flash", True),
    ("gpt-4.1", False), ("gpt-oss-120b", False), ("", False), (None, False),
])
def test_runner_requests_reasoning_summary(model, expected):
    """The runner's main tool loop asks thinking models (GPT-6 included) for a
    reasoning summary, which it streams to the UI Thought panel."""
    from core.runner import _wants_reasoning_summary

    assert _wants_reasoning_summary(model) is expected


@pytest.mark.parametrize("model,per_1k", [
    ("gpt-6-luna", (0.0001, 0.0005)),
    ("gpt-6-sol", (0.002, 0.010)),
    ("gpt-6-astra", (0.010, 0.050)),
])
def test_gpt6_trace_cost_rates(model, per_1k):
    assert MODEL_PRICING[model] == per_1k
