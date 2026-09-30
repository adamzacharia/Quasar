"""New picker model IDs reach the native Responses boundary without sampling errors."""
from types import SimpleNamespace

import pytest

from core.llm_client import ResponsesShim
from core.observability import estimate_cost
from services.model_pricing import get_model_pricing


@pytest.mark.parametrize("model", ["gpt-6.1-sol", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
    "gpt-5.5", "gpt-5.4", "gpt-5.4-pro", "gpt-5.4-nano", "gpt-5.3-codex",
    "gpt-5.2", "gpt-5.2-pro", "gpt-5.1", "gpt-5", "gpt-5-pro"])
@pytest.mark.parametrize("stream", [False, True])
def test_new_picker_models_use_native_responses_without_sampling(monkeypatch, model, stream):
    monkeypatch.delenv("QUASAR_OPENAI_REASONING_EFFORT", raising=False)
    calls = []
    fake = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: calls.append(kwargs)))
    shim = object.__new__(ResponsesShim)
    shim._llm = SimpleNamespace(_get_openai_client=lambda: fake)
    request = {"model": model, "input": "hi", "stream": stream,
               "temperature": 0.2, "top_p": 0.9, "tools": [{"type": "function", "name": "probe"}]}
    shim._call_openai(request)
    (sent,) = calls
    assert "temperature" not in sent and "top_p" not in sent
    assert sent["model"] == model and sent["stream"] is stream
    assert sent["tools"] == request["tools"]
    assert request["temperature"] == 0.2  # No mutation of the caller's request.
    rates = get_model_pricing("openai", model)
    assert rates is not None
    assert estimate_cost(model, 1000, 1000) == pytest.approx((rates["input_per_mtok"] + rates["output_per_mtok"]) / 1000)


@pytest.mark.parametrize("stream", [False, True])
def test_gpt61_reasoning_effort_preserves_summary_at_sdk_boundary(monkeypatch, stream):
    monkeypatch.setenv("QUASAR_OPENAI_REASONING_EFFORT", "High")
    calls = []
    fake = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: calls.append(kwargs)))
    shim = object.__new__(ResponsesShim)
    shim._llm = SimpleNamespace(_get_openai_client=lambda: fake)
    request = {"model": "gpt-6.1-sol", "input": "hi", "stream": stream, "reasoning": {"summary": "auto"}}
    shim._call_openai(request)
    assert calls[0]["reasoning"] == {"summary": "auto", "effort": "high"}
    assert calls[0]["stream"] is stream
    assert request["reasoning"] == {"summary": "auto"}
