import threading
from types import SimpleNamespace

from core.agent import QuasarAgent


def _bare_agent():
    agent = object.__new__(QuasarAgent)
    agent.config = SimpleNamespace(model="gpt-oss-120b", verbose=False)
    agent._conv_response_ids = {}
    agent._conv_run_tokens = {}
    agent._conv_ids_lock = threading.Lock()
    agent.client = SimpleNamespace(
        responses=SimpleNamespace(clear_history=lambda response_id=None: None)
    )
    return agent


def test_response_state_isolated_by_conversation_provider_and_model():
    agent = _bare_agent()
    agent._set_response_id("conv-1", "tacc-response", "gpt-oss-120b")
    agent._set_response_id("conv-1", "openai-response", "gpt-4o-mini")
    agent._set_response_id("conv-2", "deepseek-response", "deepseek-v4-flash")

    assert agent._get_response_id("conv-1", "gpt-oss-120b") == "tacc-response"
    assert agent._get_response_id("conv-1", "gpt-4o-mini") == "openai-response"
    assert agent._get_response_id("conv-2", "deepseek-v4-flash") == "deepseek-response"


def test_clearing_one_provider_does_not_clear_other_provider_state():
    agent = _bare_agent()
    agent._set_response_id("conv-1", "tacc-response", "gpt-oss-120b")
    agent._set_response_id("conv-1", "openai-response", "gpt-4o-mini")

    agent.clear_response_state("conv-1", "gpt-oss-120b")

    assert agent._get_response_id("conv-1", "gpt-oss-120b") is None
    assert agent._get_response_id("conv-1", "gpt-4o-mini") == "openai-response"


def test_cancelled_worker_cannot_restore_stale_response_state():
    agent = _bare_agent()
    agent._begin_response_run("conv-1", "gpt-oss-120b", "run-old")
    agent._set_response_id(
        "conv-1",
        "old-response",
        "gpt-oss-120b",
        run_token="run-old",
    )

    agent.cancel_response_run("conv-1", "gpt-oss-120b", "run-old")
    agent._set_response_id(
        "conv-1",
        "late-old-response",
        "gpt-oss-120b",
        run_token="run-old",
    )

    assert agent._get_response_id("conv-1", "gpt-oss-120b") is None


def test_newer_run_token_rejects_late_state_from_previous_worker():
    agent = _bare_agent()
    agent._begin_response_run("conv-1", "gpt-oss-120b", "run-old")
    agent._begin_response_run("conv-1", "gpt-oss-120b", "run-new")

    agent._set_response_id(
        "conv-1",
        "late-old-response",
        "gpt-oss-120b",
        run_token="run-old",
    )
    agent._set_response_id(
        "conv-1",
        "new-response",
        "gpt-oss-120b",
        run_token="run-new",
    )

    assert agent._get_response_id("conv-1", "gpt-oss-120b") == "new-response"
