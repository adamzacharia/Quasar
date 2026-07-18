"""Wall-clock budget for tool execution (slow-archive guard).

A single stuck external service must not hold a whole turn hostage (live
2026-07-18: one CADC TAP search ran 477 s behind the generic "Generating
answer" spinner). ``QuasarAgent._execute_tool_guarded`` runs tools on a
worker thread with a budget: on expiry the model receives a structured
TIMEOUT error it can answer around, and the abandoned worker's thread-local
results are discarded so they can never leak into a later round or another
request.
"""

import threading
import time
import types

import pytest

from core.agent import QuasarAgent
from core.llm_client import get_llm_request_context, llm_request_context


def _agent():
    agent = QuasarAgent.__new__(QuasarAgent)
    agent._tls = threading.local()
    return agent


def _tool(fn):
    return types.SimpleNamespace(execute=fn)


# ── budget resolution ────────────────────────────────────────────────────


def test_default_budget_applies(monkeypatch):
    monkeypatch.delenv("QUASAR_TOOL_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("QUASAR_TOOL_TIMEOUT_OVERRIDES", raising=False)
    assert (
        QuasarAgent._tool_timeout_seconds("search_cadc")
        == QuasarAgent._TOOL_TIMEOUT_DEFAULT_SECONDS
    )


def test_zero_env_disables_guard(monkeypatch):
    monkeypatch.setenv("QUASAR_TOOL_TIMEOUT_SECONDS", "0")
    assert QuasarAgent._tool_timeout_seconds("search_cadc") is None


def test_code_override_beats_default(monkeypatch):
    monkeypatch.delenv("QUASAR_TOOL_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("QUASAR_TOOL_TIMEOUT_OVERRIDES", raising=False)
    assert QuasarAgent._tool_timeout_seconds("web_research") == 420.0


def test_env_override_beats_everything(monkeypatch):
    monkeypatch.delenv("QUASAR_TOOL_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv(
        "QUASAR_TOOL_TIMEOUT_OVERRIDES", "web_research=37.5, search_cadc=90"
    )
    assert QuasarAgent._tool_timeout_seconds("web_research") == 37.5
    # Whitespace around the name must not break matching.
    assert QuasarAgent._tool_timeout_seconds("search_cadc") == 90.0


def test_invalid_env_values_fall_back(monkeypatch):
    monkeypatch.setenv("QUASAR_TOOL_TIMEOUT_SECONDS", "not-a-number")
    monkeypatch.setenv("QUASAR_TOOL_TIMEOUT_OVERRIDES", "search_cadc=fast")
    assert (
        QuasarAgent._tool_timeout_seconds("search_cadc")
        == QuasarAgent._TOOL_TIMEOUT_DEFAULT_SECONDS
    )


# ── guarded execution ────────────────────────────────────────────────────


def test_fast_tool_returns_result_and_merges_tls():
    agent = _agent()

    def execute(**kwargs):
        agent.last_run_result = {"type": "data", "rows": 3}
        agent._accumulated_run_results.append({"type": "data", "rows": 3})
        agent.last_search_results = ["row"]
        agent._alma_tap_provenance_state.update(
            {"query": "SELECT 1", "url": "https://tap"}
        )
        return {"success": True, "value": kwargs["x"]}

    result = agent._execute_tool_guarded(
        _tool(execute), {"x": 42}, tool_name="fast_tool", timeout_seconds=5.0
    )

    assert result == {"success": True, "value": 42}
    # Worker-thread TLS deltas must be visible on the calling thread.
    assert agent.last_run_result == {"type": "data", "rows": 3}
    assert agent._accumulated_run_results == [{"type": "data", "rows": 3}]
    assert agent.last_search_results == ["row"]
    assert agent._alma_tap_provenance_state["query"] == "SELECT 1"


def test_timeout_returns_structured_error_quickly():
    agent = _agent()
    release = threading.Event()

    def execute(**kwargs):
        release.wait(5.0)
        return {"success": True}

    start = time.monotonic()
    result = agent._execute_tool_guarded(
        _tool(execute), {}, tool_name="slow_tool", timeout_seconds=0.2
    )
    elapsed = time.monotonic() - start
    release.set()

    assert elapsed < 2.0, "the turn must reclaim control at the budget"
    assert result["success"] is False
    assert result["timeout"] is True
    assert "TIMEOUT" in result["error"]
    assert "slow_tool" in result["error"]


def test_late_worker_results_are_discarded():
    agent = _agent()
    release = threading.Event()
    worker_done = threading.Event()

    def execute(**kwargs):
        release.wait(5.0)
        agent.last_run_result = {"type": "data", "stale": True}
        agent._accumulated_run_results.append({"type": "data", "stale": True})
        worker_done.set()
        return {"success": True}

    result = agent._execute_tool_guarded(
        _tool(execute), {}, tool_name="slow_tool", timeout_seconds=0.1
    )
    assert result["timeout"] is True

    # Let the abandoned worker finish, then confirm nothing leaked into the
    # calling thread's request state.
    release.set()
    assert worker_done.wait(5.0)
    time.sleep(0.1)
    assert agent.last_run_result is None
    assert agent._accumulated_run_results == []


def test_tool_exception_reraised_on_calling_thread():
    agent = _agent()

    def execute(**kwargs):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        agent._execute_tool_guarded(
            _tool(execute), {}, tool_name="raiser", timeout_seconds=5.0
        )


def test_disabled_guard_runs_on_calling_thread(monkeypatch):
    monkeypatch.setenv("QUASAR_TOOL_TIMEOUT_SECONDS", "0")
    agent = _agent()
    seen = {}

    def execute(**kwargs):
        seen["ident"] = threading.get_ident()
        return "ok"

    result = agent._execute_tool_guarded(_tool(execute), {}, tool_name="inline")
    assert result == "ok"
    assert seen["ident"] == threading.get_ident()


def test_llm_request_context_reaches_worker_thread():
    agent = _agent()
    seen = {}

    def execute(**kwargs):
        ctx = get_llm_request_context()
        seen["user_id"] = ctx.user_id if ctx else None
        return "ok"

    with llm_request_context(user_id="user-123"):
        agent._execute_tool_guarded(
            _tool(execute), {}, tool_name="ctx_tool", timeout_seconds=5.0
        )

    assert seen["user_id"] == "user-123"


def test_timeout_emits_closing_status():
    agent = _agent()
    statuses = []

    def on_status(text, state):
        statuses.append((text, state))

    result = agent._execute_tool_guarded(
        _tool(lambda **kw: time.sleep(3) or "late"),
        {},
        tool_name="slow_tool",
        step_label="Searching CADC archive",
        on_status=on_status,
        timeout_seconds=0.1,
    )

    assert result["timeout"] is True
    timed_out = [s for s in statuses if "timed out" in s[0]]
    assert timed_out and timed_out[0][1] == "completed"
    assert "Searching CADC archive" in timed_out[0][0]
