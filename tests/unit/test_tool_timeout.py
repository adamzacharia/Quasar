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


# ── consecutive-timeout circuit breaker (DH-P0) ─────────────────────────
# The model retries timed-out calls despite the "Do NOT retry" error text
# (live 2026-08-04 density repro); each retry burned a full guard budget.
# After _TOOL_TIMEOUT_BREAKER_TRIPS timeouts of the same tool in one turn,
# further calls must fail instantly.


def _breaker_agent():
    agent = _agent()
    # core/runner.py resets this dict at the top of every turn.
    agent._tls.tool_timeout_breaker = {}
    return agent


def test_breaker_trips_after_two_timeouts_and_fails_instantly():
    agent = _breaker_agent()
    release = threading.Event()
    calls = {"n": 0}

    def execute(**kwargs):
        calls["n"] += 1
        release.wait(5.0)
        return {"success": True}

    for _ in range(QuasarAgent._TOOL_TIMEOUT_BREAKER_TRIPS):
        result = agent._execute_tool_guarded(
            _tool(execute), {}, tool_name="slow_tool", timeout_seconds=0.1
        )
        assert result["timeout"] is True
        assert "circuit_breaker" not in result

    # A deliberately LONG budget: the breaker must return without waiting it
    # out, and the generous bound keeps the assertion immune to scheduler
    # stalls on loaded Windows CI (CX-24) while still proving instant-fail.
    start = time.monotonic()
    result = agent._execute_tool_guarded(
        _tool(execute), {}, tool_name="slow_tool", timeout_seconds=5.0
    )
    elapsed = time.monotonic() - start
    release.set()

    assert result["circuit_breaker"] is True
    assert result["timeout"] is True
    assert "CIRCUIT BREAKER" in result["error"]
    assert "slow_tool" in result["error"]
    assert elapsed < 1.0, "a breaker-tripped call must not wait out the budget"
    assert calls["n"] == QuasarAgent._TOOL_TIMEOUT_BREAKER_TRIPS, (
        "the tripped call must never execute the tool"
    )


def test_breaker_is_per_tool():
    agent = _breaker_agent()
    agent._tls.tool_timeout_breaker = {"slow_tool": 99}

    result = agent._execute_tool_guarded(
        _tool(lambda **kw: "ok"), {}, tool_name="other_tool", timeout_seconds=5.0
    )
    assert result == "ok"


def test_breaker_resets_on_completed_call():
    agent = _breaker_agent()
    agent._tls.tool_timeout_breaker = {"flaky_tool": 1}

    result = agent._execute_tool_guarded(
        _tool(lambda **kw: "ok"), {}, tool_name="flaky_tool", timeout_seconds=5.0
    )
    assert result == "ok"
    # A completed call proves the service responds again — streak cleared.
    assert agent._tls.tool_timeout_breaker == {}


def test_breaker_disabled_without_turn_state():
    # Conductor executor threads and bare test agents never set the TLS
    # dict — the guard must behave exactly as before there (no breaker).
    agent = _agent()
    for _ in range(3):
        result = agent._execute_tool_guarded(
            _tool(lambda **kw: time.sleep(3) or "late"),
            {},
            tool_name="slow_tool",
            timeout_seconds=0.05,
        )
        assert result["timeout"] is True
        assert "circuit_breaker" not in result


def test_breaker_status_carries_timed_out_phrase():
    # ui-pro/api/sse.py withholds its deadline extension for completed
    # statuses containing " timed out " — the breaker notice must match.
    agent = _breaker_agent()
    agent._tls.tool_timeout_breaker = {"slow_tool": 2}
    statuses = []

    result = agent._execute_tool_guarded(
        _tool(lambda **kw: "never"),
        {},
        tool_name="slow_tool",
        step_label="Fetching image cutout",
        on_status=lambda t, s: statuses.append((t, s)),
        timeout_seconds=5.0,
    )

    assert result["circuit_breaker"] is True
    assert any(" timed out " in t and s == "completed" for t, s in statuses)


def test_worker_clear_of_last_run_result_propagates():
    """A guarded tool that CLEARS the stale card (Data Lab ctx provider sets
    last_run_result = None) must propagate that clear to the calling thread —
    the old `is not None` merge dropped it, so a stale data card from an
    earlier tool survived and could be re-emitted by the streaming loop."""
    agent = _agent()
    agent.last_run_result = {"type": "data", "stale": True}
    agent.last_search_results = ["stale-row"]

    def execute(**kwargs):
        agent.last_run_result = None
        agent.last_search_results = None
        return {"success": True}

    result = agent._execute_tool_guarded(
        _tool(execute), {}, tool_name="clearer", timeout_seconds=5.0
    )
    assert result == {"success": True}
    assert agent.last_run_result is None
    assert agent.last_search_results is None


def test_untouched_tls_does_not_clobber_parent_state():
    """The inverse guard: a tool that never touches card state must leave the
    calling thread's existing values alone (the set-flags must be False)."""
    agent = _agent()
    agent.last_run_result = {"type": "data", "keep": True}

    result = agent._execute_tool_guarded(
        _tool(lambda **kw: {"success": True}), {}, tool_name="hands_off",
        timeout_seconds=5.0,
    )
    assert result == {"success": True}
    assert agent.last_run_result == {"type": "data", "keep": True}


def test_result_indicates_timeout_classification():
    """CX-16/CX-27: the runner's timeout classification is central and typed —
    guard timeouts, FITS watchdog errors, and all-panels-timed-out grids all
    stamp timeout=True; everything else closes as a normal completion."""
    from core.runner import _result_indicates_timeout

    assert _result_indicates_timeout({"success": False, "timeout": True})
    assert _result_indicates_timeout(
        {"success": False, "timeout": True, "circuit_breaker": True}
    )
    assert not _result_indicates_timeout({"success": True})
    assert not _result_indicates_timeout({"success": False, "error": "boom"})
    assert not _result_indicates_timeout({"timeout": "true"})  # strict is-True
    assert not _result_indicates_timeout("TIMEOUT")
    assert not _result_indicates_timeout(None)


def test_final_composition_reraises_quota_error(monkeypatch):
    """CX-21: a quota trip during final-answer composition must reach the
    runner's dedicated handler, not vanish into an empty string."""
    from services.usage_quota_service import QuotaExceededError

    agent = _agent()

    class _Responses:
        @staticmethod
        def create(**kwargs):
            raise QuotaExceededError("weekly platform quota exhausted")

    agent.client = types.SimpleNamespace(responses=_Responses())
    agent.config = types.SimpleNamespace(
        temperature=0.2, max_tokens=1024, model="gpt-4.1"
    )
    monkeypatch.setattr(QuasarAgent, "system_prompt", "test prompt", raising=False)
    # Non-empty tool results — an empty list legitimately short-circuits to ""
    # before any LLM call is made.
    with pytest.raises(QuotaExceededError):
        agent._compose_final_answer_from_tools(
            "q", [{"output": '{"rows": 3}'}], "gpt-4.1"
        )


def test_density_vetting_override(monkeypatch):
    """Density report fix 10: vetting = SQL + cutout grid in one call and
    cannot fit the 150 s default under archive load."""
    monkeypatch.delenv("QUASAR_TOOL_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("QUASAR_TOOL_TIMEOUT_OVERRIDES", raising=False)
    assert QuasarAgent._tool_timeout_seconds("datalab_density_vetting") == 300.0
