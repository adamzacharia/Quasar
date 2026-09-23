"""The tool guard's short-cooldown auto-wait, end to end (guard CX-10 / CX-33).

core/agent.py::_execute_tool_guarded refuses a tool whose every declared
service circuit is open -- but a SHORT cooldown (<= HOST_BREAKER_AUTO_WAIT_
SECONDS) is waited out inside the turn instead (UI benchmark 2026-09-22, L10
was told "retry in 20 s" and gave up). ALMA TAP tools are declared on the
``host/tap`` SERVICE circuits, so a soft-open TAP circuit is visible to this
precheck. The wait stops on turn cancellation and is skipped when the turn's
soft deadline leaves too little time.
"""
from __future__ import annotations

import threading
import time
import types

import pytest

from services import tool_budgets as tb
from services.host_breaker import HostBreaker, _now


TAP_CIRCUITS = tb.ALMA_TAP_HOSTS  # ("almascience.nrao.edu/tap", ...)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in ("HOST_BREAKER_DISABLED", "HOST_BREAKER_AUTO_WAIT_SECONDS", "QUASAR_TOOL_TIMEOUT_SECONDS"):
        monkeypatch.delenv(var, raising=False)
    HostBreaker.reset()
    yield
    HostBreaker.reset()
    tb.end_tool_deadline()


def _open_tap_circuits(monkeypatch, cooldown: float):
    monkeypatch.setenv("HOST_BREAKER_COOLDOWN_SECONDS", str(cooldown))
    with HostBreaker._lock:
        for key in TAP_CIRCUITS:
            HostBreaker._open_locked(key, why="test", detail="HTTP 504 x3", severity="soft", count_note="3 calls", now=_now())


def _agent():
    from core.agent import QuasarAgent

    agent = QuasarAgent.__new__(QuasarAgent)
    agent._tls = threading.local()
    agent._tls.tool_timeout_breaker = {}
    return agent


def _tool(ran):
    def execute(**kw):
        ran.append(time.monotonic())
        return {"success": True, "rows": 1}
    return types.SimpleNamespace(execute=execute)


def test_tap_tools_are_declared_on_the_tap_service_circuit():
    hosts = tb.hosts_for("query_alma_science_archive")
    assert set(TAP_CIRCUITS) <= set(hosts) and all(h.endswith("/tap") for h in TAP_CIRCUITS)
    # the soft TAP circuit is visible to the precheck (bare hosts were not)
    assert HostBreaker.open_hosts(TAP_CIRCUITS) == []


def test_short_soft_cooldown_is_waited_out_then_the_tool_runs(monkeypatch):
    agent, ran, statuses = _agent(), [], []  # import core.agent BEFORE opening the circuits (it takes > 1 s)
    _open_tap_circuits(monkeypatch, 2.0)
    assert len(HostBreaker.open_hosts(tb.hosts_for("query_alma_science_archive"))) >= len(TAP_CIRCUITS)
    t0 = time.monotonic()
    out = agent._execute_tool_guarded(_tool(ran), {}, tool_name="query_alma_science_archive", step_label="ALMA query",
                                      on_status=lambda msg, state: statuses.append((msg, state)), timeout_seconds=60.0)
    assert out == {"success": True, "rows": 1}
    assert ran and ran[0] - t0 >= 1.5, "the tool started only after the cooldown"
    assert any("waiting" in m and s == "running" for m, s in statuses)


def test_auto_wait_stops_when_the_turn_is_cancelled(monkeypatch):
    agent, ran = _agent(), []
    _open_tap_circuits(monkeypatch, 20.0)
    turn = tb.TurnCancellation("autowait")
    box = {}

    def run():
        agent._tls.turn_cancellation = turn
        box["out"] = agent._execute_tool_guarded(_tool(ran), {}, tool_name="query_alma_science_archive", timeout_seconds=60.0)

    th = threading.Thread(target=run, daemon=True)
    t0 = time.monotonic()
    th.start()
    time.sleep(0.3)
    turn.cancel("stop pressed")
    th.join(3.0)
    assert not th.is_alive() and time.monotonic() - t0 < 3.0
    assert not ran, "a cancelled turn never starts the tool"
    assert box["out"]["success"] is False and box["out"].get("cancelled") is True


def test_no_wait_when_the_turn_soft_deadline_leaves_too_little_time(monkeypatch):
    agent, ran = _agent(), []
    _open_tap_circuits(monkeypatch, 20.0)
    agent._tls.turn_soft_deadline = time.monotonic() + 25.0  # 25 s left < 20 s wait + 10 s work
    t0 = time.monotonic()
    out = agent._execute_tool_guarded(_tool(ran), {}, tool_name="query_alma_science_archive", timeout_seconds=60.0)
    assert time.monotonic() - t0 < 1.0 and not ran
    assert out["success"] is False and out.get("retry_after_s", 0) > 0


def test_a_successful_sibling_clears_the_shared_host_timeout_streak():
    """Guard CX-11: host:<name> timeout counters are shared by sibling tools;
    a successful call on the same service must reset them, or two
    NON-consecutive timeouts block every sibling for the rest of the turn."""
    agent = _agent()
    slow = types.SimpleNamespace(execute=lambda **kw: (time.sleep(1.0), {"success": True})[1])
    ok = types.SimpleNamespace(execute=lambda **kw: {"success": True})
    out = agent._execute_tool_guarded(slow, {}, tool_name="search_by_position", timeout_seconds=0.2)
    assert out["timeout"] is True
    host_keys = [k for k in agent._tls.tool_timeout_breaker if k.startswith("host:")]
    assert host_keys and all(agent._tls.tool_timeout_breaker[k] == 1 for k in host_keys)
    assert agent._execute_tool_guarded(ok, {}, tool_name="search_by_target", timeout_seconds=5.0) == {"success": True}
    assert not [k for k in agent._tls.tool_timeout_breaker if k.startswith("host:")]
    agent._execute_tool_guarded(slow, {}, tool_name="search_by_position", timeout_seconds=0.2)
    # one timeout since the success: a third sibling still runs
    out = agent._execute_tool_guarded(ok, {}, tool_name="query_alma_science_archive", timeout_seconds=5.0)
    assert out == {"success": True}
