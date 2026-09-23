"""Detached tool workers die with the turn (services/tool_budgets.py
TurnCancellation / Deadline.cancel, services/http_budget_hook.py, core/agent.py).

UI benchmark 2026-09-22, L06: a turn stopped in the UI at 420 s kept its
abandoned Data Lab workers issuing 70 s ReadTimeouts for three more minutes
and slowed the next questions. Now every network request inside a guarded
tool checks the call's deadline; once the tool guard abandons the worker, the
client disconnects / presses Stop, or the runner hits its hard cap, the
deadline is cancelled and the next request raises TurnCancelled instead of
going out.
"""
from __future__ import annotations

import threading
import time
import types

import pytest
import requests

from services import http_budget_hook as hook
from services import tool_budgets as tb
from services.host_breaker import HostBreaker


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("QUASAR_HTTP_BUDGET_HOOK", raising=False)
    HostBreaker.reset()
    tb.end_tool_deadline()
    hook.reset_for_tests()
    assert hook.install() is True
    yield
    tb.end_tool_deadline()
    HostBreaker.reset()
    hook.reset_for_tests()


def _counting_fake(counter):
    def fake(self, method, url, *a, **kw):
        counter.append(url)
        time.sleep(0.02)
        return type("R", (), {"status_code": 200})()

    return fake


# ── Deadline / TurnCancellation primitives ──────────────────────────────


def test_cancelled_deadline_refuses_the_next_bounded_call():
    d = tb.Deadline(60.0, label="tool")
    assert d.bounded(30.0) == 30.0
    d.cancel("stopped")
    assert d.cancelled()
    with pytest.raises(tb.TurnCancelled) as info:
        d.bounded(30.0)
    assert "stopped" in str(info.value)
    # TurnCancelled is a BudgetExhausted (TimeoutError) so existing partial-
    # result handlers treat it as a bounded stop, not a crash.
    assert isinstance(info.value, tb.BudgetExhausted) and isinstance(info.value, TimeoutError)


def test_turn_cancellation_fans_out_to_every_live_deadline_and_counts_them():
    turn = tb.TurnCancellation(label="conv-x")
    a = tb.make_tool_deadline("tool_a", 100.0, turn=turn)
    b = tb.make_tool_deadline("tool_b", 100.0, turn=turn)
    finished = tb.make_tool_deadline("tool_c", 100.0, turn=turn)
    tb.adopt_deadline(finished)
    tb.end_tool_deadline()  # tool_c completed: released from the turn
    assert turn.live_count() == 2
    n = turn.cancel("client disconnected")
    assert n == 2
    assert a.cancelled() and b.cancelled() and turn.cancelled
    assert a.why_cancelled() == "client disconnected"
    # Idempotent: a second cancel stops nothing new.
    assert turn.cancel("again") == 0


def test_child_deadlines_share_the_call_token_and_inherit_cancellation():
    parent = tb.Deadline(50.0, label="tool")
    child = parent.child(label="helper")
    assert child.token is parent.token, "sub-requests of one tool call are one call for the host breaker"
    assert not child.cancelled()
    parent.cancel("guard gave up")
    assert child.cancelled() and child.why_cancelled() == "guard gave up"
    # ...but cancelling a child never cancels its parent.
    p2 = tb.Deadline(50.0)
    c2 = p2.child()
    c2.cancel("helper abandoned")
    assert not p2.cancelled()


def test_current_call_token_is_none_outside_a_tool_and_stable_inside():
    assert tb.current_call_token() is None
    tb.begin_tool_deadline("t", 30.0)
    tok = tb.current_call_token()
    assert tok is not None and tb.current_call_token() is tok
    tb.end_tool_deadline()
    assert tb.current_call_token() is None


# ── the requests hook refuses new requests after cancellation ───────────


def test_hook_raises_turn_cancelled_before_calling_the_transport(monkeypatch):
    calls = []
    monkeypatch.setattr(hook, "_original_request", _counting_fake(calls))
    d = tb.begin_tool_deadline("some_tool", 60.0)
    requests.Session().request("GET", "https://archive.example/x", timeout=10)
    assert len(calls) == 1
    d.cancel("stop pressed")
    with pytest.raises(tb.TurnCancelled):
        requests.Session().request("GET", "https://archive.example/y", timeout=10)
    assert len(calls) == 1, "no request may leave after cancellation"
    assert not HostBreaker.is_open("archive.example"), "our own stop is never a host failure"



def test_suppressed_and_excluded_requests_are_also_refused_after_cancellation(monkeypatch):
    """Guard CX-04: integrations that clamp their own timeout run inside
    `suppressed()`, and platform hosts are excluded from budgeting -- neither
    may start a request from a cancelled worker. CX-05: the excluded branch
    uses the defensive original, so an uninstall race never calls None."""
    calls = []
    monkeypatch.setattr(hook, "_original_request", _counting_fake(calls))
    monkeypatch.setenv("QUASAR_HTTP_BUDGET_HOOK_EXCLUDE", "platform.example")
    d = tb.begin_tool_deadline("some_tool", 60.0)
    with hook.suppressed():
        requests.Session().request("GET", "https://archive.example/a", timeout=10)
    requests.Session().request("GET", "https://platform.example/b", timeout=10)
    assert len(calls) == 2
    d.cancel("client disconnected")
    with pytest.raises(tb.TurnCancelled):
        with hook.suppressed():
            requests.Session().request("GET", "https://archive.example/c", timeout=10)
    with pytest.raises(tb.TurnCancelled):
        requests.Session().request("GET", "https://platform.example/d", timeout=10)
    assert len(calls) == 2
    tb.end_tool_deadline()

    # uninstall race: the wrapper is still in place but the captured original is gone
    tb.begin_tool_deadline("some_tool", 60.0)
    pristine = []
    monkeypatch.setattr(hook, "_original_request", None)
    monkeypatch.setattr(hook, "_pristine_request", lambda: _counting_fake(pristine))
    hook._hooked_request(requests.Session(), "GET", "https://platform.example/e", timeout=5)
    assert pristine == ["https://platform.example/e"]

def test_call_bounded_helper_stops_when_the_parent_is_cancelled(monkeypatch):
    calls = []
    monkeypatch.setattr(hook, "_original_request", _counting_fake(calls))
    parent = tb.begin_tool_deadline("some_tool", 60.0)
    stop = threading.Event()

    def helper():
        while not stop.is_set():
            requests.Session().request("GET", "https://archive.example/loop", timeout=5)
        return "done"

    def runner_thread():
        # call_bounded reads the deadline of ITS thread (thread-local), as a
        # tool worker would: adopt the tool deadline first.
        tb.adopt_deadline(parent)
        _swallow(lambda: tb.call_bounded(helper, 10.0, label="helper"))

    t = threading.Thread(target=runner_thread, daemon=True)
    t.start()
    time.sleep(0.15)
    before = len(calls)
    assert before > 0
    parent.cancel("turn over")
    time.sleep(0.2)
    after = len(calls)
    stop.set()
    t.join(2.0)
    assert after - before <= 1, f"helper kept requesting after cancellation: {before} -> {after}"


def _swallow(fn):
    try:
        return fn()
    except Exception:
        return None


# ── the tool guard cancels an abandoned worker; cancel_response_run cancels the turn ──


def _bare_agent():
    from core.agent import QuasarAgent

    agent = QuasarAgent.__new__(QuasarAgent)
    agent._tls = threading.local()
    agent._tls.tool_timeout_breaker = {}
    return agent


def _looping_tool(calls, stop):
    def execute(**kw):
        n = 0
        while not stop.is_set() and n < 1000:
            requests.Session().request("GET", f"https://archive.example/item/{n}", timeout=5)
            n += 1
        return {"success": True, "n": n}

    return types.SimpleNamespace(execute=execute)


def test_guard_timeout_cancels_the_abandoned_worker(monkeypatch):
    """After the guard gives up on a worker, the worker's deadline is
    cancelled (on top of being expired), so even a request that bypassed the
    budget clamp would be refused, and the cancel reason names the guard."""
    monkeypatch.setattr(tb, "GUARD_HEADROOM_SECONDS", 0.0)
    agent = _bare_agent()
    release = threading.Event()
    seen = {}

    def execute(**kw):
        seen["deadline"] = tb.current_deadline()
        release.wait(3.0)
        return {"success": True}

    try:
        out = agent._execute_tool_guarded(types.SimpleNamespace(execute=execute), {}, tool_name="__slow_probe__", timeout_seconds=0.3)
        assert out["timeout"] is True
        time.sleep(0.05)
        d = seen["deadline"]
        assert d is not None and d.cancelled(), "abandoned worker's deadline must be cancelled"
        assert "tool guard" in d.why_cancelled()
        with pytest.raises(tb.TurnCancelled):
            d.bounded(5.0, minimum=0.0)
    finally:
        release.set()


def test_cancel_response_run_stops_live_workers_of_that_turn(monkeypatch):
    calls = []
    monkeypatch.setattr(hook, "_original_request", _counting_fake(calls))
    agent = _bare_agent()
    agent._conv_ids_lock = threading.Lock()
    agent._conv_run_tokens = {}
    agent._conv_response_ids = {}
    agent.config = types.SimpleNamespace(model="gpt-oss-120b")
    agent.client = types.SimpleNamespace(responses=types.SimpleNamespace(clear_history=lambda *_: None))
    turn = tb.TurnCancellation(label="conv-1")
    agent._tls.turn_cancellation = turn
    agent._begin_response_run("conv-1", "gpt-oss-120b", "run-1", turn_cancellation=turn)
    stop = threading.Event()
    box = {}

    def run():
        # The runner thread sets the turn token on ITS thread-local state and
        # calls the guard on that same thread -- mirror that here.
        agent._tls.turn_cancellation = turn
        # 60 s guard -> 45 s inner budget: plenty left when the turn is cancelled.
        box["out"] = agent._execute_tool_guarded(_looping_tool(calls, stop), {}, tool_name="__slow_probe__", timeout_seconds=60.0)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(0.15)
    assert len(calls) > 0 and turn.live_count() == 1
    agent.cancel_response_run("conv-1", "gpt-oss-120b", "run-1")
    t.join(3.0)
    assert not t.is_alive(), "the worker must return promptly once cancelled"
    settled = len(calls)
    time.sleep(0.2)
    assert len(calls) == settled
    out = box["out"]
    assert out["success"] is False and out.get("cancelled") is True
    stop.set()


def test_guard_refuses_to_start_a_tool_on_an_already_cancelled_turn():
    agent = _bare_agent()
    turn = tb.TurnCancellation(label="conv-2")
    turn.cancel("client gone")
    agent._tls.turn_cancellation = turn
    ran = {"n": 0}

    def execute(**kw):
        ran["n"] += 1
        return {"success": True}

    out = agent._execute_tool_guarded(types.SimpleNamespace(execute=execute), {}, tool_name="__probe__", timeout_seconds=5.0)
    assert ran["n"] == 0 and out.get("cancelled") is True


def test_a_stale_run_token_does_not_cancel_the_current_turn():
    agent = _bare_agent()
    agent._conv_ids_lock = threading.Lock()
    agent._conv_run_tokens = {}
    agent._conv_response_ids = {}
    agent.config = types.SimpleNamespace(model="m")
    agent.client = types.SimpleNamespace(responses=types.SimpleNamespace(clear_history=lambda *_: None))
    old = tb.TurnCancellation("old")
    agent._begin_response_run("c", "m", "run-old", turn_cancellation=old)
    new = tb.TurnCancellation("new")
    agent._begin_response_run("c", "m", "run-new", turn_cancellation=new)
    agent.cancel_response_run("c", "m", "run-old")  # stale: must be a no-op
    assert not new.cancelled and not old.cancelled
    agent.cancel_response_run("c", "m", "run-new")
    assert new.cancelled


def test_guard_disabled_inline_path_keeps_call_identity_and_turn_cancellation(monkeypatch):
    """Guard CX-06: with the timeout guard disabled (budget None) the tool runs
    inline, but still under an unbounded deadline that carries the call token
    (breaker soft-failure dedup) and the turn cancellation."""
    calls = []
    monkeypatch.setattr(hook, "_original_request", _counting_fake(calls))
    agent = _bare_agent()
    agent._tool_timeout_seconds = lambda name: None
    turn = tb.TurnCancellation(label="conv-inline")
    agent._tls.turn_cancellation = turn
    seen = {}

    def execute(**kw):
        seen["token"] = tb.current_call_token()
        d = tb.current_deadline()
        seen["timeout"] = d.bounded(25.0)  # no time bound: the caller's own timeout stands
        requests.Session().request("GET", "https://archive.example/1", timeout=10)
        turn.cancel("stop pressed")
        requests.Session().request("GET", "https://archive.example/2", timeout=10)
        return {"success": True}

    out = agent._execute_tool_guarded(types.SimpleNamespace(execute=execute), {}, tool_name="__inline_probe__")
    assert seen["token"] is not None and seen["timeout"] == 25.0
    assert calls == ["https://archive.example/1"]
    assert out["success"] is False and out.get("cancelled") is True
    assert tb.current_deadline() is None and turn.live_count() == 0


def _registry_agent():
    agent = _bare_agent()
    agent._conv_ids_lock = threading.Lock()
    agent._conv_run_tokens = {}
    agent._conv_response_ids = {}
    agent.config = types.SimpleNamespace(model="gpt-oss-120b")
    agent.client = types.SimpleNamespace(responses=types.SimpleNamespace(clear_history=lambda *_: None))
    return agent


def test_cancel_with_the_requested_model_reaches_a_failover_run():
    """Guard CX-08: the runner registers the run under the failover model; the
    SSE layer cancels with the model the client requested."""
    agent = _registry_agent()
    turn = tb.TurnCancellation("failover")
    agent._begin_response_run("conv-f", "deepseek-chat", "run-f", turn_cancellation=turn)  # failover-selected
    agent.cancel_response_run("conv-f", "gpt-oss-120b", "run-f")  # requested model
    assert turn.cancelled
    assert agent._conv_run_tokens.get(agent._response_state_key("conv-f", "deepseek-chat")) is None


def test_finished_turns_release_their_token_and_cleanup_prunes_dead_ones():
    """Guard CX-12: completed turns no longer accumulate in the registry."""
    agent = _registry_agent()
    agent._begin_response_run("conv-a", "gpt-oss-120b", "run-a", turn_cancellation=tb.TurnCancellation("a"))
    agent._end_response_run("run-a")
    assert "run-a" not in agent._turn_cancellations
    agent._begin_response_run("conv-b", "gpt-oss-120b", "run-b", turn_cancellation=tb.TurnCancellation("b"))
    agent._conv_run_tokens.clear()  # the run is no longer live (e.g. evicted)
    agent._cleanup_conv_states()
    assert agent._turn_cancellations == {}


def test_background_threads_are_bound_to_the_turn(monkeypatch):
    """Guard CX-07: background web / email / RAG / Conductor threads run under
    an identity deadline registered on the turn token."""
    from core.runner import _turn_bound

    calls = []
    monkeypatch.setattr(hook, "_original_request", _counting_fake(calls))
    agent = _bare_agent()
    turn = tb.TurnCancellation("bg")
    started, go_on = threading.Event(), threading.Event()
    seen = {}

    def target():
        seen["turn_on_tls"] = agent._tls.turn_cancellation is turn
        requests.Session().request("GET", "https://search.example/1", timeout=5)
        started.set()
        go_on.wait(2)
        requests.Session().request("GET", "https://search.example/2", timeout=5)
        seen["unreachable"] = True

    t = threading.Thread(target=_turn_bound(target, "web", turn, agent), daemon=True)
    t.start()
    assert started.wait(2)
    assert turn.live_count() == 1
    turn.cancel("turn finished")
    go_on.set()
    t.join(2)
    assert not t.is_alive() and seen["turn_on_tls"] and "unreachable" not in seen
    assert calls == ["https://search.example/1"] and turn.live_count() == 0


def test_runner_wrapper_finishes_the_turn_and_clears_thread_state(monkeypatch):
    import core.runner as runner

    agent = _registry_agent()
    turn = tb.TurnCancellation("wrapped")

    def fake_impl(agent_, query, **kw):
        agent_._tls.turn_cancellation = turn
        agent_._tls.turn_soft_deadline = time.monotonic() + 5
        agent_._begin_response_run("conv-w", "gpt-oss-120b", kw["run_token"], turn_cancellation=turn)
        return "answer"

    monkeypatch.setattr(runner, "_stream_response_api_impl", fake_impl)
    assert runner.stream_response_api(agent, "q", run_token="run-w") == "answer"
    assert turn.cancelled and turn.reason == "turn finished"
    assert "run-w" not in agent._turn_cancellations
    assert agent._tls.turn_cancellation is None and agent._tls.turn_soft_deadline is None


def test_vo_deadline_worker_inherits_a_child_deadline_and_is_cancelled_when_abandoned(monkeypatch):
    """Guard CX-09: services/vo_registry._run_with_deadline runs fn on a fresh
    executor thread; it must carry a child of the caller's tool deadline."""
    from services import vo_registry

    calls = []
    monkeypatch.setattr(hook, "_original_request", _counting_fake(calls))
    parent = tb.begin_tool_deadline("vo_query", 60.0)
    seen = {}
    release = threading.Event()

    def fn():
        d = tb.current_deadline()
        seen["token_shared"] = d is not None and d.token is parent.token and d is not parent
        requests.Session().request("GET", "https://vo.example/1", timeout=5)
        release.wait(2)
        try:
            requests.Session().request("GET", "https://vo.example/2", timeout=5)
        except tb.TurnCancelled:
            seen["refused"] = True
        return "late"

    with pytest.raises(vo_registry._DeadlineExceeded):
        vo_registry._run_with_deadline(fn, 0.3, "slow TAP_SCHEMA scan")
    release.set()
    time.sleep(0.2)
    assert seen["token_shared"] and seen.get("refused") is True
    assert calls == ["https://vo.example/1"] and not parent.cancelled()
    tb.end_tool_deadline()
