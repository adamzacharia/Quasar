"""WP1 guard, Codex verify round 1 (task-bf4e790-2056) -- three incomplete
fixes, now closed:

1. SSRF by DNS rebinding / env proxy: the link probe resolved and validated
   the host, then ``requests`` resolved it AGAIN (and honoured proxy env
   vars). The probe now connects to the validated address itself.
2. Link-probe workers kept following redirects after the wall-clock cap or a
   turn cancellation: a stop event ends them before their next hop.
3. Conductor ``run_in_executor`` subtask threads did not inherit the turn:
   each now adopts a child of the Conductor thread's deadline, and the tool
   guard falls back to the current deadline's turn when the thread has no
   runner TLS.
"""
from __future__ import annotations

import asyncio
import socket
import threading
import time
import types

from core.agent import QuasarAgent
from services import tool_budgets as tb
from tests.unit.test_conductor_accounting import _make_conductor, _mini_run, _Node


def test_probe_connects_to_the_validated_address_so_rebinding_cannot_redirect_it(monkeypatch):
    answers = iter(["93.184.216.34", "169.254.169.254", "169.254.169.254"])

    def getaddrinfo(host, port, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (next(answers), port))]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    seen = []

    def head_once(url, ip, timeout):
        seen.append((url, ip))
        return 200, None

    monkeypatch.setattr(QuasarAgent, "_head_once", staticmethod(head_once))
    ok, detail = QuasarAgent._head_verify_url("https://rebind.example/page")
    assert ok and detail == "HTTP 200"
    assert seen == [("https://rebind.example/page", "93.184.216.34")], "one resolution, connection pinned to it"


def test_pinned_request_uses_no_environment_proxy_and_sends_the_hostname(monkeypatch):
    import urllib3

    captured = {}

    class _Pool:
        def __init__(self, host, port, **kw):
            captured.update(host=host, port=port, **kw)

        def request(self, method, path, headers=None, redirect=True, preload_content=True):
            captured.update(method=method, path=path, headers=headers, redirect=redirect)
            return types.SimpleNamespace(status=204, headers={}, release_conn=lambda: None)

        def close(self):
            pass

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
    monkeypatch.setattr(urllib3, "HTTPSConnectionPool", _Pool)
    status, loc = QuasarAgent._head_once("https://almascience.org/aq/?source=M83", "193.146.57.10", 5.0)
    assert status == 204 and loc is None
    assert captured["host"] == "193.146.57.10" and captured["server_hostname"] == "almascience.org"
    assert captured["assert_hostname"] == "almascience.org" and captured["headers"]["Host"] == "almascience.org"
    assert captured["method"] == "HEAD" and captured["redirect"] is False and captured["path"] == "/aq/?source=M83"


def test_stopped_probe_does_not_follow_another_redirect(monkeypatch):
    monkeypatch.setattr(QuasarAgent, "_resolve_public_target", staticmethod(lambda url: (True, "", "93.184.216.34")))
    stop = threading.Event()
    hops = []

    def head_once(url, ip, timeout):
        hops.append(url)
        stop.set()  # the cap passes while this hop is in flight
        return 302, "https://next.example/hop"

    monkeypatch.setattr(QuasarAgent, "_head_once", staticmethod(head_once))
    ok, detail = QuasarAgent._head_verify_url("https://start.example/", stop=stop)
    assert not ok and "stopped" in detail and hops == ["https://start.example/"]


def test_verify_sets_the_stop_event_when_it_gives_up(monkeypatch):
    agent = QuasarAgent.__new__(QuasarAgent)
    agent._tls = threading.local()
    monkeypatch.setattr(QuasarAgent, "_LINK_PROBE_TOTAL_S", 0.3)
    stops = []

    def slow_probe(url, timeout=5.0, stop=None):
        stops.append(stop)
        time.sleep(1.0)
        return True, "HTTP 200"

    monkeypatch.setattr(QuasarAgent, "_head_verify_url", staticmethod(slow_probe))
    out = agent._verify_requested_urls(["https://example.org/a"])
    assert out[0][1] is False and stops and stops[0].is_set()


def test_conductor_subtask_thread_inherits_the_turn_deadline():
    seen = {}

    def tool_executor(task, dep_context, model, *extra):
        d = tb.current_deadline()
        seen["deadline"] = d
        seen["turn"] = getattr(d, "turn", None)
        return "done"

    conductor = _make_conductor(tool_executor=tool_executor)
    turn = tb.TurnCancellation("conductor-turn")
    holder = {}

    def conductor_thread():
        tb.adopt_deadline(tb.make_identity_deadline("bg:conductor", turn=turn))  # what core/runner._turn_bound does
        loop = asyncio.new_event_loop()
        try:
            holder["result"] = loop.run_until_complete(conductor._execute_node(_mini_run(), _Node()))
        finally:
            loop.close()
            tb.end_tool_deadline()

    t = threading.Thread(target=conductor_thread)
    t.start()
    t.join(30)
    assert holder["result"] == "done"
    assert seen["turn"] is turn and seen["deadline"] is not None
    turn.cancel("stop pressed")
    assert seen["deadline"].cancelled(), "a Stop reaches the subtask thread's requests"


def test_guard_uses_the_deadline_turn_on_a_thread_without_runner_tls():
    agent = QuasarAgent.__new__(QuasarAgent)
    agent._tls = threading.local()
    agent._tls.tool_timeout_breaker = {}
    turn = tb.TurnCancellation("subtask")
    turn.cancel("client gone")
    ran = []
    box = {}

    def executor_thread():
        tb.adopt_deadline(tb.make_identity_deadline("conductor-subtask", turn=turn))
        try:
            box["out"] = agent._execute_tool_guarded(types.SimpleNamespace(execute=lambda **k: ran.append(1) or {"success": True}),
                                                     {}, tool_name="__probe__", timeout_seconds=5.0)
        finally:
            tb.end_tool_deadline()

    t = threading.Thread(target=executor_thread)
    t.start()
    t.join(10)
    assert not ran and box["out"].get("cancelled") is True
