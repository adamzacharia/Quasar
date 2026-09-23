"""services/http_budget_hook.py — the requests-layer safety net.

Every registered network tool ends in ``requests.Session.request`` (pyvo,
astroquery, sparclclient, tavily, plain requests). Inside a guarded tool the
hook clamps the timeout to the tool's remaining budget and routes the call
through the host breaker; outside a tool it is inert.
"""
from __future__ import annotations

import socket
import threading
import time

import pytest
import requests

from services import http_budget_hook as hook
from services.host_breaker import HostBreaker, HostCircuitOpen
from services.tool_budgets import BudgetExhausted, begin_tool_deadline, end_tool_deadline


@pytest.fixture(autouse=True)
def _installed(monkeypatch):
    monkeypatch.delenv("QUASAR_HTTP_BUDGET_HOOK", raising=False)
    HostBreaker.reset()
    end_tool_deadline()
    # Whole-suite ordering: an earlier test may have left the hook installed
    # (real agent construction) or a fake over Session.request — start clean.
    hook.reset_for_tests()
    assert hook.install() is True
    yield
    end_tool_deadline()
    HostBreaker.reset()
    hook.reset_for_tests()


class _SilentServer:
    """Accepts TCP connections and never answers — a hung archive."""

    def __init__(self):
        self.sock = socket.socket()
        # Bound on all interfaces and addressed via the 127.0.0.2 loopback alias:
        # 127.0.0.1/localhost are on the hook's platform-host exclusion list.
        self.sock.bind(("0.0.0.0", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self._stop = threading.Event()
        self._conns = []
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        self.sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
                self._conns.append(conn)
            except socket.timeout:
                continue
            except OSError:
                break

    def close(self):
        self._stop.set()
        for c in self._conns:
            try:
                c.close()
            except OSError:
                pass
        self.sock.close()


@pytest.fixture
def silent_server():
    srv = _SilentServer()
    yield srv
    srv.close()


def test_install_is_idempotent_and_uninstall_restores():
    assert hook.installed()
    assert hook.install() is True
    hook.uninstall()
    assert not hook.installed()
    assert requests.Session.request is hook._pristine_request()
    hook.install()


def test_uninstall_never_reinstalls_a_captured_fake(monkeypatch):
    # A test patches Session.request, installs the hook (capturing the fake),
    # its monkeypatch is torn down FIRST, then uninstall runs: the genuine
    # method must survive, not the fake.
    hook.reset_for_tests()
    fake = lambda self, m, u, *a, **k: None  # noqa: E731
    monkeypatch.setattr(requests.Session, "request", fake)
    hook.install()
    monkeypatch.undo()  # teardown before uninstall
    hook.uninstall()
    assert requests.Session.request is hook._pristine_request()


def test_outside_a_tool_the_timeout_is_untouched(monkeypatch):
    seen = {}

    def fake(self, method, url, *a, **kw):
        seen["timeout"] = kw.get("timeout", "absent")
        return type("R", (), {"status_code": 200})()

    monkeypatch.setattr(hook, "_original_request", fake)
    requests.Session().request("GET", "https://archive.example/x", timeout=600)
    assert seen["timeout"] == 600
    requests.Session().request("GET", "https://archive.example/x")
    assert seen["timeout"] == "absent"


def test_inside_a_tool_a_600s_timeout_is_clamped_to_the_remaining_budget(monkeypatch):
    seen = {}

    def fake(self, method, url, *a, **kw):
        seen["timeout"] = kw.get("timeout")
        return type("R", (), {"status_code": 200})()

    monkeypatch.setattr(hook, "_original_request", fake)
    begin_tool_deadline("some_tool", 20.0 + 15.0)  # inner budget 20 s
    requests.Session().request("GET", "https://archive.example/x", timeout=600)
    assert 18.0 < seen["timeout"] <= 20.0
    # A missing timeout (pyvo without a session, SkyView) becomes the budget.
    requests.Session().request("GET", "https://archive.example/x")
    assert 17.0 < seen["timeout"] <= 20.0
    # Tuple (connect, read) timeouts are clamped element-wise.
    requests.Session().request("GET", "https://archive.example/x", timeout=(15, 600))
    assert seen["timeout"][0] <= 15 and seen["timeout"][1] <= 20.0


def test_spent_budget_refuses_the_call_immediately(monkeypatch):
    calls = []
    monkeypatch.setattr(hook, "_original_request", lambda *a, **k: calls.append(1))
    begin_tool_deadline("some_tool", 0.3 + 15.0)  # inner 0.3 s
    time.sleep(0.35)
    t0 = time.perf_counter()
    with pytest.raises(BudgetExhausted):
        requests.Session().request("GET", "https://archive.example/x", timeout=600)
    assert time.perf_counter() - t0 < 0.05
    assert calls == []


def test_hung_archive_is_bounded_by_the_tool_deadline_not_the_client_timeout(silent_server, monkeypatch):
    # The client asks for a 600 s read timeout; the tool has 2 s left.
    begin_tool_deadline("some_tool", 2.0 + 15.0)
    t0 = time.perf_counter()
    with pytest.raises((requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError)):
        requests.get(f"http://127.0.0.2:{silent_server.port}/hips2fits", timeout=600)
    elapsed = time.perf_counter() - t0
    assert elapsed < 4.0, f"clamped read should time out in ~2 s, took {elapsed:.1f}s"
    # A read timeout is a SOFT failure (a heavy query on a healthy host looks
    # the same): one does not open the host...
    assert not HostBreaker.is_open("127.0.0.2")
    # ...three from DISTINCT tool calls spread over >= 20 s do, and then the
    # next call fails in microseconds. (Fake clock: no sleeping.)
    from services import host_breaker as hb
    from services.tool_budgets import begin_tool_deadline as _begin

    t = {"now": hb._now() + 30.0}
    monkeypatch.setattr(hb, "_now", lambda: t["now"])
    for _ in range(2):
        end_tool_deadline()
        _begin("another_tool", 20.0)
        HostBreaker.record_failure("http://127.0.0.2/hips2fits", requests.exceptions.ReadTimeout("read timed out"))
        t["now"] += 30.0
    assert HostBreaker.is_open("http://127.0.0.2/hips2fits")
    t1 = time.perf_counter()
    with pytest.raises(HostCircuitOpen):
        # Same service path from another tool: refused in microseconds.
        requests.get(f"http://127.0.0.2:{silent_server.port}/hips2fits?other-tool=1", timeout=600)
    assert time.perf_counter() - t1 < 0.05
    # A DIFFERENT service on the same host is not affected by a soft trip.
    assert not HostBreaker.is_open(f"http://127.0.0.2:{silent_server.port}/another-service/x")


def test_platform_hosts_are_never_clamped_or_tripped(monkeypatch):
    seen = {}

    def fake(self, method, url, *a, **kw):
        seen["timeout"] = kw.get("timeout")
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(hook, "_original_request", fake)
    begin_tool_deadline("some_tool", 5.0 + 15.0)
    with pytest.raises(requests.exceptions.ConnectionError):
        requests.Session().request("POST", "https://api.openai.com/v1/responses", timeout=600)
    assert seen["timeout"] == 600, "LLM providers keep their own timeout policy"
    assert not HostBreaker.is_open("api.openai.com")


def test_explicitly_wired_paths_are_not_double_counted(monkeypatch):
    def fake(self, method, url, *a, **kw):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(hook, "_original_request", fake)
    monkeypatch.setenv("HOST_BREAKER_TRIP_FAILURES", "2")
    begin_tool_deadline("some_tool", 5.0 + 15.0)
    with hook.suppressed():
        with pytest.raises(requests.exceptions.ConnectionError):
            requests.Session().request("GET", "https://archive.example/x", timeout=10)
    # The suppressed call recorded nothing; the wired wrapper records once itself.
    assert not HostBreaker.is_open("archive.example")


def test_env_switch_disables_the_hook(monkeypatch):
    hook.uninstall()
    monkeypatch.setenv("QUASAR_HTTP_BUDGET_HOOK", "0")
    assert hook.install() is False
    assert not hook.installed()
