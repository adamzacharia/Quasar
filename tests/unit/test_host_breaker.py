"""Host-keyed circuit breaker (services/host_breaker.py).

Live 2026-09-20: CDS (alasky.cds.unistra.fr) was down — SSLError / ReadTimeout
— and every imaging tool in a turn paid its full timeout again, because the
only breaker counted *guard timeouts per tool name*, and the model dodged
that by alternating datalab_color_image and hips_cutout. The host breaker
keys on the remote HOST, so the second call against a dead host — from any
tool — fails in milliseconds with a structured error.
"""
from __future__ import annotations

import socket
import ssl
import time

import pytest
import requests

from services import host_breaker as hb
from services.host_breaker import (
    HostBreaker,
    HostCircuitOpen,
    circuit_of,
    classify_infrastructure_error,
    guarded_request,
    host_of,
    service_path_of,
    structured_error,
)
from services.tool_budgets import begin_tool_deadline, end_tool_deadline


@pytest.fixture(autouse=True)
def _clean_breaker(monkeypatch):
    monkeypatch.delenv("HOST_BREAKER_DISABLED", raising=False)
    monkeypatch.delenv("HOST_BREAKER_COOLDOWN_SECONDS", raising=False)
    monkeypatch.delenv("HOST_BREAKER_TRIP_FAILURES", raising=False)
    monkeypatch.delenv("HOST_BREAKER_SOFT_SPREAD_SECONDS", raising=False)
    monkeypatch.delenv("HOST_BREAKER_AUTO_WAIT_SECONDS", raising=False)
    HostBreaker.reset()
    end_tool_deadline()
    yield
    end_tool_deadline()
    HostBreaker.reset()


class _Clock:
    """Controllable monotonic clock for the breaker (no sleeping in tests)."""

    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += float(seconds)


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(hb, "_now", c)
    return c


def _soft_failures_from_distinct_calls(url: str, n: int, clock: _Clock, gap: float = 25.0, *, status: int = 502):
    """``n`` soft failures, each from its OWN tool call, ``gap`` seconds apart."""
    for i in range(n):
        begin_tool_deadline(f"call-{i}", 100.0)
        try:
            HostBreaker.record_failure(url, status=status)
        finally:
            end_tool_deadline()
        if i < n - 1:
            clock.advance(gap)


# ── host normalisation ───────────────────────────────────────────────────


def test_host_of_accepts_urls_and_bare_hosts():
    assert host_of("https://alasky.cds.unistra.fr/hips-image-services/hips2fits?x=1") == "alasky.cds.unistra.fr"
    assert host_of("almascience.nrao.edu:443/tap") == "almascience.nrao.edu"
    assert host_of("HTTPS://Datalab.NOIRLab.edu/tap") == "datalab.noirlab.edu"
    assert host_of("") == ""


# ── classification: infrastructure vs request-level ─────────────────────


@pytest.mark.parametrize(
    "exc",
    [
        requests.exceptions.SSLError("HTTPSConnectionPool: Max retries exceeded with url (SSLError)"),
        requests.exceptions.ReadTimeout("HTTPSConnectionPool(host='x', port=443): Read timed out."),
        requests.exceptions.ConnectTimeout("connect timeout"),
        requests.exceptions.ConnectionError("Failed to establish a new connection: [Errno 111] Connection refused"),
        ssl.SSLError("EOF occurred in violation of protocol"),
        socket.gaierror(11001, "getaddrinfo failed"),
        ConnectionRefusedError(),
        TimeoutError("The read operation timed out"),
        # pyvo wraps the HTTP error; our TAP facade wraps pyvo again.
        RuntimeError(
            "ALMA TAP query failed: https://almascience.nrao.edu/tap: DALServiceError: "
            "502 Server Error: Proxy Error for url: https://almascience.nrao.edu/tap/sync/x/run"
        ),
    ],
)
def test_infrastructure_errors_are_classified(exc):
    assert classify_infrastructure_error(exc) is not None


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("radius must be positive"),
        ValueError("timeout must be positive"),  # a parameter, not the network
        KeyError("s_ra"),
        RuntimeError("ALMA TAP query failed: 400 Client Error: Bad Request (ADQL syntax error)"),
        RuntimeError("ALMA TAP query failed: 500 Server Error: Internal Server Error"),
    ],
)
def test_request_level_errors_are_not_infrastructure(exc):
    assert classify_infrastructure_error(exc) is None


def test_gateway_statuses_count_but_500_does_not():
    assert classify_infrastructure_error(None, status=502) == "HTTP 502"
    assert classify_infrastructure_error(None, status=503) == "HTTP 503"
    assert classify_infrastructure_error(None, status=504) == "HTTP 504"
    assert classify_infrastructure_error(None, status=500) is None
    assert classify_infrastructure_error(None, status=404) is None


def test_wrapped_cause_chain_is_inspected():
    inner = requests.exceptions.ReadTimeout("read timed out")
    try:
        try:
            raise inner
        except requests.exceptions.ReadTimeout as e:
            raise RuntimeError("hips2fits request failed") from e
    except RuntimeError as outer:
        assert classify_infrastructure_error(outer) == "ReadTimeout"


# ── trip / fast-fail / sibling sharing ───────────────────────────────────


def test_one_infrastructure_failure_opens_the_host_for_every_caller():
    url_a = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits?a=1"
    url_b = "https://alasky.cds.unistra.fr/other/service"
    assert not HostBreaker.is_open(url_a)
    why = HostBreaker.record_failure(url_a, requests.exceptions.SSLError("Max retries exceeded"))
    assert why == "SSLError"
    # datalab_color_image and hips_cutout hit the same host: BOTH are refused.
    assert HostBreaker.is_open(url_a) and HostBreaker.is_open(url_b)
    t0 = time.perf_counter()
    with pytest.raises(HostCircuitOpen) as info:
        HostBreaker.check(url_b)
    assert (time.perf_counter() - t0) < 0.05, "an open breaker must refuse in milliseconds"
    assert info.value.host == "alasky.cds.unistra.fr"
    assert info.value.retry_after > 0


def test_request_level_failure_never_trips():
    assert HostBreaker.record_failure("https://datalab.noirlab.edu/tap", ValueError("bad ADQL")) is None
    assert not HostBreaker.is_open("datalab.noirlab.edu")


def test_read_timeout_is_soft_and_needs_three_distinct_calls_spread_out(clock):
    # One heavy query timing out is not an outage (live 2026-09-22: a single
    # Data Lab aggregate ReadTimeout used to open the host for 120 s).
    rt = requests.exceptions.ReadTimeout("HTTPSConnectionPool(host='datalab.noirlab.edu'): Read timed out.")
    assert HostBreaker.record_failure("datalab.noirlab.edu", rt) == "ReadTimeout"
    assert not HostBreaker.is_open("datalab.noirlab.edu")
    clock.advance(25)
    HostBreaker.record_failure("datalab.noirlab.edu", rt)
    assert not HostBreaker.is_open("datalab.noirlab.edu")
    clock.advance(25)
    HostBreaker.record_failure("datalab.noirlab.edu", rt)
    assert HostBreaker.is_open("datalab.noirlab.edu"), "third read timeout from a third call, 50 s apart, opens the circuit"


def test_success_between_soft_failures_resets_the_streak(clock):
    rt = requests.exceptions.ReadTimeout("read timed out")
    HostBreaker.record_failure("h.example", rt)
    clock.advance(25)
    HostBreaker.record_failure("h.example", rt)
    HostBreaker.record_success("h.example")
    clock.advance(25)
    HostBreaker.record_failure("h.example", rt)
    assert not HostBreaker.is_open("h.example")


# ── 2026-09-22 semantics: per tool call, spread over time, per service path ──


def test_a_burst_of_soft_failures_inside_one_tool_call_counts_once(clock):
    """UI benchmark L07: three SIA cutout 502s within 3 s inside ONE
    datalab_density_vetting call opened datalab.noirlab.edu for 120 s."""
    begin_tool_deadline("datalab_density_vetting", 300.0)
    try:
        for _ in range(3):
            HostBreaker.record_failure("https://datalab.noirlab.edu/sia/coadd_all?POS=1,2", status=502)
            clock.advance(1.0)
    finally:
        end_tool_deadline()
    assert not HostBreaker.is_open("https://datalab.noirlab.edu/sia/coadd_all")
    assert not HostBreaker.is_open("datalab.noirlab.edu")


def test_three_distinct_calls_within_the_spread_window_do_not_trip(clock):
    _soft_failures_from_distinct_calls("https://datalab.noirlab.edu/sia/coadd_all", 3, clock, gap=1.0)
    assert not HostBreaker.is_open("https://datalab.noirlab.edu/sia/coadd_all"), "3 s apart is one incident, not an outage"


def test_three_distinct_calls_at_least_20s_apart_trip_the_service_circuit(clock):
    _soft_failures_from_distinct_calls("https://datalab.noirlab.edu/sia/coadd_all", 3, clock, gap=20.0)
    assert HostBreaker.is_open("https://datalab.noirlab.edu/sia/coadd_all")
    rec = HostBreaker.state("https://datalab.noirlab.edu/sia/coadd_all?x=1")
    assert rec is not None and rec["circuit"] == "datalab.noirlab.edu/sia" and rec["severity"] == "soft"


def test_sia_trip_leaves_tap_and_query_closed_on_the_same_host(clock):
    _soft_failures_from_distinct_calls("https://datalab.noirlab.edu/sia/coadd_all", 3, clock, gap=25.0)
    assert HostBreaker.is_open("https://datalab.noirlab.edu/sia/nsc_dr2")
    assert not HostBreaker.is_open("https://datalab.noirlab.edu/query/query?sql=select+1"), "SQL service unaffected"
    assert not HostBreaker.is_open("https://datalab.noirlab.edu/tap/sync")
    assert not HostBreaker.is_open("datalab.noirlab.edu"), "no host-level trip for soft failures"
    with pytest.raises(HostCircuitOpen) as info:
        HostBreaker.check("https://datalab.noirlab.edu/sia/coadd_all")
    assert info.value.host == "datalab.noirlab.edu/sia" and info.value.hostname == "datalab.noirlab.edu"
    HostBreaker.check("https://datalab.noirlab.edu/query/query")  # no raise
    # open_hosts with per-service declarations (services/tool_budgets.py)
    opened = HostBreaker.open_hosts(["datalab.noirlab.edu/query", "datalab.noirlab.edu/sia"])
    assert [c for c, _ in opened] == ["datalab.noirlab.edu/sia"]


def test_hard_failure_opens_the_whole_host_including_every_service():
    HostBreaker.record_failure("https://datalab.noirlab.edu/sia/coadd_all", requests.exceptions.ConnectionError("Connection refused"))
    assert HostBreaker.is_open("https://datalab.noirlab.edu/query/query")
    assert HostBreaker.is_open("datalab.noirlab.edu")
    with pytest.raises(HostCircuitOpen) as info:
        HostBreaker.check("https://datalab.noirlab.edu/tap/sync")
    assert info.value.host == "datalab.noirlab.edu"


def test_success_on_one_service_does_not_close_another_services_circuit(clock):
    _soft_failures_from_distinct_calls("https://datalab.noirlab.edu/sia/coadd_all", 3, clock, gap=25.0)
    HostBreaker.record_success("https://datalab.noirlab.edu/query/query")
    assert HostBreaker.is_open("https://datalab.noirlab.edu/sia/coadd_all"), "a healthy SQL service says nothing about SIA"
    HostBreaker.record_success("https://datalab.noirlab.edu/sia/coadd_all")
    assert not HostBreaker.is_open("https://datalab.noirlab.edu/sia/coadd_all")


def test_soft_failures_older_than_the_window_are_forgotten(clock, monkeypatch):
    monkeypatch.setenv("HOST_BREAKER_SOFT_WINDOW_SECONDS", "100")
    _soft_failures_from_distinct_calls("https://h.example/svc", 2, clock, gap=30.0)
    clock.advance(200)  # both are now stale
    begin_tool_deadline("late", 100.0)
    HostBreaker.record_failure("https://h.example/svc", status=503)
    end_tool_deadline()
    assert not HostBreaker.is_open("https://h.example/svc")


def test_spread_threshold_is_configurable(clock, monkeypatch):
    monkeypatch.setenv("HOST_BREAKER_SOFT_SPREAD_SECONDS", "0")
    _soft_failures_from_distinct_calls("https://h.example/svc", 3, clock, gap=0.5)
    assert HostBreaker.is_open("https://h.example/svc")


def test_own_budget_stops_never_count_as_failures():
    from services.tool_budgets import BudgetExhausted, TurnCancelled

    assert classify_infrastructure_error(BudgetExhausted("x", 0.1, 1.0)) is None
    assert classify_infrastructure_error(TurnCancelled("x", "stop")) is None
    assert HostBreaker.record_failure("h.example", TurnCancelled("x", "stop")) is None
    assert not HostBreaker.is_open("h.example")


def test_service_path_and_circuit_helpers():
    assert service_path_of("https://datalab.noirlab.edu/sia/coadd_all?POS=1") == "sia"
    assert service_path_of("https://datalab.noirlab.edu/query/query") == "query"
    assert service_path_of("datalab.noirlab.edu/tap") == "tap"
    assert service_path_of("datalab.noirlab.edu") == ""
    assert circuit_of("https://almascience.nrao.edu/tap/sync") == "almascience.nrao.edu/tap"
    assert circuit_of("mast.stsci.edu") == "mast.stsci.edu"
    assert host_of("datalab.noirlab.edu/sia") == "datalab.noirlab.edu"


def test_skip_result_carries_retry_after_s_and_the_wait_hint_for_short_cooldowns():
    short = structured_error(HostCircuitOpen("datalab.noirlab.edu/sia", 20.4, "HTTP 502"), tool_name="datalab_image_cutout")
    assert short["retry_after_s"] == 20 and short["retry_after_seconds"] == 20
    assert short["circuit"] == "datalab.noirlab.edu/sia" and short["hostname"] == "datalab.noirlab.edu"
    assert "retry_after_s=20" in short["error"] and "MAY retry this exact call ONCE" in short["error"]
    assert short["skipped_at_epoch"] > 0
    long = structured_error(HostCircuitOpen("datalab.noirlab.edu", 90.0, "ConnectionError"), tool_name="datalab_cone_count")
    assert long["retry_after_s"] == 90
    assert "Do NOT retry" in long["error"] and "MAY retry" not in long["error"]


def test_connection_class_failures_are_hard_and_trip_at_once():
    for exc in (requests.exceptions.ConnectTimeout("x"), requests.exceptions.SSLError("x"),
                requests.exceptions.ConnectionError("Connection refused"), socket.gaierror(11001, "getaddrinfo failed")):
        HostBreaker.reset()
        HostBreaker.record_failure("h.example", exc)
        assert HostBreaker.is_open("h.example"), type(exc).__name__


def test_gateway_statuses_are_soft(clock):
    for _ in range(2):
        HostBreaker.record_failure("gw.example", status=503)
        clock.advance(25)
    assert not HostBreaker.is_open("gw.example")
    HostBreaker.record_failure("gw.example", status=502)
    assert HostBreaker.is_open("gw.example")


def test_success_closes_and_resets():
    HostBreaker.record_failure("h.example", requests.exceptions.ConnectTimeout("x"))
    assert HostBreaker.is_open("h.example")
    HostBreaker.record_success("https://h.example/anything")
    assert not HostBreaker.is_open("h.example")


def test_cooldown_expires_then_reopens_longer_on_failing_probe(monkeypatch):
    monkeypatch.setenv("HOST_BREAKER_COOLDOWN_SECONDS", "0.05")
    HostBreaker.record_failure("h.example", requests.exceptions.ConnectTimeout("x"))
    first = HostBreaker.state("h.example")
    assert first is not None and first["trips"] == 1
    time.sleep(0.08)
    assert not HostBreaker.is_open("h.example"), "cooldown over: the next call is a live probe"
    HostBreaker.record_failure("h.example", requests.exceptions.ConnectTimeout("still down"))
    second = HostBreaker.state("h.example")
    assert second is not None and second["trips"] == 2
    assert second["cooldown"] > first["cooldown"], "a failing probe re-opens with a longer cooldown"


def test_trip_threshold_env(monkeypatch):
    monkeypatch.setenv("HOST_BREAKER_TRIP_FAILURES", "2")
    HostBreaker.record_failure("h.example", requests.exceptions.ConnectTimeout("x"))
    assert not HostBreaker.is_open("h.example")
    HostBreaker.record_failure("h.example", requests.exceptions.ConnectTimeout("x"))
    assert HostBreaker.is_open("h.example")


def test_disabled_env_makes_breaker_inert(monkeypatch):
    monkeypatch.setenv("HOST_BREAKER_DISABLED", "1")
    HostBreaker.record_failure("h.example", requests.exceptions.ConnectTimeout("x"))
    assert not HostBreaker.is_open("h.example")
    HostBreaker.check("h.example")  # no raise


def test_open_hosts_lists_only_open_ones():
    HostBreaker.record_failure("a.example", requests.exceptions.ConnectTimeout("x"))
    opened = HostBreaker.open_hosts(["a.example", "b.example"])
    assert [h for h, _ in opened] == ["a.example"]


# ── structured error the model can answer around ─────────────────────────


def test_structured_error_shape_and_siblings():
    exc = HostCircuitOpen("alasky.cds.unistra.fr", 87.4, "SSLError: Max retries exceeded")
    out = structured_error(exc, tool_name="hips_cutout", affected_tools=["hips_cutout", "datalab_color_image", "hips_rgb_composite"])
    assert out["success"] is False
    assert out["circuit_breaker"] is True and out["infrastructure_failure"] is True
    assert out["host"] == "alasky.cds.unistra.fr"
    assert out["retry_after_seconds"] == 87
    assert out["affected_tools"] == ["datalab_color_image", "hips_rgb_composite"]
    assert "NOT executed" in out["error"] and "Do NOT retry" in out["error"]
    assert "datalab_color_image" in out["error"]


# ── genuinely unreachable hosts through guarded_request ──────────────────


def _closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_connection_refused_trips_and_second_call_fails_in_milliseconds():
    port = _closed_port()
    url = f"http://127.0.0.1:{port}/hips2fits"
    with pytest.raises(requests.exceptions.ConnectionError):
        guarded_request("GET", url, timeout=2)
    assert HostBreaker.is_open("127.0.0.1")
    t0 = time.perf_counter()
    with pytest.raises(HostCircuitOpen):
        guarded_request("GET", f"http://127.0.0.1:{port}/another-tool-same-host", timeout=2)
    assert (time.perf_counter() - t0) < 0.05


def test_unroutable_host_connect_timeout_trips():
    # RFC 5737 TEST-NET-1: never routable, so this is a real connect timeout,
    # not a mocked one. Bounded at 0.5 s so the test stays fast.
    url = "http://192.0.2.1:81/hips2fits"
    with pytest.raises((requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError)):
        guarded_request("GET", url, timeout=0.5)
    assert HostBreaker.is_open("192.0.2.1")
    with pytest.raises(HostCircuitOpen):
        guarded_request("GET", url, timeout=0.5)


def test_gateway_status_trips_without_exception(monkeypatch, clock):
    class _Resp:
        status_code = 503

    # guarded_request dispatches GET/POST through requests.get/post so code
    # that patches those keeps working — patch the same surface here.
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    for _ in range(3):  # gateway statuses are soft: three calls, spread out
        guarded_request("GET", "https://gw.example/x")
        clock.advance(25)
    assert HostBreaker.is_open("https://gw.example/x")


def test_successful_response_records_success(monkeypatch):
    class _Resp:
        status_code = 200

    HostBreaker.record_failure("ok.example", requests.exceptions.ConnectTimeout("x"))
    HostBreaker.reset()  # simulate cooldown over
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    guarded_request("GET", "https://ok.example/x")
    assert not HostBreaker.is_open("ok.example")
