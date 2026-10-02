"""An unreachable NRAO TAP archive must be a visible tool error, not "no data".

MANNA-evals UI benchmark 2026-10-02: data-query.nrao.edu timed out and the
NRAO client returned a bare empty DataFrame, so search_by_target /
search_by_position / search_by_frequency (facility VLA) reported success with
0 rows and both models told users "there are no VLA observations" (even for
3C 273). These tests pin the fix: an outage or failed query is an error frame
(``attrs["quasar_error"]`` + ``quasar_error_meta``) that the capabilities turn
into ``success: False`` with ``infrastructure_failure`` and a retry hint, while
a genuine zero-row answer stays an empty success.

All offline: the connection and the TAP service are faked.
"""

import pandas as pd
import pytest
import requests
from astropy.coordinates import SkyCoord

import integrations.tap as tap
from capabilities.alma import SearchByFrequency, SearchByPosition, SearchByTarget
from capabilities.base import CallContext
from integrations.tap import NRAO_TAP_HOST, NRAOTapClient
from services.host_breaker import HostBreaker, HostCircuitOpen
from services.search import SearchService

pytestmark = pytest.mark.unit


def _timeout():
    return requests.exceptions.ConnectTimeout(
        f"HTTPSConnectionPool(host='{NRAO_TAP_HOST}', port=443): Max retries exceeded (connect timeout=10)"
    )


def _unreachable_client(monkeypatch, exc=None):
    """A lazy NRAOTapClient whose connection attempt fails with ``exc``."""
    client = NRAOTapClient()
    err = exc if exc is not None else _timeout()

    def _fail():
        client._last_connect_error = err
        client.nrao_tap = None
        client._connected = False
        return False

    monkeypatch.setattr(client, "_ensure_connected", _fail)
    return client


class _FakeTap:
    """Stands in for pyvo's TAPService: returns ``rows`` or raises ``exc``."""

    def __init__(self, rows=None, exc=None):
        self.rows = rows if rows is not None else []
        self.exc = exc
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        if self.exc is not None:
            raise self.exc
        return _FakeResult(self.rows)


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def to_table(self):
        return self

    def to_pandas(self):
        return pd.DataFrame(self._rows)


def _connected_client(monkeypatch, tap_service):
    client = NRAOTapClient()
    client.nrao_tap = tap_service
    client.obscore_table = "obscore"
    client._connected = True
    monkeypatch.setattr(
        tap.SkyCoord, "from_name", staticmethod(lambda name: SkyCoord(187.2779, 2.0524, unit="deg"))
    )
    return client


def _assert_outage_frame(df):
    assert isinstance(df, pd.DataFrame) and df.empty
    err = df.attrs.get("quasar_error")
    assert err, "an unreachable archive must not look like a genuine empty result"
    meta = df.attrs["quasar_error_meta"]
    assert meta["infrastructure_failure"] is True
    assert meta["host"] == NRAO_TAP_HOST
    assert meta["retry_hint"] and meta["retry_hint"] in err
    assert "UNKNOWN whether VLA/VLBA/GBT observations exist" in err
    assert "not a 'no observations' result" in err
    assert "retrying in a few minutes" in err
    return err, meta


# ── Client level ─────────────────────────────────────────────────────


@pytest.mark.parametrize("call", [
    lambda c: c.search_by_position(187.2779, 2.0524, 0.1, instruments=["VLA"]),
    lambda c: c.search_by_frequency_range(1.0, 2.0, instruments=["VLA"]),
    lambda c: c.search_vla_vlba("3C 273", instruments=["VLA"]),
], ids=["position", "frequency", "target"])
def test_unreachable_service_returns_error_frame(monkeypatch, call):
    client = _unreachable_client(monkeypatch)
    err, meta = _assert_outage_frame(call(client))
    assert "ConnectTimeout" in err
    assert "Do NOT retry it this turn" in err  # a plain timeout has no short cooldown
    assert "circuit_breaker" not in meta


def test_unreachable_without_recorded_cause_is_still_an_outage(monkeypatch):
    client = NRAOTapClient()
    monkeypatch.setattr(client, "_ensure_connected", lambda: False)
    _assert_outage_frame(client.search_vla_vlba("3C 273"))


def test_ensure_connected_records_the_failure(monkeypatch):
    def _boom(*a, **k):
        raise _timeout()

    monkeypatch.setattr(tap.requests, "get", _boom)
    client = NRAOTapClient()
    err, _ = _assert_outage_frame(client.search_by_position(1.0, 2.0, 0.1))
    assert isinstance(client._last_connect_error, requests.exceptions.ConnectTimeout)
    assert "connect timeout" in err


def test_open_breaker_short_cooldown_allows_one_retry(monkeypatch):
    monkeypatch.setattr(HostBreaker, "auto_wait_seconds", staticmethod(lambda: 30.0))
    client = _unreachable_client(monkeypatch, HostCircuitOpen(f"{NRAO_TAP_HOST}/tap", 12, "ConnectTimeout"))
    err, meta = _assert_outage_frame(client.search_vla_vlba("3C 273"))
    assert meta["circuit_breaker"] is True
    assert meta["retry_after_s"] == 12 and meta["retry_after_seconds"] == 12
    assert "MAY retry this exact call ONCE after ~12 s" in err


def test_open_breaker_long_cooldown_says_do_not_retry(monkeypatch):
    monkeypatch.setattr(HostBreaker, "auto_wait_seconds", staticmethod(lambda: 30.0))
    client = _unreachable_client(monkeypatch, HostCircuitOpen(NRAO_TAP_HOST, 240, "ConnectTimeout"))
    err, meta = _assert_outage_frame(client.search_by_frequency_range(1.0, 2.0))
    assert meta["retry_after_s"] == 240
    assert "Do NOT retry it this turn" in err


@pytest.mark.parametrize("call", [
    lambda c: c.search_by_position(187.2779, 2.0524, 0.1),
    lambda c: c.search_by_frequency_range(1.0, 2.0),
    lambda c: c.search_vla_vlba("3C 273"),
], ids=["position", "frequency", "target"])
def test_query_time_timeout_is_an_error_not_empty(monkeypatch, call):
    client = _connected_client(monkeypatch, _FakeTap(exc=_timeout()))
    _assert_outage_frame(call(client))


def test_query_level_failure_is_an_error_without_infra_flag(monkeypatch):
    client = _connected_client(monkeypatch, _FakeTap(exc=RuntimeError("HTTP 400: ADQL syntax error")))
    df = client.search_by_frequency_range(1.0, 2.0)
    assert df.empty and "ADQL syntax error" in df.attrs["quasar_error"]
    meta = df.attrs["quasar_error_meta"]
    assert "infrastructure_failure" not in meta
    assert meta["retry_hint"] and "not a 'no observations' result" in meta["retry_hint"]


@pytest.mark.parametrize("call", [
    lambda c: c.search_by_position(187.2779, 2.0524, 0.1),
    lambda c: c.search_by_frequency_range(1.0, 2.0),
    lambda c: c.search_vla_vlba("3C 273"),
], ids=["position", "frequency", "target"])
def test_genuine_zero_rows_stay_a_plain_empty_frame(monkeypatch, call):
    client = _connected_client(monkeypatch, _FakeTap(rows=[]))
    df = call(client)
    assert df.empty
    assert "quasar_error" not in df.attrs
    assert "quasar_error_meta" not in df.attrs


def test_rows_are_returned_without_error_marker(monkeypatch):
    rows = [{"obs_publisher_did": "nrao:1", "target_name": "3C273", "s_ra": 187.28, "s_dec": 2.05,
             "t_min": 58000.0, "t_max": 58000.1, "instrument_name": "VLA"}]
    client = _connected_client(monkeypatch, _FakeTap(rows=rows))
    df = client.search_vla_vlba("3C 273", instruments=["VLA"])
    assert len(df) == 1 and "quasar_error" not in df.attrs


# ── Tool level: the search capabilities with facility="VLA" ──────────


def _service(nrao_client):
    svc = SearchService.__new__(SearchService)  # skip ALminer construction
    svc.alminer_client = None
    svc._nrao_client = nrao_client
    return svc


def _ctx(search_service):
    state = {}
    services = {
        "search_service": search_service,
        "console_log": lambda *_: None,
        "get_last_search_results": lambda: state.get("lsr"),
        "get_last_run_result": lambda: state.get("lrr"),
        "set_last_search_results": lambda v: state.__setitem__("lsr", v),
        "set_last_run_result": lambda v: state.__setitem__("lrr", v),
        "alma_tap_provenance": {"query": None, "url": None},
        "resolve_target": lambda name: {"success": False},
    }
    return CallContext(services=services)


def _run(cap, ctx, **kwargs):
    return cap.run(cap.InputModel(**kwargs), ctx).to_native()


TOOL_CALLS = [
    (SearchByTarget, {"target_name": "3C 273", "facility": "VLA"}),
    (SearchByTarget, {"target_name": "3C 273, M87", "facility": "VLA"}),
    (SearchByPosition, {"ra": 187.2779, "dec": 2.0524, "radius": 0.1, "facility": "VLA"}),
    (SearchByFrequency, {"min_freq_ghz": 1.0, "max_freq_ghz": 2.0, "facility": "VLA"}),
]
TOOL_IDS = ["target", "multi-target", "position", "frequency"]


@pytest.mark.parametrize("cap,kwargs", TOOL_CALLS, ids=TOOL_IDS)
def test_tool_reports_unreachable_nrao_as_error_with_retry_hint(monkeypatch, cap, kwargs):
    out = _run(cap(), _ctx(_service(_unreachable_client(monkeypatch))), **kwargs)

    assert out["success"] is False
    assert "total_results" not in out
    assert out["infrastructure_failure"] is True
    assert out["host"] == NRAO_TAP_HOST
    assert "retrying in a few minutes" in out["retry_hint"]
    assert "unreachable" in out["error"] and "not a 'no observations' result" in out["error"]
    assert "archive/service error" in out["note"]


@pytest.mark.parametrize("cap,kwargs", TOOL_CALLS, ids=TOOL_IDS)
def test_tool_reports_genuine_zero_rows_as_empty_success(monkeypatch, cap, kwargs):
    out = _run(cap(), _ctx(_service(_connected_client(monkeypatch, _FakeTap(rows=[])))), **kwargs)

    assert out.get("success") is not False
    assert "infrastructure_failure" not in out
    assert "retry_hint" not in out
    assert out.get("total_results", 0) == 0


def test_breaker_fields_reach_the_tool_result(monkeypatch):
    monkeypatch.setattr(HostBreaker, "auto_wait_seconds", staticmethod(lambda: 30.0))
    client = _unreachable_client(monkeypatch, HostCircuitOpen(NRAO_TAP_HOST, 9, "ConnectTimeout"))
    out = _run(SearchByTarget(), _ctx(_service(client)), target_name="3C 273", facility="VLA")
    assert out["success"] is False
    assert out["circuit_breaker"] is True and out["retry_after_s"] == 9
    assert "MAY retry this exact call ONCE" in out["retry_hint"]
