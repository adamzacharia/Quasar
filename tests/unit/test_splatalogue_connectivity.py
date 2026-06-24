"""Connectivity-hardening contract tests for the Splatalogue client.

Covers the retry/backoff, query-result cache, and schema-drift behaviors added
to make Splatalogue access production-reliable. All offline — no real network.
"""

import threading

import pytest
import requests

from services.splatalogue import (
    SpectralLineQuery,
    SpectralWindow,
    SplatalogueClient,
    SplatalogueQueryCache,
    SplatalogueQueryCancelled,
    SplatalogueTool,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class ScriptedSession:
    """Returns/raises a scripted sequence of behaviors for post()/get()."""

    def __init__(self, behaviors):
        self.behaviors = list(behaviors)
        self.calls = 0

    def _next(self):
        self.calls += 1
        behavior = self.behaviors.pop(0)
        if isinstance(behavior, Exception):
            raise behavior
        return behavior

    def post(self, url, **kwargs):
        return self._next()

    def get(self, url, **kwargs):
        return self._next()


def _client(session, **kwargs):
    # backoff disabled so retries don't actually sleep during tests.
    params = dict(max_retries=3, backoff_base_seconds=0.0, backoff_max_seconds=0.0)
    params.update(kwargs)
    return SplatalogueClient(session=session, **params)


# -- retry / backoff ----------------------------------------------------------

def test_request_retries_transient_connection_error_then_succeeds():
    session = ScriptedSession(
        [
            requests.ConnectionError("RemoteDisconnected"),
            requests.ConnectionError("RemoteDisconnected"),
            FakeResponse({"ok": True}, 200),
        ]
    )
    client = _client(session)
    response = client._request_with_retry("POST", "https://x", json={}, timeout=5)
    assert response.status_code == 200
    assert session.calls == 3  # 2 failures + 1 success


def test_request_retries_retryable_status_then_succeeds():
    session = ScriptedSession([FakeResponse({}, 503), FakeResponse({"ok": 1}, 200)])
    client = _client(session)
    response = client._request_with_retry("GET", "https://x", timeout=5)
    assert response.status_code == 200
    assert session.calls == 2


def test_request_does_not_retry_client_error():
    # A genuine 400 must not be retried — it would mask a real query rejection.
    session = ScriptedSession([FakeResponse({"err": "bad"}, 400)])
    client = _client(session)
    response = client._request_with_retry("POST", "https://x", json={}, timeout=5)
    assert response.status_code == 400
    assert session.calls == 1


def test_request_raises_after_exhausting_retries():
    session = ScriptedSession([requests.ConnectionError("x")] * 6)
    client = _client(session, max_retries=2)
    with pytest.raises(requests.ConnectionError):
        client._request_with_retry("POST", "https://x", json={}, timeout=5)
    assert session.calls == 3  # 1 initial + 2 retries


def test_request_respects_cancellation_before_first_call():
    cancel = threading.Event()
    cancel.set()
    session = ScriptedSession([FakeResponse({}, 200)])
    client = _client(session)
    with pytest.raises(SplatalogueQueryCancelled):
        client._request_with_retry(
            "POST", "https://x", cancel_event=cancel, json={}, timeout=5
        )
    assert session.calls == 0


def test_backoff_delay_is_bounded_and_nonnegative():
    client = _client(ScriptedSession([]), backoff_base_seconds=0.5, backoff_max_seconds=4.0)
    for attempt in range(8):
        delay = client._backoff_delay(attempt)
        assert 0.0 <= delay <= 4.0


def test_call_with_retry_retries_then_succeeds():
    client = _client(ScriptedSession([]), max_retries=2)
    calls = {"n": 0}

    def func():
        calls["n"] += 1
        if calls["n"] < 2:
            raise requests.ConnectionError("transient")
        return "ok"

    assert client._call_with_retry(func, retryable=(requests.ConnectionError,)) == "ok"
    assert calls["n"] == 2


# -- query-result cache + schema-drift ---------------------------------------

class MemoryCache:
    """In-memory stand-in for SplatalogueQueryCache, keyed identically."""

    def __init__(self):
        self.store = {}

    def get(self, query):
        return self.store.get(SplatalogueQueryCache.key_for(query))

    def set(self, query, value):
        self.store[SplatalogueQueryCache.key_for(query)] = dict(value)


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def query(self, query, *, cancel_event=None):
        self.calls += 1
        return dict(self.response)


def _response(*, rows, backend, degraded=False):
    return {
        "rows": rows,
        "backend": backend,
        "degraded": degraded,
        "unsupported_filters": [],
        "warnings": [],
        "errors": [],
        "query_provenance": {},
    }


def _query():
    return SpectralLineQuery(windows=[SpectralWindow(100.0, 101.0)])


def test_non_degraded_result_is_cached_and_reused():
    cache = MemoryCache()
    client = FakeClient(_response(rows=[], backend="splatalogue_threaded_advanced"))
    tool = SplatalogueTool(client, query_cache=cache)
    query = _query()

    first = tool.query_catalog(query)
    second = tool.query_catalog(query)

    assert client.calls == 1  # second served from cache
    assert first["from_cache"] is False
    assert second["from_cache"] is True


def test_degraded_slap_result_is_not_cached():
    cache = MemoryCache()
    client = FakeClient(_response(rows=[], backend="slap", degraded=True))
    tool = SplatalogueTool(client, query_cache=cache)
    query = _query()

    tool.query_catalog(query)
    tool.query_catalog(query)

    assert client.calls == 2  # degraded results are always re-fetched


def test_schema_drift_flagged_when_rows_cannot_be_normalized():
    cache = MemoryCache()
    client = FakeClient(
        _response(
            rows=[{"totally": "unknown"}, {"shape": "changed"}],
            backend="splatalogue_advanced",
        )
    )
    tool = SplatalogueTool(client, query_cache=cache)
    query = _query()

    result = tool.query_catalog(query)

    assert result.get("schema_drift_suspected") is True
    assert any("schema drift" in w.lower() for w in result["warnings"])
    # A drift-suspected result must not be cached.
    assert cache.get(query) is None


def test_cancellation_during_slap_fallback_is_not_relabeled_as_failure(monkeypatch):
    # Regression: a SplatalogueQueryCancelled raised while the SLAP fallback runs
    # must propagate as a cancellation, not be re-wrapped as a hard query failure
    # (the job runner keys status off the exception type).
    def always_fail(url, **kwargs):
        raise requests.ConnectionError("advanced down")

    session = type("FailSession", (), {"post": staticmethod(always_fail), "get": staticmethod(always_fail)})()
    client = SplatalogueClient(
        session=session, max_retries=0, backoff_base_seconds=0.0, backoff_max_seconds=0.0
    )

    def raise_cancel(*args, **kwargs):
        raise SplatalogueQueryCancelled("canceled")

    monkeypatch.setattr(SplatalogueTool, "_query_slap", staticmethod(raise_cancel))

    # Single small window with no SLAP-unsupported filters -> reaches the SLAP branch.
    query = SpectralLineQuery(
        windows=[SpectralWindow(230.0, 230.5)], version="vall", exclude_categories=[]
    )
    with pytest.raises(SplatalogueQueryCancelled):
        client.query(query)


def test_use_cache_false_bypasses_cache():
    cache = MemoryCache()
    client = FakeClient(_response(rows=[], backend="splatalogue_threaded_advanced"))
    tool = SplatalogueTool(client, query_cache=cache)
    query = _query()

    tool.query_catalog(query, use_cache=False)
    tool.query_catalog(query, use_cache=False)

    assert client.calls == 2
    assert cache.get(query) is None
