"""Regression tests for the Codex guard review of the 2026-09-21 budget hierarchy
(tmp/codex/tasks/task-bf4e790-3838/review.md). One test per finding, named
after it, plus the tests the review listed as missing.
"""
from __future__ import annotations

import threading
import time

import pandas as pd
import pytest
import requests

from services import http_budget_hook as hook
from services import tool_budgets as tb
from services.host_breaker import HostBreaker, HostCircuitOpen, classify_infrastructure_error


@pytest.fixture(autouse=True)
def _clean():
    HostBreaker.reset()
    tb.end_tool_deadline()
    yield
    HostBreaker.reset()
    tb.end_tool_deadline()


# ── CX-01: the cross-match's outage reaches the runner as a structured, cacheable failure ──


def _xmatch_ctx(tap_exc, mast_df):
    from capabilities.base import CallContext

    class _Tap:
        def search(self, q):
            raise tap_exc

    class _Alminer:
        def _get_tap_service(self):
            return _Tap()

        def _standardize_columns(self, df):
            return df

    class _Search:
        alminer_client = _Alminer()

    class _Mast:
        calls = []

        def search_by_position(self, ra, dec, radius_arcmin=1.0, mission=None, max_results=500):
            self.calls.append((ra, dec))
            return mast_df.copy()

    state = {}
    services = {
        "search_service": _Search(),
        "mast_client": _Mast(),
        "alma_tap_provenance": {"query": None, "url": None, "note": None},
        "set_last_search_results": lambda v: state.__setitem__("lsr", v),
        "set_last_run_result": lambda v: state.__setitem__("lrr", v),
        "get_last_run_result": lambda: state.get("lrr"),
    }
    return CallContext(services=services), state


SOURCES = [{"source_name": "S1", "ra": 51.4126, "dec": 30.7343}, {"source_name": "S2", "ra": 52.2657, "dec": 31.2420}]


def test_cx01_open_alma_host_with_no_matches_is_a_structured_infrastructure_failure():
    from capabilities.alma import MatchCrossArchiveSources
    from core.runner import _result_indicates_timeout, _result_is_dead_end

    ctx, _ = _xmatch_ctx(HostCircuitOpen("almascience.nrao.edu", 100.0, "SSLError"), pd.DataFrame())
    cap = MatchCrossArchiveSources()
    out = cap.run(cap.InputModel(catalog_name="inline", sources=SOURCES, archives=["ALMA", "JWST"], require_all_archives=True), ctx).to_native()
    assert out["success"] is False
    assert out["infrastructure_failure"] is True and out["circuit_breaker"] is True
    assert out["host"] == "almascience.nrao.edu" and out["retry_after_seconds"] == 100
    assert out["dead_hosts"] == ["almascience.nrao.edu"]
    assert _result_indicates_timeout(out) is True, "runner classifies it as non-progress"
    assert _result_is_dead_end(out) is True, "an identical re-call would be refused from cache"


def test_cx01_partial_success_with_a_dead_host_is_progress_but_not_worth_repeating():
    from capabilities.alma import MatchCrossArchiveSources
    from core.runner import _result_indicates_timeout, _result_is_dead_end

    mast = pd.DataFrame([{"target_name": "x", "telescope": "JWST", "instrument_name": "NIRCam", "project_code": "1"}])
    ctx, _ = _xmatch_ctx(HostCircuitOpen("almascience.nrao.edu", 60.0, "ReadTimeout"), mast)
    cap = MatchCrossArchiveSources()
    out = cap.run(cap.InputModel(catalog_name="inline", sources=SOURCES, archives=["ALMA", "JWST"]), ctx).to_native()
    assert out["success"] is True and out["partial"] is True
    assert out["dead_hosts"] == ["almascience.nrao.edu"]
    assert "unreachable during this call" in out["note"]
    # Real MAST rows came back: the SSE deadline may extend for it...
    assert _result_indicates_timeout(out) is False
    # ...but re-issuing the identical call cannot improve on it this turn.
    assert _result_is_dead_end(out) is True


def test_cx01_runner_dead_end_classification():
    from core.runner import _result_is_dead_end

    assert _result_is_dead_end({"success": False, "timeout": True})
    assert _result_is_dead_end({"success": False, "infrastructure_failure": True})
    assert _result_is_dead_end({"success": True, "dead_hosts": ["h"]})
    assert _result_is_dead_end({"success": True, "budget_exhausted": True})
    assert not _result_is_dead_end({"success": True, "rows": 3})
    assert not _result_is_dead_end({"success": False, "error": "bad radius"})
    assert not _result_is_dead_end("not a dict")


# ── CX-02: ALminer's bounded source search inherits the tool deadline ────


def test_cx02_alminer_bounded_source_search_adopts_the_deadline():
    from integrations.alminer_client import run_bounded_source

    d = tb.begin_tool_deadline("search_by_target", 10.0 + tb.GUARD_HEADROOM_SECONDS)
    seen = {}

    def fn():
        seen["deadline"] = tb.current_deadline()
        return pd.DataFrame([{"a": 1}])

    df, err, timed_out = run_bounded_source(fn, 85)
    assert err is None and not timed_out and len(df) == 1
    # The helper adopts a CHILD of the tool deadline: same budget and call token.
    assert seen["deadline"].token is d.token and abs(seen["deadline"].remaining() - d.remaining()) < 0.5

    def hang():
        time.sleep(3.0)
        return pd.DataFrame()

    t0 = time.perf_counter()
    df, err, timed_out = run_bounded_source(hang, 0.2)
    assert df is None and timed_out and "timed out" in err
    assert time.perf_counter() - t0 < 1.5


def test_cx02_spent_budget_refuses_the_source_search_instantly():
    from integrations.alminer_client import run_bounded_source

    tb.begin_tool_deadline("search_by_target", 0.1 + tb.GUARD_HEADROOM_SECONDS)
    time.sleep(0.15)
    df, err, timed_out = run_bounded_source(lambda: pd.DataFrame(), 85)
    assert df is None and timed_out and "budget exhausted" in err


# ── CX-03: the hierarchy test bites on REAL constants ────────────────────


def test_cx03_reverting_the_tile_timeout_to_180s_fails_the_hierarchy(monkeypatch):
    monkeypatch.setenv("DATALAB_IMAGE_DOWNLOAD_TIMEOUT_SECONDS", "180")
    violations = tb.check_hierarchy(["datalab_color_image", "datalab_cutout_grid", "datalab_image_cutout"])
    assert len(violations) == 3 and all("270" in v for v in violations), violations


def test_cx03_raising_a_module_constant_fails_the_hierarchy(monkeypatch):
    from services import fits_service

    monkeypatch.setattr(fits_service, "DOWNLOAD_TIMEOUT_S", 180)
    assert tb.check_hierarchy(["render_fits_image"]), "a 180 s download under a 150 s guard must be a violation"
    assert tb.check_hierarchy(["overlay_fits_images"])


def test_cx03_alma_attempt_timeout_env_override_is_checked(monkeypatch):
    monkeypatch.setenv("ALMA_TAP_TIMEOUT_SECONDS", "120")
    # Tools that call the TAP facade directly see the raw 2 x 120 s.
    assert tb.check_hierarchy(["search_by_frequency", "advanced_search"]), "2 x 120 s attempts cannot fit a 180 s guard"
    # search_by_target is additionally capped by ALminerClient.SOURCE_TIMEOUT_S (85 s
    # join) — the declaration says so via min(), so it stays within budget.
    assert not tb.check_hierarchy(["search_by_target"])


def test_cx03_splatalogue_chain_includes_the_non_threaded_fallback():
    parts = tb.INNER_BUDGETS["search_spectral_lines"]()
    assert any("non-threaded" in k for k in parts)
    assert sum(parts.values()) <= 135.0


def test_cx03_loop_and_deadline_only_declarations_are_deliberate():
    for name in tb.LOOP_TOOLS:
        assert name in tb.INNER_BUDGETS and tb.LOOP_TOOLS[name]
        parts = tb.INNER_BUDGETS[name]()
        assert parts and all(v > 0 for v in parts.values()), name
        assert "deadline-driven" not in " ".join(parts), f"{name}: loop tools declare a real per-call chain"
    for name, reason in tb.DEADLINE_ONLY_TOOLS.items():
        assert reason and len(reason) > 20, name
    assert not (set(tb.LOOP_TOOLS) & set(tb.DEADLINE_ONLY_TOOLS))


# ── CX-04: a heavy-query 500 with "timed out" wording never trips the breaker ──


@pytest.mark.parametrize("msg", [
    "HTTP 500: heavy ADQL query timed out",
    "DALQueryError: 400 Client Error: query execution timed out",
    "500 Server Error: Internal Server Error for url: https://x/tap/sync (query aborted after 60 s)",
])
def test_cx04_query_level_timeouts_are_not_infrastructure(msg):
    assert classify_infrastructure_error(RuntimeError(msg)) is None
    assert HostBreaker.record_failure("x.example", RuntimeError(msg)) is None
    assert not HostBreaker.is_open("x.example")


@pytest.mark.parametrize("msg", [
    "DALFormatError: ReadTimeout: HTTPSConnectionPool(host='x', port=443): Read timed out. (read timeout=400.0)",
    "ALMA TAP query failed: DALServiceError: 502 Server Error: Proxy Error for url: https://x/tap/sync/run",
    "MAST query did not answer within 30 s (worker abandoned)",
])
def test_cx04_transport_wording_still_trips_even_with_status_like_numbers(msg):
    assert classify_infrastructure_error(RuntimeError(msg)) is not None


# ── CX-05: tuple timeouts with None are clamped too ──────────────────────


def test_cx05_tuple_timeout_with_none_is_clamped(monkeypatch):
    seen = {}

    def fake(self, method, url, *a, **kw):
        seen["timeout"] = kw.get("timeout")
        return type("R", (), {"status_code": 200})()

    # Start from the genuine method (an earlier test may have left the hook
    # installed), patch it with the fake, then let install() wrap the fake;
    # uninstall() restores the fake and monkeypatch restores the genuine one.
    hook.reset_for_tests()
    monkeypatch.setattr(requests.Session, "request", fake)
    assert hook.install()
    try:
        tb.begin_tool_deadline("t", 20.0 + tb.GUARD_HEADROOM_SECONDS)
        requests.Session().request("GET", "https://archive.example/x", timeout=(None, 600))
        connect, read = seen["timeout"]
        assert connect is not None and connect <= 20.0 and read <= 20.0
    finally:
        hook.uninstall()


def test_cx05_tap_session_clamps_none_tuple_elements(monkeypatch):
    from integrations.tap import _TimeoutHTTPSession

    captured = {}

    def fake_request(self, method, url, **kwargs):
        captured["timeout"] = kwargs["timeout"]
        return type("R", (), {"status_code": 200})()

    monkeypatch.setattr(requests.Session, "request", fake_request)
    tb.begin_tool_deadline("t", 20.0 + tb.GUARD_HEADROOM_SECONDS)
    _TimeoutHTTPSession(timeout=30.0).request("GET", "https://almascience.nrao.edu/tap/sync", timeout=(None, 600))
    connect, read = captured["timeout"]
    assert connect is not None and connect <= 30.0 and read <= 20.0


# ── CX-06: a disabled guard means NO cross-match deadline ────────────────


def test_cx06_disabled_guard_disables_the_cross_match_deadline(monkeypatch):
    monkeypatch.setenv("QUASAR_TOOL_TIMEOUT_SECONDS", "0")
    assert tb.inner_ceiling_seconds("match_cross_archive_sources") is None
    from capabilities.alma import MatchCrossArchiveSources

    mast = pd.DataFrame([{"target_name": "x", "telescope": "JWST", "instrument_name": "NIRCam", "project_code": "1"}])
    ctx, _ = _xmatch_ctx(RuntimeError("ALMA TAP query failed: 400 bad column"), mast)
    cap = MatchCrossArchiveSources()
    out = cap.run(cap.InputModel(catalog_name="inline", sources=SOURCES, archives=["ALMA", "JWST"]), ctx).to_native()
    assert out["success"] is True
    assert out["budget_exhausted"] is False, "no deadline: nothing can be cut by it"


# ── CX-07: platform-host exclusion is DNS-label bounded ──────────────────


@pytest.mark.parametrize("host,excluded", [
    ("api.openai.com", True), ("openai.com", True), ("notopenai.com", False), ("evilturso.io", False),
    ("db.turso.io", True), ("localhost", True), ("127.0.0.1", True), ("alasky.cds.unistra.fr", False),
])
def test_cx07_exclusion_boundaries(host, excluded):
    assert hook._excluded(host) is excluded


# ── missing tests from the review ─────────────────────────────────────────


def test_sibling_tool_on_the_same_host_trips_the_outer_timeout_breaker():
    """Per-declared-host counters (core/agent.py): two guard timeouts of tool A
    open the door for sibling B on the same archive to be refused instantly."""
    import types

    from core.agent import QuasarAgent

    tb.declare(["__probe_a__"], lambda: {"x": 1.0}, ["probe.archive.example"])
    tb.declare(["__probe_b__"], lambda: {"x": 1.0}, ["probe.archive.example"])
    try:
        agent = QuasarAgent.__new__(QuasarAgent)
        agent._tls = threading.local()
        agent._tls.tool_timeout_breaker = {}
        release = threading.Event()

        def slow(**kw):
            release.wait(5.0)
            return {"success": True}

        tool = types.SimpleNamespace(execute=slow)
        for _ in range(QuasarAgent._TOOL_TIMEOUT_BREAKER_TRIPS):
            out = agent._execute_tool_guarded(tool, {}, tool_name="__probe_a__", timeout_seconds=0.1)
            assert out["timeout"] is True
        t0 = time.perf_counter()
        out = agent._execute_tool_guarded(tool, {}, tool_name="__probe_b__", timeout_seconds=30.0)
        assert out["circuit_breaker"] is True and "sibling" in out["error"]
        assert time.perf_counter() - t0 < 1.0
        release.set()
    finally:
        for n in ("__probe_a__", "__probe_b__"):
            tb.INNER_BUDGETS.pop(n, None)
            tb.TOOL_HOSTS.pop(n, None)


def test_open_host_precheck_refuses_before_starting_a_worker():
    import types

    from core.agent import QuasarAgent

    tb.declare(["__probe_c__"], lambda: {"x": 1.0}, ["dead.archive.example"])
    try:
        HostBreaker.record_failure("dead.archive.example", requests.exceptions.SSLError("down"))
        agent = QuasarAgent.__new__(QuasarAgent)
        agent._tls = threading.local()
        ran = {"n": 0}

        def execute(**kw):
            ran["n"] += 1
            return {"success": True}

        out = agent._execute_tool_guarded(types.SimpleNamespace(execute=execute), {}, tool_name="__probe_c__", timeout_seconds=30.0)
        assert ran["n"] == 0
        assert out["infrastructure_failure"] is True and out["host"] == "dead.archive.example"
    finally:
        tb.INNER_BUDGETS.pop("__probe_c__", None)
        tb.TOOL_HOSTS.pop("__probe_c__", None)
