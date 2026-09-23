"""match_cross_archive_sources under the budget hierarchy (2026-09-21, AM-H-01).

Live trace: the 12-source Perseus cross-match ran a footprint-aware bulk ALMA
query through three 120 s mirror attempts (310 s to fail) and then twelve
sequential MAST queries, under a 150 s tool guard. The guard always fired, the
model re-called the tool (then its sibling), and one question cost 490 s.

Contract now: ALMA and MAST phases run concurrently under the tool's inner
deadline; a failed bulk footprint query falls back to bounded per-source point
cones; MAST stops when the budget is spent or its host breaker is open; the
tool ALWAYS returns before the guard with a partial, disclosed result.
"""
from __future__ import annotations

import threading
import time

import pandas as pd
import pytest

from capabilities.alma import MatchCrossArchiveSources, MatchPerseusProtostarsAlmaJwst
from capabilities.base import CallContext
from services.host_breaker import HostBreaker, HostCircuitOpen
from services.tool_budgets import Deadline, begin_tool_deadline, end_tool_deadline


@pytest.fixture(autouse=True)
def _clean():
    HostBreaker.reset()
    end_tool_deadline()
    yield
    HostBreaker.reset()
    end_tool_deadline()


class _State:
    def __init__(self):
        self.last_run_result = None
        self.last_search_results = None


def _ctx(search_service, mast_client):
    state = _State()
    services = {
        "search_service": search_service,
        "mast_client": mast_client,
        "alma_tap_provenance": {"query": None, "url": None, "note": None},
        "get_last_run_result": lambda: state.last_run_result,
    }

    def _set_lsr(v):
        state.last_search_results = v

    def _set_lrr(v):
        state.last_run_result = v

    services["set_last_search_results"] = _set_lsr
    services["set_last_run_result"] = _set_lrr
    return CallContext(services=services), state


class _TapResult:
    def __init__(self, df):
        self._df = df

    def to_table(self):
        return self

    def to_pandas(self):
        return self._df.copy()


class _Tap:
    """Scriptable ALMA TAP: `bulk` handles the 12-way query, `point` the
    per-source fallback. Either may be a DataFrame, an exception, or a
    callable(query) -> DataFrame."""

    def __init__(self, bulk, point=None):
        self.bulk, self.point = bulk, point
        self.queries = []
        self.lock = threading.Lock()

    def search(self, query):
        with self.lock:
            self.queries.append(query)
        is_bulk = query.count("CIRCLE(") > 1 or "INTERSECTS" in query
        target = self.bulk if is_bulk else self.point
        if isinstance(target, BaseException):
            raise target
        if callable(target):
            return _TapResult(target(query))
        return _TapResult(target if target is not None else pd.DataFrame())


class _Alminer:
    def __init__(self, tap):
        self._tap = tap

    def _get_tap_service(self):
        return self._tap

    def _standardize_columns(self, df):
        return df


class _Search:
    def __init__(self, tap):
        self.alminer_client = _Alminer(tap)


class _Mast:
    def __init__(self, df=None, delay=0.0, error=None):
        self._df = df if df is not None else pd.DataFrame()
        self.delay, self.error = delay, error
        self.calls = []

    def search_by_position(self, ra, dec, radius_arcmin=1.0, mission=None, max_results=500):
        self.calls.append((ra, dec, mission))
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            out = pd.DataFrame()
            out.attrs["error"] = self.error
            return out
        return self._df.copy()


def _alma_row(name, ra, dec):
    return {"target_name": name, "proposal_id": "2023.1.00001.S", "s_ra": ra, "s_dec": dec}


def _mast_row(name):
    return {"target_name": name, "telescope": "JWST", "instrument_name": "NIRCam", "project_code": "1234"}


def _run(cap, ctx, **kwargs):
    return cap.run(cap.InputModel(**kwargs), ctx).to_native()


SOURCES = [
    {"source_name": "S1", "ra": 51.4126, "dec": 30.7343},
    {"source_name": "S2", "ra": 52.2657, "dec": 31.2420},
    {"source_name": "S3", "ra": 55.9803, "dec": 32.0030},
]


# ── happy path: bulk footprint query works, phases run concurrently ──────


def test_bulk_footprint_query_is_preferred_and_phases_overlap():
    alma = pd.DataFrame([_alma_row(s["source_name"], s["ra"], s["dec"]) for s in SOURCES])
    tap = _Tap(bulk=alma)
    mast = _Mast(pd.DataFrame([_mast_row("x")]), delay=0.15)
    ctx, state = _ctx(_Search(tap), mast)
    t0 = time.perf_counter()
    out = _run(MatchCrossArchiveSources(), ctx, catalog_name="inline", sources=SOURCES, archives=["ALMA", "JWST"])
    elapsed = time.perf_counter() - t0
    assert out["success"] and not out["partial"]
    assert out["matched_sources"] == 3
    assert out["footprint_mode"] == "intersects_or_point"
    assert len(tap.queries) == 1, "one bulk query when the footprint path works"
    assert len(mast.calls) == 3
    assert out["archive_errors"] == []
    assert state.last_run_result["partial"] is False


# ── bulk footprint query fails -> bounded per-source point cones ─────────


def test_failed_bulk_query_falls_back_to_point_cones_and_discloses_it():
    def point(query):
        # answer only S1 and S3's cones
        for s in SOURCES:
            if f"{s['ra']:.8f}" in query and s["source_name"] in ("S1", "S3"):
                return pd.DataFrame([_alma_row(s["source_name"], s["ra"], s["dec"])])
        return pd.DataFrame()

    tap = _Tap(bulk=RuntimeError("ALMA TAP query failed: DALFormatError: ReadTimeout (read timeout=40.0)"), point=point)
    mast = _Mast(pd.DataFrame([_mast_row("x")]))
    ctx, state = _ctx(_Search(tap), mast)
    out = _run(MatchCrossArchiveSources(), ctx, catalog_name="inline", sources=SOURCES, archives=["ALMA", "JWST"])
    assert out["success"] and out["partial"]
    assert out["footprint_mode"] == "point_only"
    assert "pointing centres only" in out["note"]
    assert len(tap.queries) == 1 + 3, "bulk attempt then one point cone per source"
    assert any("fell back to per-source point cones" in e for e in out["archive_errors"])
    names = {r["source_name"] for r in out["results"]}
    assert {"S1", "S3"} <= names
    matched = {r["source_name"]: r for r in out["results"]}
    assert matched["S1"]["alma_observations"] == 1 and matched["S2"]["alma_observations"] == 0
    assert ctx.service("alma_tap_provenance")["note"], "provenance discloses the point-only execution"


# ── the tool returns BEFORE the guard when ALMA hangs ────────────────────


def test_hanging_alma_phase_is_abandoned_at_the_deadline_with_partial_mast_results():
    release = threading.Event()

    def hang(query):
        release.wait(20.0)
        return pd.DataFrame()

    tap = _Tap(bulk=hang, point=hang)
    mast = _Mast(pd.DataFrame([_mast_row("x")]))
    ctx, state = _ctx(_Search(tap), mast)
    # Simulate running under a tool guard with a small inner budget (4 s):
    # enough for the MAST loop (which needs >= 3 s left per query), while the
    # ALMA phase never answers.
    begin_tool_deadline("match_cross_archive_sources", 4.0 + 15.0)  # -> inner 4 s
    try:
        t0 = time.perf_counter()
        out = _run(MatchCrossArchiveSources(), ctx, catalog_name="inline", sources=SOURCES, archives=["ALMA", "JWST"])
        elapsed = time.perf_counter() - t0
    finally:
        release.set()
    assert elapsed < 7.0, f"tool must return at its deadline, took {elapsed:.1f}s"
    assert out["success"] and out["partial"]
    assert any("did not finish within the tool budget" in e for e in out["archive_errors"])
    # MAST side still delivered: every source has its JWST count.
    assert len(mast.calls) == 3
    assert all(r["mast_observations"] == 1 for r in out["results"])


# ── MAST budget exhaustion names the sources that were not queried ───────


def test_mast_loop_stops_when_budget_is_spent():
    alma = pd.DataFrame([_alma_row(s["source_name"], s["ra"], s["dec"]) for s in SOURCES])
    tap = _Tap(bulk=alma)
    mast = _Mast(pd.DataFrame([_mast_row("x")]), delay=0.35)
    ctx, _ = _ctx(_Search(tap), mast)
    begin_tool_deadline("match_cross_archive_sources", 3.4 + 15.0)  # inner 3.4 s: room for ~1 MAST call (<3 s left after it)
    out = _run(MatchCrossArchiveSources(), ctx, catalog_name="inline", sources=SOURCES, archives=["ALMA", "JWST"])
    assert out["success"] and out["partial"]
    assert len(mast.calls) < 3
    assert any("MAST not queried for" in e and "tool budget exhausted" in e for e in out["archive_errors"])


# ── an open MAST host breaker stops the loop after the first refusal ─────


def test_open_mast_breaker_short_circuits_remaining_sources():
    alma = pd.DataFrame([_alma_row(s["source_name"], s["ra"], s["dec"]) for s in SOURCES])
    tap = _Tap(bulk=alma)
    mast = _Mast(error="HostCircuitOpen: circuit breaker open for mast.stsci.edu (retry in 118 s): ReadTimeout")
    ctx, _ = _ctx(_Search(tap), mast)
    out = _run(MatchCrossArchiveSources(), ctx, catalog_name="inline", sources=SOURCES, archives=["ALMA", "JWST"])
    assert len(mast.calls) == 1, "after the first refusal the remaining sources are not attempted"
    assert out["success"] and out["partial"]
    assert any("MAST host unreachable (circuit open)" in e for e in out["archive_errors"])


# ── ALMA breaker open -> loud, fast, structured ──────────────────────────


def test_open_alma_breaker_fails_fast_and_honestly_when_nothing_matches():
    tap = _Tap(bulk=HostCircuitOpen("almascience.nrao.edu", 100.0, "SSLError"))
    mast = _Mast(pd.DataFrame())  # nothing on the MAST side either
    ctx, _ = _ctx(_Search(tap), mast)
    t0 = time.perf_counter()
    out = _run(MatchCrossArchiveSources(), ctx, catalog_name="inline", sources=SOURCES, archives=["ALMA", "JWST"],
               require_all_archives=True)
    assert time.perf_counter() - t0 < 2.0
    assert out["success"] is False
    assert "outage" in out["error"]
    assert any("ALMA TAP unreachable (almascience.nrao.edu)" in e for e in out["archive_errors"])


def test_perseus_wrapper_shares_the_budgeted_implementation():
    from services.cross_archive_matcher import PERSEUS_PROTOSTARS

    alma = pd.DataFrame([_alma_row(s["source_name"], s["ra"], s["dec"]) for s in PERSEUS_PROTOSTARS[:2]])
    tap = _Tap(bulk=alma)
    mast = _Mast(pd.DataFrame([_mast_row("x")]))
    ctx, state = _ctx(_Search(tap), mast)
    out = _run(MatchPerseusProtostarsAlmaJwst(), ctx, max_sources=2)
    assert out["success"] and out["mode"] == "perseus_alma_jwst_cross_match"
    assert out["elapsed_seconds"] >= 0
    assert state.last_run_result["tool_name"] == "match_perseus_protostars_alma_jwst"
