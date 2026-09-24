"""Regression tests for the Codex guard review task-25bee13-12894 (2026-09-24,
gpt-6-luna high) of the DataLabBench push toward 100."""
from __future__ import annotations

import json
import threading
import time

import numpy as np
import pandas as pd
import pytest


# CX-01: a low remaining budget is split, never floored into the reserve.
def test_cx01_scan_budget_split_when_little_time_is_left(monkeypatch):
    from capabilities import datalab_tools
    from services import datalab_orchestration as orch
    from services.tool_budgets import begin_tool_deadline, end_tool_deadline
    from tests.unit.test_dlb_100_fixes import _SatClient, _sat_ctx

    seen = {}
    monkeypatch.setattr(orch, "tiled_density_aggregate",
                        lambda *a, **k: seen.setdefault("max_seconds", k.get("max_seconds")) and {"result_id": None, "error": "stop"})
    monkeypatch.setenv("DATALAB_SATELLITE_ASYNC", "0")
    cap = datalab_tools.SatelliteSearch()
    begin_tool_deadline("datalab_satellite_search", 115.0)   # inner 100 s
    try:
        cap.run(cap.InputModel(survey="delve", preset="delve_south"), _sat_ctx(_SatClient(), []))
    finally:
        end_tool_deadline()
    # reserve = min(90, (100 - 8) / 2) = 46 s, scan = 100 - 46 - 8 = 46 s (documented split)
    assert seen["max_seconds"] == pytest.approx(0.5 * (100.0 - 8.0), abs=1.5)


# CX-02: an abandoned concurrent worker's own deadline is cancelled.
def test_cx02_run_concurrently_cancels_abandoned_workers():
    from services.alma_server_side import run_concurrently
    from services.tool_budgets import is_cancelled

    stopped = threading.Event()

    def slow():
        t0 = time.monotonic()
        while time.monotonic() - t0 < 10:
            if is_cancelled():
                stopped.set()
                return "stopped"
            time.sleep(0.02)
        return "ran to the end"

    out = run_concurrently([slow, lambda: "fast"], wall_seconds=0.3)
    assert isinstance(out[0], TimeoutError) and out[1] == "fast"
    assert stopped.wait(2.0), "the abandoned worker must see its cancelled deadline"


# CX-03: a timed-out CMD population test is cancelled, not left querying.
def test_cx03_abandoned_population_test_is_cancelled(monkeypatch):
    from capabilities import datalab_tools

    box = {}
    from services.tool_budgets import Deadline
    d = Deadline(60.0)
    t = threading.Thread(target=lambda: time.sleep(0.5), daemon=True)
    t.start()
    cand = {"rank": 1}
    monkeypatch.setattr("services.tool_budgets.remaining_seconds", lambda: 12.1)  # join wait -> ~0.1 s
    notes = []
    got = list(datalab_tools.SatelliteSearch._join_population_tests([(cand, t, {**box, "deadline": d})], notes))
    assert got == [] and d.cancelled() and cand["population_verdict"] == "not tested (budget)"


# CX-04: an operational failure is "not checked", not a coverage gap.
def test_cx04_failed_panel_is_not_a_coverage_gap():
    from services.datalab_image_service import DatalabImageService
    from tests.unit.test_datalab_p1 import _MemoryPlottingService

    svc = DatalabImageService.__new__(DatalabImageService)
    svc.plotting_service = _MemoryPlottingService()

    def search(ra, dec, fov, **k):
        if ra == 1.0:
            raise RuntimeError("SIA HTTP 500")
        return {"coverage_gap": True, "rows": [], "used_endpoint": "sia"}

    svc.search = search
    out = svc.cutout_grid([{"ra": 1.0, "dec": 0.0}, {"ra": 2.0, "dec": 0.0}], 0.05, band="g")
    assert out["panels"][0]["coverage_gap"] is False and out["panels"][0]["not_checked"] is True
    assert out["panels"][1]["coverage_gap"] is True
    assert out["coverage_gap"] is False and out["panels_not_checked"] == 1


# CX-05: the Hess aggregate is bounded well below its row cap.
def test_cx05_hess_plane_is_bounded():
    from services import cmd_population as cp

    sql, _ = cp.build_hess_aggregate("nsc_dr2", "object", 185.43, -31.99)
    assert "(gmag - rmag) BETWEEN -1 AND 3" in sql and "gmag BETWEEN 14 AND 27" in sql
    n_cells = ((cp.COLOR_WINDOW[1] - cp.COLOR_WINDOW[0]) / cp.COLOR_BIN + 1) * ((cp.MAG_WINDOW[1] - cp.MAG_WINDOW[0]) / cp.MAG_BIN + 1)
    assert n_cells < cp.HESS_ROW_LIMIT


# CX-06: satellite peak geometry survives the RA 0/360 seam.
def test_cx06_aperture_peaks_across_ra_zero():
    from capabilities.datalab_tools import SatelliteSearch

    step = 0.05
    rows = []
    for i in range(-20, 21):
        for j in range(-20, 21):
            ra = (0.0 + i * step) % 360.0
            c = 20 + (40 if abs(i) <= 1 and abs(j) <= 1 else 0)
            rows.append({"ra_bin": round(ra, 4), "dec_bin": round(-30.0 + j * step, 4), "source_count": c})
    peaks = SatelliteSearch.aperture_peaks(pd.DataFrame(rows), step, max_candidates=3)
    assert peaks and (peaks[0]["ra"] < 0.1 or peaks[0]["ra"] > 359.9) and peaks[0]["significance"] > 5


# CX-07: a capped sample makes no completeness claim.
def test_cx07_capped_cmd_sample_claims_no_turnover():
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore
    from tests.unit.test_datalab_p1 import _MemoryPlottingService
    from tests.unit.test_reb_science_correctness import _FakeNSCClient

    out = orch.color_magnitude_diagram("nsc_dr2", "object", 260.06, 57.92, 0.4, limit=150, client=_FakeNSCClient(),
                                       result_store=DatalabResultStore(enable_disk_cache=False),
                                       plotting_service=_MemoryPlottingService())
    assert out["depth"]["sample_capped"] is True and out["depth"]["turnover_mag"] is None
    assert "does not measure the catalogue depth" in out["depth"]["note"]


# CX-09: an UNKNOWN async job is aborted before the sync fallback.
def test_cx09_unknown_job_is_aborted_before_fallback(monkeypatch):
    from capabilities import datalab_tools
    from capabilities.base import CallContext
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore

    class _Client:
        aborted = []

        def abort(self, jobid):
            self.aborted.append(jobid)

    client = _Client()
    store = DatalabResultStore(enable_disk_cache=False)
    monkeypatch.setenv("DATALAB_HEALPIX_ASYNC", "1")
    monkeypatch.setattr(orch, "async_density_aggregate", lambda *a, **k: {
        "success": False, "jobid": "job-u", "job_state": "UNKNOWN", "error": "status polling failed 3 times"})
    rid = store.put(pd.DataFrame({"healpix": [1, 2], "source_count": [10, 20]}), {"provenance": {}})
    monkeypatch.setattr(orch, "tiled_density_aggregate", lambda *a, **k: {"success": True, "result_id": rid, "warnings": []})
    monkeypatch.setattr(datalab_tools, "_run_analysis_plot",
                        lambda name, args, ctx: type("R", (), {"to_native": lambda self: {"success": True}})())
    cap = datalab_tools.HealpixDensityMap()
    out = cap.run(cap.InputModel(preset="south_gradient"), CallContext(services={"datalab_client": client}, result_store=store)).to_native()
    assert client.aborted == ["job-u"] and out["async_job"]["abort"] == "abort requested"


# CX-10: a project-grain query in one call does not support another call's row count.
def test_cx10_grain_support_is_attributed_per_call():
    from core.answer_verifier import build_trace_summary, verify_answer

    outputs = [
        {"output": json.dumps({"success": True, "rowcount": 19, "validated_sql": "SELECT obs_id FROM ivoa.obscore"})},
        {"output": json.dumps({"success": True, "rowcount": 4,
                               "validated_sql": "SELECT DISTINCT proposal_id FROM ivoa.obscore WHERE band_list LIKE '%6%'"})},
    ]
    ts = build_trace_summary(outputs, [], [])
    flagged = [c.text for c in verify_answer("The archive lists 19 projects.", ts).by_kind("count")]
    assert flagged == ["19 projects"]
    assert not verify_answer("The archive lists 4 projects.", ts).by_kind("count")


# CX-11: a failed run_result contributes no counts.
def test_cx11_failed_run_results_contribute_nothing():
    from core.answer_verifier import build_trace_summary

    ts = build_trace_summary([], [], [{"type": "data", "success": False, "rowcount": 77, "rows": [{"a": 1}] * 5}])
    assert 77 not in ts.counts and 5 not in ts.counts


# CX-12: special-float cast stripping is quote-aware.
def test_cx12_cast_strip_leaves_string_literals_alone():
    from integrations.datalab_client import DatalabClient

    sql = "SELECT a FROM t WHERE x < 'Infinity'::float8 AND s = 'it''s ''Infinity''::float8' AND n < 'NaN' :: real"
    out = DatalabClient.strip_materialized_for_async(sql)
    assert "x < 'Infinity' AND" in out and "'it''s ''Infinity''::float8'" in out and "n < 'NaN'" in out and "::" not in out.replace("''::", "")


# CX-13: no DECam explanation for a non-DECam catalog.
def test_cx13_decam_sentence_only_for_decam_catalogs():
    from capabilities.datalab import ColorImage

    gap = {"success": True, "coverage_gap": True, "available_bands": [], "provenance": {}}
    other = ColorImage.explain_outcome(gap, catalog="twomass", ra=10.0, dec=45.0, fov_deg=0.1, label="X", q=8, stretch=0.5)
    assert "DECam" not in other["coverage_reason"]
    ls = ColorImage.explain_outcome(gap, catalog="ls_dr9", ra=10.0, dec=45.0, fov_deg=0.1, label="X", q=8, stretch=0.5)
    assert "DECam" in ls["coverage_reason"]


# CX-14: a malformed worker count falls back to the default.
def test_cx14_malformed_worker_env_does_not_break_the_grid(monkeypatch):
    from services.datalab_image_service import DatalabImageService
    from tests.unit.test_datalab_p1 import _MemoryPlottingService

    monkeypatch.setenv("DATALAB_CUTOUT_GRID_WORKERS", "four")
    svc = DatalabImageService.__new__(DatalabImageService)
    svc.plotting_service = _MemoryPlottingService()
    svc.search = lambda ra, dec, fov, **k: {"coverage_gap": True, "rows": [], "used_endpoint": "sia"}
    out = svc.cutout_grid([{"ra": 1.0, "dec": 0.0}], 0.05, band="g")
    assert out["panels"][0]["coverage_gap"] is True


def test_cx07_real_turnover_is_reported():
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore
    from tests.unit.test_datalab_p1 import _MemoryPlottingService
    from integrations.datalab_client import DatalabResult

    class _Rising:
        def __init__(self):
            self.sql = []

        def query(self, *, sql=None, **k):
            rng = np.random.default_rng(2)
            g = np.concatenate([rng.uniform(17, 23, 300), rng.uniform(22, 23.2, 900), rng.uniform(23.2, 23.8, 40)])
            df = pd.DataFrame({"ra": np.full(g.size, 260.0), "dec": np.full(g.size, 57.9), "gmag": g, "rmag": g - 0.5})
            return DatalabResult.from_dataframe(df, {"query": sql})

    out = orch.color_magnitude_diagram("nsc_dr2", "object", 260.06, 57.92, 0.4, limit=5000, client=_Rising(),
                                       result_store=DatalabResultStore(enable_disk_cache=False),
                                       plotting_service=_MemoryPlottingService())
    assert out["depth"]["turnover_mag"] is not None and 22.0 <= out["depth"]["turnover_mag"] <= 23.3


# ── verify round 1 reopens (2026-09-24) ───────────────────────────────────
def test_cx02_verify_worker_deadline_ends_at_the_wall_clock():
    """An in-flight request is clamped by the requests hook to the worker's
    deadline, so the deadline itself must END at the wall clock."""
    from services.alma_server_side import run_concurrently
    from services.tool_budgets import begin_tool_deadline, current_deadline, end_tool_deadline

    begin_tool_deadline("datalab_satellite_search", 300.0)
    try:
        out = run_concurrently([lambda: current_deadline().remaining()], wall_seconds=3.0)
    finally:
        end_tool_deadline()
    assert out[0] <= 5.1


def test_cx03_verify_population_deadline_is_bounded():
    from services.tool_budgets import Deadline

    parent = Deadline(250.0)
    child = parent.child_until(60.0)
    assert child.remaining() <= 60.1 and child.token is parent.token and not child.cancelled()
    parent.cancel("turn stop")
    assert child.cancelled()


def test_cx07_verify_no_bound_note_makes_no_bound_claim():
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore
    from tests.unit.test_datalab_p1 import _MemoryPlottingService
    from tests.unit.test_reb_science_correctness import _FakeNSCClient

    out = orch.color_magnitude_diagram("nsc_dr2", "object", 260.06, 57.92, 0.4, limit=5000,
                                       value_cuts=[{"column": "gmag", "op": ">", "value": 15.0},
                                                   {"column": "rmag", "op": ">", "value": 15.0}],
                                       client=_FakeNSCClient(), result_store=DatalabResultStore(enable_disk_cache=False),
                                       plotting_service=_MemoryPlottingService())
    note = out["depth"]["note"]
    assert "bound applied" not in note and "does not show where the catalogue becomes incomplete" in note


def test_cx09_verify_failed_abort_blocks_the_duplicate_scan(monkeypatch):
    from capabilities import datalab_tools
    from capabilities.base import CallContext
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore

    class _Client:
        def abort(self, jobid):
            raise ConnectionError("down")

    monkeypatch.setenv("DATALAB_HEALPIX_ASYNC", "1")
    monkeypatch.setattr(orch, "async_density_aggregate", lambda *a, **k: {
        "success": False, "jobid": "job-u", "job_state": "UNKNOWN", "error": "status polling failed 3 times"})
    monkeypatch.setattr(orch, "tiled_density_aggregate",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no duplicate scan")))
    cap = datalab_tools.HealpixDensityMap()
    out = cap.run(cap.InputModel(preset="south_gradient"),
                  CallContext(services={"datalab_client": _Client()}, result_store=DatalabResultStore(enable_disk_cache=False))).to_native()
    assert out["success"] is False and out["async_job"]["abort"].startswith("abort failed")


def test_cx12_verify_dollar_quoted_strings_are_untouched():
    from integrations.datalab_client import DatalabClient

    sql = "SELECT $$'Infinity'::float8$$ AS s, $t$x 'NaN'::real$t$ AS u FROM t WHERE x < 'Infinity'::float8"
    out = DatalabClient.strip_materialized_for_async(sql)
    assert "$$'Infinity'::float8$$" in out and "$t$x 'NaN'::real$t$" in out and out.endswith("x < 'Infinity'")


def test_cx16_background_job_states_are_honest():
    from capabilities.datalab_tools import SatelliteSearch

    class _T:
        def is_alive(self):
            return False

        def join(self, t=None):
            pass

    from services.tool_budgets import Deadline
    done = {"thread": _T(), "deadline": Deadline(10), "box": {"job": {"state": "COMPLETED", "jobid": "j1"}, "meta": {}, "validated_sql": ""},
            "started": 0.0}
    rec, _ = SatelliteSearch()._finish_background_job(None, done, {"result_id": "r", "partial": False}, reserve=0)
    assert rec["state"] == "COMPLETED" and "was not needed" in rec["result_source"]
    lost = {"thread": _T(), "deadline": Deadline(10), "box": {"meta": {}, "validated_sql": ""}, "started": 0.0}
    rec2, _ = SatelliteSearch()._finish_background_job(None, lost, {"result_id": "r", "partial": False}, reserve=0)
    assert rec2["state"] == "ABORT REQUESTED (not confirmed)" and "aborted" not in rec2["result_source"]


# ── verify round 2 residuals (fixed without a further Codex round) ─────────
def test_cx17_double_quoted_identifiers_are_untouched():
    from integrations.datalab_client import DatalabClient

    sql = "SELECT \"col 'Infinity'::float8\" FROM t WHERE x < 'Infinity'::float8"
    out = DatalabClient.strip_materialized_for_async(sql)
    assert "\"col 'Infinity'::float8\"" in out and out.endswith("x < 'Infinity'")


def test_cx18_unaborted_job_is_not_a_coverage_gap(monkeypatch):
    from capabilities import datalab_tools
    from capabilities.base import CallContext
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore

    class _Client:
        def abort(self, jobid):
            raise ConnectionError("down")

    monkeypatch.setenv("DATALAB_HEALPIX_ASYNC", "1")
    monkeypatch.setattr(orch, "async_density_aggregate", lambda *a, **k: {
        "success": False, "jobid": "job-u", "job_state": "UNKNOWN", "error": "status polling failed"})
    cap = datalab_tools.HealpixDensityMap()
    out = cap.run(cap.InputModel(preset="south_gradient"),
                  CallContext(services={"datalab_client": _Client()}, result_store=DatalabResultStore(enable_disk_cache=False))).to_native()
    assert out["status"] == "job_state_unknown"


def test_cx07_capped_note_names_no_bound_without_one():
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore
    from tests.unit.test_datalab_p1 import _MemoryPlottingService
    from tests.unit.test_reb_science_correctness import _FakeNSCClient

    out = orch.color_magnitude_diagram("nsc_dr2", "object", 260.06, 57.92, 0.4, limit=150,
                                       value_cuts=[{"column": "gmag", "op": ">", "value": 15.0}, {"column": "rmag", "op": ">", "value": 15.0}],
                                       client=_FakeNSCClient(), result_store=DatalabResultStore(enable_disk_cache=False),
                                       plotting_service=_MemoryPlottingService())
    assert out["depth"]["sample_capped"] and "bound applied" not in out["depth"]["note"]
