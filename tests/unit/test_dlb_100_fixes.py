"""DataLabBench push toward 100 (handoff 2026-09-24, work items F1-F12).

Each section names the work item and the DataLabBench question it targets.
All tests are offline: Data Lab is replaced by fake clients.
"""

import numpy as np
import pandas as pd
import pytest

from services import datalab_orchestration as orch
from services import datalab_registry as reg
from services.datalab_result_store import DatalabResultStore


def _plotting():
    from tests.unit.test_datalab_p1 import _MemoryPlottingService
    return _MemoryPlottingService()


# ─────────────────────────────────────────────────────────────────────────────
# F2 (L05): star/galaxy split IN the SQL + DES quality cuts
# ─────────────────────────────────────────────────────────────────────────────
class _FakeDESMorphClient:
    """Returns the columns the executed SQL selects, including the CASE column."""

    def __init__(self):
        self.sql = []

    def query(self, *, sql=None, adql=None, fmt="pandas", **kwargs):
        from integrations.datalab_client import DatalabResult
        self.sql.append(sql or "")
        rng = np.random.default_rng(7)
        n = 200
        sm = rng.normal(0.0, 0.004, n)
        df = pd.DataFrame({
            "ra": rng.uniform(29.5, 30.5, n), "dec": rng.uniform(-50.5, -49.5, n),
            "mag_auto_g": rng.uniform(18, 23, n), "mag_auto_r": rng.uniform(17, 22, n),
            "mag_auto_i": rng.uniform(16.5, 22.5, n), "spread_model_r": sm,
        })
        if "AS morph" in (sql or ""):
            df["morph"] = np.where(np.abs(sm) <= 0.003, "star", "galaxy")
        return DatalabResult.from_dataframe(df, {"catalog": "des_dr1", "table": "main", "query": sql})


def test_f2_ccd_sql_carries_case_split_and_all_three_quality_groups():
    client = _FakeDESMorphClient()
    out = orch.color_color_diagram(
        "des_dr1", "main", 30.0, -50.0, 0.5,
        client=client, result_store=DatalabResultStore(enable_disk_cache=False),
        plotting_service=_plotting(),
    )
    assert out["success"] is True
    sql = client.sql[0]
    assert "CASE WHEN spread_model_r BETWEEN -0.003 AND 0.003 THEN 'star'" in sql
    assert "END AS morph" in sql
    for col in ("flags_g", "flags_r", "flags_i"):
        assert f"{col} = 0" in sql
    for col in ("fluxerr_auto_g", "fluxerr_auto_r", "fluxerr_auto_i"):
        assert f"{col} > 0" in sql
    assert "mag_auto_i > 16" in sql and "mag_auto_i < 23" in sql
    cuts = " | ".join(out["cuts_applied"])
    assert "fluxerr_auto_i > 0" in cuts and "mag_auto_i > 16" in cuts and "AS morph" in cuts
    assert out["split_sql"].startswith("CASE WHEN spread_model_r")
    # The panels follow the server's morph column.
    pops = {p["population"]: p["n"] for p in out["populations"]}
    assert pops["stars"] > 0 and pops["galaxies"] > 0


def test_f2_caller_magnitude_cut_overrides_the_window():
    client = _FakeDESMorphClient()
    orch.color_color_diagram(
        "des_dr1", "main", 30.0, -50.0, 0.5,
        value_cuts=[{"column": "mag_auto_i", "op": "<", "value": 21.0}],
        client=client, result_store=DatalabResultStore(enable_disk_cache=False),
        plotting_service=_plotting(),
    )
    sql = client.sql[0]
    assert "mag_auto_i < 21" in sql
    assert "mag_auto_i > 16" not in sql and "mag_auto_i < 23" not in sql


def test_f2_split_case_sql_conventions():
    assert orch._split_case_sql("ext_coadd", 0.0).startswith("CASE WHEN ext_coadd IN (0, 1) THEN 'star'")
    assert "class_star > 0.5 THEN 'star'" in orch._split_case_sql("class_star", 0.5)
    with pytest.raises(ValueError):
        orch._split_case_sql("x; DROP TABLE y", 0.1)


def test_f1a_cutout_grid_fetches_panels_concurrently_with_per_panel_timeout(monkeypatch):
    import threading
    import time as _time

    from services.datalab_image_service import DatalabImageService

    monkeypatch.setenv("DATALAB_CUTOUT_GRID_WORKERS", "4")
    monkeypatch.setenv("DATALAB_CUTOUT_PANEL_SECONDS", "1.5")
    svc = DatalabImageService.__new__(DatalabImageService)
    svc.plotting_service = _plotting()
    live = {"now": 0, "max": 0}
    lock = threading.Lock()

    def fake_search(ra, dec, fov, **kwargs):
        with lock:
            live["now"] += 1
            live["max"] = max(live["max"], live["now"])
        try:
            _time.sleep(5.0 if ra == 3.0 else 0.4)  # panel 3 exceeds its panel timeout
        finally:
            with lock:
                live["now"] -= 1
        return {"coverage_gap": True, "rows": [], "used_endpoint": "sia"}

    svc.search = fake_search
    t0 = _time.monotonic()
    out = svc.cutout_grid([{"ra": float(i), "dec": 0.0} for i in range(5)], 0.05, band="g")
    elapsed = _time.monotonic() - t0
    assert live["max"] >= 3                    # panels overlapped
    assert elapsed < 4.0                       # serial would be >= 6.6 s
    assert out["panels"][3].get("timeout") is True
    assert sum(1 for p in out["panels"] if p.get("timeout")) == 1
    assert out["panel_timeouts"] == 1


def test_f11_wedge_radial_profile_finds_the_wall():
    from services.datalab_analysis import radial_density_profile

    rng = np.random.default_rng(1)
    d = rng.uniform(50, 450, 4000)
    wall = rng.normal(330, 4, 600)          # a wall at ~330 Mpc
    dist = np.concatenate([d, wall])
    z = dist / 4300.0
    prof = radial_density_profile(dist, z)
    lo, hi = prof["most_overdense_shell"]["distance_mpc"]
    assert 320 <= lo <= 340 and hi - lo == 10
    assert prof["most_overdense_shell"]["overdensity_vs_neighbours"] > 1.5


def test_f2_ccd_window_only_registered_for_des():
    assert reg.ccd_magnitude_window("des_dr1", "main") == {"column": "mag_auto_i", "min": 16.0, "max": 23.0}
    assert reg.ccd_magnitude_window("nsc_dr2", "object") is None


# ─────────────────────────────────────────────────────────────────────────────
# F1 (L15) + F8 (L12) + F11 (L07): satellite search vetting, background job,
# counts next to significance, method in plain units
# ─────────────────────────────────────────────────────────────────────────────
from services import cmd_population as cp  # noqa: E402


def _hess_frame(*, old_excess: bool, ratio: float, n_out_per_cell: float = 40.0):
    """Synthetic Hess aggregate: a smooth field in every (colour, mag) cell,
    plus (optionally) an old-population excess in the turnoff / RGB / BHB boxes."""
    rows = []
    for cbin in range(-6, 18):          # g-r = -0.6 .. 1.8
        for mbin in range(64, 102):     # g = 16 .. 25.5
            color, mag = (cbin + 0.5) * cp.COLOR_BIN, (mbin + 0.5) * cp.MAG_BIN
            n_out = n_out_per_cell
            n_in = n_out * ratio
            if old_excess:
                if 0.0 <= color < 0.45 and mag >= 22.5:
                    n_in += 6.0      # turnoff / upper MS
                elif 0.45 <= color < 1.0 and 18.0 <= mag < 22.0:
                    n_in += 3.0      # RGB
                elif -0.4 <= color < 0.0 and 20.5 <= mag < 21.5:
                    n_in += 4.0      # BHB
            rows.append({"cbin": cbin, "mbin": mbin, "n_in": round(n_in), "n_out": n_out})
    return pd.DataFrame(rows)


def test_f1c_population_test_old_population_vs_field():
    ratio = cp.area_ratio(0.1, (0.25, 0.5))
    old = cp.population_test(_hess_frame(old_excess=True, ratio=ratio))
    assert old["verdict"] == "coherent old population"
    assert {"main-sequence turnoff / blue main sequence", "red-giant branch", "blue horizontal branch"} <= set(old["features_detected"])
    field = cp.population_test(_hess_frame(old_excess=False, ratio=ratio))
    assert field["verdict"] == "field-like"
    assert abs(field["old_population_significance"]) < 2.5
    empty = cp.population_test(pd.DataFrame())
    assert empty["verdict"] == "inconclusive"


def test_f1c_red_excess_is_not_an_old_population():
    ratio = cp.area_ratio(0.1, (0.25, 0.5))
    frame = _hess_frame(old_excess=False, ratio=ratio)
    red = (frame["cbin"] >= 10) & (frame["cbin"] < 18) & (frame["mbin"] >= 70)
    frame.loc[red, "n_in"] = frame.loc[red, "n_in"] + 4
    out = cp.population_test(frame)
    assert out["verdict"] == "field-like" and "red" in out["reason"]


def test_f1c_hess_sql_is_one_bounded_aggregate():
    sql, meta = cp.build_hess_aggregate("nsc_dr2", "object", 185.43, -31.99, predicates=["class_star > 0.5"])
    assert "q3c_radial_query(ra, dec, 185.43, -31.99, 0.5)" in sql and "GROUP BY cbin, mbin" in sql
    assert "q3c_dist(ra, dec, 185.43, -31.99) <= 0.1" in sql and ">= 0.25" in sql
    assert meta["aggregate"] is True
    with pytest.raises(ValueError):
        cp.build_hess_aggregate("nsc_dr2", "object", 1.0, 2.0, aperture_deg=0.3, annulus_deg=(0.25, 0.5))


class _SatClient:
    """Fake Data Lab: a density grid with one peak, clean LS masks, and an
    old-population Hess aggregate at every candidate."""

    def __init__(self, token="anonymous.0.0.anon_access"):
        self.token = token
        self.sql = []
        self.timeout = 60.0

    def query(self, *, sql=None, fmt="pandas", **kwargs):
        from integrations.datalab_client import DatalabResult
        self.sql.append(sql or "")
        if "AS cbin" in sql:
            df = _hess_frame(old_excess=True, ratio=cp.area_ratio(0.1, (0.25, 0.5)))
        elif "maskbits" in sql:
            df = pd.DataFrame({"n": [120], "n_galaxy": [0], "n_bright": [0]})
        else:
            df = _grid_frame()
        return DatalabResult.from_dataframe(df, {"query": sql})


def _grid_frame(ra0=185.425, dec0=-31.985, step=0.05, n=20):
    rows = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            c = 20
            if abs(i) <= 1 and abs(j) <= 1:
                c += 25
            rows.append({"ra_bin": round(ra0 + i * step, 4), "dec_bin": round(dec0 + j * step, 4), "source_count": c})
    return pd.DataFrame(rows)


class _Grid:
    def cutout_grid(self, peaks, fov, **k):
        return {"success": True, "image_base64": "aGk=", "panels": [{"label": p["label"]} for p in peaks]}


def _sat_ctx(client, cards):
    from capabilities.base import CallContext
    return CallContext(services={"datalab_client": client, "datalab_image_service": _Grid(),
                                 "append_run_result": cards.append},
                       result_store=DatalabResultStore(enable_disk_cache=False))


def _patch_run_builder(monkeypatch, client):
    def fake_run(sql, meta, *, client=None, result_store=None, owner_id=None, **kw):
        res = client.query(sql=sql)
        return result_store.put(res.dataframe, {"provenance": {}}), res
    monkeypatch.setattr(orch, "_run_builder_sql", fake_run)


def test_f1_satellite_search_attaches_population_verdicts_and_counts(monkeypatch):
    from capabilities import datalab_tools

    monkeypatch.setattr(datalab_tools, "_run_analysis_plot",
                        lambda name, args, ctx: type("R", (), {"to_native": lambda self: {"success": False}})())
    client = _SatClient()
    _patch_run_builder(monkeypatch, client)
    cards = []
    cap = datalab_tools.SatelliteSearch()
    out = cap.run(cap.InputModel(survey="nsc", preset="hydra2"), _sat_ctx(client, cards)).to_native()
    assert out["success"] is True and out["candidates"]
    top = out["candidates"][0]
    # F8: raw counts next to significance, in the candidate and the headline
    assert top["aperture_count"] > top["background_in_aperture"] > 0
    assert f"{top['aperture_count']} stars in the aperture" in out["headline"]
    # F1c: the CMD test ran and its verdict reached the candidate and the summary
    assert top["population_verdict"] == "coherent old population"
    assert top["population_test"]["old_population_significance"] >= 4
    assert "CMD population test per rank: #1: coherent old population" in out["vetting_summary"]
    assert any("Hess" in c.get("caption", "") or "CMD" in c.get("caption", "") for c in cards)
    assert any("AS cbin" in s for s in client.sql)
    # F11 (L07): the method in plain units
    m = out["method"]
    assert m["cell_size_arcmin"] == 3.0 and m["aperture_radius_arcmin"] == 3.3
    assert m["background_annulus_arcmin"] == [15.0, 36.0] and "sqrt" in m["significance"]


def test_f1b_scan_budget_keeps_the_vetting_reserve(monkeypatch):
    from capabilities import datalab_tools
    from services.tool_budgets import begin_tool_deadline, end_tool_deadline

    seen = {}

    def fake_tiled(*a, **k):
        seen["max_seconds"] = k.get("max_seconds")
        return {"result_id": None, "error": "stop"}

    monkeypatch.setattr(orch, "tiled_density_aggregate", fake_tiled)
    monkeypatch.setenv("DATALAB_SATELLITE_ASYNC", "0")
    cap = datalab_tools.SatelliteSearch()
    begin_tool_deadline("datalab_satellite_search", 215.0)   # inner 200 s
    try:
        cap.run(cap.InputModel(survey="delve", preset="delve_south"), _sat_ctx(_SatClient(), []))
    finally:
        end_tool_deadline()
    assert seen["max_seconds"] == pytest.approx(200.0 - cap._VET_RESERVE_S - 8.0, abs=1.5)


def _fake_async(result_frame=None, delay=0.0):
    def run(sql, *, client=None, max_seconds=0, poll_seconds=5.0, on_event=None):
        import time as _t

        from services.tool_budgets import is_cancelled
        t0 = _t.monotonic()
        hist = [{"t_s": 0.0, "state": "SUBMITTED"}, {"t_s": 0.1, "state": "EXECUTING"}]
        for h in hist:
            if on_event:
                on_event({**h, "jobid": "job-7"})
        while _t.monotonic() - t0 < max(delay, 0.0):
            if is_cancelled():
                return {"state": "CANCELLED", "jobid": "job-7", "status_history": hist, "elapsed_s": 0.2,
                        "error": "turn cancelled; async job aborted"}
            _t.sleep(0.02)
        if result_frame is None:
            return {"state": "RUNNING", "jobid": "job-7", "status_history": hist, "elapsed_s": delay}
        from integrations.datalab_client import DatalabResult
        return {"state": "COMPLETED", "jobid": "job-7", "status_history": hist + [{"t_s": delay, "state": "COMPLETED"}],
                "elapsed_s": delay, "executed_sql": sql, "result": DatalabResult.from_dataframe(result_frame, {})}
    return run


def test_f1d_tiles_finish_first_and_the_background_job_is_aborted(monkeypatch):
    from capabilities import datalab_tools

    store_holder = {}

    def fake_tiled(*a, result_store=None, **k):
        rid = result_store.put(_grid_frame(30.0, -50.0), {"provenance": {}})
        store_holder["rid"] = rid
        return {"success": True, "result_id": rid, "partial": False, "warnings": []}

    monkeypatch.setattr(orch, "tiled_density_aggregate", fake_tiled)
    monkeypatch.setattr(orch, "run_async_sql", _fake_async(delay=30.0))
    monkeypatch.setattr(datalab_tools, "_run_analysis_plot",
                        lambda name, args, ctx: type("R", (), {"to_native": lambda self: {"success": False}})())
    client = _SatClient(token="real.login.token")
    _patch_run_builder(monkeypatch, client)
    cap = datalab_tools.SatelliteSearch()
    import time as _t
    t0 = _t.monotonic()
    out = cap.run(cap.InputModel(survey="delve", preset="delve_south", vet=False), _sat_ctx(client, [])).to_native()
    assert _t.monotonic() - t0 < 10.0          # the job never slows the turn
    job = out["background_job"]
    assert job["jobid"] == "job-7" and job["state"].startswith("ABORT REQUESTED") and job["used"] is False
    assert job["result_source"].startswith("synchronous tiles (finished first")
    assert [h["state"] for h in job["status_history"]][:2] == ["SUBMITTED", "EXECUTING"]
    assert out["result_id"] == store_holder["rid"]


def test_f1d_partial_tiles_use_the_completed_background_job(monkeypatch):
    from capabilities import datalab_tools

    def fake_tiled(*a, result_store=None, **k):
        rid = result_store.put(_grid_frame(30.0, -50.0, n=5), {"provenance": {}})
        return {"success": True, "result_id": rid, "partial": True, "warnings": ["Partial map: 3 of 9 tiles"]}

    monkeypatch.setattr(orch, "tiled_density_aggregate", fake_tiled)
    monkeypatch.setattr(orch, "run_async_sql", _fake_async(result_frame=_grid_frame(30.0, -50.0), delay=0.3))
    monkeypatch.setattr(datalab_tools, "_run_analysis_plot",
                        lambda name, args, ctx: type("R", (), {"to_native": lambda self: {"success": False}})())
    client = _SatClient(token="real.login.token")
    _patch_run_builder(monkeypatch, client)
    cap = datalab_tools.SatelliteSearch()
    out = cap.run(cap.InputModel(survey="delve", preset="delve_south", vet=False), _sat_ctx(client, [])).to_native()
    job = out["background_job"]
    assert job["used"] is True and job["state"] == "COMPLETED"
    assert job["result_source"].startswith("background job")
    assert out["status"] == "ok" and out["background"]["cells"] == len(_grid_frame(30.0, -50.0))


def test_f1d_anonymous_token_reports_no_job_honestly(monkeypatch):
    from capabilities import datalab_tools

    monkeypatch.setattr(orch, "tiled_density_aggregate", lambda *a, **k: {"result_id": None, "error": "stop"})
    cap = datalab_tools.SatelliteSearch()
    out = cap.run(cap.InputModel(survey="delve", preset="delve_south"), _sat_ctx(_SatClient(), [])).to_native()
    assert out["background_job"]["state"] == "UNAVAILABLE" and "anonymous" in out["background_job"]["note"]


# ─────────────────────────────────────────────────────────────────────────────
# F3 (L03): CMD depth bound in the SQL + depth block
# ─────────────────────────────────────────────────────────────────────────────
def test_f3_nsc_cmd_applies_the_depth_bound_and_reports_the_turnover():
    from tests.unit.test_reb_science_correctness import _FakeNSCClient

    client = _FakeNSCClient()
    out = orch.color_magnitude_diagram(
        "nsc_dr2", "object", 260.06, 57.92, 0.4, blue_band="g", red_band="r", point_sources=True,
        client=client, result_store=DatalabResultStore(enable_disk_cache=False), plotting_service=_plotting(),
    )
    sql = client.sql[0]
    assert "gmag < 24" in sql and "rmag < 24" in sql and "class_star > 0.5" in sql
    assert out["depth"]["bound_applied"] == ["gmag < 24", "rmag < 24"]
    # uniform synthetic magnitudes never turn over: no completeness claim (CX-07)
    assert out["depth"]["turnover_mag"] is None and "the bound applied" in out["depth"]["note"]
    # a caller cut on the band wins
    client2 = _FakeNSCClient()
    orch.color_magnitude_diagram(
        "nsc_dr2", "object", 260.06, 57.92, 0.4, blue_band="g", red_band="r",
        value_cuts=[{"column": "gmag", "op": "<", "value": 23.0}],
        client=client2, result_store=DatalabResultStore(enable_disk_cache=False), plotting_service=_plotting(),
    )
    assert "gmag < 23" in client2.sql[0] and "gmag < 24" not in client2.sql[0] and "rmag < 24" in client2.sql[0]


# ─────────────────────────────────────────────────────────────────────────────
# F4 (L04): colour-image outcome facts
# ─────────────────────────────────────────────────────────────────────────────
def test_f4_color_image_gap_explains_reason_fov_and_band_mapping():
    from capabilities.datalab import ColorImage

    gap = {"success": True, "coverage_gap": True, "available_bands": ["z"],
           "provenance": {"color_hips_attempted": "CDS/P/DESI-Legacy-Surveys/DR10/color"}}
    out = ColorImage.explain_outcome(gap, catalog="ls_dr9", ra=10.6847, dec=41.2688, fov_deg=0.1, label="M31",
                                     q=8.0, stretch=0.5)
    assert out["bands_found"] == ["z"]
    assert "only in band(s) z" in out["coverage_reason"] and "northern edge of DECam" in out["coverage_reason"]
    assert "coverage gap, not a processing error" in out["coverage_reason"]
    assert "D25 = 190 arcmin" in out["fov_rationale"] and "bulge and nucleus" in out["fov_rationale"]
    assert out["rgb_mapping"]["red"] == "z" and out["rgb_mapping"]["green"] == "r" and out["rgb_mapping"]["blue"] == "g"
    assert out["rgb_mapping"]["stretch"]["Q"] == 8.0
    assert any("Pan-STARRS1" in a for a in out["alternatives"])
    assert "footprint_note" in out


def test_f4_color_image_success_reports_the_used_mapping():
    from capabilities.datalab import ColorImage

    ok = {"success": True, "coverage_gap": False, "bands_used": ["z", "r", "g"],
          "provenance": {"lupton_rgb_order": "z,r,g"}}
    out = ColorImage.explain_outcome(ok, catalog="ls_dr9", ra=150.0, dec=2.0, fov_deg=0.2, label="COSMOS",
                                     q=8.0, stretch=0.5)
    assert out["rgb_mapping"]["state"] == "used" and out["rgb_mapping"]["red"] == "z"
    assert "coverage_reason" not in out and "0.2 deg" in out["fov_rationale"]


# ─────────────────────────────────────────────────────────────────────────────
# F5 (L08): HEALPix pixel polygons + robust colour limits
# ─────────────────────────────────────────────────────────────────────────────
def test_f5_healpix_polygons_count_and_ra_wrap():
    from astropy_healpix import HEALPix

    from services.datalab_analysis import _healpix_polygons

    hp = HEALPix(nside=64, order="ring")
    import astropy.units as u
    pix = hp.cone_search_lonlat(0.5 * u.deg, -30 * u.deg, radius=3 * u.deg)   # straddles RA 0/360
    verts, ref = _healpix_polygons(pix, 64, "ring")
    assert len(verts) == len(pix) and all(v.shape == (4, 2) for v in verts)
    ras = np.concatenate([v[:, 0] for v in verts])
    assert ras.max() - ras.min() < 10.0            # contiguous, not split across 0/360
    assert ref < 5.0 or ref > 355.0


def test_f5_sky_density_map_draws_pixel_polygons_with_robust_limits():
    from astropy_healpix import HEALPix
    import astropy.units as u

    from services import datalab_analysis as da

    hp = HEALPix(nside=64, order="ring")
    pix = hp.cone_search_lonlat(75 * u.deg, -30 * u.deg, radius=5 * u.deg)
    counts = np.linspace(50, 150, len(pix)).round()
    counts[0] = 50000                                  # one cluster pixel
    store = DatalabResultStore(enable_disk_cache=False)
    rid = store.put(pd.DataFrame({"healpix": pix, "source_count": counts}),
                    {"provenance": {"healpix": {"column": "ring256", "nside": 64, "scheme": "RING"}}})
    out = da.sky_density_map(rid, mode="healpix", healpix_col="healpix", log_scale=True, result_store=store,
                             plotting_service=_plotting())
    assert out["pixel_geometry"] == "healpix_polygons"
    assert out["color_limits"]["vmax"] < 1000         # the cluster pixel does not set the top of the scale


# ─────────────────────────────────────────────────────────────────────────────
# F7 (L10): datalab_sed_sample builds the rubric-like single-table query
# ─────────────────────────────────────────────────────────────────────────────
def test_f7_sed_sample_sql_has_extended_snr_and_colour_cuts():
    from capabilities.datalab_tools import SedSample

    sql, meta, applied = SedSample.build_sql(194.95, 27.98, 1.0, gr_min=0.8, rz_min=0.5, snr_min=5.0,
                                            snr_wise_min=3.0, extended_only=True, limit=300)
    assert "FROM ls_dr9.tractor" in sql and "q3c_radial_query(ra, dec, 194.95, 27.98, 1)" in sql
    assert "type != 'PSF'" in sql
    for b in ("g", "r", "z"):
        assert f"snr_{b} > 5" in sql
    assert "snr_w1 > 3" in sql and "snr_w2 > 3" in sql
    assert "(dered_mag_g - dered_mag_r) >= 0.8" in sql and "(dered_mag_r - dered_mag_z) >= 0.5" in sql
    assert "dered_mag_w1" in sql and "dered_mag_w2" in sql and "JOIN" not in sql.upper()
    assert sql.rstrip().endswith("LIMIT 300")      # the user's "few hundred", not a flagged platform cap
    assert any("type != 'PSF'" in a for a in applied)


# ─────────────────────────────────────────────────────────────────────────────
# F9 (L14): light-curve shape + a marked DECam field cutout
# ─────────────────────────────────────────────────────────────────────────────
def _rrab(phase):
    """Sawtooth: rise over 0.15 of the cycle, slow linear decline (1 mag)."""
    ph = np.asarray(phase) % 1.0
    return np.where(ph < 0.15, 18.0 - (ph / 0.15), 17.0 + (ph - 0.15) / 0.85)


def test_f9_shape_classifies_rrab_sawtooth_and_sinusoid():
    from services.datalab_analysis import light_curve_shape

    rng = np.random.default_rng(3)
    ph = rng.uniform(0, 1, 200)
    saw = light_curve_shape(ph, _rrab(ph) + rng.normal(0, 0.02, ph.size), period_days=0.65)
    assert saw["classification"].startswith("RRab-like sawtooth") and saw["rise_fraction"] <= 0.3
    assert saw["amplitude_mag"] > 0.8
    sine = light_curve_shape(ph, 17.5 + 0.25 * np.sin(2 * np.pi * ph) + rng.normal(0, 0.01, ph.size), period_days=0.33)
    assert "near-sinusoidal" in sine["classification"]
    assert light_curve_shape(ph[:5], ph[:5])["classification"].startswith("not assessed")


def test_f9_l14_route_asks_for_a_marked_decam_cutout():
    from core.oneshot_routing import detect_oneshot_intent

    q = ("I have a candidate variable star at RA = 185.4311, Dec = −31.9953 (an RR Lyrae in the Hydra II field). Find its "
         "multi-epoch SMASH photometry, phase-fold the light curve to get the period, and pull an image cutout of the field.")
    route = detect_oneshot_intent(q)
    assert route["tool"] == "datalab_star_lightcurve"
    assert route["args"] == {"catalog": "smash_dr1", "ra": 185.4311, "dec": -31.9953}
    assert "`datalab_image_cutout`" in route["directive"] and "not DSS" in route["directive"]
    assert "sawtooth" in route["directive"]


# ─────────────────────────────────────────────────────────────────────────────
# F6 (L09): tail detection beyond the scatter plot
# ─────────────────────────────────────────────────────────────────────────────
def test_f6_tail_test_finds_an_injected_tail_and_not_a_uniform_field():
    from capabilities.datalab_tools import StreamSelection

    rng = np.random.default_rng(5)
    ra0, dec0 = 229.022, -0.112
    n = 3000
    field = pd.DataFrame({"ra": ra0 + rng.uniform(-5, 5, n), "dec": dec0 + rng.uniform(-5, 5, n)})
    none = StreamSelection.tail_test(field, ra0, dec0)
    assert none["verdict"].startswith("no significant elongation")
    # inject a tail along PA 40 deg (east of north), both sides, 0.5-4 deg
    s = rng.uniform(0.5, 4.0, 400) * rng.choice([-1, 1], 400)
    t = np.radians(40.0)
    tail = pd.DataFrame({"ra": ra0 + s * np.sin(t) + rng.normal(0, 0.08, 400),
                         "dec": dec0 + s * np.cos(t) + rng.normal(0, 0.08, 400)})
    got = StreamSelection.tail_test(pd.concat([field, tail]), ra0, dec0)
    assert got["verdict"].startswith("tail-like elongation") and abs(got["best_pa_deg_east_of_north"] - 40.0) <= 10.0
    assert got["n_in_best_strip"] > got["n_in_perpendicular_strip"] and got["significance_vs_perpendicular"] >= 3


def test_f6_stream_output_explains_the_execution_model():
    import inspect

    from capabilities import datalab_tools

    src = inspect.getsource(datalab_tools.StreamSelection.run)
    assert '"execution_model"' in src and "MATERIALIZED CTE" in src and '"tail_test": tails' in src


def test_f1_strong_peak_with_field_like_cmd_is_not_a_new_candidate(monkeypatch):
    from capabilities import datalab_tools
    from integrations.datalab_client import DatalabResult

    class _FieldClient(_SatClient):
        def query(self, *, sql=None, fmt="pandas", **kwargs):
            self.sql.append(sql or "")
            if "AS cbin" in sql:
                df = _hess_frame(old_excess=False, ratio=cp.area_ratio(0.1, (0.25, 0.5)))
            elif "maskbits" in sql:
                df = pd.DataFrame({"n": [120], "n_galaxy": [0], "n_bright": [0]})
            else:
                df = _grid_frame(20.0, -40.0)
                hot = ((df.ra_bin - 20.0).abs() < 0.06) & ((df.dec_bin + 40.0).abs() < 0.06)
                df.loc[hot, "source_count"] += 60
            return DatalabResult.from_dataframe(df, {"query": sql})

    monkeypatch.setattr(datalab_tools, "_run_analysis_plot",
                        lambda name, args, ctx: type("R", (), {"to_native": lambda self: {"success": False}})())
    client = _FieldClient()
    _patch_run_builder(monkeypatch, client)
    cap = datalab_tools.SatelliteSearch()
    out = cap.run(cap.InputModel(survey="nsc", region={"ra": 20.0, "dec": -40.0, "radius": 1.0}), _sat_ctx(client, [])).to_native()
    top = out["candidates"][0]
    assert top["significance"] >= 5 and top["population_verdict"] == "field-like"
    assert top["verdict"].startswith("rejected by the CMD test")
    assert out["n_candidates_over_5sigma"] == 0 and "rejected by a field-like CMD" in out["headline"]


def test_f1d_async_grid_sql_uses_positional_group_by():
    from capabilities.datalab_tools import SatelliteSearch
    from services import datalab_query_builders as B

    sql, _ = B.build_density_aggregate("delve_dr3", "coadd_objects", mode="grid", step_deg=0.05, ra=30, dec=-50,
                                       radius_deg=5, predicates=["ext_coadd BETWEEN 0 AND 1"], max_cells=40000)
    out = SatelliteSearch.async_grid_sql(sql)
    assert "GROUP BY 1, 2" in out and "ORDER BY 3 DESC" in out and "GROUP BY ra_bin" not in out
    assert "AS ra_bin" in out and "AS dec_bin" in out          # the result columns keep their names


# ─────────────────────────────────────────────────────────────────────────────
# Found in the 2026-09-24 UI run 1 (fixed before the domain suite and run 2)
# ─────────────────────────────────────────────────────────────────────────────
def test_ui_l02_rounded_radius_is_not_flagged():
    from core.answer_verifier import build_trace_summary, verify_answer

    sql = "SELECT COUNT(*) AS row_count FROM gaia_dr3.gaia_source WHERE q3c_radial_query(ra, dec, 229.022, -0.112, 0.1666667)"
    ts = build_trace_summary([{"output": '{"success": true, "row_count": 1313}'}], [], [], extra_sql=[sql])
    rep = verify_answer("The cone count used a radius of 0.1667 deg (10 arcmin) and found 1 313 sources.", ts)
    assert not rep.by_kind("cut")
    assert verify_answer("The cone count used a radius of 0.25 deg.", ts).by_kind("cut")


def test_ui_l03_selection_diagram_applies_point_sources_and_sentinel_guard(monkeypatch):
    from capabilities import datalab_tools
    from capabilities.base import CallContext

    seen = {}

    class _Stop(Exception):
        pass

    def fake_run(sql, meta, **kw):
        seen["sql"] = sql
        raise _Stop()

    monkeypatch.setattr(orch, "_run_builder_sql", fake_run)
    cap = datalab_tools.SelectionDiagram()
    ctx = CallContext(services={"datalab_client": object(), "resolve_coordinates": lambda target_name=None, ra=None, dec=None: (ra, dec, "x")},
                      result_store=DatalabResultStore(enable_disk_cache=False))
    cap.run(cap.InputModel(catalog="nsc_dr2", ra=260.06, dec=57.92, radius_deg=0.4, x_expr="gmag - rmag", y_expr="gmag",
                           point_sources=True, quality_preset="none"), ctx)
    sql = seen["sql"]
    assert "class_star > 0.5" in sql
    assert "gmag > -5" in sql and "gmag < 50" in sql and "rmag < 50" in sql


def test_ui_l06_interactive_card_shows_the_wd_locus():
    from tests.unit.test_datalab_p1 import _MemoryPlottingService
    from integrations.datalab_client import DatalabResult

    class _G:
        def query(self, *, sql=None, **k):
            rng = np.random.default_rng(1)
            n = 300
            df = pd.DataFrame({"ra": np.full(n, 60.0), "dec": np.full(n, -50.0), "bp_rp": rng.uniform(-0.2, 3.5, n),
                               "phot_g_mean_mag": rng.uniform(14, 20, n), "parallax": rng.uniform(5, 50, n)})
            return DatalabResult.from_dataframe(df, {"query": sql})

    out = orch.color_magnitude_diagram("gaia_dr3", "gaia_source", 60.0, -50.0, 1.0, x_expr="bp_rp",
                                       y_expr="phot_g_mean_mag + 5*log10(parallax) - 10", overlay_locus="white_dwarf",
                                       client=_G(), result_store=DatalabResultStore(enable_disk_cache=False),
                                       plotting_service=_MemoryPlottingService())
    names = [t.get("name", "") for t in out["plotly_spec"]["data"]]
    assert any(n.startswith("WD candidates") for n in names) and "WD locus" in names
    wd = next(t for t in out["plotly_spec"]["data"] if t.get("name", "").startswith("WD candidates"))
    assert len(wd["x"]) == out["n_wd_candidates"]


def test_ui_l08_polygon_map_ships_no_marker_spec():
    """The UI shows the interactive spec instead of the PNG; a polygon map must ship the PNG only."""
    from astropy_healpix import HEALPix
    import astropy.units as u

    from services import datalab_analysis as da

    hp = HEALPix(nside=64, order="ring")
    pix = hp.cone_search_lonlat(75 * u.deg, -30 * u.deg, radius=5 * u.deg)
    store = DatalabResultStore(enable_disk_cache=False)
    rid = store.put(pd.DataFrame({"healpix": pix, "source_count": np.linspace(50, 150, len(pix))}),
                    {"provenance": {"healpix": {"column": "ring256", "nside": 64, "scheme": "RING"}}})
    out = da.sky_density_map(rid, mode="healpix", healpix_col="healpix", log_scale=True, result_store=store,
                             plotting_service=_plotting())
    assert out["pixel_geometry"] == "healpix_polygons" and "plotly_spec" not in out


def test_ui_l08_median_statistic_is_not_a_count_claim():
    from core.answer_verifier import build_trace_summary, verify_answer

    ts = build_trace_summary([{"output": '{"success": true, "rowcount": 6141, "structure_notes": ["6141 cells; median 282, max 4023 sources per cell"]}'}],
                             [], [], extra_sql=["SELECT ring256 AS healpix, COUNT(*) FROM nsc_dr2.object GROUP BY ring256"])
    assert not verify_answer("Count statistics: median ≈ 282 sources per cell.", ts).by_kind("count")
    assert verify_answer("The map holds 999 sources.", ts).by_kind("count")


def test_ui_d18_paraphrased_key_and_generic_entries_are_supported():
    import json as _json
    from core.answer_verifier import build_trace_summary, verify_answer

    out = {"success": True, "sources_tested": 12, "matched_sources": 4, "alma_status_unknown_for": 0,
           "headline": "4 of 12 sources have data in ALL of ALMA, JWST"}
    ts = build_trace_summary([{"output": _json.dumps(out)}], [], [], extra_sql=["SELECT COUNT(*) FROM ivoa.obscore"])
    text = ("The cross-match examined the Perseus catalog (12 entries). "
            "ALMA status unknown for 0 sources, all ALMA queries completed.")
    assert not verify_answer(text, ts).by_kind("count")
    assert verify_answer("The catalog has 13 entries.", ts).by_kind("count")


def test_ui_d20_protostar_class_is_not_a_count():
    from core.answer_verifier import build_trace_summary, verify_answer

    ts = build_trace_summary([{"output": '{"success": true, "n": 10}'}], [], [], extra_sql=["SELECT 1 FROM x"])
    assert not verify_answer("It studies the jet of the Class 0 source IRAS 04166+2706.", ts).by_kind("count")


def test_ui_l11_run2_synonym_count_key_supports_objects():
    import json as _json
    from core.answer_verifier import build_trace_summary, verify_answer

    hist = [{"z_bin": 0.40, "source_count": 64857}, {"z_bin": 0.78, "source_count": 108444}]
    ts = build_trace_summary([{"output": _json.dumps({"success": True, "histogram": hist})}], [], [],
                             extra_sql=["SELECT FLOOR(z / 0.02) AS z_bin, COUNT(*) AS source_count FROM t GROUP BY z_bin"])
    text = "The largest bin holds 108 444 objects in the 0.78 - 0.80 bin."
    assert not verify_answer(text, ts).by_kind("count")
    assert verify_answer("The largest bin holds 108 445 objects.", ts).by_kind("count")
    # A synonym never crosses grains: 19 observations are not 19 projects.
    ts2 = build_trace_summary([{"output": '{"success": true, "n_observations": 19}'}], [], [], extra_sql=["SELECT 1 FROM x"])
    assert verify_answer("The query returned 19 projects.", ts2).by_kind("count")


def test_ui_l13_run2_wedge_reports_slice_count_and_thinning(monkeypatch):
    from types import SimpleNamespace
    from capabilities import datalab as cap_mod
    from capabilities.base import ToolResult

    frame = pd.DataFrame([{"n": 8974}])
    monkeypatch.setattr(cap_mod.datalab_orchestration, "_run_builder_sql",
                        lambda *a, **k: ("cid", SimpleNamespace(dataframe=frame)))
    seen = {}

    class _Q:
        def to_native(self):
            return {"success": True, "result_id": "rid"}

    def fake_exec(sql, meta, **k):
        seen["sql"] = sql
        return _Q()

    monkeypatch.setattr(cap_mod, "execute_datalab_sql", fake_exec)
    monkeypatch.setattr(cap_mod, "_run_analysis_plot",
                        lambda fn, params, ctx: ToolResult(success=True, native={"success": True, "points": 4487}))
    ctx = SimpleNamespace(service=lambda name: None, result_store=None, user_id=None)
    res = cap_mod.LssWedge().run(cap_mod.LssWedgeInput(), ctx)
    sample = res.native["sample"]
    assert "MOD(fiberid, 2) = 0" in seen["sql"]
    assert sample["galaxies_in_slice"] == 8974 and sample["thinning"] == 2 and sample["plotted"] == 4487
    assert "1-in-2 subsample" in sample["statement"]


def test_ui_l14_run2_measured_frequency_is_not_a_cut():
    import json as _json
    from core.answer_verifier import build_trace_summary, verify_answer

    out = {"success": True, "best_period_days": 0.64873, "best_frequency_per_day": 1.541474, "points": 127}
    trace = [{"name": "datalab_period_fold", "ok": True, "output": _json.dumps(out),
              "arguments": _json.dumps({"result_id": "r", "band": "g", "min_frequency": 1, "max_frequency": 10})}]
    ts = build_trace_summary([{"output": _json.dumps(out)}], trace, [], extra_sql=["SELECT mjd, cmag FROM smash_dr1.source WHERE q3c_radial_query(ra, dec, 185.4311, -31.9953, 0.0003)"])
    good = "Best period: 0.6487 d (frequency = 1.5415 d⁻¹)."
    assert not verify_answer(good, ts).by_kind("cut")
    # A value the fit did not produce is still flagged.
    assert verify_answer("Best period: 0.6487 d (frequency = 1.7 d⁻¹).", ts).by_kind("cut")


def test_ui_d22_preview_size_supports_rows():
    import json as _json
    from core.answer_verifier import build_trace_summary, verify_answer

    out = {"success": True, "n_projects": 2567, "results": [{"proposal_id": f"p{i}"} for i in range(60)],
           "results_returned": 60, "results_truncated": True}
    ts = build_trace_summary([{"output": _json.dumps(out)}], [], [], extra_sql=["SELECT proposal_id FROM ivoa.obscore"])
    text = "The full table (60 rows in this preview, up to 2567 total projects) is in the data card."
    assert not verify_answer(text, ts).by_kind("count")
    # The preview size is not a project count.
    assert verify_answer("The census found 60 projects.", ts).by_kind("count")
