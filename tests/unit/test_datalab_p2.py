from __future__ import annotations

import re
import threading
import time

import numpy as np
import pandas as pd
import pytest

from integrations.datalab_client import DatalabResult


# ── fakes ──────────────────────────────────────────────────────────────────────
class _FakeTiledClient:
    """Returns a low-density grid per tile, with a planted dense cell in the hot tile."""

    def __init__(self, hot=(150.0, 2.0), hot_radius=1.1):
        self.hot = hot
        self.hot_radius = hot_radius
        self.calls = 0

    def query(self, *, sql=None, adql=None, fmt="pandas", **kwargs):
        self.calls += 1
        m = re.search(r"q3c_radial_query\([^,]+,\s*[^,]+,\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)", sql or "")
        ra_c, dec_c = (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)
        rows = []
        for r in np.round(np.linspace(ra_c - 0.4, ra_c + 0.4, 5), 3):
            for d in np.round(np.linspace(dec_c - 0.4, dec_c + 0.4, 5), 3):
                rows.append({"ra_bin": float(r), "dec_bin": float(d), "source_count": 2})
        if abs(ra_c - self.hot[0]) <= self.hot_radius and abs(dec_c - self.hot[1]) <= self.hot_radius:
            rows.append({"ra_bin": float(self.hot[0]), "dec_bin": float(self.hot[1]), "source_count": 500})
        return DatalabResult.from_dataframe(pd.DataFrame(rows), {"catalog": "x", "table": "y", "query": sql})


class _FakeAnalysis:
    """Stand-in for datalab_analysis: returns a peak only for a clearly dense tile."""

    @staticmethod
    def _matched_filter_peaks(hist, xe, ye, ss, sl, threshold, max_peaks):
        if hist.size == 0 or float(np.nanmax(hist)) < 50:
            return []
        i, j = np.unravel_index(int(np.nanargmax(hist)), hist.shape)
        return [{"ra": float((xe[i] + xe[i + 1]) / 2.0), "dec": float((ye[j] + ye[j + 1]) / 2.0),
                 "significance": float(np.nanmax(hist))}]


class _FakeDensityClient:
    def query(self, *, sql=None, adql=None, fmt="pandas", **kwargs):
        rows = [{"ra_bin": 185.0 + i * 0.05, "dec_bin": -32.0, "source_count": 100 - i} for i in range(10)]
        return DatalabResult.from_dataframe(pd.DataFrame(rows), {"catalog": "nsc_dr2", "table": "object", "query": sql})


class _FakeImageService:
    def __init__(self):
        self.calls = []

    def cutout_grid(self, peaks, fov_deg, *, band="g", catalog=None, **kwargs):
        peaks = list(peaks)
        self.calls.append((peaks, fov_deg, band))
        return {"success": True, "panels": [{"ra": p["ra"], "dec": p["dec"]} for p in peaks]}


# ── tests ──────────────────────────────────────────────────────────────────────
def test_build_catalog_predicates_and_density_with_cuts():
    from services import datalab_query_builders as B
    preds = B.build_catalog_predicates(
        "nsc_dr2", "object",
        color_cut={"bands": ["gmag", "rmag"], "min": -0.5, "max": 0.5},
        value_cuts=[{"column": "gmag", "op": ">", "value": 19.5}],
        morphology={"column": "class_star", "op": ">", "value": 0.5},
    )
    assert any("gmag - rmag" in p for p in preds)
    sql, meta = B.build_density_aggregate("nsc_dr2", "object", mode="grid", ra=10, dec=0, radius_deg=1.0, predicates=preds)
    assert "q3c_radial_query" in sql and "class_star" in sql and "GROUP BY ra_bin, dec_bin" in sql
    with pytest.raises(ValueError):  # unregistered column rejected
        B.build_catalog_predicates("nsc_dr2", "object", value_cuts=[{"column": "not_a_col", "op": "<", "value": 1}])
    with pytest.raises(ValueError):  # bad operator rejected
        B.build_catalog_predicates("nsc_dr2", "object", value_cuts=[{"column": "gmag", "op": "DROP", "value": 1}])


def test_confirm_sky_area_gates_wide_scans():
    from services import datalab_orchestration as orch
    small = orch.confirm_sky_area({"ra_min": 10, "ra_max": 11, "dec_min": 0, "dec_max": 1}, 1.0, max_tiles=64)
    assert small["needs_confirmation"] is False
    wide = orch.confirm_sky_area({"ra_min": 0, "ra_max": 40, "dec_min": -20, "dec_max": 20}, 1.0, max_tiles=64)
    assert wide["needs_confirmation"] is True and wide["tiles"] > 64 and wide["area_deg2"] > 0


def test_tiled_sky_scan_finds_and_ranks_planted_overdensity():
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore
    fp = {"ra_min": 149.0, "ra_max": 152.0, "dec_min": 1.0, "dec_max": 3.0}
    client = _FakeTiledClient(hot=(150.0, 2.0))
    out = orch.tiled_sky_scan(
        "nsc_dr2", "object", fp, tile_radius_deg=1.0, step_deg=0.2,
        morphology={"column": "class_star", "op": ">", "value": 0.5},
        client=client, result_store=DatalabResultStore(enable_disk_cache=False),
        analysis=_FakeAnalysis(), max_tiles=64, candidate_budget=10, confirm=True,
    )
    assert out["success"] is True and out["candidates"]
    assert client.calls == out["tiles_scanned"] > 1
    top = out["candidates"][0]
    assert abs(top["ra"] - 150.0) < 0.6 and abs(top["dec"] - 2.0) < 0.6


def test_tiled_sky_scan_requires_confirmation_when_wide():
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore
    fp = {"ra_min": 0, "ra_max": 40, "dec_min": -20, "dec_max": 20}
    out = orch.tiled_sky_scan(
        "nsc_dr2", "object", fp, tile_radius_deg=1.0,
        client=_FakeTiledClient(), result_store=DatalabResultStore(enable_disk_cache=False),
        analysis=_FakeAnalysis(), max_tiles=64, confirm=False,
    )
    assert out["success"] is False and out["needs_confirmation"] is True


def test_tiled_sky_scan_respects_max_tiles_and_reports_dropped():
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore
    fp = {"ra_min": 0, "ra_max": 40, "dec_min": -20, "dec_max": 20}
    out = orch.tiled_sky_scan(
        "nsc_dr2", "object", fp, tile_radius_deg=1.0,
        client=_FakeTiledClient(), result_store=DatalabResultStore(enable_disk_cache=False),
        analysis=_FakeAnalysis(), max_tiles=8, confirm=True,
    )
    assert out["success"] is True and out["dropped_tiles"] > 0
    assert out["tiles_scanned"] <= 8 and any("Capped at 8 tiles" in n for n in out["notes"])


def test_density_then_cutouts_chains_top_n():
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore
    img = _FakeImageService()
    out = orch.density_then_cutouts(
        "nsc_dr2", "object", 185.41, -31.98, 1.0, step_deg=0.05,
        morphology={"column": "class_star", "op": ">", "value": 0.5}, top_n=3, fov_deg=0.05, band="g",
        client=_FakeDensityClient(), result_store=DatalabResultStore(enable_disk_cache=False), image_service=img,
    )
    assert out["success"] is True and out["n_peaks"] == 3
    assert len(img.calls) == 1 and len(img.calls[0][0]) == 3
    assert out["cutout_grid"]["success"] is True


def test_job_service_succeeds_and_cancels():
    from services.datalab_job_service import DatalabJobService
    svc = DatalabJobService(max_workers=2)

    jid = svc.start("unit", lambda cancel_check: {"ok": True})
    for _ in range(100):
        if svc.status(jid)["status"] in ("succeeded", "failed", "canceled"):
            break
        time.sleep(0.02)
    assert svc.status(jid)["status"] == "succeeded"
    assert svc.results(jid)["result"] == {"ok": True}

    started = threading.Event()

    def _long(cancel_check):
        started.set()
        for _ in range(2000):
            if cancel_check():
                return {"canceled": True}
            time.sleep(0.005)
        return {"done": True}

    jid2 = svc.start("unit", _long)
    assert started.wait(2.0)
    svc.cancel(jid2)
    for _ in range(200):
        if svc.status(jid2)["status"] == "canceled":
            break
        time.sleep(0.02)
    assert svc.status(jid2)["status"] == "canceled"


def test_agent_registers_p2_tools():
    from tests.unit.test_datalab_p0 import _make_agent
    agent = _make_agent()
    agent._register_tools()
    for name in [
        "datalab_density_vetting",
        "datalab_tiled_search",
        "datalab_confirm_sky_area",
        "datalab_job_status",
        "datalab_job_results",
        "datalab_job_cancel",
        "datalab_export_notebook",
    ]:
        assert agent.tool_registry.get_tool(name) is not None


# ── Polish items: provenance + reproducible-notebook recipe ────────────────────
def test_provenance_ledger_renders_datalab_catalog_source():
    from services.provenance import ProvenanceLedger
    ledger = ProvenanceLedger()
    ledger.extract_from_results({"catalog": "nsc_dr2", "table": "object", "rowcount": 817, "query": "SELECT 1"})
    md = ledger.render_markdown()
    assert "Data Lab catalogs" in md and "nsc_dr2.object" in md and "817 rows" in md


def test_provenance_ignores_non_datalab_catalog():
    from services.provenance import ProvenanceLedger
    ledger = ProvenanceLedger()
    ledger.extract_from_results({"catalog": "some_alma_collection", "table": "obscore"})
    assert ledger.is_empty()


def test_datalab_notebook_steps_recipe():
    from services.notebook_gen import datalab_notebook_steps
    steps = datalab_notebook_steps(
        sql="SELECT COUNT(*) FROM gaia_dr3.gaia_source WHERE q3c_radial_query(ra,dec,1,2,0.1)",
        catalog="gaia_dr3", table="gaia_source",
        sia={"ra": 10.68, "dec": 41.27, "fov_deg": 0.15},
        svo_filters=["CTIO/DECam.g", "WISE/WISE.W1"],
        citation={"text": "Gaia DR3", "url": "https://example/dr3"},
    )
    assert all(s.get("type") in ("markdown", "code") and "content" in s for s in steps)
    blob = "\n".join(s["content"] for s in steps)
    assert "qc.query(sql=q" in blob and "sia.SIAService" in blob and "SvoFps" in blob and "Gaia DR3" in blob
