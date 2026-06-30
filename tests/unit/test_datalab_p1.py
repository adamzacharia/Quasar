from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from integrations.svo_fps_client import SvoFpsClient
from services.datalab_image_service import DatalabImageService, FitsImage
from services.datalab_result_store import DatalabResultStore



class _MemoryPlottingService:
    def _apply_style(self, dark=False):
        from services.plotting import PlottingService

        return PlottingService()._apply_style(dark=dark)

    def _save_and_encode(self, fig, filename):
        import base64
        import io
        import matplotlib.pyplot as plt

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return {
            "success": True,
            "base64_png": base64.b64encode(buf.getvalue()).decode(),
            "web_url": f"/plots/{filename}.png",
            "png_path": None,
            "pdf_path": None,
        }

class _FakeSia:
    def __init__(self, rows):
        self.rows = list(rows)

    def search(self, ra, dec, fov_deg, *, catalog=None, endpoint=None):
        return {
            "success": True,
            "rows": list(self.rows),
            "coverage_gap": len(self.rows) == 0,
            "used_endpoint": endpoint or "fake://sia",
            "provenance": {"ra": ra, "dec": dec, "fov_deg": fov_deg, "catalog": catalog},
        }


class _FakeTable:
    def __init__(self, rows):
        self.rows = rows

    def to_pandas(self):
        return pd.DataFrame(self.rows)


def _rows():
    return [
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "g", "exptime": 30, "access_url": "g30.fits"},
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "g", "exptime": 90, "access_url": "g90.fits"},
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "g", "exptime": 0, "access_url": "g0.fits"},
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "r", "exptime": 60, "access_url": "r60.fits"},
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "i", "exptime": 70, "access_url": "i70.fits"},
        {"proctype": "Preview", "prodtype": "image", "obs_bandpass": "g", "exptime": 999, "access_url": "bad.fits"},
        {"proctype": "Stack", "prodtype": "catalog", "obs_bandpass": "g", "exptime": 999, "access_url": "bad2.fits"},
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "g", "exptime": None, "access_url": "bad3.fits"},
    ]


def test_image_service_deepest_selection_and_coverage_gap():
    service = DatalabImageService(sia_client=_FakeSia(_rows()))
    chosen = service.deepest_by_band(_rows(), ["g", "r"])
    assert chosen["g"]["row"]["access_url"] == "g90.fits"
    assert chosen["r"]["row"]["access_url"] == "r60.fits"

    gap_service = DatalabImageService(sia_client=_FakeSia([]))
    gap = gap_service.cutout(10.0, 0.0, 0.01, band="g")
    assert gap["success"] is True
    assert gap["coverage_gap"] is True
    assert gap["image_base64"] is None


def test_color_image_reprojects_all_bands_before_lupton(monkeypatch):
    calls = []
    service = DatalabImageService(sia_client=_FakeSia(_rows()))

    def fake_load(row, **kwargs):
        value = {"g90.fits": 1.0, "r60.fits": 2.0, "i70.fits": 3.0}[row["access_url"]]
        return FitsImage(data=np.full((4, 4), value), wcs=object(), header={}, path=row["access_url"], source_url=row["access_url"])

    def fake_reproject(image_wcs, ref_wcs, shape_out=None):
        calls.append("reproject")
        return np.asarray(image_wcs[0]), np.ones(shape_out or image_wcs[0].shape)

    def fake_lupton(i_img, r_img, g_img, Q=8.0, stretch=0.5):
        calls.append("lupton")
        assert calls[:3] == ["reproject", "reproject", "reproject"]
        return np.zeros((4, 4, 3), dtype=float)

    fake_reproject_module = types.ModuleType("reproject")
    fake_reproject_module.reproject_interp = fake_reproject
    monkeypatch.setitem(sys.modules, "reproject", fake_reproject_module)

    import astropy.visualization as visualization

    monkeypatch.setattr(visualization, "make_lupton_rgb", fake_lupton)
    monkeypatch.setattr(service, "_load_image", fake_load)
    monkeypatch.setattr(service, "_cutout_image", lambda image, **kwargs: image)
    monkeypatch.setattr(service, "_render_rgb", lambda rgb, wcs, title: {"base64_png": "abc", "web_url": "/plots/color.png"})

    result = service.color_image(10.0, 0.0, 0.01)
    assert result["success"] is True
    # Auto-selection yields RGB order (red, green, blue) = (i, r, g) when i is present.
    assert result["bands_used"] == ["i", "r", "g"]
    assert calls == ["reproject", "reproject", "reproject", "lupton"]


def test_svo_client_returns_cached_wavelengths(monkeypatch):
    calls = {"n": 0}

    class FakeSvoFps:
        @staticmethod
        def get_filter_list(**kwargs):
            calls["n"] += 1
            return _FakeTable([
                {"filterID": "CTIO/DECam.g", "WavelengthEff": 4770.0, "WavelengthPivot": 4820.0},
            ])

    astroquery = types.ModuleType("astroquery")
    svo_mod = types.ModuleType("astroquery.svo_fps")
    svo_mod.SvoFps = FakeSvoFps
    monkeypatch.setitem(sys.modules, "astroquery", astroquery)
    monkeypatch.setitem(sys.modules, "astroquery.svo_fps", svo_mod)

    client = SvoFpsClient(enable_disk_cache=False)
    first = client.wavelength("g")
    second = client.wavelength("g")
    assert first["effective_angstrom"] == 4770.0
    assert first["pivot_micron"] == pytest.approx(0.482)
    assert second["provenance"]["cached"] is True
    assert calls["n"] == 1


def test_datalab_analysis_functions_render_from_synthetic_result_store(monkeypatch):
    from services import datalab_analysis

    plotter = _MemoryPlottingService()
    store = DatalabResultStore(enable_disk_cache=False)

    phot = pd.DataFrame({"g": [15.0, 16.0, 17.0], "r": [14.5, 15.7, 16.8], "ra": [10.0, 10.1, 10.2], "dec": [0.0, 0.1, 0.2]})
    phot_id = store.put(phot, {"catalog": "synthetic", "table": "phot"})
    scatter = datalab_analysis.catalog_scatter(phot_id, "g-r", "g", invert_y=True, result_store=store, plotting_service=plotter)
    assert scatter["success"] and scatter["image_base64"]

    rng = np.random.default_rng(4)
    field = pd.DataFrame({
        "ra": np.concatenate([rng.normal(10.0, 0.4, 250), rng.normal(12.0, 0.03, 90)]),
        "dec": np.concatenate([rng.normal(0.0, 0.4, 250), rng.normal(1.0, 0.03, 90)]),
    })
    density_id = store.put(field, {"catalog": "synthetic", "table": "sky"})
    density = datalab_analysis.sky_density_map(density_id, matched_filter=True, bins=45, peak_threshold=2.0, result_store=store, plotting_service=plotter)
    assert density["success"] and density["image_base64"]
    assert density["peaks"]
    assert abs(density["peaks"][0]["ra"] - 12.0) < 0.25
    assert abs(density["peaks"][0]["dec"] - 1.0) < 0.25

    period = 0.6
    t = np.linspace(0, 18, 160)
    mag = 15.0 + 0.25 * np.sin(2 * np.pi * t / period)
    lc_id = store.put(pd.DataFrame({"mjd": t, "cmag": mag, "cerr": np.full_like(t, 0.03)}), {"catalog": "synthetic", "table": "lc"})
    folded = datalab_analysis.period_fold(lc_id, min_frequency=1.0, max_frequency=4.0, result_store=store, plotting_service=plotter)
    assert folded["success"] and folded["image_base64"]
    assert folded["best_period_days"] == pytest.approx(period, abs=0.02)

    class FakeSvoClient:
        def wavelengths(self, filters):
            values = {"g": 0.48, "r": 0.62, "z": 0.91, "w1": 3.4, "w2": 4.6}
            return {f: {"effective_micron": values[f], "pivot_micron": values[f]} for f in filters}

    sed_frame = pd.DataFrame([{ "dered_mag_g": 18.0, "dered_mag_r": 17.5, "dered_mag_z": 17.1, "dered_mag_w1": 16.0, "dered_mag_w2": 15.8 }])
    sed_id = store.put(sed_frame, {"catalog": "ls_dr9", "table": "tractor"})
    sed = datalab_analysis.sed_plot(sed_id, result_store=store, svo_client=FakeSvoClient(), plotting_service=plotter)
    assert sed["success"] and sed["image_base64"]

    lss_frame = pd.DataFrame({"mean_fiber_ra": [10.0, 10.2, 10.4], "mean_fiber_dec": [0.0, 0.2, 0.4], "z": [0.1, 0.2, 0.3], "spectype": ["GALAXY", "QSO", "GALAXY"]})
    lss_id = store.put(lss_frame, {"catalog": "desi_dr1", "table": "zpix"})
    lss = datalab_analysis.lss_wedge(lss_id, class_col="spectype", result_store=store, plotting_service=plotter)
    assert lss["success"] and lss["image_base64"]


def test_agent_registers_p1_tools():
    from tests.unit.test_datalab_p0 import _make_agent

    agent = _make_agent()
    agent._register_tools()
    for name in [
        "datalab_image_cutout",
        "datalab_color_image",
        "datalab_cutout_grid",
        "svo_filter_wavelength",
        "datalab_catalog_scatter",
        "datalab_sky_density_map",
        "datalab_period_fold",
        "datalab_sed_plot",
        "datalab_lss_wedge",
    ]:
        assert agent.tool_registry.get_tool(name) is not None


def test_sandbox_allowlist_adds_exact_p1_astropy_modules():
    from core import sandbox

    for name in ["astropy.timeseries", "astropy.visualization", "astropy.table"]:
        assert name in sandbox._ALLOWED_MODULES
    for name in ["healpy", "matplotlib", "requests"]:
        assert name not in sandbox._ALLOWED_MODULES


# ── Regression tests for the @cx-build P1 self-review findings ─────────────────

def test_color_image_auto_selects_z_when_i_absent(monkeypatch):
    # LS DR9 registered bands are g/r/z; a fixed i/r/g triplet would wrongly report a gap.
    rows = [
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "g", "exptime": 90, "access_url": "g.fits"},
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "r", "exptime": 60, "access_url": "r.fits"},
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "z", "exptime": 70, "access_url": "z.fits"},
    ]
    service = DatalabImageService(sia_client=_FakeSia(rows))
    monkeypatch.setattr(service, "_load_image", lambda row, **kw: FitsImage(data=np.ones((4, 4)), wcs=object(), header={}, path=row["access_url"], source_url=row["access_url"]))
    monkeypatch.setattr(service, "_cutout_image", lambda image, **kw: image)
    monkeypatch.setattr(service, "_render_rgb", lambda rgb, wcs, title: {"base64_png": "abc", "web_url": "/p.png"})
    fake_reproject_module = types.ModuleType("reproject")
    fake_reproject_module.reproject_interp = lambda image_wcs, ref_wcs, shape_out=None: (np.asarray(image_wcs[0]), None)
    monkeypatch.setitem(sys.modules, "reproject", fake_reproject_module)
    import astropy.visualization as visualization
    monkeypatch.setattr(visualization, "make_lupton_rgb", lambda r, g, b, Q=8.0, stretch=0.5: np.zeros((4, 4, 3)))

    result = service.color_image(10.0, 0.0, 0.01)
    assert result["success"] is True
    assert result.get("coverage_gap") is not True
    assert result["bands_used"] == ["z", "r", "g"]


def test_period_fold_tolerates_nonfinite_uncertainties():
    from services import datalab_analysis
    store = DatalabResultStore(enable_disk_cache=False)
    plotter = _MemoryPlottingService()
    period = 0.6
    t = np.linspace(0, 18, 160)
    mag = 15.0 + 0.25 * np.sin(2 * np.pi * t / period)
    cerr = np.full_like(t, np.nan)   # all uncertainties invalid ...
    cerr[::2] = 0.0                   # ... or zero -> must fall back to unweighted, not crash
    lc_id = store.put(pd.DataFrame({"mjd": t, "cmag": mag, "cerr": cerr}), {"catalog": "synthetic", "table": "lc"})
    folded = datalab_analysis.period_fold(lc_id, min_frequency=1.0, max_frequency=4.0, result_store=store, plotting_service=plotter)
    assert folded["success"]
    assert folded["best_period_days"] == pytest.approx(period, abs=0.02)


def test_try_healpy_plot_guards_high_nside():
    from services import datalab_analysis
    frame = pd.DataFrame({"healpix": [0, 1, 2], "source_count": [5, 10, 3]})
    # nside above the dense cap must short-circuit to the sparse path (no 2e8 alloc).
    assert datalab_analysis._try_healpy_plot(None, frame, "healpix", "source_count", 4096, "nested", "t") is False


def test_rag_gate_tiered_accept_reject():
    from services.rag_service import is_domain_relevant
    assert is_domain_relevant("what can you tell me about ALMA Band 6 sensitivity") is True   # strong keyword: alma
    assert is_domain_relevant("explain the correlator") is True                               # strong keyword: correlator
    assert is_domain_relevant("what can you tell me about my rock band?") is False             # off-domain stem + broad 'band'
    assert is_domain_relevant("ok, explain phase margin in my amplifier") is False             # off-domain stem + broad 'phase'
    assert is_domain_relevant("show the proposal deadlines for cycle 12") is True              # broad keyword, no off-domain stem


# ── Image display wiring (why Data Lab images render in chat) ──────────────────
def test_attach_image_result_strips_base64_and_sets_card():
    from tests.unit.test_datalab_p0 import _make_agent
    agent = _make_agent()
    agent.last_run_result = None
    result = {"success": True, "path": "/plots/x.png", "image_base64": "AAAA" * 5000}
    out = agent._datalab_attach_image_result(result, "cap")
    # Heavy base64 is stripped from the LLM-facing result (prevents 8000-char truncation → empty reply).
    assert "image_base64" not in out and out["image_attached"] is True
    # UI image card is set with the servable /plots path.
    assert agent.last_run_result == {"type": "image", "image_url": "/plots/x.png", "caption": "cap"}


def test_attach_image_result_falls_back_to_data_uri():
    from tests.unit.test_datalab_p0 import _make_agent
    agent = _make_agent()
    out = agent._datalab_attach_image_result({"success": True, "path": None, "image_base64": "QUJD"}, "c")
    assert agent.last_run_result["image_url"].startswith("data:image/png;base64,QUJD")
    assert "image_base64" not in out


def test_attach_image_result_no_card_on_coverage_gap():
    from tests.unit.test_datalab_p0 import _make_agent
    agent = _make_agent()
    agent.last_run_result = "PRIOR"
    out = agent._datalab_attach_image_result(
        {"success": True, "coverage_gap": True, "image_base64": None, "path": None, "bands_used": []}, "c"
    )
    assert agent.last_run_result == "PRIOR"  # coverage gap → no image card forced
    assert out["coverage_gap"] is True





