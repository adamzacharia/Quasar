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
        # Real tiles always carry noise; a CONSTANT fake trips the degenerate-tile
        # guard (all-NaN/single-valued tiles are skipped as broken placeholders).
        data = np.full((4, 4), value) + np.arange(16).reshape(4, 4) * 0.01
        return FitsImage(data=data, wcs=object(), header={}, path=row["access_url"], source_url=row["access_url"])

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


def _overlap_service(monkeypatch, rows, partial_bases, captured=None):
    """Color-image harness where fake tiles reproject with NaN gaps.

    partial_bases: {base_value: keep_columns} — tiles whose pixel base value is
    listed reproject with NaN outside their first keep_columns columns.
    """
    service = DatalabImageService(sia_client=_FakeSia(rows))
    grad = np.arange(16).reshape(4, 4) * 0.01

    def fake_load(row, **kw):
        base = float(row["exptime"])  # encode identity in the pixel values
        return FitsImage(data=np.full((4, 4), base) + grad, wcs=object(), header={}, path=row["access_url"], source_url=row["access_url"])

    def fake_reproject(image_wcs, ref_wcs, shape_out=None):
        base = round(float(np.asarray(image_wcs[0]).flat[0]))
        out = np.full(shape_out or (4, 4), float(base)) + grad
        keep = partial_bases.get(base)
        if keep is not None:
            out[:, keep:] = np.nan
        return out, None

    def fake_render(rgb, wcs, title):
        if captured is not None:
            captured["rgb_shape"] = np.asarray(rgb).shape
        return {"base64_png": "abc", "web_url": "/p.png"}

    fake_reproject_module = types.ModuleType("reproject")
    fake_reproject_module.reproject_interp = fake_reproject
    monkeypatch.setitem(sys.modules, "reproject", fake_reproject_module)
    import astropy.visualization as visualization
    monkeypatch.setattr(visualization, "make_lupton_rgb", lambda r, g, b, Q=8.0, stretch=0.5: np.stack([r, g, b], axis=-1))
    monkeypatch.setattr(service, "_load_image", fake_load)
    monkeypatch.setattr(service, "_cutout_image", lambda image, **kw: image)
    monkeypatch.setattr(service, "_render_rgb", fake_render)
    return service


def test_color_image_partial_band_falls_through_to_covering_tile(monkeypatch):
    """F-12 deferred item (live P14 07-12: half-black composite from a SMASH
    tile partially overlapping the Hydra II field). A band whose deepest tile
    covers <90% of the field must fall through to the next candidate."""
    rows = [
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "g", "exptime": 10, "access_url": "g.fits"},
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "r", "exptime": 20, "access_url": "r.fits"},
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "i", "exptime": 900, "access_url": "i_partial.fits"},
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "i", "exptime": 500, "access_url": "i_full.fits"},
    ]
    service = _overlap_service(monkeypatch, rows, partial_bases={900: 2})  # deepest i: half covered
    result = service.color_image(10.0, 0.0, 0.01)
    assert result["success"] is True
    assert result["provenance"]["band_coverage"] == {"i": 1.0, "r": 1.0, "g": 1.0}
    assert "i_full.fits" in str(result["provenance"]["selected_rows"]["i"])
    assert result.get("note") is None


def test_color_image_crops_to_common_coverage_with_note(monkeypatch):
    """No better tile exists: the composite is cropped to the jointly covered
    region and the result says so, instead of rendering half the field black."""
    rows = [
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "g", "exptime": 10, "access_url": "g.fits"},
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "r", "exptime": 20, "access_url": "r.fits"},
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "i", "exptime": 900, "access_url": "i_partial.fits"},
    ]
    captured = {}
    service = _overlap_service(monkeypatch, rows, partial_bases={900: 2}, captured=captured)
    result = service.color_image(10.0, 0.0, 0.01)
    assert result["success"] is True
    assert "cropped" in str(result.get("note"))
    assert result["provenance"]["band_coverage"]["i"] == 0.5
    assert captured["rgb_shape"] == (4, 2, 3)  # only the covered half is rendered


def test_color_image_refuses_sliver_common_coverage(monkeypatch):
    """Bands that barely overlap each other -> structured refusal with per-band
    coverage, not a mostly-black composite."""
    rows = [
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "g", "exptime": 10, "access_url": "g.fits"},
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "r", "exptime": 20, "access_url": "r.fits"},
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "i", "exptime": 900, "access_url": "i_sliver.fits"},
    ]
    service = _overlap_service(monkeypatch, rows, partial_bases={900: 0})  # i covers nothing
    result = service.color_image(10.0, 0.0, 0.01)
    assert result["success"] is False and result["coverage_gap"] is True
    assert result["band_coverage"]["i"] == 0.0
    assert "mostly black" in result["error"]


# ── Same-survey color completion (no product substitution on color requests) ──
# Beta eval 2026-07: 'color image of M31 from the DECam Legacy Surveys' was
# answered with an optical/2MASS/WISE panel. When SIA has <3 usable bands the
# service now completes IN-SURVEY from that survey's official color HiPS, and
# when even that fails it says the product cannot be made — never substitutes.

class _FakeColorHips:
    def __init__(self, blank=False, success=True):
        self.blank = blank
        self.success = success
        self.calls = []

    def cutout(self, ra, dec, fov_deg=0.25, survey="optical", width=512, title=None, detect_blank=False):
        self.calls.append({"ra": ra, "dec": dec, "fov_deg": fov_deg, "survey": survey,
                           "width": width, "detect_blank": detect_blank})
        if not self.success:
            return {"success": False, "error": "hips2fits down"}
        return {
            "success": True,
            "image_base64": "abc",
            "path": "/plots/hips_color.png",
            "png_path": None,
            "blank": self.blank,
            "provenance": {"params": {"hips": survey}},
        }


def _lsdr9_m31_rows():
    # M31-like inventory: only z has a healthy tile; the g/r refs carry the
    # known-broken empty col= pattern (LGS tiles that 500 server-side).
    return [
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "g", "exptime": 900, "access_url": "https://x/cutout?col=&siaRef=g.fits"},
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "r", "exptime": 900, "access_url": "https://x/cutout?col=&siaRef=r.fits"},
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "z", "exptime": 300, "access_url": "https://x/cutout?col=mzls&siaRef=z.fits"},
    ]


def test_color_image_completes_in_survey_from_color_hips():
    service = DatalabImageService(sia_client=_FakeSia(_lsdr9_m31_rows()))
    fake = _FakeColorHips()
    service._hips_service = fake
    out = service.color_image(10.68, 41.27, 0.1, catalog="ls_dr9")
    assert out["success"] is True and out["coverage_gap"] is False
    assert out["color_hips_completion"] == "CDS/P/DESI-Legacy-Surveys/DR10/color"
    assert out["bands_used"] == ["g", "r", "z"]
    assert "True-color" in out["note"] and "hips2fits" in out["note"]
    assert out["provenance"]["color_hips"] == "CDS/P/DESI-Legacy-Surveys/DR10/color"
    assert out["provenance"]["healthy_sia_bands"] == ["z"]
    # The completion fetched the SAME survey's color HiPS with blank detection on.
    assert fake.calls[0]["survey"] == "CDS/P/DESI-Legacy-Surveys/DR10/color"
    assert fake.calls[0]["detect_blank"] is True


def test_color_image_blank_color_hips_reports_impossible_not_substitute():
    # Live M31 case: the DR10 color HiPS MOC does not reach the DR9-north area,
    # so hips2fits serves a uniform PNG there — must NOT be shipped as the image.
    service = DatalabImageService(sia_client=_FakeSia(_lsdr9_m31_rows()))
    service._hips_service = _FakeColorHips(blank=True)
    out = service.color_image(10.68, 41.27, 0.1, catalog="ls_dr9")
    assert out["coverage_gap"] is True and out["image_base64"] is None
    assert "cannot be produced" in out["note"]
    assert "hips_multiband_panel" in out["note"]  # the substitution ban is explicit
    assert out["provenance"]["color_hips_attempted"] == "CDS/P/DESI-Legacy-Surveys/DR10/color"


def test_color_image_no_color_hips_survey_says_so_and_asks():
    class _NeverHips:
        def cutout(self, *a, **k):
            raise AssertionError("catalog has no color HiPS — completion must not be attempted")

    service = DatalabImageService(sia_client=_FakeSia(_lsdr9_m31_rows()))
    service._hips_service = _NeverHips()
    out = service.color_image(10.68, 41.27, 0.1, catalog="smash_dr2")
    assert out["coverage_gap"] is True
    assert "publishes no full-color HiPS" in out["note"]
    assert "do NOT silently substitute" in out["note"]


def test_color_image_zero_row_gap_also_completes_in_survey():
    service = DatalabImageService(sia_client=_FakeSia([]))
    fake = _FakeColorHips()
    service._hips_service = fake
    out = service.color_image(150.0, 2.2, 0.1, catalog="ls_dr10")
    assert out["success"] is True
    assert out["color_hips_completion"] == "CDS/P/DESI-Legacy-Surveys/DR10/color"
    assert fake.calls  # the completion path ran off the empty SIA search too


def test_sia_hips_profile_routes_color_requests_away_from_multiband_panel():
    from services.archive_profiles.sia_hips import PROFILE

    ranked = " ".join(PROFILE.prompt_pitfalls())
    assert "datalab_color_image" in ranked
    assert "NEVER hips_multiband_panel" in ranked


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


class _NormCapturingPlottingService(_MemoryPlottingService):
    """Records the color-scale norm class used on each mappable, so tests can
    assert whether LogNorm (vs a linear Normalize) was actually applied."""

    def _save_and_encode(self, fig, filename):
        self.norms = [type(coll.norm).__name__ for ax in fig.axes for coll in ax.collections]
        return super()._save_and_encode(fig, filename)


def test_sky_density_map_log_scale_default_and_toggle():
    """DLB-08: density colorbars are log-scaled by default; log_scale=False stays linear."""
    from services import datalab_analysis

    store = DatalabResultStore(enable_disk_cache=False)
    field = pd.DataFrame({"ra": [10.0, 10.0, 10.1, 12.0], "dec": [0.0, 0.0, 0.1, 1.0]})
    did = store.put(field, {"catalog": "synthetic", "table": "sky"})

    p = _NormCapturingPlottingService()
    r = datalab_analysis.sky_density_map(did, bins=20, result_store=store, plotting_service=p)
    assert r["success"] and r["log_scale"] is True
    assert "LogNorm" in p.norms

    p = _NormCapturingPlottingService()
    r = datalab_analysis.sky_density_map(did, bins=20, log_scale=False, result_store=store, plotting_service=p)
    assert r["log_scale"] is False
    assert "LogNorm" not in p.norms


def test_sky_density_map_log_scale_falls_back_to_linear_on_nonpositive_counts():
    """A map with no positive finite counts cannot be log-scaled — degrade to
    linear and report log_scale=False (never claim a log scale that wasn't applied)."""
    from services import datalab_analysis

    store = DatalabResultStore(enable_disk_cache=False)
    frame = pd.DataFrame({"healpix": [1, 2, 3], "source_count": [0.0, 0.0, 0.0]})
    hid = store.put(frame, {"catalog": "synthetic", "table": "hpx0"})
    p = _NormCapturingPlottingService()
    r = datalab_analysis.sky_density_map(hid, mode="healpix", nside=4, result_store=store, plotting_service=p)
    assert r["success"] and r["log_scale"] is False
    assert "LogNorm" not in p.norms


def test_try_healpy_plot_masks_zero_pixels_and_never_passes_nonpositive_vmin(monkeypatch):
    """healpy derives vmin from ALL finite pixels, so a bare norm='log' crashes on any
    map containing a zero-count pixel. Guard: mask non-positive pixels and pin min/max
    to the positive range; on an all-nonpositive map, fall back to a linear scale."""
    from services import datalab_analysis

    captured = {}

    class _FakeHP:
        UNSEEN = -1.6375e30

        @staticmethod
        def nside2npix(nside):
            return 12 * nside * nside

        @staticmethod
        def mollview(values, **kwargs):
            captured["values"] = np.asarray(values, dtype=float)
            captured["kwargs"] = dict(kwargs)

    monkeypatch.setitem(sys.modules, "healpy", _FakeHP)
    store = DatalabResultStore(enable_disk_cache=False)

    # Mixed zeros + positives, log requested → norm='log', min from POSITIVE values, zeros masked.
    mix = pd.DataFrame({"healpix": [0, 1, 2, 3], "source_count": [0.0, 5.0, 0.0, 20.0]})
    mid = store.put(mix, {"catalog": "synthetic", "table": "mix"})
    r = datalab_analysis.sky_density_map(mid, mode="healpix", nside=4, result_store=store,
                                         plotting_service=_MemoryPlottingService())
    assert r["log_scale"] is True
    assert captured["kwargs"].get("norm") == "log"
    assert captured["kwargs"].get("min") == 5.0 and captured["kwargs"].get("max") == 20.0
    assert np.isnan(captured["values"][0]) and np.isnan(captured["values"][2])  # zero-count → masked
    assert captured["values"][1] == 5.0 and captured["values"][3] == 20.0

    # All-zero map, log requested → linear (no norm kwarg), no ValueError.
    zero = pd.DataFrame({"healpix": [0, 1, 2], "source_count": [0.0, 0.0, 0.0]})
    zid = store.put(zero, {"catalog": "synthetic", "table": "z"})
    r = datalab_analysis.sky_density_map(zid, mode="healpix", nside=4, result_store=store,
                                         plotting_service=_MemoryPlottingService())
    assert r["log_scale"] is False
    assert "norm" not in captured["kwargs"]


def test_tool_call_args_html_entities_are_unescaped():
    """deepseek emitted a plot title 'g&lt;18' that rendered the literal entity; the
    agent loop unescapes common HTML entities in string args (values, recursively)."""
    from core.agent import _unescape_tool_args, _unescape_html_entities

    assert _unescape_html_entities("g&lt;18") == "g<18"
    assert _unescape_html_entities("x &gt;= 5 &amp;&amp; y &lt; 2") == "x >= 5 && y < 2"
    assert _unescape_html_entities("name = &#39;foo&#39;") == "name = 'foo'"
    # A '&copy'-style non-target sequence in a URL query string is left intact.
    assert _unescape_html_entities("http://x/?a=1&copy=2") == "http://x/?a=1&copy=2"

    out = _unescape_tool_args({
        "title": "counts g&lt;18",
        "value_cuts": [{"column": "g", "op": "&lt;", "value": 18}],
        "n": 3,
        "flag": True,
    })
    assert out["title"] == "counts g<18"
    assert out["value_cuts"][0]["op"] == "<"
    assert out["n"] == 3 and out["flag"] is True  # non-strings untouched


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
    # Non-constant data: constant fakes trip the degenerate-tile guard.
    monkeypatch.setattr(service, "_load_image", lambda row, **kw: FitsImage(data=np.ones((4, 4)) + np.arange(16).reshape(4, 4) * 0.01, wcs=object(), header={}, path=row["access_url"], source_url=row["access_url"]))
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


def test_period_fold_single_band_selection():
    """A multi-band light curve must fold ONE band cleanly; the mixed set smears it."""
    from services import datalab_analysis
    store = DatalabResultStore(enable_disk_cache=False)
    plotter = _MemoryPlottingService()
    period = 0.6
    t = np.linspace(0, 18, 160)
    g_mag = 15.0 + 0.25 * np.sin(2 * np.pi * t / period)
    # r-band epochs at a totally different period + zero-point (would corrupt a mixed fold).
    r_mag = 17.0 + 0.4 * np.sin(2 * np.pi * t / 0.21)
    frame = pd.DataFrame({
        "mjd": np.concatenate([t, t]),
        "cmag": np.concatenate([g_mag, r_mag]),
        "cerr": np.full(2 * len(t), 0.03),
        "filter": ["g"] * len(t) + ["r"] * len(t),
    })
    lc_id = store.put(frame, {"catalog": "nsc_dr2", "table": "meas"})
    folded = datalab_analysis.period_fold(
        lc_id, band="g", min_frequency=1.0, max_frequency=4.0,
        result_store=store, plotting_service=plotter,
    )
    assert folded["success"]
    assert folded["band"] == "g"
    assert folded["points"] == len(t)  # only the g-band epochs were folded
    assert folded["best_period_days"] == pytest.approx(period, abs=0.02)


def test_period_fold_band_is_case_insensitive_and_validates_absent_rows():
    from services import datalab_analysis
    store = DatalabResultStore(enable_disk_cache=False)
    plotter = _MemoryPlottingService()
    t = np.linspace(0, 18, 120)
    frame = pd.DataFrame({
        "mjd": t,
        "cmag": 15.0 + 0.25 * np.sin(2 * np.pi * t / 0.6),
        "cerr": np.full_like(t, 0.03),
        "filter": ["G"] * len(t),   # stored upper-case; request lower-case
    })
    lc_id = store.put(frame, {"catalog": "nsc_dr2", "table": "meas"})
    folded = datalab_analysis.period_fold(lc_id, band="g", result_store=store, plotting_service=plotter)
    assert folded["success"] and folded["band"] == "g" and folded["points"] == len(t)
    # A band with no matching rows must raise, not silently fold nothing.
    with pytest.raises(ValueError):
        datalab_analysis.period_fold(lc_id, band="z", result_store=store, plotting_service=plotter)
    # A band requested against a frame that has NO band column must raise, not
    # silently fold every band (which would smear the phased curve).
    no_band = pd.DataFrame({"mjd": t, "cmag": 15.0 + 0.2 * np.sin(2 * np.pi * t / 0.6), "cerr": np.full_like(t, 0.03)})
    nb_id = store.put(no_band, {"catalog": "synthetic", "table": "lc"})
    with pytest.raises(ValueError):
        datalab_analysis.period_fold(nb_id, band="g", result_store=store, plotting_service=plotter)


def test_period_fold_rejects_sentinel_magnitudes():
    """99.99 / -99 padding must be dropped before the periodogram, not folded."""
    from services import datalab_analysis
    store = DatalabResultStore(enable_disk_cache=False)
    plotter = _MemoryPlottingService()
    period = 0.6
    t = np.linspace(0, 18, 160)
    mag = 15.0 + 0.25 * np.sin(2 * np.pi * t / period)
    mag[::5] = 99.99   # sentinel padding on 1/5 of epochs
    lc_id = store.put(pd.DataFrame({"mjd": t, "cmag": mag, "cerr": np.full_like(t, 0.03)}),
                      {"catalog": "synthetic", "table": "lc"})
    folded = datalab_analysis.period_fold(lc_id, min_frequency=1.0, max_frequency=4.0,
                                          result_store=store, plotting_service=plotter)
    assert folded["success"]
    assert folded["points"] == int(np.sum(np.abs(mag) < 90.0))  # sentinels excluded
    assert folded["best_period_days"] == pytest.approx(period, abs=0.02)


def test_clean_label_unescapes_html_entities_in_titles():
    """Models sometimes HTML-escape angle brackets in a plot title (e.g. 'g&lt;18')."""
    from services import datalab_analysis
    assert datalab_analysis._clean_label("NSC density g&lt;18") == "NSC density g<18"
    assert datalab_analysis._clean_label("M31 &amp; M32") == "M31 & M32"
    assert datalab_analysis._clean_label("  plain title  ") == "plain title"
    assert datalab_analysis._clean_label(None) == ""


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


def test_cutout_falls_through_broken_tiles(monkeypatch):
    """A tile whose download 500s must NOT kill the cutout — the next-deepest candidate is used.
    (Observed live: Local Group Survey refs in coadd_all at M31 500 on /svc/cutout.)"""
    rows = [
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "r", "exptime": 900, "access_url": "https://x/cutout?col=lgs&siaRef=broken.fits"},
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "r", "exptime": 500, "access_url": "https://x/cutout?col=ls_dr9&siaRef=good.fits"},
    ]
    service = DatalabImageService(sia_client=_FakeSia(rows))

    def fake_load(row, **kwargs):
        if "broken" in row["access_url"]:
            raise RuntimeError("500 Server Error")
        # Non-constant data: constant fakes trip the degenerate-tile guard.
        data = np.ones((4, 4)) + np.arange(16).reshape(4, 4) * 0.01
        return FitsImage(data=data, wcs=object(), header={}, path="good.fits", source_url=row["access_url"])

    monkeypatch.setattr(service, "_load_image", fake_load)
    monkeypatch.setattr(service, "_cutout_image", lambda image, **kw: image)
    monkeypatch.setattr(service, "_render_single_band", lambda data, wcs, title: {"base64_png": "abc", "web_url": "/p.png"})

    out = service.cutout(10.0, 41.0, 0.05, band="r")
    assert out["success"] is True and out.get("coverage_gap") is not True
    assert out["provenance"]["selected_rows"]["r"]  # a row was used
    assert "good.fits" in str(out["provenance"].get("source_url"))
    assert out["provenance"].get("skipped_broken_tiles")  # the broken one is reported


def test_cutout_reports_structured_error_when_all_tiles_fail(monkeypatch):
    rows = [
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "r", "exptime": 900, "access_url": "https://x/cutout?col=&siaRef=b1.fits"},
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "r", "exptime": 500, "access_url": "https://x/cutout?col=&siaRef=b2.fits"},
    ]
    service = DatalabImageService(sia_client=_FakeSia(rows))
    monkeypatch.setattr(service, "_load_image", lambda row, **kw: (_ for _ in ()).throw(RuntimeError("500")))
    out = service.cutout(10.0, 41.0, 0.05, band="r")
    # Structured failure, NOT a raised exception; distinguishes service errors from coverage gaps.
    assert out["success"] is False and out["coverage_gap"] is False
    assert "failed to download" in out["error"] and len(out["download_errors"]) == 2


def test_candidates_deprioritize_empty_col_urls():
    rows = [
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "g", "exptime": 900, "access_url": "https://x/cutout?col=&siaRef=lgs.fits"},
        {"proctype": "Stack", "prodtype": "image", "obs_bandpass": "g", "exptime": 100, "access_url": "https://x/cutout?col=ls_dr9&siaRef=ok.fits"},
    ]
    service = DatalabImageService(sia_client=_FakeSia(rows))
    cands = service.candidates_by_band(rows, ["g"])["g"]
    # Known-broken empty-col URL ranks BELOW the shallower but healthy ref.
    assert "ok.fits" in cands[0]["row"]["access_url"]


def test_attach_image_result_no_card_on_coverage_gap():
    from tests.unit.test_datalab_p0 import _make_agent
    agent = _make_agent()
    agent.last_run_result = "PRIOR"
    out = agent._datalab_attach_image_result(
        {"success": True, "coverage_gap": True, "image_base64": None, "path": None, "bands_used": []}, "c"
    )
    assert agent.last_run_result == "PRIOR"  # coverage gap → no image card forced
    assert out["coverage_gap"] is True







def test_period_fold_declares_significance_and_null_results():
    """The tool itself must declare a null result: live P14 claimed 'consistent
    with an RR Lyrae' for a grid-edge period at FAP=0.28, and the post-fix run
    presented FAP=1.0 at the 1-day alias."""
    from services import datalab_analysis
    rng = np.random.default_rng(3)
    store = DatalabResultStore(enable_disk_cache=False)
    plotter = _MemoryPlottingService()

    # A clean 0.6487d sinusoid inside the search window -> significant.
    period = 0.6487
    t = np.sort(rng.uniform(0, 60, 240))
    mag = 21.5 + 0.5 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.02, t.size)
    lc = store.put(pd.DataFrame({"mjd": t, "cmag": mag}), {"catalog": "synthetic", "table": "lc"})
    good = datalab_analysis.period_fold(lc, min_frequency=1.0, max_frequency=10.0,
                                        result_store=store, plotting_service=plotter)
    assert good["period_significant"] is True
    assert good["significance"] == "significant"
    assert good["best_period_days"] == pytest.approx(period, abs=0.005)
    assert good["searched_period_days"] == [0.1, 1.0]
    assert not good["warnings"]

    # Pure noise -> FAP is large -> declared no_significant_period with warnings.
    # Seed picked for a robustly in-window, high-FAP max peak (FAP~0.78): the
    # Baluev FAP is deterministic for fixed data, so this cannot flake.
    t_noise = np.sort(np.random.default_rng(0).uniform(0, 60, 240))
    noise_mag = 21.5 + np.random.default_rng(100).normal(0, 0.3, t_noise.size)
    lc2 = store.put(pd.DataFrame({"mjd": t_noise, "cmag": noise_mag}), {"catalog": "synthetic", "table": "lc"})
    bad = datalab_analysis.period_fold(lc2, min_frequency=1.0, max_frequency=10.0,
                                       result_store=store, plotting_service=plotter)
    assert bad["period_significant"] is False
    assert bad["significance"] == "no_significant_period"
    assert any("NO SIGNIFICANT PERIOD" in w for w in bad["warnings"])

    # A best peak sitting ON the searched-window boundary is a window artifact
    # and must be flagged even when its FAP is tiny (the strong signal lands at
    # freq[0] here because min_frequency == the signal frequency).
    lc3 = store.put(pd.DataFrame({"mjd": t, "cmag": mag}), {"catalog": "synthetic", "table": "lc"})
    edge = datalab_analysis.period_fold(lc3, min_frequency=1.0 / period, max_frequency=10.0,
                                        result_store=store, plotting_service=plotter)
    assert edge["period_significant"] is False
    assert any("edge of the searched window" in n for n in edge["significance_notes"])


def test_period_fold_mixed_bands_default_to_best_sampled():
    """No band requested + multi-band frame: fold ONLY the best-sampled band and
    say so, instead of silently smearing bands together (live P14)."""
    from services import datalab_analysis
    rng = np.random.default_rng(4)
    store = DatalabResultStore(enable_disk_cache=False)
    plotter = _MemoryPlottingService()
    period = 0.6487
    t_g = np.sort(rng.uniform(0, 60, 200))
    t_r = np.sort(rng.uniform(0, 60, 40))
    frame = pd.DataFrame({
        "mjd": np.concatenate([t_g, t_r]),
        "cmag": np.concatenate([
            21.5 + 0.5 * np.sin(2 * np.pi * t_g / period) + rng.normal(0, 0.02, t_g.size),
            21.0 + 0.4 * np.sin(2 * np.pi * t_r / period) + rng.normal(0, 0.02, t_r.size),
        ]),
        "filter": ["g"] * t_g.size + ["r"] * t_r.size,
    })
    lc = store.put(frame, {"catalog": "synthetic", "table": "lc"})
    out = datalab_analysis.period_fold(lc, min_frequency=1.0, max_frequency=10.0,
                                       result_store=store, plotting_service=plotter)
    assert out["band"] == "g"
    assert out["points"] == 200
    assert any("best-sampled" in w for w in out["warnings"])
    assert out["best_period_days"] == pytest.approx(period, abs=0.005)


class _FakeSvo:
    def wavelengths(self, filters):
        values = {"g": 0.48, "r": 0.62, "z": 0.91, "w1": 3.4, "w2": 4.6}
        return {f: {"effective_micron": values[f], "pivot_micron": values[f]} for f in filters}


def test_plotly_specs_on_remaining_plot_tools():
    """F-9: every Data Lab plot tool must emit a basic-bundle plotly_spec
    (scatter traces only — the frontend ships plotly.js-basic-dist-min)."""
    from services import datalab_analysis
    rng = np.random.default_rng(11)
    store = DatalabResultStore(enable_disk_cache=False)
    plotter = _MemoryPlottingService()

    # catalog_scatter
    n = 50
    df = pd.DataFrame({"bp_rp": rng.uniform(0, 2, n), "mg": rng.uniform(8, 15, n),
                       "pm": rng.uniform(50, 300, n)})
    rid = store.put(df, {"catalog": "gaia_dr3", "table": "gaia_source"})
    out = datalab_analysis.catalog_scatter(rid, "bp_rp", "mg", color_by="pm", invert_y=True,
                                           result_store=store, plotting_service=plotter)
    spec = out["plotly_spec"]
    assert spec["data"][0]["type"] == "scatter"
    assert spec["layout"]["yaxis"]["autorange"] == "reversed"
    assert spec["data"][0]["marker"]["showscale"] is True  # color_by wired

    # sky_density_map (hist2d): square-marker scatter, RA reversed
    dfd = pd.DataFrame({"ra_bin": rng.uniform(100, 102, 400), "dec_bin": rng.uniform(-1, 1, 400),
                        "source_count": rng.integers(1, 50, 400)})
    rid2 = store.put(dfd, {"catalog": "nsc_dr2", "table": "object"})
    out2 = datalab_analysis.sky_density_map(rid2, ra_col="ra_bin", dec_col="dec_bin",
                                            count_col="source_count",
                                            result_store=store, plotting_service=plotter)
    spec2 = out2["plotly_spec"]
    assert spec2["data"][0]["type"] == "scatter"
    assert spec2["data"][0]["marker"]["symbol"] == "square"
    assert spec2["layout"]["xaxis"]["autorange"] == "reversed"

    # period_fold: two-panel spec (periodogram line + folded markers)
    period = 0.6487
    t = np.sort(rng.uniform(0, 60, 200))
    mag = 21.5 + 0.5 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.02, t.size)
    rid3 = store.put(pd.DataFrame({"mjd": t, "cmag": mag}), {"catalog": "synthetic", "table": "lc"})
    out3 = datalab_analysis.period_fold(rid3, result_store=store, plotting_service=plotter)
    spec3 = out3["plotly_spec"]
    assert [tr["type"] for tr in spec3["data"]] == ["scatter", "scatter"]
    assert spec3["layout"]["grid"]["columns"] == 2
    assert spec3["layout"]["yaxis2"]["autorange"] == "reversed"

    # lss_wedge: 2D projection scatter with equal aspect
    lss = pd.DataFrame({"ra": rng.uniform(150, 155, 40), "dec": rng.uniform(-2, 2, 40),
                        "z": rng.uniform(0.05, 0.2, 40)})
    rid4 = store.put(lss, {"catalog": "sdss_dr17", "table": "specobj"})
    out4 = datalab_analysis.lss_wedge(rid4, result_store=store, plotting_service=plotter)
    spec4 = out4["plotly_spec"]
    assert spec4["data"][0]["type"] == "scatter"
    assert spec4["layout"]["yaxis"]["scaleanchor"] == "x"


def test_sed_plot_multi_object_overlay():
    """F-10: sample_n/row_indices overlay many SEDs with a median line (live
    P10 could only draw one 'example' SED for a 500-object sample)."""
    from services import datalab_analysis
    rng = np.random.default_rng(12)
    store = DatalabResultStore(enable_disk_cache=False)
    plotter = _MemoryPlottingService()
    n = 40
    frame = pd.DataFrame({
        "dered_mag_g": rng.uniform(20, 22, n),
        "dered_mag_r": rng.uniform(19, 21, n),
        "dered_mag_z": rng.uniform(18.5, 20.5, n),
        "dered_mag_w1": rng.uniform(18, 20, n),
        "dered_mag_w2": rng.uniform(18, 20, n),
    })
    rid = store.put(frame, {"catalog": "ls_dr9", "table": "tractor"})
    out = datalab_analysis.sed_plot(rid, sample_n=n, result_store=store,
                                    svo_client=_FakeSvo(), plotting_service=plotter)
    assert out["success"] and out["objects_plotted"] == n
    spec = out["plotly_spec"]
    # n faint per-object lines + 1 median trace
    assert len(spec["data"]) == n + 1
    assert spec["data"][-1]["name"].startswith("median of")
    assert spec["layout"]["xaxis"]["type"] == "log"
    # Single-object path unchanged (back-compat).
    single = datalab_analysis.sed_plot(rid, row_index=0, result_store=store,
                                       svo_client=_FakeSvo(), plotting_service=plotter)
    assert single["objects_plotted"] == 1
    assert len(single["plotly_spec"]["data"]) == 1
    # The 300-object cap is enforced and disclosed.
    big = pd.concat([frame] * 10, ignore_index=True)
    rid_big = store.put(big, {"catalog": "ls_dr9", "table": "tractor"})
    capped = datalab_analysis.sed_plot(rid_big, sample_n=400, result_store=store,
                                       svo_client=_FakeSvo(), plotting_service=plotter)
    assert capped["objects_plotted"] == 300
    assert any("capped" in w for w in capped["warnings"])
    # NO selector on a multi-row result -> sample-first default (two live P10
    # runs: the model never passed sample_n despite the description/prompt).
    auto = datalab_analysis.sed_plot(rid, result_store=store,
                                     svo_client=_FakeSvo(), plotting_service=plotter)
    assert auto["objects_plotted"] == n and auto["auto_sampled"] is True
    assert any("sample overlay" in w for w in auto["warnings"])
    assert len(auto["plotly_spec"]["data"]) == n + 1
    # ...but a single-row result still draws one object with no selector.
    rid_one = store.put(frame.iloc[:1], {"catalog": "ls_dr9", "table": "tractor"})
    one = datalab_analysis.sed_plot(rid_one, result_store=store,
                                    svo_client=_FakeSvo(), plotting_service=plotter)
    assert one["objects_plotted"] == 1 and "auto_sampled" not in one
