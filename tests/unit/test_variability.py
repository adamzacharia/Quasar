"""Generic variability analysis (services/variability.py) on synthetic light
curves with known answers, plus the source adapters. Offline."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from services import variability as va

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "mmdc"


def _lc(t, x, e, band="B", ul=None, idx=None, idx_err=None):
    n = len(t)
    return pd.DataFrame({"time_mjd": t, "value": x, "error": e, "band": band,
                         "is_upper_limit": ul if ul is not None else [False] * n,
                         "spectral_index": idx if idx is not None else np.nan,
                         "spectral_index_err": idx_err if idx_err is not None else np.nan})


# ── Fvar ────────────────────────────────────────────────────────────────────
def test_constant_light_curve_has_no_excess_variance():
    rng = np.random.default_rng(3)
    t = np.arange(200.0)
    x = 1.0 + rng.normal(0, 0.05, t.size)   # pure measurement noise
    fv = va.fvar(x, np.full(t.size, 0.05))
    # Fvar is undefined (excess variance <= 0) or consistent with zero
    assert "skipped" in fv or fv["Fvar"] < 2 * fv["Fvar_err"]


def test_fvar_matches_injected_intrinsic_scatter():
    rng = np.random.default_rng(4)
    n = 4000
    intrinsic = 0.30   # fractional rms of the source
    noise = 0.05
    x = 1.0 * (1 + rng.normal(0, intrinsic, n)) + rng.normal(0, noise, n)
    fv = va.fvar(x, np.full(n, noise))
    assert fv["Fvar"] == pytest.approx(intrinsic, rel=0.05)
    assert 0 < fv["Fvar_err"] < 0.02
    bs = va.fvar_bootstrap(x, np.full(n, noise), n_boot=300)
    # Vaughan's analytic error covers measurement noise only; the bootstrap also
    # carries the sampling error of the intrinsic scatter, ~ sigma / sqrt(2N).
    assert bs["Fvar_boot_std"] > fv["Fvar_err"]
    assert bs["Fvar_boot_std"] == pytest.approx(intrinsic / math.sqrt(2 * n), rel=0.35)


def test_fvar_vaughan_formula_by_hand():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    e = np.array([0.1] * 5)
    fv = va.fvar(x, e)
    s2, mse, mean, n = 2.5, 0.01, 3.0, 5
    f = math.sqrt((s2 - mse) / mean ** 2)
    err = math.sqrt((math.sqrt(1 / (2 * n)) * mse / (mean ** 2 * f)) ** 2 + (math.sqrt(mse / n) / mean) ** 2)
    assert fv["Fvar"] == pytest.approx(f) and fv["Fvar_err"] == pytest.approx(err)


def test_fvar_without_errors_is_skipped_with_reason():
    fv = va.fvar(np.array([1.0, 2.0, 3.0]), np.array([np.nan] * 3))
    assert "errors are missing" in fv["skipped"]


# ── Bayesian-block flares ───────────────────────────────────────────────────
def test_injected_flare_is_recovered():
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(0, 1000, 1500))
    x = np.ones_like(t)
    x[(t > 400) & (t < 430)] = 6.0
    x[(t > 700) & (t < 705)] = 1.0   # no second flare
    e = np.full_like(t, 0.2)
    x = x + rng.normal(0, 0.2, t.size)
    bb = va.bayesian_block_flares(t, x, e, p0=0.05, k=3.0)
    assert bb["n_blocks"] >= 3 and len(bb["flares"]) == 1
    f = bb["flares"][0]
    assert 395 <= f["start_mjd"] <= 405 and 425 <= f["end_mjd"] <= 435 and 400 <= f["peak_mjd"] <= 430
    assert f["peak_block_flux"] == pytest.approx(6.0, abs=0.1) and f["significance_sigma"] > 3
    assert bb["quiescent_flux"] == pytest.approx(1.0, abs=0.1)


def test_flat_light_curve_yields_no_flares_and_says_why():
    rng = np.random.default_rng(6)
    t = np.arange(300.0)
    bb = va.bayesian_block_flares(t, 1 + rng.normal(0, 0.1, t.size), np.full(t.size, 0.1))
    assert bb["flares"] == [] and "block" in bb["flare_note"]


def test_many_points_are_time_binned_before_blocks():
    t = np.linspace(0, 100, va.BB_MAX_POINTS * 2)
    x = 1 + 0.01 * np.sin(t)
    bb = va.bayesian_block_flares(t, x, np.full(t.size, 0.05))
    assert any("time-binned" in n for n in bb["notes"])


# ── lags ────────────────────────────────────────────────────────────────────
def _red_noise(n, rng, tau=20.0):
    y = np.zeros(n)
    a = math.exp(-1.0 / tau)
    for i in range(1, n):
        y[i] = a * y[i - 1] + rng.normal()
    return y


def test_injected_lag_is_recovered_by_the_dcf():
    rng = np.random.default_rng(7)
    true_lag = 8.0
    grid = np.arange(0, 1200.0, 1.0)
    sig = _red_noise(grid.size, rng)
    ta = np.sort(rng.choice(grid[50:-50], 400, replace=False))
    tb = np.sort(rng.choice(grid[50:-50], 400, replace=False))
    xa = np.interp(ta, grid, sig) + 10 + rng.normal(0, 0.1, ta.size)
    xb = np.interp(tb - true_lag, grid, sig) + 10 + rng.normal(0, 0.1, tb.size)  # b lags a
    d = va.dcf_lag(ta, xa, np.full(ta.size, 0.1), tb, xb, np.full(tb.size, 0.1), max_lag=40, bin_width=2.0,
                   n_boot=100, seed=1)
    ci = d["centroid_lag_bootstrap"]
    assert ci is not None
    assert abs(d["centroid_lag_d"] - true_lag) <= max(2.5, 2 * max(ci["minus"], ci["plus"]))
    assert ci["median"] - 3 * ci["minus"] <= true_lag <= ci["median"] + 3 * ci["plus"]
    assert d["peak_dcf"] > 0.5


def test_flare_matching_lag():
    a = [{"peak_mjd": 100.0}, {"peak_mjd": 500.0}]
    b = [{"peak_mjd": 104.0}, {"peak_mjd": 900.0}]
    m = va.match_flare_lags(a, b, window_d=30)
    assert m == [{"peak_mjd_a": 100.0, "peak_mjd_b": 104.0, "lag_d": 4.0}]


def test_analyze_computes_rest_frame_lag_and_sign_convention():
    rng = np.random.default_rng(8)
    grid = np.arange(0, 800.0, 1.0)
    sig = _red_noise(grid.size, rng)
    t = grid[40:-40:2]
    a = _lc(t, np.interp(t, grid, sig) + 20, np.full(t.size, 0.1), band="MMDCXRT")
    b = _lc(t, np.interp(t - 6, grid, sig) + 20, np.full(t.size, 0.1), band="MMDCGR")
    res = va.analyze(pd.concat([a, b]), z=0.5, n_boot=100, n_boot_dcf=60)
    lag = res["lag"]
    assert lag["band_a"] == "MMDCXRT" and lag["band_b"] == "MMDCGR" and "MMDCGR lags MMDCXRT" in lag["sign_convention"]
    assert lag["rest_frame_centroid_lag_d"] == pytest.approx(lag["dcf"]["centroid_lag_d"] / 1.5)
    assert abs(lag["dcf"]["centroid_lag_d"] - 6) < 4


# ── spectral trend ──────────────────────────────────────────────────────────
def test_injected_index_flux_slope_is_recovered_and_labelled():
    rng = np.random.default_rng(9)
    flux = 10 ** rng.uniform(-11, -9.5, 300)
    idx = 2.0 - 0.3 * np.log10(flux / 1e-10) + rng.normal(0, 0.03, flux.size)   # harder when brighter
    tr = va.index_flux_trend(flux, idx)
    assert tr["slope"] == pytest.approx(-0.3, abs=3 * tr["slope_err"] + 0.01)
    assert tr["pearson_p"] < 1e-6 and tr["spearman_p"] < 1e-6 and tr["trend"] == "harder when brighter"
    noise = va.index_flux_trend(flux, 2.0 + rng.normal(0, 0.1, flux.size))
    labelled = noise["pearson_p"] < 0.05 and noise["spearman_p"] < 0.05
    assert (noise["trend"] != "no significant trend") == labelled   # the label follows the stated rule exactly
    soft = va.index_flux_trend(flux, 2.0 + 0.3 * np.log10(flux / 1e-10) + rng.normal(0, 0.03, flux.size))
    assert soft["trend"] == "softer when brighter"
    assert "fewer than 5" in va.index_flux_trend(flux[:3], idx[:3])["skipped"]


# ── orchestration, cleaning, adapters ───────────────────────────────────────
def test_analyze_skips_sparse_bands_and_excludes_upper_limits():
    t = np.arange(50.0)
    good = _lc(t, 1 + 0.5 * np.sin(t / 5), np.full(t.size, 0.05), band="ZTF:r")
    ul = _lc(t[:10], np.full(10, 9.0), np.full(10, np.nan), band="ZTF:r", ul=[True] * 10)
    sparse = _lc(t[:3], [1, 2, 3], [0.1] * 3, band="ZTF:g")
    res = va.analyze(pd.concat([good, ul, sparse]), n_boot=50)
    rows = {r["band"]: r for r in res["per_band"]}
    assert rows["ZTF:r"]["n_upper_limits_excluded"] == 10 and rows["ZTF:r"]["n_used"] == 50
    assert "Fvar" in rows["ZTF:r"] and "only 3 detections" in rows["ZTF:g"]["skipped"]


def test_duplicate_epochs_are_merged_by_inverse_variance():
    d = _lc([1.0, 1.0, 2.0], [1.0, 3.0, 5.0], [1.0, 1.0, 1.0])
    out, st = va.clean(d)
    assert st["n_duplicates_merged"] == 1 and len(out) == 2
    assert out.iloc[0]["value"] == pytest.approx(2.0) and out.iloc[0]["error"] == pytest.approx(1 / math.sqrt(2))


def test_upper_limit_never_merges_with_a_detection_at_the_same_epoch():
    d = _lc([1.0, 1.0], [1.0, 9.0], [0.1, np.nan], ul=[False, True])
    out, st = va.clean(d)
    assert st["n_duplicates_merged"] == 0 and len(out) == 2
    assert out.loc[~out["is_upper_limit"], "value"].tolist() == [1.0]


def test_band_aliases():
    avail = ["MMDCGR", "MMDCXRT", "ZTF:R", "MMDCOUV:W1"]
    assert va.resolve_bands(["gamma", "x-ray"], avail) == ["MMDCGR", "MMDCXRT"]
    assert va.resolve_bands(["optical"], avail) == ["ZTF:R"]
    assert va.resolve_bands(["W1"], avail) == ["MMDCOUV:W1"]


def test_adapter_for_datalab_magnitudes_and_generic_flux_tables():
    smash = pd.DataFrame({"mjd": [56000.0, 56001.0], "filter": ["g", "g"], "cmag": [20.0, 19.0], "cerr": [0.1, 0.1]})
    t, prov = va.adapt_table(smash)
    assert t["value"].iloc[1] / t["value"].iloc[0] == pytest.approx(10 ** 0.4)
    assert t["error"].iloc[0] == pytest.approx(t["value"].iloc[0] * 0.4 * math.log(10) * 0.1)
    assert "magnitude" in prov["adapter"]
    flux = pd.DataFrame({"time": [1.0, 2.0], "flux": [3.0, 4.0], "flux_err": [0.1, 0.2]})
    t2, _ = va.adapt_table(flux)
    assert list(t2["band"]) == ["all", "all"] and t2["error"].tolist() == [0.1, 0.2]
    with pytest.raises(ValueError, match="no time column"):
        va.adapt_table(pd.DataFrame({"x": [1]}))


def test_adapter_for_mmdc_contract_fixture():
    from services import mmdc_service as ms

    lc = ms.normalize_lightcurve_rows(json.loads((FIX / "lc_1es1959_trim.json").read_text()))
    t, prov = va.adapt_table(lc)
    assert list(t.columns) == va.CONTRACT and prov["adapter"].startswith("light-curve contract")
    assert t.loc[t["band"] == "MMDCGR", "spectral_index"].notna().all()


def test_alerce_adapter_uses_reference_corrected_magnitudes():
    """review CX-B02: magpsf is a difference-image magnitude; only magpsf_corr measures the source."""
    lc = {"detections": [{"mjd": 1.0, "magpsf": 21.5, "sigmapsf": 0.2, "magpsf_corr": 18.0, "sigmapsf_corr": 0.05, "fid": 1},
                         {"mjd": 2.0, "magpsf": 21.0, "sigmapsf": 0.2, "fid": 2}],   # no corrected magnitude
          "non_detections": [{"mjd": 3.0, "diffmaglim": 20.0, "fid": 2}]}
    t, info = va.from_alerce(lc)
    assert t["band"].tolist() == ["ZTF:g"] and t["value"].iloc[0] == pytest.approx(10 ** (-0.4 * 18.0))
    assert info["n_without_corrected_magnitude"] == 1 and "difference-image" in info["non_detections"]
    assert not t["is_upper_limit"].any()


# ── service (fake sources) and plots ────────────────────────────────────────
class _Plot:
    def _apply_style(self, dark=False):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt

    def _save_and_encode(self, fig, name):
        import matplotlib.pyplot as plt

        plt.close(fig)
        return {"web_url": f"/plots/{name}.png", "base64_png": "AAAA"}


def test_service_on_ztf_source_builds_table_and_figures():
    rng = np.random.default_rng(10)
    mags = 18 + 0.4 * np.sin(np.arange(120) / 8) + rng.normal(0, 0.02, 120)
    det = [{"mjd": 58000.0 + i, "magpsf": float(m) + 2, "sigmapsf": 0.1, "magpsf_corr": float(m),
            "sigmapsf_corr": 0.02, "fid": 2} for i, m in enumerate(mags)]
    alerce = SimpleNamespace(light_curve=lambda oid: {"success": True, "detections": det, "non_detections": []})
    svc = va.VariabilityService(mmdc=SimpleNamespace(plotting_service=_Plot()), alerce=alerce, plotting_service=_Plot())
    out = svc.run(source="ztf", identifier="ZTF18aaaaaaa")
    assert out["success"] and out["bands_analysed"] == ["ZTF:r"] and "attribution" not in out
    table, label = out["_table"]
    assert list(table["band"]) == ["ZTF:r"] and table["Fvar"].iloc[0] > 0
    assert out["_figures"][0]["plotly_spec"]["data"] and "Bayesian blocks" in out["_figures"][0]["caption"]
    assert all(tr["type"] == "scatter" for tr in out["_figures"][0]["plotly_spec"]["data"])
    with pytest.raises(ValueError, match="identifier"):
        svc.run(source="ztf")


def test_service_rejects_unknown_input():
    svc = va.VariabilityService(mmdc=SimpleNamespace(plotting_service=_Plot()), plotting_service=_Plot())
    with pytest.raises(ValueError, match="give result_id"):
        svc.run()


def test_answer_verifier_accepts_the_cleaning_row_count():
    """UI 2026-09-26 Q5/Q6/Q9: 'N rows' was flagged although cleaning carried the count."""
    from core.answer_verifier import build_trace_summary, verify_answer

    t = np.arange(40.0)
    res = va.analyze(_lc(t, 1 + 0.3 * np.sin(t), np.full(t.size, 0.05)), n_boot=20)
    ts = build_trace_summary([{"output": json.dumps({"success": True, "cleaning": res["cleaning"]})}], [], [])
    ts.query_arg_texts.append("{}")
    assert not verify_answer(f"The light curve had {res['cleaning']['n_rows']} rows.", ts).unsupported


def test_output_rounding_keeps_mjd_decimals():
    out = va._r({"start_mjd": 59770.123456, "Fvar": 0.2867481, "flares": [{"peak_mjd": 58000.98765}],
                 "duration_d": 2.51999, "lag": {"median": -5.9500000001935, "n": 200}})
    assert out["start_mjd"] == 59770.123 and out["flares"][0]["peak_mjd"] == 58000.988   # not 59770.0
    assert out["Fvar"] == 0.2867 and out["duration_d"] == 2.52 and out["lag"] == {"median": -5.95, "n": 200}


def test_dcf_and_index_plot_specs():
    lag = {"band_a": "A", "band_b": "B", "dcf": {"lag_bins_d": [-1.0, 0.0, 1.0], "dcf": [0.1, 0.9, None],
                                                 "dcf_err": [0.05, 0.05, None], "peak_lag_d": 0.0, "centroid_lag_d": 0.1}}
    spec = va.dcf_plotly_spec(lag, "t")
    assert spec["data"][0]["x"] == [-1.0, 0.0] and len(spec["layout"]["shapes"]) == 2


# ── Round B independent review (reviewB ledger) regressions ─────────────────
def test_b01_dcf_zero_lag_peaks_at_zero():
    rng = np.random.default_rng(11)
    grid = np.arange(0, 1000.0, 1.0)
    sig = _red_noise(grid.size, rng)
    t = np.sort(rng.choice(grid[20:-20], 400, replace=False))
    x = np.interp(t, grid, sig) + 10
    d = va.dcf_lag(t, x + rng.normal(0, 0.05, t.size), np.full(t.size, 0.05), t, x + rng.normal(0, 0.05, t.size),
                   np.full(t.size, 0.05), max_lag=100, bin_width=10.0, n_boot=60, seed=2)
    assert d["peak_lag_d"] == 0.0 and 0.0 in d["lag_bins_d"]   # bins are centred on zero lag


def test_b03_b05_window_applies_to_every_source_with_exclusive_expanded_end():
    det = [{"mjd": m, "magpsf": 20.0, "sigmapsf": 0.1, "magpsf_corr": 18 + 0.1 * np.sin(m), "sigmapsf_corr": 0.02,
            "fid": 2} for m in np.arange(58300.0, 60000.0, 5.0)] + [
           {"mjd": 58849.0, "magpsf": 20.0, "sigmapsf": 0.1, "magpsf_corr": 18.0, "sigmapsf_corr": 0.02, "fid": 2}]
    alerce = SimpleNamespace(light_curve=lambda oid: {"success": True, "detections": det, "non_detections": []})
    svc = va.VariabilityService(mmdc=SimpleNamespace(plotting_service=_Plot()), alerce=alerce, plotting_service=_Plot())
    out = svc.run(source="ztf", identifier="ZTF18x", start_date="2019", end_date="2019")
    w = out["window"]
    assert w["applied"] and w["end_exclusive"] and w["end_date"] == "2019-12-31" and w["rows_kept"] < w["rows_before"]
    rng = out["per_band"][0]["MJD_range"]
    assert rng[0] >= 58484.0 and rng[1] < 58849.0            # 2020-01-01 00:00 (MJD 58849) is excluded
    tess = SimpleNamespace(fetch_lightcurve_arrays=lambda *a, **k: (np.arange(100.0), np.ones(100), "flux", {}))
    svc2 = va.VariabilityService(mmdc=SimpleNamespace(plotting_service=_Plot()), lightcurves=tess, plotting_service=_Plot())
    with pytest.raises(ValueError, match="mission days"):
        svc2.run(source="tess", identifier="x", start_date="2019")


def test_b09_merge_off_with_duplicate_times_does_not_crash():
    t = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    res = va.analyze(_lc(t, [1, 1.2, 1.1, 5, 1, 1, 1.1], [0.05] * 7), merge_duplicates=False, n_boot=20)
    assert res["per_band"][0]["n_blocks"] >= 1
    assert va._time_bin(np.ones(5000), np.ones(5000), np.ones(5000), 10)[3] is None   # zero span


@pytest.mark.parametrize("kw,msg", [({"max_lag_d": -5}, "max_lag_days"), ({"dcf_bin_d": -1}, "dcf_bin_days"),
                                    ({"max_lag_d": 5000, "dcf_bin_d": 0.1}, "DCF bins")])
def test_b10_dcf_parameters_are_validated(kw, msg):
    t = np.arange(0.0, 300.0)
    a = _lc(t, 1 + 0.1 * np.sin(t / 7), np.full(t.size, 0.02), band="MMDCXRT")
    b = _lc(t, 1 + 0.1 * np.sin(t / 7), np.full(t.size, 0.02), band="MMDCGR")
    with pytest.raises(ValueError, match=msg):
        va.analyze(pd.concat([a, b]), n_boot=10, n_boot_dcf=5, **kw)


def test_b14_upper_limit_strings_and_nan_parse_correctly():
    s = va.as_bool_series(["False", "True", "0", "1", np.nan, None, "UL", True])
    assert s.tolist() == [False, True, False, True, False, False, True, True]
    t, _ = va.adapt_table(pd.DataFrame({"time": [1.0, 2.0, 3.0], "flux": [1.0, 2.0, 3.0],
                                        "upper_limit": ["False", "False", "True"]}))
    assert t["is_upper_limit"].tolist() == [False, False, True]
