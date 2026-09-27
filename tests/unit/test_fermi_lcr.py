"""Fermi LAT Light Curve Repository tool (services/fermi_lcr.py). Offline, on a
trimmed live response (Mkn 421 weekly, free index, 2026-09-26)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services import fermi_lcr as fl
from services import mmdc_service as ms

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "mmdc"
RAW = json.loads((FIX / "fermi_lcr_mkn421_weekly_trim.json").read_text())
SOURCES = json.loads((FIX / "fermi_lcr_sources_trim.json").read_text())


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


def _svc(tmp_path, monkeypatch, body=RAW):
    monkeypatch.setenv("MMDC_CACHE_DIR", str(tmp_path))
    calls = []

    def fetch(params, seconds):
        calls.append(params)
        if params["typeOfRequest"] == "SourceList":
            return json.dumps([{"Source_Name": s["name"], "ASSOC1": s["assoc"], "RAJ2000": str(s["ra"]),
                                "DEJ2000": str(s["dec"]), "CLASS1": s["class"]} for s in SOURCES])
        return "<html>\n" + json.dumps(body)   # served as text/html with JSON inside

    return fl.FermiLcrService(fetch=fetch, plotting_service=_Plot()), calls


def test_met_to_mjd():
    assert fl.met_to_mjd(0) == 51910.0
    assert fl.met_to_mjd(86400 * 365) == pytest.approx(52275.0)


def test_normalize_follows_the_light_curve_contract_and_drops_zero_width_bins():
    df, dropped = fl.normalize_lcr(RAW)
    assert list(df.columns) == ms.LC_COLUMNS
    assert dropped == 1   # the live 1.04e-3 bin with lo == hi
    assert df["value"].max() < 1e-5
    det = df[~df["is_upper_limit"]]
    assert (det["error"] > 0).all() and det["spectral_index"].notna().all()
    assert int(df["is_upper_limit"].sum()) == len(RAW["flux_upper_limits"])
    assert df["time_mjd"].is_monotonic_increasing


def test_lightcurve_by_association_name_with_window(tmp_path, monkeypatch):
    svc, calls = _svc(tmp_path, monkeypatch)
    out = svc.lightcurve(target_name="Mkn 421", cadence="weekly", index_type="free")
    assert out["success"] and out["lcr_source"]["name"] == "4FGL J1104.4+3812"
    assert out["lcr_source"]["matched_by"].startswith("4FGL name")
    lc_call = [c for c in calls if c["typeOfRequest"] == "lightCurveData"][0]
    assert lc_call["source_name"] == "4FGL J1104.4+3812" and lc_call["index_type"] == "free"
    table, _ = out["_table"]
    assert out["rows"] == len(table) and out["n_bins_dropped_without_error"] == 1
    assert out["attribution"]["citation"].startswith("Abdollahi")
    assert all(t["type"] == "scatter" for t in out["_figure"]["plotly_spec"]["data"])
    mid = float(table["time_mjd"].median())
    win = svc.lightcurve(target_name="Mkn 421", start_date=f"{mid:.1f}")
    assert win["_table"][0]["time_mjd"].min() >= mid - 1e-6
    # the source list is cached on disk: a new service does not refetch it
    svc2, calls2 = _svc(tmp_path, monkeypatch)
    svc2.lightcurve(target_name="Mkn 421")
    assert not any(c["typeOfRequest"] == "SourceList" for c in calls2)


def test_unmonitored_4fgl_source_is_refused(tmp_path, monkeypatch):
    svc, _ = _svc(tmp_path, monkeypatch, body={"flux": [], "flux_upper_limits": []})
    out = svc.lightcurve(target_name="M82")
    assert out["success"] is False and out["not_in_lcr"] and "monitors only variable" in out["error"]


def test_position_match_and_no_match(tmp_path, monkeypatch):
    svc, _ = _svc(tmp_path, monkeypatch)
    hit = svc.lightcurve(ra=166.119, dec=38.207)
    assert hit["lcr_source"]["matched_by"] == "position"
    miss = svc.lightcurve(ra=10.0, dec=-40.0)
    assert miss["success"] is False and miss["not_in_lcr"]


@pytest.mark.parametrize("kw", [{"cadence": "3-day"}, {"flux_type": "counts"}, {"index_type": "variable"}, {"ts_min": 0}])
def test_bad_parameters_raise(tmp_path, monkeypatch, kw):
    svc, _ = _svc(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        svc.lightcurve(target_name="Mkn 421", **kw)


def test_variability_runs_on_the_lcr_table(tmp_path, monkeypatch):
    from services.variability import adapt_table, analyze

    svc, _ = _svc(tmp_path, monkeypatch)
    table, _ = svc.lightcurve(target_name="Mkn 421", index_type="free")["_table"]
    t, prov = adapt_table(table)
    assert prov["adapter"].startswith("light-curve contract")
    row = analyze(t, n_boot=50)["per_band"][0]
    assert row["band"] == "FermiLAT-LCR" and row["n_upper_limits_excluded"] == len(RAW["flux_upper_limits"])
    assert "Fvar" in row and 0 < row["Fvar"] < 3   # the zero-width outlier bin is gone (it drove Fvar to 15 live)


def test_budget_declared():
    from services import tool_budgets as tb

    assert "fermi_lcr_lightcurve" in tb.INNER_BUDGETS and not tb.check_hierarchy(["fermi_lcr_lightcurve"])


# ── Round B independent review (reviewB ledger) regressions ─────────────────
def test_b05_expanded_end_is_exclusive(tmp_path, monkeypatch):
    df, _ = fl.normalize_lcr(RAW)
    edge = float(df["time_mjd"].iloc[len(df) // 2])
    body = {k: list(v) for k, v in RAW.items()}
    met = int(round((edge - fl.MET_EPOCH_MJD) * 86400))
    svc, _ = _svc(tmp_path, monkeypatch, body=body)
    from services.mmdc_service import mjd_to_date

    day = mjd_to_date(edge - 1)   # an end DATE that expands to exactly the next day's 00:00
    out = svc.lightcurve(target_name="Mkn 421", end_date=day)
    assert out["window"]["end_exclusive"] and (out["_table"][0]["time_mjd"] < out["window"]["end_mjd"]).all()


def test_b04_non_object_body_is_a_clear_error(tmp_path, monkeypatch):
    svc, _ = _svc(tmp_path, monkeypatch)
    svc._fetch_fn = lambda params, s: "[]" if params["typeOfRequest"] == "lightCurveData" else json.dumps(
        [{"Source_Name": "4FGL J1104.4+3812", "ASSOC1": "Mkn 421", "RAJ2000": "166.119", "DEJ2000": "38.207"}])
    with pytest.raises(ValueError, match="unexpected body"):
        svc.lightcurve(target_name="Mkn 421")


def test_b15_all_bins_dropped_note_is_honest(tmp_path, monkeypatch):
    body = {"flux": [[1, 1e-3]], "flux_error": [[1, 1e-3, 1e-3]], "flux_upper_limits": []}
    svc, _ = _svc(tmp_path, monkeypatch, body=body)
    out = svc.lightcurve(target_name="Mkn 421")
    assert out["rows"] == 0 and "zero-width" in out["note"] and "window" not in out["note"]
