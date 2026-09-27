"""MMDC SED + light-curve service (services/mmdc_service.py) and its agent wiring.

Offline: every test uses the trimmed live fixtures in tests/fixtures/mmdc/
(recorded 2026-09-26, see tmp/mmdc-integration-2026-09-26/PHASE0.md) or a fake
SDK client. Live checks are in test_mmdc_live.py behind RUN_LIVE_MMDC_TESTS=1.
"""

from __future__ import annotations

import io
import json
import math
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from services import mmdc_service as ms

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "mmdc"


def _csv(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def _json(name: str):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def _loader():
    return _json("known_sources_trim.json"), _json("gamma_ray_sources_trim.json")


def _row(fs, fe, flag="", cat="X", freq=1e17, flux=1e-11, err=1e-12):
    return {"frequency": freq, "flux": flux, "flux_err": err, "MJD_start": fs, "MJD_end": fe, "flag": flag,
            "catalog": cat, "reference": "ref"}


# ── CSV normalization ───────────────────────────────────────────────────────
def test_normalize_marks_undated_placeholder_and_all_ul_vocabularies():
    df = ms.normalize_sed_csv(_csv("sed_1es1959_trim.csv"))
    assert list(df.columns) == ms.SED_COLUMNS
    raw = pd.read_csv(FIX / "sed_1es1959_trim.csv", dtype={"flag": str})
    placeholder = (raw.MJD_start == 55000.0) & (raw.MJD_end == 55000.0)
    assert int(df["undated"].sum()) == int(placeholder.sum()) > 0
    assert df.loc[df["undated"], "MJD_start"].isna().all()
    assert int(df["upper_limit"].sum()) == int((raw.flag.fillna("").str.strip() == "UL").sum()) > 0
    # Mkn 421 uses the 'Det' / 'UL' vocabulary instead of NaN / blank / 'UL'
    dm = ms.normalize_sed_csv(_csv("sed_mkn421_trim.csv"))
    rawm = pd.read_csv(FIX / "sed_mkn421_trim.csv", dtype={"flag": str})
    assert set(rawm.flag.dropna().str.strip()) <= {"Det", "UL"}
    assert int(dm["upper_limit"].sum()) == int((rawm.flag.str.strip() == "UL").sum()) > 0


def test_normalize_units_energy_and_ranges():
    df = ms.normalize_sed_csv(pd.DataFrame([_row(55100, 55101, freq=2.418e17), _row(55100, 55101, freq=1.4e9)]))
    assert df["energy_eV"].iloc[0] == pytest.approx(2.418e17 * 4.135667696e-15)  # ~1 keV
    assert list(df["range"]) == ["xray", "radio"]
    assert ms.classify_range(2.5e22) == "gamma" and ms.classify_range(5e14) == "optical"
    assert ms.classify_range(1.2e15) == "uv" and ms.classify_range(1e13) == "infrared"


def test_normalize_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        ms.normalize_sed_csv("frequency,flux\n1,2\n")


# ── window semantics ────────────────────────────────────────────────────────
def test_window_edge_equality_is_contained_and_straddlers_are_overlap_only():
    df = ms.normalize_sed_csv(pd.DataFrame([
        _row(55000.5, 55010.0, cat="A"),      # exactly [start, end] -> contained
        _row(55005.0, 55010.01, cat="B"),     # ends just after -> overlap only
        _row(54990.0, 55001.0, cat="C"),      # starts before -> overlap only
        _row(55011.0, 55012.0, cat="D"),      # outside
        _row(55000.0, 55000.0, cat="NVSS"),   # undated placeholder
    ]))
    kept, rep = ms.apply_window(df, 55000.5, 55010.0, "contained")
    assert list(kept["catalog"]) == ["A"]
    assert rep["rows_kept_by_contained_rule"] == 1 and rep["rows_kept_by_overlap_rule"] == 3
    assert rep["straddling_rows"] == 2 and set(rep["straddling_by_catalog"]) == {"B", "C"}
    assert rep["dated_rows_outside_window"] == 1 and rep["undated_archival_rows"] == 1
    assert rep["undated_included"] is False and "include_undated=true" in rep["undated_note"]
    assert rep["rule_applied"] == "contained" and "MJD_end <= window end" in rep["rule_text"]
    kept_o, rep_o = ms.apply_window(df, 55000.5, 55010.0, "overlap")
    assert set(kept_o["catalog"]) == {"A", "B", "C"} and "extend outside" in rep_o["straddling_note"]
    kept_u, rep_u = ms.apply_window(df, 55000.5, 55010.0, "contained", include_undated=True)
    assert set(kept_u["catalog"]) == {"A", "NVSS"} and "NOT evidence" in rep_u["undated_note"]


def test_no_window_keeps_everything_including_undated():
    df = ms.normalize_sed_csv(_csv("sed_1es1959_trim.csv"))
    kept, rep = ms.apply_window(df, None, None)
    assert len(kept) == len(df) and rep["rule_applied"] == "none" and rep["undated_included"] is True


def test_competitor_window_contains_no_bin_ending_after_2020():
    """AstroGenesis returned max MJD_end 59694.67 for 2008-2020: the server's
    overlap filter kept a Fermi bin ending in April 2022."""
    df = ms.normalize_sed_csv(_csv("sed_1es1959_trim.csv"))
    s, _ = ms.parse_window_bound("2008", is_end=False)
    e, note = ms.parse_window_bound("2020", is_end=True)
    assert (s, e) == (54466.0, 59215.0) and "expanded" in note
    kept, rep = ms.apply_window(df, s, e, "contained")
    assert kept["MJD_end"].max() <= 59214.0 + 1e-9
    assert rep["kept_max_MJD_end"] <= 59214.0
    assert rep["straddling_by_catalog"]["MMDCGR"]["latest_MJD_end"] == pytest.approx(59694.67)
    kept_o, _ = ms.apply_window(df, s, e, "overlap")
    assert kept_o["MJD_end"].max() == pytest.approx(59694.67)  # what the server-side filter returns
    assert rep["rows_kept_by_overlap_rule"] - rep["rows_kept_by_contained_rule"] == rep["straddling_rows"]


def test_empty_window_reports_undated_honestly():
    df = ms.normalize_sed_csv(_csv("sed_1es1959_trim.csv"))
    s, _ = ms.parse_window_bound("1990", is_end=False)
    e, _ = ms.parse_window_bound("1991", is_end=True)
    kept, rep = ms.apply_window(df, s, e)
    assert kept.empty and rep["undated_archival_rows"] > 0 and rep["kept_max_MJD_end"] is None


def test_window_rejects_bad_mode_and_reversed_bounds():
    df = ms.normalize_sed_csv(pd.DataFrame([_row(55100, 55101)]))
    with pytest.raises(ValueError):
        ms.apply_window(df, 55100, 55200, "loose")
    with pytest.raises(ValueError):
        ms.apply_window(df, 55200, 55100)


@pytest.mark.parametrize("value,is_end,expected", [
    ("2008", False, 54466.0), ("2020", True, 59215.0), (2020, True, 59215.0),
    ("2010-02", False, 55228.0), ("2010-02", True, 55256.0), ("2010-12", True, 55562.0),
    ("2020-12-31", True, 59215.0), ("2020-12-31", False, 59214.0),
    ("59214", True, 59214.0), (59214.5, False, 59214.5),
])
def test_parse_window_bound(value, is_end, expected):
    assert ms.parse_window_bound(value, is_end=is_end)[0] == pytest.approx(expected)


@pytest.mark.parametrize("value", ["20-20", 123, "not a date", float("nan"), "2010-13"])
def test_parse_window_bound_rejects(value):
    with pytest.raises(ValueError):
        ms.parse_window_bound(value, is_end=False)


def test_filter_catalogs_and_ranges():
    df = ms.normalize_sed_csv(_csv("sed_1es1959_trim.csv"))
    out, info = ms.filter_catalogs_and_ranges(df, ["mmdcxrt", "NOPE"], ["gamma", "X-ray"])
    assert "MMDCXRT" not in set(out["catalog"]) and set(out["range"]) <= {"gamma", "xray"}
    assert info["exclude_catalogs_not_present"] == ["nope"]
    with pytest.raises(ValueError, match="unknown include_ranges"):
        ms.filter_catalogs_and_ranges(df, None, ["submillimetre-ish"])


# ── summary, downsampling, plot specs ───────────────────────────────────────
def test_summary_counts_match_table():
    df = ms.normalize_sed_csv(_csv("sed_1es1959_trim.csv"))
    summ = ms.summarize_sed(df)
    assert summ["totals"]["rows"] == len(df)
    assert summ["totals"]["upper_limits"] == int(df["upper_limit"].sum())
    assert sum(g["n"] for g in summ["by_catalog_and_range"]) + summ.get("other_groups", {}).get("rows", 0) == len(df)
    xrt = next(g for g in summ["by_catalog_and_range"] if g["catalog"] == "MMDCXRT")
    assert xrt["date_range"][0] <= xrt["date_range"][1]


def test_downsample_keeps_every_upper_limit_and_extremes():
    rng = np.random.default_rng(1)
    n = 9000
    rows = [_row(55000.5 + i * 0.1, 55000.5 + i * 0.1, cat="MMDCXRT", freq=10 ** rng.uniform(17, 18.4),
                 flux=10 ** rng.uniform(-11, -9)) for i in range(n)]
    rows += [_row(56000.5 + i, 56000.5 + i, flag="UL", cat="MMDCGR", freq=1e23, flux=1e-12) for i in range(40)]
    df = ms.normalize_sed_csv(pd.DataFrame(rows))
    sub, info = ms.downsample_sed(df, 2000)
    assert info["downsampled"] and info["table_rows"] == len(df) and len(sub) <= 2000 + 10
    assert int(sub["upper_limit"].sum()) == 40
    det = df[~df["upper_limit"]]
    for col in ("nuFnu_erg_cm2_s", "frequency_Hz", "MJD_mid"):
        assert det[col].idxmax() in sub.index and det[col].idxmin() in sub.index
    assert "CSV export hold all rows" in info["note"]
    small, info2 = ms.downsample_sed(df.head(50))
    assert len(small) == 50 and not info2["downsampled"]


def test_sed_plotly_spec_epoch_colorbar_log_axes_and_ul_triangles():
    df = ms.normalize_sed_csv(_csv("sed_1es1959_trim.csv"))
    spec = ms.sed_plotly_spec(df, "t", color_by="epoch")
    lay = spec["layout"]
    assert lay["xaxis"]["type"] == "log" and lay["yaxis"]["type"] == "log"
    assert "erg" in lay["yaxis"]["title"]["text"] and "Hz" in lay["xaxis"]["title"]["text"]
    assert lay["xaxis2"]["range"][0] == pytest.approx(lay["xaxis"]["range"][0] + math.log10(ms.H_EV_S))
    names = [t["name"] for t in spec["data"]]
    det = spec["data"][names.index("detections (colour = epoch)")]
    ul = spec["data"][names.index("upper limits")]
    cb = det["marker"]["colorbar"]
    assert all(txt.isdigit() and 1990 < int(txt) < 2100 for txt in cb["ticktext"])
    assert ul["marker"]["symbol"] == "triangle-down" and ul["error_y"]["arrayminus"][0] > 0
    dated = df[~df["undated"]]
    assert len(ul["x"]) == int(dated["upper_limit"].sum())
    assert len(det["x"]) == int((~dated["upper_limit"]).sum())
    assert "archival, undated" in names
    assert all(t["type"] == "scatter" for t in spec["data"])  # basic plotly bundle has no scattergl
    assert "UPPER LIMIT" in {c[6] for c in ul["customdata"]}


def test_sed_plotly_spec_catalog_mode_and_model_curve():
    df = ms.normalize_sed_csv(_csv("sed_mkn421_trim.csv"))
    spec = ms.sed_plotly_spec(df, "t", color_by="catalog",
                              model_curves=[{"name": "SSC best fit", "nu": [1e10, 1e20], "nuFnu": [1e-12, 1e-11]}])
    names = [t["name"] for t in spec["data"]]
    assert "SSC best fit" in names and any(n.endswith("upper limits") for n in names)
    assert spec["data"][names.index("SSC best fit")]["mode"] == "lines"


# ── light curves ────────────────────────────────────────────────────────────
def test_lightcurve_contract_and_spectral_indices():
    df = ms.normalize_lightcurve_rows(_json("lc_1es1959_trim.json"))
    assert list(df.columns) == ms.LC_COLUMNS
    gr = df[df["catalog"] == "MMDCGR"]
    assert len(gr) and gr["spectral_index"].notna().all() and gr["unit"].str.contains("ph").all()
    assert (df[df["catalog"] == "MMDCXRT"]["spectral_index"].notna()).all()
    assert (df[df["catalog"] == "ZTF"]["spectral_index"].isna()).all()
    assert set(df.loc[df["catalog"] == "MMDCOUV", "band"]) <= {f"MMDCOUV:{b}" for b in ("U", "B", "V", "W1", "M2", "W2")}
    assert df["date"].str.match(r"\d{4}-\d{2}-\d{2}").all()
    summ = ms.summarize_lightcurve(df)
    assert sum(b["n"] for b in summ) == len(df)


def test_lightcurve_plotly_panels_and_upper_limits():
    rows = _json("lc_1es1959_trim.json")
    rows.append({**rows[0], "is_upper_limit": True, "mjd_mid": 60000.0})
    df = ms.normalize_lightcurve_rows(rows)
    spec, _ = ms.lightcurve_plotly_spec(df, "lc")
    lay = spec["layout"]
    ytitles = [lay[k]["title"]["text"] for k in lay if k.startswith("yaxis")]
    assert any("gamma-ray" in t for t in ytitles) and any("X-ray" in t for t in ytitles)
    assert any(t.get("marker", {}).get("symbol") == "triangle-down" for t in spec["data"])
    assert lay["xaxis2"]["ticktext"] and lay["xaxis"]["title"]["text"] == "MJD"


def test_downsample_trace_keeps_upper_limits_and_extremes():
    g = pd.DataFrame({"time_mjd": np.arange(1000.0), "value": np.sin(np.arange(1000.0)),
                      "is_upper_limit": [i % 97 == 0 for i in range(1000)]})
    sub = ms.downsample_trace(g, 100)
    assert set(g.index[g["is_upper_limit"]]) <= set(sub.index)
    assert g["value"].idxmax() in sub.index and g["time_mjd"].idxmax() in sub.index


# ── service with a fake SDK client ──────────────────────────────────────────
class _FakeSed:
    def __init__(self, csv_text, statuses=("done",), redshift=0.047, name="1ES 1959+650"):
        self.csv_text = csv_text
        self.statuses = list(statuses)
        self.calls = []
        self.redshift = redshift
        self.name = name

    def _job(self):
        st = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return SimpleNamespace(uuid="job-1", status=st, logs="log tail")

    def prepare(self, **kw):
        self.calls.append(("prepare", kw))
        return self._job()

    def get_status(self, uuid):
        self.calls.append(("status", uuid))
        return self._job()

    def download_csv(self, uuid, dest, **kw):
        """Emulates MMDC's server filter as probed live: rows whose
        [MJD_start, MJD_end] OVERLAPS the window, placeholder dates included."""
        self.calls.append(("csv", kw))
        text = self.csv_text
        if kw and text:
            raw = pd.read_csv(io.StringIO(text), dtype=str)
            s = pd.to_numeric(raw["MJD_start"])
            e = pd.to_numeric(raw["MJD_end"])
            keep = pd.Series(True, index=raw.index)
            if "mjd_start" in kw:
                keep &= e >= kw["mjd_start"]
            if "mjd_end" in kw:
                keep &= s <= kw["mjd_end"]
            lines = text.splitlines()
            text = "\n".join([lines[0]] + [lines[i + 1] for i in raw.index[keep]]) + "\n"
        Path(dest).write_text(text, encoding="utf-8")

    def get_info(self, uuid):
        return SimpleNamespace(model_dump=lambda: {"source_name": self.name, "ra": 299.99938, "dec": 65.14851,
                                                   "redshift": self.redshift, "gal_lat": 17.7, "gal_long": 98.0})


class _FakePlot:
    def _apply_style(self, dark=False):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt

    def _save_and_encode(self, fig, name):
        import matplotlib.pyplot as plt

        plt.close(fig)
        return {"web_url": f"/plots/{name}.png", "base64_png": "AAAA", "png_path": "x.png", "pdf_path": "x.pdf"}


def _svc(fake_sed, observations=None):
    client = SimpleNamespace(sed=fake_sed, observations=observations)
    return ms.MmdcService(plotting_service=_FakePlot(), client_factory=lambda: client, known_sources_loader=_loader)


def test_sed_end_to_end_with_fake_client():
    fake = _FakeSed(_csv("sed_1es1959_trim.csv"))
    out = _svc(fake).sed(target_name="1ES 1959+650", start_date="2008", end_date="2020")
    assert out["success"] and out["mmdc_job_uuid"] == "job-1"
    assert out["resolved_position"]["method"].startswith("MMDC gamma_ray_sources name match")
    assert out["redshift"] == 0.047 and out["redshift_source"].startswith("MMDC")
    w = out["window"]
    assert w["rule_applied"] == "contained" and w["window_end_date"] == "2020-12-31"
    assert w["kept_max_MJD_end"] <= 59214 and w["straddling_rows"] > 0
    assert out["attribution"]["acknowledgment"] == ms.ACKNOWLEDGMENT and out["attribution"]["bibcode"] == ms.MMDC_BIBCODE
    table, label = out["_table"]
    assert len(table) == out["summary"]["totals"]["rows"] and "2008-01-01" in label
    assert out["_figure"]["plotly_spec"]["data"] and out["_figure"]["path"].startswith("/plots/")
    assert any("contained" in r for r in out["answer_requirements"])


@pytest.mark.parametrize("start,end,mode,undated", [
    ("2008", "2020", "contained", False), ("2008", "2020", "overlap", True), ("1995", "1997", "contained", True),
    ("2010-02", "2010-02", "overlap", False), (None, "2012", "contained", True), ("2019", None, "overlap", False),
])
def test_windowed_server_fetch_gives_the_same_rows_as_the_full_table(start, end, mode, undated):
    """The windowed fetch (server overlap superset + undated probe) must keep
    exactly the rows the full-table computation keeps, with the same counts."""
    text = _csv("sed_1es1959_trim.csv")
    full = ms.normalize_sed_csv(text)
    s, _ = ms.parse_window_bound(start, is_end=False)
    e, _ = ms.parse_window_bound(end, is_end=True)
    want, want_rep = ms.apply_window(full, s, e, mode, undated)
    fake = _FakeSed(text)
    got = _svc(fake).sed(target_name="1ES 1959+650", start_date=start, end_date=end, window_mode=mode,
                         include_undated=undated)
    if want.empty:
        assert "_table" not in got and got["summary"]["totals"]["rows"] == 0
        table = want
    else:
        table, _ = got["_table"]
    key = ["catalog", "frequency_Hz", "nuFnu_erg_cm2_s", "MJD_start"]
    assert sorted(map(tuple, table[key].astype(str).values)) == sorted(map(tuple, want[key].astype(str).values))
    for k in ("rows_kept_by_contained_rule", "rows_kept_by_overlap_rule", "straddling_rows", "undated_archival_rows"):
        assert got["window"][k] == want_rep[k], k
    csv_calls = [c for c in fake.calls if c[0] == "csv"]
    assert csv_calls and all(c[1] for c in csv_calls)   # never the full table when a window is given
    assert got["window"]["dated_rows_available"] is None and "superset" in got["window"]["fetch_scope"]


def test_no_window_downloads_the_full_table_once():
    fake = _FakeSed(_csv("sed_1es1959_trim.csv"))
    svc = _svc(fake)
    svc.sed(target_name="1ES 1959+650")
    svc.sed(target_name="1ES 1959+650", color_by="catalog")   # served from the frame cache
    assert [c for c in fake.calls if c[0] == "csv"] == [("csv", {})]


def test_not_an_mmdc_source_falls_back_without_calling_mmdc():
    fake = _FakeSed(_csv("sed_1es1959_trim.csv"))
    out = _svc(fake).sed(ra=148.96846, dec=69.6797, target_name="M82")
    assert out["success"] is False and out["not_mmdc_source"] and out["fallback_tool"] == "ned_sed_plot"
    assert "Do not describe any MMDC data" in out["error"] and fake.calls == []
    lc = _svc(fake).lightcurve(ra=148.96846, dec=69.6797, target_name="M82")
    assert lc["not_mmdc_source"] and "ztf" in lc["fallback_tool"]


def test_require_known_source_false_proceeds_with_warning():
    fake = _FakeSed(_csv("sed_1es1959_trim.csv"), name="M82")
    out = _svc(fake).sed(ra=148.96846, dec=69.6797, target_name="M82", require_known_source=False)
    assert out["success"] and out["mmdc_source_match"] is None and "not in MMDC" in out["warnings"][0]


def test_pending_prepare_returns_pollable_job_and_resume_collects():
    fake = _FakeSed(_csv("sed_1es1959_trim.csv"), statuses=["processing"])
    svc = _svc(fake)
    pend = svc.prepare(299.99938, 65.14851, "1ES 1959+650", poll_seconds=0)
    assert pend["pending"] and pend["uuid"] == "job-1"
    fake.statuses = ["processing"]
    old = ms.PREPARE_POLL_S
    try:
        ms.PREPARE_POLL_S = 0.0
        out = svc.sed(target_name="1ES 1959+650")
    finally:
        ms.PREPARE_POLL_S = old
    assert out["success"] is False and out["pending"] and "job_uuid='job-1'" in out["error"]
    fake.statuses = ["done"]
    out2 = svc.sed(target_name="1ES 1959+650", job_uuid="job-1")
    assert out2["success"] and ("status", "job-1") in fake.calls


def test_no_data_and_error_statuses():
    out = _svc(_FakeSed("", statuses=["no_data"])).sed(target_name="1ES 1959+650")
    assert out["no_data"] and out["fallback_tool"] == "ned_sed_plot"
    out = _svc(_FakeSed("", statuses=["error"])).sed(target_name="1ES 1959+650")
    assert out["success"] is False and "log tail" in out["error"]


def test_empty_window_has_no_table_card():
    out = _svc(_FakeSed(_csv("sed_1es1959_trim.csv"))).sed(target_name="1ES 1959+650", start_date="1990",
                                                          end_date="1991")
    assert out["success"] and "_table" not in out and "No MMDC rows" in out["note"]
    assert out["window"]["undated_archival_rows"] > 0


def test_resolution_order_explicit_then_mmdc_list_then_resolver():
    svc = _svc(_FakeSed(""))
    assert svc.resolve_position("x", 10.0, 20.0, None)["method"] == "explicit ra/dec"
    assert "name match" in svc.resolve_position("Mrk 421", None, None, None)["method"]  # Mrk == Mkn
    called = []

    def resolver(name):
        called.append(name)
        return {"success": True, "ra_deg": 1.0, "dec_deg": 2.0, "resolver": "SIMBAD"}

    pos = svc.resolve_position("Some Galaxy", None, None, resolver)
    assert called == ["Some Galaxy"] and pos["method"].startswith("SIMBAD")
    with pytest.raises(ValueError):
        svc.resolve_position("x", 400.0, 0.0, None)


def test_lightcurve_end_to_end_window_and_bands():
    rows = [SimpleNamespace(model_dump=lambda r=r: r) for r in _json("lc_1es1959_trim.json")]
    seen = {}

    def cone_search(ra, dec, **kw):
        seen.update(kw)
        return rows

    svc = _svc(_FakeSed(""), observations=SimpleNamespace(cone_search=cone_search))
    out = svc.lightcurve(target_name="1ES 1959+650", start_date="2008", end_date="2020", bands=["MMDCGR", "W1", "ZTF"])
    assert seen["is_lightcurve"] is True and seen["mjd_min"] == 54466.0 and seen["mjd_max"] == 59215.0
    table, _ = out["_table"]
    # the fixture's UVOT W1 rows are from 2005: the window re-applied client-side drops them
    assert set(table["band"]) == {"MMDCGR", "ZTF:G", "ZTF:R"} and len(table) == out["rows"]
    assert table["time_mjd"].between(54466.0, 59215.0).all()
    assert out["attribution"]["citation"] == ms.MMDC_CITATION
    with pytest.raises(ValueError, match="unknown MMDC light-curve catalogs"):
        svc.lightcurve(target_name="1ES 1959+650", catalogs=["FERMI"])


def test_hard_transport_failure_opens_the_mmdc_breaker():
    from services.host_breaker import HostBreaker, HostCircuitOpen

    class ConnectError(Exception):  # httpx.ConnectError's class name = a hard infrastructure failure
        pass

    def boom():
        raise ConnectError("connection refused")

    svc = _svc(_FakeSed(""))
    with pytest.raises(ConnectError):
        svc.call("x", boom, 5)
    assert HostBreaker.is_open(f"{ms.MMDC_BASE_URL}/api/")
    with pytest.raises(HostCircuitOpen):
        svc.call("y", lambda: 1, 5)


def test_known_sources_disk_cache_and_stale_fallback(monkeypatch, tmp_path):
    """Live 2026-09-26: the 1.3 MB lists took >15 s after a backend restart."""
    monkeypatch.setenv("MMDC_CACHE_DIR", str(tmp_path))
    known, gamma = _loader()
    fetches = []

    def fake_call(self, label, fn, seconds):
        fetches.append(label)
        return known, gamma

    monkeypatch.setattr(ms.MmdcService, "call", fake_call)
    a = ms.MmdcService()
    assert a.known_sources() == (known, gamma) and fetches == ["known-source list"]
    b = ms.MmdcService()   # a fresh process: served from disk, no fetch
    assert b.known_sources() == (known, gamma) and len(fetches) == 1
    # expire the disk copy and make mmdc.am time out: the stale copy is used, not an error
    path = tmp_path / "known_sources.json"
    d = json.loads(path.read_text())
    d["fetched_at"] = 0
    path.write_text(json.dumps(d))

    def slow(self, label, fn, seconds):
        raise TimeoutError("MMDC known-source list did not answer within 30 s")

    monkeypatch.setattr(ms.MmdcService, "call", slow)
    c = ms.MmdcService()
    assert c.known_sources() == (known, gamma)
    path.unlink()
    with pytest.raises(TimeoutError):
        ms.MmdcService().known_sources()


def test_request_level_sdk_error_does_not_open_the_breaker():
    from services.host_breaker import HostBreaker

    class ValidationError(Exception):
        pass

    def bad():
        raise ValidationError("CSV validation failed")

    svc = _svc(_FakeSed(""))
    with pytest.raises(ValidationError):
        svc.call("x", bad, 5)
    assert not HostBreaker.is_open(f"{ms.MMDC_BASE_URL}/api/")
    assert svc.call("y", lambda: 7, 5) == 7


# ── lazy import ────────────────────────────────────────────────────────────
def test_missing_sdk_degrades_to_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "astro_mmdc", None)
    with pytest.raises(ms.MmdcUnavailable, match="astro-mmdc"):
        ms.import_sdk()
    with pytest.raises(ms.MmdcUnavailable):
        ms.MmdcService(known_sources_loader=_loader).client()


# ── agent wiring ───────────────────────────────────────────────────────────
@pytest.fixture()
def bare_agent(monkeypatch):
    from services import datalab_result_store as store_mod
    from tests.integration.test_agent_archive_tools import _load_agent_module

    module = _load_agent_module()
    agent = module.QuasarAgent.__new__(module.QuasarAgent)
    agent._tls = threading.local()
    agent._tls.current_user_id = "user-42"
    agent.last_run_result = None
    store = store_mod.DatalabResultStore(enable_disk_cache=False)
    monkeypatch.setattr(store_mod, "default_result_store", lambda: store)
    return agent, store


def test_agent_cards_store_table_with_owner_and_strip_heavy_keys(bare_agent):
    agent, store = bare_agent
    df = ms.normalize_sed_csv(_csv("sed_1es1959_trim.csv"))
    res = {"success": True, "_table": (df, "label"), "_figure": {"path": "/plots/a.png", "plotly_spec": {"data": [1]},
                                                                 "caption": "cap", "image_base64": "AAAA"}}
    out = agent._mmdc_cards(res, tool_name="mmdc_sed", store_meta={"mmdc": {"kind": "sed"}})
    assert "_table" not in out and "_figure" not in out and out["figure_attached"]
    frame, meta, status = store.lookup(out["result_id"])
    assert status == "ok" and len(frame) == len(df) and meta["owner_id"] == "user-42" and meta["mmdc"]["kind"] == "sed"
    cards = agent._accumulated_run_results
    assert [c["type"] for c in cards] == ["data", "image"]
    assert cards[0]["result_id"] == out["result_id"] and cards[1]["plotly_spec"] == {"data": [1]}
    assert agent.last_run_result is cards[-1]
    assert "image_base64" not in json.dumps({k: v for k, v in out.items()}, default=str)


def test_agent_call_maps_failures(bare_agent):
    from services.host_breaker import HostCircuitOpen
    from services.tool_budgets import BudgetExhausted

    agent, _ = bare_agent

    def raise_(exc):
        def f():
            raise exc
        return f

    out = agent._mmdc_call("mmdc_sed", raise_(ms.MmdcUnavailable("astro-mmdc missing")))
    assert out["dependency_missing"] == "astro-mmdc"
    out = agent._mmdc_call("mmdc_sed", raise_(HostCircuitOpen("mmdc.am", 30.0, "ConnectError")))
    assert out["circuit_breaker"] and out["infrastructure_failure"]
    out = agent._mmdc_call("mmdc_sed", raise_(TimeoutError("hung")))
    assert out["infrastructure_failure"] and out["host"] == "mmdc.am"
    out = agent._mmdc_call("mmdc_sed", raise_(BudgetExhausted("MMDC CSV", 1.0, 2.0)))
    assert out["budget_exhausted"]
    out = agent._mmdc_call("mmdc_sed", raise_(ValueError("bad window")))
    assert out["success"] is False and "bad window" in out["error"]


def test_data_card_shows_tiny_and_huge_floats_with_significant_figures():
    """UI 2026-09-26: the table card displayed every nuFnu (~1e-12) as '0'."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ui-pro"))
    from api.serializers.data_card import _build_data_card_event

    df = pd.DataFrame({"frequency_Hz": [1.686e18], "nuFnu_erg_cm2_s": [5.616e-12], "energy_eV": [697.274],
                       "MJD_start": [50572.9003], "zero": [0.0]})
    ev = _build_data_card_event({"type": "data", "data": df, "source": "MMDC", "tool_name": "mmdc_sed", "filter_label": "x"})
    payload = json.loads(ev[0][ev[0].index("{"):ev[0].rindex("}") + 1])
    row = payload["rows"][0]
    # tiny values get significant figures; large ones keep every digit (review CX-B12: float-stored IDs)
    assert row["nuFnu_erg_cm2_s"] == "5.616e-12" and row["frequency_Hz"] == "1686000000000000000"
    assert row["energy_eV"] == "697.274" and row["MJD_start"] == "50572.9" and row["zero"] == "0"


def test_tools_are_registered_budgeted_and_hosted():
    from services import tool_budgets as tb
    from services.host_breaker import host_of

    for name in ("mmdc_sed", "mmdc_lightcurve"):
        assert name in tb.INNER_BUDGETS and "mmdc.am" in {host_of(h) for h in tb.hosts_for(name)}
    assert not tb.check_hierarchy(["mmdc_sed", "mmdc_lightcurve"])
    assert "mmdc_sed" in tb.tools_for_host("mmdc.am")


# ── routing ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("query,tool,args", [
    ("Retrieve multi-epoch spectral energy distributions of 1ES 1959+650 between 2008 and 2020", "mmdc_sed",
     {"target_name": "1ES 1959+650", "start_date": "2008", "end_date": "2020"}),
    ("Show the SED of Mkn 421 during February 2010 and mark upper limits.", "mmdc_sed",
     {"target_name": "Mkn 421", "start_date": "2010-02", "end_date": "2010-02"}),
    ("SED of 1ES 1959+650 for 1995 to 1997.", "mmdc_sed", {"start_date": "1995", "end_date": "1997"}),
    ("Fit an SSC model to the 3C 279 SED from 2015", "mmdc_sed", {"target_name": "3C 279"}),
    ("Compute the fractional variability of Mkn 501 in each band from 2018 to 2022 and find flares.",
     "variability_analysis", {"target_name": "Mkn 501"}),
    ("Is there a lag between the X-ray and gamma-ray flares of Mkn 421?", "variability_analysis", {}),
])
def test_blazar_questions_route_to_mmdc(query, tool, args):
    from core.oneshot_routing import detect_oneshot_intent

    got = detect_oneshot_intent(query)
    assert got and got["tool"] == tool
    for k, v in args.items():
        assert got["args"][k] == v
    assert "MANDATORY INSTRUCTION" in got["directive"]


@pytest.mark.parametrize("query", [
    "Plot the SED of M82.",
    "Show the TESS light curve of Kepler-10",
    "How many Gaia DR3 sources lie within 10 arcminutes of Palomar 5?",
])
def test_non_blazar_questions_are_not_routed_to_mmdc(query):
    from core.oneshot_routing import detect_oneshot_intent

    got = detect_oneshot_intent(query)
    assert not got or not str(got["tool"]).startswith(("mmdc_", "variability"))


def test_hadronic_fit_directive_names_neutrino_inputs():
    from core.oneshot_routing import detect_oneshot_intent

    got = detect_oneshot_intent("Fit a hadronic model to TXS 0506+056 for 2017 including the one IceCube neutrino")
    assert got["tool"] == "mmdc_sed" and "HADRONIC" in got["directive"] and "n_icecube" in got["directive"]


def test_multi_model_fit_directive_asks_for_parallel_fits():
    from core.oneshot_routing import detect_oneshot_intent

    got = detect_oneshot_intent("Fit both an SSC and an EIC model to the 3C 279 SED from 2015 and compare the two fits.")
    assert got["tool"] == "mmdc_sed" and "(SSC and EIC)" in got["directive"] and "IN THE SAME ROUND" in got["directive"]


def test_v2_playbook_and_routing_row_present():
    from core.prompts import playbooks as pb
    from core.prompts import system_core as sc

    p = pb.get_playbook("blazar_mmdc")
    assert p is not None and p.token_cost <= pb.INITIAL_BUDGET
    assert "\x08" not in p.text and all("\x08" not in r.pattern for r in p.query_patterns)
    assert any(r.search("multi-epoch SED of a blazar") for r in p.query_patterns)
    assert "mmdc_sed" in sc.core_body() and sc.count_tokens(sc.core_body()) < sc.CORE_TOKEN_BUDGET


# ── guard Round A (task-dd87861-19423) regressions ─────────────────────────
def test_cx01_model_cannot_bypass_the_non_member_fallback(bare_agent):
    agent, _ = bare_agent
    fake = _FakeSed(_csv("sed_1es1959_trim.csv"), name="M82")
    agent._mmdc_service_instance = _svc(fake)
    out = agent._mmdc_sed(ra=148.96846, dec=69.6797, target_name="M82", require_known_source=False)
    assert out["success"] is False and out["not_mmdc_source"] and fake.calls == []
    src = (Path(__file__).resolve().parents[2] / "core" / "tool_registrations.py").read_text(encoding="utf-8")
    block = src[src.index('name="mmdc_sed"'):src.index('name="mmdc_model"')]
    assert "require_known_source" not in block   # not offered to the model at all


def test_cx02_expanded_end_excludes_the_next_periods_first_instant():
    df = ms.normalize_sed_csv(pd.DataFrame([_row(59214.5, 59214.9, cat="IN"), _row(59215.0, 59215.0, cat="EDGE"),
                                            _row(59214.0, 59215.0, cat="ENDS_AT_EDGE")]))
    e, note = ms.parse_window_bound("2020", is_end=True)
    kept, rep = ms.apply_window(df, 54466.0, e, "contained", end_exclusive=True)
    assert set(kept["catalog"]) == {"IN"} and rep["window_end_exclusive"] and "excluded" in rep["rule_text"]
    kept_o, _ = ms.apply_window(df, 54466.0, e, "overlap", end_exclusive=True)
    assert set(kept_o["catalog"]) == {"IN", "ENDS_AT_EDGE"}   # EDGE starts exactly at 2021-01-01 00:00
    kept_mjd, _ = ms.apply_window(df, 54466.0, 59215.0, "contained")   # an explicit MJD end stays inclusive
    assert set(kept_mjd["catalog"]) == {"IN", "EDGE", "ENDS_AT_EDGE"}


def test_cx02_sed_and_lightcurve_apply_the_exclusive_end():
    rows = pd.read_csv(FIX / "sed_1es1959_trim.csv", dtype=str)
    extra = rows.iloc[[0]].copy()
    extra["MJD_start"], extra["MJD_end"], extra["catalog"], extra["flag"] = "59215.0", "59215.0", "EDGECAT", ""
    text = pd.concat([rows, extra]).to_csv(index=False)
    out = _svc(_FakeSed(text)).sed(target_name="1ES 1959+650", start_date="2008", end_date="2020", window_mode="overlap")
    assert "EDGECAT" not in set(out["_table"][0]["catalog"])
    lc_rows = [SimpleNamespace(model_dump=lambda r=r: r) for r in _json("lc_1es1959_trim.json")]
    edge = dict(_json("lc_1es1959_trim.json")[0], mjd_mid=59215.0)
    lc_rows.append(SimpleNamespace(model_dump=lambda: edge))
    svc = _svc(_FakeSed(""), observations=SimpleNamespace(cone_search=lambda ra, dec, **kw: lc_rows))
    lc = svc.lightcurve(target_name="1ES 1959+650", start_date="2008", end_date="2020")
    assert (lc["_table"][0]["time_mjd"] < 59215.0).all()


def test_cx03_ownerless_results_are_not_loadable_by_a_signed_in_user(monkeypatch):
    from services import datalab_result_store as store_mod

    store = store_mod.DatalabResultStore(enable_disk_cache=False)
    monkeypatch.setattr(store_mod, "default_result_store", lambda: store)
    df = ms.normalize_sed_csv(_csv("sed_1es1959_trim.csv"))
    rid_none = store.put(df, {"source": "MMDC"})
    rid_owned = store.put(df, {"source": "MMDC", "owner_id": "alice"})
    with pytest.raises(KeyError):
        ms.load_result(rid_none, "bob")
    assert len(ms.load_result(rid_none, None)[0]) == len(df)   # scripts / tests without a user
    assert len(ms.load_result(rid_owned, "alice")[0]) == len(df)
    for caller in ("bob", None):
        with pytest.raises(KeyError):
            ms.load_result(rid_owned, caller)


@pytest.mark.parametrize("query", ["Show the SED of M82 at epoch J2000", "Fit an SSC model to the SED of M82",
                                   "Plot the SED of NGC 1068 at MJD 55000"])
def test_cx04_non_blazar_questions_are_not_forced_to_mmdc(query):
    from core.oneshot_routing import detect_oneshot_intent

    got = detect_oneshot_intent(query)
    assert not got or not str(got["tool"]).startswith(("mmdc_", "variability"))


def test_cx05_upper_limits_are_arrows_in_every_figure():
    rows = _json("lc_1es1959_trim.json")
    rows.append({**rows[0], "is_upper_limit": True, "mjd_mid": 60000.0})
    spec, _ = ms.lightcurve_plotly_spec(ms.normalize_lightcurve_rows(rows), "lc")
    ul = [t for t in spec["data"] if t.get("marker", {}).get("symbol") == "triangle-down"]
    assert ul and all(t["error_y"]["arrayminus"][0] > 0 and not t["error_y"]["symmetric"] for t in ul)

    class _Ax:
        def __init__(self):
            self.calls = []

        def vlines(self, *a, **k):
            self.calls.append("vlines")

        def scatter(self, *a, **k):
            self.calls.append(k.get("marker"))

    ax = _Ax()
    ms.ul_arrows(ax, [1e17], [1e-11])
    assert ax.calls == ["vlines", "_", "v"]
    # the SED PNG path draws them (no bare triangles)
    df = ms.normalize_sed_csv(_csv("sed_1es1959_trim.csv"))
    drawn = []
    orig = ms.ul_arrows
    try:
        ms.ul_arrows = lambda *a, **k: drawn.append(len(a[1]))
        ms.sed_png(df, "t", _FakePlot())
        ms.sed_png(df, "t", _FakePlot(), color_by="catalog")
    finally:
        ms.ul_arrows = orig
    assert sum(drawn) >= int(df["upper_limit"].sum())


def test_cx06_failed_source_info_is_not_reported_as_missing_redshift():
    fake = _FakeSed(_csv("sed_1es1959_trim.csv"))

    def boom(uuid):
        raise TimeoutError("source info hung")

    fake.get_info = boom
    out = _svc(fake).sed(target_name="1ES 1959+650")
    assert out["redshift"] is None and "FAILED" in out["redshift_source"]
    assert any("redshift is unknown" in w for w in out["warnings"])


def test_cx07_plot_cap_cannot_be_raised_above_5000():
    rows = [_row(55000.5 + i * 0.01, 55000.5 + i * 0.01, cat="X", freq=1e17 * (1 + i % 50)) for i in range(6000)]
    df = ms.normalize_sed_csv(pd.DataFrame(rows))
    sub, info = ms.downsample_sed(df, 6000)
    assert info["downsampled"] and len(sub) <= ms.PLOTLY_POINT_CAP + 20
