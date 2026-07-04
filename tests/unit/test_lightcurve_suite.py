import math
import os

import numpy as np
import pytest

from services import plotting
from services.lightcurve_suite import FEW_POINTS_WARNING, LightCurveSuite


def _plot_dir(name):
    path = os.path.join("test_results", "lightcurve_suite", name)
    os.makedirs(path, exist_ok=True)
    return path


class FakeArray:
    def __init__(self, values):
        self.value = np.asarray(values, dtype=float)


class FakeLightCurve:
    def __init__(self, time, flux, meta=None):
        self.time = FakeArray(time)
        self.flux = FakeArray(flux)
        self.meta = meta or {}

    def remove_nans(self):
        return self

    def normalize(self):
        values = np.asarray(self.flux.value, dtype=float)
        finite = values[np.isfinite(values)]
        median = np.nanmedian(finite) if len(finite) else np.nan
        if np.isfinite(median) and median != 0:
            values = values / median
        return FakeLightCurve(self.time.value, values, meta=self.meta)


class FakeSearchEntry:
    def __init__(self, lc):
        self.lc = lc

    def download(self):
        return self.lc


class FakeSearchResult:
    def __init__(self, rows, lightcurves=None):
        self.table = list(rows)
        self._lightcurves = list(lightcurves or [])

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        return FakeSearchEntry(self._lightcurves[index])


class FakeAlerce:
    def __init__(self, detections):
        self.detections = list(detections)
        self.calls = []

    def light_curve(self, oid):
        self.calls.append(oid)
        return {
            "success": True,
            "oid": oid,
            "detections": list(self.detections),
            "non_detections": [],
            "n_detections": len(self.detections),
            "n_non_detections": 0,
            "provenance": {"service": "fake alerce"},
        }


def _ztf_detections(period, n, fid, *, span=80.0, mag0=17.0, amp=0.35, seed=3):
    rng = np.random.default_rng(seed + fid)
    t = np.sort(rng.uniform(0, span, n))
    mag = mag0 + amp * np.sin(2 * np.pi * t / period)
    return [{"mjd": float(x), "magpsf": float(y), "sigmapsf": 0.03, "fid": fid} for x, y in zip(t, mag)]


def test_search_space_lightcurves_normalizes_rows_and_empty_warns():
    rows = [
        {"mission": "TESS Sector 01", "year": np.int64(2018), "author": "SPOC", "exptime": np.float64(120.0), "target_name": "Pi Mensae", "distance": np.float64(0.12)},
        {"mission": "TESS Sector 02", "year": 2018, "author": "QLP", "t_exptime": 1800.0, "target_name": "Pi Mensae", "distance": np.ma.masked},
        {"mission": "K2 Campaign 01", "year": 2014, "author": "K2SFF", "exptime": 60.0, "target_name": "EPIC", "distance": 1.5},
    ]
    calls = []

    def fake_search(target, mission=None):
        calls.append((target, mission))
        return FakeSearchResult(rows)

    out = LightCurveSuite(search_fn=fake_search).search_space_lightcurves("Pi Mensae", mission="tess", max_rows=2)

    assert out["success"] is True
    assert calls == [("Pi Mensae", "TESS")]
    assert out["count"] == 2
    assert out["total_available"] == 3
    assert out["rows"][0] == {
        "index": 0,
        "mission": "TESS Sector 01",
        "year": 2018,
        "author": "SPOC",
        "exptime_s": 120.0,
        "target_name": "Pi Mensae",
        "distance_arcsec": 0.12,
    }
    assert out["rows"][1]["exptime_s"] == 1800.0
    assert out["rows"][1]["distance_arcsec"] is None
    assert any("Returned first 2" in warning for warning in out["warnings"])

    empty = LightCurveSuite(search_fn=lambda target, mission=None: FakeSearchResult([])).search_space_lightcurves("Nope")
    assert empty["success"] is True
    assert empty["count"] == 0
    assert any("No TESS/Kepler/K2" in warning for warning in empty["warnings"])


def test_plot_space_lightcurve_downloads_normalizes_and_writes_png(monkeypatch):
    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", _plot_dir("lightcurve"))
    t = np.linspace(0, 27, 150)
    flux = 1000.0 + 20.0 * np.sin(2 * np.pi * t / 3.0)
    rows = [{"mission": "TESS Sector 01", "author": "SPOC", "exptime": 120.0, "target_name": "Synthetic", "distance": 0.0, "year": 2018}]
    result = FakeSearchResult(rows, [FakeLightCurve(t, flux, {"MISSION": "TESS"})])
    out = LightCurveSuite(search_fn=lambda target, mission=None: result).plot_space_lightcurve("Synthetic", mission="TESS", index=0)

    assert out["success"] is True
    assert out["path"].endswith(".png")
    assert os.path.exists(out["png_path"])
    assert out["n_points"] == 150
    assert out["time_span_days"] == pytest.approx(27.0)
    assert out["provenance"]["index"] == 0


def test_period_search_ztf_auto_selects_most_populated_fid(monkeypatch):
    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", _plot_dir("period_ztf"))
    detections = _ztf_detections(1.7, 60, 1, seed=10) + _ztf_detections(2.9, 24, 2, seed=20)
    out = LightCurveSuite(alerce_client=FakeAlerce(detections)).period_search("ztf", "ZTF1", min_period_d=0.5, max_period_d=4.0)

    assert out["success"] is True
    assert out["value_kind"] == "mag"
    assert out["n_points"] == 60
    assert out["path"].endswith(".png")
    assert os.path.exists(out["png_path"])
    assert any("fid 1" in warning for warning in out["warnings"])
    assert out["best_period_d"] == pytest.approx(1.7, rel=0.01)


def test_period_search_tess_stub_recovers_period_and_honors_index(monkeypatch):
    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", _plot_dir("period_tess"))
    svc = LightCurveSuite(search_fn=lambda target, mission=None: FakeSearchResult([]))
    t = np.linspace(0, 90, 300)
    flux = 1.0 + 0.04 * np.sin(2 * np.pi * t / 3.21)

    def fake_fetch(target, mission=None, index=0):
        assert target == "Synthetic TESS"
        assert mission == "TESS"
        assert index == 2
        return t, flux, "flux", {"mission": "TESS Sector 09", "author": "SPOC", "index": index}

    monkeypatch.setattr(svc, "fetch_lightcurve_arrays", fake_fetch)
    out = svc.period_search("tess", "Synthetic TESS", min_period_d=1.0, max_period_d=8.0, index=2)

    assert out["success"] is True
    assert out["path"].endswith(".png")
    assert out["best_period_d"] == pytest.approx(3.21, rel=0.01)
    assert out["provenance"]["index"] == 2


def test_few_points_warns_and_all_nan_fails(monkeypatch):
    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", _plot_dir("period_edges"))
    svc = LightCurveSuite(search_fn=lambda target, mission=None: FakeSearchResult([]))
    t = np.linspace(0, 20, 15)
    y = 1.0 + 0.05 * np.sin(2 * np.pi * t / 2.0)
    monkeypatch.setattr(svc, "fetch_lightcurve_arrays", lambda target, mission=None, index=0: (t, y, "flux", {"mission": "TESS"}))
    out = svc.period_search("tess", "few", min_period_d=0.8, max_period_d=4.0)
    assert out["success"] is True
    assert FEW_POINTS_WARNING in out["warnings"]

    monkeypatch.setattr(svc, "fetch_lightcurve_arrays", lambda target, mission=None, index=0: (np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), "flux", {}))
    failed = svc.period_search("tess", "nan", min_period_d=0.8, max_period_d=4.0)
    assert failed["success"] is False
    assert "five finite" in failed["error"]


def test_period_search_ztf_fid_filter_honored(monkeypatch):
    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", _plot_dir("period_fid"))
    detections = _ztf_detections(1.3, 55, 1, seed=30) + _ztf_detections(2.2, 42, 2, seed=40)
    out = LightCurveSuite(alerce_client=FakeAlerce(detections)).period_search("ztf", "ZTF2", min_period_d=1.0, max_period_d=3.0, fid="2")

    assert out["success"] is True
    assert out["n_points"] == 42
    assert out["provenance"]["fid"] == 2
    assert out["best_period_d"] == pytest.approx(2.2, rel=0.01)


def test_period_search_validation_and_constant_signal(monkeypatch):
    svc = LightCurveSuite(search_fn=lambda target, mission=None: FakeSearchResult([]))
    out = svc.period_search("bad", "x")
    assert out["success"] is False
    assert "source must" in out["error"]

    t = np.linspace(0, 10, 30)
    monkeypatch.setattr(svc, "fetch_lightcurve_arrays", lambda target, mission=None, index=0: (t, np.ones_like(t), "flux", {}))
    const = svc.period_search("tess", "constant", min_period_d=2.0, max_period_d=1.0)
    assert const["success"] is False
    assert "min_period_d" in const["error"]

    const = svc.period_search("tess", "constant", min_period_d=0.5, max_period_d=4.0)
    assert const["success"] is False
    assert "constant" in const["error"]