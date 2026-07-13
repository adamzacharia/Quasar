"""Unit tests for services/sparcl_stacking.py (spectral stacking workflow)."""

import os

import numpy as np

from services import plotting
from services.sparcl_stacking import SparclStackingService


def _plot_dir(monkeypatch):
    path = os.path.join("test_results", "live_imagery", "sparcl_stack")
    os.makedirs(path, exist_ok=True)
    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", path)


def _make_spectrum(sid, z, *, n=200, level=1.0, wl_lo=3600.0, wl_hi=9800.0, ivar=True):
    wave = np.linspace(wl_lo, wl_hi, n)
    flux = np.full(n, level)
    out = {
        "sparcl_id": sid,
        "wavelength": wave,
        "flux": flux,
        "ivar": np.full(n, 4.0) if ivar else None,
        "model": None,
        "mask": None,
        "redshift": z,
        "spectype": "GALAXY",
        "data_release": "DESI-DR1",
        "n_points": n,
        "wavelength_min": wl_lo,
        "wavelength_max": wl_hi,
        "median_snr": 2.0,
    }
    return out


class _FakeSpectraService:
    """Fake SparclSpectraService: search returns ids per redshift bin;
    retrieve returns constant-flux spectra at the bin's redshift."""

    def __init__(self):
        self.search_calls = []
        self.retrieve_calls = []
        # sid -> retrieved spectrum payload; sid -> catalog redshift (search side)
        self.spectra = {}
        self.catalog_z = {}

    def add(self, sid, z, **kw):
        self.spectra[sid] = _make_spectrum(sid, z, **kw)
        self.catalog_z[sid] = z

    def search_spectra(self, spectype=None, redshift_min=None, redshift_max=None,
                       data_release=None, limit=100, **kw):
        self.search_calls.append((redshift_min, redshift_max, limit))
        rows = [
            {"sparcl_id": sid, "ra": 1.0, "dec": 2.0,
             "redshift": z, "spectype": "GALAXY", "data_release": "DESI-DR1"}
            for sid, z in self.catalog_z.items()
            if redshift_min <= z < redshift_max
        ][: int(limit)]
        return {"success": True, "rows": rows, "count": len(rows), "warnings": [],
                "provenance": {}}

    def retrieve_spectra(self, sparcl_ids, include=None):
        self.retrieve_calls.append(list(sparcl_ids))
        spectra = [self.spectra[sid] for sid in sparcl_ids if sid in self.spectra]
        if not spectra:
            return {"success": False, "error": "no spectra"}
        return {"success": True, "spectra": spectra, "count": len(spectra),
                "warnings": [], "provenance": {}}


def test_stack_constraint_mode_two_bins(monkeypatch):
    _plot_dir(monkeypatch)
    svc = _FakeSpectraService()
    # Bin 1 (0.1 <= z < 0.2): two spectra with different brightness — the
    # median normalization must equalize them so the stack is ~1.0.
    svc.add("a1", 0.12, level=2.0)
    svc.add("a2", 0.15, level=10.0)
    # Bin 2 (0.2 <= z < 0.3)
    svc.add("b1", 0.25, level=5.0)

    out = SparclStackingService(spectra_service=svc).stack(
        spectype="GALAXY", redshift_min=0.1, redshift_max=0.3, n_bins=2, n_per_bin=10,
    )

    assert out["success"] is True, out.get("error")
    assert len(out["bins"]) == 2
    assert out["bins"][0]["n_spectra"] == 2
    assert out["bins"][1]["n_spectra"] == 1
    assert out["grid"]["frame"] == "rest"
    # rest-frame coverage shrinks by (1+z)
    assert out["grid"]["lo"] < 3600.0
    assert out["path"].endswith(".png")
    # normalized constant spectra stack to ~1
    flux_values = [r["flux_stacked"] for r in out["stack_rows"] if r["bin_index"] == 0]
    assert flux_values and abs(np.median(flux_values) - 1.0) < 1e-6
    # long rows carry bin labels
    assert any("z" in r["bin_label"] for r in out["stack_rows"])


def test_stack_rows_mode_with_bin_column(monkeypatch):
    _plot_dir(monkeypatch)
    svc = _FakeSpectraService()
    for k, (sid, color) in enumerate([("s1", 0.3), ("s2", 0.5), ("s3", 1.1), ("s4", 1.3)]):
        svc.add(sid, 0.2)
    rows = [
        {"sparcl_id": "s1", "g_r": 0.3},
        {"sparcl_id": "s2", "g_r": 0.5},
        {"sparcl_id": "s3", "g_r": 1.1},
        {"sparcl_id": "s4", "g_r": 1.3},
        {"sparcl_id": "", "g_r": 0.4},          # dropped: no id
        {"sparcl_id": "s5", "g_r": float("nan")},  # dropped: NaN bin value
    ]
    out = SparclStackingService(spectra_service=svc).stack(
        rows=rows, bin_column="g_r", bin_edges=[0.0, 0.8, 1.6], n_per_bin=10,
    )

    assert out["success"] is True, out.get("error")
    assert len(out["bins"]) == 2
    assert out["bins"][0]["n_spectra"] == 2 and out["bins"][1]["n_spectra"] == 2
    assert "g_r" in out["bins"][0]["label"]
    # search was never called in chained mode
    assert svc.search_calls == []


def test_stack_errors_are_typed(monkeypatch):
    _plot_dir(monkeypatch)
    svc = _FakeSpectraService()

    out = SparclStackingService(spectra_service=svc).stack(rows=[{"x": 1}], bin_column="x")
    assert out["success"] is False and "sparcl_id" in out["error"]

    out2 = SparclStackingService(spectra_service=svc).stack(spectype="GALAXY")
    assert out2["success"] is False and "redshift_min" in out2["error"]

    out3 = SparclStackingService(spectra_service=svc).stack(
        spectype="GALAXY", redshift_min=0.1, redshift_max=0.3, weighting="bogus"
    )
    assert out3["success"] is False and "weighting" in out3["error"]


def test_stack_drops_spectra_without_redshift_in_rest_frame(monkeypatch):
    _plot_dir(monkeypatch)
    svc = _FakeSpectraService()
    svc.add("good", 0.15)
    svc.add("noz", 0.18)
    svc.spectra["noz"]["redshift"] = None

    out = SparclStackingService(spectra_service=svc).stack(
        spectype="GALAXY", redshift_min=0.1, redshift_max=0.2, n_bins=1, n_per_bin=10,
    )
    assert out["success"] is True, out.get("error")
    assert out["bins"][0]["n_spectra"] == 1
    assert any("without a finite redshift" in w for w in out["warnings"])


def test_stack_total_budget_is_capped(monkeypatch):
    _plot_dir(monkeypatch)
    svc = _FakeSpectraService()
    svc.add("x", 0.15)
    out = SparclStackingService(spectra_service=svc).stack(
        spectype="GALAXY", redshift_min=0.1, redshift_max=0.2,
        n_bins=12, n_per_bin=200,  # 2400 > 600 budget
    )
    # search was asked for at most floor(600/12)=50 per bin
    assert all(call[2] <= 50 for call in svc.search_calls)
    if out["success"]:
        assert any("capped" in w for w in out["warnings"])
