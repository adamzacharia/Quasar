"""Unit tests for capabilities/spectra.py (SPARCL survey-scale spectroscopy family)."""

import numpy as np
import pandas as pd

from capabilities.base import CallContext
from capabilities.spectra import CAPABILITIES, SparclGetSpectrum, SparclSearchSpectra


class _FakeSparclService:
    """Recording fake for services.sparcl_spectra.SparclSpectraService."""

    def __init__(self, search_result=None, retrieve_result=None):
        self.search_result = search_result or {"success": True, "rows": []}
        self.retrieve_result = retrieve_result or {"success": True, "spectra": []}
        self.calls = []

    def search_spectra(self, **kwargs):
        self.calls.append(("search_spectra", kwargs))
        return dict(self.search_result)

    def retrieve_spectra(self, sparcl_ids, include=None):
        self.calls.append(("retrieve_spectra", list(sparcl_ids), include))
        return dict(self.retrieve_result)


class _TableRecorder:
    def __init__(self):
        self.kwargs = None

    def __call__(self, rows, **kwargs):
        self.kwargs = dict(kwargs, rows=rows)
        return {
            "success": True,
            "total_results": len(rows),
            "tool_name": kwargs["tool_name"],
            "note": "Full table is rendered as a data card in the UI; do not repeat the rows in text.",
        }


class _FakeStore:
    def __init__(self):
        self.frames = []

    def put(self, dataframe, meta=None):
        self.frames.append((dataframe, dict(meta or {})))
        return f"dlr_test{len(self.frames)}"

    def get(self, result_id):  # pragma: no cover - protocol completeness
        return None


def _ctx(service, *, table=None, store=None, resolve=None):
    table = table or _TableRecorder()
    services = {
        "get_sparcl_spectra_service": lambda: service,
        "external_catalog_table_result": table,
        "live_imagery_coordinates": resolve or (lambda **kw: (150.1, 2.2, "COSMOS")),
    }
    return CallContext(services=services, result_store=store), table


def _run(cap, ctx, **kwargs):
    return cap.run(cap.InputModel(**kwargs), ctx).to_native()


def test_registry_exposes_all_capabilities():
    names = {cap.name for cap in CAPABILITIES}
    assert names == {"sparcl_search_spectra", "sparcl_get_spectrum", "sparcl_stack_spectra"}


def test_search_constraint_mode_stores_result_and_builds_label():
    rows = [
        {"sparcl_id": "a", "ra": 10.0, "dec": 1.0, "redshift": 0.15,
         "spectype": "GALAXY", "data_release": "DESI-DR1"},
    ]
    svc = _FakeSparclService(search_result={
        "success": True, "rows": rows, "count": 1, "warnings": [],
        "provenance": {"service": "NOIRLab SparCL",
                       "constraints": {"spectype": ["GALAXY"], "redshift": [0.1, 0.3],
                                       "data_release": ["DESI-DR1", "SDSS-DR17"]}},
    })
    store = _FakeStore()
    ctx, table = _ctx(svc, store=store)
    out = _run(SparclSearchSpectra(), ctx,
               spectype="GALAXY", redshift_min=0.1, redshift_max=0.3)

    assert out["success"] is True
    assert out["result_id"] == "dlr_test1"
    assert "chaining" in out["note"]
    # no cone was given: the service must not receive ra/dec and the table
    # must not carry a distance column
    _, kwargs = svc.calls[0]
    assert "ra" not in kwargs and "dec" not in kwargs
    assert "distance_arcsec" not in table.kwargs["columns"]
    label = table.kwargs["filter_label"]
    assert "spectype=GALAXY" in label and "0.1 <= z <= 0.3" in label
    stored_df, stored_meta = store.frames[0]
    assert isinstance(stored_df, pd.DataFrame) and len(stored_df) == 1
    assert stored_meta["tool_name"] == "sparcl_search_spectra"


def test_search_with_target_resolves_cone_and_adds_distance_column():
    svc = _FakeSparclService(search_result={
        "success": True, "rows": [], "count": 0, "warnings": [],
        "provenance": {"constraints": {"data_release": ["DESI-DR1"]},
                       "radius_arcsec": 60.0},
    })
    ctx, table = _ctx(svc)
    out = _run(SparclSearchSpectra(), ctx, target_name="COSMOS", spectype="QSO")

    assert out["success"] is True
    _, kwargs = svc.calls[0]
    assert kwargs["ra"] == 150.1 and kwargs["dec"] == 2.2
    assert "distance_arcsec" in table.kwargs["columns"]
    assert "cone COSMOS" in table.kwargs["filter_label"]


def test_search_service_failure_passes_through():
    svc = _FakeSparclService(search_result={"success": False, "error": "boom"})
    ctx, _ = _ctx(svc)
    out = _run(SparclSearchSpectra(), ctx, spectype="GALAXY")
    assert out == {"success": False, "error": "boom"}


def test_search_survives_missing_result_store():
    svc = _FakeSparclService(search_result={
        "success": True,
        "rows": [{"sparcl_id": "a", "ra": 1.0, "dec": 2.0, "redshift": 0.1,
                  "spectype": "GALAXY", "data_release": "DESI-DR1"}],
        "count": 1, "warnings": [], "provenance": {"constraints": {}},
    })
    ctx, _ = _ctx(svc, store=None)
    out = _run(SparclSearchSpectra(), ctx, spectype="GALAXY")
    assert out["success"] is True
    assert "result_id" not in out


def test_get_spectrum_stores_long_format_and_returns_scalars_only():
    spectra = [
        {"sparcl_id": "s1",
         "wavelength": np.array([1.0, 2.0, 3.0]),
         "flux": np.array([5.0, 6.0, 7.0]),
         "ivar": np.array([1.0, 1.0, 0.0]),
         "model": None, "mask": None,
         "redshift": 0.2, "spectype": "GALAXY", "data_release": "DESI-DR1",
         "n_points": 3, "wavelength_min": 1.0, "wavelength_max": 3.0,
         "median_snr": 5.5},
    ]
    svc = _FakeSparclService(retrieve_result={
        "success": True, "spectra": spectra, "count": 1,
        "warnings": [], "provenance": {"include": ["sparcl_id"]},
    })
    store = _FakeStore()
    ctx, _ = _ctx(svc, store=store)
    out = _run(SparclGetSpectrum(), ctx, sparcl_ids=["s1"])

    assert out["success"] is True
    assert out["count"] == 1
    assert out["result_id"] == "dlr_test1"
    # scalars only in the LLM-facing payload — no arrays
    summary = out["spectra"][0]
    assert summary["median_snr"] == 5.5
    assert "wavelength" not in summary and "flux" not in summary
    stored_df, _ = store.frames[0]
    assert list(stored_df.columns) == ["sparcl_id", "wavelength", "flux", "ivar"]
    assert len(stored_df) == 3


def test_get_spectrum_warns_when_store_unavailable():
    spectra = [
        {"sparcl_id": "s1", "wavelength": np.array([1.0]), "flux": np.array([1.0]),
         "ivar": None, "model": None, "mask": None,
         "redshift": None, "spectype": "STAR", "data_release": "SDSS-DR17",
         "n_points": 1, "wavelength_min": 1.0, "wavelength_max": 1.0,
         "median_snr": None},
    ]
    svc = _FakeSparclService(retrieve_result={
        "success": True, "spectra": spectra, "count": 1,
        "warnings": [], "provenance": {},
    })
    ctx, _ = _ctx(svc, store=None)
    out = _run(SparclGetSpectrum(), ctx, sparcl_ids=["s1"])

    assert out["success"] is True
    assert "result_id" not in out
    assert any("not persisted" in w for w in out["warnings"])


def test_get_spectrum_failure_passes_through():
    svc = _FakeSparclService(retrieve_result={"success": False, "error": "nope"})
    ctx, _ = _ctx(svc)
    out = _run(SparclGetSpectrum(), ctx, sparcl_ids=["x"])
    assert out == {"success": False, "error": "nope"}
