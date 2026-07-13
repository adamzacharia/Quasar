import json
import os
import sys
import types

import pandas as pd
import pytest

from services.mmu_hats import (
    MMU_HATS_CATALOGS,
    MMUHatsError,
    MMUHatsService,
    MMUHatsUnavailableError,
    compact_preview_frame,
)


pytestmark = pytest.mark.slow


class NestedDType:
    def __str__(self):
        return "nested<float64>"


class FakeLazyCatalog:
    def __init__(self, frame, dtypes=None, calls=None, uri="fake"):
        self._frame = frame.copy()
        self.columns = list(frame.columns)
        self.dtypes = dtypes or {col: "float64" for col in self.columns}
        self.calls = calls if calls is not None else {"selected": [], "computed": 0, "crossmatch": []}
        self.uri = uri

    def __getitem__(self, columns):
        selected = list(columns)
        self.calls.setdefault("selected", []).append(selected)
        frame = self._frame[[col for col in selected if col in self._frame.columns]].copy()
        dtypes = {col: self.dtypes.get(col, "float64") for col in frame.columns}
        return FakeLazyCatalog(frame, dtypes=dtypes, calls=self.calls, uri=self.uri)

    def compute(self):
        self.calls["computed"] = self.calls.get("computed", 0) + 1
        return self._frame.copy()

    def crossmatch(self, other, radius_arcsec, suffixes):
        self.calls.setdefault("crossmatch", []).append({"radius_arcsec": radius_arcsec, "suffixes": suffixes})
        left = self._frame.reset_index(drop=True)
        right = other._frame.reset_index(drop=True)
        n = min(len(left), len(right))
        data = {}
        dtypes = {}
        for col in left.columns:
            out = f"{col}{suffixes[0]}"
            data[out] = left[col].head(n).tolist()
            dtypes[out] = self.dtypes.get(col, "float64")
        for col in right.columns:
            out = f"{col}{suffixes[1]}"
            data[out] = right[col].head(n).tolist()
            dtypes[out] = other.dtypes.get(col, "float64")
        return FakeLazyCatalog(pd.DataFrame(data), dtypes=dtypes, calls=self.calls, uri="matched")


def _frame(rows=3, include_nested=True):
    data = {
        "source_id": list(range(rows)),
        "object_id": [f"obj-{i}" for i in range(rows)],
        "ra": [10.0 + i * 0.001 for i in range(rows)],
        "dec": [-2.0 - i * 0.001 for i in range(rows)],
        "parallax": [0.1] * rows,
    }
    if include_nested:
        data["nested_blob"] = [[1, 2]] * rows
    df = pd.DataFrame(data)
    df.index.name = "_healpix_29"
    return df


def _install_fake_lsdb(monkeypatch, frame=None, dtypes=None, catalogs=None, open_raises=None):
    calls = {"open": [], "cone": [], "selected": [], "computed": 0, "crossmatch": []}
    default_frame = frame if frame is not None else _frame()
    default_dtypes = dtypes or {col: (NestedDType() if col == "nested_blob" else "float64") for col in default_frame.columns}

    class ConeSearch:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls["cone"].append(kwargs)

    def open_catalog(uri, **kwargs):
        calls["open"].append({"uri": uri, "kwargs": kwargs})
        if open_raises:
            raise open_raises
        if catalogs and uri in catalogs:
            spec = catalogs[uri]
            return FakeLazyCatalog(spec["frame"], dtypes=spec.get("dtypes"), calls=calls, uri=uri)
        return FakeLazyCatalog(default_frame, dtypes=default_dtypes, calls=calls, uri=uri)

    fake_lsdb = types.SimpleNamespace(ConeSearch=ConeSearch, open_catalog=open_catalog, __version__="0.9.2")
    monkeypatch.setitem(sys.modules, "lsdb", fake_lsdb)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace())
    return calls


def _service(**overrides):
    defaults = dict(enabled=True, max_radius_arcsec=3600.0, max_rows=5000, default_radius_arcsec=120.0, default_rows=500)
    defaults.update(overrides)
    return MMUHatsService(**defaults)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ra": -1, "dec": 0},
        {"ra": 360, "dec": 0},
        {"ra": 1, "dec": 91},
        {"ra": 1, "dec": -91},
        {"ra": None, "dec": 0},
        {"ra": float("nan"), "dec": 0},
        {"ra": 1, "dec": float("inf")},
    ],
)
def test_ra_dec_validation(kwargs):
    with pytest.raises(MMUHatsError):
        _service().cone_search("gaia", radius_arcsec=1, **kwargs)


def test_unknown_catalog_names_valid_keys():
    with pytest.raises(MMUHatsError, match="gaia"):
        _service().cone_search("bad", ra=1, dec=0)


def test_radius_default_and_clamp(monkeypatch):
    calls = _install_fake_lsdb(monkeypatch)
    svc = _service()
    defaulted = svc.cone_search("gaia", 10, 0, radius_arcsec=None)
    assert defaulted["provenance"]["cone"]["radius_arcsec"] == 120.0
    clamped = svc.cone_search("gaia", 10, 0, radius_arcsec=7200)
    assert clamped["provenance"]["cone"]["radius_arcsec"] == 3600.0
    assert any("clamped" in w for w in clamped["warnings"])
    assert calls["cone"][-1]["radius_arcsec"] == 3600.0


def test_max_rows_clamp_and_truncation(monkeypatch):
    frame = _frame(rows=6001, include_nested=False)
    _install_fake_lsdb(monkeypatch, frame=frame)
    out = _service().cone_search("gaia", 10, 0, max_rows=999999)
    assert out["rowcount"] == 6001
    assert out["returned_rows"] == 5000
    assert out["provenance"]["max_rows"] == 5000
    assert any("max_rows" in w and "clamped" in w for w in out["warnings"])
    assert any("truncated" in w for w in out["warnings"])


def test_import_failure_unavailable(monkeypatch):
    svc = _service()

    def boom():
        raise RuntimeError("no lsdb")

    monkeypatch.setattr(svc, "_import_lsdb", boom)
    available, reason = svc.is_available()
    assert available is False
    assert "no lsdb" in reason
    with pytest.raises(MMUHatsUnavailableError, match="lsdb"):
        svc.cone_search("gaia", 10, 0)


def test_disabled_feature_unavailable():
    svc = _service(enabled=False)
    available, reason = svc.is_available()
    assert available is False
    assert "ENABLE_MMU_HATS" in reason
    with pytest.raises(MMUHatsUnavailableError):
        svc.cone_search("gaia", 10, 0)


def test_registry_lists_five_catalogs_without_import(monkeypatch):
    svc = _service()
    monkeypatch.setattr(svc, "_import_lsdb", lambda: (_ for _ in ()).throw(RuntimeError("should not import")))
    catalogs = svc.list_catalogs()
    keys = {row["key"] for row in catalogs}
    assert keys == {"gaia", "desi_edr_sv3", "sdss", "tess_spoc", "chandra_spectra"}
    for row in catalogs:
        assert {"key", "label", "uri", "description", "preferred_columns"} <= set(row)
        assert row["uri"].startswith("hf://datasets/UniverseTBD/")


def test_column_selection_nested_exclusion_and_healpix_drop(monkeypatch):
    frame = _frame(rows=2, include_nested=True)
    dtypes = {col: (NestedDType() if col == "nested_blob" else "float64") for col in frame.columns}
    _install_fake_lsdb(monkeypatch, frame=frame, dtypes=dtypes)
    out = _service().cone_search("gaia", 10, 0, columns=None)
    assert "nested_blob" not in out["columns"]
    assert "_healpix_29" not in out["columns"]
    assert {"object_id", "ra", "dec"} <= set(out["columns"])


def test_requested_columns_warn_missing_and_allow_explicit_nested(monkeypatch):
    frame = _frame(rows=2, include_nested=True)
    dtypes = {col: (NestedDType() if col == "nested_blob" else "float64") for col in frame.columns}
    _install_fake_lsdb(monkeypatch, frame=frame, dtypes=dtypes)
    out = _service().cone_search("gaia", 10, 0, columns=["source_id", "missing", "nested_blob"])
    assert "nested_blob" in out["columns"]
    assert "ra" in out["columns"] and "dec" in out["columns"]
    assert any("missing" in w for w in out["warnings"])


def test_all_requested_columns_missing_errors(monkeypatch):
    _install_fake_lsdb(monkeypatch)
    with pytest.raises(MMUHatsError, match="None of the requested"):
        _service().cone_search("gaia", 10, 0, columns=["not_a_column"])


def test_crossmatch_bounded_margin_cache_and_coords(monkeypatch):
    left_uri = MMU_HATS_CATALOGS["gaia"]["uri"]
    right_uri = MMU_HATS_CATALOGS["desi_edr_sv3"]["uri"]
    calls = _install_fake_lsdb(
        monkeypatch,
        catalogs={
            left_uri: {"frame": _frame(rows=2, include_nested=True)},
            right_uri: {"frame": pd.DataFrame({"targetid": [1, 2], "ra": [10.0, 10.1], "dec": [-2.0, -2.1], "z": [0.1, 0.2]})},
        },
    )
    svc = _service()
    with pytest.raises(MMUHatsError):
        svc.crossmatch_catalogs("gaia", "desi_edr_sv3", None, 0)
    out = svc.crossmatch_catalogs("gaia", "desi_edr_sv3", 10, 0, match_radius_arcsec=25)
    assert out["provenance"]["match_radius_arcsec"] == 10.0
    assert any("10 arcsec" in w for w in out["warnings"])
    assert calls["open"][1]["kwargs"]["margin_cache"] == MMU_HATS_CATALOGS["desi_edr_sv3"]["margin_uri"]
    assert {"ra", "dec"} <= set(out["columns"])
    assert calls["crossmatch"][-1]["radius_arcsec"] == 10.0


def test_hf_home_setdefault_preserves_existing_value(monkeypatch):
    _install_fake_lsdb(monkeypatch)
    monkeypatch.setenv("HF_HOME", "custom-hf-cache")
    assert _service().is_available()[0] is True
    assert os.environ["HF_HOME"] == "custom-hf-cache"


def test_hf_home_defaults_under_configured_cache_dir(monkeypatch, tmp_path):
    _install_fake_lsdb(monkeypatch)
    # setenv-then-delenv so monkeypatch restores the pre-test state even though
    # the service itself writes HF_HOME via os.environ.setdefault.
    monkeypatch.setenv("HF_HOME", "sentinel")
    monkeypatch.delenv("HF_HOME")
    svc = _service(cache_dir=tmp_path / "mmu")
    assert svc.is_available()[0] is True
    assert os.environ["HF_HOME"] == str(tmp_path / "mmu" / "huggingface")


def test_hf_token_lowercase_alias_maps_to_hf_token(monkeypatch):
    _install_fake_lsdb(monkeypatch)
    # setenv-then-delenv so monkeypatch restores pre-test state even though
    # the service writes HF_TOKEN directly (same trick as the HF_HOME test).
    monkeypatch.setenv("HF_TOKEN", "sentinel")
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setenv("HF_token", "hf_dummy")
    assert _service().is_available()[0] is True
    assert os.environ["HF_TOKEN"] == "hf_dummy"


def test_hf_token_alias_does_not_override_existing_hf_token(monkeypatch):
    _install_fake_lsdb(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "canonical")
    monkeypatch.setenv("HF_token", "alias")
    assert _service().is_available()[0] is True
    # On Windows os.environ is case-insensitive so both names collapse to one
    # entry; on Linux the canonical spelling must win.
    assert os.environ["HF_TOKEN"] in {"canonical", "alias"}
    if os.name != "nt":
        assert os.environ["HF_TOKEN"] == "canonical"


def test_no_coverage_returns_empty_result_not_error(monkeypatch):
    _install_fake_lsdb(monkeypatch, open_raises=ValueError("The selected sky region has no coverage"))
    out = _service().cone_search("gaia", 10, 0)
    assert out["rowcount"] == 0
    assert out["returned_rows"] == 0
    assert any("no coverage" in w for w in out["warnings"])


def test_struct_fields_are_expanded_to_flat_columns(monkeypatch):
    frame = pd.DataFrame({
        "object_id": [1, 2],
        "ra": [10.0, 10.1],
        "dec": [-2.0, -2.1],
        "photometry": [
            {"phot_g_mean_mag": 12.5, "phot_bp_mean_mag": 12.9, "phot_rp_mean_mag": 11.9},
            {"phot_g_mean_mag": 15.0, "phot_bp_mean_mag": 15.4, "phot_rp_mean_mag": 14.4},
        ],
        "astrometry": [
            {"parallax": 7.4, "parallax_error": 0.02, "pmra": 20.0, "pmdec": -45.0},
            {"parallax": 2.1, "parallax_error": 0.05, "pmra": 1.0, "pmdec": -2.0},
        ],
    })
    frame.index.name = "_healpix_29"
    dtypes = {col: "float64" for col in frame.columns}
    _install_fake_lsdb(monkeypatch, frame=frame, dtypes=dtypes)
    out = _service().cone_search("gaia", 10, 0)
    assert "phot_g_mean_mag" in out["columns"]
    assert "parallax" in out["columns"]
    assert "photometry" not in out["columns"]
    assert "astrometry" not in out["columns"]
    df = out["dataframe"]
    assert df["phot_g_mean_mag"].iloc[0] == 12.5
    assert df["pmdec"].iloc[1] == -2.0


def test_compact_preview_frame_summarizes_nested_cells():
    df = pd.DataFrame({
        "source_id": [1, 2],
        "flux": [list(range(100000)), list(range(50))],
        "note": ["x" * 500, "short"],
    })
    out = compact_preview_frame(df)
    assert out["flux"].iloc[0] == "<list[100000]>"
    assert len(out["note"].iloc[0]) <= 203
    assert out["note"].iloc[1] == "short"
    assert out["source_id"].iloc[0] == 1
    assert len(json.dumps(out.to_dict("records"), default=str)) < 8000


@pytest.mark.skipif(os.getenv("RUN_LIVE_MMU_HATS_TESTS") != "1", reason="set RUN_LIVE_MMU_HATS_TESTS=1 to query live LSDB/HF catalogs")
def test_live_gaia_cone_search():
    pytest.importorskip("lsdb")
    pytest.importorskip("huggingface_hub")
    # Pleiades: verified covered in the MMU Gaia bright-star subset (150 rows
    # at 600 arcsec on 2026-07-02). MMU catalogs have real coverage holes, so
    # arbitrary coordinates are NOT a valid live probe.
    out = _service().cone_search("gaia", ra=56.75, dec=24.11, radius_arcsec=600)
    assert out["returned_rows"] > 0
    assert {"ra", "dec", "object_id"} <= set(out["columns"])
    assert "phot_g_mean_mag" in out["columns"]  # struct expansion worked live


@pytest.mark.skipif(os.getenv("RUN_LIVE_MMU_HATS_TESTS") != "1", reason="set RUN_LIVE_MMU_HATS_TESTS=1 to query live LSDB/HF catalogs")
def test_live_no_coverage_is_graceful():
    pytest.importorskip("lsdb")
    pytest.importorskip("huggingface_hub")
    out = _service().cone_search("gaia", ra=187.70593, dec=12.39112, radius_arcsec=60)  # M87: verified hole
    assert out["rowcount"] == 0
    assert any("no coverage" in w for w in out["warnings"])
