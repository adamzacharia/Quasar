# tests/unit/test_line_lists_and_xmatch.py
"""RANK 8 backend: services/line_lists.py (vacuum wavelengths by spectype),
the interactive SPARCL spectrum plotly spec + SLE deep-link meta, and the
xmatch_user_list capability over the CDS X-Match service (transport faked)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import capabilities.datalab as dl
from capabilities.base import CallContext
from services import cds_xmatch, line_lists
from services import sparcl_spectra as sps
from services.datalab_result_store import DatalabResultStore


# ── line_lists ────────────────────────────────────────────────────────────────
def test_spectype_sets_cover_the_sparcl_vocabulary():
    assert set(line_lists.SPECTYPE_LINES) == {"GALAXY", "STAR", "QSO"}
    for entries in line_lists.SPECTYPE_LINES.values():
        assert entries == sorted(entries, key=lambda e: e["wavelength_vac"])
        for entry in entries:
            assert entry["kind"] in {line_lists.EMISSION, line_lists.ABSORPTION, line_lists.BOTH}


def test_air_values_corrected_to_vacuum():
    # The legacy COMMON_LINES carried AIR values for exactly these three; the
    # vacuum corrections are ~+1.1 A (Morton 1991 convention, SDSS/DESI grids).
    galaxy = {(e["label"], e["wavelength_vac"]) for e in line_lists.lines_for_spectype("GALAXY")}
    assert ("Ca K", 3934.78) in galaxy
    assert ("Ca H", 3969.59) in galaxy
    assert ("[O II]", 3728.48) in galaxy
    # And the always-vacuum anchors are unchanged from the SDSS list.
    assert ("H alpha", 6564.61) in galaxy
    assert ("[O III]", 5008.24) in galaxy


def test_duplicate_labels_survive():
    labels = [e["label"] for e in line_lists.lines_for_spectype("GALAXY")]
    assert labels.count("[O III]") == 2 and labels.count("[S II]") == 2


def test_union_for_unknown_spectype_and_backcompat_tuples():
    union = line_lists.lines_for_spectype(None)
    assert {"Ly alpha", "Ca K", "H alpha"} <= {e["label"] for e in union}
    tuples = line_lists.common_lines()
    assert all(isinstance(t, tuple) and len(t) == 2 for t in tuples)
    # sparcl_spectra re-exports the same shape for legacy consumers
    assert sps.COMMON_LINES == tuples


def test_observed_lines_redshift_and_range():
    assert line_lists.observed_lines("GALAXY", None) == []
    lines = line_lists.observed_lines("GALAXY", 1.0, wavelength_min=9000.0, wavelength_max=11000.0)
    assert lines and all(9000.0 <= e["wavelength_observed"] <= 11000.0 for e in lines)
    ha = [e for e in line_lists.observed_lines("STAR", 0.0) if e["label"] == "H alpha"]
    assert ha and ha[0]["wavelength_observed"] == pytest.approx(6564.61)


def test_qso_set_has_uv_lines_star_set_does_not():
    qso = {e["label"] for e in line_lists.lines_for_spectype("QSO")}
    star = {e["label"] for e in line_lists.lines_for_spectype("STAR")}
    assert {"Ly alpha", "C IV", "Mg II"} <= qso
    assert "Ly alpha" not in star and "Ca II" in star


# ── interactive spectrum spec + deep-link meta ───────────────────────────────
def test_spectrum_plotly_spec_shapes_and_json_safety():
    wavelength = np.linspace(3600.0, 9800.0, 6000)
    flux = np.ones_like(wavelength)
    flux[10] = np.nan  # masked pixel must serialize as None, never NaN
    spec = sps._spectrum_plotly_spec(
        wavelength, flux, None, sid="abc12345", spectype="GALAXY",
        redshift=0.1, mark_lines=True,
    )
    assert spec["data"][0]["type"] == "scatter"  # basic plotly bundle has no scattergl
    xs = spec["data"][0]["x"]
    assert len(xs) <= sps.PLOTLY_MAX_POINTS
    assert None in spec["data"][0]["y"]
    import json
    json.dumps(spec, allow_nan=False)  # raises if any NaN survived
    labels = {a["text"] for a in spec["layout"]["annotations"]}
    assert "H alpha" in labels
    assert len(spec["layout"]["shapes"]) == len(spec["layout"]["annotations"]) > 0


def test_spectrum_plotly_spec_no_lines_without_redshift():
    wavelength = np.linspace(3600.0, 9800.0, 100)
    spec = sps._spectrum_plotly_spec(
        wavelength, np.ones_like(wavelength), None, sid="x", spectype="GALAXY",
        redshift=None, mark_lines=True,
    )
    assert spec["layout"]["shapes"] == [] and spec["layout"]["annotations"] == []


def test_spectrum_card_meta_deep_link():
    meta = sps._spectrum_card_meta("abc", spectype="GALAXY", redshift=0.1234, ra=150.1, dec=2.2)
    assert meta["sparcl_id"] == "abc"
    url = meta["line_explorer_url"]
    assert url.startswith("/spectral-lines?")
    assert "target=150.10000+%2B2.20000" in url and "z=0.1234" in url
    assert "ra_deg=150.1" in url and "dec_deg=2.2" in url
    assert "autorun" not in url  # coordinate targets must not auto-fire a job
    # No coordinates -> no link, but identity still present
    meta2 = sps._spectrum_card_meta("abc", spectype="STAR", redshift=None, ra=None, dec=None)
    assert "line_explorer_url" not in meta2


# ── CDS X-Match service (transport faked) ────────────────────────────────────
def test_resolve_catalog_aliases():
    assert cds_xmatch.resolve_catalog("2mass") == "vizier:II/246/out"
    assert cds_xmatch.resolve_catalog("SIMBAD") == "simbad"
    assert cds_xmatch.resolve_catalog("vizier:I/355/gaiadr3") == "vizier:I/355/gaiadr3"
    assert cds_xmatch.resolve_catalog("II/246/out") == "vizier:II/246/out"
    with pytest.raises(ValueError):
        cds_xmatch.resolve_catalog("not_a_catalog")


class _Resp:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text


def test_xmatch_dataframe_contract(monkeypatch):
    captured = {}

    def fake_post(url, data=None, files=None, timeout=None):
        captured.update({"url": url, "data": data, "files": files})
        return _Resp(200, "angDist,id,ra,dec,Jmag\n0.2,obj1,10.0,20.0,12.3\n")

    monkeypatch.setattr(cds_xmatch.requests, "post", fake_post)
    frame = pd.DataFrame({"id": ["obj1"], "ra": [10.0], "dec": [20.0]})
    out = cds_xmatch.xmatch_dataframe(frame, catalog="2mass", radius_arcsec=5.0)
    assert out["matched_rows"] == 1 and out["cat2"] == "vizier:II/246/out"
    assert captured["data"]["request"] == "xmatch"
    assert captured["data"]["colRA1"] == "ra" and captured["data"]["colDec1"] == "dec"
    assert captured["data"]["responseFormat"] == "csv"
    assert "cat1" in captured["files"]


def test_xmatch_validations():
    frame = pd.DataFrame({"ra": [1.0], "dec": [2.0]})
    with pytest.raises(ValueError):
        cds_xmatch.xmatch_dataframe(frame, catalog="2mass", radius_arcsec=999)  # > 180
    with pytest.raises(ValueError):
        cds_xmatch.xmatch_dataframe(frame, catalog="2mass", selection="worst")
    with pytest.raises(ValueError):
        cds_xmatch.xmatch_dataframe(pd.DataFrame({"x": [1]}), catalog="2mass")  # no coords
    with pytest.raises(ValueError):
        cds_xmatch.xmatch_dataframe(pd.DataFrame(), catalog="2mass")


def test_xmatch_error_body_surfaced(monkeypatch):
    def fake_post(url, data=None, files=None, timeout=None):
        return _Resp(400, "<VOTABLE><INFO name='QUERY_STATUS'>Missing parameter colRA1</INFO></VOTABLE>")

    monkeypatch.setattr(cds_xmatch.requests, "post", fake_post)
    frame = pd.DataFrame({"ra": [1.0], "dec": [2.0]})
    with pytest.raises(cds_xmatch.CdsXmatchError) as excinfo:
        cds_xmatch.xmatch_dataframe(frame, catalog="2mass")
    assert "Missing parameter colRA1" in str(excinfo.value)


# ── xmatch_user_list capability ───────────────────────────────────────────────
def _fake_xmatch(monkeypatch, response_csv="angDist,id,ra,dec,Jmag\n0.2,obj1,10.0,20.0,12.3\n"):
    def fake_post(url, data=None, files=None, timeout=None):
        return _Resp(200, response_csv)

    monkeypatch.setattr(cds_xmatch.requests, "post", fake_post)


def test_xmatch_capability_objects_path(monkeypatch):
    _fake_xmatch(monkeypatch)
    store = DatalabResultStore(enable_disk_cache=False)
    ctx = CallContext(services={}, result_store=store)
    out = dl.XmatchUserList().run(
        dl.XmatchUserListInput(catalog="2mass", objects=[{"ra": 10.0, "dec": 20.0, "name": "t1"}]),
        ctx,
    ).to_native()
    assert out["success"] is True and out["result_id"].startswith("dlr_")
    assert out["matched_rows"] == 1 and out["catalog"] == "vizier:II/246/out"
    stored = store.get(out["result_id"])
    assert stored.provenance["service"] == "cds_xmatch"


def test_xmatch_capability_result_id_path(monkeypatch):
    _fake_xmatch(monkeypatch)
    store = DatalabResultStore(enable_disk_cache=False)
    source_id = store.put(pd.DataFrame({"ra": [10.0], "dec": [20.0]}), {})
    ctx = CallContext(services={}, result_store=store)
    out = dl.XmatchUserList().run(
        dl.XmatchUserListInput(catalog="gaia_dr3", result_id=source_id, radius_arcsec=2.0),
        ctx,
    ).to_native()
    assert out["success"] is True
    assert store.get(out["result_id"]).provenance["source_result_id"] == source_id


def test_xmatch_capability_requires_exactly_one_input():
    ctx = CallContext(services={}, result_store=DatalabResultStore(enable_disk_cache=False))
    neither = dl.XmatchUserList().run(dl.XmatchUserListInput(catalog="2mass"), ctx).to_native()
    assert neither["success"] is False and "exactly one" in neither["error"]
    both = dl.XmatchUserList().run(
        dl.XmatchUserListInput(catalog="2mass", objects=[{"ra": 1, "dec": 2}], result_id="dlr_x"),
        ctx,
    ).to_native()
    assert both["success"] is False and "exactly one" in both["error"]


def test_new_capabilities_registered_in_module_lists():
    names = {cap.name for cap in dl.CAPABILITIES}
    assert {"xmatch_user_list", "datalab_save_result", "datalab_list_my_tables", "datalab_load_my_table"} <= names


# ── guard-review regressions (task-097819d-1643) ─────────────────────────────
def test_xmatch_preserves_leading_zero_identifiers(monkeypatch):
    """CX-11: uploaded string ids like '00123' must not come back as int 123."""
    def fake_post(url, data=None, files=None, timeout=None):
        return _Resp(200, "angDist,id,ra,dec,Jmag\n0.2,00123,10.0,20.0,12.3\n")

    monkeypatch.setattr(cds_xmatch.requests, "post", fake_post)
    frame = pd.DataFrame({"id": ["00123"], "ra": [10.0], "dec": [20.0]})
    out = cds_xmatch.xmatch_dataframe(frame, catalog="2mass")
    assert list(out["dataframe"]["id"]) == ["00123"]
    # numeric response columns keep numeric dtypes
    assert out["dataframe"]["angDist"].dtype.kind == "f"


def test_period_fold_fap_uses_the_searched_frequency_range():
    """CX-05: the reported FAP must be computed for the same frequency range
    the periodogram actually searched, not astropy's derived default."""
    import numpy as np
    from astropy.timeseries import LombScargle

    from services import datalab_analysis as da
    from services.datalab_result_store import DatalabResultStore

    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(0, 60, 120))
    mag = 18.0 + 0.3 * np.sin(2 * np.pi * t / 0.7) + rng.normal(0, 0.05, t.size)
    err = np.full(t.size, 0.05)
    store = DatalabResultStore(enable_disk_cache=False)
    rid = store.put(pd.DataFrame({"mjd": t, "cmag": mag, "cerr": err}), {})
    out = da.period_fold(rid, result_store=store, min_frequency=0.5, max_frequency=4.0)
    assert out["success"] is True

    ls = LombScargle(t, mag, dy=err)
    freq, power = ls.autopower(minimum_frequency=0.5, maximum_frequency=4.0)
    expected = float(ls.false_alarm_probability(
        power.max(), minimum_frequency=0.5, maximum_frequency=4.0,
    ))
    assert out["false_alarm_probability"] == pytest.approx(expected, rel=1e-6)
    for alt in out["alternate_periods"]:
        assert 0.25 <= alt["period_days"] <= 2.0  # inside the searched range


def test_target_resolver_short_circuits_coordinate_pairs(monkeypatch):
    """CX-09: 'RA DEC' targets (the SPARCL deep-link form) must resolve as
    literal coordinates without any SIMBAD/NED name lookup."""
    from services.spectral_line_explorer import TargetResolver, _parse_coordinate_pair

    assert _parse_coordinate_pair("150.09561 +2.20013") == (150.09561, 2.20013)
    assert _parse_coordinate_pair("150.1, -69.75") == (150.1, -69.75)
    assert _parse_coordinate_pair("NGC 4151") is None
    assert _parse_coordinate_pair("400.0 +2.0") is None  # RA out of range

    resolver = TargetResolver()

    def _no_network(*args, **kwargs):
        raise AssertionError("name resolution must not run for coordinate targets")

    monkeypatch.setattr(resolver, "_query_simbad", _no_network)
    monkeypatch.setattr(resolver, "_query_ned", _no_network)
    out = resolver.resolve("150.09561 +2.20013", explicit_redshift=0.186)
    assert out["state"] == "resolved"
    assert out["ra_deg"] == pytest.approx(150.09561)
    assert out["dec_deg"] == pytest.approx(2.20013)
    assert out["redshift"] == pytest.approx(0.186)
    # No explicit redshift -> coordinates known, z is the user's call.
    out2 = resolver.resolve("150.09561 +2.20013")
    assert out2["state"] == "needs_input" and out2["ra_deg"] is not None
