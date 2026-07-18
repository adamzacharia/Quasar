import os
import sys
import types

import numpy as np
import pytest

from services import plotting
from services.sparcl_spectra import SparclSpectraService


def _plot_dir(name):
    path = os.path.join("test_results", "live_imagery", name)
    os.makedirs(path, exist_ok=True)
    return path


class FakeResult:
    def __init__(self, records):
        self.records = records


def install_fake_sparcl(monkeypatch, fake_client_cls):
    sparcl_mod = types.ModuleType("sparcl")
    client_mod = types.ModuleType("sparcl.client")
    client_mod.SparclClient = fake_client_cls
    monkeypatch.setitem(sys.modules, "sparcl", sparcl_mod)
    monkeypatch.setitem(sys.modules, "sparcl.client", client_mod)


def test_find_spectra_cos_dec_box_and_post_filter(monkeypatch):
    class FakeClient:
        calls = []

        def find(self, outfields=None, constraints=None, limit=None):
            self.calls.append({"outfields": outfields, "constraints": constraints, "limit": limit})
            return FakeResult(
                [
                    {"sparcl_id": "near", "ra": 10.0001, "dec": 60.0001, "redshift": 0.1, "spectype": "GALAXY", "data_release": "DESI-EDR"},
                    {"sparcl_id": "far", "ra": 12.0, "dec": 60.0, "redshift": 0.2, "spectype": "QSO", "data_release": "DESI-EDR"},
                ]
            )

    install_fake_sparcl(monkeypatch, FakeClient)
    out = SparclSpectraService().find_spectra(10.0, 60.0, radius_arcsec=60, limit=10)

    assert out["success"] is True
    assert [row["sparcl_id"] for row in out["rows"]] == ["near"]
    call = FakeClient.calls[0]
    ra_min, ra_max = call["constraints"]["ra"]
    dec_min, dec_max = call["constraints"]["dec"]
    assert abs(ra_min - 9.9666667) < 1e-4
    assert abs(ra_max - 10.0333333) < 1e-4
    assert abs(dec_min - 59.9833333) < 1e-4
    assert abs(dec_max - 60.0166667) < 1e-4
    assert call["constraints"]["data_release"] == ["DESI-DR1", "SDSS-DR17"]


def test_find_spectra_splits_ra_wrap(monkeypatch):
    class FakeClient:
        calls = []

        def find(self, outfields=None, constraints=None, limit=None):
            self.calls.append(constraints)
            return FakeResult([])

    install_fake_sparcl(monkeypatch, FakeClient)
    out = SparclSpectraService().find_spectra(359.99, 0.0, radius_arcsec=120, limit=5)

    assert out["success"] is True
    assert len(FakeClient.calls) == 2
    assert FakeClient.calls[0]["ra"][1] == 360.0
    assert FakeClient.calls[1]["ra"][0] == 0.0
    assert any("split" in warning for warning in out["warnings"])


def test_plot_spectrum_with_model_and_without_redshift(monkeypatch):
    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", _plot_dir("sparcl"))
    wave = np.linspace(3600.0, 9800.0, 100)
    flux = np.sin(np.linspace(0, 6.0, 100))

    class FakeClient:
        def find(self, *args, **kwargs):
            return FakeResult([])

        def retrieve(self, uuid_list=None, include=None):
            sid = uuid_list[0]
            if sid == "with-model":
                return FakeResult([{"wavelength": wave, "flux": flux, "model": flux * 0.9, "redshift": 0.1, "spectype": "GALAXY"}])
            return FakeResult([{"wavelength": wave, "flux": flux, "spectype": "STAR"}])

    install_fake_sparcl(monkeypatch, FakeClient)
    svc = SparclSpectraService()
    out = svc.plot_spectrum("with-model", smooth=3)
    assert out["success"] is True
    assert out["n_points"] == 100
    assert out["redshift"] == 0.1
    assert out["path"].endswith(".png")

    out2 = svc.plot_spectrum("without-model", mark_lines=True)
    assert out2["success"] is True
    assert out2["redshift"] is None


def test_plot_spectrum_masks_bad_ivar_before_smoothing(monkeypatch):
    # sparcl-smooth-before-ivar-mask: a spike at an ivar=0 pixel must not
    # contaminate the smoothed values of its (good) neighbors.
    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", _plot_dir("sparcl"))
    wave = np.linspace(3600.0, 9800.0, 100)
    flux = np.ones(100)
    flux[50] = 1.0e6  # cosmic-ray spike
    ivar = np.ones(100)
    ivar[50] = 0.0  # masked pixel

    class FakeClient:
        def retrieve(self, uuid_list=None, include=None):
            return FakeResult(
                [{"wavelength": wave, "flux": flux, "ivar": ivar, "spectype": "GALAXY"}]
            )

    install_fake_sparcl(monkeypatch, FakeClient)
    out = SparclSpectraService().plot_spectrum("spiked", smooth=5)

    assert out["success"] is True
    y = out["plotly_spec"]["data"][0]["y"]
    assert y[50] is None  # masked pixel stays blanked
    # Neighbors are smoothed over good pixels only: all 1.0, not ~2e5.
    for index in (48, 49, 51, 52):
        assert y[index] == pytest.approx(1.0, abs=1e-9)


def test_find_spectra_warns_when_remote_box_hits_limit(monkeypatch):
    # sparcl-box-limit-silent-truncation: a capped box query must disclose
    # possible truncation before the cone cut.
    class FakeClient:
        def find(self, outfields=None, constraints=None, limit=None):
            return FakeResult(
                [
                    {"sparcl_id": f"s{i}", "ra": 10.0 + i * 1e-4, "dec": 60.0,
                     "redshift": 0.1, "spectype": "GALAXY", "data_release": "DESI-DR1"}
                    for i in range(limit)
                ]
            )

    install_fake_sparcl(monkeypatch, FakeClient)
    out = SparclSpectraService().find_spectra(10.0, 60.0, radius_arcsec=60, limit=3)

    assert out["success"] is True
    assert any("row cap" in warning for warning in out["warnings"])


def test_find_spectra_no_truncation_warning_below_limit(monkeypatch):
    class FakeClient:
        def find(self, outfields=None, constraints=None, limit=None):
            return FakeResult(
                [{"sparcl_id": "only", "ra": 10.0, "dec": 60.0, "redshift": 0.1,
                  "spectype": "GALAXY", "data_release": "DESI-DR1"}]
            )

    install_fake_sparcl(monkeypatch, FakeClient)
    out = SparclSpectraService().find_spectra(10.0, 60.0, radius_arcsec=60, limit=10)

    assert out["success"] is True
    assert not any("row cap" in warning for warning in out["warnings"])


def test_search_spectra_cone_warns_when_remote_box_hits_limit(monkeypatch):
    class FakeClient:
        def find(self, outfields=None, constraints=None, limit=None):
            return FakeResult(
                [
                    {"sparcl_id": f"s{i}", "ra": 10.0 + i * 1e-4, "dec": 60.0,
                     "redshift": 0.1, "spectype": "GALAXY", "data_release": "DESI-DR1"}
                    for i in range(limit)
                ]
            )

    install_fake_sparcl(monkeypatch, FakeClient)
    out = SparclSpectraService().search_spectra(
        spectype="GALAXY", ra=10.0, dec=60.0, radius_arcsec=60, limit=3
    )

    assert out["success"] is True
    assert any("row cap" in warning for warning in out["warnings"])


def test_plot_spectrum_downsampling_is_annotated(monkeypatch):
    # plotly-downsample-unannotated: stride decimation must be disclosed in
    # the interactive figure and in the card warnings.
    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", _plot_dir("sparcl"))
    n = 9000  # stride 3 above the 4000-point cap
    wave = np.linspace(3600.0, 9800.0, n)
    flux = np.ones(n)

    class FakeClient:
        def retrieve(self, uuid_list=None, include=None):
            return FakeResult([{"wavelength": wave, "flux": flux, "spectype": "QSO"}])

    install_fake_sparcl(monkeypatch, FakeClient)
    out = SparclSpectraService().plot_spectrum("big")

    assert out["success"] is True
    layout = out["plotly_spec"]["layout"]
    downsample = layout["meta"]["downsample"]
    assert downsample["stride"] == 3
    assert downsample["points_total"] == n
    assert downsample["points_shown"] == len(out["plotly_spec"]["data"][0]["x"])
    assert any("downsampled" in a["text"].lower() for a in layout["annotations"])
    assert any("downsampled" in warning.lower() for warning in out["warnings"])


def test_search_spectra_constraint_mode(monkeypatch):
    class FakeClient:
        calls = []

        def find(self, outfields=None, constraints=None, limit=None):
            self.calls.append({"outfields": outfields, "constraints": constraints, "limit": limit})
            return FakeResult(
                [
                    {"sparcl_id": "a", "ra": 10.0, "dec": 1.0, "redshift": 0.15, "spectype": "GALAXY", "data_release": "DESI-DR1"},
                    {"sparcl_id": "a", "ra": 10.0, "dec": 1.0, "redshift": 0.15, "spectype": "GALAXY", "data_release": "DESI-DR1"},
                    {"sparcl_id": "b", "ra": 210.0, "dec": -3.0, "redshift": 0.28, "spectype": "GALAXY", "data_release": "SDSS-DR17"},
                ]
            )

    FakeClient.calls = []
    install_fake_sparcl(monkeypatch, FakeClient)
    out = SparclSpectraService().search_spectra(
        spectype="galaxy", redshift_min=0.1, redshift_max=0.3, limit=10
    )

    assert out["success"] is True
    # deduped by sparcl_id, no distance column in constraint mode
    assert [row["sparcl_id"] for row in out["rows"]] == ["a", "b"]
    assert "distance_arcsec" not in out["rows"][0]
    call = FakeClient.calls[0]
    assert call["constraints"]["spectype"] == ["GALAXY"]
    assert call["constraints"]["redshift"] == [0.1, 0.3]
    assert call["constraints"]["data_release"] == ["DESI-DR1", "SDSS-DR17"]
    assert "ra" not in call["constraints"]
    assert out["provenance"]["constraints"]["redshift"] == [0.1, 0.3]


def test_search_spectra_with_cone_filters_and_sorts(monkeypatch):
    class FakeClient:
        calls = []

        def find(self, outfields=None, constraints=None, limit=None):
            self.calls.append(constraints)
            return FakeResult(
                [
                    {"sparcl_id": "far", "ra": 12.0, "dec": 60.0, "redshift": 0.2, "spectype": "GALAXY", "data_release": "DESI-DR1"},
                    {"sparcl_id": "near", "ra": 10.0001, "dec": 60.0001, "redshift": 0.1, "spectype": "GALAXY", "data_release": "DESI-DR1"},
                ]
            )

    FakeClient.calls = []
    install_fake_sparcl(monkeypatch, FakeClient)
    out = SparclSpectraService().search_spectra(
        spectype="GALAXY", ra=10.0, dec=60.0, radius_arcsec=60, limit=10
    )

    assert out["success"] is True
    assert [row["sparcl_id"] for row in out["rows"]] == ["near"]
    assert out["rows"][0]["distance_arcsec"] < 60
    assert FakeClient.calls[0]["spectype"] == ["GALAXY"]
    assert "ra" in FakeClient.calls[0] and "dec" in FakeClient.calls[0]


def test_search_spectra_rejects_bad_inputs(monkeypatch):
    class FakeClient:
        def find(self, *args, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("find should not be called on invalid input")

    install_fake_sparcl(monkeypatch, FakeClient)
    svc = SparclSpectraService()

    out = svc.search_spectra(spectype="NEBULA")
    assert out["success"] is False and "spectype" in out["error"]

    out2 = svc.search_spectra(redshift_min=0.5, redshift_max=0.1)
    assert out2["success"] is False and "redshift_min" in out2["error"]


def test_retrieve_spectra_arrays_and_median_snr(monkeypatch):
    wave = np.linspace(3600.0, 9800.0, 50)
    flux = np.full(50, 2.0)
    ivar = np.full(50, 4.0)
    ivar[:5] = 0.0  # bad pixels excluded from S/N

    class FakeClient:
        calls = []

        def retrieve(self, uuid_list=None, include=None):
            self.calls.append({"uuid_list": uuid_list, "include": include})
            return FakeResult(
                [
                    {"sparcl_id": "s1", "wavelength": wave, "flux": flux, "ivar": ivar,
                     "redshift": 0.2, "spectype": "GALAXY", "data_release": "DESI-DR1"},
                ]
            )

    FakeClient.calls = []
    install_fake_sparcl(monkeypatch, FakeClient)
    out = SparclSpectraService().retrieve_spectra(["s1", "gone"])

    assert out["success"] is True
    assert out["count"] == 1
    spectrum = out["spectra"][0]
    assert spectrum["n_points"] == 50
    assert spectrum["wavelength_min"] == 3600.0
    assert spectrum["wavelength_max"] == 9800.0
    # flux * sqrt(ivar) = 2 * 2 = 4 on good pixels
    assert abs(spectrum["median_snr"] - 4.0) < 1e-9
    assert "sparcl_id" in FakeClient.calls[0]["include"]
    assert any("returned no spectrum" in w for w in out["warnings"])


def test_retrieve_spectra_caps_ids_and_requires_ids(monkeypatch):
    class FakeClient:
        calls = []

        def retrieve(self, uuid_list=None, include=None):
            self.calls.append(list(uuid_list))
            records = [
                {"sparcl_id": sid, "wavelength": [1.0, 2.0], "flux": [1.0, 1.0]}
                for sid in uuid_list
            ]
            return FakeResult(records)

    FakeClient.calls = []
    install_fake_sparcl(monkeypatch, FakeClient)
    svc = SparclSpectraService()

    out = svc.retrieve_spectra([f"id{i}" for i in range(60)])
    assert out["success"] is True
    assert len(FakeClient.calls[0]) == 50
    assert any("first 50" in w for w in out["warnings"])

    out2 = svc.retrieve_spectra([])
    assert out2["success"] is False and "sparcl_ids" in out2["error"]


def test_missing_sparcl_client_error(monkeypatch):
    from services import sparcl_spectra

    def missing(name):
        raise ImportError("no sparcl")

    monkeypatch.setattr(sparcl_spectra.importlib, "import_module", missing)
    out = SparclSpectraService().find_spectra(10.0, 0.0)
    assert out["success"] is False
    assert "pip install sparclclient" in out["error"]
