import os
import sys
import types

import numpy as np

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
    assert call["constraints"]["data_release"] == ["DESI-EDR"]


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


def test_missing_sparcl_client_error(monkeypatch):
    from services import sparcl_spectra

    def missing(name):
        raise ImportError("no sparcl")

    monkeypatch.setattr(sparcl_spectra.importlib, "import_module", missing)
    out = SparclSpectraService().find_spectra(10.0, 0.0)
    assert out["success"] is False
    assert "pip install sparclclient" in out["error"]
