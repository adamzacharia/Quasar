import os
import sys
import types

import numpy as np
import pytest

from services import plotting
from services.ned_photometry import NedPhotometryService


pytestmark = pytest.mark.slow


def _plot_dir(name):
    path = os.path.join("test_results", "live_imagery", name)
    os.makedirs(path, exist_ok=True)
    return path


def test_ned_sed_plot_cleans_masked_rows_and_plots(monkeypatch):
    pytest.importorskip("astropy.table")
    from astropy.table import MaskedColumn, Table

    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", _plot_dir("ned"))

    table = Table()
    table["Frequency"] = MaskedColumn([1.0e9, 2.0e9, -1.0, 4.0e9], mask=[False, False, False, True])
    table["Flux Density"] = MaskedColumn([1.5, 0.0, 2.0, 3.0], mask=[False, False, False, True])
    table["Observed Passband"] = ["radio", "badzero", "badfreq", "masked"]
    table["Refcode"] = ["ref1", "ref2", "ref3", "ref4"]

    class FakeNed:
        calls = []

        @staticmethod
        def get_table(target, table=None):
            FakeNed.calls.append((target, table))
            return table_obj

    table_obj = table
    ned_mod = types.ModuleType("astroquery.ipac.ned")
    ned_mod.Ned = FakeNed
    ipac_mod = types.ModuleType("astroquery.ipac")
    astroquery_mod = types.ModuleType("astroquery")
    monkeypatch.setitem(sys.modules, "astroquery", astroquery_mod)
    monkeypatch.setitem(sys.modules, "astroquery.ipac", ipac_mod)
    monkeypatch.setitem(sys.modules, "astroquery.ipac.ned", ned_mod)

    out = NedPhotometryService().sed_plot("M87")

    assert out["success"] is True
    assert out["rowcount"] == 1
    assert out["preview"] == [{"passband": "radio", "frequency": 1.0e9, "flux_jy": 1.5, "refcode": "ref1"}]
    assert out["path"].endswith(".png")
    assert FakeNed.calls == [("M87", "photometry")]
