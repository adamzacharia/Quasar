"""Live MMDC checks (network). Run with RUN_LIVE_MMDC_TESTS=1.

One SED fetch (the competitor's 1ES 1959+650 2008-2020 question), one light
curve and one synchronous model inference against https://mmdc.am.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(os.getenv("RUN_LIVE_MMDC_TESTS") != "1",
                                reason="live MMDC tests: set RUN_LIVE_MMDC_TESTS=1")


def test_live_sed_window_is_contained():
    from services.mmdc_service import MmdcService

    out = MmdcService().sed(target_name="1ES 1959+650", start_date="2008", end_date="2020")
    assert out["success"], out.get("error")
    table, _ = out["_table"]
    assert table["MJD_end"].max() <= 59214.0 + 1e-6          # AstroGenesis reported 59694.67 here
    w = out["window"]
    assert w["rule_applied"] == "contained" and w["straddling_rows"] >= 1
    assert "MMDCGR" in w["straddling_by_catalog"] and w["undated_archival_rows"] > 0
    assert out["redshift"] == pytest.approx(0.047, abs=1e-3)
    assert out["summary"]["totals"]["rows"] == len(table)


def test_live_lightcurve_has_fermi_indices():
    from services.mmdc_service import MmdcService

    out = MmdcService().lightcurve(target_name="1ES 1959+650", start_date="2018", end_date="2019", make_plot=False)
    assert out["success"], out.get("error")
    table, _ = out["_table"]
    gr = table[table["catalog"] == "MMDCGR"]
    assert len(gr) > 0 and gr["spectral_index"].notna().all()
    assert table["time_mjd"].between(58119.0, 58849.0).all()


def test_live_ssc_inference():
    from services.mmdc_modeling import MmdcModelingService

    params = {"log_B": -1.5, "log_electron_luminosity": 44.0, "log_gamma_cut": 5.0, "log_gamma_min": 2.0,
              "log_radius": 16.0, "lorentz_factor": 20.0, "spectral_index": 2.2}
    out = MmdcModelingService().spectrum(model_type="SSC", z=0.03179, parameters=params)
    assert out["success"] and out["model_points"] >= 50 and out["spectral_peaks"]
