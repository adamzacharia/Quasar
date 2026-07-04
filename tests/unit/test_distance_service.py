import math

import pytest

from services.distance_service import (
    LOW_PARALLAX_WARNING,
    PECULIAR_VELOCITY_WARNING,
    DistanceService,
)


class FakeResponse:
    def __init__(self, payload=None, status_code=200, text="", json_error=None):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code
        self.text = text
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise self._json_error
        return self._payload


GAIA_METADATA = [
    {"name": "source_id"},
    {"name": "ra"},
    {"name": "dec"},
    {"name": "parallax"},
    {"name": "parallax_error"},
    {"name": "phot_g_mean_mag"},
    {"name": "pmra"},
    {"name": "pmdec"},
    {"name": "r_med_geo"},
    {"name": "r_lo_geo"},
    {"name": "r_hi_geo"},
    {"name": "r_med_photogeo"},
    {"name": "r_lo_photogeo"},
    {"name": "r_hi_photogeo"},
    {"name": "sep_deg"},
]


def test_gaia_distances_posts_form_and_normalizes_rows():
    calls = []

    def fake_post(url, data=None, timeout=None):
        calls.append((url, dict(data or {}), timeout))
        return FakeResponse(
            {
                "metadata": GAIA_METADATA,
                "data": [
                    [
                        5853498713160606720,
                        12.31,
                        -4.51,
                        1.0,
                        0.5,
                        18.2,
                        1.2,
                        -3.4,
                        1000.0,
                        900.0,
                        1100.0,
                        950.0,
                        850.0,
                        1050.0,
                        0.0012345,
                    ]
                ],
            }
        )

    out = DistanceService(base_url="https://gaia.test/tap/sync", timeout=7, http_post=fake_post).gaia_distances(
        12.3, -4.5, radius_arcsec=10, max_rows=5
    )

    assert out["success"] is True
    assert out["count"] == 1
    assert out["rows"][0]["source_id"] == "5853498713160606720"
    assert out["rows"][0]["parallax_mas"] == 1.0
    assert out["rows"][0]["r_geo_pc"] == 1000.0
    assert out["rows"][0]["r_photogeo_lo_pc"] == 850.0
    assert out["rows"][0]["sep_arcsec"] == 4.444
    assert LOW_PARALLAX_WARNING in out["warnings"]
    assert calls[0][0] == "https://gaia.test/tap/sync"
    assert calls[0][1]["REQUEST"] == "doQuery"
    assert calls[0][1]["LANG"] == "ADQL"
    assert calls[0][1]["FORMAT"] == "json"
    assert calls[0][2] == 7
    query = calls[0][1]["QUERY"]
    assert "SELECT TOP 5" in query
    assert "CIRCLE('ICRS', 12.3, -4.5," in query
    assert "0.00277777777778" in query


def test_gaia_distances_clamps_radius_and_max_rows():
    calls = []

    def fake_post(url, data=None, timeout=None):
        calls.append(dict(data or {}))
        return FakeResponse({"metadata": GAIA_METADATA, "data": []})

    out = DistanceService(http_post=fake_post).gaia_distances(10, 20, radius_arcsec=999, max_rows=999)

    assert out["success"] is True
    assert out["count"] == 0
    assert any("radius_arcsec 999" in warning for warning in out["warnings"])
    assert any("max_rows 999" in warning for warning in out["warnings"])
    assert "SELECT TOP 50" in calls[0]["QUERY"]
    assert "0.0833333333333" in calls[0]["QUERY"]


def test_gaia_distances_http_and_parse_failures_return_success_false():
    def http_400(url, data=None, timeout=None):
        return FakeResponse(status_code=400, text="bad adql")

    out = DistanceService(http_post=http_400).gaia_distances(10, 20)
    assert out["success"] is False
    assert "HTTP 400" in out["error"]

    def invalid_json(url, data=None, timeout=None):
        return FakeResponse(json_error=ValueError("no json"))

    out = DistanceService(http_post=invalid_json).gaia_distances(10, 20)
    assert out["success"] is False
    assert "invalid JSON" in out["error"]


def test_gaia_distances_empty_result_is_success():
    def fake_post(url, data=None, timeout=None):
        return FakeResponse({"metadata": GAIA_METADATA, "data": []})

    out = DistanceService(http_post=fake_post).gaia_distances(10, 20)

    assert out["success"] is True
    assert out["rows"] == []
    assert out["count"] == 0


def test_ned_distances_loader_normalizes_awkward_columns_and_summary():
    from astropy.table import MaskedColumn, Table

    table = Table(
        [
            MaskedColumn([28.0, 29.0, 30.0], mask=[False, True, False]),
            MaskedColumn([0.1, 0.2, 0.3], mask=[False, False, True]),
            ["TRGB", "Cepheids", "SNIa"],
            ["1999A", "2001B", "2020C"],
        ],
        names=["Distance Modulus", "Distance Modulus Err", "Method", "Refcode"],
    )

    def loader(name):
        assert name == "M83"
        return table

    out = DistanceService(ned_table_loader=loader).ned_distances("M83")

    assert out["success"] is True
    assert out["count"] == 2
    assert out["rows"][0] == {
        "dist_mpc": 3.981,
        "dist_modulus": 28.0,
        "dist_modulus_err": 0.1,
        "method": "TRGB",
        "refcode": "1999A",
    }
    assert out["rows"][1]["dist_mpc"] == 10.0
    assert out["rows"][1]["dist_modulus_err"] is None
    assert out["summary"] == {"n": 2, "median_mpc": 6.99, "min_mpc": 3.981, "max_mpc": 10.0, "n_methods": 2}
    assert any("Skipped 1" in warning for warning in out["warnings"])


def test_ned_distances_loader_accepts_pandas_ndistance_table_and_modulus_fallback():
    import pandas as pd

    table = pd.DataFrame(
        {
            "NGC 253 NGC 0253": ["NGC 253", "NGC 253"],
            "Distance Modulus (mag)": [27.73, None],
            "Metric Distance (Mpc)": [3.52, 3.6],
        }
    )

    out = DistanceService(ned_table_loader=lambda name: table).ned_distances("NGC 253")

    assert out["success"] is True
    assert out["count"] == 2
    assert out["rows"][0] == {
        "dist_mpc": 3.52,
        "dist_modulus": 27.73,
        "dist_modulus_err": None,
        "method": None,
        "refcode": None,
    }
    assert out["rows"][1]["dist_mpc"] == 3.6
    assert out["rows"][1]["dist_modulus"] is None
    assert out["summary"] == {"n": 2, "median_mpc": 3.56, "min_mpc": 3.52, "max_mpc": 3.6, "n_methods": 0}
    assert out["provenance"]["table"] == "nDistance"
    assert out["provenance"]["endpoint"] == "https://ned.ipac.caltech.edu/cgi-bin/nDistance"

    fallback_table = table.drop(columns=["Metric Distance (Mpc)"])
    fallback = DistanceService(ned_table_loader=lambda name: fallback_table).ned_distances("NGC 253")
    expected_mpc = float(f"{10.0 ** ((27.73 - 25.0) / 5.0):.4g}")

    assert fallback["success"] is True
    assert fallback["count"] == 1
    assert fallback["rows"][0]["dist_mpc"] == expected_mpc
    assert fallback["rows"][0]["dist_modulus"] == 27.73
    assert fallback["summary"] == {
        "n": 1,
        "median_mpc": expected_mpc,
        "min_mpc": expected_mpc,
        "max_mpc": expected_mpc,
        "n_methods": 0,
    }
    assert any("Skipped 1" in warning for warning in fallback["warnings"])


def test_ned_distances_empty_valid_table_is_success():
    from astropy.table import MaskedColumn, Table

    table = Table([MaskedColumn([28.0], mask=[True])], names=["Distance Modulus"])
    out = DistanceService(ned_table_loader=lambda name: table).ned_distances("EmptyGalaxy")

    assert out["success"] is True
    assert out["count"] == 0
    assert out["summary"] == {"n": 0, "median_mpc": None, "min_mpc": None, "max_mpc": None, "n_methods": 0}


def test_velocity_frames_cmb_apex_known_value():
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    apex = SkyCoord(l=264.021 * u.deg, b=48.253 * u.deg, frame="galactic").icrs
    out = DistanceService().velocity_frames(apex.ra.deg, apex.dec.deg, v_helio_kms=1000)

    assert out["success"] is True
    cmb = next(row for row in out["rows"] if row["frame"] == "CMB")
    assert abs(cmb["v_kms"] - 1369.82) < 0.5


def test_velocity_frames_redshift_conversion_and_warning():
    out = DistanceService().velocity_frames(0, 0, z=0.01)

    assert out["success"] is True
    helio = next(row for row in out["rows"] if row["frame"] == "heliocentric")
    assert abs(helio["v_kms"] - 2982.6) < 1
    assert PECULIAR_VELOCITY_WARNING in out["warnings"]


def test_velocity_frames_requires_exactly_one_velocity_input():
    service = DistanceService()

    neither = service.velocity_frames(0, 0)
    both = service.velocity_frames(0, 0, v_helio_kms=1, z=0.1)
    invalid_z = service.velocity_frames(0, 0, z=-1)

    assert neither["success"] is False
    assert "exactly one" in neither["error"]
    assert both["success"] is False
    assert "exactly one" in both["error"]
    assert invalid_z["success"] is False
    assert "greater than -1" in invalid_z["error"]