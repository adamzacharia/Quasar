"""Tests for the Spectral Line Explorer TOPO->frame correction.

The sign of the Doppler-frame offset is the critical correctness property here
(a flipped sign is worse than no correction), so the sign test validates against
an INDEPENDENT astropy computation and fails loudly if the convention is wrong.
"""

import pytest

pytestmark = pytest.mark.unit

astropy = pytest.importorskip("astropy")

import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from services.spectral_line_explorer import (
    ALMACoverageService,
    SPEED_OF_LIGHT_KMS,
    _resolve_target_frame_name,
    compute_topo_to_frame_offset_kms,
)
import pandas as pd


_ALMA = EarthLocation.from_geodetic(
    lon=-67.7549 * u.deg, lat=-23.0229 * u.deg, height=5058.7 * u.m
)


def test_barycentric_offset_matches_independent_astropy_and_sign():
    ra, dec, mjd = 180.0, 0.0, 58000.0
    v = compute_topo_to_frame_offset_kms(ra, dec, mjd, target_frame="icrs")
    assert v is not None

    # Independent ground truth: astropy's own barycentric radial-velocity
    # correction. Our convention is f_frame = f_topo*(1 + v/c); the barycentric
    # correction has the OPPOSITE sign to our frequency-domain offset.
    target = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    barycorr = target.radial_velocity_correction(
        kind="barycentric", obstime=Time(mjd, format="mjd"), location=_ALMA
    ).to_value(u.km / u.s)

    assert v == pytest.approx(-barycorr, abs=0.05)
    # Sign must be opposite and non-trivial — this is the test that fails if the
    # frequency/velocity sign convention is ever flipped.
    assert v * barycorr < 0
    assert abs(v) > 1.0


def test_lsrk_differs_from_barycentric_within_solar_motion_bound():
    ra, dec, mjd = 150.0, -20.0, 58200.0
    v_icrs = compute_topo_to_frame_offset_kms(ra, dec, mjd, target_frame="icrs")
    v_lsrk = compute_topo_to_frame_offset_kms(ra, dec, mjd, target_frame="lsrk")
    assert v_icrs is not None and v_lsrk is not None
    # LSRK adds the projected standard solar motion (~<= 20 km/s) on top of the
    # barycentric term; it must be finite and differ from barycentric.
    assert v_lsrk != v_icrs
    assert abs(v_lsrk - v_icrs) <= 25.0


def test_fallback_returns_none_and_never_raises():
    assert compute_topo_to_frame_offset_kms(180.0, 0.0, None, "lsrk") is None
    assert compute_topo_to_frame_offset_kms(None, 0.0, 58000.0, "lsrk") is None
    assert compute_topo_to_frame_offset_kms(float("nan"), 0.0, 58000.0, "lsrk") is None
    # An empty/topocentric target frame means "no correction".
    assert compute_topo_to_frame_offset_kms(180.0, 0.0, 58000.0, "") is None


def test_resolve_target_frame_name():
    assert _resolve_target_frame_name("observed (source frame)") == "lsrk"
    assert _resolve_target_frame_name("LSRK") == "lsrk"
    assert _resolve_target_frame_name("barycentric") == "icrs"
    assert _resolve_target_frame_name("topocentric (TOPO)") is None


def _frame_with_window(window: str, mjd: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "proposal_id": "P1",
                "member_ous_uid": "uid://A",
                "frequency_support": window,
                "s_ra": 10.0,
                "s_dec": -20.0,
                "t_exptime": 100,
                "s_resolution": 0.2,
                "t_min": mjd,
                "t_max": mjd,
            }
        ]
    )


def test_correction_flips_edge_case_in_right_direction():
    # A line that sits just OUTSIDE a topocentric SPW edge must become covered
    # once the window is converted into the line's rest frame — and must stay
    # uncovered when no correction is applied. This proves the shift direction.
    ra, dec, mjd = 10.0, -20.0, 58000.0
    v = compute_topo_to_frame_offset_kms(ra, dec, mjd, target_frame="lsrk")
    assert v is not None and abs(v) > 0.5  # non-trivial offset for this geometry

    w_lo, w_hi = 230.000, 230.010  # topocentric SPW window (GHz)
    factor = 1.0 + v / SPEED_OF_LIGHT_KMS
    # Place the line in the sliver that is OUTSIDE the topocentric window but
    # INSIDE the frame-corrected window. Sign-agnostic: when v>0 the window
    # shifts up (use the high edge); when v<0 it shifts down (use the low edge).
    if v > 0:
        line_center = w_hi + 0.4 * (w_hi * factor - w_hi)
    else:
        line_center = w_lo + 0.4 * (w_lo * factor - w_lo)
    line = {
        "line_id": "co21",
        "species": "CO",
        "transition": "2-1",
        "frequency_ghz": 230.538,
        "observed_frequency_ghz": line_center,
    }
    # _classify_rows consumes processed request intervals (with minimum/maximum).
    requested = [
        ALMACoverageService._requested_interval(
            line, tolerance_mhz=0, velocity_width_kms=None
        )
    ]
    frame = _frame_with_window(f"{w_lo}..{w_hi}GHz", mjd)
    service = ALMACoverageService()

    corrected = service._classify_rows(
        frame, requested, ra, dec, target_frame="lsrk"
    )
    assert len(corrected) == 1
    assert corrected[0]["frame_corrected"] is True
    assert corrected[0]["frame_offset_kms"] == pytest.approx(round(v, 4), abs=1e-3)
    assert corrected[0]["matching_lines"][0]["best_classification"] in {"full", "edge"}

    uncorrected = service._classify_rows(
        frame, requested, ra, dec, target_frame=None
    )
    assert uncorrected == []  # line falls outside the un-shifted TOPO window
