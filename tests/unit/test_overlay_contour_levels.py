"""Overlay contours at significance levels (UI 2026-09-23 D19): levels spread
between the 30th and 99.5th percentile put most contours in the noise and a
single ALMA pointing rendered as a filled disc."""
from __future__ import annotations

import numpy as np

from services.fits_service import significance_contour_levels


def test_levels_are_multiples_of_a_robust_rms_and_only_below_the_peak():
    rng = np.random.default_rng(1)
    img = rng.normal(0.0, 1.0, (200, 200))
    img[100, 100] = 20.0  # one 20-sigma source
    out = significance_contour_levels(img)
    assert 0.9 < out["rms"] < 1.1
    assert out["sigmas"] == [3.0, 5.0, 8.0, 13.0]
    assert all(abs(lvl - (out["median"] + k * out["rms"])) < 1e-9 for lvl, k in zip(out["positive"], out["sigmas"]))
    assert out["peak_snr"] > 15


def test_pure_noise_has_no_positive_contours_beyond_its_own_fluctuations():
    img = np.random.default_rng(2).normal(0.0, 1.0, (50, 50))
    out = significance_contour_levels(img)
    assert out["positive"] == [] or max(out["sigmas"]) <= 3.0
    nan = significance_contour_levels(np.full((10, 10), np.nan))
    assert nan["rms"] is None and nan["positive"] == []
