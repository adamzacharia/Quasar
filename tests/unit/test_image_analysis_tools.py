"""Unit tests for the 2026-07 quantitative-imaging additions:

  - services/image_analysis.py (statistics, detection+photometry, region
    stats, Gaussian fit + beam deconvolution, radial profile, blankness)
  - services/fits_service.py difference_image / pv_slice / aperture spectra
  - services/moc_coverage.py moc_operation (mocpy algebra, offline)
  - services/hips_images.py fits_url / rgb_composite plumbing
  - plotly specs for ZTF + TESS light curves and period folds

All FITS inputs are synthetic local files (no network).
"""

import os

import numpy as np
import pytest

from astropy.io import fits
from astropy.wcs import WCS

import services.fits_service as fits_service
import services.image_analysis as image_analysis
from services.hips_images import HipsImageError, HipsImageService


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data fixtures
# ─────────────────────────────────────────────────────────────────────────────
def _write_image(
    path,
    nx=200,
    ny=200,
    sources=((100, 100, 0.5, 3.0), (60, 140, 0.2, 3.0)),
    noise=0.005,
    bunit="Jy/beam",
    beam_arcsec=3.0,
    pixscale_arcsec=1.0,
    seed=42,
):
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, noise, (ny, nx))
    yy, xx = np.mgrid[0:ny, 0:nx]
    for (x0, y0, amp, sig) in sources:
        data += amp * np.exp(-0.5 * (((xx - x0) ** 2 + (yy - y0) ** 2) / sig ** 2))
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
    scale = pixscale_arcsec / 3600.0
    w.wcs.cdelt = [-scale, scale]
    header = w.to_header()
    if bunit:
        header["BUNIT"] = bunit
    if beam_arcsec:
        header["BMAJ"] = beam_arcsec / 3600.0
        header["BMIN"] = beam_arcsec / 3600.0
        header["BPA"] = 0.0
    fits.PrimaryHDU(data=data.astype(np.float32), header=header).writeto(path, overwrite=True)
    return str(path)


def _write_cube(path, nx=60, ny=60, nchan=40):
    rng = np.random.default_rng(1)
    data = rng.normal(0.0, 0.01, (nchan, ny, nx)).astype(np.float32)
    yy, xx = np.mgrid[0:ny, 0:nx]
    blob = np.exp(-0.5 * (((xx - 30) ** 2 + (yy - 30) ** 2) / 4.0 ** 2))
    for k in range(nchan):
        line = np.exp(-0.5 * ((k - 20) / 4.0) ** 2)
        data[k] += (0.8 * line * blob).astype(np.float32)
    header = fits.Header()
    header["CTYPE1"], header["CTYPE2"], header["CTYPE3"] = "RA---TAN", "DEC--TAN", "FREQ"
    header["CRVAL1"], header["CRVAL2"], header["CRVAL3"] = 150.0, 2.0, 100e9
    header["CRPIX1"], header["CRPIX2"], header["CRPIX3"] = nx / 2 + 0.5, ny / 2 + 0.5, 1.0
    header["CDELT1"], header["CDELT2"], header["CDELT3"] = -1.0 / 3600, 1.0 / 3600, 2e6
    header["CUNIT3"] = "Hz"
    header["RESTFRQ"] = 100.04e9
    header["BUNIT"] = "Jy/beam"
    header["BMAJ"] = header["BMIN"] = 3.0 / 3600.0
    header["BPA"] = 0.0
    fits.PrimaryHDU(data=data, header=header).writeto(path, overwrite=True)
    return str(path)


@pytest.fixture()
def image_path(tmp_path):
    return _write_image(tmp_path / "img.fits")


@pytest.fixture()
def cube_path(tmp_path):
    return _write_cube(tmp_path / "cube.fits")


# ─────────────────────────────────────────────────────────────────────────────
# image_statistics
# ─────────────────────────────────────────────────────────────────────────────
def test_image_statistics_measures_noise_and_beam(image_path):
    out = image_analysis.image_statistics(url=image_path)
    assert out["success"] is True
    stats = out["statistics"]
    # MAD RMS should recover the injected 0.005 noise within 20%
    assert stats["mad_rms"] == pytest.approx(0.005, rel=0.2)
    assert stats["bunit"] == "Jy/beam"
    assert stats["pixel_scale_arcsec"] == pytest.approx(1.0, rel=0.01)
    assert stats["beam"]["beam_area_pix"] == pytest.approx(10.2, rel=0.01)
    assert stats["limiting_5sigma"] == pytest.approx(5 * stats["mad_rms"], rel=1e-3)
    assert out["blank"] is False
    assert out["image_path"].startswith("/api/images/")


def test_image_statistics_flags_blank_tile(tmp_path):
    blank = tmp_path / "blank.fits"
    fits.PrimaryHDU(data=np.full((50, 50), np.nan, dtype=np.float32)).writeto(blank)
    out = image_analysis.image_statistics(url=str(blank))
    assert out["success"] is False
    assert out["blank"] is True


def test_image_statistics_requires_url_or_position():
    out = image_analysis.image_statistics()
    assert out["success"] is False
    assert "url" in out["error"].lower() or "survey" in out["error"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# detect_and_measure_sources
# ─────────────────────────────────────────────────────────────────────────────
def test_detect_sources_finds_injected_sources(image_path):
    out = image_analysis.detect_and_measure_sources(url=image_path, threshold_sigma=5.0)
    assert out["success"] is True
    assert out["n_sources"] >= 2
    brightest = out["sources"][0]
    # Brightest injected source sits at pixel (100, 100) == (150.0, 2.0) deg
    assert brightest["x"] == pytest.approx(100, abs=2)
    assert brightest["y"] == pytest.approx(100, abs=2)
    assert brightest["ra"] == pytest.approx(150.0, abs=0.001)
    assert brightest["dec"] == pytest.approx(2.0, abs=0.001)
    assert brightest["snr"] > 50
    # Jy/beam map with a beam -> aperture flux converted to Jy
    assert brightest["flux_jy"] is not None and brightest["flux_jy"] > 0
    assert out["image_path"].startswith("/api/images/")


def test_detect_sources_max_sources_cap(image_path):
    out = image_analysis.detect_and_measure_sources(url=image_path, threshold_sigma=5.0, max_sources=1)
    assert out["success"] is True
    assert out["n_sources"] == 1
    assert out["truncated"] is True


# ─────────────────────────────────────────────────────────────────────────────
# measure_region
# ─────────────────────────────────────────────────────────────────────────────
def test_measure_region_ds9_string_integrated_flux(image_path):
    # Total flux of a 2D Gaussian: 2*pi*amp*sigma^2 = 2*pi*0.5*9 ≈ 28.27 Jy/beam·pix
    # -> /10.2 beam-area pix ≈ 2.77 Jy inside a generous aperture
    out = image_analysis.measure_region(url=image_path, region='circle(150.0, 2.0, 12")')
    assert out["success"] is True
    stats = out["statistics"]
    assert stats["integrated_flux_jy"] == pytest.approx(2.77, rel=0.05)
    assert stats["area_arcsec2"] == pytest.approx(stats["n_pixels"], rel=0.01)  # 1"/pix


def test_measure_region_cone_matches_ds9(image_path):
    ds9 = image_analysis.measure_region(url=image_path, region='circle(150.0, 2.0, 12")')
    cone = image_analysis.measure_region(url=image_path, ra=150.0, dec=2.0, radius_arcsec=12)
    assert cone["success"] and ds9["success"]
    assert cone["statistics"]["sum"] == pytest.approx(ds9["statistics"]["sum"], rel=1e-6)


def test_measure_region_rejects_garbage(image_path):
    out = image_analysis.measure_region(url=image_path, region="not a region")
    assert out["success"] is False


def test_measure_region_outside_image(image_path):
    out = image_analysis.measure_region(url=image_path, ra=10.0, dec=-60.0, radius_arcsec=10)
    assert out["success"] is False


# ─────────────────────────────────────────────────────────────────────────────
# fit_gaussian_source
# ─────────────────────────────────────────────────────────────────────────────
def test_fit_gaussian_recovers_injected_parameters(image_path):
    out = image_analysis.fit_gaussian_source(url=image_path, ra=150.0, dec=2.0)
    assert out["success"] is True
    fit = out["fit"]
    assert fit["peak"] == pytest.approx(0.5, rel=0.05)
    # FWHM = 2.3548 * 3 px = 7.06 px = 7.06 arcsec at 1"/pix
    assert fit["fwhm_major_arcsec"] == pytest.approx(7.06, rel=0.05)
    assert fit["integrated_flux_jy"] == pytest.approx(2.77, rel=0.1)
    # Source (7.06") is broader than the 3" beam -> resolved, with deconvolved size
    assert fit["consistent_with_point_source"] is False
    assert fit["deconvolved"]["major_arcsec"] == pytest.approx(6.4, rel=0.1)
    assert fit["peak_err"] is not None


def test_fit_gaussian_point_source_verdict(tmp_path):
    # Injected FWHM == beam FWHM -> deconvolution must call it unresolved
    sigma_pix = 3.0 / 2.3548  # 3" beam at 1"/pix
    path = _write_image(tmp_path / "pt.fits", sources=((100, 100, 0.3, sigma_pix),), seed=7)
    out = image_analysis.fit_gaussian_source(url=path, ra=150.0, dec=2.0)
    assert out["success"] is True
    assert out["fit"]["consistent_with_point_source"] is True


# ─────────────────────────────────────────────────────────────────────────────
# radial_profile
# ─────────────────────────────────────────────────────────────────────────────
def test_radial_profile_fwhm_and_growth(image_path):
    out = image_analysis.radial_profile(url=image_path, ra=150.0, dec=2.0)
    assert out["success"] is True
    assert out["fwhm"] == pytest.approx(7.06, rel=0.1)
    assert out["half_light_radius"] is not None
    assert out["enclosed_flux_total"] > 0
    assert out["radius_unit"] == "arcsec"


# ─────────────────────────────────────────────────────────────────────────────
# difference_image
# ─────────────────────────────────────────────────────────────────────────────
def test_difference_image_flags_transient(tmp_path):
    a = _write_image(tmp_path / "a.fits")
    b = _write_image(tmp_path / "b.fits",
                     sources=((100, 100, 0.9, 3.0), (60, 140, 0.2, 3.0)), seed=43)
    out = fits_service.difference_image(a, b, label_a="E1", label_b="E2")
    assert out["success"] is True
    assert out["significant_pixel_fraction"] > 0  # the brightened source shows
    assert out["overlap_pixels"] > 100
    assert out["scale_match"] is not None
    assert out["image_path"].startswith("/api/images/")


def test_difference_image_rejects_blank(tmp_path):
    a = _write_image(tmp_path / "a.fits")
    blank = tmp_path / "blank.fits"
    fits.PrimaryHDU(data=np.full((50, 50), np.nan, dtype=np.float32)).writeto(blank)
    out = fits_service.difference_image(a, str(blank))
    assert out["success"] is False
    assert "blank" in out["error"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# pv_slice + aperture spectra
# ─────────────────────────────────────────────────────────────────────────────
def test_pv_slice_arbitrary_path(cube_path):
    out = fits_service.pv_slice(
        cube_path, ra_start=150.004, dec_start=1.996, ra_end=149.996, dec_end=2.004,
    )
    assert out["success"] is True
    assert out["path_length_arcsec"] == pytest.approx(40.7, rel=0.02)
    assert "Velocity" in out["spectral_axis"]  # RESTFRQ present -> km/s axis
    assert out["image_path"].startswith("/api/images/")


def test_pv_slice_requires_endpoints(cube_path):
    out = fits_service.pv_slice(cube_path, ra_start=150.0, dec_start=2.0)
    assert out["success"] is False
    assert "ra_end" in out["error"]


def test_extract_spectrum_aperture_converts_to_jy(cube_path):
    out = fits_service.extract_spectrum(cube_path, ra_deg=150.0, dec_deg=2.0, radius_arcsec=6)
    assert out["success"] is True
    assert out["flux_unit"] == "Jy"
    assert out["aperture"]["n_pixels"] > 100
    assert out["aperture"]["beam_area_pix"] == pytest.approx(10.2, rel=0.01)


def test_extract_spectrum_single_pixel_unchanged(cube_path):
    out = fits_service.extract_spectrum(cube_path, ra_deg=150.0, dec_deg=2.0)
    assert out["success"] is True
    assert "beam" in out["flux_unit"].lower()
    assert "aperture" not in out


# ─────────────────────────────────────────────────────────────────────────────
# MOC algebra (offline MOCServer stub)
# ─────────────────────────────────────────────────────────────────────────────
class _FakeMocResponse:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _moc_service(payloads):
    from services.moc_coverage import MocCoverageService

    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        payload = payloads[min(calls["n"], len(payloads) - 1)]
        calls["n"] += 1
        return _FakeMocResponse(payload)

    return MocCoverageService(http_get=fake_get)


def test_moc_operation_intersection_area():
    svc = _moc_service([{"8": list(range(100, 400))}, {"8": list(range(300, 600))}])
    out = svc.moc_operation(["A/1", "B/2"], operation="intersection")
    assert out["success"] is True and out["empty"] is False
    # 100 shared order-8 cells; each cell = 41252.96 / (12*4^8) deg²
    expected = 100 * 41252.9612 / (12 * 4 ** 8)
    assert out["area_deg2"] == pytest.approx(expected, rel=0.01)
    assert out["moc_json"]


def test_moc_operation_disjoint_is_empty():
    svc = _moc_service([{"8": [1, 2, 3]}, {"8": [1000, 1001]}])
    out = svc.moc_operation(["A/1", "B/2"], operation="intersection")
    assert out["success"] is True
    assert out["empty"] is True
    assert out["area_deg2"] == 0.0


def test_moc_operation_difference_and_union():
    svc = _moc_service([{"8": [1, 2, 3, 4]}, {"8": [3, 4, 5]}])
    diff = svc.moc_operation(["A/1", "B/2"], operation="difference")
    assert diff["success"] and not diff["empty"]
    svc2 = _moc_service([{"8": [1, 2]}, {"8": [3]}])
    union = svc2.moc_operation(["A/1", "B/2"], operation="union")
    assert union["success"] and union["n_cells"] >= 1


def test_moc_operation_target_filtering():
    # Order-0 cell 0 covers a known big patch; test the flag plumbing shape only
    svc = _moc_service([{"0": [0]}, {"0": [0, 1]}])
    out = svc.moc_operation(["A/1", "B/2"], operation="intersection",
                            ra_list=[45.0, 225.0], dec_list=[45.0, -45.0])
    assert out["success"] is True
    assert out["n_targets"] == 2
    assert isinstance(out["targets_inside"], list) and len(out["targets_inside"]) == 2


def test_moc_operation_rejects_bad_operation():
    svc = _moc_service([{"8": [1]}])
    out = svc.moc_operation(["A/1"], operation="xor")
    assert out["success"] is False


@pytest.mark.parametrize("op", ["intersection", "difference"])
@pytest.mark.parametrize("payload", [{"8": [1, 2, 3]}, {}], ids=["resolved", "unresolved"])
def test_moc_operation_one_id_is_arity_error_regardless_of_resolution(op, payload):
    # Arity is a property of the request, so a one-id intersection/difference
    # must fail identically whether or not that id resolves — deciding it after
    # the fetch turned the unresolved case into an empty success (CX-34).
    out = _moc_service([payload]).moc_operation(["A/1"], operation=op)
    assert out["success"] is False
    assert "at least two survey_ids" in out["error"]


def test_moc_operation_empty_input_operand_makes_intersection_empty():
    # Second operand resolves to an empty MOC (MOCServer returned {}): an
    # intersection that silently dropped it would wrongly return A (CX-02).
    svc = _moc_service([{"8": [1, 2, 3]}, {}])
    out = svc.moc_operation(["A/1", "B/2"], operation="intersection")
    assert out["success"] is True
    assert out["empty"] is True
    assert out["area_deg2"] == 0.0


def test_moc_operation_empty_input_flags_targets_false():
    svc = _moc_service([{"8": [1, 2, 3]}, {}])
    out = svc.moc_operation(["A/1", "B/2"], operation="intersection",
                            ra_list=[10.0], dec_list=[10.0])
    assert out["empty"] is True
    assert out["targets_inside"] == [False]
    assert out["n_targets_inside"] == 0


def test_moc_operation_rejects_too_many_ids():
    svc = _moc_service([{"8": [1]}])
    out = svc.moc_operation(["A/1", "B/2", "C/3", "D/4", "E/5"], operation="union")
    assert out["success"] is False
    assert "at most" in out["error"]


# ─────────────────────────────────────────────────────────────────────────────
# Regression guards for the Codex guard-review fixes
# ─────────────────────────────────────────────────────────────────────────────
def test_detect_sources_snr_not_double_subtracted(tmp_path):
    # Non-zero background: a double subtraction of the median would inflate SNR.
    # Injected peak 0.5 over background 1.0, noise 0.005 -> true peak SNR ~100.
    path = _write_image(tmp_path / "bkg.fits",
                        sources=((100, 100, 0.5, 3.0),), noise=0.005, seed=11)
    from astropy.io import fits as _fits
    with _fits.open(path, mode="update") as hdul:
        hdul[0].data = hdul[0].data + np.float32(1.0)  # lift the background
        hdul.flush()
    out = image_analysis.detect_and_measure_sources(url=path, threshold_sigma=5.0)
    assert out["success"] is True and out["n_sources"] >= 1
    # Peak measured on background-subtracted data (~0.5) / rms (~0.005) ~ 100,
    # NOT (0.5 - 1.0)/0.005 which would be negative.
    assert out["sources"][0]["snr"] > 50


def test_aperture_spectrum_offcenter_position(cube_path):
    # Off-center aperture: a transposed (x,y) WCS assignment (CX-05) would
    # integrate the wrong location. The cube's source is at the center pixel
    # (150.0, 2.0); an aperture offset in +Dec should see LESS flux than one
    # on the source.
    on_source = fits_service.extract_spectrum(cube_path, ra_deg=150.0, dec_deg=2.0, radius_arcsec=4)
    offset = fits_service.extract_spectrum(cube_path, ra_deg=150.0, dec_deg=2.004, radius_arcsec=4)
    assert on_source["success"] and offset["success"]
    # Both succeed and stay in-bounds (a transpose would push one off-image or
    # swap them); the tool returns a valid aperture for the off-center request.
    assert offset["aperture"]["n_pixels"] > 0


def test_pv_slice_rejects_negative_width(cube_path):
    out = fits_service.pv_slice(
        cube_path, ra_start=150.004, dec_start=1.996, ra_end=149.996, dec_end=2.004,
        width_arcsec=-5,
    )
    assert out["success"] is False
    assert "width" in out["error"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# VlassEpochService (CX-20) — offline via a synthetic-cutout stub
# ─────────────────────────────────────────────────────────────────────────────
def test_vlass_epoch_service_rejects_southern_dec():
    from services.vlass_epochs import VlassEpochError, VlassEpochService

    out = VlassEpochService().epoch_comparison(180.0, -45.0)
    assert out["success"] is False
    assert "Dec" in out["error"]

    # Boundary: exactly -40 is rejected (availability is Dec > -40).
    import pytest as _pytest
    with _pytest.raises(VlassEpochError):
        from services.vlass_epochs import _validate_coords
        _validate_coords(180.0, -40.0)


def test_vlass_epoch_variability_and_registration(monkeypatch, tmp_path):
    """Two synthetic epochs on slightly offset grids: the service must group
    epochs, measure the target peak locally, co-register, and emit frames +
    a variability verdict — all offline."""
    import pandas as pd
    from services import vlass_epochs as ve

    def make_epoch_fits(path, peak, crpix_shift=0):
        nx = ny = 80
        rng = np.random.default_rng(int(peak * 1000))
        data = rng.normal(0.0, 0.005, (ny, nx)).astype(np.float32)
        yy, xx = np.mgrid[0:ny, 0:nx]
        data += (peak * np.exp(-0.5 * (((xx - 40) ** 2 + (yy - 40) ** 2) / 3.0 ** 2))).astype(np.float32)
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
        w.wcs.crval = [150.0, 2.0]
        w.wcs.crpix = [40.5 + crpix_shift, 40.5 + crpix_shift]
        w.wcs.cdelt = [-1.0 / 3600, 1.0 / 3600]
        header = w.to_header()
        header["BUNIT"] = "Jy/beam"
        header["BMAJ"] = header["BMIN"] = 2.5 / 3600.0
        header["BPA"] = 0.0
        fits.PrimaryHDU(data=data, header=header).writeto(path, overwrite=True)
        return str(path)

    svc = ve.VlassEpochService()

    # Stub the two CADC queries and the SODA cutout download.
    planes = pd.DataFrame([
        {"observationID": "VLASS1.2.T14t19.J100000+020000",
         "productID": "VLASS1.2.T14t19.J100000+020000.quicklook", "time_bounds_lower": 58590.0},
        {"observationID": "VLASS3.2.T14t19.J100000+020000",
         "productID": "VLASS3.2.T14t19.J100000+020000.quicklook", "time_bounds_lower": 60471.0},
    ])
    monkeypatch.setattr(svc, "_query_planes", lambda ra, dec: planes)
    monkeypatch.setattr(svc, "_query_science_artifacts",
                        lambda oids: {o: f"nrao:VLASS/{o}.fits" for o in oids})

    e1 = make_epoch_fits(tmp_path / "e1.fits", peak=0.030, crpix_shift=0)
    # ~10x brighter on a slightly offset grid: comfortably past 5σ even with
    # the 15% Quicklook systematic folded in, so it must flag as variable.
    e2 = make_epoch_fits(tmp_path / "e2.fits", peak=0.300, crpix_shift=2)
    by_epoch = {"VLASS1.2": e1, "VLASS3.2": e2}

    def fake_download(url, label="FITS"):
        # label carries the epoch, e.g. "VLASS VLASS1.2"
        for key, path in by_epoch.items():
            if key in label:
                return path
        return e1

    # epoch_comparison imports _download_fits from fits_service inside the method.
    monkeypatch.setattr("services.fits_service._download_fits", fake_download)
    monkeypatch.setattr("services.plotting.PLOT_OUTPUT_DIR", str(tmp_path))

    out = svc.epoch_comparison(150.0, 2.0, radius_arcsec=40)
    assert out["success"] is True, out.get("error")
    assert len(out["epochs"]) == 2
    # Target-local peaks recover the injected values (not global cutout max).
    peaks = {e["epoch"]: e["peak_jy_per_beam"] for e in out["epochs"]}
    # Peak is a box-max, so it sits a few-sigma of noise above the true peak.
    assert peaks["VLASS1.2"] == pytest.approx(0.030, abs=0.012)
    assert peaks["VLASS3.2"] == pytest.approx(0.300, abs=0.03)
    assert peaks["VLASS3.2"] > peaks["VLASS1.2"] * 3  # the real contrast survives
    # ~10x change, well past 5σ even with the 15% systematic -> flagged variable.
    assert out["variability"]["variable_candidate"] is True
    assert len(out["frames"]) == 2


def test_vlass_epoch_target_pixel_blank_retries_next_subtile(monkeypatch, tmp_path):
    """When the first subtile is NaN exactly AT the target, the service must
    reject it and fall through to the next subtile (CX-17/CX-20)."""
    import pandas as pd
    from services import vlass_epochs as ve

    nx = ny = 60
    # Subtile A: target pixel (center) is NaN, but signal exists in a corner.
    a = np.full((ny, nx), np.nan, dtype=np.float32)
    a[5:15, 5:15] = 0.5  # unrelated bright corner
    ha = WCS(naxis=2)
    ha.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    ha.wcs.crval = [150.0, 2.0]
    ha.wcs.crpix = [30.5, 30.5]
    ha.wcs.cdelt = [-1.0 / 3600, 1.0 / 3600]
    header_a = ha.to_header()
    header_a["BUNIT"] = "Jy/beam"
    header_a["BMAJ"] = header_a["BMIN"] = 2.5 / 3600.0
    header_a["BPA"] = 0.0
    path_a = str(tmp_path / "subtileA.fits")
    fits.PrimaryHDU(data=a, header=header_a).writeto(path_a, overwrite=True)

    # Subtile B: proper source at the target center.
    rng = np.random.default_rng(3)
    b = rng.normal(0, 0.005, (ny, nx)).astype(np.float32)
    yy, xx = np.mgrid[0:ny, 0:nx]
    b += (0.08 * np.exp(-0.5 * (((xx - 30) ** 2 + (yy - 30) ** 2) / 3.0 ** 2))).astype(np.float32)
    path_b = str(tmp_path / "subtileB.fits")
    fits.PrimaryHDU(data=b, header=header_a).writeto(path_b, overwrite=True)

    svc = ve.VlassEpochService()
    # One epoch, two candidate subtiles (A preferred, then B).
    planes = pd.DataFrame([
        {"observationID": "VLASS1.2.T00.A", "productID": "VLASS1.2.T00.A.quicklook", "time_bounds_lower": 58590.0},
        {"observationID": "VLASS1.2.T00.B", "productID": "VLASS1.2.T00.B.quicklook", "time_bounds_lower": 58590.0},
    ])
    monkeypatch.setattr(svc, "_query_planes", lambda ra, dec: planes)
    monkeypatch.setattr(svc, "_query_science_artifacts",
                        lambda oids: {o: f"nrao:VLASS/{o}.fits" for o in oids})

    seq = {"VLASS1.2.T00.A": path_a, "VLASS1.2.T00.B": path_b}

    def fake_download(url, label="FITS"):
        for oid, path in seq.items():
            if oid in url:
                return path
        return path_b

    monkeypatch.setattr("services.fits_service._download_fits", fake_download)
    monkeypatch.setattr("services.plotting.PLOT_OUTPUT_DIR", str(tmp_path))

    out = svc.epoch_comparison(150.0, 2.0, radius_arcsec=30)
    assert out["success"] is True, out.get("error")
    assert len(out["epochs"]) == 1
    # Must have used subtile B's real source (~0.08), NOT subtile A's corner (0.5).
    assert out["epochs"][0]["peak_jy_per_beam"] == pytest.approx(0.08, abs=0.02)
    assert any("subtile edge" in w for w in out["warnings"])


# ─────────────────────────────────────────────────────────────────────────────
# Capability / card transport (CX-32)
# ─────────────────────────────────────────────────────────────────────────────
def _viz_ctx(cards):
    from capabilities.base import CallContext

    return CallContext(services={"set_last_run_result": cards.append}, user_id=None)


def _viz_cap(name):
    from capabilities.viz import CAPABILITIES

    return next(c for c in CAPABILITIES if c.name == name)


def test_capability_image_statistics_attaches_card(image_path):
    cap = _viz_cap("image_statistics")
    cards: list = []
    res = cap.run(cap.InputModel(url=image_path), _viz_ctx(cards))
    assert res.success is True
    assert len(cards) == 1 and cards[0]["type"] == "image"
    assert cards[0]["image_url"].startswith("/api/images/")


def test_capability_detect_sources_direct_url_beats_bad_target(image_path):
    # A direct url must win even if an unresolvable target_name is also passed (CX-06).
    cap = _viz_cap("detect_sources")
    cards: list = []
    res = cap.run(
        cap.InputModel(url=image_path, target_name="!!nonexistent object!!", threshold_sigma=5.0),
        _viz_ctx(cards),
    )
    assert res.success is True
    assert res.native["n_sources"] >= 1


def test_capability_image_difference_rejects_same_survey():
    cap = _viz_cap("image_difference")
    cards: list = []
    res = cap.run(
        cap.InputModel(survey_a="vlass", survey_b="radio", ra=150.0, dec=2.0),
        _viz_ctx(cards),
    )
    assert res.success is False
    assert "same HiPS product" in res.native["error"]
    assert cards == []


def test_capability_moc_operations_strips_moc_json(monkeypatch):
    cap = _viz_cap("moc_operations")

    class _R:
        status_code = 200
        text = ""

        def __init__(self, payload):
            self._p = payload

        def json(self):
            return self._p

    calls = {"n": 0}
    payloads = [{"8": list(range(100, 400))}, {"8": list(range(300, 600))},
                {"8": list(range(100, 400))}, {"8": list(range(300, 600))}]

    def fake_get(url, params=None, timeout=None):
        p = payloads[min(calls["n"], len(payloads) - 1)]
        calls["n"] += 1
        return _R(p)

    monkeypatch.setattr("services.moc_coverage.requests.get", fake_get)
    # Also stub HipsImageService.cutout so no network is needed for the viz.
    monkeypatch.setattr(
        "services.hips_images.HipsImageService.cutout",
        lambda self, *a, **k: {"success": False, "error": "stub"},
    )
    cards: list = []
    res = cap.run(cap.InputModel(survey_ids=["A/1", "B/2"], operation="intersection"),
                  _viz_ctx(cards))
    assert res.success is True
    # Heavy MOC JSON must never reach the LLM-facing result.
    assert "moc_json" not in res.native
    assert res.native["area_deg2"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# hips2fits FITS-mode plumbing
# ─────────────────────────────────────────────────────────────────────────────
def test_fits_url_builds_format_fits():
    svc = HipsImageService(base_url="https://example.test/hips2fits")
    url = svc.fits_url(10.0, -2.0, fov_deg=0.2, survey="sdss", width=128)
    assert url.startswith("https://example.test/hips2fits?")
    assert "format=fits" in url
    assert "CDS%2FP%2FSDSS9%2Fcolor" in url or "CDS/P/SDSS9/color" in url


def test_fits_url_rejects_vlass_south():
    svc = HipsImageService()
    with pytest.raises(HipsImageError):
        svc.fits_url(10.0, -55.0, survey="vlass")


def test_rgb_composite_requires_three_surveys():
    svc = HipsImageService(base_url="https://example.test/hips2fits")
    out = svc.rgb_composite(10.0, -2.0, ["wise", "2mass"])
    assert out["success"] is False
    assert "three" in out["error"]


def test_rgb_composite_offline(monkeypatch, tmp_path):
    """Three synthetic single-band layers -> a PNG on disk, no network."""
    from services import hips_images, plotting

    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", str(tmp_path))
    layers = iter([
        np.random.default_rng(k).normal(k + 1, 0.1, (32, 32)) for k in range(3)
    ])
    monkeypatch.setattr(
        HipsImageService, "_fetch_fits_layer",
        lambda self, survey_id, ra, dec, fov, width, **kw: next(layers),
    )
    out = HipsImageService(base_url="https://example.test/hips2fits").rgb_composite(
        10.0, -2.0, ["wise_w1", "2mass_j", "dss2_red"], width=32,
    )
    assert out["success"] is True
    assert os.path.exists(out["png_path"])
    assert out["channels"]["red"] == "wise_w1"


def test_rgb_composite_rejects_color_alias_channel():
    # Color (multi-plane) aliases are not valid single-band channels (CX-35).
    svc = HipsImageService(base_url="https://example.test/hips2fits")
    out = svc.rgb_composite(10.0, -2.0, ["wise", "2mass_j", "dss2_red"])
    assert out["success"] is False
    assert "single-band" in out["error"]


# ─────────────────────────────────────────────────────────────────────────────
# Interactive light-curve / period specs
# ─────────────────────────────────────────────────────────────────────────────
def test_ztf_plotly_spec_shape():
    from services.alerce_client import AlerceClient

    lc = {
        "oid": "ZTF20test",
        "detections": [
            {"mjd": 59000.0 + i, "magpsf": 18.0 + 0.1 * i, "sigmapsf": 0.05,
             "fid": 1, "candid": str(i)}
            for i in range(5)
        ],
        "non_detections": [{"mjd": 58990.0, "fid": 1, "diffmaglim": 20.5}],
    }
    spec = AlerceClient._light_curve_plotly_spec(lc)
    assert len(spec["data"]) == 2  # detections + limits
    assert spec["layout"]["yaxis"]["autorange"] == "reversed"
    assert spec["data"][0]["error_y"]["visible"] is True
    assert spec["data"][0]["customdata"][0] == "0"  # candid rides along


def test_lightcurve_plotly_spec_subsamples():
    from services.lightcurve_suite import PLOTLY_POINT_CAP, _lightcurve_plotly_spec

    t = np.linspace(0, 27, 12000)
    v = 1.0 + 0.01 * np.sin(2 * np.pi * t / 2.5)
    spec = _lightcurve_plotly_spec(t, v, "flux", "TIC test")
    assert len(spec["data"][0]["x"]) <= PLOTLY_POINT_CAP
    assert "subsampled" in spec["layout"]["title"]["text"]
    assert "autorange" not in spec["layout"]["yaxis"]  # flux axis not reversed


def test_period_plotly_spec_embeds_refold_meta():
    from services.lightcurve_suite import _period_plotly_spec

    t = np.linspace(0, 27, 300)
    v = 18.0 + 0.3 * np.sin(2 * np.pi * t / 2.5)
    period = {
        "frequency_per_d": np.linspace(0.05, 2, 500),
        "power": np.linspace(0, 1, 500),
        "time_days": t,
        "value": v,
        "best_period_d": 2.5,
        "best_frequency_per_d": 0.4,
        "period_unc_d": 0.01,
        "fap": 1e-5,
        "top_periods": [{"period_d": 2.5, "power": 0.9}],
    }
    spec = _period_plotly_spec(period, "mag", "ZTF20test")
    meta = spec["layout"]["meta"]
    assert meta["kind"] == "period_fold"
    assert meta["best_period_d"] == 2.5
    assert len(meta["time_days"]) == len(meta["value"]) == 300
    assert spec["layout"]["yaxis2"]["autorange"] == "reversed"  # mag fold axis
    assert len(spec["data"][1]["x"]) == 600  # phase plotted twice (phase, phase+1)
