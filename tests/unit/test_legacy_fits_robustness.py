"""R7 — legacy FITS convention robustness (ReplicationBench failure mode #3:
technical execution failures on legacy astronomy formats).

Old archives serve FITS shapes modern cutout services no longer emit:
AIPS/CASA 4-axis degenerate cubes, CD-matrix WCS (no CDELT), FK4/B1950
equinoxes, BSCALE/BZERO integer-scaled data, and headers without BUNIT.
These tests pin that the image-analysis tools handle each without crashing
and with correct scaling/geometry.
"""

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import services.image_analysis as image_analysis


def _base_data(nx=120, ny=120, amp=0.5, sigma=3.0, noise=0.005, seed=11):
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, noise, (ny, nx))
    yy, xx = np.mgrid[0:ny, 0:nx]
    data += amp * np.exp(-0.5 * (((xx - 60) ** 2 + (yy - 60) ** 2) / sigma ** 2))
    return data


def _tan_header(nx=120, ny=120, scale_arcsec=1.0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
    s = scale_arcsec / 3600.0
    w.wcs.cdelt = [-s, s]
    return w.to_header()


def test_casa_style_4axis_degenerate_cube(tmp_path):
    # Classic radio FITS: NAXIS=4 with degenerate FREQ and STOKES axes.
    data = _base_data().astype(np.float32)[np.newaxis, np.newaxis, :, :]
    header = fits.Header()
    header["CTYPE1"], header["CTYPE2"] = "RA---SIN", "DEC--SIN"
    header["CTYPE3"], header["CTYPE4"] = "FREQ", "STOKES"
    header["CRVAL1"], header["CRVAL2"] = 150.0, 2.0
    header["CRVAL3"], header["CRVAL4"] = 230e9, 1.0
    header["CRPIX1"], header["CRPIX2"] = 60.5, 60.5
    header["CRPIX3"], header["CRPIX4"] = 1.0, 1.0
    header["CDELT1"], header["CDELT2"] = -1.0 / 3600, 1.0 / 3600
    header["CDELT3"], header["CDELT4"] = 2e6, 1.0
    header["BUNIT"] = "JY/BEAM"          # AIPS-style uppercase
    header["BMAJ"] = header["BMIN"] = 3.0 / 3600.0
    header["BPA"] = 0.0
    path = tmp_path / "casa4axis.fits"
    fits.PrimaryHDU(data=data, header=header).writeto(path)

    out = image_analysis.image_statistics(url=str(path))
    assert out["success"] is True
    assert out["statistics"]["mad_rms"] == pytest.approx(0.005, rel=0.3)
    # Uppercase JY/BEAM must still be recognized for the beam conversion.
    assert out["statistics"]["beam"] is not None


def test_cd_matrix_wcs_without_cdelt(tmp_path):
    # Pre-CDELT convention: the WCS is a CD matrix (with rotation here).
    data = _base_data().astype(np.float32)
    header = fits.Header()
    header["CTYPE1"], header["CTYPE2"] = "RA---TAN", "DEC--TAN"
    header["CRVAL1"], header["CRVAL2"] = 150.0, 2.0
    header["CRPIX1"], header["CRPIX2"] = 60.5, 60.5
    theta = np.deg2rad(30.0)
    s = 1.0 / 3600.0
    header["CD1_1"] = -s * np.cos(theta)
    header["CD1_2"] = s * np.sin(theta)
    header["CD2_1"] = s * np.sin(theta)
    header["CD2_2"] = s * np.cos(theta)
    header["BUNIT"] = "Jy/beam"
    path = tmp_path / "cdmatrix.fits"
    fits.PrimaryHDU(data=data, header=header).writeto(path)

    out = image_analysis.image_statistics(url=str(path))
    assert out["success"] is True
    # The rotated CD matrix still yields a ~1 arcsec pixel scale.
    assert out["statistics"]["pixel_scale_arcsec"] == pytest.approx(1.0, rel=0.05)


def test_fk4_b1950_equinox_header(tmp_path):
    data = _base_data().astype(np.float32)
    header = _tan_header()
    header["EQUINOX"] = 1950.0
    header["RADESYS"] = "FK4"
    header["EPOCH"] = 1950.0              # deprecated legacy keyword
    header["BUNIT"] = "Jy/beam"
    path = tmp_path / "b1950.fits"
    fits.PrimaryHDU(data=data, header=header).writeto(path)

    out = image_analysis.image_statistics(url=str(path))
    assert out["success"] is True
    assert out["statistics"]["pixel_scale_arcsec"] == pytest.approx(1.0, rel=0.01)


def test_int16_bscale_bzero_scaling(tmp_path):
    # Old archives store int16 with BSCALE/BZERO; astropy must rescale and
    # the statistics must come out in PHYSICAL units, not raw counts.
    physical = _base_data(amp=0.5, noise=0.005)
    bscale, bzero = 1e-4, 0.0
    raw = np.round((physical - bzero) / bscale).astype(np.int16)
    header = _tan_header()
    header["BUNIT"] = "Jy/beam"
    hdu = fits.PrimaryHDU(data=raw, header=header)
    hdu.header["BSCALE"] = bscale
    hdu.header["BZERO"] = bzero
    path = tmp_path / "int16.fits"
    hdu.writeto(path)

    out = image_analysis.image_statistics(url=str(path))
    assert out["success"] is True
    stats = out["statistics"]
    # Physical peak ~0.5 Jy/beam (not the raw ~5000 counts).
    assert stats["max"] == pytest.approx(0.5, rel=0.15)
    assert stats["mad_rms"] == pytest.approx(0.005, rel=0.35)


def test_missing_bunit_is_tolerated(tmp_path):
    data = _base_data().astype(np.float32)
    header = _tan_header()
    path = tmp_path / "nobunit.fits"
    fits.PrimaryHDU(data=data, header=header).writeto(path)

    out = image_analysis.image_statistics(url=str(path))
    assert out["success"] is True
    assert out["statistics"]["bunit"] is None


def test_measure_region_on_legacy_cd_matrix_image(tmp_path):
    data = _base_data().astype(np.float32)
    header = fits.Header()
    header["CTYPE1"], header["CTYPE2"] = "RA---TAN", "DEC--TAN"
    header["CRVAL1"], header["CRVAL2"] = 150.0, 2.0
    header["CRPIX1"], header["CRPIX2"] = 60.5, 60.5
    s = 1.0 / 3600.0
    header["CD1_1"], header["CD1_2"] = -s, 0.0
    header["CD2_1"], header["CD2_2"] = 0.0, s
    header["BUNIT"] = "Jy/beam"
    header["BMAJ"] = header["BMIN"] = 3.0 / 3600.0
    header["BPA"] = 0.0
    path = tmp_path / "cd_region.fits"
    fits.PrimaryHDU(data=data, header=header).writeto(path)

    out = image_analysis.measure_region(url=str(path), region="circle(150.0, 2.0, 10\")")
    assert out["success"] is True
    assert out["statistics"]["max"] == pytest.approx(0.5, rel=0.15)
