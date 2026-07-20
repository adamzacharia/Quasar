"""R5 — hips_aperture_photometry: per-survey validation gate + photometry."""

import shutil

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import services.image_analysis as image_analysis
from services.hips_images import HipsImageService
from services.image_analysis import (
    HIPS_PHOTOMETRY_CAVEAT,
    HIPS_PHOTOMETRY_VALIDATION,
    hips_aperture_photometry,
)

RA0, DEC0 = 150.0, 2.0
AMP, SIGMA_PIX = 0.5, 3.0
TRUE_FLUX = AMP * 2.0 * np.pi * SIGMA_PIX ** 2   # ≈ 28.27 units·pix


def _write_source_fits(path, nx=200, ny=200, background=1.0):
    yy, xx = np.mgrid[0:ny, 0:nx]
    data = np.full((ny, nx), background, dtype=float)
    data += AMP * np.exp(-0.5 * (((xx - nx / 2) ** 2 + (yy - ny / 2) ** 2) / SIGMA_PIX ** 2))
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = [nx / 2 + 1, ny / 2 + 1]
    s = 1.0 / 3600.0
    w.wcs.cdelt = [-s, s]
    header = w.to_header()
    fits.PrimaryHDU(data=data.astype(np.float32), header=header).writeto(path, overwrite=True)
    return str(path)


@pytest.fixture()
def fake_hips(tmp_path, monkeypatch):
    """Serve the synthetic FITS for every hips2fits fetch; count per-band calls."""
    master = _write_source_fits(tmp_path / "master.fits")
    calls = {"urls": [], "n": 0}

    def _fake_fits_url(self, ra, dec, fov_deg=0.25, survey="optical", width=512):
        from services.hips_images import resolve_survey
        url = f"http://fake-hips2fits/{resolve_survey(survey)}"
        calls["urls"].append(url)
        return url

    def _fake_download(url, label):
        calls["n"] += 1
        copy = tmp_path / f"copy_{calls['n']}.fits"
        shutil.copy(master, copy)
        return str(copy)

    monkeypatch.setattr(HipsImageService, "fits_url", _fake_fits_url)
    monkeypatch.setattr(image_analysis, "_download_fits", _fake_download)
    return calls


def test_photometry_recovers_injected_flux(fake_hips):
    out = hips_aperture_photometry(ra=RA0, dec=DEC0, bands=["sdss_g"],
                                   radius_arcsec=15.0)
    assert out["success"] is True
    (m,) = [m for m in out["measurements"] if m["status"] == "measured"]
    assert m["band"] == "sdss_g"
    assert m["validation"]["status"] == "validated"
    # Background-subtracted aperture sum recovers the injected Gaussian flux.
    assert m["net_sum_native"] == pytest.approx(TRUE_FLUX, rel=0.05)
    assert m["pixel_scale_arcsec"] == pytest.approx(1.0, rel=0.01)
    assert out["image_path"].startswith("/api/images/") or out["image_path"].startswith("/plots/")


def test_mandatory_accuracy_caveat_always_present(fake_hips):
    out = hips_aperture_photometry(ra=RA0, dec=DEC0, bands=["sdss_r"])
    assert out["accuracy_caveat"] == HIPS_PHOTOMETRY_CAVEAT
    assert HIPS_PHOTOMETRY_CAVEAT in out["warnings"]
    assert "~10%" in HIPS_PHOTOMETRY_CAVEAT


def test_known_bad_pacs100_skipped_by_default(fake_hips):
    out = hips_aperture_photometry(ra=RA0, dec=DEC0,
                                   bands=["pacs100", "sdss_g"])
    assert out["success"] is True
    skipped = [m for m in out["measurements"] if m["status"] == "skipped_known_bad"]
    assert len(skipped) == 1 and skipped[0]["band"] == "pacs100"
    assert any("skipped" in w and "pacs100" in w for w in out["warnings"])
    # The good band still measured.
    assert any(m["status"] == "measured" for m in out["measurements"])


def test_known_bad_forced_with_flag_warns_loudly(fake_hips):
    out = hips_aperture_photometry(ra=RA0, dec=DEC0, bands=["pacs100"],
                                   include_known_bad=True)
    assert out["success"] is True
    (m,) = out["measurements"]
    assert m["status"] == "measured"
    assert m["validation"]["status"] == "known_bad"
    assert any("KNOWN-BAD" in w for w in out["warnings"])


def test_raw_hips_id_still_hits_known_bad_gate(fake_hips):
    # Passing the raw HiPS ID must not bypass the validation table.
    out = hips_aperture_photometry(ra=RA0, dec=DEC0,
                                   bands=["ESAVO/P/HERSCHEL/PACS100"])
    assert out["success"] is False  # nothing measurable
    assert out["measurements"][0]["status"] == "skipped_known_bad"


def test_unvalidated_band_is_flagged(fake_hips):
    out = hips_aperture_photometry(ra=RA0, dec=DEC0,
                                   bands=["wise_w1", "sdss_g"])
    m = next(m for m in out["measurements"] if m["band"] == "wise_w1")
    assert m["validation"]["status"] == "unvalidated"
    assert any("indicative only" in w for w in out["warnings"])


def test_unknown_alias_is_per_band_error_not_fatal(fake_hips):
    out = hips_aperture_photometry(ra=RA0, dec=DEC0,
                                   bands=["not_a_survey", "sdss_g"])
    assert out["success"] is True
    err = next(m for m in out["measurements"] if m["band"] == "not_a_survey")
    assert err["status"] == "error"


def test_requires_center():
    out = hips_aperture_photometry(ra=None, dec=None)
    assert out["success"] is False and "ra and dec" in out["error"]


def test_validation_table_has_nine_validated_and_pacs100_bad():
    statuses = [v["status"] for v in HIPS_PHOTOMETRY_VALIDATION.values()]
    assert statuses.count("validated") == 9
    assert HIPS_PHOTOMETRY_VALIDATION["ESAVO/P/HERSCHEL/PACS100"]["status"] == "known_bad"


def test_provenance_carries_hips2fits_request(fake_hips):
    out = hips_aperture_photometry(ra=RA0, dec=DEC0, bands=["sdss_g"])
    prov = out["provenance"]
    assert prov["service"] == "CDS hips2fits"
    assert prov["params"]["ra"] == RA0 and prov["params"]["dec"] == DEC0
    assert "CDS/P/SDSS9/g" in prov["params"]["hips"]


def test_capability_wiring_sets_image_card(fake_hips):
    from capabilities.base import CallContext
    from capabilities.viz import HipsAperturePhotometry

    state = {}
    ctx = CallContext(services={"set_last_run_result": lambda v: state.update(card=v)})
    cap = HipsAperturePhotometry()
    out = cap.run(cap.InputModel(ra=RA0, dec=DEC0, bands=["sdss_g"]), ctx).to_native()
    assert out["success"] is True
    assert state["card"]["type"] == "image"


# ─────────────────────────────────────────────────────────────────────────────
# Guard round CX-03/04/05/06/08 fixes
# ─────────────────────────────────────────────────────────────────────────────
def test_cx03_error_paths_carry_the_caveat(fake_hips):
    # Missing center.
    out = hips_aperture_photometry(ra=None, dec=None)
    assert out["success"] is False
    assert out["accuracy_caveat"] == HIPS_PHOTOMETRY_CAVEAT
    # Invalid radius.
    out = hips_aperture_photometry(ra=RA0, dec=DEC0, radius_arcsec=0)
    assert out["success"] is False and out["accuracy_caveat"] == HIPS_PHOTOMETRY_CAVEAT
    # No measurable band (known-bad only).
    out = hips_aperture_photometry(ra=RA0, dec=DEC0,
                                   bands=["ESAVO/P/HERSCHEL/PACS100"])
    assert out["success"] is False and out["accuracy_caveat"] == HIPS_PHOTOMETRY_CAVEAT


def test_cx05_explicit_zero_radius_is_an_error_not_default(fake_hips):
    out = hips_aperture_photometry(ra=RA0, dec=DEC0, bands=["sdss_g"],
                                   radius_arcsec=0)
    assert out["success"] is False and "radius_arcsec" in out["error"]


def test_cx05_band_cap_warns_about_dropped_bands(fake_hips):
    bands = [f"CDS/P/SDSS9/g" for _ in range(11)]
    bands[10] = "wise_w1"
    out = hips_aperture_photometry(ra=RA0, dec=DEC0, bands=bands)
    assert out["success"] is True
    assert any("dropped: wise_w1" in w for w in out["warnings"])


def test_cx04_empty_annulus_falls_back_loudly(fake_hips):
    # r=200" → annulus at 300-450 px lies fully outside the 200x200 image.
    out = hips_aperture_photometry(ra=RA0, dec=DEC0, bands=["sdss_g"],
                                   radius_arcsec=200.0, fov_deg=0.2)
    assert out["success"] is True
    (m,) = [m for m in out["measurements"] if m["status"] == "measured"]
    assert m["background_source"] == "global_median_annulus_empty"
    assert any("annulus had no usable pixels" in w for w in out["warnings"])


def test_cx04_normal_annulus_labelled(fake_hips):
    out = hips_aperture_photometry(ra=RA0, dec=DEC0, bands=["sdss_g"],
                                   radius_arcsec=15.0)
    (m,) = [m for m in out["measurements"] if m["status"] == "measured"]
    assert m["background_source"] == "annulus"


def test_cx06_provenance_records_clamped_fov(fake_hips):
    # fov 99 deg clamps to 10; provenance must record the clamped value.
    out = hips_aperture_photometry(ra=RA0, dec=DEC0, bands=["sdss_g"],
                                   fov_deg=99.0)
    assert out["provenance"]["params"]["fov"] == 10.0
    assert any("clamped" in w for w in out["warnings"])


def test_cx08_all_herschel_aliases_resolve():
    from services.hips_images import resolve_survey
    assert resolve_survey("pacs70") == "ESAVO/P/HERSCHEL/PACS70"
    assert resolve_survey("pacs100") == "ESAVO/P/HERSCHEL/PACS100"
    assert resolve_survey("pacs160") == "ESAVO/P/HERSCHEL/PACS160"
    assert resolve_survey("spire250") == "ESAVO/P/HERSCHEL/SPIRE-250"
    assert resolve_survey("spire350") == "ESAVO/P/HERSCHEL/SPIRE-350"
    assert resolve_survey("spire500") == "ESAVO/P/HERSCHEL/SPIRE-500"


# ─────────────────────────────────────────────────────────────────────────────
# Verify-round CX-03/CX-05 reopens
# ─────────────────────────────────────────────────────────────────────────────
def test_cx03_verify_capability_errors_carry_caveat(fake_hips):
    from capabilities.base import CallContext
    from capabilities.viz import HipsAperturePhotometry

    cap = HipsAperturePhotometry()
    ctx = CallContext(services={"set_last_run_result": lambda v: None})
    # Unresolvable center (no target, no coords).
    out = cap.run(cap.InputModel(), ctx).to_native()
    assert out["success"] is False
    assert out["accuracy_caveat"] == HIPS_PHOTOMETRY_CAVEAT


def test_cx05_verify_explicit_empty_band_list_is_an_error(fake_hips):
    out = hips_aperture_photometry(ra=RA0, dec=DEC0, bands=[])
    assert out["success"] is False and "at least one band" in out["error"]
    assert out["accuracy_caveat"] == HIPS_PHOTOMETRY_CAVEAT
    # Whitespace-only entries count as empty too.
    out = hips_aperture_photometry(ra=RA0, dec=DEC0, bands=["  "])
    assert out["success"] is False and "at least one band" in out["error"]


def test_cx05_verify_explicit_zero_fov_and_width_are_errors(fake_hips):
    out = hips_aperture_photometry(ra=RA0, dec=DEC0, bands=["sdss_g"], fov_deg=0)
    assert out["success"] is False and "fov_deg" in out["error"]
    out = hips_aperture_photometry(ra=RA0, dec=DEC0, bands=["sdss_g"], width=0)
    assert out["success"] is False and "width" in out["error"]


def test_cx05_verify_omitted_inputs_still_default(fake_hips):
    out = hips_aperture_photometry(ra=RA0, dec=DEC0, bands=["sdss_g"])
    assert out["success"] is True
    assert out["aperture_radius_arcsec"] == 15.0


def test_cx03_second_reopen_caveat_survives_import_failure(fake_hips, monkeypatch):
    # Even if services.image_analysis fails to import inside run(), the error
    # response still carries a caveat rather than raising.
    import builtins

    from capabilities.base import CallContext
    from capabilities.viz import HipsAperturePhotometry

    real_import = builtins.__import__

    def _broken_import(name, *args, **kwargs):
        if name == "services.image_analysis" or name.endswith("image_analysis"):
            raise ImportError("simulated import failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(__import__("sys").modules, "services.image_analysis", raising=False)
    monkeypatch.setattr(builtins, "__import__", _broken_import)
    cap = HipsAperturePhotometry()
    ctx = CallContext(services={"set_last_run_result": lambda v: None})
    out = cap.run(cap.InputModel(ra=RA0, dec=DEC0), ctx).to_native()
    assert out["success"] is False
    assert "~10%" in out["accuracy_caveat"]


def test_cx02_r9round_infinite_fov_is_an_error(fake_hips):
    out = hips_aperture_photometry(ra=RA0, dec=DEC0, bands=["sdss_g"],
                                   fov_deg=float("inf"))
    assert out["success"] is False and "fov_deg" in out["error"]
    assert out["accuracy_caveat"] == HIPS_PHOTOMETRY_CAVEAT
    out = hips_aperture_photometry(ra=RA0, dec=DEC0, bands=["sdss_g"],
                                   width=float("inf"))
    assert out["success"] is False and "width" in out["error"]
    assert out["accuracy_caveat"] == HIPS_PHOTOMETRY_CAVEAT


def test_cx04_r9round_bands_none_uses_default_set(fake_hips):
    out = hips_aperture_photometry(ra=RA0, dec=DEC0)
    assert out["success"] is True
    assert out["n_bands_measured"] == 5
    measured_bands = [m["band"] for m in out["measurements"] if m["status"] == "measured"]
    assert measured_bands == ["galex_fuv", "galex_nuv", "sdss_g", "sdss_r", "sdss_i"]


def test_cx05_r9round_zero_fov_width_errors_carry_caveat(fake_hips):
    out = hips_aperture_photometry(ra=RA0, dec=DEC0, bands=["sdss_g"], fov_deg=0)
    assert out["success"] is False
    assert out["accuracy_caveat"] == HIPS_PHOTOMETRY_CAVEAT
    out = hips_aperture_photometry(ra=RA0, dec=DEC0, bands=["sdss_g"], width=0)
    assert out["success"] is False
    assert out["accuracy_caveat"] == HIPS_PHOTOMETRY_CAVEAT
