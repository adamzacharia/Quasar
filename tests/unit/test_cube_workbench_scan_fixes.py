"""Regression tests for the 2026-07-17 scan fixes in services/cube_workbench.py:

  - IMG-03: CTYPE-based spectral-axis detection (CASA STOKES-on-3 cubes)
  - IMG-04: moment maps in spectral-axis units, not raw BUNIT
  - IMG-07: aperture_radius_arcsec honored via the WCS pixel scale
  - IMG-08: eviction guard keyed on live jobs per SESSION, not job ids
  - IMG-09: atomic locked read-modify-write of session JSON (cancel survives)
  - sle-line-overlays-restfrq-as-observed: RESTFRQ is rest-frame
  - sle-overlay-labels-velocity-axis: no GHz matching on velocity axes
  - sle-overlay-truncation-undisclosed: total_matches + truncation warning

All FITS inputs are synthetic local files (no network).
"""

import numpy as np
import pytest

from astropy.io import fits

from services.cube_workbench import CubeWorkbenchService


NCHAN = 25


def _casa_cube_path(tmp_path, nchan=NCHAN, nx=32, ny=32):
    """CASA-convention 4-axis cube: (RA, Dec, STOKES[1], FREQ[nchan])."""
    data = np.zeros((nchan, 1, ny, nx), dtype=np.float32)
    for k in range(nchan):
        data[k] += np.float32(np.exp(-0.5 * ((k - 12) / 3.0) ** 2))
    header = fits.Header()
    header["CTYPE1"], header["CTYPE2"] = "RA---SIN", "DEC--SIN"
    header["CRVAL1"], header["CRVAL2"] = 150.0, 2.0
    header["CRPIX1"], header["CRPIX2"] = nx / 2 + 0.5, ny / 2 + 0.5
    header["CDELT1"], header["CDELT2"] = -1.0 / 3600, 1.0 / 3600
    header["CTYPE3"] = "STOKES"
    header["CRVAL3"], header["CDELT3"], header["CRPIX3"] = 1.0, 1.0, 1.0
    header["CTYPE4"] = "FREQ"
    header["CRVAL4"], header["CDELT4"], header["CRPIX4"] = 230.0e9, 2.0e6, 1.0
    header["CUNIT4"] = "Hz"
    header["RESTFRQ"] = 230.538e9
    header["BUNIT"] = "Jy/beam"
    path = tmp_path / "casa_cube.fits"
    fits.PrimaryHDU(data=data, header=header).writeto(path)
    return path


@pytest.fixture()
def service(tmp_path):
    svc = CubeWorkbenchService(session_dir=tmp_path / "sessions", cache_dir=tmp_path / "cache")
    yield svc
    svc.shutdown()


def _prepared_casa_session(service, tmp_path):
    path = _casa_cube_path(tmp_path)
    session = service.create_session(user_id="u1", source_url=path.as_uri(),
                                     filename="casa_cube.fits")
    service.prepare_product(session_id=session["session_id"], user_id="u1")
    return session["session_id"]


# ─────────────────────────────────────────────────────────────────────────────
# IMG-03 — spectral axis by CTYPE, not hardcoded axis 3
# ─────────────────────────────────────────────────────────────────────────────
def test_spectral_axis_number_prefers_ctype_match():
    casa = {"CTYPE3": "STOKES", "CTYPE4": "FREQ"}
    plain = {"CTYPE3": "FREQ"}
    velocity = {"CTYPE3": "VRAD"}
    unknown = {}
    assert CubeWorkbenchService._spectral_axis_number(casa) == 4
    assert CubeWorkbenchService._spectral_axis_number(plain) == 3
    assert CubeWorkbenchService._spectral_axis_number(velocity) == 3
    assert CubeWorkbenchService._spectral_axis_number(unknown) == 3  # legacy fallback


def test_casa_cube_spectrum_uses_freq_axis(service, tmp_path):
    """A STOKES-on-3 cube must produce a real frequency axis with the true
    channel count, not a one-point 'STOKES' axis from NAXIS3=1."""
    session_id = _prepared_casa_session(service, tmp_path)
    spectrum = service.spectrum_plan(session_id=session_id, user_id="u1",
                                     x_pixel=16, y_pixel=16)
    axis = spectrum["spectral_axis"]
    assert axis["channel_count"] == NCHAN
    assert axis["ctype"].startswith("FREQ")
    assert axis["unit"] == "GHz"
    assert axis["basis"] == "wcs"
    assert axis["values"][0] == pytest.approx(230.0, abs=1e-6)
    assert len(spectrum["series"]["y"]) == len(axis["indices"])


def test_scale_preview_header_scales_the_freq_axis_not_stokes():
    header = fits.Header()
    header["CTYPE3"] = "STOKES"
    header["CDELT3"], header["CRPIX3"] = 1.0, 1.0
    header["CTYPE4"] = "FREQ"
    header["CDELT4"], header["CRPIX4"] = 2.0e6, 1.0
    CubeWorkbenchService._scale_preview_header(
        header, spectral_stride=5, spatial_stride_y=1, spatial_stride_x=1,
    )
    assert header["CDELT4"] == pytest.approx(1.0e7)  # freq CDELT scaled
    assert header["CDELT3"] == pytest.approx(1.0)    # Stokes untouched


def test_metadata_channel_count_from_spectral_axis():
    header = fits.Header()
    header["NAXIS"] = 4
    header["NAXIS1"], header["NAXIS2"] = 32, 32
    header["NAXIS3"], header["NAXIS4"] = 1, NCHAN
    header["CTYPE3"], header["CTYPE4"] = "STOKES", "FREQ"
    meta = CubeWorkbenchService._metadata_from_header(header, "file:///x.fits", "x.fits")
    assert meta["channel_count"] == NCHAN


# ─────────────────────────────────────────────────────────────────────────────
# IMG-04 — moment maps carry axis units, not BUNIT
# ─────────────────────────────────────────────────────────────────────────────
def test_moment1_stats_are_in_ghz_not_jy_beam(service, tmp_path):
    session_id = _prepared_casa_session(service, tmp_path)
    rendered = service.render_plan(session_id=session_id, user_id="u1",
                                   mode="moment", moment=1)
    stats = rendered["stats"]
    assert stats["unit"] == "GHz"
    assert "Moment 1 (GHz)" in rendered["image"]["label"]
    # Symmetric Gaussian line centered on channel 12 -> intensity-weighted
    # mean frequency = 230 GHz + 12 * 2 MHz = 230.024 GHz at every pixel.
    assert stats["mean"] == pytest.approx(230.024, abs=1e-4)


def test_moment0_unit_is_bunit_times_axis_unit(service, tmp_path):
    session_id = _prepared_casa_session(service, tmp_path)
    rendered = service.render_plan(session_id=session_id, user_id="u1",
                                   mode="moment", moment=0)
    assert rendered["stats"]["unit"] == "Jy/beam·GHz"
    # Moment 0 = channel sum x channel width (2 MHz = 0.002 GHz).
    expected = float(sum(np.exp(-0.5 * ((k - 12) / 3.0) ** 2) for k in range(NCHAN))) * 0.002
    assert rendered["stats"]["mean"] == pytest.approx(expected, rel=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# IMG-07 — aperture_radius_arcsec honored
# ─────────────────────────────────────────────────────────────────────────────
def test_aperture_radius_arcsec_converted_via_pixel_scale(service, tmp_path):
    session_id = _prepared_casa_session(service, tmp_path)
    spectrum = service.spectrum_plan(
        session_id=session_id, user_id="u1",
        x_pixel=16, y_pixel=16, aperture_radius_arcsec=5.0,
    )
    # 1 arcsec/pixel cube -> a 5" aperture is 5 pixels, not the 3-px default.
    assert spectrum["series"]["aperture_radius_pixels_used"] == pytest.approx(5.0)


def test_aperture_radius_pixels_default_still_applies(service, tmp_path):
    session_id = _prepared_casa_session(service, tmp_path)
    spectrum = service.spectrum_plan(session_id=session_id, user_id="u1",
                                     x_pixel=16, y_pixel=16)
    assert spectrum["series"]["aperture_radius_pixels_used"] == pytest.approx(3.0)


# ─────────────────────────────────────────────────────────────────────────────
# IMG-08 — eviction guard uses per-session live jobs
# ─────────────────────────────────────────────────────────────────────────────
def test_eviction_blocked_while_session_has_active_job(service, tmp_path):
    path = _casa_cube_path(tmp_path)
    session = service.create_session(user_id="u1", source_url=path.as_uri(),
                                     filename="casa_cube.fits")
    session_id = session["session_id"]
    service.prepare_product(session_id=session_id, user_id="u1")
    records = service._cached_session_records(user_id="u1")
    assert len(records) == 1
    cache_file = records[0]["cache_path"]
    assert cache_file.exists()

    with service._job_lock:
        service._track_session_job(session_id)  # simulate a queued/running job
    try:
        assert service._evict_cached_record(records[0], reason="quota_pressure") is None
        assert cache_file.exists()  # staged FITS survived
    finally:
        with service._job_lock:
            service._release_session_job(session_id)

    # With no live job the same record evicts normally.
    evicted = service._evict_cached_record(records[0], reason="quota_pressure")
    assert evicted is not None and evicted["session_id"] == session_id
    assert not cache_file.exists()


# ─────────────────────────────────────────────────────────────────────────────
# IMG-09 — cancel flag survives a concurrent worker write
# ─────────────────────────────────────────────────────────────────────────────
def test_cancel_requested_survives_worker_state_update(service, tmp_path):
    path = _casa_cube_path(tmp_path)
    session = service.create_session(user_id="u1", source_url=path.as_uri(),
                                     filename="casa_cube.fits")
    session_id = session["session_id"]
    session["jobs"] = [{
        "job_id": "job-1", "operation": "render", "status": "running",
        "phase": "running", "progress": 10, "created_at": 1, "started_at": 1,
        "finished_at": None, "cancel_requested": False, "request": {},
        "result": None, "error": "", "metrics": {},
    }]
    service._write_session(session)

    cancel = service.cancel_job(session_id=session_id, user_id="u1", job_id="job-1")
    assert cancel["job"]["cancel_requested"] is True

    # Worker-style progress update AFTER the cancel: it must re-read under the
    # lock and keep cancel_requested, not write back a stale pre-cancel copy.
    updated = service._update_job_state(
        session_id=session_id, user_id="u1", job_id="job-1",
        status="running", phase="downloading", progress=50,
    )
    assert updated["cancel_requested"] is True
    assert service._job_cancel_requested(session_id=session_id, user_id="u1",
                                         job_id="job-1") is True

    # And the worker's terminal write also lands intact.
    finished = service._update_job_state(
        session_id=session_id, user_id="u1", job_id="job-1",
        status="succeeded", progress=100, result={"ok": True}, finished=True,
    )
    assert finished["status"] == "succeeded"
    assert finished["cancel_requested"] is True
    assert finished["result"] == {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# sle-line-overlays-restfrq-as-observed
# ─────────────────────────────────────────────────────────────────────────────
def _overlay_session(service, metadata):
    service._write_session({
        "session_id": "line-session",
        "user_id": "u1",
        "metadata": metadata,
        "state": {},
    })
    captured = {}

    def fake_identify(**kwargs):
        captured.update(kwargs)
        return {
            "backend": "astroquery",
            "lines": [{"species": "CO v=0", "transition": "2-1",
                       "frequency_ghz": 230.538, "source": "CDMS"}],
            "total_matches": 1,
        }

    service.splatalogue.identify_spectral_line = fake_identify
    return captured


def test_header_restfrq_is_treated_as_rest_frame(service):
    captured = _overlay_session(service, {"rest_freq_ghz": 230.538})
    result = service.line_overlays(session_id="line-session", user_id="u1",
                                   redshift=0.03)
    # RESTFRQ is REST-frame: search at rest (not rest*(1+z)) and report
    # observed = rest / (1 + z).
    assert captured["frequency_ghz"] == pytest.approx(230.538)
    assert result["query_rest_frequency_ghz"] == pytest.approx(230.538)
    assert result["query_observed_frequency_ghz"] == pytest.approx(230.538 / 1.03)
    assert result["frequency_basis"] == "header_restfrq"


def test_header_restfrq_z0_assumption_is_disclosed(service):
    _overlay_session(service, {"rest_freq_ghz": 230.538})
    result = service.line_overlays(session_id="line-session", user_id="u1")
    assert result["query_observed_frequency_ghz"] == pytest.approx(230.538)
    assert any("z=0" in a for a in result["evidence"]["assumptions"])


def test_user_observed_frequency_path_unchanged(service):
    captured = _overlay_session(service, {})
    result = service.line_overlays(session_id="line-session", user_id="u1",
                                   observed_frequency_ghz=115.269, redshift=1.0)
    assert captured["frequency_ghz"] == pytest.approx(230.538)
    assert result["frequency_basis"] == "user_observed"


# ─────────────────────────────────────────────────────────────────────────────
# sle-overlay-truncation-undisclosed
# ─────────────────────────────────────────────────────────────────────────────
def test_truncated_candidate_list_is_disclosed(service):
    service._write_session({
        "session_id": "line-session",
        "user_id": "u1",
        "metadata": {},
        "state": {},
    })

    def fake_identify(**kwargs):
        return {
            "backend": "astroquery",
            "lines": [{"species": f"S{i}", "transition": "t",
                       "frequency_ghz": 230.5 + i * 1e-4, "source": "CDMS"}
                      for i in range(8)],
            "total_matches": 40,
        }

    service.splatalogue.identify_spectral_line = fake_identify
    result = service.line_overlays(session_id="line-session", user_id="u1",
                                   observed_frequency_ghz=230.5)
    assert result["total_matches"] == 40
    assert result["n_matches"] == 8
    assert any("40 transitions" in w for w in result["warnings"])
    # Persisted for the render/export paths too.
    state = service._read_session("line-session")["state"]["line_overlays"]
    assert state["total_matches"] == 40
    assert state["warnings"]


# ─────────────────────────────────────────────────────────────────────────────
# sle-overlay-labels-velocity-axis
# ─────────────────────────────────────────────────────────────────────────────
def _label_session(observed_ghz=230.5):
    return {
        "session_id": "s", "user_id": "u1", "metadata": {},
        "state": {"line_overlays": {
            "tolerance_ghz": 0.01,
            "lines": [{"species": "CO v=0", "observed_frequency_ghz": observed_ghz,
                       "rest_frequency_ghz": observed_ghz, "redshift": 0.0}],
        }},
    }


def test_line_labels_skip_velocity_axis_with_warning(service):
    header = {"CTYPE3": "VRAD", "CUNIT3": "km/s", "NAXIS3": 100,
              "CRVAL3": -500.0, "CDELT3": 10.0, "CRPIX3": 1.0}
    labels, warnings = service._line_labels_for_render(
        _label_session(), header=header, channel_index=73, nchan=100, mode="channel",
    )
    # Old code compared 230.5 GHz against km/s values: channel 73 sits at
    # ~230 km/s and matched spuriously. Now: no labels + a disclosure.
    assert labels == []
    assert any("not a frequency axis" in w for w in warnings)


def test_line_labels_skip_constant_restfrq_axis(service):
    session = _label_session()
    session["metadata"] = {"rest_freq_ghz": 230.5}
    header = {"NAXIS3": 100}  # no CRVAL/CDELT -> constant-RESTFRQ fallback axis
    labels, warnings = service._line_labels_for_render(
        session, header=header, channel_index=5, nchan=100, mode="channel",
    )
    assert labels == []
    assert warnings


def test_line_labels_match_on_true_frequency_axis(service):
    header = {"CTYPE3": "FREQ", "CUNIT3": "Hz", "NAXIS3": 100,
              "CRVAL3": 230.5e9, "CDELT3": 2.0e6, "CRPIX3": 1.0}
    labels, warnings = service._line_labels_for_render(
        _label_session(), header=header, channel_index=0, nchan=100, mode="channel",
    )
    assert warnings == []
    assert len(labels) == 1
    assert labels[0]["label"] == "CO v=0"
