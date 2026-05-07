"""
Astronomy Calculator Tools for Quasar AI.

Pure-computation tools that require no external API calls -- they use astropy
and standard physics formulas to give instant results.

Features:
  U9  -- Redshift Calculator (cosmological distances, lookback time, scale)
  U10 -- Coordinate Converter (ICRS, Galactic, Ecliptic, epoch precession)
  R6  -- Beam Calculator (synthesized beam from baseline + frequency)
  R4  -- ALMA Sensitivity Calculator (radiometer equation per band)

CALLED BY: core/agent.py (tool execution)
CALLS:     astropy.cosmology, astropy.coordinates, math
"""

import math
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports -- astropy is heavy, only import when actually called
# ---------------------------------------------------------------------------

def _get_cosmology():
    """Return Planck18 cosmology, lazy-loaded."""
    from astropy.cosmology import Planck18
    return Planck18


# ═══════════════════════════════════════════════════════════════════════════
# U9 -- Redshift Calculator
# ═══════════════════════════════════════════════════════════════════════════

def calculate_redshift(z: float) -> Dict[str, Any]:
    """
    Given a redshift z, compute key cosmological quantities using Planck18.

    Returns luminosity distance, angular diameter distance, comoving distance,
    lookback time, age of the universe at that redshift, and the physical
    scale (kpc per arcsecond).

    Parameters
    ----------
    z : float
        Cosmological redshift (must be >= 0).

    Returns
    -------
    dict
        All computed quantities with units.
    """
    if z < 0:
        return {"success": False, "error": "Redshift must be >= 0."}

    try:
        cosmo = _get_cosmology()
        import astropy.units as u

        d_L = cosmo.luminosity_distance(z).to(u.Mpc).value
        d_A = cosmo.angular_diameter_distance(z).to(u.Mpc).value
        d_C = cosmo.comoving_distance(z).to(u.Mpc).value
        lookback = cosmo.lookback_time(z).to(u.Gyr).value
        age_at_z = cosmo.age(z).to(u.Gyr).value
        age_now = cosmo.age(0).to(u.Gyr).value

        # Physical scale: kpc per arcsecond at this redshift
        if z > 0:
            kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec).value
        else:
            kpc_per_arcsec = 0.0

        # Distance modulus
        if d_L > 0:
            dist_modulus = 5 * math.log10(d_L * 1e6 / 10)  # d_L in pc
        else:
            dist_modulus = 0.0

        return {
            "success": True,
            "redshift": z,
            "cosmology": "Planck18 (H0=67.66, Om0=0.3111)",
            "luminosity_distance_mpc": round(d_L, 2),
            "angular_diameter_distance_mpc": round(d_A, 2),
            "comoving_distance_mpc": round(d_C, 2),
            "lookback_time_gyr": round(lookback, 3),
            "age_of_universe_at_z_gyr": round(age_at_z, 3),
            "age_of_universe_now_gyr": round(age_now, 3),
            "physical_scale_kpc_per_arcsec": round(kpc_per_arcsec, 4),
            "distance_modulus_mag": round(dist_modulus, 2),
        }

    except Exception as e:
        logger.error("Redshift calculation failed: %s", e)
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# U10 -- Coordinate Converter
# ═══════════════════════════════════════════════════════════════════════════

def convert_coordinates(
    ra: float = None,
    dec: float = None,
    l: float = None,
    b: float = None,
    input_frame: str = "icrs",
    output_frame: str = "galactic",
    epoch_in: str = "J2000",
    epoch_out: str = None,
) -> Dict[str, Any]:
    """
    Convert sky coordinates between ICRS, Galactic, and Ecliptic frames.
    Optionally precess between epochs (J2000, B1950).

    Provide EITHER (ra, dec) OR (l, b) depending on input_frame.

    Parameters
    ----------
    ra, dec : float
        Input coordinates in degrees (for ICRS or Ecliptic input).
    l, b : float
        Input coordinates in degrees (for Galactic input).
    input_frame : str
        Source frame: 'icrs', 'galactic', 'ecliptic', 'fk5', 'fk4'
    output_frame : str
        Target frame: 'icrs', 'galactic', 'ecliptic', 'fk5', 'fk4'
    epoch_in : str
        Input epoch (e.g., 'J2000', 'B1950'). Only for FK4/FK5.
    epoch_out : str
        Output epoch. If None, matches output frame default.

    Returns
    -------
    dict
        Converted coordinates in all standard frames.
    """
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        # Build the input coordinate
        frame_lower = input_frame.lower().strip()

        if frame_lower == "galactic":
            if l is None or b is None:
                return {"success": False, "error": "Galactic input requires l and b in degrees."}
            coord = SkyCoord(l=l * u.deg, b=b * u.deg, frame="galactic")
        elif frame_lower in ("fk4",):
            if ra is None or dec is None:
                return {"success": False, "error": "FK4 input requires ra and dec in degrees."}
            coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="fk4", equinox="B1950")
        elif frame_lower in ("ecliptic", "barycentricmeanecliptic"):
            if ra is None or dec is None:
                return {"success": False, "error": "Ecliptic input requires lon (ra) and lat (dec) in degrees."}
            from astropy.coordinates import BarycentricMeanEcliptic
            coord = SkyCoord(
                lon=ra * u.deg, lat=dec * u.deg,
                frame=BarycentricMeanEcliptic
            )
        else:
            # Default: ICRS or FK5
            if ra is None or dec is None:
                return {"success": False, "error": "ICRS/FK5 input requires ra and dec in degrees."}
            coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame=frame_lower)

        # Convert to all frames
        icrs = coord.icrs
        gal = coord.galactic
        fk5_j2000 = coord.fk5
        fk4_b1950 = coord.fk4

        try:
            from astropy.coordinates import BarycentricMeanEcliptic
            ecl = coord.transform_to(BarycentricMeanEcliptic())
            ecl_lon = round(ecl.lon.deg, 6)
            ecl_lat = round(ecl.lat.deg, 6)
        except Exception:
            ecl_lon = None
            ecl_lat = None

        # Format RA as HMS, Dec as DMS
        ra_hms = icrs.ra.to_string(unit=u.hourangle, sep=":", precision=2)
        dec_dms = icrs.dec.to_string(unit=u.deg, sep=":", precision=2)

        return {
            "success": True,
            "input_frame": input_frame,
            "icrs": {
                "ra_deg": round(icrs.ra.deg, 6),
                "dec_deg": round(icrs.dec.deg, 6),
                "ra_hms": ra_hms,
                "dec_dms": dec_dms,
            },
            "galactic": {
                "l_deg": round(gal.l.deg, 6),
                "b_deg": round(gal.b.deg, 6),
            },
            "ecliptic": {
                "lon_deg": ecl_lon,
                "lat_deg": ecl_lat,
            },
            "fk5_j2000": {
                "ra_deg": round(fk5_j2000.ra.deg, 6),
                "dec_deg": round(fk5_j2000.dec.deg, 6),
            },
            "fk4_b1950": {
                "ra_deg": round(fk4_b1950.ra.deg, 6),
                "dec_deg": round(fk4_b1950.dec.deg, 6),
            },
        }

    except Exception as e:
        logger.error("Coordinate conversion failed: %s", e)
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# R6 -- Beam Calculator
# ═══════════════════════════════════════════════════════════════════════════

# Real ALMA array configurations: name -> max baseline in meters
# Source: ALMA Technical Handbook, Cycle 11
ALMA_CONFIGS = {
    "C-1":  161,
    "C-2":  314,
    "C-3":  500,
    "C-4":  784,
    "C-5":  1398,
    "C-6":  2517,
    "C-7":  3638,
    "C-8":  8548,
    "C-9":  13894,
    "C-10": 16196,
}

# ALMA band center frequencies (GHz)
ALMA_BANDS = {
    1: 43.0,
    2: 75.0,
    3: 100.0,
    4: 144.0,
    5: 187.0,
    6: 233.0,
    7: 343.5,
    8: 405.0,
    9: 661.0,
    10: 868.5,
}


def calculate_beam(
    frequency_ghz: float,
    max_baseline_m: float = None,
    array_config: str = None,
) -> Dict[str, Any]:
    """
    Calculate the synthesized beam size for a radio interferometer.

    Uses theta = 1.22 * lambda / B_max (diffraction limit).
    For ALMA, you can specify an array configuration name (C-1 through C-10)
    instead of a raw baseline length.

    Parameters
    ----------
    frequency_ghz : float
        Observing frequency in GHz.
    max_baseline_m : float, optional
        Maximum baseline length in meters.
    array_config : str, optional
        ALMA array config name (e.g., 'C-6'). Overrides max_baseline_m.

    Returns
    -------
    dict
        Synthesized beam size in arcseconds, plus context.
    """
    if frequency_ghz <= 0:
        return {"success": False, "error": "Frequency must be positive."}

    try:
        # Resolve array config
        if array_config:
            config_upper = array_config.upper().strip()
            if config_upper in ALMA_CONFIGS:
                max_baseline_m = ALMA_CONFIGS[config_upper]
            else:
                return {
                    "success": False,
                    "error": f"Unknown array config '{array_config}'. "
                             f"Valid configs: {', '.join(ALMA_CONFIGS.keys())}",
                }

        if max_baseline_m is None or max_baseline_m <= 0:
            return {
                "success": False,
                "error": "Provide max_baseline_m (meters) or array_config (e.g., 'C-6').",
            }

        # Speed of light
        c = 299792458.0  # m/s
        wavelength_m = c / (frequency_ghz * 1e9)
        wavelength_mm = wavelength_m * 1000

        # Diffraction limit: theta = 1.22 * lambda / B (radians)
        theta_rad = 1.22 * wavelength_m / max_baseline_m
        theta_arcsec = math.degrees(theta_rad) * 3600
        theta_mas = theta_arcsec * 1000

        # Identify which ALMA band this frequency falls in
        alma_band = None
        for band_num, center_freq in ALMA_BANDS.items():
            # Rough matching: within 30% of band center
            if abs(frequency_ghz - center_freq) / center_freq < 0.30:
                alma_band = band_num
                break

        result = {
            "success": True,
            "frequency_ghz": frequency_ghz,
            "wavelength_mm": round(wavelength_mm, 3),
            "max_baseline_m": max_baseline_m,
            "max_baseline_km": round(max_baseline_m / 1000, 2),
            "synthesized_beam_arcsec": round(theta_arcsec, 4),
            "synthesized_beam_mas": round(theta_mas, 2),
            "formula": "theta = 1.22 * lambda / B_max",
            "note": "This is the diffraction limit. Actual beam depends on uv-coverage, weighting, and tapering.",
        }

        if array_config:
            result["array_config"] = array_config.upper()
        if alma_band:
            result["alma_band"] = alma_band

        # Add all ALMA configs for reference if no specific config was given
        if not array_config:
            result["alma_configs_reference"] = {
                k: f"{v}m ({v/1000:.1f}km)" for k, v in ALMA_CONFIGS.items()
            }

        return result

    except Exception as e:
        logger.error("Beam calculation failed: %s", e)
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# R4 -- ALMA Sensitivity Calculator
# ═══════════════════════════════════════════════════════════════════════════

# System temperature (Tsys in K) per ALMA band at typical weather
# Source: ALMA Technical Handbook, median conditions
ALMA_TSYS = {
    3: 70,
    4: 90,
    5: 110,
    6: 120,
    7: 200,
    8: 400,
    9: 800,
    10: 1500,
}

# Aperture efficiency per band (approximate)
ALMA_ETA_A = {
    3: 0.71,
    4: 0.69,
    5: 0.67,
    6: 0.65,
    7: 0.57,
    8: 0.49,
    9: 0.35,
    10: 0.25,
}

ALMA_DISH_DIAMETER = 12.0  # meters
ALMA_TOTAL_ANTENNAS = 50   # 12-m array (typical)


def calculate_alma_sensitivity(
    band: int,
    bandwidth_ghz: float = 7.5,
    t_integration_s: float = 60.0,
    n_antennas: int = None,
    n_polarizations: int = 2,
    channel_width_khz: float = None,
    pwv_mm: float = 1.0,
) -> Dict[str, Any]:
    """
    Estimate ALMA continuum and line sensitivity using the radiometer equation.

    sigma = (2 * k * Tsys) / (eta_A * A_eff * sqrt(N*(N-1) * delta_nu * t_int * n_pol))

    Parameters
    ----------
    band : int
        ALMA band number (3-10).
    bandwidth_ghz : float
        Total continuum bandwidth in GHz (default 7.5 for dual-sideband).
    t_integration_s : float
        On-source integration time in seconds.
    n_antennas : int
        Number of antennas (default 50 for 12-m array).
    n_polarizations : int
        Number of polarizations (1 or 2, default 2).
    channel_width_khz : float
        Spectral channel width in kHz (for line sensitivity). If None,
        only continuum sensitivity is computed.
    pwv_mm : float
        Precipitable water vapor in mm (affects Tsys scaling). Default 1.0.

    Returns
    -------
    dict
        Continuum and optionally line sensitivity in mJy/beam.
    """
    if band not in ALMA_TSYS:
        return {
            "success": False,
            "error": f"Band {band} not supported. Valid bands: {list(ALMA_TSYS.keys())}",
        }

    if t_integration_s <= 0:
        return {"success": False, "error": "Integration time must be positive."}

    if bandwidth_ghz <= 0:
        return {"success": False, "error": "Bandwidth must be positive."}

    try:
        n_ant = n_antennas or ALMA_TOTAL_ANTENNAS
        Tsys = ALMA_TSYS[band]
        eta_a = ALMA_ETA_A.get(band, 0.5)

        # Scale Tsys with PWV (rough: Tsys increases ~20% per mm PWV above Band 6)
        if pwv_mm > 1.0 and band >= 6:
            pwv_factor = 1.0 + 0.2 * (pwv_mm - 1.0)
            Tsys = Tsys * pwv_factor

        # Boltzmann constant
        k_B = 1.380649e-23  # J/K

        # Effective collecting area of one antenna
        A_eff = eta_a * math.pi * (ALMA_DISH_DIAMETER / 2) ** 2  # m^2

        # Number of baselines
        n_baselines = n_ant * (n_ant - 1) / 2

        # Continuum sensitivity (Jy)
        delta_nu_cont = bandwidth_ghz * 1e9  # Hz
        sigma_cont_jy = (
            (2 * k_B * Tsys)
            / (A_eff * math.sqrt(n_baselines * delta_nu_cont * t_integration_s * n_polarizations))
        ) * 1e26  # Convert W/m^2/Hz to Jy

        sigma_cont_mjy = sigma_cont_jy * 1000
        sigma_cont_ujy = sigma_cont_jy * 1e6

        result = {
            "success": True,
            "band": band,
            "frequency_ghz": ALMA_BANDS.get(band, "N/A"),
            "tsys_k": round(Tsys, 1),
            "aperture_efficiency": eta_a,
            "n_antennas": n_ant,
            "n_baselines": int(n_baselines),
            "n_polarizations": n_polarizations,
            "bandwidth_ghz": bandwidth_ghz,
            "integration_time_s": t_integration_s,
            "integration_time_min": round(t_integration_s / 60, 2),
            "pwv_mm": pwv_mm,
            "continuum_sensitivity_mjy_beam": round(sigma_cont_mjy, 4),
            "continuum_sensitivity_ujy_beam": round(sigma_cont_ujy, 1),
            "formula": "sigma = 2*k*Tsys / (eta*A * sqrt(N*(N-1) * dv * t * npol))",
            "note": (
                "This is a theoretical estimate. Actual sensitivity depends on "
                "weather, calibration overhead, flagging, and imaging parameters."
            ),
        }

        # Line sensitivity (if channel width provided)
        if channel_width_khz is not None and channel_width_khz > 0:
            delta_nu_line = channel_width_khz * 1e3  # Hz
            sigma_line_jy = (
                (2 * k_B * Tsys)
                / (A_eff * math.sqrt(n_baselines * delta_nu_line * t_integration_s * n_polarizations))
            ) * 1e26
            sigma_line_mjy = sigma_line_jy * 1000

            # Convert channel width to velocity (km/s)
            freq_ghz = ALMA_BANDS.get(band, 230.0)
            channel_vel_kms = (channel_width_khz * 1e-6 / freq_ghz) * 299792.458

            result["channel_width_khz"] = channel_width_khz
            result["channel_velocity_km_s"] = round(channel_vel_kms, 3)
            result["line_sensitivity_mjy_beam"] = round(sigma_line_mjy, 4)

        return result

    except Exception as e:
        logger.error("ALMA sensitivity calculation failed: %s", e)
        return {"success": False, "error": str(e)}
