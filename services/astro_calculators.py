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

# Nominal receiver band limits (GHz) — Technical Handbook / skill
# cycle-capabilities.md. Band 2 (67-116 GHz, Cycle 13 12-m only) overlaps
# Band 3 (84-116); WVR windows sit near 183 GHz in every band and calibration
# SPWs can fall in inter-band gaps, so lookups must tolerate out-of-band input.
ALMA_BAND_LIMITS_GHZ = {
    1: (35.0, 50.0),
    2: (67.0, 116.0),
    3: (84.0, 116.0),
    4: (125.0, 163.0),
    5: (158.0, 211.0),
    6: (211.0, 275.0),
    7: (275.0, 373.0),
    8: (385.0, 500.0),
    9: (602.0, 720.0),
    10: (787.0, 950.0),
}

# Technical Handbook (Cycle 13) Table 7.3: 80th-percentile baseline L80 of the
# notional configurations, metres. The Handbook's angular resolution is set by
# L80 (and robust=0.5 weighting), NOT by the maximum baseline.
ALMA_L80_M = {
    "7-M": 30.7, "C-1": 107.1, "C-2": 143.8, "C-3": 235.4, "C-4": 369.2, "C-5": 623.8,
    "C-6": 1172.5, "C-7": 1673.1, "C-8": 3527.3, "C-9": 6482.6, "C-10": 8685.9,
}


def bands_for_frequency(frequency_ghz: float) -> list:
    """All ALMA bands whose nominal range contains the frequency (Band 3
    preferred over the overlapping Band 2); [] for inter-band gaps."""
    hits = [b for b, (lo, hi) in ALMA_BAND_LIMITS_GHZ.items() if lo <= frequency_ghz <= hi]
    if 2 in hits and 3 in hits:
        hits = [3] + [b for b in hits if b not in (2, 3)] + [2]
    return hits


def calculate_beam(
    frequency_ghz: float,
    max_baseline_m: float = None,
    array_config: str = None,
) -> Dict[str, Any]:
    """
    Calculate the synthesized beam size for a radio interferometer.

    Approximation: theta ~ lambda / B (order-of-magnitude synthesized beam).
    1.22*lambda/D is the Rayleigh criterion for a SINGLE filled aperture and
    is NOT an interferometer FWHM; the Handbook's configuration resolutions
    are set by the 80th-percentile baseline L80 with Briggs robust=0.5
    weighting. For an ALMA configuration name (C-1 .. C-10) the L80 estimate
    is reported alongside, since uv-coverage, weighting and elevation move
    the real beam by tens of percent.

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

        # Order-of-magnitude synthesized beam: theta ~ 1.22 * lambda / B_max.
        # (Kept for continuity; labelled as an approximation, not a
        # diffraction limit — see the L80 estimate below.)
        theta_rad = 1.22 * wavelength_m / max_baseline_m
        theta_arcsec = math.degrees(theta_rad) * 3600
        theta_mas = theta_arcsec * 1000

        # Identify which ALMA band(s) this frequency falls in — nominal band
        # limits, all matches, None for inter-band gaps (A-74).
        bands = bands_for_frequency(frequency_ghz)
        alma_band = bands[0] if bands else None

        result = {
            "success": True,
            "frequency_ghz": frequency_ghz,
            "wavelength_mm": round(wavelength_mm, 3),
            "max_baseline_m": max_baseline_m,
            "max_baseline_km": round(max_baseline_m / 1000, 2),
            "synthesized_beam_arcsec": round(theta_arcsec, 4),
            "synthesized_beam_mas": round(theta_mas, 2),
            "formula": "theta ~ 1.22 * lambda / B_max (approximate proxy; NOT the interferometer FWHM)",
            "note": (
                "Approximate proxy only. 1.22*lambda/D is the single-aperture Rayleigh criterion; an "
                "interferometer's synthesized beam depends on uv-coverage, weighting (ALMA QA2 uses "
                "Briggs robust=0.5), elevation and tapering — the Handbook resolution is set by the "
                "80th-percentile baseline (L80), reported as beam_l80_arcsec when a configuration is given."
            ),
        }

        if array_config:
            result["array_config"] = array_config.upper()
            l80 = ALMA_L80_M.get(array_config.upper())
            if l80:
                # theta ~ 0.574 lambda / L80 reproduces the Handbook's tabulated
                # per-configuration resolutions (robust = 0.5) to ~1%.
                result["l80_m"] = l80
                result["beam_l80_arcsec"] = round(math.degrees(0.574 * wavelength_m / l80) * 3600, 4)
                result["beam_l80_note"] = (
                    "theta_res ~ 0.574 * lambda / L80 (L80 = 80th-percentile baseline of the notional "
                    "configuration, Technical Handbook Table 7.3; Briggs robust 0.5). Use this as the "
                    "expected synthesized beam; the B_max proxy above is only order-of-magnitude."
                )
        if alma_band:
            result["alma_band"] = alma_band
            result["alma_bands_containing_frequency"] = bands
        else:
            result["alma_band"] = None
            result["alma_band_note"] = (
                f"{frequency_ghz} GHz lies outside every nominal ALMA receiver band "
                "(inter-band gap or WVR/calibration window)."
            )

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
# R4 -- ALMA Sensitivity Calculator (Technical Handbook Cycle 13, Chapter 9)
# ═══════════════════════════════════════════════════════════════════════════
#
# Point-source sensitivity for the 12-m and 7-m Arrays (Handbook eq. 9.8):
#
#     dS = w_r * 2 k Tsys / ( eta_q * eta_c * A_eff * (1 - f_s) * sqrt( N (N-1) n_p dnu t_int ) )
#
# and for the Total Power Array (eq. 9.11):
#
#     dS_TP = 2 k Tsys / ( eta_q * eta_c * A_eff * sqrt( N n_p dnu t_int ) )
#
# with eta_q = 0.96 (3-bit digitizer quantization), eta_c = 0.88 (64-input
# correlator, all modes), N defaulting to 43 / 10 / 3 antennas for the 12-m /
# 7-m / TP Arrays, A_eff = eta_ap x geometric area (113.1 m^2 for 12-m,
# 38.5 m^2 for 7-m; Table 9.3 aperture efficiencies), w_r the robust-weighting
# factor (1.0 = natural; the OT assumes Briggs robust 0.5), f_s the shadowed
# antenna fraction. N(N-1) sits under the root WITHOUT a /2: the 2kT numerator
# already carries the single-baseline sqrt(2). The previous implementation used
# 2kT with N(N-1)/2 under the root, inflating every estimate by sqrt(2)=1.41
# (report G-12).
#
# What this calculator does NOT model: the ALMA Sensitivity Calculator (ASC)
# derives Tsys from the receiver temperature, the atmospheric model at the
# requested PWV octile and the source elevation; the values below are
# representative median-weather Tsys per band and the PWV scaling is a coarse
# proxy. The ASC / OT remain authoritative for proposals.

ALMA_HANDBOOK_VERSION = "ALMA Technical Handbook, Cycle 13 (Doc 13.3), Sec. 9.2, eq. 9.8 / 9.9 / 9.11, Table 9.3"

# Representative Tsys (K) at the Table 9.3 continuum frequencies for good
# median weather (PWV ~1 mm at Bands 3-7; drier octiles for 8-10). These are
# calculator ASSUMPTIONS, versioned by handbook cycle; the ASC computes Tsys
# from Trx + atmosphere at the requested PWV.
ALMA_TSYS = {
    1: 55,
    2: 65,
    3: 70,
    4: 90,
    5: 110,
    6: 120,
    7: 200,
    8: 400,
    9: 800,
    10: 1500,
}

# Handbook Table 9.3: aperture efficiency eta_ap at typical continuum
# frequencies, 12-m and 7-m antennas.
ALMA_TABLE_9_3 = {
    # band: (frequency_ghz, eta_ap_12m, eta_ap_7m)
    1: (40.0, 0.72, 0.72),
    2: (75.0, 0.71, 0.71),
    3: (100.0, 0.71, 0.71),
    4: (145.0, 0.70, 0.71),
    5: (183.0, 0.69, 0.70),
    6: (230.0, 0.68, 0.69),
    7: (345.0, 0.63, 0.66),
    8: (405.0, 0.60, 0.64),
    9: (690.0, 0.43, 0.52),
    10: (870.0, 0.31, 0.42),
}
ALMA_ETA_A = {band: row[1] for band, row in ALMA_TABLE_9_3.items()}      # 12-m, back-compat name
ALMA_ETA_A_7M = {band: row[2] for band, row in ALMA_TABLE_9_3.items()}

ALMA_ETA_Q = 0.96          # digitizer quantization efficiency (3-bit)
ALMA_ETA_C = 0.88          # 64-input correlator efficiency (all modes; ACA assumed equal by the web ASC)
ALMA_DISH_DIAMETER = 12.0  # meters
ALMA_GEOMETRIC_AREA_M2 = {"12m": 113.1, "7m": 38.5, "TP": 113.1}
ALMA_DEFAULT_ANTENNAS = {"12m": 43, "7m": 10, "TP": 3}
ALMA_TOTAL_ANTENNAS = ALMA_DEFAULT_ANTENNAS["12m"]  # back-compat name (was 50)
ALMA_ARRAY_ALIASES = {
    "12M": "12m", "12-M": "12m", "MAIN": "12m", "TM": "12m", "TM1": "12m", "TM2": "12m",
    "7M": "7m", "7-M": "7m", "ACA": "7m", "MORITA": "7m",
    "TP": "TP", "TOTALPOWER": "TP", "TOTAL_POWER": "TP", "TOTAL-POWER": "TP", "SD": "TP",
}
K_B = 1.380649e-23  # J/K
C_KMS = 299792.458


def _normalize_array(array: Optional[str]) -> Optional[str]:
    key = str(array or "12m").strip().upper().replace(" ", "")
    return ALMA_ARRAY_ALIASES.get(key)


def _tsys_for(band: int, frequency_ghz: float, pwv_mm: float) -> tuple:
    """Representative Tsys with a coarse PWV scaling; returns (tsys, note)."""
    tsys = float(ALMA_TSYS[band])
    note = f"representative median-weather Tsys for Band {band}"
    if pwv_mm is not None and pwv_mm > 0:
        # Coarse proxy: opacity-driven Tsys growth with water column. Steeper
        # at higher frequency; ~+20%/mm above 1 mm at Band 6, ~+8%/mm at
        # Band 3, ~+40%/mm at Bands 8-10. Not the ASC atmospheric model.
        slope = 0.08 if band <= 3 else 0.12 if band <= 5 else 0.20 if band <= 7 else 0.40
        factor = max(0.6, 1.0 + slope * (float(pwv_mm) - 1.0))
        tsys *= factor
        note += f", scaled x{factor:.2f} for PWV={pwv_mm} mm (coarse proxy)"
    return tsys, note


def calculate_alma_sensitivity(
    band: int,
    bandwidth_ghz: float = 7.5,
    t_integration_s: float = 60.0,
    n_antennas: int = None,
    n_polarizations: int = 2,
    channel_width_khz: float = None,
    pwv_mm: float = 1.0,
    array: str = "12m",
    frequency_ghz: float = None,
    robust_weighting_factor: float = 1.0,
    shadowing_fraction: float = 0.0,
    tsys_k: float = None,
) -> Dict[str, Any]:
    """
    Estimate ALMA continuum and line point-source sensitivity per the ALMA
    Technical Handbook (Cycle 13) equation 9.8 (12-m / 7-m Arrays) or 9.11
    (Total Power Array). See the module comment above for the equations.

    Parameters
    ----------
    band : int
        ALMA band number (1-10).
    bandwidth_ghz : float
        Continuum bandwidth per polarization in GHz (Handbook: 7.5 GHz for
        continuum; 4 x 1.875 GHz SPWs).
    t_integration_s : float
        On-source integration time in seconds (no overheads).
    n_antennas : int
        Number of antennas. Defaults 43 (12-m), 10 (7-m), 3 (TP) — the ASC
        assumptions, NOT the array's full complement.
    n_polarizations : int
        1 for single polarization, 2 for dual/full polarization.
    channel_width_khz : float
        Spectral channel width in kHz for the line sensitivity (optional).
    pwv_mm : float
        Precipitable water vapour in mm; coarse Tsys scaling proxy (default 1.0).
    array : str
        '12m' (default), '7m' or 'TP'.
    frequency_ghz : float
        Observing frequency; defaults to the Table 9.3 continuum frequency of
        the band. Used for the channel-velocity conversion and band validation.
    robust_weighting_factor : float
        w_r in eq. 9.8 (1.0 = natural weighting; robust=0.5 is ~1.1-1.2,
        uniform ~1.5-2).
    shadowing_fraction : float
        f_s in eq. 9.8 (0-0.5), relevant for compact 12-m configs and the ACA
        at low elevation; ignored for TP.
    tsys_k : float
        Override the representative Tsys (e.g. the value the ASC reports).

    Returns
    -------
    dict
        Continuum (and optionally line) sensitivity in Jy/mJy/uJy per beam,
        every assumption used, the formula string that matches the arithmetic,
        and a note pointing to the official ALMA Sensitivity Calculator.
    """
    try:
        band = int(band)
    except (TypeError, ValueError):
        return {"success": False, "error": f"Band must be an integer 1-10, got {band!r}."}
    if band not in ALMA_TSYS:
        return {
            "success": False,
            "error": f"Band {band} not supported. Valid bands: {list(ALMA_TSYS.keys())}",
        }
    if t_integration_s is None or t_integration_s <= 0:
        return {"success": False, "error": "Integration time must be positive."}
    if bandwidth_ghz is None or bandwidth_ghz <= 0:
        return {"success": False, "error": "Bandwidth must be positive."}
    if n_polarizations not in (1, 2):
        return {"success": False, "error": "n_polarizations must be 1 (single) or 2 (dual/full)."}
    array_key = _normalize_array(array)
    if array_key is None:
        return {"success": False, "error": f"Unknown array {array!r}; use '12m', '7m' or 'TP'."}
    try:
        w_r = float(robust_weighting_factor if robust_weighting_factor is not None else 1.0)
        f_s = float(shadowing_fraction or 0.0)
    except (TypeError, ValueError):
        return {"success": False, "error": "robust_weighting_factor and shadowing_fraction must be numbers."}
    if w_r < 1.0:
        return {"success": False, "error": "robust_weighting_factor w_r must be >= 1 (1.0 = natural weighting)."}
    if not (0.0 <= f_s < 1.0):
        return {"success": False, "error": "shadowing_fraction must be in [0, 1)."}

    try:
        n_ant = int(n_antennas) if n_antennas else ALMA_DEFAULT_ANTENNAS[array_key]
        if array_key == "TP":
            if n_ant < 1:
                return {"success": False, "error": "TP needs at least one antenna."}
        elif n_ant < 2:
            return {"success": False, "error": "An interferometer needs at least two antennas."}

        table_freq, eta_12m, eta_7m = ALMA_TABLE_9_3[band]
        nu_ghz = float(frequency_ghz) if frequency_ghz else table_freq
        band_lo, band_hi = ALMA_BAND_LIMITS_GHZ[band]
        warnings_out = []
        if not (band_lo <= nu_ghz <= band_hi):
            warnings_out.append(
                f"frequency_ghz={nu_ghz} lies outside the nominal Band {band} range {band_lo}-{band_hi} GHz."
            )
        if band == 2:
            warnings_out.append("Band 2 is offered from Cycle 13 on the 12-m Array only; efficiencies are provisional.")
        if array_key == "TP" and bandwidth_ghz > 2.0 and not channel_width_khz:
            warnings_out.append("TP continuum sensitivity is over-optimistic (Handbook 9.2.2): only spectral-line TP is offered.")

        eta_ap = eta_7m if array_key == "7m" else eta_12m
        geom_area = ALMA_GEOMETRIC_AREA_M2[array_key]
        a_eff = eta_ap * geom_area  # m^2 per antenna

        if tsys_k is not None and float(tsys_k) > 0:
            tsys = float(tsys_k)
            tsys_note = "Tsys supplied by caller"
        else:
            tsys, tsys_note = _tsys_for(band, nu_ghz, pwv_mm)

        if array_key == "TP":
            baseline_term = float(n_ant)                      # eq. 9.11: sqrt(N)
            shadow_term = 1.0
            w_term = 1.0
            equation = "9.11"
            formula = ("dS_TP = 2*k*Tsys / (eta_q*eta_c*A_eff * sqrt(N * n_pol * dnu * t))")
        else:
            baseline_term = float(n_ant) * (n_ant - 1)        # eq. 9.8: sqrt(N(N-1)), NO /2
            shadow_term = 1.0 - f_s
            w_term = w_r
            equation = "9.8"
            formula = ("dS = w_r * 2*k*Tsys / (eta_q*eta_c*A_eff*(1-f_s) * sqrt(N*(N-1) * n_pol * dnu * t))")

        def _sigma_jy(delta_nu_hz: float) -> float:
            denominator = (ALMA_ETA_Q * ALMA_ETA_C * a_eff * shadow_term
                           * math.sqrt(baseline_term * n_polarizations * delta_nu_hz * t_integration_s))
            return w_term * 2.0 * K_B * tsys / denominator * 1e26  # W m^-2 Hz^-1 -> Jy

        delta_nu_cont = float(bandwidth_ghz) * 1e9
        sigma_cont_jy = _sigma_jy(delta_nu_cont)

        result = {
            "success": True,
            "band": band,
            "array": array_key,
            "frequency_ghz": nu_ghz,
            "tsys_k": round(tsys, 1),
            "tsys_note": tsys_note,
            "aperture_efficiency": eta_ap,
            "effective_area_m2_per_antenna": round(a_eff, 2),
            "quantization_efficiency": ALMA_ETA_Q,
            "correlator_efficiency": ALMA_ETA_C,
            "robust_weighting_factor": w_term,
            "shadowing_fraction": f_s if array_key != "TP" else 0.0,
            "n_antennas": n_ant,
            "n_baselines": int(n_ant * (n_ant - 1) / 2) if array_key != "TP" else 0,
            "n_polarizations": n_polarizations,
            "bandwidth_ghz": bandwidth_ghz,
            "integration_time_s": t_integration_s,
            "integration_time_min": round(t_integration_s / 60, 2),
            "pwv_mm": pwv_mm,
            "continuum_sensitivity_jy_beam": sigma_cont_jy,
            "continuum_sensitivity_mjy_beam": round(sigma_cont_jy * 1e3, 4),
            "continuum_sensitivity_ujy_beam": round(sigma_cont_jy * 1e6, 1),
            "equation": f"Technical Handbook eq. {equation}",
            "formula": formula,
            "assumptions": {
                "handbook": ALMA_HANDBOOK_VERSION,
                "eta_q": ALMA_ETA_Q,
                "eta_c": ALMA_ETA_C,
                "aperture_efficiency_table": "Table 9.3",
                "default_antennas": ALMA_DEFAULT_ANTENNAS,
                "geometric_area_m2": geom_area,
                "tsys_source": tsys_note,
                "no_overheads": "on-source time only; calibration/latency overheads are not included",
            },
            "warnings": warnings_out,
            "note": (
                "Theoretical estimate following the Technical Handbook radiometer equation; Tsys here "
                "is a representative median-weather value with a coarse PWV proxy. The official ALMA "
                "Sensitivity Calculator (Trx + atmospheric model at the PWV octile and source elevation) "
                "is authoritative for proposals; imaging losses (dynamic range, flagging, pointing) are "
                "not modelled."
            ),
        }

        if channel_width_khz is not None and channel_width_khz > 0:
            delta_nu_line = float(channel_width_khz) * 1e3
            sigma_line_jy = _sigma_jy(delta_nu_line)
            channel_vel_kms = (float(channel_width_khz) * 1e-6 / nu_ghz) * C_KMS
            result["channel_width_khz"] = channel_width_khz
            result["channel_velocity_km_s"] = round(channel_vel_kms, 3)
            result["line_sensitivity_jy_beam"] = sigma_line_jy
            result["line_sensitivity_mjy_beam"] = round(sigma_line_jy * 1e3, 4)
            result["line_sensitivity_ujy_beam"] = round(sigma_line_jy * 1e6, 1)

        return result

    except Exception as e:
        logger.error("ALMA sensitivity calculation failed: %s", e)
        return {"success": False, "error": str(e)}


def brightness_temperature_sensitivity_k(sigma_jy: float, frequency_ghz: float, beam_fwhm_arcsec: float) -> float:
    """Surface-brightness sensitivity (K) from a point-source sensitivity (Jy)
    for a circular Gaussian beam of FWHM theta (Handbook eq. 9.9 / 9.10):
    dT = dS lambda^2 / (2 k Omega), Omega = pi theta^2 / (4 ln 2)."""
    lam_m = 299792458.0 / (float(frequency_ghz) * 1e9)
    theta_rad = math.radians(float(beam_fwhm_arcsec) / 3600.0)
    omega = math.pi * theta_rad ** 2 / (4.0 * math.log(2.0))
    return float(sigma_jy) * 1e-26 * lam_m ** 2 / (2.0 * K_B * omega)


# ═══════════════════════════════════════════════════════════════════════════
# Doppler / spectral-frame helper (A-72)
# ═══════════════════════════════════════════════════════════════════════════

DOPPLER_CONVENTIONS = ("radio", "optical", "relativistic")


def calculate_doppler_shift(
    rest_frequency_ghz: float = None,
    observed_frequency_ghz: float = None,
    redshift: float = None,
    velocity_kms: float = None,
    convention: str = "radio",
    frame: str = "LSRK",
) -> Dict[str, Any]:
    """
    Convert between rest frequency, observed (sky) frequency, redshift and
    velocity, stating the Doppler convention and the reference frame.

    Give the rest frequency plus ONE of (observed_frequency_ghz, redshift,
    velocity_kms), or observed + redshift/velocity to recover the rest
    frequency. Conventions (all exact for the quantity they define):

      radio:         v = c (nu_rest - nu_obs) / nu_rest
      optical:       v = c (nu_rest - nu_obs) / nu_obs = c z
      relativistic:  v = c (nu_rest^2 - nu_obs^2) / (nu_rest^2 + nu_obs^2)

    ``frame`` is a label carried through for disclosure (ALMA products are
    typically LSRK; native visibilities are TOPO per execution block via
    Doppler setting, so cross-EB frequencies differ until regridded).
    """
    conv = str(convention or "radio").strip().lower()
    if conv not in DOPPLER_CONVENTIONS:
        return {"success": False, "error": f"convention must be one of {DOPPLER_CONVENTIONS}"}
    c = C_KMS
    try:
        given = {k: v for k, v in {
            "rest_frequency_ghz": rest_frequency_ghz,
            "observed_frequency_ghz": observed_frequency_ghz,
            "redshift": redshift,
            "velocity_kms": velocity_kms,
        }.items() if v is not None}
        nu_rest = float(rest_frequency_ghz) if rest_frequency_ghz is not None else None
        nu_obs = float(observed_frequency_ghz) if observed_frequency_ghz is not None else None
        z = float(redshift) if redshift is not None else None
        v = float(velocity_kms) if velocity_kms is not None else None

        # Velocity -> ratio nu_obs/nu_rest under the convention.
        def ratio_from_velocity(vel: float) -> float:
            beta = vel / c
            if conv == "radio":
                return 1.0 - beta
            if conv == "optical":
                return 1.0 / (1.0 + beta)
            if abs(beta) >= 1.0:
                raise ValueError("relativistic velocity must be |v| < c")
            return math.sqrt((1.0 - beta) / (1.0 + beta))

        if nu_rest is not None and nu_obs is None:
            if z is not None:
                nu_obs = nu_rest / (1.0 + z)
            elif v is not None:
                nu_obs = nu_rest * ratio_from_velocity(v)
            else:
                return {"success": False, "error": "Give observed_frequency_ghz, redshift or velocity_kms with the rest frequency."}
        elif nu_obs is not None and nu_rest is None:
            if z is not None:
                nu_rest = nu_obs * (1.0 + z)
            elif v is not None:
                nu_rest = nu_obs / ratio_from_velocity(v)
            else:
                return {"success": False, "error": "Give rest_frequency_ghz, redshift or velocity_kms with the observed frequency."}
        elif nu_rest is None and nu_obs is None:
            return {"success": False, "error": "Give at least one of rest_frequency_ghz / observed_frequency_ghz."}
        if nu_rest <= 0 or nu_obs <= 0:
            return {"success": False, "error": "Frequencies must be positive."}

        z_out = nu_rest / nu_obs - 1.0
        v_radio = c * (nu_rest - nu_obs) / nu_rest
        v_optical = c * (nu_rest - nu_obs) / nu_obs
        v_rel = c * (nu_rest ** 2 - nu_obs ** 2) / (nu_rest ** 2 + nu_obs ** 2)
        velocities = {"radio": v_radio, "optical": v_optical, "relativistic": v_rel}
        return {
            "success": True,
            "given": given,
            "convention": conv,
            "frame": str(frame or "LSRK").upper(),
            "rest_frequency_ghz": round(nu_rest, 9),
            "observed_frequency_ghz": round(nu_obs, 9),
            "redshift": round(z_out, 9),
            "velocity_kms": round(velocities[conv], 4),
            "velocity_kms_by_convention": {k: round(val, 4) for k, val in velocities.items()},
            "channel_kms_per_mhz_at_observed": round((1e-3 / nu_obs) * c, 6),
            "note": (
                "Redshift z = nu_rest/nu_obs - 1 is convention-free; velocities differ by convention "
                "(radio and optical diverge at high z). ALMA spectral products are typically LSRK; "
                "native ALMA visibilities are TOPO per execution block (Doppler setting), so sky "
                "frequencies shift between EBs and must be regridded to a common frame."
            ),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
