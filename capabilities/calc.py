"""
capabilities/calc.py — the astronomy calculator / target-resolution family as
transport-pure capabilities (P1 family migration #7).

Logic relocated VERBATIM from ``core/agent.py`` (``_calculate_redshift`` /
``_convert_coordinates`` / ``_calculate_beam`` / ``_calculate_alma_sensitivity``
/ ``_resolve_target``). These are pure computations over
``services.astro_calculators`` plus the SIMBAD resolver — no clients, no
result-state writes, so the CallContext is unused beyond validation.

``resolve_target`` is ALSO an internal dependency of other families (the ALMA
capabilities' dead positional fallback + CADC fallback, ``_datalab_coordinates``
and ``_live_imagery_coordinates`` on the agent). The single implementation is
the module function :func:`resolve_target` below; the agent keeps a thin
``_resolve_target`` delegate over it for those injection sites, and the
``ResolveTarget`` capability calls it directly — one body, two entry points.

None of these methods carried ``@log_tool``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from capabilities.base import BaseCapability, ToolResult
from services.astro_calculators import (
    calculate_alma_sensitivity,
    calculate_beam,
    calculate_doppler_shift,
    calculate_redshift,
    convert_coordinates,
)

logger = logging.getLogger(__name__)


def _native(out: Dict[str, Any]) -> ToolResult:
    """Wrap a legacy output dict as a byte-parity ToolResult."""
    ok = bool(isinstance(out, dict) and out.get("success"))
    err = out.get("error") if isinstance(out, dict) else None
    return ToolResult(
        success=ok,
        error=(str(err) if (err is not None and not ok) else None),
        native=out,
    )


def resolve_target(target_name: str) -> Dict[str, Any]:
    """Resolve a target name to ICRS RA/Dec in degrees.

    Resolution order:
      1. CDS Sesame through ``astropy.coordinates.SkyCoord.from_name`` — the
         astropy-ecosystem standard resolver (SIMBAD, then NED, then VizieR).
         Any astropy/pyvo user resolving the same name obtains byte-identical
         coordinates, so archive requests Quasar derives from them (SIA POS /
         SIZE, cone centres) reproduce exactly.
      2. SIMBAD via astroquery when Sesame is unreachable or has no match.

    Coordinates are returned exactly as the resolver provides them. They used to
    be rounded to six decimals (~4 mas); that silently changed every derived
    request parameter and made Quasar's archive queries irreproducible against
    the same query issued directly.
    """
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = None
        resolver = None
        sesame_error = None
        try:
            coord = SkyCoord.from_name(target_name)
            resolver = "CDS Sesame"
        except Exception as exc:  # NameResolveError, network failure, bad input
            sesame_error = f"{type(exc).__name__}: {exc}"

        if coord is None:
            try:
                from astroquery.simbad import Simbad
            except ImportError:
                return {"success": False, "error": "astroquery not installed. Cannot resolve target names."}

            try:
                result = Simbad.query_object(target_name)
            except Exception as exc:
                detail = f" (Sesame: {sesame_error})" if sesame_error else ""
                return {"success": False,
                        "error": f"Resolution failed: SIMBAD: {type(exc).__name__}: {exc}{detail}"}
            if result is None or len(result) == 0:
                detail = f" (Sesame: {sesame_error})" if sesame_error else ""
                return {
                    "success": False,
                    "error": (
                        f"SIMBAD could not resolve '{target_name}'{detail}. "
                        "Check spelling or try alternate designation."
                    ),
                }

            # Get coordinates from first match. astroquery <0.4.8 returns
            # 'RA'/'DEC' sexagesimal strings; newer versions return lowercase
            # 'ra'/'dec' already in degrees.
            cols = {c.lower(): c for c in result.colnames}
            ra_val = result[cols['ra']][0]
            dec_val = result[cols['dec']][0]
            try:
                coord = SkyCoord(ra=float(ra_val) * u.deg, dec=float(dec_val) * u.deg)
            except (TypeError, ValueError):
                coord = SkyCoord(str(ra_val), str(dec_val), unit=(u.hourangle, u.deg))
            resolver = "SIMBAD"

        ra_deg = float(coord.ra.deg)
        dec_deg = float(coord.dec.deg)
        return {
            "success": True,
            "target_name": target_name,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "resolver": resolver,
            "message": (
                f"Resolved '{target_name}' via {resolver} to RA={ra_deg:.6f}°, Dec={dec_deg:.6f}° "
                "(full-precision values in ra_deg/dec_deg). Use search_by_position with these coordinates."
            ),
        }
    except ImportError:
        return {"success": False, "error": "astropy/astroquery not installed. Cannot resolve target names."}
    except Exception as e:
        return {"success": False, "error": f"Resolution failed: {str(e)}"}


class _In(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ResolveTargetInput(_In):
    target_name: Optional[str]


class ResolveTarget(BaseCapability):
    name = "resolve_target"
    description = (
        "Resolve a target name to ICRS RA/Dec degrees (CDS Sesame: SIMBAD, then NED, then VizieR; "
        "SIMBAD fallback). Returns full-precision coordinates and the resolver used. Use this "
        "before positional searches or if search_by_target returns empty for a valid name."
    )
    category = "general"
    InputModel = ResolveTargetInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        return _native(resolve_target(inp.target_name))


class CalculateRedshiftInput(_In):
    z: Optional[float]


class CalculateRedshift(BaseCapability):
    name = "calculate_redshift"
    description = (
        "Compute cosmological quantities for a given redshift z using "
        "Planck18 cosmology. Returns luminosity distance, angular diameter "
        "distance, comoving distance, lookback time, age of the universe "
        "at that epoch, and the physical scale (kpc per arcsecond). "
        "Use for any question about distances, ages, or scales at a "
        "given redshift."
    )
    category = "analysis"
    InputModel = CalculateRedshiftInput
    annotations = {"read_only": True, "cost": "cpu"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            return _native(calculate_redshift(inp.z))
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class ConvertCoordinatesInput(_In):
    ra: Optional[float] = None
    dec: Optional[float] = None
    l: Optional[float] = None
    b: Optional[float] = None
    input_frame: Optional[str] = "icrs"
    output_frame: Optional[str] = "galactic"


class ConvertCoordinates(BaseCapability):
    name = "convert_coordinates"
    description = (
        "Convert sky coordinates between ICRS (RA/Dec), Galactic (l/b), "
        "Ecliptic (lon/lat), FK5 (J2000), and FK4 (B1950) frames. "
        "Returns the position in ALL frames at once. "
        "Use for coordinate transformations, epoch precession, or when "
        "the user gives Galactic coordinates and needs RA/Dec."
    )
    category = "analysis"
    InputModel = ConvertCoordinatesInput
    annotations = {"read_only": True, "cost": "cpu"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            return _native(convert_coordinates(
                ra=inp.ra, dec=inp.dec, l=inp.l, b=inp.b,
                input_frame=inp.input_frame, output_frame=inp.output_frame,
            ))
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class CalculateBeamInput(_In):
    frequency_ghz: Optional[float]
    max_baseline_m: Optional[float] = None
    array_config: Optional[str] = None


class CalculateBeam(BaseCapability):
    name = "calculate_beam"
    description = (
        "Calculate the synthesized beam size for a radio interferometer "
        "given the maximum baseline and observing frequency. For ALMA, "
        "you can specify an array configuration name (C-1 through C-10) "
        "instead of a raw baseline length. Returns beam size in arcsec "
        "and milliarcsec."
    )
    category = "analysis"
    InputModel = CalculateBeamInput
    annotations = {"read_only": True, "cost": "cpu"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            return _native(calculate_beam(
                frequency_ghz=inp.frequency_ghz,
                max_baseline_m=inp.max_baseline_m,
                array_config=inp.array_config,
            ))
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class CalculateAlmaSensitivityInput(_In):
    band: Optional[int]
    bandwidth_ghz: Optional[float] = 7.5
    t_integration_s: Optional[float] = 60.0
    n_antennas: Optional[int] = None
    n_polarizations: Optional[int] = 2
    channel_width_khz: Optional[float] = None
    pwv_mm: Optional[float] = 1.0
    array: Optional[str] = "12m"
    frequency_ghz: Optional[float] = None
    robust_weighting_factor: Optional[float] = 1.0
    shadowing_fraction: Optional[float] = 0.0
    tsys_k: Optional[float] = None


class CalculateAlmaSensitivity(BaseCapability):
    name = "calculate_alma_sensitivity"
    description = (
        "Estimate ALMA point-source sensitivity with the Technical Handbook radiometer "
        "equation (eq. 9.8 for the 12-m/7-m Arrays, eq. 9.11 for Total Power): "
        "quantization 0.96 and correlator 0.88 efficiencies, Table 9.3 aperture "
        "efficiencies, N(N-1) baselines, default 43/10/3 antennas. Returns continuum "
        "(and optional line) rms in mJy/beam and uJy/beam with every assumption stated. "
        "Bands 1-10; array '12m'|'7m'|'TP'. The official ALMA Sensitivity Calculator "
        "remains authoritative for proposals (real Tsys from PWV octile + elevation)."
    )
    category = "analysis"
    InputModel = CalculateAlmaSensitivityInput
    annotations = {"read_only": True, "cost": "cpu"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            return _native(calculate_alma_sensitivity(
                band=inp.band, bandwidth_ghz=inp.bandwidth_ghz,
                t_integration_s=inp.t_integration_s, n_antennas=inp.n_antennas,
                n_polarizations=inp.n_polarizations,
                channel_width_khz=inp.channel_width_khz, pwv_mm=inp.pwv_mm,
                array=inp.array or "12m", frequency_ghz=inp.frequency_ghz,
                robust_weighting_factor=inp.robust_weighting_factor if inp.robust_weighting_factor is not None else 1.0,
                shadowing_fraction=inp.shadowing_fraction or 0.0,
                tsys_k=inp.tsys_k,
            ))
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class CalculateDopplerShiftInput(_In):
    rest_frequency_ghz: Optional[float] = None
    observed_frequency_ghz: Optional[float] = None
    redshift: Optional[float] = None
    velocity_kms: Optional[float] = None
    convention: Optional[str] = "radio"
    frame: Optional[str] = "LSRK"


class CalculateDopplerShift(BaseCapability):
    name = "calculate_doppler_shift"
    description = (
        "Convert between rest frequency, observed (sky) frequency, redshift and velocity "
        "with an explicit Doppler convention (radio | optical | relativistic) and reference-"
        "frame label (LSRK default; ALMA native visibilities are TOPO per execution block). "
        "Give rest_frequency_ghz plus one of observed_frequency_ghz / redshift / velocity_kms "
        "(or observed + redshift/velocity to recover the rest frequency). Use before any "
        "line-coverage or channel-width-to-velocity statement."
    )
    category = "analysis"
    InputModel = CalculateDopplerShiftInput
    annotations = {"read_only": True, "cost": "cpu"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            return _native(calculate_doppler_shift(
                rest_frequency_ghz=inp.rest_frequency_ghz,
                observed_frequency_ghz=inp.observed_frequency_ghz,
                redshift=inp.redshift, velocity_kms=inp.velocity_kms,
                convention=inp.convention or "radio", frame=inp.frame or "LSRK",
            ))
        except Exception as e:
            return _native({"success": False, "error": str(e)})


CAPABILITIES: List[BaseCapability] = [
    ResolveTarget(),
    CalculateRedshift(),
    ConvertCoordinates(),
    CalculateBeam(),
    CalculateAlmaSensitivity(),
    CalculateDopplerShift(),
]

__all__ = [
    "CAPABILITIES",
    "resolve_target",
    "ResolveTarget", "CalculateRedshift", "ConvertCoordinates",
    "CalculateBeam", "CalculateAlmaSensitivity", "CalculateDopplerShift",
]
