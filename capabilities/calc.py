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
    """Resolve target name to RA/Dec using SIMBAD (Fix 3)"""
    try:
        from astroquery.simbad import Simbad
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        result = Simbad.query_object(target_name)

        if result is None or len(result) == 0:
            return {
                "success": False,
                "error": f"SIMBAD could not resolve '{target_name}'. Check spelling or try alternate designation."
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

        return {
            "success": True,
            "target_name": target_name,
            "ra_deg": round(coord.ra.deg, 6),
            "dec_deg": round(coord.dec.deg, 6),
            "message": f"Resolved '{target_name}' to RA={coord.ra.deg:.4f}°, Dec={coord.dec.deg:.4f}°. Use search_by_position with these coordinates."
        }
    except ImportError:
        return {"success": False, "error": "astroquery not installed. Cannot resolve target names."}
    except Exception as e:
        return {"success": False, "error": f"Resolution failed: {str(e)}"}


class _In(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ResolveTargetInput(_In):
    target_name: Optional[str]


class ResolveTarget(BaseCapability):
    name = "resolve_target"
    description = (
        "Resolve a target name to RA/Dec coordinates using SIMBAD. Use this if "
        "search_by_target returns empty for a valid target name."
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


class CalculateAlmaSensitivity(BaseCapability):
    name = "calculate_alma_sensitivity"
    description = (
        "Estimate ALMA continuum and spectral line sensitivity using "
        "the radiometer equation. Returns noise level in mJy/beam and "
        "uJy/beam for given band, bandwidth, and integration time. "
        "Includes Tsys scaling for weather (PWV). Use when the user "
        "asks about ALMA sensitivity, noise levels, or integration "
        "time estimates."
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
            ))
        except Exception as e:
            return _native({"success": False, "error": str(e)})


CAPABILITIES: List[BaseCapability] = [
    ResolveTarget(),
    CalculateRedshift(),
    ConvertCoordinates(),
    CalculateBeam(),
    CalculateAlmaSensitivity(),
]

__all__ = [
    "CAPABILITIES",
    "resolve_target",
    "ResolveTarget", "CalculateRedshift", "ConvertCoordinates",
    "CalculateBeam", "CalculateAlmaSensitivity",
]
