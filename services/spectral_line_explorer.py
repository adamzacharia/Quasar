"""Shared spectral-line workflows for the API, UI, workbench, and chat."""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.parse import quote_plus, urlencode

import pandas as pd
from diskcache import Cache

from services.alma_science_queries import (
    observation_intervals_ghz,
    observation_windows_ghz,
)
from services.splatalogue import (
    ALMA_BAND_LIMITS_GHZ,
    SPLATALOGUE_SPECIES_URL,
    SpectralLineQuery,
    SpectralWindow,
    SplatalogueQueryCancelled,
    SplatalogueTool,
    formula_matches,
    transition_matches,
)


SPEED_OF_LIGHT_KMS = 299792.458
SPECTRAL_LINE_JOB_STATUSES = {
    "queued",
    "running",
    "needs_input",
    "succeeded",
    "partial",
    "failed",
    "canceled",
    "orphaned",
}
TERMINAL_JOB_STATUSES = {
    "needs_input",
    "succeeded",
    "partial",
    "failed",
    "canceled",
    "orphaned",
}
DEFAULT_DISPLAY_FIELDS = [
    "formula",
    "chemical_name",
    "resolved_quantum_numbers",
    "frequency_ghz",
    "predicted_frequency_ghz",
    "measured_frequency_ghz",
    "frequency_uncertainty_mhz",
    "observed_frequency_ghz",
    "lower_energy_k",
    "upper_energy_k",
    "log_intensity",
    "lovas_astronomical_intensity",
    "catalogs",
]
logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def spectral_line_explorer_enabled() -> bool:
    """Single source of truth for the Spectral Line Explorer feature flag.

    Enabled by default; disabled only when ENABLE_SPECTRAL_LINE_EXPLORER is set
    to an explicit falsy token. Both the API route gate (ui-pro/api/main.py) and
    metadata() must use this so the reported ``enabled`` flag can never disagree
    with the actual 404 gate.
    """
    value = os.getenv("ENABLE_SPECTRAL_LINE_EXPLORER", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def radial_velocity_to_redshift(
    velocity_kms: float, convention: str = "radio"
) -> float:
    beta = float(velocity_kms) / SPEED_OF_LIGHT_KMS
    convention = convention.lower()
    if convention == "radio":
        if beta >= 1:
            raise ValueError("Radio velocity must be below the speed of light")
        return beta / (1 - beta)
    if convention == "optical":
        return beta
    if convention == "relativistic":
        if abs(beta) >= 1:
            raise ValueError("Relativistic velocity magnitude must be below c")
        return math.sqrt((1 + beta) / (1 - beta)) - 1
    raise ValueError("velocity convention must be radio, optical, or relativistic")


def query_redshift(query: SpectralLineQuery) -> Optional[float]:
    if query.redshift is not None:
        return query.redshift
    if query.radial_velocity_kms is not None:
        return radial_velocity_to_redshift(
            query.radial_velocity_kms, query.velocity_convention
        )
    return None


def apply_frequency_frame(
    line: Mapping[str, Any], redshift: Optional[float]
) -> Dict[str, Any]:
    result = dict(line)
    rest = _finite_float(result.get("frequency_ghz"))
    if rest is not None and redshift is not None:
        if redshift <= -1:
            raise ValueError("redshift must be greater than -1")
        result["observed_frequency_ghz"] = rest / (1 + redshift)
        result["redshift"] = redshift
    elif rest is not None and result.get("observed_frequency_ghz") is None:
        result["observed_frequency_ghz"] = rest
    return result


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class SpeciesMetadataCache:
    """Twenty-four-hour, stale-on-error cache for Splatalogue species metadata."""

    CACHE_KEY = "splatalogue_species_v1"

    def __init__(
        self,
        cache_dir: Path | str = Path("cache") / "spectral_line_species",
        *,
        ttl_seconds: int = 24 * 60 * 60,
    ):
        self.cache = Cache(str(cache_dir))
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()

    def all(self, *, force_refresh: bool = False) -> List[Dict[str, Any]]:
        cached = self.cache.get(self.CACHE_KEY)
        refreshed_at = self.cache.get(f"{self.CACHE_KEY}:refreshed_at", 0)
        fresh = bool(cached) and time.time() - float(refreshed_at or 0) < self.ttl_seconds
        if fresh and not force_refresh:
            return list(cached)
        with self._lock:
            cached = self.cache.get(self.CACHE_KEY)
            refreshed_at = self.cache.get(f"{self.CACHE_KEY}:refreshed_at", 0)
            fresh = bool(cached) and time.time() - float(refreshed_at or 0) < self.ttl_seconds
            if fresh and not force_refresh:
                return list(cached)
            try:
                import requests

                response = requests.get(SPLATALOGUE_SPECIES_URL, timeout=30)
                response.raise_for_status()
                payload = response.json()
                records = []
                for item in payload:
                    if not isinstance(item, Mapping) or item.get("species_id") is None:
                        continue
                    try:
                        records.append(self._normalize(item))
                    except (TypeError, ValueError):
                        continue
                with self.cache.transact():
                    self.cache.set(self.CACHE_KEY, records)
                    self.cache.set(f"{self.CACHE_KEY}:refreshed_at", time.time())
                return records
            except Exception:
                logger.warning(
                    "Splatalogue species refresh failed; serving stale cache=%s",
                    bool(cached),
                    exc_info=True,
                )
                if cached:
                    return list(cached)
                raise

    def search(self, query: str = "", limit: int = 25) -> List[Dict[str, Any]]:
        needle = "".join(char for char in query.lower() if char.isalnum())
        matches = []
        for item in self.all():
            raw_formula = str(item.get("formula") or "").strip().lower()
            base_formula = "".join(
                char
                for char in (raw_formula.split()[0] if raw_formula else "")
                if char.isalnum()
            )
            formula = "".join(
                char
                for char in raw_formula
                if char.isalnum()
            )
            chemical = "".join(
                char
                for char in str(item.get("chemical_name") or "").lower()
                if char.isalnum()
            )
            tag = "".join(
                char
                for char in str(item.get("tag") or "").lower()
                if char.isalnum()
            )
            species_id = str(item.get("species_id") or "")
            if not needle:
                score = (5, formula)
            elif needle in {base_formula, formula, species_id, tag}:
                score = (0, formula)
            elif formula.startswith(needle):
                score = (1, formula)
            elif needle in formula:
                score = (2, formula)
            elif chemical.startswith(needle):
                score = (3, chemical)
            elif needle in chemical:
                score = (4, chemical)
            else:
                continue
            matches.append((score, item))
        matches.sort(key=lambda match: match[0])
        return [
            item
            for _, item in matches[: max(1, min(int(limit or 25), 100))]
        ]

    @staticmethod
    def _normalize(item: Mapping[str, Any]) -> Dict[str, Any]:
        formula = SplatalogueTool._clean_text(item.get("s_name") or item.get("htmlname"))
        chemical_name = str(item.get("chemical_name") or "").strip()
        if _truthy(item.get("know_ast_molecules")):
            status = "known"
        elif _truthy(item.get("probable")):
            status = "probable"
        elif _truthy(item.get("potential")):
            status = "potential"
        elif _truthy(item.get("atmos")):
            status = "atmospheric"
        else:
            status = "unknown"
        tag = str(item.get("SPLAT_ID") or "").strip()
        return {
            "species_id": int(item["species_id"]),
            "tag": tag,
            "formula": formula,
            "chemical_name": chemical_name,
            "molecular_mass": _mass_from_tag(tag),
            "status": status,
            "label": f"{formula} — {chemical_name}" if chemical_name else formula,
        }


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _mass_from_tag(tag: str) -> Optional[int]:
    digits = "".join(char for char in str(tag) if char.isdigit())
    if len(digits) < 3:
        return None
    try:
        return int(digits[:3])
    except ValueError:
        return None


# Decimal-degree coordinate pair, space or comma separated ("150.096 +2.220").
_COORD_PAIR_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)[\s,]+([+-]?\d+(?:\.\d+)?)\s*$")


def _parse_coordinate_pair(text: str) -> Optional[tuple[float, float]]:
    """Parse an ICRS decimal-degree "RA DEC" string, or None if it isn't one."""
    match = _COORD_PAIR_RE.match(str(text or ""))
    if not match:
        return None
    ra, dec = float(match.group(1)), float(match.group(2))
    if 0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0:
        return ra, dec
    return None


class TargetResolver:
    """Resolve coordinates and redshift from SIMBAD and NED with provenance."""

    def resolve(
        self,
        target_name: str,
        *,
        explicit_redshift: Optional[float] = None,
        explicit_ra_deg: Optional[float] = None,
        explicit_dec_deg: Optional[float] = None,
    ) -> Dict[str, Any]:
        target = str(target_name or "").strip()
        if not target and (explicit_ra_deg is None or explicit_dec_deg is None):
            raise ValueError("target_name or explicit coordinates are required")
        # Coordinate-pair targets ("150.09561 +2.20013") short-circuit name
        # resolution: SIMBAD/NED name lookups cannot resolve raw coordinate
        # strings, and the SPARCL→SLE deep link hands exactly this form.
        # Redshift must then come from explicit_redshift (the deep link carries
        # it) or the user — state falls to needs_input otherwise. (guard CX-09)
        coordinate_pair = _parse_coordinate_pair(target) if target else None
        if coordinate_pair is not None and (explicit_ra_deg is None or explicit_dec_deg is None):
            explicit_ra_deg, explicit_dec_deg = coordinate_pair
        run_name_queries = bool(target) and coordinate_pair is None
        queried_at = utc_now_iso()
        executor = ThreadPoolExecutor(max_workers=2)
        try:
            simbad_future = executor.submit(self._query_simbad, target) if run_name_queries else None
            ned_future = executor.submit(self._query_ned, target) if run_name_queries else None
            deadline = time.monotonic() + 45
            simbad = self._future_result(
                simbad_future,
                "SIMBAD",
                timeout=max(0.1, deadline - time.monotonic()),
            )
            ned = self._future_result(
                ned_future,
                "NED",
                timeout=max(0.1, deadline - time.monotonic()),
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        ra = explicit_ra_deg
        dec = explicit_dec_deg
        coordinate_source = "explicit"
        if ra is None or dec is None:
            if simbad.get("ra_deg") is not None and simbad.get("dec_deg") is not None:
                ra, dec, coordinate_source = simbad["ra_deg"], simbad["dec_deg"], "SIMBAD"
            elif ned.get("ra_deg") is not None and ned.get("dec_deg") is not None:
                ra, dec, coordinate_source = ned["ra_deg"], ned["dec_deg"], "NED"

        if explicit_redshift is not None:
            selected_redshift = float(explicit_redshift)
            redshift_source = "explicit"
            state = "resolved"
        else:
            selected_redshift, redshift_source, state = self._select_redshift(simbad, ned)

        if selected_redshift is not None and selected_redshift <= -1:
            raise ValueError("redshift must be greater than -1")
        return {
            "target_name": target,
            "ra_deg": ra,
            "dec_deg": dec,
            "coordinate_source": coordinate_source if ra is not None else None,
            "redshift": selected_redshift,
            "redshift_source": redshift_source,
            "state": state,
            "candidates": {"simbad": simbad, "ned": ned},
            "queried_at": queried_at,
        }

    @staticmethod
    def _select_redshift(
        simbad: Mapping[str, Any], ned: Mapping[str, Any]
    ) -> tuple[Optional[float], Optional[str], str]:
        z_simbad = _finite_float(simbad.get("redshift"))
        z_ned = _finite_float(ned.get("redshift"))
        if z_simbad is not None and z_ned is not None:
            tolerance = max(0.001, 0.02 * max(abs(z_simbad), abs(z_ned)))
            if abs(z_simbad - z_ned) <= tolerance:
                return z_simbad, "SIMBAD (agrees with NED)", "resolved"
            return None, None, "needs_input"
        if z_simbad is not None:
            return z_simbad, "SIMBAD", "resolved"
        if z_ned is not None:
            return z_ned, "NED", "resolved"
        return None, None, "needs_input"

    @staticmethod
    def _future_result(
        future: Any, service: str, *, timeout: float = 45
    ) -> Dict[str, Any]:
        if future is None:
            return {"service": service}
        try:
            return future.result(timeout=timeout)
        except Exception as exc:
            return {"service": service, "error": str(exc)}

    @staticmethod
    def _query_simbad(target: str) -> Dict[str, Any]:
        from astroquery.simbad import Simbad
        from astropy.coordinates import SkyCoord
        from astropy import units as u

        client = Simbad()
        for field in ("rvz_redshift", "rvz_radvel"):
            try:
                client.add_votable_fields(field)
            except Exception:
                pass
        table = client.query_object(target)
        if table is None or len(table) == 0:
            return {"service": "SIMBAD"}
        row = table[0]
        names = list(table.colnames)
        ra = _first_numeric(row, names, ["ra", "RA_d", "ra_d"])
        dec = _first_numeric(row, names, ["dec", "DEC_d", "dec_d"])
        if ra is None or dec is None:
            ra_text = _first_value(row, names, ["RA"])
            dec_text = _first_value(row, names, ["DEC"])
            if ra_text is not None and dec_text is not None:
                coord = SkyCoord(str(ra_text), str(dec_text), unit=(u.hourangle, u.deg))
                ra, dec = coord.ra.deg, coord.dec.deg
        redshift = _first_numeric(
            row,
            names,
            [
                "redshift",
                "z_value",
                "rvz_redshift",
                "RVZ_REDSHIFT",
                "Z_VALUE",
            ],
        )
        return {
            "service": "SIMBAD",
            "ra_deg": ra,
            "dec_deg": dec,
            "redshift": redshift,
        }

    @staticmethod
    def _query_ned(target: str) -> Dict[str, Any]:
        from astroquery.ipac.ned import Ned

        table = Ned.query_object(target)
        if table is None or len(table) == 0:
            return {"service": "NED"}
        row = table[0]
        names = list(table.colnames)
        return {
            "service": "NED",
            "ra_deg": _first_numeric(row, names, ["RA", "ra"]),
            "dec_deg": _first_numeric(row, names, ["DEC", "dec"]),
            "redshift": _first_numeric(row, names, ["Redshift", "redshift", "z"]),
        }


def _first_value(row: Any, names: Sequence[str], candidates: Sequence[str]) -> Any:
    lookup = {name.lower(): name for name in names}
    for candidate in candidates:
        name = lookup.get(candidate.lower())
        if name is None:
            continue
        value = row[name]
        if value is not None and not getattr(value, "mask", False):
            return value
    return None


def _first_numeric(
    row: Any, names: Sequence[str], candidates: Sequence[str]
) -> Optional[float]:
    return _finite_float(_first_value(row, names, candidates))


# ALMA array centre (used as the EarthLocation fallback so we never depend on a
# network site-registry lookup). Values per the IAU/ALMA reference position.
_ALMA_LON_DEG = -67.7549
_ALMA_LAT_DEG = -23.0229
_ALMA_HEIGHT_M = 5058.7

# Bounded safety margin (km/s) used to widen the ADQL spectral pre-filter when no
# explicit Doppler margin was supplied, so the TOPO-vs-source frame mismatch can
# never exclude a genuinely-covering observation before per-row correction.
_FRAME_PREFILTER_CAP_KMS = 40.0


def _resolve_target_frame_name(doppler_frame: Any) -> Optional[str]:
    """Map a human Doppler-frame label to an astropy frame for correction.

    Returns an astropy frame name suitable for
    ``SpectralCoord.with_observer_stationary_relative_to`` ("lsrk" or "icrs"),
    or ``None`` when no correction should be applied (already topocentric).
    """
    text = str(doppler_frame or "").strip().lower()
    if "topo" in text:
        return None
    if any(token in text for token in ("bary", "icrs", "helio")):
        # SpectralCoord treats an ICRS-stationary observer as barycentric;
        # heliocentric differs by ~0.01 km/s (negligible here).
        return "icrs"
    # Default (incl. "observed (source frame)") → kinematic radio LSRK, which is
    # the convention CASA/ALMA velocities are quoted in.
    return "lsrk"


def compute_topo_to_frame_offset_kms(
    ra_deg: float,
    dec_deg: float,
    obs_mjd: float,
    target_frame: str = "lsrk",
) -> Optional[float]:
    """Per-observation TOPO→target-frame radial-velocity offset, in km/s.

    Returns ``v`` such that ``f_frame = f_topo * (1 + v/c)`` — i.e. the velocity
    of the requested rest frame relative to the topocentric observer along the
    line of sight at ``obs_mjd``. Computed with astropy's ``SpectralCoord`` frame
    machinery (the authoritative implementation), so the sign/magnitude need no
    hand-derivation. Returns ``None`` (caller falls back to no shift) when astropy
    is unavailable, the epoch/coords are missing or non-finite, or the frame is
    topocentric. Never raises.
    """
    if not target_frame:
        return None
    ra = _finite_float(ra_deg)
    dec = _finite_float(dec_deg)
    mjd = _finite_float(obs_mjd)
    if ra is None or dec is None or mjd is None:
        return None
    try:
        import warnings

        import astropy.units as u
        from astropy.coordinates import EarthLocation, SkyCoord, SpectralCoord
        from astropy.time import Time

        location = EarthLocation.from_geodetic(
            lon=_ALMA_LON_DEG * u.deg,
            lat=_ALMA_LAT_DEG * u.deg,
            height=_ALMA_HEIGHT_M * u.m,
        )
        obstime = Time(float(mjd), format="mjd")
        target = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
        ref_ghz = 100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = SpectralCoord(
                ref_ghz * u.GHz,
                observer=location.get_itrs(obstime=obstime),
                target=target,
            )
            shifted = spec.with_observer_stationary_relative_to(target_frame)
            f_frame = float(shifted.quantity.to_value(u.GHz))
        return (f_frame / ref_ghz - 1.0) * SPEED_OF_LIGHT_KMS
    except Exception:  # pragma: no cover - defensive: astropy edge cases
        logger.debug("Frame offset computation failed", exc_info=True)
        return None


def _widen_intervals(
    intervals: Sequence[Mapping[str, Any]], extra_kms: float
) -> List[Dict[str, Any]]:
    """Return copies of intervals widened symmetrically by ``extra_kms``.

    Used only for the ADQL spectral pre-filter, never for classification, so a
    frame-shifted line is not excluded before its exact per-row correction.
    """
    if extra_kms <= 0:
        return [dict(item) for item in intervals]
    widened: List[Dict[str, Any]] = []
    for item in intervals:
        center = _finite_float(item.get("observed_frequency_ghz"))
        copy = dict(item)
        if center is not None:
            delta = center * float(extra_kms) / SPEED_OF_LIGHT_KMS
            copy["minimum_ghz"] = float(item["minimum_ghz"]) - delta
            copy["maximum_ghz"] = float(item["maximum_ghz"]) + delta
        widened.append(copy)
    return widened


class ALMACoverageService:
    """Exact local SPW coverage classification over ALMA ObsCore results."""

    def query(
        self,
        *,
        target: Mapping[str, Any],
        lines: Sequence[Mapping[str, Any]],
        radius_arcsec: float = 60.0,
        tolerance_mhz: float = 0.0,
        velocity_width_kms: Optional[float] = None,
        coverage_mode: str = "any",
        frame_uncertainty_kms: float = 0.0,
        min_calib_level: int = 2,
        public_only: bool = False,
        edge_channels: float = 0.0,
        edge_margin_ghz: float = 0.0,
        max_channel_width_khz: Optional[float] = None,
        doppler_frame: str = "observed (source frame)",
    ) -> Dict[str, Any]:
        radius = float(radius_arcsec)
        if radius < 1 or radius > 600:
            raise ValueError("radius_arcsec must be between 1 and 600")
        if coverage_mode not in {"any", "all"}:
            raise ValueError("coverage_mode must be any or all")
        ra = _finite_float(target.get("ra_deg"))
        dec = _finite_float(target.get("dec_deg"))
        if ra is None or dec is None:
            raise ValueError("Resolved target coordinates are required for ALMA coverage")
        requested = [
            self._requested_interval(
                line,
                tolerance_mhz=tolerance_mhz,
                velocity_width_kms=velocity_width_kms,
                frame_uncertainty_kms=frame_uncertainty_kms,
            )
            for line in lines
        ]
        requested = [item for item in requested if item is not None]
        if not requested:
            raise ValueError("At least one line with an observed frequency is required")
        now_iso = utc_now_iso()
        target_frame_name = _resolve_target_frame_name(doppler_frame)
        # Pre-filter widening: ensure the ADQL spectral predicate is at least as
        # wide as the bounded frame cap so no covering observation is excluded
        # before per-row correction. The precise `requested` intervals are kept
        # for classification; only the pre-filter copy is widened.
        if target_frame_name is None:
            prefilter_extra_kms = 0.0  # caller wants TOPO; no frame margin needed
        else:
            prefilter_extra_kms = max(
                0.0, _FRAME_PREFILTER_CAP_KMS - float(frame_uncertainty_kms or 0.0)
            )
        prefilter_intervals = _widen_intervals(requested, prefilter_extra_kms)
        frame = self._query_obscore(
            ra_deg=ra,
            dec_deg=dec,
            radius_arcsec=radius,
            intervals=prefilter_intervals,
            min_calib_level=min_calib_level,
            public_only=public_only,
            now_iso=now_iso,
        )
        rows = self._classify_rows(
            frame,
            requested,
            ra,
            dec,
            now_iso=now_iso,
            edge_channels=float(edge_channels or 0.0),
            edge_margin_ghz=float(edge_margin_ghz or 0.0),
            max_channel_width_khz=(
                float(max_channel_width_khz)
                if max_channel_width_khz is not None
                else None
            ),
            target_frame=target_frame_name,
        )
        projects = self._group_projects(rows, requested, coverage_mode)
        # Summarise the per-observation frame corrections that were actually
        # applied so nothing is shifted silently.
        applied_offsets = [
            row["frame_offset_kms"]
            for row in rows
            if row.get("frame_offset_kms") is not None
        ]
        any_corrected = bool(applied_offsets)
        warnings: List[str] = []
        if target_frame_name is None:
            pass  # caller explicitly requested the topocentric frame; no correction
        elif not any_corrected and frame_uncertainty_kms <= 0:
            warnings.append(
                "ALMA frequency_support is topocentric (TOPO) sky frequency while the "
                "requested line is in the observed source frame; per-observation "
                "Doppler-frame correction could not be applied (no observation epoch "
                "or astropy unavailable) and frame_uncertainty_kms=0. Lines near a "
                "spectral window edge may be mis-classified by the TOPO–LSRK offset "
                "(up to a few tens of km/s)."
            )
        return {
            "target": dict(target),
            "coverage_mode": coverage_mode,
            "radius_arcsec": radius,
            "requested_lines": requested,
            "project_count": len(projects),
            "observation_count": sum(len(item["observations"]) for item in projects),
            "projects": projects,
            "queried_at": now_iso,
            "archive": "ALMA ObsCore",
            "coverage_filters": {
                "min_calib_level": min_calib_level,
                "public_only": bool(public_only),
                "edge_channels": float(edge_channels or 0.0),
                "edge_margin_ghz": float(edge_margin_ghz or 0.0),
                "max_channel_width_khz": (
                    float(max_channel_width_khz)
                    if max_channel_width_khz is not None
                    else None
                ),
            },
            "doppler": {
                "archive_frame": "TOPO",
                "search_frame": doppler_frame,
                "target_frame": target_frame_name,
                "frame_uncertainty_kms": float(frame_uncertainty_kms or 0.0),
                "prefilter_margin_kms": float(_FRAME_PREFILTER_CAP_KMS)
                if target_frame_name is not None
                else 0.0,
                "offset_kms_range": (
                    [min(applied_offsets), max(applied_offsets)]
                    if applied_offsets
                    else None
                ),
                "method": (
                    "astropy SpectralCoord (per-observation TOPO→%s)" % target_frame_name.upper()
                    if any_corrected
                    else "fallback: symmetric widening (no per-observation correction applied)"
                ),
            },
            "warnings": warnings,
        }

    @staticmethod
    def _requested_interval(
        line: Mapping[str, Any],
        *,
        tolerance_mhz: float,
        velocity_width_kms: Optional[float],
        frame_uncertainty_kms: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        center = _finite_float(line.get("observed_frequency_ghz"))
        if center is None:
            return None
        half_width = max(0.0, float(tolerance_mhz or 0)) / 1000.0
        uncertainty = _finite_float(line.get("frequency_uncertainty_mhz"))
        if uncertainty is not None:
            half_width += max(0.0, uncertainty) / 1000.0
        if velocity_width_kms is not None:
            half_width += (
                center * max(0.0, float(velocity_width_kms)) / SPEED_OF_LIGHT_KMS / 2
            )
        # Absorb the topocentric-vs-source Doppler-frame offset of the archive
        # frequencies by widening the search interval (does not shift the center).
        if frame_uncertainty_kms:
            half_width += (
                center
                * max(0.0, float(frame_uncertainty_kms))
                / SPEED_OF_LIGHT_KMS
            )
        return {
            "line_id": str(line.get("line_id") or line.get("unique_line_id") or ""),
            "species": line.get("species") or line.get("formula"),
            "transition": line.get("transition"),
            "rest_frequency_ghz": line.get("frequency_ghz"),
            "observed_frequency_ghz": center,
            "minimum_ghz": center - half_width,
            "maximum_ghz": center + half_width,
            "half_width_mhz": half_width * 1000,
        }

    @staticmethod
    def _build_obscore_adql(
        *,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float,
        intervals: Sequence[Mapping[str, Any]],
        min_calib_level: Optional[int] = 2,
        public_only: bool = False,
        now_iso: Optional[str] = None,
    ) -> str:
        """Build the ObsCore ADQL with spatial, spectral, and data-quality filters.

        Pure/string-only so it can be unit-tested without the network. The
        calib_level filter excludes raw (uncalibrated) products by default; the
        public filter restricts to data past its proprietary period.
        """
        radius_deg = radius_arcsec / 3600.0
        spectral_predicates = []
        for item in intervals:
            high_hz = float(item["maximum_ghz"]) * 1e9
            low_hz = float(item["minimum_ghz"]) * 1e9
            lambda_min = 299792458.0 / high_hz
            lambda_max = 299792458.0 / low_hz
            spectral_predicates.append(
                f"(em_min <= {lambda_max:.15g} AND em_max >= {lambda_min:.15g})"
            )
        clauses = [
            "CONTAINS("
            "POINT('ICRS', s_ra, s_dec), "
            f"CIRCLE('ICRS', {ra_deg:.10f}, {dec_deg:.10f}, {radius_deg:.10f})"
            ") = 1",
            "(" + " OR ".join(spectral_predicates) + ")",
        ]
        if min_calib_level is not None and int(min_calib_level) > 0:
            clauses.append(f"calib_level >= {int(min_calib_level)}")
        if public_only and now_iso:
            clauses.append(f"obs_release_date <= '{now_iso[:19]}'")
        where = "\n  AND ".join(clauses)
        return f"""
SELECT TOP 20000
       target_name, proposal_id, member_ous_uid, obs_publisher_did,
       frequency, bandwidth, frequency_support, band_list,
       antenna_arrays, dataproduct_type, calib_level,
       scientific_category, science_keyword, obs_title, pi_name,
       s_ra, s_dec, t_exptime, s_resolution, spatial_resolution,
       obs_release_date, em_min, em_max, t_min, t_max
FROM ivoa.obscore
WHERE {where}
"""

    @staticmethod
    def _query_obscore(
        *,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float,
        intervals: Sequence[Mapping[str, Any]],
        min_calib_level: Optional[int] = 2,
        public_only: bool = False,
        now_iso: Optional[str] = None,
    ) -> pd.DataFrame:
        import requests

        query = ALMACoverageService._build_obscore_adql(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            radius_arcsec=radius_arcsec,
            intervals=intervals,
            min_calib_level=min_calib_level,
            public_only=public_only,
            now_iso=now_iso,
        )
        configured = os.getenv("SPECTRAL_LINE_ALMA_TAP_URL", "").strip()
        endpoints = [
            configured.rstrip("/") + "/sync" if configured else "",
            "https://almascience.eso.org/tap/sync",
            "https://almascience.nrao.edu/tap/sync",
        ]
        errors = []
        for endpoint in dict.fromkeys(item for item in endpoints if item):
            try:
                response = requests.post(
                    endpoint,
                    data={
                        "REQUEST": "doQuery",
                        "LANG": "ADQL",
                        "FORMAT": "csv",
                        "QUERY": query,
                    },
                    timeout=180,
                )
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "html" in content_type or response.text.lstrip().startswith("<!DOCTYPE"):
                    raise RuntimeError("ALMA TAP returned an HTML error response")
                return pd.read_csv(io.StringIO(response.text))
            except Exception as exc:
                errors.append(f"{endpoint}: {exc}")
        raise RuntimeError("ALMA TAP coverage query failed: " + " | ".join(errors))

    def _classify_rows(
        self,
        frame: pd.DataFrame,
        requested: Sequence[Mapping[str, Any]],
        target_ra: float,
        target_dec: float,
        *,
        now_iso: Optional[str] = None,
        edge_channels: float = 0.0,
        edge_margin_ghz: float = 0.0,
        max_channel_width_khz: Optional[float] = None,
        target_frame: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if frame is None or frame.empty:
            return rows
        rank = {"full": 0, "edge": 1, "center_only": 2, "partial": 3}
        # Per-observation TOPO→frame offset, memoised by epoch (target is fixed
        # for the whole query, so the offset depends only on the observation date).
        offset_cache: Dict[Any, Optional[float]] = {}

        def _offset_for_epoch(mid_mjd: Optional[float]) -> Optional[float]:
            if target_frame is None or mid_mjd is None:
                return None
            key = round(float(mid_mjd), 3)
            if key not in offset_cache:
                offset_cache[key] = compute_topo_to_frame_offset_kms(
                    target_ra, target_dec, mid_mjd, target_frame
                )
            return offset_cache[key]

        for index, row in frame.iterrows():
            windows = observation_windows_ghz(row)
            if not windows:
                continue
            # Convert this observation's topocentric SPW edges into the requested
            # rest frame using its own epoch, so coverage is judged in a single
            # consistent frame. f_frame = f_topo * (1 + v/c).
            t_lo = _finite_float(row.get("t_min") if hasattr(row, "get") else None)
            t_hi = _finite_float(row.get("t_max") if hasattr(row, "get") else None)
            if t_lo is not None and t_hi is not None:
                mid_mjd: Optional[float] = (t_lo + t_hi) / 2.0
            else:
                mid_mjd = t_lo if t_lo is not None else t_hi
            frame_offset_kms = _offset_for_epoch(mid_mjd)
            if frame_offset_kms is not None:
                factor = 1.0 + frame_offset_kms / SPEED_OF_LIGHT_KMS
                windows = [
                    {
                        **window,
                        "low_ghz": float(window["low_ghz"]) * factor,
                        "high_ghz": float(window["high_ghz"]) * factor,
                    }
                    for window in windows
                ]
            line_matches = []
            for line in requested:
                matches = []
                for window in windows:
                    classified = self._classify_spw(
                        window,
                        line,
                        edge_channels=edge_channels,
                        edge_margin_ghz=edge_margin_ghz,
                        max_channel_width_khz=max_channel_width_khz,
                    )
                    if classified["classification"] != "none":
                        matches.append(classified)
                if matches:
                    matches.sort(
                        key=lambda item: (
                            rank.get(item["classification"], 4),
                            0 if item.get("resolution_ok", True) else 1,
                            -item["edge_margin_mhz"],
                        )
                    )
                    line_matches.append(
                        {
                            "line": dict(line),
                            "best_classification": matches[0]["classification"],
                            "usable": bool(matches[0].get("usable")),
                            "spws": matches,
                        }
                    )
            if not line_matches:
                continue
            row_dict = {
                key: _json_value(value)
                for key, value in row.to_dict().items()
            }
            row_dict.update(
                {
                    "observation_id": str(
                        row_dict.get("obs_publisher_did") or index
                    ),
                    "is_public": self._observation_is_public(
                        row_dict.get("obs_release_date"), now_iso
                    ),
                    "matching_lines": line_matches,
                    "frame_offset_kms": (
                        round(frame_offset_kms, 4)
                        if frame_offset_kms is not None
                        else None
                    ),
                    "frame_corrected": frame_offset_kms is not None,
                    "angular_separation_arcsec": _angular_separation_arcsec(
                        target_ra,
                        target_dec,
                        _finite_float(row_dict.get("s_ra")),
                        _finite_float(row_dict.get("s_dec")),
                    ),
                    "archive_url": _alma_archive_url(
                        row_dict.get("member_ous_uid"),
                        row_dict.get("proposal_id"),
                    ),
                }
            )
            rows.append(row_dict)
        return rows

    @staticmethod
    def _observation_is_public(release: Any, now_iso: Optional[str]) -> Optional[bool]:
        release_text = str(release if release is not None else "").strip()
        if not release_text or release_text.lower() == "nan":
            return None
        reference = (now_iso or utc_now_iso())[:19]
        return release_text[:19] <= reference

    @staticmethod
    def _classify_spw(
        window: Mapping[str, Any],
        line: Mapping[str, Any],
        *,
        edge_channels: float = 0.0,
        edge_margin_ghz: float = 0.0,
        max_channel_width_khz: Optional[float] = None,
    ) -> Dict[str, Any]:
        low = float(window["low_ghz"])
        high = float(window["high_ghz"])
        resolution_khz = window.get("resolution_khz")
        resolution_khz = (
            float(resolution_khz) if resolution_khz is not None else None
        )
        required_low = float(line["minimum_ghz"])
        required_high = float(line["maximum_ghz"])
        center = float(line["observed_frequency_ghz"])

        # Effective edge-channel safety margin: the larger of an absolute GHz
        # margin and N channels (when the channel width is known).
        margin = max(0.0, float(edge_margin_ghz or 0.0))
        if edge_channels and resolution_khz is not None:
            margin = max(margin, float(edge_channels) * resolution_khz / 1e6)
        usable_low = low + margin
        usable_high = high - margin

        if low <= required_low and required_high <= high:
            # Fully inside the band; "edge" if within the safety margin of an edge.
            if usable_low <= required_low and required_high <= usable_high:
                classification = "full"
            else:
                classification = "edge"
        elif low <= center <= high:
            classification = "center_only"
        elif max(low, required_low) <= min(high, required_high):
            classification = "partial"
        else:
            classification = "none"

        # Resolution requirement: if a max channel width is set we can only call a
        # window resolution-OK when its (known) channel width is fine enough.
        resolution_ok = max_channel_width_khz is None or (
            resolution_khz is not None
            and resolution_khz <= float(max_channel_width_khz)
        )
        usable = classification == "full" and resolution_ok
        edge_margin = min(center - low, high - center) * 1000
        return {
            "minimum_ghz": low,
            "maximum_ghz": high,
            "classification": classification,
            "edge_margin_mhz": round(edge_margin, 6),
            "resolution_khz": resolution_khz,
            "resolution_ok": resolution_ok,
            "usable": usable,
        }

    @staticmethod
    def _group_projects(
        rows: Sequence[Mapping[str, Any]],
        requested: Sequence[Mapping[str, Any]],
        coverage_mode: str,
    ) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Mapping[str, Any]]] = {}
        for row in rows:
            project = str(row.get("proposal_id") or "Unknown project")
            grouped.setdefault(project, []).append(row)
        requested_keys = {
            (
                item.get("line_id"),
                item.get("species"),
                item.get("transition"),
                item.get("observed_frequency_ghz"),
            )
            for item in requested
        }
        projects: List[Dict[str, Any]] = []
        for project_id, observations in grouped.items():
            covered_keys = set()
            full_keys = set()
            usable_keys = set()
            margins = []
            for observation in observations:
                for match in observation.get("matching_lines") or []:
                    line = match["line"]
                    key = (
                        line.get("line_id"),
                        line.get("species"),
                        line.get("transition"),
                        line.get("observed_frequency_ghz"),
                    )
                    covered_keys.add(key)
                    if match["best_classification"] == "full":
                        full_keys.add(key)
                    if match.get("usable"):
                        usable_keys.add(key)
                    for spw in match.get("spws") or []:
                        margins.append(float(spw.get("edge_margin_mhz") or 0))
            collective_all = requested_keys <= covered_keys
            if coverage_mode == "all" and not collective_all:
                continue
            first = observations[0]
            resolutions = [
                value
                for item in observations
                for value in (
                    _finite_float(item.get("s_resolution")),
                    _finite_float(item.get("spatial_resolution")),
                )
                if value is not None
            ]
            separations = [
                float(item["angular_separation_arcsec"])
                for item in observations
                if item.get("angular_separation_arcsec") is not None
            ]
            exposures = [
                _finite_float(item.get("t_exptime")) or 0 for item in observations
            ]
            projects.append(
                {
                    "proposal_id": project_id,
                    "target_name": first.get("target_name"),
                    "pi_name": first.get("pi_name"),
                    "obs_title": first.get("obs_title"),
                    "bands": sorted(
                        {
                            part.strip()
                            for item in observations
                            for part in str(item.get("band_list") or "").split(",")
                            if part.strip()
                        }
                    ),
                    "covers_all_lines": collective_all,
                    "all_lines_full": requested_keys <= full_keys,
                    "all_lines_usable": requested_keys <= usable_keys,
                    "covered_line_count": len(covered_keys),
                    "requested_line_count": len(requested_keys),
                    "minimum_edge_margin_mhz": min(margins) if margins else None,
                    "angular_separation_arcsec": min(separations) if separations else None,
                    "best_angular_resolution_arcsec": min(resolutions) if resolutions else None,
                    "total_exposure_seconds": sum(exposures),
                    "archive_url": _alma_archive_url(None, project_id),
                    "observations": list(observations),
                }
            )
        projects.sort(
            key=lambda item: (
                0 if item["all_lines_usable"] else 1,
                0 if item["all_lines_full"] else 1,
                0 if item["covers_all_lines"] else 1,
                -float(item["minimum_edge_margin_mhz"] or -1e12),
                float(item["angular_separation_arcsec"] or 1e12),
                float(item["best_angular_resolution_arcsec"] or 1e12),
                -float(item["total_exposure_seconds"] or 0),
            )
        )
        return projects


def _angular_separation_arcsec(
    ra1: float,
    dec1: float,
    ra2: Optional[float],
    dec2: Optional[float],
) -> Optional[float]:
    if ra2 is None or dec2 is None:
        return None
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    return float(
        SkyCoord(ra1 * u.deg, dec1 * u.deg)
        .separation(SkyCoord(ra2 * u.deg, dec2 * u.deg))
        .arcsec
    )


def _alma_archive_url(member_ous_uid: Any, proposal_id: Any) -> str:
    if member_ous_uid:
        return (
            "https://almascience.nrao.edu/aq/?member_ous_id="
            + quote_plus(str(member_ous_uid))
        )
    return (
        "https://almascience.nrao.edu/aq/?project_code="
        + quote_plus(str(proposal_id or ""))
    )


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def confusion_score(
    candidate: Mapping[str, Any],
    *,
    target_frequency_ghz: float,
    window_mhz: float,
) -> Dict[str, Any]:
    frequency = _finite_float(candidate.get("observed_frequency_ghz"))
    if frequency is None:
        frequency = _finite_float(candidate.get("frequency_ghz"))
    delta_mhz = (
        abs(frequency - target_frequency_ghz) * 1000
        if frequency is not None
        else float("inf")
    )
    effective_window = max(float(window_mhz), 1e-9)
    proximity = max(0.0, 60.0 * (1 - delta_mhz / effective_window))
    uncertainty = _finite_float(candidate.get("frequency_uncertainty_mhz"))
    if uncertainty is None:
        uncertainty_points = 7.5
    else:
        uncertainty_points = max(
            0.0, 15.0 * (1 - min(uncertainty / effective_window, 1.0))
        )
    recommended = 10.0 if candidate.get("nrao_recommended") is True else 0.0
    observed = 10.0 if candidate.get("astronomically_observed") is True else 0.0
    catalog_count = len(candidate.get("catalogs") or [])
    agreement = min(5.0, catalog_count / 3 * 5.0)
    score = round(
        proximity + uncertainty_points + recommended + observed + agreement, 2
    )
    classification = "High" if score >= 75 else "Medium" if score >= 45 else "Low"
    return {
        "score": score,
        "classification": classification,
        "velocity_offset_kms": (
            (frequency - target_frequency_ghz)
            / target_frequency_ghz
            * SPEED_OF_LIGHT_KMS
            if frequency is not None
            else None
        ),
        "score_components": {
            "frequency_proximity": round(proximity, 2),
            "frequency_uncertainty_quality": round(uncertainty_points, 2),
            "nrao_recommendation": recommended,
            "astronomically_observed": observed,
            "multi_catalog_agreement": round(agreement, 2),
        },
        "scientific_limitation": (
            "Catalog-evidence heuristic only; this is not a flux, abundance, "
            "detectability, or LTE prediction."
        ),
    }


def analyze_confusion(
    selected_line: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    window_mhz: float,
) -> Dict[str, Any]:
    center = _finite_float(selected_line.get("observed_frequency_ghz"))
    if center is None:
        center = _finite_float(selected_line.get("frequency_ghz"))
    if center is None:
        raise ValueError("Selected line requires a frequency")
    selected_ids = set(selected_line.get("raw_line_ids") or [])
    selected_key = (
        SplatalogueTool._search_token(selected_line.get("species")),
        SplatalogueTool._search_token(selected_line.get("transition")),
        round(float(selected_line.get("frequency_ghz") or center), 6),
    )
    results = []
    for candidate in candidates:
        candidate_ids = set(candidate.get("raw_line_ids") or [])
        candidate_key = (
            SplatalogueTool._search_token(candidate.get("species")),
            SplatalogueTool._search_token(candidate.get("transition")),
            round(float(candidate.get("frequency_ghz") or 0), 6),
        )
        if (selected_ids and selected_ids & candidate_ids) or candidate_key == selected_key:
            continue
        item = dict(candidate)
        item.update(
            confusion_score(
                item,
                target_frequency_ghz=center,
                window_mhz=window_mhz,
            )
        )
        results.append(item)
    results.sort(key=lambda item: (-float(item["score"]), abs(float(item["velocity_offset_kms"] or 0))))
    return {
        "selected_line": dict(selected_line),
        "window_mhz": float(window_mhz),
        "candidates": results,
        "candidate_count": len(results),
        "scientific_limitation": (
            "Scores rank catalog evidence near the selected line. They do not model "
            "excitation, chemistry, optical depth, abundance, or sensitivity."
        ),
    }


class SpectralLineJobService:
    """Ephemeral asynchronous job service backed by DiskCache."""

    def __init__(
        self,
        cache_dir: Path | str = Path("cache") / "spectral_line_jobs",
        *,
        ttl_seconds: Optional[int] = None,
        max_workers: Optional[int] = None,
        max_rows: Optional[int] = None,
    ):
        self.cache = Cache(str(cache_dir))
        self.ttl_seconds = int(
            ttl_seconds
            if ttl_seconds is not None
            else os.getenv("SPECTRAL_LINE_JOB_TTL_SECONDS", "3600")
        )
        self.max_rows = int(
            max_rows
            if max_rows is not None
            else os.getenv("SPECTRAL_LINE_MAX_ROWS", "50000")
        )
        worker_count = int(
            max_workers
            if max_workers is not None
            else os.getenv("SPECTRAL_LINE_GLOBAL_WORKERS", "2")
        )
        self.executor = ThreadPoolExecutor(
            max_workers=max(1, worker_count), thread_name_prefix="spectral-lines"
        )
        self.splatalogue = SplatalogueTool()
        self.species = SpeciesMetadataCache()
        self.resolver = TargetResolver()
        self.coverage = ALMACoverageService()
        self._events: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._orphan_running_jobs()

    def metadata(self) -> Dict[str, Any]:
        return {
            "enabled": spectral_line_explorer_enabled(),
            "band_registry": {
                "version": "ALMA Cycle 13",
                "source": "https://almascience.nrao.edu/proposing/proposers-guide",
                "bands": [
                    {"band": band, "minimum_ghz": limits[0], "maximum_ghz": limits[1]}
                    for band, limits in ALMA_BAND_LIMITS_GHZ.items()
                ],
            },
            "units": [
                "Hz",
                "kHz",
                "MHz",
                "GHz",
                "THz",
                "m",
                "cm",
                "mm",
                "um",
                "nm",
                "Angstrom",
            ],
            "catalogs": [
                "JPL",
                "CDMS",
                "SLAIM",
                "LovasNIST",
                "ToyaMA",
                "OSU",
                "TopModel",
                "Recombination",
                "RFI",
            ],
            "versions": ["v1.0", "v2.0", "v3.0", "vall"],
            "defaults": {
                "mode": "basic",
                "version": "v3.0",
                "line_lists": ["JPL", "CDMS", "SLAIM", "LovasNIST"],
                "exclude_categories": ["atmospheric", "potential", "probable"],
                "only_nrao_recommended": True,
                "output_mode": "merged",
                "page_size": 100,
                "display_fields": DEFAULT_DISPLAY_FIELDS,
            },
            "limits": {
                "species": 50,
                "windows": 30,
                "rows": self.max_rows,
                "page_size": 500,
                "remote_seconds": 240,
                "concurrent_jobs_per_user": 2,
                "target_radius_arcsec": [1, 600],
            },
        }

    def create_job(
        self,
        *,
        user_id: str,
        operation: str,
        payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if operation not in {"catalog_search", "alma_coverage", "confusion"}:
            raise ValueError("Unsupported spectral-line operation")
        if self._active_jobs_for_user(user_id) >= 2:
            raise ValueError("At most two spectral-line jobs may run concurrently per user")
        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "user_id": user_id,
            "operation": operation,
            "status": "queued",
            "phase": "queued",
            "progress": 0,
            "payload": dict(payload),
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "warnings": [],
            "summary": {},
            "result": None,
            "error": None,
            "cancel_requested": False,
        }
        event = threading.Event()
        with self._lock:
            self._events[job_id] = event
            self._save(job)
        self.executor.submit(self._run_job, job_id)
        return self._public_job(job)

    def get_job(
        self,
        *,
        user_id: str,
        job_id: str,
        page: int = 1,
        page_size: int = 100,
        dataset: Optional[str] = None,
    ) -> Dict[str, Any]:
        job = self._owned_job(user_id, job_id)
        size = max(1, min(int(page_size or 100), 500))
        page_number = max(1, int(page or 1))
        public = self._public_job(job)
        result = job.get("result")
        if isinstance(result, Mapping):
            if dataset:
                rows = self._dataset_rows(result, dataset)
                selected_dataset = dataset
            else:
                rows, selected_dataset = self._primary_rows(job)
            start = (page_number - 1) * size
            public["rows"] = rows[start : start + size]
            public["pagination"] = {
                "dataset": selected_dataset,
                "page": page_number,
                "page_size": size,
                "total_rows": len(rows),
                "total_pages": math.ceil(len(rows) / size) if rows else 0,
            }
            public["result_context"] = {
                key: value
                for key, value in result.items()
                if key not in {"lines", "raw_lines", "projects", "candidates"}
            }
        return public

    def cancel_job(self, *, user_id: str, job_id: str) -> Dict[str, Any]:
        job = self._owned_job(user_id, job_id)
        if job["status"] in TERMINAL_JOB_STATUSES:
            return self._public_job(job)
        job["cancel_requested"] = True
        job["updated_at"] = utc_now_iso()
        event = self._events.get(job_id)
        if event:
            event.set()
        self._save(job)
        return self._public_job(job)

    def export(
        self,
        *,
        user_id: str,
        job_id: str,
        dataset: str,
        format_name: str,
    ) -> tuple[str, str, str]:
        job = self._owned_job(user_id, job_id)
        if job["status"] not in {"succeeded", "partial"}:
            raise ValueError("Only completed jobs can be exported")
        result = job.get("result") or {}
        rows = self._dataset_rows(result, dataset)
        format_name = format_name.lower()
        stem = f"spectral-lines-{job_id[:8]}-{dataset}"
        if format_name == "json":
            content = json.dumps(
                {
                    "query": job.get("payload"),
                    "provenance": result.get("query_provenance"),
                    "target": result.get("target"),
                    "warnings": job.get("warnings"),
                    dataset: rows,
                },
                indent=2,
                ensure_ascii=False,
            )
            return f"{stem}.json", "application/json", content
        if format_name in {"csv", "tsv"}:
            delimiter = "," if format_name == "csv" else "\t"
            content = _delimited_text(rows, delimiter)
            mime = "text/csv" if format_name == "csv" else "text/tab-separated-values"
            return f"{stem}.{format_name}", mime, content
        if format_name == "casa":
            if dataset not in {"lines", "raw_lines", "candidates"}:
                raise ValueError("CASA export is available for line datasets")
            return (
                f"{stem}-casa.txt",
                "text/plain",
                _casa_bundle(rows, result.get("target")),
            )
        raise ValueError("format must be csv, tsv, json, or casa")

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.cache.close()

    def _run_job(self, job_id: str) -> None:
        job = self.cache.get(job_id)
        if not job:
            return
        started_monotonic = time.monotonic()
        event = self._events.get(job_id) or threading.Event()
        try:
            self._update(job, status="running", phase="validating query", progress=5)
            if job["operation"] == "catalog_search":
                result = self._run_catalog_search(job, event)
            elif job["operation"] == "alma_coverage":
                result = self._run_coverage(job, event)
                if result.get("needs_input"):
                    self._update(
                        job,
                        status="needs_input",
                        phase="redshift selection required",
                        progress=100,
                        result=result,
                        summary={"reason": result.get("reason")},
                    )
                    logger.info(
                        "spectral_line_job operation=%s status=needs_input reason=%s",
                        job["operation"],
                        result.get("reason"),
                    )
                    return
            else:
                result = self._run_confusion(job, event)
            if event.is_set():
                raise SplatalogueQueryCancelled("Job canceled")
            warnings = list(result.get("warnings") or [])
            status = "partial" if result.get("degraded") or warnings else "succeeded"
            rows, _ = self._primary_rows({"operation": job["operation"], "result": result})
            self._update(
                job,
                status=status,
                phase="complete",
                progress=100,
                result=result,
                warnings=warnings,
                summary={"row_count": len(rows), "operation": job["operation"]},
            )
            logger.info(
                "spectral_line_job operation=%s status=%s rows=%s backend=%s degraded=%s duration_seconds=%.3f",
                job["operation"],
                status,
                len(rows),
                result.get("backend"),
                bool(result.get("degraded")),
                time.monotonic() - started_monotonic,
            )
        except SplatalogueQueryCancelled:
            self._update(job, status="canceled", phase="canceled", progress=100)
            logger.info(
                "spectral_line_job operation=%s status=canceled duration_seconds=%.3f",
                job["operation"],
                time.monotonic() - started_monotonic,
            )
        except Exception as exc:
            self._update(
                job,
                status="failed",
                phase="failed",
                progress=100,
                error=str(exc),
            )
            logger.exception(
                "spectral_line_job operation=%s status=failed duration_seconds=%.3f",
                job["operation"],
                time.monotonic() - started_monotonic,
            )
        finally:
            with self._lock:
                self._events.pop(job_id, None)

    def _run_catalog_search(
        self, job: Dict[str, Any], event: threading.Event
    ) -> Dict[str, Any]:
        payload = job["payload"]
        mode = str(payload.get("mode") or "basic")
        query = (
            SpectralLineQuery.basic(payload.get("query") or payload)
            if mode == "basic"
            else SpectralLineQuery.from_payload(payload.get("query") or payload)
        )
        z = query_redshift(query)
        segments = self._segments(query)
        all_raw: List[Dict[str, Any]] = []
        backend_names = []
        warnings: List[str] = []
        degraded = False
        provenance = []
        for index, segment in enumerate(segments):
            if event.is_set():
                raise SplatalogueQueryCancelled("Job canceled")
            self._update(
                job,
                phase=f"querying Splatalogue segment {index + 1}/{len(segments)}",
                progress=10 + int(75 * index / max(1, len(segments))),
            )
            response = self.splatalogue.query_catalog(segment, cancel_event=event)
            all_raw.extend(response.get("raw_lines") or response.get("lines") or [])
            backend_names.append(response.get("backend"))
            warnings.extend(response.get("warnings") or [])
            degraded = degraded or bool(response.get("degraded"))
            provenance.append(response.get("query_provenance"))
            if len(all_raw) >= self.max_rows:
                all_raw = all_raw[: self.max_rows]
                warnings.append(
                    f"Results were truncated at the {self.max_rows:,}-row job limit."
                )
                break
        raw_by_id: Dict[str, Dict[str, Any]] = {}
        for index, line in enumerate(all_raw):
            key = str(
                line.get("unique_line_id")
                or line.get("line_id")
                or f"{line.get('species_id')}:{line.get('frequency_ghz')}:{line.get('source')}:{index}"
            )
            raw_by_id.setdefault(key, line)
        all_raw = list(raw_by_id.values())
        cached_species = self.species.cache.get(self.species.CACHE_KEY) or []
        species_metadata = {
            str(item["species_id"]): item
            for item in cached_species
            if isinstance(item, Mapping) and item.get("species_id") is not None
        }
        enriched_raw = []
        for line in all_raw:
            item = dict(line)
            metadata = species_metadata.get(str(item.get("species_id") or ""))
            if metadata:
                item["formula"] = metadata.get("formula") or item.get("formula")
                item["chemical_name"] = (
                    metadata.get("chemical_name") or item.get("chemical_name")
                )
                item["species_status"] = metadata.get("status")
                item["molecular_mass"] = metadata.get("molecular_mass")
                item["species_tag"] = metadata.get("tag")
            enriched_raw.append(apply_frequency_frame(item, z))
        all_raw = enriched_raw
        lines = (
            self.splatalogue._deduplicate(all_raw)
            if query.output_mode == "merged"
            else all_raw
        )
        lines.sort(key=lambda item: float(item.get("frequency_ghz") or 0))
        if query.sort_order == "frequency_desc":
            lines.reverse()
        return {
            "query": query.to_dict(),
            "lines": lines,
            "raw_lines": all_raw,
            "total_matches": len(lines),
            "raw_total_matches": len(all_raw),
            "backend": ", ".join(sorted(set(filter(None, backend_names)))),
            "degraded": degraded,
            "warnings": list(dict.fromkeys(warnings)),
            "query_provenance": provenance,
        }

    def _run_coverage(
        self, job: Dict[str, Any], event: threading.Event
    ) -> Dict[str, Any]:
        payload = job["payload"]
        self._update(job, phase="resolving target", progress=10)
        explicit_z = _finite_float(payload.get("redshift"))
        initial_query_payload = dict(payload.get("query") or {})
        if explicit_z is None and initial_query_payload.get("radial_velocity_kms") not in (None, ""):
            explicit_z = radial_velocity_to_redshift(
                float(initial_query_payload["radial_velocity_kms"]),
                str(initial_query_payload.get("velocity_convention") or "radio"),
            )
        target = self.resolver.resolve(
            str(payload.get("target_name") or ""),
            explicit_redshift=explicit_z,
            explicit_ra_deg=_finite_float(payload.get("ra_deg")),
            explicit_dec_deg=_finite_float(payload.get("dec_deg")),
        )
        if target["state"] == "needs_input":
            return {
                "needs_input": True,
                "reason": "SIMBAD and NED did not provide one unambiguous redshift.",
                "target": target,
            }
        if event.is_set():
            raise SplatalogueQueryCancelled("Job canceled")
        query_payload = initial_query_payload
        query_payload["redshift"] = target["redshift"]
        query_payload["radial_velocity_kms"] = None
        query_payload.setdefault("output_mode", "merged")
        catalog_job = {
            **job,
            "payload": {
                "mode": payload.get("mode") or "basic",
                "query": query_payload,
            },
        }
        catalog = self._run_catalog_search(catalog_job, event)
        selected_ids = set(str(value) for value in payload.get("selected_line_ids") or [])
        lines = catalog["lines"]
        if selected_ids:
            lines = [
                line
                for line in lines
                if str(line.get("line_id")) in selected_ids
                or bool(selected_ids & set(str(value) for value in line.get("raw_line_ids") or []))
            ]
        if not lines:
            raise ValueError("No catalog line matched the coverage request")
        self._update(job, phase="querying exact ALMA spectral-window coverage", progress=75)
        coverage = self.coverage.query(
            target=target,
            lines=lines,
            radius_arcsec=float(payload.get("radius_arcsec") or 60),
            tolerance_mhz=float(payload.get("tolerance_mhz") or 0),
            velocity_width_kms=_finite_float(payload.get("velocity_width_kms")),
            coverage_mode=str(payload.get("coverage_mode") or "any"),
            frame_uncertainty_kms=float(payload.get("frame_uncertainty_kms") or 0),
            min_calib_level=int(payload.get("min_calib_level", 2) or 0),
            public_only=bool(payload.get("public_only", False)),
            edge_channels=float(payload.get("edge_channels") or 0),
            edge_margin_ghz=float(payload.get("edge_margin_ghz") or 0),
            max_channel_width_khz=_finite_float(payload.get("max_channel_width_khz")),
        )
        coverage.update(
            {
                "lines": lines,
                "raw_lines": catalog["raw_lines"],
                "backend": catalog["backend"],
                "degraded": catalog["degraded"],
                "warnings": catalog["warnings"],
                "query_provenance": catalog["query_provenance"],
            }
        )
        return coverage

    def _run_confusion(
        self, job: Dict[str, Any], event: threading.Event
    ) -> Dict[str, Any]:
        payload = job["payload"]
        selected = payload.get("selected_line")
        if not isinstance(selected, Mapping):
            raise ValueError("selected_line is required for confusion analysis")
        selected = apply_frequency_frame(selected, _finite_float(payload.get("redshift")))
        center = _finite_float(selected.get("observed_frequency_ghz"))
        if center is None:
            raise ValueError("selected_line requires an observed frequency")
        if payload.get("window_mhz") not in (None, ""):
            window_mhz = float(payload["window_mhz"])
        elif payload.get("velocity_width_kms") not in (None, ""):
            window_mhz = (
                center
                * float(payload["velocity_width_kms"])
                / SPEED_OF_LIGHT_KMS
                * 1000
            )
        else:
            window_mhz = 10.0
        query_payload = dict(payload.get("query") or {})
        query_payload["windows"] = [
            {
                "minimum": center - window_mhz / 1000,
                "maximum": center + window_mhz / 1000,
                "unit": "GHz",
            }
        ]
        query_payload["frame"] = "observed"
        query_payload["redshift"] = _finite_float(payload.get("redshift"))
        query_payload["output_mode"] = "merged"
        catalog = self._run_catalog_search(
            {
                **job,
                "payload": {
                    "mode": payload.get("mode") or "advanced",
                    "query": query_payload,
                },
            },
            event,
        )
        result = analyze_confusion(
            selected,
            catalog["lines"],
            window_mhz=window_mhz,
        )
        result.update(
            {
                "backend": catalog["backend"],
                "degraded": catalog["degraded"],
                "warnings": catalog["warnings"],
                "query_provenance": catalog["query_provenance"],
            }
        )
        return result

    @staticmethod
    def _segments(query: SpectralLineQuery) -> List[SpectralLineQuery]:
        if query.species_ids:
            species_batches = [
                {
                    "species_ids": query.species_ids[index : index + 10],
                    "species_names": [],
                }
                for index in range(0, len(query.species_ids), 10)
            ]
        elif query.species_names:
            species_batches = [
                {"species_ids": [], "species_names": [name]}
                for name in query.species_names
            ]
        else:
            species_batches = [{"species_ids": [], "species_names": []}]
        segments = []
        for window in query.windows:
            for species_batch in species_batches:
                values = query.to_dict()
                values["windows"] = [
                    {
                        "minimum": window.minimum_ghz,
                        "maximum": window.maximum_ghz,
                        "unit": "GHz",
                    }
                ]
                values.update(species_batch)
                segments.append(SpectralLineQuery.from_payload(values))
        return segments

    def _active_jobs_for_user(self, user_id: str) -> int:
        total = 0
        for key in self.cache.iterkeys():
            job = self.cache.get(key)
            if (
                isinstance(job, Mapping)
                and job.get("user_id") == user_id
                and job.get("status") in {"queued", "running"}
            ):
                total += 1
        return total

    def _owned_job(self, user_id: str, job_id: str) -> Dict[str, Any]:
        job = self.cache.get(job_id)
        if not isinstance(job, dict):
            raise KeyError(f"Spectral-line job not found: {job_id}")
        if job.get("user_id") != user_id:
            raise PermissionError("Spectral-line job belongs to another user")
        return job

    def _save(self, job: Dict[str, Any]) -> None:
        self.cache.set(job["job_id"], job, expire=self.ttl_seconds)

    def _update(self, job: Dict[str, Any], **changes: Any) -> None:
        job.update(changes)
        job["updated_at"] = utc_now_iso()
        self._save(job)

    @staticmethod
    def _public_job(job: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in job.items()
            if key not in {"user_id", "result"}
        }

    def _orphan_running_jobs(self) -> None:
        for key in list(self.cache.iterkeys()):
            job = self.cache.get(key)
            if isinstance(job, dict) and job.get("status") in {"queued", "running"}:
                job["status"] = "orphaned"
                job["phase"] = "server restarted"
                job["updated_at"] = utc_now_iso()
                self._save(job)

    @staticmethod
    def _dataset_rows(result: Mapping[str, Any], dataset: str) -> List[Dict[str, Any]]:
        mapping = {
            "lines": result.get("lines"),
            "raw_lines": result.get("raw_lines"),
            "coverage": result.get("projects"),
            "candidates": result.get("candidates"),
        }
        rows = mapping.get(dataset)
        if not isinstance(rows, list):
            raise ValueError("Unknown or unavailable export dataset")
        return rows

    def _primary_rows(self, job: Mapping[str, Any]) -> tuple[List[Dict[str, Any]], str]:
        result = job.get("result") or {}
        operation = job.get("operation")
        if operation == "alma_coverage":
            return list(result.get("projects") or []), "coverage"
        if operation == "confusion":
            return list(result.get("candidates") or []), "candidates"
        return list(result.get("lines") or []), "lines"


def _delimited_text(rows: Sequence[Mapping[str, Any]], delimiter: str) -> str:
    if not rows:
        return ""
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter=delimiter, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in row.items()
            }
        )
    return stream.getvalue()


def _casa_bundle(
    rows: Sequence[Mapping[str, Any]], target: Optional[Mapping[str, Any]]
) -> str:
    redshift = _finite_float((target or {}).get("redshift"))
    tabular = _delimited_text(rows, "\t")
    snippet = [
        "",
        "# CASA-oriented helper snippet (plain Python, not a native CASA table)",
        "from astropy import units as u",
        f"redshift = {redshift!r}",
        "lines = [",
    ]
    for row in rows:
        rest = _finite_float(row.get("frequency_ghz"))
        observed = _finite_float(row.get("observed_frequency_ghz"))
        comment = " | ".join(
            str(value or "")
            for value in (
                row.get("species"),
                row.get("transition"),
                ", ".join(row.get("catalogs") or []),
                f"z={redshift}" if redshift is not None else "z=not supplied",
            )
        )
        snippet.append(
            "    {"
            f"'restfreq': {rest!r} * u.GHz, "
            f"'observedfreq': {observed!r} * u.GHz, "
            f"'comment': {comment!r}"
            "},"
        )
    snippet.extend(["]", "restfreqs = [line['restfreq'] for line in lines]"])
    return tabular + "\n".join(snippet) + "\n"


def spectral_line_deep_link(
    *,
    target_name: str,
    species_id: Optional[int] = None,
    species: Optional[str] = None,
    transition: Optional[str] = None,
    redshift: Optional[float] = None,
) -> str:
    params = {
        "mode": "basic",
        "target": target_name,
        "z": "auto" if redshift is None else str(redshift),
        "autorun": "coverage",
    }
    if species_id is not None:
        params["species"] = str(species_id)
    elif species:
        params["species_name"] = species
    if transition:
        params["transition"] = transition
    return "/spectral-lines?" + urlencode(params)


def find_alma_line_coverage(
    *,
    target_name: str,
    species: str,
    transition: str,
    redshift: Optional[float] = None,
    tolerance_mhz: float = 0.0,
    velocity_width_kms: Optional[float] = None,
    radius_arcsec: float = 60.0,
) -> Dict[str, Any]:
    """Composed synchronous workflow used by chat for one named transition."""
    resolver = TargetResolver()
    target = resolver.resolve(target_name, explicit_redshift=redshift)
    if target["state"] == "needs_input":
        return {
            "success": False,
            "needs_input": True,
            "target": target,
            "error": "An explicit redshift is required because SIMBAD/NED values conflict or are missing.",
            "deep_link": spectral_line_deep_link(
                target_name=target_name,
                species=species,
                transition=transition,
                redshift=redshift,
            ),
        }
    species_id = None
    try:
        metadata_matches = SpeciesMetadataCache().search(species, limit=100)
        exact = [
            item
            for item in metadata_matches
            if formula_matches(species, item.get("formula"))
            or formula_matches(species, item.get("base_formula"))
        ]
        preferred = next(
            (item for item in exact if item.get("status") == "known"),
            exact[0] if exact else None,
        )
        species_id = preferred.get("species_id") if preferred else None
    except Exception:
        species_id = None
    query = SpectralLineQuery.basic(
        {
            "windows": [{"minimum": 1, "maximum": 1000, "unit": "GHz"}],
            "species_ids": [species_id] if species_id is not None else [],
            "species_names": [] if species_id is not None else [species],
            "transition": transition,
            "redshift": target["redshift"],
            "only_nrao_recommended": True,
            "output_mode": "merged",
            "page_size": 100,
        }
    )
    catalog = SplatalogueTool().query_catalog(query)
    candidates = [
        apply_frequency_frame(line, target["redshift"])
        for line in catalog.get("lines") or []
        if _transition_matches(line.get("transition"), transition)
    ]
    if not candidates:
        return {
            "success": False,
            "target": target,
            "error": f"Splatalogue returned no recommended {species} {transition} transition.",
            "backend": catalog.get("backend"),
            "warnings": catalog.get("warnings") or [],
            "deep_link": spectral_line_deep_link(
                target_name=target_name,
                species=species,
                transition=transition,
                redshift=redshift,
            ),
        }
    candidates.sort(
        key=lambda line: (
            0 if line.get("nrao_recommended") is True else 1,
            _finite_float(line.get("frequency_uncertainty_mhz"))
            if _finite_float(line.get("frequency_uncertainty_mhz")) is not None
            else float("inf"),
            float(line.get("frequency_ghz") or 0),
        )
    )
    selected = candidates[0]
    coverage = ALMACoverageService().query(
        target=target,
        lines=[selected],
        radius_arcsec=radius_arcsec,
        tolerance_mhz=tolerance_mhz,
        velocity_width_kms=velocity_width_kms,
        coverage_mode="any",
    )
    return {
        "success": True,
        "selected_line": selected,
        "target": target,
        "project_count": coverage["project_count"],
        "projects": coverage["projects"],
        "backend": catalog.get("backend"),
        "degraded": catalog.get("degraded"),
        "warnings": catalog.get("warnings") or [],
        "deep_link": spectral_line_deep_link(
            target_name=target_name,
            species_id=species_id,
            species=species,
            transition=transition,
            redshift=redshift,
        ),
    }


def _transition_matches(value: Any, requested: str) -> bool:
    """Structural quantum-number match (delegates to the shared matcher).

    ``value`` is the candidate line's transition; ``requested`` is the query.
    """
    return transition_matches(requested, value)
