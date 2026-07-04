"""Solar-system tools: JPL Horizons ephemerides + IMCCE SkyBoT cone search."""

from __future__ import annotations

import math
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests


SKYBOT_DEFAULT_BASE_URL = "https://ssp.imcce.fr/webservices/skybot/api/conesearch.php"
_MAX_EPHEM_ROWS = 400
_MAX_SKYBOT_RADIUS_DEG = 10.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


class SolarSystemService:
    """Timeout-bound client for JPL Horizons ephemerides and SkyBoT cone search."""

    def __init__(
        self,
        *,
        skybot_base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        horizons_factory: Optional[Callable[..., Any]] = None,
    ):
        self.skybot_base_url = str(
            skybot_base_url or os.getenv("SKYBOT_BASE_URL") or SKYBOT_DEFAULT_BASE_URL
        )
        self.timeout = float(timeout if timeout is not None else _env_float("SKYBOT_TIMEOUT", 30.0))
        self._horizons_factory = horizons_factory

    # ── Part A: JPL Horizons ────────────────────────────────────────────────
    def horizons_ephemeris(
        self,
        target: Any,
        start: Any,
        stop: Any,
        step: Any = "1d",
        location: Any = "500",
        id_type: Any = "smallbody",
    ) -> Dict[str, Any]:
        try:
            target_s = _require_str(target, "target")
            start_s = _require_str(start, "start")
            stop_s = _require_str(stop, "stop")
            step_s = str(step or "1d").strip() or "1d"
            # 'smallbody' is the default because it is the common case (asteroids/
            # comets) AND because id_type=None resolves ambiguous names to the
            # major-body ephemeris, which masks the V magnitude (verified live
            # 2026-07-03: Ceres V is masked with default resolution, 8.98 with
            # smallbody). None/'' -> let astroquery auto-resolve. Major bodies
            # (planets/moons) fail the small-body resolver, so a failed
            # smallbody attempt falls back to auto resolution below.
            id_type_s = str(id_type).strip() or None if id_type is not None else None
            warnings: List[str] = []

            # Bound the request BEFORE any network call (codex review P2):
            # truncate spans > 370 days and reject step/span combos that would
            # ask Horizons for an absurd number of rows.
            start_s, stop_s, guard_warnings = _guard_ephemeris_range(start_s, stop_s, step_s)
            warnings.extend(guard_warnings)

            try:
                obj = self._make_horizons(target_s, location, start_s, stop_s, step_s, id_type_s)
                table = obj.ephemerides()
            except Exception as first_err:
                if id_type_s != "smallbody":
                    raise
                # planets/moons are not in the small-body system — retry with
                # Horizons' automatic resolution (codex review P2)
                obj = self._make_horizons(target_s, location, start_s, stop_s, step_s, None)
                try:
                    table = obj.ephemerides()
                except Exception:
                    raise first_err
                id_type_s = None
                warnings.append(
                    "Small-body lookup failed; resolved via Horizons major-body/auto resolution."
                )
            rows, thin_warn = _normalize_ephemeris(table)
            warnings.extend(thin_warn)
            return {
                "success": True,
                "rows": rows,
                "count": len(rows),
                "warnings": warnings,
                "provenance": {
                    "service": "JPL Horizons",
                    "target": target_s,
                    "location": str(location),
                    "start": start_s,
                    "stop": stop_s,
                    "step": step_s,
                    "id_type": id_type_s,
                },
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _make_horizons(self, target: str, location: Any, start: str, stop: str, step: str, id_type: Any):
        kwargs = dict(
            id=target,
            location=str(location),
            epochs={"start": start, "stop": stop, "step": step},
            id_type=id_type,
        )
        if self._horizons_factory is not None:
            return self._horizons_factory(**kwargs)
        from astroquery.jplhorizons import Horizons  # lazy

        return Horizons(**kwargs)

    # ── Part B: IMCCE SkyBoT cone search ────────────────────────────────────
    def skybot_cone(
        self,
        ra: Any,
        dec: Any,
        radius_deg: Any = 0.2,
        epoch: Any = None,
    ) -> Dict[str, Any]:
        try:
            ra_f, dec_f = _validate_coords(ra, dec)
            radius, warnings = _normalize_radius(radius_deg)
            jd, epoch_iso = _epoch_to_jd(epoch)
            params = {
                "-ep": f"{jd:.6f}",
                "-ra": ra_f,
                "-dec": dec_f,
                "-rd": radius,
                "-mime": "json",
                "-output": "object",
                "-loc": "500",
            }
            resp = requests.get(self.skybot_base_url, params=params, timeout=self.timeout)
            if resp.status_code != 200:
                return {"success": False, "error": f"SkyBoT returned HTTP {resp.status_code}"}
            payload = _parse_skybot_json(resp)
            if payload.get("no_solution"):
                warnings.append(payload.get("message") or "SkyBoT reported no solution for this field.")
                records = []
            else:
                records = payload.get("records", [])
            rows = [row for row in (_normalize_skybot_record(rec) for rec in records) if row]
            return {
                "success": True,
                "rows": rows,
                "count": len(rows),
                "warnings": warnings,
                "provenance": {
                    "service": "IMCCE SkyBoT",
                    "base_url": self.skybot_base_url,
                    "ra": ra_f,
                    "dec": dec_f,
                    "radius_deg": radius,
                    "epoch_jd": jd,
                    "epoch_iso": epoch_iso,
                },
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}


# ── Horizons helpers ────────────────────────────────────────────────────────
_MAX_SPAN_DAYS = 370.0
_MAX_REQUEST_ROWS = 5000
_STEP_UNIT_SECONDS = {"d": 86400.0, "h": 3600.0, "m": 60.0}


def _guard_ephemeris_range(start_s: str, stop_s: str, step_s: str) -> Tuple[str, str, List[str]]:
    """Truncate over-long spans and reject absurd row counts BEFORE fetching."""
    from astropy.time import Time  # lazy

    warnings: List[str] = []
    t0 = Time(start_s, scale="utc")
    t1 = Time(stop_s, scale="utc")
    if float(t1.jd) <= float(t0.jd):
        raise ValueError(f"stop ({stop_s}) must be after start ({start_s}).")
    span_days = float(t1.jd) - float(t0.jd)
    if span_days > _MAX_SPAN_DAYS:
        t1 = Time(float(t0.jd) + _MAX_SPAN_DAYS, format="jd", scale="utc")
        stop_s = str(t1.utc.isot)
        warnings.append(
            f"Date span {span_days:.0f} d exceeds {_MAX_SPAN_DAYS:.0f} d; stop truncated to {stop_s}."
        )
        span_days = _MAX_SPAN_DAYS
    match = re.fullmatch(r"\s*(\d+)\s*([dhm])\s*", str(step_s), flags=re.IGNORECASE)
    if match:
        step_seconds = int(match.group(1)) * _STEP_UNIT_SECONDS[match.group(2).lower()]
        if step_seconds > 0:
            estimated_rows = span_days * 86400.0 / step_seconds
            if estimated_rows > _MAX_REQUEST_ROWS:
                raise ValueError(
                    f"Requested range would return ~{estimated_rows:.0f} ephemeris rows "
                    f"(cap {_MAX_REQUEST_ROWS}); use a coarser step than {step_s!r} or a shorter range."
                )
    return start_s, stop_s, warnings


def _normalize_ephemeris(table: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    colnames = list(getattr(table, "colnames", []) or [])

    def col(*candidates: str) -> Optional[str]:
        for cand in candidates:
            for name in colnames:
                if name.lower() == cand.lower():
                    return name
        return None

    c_dt = col("datetime_str", "datetime")
    c_ra = col("RA")
    c_dec = col("DEC")
    c_delta = col("delta")
    c_r = col("r")
    c_v = col("V", "Tmag", "Nmag")
    c_elong = col("elong")
    c_alpha = col("alpha", "alpha_true")

    n = len(table)
    indices = range(n)
    if n > _MAX_EPHEM_ROWS:
        stride = math.ceil(n / _MAX_EPHEM_ROWS)
        indices = range(0, n, stride)
        warnings.append(
            f"Horizons returned {n} rows; thinned to every {stride}th row (cap {_MAX_EPHEM_ROWS})."
        )

    rows: List[Dict[str, Any]] = []
    for i in indices:
        rows.append(
            {
                "datetime": _cell_str(table, c_dt, i),
                "ra_deg": _cell_float(table, c_ra, i),
                "dec_deg": _cell_float(table, c_dec, i),
                "delta_au": _cell_float(table, c_delta, i),
                "r_au": _cell_float(table, c_r, i),
                "v_mag": _cell_float(table, c_v, i),
                "elong_deg": _cell_float(table, c_elong, i),
                "alpha_deg": _cell_float(table, c_alpha, i),
            }
        )
    return rows, warnings


def _cell_str(table: Any, colname: Optional[str], i: int) -> Optional[str]:
    if not colname:
        return None
    try:
        value = table[colname][i]
    except Exception:
        return None
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def _cell_float(table: Any, colname: Optional[str], i: int) -> Optional[float]:
    if not colname:
        return None
    try:
        value = table[colname][i]
    except Exception:
        return None
    return _to_float(value)


# ── SkyBoT helpers ──────────────────────────────────────────────────────────
def _parse_skybot_json(resp: Any) -> Dict[str, Any]:
    try:
        payload = resp.json()
    except Exception:
        return {"no_solution": True, "message": "SkyBoT returned an unparseable response."}
    if isinstance(payload, list):
        return {"records": payload}
    if isinstance(payload, dict):
        flag = payload.get("flag")
        data = payload.get("data")
        if flag == -1 or data is None:
            return {"no_solution": True, "message": payload.get("message")}
        if isinstance(data, list):
            return {"records": data}
    return {"no_solution": True, "message": "SkyBoT returned no data array."}


def _skget(record: Dict[str, Any], *keys: str) -> Any:
    """Case-insensitive, unit-suffix-insensitive key lookup on a SkyBoT record."""
    norm = {_normkey(k): v for k, v in record.items()}
    for key in keys:
        val = norm.get(_normkey(key))
        if val is not None:
            return val
    return None


def _normkey(key: str) -> str:
    # drop a trailing " (unit)" and collapse to lowercase alnum
    base = re.sub(r"\s*\(.*?\)\s*$", "", str(key))
    return re.sub(r"[^a-z0-9]", "", base.lower())


def _normalize_skybot_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(record, dict):
        return None
    ra_raw = _skget(record, "RA", "RA (hms)", "RA(h)")
    dec_raw = _skget(record, "DEC", "DEC (dms)", "DE(deg)")
    ra_deg, dec_deg = _skybot_coords(ra_raw, dec_raw)
    num = _skget(record, "Num", "Number")
    return {
        "number": _to_int(num),
        "name": _clean_str(_skget(record, "Name")),
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "class": _clean_str(_skget(record, "Class")),
        "v_mag": _to_float(_skget(record, "VMag", "Mv", "V")),
        "sep_arcsec": _to_float(_skget(record, "d", "d (arcsec)")),
        "pos_err_arcsec": _to_float(_skget(record, "Err", "ErrPos", "Err (arcsec)")),
    }


def _skybot_coords(ra_raw: Any, dec_raw: Any) -> Tuple[Optional[float], Optional[float]]:
    if ra_raw is None or dec_raw is None:
        return None, None
    ra_s = str(ra_raw).strip()
    dec_s = str(dec_raw).strip()
    # sexagesimal if it contains a space or colon (RA in hours, DEC in degrees)
    if re.search(r"[\s:]", ra_s) or re.search(r"[\s:]", dec_s):
        try:
            from astropy.coordinates import SkyCoord  # lazy
            from astropy import units as u

            c = SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg))
            return float(c.ra.deg), float(c.dec.deg)
        except Exception:
            return None, None
    return _to_float(ra_raw), _to_float(dec_raw)


def _epoch_to_jd(epoch: Any) -> Tuple[float, str]:
    """Return (jd, iso). MUST be a JD for SkyBoT — ISO with 'T' can 500 the API."""
    from astropy.time import Time  # lazy

    if epoch is None or (isinstance(epoch, str) and not epoch.strip()):
        t = Time.now()
    else:
        try:
            t = Time(float(epoch), format="jd", scale="utc")
        except (TypeError, ValueError):
            t = Time(str(epoch), scale="utc")
    return float(t.jd), str(t.utc.isot)


# ── shared coercion helpers ─────────────────────────────────────────────────
def _validate_coords(ra: Any, dec: Any) -> Tuple[float, float]:
    ra_f = float(ra)
    dec_f = float(dec)
    if not (0.0 <= ra_f < 360.0):
        raise ValueError(f"RA {ra_f} out of range [0, 360).")
    if not (-90.0 <= dec_f <= 90.0):
        raise ValueError(f"Dec {dec_f} out of range [-90, 90].")
    return ra_f, dec_f


def _normalize_radius(radius_deg: Any) -> Tuple[float, List[str]]:
    warnings: List[str] = []
    try:
        radius = float(radius_deg)
    except (TypeError, ValueError):
        return 0.2, ["radius_deg was not numeric; using 0.2 deg."]
    if radius <= 0:
        warnings.append(f"radius_deg {radius:g} is non-positive; using 0.2 deg.")
        radius = 0.2
    elif radius > _MAX_SKYBOT_RADIUS_DEG:
        warnings.append(f"radius_deg {radius:g} exceeds {_MAX_SKYBOT_RADIUS_DEG:g}; clamped.")
        radius = _MAX_SKYBOT_RADIUS_DEG
    return radius, warnings


def _require_str(value: Any, label: str) -> str:
    s = str(value or "").strip()
    if not s:
        raise ValueError(f"{label} is required.")
    return s


def _clean_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if hasattr(value, "mask") and getattr(value, "mask", False):
        return None
    return f if math.isfinite(f) else None


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        s = str(value).strip()
        if not s or s == "-":
            return None
        return int(float(s))
    except (TypeError, ValueError):
        return None


__all__ = ["SolarSystemService"]
