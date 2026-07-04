"""Distance tools for Gaia/Bailer-Jones, NED-D, and velocity frames."""

from __future__ import annotations

import math
import os
import statistics
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


GAIA_TAP_DEFAULT_BASE_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
GAIA_TAP_DEFAULT_TIMEOUT = 60.0
NED_DISTANCE_ENDPOINT = "https://ned.ipac.caltech.edu/cgi-bin/nDistance"
LOW_PARALLAX_WARNING = "low-significance parallax: prefer r_photogeo / treat as uncertain"
PECULIAR_VELOCITY_WARNING = "peculiar velocities significant; Hubble-flow distance unreliable"
SPEED_OF_LIGHT_KMS = 299792.458
PLANCK18_H0 = 67.66

# Apex vectors for heliocentric corrections: NED GSR convention,
# Karachentsev & Makarov 1996 Local Group, and Planck 2018 CMB dipole.
APEX_CORRECTIONS: Tuple[Tuple[str, float, float, float], ...] = (
    ("GSR", 232.3, 87.8, 1.7),
    ("LocalGroup", 316.0, 93.0, -4.0),
    ("CMB", 369.82, 264.021, 48.253),
)


class DistanceService:
    """Timeout-bound distance service with offline-testable network hooks."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        http_post: Optional[Callable[..., Any]] = None,
        ned_table_loader: Optional[Callable[[str], Any]] = None,
    ):
        self.base_url = str(base_url or os.getenv("GAIA_TAP_BASE_URL") or GAIA_TAP_DEFAULT_BASE_URL).rstrip("/")
        self.timeout = float(timeout if timeout is not None else _env_float("GAIA_TAP_TIMEOUT", GAIA_TAP_DEFAULT_TIMEOUT))
        if http_post is None:
            def _http_post(*args: Any, **kwargs: Any) -> Any:
                import requests

                return requests.post(*args, **kwargs)

            self.http_post = _http_post
        else:
            self.http_post = http_post
        self.ned_table_loader = ned_table_loader

    def gaia_distances(self, ra: Any, dec: Any, radius_arcsec: Any = 10, max_rows: Any = 10) -> Dict[str, Any]:
        """Return Bailer-Jones geometric/photogeometric Gaia distances near a sky position."""
        try:
            warnings: List[str] = []
            ra_f, dec_f = _validate_coords(ra, dec)
            radius, radius_warnings = _normalize_radius_arcsec(radius_arcsec, default=10.0, cap=300.0)
            row_limit, row_warnings = _normalize_positive_int(max_rows, default=10, cap=50, label="max_rows")
            warnings.extend(radius_warnings)
            warnings.extend(row_warnings)

            radius_deg = radius / 3600.0
            adql = _gaia_adql(ra_f, dec_f, radius_deg, row_limit)
            form = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": adql}
            payload = self._post_gaia_json(form)
            rows, row_warnings = _normalize_gaia_payload(payload)
            warnings.extend(row_warnings)
            return {
                "success": True,
                "rows": rows,
                "count": len(rows),
                "warnings": warnings,
                "provenance": {
                    "service": "ESA Gaia DR3 x Bailer-Jones EDR3 distances",
                    "base_url": self.base_url,
                    "ra": ra_f,
                    "dec": dec_f,
                    "radius_arcsec": radius,
                    "radius_deg": radius_deg,
                    "max_rows": row_limit,
                    "query": adql,
                },
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def ned_distances(self, target_name: Any) -> Dict[str, Any]:
        """Return NED redshift-independent distance measurements for a named galaxy."""
        try:
            target = _require_target(target_name)
            table = self.ned_table_loader(target) if self.ned_table_loader else self._load_ned_distance_table(target)
            rows, warnings = _normalize_ned_table(table)
            summary = _ned_summary(rows)
            return {
                "success": True,
                "rows": rows,
                "count": len(rows),
                "summary": summary,
                "warnings": warnings,
                "provenance": {
                    "service": "NED-D redshift-independent distances",
                    "target_name": target,
                    "table": "nDistance",
                    "endpoint": NED_DISTANCE_ENDPOINT,
                    "timeout": self.timeout,
                },
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def velocity_frames(
        self,
        ra: Any,
        dec: Any,
        v_helio_kms: Optional[Any] = None,
        z: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Convert heliocentric velocity/redshift to GSR, Local Group, and CMB frames."""
        try:
            warnings: List[str] = []
            ra_f, dec_f = _validate_coords(ra, dec)
            if (v_helio_kms is None) == (z is None):
                raise ValueError("Provide exactly one of v_helio_kms or z.")
            if z is not None:
                z_f = _finite_float(z, "z")
                if z_f <= -1.0:
                    raise ValueError("z must be greater than -1.")
                one_plus_z_sq = (1.0 + z_f) ** 2
                v_helio = SPEED_OF_LIGHT_KMS * ((one_plus_z_sq - 1.0) / (one_plus_z_sq + 1.0))
            else:
                z_f = None
                v_helio = _finite_float(v_helio_kms, "v_helio_kms")

            gal_l, gal_b = _galactic_lon_lat(ra_f, dec_f)
            rows = [
                {
                    "frame": "heliocentric",
                    "v_kms": float(v_helio),
                    "d_hubble_mpc": _hubble_distance(v_helio),
                }
            ]
            frame_velocities: Dict[str, float] = {}
            for frame, apex_v, apex_l, apex_b in APEX_CORRECTIONS:
                corrected = v_helio + apex_v * _apex_projection(gal_l, gal_b, apex_l, apex_b)
                frame_velocities[frame] = corrected
                rows.append(
                    {
                        "frame": frame,
                        "v_kms": float(corrected),
                        "d_hubble_mpc": _hubble_distance(corrected),
                    }
                )
            if abs(frame_velocities["CMB"]) < 3000.0:
                warnings.append(PECULIAR_VELOCITY_WARNING)
            return {
                "success": True,
                "rows": rows,
                "count": len(rows),
                "warnings": warnings,
                "provenance": {
                    "service": "Velocity-frame corrections",
                    "ra": ra_f,
                    "dec": dec_f,
                    "galactic_l_deg": gal_l,
                    "galactic_b_deg": gal_b,
                    "v_helio_kms": float(v_helio),
                    "z": z_f,
                    "H0_km_s_Mpc": PLANCK18_H0,
                    "apex_conventions": "NED GSR, Karachentsev & Makarov 1996 Local Group, Planck 2018 CMB dipole",
                },
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _post_gaia_json(self, form: Dict[str, Any]) -> Dict[str, Any]:
        import requests

        try:
            response = self.http_post(self.base_url, data=form, timeout=self.timeout)
        except requests.RequestException as exc:
            raise RuntimeError(f"Gaia TAP request failed: {exc}") from exc
        status = int(getattr(response, "status_code", 0) or 0)
        if status != 200:
            text = str(getattr(response, "text", "") or "")[:200]
            raise RuntimeError(f"Gaia TAP returned HTTP {status}: {text}")
        try:
            payload = response.json()
        except Exception as exc:
            raise RuntimeError("Gaia TAP returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Gaia TAP returned an unexpected JSON payload.")
        return payload

    def _load_ned_distance_table(self, target: str) -> Any:
        import pandas as pd
        import requests

        response = requests.get(NED_DISTANCE_ENDPOINT, params={"name": target}, timeout=self.timeout)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        for table in tables:
            columns = list(getattr(table, "columns", []))
            if _find_ned_column(columns, "modulus") or _find_ned_column(columns, "mpc"):
                return table
        raise ValueError(f"NED returned no distance table for {target!r}")


def _gaia_adql(ra: float, dec: float, radius_deg: float, max_rows: int) -> str:
    ra_s = _fmt_adql_float(ra)
    dec_s = _fmt_adql_float(dec)
    radius_s = _fmt_adql_float(radius_deg)
    return f"""SELECT TOP {max_rows}
  g.source_id, g.ra, g.dec, g.parallax, g.parallax_error,
  g.phot_g_mean_mag, g.pmra, g.pmdec,
  d.r_med_geo, d.r_lo_geo, d.r_hi_geo,
  d.r_med_photogeo, d.r_lo_photogeo, d.r_hi_photogeo,
  DISTANCE(POINT('ICRS', g.ra, g.dec), POINT('ICRS', {ra_s}, {dec_s})) AS sep_deg
FROM gaiadr3.gaia_source AS g
JOIN external.gaiaedr3_distance AS d ON g.source_id = d.source_id
WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec),
                   CIRCLE('ICRS', {ra_s}, {dec_s}, {radius_s}))
ORDER BY sep_deg ASC"""


def _normalize_gaia_payload(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    metadata = payload.get("metadata")
    data = payload.get("data")
    if not isinstance(metadata, list) or not isinstance(data, list):
        message = payload.get("message") or payload.get("error") or "Gaia TAP returned JSON without metadata/data arrays."
        raise RuntimeError(str(message))
    names = []
    for item in metadata:
        if not isinstance(item, dict) or not item.get("name"):
            raise RuntimeError("Gaia TAP metadata did not include usable column names.")
        names.append(str(item["name"]).lower())
    rows: List[Dict[str, Any]] = []
    low_significance = False
    for raw_row in data:
        if not isinstance(raw_row, (list, tuple)) or len(raw_row) != len(names):
            raise RuntimeError("Gaia TAP data row did not match metadata columns.")
        item = dict(zip(names, raw_row))
        sep_deg = _float_or_none(item.get("sep_deg"))
        row = {
            "source_id": _str_or_none(item.get("source_id")),
            "ra": _float_or_none(item.get("ra")),
            "dec": _float_or_none(item.get("dec")),
            "parallax_mas": _float_or_none(item.get("parallax")),
            "parallax_err_mas": _float_or_none(item.get("parallax_error")),
            "g_mag": _float_or_none(item.get("phot_g_mean_mag")),
            "pm_ra": _float_or_none(item.get("pmra")),
            "pm_dec": _float_or_none(item.get("pmdec")),
            "r_geo_pc": _float_or_none(item.get("r_med_geo")),
            "r_geo_lo_pc": _float_or_none(item.get("r_lo_geo")),
            "r_geo_hi_pc": _float_or_none(item.get("r_hi_geo")),
            "r_photogeo_pc": _float_or_none(item.get("r_med_photogeo")),
            "r_photogeo_lo_pc": _float_or_none(item.get("r_lo_photogeo")),
            "r_photogeo_hi_pc": _float_or_none(item.get("r_hi_photogeo")),
            "sep_arcsec": round(sep_deg * 3600.0, 3) if sep_deg is not None else None,
        }
        if _is_low_parallax_significance(row["parallax_mas"], row["parallax_err_mas"]):
            low_significance = True
        rows.append(row)
    warnings = [LOW_PARALLAX_WARNING] if low_significance else []
    return rows, warnings


def _normalize_ned_table(table: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    if hasattr(table, "colnames"):
        table = table.to_pandas()

    warnings: List[str] = []
    try:
        colnames = list(table.columns)
    except Exception:
        colnames = []
    modulus_col = _find_ned_column(colnames, "modulus")
    err_col = _find_ned_column(colnames, "modulus_error")
    mpc_col = _find_ned_column(colnames, "mpc")
    method_col = _find_ned_column(colnames, "method")
    refcode_col = _find_ned_column(colnames, "refcode")
    if not modulus_col and not mpc_col:
        raise RuntimeError("NED-D distance modulus/metric distance column not found.")
    if modulus_col and not err_col:
        warnings.append("NED-D distance modulus error column not found; using null errors.")
    if not method_col:
        warnings.append("NED-D method column not found; using null methods.")
    if not refcode_col:
        warnings.append("NED-D refcode column not found; using null refcodes.")

    rows: List[Dict[str, Any]] = []
    skipped = 0
    for _, source_row in table.iterrows():
        modulus = _float_or_none(_row_cell(source_row, modulus_col)) if modulus_col else None
        metric_distance = _float_or_none(_row_cell(source_row, mpc_col)) if mpc_col else None
        if modulus is None and metric_distance is None:
            skipped += 1
            continue
        if metric_distance is not None:
            dist_mpc = _round_sigfig(metric_distance, 4)
        else:
            dist_mpc = _round_sigfig(10.0 ** ((modulus - 25.0) / 5.0), 4)
        rows.append(
            {
                "dist_mpc": dist_mpc,
                "dist_modulus": float(modulus) if modulus is not None else None,
                "dist_modulus_err": _float_or_none(_row_cell(source_row, err_col)) if err_col else None,
                "method": _str_or_none(_row_cell(source_row, method_col)) if method_col else None,
                "refcode": _str_or_none(_row_cell(source_row, refcode_col)) if refcode_col else None,
            }
        )
    if skipped:
        warnings.append(f"Skipped {skipped} NED-D row(s) with masked/invalid distance modulus/metric distance.")
    return rows, warnings


def _ned_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    distances = [float(row["dist_mpc"]) for row in rows if row.get("dist_mpc") is not None]
    methods = {
        str(row.get("method")).strip().lower()
        for row in rows
        if row.get("method") is not None and str(row.get("method")).strip()
    }
    if not distances:
        return {"n": 0, "median_mpc": None, "min_mpc": None, "max_mpc": None, "n_methods": 0}
    return {
        "n": len(distances),
        "median_mpc": _round_sigfig(statistics.median(distances), 4),
        "min_mpc": _round_sigfig(min(distances), 4),
        "max_mpc": _round_sigfig(max(distances), 4),
        "n_methods": len(methods),
    }


def _find_ned_column(colnames: Sequence[Any], role: str) -> Optional[Any]:
    for name in colnames:
        lower = str(name).lower()
        compact = "".join(ch for ch in lower if ch.isalnum())
        is_error = "err" in lower or "uncert" in lower
        is_modulus = "modulus" in lower
        if role == "modulus" and is_modulus and not is_error:
            return name
        if role == "modulus_error" and is_modulus and is_error:
            return name
        if role == "mpc" and "mpc" in lower:
            return name
        if role == "method" and "method" in lower:
            return name
        if role == "refcode" and "refcode" in compact:
            return name
    return None


def _row_cell(row: Any, column: Optional[Any]) -> Any:
    if not column:
        return None
    try:
        return row[column]
    except Exception:
        return None


def _galactic_lon_lat(ra: float, dec: float) -> Tuple[float, float]:
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    return float(coord.galactic.l.deg), float(coord.galactic.b.deg)


def _apex_projection(l_deg: float, b_deg: float, l_apex_deg: float, b_apex_deg: float) -> float:
    l = math.radians(l_deg)
    b = math.radians(b_deg)
    l_apex = math.radians(l_apex_deg)
    b_apex = math.radians(b_apex_deg)
    return math.sin(b) * math.sin(b_apex) + math.cos(b) * math.cos(b_apex) * math.cos(l - l_apex)


def _hubble_distance(v_kms: float) -> Optional[float]:
    if v_kms <= 0.0:
        return None
    return float(v_kms / PLANCK18_H0)


def _validate_coords(ra: Any, dec: Any) -> Tuple[float, float]:
    ra_f = _finite_float(ra, "ra")
    dec_f = _finite_float(dec, "dec")
    if not 0.0 <= ra_f < 360.0:
        raise ValueError("ra must satisfy 0 <= ra < 360 degrees.")
    if not -90.0 <= dec_f <= 90.0:
        raise ValueError("dec must satisfy -90 <= dec <= 90 degrees.")
    return ra_f, dec_f


def _finite_float(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def _normalize_radius_arcsec(value: Any, *, default: float, cap: float) -> Tuple[float, List[str]]:
    warnings: List[str] = []
    try:
        radius = float(value)
    except (TypeError, ValueError):
        radius = default
        warnings.append(f"radius_arcsec was not numeric; using {default:g} arcsec.")
    if not math.isfinite(radius) or radius <= 0:
        radius = default
        warnings.append(f"radius_arcsec must be positive; using {default:g} arcsec.")
    elif radius > cap:
        warnings.append(f"radius_arcsec {radius:g} exceeds {cap:g}; clamped to {cap:g} arcsec.")
        radius = cap
    return radius, warnings


def _normalize_positive_int(value: Any, *, default: int, cap: int, label: str) -> Tuple[int, List[str]]:
    warnings: List[str] = []
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = default
        warnings.append(f"{label} was not an integer; using {default}.")
    if out <= 0:
        warnings.append(f"{label} must be positive; using {default}.")
        out = default
    elif out > cap:
        warnings.append(f"{label} {out} exceeds {cap}; clamped to {cap}.")
        out = cap
    return out, warnings


def _is_low_parallax_significance(parallax: Optional[float], error: Optional[float]) -> bool:
    if parallax is None:
        return False
    if error is None or error <= 0:
        return True
    return abs(parallax / error) < 5.0


def _require_target(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("target_name is required.")
    return text


def _fmt_adql_float(value: float) -> str:
    return format(float(value), ".12g")


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value > 0 else default


def _float_or_none(value: Any) -> Optional[float]:
    if value is None or _is_masked(value):
        return None
    try:
        if hasattr(value, "to_value"):
            value = value.to_value()
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            value = value.item()
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _str_or_none(value: Any) -> Optional[str]:
    if value is None or _is_masked(value):
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None


def _is_masked(value: Any) -> bool:
    try:
        import numpy as np

        return bool(np.ma.is_masked(value))
    except Exception:
        return False


def _round_sigfig(value: float, digits: int) -> float:
    return float(f"{float(value):.{digits}g}")


__all__ = ["DistanceService"]