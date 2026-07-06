"""CDS MOCServer sky-coverage queries for survey footprint preflights."""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests


# NB: the path is case-sensitive on the CDS side — "MocServer", not "MOCServer".
MOCSERVER_DEFAULT_BASE_URL = "https://alasky.unistra.fr/MocServer/query"
MOCSERVER_FIELDS = "ID,obs_title,dataproduct_type,em_min,em_max,moc_sky_fraction"
VALID_DATAPRODUCT_TYPES = {"image", "catalog", "cube"}
VALID_REGIMES = {"radio", "mm/sub-mm", "infrared", "optical", "uv", "x-ray", "gamma"}
# MocServer accepts SR=0 (point query) but returns HTTP 500 for 0 < SR < ~1e-4
# (verified live 2026-07-03), so tiny nonzero radii are clamped up to this.
MIN_NONZERO_RADIUS_DEG = 1e-4


class MocCoverageService:
    """Timeout-bound client for CDS MOCServer coverage records."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        http_get: Optional[Callable[..., Any]] = None,
    ):
        self.base_url = str(base_url or os.getenv("MOCSERVER_BASE_URL") or MOCSERVER_DEFAULT_BASE_URL).rstrip("/")
        self.timeout = float(timeout if timeout is not None else _env_float("MOCSERVER_TIMEOUT", 30.0))
        self.http_get = http_get or requests.get

    def coverage_at(
        self,
        ra: Any,
        dec: Any,
        radius_deg: Any = 0.0,
        dataproduct_type: Optional[str] = None,
        keyword: Optional[str] = None,
        max_rows: Any = 50,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return MOCServer records whose sky coverage intersects a cone."""
        try:
            warnings: List[str] = []
            ra_f, dec_f = _validate_coords(ra, dec)
            radius, radius_warnings = _normalize_radius_deg(radius_deg)
            warnings.extend(radius_warnings)
            row_limit, row_warnings = _normalize_max_rows(max_rows)
            warnings.extend(row_warnings)
            dtype = _normalize_dataproduct_type(dataproduct_type)
            regime_filter = _normalize_regime(regime)
            keyword_s = _normalize_keyword(keyword)

            params = self._query_params(ra_f, dec_f, radius)
            if dtype:
                params["expr"] = f"dataproduct_type={dtype}"

            records, request_meta, request_warnings = self._get_records_with_expr_fallback(params)
            warnings.extend(request_warnings)

            rows = [self._normalize_record(record) for record in records if isinstance(record, dict)]
            skipped = len(records) - len(rows)
            if skipped:
                warnings.append(f"Skipped {skipped} non-record MOCServer item(s).")
            rows = _filter_rows(rows, dataproduct_type=dtype, keyword=keyword_s, regime=regime_filter)
            rows.sort(key=_specificity_sort_key)

            total_matches = len(rows)
            capped_rows = rows[:row_limit]
            if total_matches > row_limit:
                warnings.append(f"MOCServer returned {total_matches} matching record(s); showing first {row_limit}.")

            return {
                "success": True,
                "rows": capped_rows,
                "count": len(capped_rows),
                "total_matches": total_matches,
                "warnings": warnings,
                "provenance": {
                    "service": "CDS MOCServer",
                    "base_url": self.base_url,
                    "params": request_meta["params"],
                    "attempted_params": request_meta["attempted_params"],
                    "ra": ra_f,
                    "dec": dec_f,
                    "radius_deg": radius,
                    "requested_radius_deg": _float_or_none(radius_deg),
                    "dataproduct_type": dtype,
                    "keyword": keyword_s,
                    "regime": regime_filter,
                    "max_rows": row_limit,
                },
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def survey_covers(self, survey_keyword: Any, ra: Any, dec: Any) -> Dict[str, Any]:
        """Return whether any MOCServer record matching a survey keyword covers a point."""
        try:
            keyword = _require_keyword(survey_keyword)
            result = self.coverage_at(ra, dec, radius_deg=0.0, keyword=keyword, max_rows=200)
            if not result.get("success"):
                return result
            ids = [str(row.get("id")) for row in result.get("rows", []) if row.get("id")]
            total_matches = int(result.get("total_matches") or 0)
            return {
                "success": True,
                "covered": total_matches > 0,
                "matches": ids[:10],
                "matched_count": total_matches,
                "warnings": result.get("warnings", []),
                "provenance": result.get("provenance", {}),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _query_params(self, ra: float, dec: float, radius_deg: float) -> Dict[str, Any]:
        return {
            "RA": ra,
            "DEC": dec,
            "SR": radius_deg,
            "intersect": "overlaps",
            "get": "record",
            "fmt": "json",
            "fields": MOCSERVER_FIELDS,
        }

    def _get_records_with_expr_fallback(self, params: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
        attempted = [dict(params)]
        warnings: List[str] = []
        try:
            payload = self._get_json(params)
            return payload, {"params": dict(params), "attempted_params": attempted}, warnings
        except Exception as first_exc:
            if "expr" not in params:
                raise
            retry_params = dict(params)
            retry_params.pop("expr", None)
            attempted.append(dict(retry_params))
            warnings.append(f"MOCServer rejected expr filter; retried without expr ({first_exc}).")
            payload = self._get_json(retry_params)
            return payload, {"params": retry_params, "attempted_params": attempted}, warnings

    def _get_json(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            response = self.http_get(self.base_url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise RuntimeError(f"MOCServer request failed: {exc}") from exc
        status = int(getattr(response, "status_code", 0) or 0)
        if status != 200:
            text = str(getattr(response, "text", "") or "")[:200]
            raise RuntimeError(f"MOCServer returned HTTP {status}: {text}")
        try:
            payload = response.json()
        except Exception as exc:
            raise RuntimeError("MOCServer returned invalid JSON.") from exc
        if not isinstance(payload, list):
            raise RuntimeError("MOCServer returned an unexpected JSON payload.")
        return payload

    @staticmethod
    def _normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
        em_min = _float_or_none(record.get("em_min"))
        em_max = _float_or_none(record.get("em_max"))
        return {
            "id": _str_or_none(record.get("ID", record.get("id"))),
            "title": _str_or_none(record.get("obs_title", record.get("title"))),
            "dataproduct_type": _lower_str_or_none(record.get("dataproduct_type")),
            "regime": _regime_label(em_min, em_max),
            "moc_sky_fraction": _float_or_none(record.get("moc_sky_fraction")),
        }


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value > 0 else default


def _validate_coords(ra: Any, dec: Any) -> Tuple[float, float]:
    try:
        ra_f = float(ra)
        dec_f = float(dec)
    except (TypeError, ValueError) as exc:
        raise ValueError("ra and dec must be numeric ICRS degrees.") from exc
    if not math.isfinite(ra_f) or not math.isfinite(dec_f):
        raise ValueError("ra and dec must be finite ICRS degrees.")
    if not 0.0 <= ra_f < 360.0:
        raise ValueError("ra must satisfy 0 <= ra < 360 degrees.")
    if not -90.0 <= dec_f <= 90.0:
        raise ValueError("dec must satisfy -90 <= dec <= 90 degrees.")
    return ra_f, dec_f


def _normalize_radius_deg(radius_deg: Any) -> Tuple[float, List[str]]:
    warnings: List[str] = []
    try:
        radius = float(radius_deg)
    except (TypeError, ValueError):
        radius = 0.0
        warnings.append("radius_deg was not numeric; using a point query.")
    if not math.isfinite(radius):
        radius = 0.0
        warnings.append("radius_deg must be finite; using a point query.")
    elif radius < 0:
        warnings.append(f"radius_deg {radius:g} is negative; using a point query.")
        radius = 0.0
    elif radius > 30.0:
        warnings.append(f"radius_deg {radius:g} exceeds 30; clamped to 30 deg.")
        radius = 30.0
    if 0.0 < radius < MIN_NONZERO_RADIUS_DEG:
        warnings.append(
            f"radius_deg {radius:g} is below MocServer's working minimum; "
            f"clamped up to {MIN_NONZERO_RADIUS_DEG:g} deg."
        )
        radius = MIN_NONZERO_RADIUS_DEG
    return radius, warnings


def _normalize_max_rows(max_rows: Any) -> Tuple[int, List[str]]:
    warnings: List[str] = []
    try:
        out = int(max_rows)
    except (TypeError, ValueError):
        out = 50
        warnings.append("max_rows was not an integer; using 50.")
    if out <= 0:
        warnings.append("max_rows must be positive; using 50.")
        out = 50
    elif out > 200:
        warnings.append(f"max_rows {out} exceeds 200; clamped to 200.")
        out = 200
    return out, warnings


def _normalize_dataproduct_type(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text not in VALID_DATAPRODUCT_TYPES:
        raise ValueError("dataproduct_type must be one of image, catalog, cube.")
    return text


def _normalize_regime(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text not in VALID_REGIMES:
        raise ValueError("regime must be one of radio, mm/sub-mm, infrared, optical, UV, X-ray, gamma.")
    if text == "uv":
        return "UV"
    if text == "x-ray":
        return "X-ray"
    return text


def _normalize_keyword(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _require_keyword(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("survey_keyword is required.")
    return text


def _filter_rows(
    rows: List[Dict[str, Any]],
    *,
    dataproduct_type: Optional[str],
    keyword: Optional[str],
    regime: Optional[str],
) -> List[Dict[str, Any]]:
    out = rows
    if dataproduct_type:
        out = [row for row in out if row.get("dataproduct_type") == dataproduct_type]
    if keyword:
        needle = keyword.lower()
        out = [row for row in out if needle in _row_text(row)]
    if regime:
        out = [row for row in out if row.get("regime") == regime]
    return out


def _row_text(row: Dict[str, Any]) -> str:
    return f"{row.get('id') or ''} {row.get('title') or ''}".lower()


def _specificity_sort_key(row: Dict[str, Any]) -> Tuple[float, str]:
    sky_fraction = row.get("moc_sky_fraction")
    if sky_fraction is None:
        sky_fraction = float("inf")
    return float(sky_fraction), str(row.get("id") or row.get("title") or "")


def _regime_label(em_min: Optional[float], em_max: Optional[float]) -> Optional[str]:
    if em_min is None or em_max is None:
        return None
    midpoint = (em_min + em_max) / 2.0
    if not math.isfinite(midpoint) or midpoint <= 0:
        return None
    if midpoint > 1e-2:
        return "radio"
    if midpoint >= 1e-3:
        return "mm/sub-mm"
    if midpoint >= 1e-6:
        return "infrared"
    if midpoint >= 3e-7:
        return "optical"
    if midpoint >= 1e-8:
        return "UV"
    if midpoint >= 1e-11:
        return "X-ray"
    return "gamma"


def _float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _lower_str_or_none(value: Any) -> Optional[str]:
    text = _str_or_none(value)
    return text.lower() if text else None


__all__ = ["MocCoverageService"]