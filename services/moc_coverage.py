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
# MOC geometry fetch limits: order caps payload size (all-sky MOCs at order 10+
# can be MBs); the cell cap triggers an automatic order downgrade.
MAX_MOC_ORDER = 10
MIN_MOC_ORDER = 3
MAX_MOC_IDS = 4
MAX_MOC_CELLS = 20000


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

    def moc_geometry(self, ids: Any, order: Any = 8) -> Dict[str, Any]:
        """Fetch MOC geometry (HEALPix cell maps) for survey/dataset IDs.

        Returns, per id, the exact JSON format Aladin Lite's ``A.MOCFromJSON``
        ingests ({"<order>": [cell, ...], ...}; verified live 2026-07-11).
        Oversized MOCs are automatically downsampled to a coarser order."""
        try:
            id_list = [str(i).strip() for i in (ids or []) if str(i).strip()]
            if not id_list:
                return {"success": False, "error": "ids is required (MOCServer dataset IDs, e.g. 'CDS/P/SDSS9/color')."}
            warnings: List[str] = []
            if len(id_list) > MAX_MOC_IDS:
                warnings.append(f"Fetching the first {MAX_MOC_IDS} of {len(id_list)} MOC geometries.")
                id_list = id_list[:MAX_MOC_IDS]
            try:
                order_i = int(order)
            except (TypeError, ValueError):
                order_i = 8
            order_i = max(MIN_MOC_ORDER, min(order_i, MAX_MOC_ORDER))

            mocs: List[Dict[str, Any]] = []
            for moc_id in id_list:
                local_order = order_i
                geometry = self._get_json_dict(
                    {"ID": moc_id, "get": "moc", "fmt": "json", "order": local_order}
                )
                n_cells = _moc_cell_count(geometry)
                while n_cells > MAX_MOC_CELLS and local_order > MIN_MOC_ORDER:
                    local_order -= 1
                    geometry = self._get_json_dict(
                        {"ID": moc_id, "get": "moc", "fmt": "json", "order": local_order}
                    )
                    n_cells = _moc_cell_count(geometry)
                if n_cells == 0:
                    warnings.append(f"MOCServer returned an empty MOC for {moc_id!r}; skipped.")
                    continue
                if local_order != order_i:
                    warnings.append(
                        f"MOC for {moc_id!r} downsampled to order {local_order} ({n_cells} cells)."
                    )
                mocs.append({
                    "id": moc_id,
                    "moc_json": geometry,
                    "n_cells": n_cells,
                    "order": local_order,
                })
            if not mocs:
                return {"success": False, "error": "No MOC geometry could be fetched for the given ids.", "warnings": warnings}
            return {
                "success": True,
                "mocs": mocs,
                "warnings": warnings,
                "provenance": {
                    "service": "CDS MOCServer",
                    "endpoint": self.base_url,
                    "order": order_i,
                },
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def moc_operation(
        self,
        ids: Any,
        operation: str = "intersection",
        order: Any = 8,
        ra_list: Optional[List[Any]] = None,
        dec_list: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """MOC boolean algebra over MOCServer dataset ids (mocpy-backed).

        operation: 'intersection' | 'union' | 'difference' (first minus the
        union of the rest). Returns the derived MOC in Aladin JSON format with
        its sky area in deg². When ra_list/dec_list are given, each position is
        additionally flagged as inside/outside the derived MOC.
        """
        try:
            op = str(operation or "intersection").strip().lower()
            if op not in {"intersection", "union", "difference"}:
                return {"success": False,
                        "error": "operation must be one of intersection, union, difference."}
            id_list = [str(i).strip() for i in (ids or []) if str(i).strip()]
            if len(id_list) > MAX_MOC_IDS:
                # Silent truncation would corrupt the algebra ("first minus the
                # union of the REST"), so an oversized request is an error.
                return {"success": False,
                        "error": f"moc_operation accepts at most {MAX_MOC_IDS} survey_ids per call; "
                                 f"got {len(id_list)}. Split the request or pre-combine with union."}
            geometry = self.moc_geometry(id_list, order=order)
            if not geometry.get("success"):
                return geometry
            fetched = geometry.get("mocs") or []
            warnings = list(geometry.get("warnings") or [])

            def _target_flags(moc_or_none) -> Dict[str, Any]:
                """inside/outside flags against the derived MOC (all-False when empty)."""
                extra: Dict[str, Any] = {}
                if ra_list and dec_list and len(ra_list) == len(dec_list):
                    import astropy.units as u
                    import numpy as np

                    if moc_or_none is None:
                        flags = [False] * len(ra_list)
                    else:
                        lon = np.asarray([float(v) for v in ra_list], dtype=float) * u.deg
                        lat = np.asarray([float(v) for v in dec_list], dtype=float) * u.deg
                        try:
                            inside = moc_or_none.contains_lonlat(lon, lat)
                        except AttributeError:  # older mocpy API
                            inside = moc_or_none.contains(lon, lat)
                        flags = [bool(f) for f in np.asarray(inside).ravel()]
                    extra["targets_inside"] = flags
                    extra["n_targets_inside"] = int(sum(flags))
                    extra["n_targets"] = len(flags)
                elif ra_list or dec_list:
                    warnings.append("ra_list/dec_list ignored: lengths differ or one is missing.")
                return extra

            def _empty_result(note: str) -> Dict[str, Any]:
                out = {
                    "success": True,
                    "operation": op,
                    "ids": id_list,
                    "empty": True,
                    "area_deg2": 0.0,
                    "sky_fraction": 0.0,
                    "note": note,
                    "warnings": warnings,
                }
                out.update(_target_flags(None))
                return out

            # An id MOCServer returned nothing for is an EMPTY operand — it
            # must participate in the algebra, not silently vanish (CX-02):
            # intersection with empty is empty; difference from empty is empty;
            # an empty subtrahend / union operand is a no-op.
            fetched_ids = {m["id"] for m in fetched}
            missing = [i for i in id_list if i not in fetched_ids]
            if missing:
                if op == "intersection":
                    return _empty_result(
                        f"No MOC coverage found for {', '.join(missing)} — an intersection "
                        "with an empty footprint is empty."
                    )
                if op == "difference" and id_list and id_list[0] in missing:
                    return _empty_result(
                        f"No MOC coverage found for the first operand {id_list[0]!r} — "
                        "the difference of an empty footprint is empty."
                    )
                warnings.append(
                    f"Empty/unresolvable operand(s) treated as empty sky: {', '.join(missing)}."
                )
            if op in {"intersection", "difference"} and len(id_list) < 2:
                return {"success": False,
                        "error": f"{op} needs at least two survey_ids; got {len(id_list)}."}
            if not fetched:
                # Every operand resolved empty. That is a well-defined result —
                # the union (or any op) of empty footprints is empty sky, not an
                # error (CX-21).
                return _empty_result(
                    f"No MOC coverage found for any of {', '.join(id_list)} — result is empty sky."
                )

            from mocpy import MOC

            by_id = {m["id"]: MOC.from_json(m["moc_json"]) for m in fetched}
            present = [by_id[i] for i in id_list if i in by_id]
            if op == "union":
                combined = present[0]
                for m in present[1:]:
                    combined = combined.union(m)
            elif op == "intersection":
                combined = present[0]
                for m in present[1:]:
                    combined = combined.intersection(m)
            else:
                combined = by_id[id_list[0]]
                rest_mocs = [by_id[i] for i in id_list[1:] if i in by_id]
                if rest_mocs:
                    rest = rest_mocs[0]
                    for m in rest_mocs[1:]:
                        rest = rest.union(m)
                    combined = combined.difference(rest)

            if combined.empty():
                return _empty_result(
                    "The derived MOC is empty — the footprints do not "
                    + ("overlap." if op == "intersection" else "leave any residual sky.")
                )

            serialized = combined.serialize(format="json")
            n_cells = _moc_cell_count(serialized)
            # Keep the overlay payload bounded like moc_geometry does.
            local = combined
            while n_cells > 25000:
                max_order = max(int(k) for k in serialized.keys())
                if max_order <= MIN_MOC_ORDER:
                    break
                local = local.degrade_to_order(max_order - 1)
                serialized = local.serialize(format="json")
                n_cells = _moc_cell_count(serialized)
                warnings.append(f"Derived MOC downsampled to order {max_order - 1} for display.")

            sky_fraction = float(combined.sky_fraction)
            input_order = geometry.get("provenance", {}).get("order")
            result: Dict[str, Any] = {
                "success": True,
                "operation": op,
                "ids": id_list,
                "empty": False,
                "moc_json": serialized,
                "n_cells": n_cells,
                "sky_fraction": round(sky_fraction, 8),
                "area_deg2": round(sky_fraction * 41252.9612, 3),
                # Inputs are fetched at a capped HEALPix order, so the algebra
                # (area, membership) is exact only to that cell size.
                "area_precision_note": (
                    f"Computed from order-{input_order} input MOCs "
                    f"(cell ≈ {41252.9612 / (12 * 4 ** int(input_order or 8)):.4g} deg²); "
                    "boundaries finer than this are smoothed."
                ),
                "warnings": warnings,
                "provenance": {
                    "service": "CDS MOCServer + mocpy",
                    "endpoint": self.base_url,
                    "order": input_order,
                },
            }
            result.update(_target_flags(combined))
            return result
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _get_json_dict(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Like _get_json but for endpoints whose payload is a JSON object
        (MOC geometry responses are dicts, not record lists)."""
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
        if not isinstance(payload, dict):
            raise RuntimeError("MOCServer returned an unexpected JSON payload for a MOC geometry.")
        return payload

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


def _moc_cell_count(geometry: Dict[str, Any]) -> int:
    return sum(len(v) for v in geometry.values() if isinstance(v, list))


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