"""ATNF pulsar catalogue search and lookup via psrqpy."""

from __future__ import annotations

import math
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd


ATNF_DEFAULT_BASE_URL = "https://www.atnf.csiro.au/research/pulsar/psrcat/"
PSR_PARAMS = [
    "JNAME",
    "NAME",
    "BNAME",
    "RAJD",
    "DECJD",
    "P0",
    "P1",
    "DM",
    "DIST",
    "AGE",
    "BSURF",
    "EDOT",
    "S1400",
    "BINARY",
    "ASSOC",
    "TYPE",
]
FULL_COLUMNS = [
    "jname",
    "name",
    "bname",
    "ra_deg",
    "dec_deg",
    "p0_s",
    "p1",
    "dm_pc_cm3",
    "dist_kpc",
    "age_yr",
    "bsurf_g",
    "edot_erg_s",
    "s1400_mjy",
    "binary",
    "assoc",
    "type",
]
FIELD_MAP = {
    "JNAME": "jname",
    "NAME": "name",
    "BNAME": "bname",
    "RAJD": "ra_deg",
    "DECJD": "dec_deg",
    "P0": "p0_s",
    "P1": "p1",
    "DM": "dm_pc_cm3",
    "DIST": "dist_kpc",
    "AGE": "age_yr",
    "BSURF": "bsurf_g",
    "EDOT": "edot_erg_s",
    "S1400": "s1400_mjy",
    "BINARY": "binary",
    "ASSOC": "assoc",
    "TYPE": "type",
}


class PulsarCatalogService:
    """Cached, locally filtered ATNF pulsar catalogue service."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        table_loader: Optional[Callable[[], Any]] = None,
    ):
        self.base_url = str(base_url or os.getenv("ATNF_PULSAR_BASE_URL") or ATNF_DEFAULT_BASE_URL).rstrip("/")
        self.timeout = float(timeout if timeout is not None else _env_float("ATNF_PULSAR_TIMEOUT", 30.0))
        self.table_loader = table_loader
        self._table: Optional[pd.DataFrame] = None
        self._catalogue_version: Optional[str] = None
        self._load_lock = threading.Lock()

    def search_pulsars(self, ra: Any, dec: Any, radius_deg: Any = 1.0, max_rows: Any = 25) -> Dict[str, Any]:
        """Return pulsars inside an ICRS cone, sorted by angular separation."""
        try:
            warnings: List[str] = []
            ra_f, dec_f = _validate_coords(ra, dec)
            radius, radius_warnings = _normalize_radius_deg(radius_deg)
            row_limit, row_warnings = _normalize_max_rows(max_rows)
            warnings.extend(radius_warnings)
            warnings.extend(row_warnings)

            table = self._load_table()
            positioned_rows: List[Tuple[int, float, float]] = []
            skipped_positions = 0
            for pos, (_, row) in enumerate(table.iterrows()):
                psr_ra = _float_or_none(row.get("RAJD"))
                psr_dec = _float_or_none(row.get("DECJD"))
                if psr_ra is None or psr_dec is None:
                    skipped_positions += 1
                    continue
                positioned_rows.append((pos, psr_ra, psr_dec))
            if skipped_positions:
                warnings.append(f"Skipped {skipped_positions} ATNF row(s) without RAJD/DECJD.")

            matches: List[Tuple[float, Any]] = []
            if positioned_rows:
                sep_arcmin = _separations_arcmin(
                    ra_f,
                    dec_f,
                    [item[1] for item in positioned_rows],
                    [item[2] for item in positioned_rows],
                )
                for (pos, _, _), sep in zip(positioned_rows, sep_arcmin):
                    if sep / 60.0 <= radius:
                        matches.append((float(sep), pos))
            matches.sort(key=lambda item: item[0])
            total_matches = len(matches)
            if total_matches > row_limit:
                warnings.append(f"ATNF cone matched {total_matches} pulsar(s); showing first {row_limit}.")
            rows = [_normalize_row(table.iloc[pos], sep_arcmin=round(sep, 2)) for sep, pos in matches[:row_limit]]
            return {
                "success": True,
                "rows": rows,
                "count": len(rows),
                "total_matches": total_matches,
                "warnings": warnings,
                "provenance": self._provenance(
                    ra=ra_f,
                    dec=dec_f,
                    radius_deg=radius,
                    requested_radius_deg=_float_or_none(radius_deg),
                    max_rows=row_limit,
                ),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def pulsar_lookup(self, name: Any) -> Dict[str, Any]:
        """Return ATNF rows matching a pulsar J-name, B-name, or NAME."""
        try:
            query = _canonical_name(name)
            if not query:
                raise ValueError("name is required.")
            table = self._load_table()
            rows: List[Dict[str, Any]] = []
            seen: set = set()
            for idx, row in table.iterrows():
                variants: set = set()
                for column in ("JNAME", "BNAME", "NAME"):
                    variants.update(_name_variants(row.get(column)))
                if query not in variants:
                    continue
                dedupe_key = (
                    _str_or_none(row.get("JNAME")),
                    _str_or_none(row.get("BNAME")),
                    _str_or_none(row.get("NAME")),
                    idx,
                )
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                rows.append(_normalize_row(row))
            return {
                "success": True,
                "rows": rows,
                "count": len(rows),
                "warnings": [],
                "provenance": self._provenance(name=str(name).strip(), canonical_name=query),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _load_table(self) -> pd.DataFrame:
        if self._table is not None:
            return self._table
        with self._load_lock:
            if self._table is not None:
                return self._table
            try:
                if self.table_loader is not None:
                    raw_table = self.table_loader()
                    version = None
                else:
                    raw_table, version = self._load_default_table()
                table = _coerce_table(raw_table)
            except Exception as exc:
                raise RuntimeError(f"ATNF catalogue download failed: {exc}") from exc
            self._table = table
            self._catalogue_version = _str_or_none(version)
            return table

    def _load_default_table(self) -> Tuple[Any, Optional[str]]:
        from psrqpy import QueryATNF

        query = QueryATNF(params=PSR_PARAMS)
        raw_table = query.pandas
        if callable(raw_table):
            raw_table = raw_table()
        return raw_table, _query_version(query)

    def _provenance(self, **params: Any) -> Dict[str, Any]:
        provenance = {
            "service": "ATNF Pulsar Catalogue (psrqpy)",
            "base_url": self.base_url,
            "catalogue_version": self._catalogue_version,
            "timeout_s": self.timeout,
        }
        provenance.update(params)
        return provenance


def _query_version(query: Any) -> Optional[str]:
    for name in ("get_version", "version"):
        try:
            value = getattr(query, name, None)
            if callable(value):
                value = value()
            text = _str_or_none(value)
            if text:
                return text
        except Exception:
            continue
    return None


def _coerce_table(raw_table: Any) -> pd.DataFrame:
    if hasattr(raw_table, "to_pandas"):
        raw_table = raw_table.to_pandas()
    if isinstance(raw_table, pd.DataFrame):
        table = raw_table.copy()
    else:
        table = pd.DataFrame(raw_table)
    table.columns = [str(column).strip().upper() for column in table.columns]
    table = table.loc[:, ~table.columns.duplicated()].copy()
    for column in PSR_PARAMS:
        if column not in table.columns:
            table[column] = None
    return table[PSR_PARAMS].copy()


def _validate_coords(ra: Any, dec: Any) -> Tuple[float, float]:
    ra_f = _finite_float(ra, "ra")
    dec_f = _finite_float(dec, "dec")
    if not 0.0 <= ra_f < 360.0:
        raise ValueError("ra must satisfy 0 <= ra < 360 degrees.")
    if not -90.0 <= dec_f <= 90.0:
        raise ValueError("dec must satisfy -90 <= dec <= 90 degrees.")
    return ra_f, dec_f


def _normalize_radius_deg(value: Any) -> Tuple[float, List[str]]:
    warnings: List[str] = []
    try:
        radius = float(value)
    except (TypeError, ValueError):
        radius = 1.0
        warnings.append("radius_deg was not numeric; using 1 deg.")
    if not math.isfinite(radius) or radius <= 0:
        radius = 1.0
        warnings.append("radius_deg must be positive; using 1 deg.")
    elif radius > 30.0:
        warnings.append(f"radius_deg {radius:g} exceeds 30; clamped to 30 deg.")
        radius = 30.0
    return radius, warnings


def _normalize_max_rows(value: Any) -> Tuple[int, List[str]]:
    warnings: List[str] = []
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = 25
        warnings.append("max_rows was not an integer; using 25.")
    if out <= 0:
        out = 25
        warnings.append("max_rows must be positive; using 25.")
    elif out > 200:
        warnings.append(f"max_rows {out} exceeds 200; clamped to 200.")
        out = 200
    return out, warnings


def _separations_arcmin(ra: float, dec: float, catalog_ra: List[float], catalog_dec: List[float]) -> List[float]:
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    catalog = SkyCoord(ra=catalog_ra * u.deg, dec=catalog_dec * u.deg, frame="icrs")
    return [float(value) for value in target.separation(catalog).arcmin]


def _normalize_row(row: Any, sep_arcmin: Optional[float] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for source_key, output_key in FIELD_MAP.items():
        value = row.get(source_key)
        if source_key in {"JNAME", "NAME", "BNAME", "BINARY", "ASSOC", "TYPE"}:
            out[output_key] = _str_or_none(value)
        else:
            out[output_key] = _rounded_float(source_key, value)
    if sep_arcmin is not None:
        out["sep_arcmin"] = float(sep_arcmin)
    return out


def _rounded_float(source_key: str, value: Any) -> Optional[float]:
    out = _float_or_none(value)
    if out is None:
        return None
    if source_key in {"RAJD", "DECJD"}:
        return round(out, 6)
    if source_key == "P0":
        return _round_sigfig(out, 6)
    if source_key == "DM":
        return round(out, 2)
    return _round_sigfig(out, 4)


def _canonical_name(value: Any) -> str:
    text = _str_or_none(value)
    if not text:
        return ""
    compact = "".join(str(text).upper().strip().split())
    if compact.startswith("PSR"):
        compact = compact[3:]
    return compact


def _name_variants(value: Any) -> set:
    canonical = _canonical_name(value)
    variants = {canonical} if canonical else set()
    if len(canonical) > 1 and canonical[0] in {"J", "B"}:
        variants.add(canonical[1:])
    return variants


def _finite_float(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def _float_or_none(value: Any) -> Optional[float]:
    if _is_null(value):
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
    if _is_null(value):
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    try:
        import numpy as np

        return bool(np.ma.is_masked(value))
    except Exception:
        return False


def _round_sigfig(value: float, digits: int) -> float:
    return float(f"{float(value):.{digits}g}")


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value > 0 else default


__all__ = ["FULL_COLUMNS", "PSR_PARAMS", "PulsarCatalogService"]