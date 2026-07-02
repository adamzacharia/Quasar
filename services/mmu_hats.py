"""Multimodal Universe HATS catalog access through LSDB/Hugging Face."""

from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


# AUTHORITATIVE for Quasar's MMU/HATS tool surface. Preferred columns are
# best-effort hints and are intersected with live catalog schemas at query time.
MMU_HATS_CATALOGS: Dict[str, Dict[str, Any]] = {
    # preferred_columns/struct_fields verified against the LIVE schemas on
    # 2026-07-02 (lsdb.open_catalog(...).dtypes). MMU catalogs are ML subsets:
    # scalar science values often live inside pyarrow struct columns
    # (struct_fields maps struct column -> scalar fields to flatten), and big
    # per-object arrays live in nested<> columns (excluded by default).
    "gaia": {
        "label": "Gaia DR3 (MMU)",
        "uri": "hf://datasets/UniverseTBD/mmu_gaia_gaia@main/mmu_gaia_gaia",
        "margin_uri": "hf://datasets/UniverseTBD/mmu_gaia_gaia@main/mmu_gaia_gaia_10arcs",
        "description": "Gaia astrometry/photometry source catalog (Multimodal Universe HATS; BP/RP-spectra bright-star subset, not full DR3).",
        "preferred_columns": ["object_id", "ra", "dec", "photometry", "astrometry", "flags", "radial_velocity"],
        "struct_fields": {
            "photometry": ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"],
            "astrometry": ["parallax", "parallax_error", "pmra", "pmdec"],
            "flags": ["ruwe"],
            "radial_velocity": ["radial_velocity"],
        },
    },
    "desi_edr_sv3": {
        "label": "DESI EDR SV3 (MMU)",
        "uri": "hf://datasets/UniverseTBD/mmu_desi_edr_sv3@main/mmu_desi_edr_sv3",
        "margin_uri": "hf://datasets/UniverseTBD/mmu_desi_edr_sv3@main/mmu_desi_edr_sv3_10arcs",
        "description": "DESI EDR SV3 redshift/source catalog (Multimodal Universe HATS).",
        "preferred_columns": ["object_id", "ra", "dec", "Z", "ZERR", "ZWARN", "EBV", "FLUX_G", "FLUX_R", "FLUX_Z"],
    },
    "sdss": {
        "label": "SDSS (MMU)",
        "uri": "hf://datasets/UniverseTBD/mmu_sdss_sdss@main/mmu_sdss_sdss",
        "margin_uri": "hf://datasets/UniverseTBD/mmu_sdss_sdss@main/mmu_sdss_sdss_10arcs",
        "description": "SDSS spectroscopic source/redshift catalog (Multimodal Universe HATS).",
        "preferred_columns": ["object_id", "ra", "dec", "Z", "Z_ERR", "ZWARNING", "VDISP", "SPECTROFLUX_G", "SPECTROFLUX_R", "SPECTROFLUX_I"],
    },
    "tess_spoc": {
        "label": "TESS SPOC (MMU)",
        "uri": "hf://datasets/UniverseTBD/mmu_tess_spoc@main/mmu_tess_spoc",
        "margin_uri": "hf://datasets/UniverseTBD/mmu_tess_spoc@main/mmu_tess_spoc_10arcs",
        "description": "TESS SPOC light-curve source catalog (Multimodal Universe HATS; scalar columns are object_id/ra/dec, light curves are nested).",
        "preferred_columns": ["object_id", "ra", "dec"],
    },
    "chandra_spectra": {
        "label": "Chandra Spectra (MMU)",
        "uri": "hf://datasets/UniverseTBD/mmu_chandra_spectra@main/mmu_chandra_spectra",
        "margin_uri": "hf://datasets/UniverseTBD/mmu_chandra_spectra@main/mmu_chandra_spectra_10arcs",
        "description": "Chandra source/spectra catalog (Multimodal Universe HATS).",
        "preferred_columns": [
            "object_id", "name", "ra", "dec", "flux_aper_b", "flux_significance_b",
            "hard_hm", "hard_hs", "hard_ms", "var_index_b", "var_prob_b",
        ],
    },
}


class MMUHatsError(ValueError):
    """Base error for MMU/HATS service failures."""


class MMUHatsUnavailableError(MMUHatsError):
    """Raised when the MMU/HATS optional runtime is disabled or unavailable."""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _is_nested_dtype(dtype: Any) -> bool:
    return "nested" in str(dtype).lower()


def compact_preview_frame(frame: pd.DataFrame, max_rows: int = 12) -> pd.DataFrame:
    """Preview-safe head slice: clip long strings, summarize nested/array cells.

    Tool JSON is hard-sliced at 8000 chars downstream; a single nested HATS
    cell (an embedded spectrum or light curve) could otherwise blow the
    payload into invalid truncated JSON.
    """
    head = frame.head(max_rows).copy()

    def _compact(value: Any) -> Any:
        if isinstance(value, str):
            return value if len(value) <= 200 else value[:200] + "..."
        if pd.api.types.is_scalar(value):
            return value
        try:
            size = len(value)
        except TypeError:
            size = None
        name = type(value).__name__
        return f"<{name}[{size}]>" if size is not None else f"<{name}>"

    for col in head.columns:
        head[col] = head[col].map(_compact)
    return head


class MMUHatsService:
    """Lazy LSDB-backed access to public MMU/HATS catalogs."""

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        max_radius_arcsec: Optional[float] = None,
        max_rows: Optional[int] = None,
        default_radius_arcsec: Optional[float] = None,
        default_rows: Optional[int] = None,
        cache_dir: Optional[Path | str] = None,
    ):
        self.enabled = _env_bool("ENABLE_MMU_HATS", True) if enabled is None else bool(enabled)
        self.max_radius_arcsec = float(
            max_radius_arcsec if max_radius_arcsec is not None else _env_float("MMU_HATS_MAX_RADIUS_ARCSEC", 3600.0)
        )
        self.max_rows_cap = int(max_rows if max_rows is not None else _env_int("MMU_HATS_MAX_ROWS", 5000))
        self.default_radius_arcsec = float(
            default_radius_arcsec
            if default_radius_arcsec is not None
            else _env_float("MMU_HATS_DEFAULT_RADIUS_ARCSEC", 120.0)
        )
        self.default_rows = int(default_rows if default_rows is not None else _env_int("MMU_HATS_DEFAULT_ROWS", 500))
        self.cache_dir = Path(cache_dir if cache_dir is not None else os.getenv("MMU_HATS_CACHE_DIR", Path("cache") / "mmu_hats"))
        self._lsdb_module = None

    def _import_lsdb(self):
        if self._lsdb_module is not None:
            return self._lsdb_module
        # Keep HF downloads under the configured MMU cache dir unless the
        # deployment already pinned HF_HOME elsewhere.
        os.environ.setdefault("HF_HOME", str(self.cache_dir / "huggingface"))
        # Deployments (Render) set the token as `HF_token`; env names are
        # case-sensitive on Linux and huggingface_hub only reads HF_TOKEN.
        alias_token = os.environ.get("HF_token")
        if alias_token and not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = alias_token
        importlib.import_module("huggingface_hub")
        self._lsdb_module = importlib.import_module("lsdb")
        return self._lsdb_module

    def is_available(self) -> Tuple[bool, str]:
        if not self.enabled:
            return False, "MMU HATS is disabled by ENABLE_MMU_HATS."
        try:
            self._import_lsdb()
            return True, ""
        except Exception as exc:
            return (
                False,
                "MMU HATS unavailable: install lsdb==0.9.2 and huggingface_hub, "
                f"or check ENABLE_MMU_HATS. Underlying error: {exc}",
            )

    def list_catalogs(self) -> List[Dict[str, Any]]:
        rows = []
        for key, entry in MMU_HATS_CATALOGS.items():
            rows.append(
                {
                    "key": key,
                    "label": entry["label"],
                    "uri": entry["uri"],
                    "description": entry["description"],
                    "preferred_columns": list(entry.get("preferred_columns", [])),
                }
            )
        return rows

    def cone_search(
        self,
        catalog_key: str,
        ra: Any,
        dec: Any,
        radius_arcsec: Optional[Any] = None,
        columns: Optional[Sequence[str] | str] = None,
        max_rows: Optional[Any] = None,
    ) -> Dict[str, Any]:
        entry = self._catalog_entry(catalog_key)
        ra_f, dec_f = self._validate_coords(ra, dec)
        warnings: List[str] = []
        radius = self._normalize_radius(radius_arcsec, warnings)
        row_cap = self._normalize_max_rows(max_rows, warnings)
        lsdb = self._available_lsdb()

        selected: List[str] = []
        try:
            lazy = self._open_raw_catalog(entry["uri"], lsdb, cone=(ra_f, dec_f, radius))
            selected = self._select_columns(lazy, entry, columns, warnings, ensure_coords=True)
            explicit_nested = self._explicit_nested_columns(lazy, selected, columns)
            if explicit_nested and row_cap > 50:
                warnings.append(
                    "Explicit nested MMU/HATS columns can be large; returned rows capped at 50 for this query."
                )
                row_cap = 50
            lazy = self._apply_column_selection(lazy, selected)
            df = self._normalize_dataframe(lazy.compute())
            df = self._expand_struct_columns(df, entry, warnings)
        except MMUHatsError:
            raise
        except Exception as exc:
            if self._is_no_coverage_error(exc):
                # MMU catalogs are ML subsets with real sky holes; an empty
                # table is the correct answer, not a failure.
                warnings.append(
                    f"{entry['label']} has no coverage at this position (MMU catalogs are "
                    "ML-focused survey subsets, not full-sky); 0 rows returned."
                )
                df = pd.DataFrame()
            else:
                raise MMUHatsError(f"MMU HATS query failed for catalog '{catalog_key}': {exc}") from exc

        df, rowcount = self._cap_dataframe(df, row_cap, warnings)
        return {
            "dataframe": df,
            "rowcount": rowcount,
            "returned_rows": int(len(df)),
            "columns": list(df.columns),
            "warnings": warnings,
            "provenance": {
                "backend": "lsdb+huggingface",
                "lsdb_version": str(getattr(lsdb, "__version__", "unknown") or "unknown"),
                "catalog_key": catalog_key,
                "catalog_label": entry["label"],
                "uri": entry["uri"],
                "cone": {"ra_deg": ra_f, "dec_deg": dec_f, "radius_arcsec": radius},
                "max_rows": row_cap,
                "requested_columns": self._normalize_columns(columns),
                "selected_columns": selected,
                "cache_dir": str(self.cache_dir),
            },
        }

    def open_catalog(
        self,
        catalog_key: str,
        columns: Optional[Sequence[str] | str] = None,
        cone: Optional[Tuple[Any, Any, Any]] = None,
        margin_cache: Optional[str] = None,
    ):
        entry = self._catalog_entry(catalog_key)
        lsdb = self._available_lsdb()
        cone_tuple = self._validate_cone_tuple(cone) if cone is not None else None
        lazy = self._open_raw_catalog(entry["uri"], lsdb, cone=cone_tuple, margin_cache=margin_cache)
        if columns is None:
            return lazy
        requested = self._normalize_columns(columns) or []
        schema = list(getattr(lazy, "columns", []))
        missing = [col for col in requested if col not in schema]
        if missing:
            raise MMUHatsError(f"Requested MMU/HATS columns not present in {catalog_key}: {missing}")
        return self._apply_column_selection(lazy, requested)

    def crossmatch_catalogs(
        self,
        left_key: str,
        right_key: str,
        ra: Any,
        dec: Any,
        radius_arcsec: Optional[Any] = None,
        match_radius_arcsec: Any = 1.0,
        columns_left: Optional[Sequence[str] | str] = None,
        columns_right: Optional[Sequence[str] | str] = None,
        max_rows: Optional[Any] = None,
    ) -> Dict[str, Any]:
        left_entry = self._catalog_entry(left_key)
        right_entry = self._catalog_entry(right_key)
        ra_f, dec_f = self._validate_coords(ra, dec)
        warnings: List[str] = []
        radius = self._normalize_radius(radius_arcsec, warnings)
        row_cap = self._normalize_max_rows(max_rows, warnings)
        match_radius = self._normalize_match_radius(match_radius_arcsec, warnings)
        lsdb = self._available_lsdb()

        try:
            cone = (ra_f, dec_f, radius)
            left = self._open_raw_catalog(left_entry["uri"], lsdb, cone=cone)
            right = self._open_raw_catalog(right_entry["uri"], lsdb, cone=cone, margin_cache=right_entry["margin_uri"])
            left_selected = self._select_columns(left, left_entry, columns_left, warnings, ensure_coords=True)
            right_selected = self._select_columns(right, right_entry, columns_right, warnings, ensure_coords=True)
            left = self._apply_column_selection(left, left_selected)
            right = self._apply_column_selection(right, right_selected)
            matched = left.crossmatch(
                right,
                radius_arcsec=match_radius,
                suffixes=("_" + left_key, "_" + right_key),
            )
            df = self._normalize_dataframe(matched.compute())
            self._add_canonical_crossmatch_coords(df, left_key)
        except MMUHatsError:
            raise
        except Exception as exc:
            if self._is_no_coverage_error(exc):
                warnings.append(
                    f"{left_entry['label']} x {right_entry['label']} have no overlapping coverage "
                    "at this position (MMU catalogs are ML-focused survey subsets); 0 rows returned."
                )
                df = pd.DataFrame()
                left_selected, right_selected = [], []
            else:
                raise MMUHatsError(
                    f"MMU HATS crossmatch failed for {left_key} x {right_key}: {exc}. "
                    "Fall back to two bounded search_mmu_hats_catalog cone searches."
                ) from exc

        df, rowcount = self._cap_dataframe(df, row_cap, warnings)
        return {
            "dataframe": df,
            "rowcount": rowcount,
            "returned_rows": int(len(df)),
            "columns": list(df.columns),
            "warnings": warnings,
            "provenance": {
                "backend": "lsdb+huggingface",
                "lsdb_version": str(getattr(lsdb, "__version__", "unknown") or "unknown"),
                "left_catalog_key": left_key,
                "right_catalog_key": right_key,
                "left_catalog_label": left_entry["label"],
                "right_catalog_label": right_entry["label"],
                "left_uri": left_entry["uri"],
                "right_uri": right_entry["uri"],
                "margin_uri": right_entry["margin_uri"],
                "cone": {"ra_deg": ra_f, "dec_deg": dec_f, "radius_arcsec": radius},
                "match_radius_arcsec": match_radius,
                "max_rows": row_cap,
                "requested_columns_left": self._normalize_columns(columns_left),
                "requested_columns_right": self._normalize_columns(columns_right),
                "selected_columns_left": left_selected,
                "selected_columns_right": right_selected,
                "cache_dir": str(self.cache_dir),
            },
        }

    def _available_lsdb(self):
        available, reason = self.is_available()
        if not available:
            raise MMUHatsUnavailableError(reason)
        return self._lsdb_module

    def _catalog_entry(self, catalog_key: str) -> Dict[str, Any]:
        key = str(catalog_key or "").strip()
        if key not in MMU_HATS_CATALOGS:
            raise MMUHatsError(f"Unknown MMU/HATS catalog '{catalog_key}'. Valid catalogs: {sorted(MMU_HATS_CATALOGS)}")
        return MMU_HATS_CATALOGS[key]

    def _validate_cone_tuple(self, cone: Tuple[Any, Any, Any]) -> Tuple[float, float, float]:
        if not cone or len(cone) != 3:
            raise MMUHatsError("MMU/HATS cone must be (ra, dec, radius_arcsec).")
        ra_f, dec_f = self._validate_coords(cone[0], cone[1])
        warnings: List[str] = []
        radius = self._normalize_radius(cone[2], warnings)
        return ra_f, dec_f, radius

    @staticmethod
    def _validate_coords(ra: Any, dec: Any) -> Tuple[float, float]:
        try:
            ra_f = float(ra)
            dec_f = float(dec)
        except (TypeError, ValueError):
            raise MMUHatsError("MMU/HATS ra and dec must be numeric ICRS degrees.")
        if not math.isfinite(ra_f) or not math.isfinite(dec_f):
            raise MMUHatsError("MMU/HATS ra and dec must be finite ICRS degrees.")
        if not 0.0 <= ra_f < 360.0:
            raise MMUHatsError("MMU/HATS ra must satisfy 0 <= ra < 360 degrees.")
        if not -90.0 <= dec_f <= 90.0:
            raise MMUHatsError("MMU/HATS dec must satisfy -90 <= dec <= 90 degrees.")
        return ra_f, dec_f

    def _normalize_radius(self, radius_arcsec: Optional[Any], warnings: List[str]) -> float:
        if radius_arcsec is None:
            return self.default_radius_arcsec
        try:
            radius = float(radius_arcsec)
        except (TypeError, ValueError):
            raise MMUHatsError("MMU/HATS radius_arcsec must be numeric.")
        if not math.isfinite(radius):
            raise MMUHatsError("MMU/HATS radius_arcsec must be finite.")
        if radius <= 0:
            warnings.append(
                f"radius_arcsec <= 0; using default radius {self.default_radius_arcsec:g} arcsec."
            )
            return self.default_radius_arcsec
        if radius > self.max_radius_arcsec:
            warnings.append(
                f"radius_arcsec {radius:g} exceeds cap {self.max_radius_arcsec:g}; clamped."
            )
            return self.max_radius_arcsec
        return radius

    def _normalize_max_rows(self, max_rows: Optional[Any], warnings: List[str]) -> int:
        if max_rows is None:
            rows = self.default_rows
        else:
            try:
                rows = int(max_rows)
            except (TypeError, ValueError):
                raise MMUHatsError("MMU/HATS max_rows must be an integer.")
        if rows <= 0:
            warnings.append(f"max_rows <= 0; using default row cap {self.default_rows}.")
            rows = self.default_rows
        if rows > self.max_rows_cap:
            warnings.append(f"max_rows {rows} exceeds cap {self.max_rows_cap}; clamped.")
            rows = self.max_rows_cap
        return rows

    @staticmethod
    def _normalize_match_radius(match_radius_arcsec: Any, warnings: List[str]) -> float:
        try:
            radius = float(match_radius_arcsec)
        except (TypeError, ValueError):
            raise MMUHatsError("MMU/HATS match_radius_arcsec must be numeric.")
        if not math.isfinite(radius):
            raise MMUHatsError("MMU/HATS match_radius_arcsec must be finite.")
        if radius <= 0:
            warnings.append("match_radius_arcsec <= 0; using 1 arcsec.")
            return 1.0
        if radius > 10.0:
            warnings.append("match_radius_arcsec exceeds the 10 arcsec margin-cache limit; clamped to 10.")
            return 10.0
        return radius

    @staticmethod
    def _normalize_columns(columns: Optional[Sequence[str] | str]) -> Optional[List[str]]:
        if columns is None:
            return None
        if isinstance(columns, str):
            return [part.strip() for part in columns.split(",") if part.strip()]
        try:
            return [str(col).strip() for col in columns if str(col).strip()]
        except TypeError:
            raise MMUHatsError("MMU/HATS columns must be a list of column names.")

    def _open_raw_catalog(self, uri: str, lsdb: Any, *, cone: Optional[Tuple[float, float, float]] = None, margin_cache: Optional[str] = None):
        kwargs: Dict[str, Any] = {}
        if cone is not None:
            kwargs["search_filter"] = lsdb.ConeSearch(ra=cone[0], dec=cone[1], radius_arcsec=cone[2])
        if margin_cache:
            kwargs["margin_cache"] = margin_cache
        return lsdb.open_catalog(uri, **kwargs)

    def _select_columns(
        self,
        lazy: Any,
        entry: Dict[str, Any],
        columns: Optional[Sequence[str] | str],
        warnings: List[str],
        *,
        ensure_coords: bool,
    ) -> List[str]:
        schema = list(getattr(lazy, "columns", []))
        dtypes = getattr(lazy, "dtypes", {})
        non_nested = [col for col in schema if not _is_nested_dtype(self._dtype_for(dtypes, col))]
        requested = self._normalize_columns(columns)

        if requested is not None:
            missing = [col for col in requested if col not in schema]
            if missing:
                warnings.append(f"Requested columns not present in {entry['label']}: {missing}")
            selected = [col for col in requested if col in schema]
            if not selected:
                raise MMUHatsError(
                    f"None of the requested MMU/HATS columns are present in {entry['label']}. "
                    f"Available non-nested columns include: {non_nested[:24]}"
                )
        else:
            preferred = list(entry.get("preferred_columns", []))
            selected = [
                col for col in preferred
                if col in schema and not _is_nested_dtype(self._dtype_for(dtypes, col))
            ]
            dropped = [col for col in preferred if col not in schema]
            if dropped:
                warnings.append(f"Preferred columns not present in {entry['label']}: {dropped}")
            if not selected:
                selected = non_nested[:24]
                warnings.append(f"No preferred columns matched {entry['label']}; auto-selected first non-nested columns.")

        if ensure_coords:
            for coord in ("ra", "dec"):
                if coord in schema and coord not in selected and coord in non_nested:
                    selected.append(coord)
        if not selected:
            raise MMUHatsError(f"No selectable non-nested columns found for {entry['label']}.")
        return selected

    @staticmethod
    def _dtype_for(dtypes: Any, column: str) -> Any:
        try:
            return dtypes[column]
        except Exception:
            try:
                return getattr(dtypes, column)
            except Exception:
                return ""

    @staticmethod
    def _apply_column_selection(lazy: Any, selected: List[str]):
        schema = list(getattr(lazy, "columns", []))
        if selected and (len(selected) != len(schema) or set(selected) != set(schema)):
            return lazy[selected]
        return lazy

    def _explicit_nested_columns(self, lazy: Any, selected: List[str], columns: Optional[Sequence[str] | str]) -> List[str]:
        if columns is None:
            return []
        dtypes = getattr(lazy, "dtypes", {})
        return [col for col in selected if _is_nested_dtype(self._dtype_for(dtypes, col))]

    @staticmethod
    def _is_no_coverage_error(exc: Exception) -> bool:
        return "no coverage" in str(exc).lower()

    @staticmethod
    def _expand_struct_columns(df: pd.DataFrame, entry: Dict[str, Any], warnings: List[str]) -> pd.DataFrame:
        """Flatten registry-listed scalar fields out of pyarrow struct columns.

        The useful Gaia values (parallax, G mag, ...) live inside struct
        columns; flat columns are what the data card and the model can use.
        """
        struct_fields: Dict[str, List[str]] = entry.get("struct_fields") or {}
        for col, fields in struct_fields.items():
            if col not in df.columns:
                continue
            series = df[col]
            for field in fields:
                out_name = field if field not in df.columns else f"{col}_{field}"
                try:
                    try:
                        df[out_name] = series.struct.field(field)
                    except AttributeError:
                        df[out_name] = series.map(
                            lambda v, _f=field: v.get(_f) if isinstance(v, dict) else None
                        )
                except Exception:
                    warnings.append(f"Could not expand struct field {col}.{field}.")
            df = df.drop(columns=[col])
        return df

    @staticmethod
    def _normalize_dataframe(frame: Any) -> pd.DataFrame:
        df = pd.DataFrame(frame)
        if not isinstance(df.index, pd.RangeIndex) or any(name is not None for name in df.index.names):
            df = df.reset_index()
        if "_healpix_29" in df.columns:
            df = df.drop(columns=["_healpix_29"])
        return pd.DataFrame(df)

    @staticmethod
    def _cap_dataframe(df: pd.DataFrame, max_rows: int, warnings: List[str]) -> Tuple[pd.DataFrame, int]:
        rowcount = int(len(df))
        if rowcount > max_rows:
            warnings.append(f"Returned rows truncated from {rowcount} to {max_rows}.")
            df = df.head(max_rows).copy()
        return df, rowcount

    @staticmethod
    def _add_canonical_crossmatch_coords(df: pd.DataFrame, left_key: str) -> None:
        ra_col = f"ra_{left_key}"
        dec_col = f"dec_{left_key}"
        if "ra" not in df.columns and ra_col in df.columns:
            df.insert(0, "ra", df[ra_col])
        if "dec" not in df.columns and dec_col in df.columns:
            insert_at = 1 if "ra" in df.columns else 0
            df.insert(insert_at, "dec", df[dec_col])


_DEFAULT_SERVICE: Optional[MMUHatsService] = None


def default_mmu_hats_service() -> MMUHatsService:
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = MMUHatsService()
    return _DEFAULT_SERVICE


__all__ = [
    "MMU_HATS_CATALOGS",
    "MMUHatsError",
    "MMUHatsService",
    "MMUHatsUnavailableError",
    "compact_preview_frame",
    "default_mmu_hats_service",
]
