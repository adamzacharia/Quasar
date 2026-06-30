"""SVO Filter Profile Service client for photometric wavelengths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


LS_DR9_FILTER_IDS = {
    "g": "CTIO/DECam.g",
    "r": "CTIO/DECam.r",
    "z": "CTIO/DECam.z",
    "w1": "WISE/WISE.W1",
    "w2": "WISE/WISE.W2",
}


class SvoFpsClientError(RuntimeError):
    """Raised when SVO FPS cannot provide a requested filter wavelength."""


class SvoFpsClient:
    """Small cached wrapper around ``astroquery.svo_fps.SvoFps``."""

    def __init__(
        self,
        *,
        cache_dir: Path | str = Path("cache") / "svo_fps",
        ttl_seconds: Optional[int] = None,
        enable_disk_cache: Optional[bool] = None,
    ):
        self.ttl_seconds = int(
            ttl_seconds
            if ttl_seconds is not None
            else os.getenv("SVO_FPS_CACHE_TTL_SECONDS", "2592000")
        )
        if enable_disk_cache is None:
            flag = os.getenv("SVO_FPS_DISKCACHE_ENABLED", "true").strip().lower()
            enable_disk_cache = flag not in {"0", "false", "no", "off"}
        self._memory: Dict[str, Dict[str, Any]] = {}
        self._cache = None
        if enable_disk_cache:
            try:
                from diskcache import Cache

                self._cache = Cache(str(cache_dir))
            except Exception:
                self._cache = None

    def wavelength(self, filter_id: str) -> Dict[str, Any]:
        resolved = self.resolve_filter_id(filter_id)
        key = f"svo:{resolved}"
        cached = self._cache_get(key)
        if cached is not None:
            result = dict(cached)
            result.setdefault("provenance", {})["cached"] = True
            return result

        row = self._fetch_filter_row(resolved)
        result = self._row_to_wavelength(filter_id, resolved, row)
        self._cache_set(key, result)
        return result

    def wavelengths(self, filter_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        return {str(fid): self.wavelength(str(fid)) for fid in filter_ids}

    @staticmethod
    def resolve_filter_id(filter_id: str) -> str:
        text = str(filter_id or "").strip()
        if not text:
            raise ValueError("filter_id is required")
        return LS_DR9_FILTER_IDS.get(text.lower(), text)

    def _fetch_filter_row(self, filter_id: str) -> Mapping[str, Any]:
        try:
            from astroquery.svo_fps import SvoFps
        except ImportError as exc:
            raise ImportError("astroquery is required for SVO FPS wavelength lookups") from exc

        parts = filter_id.split("/", 1)
        facility = parts[0] if parts else None
        instrument = None
        if len(parts) == 2 and "." in parts[1]:
            instrument = parts[1].split(".", 1)[0]

        table = None
        errors = []
        for kwargs in (
            {"facility": facility, "instrument": instrument},
            {"facility": facility},
            {},
        ):
            try:
                table = SvoFps.get_filter_list(**{k: v for k, v in kwargs.items() if v})
                if table is not None:
                    row = self._find_filter_row(table, filter_id)
                    if row is not None:
                        return row
            except TypeError as exc:
                errors.append(str(exc))
                continue
            except Exception as exc:  # noqa: BLE001 - try broader fallback query shapes
                errors.append(str(exc))
                continue

        raise SvoFpsClientError(f"SVO FPS filter_id not found: {filter_id}; errors: {'; '.join(errors[:3])}")

    @staticmethod
    def _find_filter_row(table: Any, filter_id: str) -> Optional[Mapping[str, Any]]:
        rows = []
        if hasattr(table, "to_pandas"):
            rows = table.to_pandas().to_dict(orient="records")
        else:
            colnames = list(getattr(table, "colnames", []) or getattr(table, "columns", []) or [])
            for row in table:
                if isinstance(row, Mapping):
                    rows.append(dict(row))
                elif colnames:
                    rows.append({name: row[name] for name in colnames})
        for row in rows:
            row_filter = _lookup(row, "filterID", "filter_id", "FilterID")
            if str(row_filter).strip().lower() == filter_id.lower():
                return row
        return None

    @staticmethod
    def _row_to_wavelength(original: str, resolved: str, row: Mapping[str, Any]) -> Dict[str, Any]:
        effective = _to_float(_lookup(row, "WavelengthEff", "wavelength_eff", "lambda_eff"))
        pivot = _to_float(_lookup(row, "WavelengthPivot", "wavelength_pivot", "lambda_pivot"))
        if effective is None:
            effective = _to_float(_lookup(row, "WavelengthMean", "WavelengthCen"))
        if pivot is None:
            pivot = _to_float(_lookup(row, "WavelengthCen", "WavelengthMean"))
        if effective is None or pivot is None:
            raise SvoFpsClientError(f"SVO FPS row for {resolved} lacks effective or pivot wavelength")
        return {
            "success": True,
            "filter_id": original,
            "resolved_filter_id": resolved,
            "effective_angstrom": effective,
            "effective_micron": effective / 10000.0,
            "pivot_angstrom": pivot,
            "pivot_micron": pivot / 10000.0,
            "provenance": {"source": "SVO Filter Profile Service", "cached": False},
        }

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self._memory:
            return dict(self._memory[key])
        if self._cache is None:
            return None
        try:
            value = self._cache.get(key)
        except Exception:
            return None
        return dict(value) if isinstance(value, Mapping) else None

    def _cache_set(self, key: str, value: Mapping[str, Any]) -> None:
        payload = dict(value)
        self._memory[key] = payload
        if self._cache is None:
            return
        try:
            self._cache.set(key, payload, expire=self.ttl_seconds)
        except Exception:
            pass


def _lookup(row: Mapping[str, Any], *names: str) -> Any:
    lowered = {str(k).lower(): k for k in row.keys()}
    for name in names:
        key = lowered.get(str(name).lower())
        if key is not None:
            return row[key]
    return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["LS_DR9_FILTER_IDS", "SvoFpsClient", "SvoFpsClientError"]
