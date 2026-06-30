"""Stable Data Lab result store for follow-up tools."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from integrations.datalab_client import DatalabResult


class DatalabResultStore:
    """In-memory result store with optional DiskCache persistence."""

    def __init__(
        self,
        *,
        cache_dir: Path | str = Path("cache") / "datalab_results",
        ttl_seconds: Optional[int] = None,
        enable_disk_cache: Optional[bool] = None,
    ):
        self.ttl_seconds = int(
            ttl_seconds
            if ttl_seconds is not None
            else os.getenv("DATALAB_RESULT_TTL_SECONDS", "3600")
        )
        if enable_disk_cache is None:
            flag = os.getenv("DATALAB_RESULT_DISKCACHE_ENABLED", "true").strip().lower()
            enable_disk_cache = flag not in {"0", "false", "no", "off"}
        self._memory: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._cache = None
        if enable_disk_cache:
            try:
                from diskcache import Cache

                self._cache = Cache(str(cache_dir))
            except Exception as exc:
                self._cache = None
                logging.getLogger(__name__).warning(
                    "Data Lab result store disk cache unavailable (%s); result_ids will be "
                    "process-local and may not resolve across workers.", exc,
                )

    def put(self, dataframe: pd.DataFrame, meta: Optional[Mapping[str, Any]] = None) -> str:
        result_id = "dlr_" + uuid.uuid4().hex
        frame = dataframe.copy()
        payload = {
            "dataframe": frame,
            "meta": dict(meta or {}),
            "created_at": time.time(),
        }
        with self._lock:
            self._memory[result_id] = payload
        if self._cache is not None:
            try:
                self._cache.set(result_id, payload, expire=self.ttl_seconds)
            except Exception:
                pass
        return result_id

    def get(self, result_id: str) -> DatalabResult:
        key = str(result_id or "").strip()
        if not key:
            raise KeyError("Data Lab result_id is required")
        payload = None
        with self._lock:
            payload = self._memory.get(key)
            if payload is not None and self._is_expired(payload):
                # Evict expired in-memory results so long-running servers don't
                # retain DataFrames past their TTL.
                self._memory.pop(key, None)
                payload = None
        if payload is None and self._cache is not None:
            try:
                payload = self._cache.get(key)
            except Exception:
                payload = None
        if not isinstance(payload, Mapping):
            raise KeyError(f"Unknown Data Lab result_id: {result_id}")
        frame = payload.get("dataframe")
        if not isinstance(frame, pd.DataFrame):
            raise KeyError(f"Data Lab result_id has no DataFrame payload: {result_id}")
        meta = dict(payload.get("meta") or {})
        provenance = dict(meta.get("provenance") or meta)
        provenance.setdefault("result_id", key)
        provenance.setdefault("rowcount", int(len(frame)))
        return DatalabResult.from_dataframe(frame, provenance)

    def _is_expired(self, payload: Mapping[str, Any]) -> bool:
        if not self.ttl_seconds or self.ttl_seconds <= 0:
            return False
        created = float(payload.get("created_at") or 0.0)
        return (time.time() - created) > self.ttl_seconds


_DEFAULT_STORE: Optional[DatalabResultStore] = None


def default_result_store() -> DatalabResultStore:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = DatalabResultStore()
    return _DEFAULT_STORE


def put(dataframe: pd.DataFrame, meta: Optional[Mapping[str, Any]] = None) -> str:
    return default_result_store().put(dataframe, meta)


def get(result_id: str) -> DatalabResult:
    return default_result_store().get(result_id)


__all__ = ["DatalabResultStore", "default_result_store", "get", "put"]
