"""Stable Data Lab result store for follow-up tools, plus MyDB-lite: durable
user-named tables saved from ephemeral result_ids (SQL-side MYDB/VOSPACE stays
blocked by the governor — persistence is client-side on the DiskCache)."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import uuid
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from integrations.datalab_client import DatalabResult

# My-tables namespace: names are slugs so they can never collide with dlr_ ids
# or carry path/SQL-hostile characters; the index key lists them for the panel.
_MY_TABLE_KEY_PREFIX = "dlt_"
_MY_TABLES_INDEX_KEY = "dlt_index_v1"
_MY_TABLE_NAME_RE = re.compile(r"^[a-z][a-z0-9_\-]{0,63}$")
_MY_TABLES_MAX = int(os.getenv("DATALAB_MY_TABLES_MAX", "50"))


def normalize_my_table_name(name: str) -> str:
    text = str(name or "").strip().lower().replace(" ", "_")
    if not _MY_TABLE_NAME_RE.match(text):
        raise ValueError(
            "My-table names must be 1-64 chars: start with a letter, then "
            "letters/digits/underscore/hyphen (e.g. 'lmc_rr_lyrae_candidates')."
        )
    return text


def _my_table_namespace(user_id: Optional[str]) -> str:
    """Return a stable storage-key namespace without embedding the raw user ID."""
    identity = "" if user_id is None else str(user_id).strip()
    if not identity:
        return ""
    return sha256(identity.encode("utf-8")).hexdigest()


def _my_table_keys(
    name: Optional[str] = None, *, user_id: Optional[str] = None
) -> tuple[str, Optional[str]]:
    """Return index/payload keys while preserving the legacy default namespace."""
    namespace = _my_table_namespace(user_id)
    if not namespace:
        return _MY_TABLES_INDEX_KEY, (
            _MY_TABLE_KEY_PREFIX + name if name is not None else None
        )
    index_key = f"{_MY_TABLES_INDEX_KEY}:user:{namespace}"
    table_key = (
        f"{_MY_TABLE_KEY_PREFIX}user:{namespace}:{name}"
        if name is not None
        else None
    )
    return index_key, table_key


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
        # Hard byte cap on the in-memory dict: eviction used to be LAZY (only
        # on a get() of that exact key), so never-re-read results lingered past
        # TTL — measured ~1-2 MB/result ⇒ a busy hour ≈ 1 GB retained, OOM
        # territory on a 2 GB Render instance. Evicted entries still resolve
        # from the DiskCache.
        self.memory_cap_bytes = int(
            float(os.getenv("DATALAB_RESULT_MEMORY_CAP_MB", "200")) * 1024 * 1024
        )
        if enable_disk_cache is None:
            flag = os.getenv("DATALAB_RESULT_DISKCACHE_ENABLED", "true").strip().lower()
            enable_disk_cache = flag not in {"0", "false", "no", "off"}
        self._memory: Dict[str, Dict[str, Any]] = {}
        # Compound my-table operations call internal index helpers under this
        # lock. RLock makes helper reuse safe without sacrificing atomicity.
        self._lock = threading.RLock()
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
        try:
            nbytes = int(frame.memory_usage(deep=True).sum())
        except Exception:  # noqa: BLE001 - sizing is best-effort
            nbytes = 0
        now = time.time()
        payload = {
            "dataframe": frame,
            "meta": dict(meta or {}),
            "created_at": now,
            "last_used": now,
            "nbytes": nbytes,
        }
        with self._lock:
            self._memory[result_id] = payload
            self._enforce_memory_bounds_locked()
        if self._cache is not None:
            try:
                self._cache.set(result_id, payload, expire=self.ttl_seconds)
            except Exception:
                pass
        return result_id

    def _enforce_memory_bounds_locked(self) -> None:
        """Sweep expired entries and LRU-evict past the byte cap (lock held).

        Runs on every put(), so retention is bounded by write activity rather
        than by whether anyone happens to re-read an expired key. My-table
        (dlt_) payloads are durable and never swept here.
        """
        expired = [
            key for key, payload in self._memory.items()
            if key.startswith("dlr_") and self._is_expired(payload)
        ]
        for key in expired:
            self._memory.pop(key, None)
        if self.memory_cap_bytes <= 0:
            return
        ephemeral = [
            (key, payload) for key, payload in self._memory.items() if key.startswith("dlr_")
        ]
        total = sum(int(p.get("nbytes") or 0) for _k, p in ephemeral)
        if total <= self.memory_cap_bytes:
            return
        for key, payload in sorted(ephemeral, key=lambda kv: float(kv[1].get("last_used") or 0.0)):
            self._memory.pop(key, None)
            total -= int(payload.get("nbytes") or 0)
            if total <= self.memory_cap_bytes:
                break

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
            elif payload is not None:
                payload["last_used"] = time.time()
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

    # ── My-tables (MyDB-lite): durable named tables, no TTL ────────────────────
    def save_result(
        self,
        result_id: str,
        name: str,
        *,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Promote an ephemeral result_id to a durable named table.

        The payload is COPIED under a dlt_ key with no expiry (the dlr_ key
        keeps its 1-hour TTL), in memory and — when available — on the same
        DiskCache, so named tables survive restarts. ``user_id`` selects an
        isolated namespace; omitted IDs retain the legacy shared namespace.
        """
        table_name = normalize_my_table_name(name)
        result = self.get(result_id)  # raises KeyError for unknown/expired ids
        frame = result.dataframe
        payload = {
            "dataframe": frame.copy(),
            "meta": {
                "provenance": dict(result.provenance),
                "my_table": table_name,
                "saved_from": str(result_id),
                "description": (str(description).strip() if description else None),
            },
            "created_at": time.time(),
        }
        entry = {
            "name": table_name,
            "rowcount": int(len(frame)),
            "columns": [str(col) for col in frame.columns][:40],
            "saved_at": payload["created_at"],
            "saved_from": str(result_id),
            "description": payload["meta"]["description"],
            "catalog": result.provenance.get("catalog"),
            "table": result.provenance.get("table"),
        }
        index_key, key = _my_table_keys(table_name, user_id=user_id)
        assert key is not None
        with self._lock:
            existing = self._read_index_unlocked(index_key)
            try:
                updated = self._persist_saved_table_unlocked(
                    index_key, key, table_name, payload, entry, existing
                )
            except ValueError:
                raise
            except Exception:
                logging.getLogger(__name__).warning(
                    "Could not persist my-table %r to disk; it will be process-local.",
                    table_name,
                )
                self._check_table_limit(existing, table_name)
                updated = {**existing, table_name: entry}
            self._memory[key] = payload
            self._memory[index_key] = updated
        return dict(entry)

    def list_my_tables(self, *, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        index = self._read_index(user_id=user_id)
        return sorted(
            index.values(),
            key=lambda entry: float(entry.get("saved_at") or 0.0),
            reverse=True,
        )

    def load_my_table(self, name: str, *, user_id: Optional[str] = None) -> DatalabResult:
        table_name = normalize_my_table_name(name)
        index_key, key = _my_table_keys(table_name, user_id=user_id)
        assert key is not None
        with self._lock:
            payload = self._memory.get(key)
            if payload is None and self._cache is not None:
                try:
                    payload = self._cache.get(key)
                except Exception:
                    payload = None
                if isinstance(payload, Mapping):
                    self._memory[key] = dict(payload)
            if not isinstance(payload, Mapping) or not isinstance(
                payload.get("dataframe"), pd.DataFrame
            ):
                known = ", ".join(
                    sorted(self._read_index_unlocked(index_key))
                ) or "(none saved yet)"
                raise KeyError(f"Unknown my-table: {name!r}. Saved tables: {known}")
        meta = dict(payload.get("meta") or {})
        provenance = dict(meta.get("provenance") or {})
        provenance.update({"my_table": table_name, "saved_from": meta.get("saved_from")})
        return DatalabResult.from_dataframe(payload["dataframe"], provenance)

    def delete_my_table(self, name: str, *, user_id: Optional[str] = None) -> Dict[str, Any]:
        table_name = normalize_my_table_name(name)
        index_key, key = _my_table_keys(table_name, user_id=user_id)
        assert key is not None
        with self._lock:
            index = self._read_index_unlocked(index_key)
            entry = index.get(table_name)
            if entry is None:
                raise KeyError(f"Unknown my-table: {name!r}")
            updated = dict(index)
            updated.pop(table_name)
            try:
                updated = self._persist_deleted_table_unlocked(
                    index_key, key, table_name, updated
                )
            except Exception:
                logging.getLogger(__name__).warning(
                    "Could not delete persisted my-table %r; deletion is process-local.",
                    table_name,
                )
            self._memory.pop(key, None)
            self._memory[index_key] = updated
        return dict(entry)

    @staticmethod
    def _check_table_limit(index: Mapping[str, Any], table_name: str) -> None:
        if table_name not in index and len(index) >= _MY_TABLES_MAX:
            raise ValueError(
                f"My-tables limit reached ({_MY_TABLES_MAX}). Delete one first "
                "(see the My tables panel) or overwrite an existing name."
            )

    def _cache_transaction(self):
        transact = getattr(self._cache, "transact", None)
        return transact() if callable(transact) else nullcontext()

    def _disk_index_unlocked(self, index_key: str) -> Optional[Dict[str, Dict[str, Any]]]:
        if self._cache is None:
            return None
        index = self._cache.get(index_key)
        return dict(index) if isinstance(index, Mapping) else None

    def _persist_saved_table_unlocked(
        self,
        index_key: str,
        table_key: str,
        table_name: str,
        payload: Dict[str, Any],
        entry: Dict[str, Any],
        fallback_index: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        if self._cache is None:
            self._check_table_limit(fallback_index, table_name)
            return {**fallback_index, table_name: entry}
        with self._cache_transaction():
            existing = self._disk_index_unlocked(index_key)
            if existing is None:
                existing = fallback_index
            self._check_table_limit(existing, table_name)
            updated = {**existing, table_name: entry}
            self._cache.set(table_key, payload)  # no expiry — durable
            self._cache.set(index_key, updated)  # no expiry — durable
        return updated

    def _persist_deleted_table_unlocked(
        self,
        index_key: str,
        table_key: str,
        table_name: str,
        fallback_index: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        if self._cache is None:
            return fallback_index
        with self._cache_transaction():
            existing = self._disk_index_unlocked(index_key)
            if existing is not None:
                updated = dict(existing)
                updated.pop(table_name, None)
            else:
                updated = fallback_index
            self._cache.delete(table_key)
            self._cache.set(index_key, updated)  # no expiry — durable
        return updated

    def _read_index(self, *, user_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        index_key, _ = _my_table_keys(user_id=user_id)
        with self._lock:
            return self._read_index_unlocked(index_key)

    def _read_index_unlocked(self, index_key: str) -> Dict[str, Dict[str, Any]]:
        index = self._memory.get(index_key)
        if self._cache is not None:
            try:
                persisted = self._cache.get(index_key)
            except Exception:
                persisted = None
            if isinstance(persisted, Mapping):
                index = dict(persisted)
                self._memory[index_key] = index
        return dict(index) if isinstance(index, Mapping) else {}

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


__all__ = ["DatalabResultStore", "default_result_store", "get", "normalize_my_table_name", "put"]
