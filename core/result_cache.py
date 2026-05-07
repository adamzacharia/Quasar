# core/result_cache.py
"""
Result Cache -- Avoid redundant tool calls across Conductor subtasks.

If a user asks two similar complex queries in one session (e.g. "Compare ALMA
data on M87 with papers" then "What CO lines does M87 cover in Band 6?"),
the Conductor would run completely fresh DAGs -- re-searching ALMA for M87
data both times.

This cache stores tool call results keyed on (tool_name, canonical_args)
with a configurable TTL.  Cache hits skip the API call entirely.

Design decisions to avoid past issues:
  - Keys are STRICT: exact tool_name + sorted canonical args hash.
    No fuzzy matching -- similar-but-different queries always miss.
  - Short TTL (5 min default) prevents stale data from being served.
  - Error/empty results are NEVER cached.
  - A force_fresh flag lets callers bypass the cache explicitly.
  - Per-conversation scoping: cache is cleared between conversations.
  - Thread-safe for Conductor's parallel execution.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ResultCache:
    """
    Strict-match cache for tool call results with TTL expiration.

    Keys are computed from (tool_name, sorted_args_hash) -- NO fuzzy
    matching.  This avoids the "similar question returns wrong data"
    problem entirely.

    Usage
    -----
    cache = ResultCache(ttl_seconds=300)

    # Before calling a tool:
    key = cache.make_key("search_by_target", {"target": "M87"})
    cached = cache.get(key)
    if cached is not None:
        return cached  # Skip the API call!

    # After calling a tool:
    result = tool.execute(**args)
    cache.set(key, result)
    """

    def __init__(self, ttl_seconds: int = 300, max_entries: int = 100):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @staticmethod
    def make_key(tool_name: str, args: Dict[str, Any]) -> str:
        """
        Create a deterministic cache key from tool name + arguments.

        STRICT matching only -- arguments are sorted, serialized, and
        hashed.  Different arg values = different key, always.
        """
        # Normalize: remove None values, sort keys, stringify
        clean_args = {
            k: v for k, v in sorted(args.items())
            if v is not None and v != "" and v != []
        }
        args_str = json.dumps(clean_args, sort_keys=True, default=str)
        args_hash = hashlib.sha256(args_str.encode()).hexdigest()[:16]
        return f"{tool_name}:{args_hash}"

    def get(self, key: str, force_fresh: bool = False) -> Optional[Any]:
        """
        Get a cached result.  Returns None on miss, expiration, or force_fresh.

        Parameters
        ----------
        key : str
            Cache key from make_key().
        force_fresh : bool
            If True, always return None (bypass cache).  Useful when the
            user explicitly re-asks a question.
        """
        if force_fresh:
            self._misses += 1
            return None

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            age = time.time() - entry["timestamp"]
            if age > self.ttl:
                # Expired -- remove and miss
                del self._cache[key]
                self._misses += 1
                self._evictions += 1
                logger.debug("Cache EXPIRED: %s (age=%.0fs > ttl=%ds)", key, age, self.ttl)
                return None

            self._hits += 1
            logger.debug("Cache HIT: %s (age=%.0fs)", key, age)
            return entry["result"]

    def set(self, key: str, result: Any) -> None:
        """
        Store a result in the cache.

        NEVER caches:
          - None results
          - Empty strings
          - Dicts with success=False
          - Strings containing error markers
          - Empty lists
          - Results with 0 total_results

        Evicts oldest entries if max_entries is exceeded.
        """
        if not self._is_cacheable(result):
            logger.debug("Cache SKIP (not cacheable): %s", key)
            return

        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_entries and key not in self._cache:
                oldest_key = min(
                    self._cache, key=lambda k: self._cache[k]["timestamp"]
                )
                del self._cache[oldest_key]
                self._evictions += 1

            self._cache[key] = {
                "result": result,
                "timestamp": time.time(),
            }
            logger.debug("Cache SET: %s", key)

    def _is_cacheable(self, result: Any) -> bool:
        """Check if a result is worth caching (not an error/empty)."""
        if result is None:
            return False
        if isinstance(result, str):
            if not result.strip():
                return False
            if "[Tool execution error:" in result:
                return False
            if "[Error:" in result:
                return False
            if "recovery strategies failed" in result.lower():
                return False
        if isinstance(result, dict):
            if result.get("success") is False:
                return False
            if "error" in result and not result.get("data"):
                return False
            if result.get("total_results", -1) == 0:
                return False
            if result.get("file_count", -1) == 0:
                return False
        if isinstance(result, list) and len(result) == 0:
            return False
        return True

    def invalidate(self, tool_name: str) -> int:
        """
        Invalidate all cached results for a specific tool.
        Returns the number of entries removed.
        """
        with self._lock:
            prefix = f"{tool_name}:"
            keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._cache[k]
            return len(keys_to_remove)

    def invalidate_key(self, key: str) -> bool:
        """Remove a specific cache entry.  Returns True if it existed."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear the entire cache (call between conversations)."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics for observability."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                "entries": len(self._cache),
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate_pct": round(hit_rate, 1),
                "estimated_api_calls_saved": self._hits,
            }
