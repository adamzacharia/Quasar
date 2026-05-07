# core/dag_cache.py
"""
DAG Cache — Cross-session DAG decomposition learning.

Stores successful DAG decompositions keyed by query pattern. When a similar
query comes in, the cached decomposition is offered as a starting point
instead of calling GPT-5.4 for planning — saving ~$0.01 per decomposition
and ~2s latency.

Pattern matching uses normalized query fingerprints (lowercased, targets
replaced with placeholders) to match structurally similar queries.

Storage: JSON file on disk for simplicity (no database dependency).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DAGCache:
    """
    Cache and reuse successful DAG decompositions across sessions.

    Usage
    -----
    cache = DAGCache("data/dag_cache.json")

    # Before decomposing:
    cached = cache.find_similar("Compare ALMA data for NGC 1068 with papers")
    if cached:
        subtasks = cached["subtasks"]  # reuse the template!
    else:
        subtasks = decompose_with_llm(query)
        cache.store(query, subtasks, reasoning)

    # The cache automatically generalizes query patterns so
    # "Compare ALMA data for M87 with papers" matches the template
    # from "Compare ALMA data for NGC 1068 with papers".
    """

    def __init__(self, cache_path: str = "data/dag_cache.json", max_entries: int = 200):
        self.cache_path = cache_path
        self.max_entries = max_entries
        self._cache: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load the cache from disk."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    self._cache = json.load(f)
                logger.info("DAGCache: loaded %d entries from %s", len(self._cache), self.cache_path)
            except Exception as e:
                logger.warning("DAGCache: failed to load %s: %s", self.cache_path, e)
                self._cache = []
        else:
            self._cache = []

    def _save(self) -> None:
        """Persist the cache to disk."""
        try:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self._cache[-self.max_entries:], f, indent=2)
        except Exception as e:
            logger.warning("DAGCache: failed to save: %s", e)

    @staticmethod
    def _normalize_query(query: str) -> str:
        """
        Normalize a query into a pattern fingerprint.

        Replaces specific target names, bands, and numbers with placeholders
        so structurally similar queries match.

        Examples:
          "Compare ALMA Band 6 observations of M87 with papers on its jet"
          → "compare alma band {BAND} observations of {TARGET} with papers on its jet"

          "Search ALMA for NGC 1068 and find recent papers"
          → "search alma for {TARGET} and find recent papers"
        """
        q = query.lower().strip()

        # Replace common astronomical target names with {TARGET}
        q = re.sub(
            r'\b(?:m\d{1,3}|ngc\s*\d{1,5}|ic\s*\d{1,5}|sz\s*\d{1,3}|'
            r'sgr\s*[ab]\*?|3c\s*\d{1,3}|'
            r'[a-z]{2,}\s*j?\d{4}[+-]\d{2,6})\b',
            '{TARGET}', q
        )

        # Replace band numbers
        q = re.sub(r'\bband\s*\d+\b', 'band {BAND}', q)

        # Replace standalone large numbers (frequencies, etc.)
        q = re.sub(r'\b\d{2,}\s*(?:ghz|mhz|khz)\b', '{FREQ}', q)

        # Collapse whitespace
        q = re.sub(r'\s+', ' ', q).strip()

        return q

    @staticmethod
    def _fingerprint(normalized: str) -> str:
        """Create a short hash from a normalized query."""
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def find_similar(self, query: str, similarity_threshold: float = 0.8) -> Optional[Dict[str, Any]]:
        """
        Find a cached DAG decomposition for a similar query.

        Uses normalized fingerprint matching first (exact pattern match),
        then falls back to word-overlap similarity.

        Returns the cached entry dict or None.
        """
        normalized = self._normalize_query(query)
        fingerprint = self._fingerprint(normalized)

        # 1. Exact pattern match (fastest)
        for entry in reversed(self._cache):
            if entry.get("fingerprint") == fingerprint:
                logger.info("DAGCache: exact pattern match for '%s'", query[:60])
                entry["hits"] = entry.get("hits", 0) + 1
                return entry

        # 2. Word-overlap similarity (fallback)
        query_words = set(normalized.split())
        if len(query_words) < 3:
            return None  # Too short to match meaningfully

        best_match = None
        best_score = 0.0

        for entry in reversed(self._cache):
            cached_words = set(entry.get("normalized", "").split())
            if not cached_words:
                continue
            overlap = len(query_words & cached_words)
            union = len(query_words | cached_words)
            score = overlap / union if union > 0 else 0

            if score > best_score:
                best_score = score
                best_match = entry

        if best_score >= similarity_threshold and best_match:
            logger.info(
                "DAGCache: similarity match (%.0f%%) for '%s'",
                best_score * 100, query[:60],
            )
            best_match["hits"] = best_match.get("hits", 0) + 1
            return best_match

        return None

    def store(
        self,
        query: str,
        subtasks: List[dict],
        reasoning: str = "",
        execution_summary: Optional[Dict] = None,
    ) -> None:
        """
        Store a successful DAG decomposition for future reuse.

        Only stores if the DAG had ≥2 subtasks and completed successfully.
        """
        if len(subtasks) < 2:
            return  # Not worth caching trivial decompositions

        # Check if execution was successful (if summary provided)
        if execution_summary:
            failed = execution_summary.get("failed", 0)
            total = execution_summary.get("total_tasks", 0)
            if failed > 0 and failed >= total / 2:
                logger.debug("DAGCache: skipping failed decomposition")
                return  # Don't cache mostly-failed plans

        normalized = self._normalize_query(query)
        fingerprint = self._fingerprint(normalized)

        # Don't duplicate
        for entry in self._cache:
            if entry.get("fingerprint") == fingerprint:
                # Update with latest version
                entry["subtasks"] = subtasks
                entry["reasoning"] = reasoning
                entry["updated_at"] = time.time()
                entry["hits"] = entry.get("hits", 0)
                self._save()
                return

        entry = {
            "query": query,
            "normalized": normalized,
            "fingerprint": fingerprint,
            "subtasks": subtasks,
            "reasoning": reasoning,
            "created_at": time.time(),
            "updated_at": time.time(),
            "hits": 0,
        }
        self._cache.append(entry)

        # Evict oldest if over limit
        if len(self._cache) > self.max_entries:
            self._cache = self._cache[-self.max_entries:]

        self._save()
        logger.info("DAGCache: stored decomposition for '%s' (%d subtasks)", query[:60], len(subtasks))

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total_hits = sum(e.get("hits", 0) for e in self._cache)
        return {
            "entries": len(self._cache),
            "total_hits": total_hits,
            "top_patterns": sorted(
                [
                    {"query": e["query"][:60], "hits": e.get("hits", 0)}
                    for e in self._cache
                ],
                key=lambda x: x["hits"],
                reverse=True,
            )[:5],
        }

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._save()
