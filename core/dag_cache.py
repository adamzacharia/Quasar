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

import copy
import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

# Target regex shared by _normalize_query and _extract_targets so the tokens we
# generalise into {TARGET} are exactly the ones we can re-substitute later. (C9)
_TARGET_RE = re.compile(
    r'\b(?:m\d{1,3}|ngc\s*\d{1,5}|ic\s*\d{1,5}|sz\s*\d{1,3}|'
    r'sgr\s*[ab]\*?|3c\s*\d{1,3}|'
    r'[a-z]{2,}\s*j?\d{4}[+-]\d{2,6})\b',
    re.IGNORECASE,
)

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
        q = _TARGET_RE.sub('{TARGET}', q)

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

    @staticmethod
    def _extract_targets(query: str) -> List[str]:
        """Return the astronomical target tokens in a query, in order.

        Uses the same regex that _normalize_query generalises into {TARGET},
        so a cached plan can be re-substituted for a *new* target instead of
        replaying the stale one baked into its subtasks. Tokens are whitespace-
        collapsed but keep their original casing. (C9)
        """
        out: List[str] = []
        seen = set()
        for m in _TARGET_RE.finditer(query or ""):
            tok = re.sub(r"\s+", " ", m.group(0)).strip()
            key = tok.lower()
            if key not in seen:
                seen.add(key)
                out.append(tok)
        return out

    @staticmethod
    def _target_pattern(tok: str) -> "re.Pattern":
        """Word-boundary-ish matcher for a target token.

        Uses lookarounds that treat word chars AND '*' (for "Sgr A*") as
        boundary characters, so a short target ("M8") never matches inside a
        longer one ("M87"). (C9)
        """
        return re.compile(r'(?<![\w*])' + re.escape(tok) + r'(?![\w*])', re.IGNORECASE)

    def _reconcile_targets(
        self, entry: Dict[str, Any], query: str, is_exact: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Return a copy of ``entry`` whose subtasks target the *new* query, or
        ``None`` if it cannot be safely re-targeted.

        The cached subtasks bake in the original target name(s); reusing them
        verbatim for a different object silently returns wrong-target results.
        Guards (C9):
          * Both target sets empty is only safe on an EXACT fingerprint match.
            On a *fuzzy* match the queries differ by something the catalog-
            designation regex could not parse (e.g. a proper name — Betelgeuse
            vs Fomalhaut), so we decompose fresh rather than reuse verbatim.
          * Substitution is word-boundary + two-pass placeholder based, so a
            short target (M8) never rewrites a longer one (M87), regardless of
            ordering.
          * Safety net: if any old target token still lingers, or a placeholder
            was left unreplaced, we bail to a fresh decomposition.
        """
        new_targets = self._extract_targets(query)
        old_targets = entry.get("targets")
        if old_targets is None:  # older cache entries didn't store this
            old_targets = self._extract_targets(entry.get("query", ""))

        old_lower = {t.lower() for t in old_targets}
        new_lower = {t.lower() for t in new_targets}

        if old_lower == new_lower:
            # Identical recognised targets. If BOTH are empty we cannot see what
            # actually differs between the two queries — only an exact
            # fingerprint match guarantees they are the same query. (C9)
            if not old_lower and not is_exact:
                return None
            return entry

        # Differing targets are only safe with a clean 1:1 positional mapping.
        if not new_targets or len(old_targets) != len(new_targets):
            return None

        reconciled = copy.deepcopy(entry)
        subtasks = reconciled.get("subtasks", [])

        # Pass 1: each old target → a unique sentinel (word-boundary matched).
        for i, old_t in enumerate(old_targets):
            pat = self._target_pattern(old_t)
            sentinel = f"\x00__TGT{i}__\x00"
            for st in subtasks:
                for key, val in list(st.items()):
                    if isinstance(val, str):
                        st[key] = pat.sub(sentinel, val)
        # Pass 2: sentinel → the corresponding new target.
        for i, new_t in enumerate(new_targets):
            sentinel = f"\x00__TGT{i}__\x00"
            for st in subtasks:
                for key, val in list(st.items()):
                    if isinstance(val, str):
                        st[key] = val.replace(sentinel, new_t)

        # Safety net: unreplaced sentinel or a lingering old target → don't trust.
        for st in subtasks:
            for val in st.values():
                if not isinstance(val, str):
                    continue
                if "\x00__TGT" in val:
                    return None
                for old_t in old_targets:
                    if old_t.lower() in new_lower:
                        continue
                    if self._target_pattern(old_t).search(val):
                        return None
        return reconciled

    def find_similar(self, query: str, similarity_threshold: float = 0.8) -> Optional[Dict[str, Any]]:
        """
        Find a cached DAG decomposition for a similar query.

        Uses normalized fingerprint matching first (exact pattern match),
        then falls back to word-overlap similarity.

        Returns the cached entry dict or None.
        """
        normalized = self._normalize_query(query)
        fingerprint = self._fingerprint(normalized)

        candidate: Optional[Dict[str, Any]] = None
        match_kind = ""

        # 1. Exact pattern match (fastest)
        for entry in reversed(self._cache):
            if entry.get("fingerprint") == fingerprint:
                candidate = entry
                match_kind = "exact pattern"
                break

        # 2. Word-overlap similarity (fallback)
        if candidate is None:
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
                candidate = best_match
                match_kind = f"similarity {best_score * 100:.0f}%"

        if candidate is None:
            return None

        # Re-target the cached plan for THIS query's object(s). Returns None if
        # the plan cannot be safely re-substituted, so we never replay a
        # wrong-target decomposition. (C9)
        reconciled = self._reconcile_targets(
            candidate, query, is_exact=match_kind.startswith("exact")
        )
        if reconciled is None:
            logger.info(
                "DAGCache: %s match for '%s' rejected — target mismatch, decomposing fresh",
                match_kind, query[:60],
            )
            return None

        logger.info("DAGCache: %s match for '%s'", match_kind, query[:60])
        candidate["hits"] = candidate.get("hits", 0) + 1
        return reconciled

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
                entry["targets"] = self._extract_targets(query)
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
            "targets": self._extract_targets(query),
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
