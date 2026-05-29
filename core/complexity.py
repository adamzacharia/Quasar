# core/complexity.py
"""
Query Complexity Detection

Two-stage complexity detector that determines whether a user query requires
multi-step (DAG) orchestration via the Conductor, or can be handled directly
by the standard tool-calling loop.

  Stage 1: Fast heuristic keyword scan (free — no LLM call).
  Stage 2: LLM-based assessment (only triggered when heuristic is ambiguous).

Used by: agent.py → stream_response_api() to gate Conductor activation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List

from openai import OpenAI


# ---------------------------------------------------------------------------
# Multi-hop reasoning indicators
# ---------------------------------------------------------------------------

_MULTI_HOP_INDICATORS = [
    "coverage across",
    "overlap",
    "redshift",
    "observed frequency",
    "rest frequency",
    "line coverage",
    "compare",
    "correlate",
    "combine",
    "both",
    "relationship between",
    "across multiple",
    "step by step",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ComplexityResult:
    """Output of the complexity detector."""
    is_complex: bool
    score: float  # 0.0 (trivial) – 1.0 (very complex)
    reasoning: str = ""
    suggested_subtasks: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Complexity Detector
# ---------------------------------------------------------------------------

class ComplexityDetector:
    """
    Determines whether a query requires DAG orchestration.

    Uses a two-stage approach:
      1. Fast heuristic check (keyword scan) — cheap, no LLM call.
      2. LLM-based assessment (only triggered when heuristic is ambiguous).
    """

    THRESHOLD = 0.7  # queries scoring above this are treated as complex

    def __init__(self, client: OpenAI, model: str = "deepseek-v4-flash"):
        self.client = client
        self.model = model

    # --- public API ---------------------------------------------------------

    def assess(self, query: str) -> ComplexityResult:
        """Return a ComplexityResult for *query*."""
        # Stage 1: fast heuristic
        heuristic_score = self._heuristic_score(query)
        if heuristic_score >= 0.8:
            return ComplexityResult(
                is_complex=True,
                score=heuristic_score,
                reasoning="Strong multi-hop indicators detected",
            )
        if heuristic_score <= 0.2:
            return ComplexityResult(
                is_complex=False,
                score=heuristic_score,
                reasoning="Simple single-step query",
            )

        # Stage 2: LLM probe
        return self._llm_assess(query, heuristic_score)

    # --- internals ----------------------------------------------------------

    @staticmethod
    def _heuristic_score(query: str) -> float:
        q = query.lower()

        # Fast-track obvious single-step queries → always non-complex.
        _simple_patterns = [
            r'^(?:look\s*up|search|find|show|get|list|display|give\s+me|query)\b',
            r'^(?:what|where|who|when|how)\s+(?:is|are|was|were|do|does)\b',
            r'^(?:tell\s+me\s+about|explain|describe|define)\b',
            # Archive-query patterns: "any observations of X", "ALMA data for X"
            r'(?:any|are\s+there)\s+(?:observations|data)\b',
        ]
        _multi_hop_hits = sum(1 for kw in _MULTI_HOP_INDICATORS if kw in q)
        for pat in _simple_patterns:
            if re.search(pat, q):
                if _multi_hop_hits == 0:
                    return 0.0
                break  # has multi-hop indicators — fall through to scoring

        # Explicit single-archive queries
        _has_target_kw = bool(re.search(
            r'\b(?:observation|observations|data|archive)\b', q
        ))
        _has_band_or_target = bool(re.search(
            r'\b(?:band\s*\d|m\d{1,3}|ngc|ic\s*\d|alma|vla|jwst)\b', q
        ))
        if _has_target_kw and _has_band_or_target:
            if _multi_hop_hits == 0:
                return 0.0

        # ── Multi-target / multi-band archive queries ──────────────────
        _is_archive_lookup = bool(re.search(
            r'\b(?:find|search|show|get|list|look\s*up|query|give\s+me|any)\b'
            r'.*\b(?:observation|observations|data|archive|result)\b',
            q,
        ))
        if _is_archive_lookup and _multi_hop_hits == 0:
            return 0.0

        # Starts with telescope/archive name
        _starts_with_telescope = bool(re.search(
            r'^(?:alma|vla|vlba|gbt|jwst|hst|hubble|gemini|jcmt|cfht|cadc|chandra|xmm)\b',
            q,
        ))
        if _starts_with_telescope and _multi_hop_hits == 0:
            return 0.0

        # Normalize: 3+ hits → 1.0
        return min(_multi_hop_hits / 3.0, 1.0)

    def _llm_assess(self, query: str, heuristic_score: float) -> ComplexityResult:
        prompt = (
            "You are a complexity analyst for an astronomy research assistant.\n"
            "Given the user's query, decide whether it requires MULTIPLE sequential\n"
            "reasoning steps (e.g. look up a value, then compute something, then search).\n\n"
            "Respond with JSON only:\n"
            '{"is_complex": bool, "score": float 0-1, "reasoning": "...", '
            '"subtasks": ["task1", "task2", ...]}\n\n'
            f"User query: \"{query}\"\n"
        )
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=prompt,
                temperature=0,
                max_output_tokens=300,
                text={"format": {"type": "json_object"}},
            )
            data = json.loads(resp.output_text)
            score = float(data.get("score", heuristic_score))
            return ComplexityResult(
                is_complex=score >= self.THRESHOLD,
                score=score,
                reasoning=data.get("reasoning", ""),
                suggested_subtasks=data.get("subtasks", []),
            )
        except Exception:
            # Fallback to heuristic
            return ComplexityResult(
                is_complex=heuristic_score >= self.THRESHOLD,
                score=heuristic_score,
                reasoning="LLM assessment failed; using heuristic",
            )
