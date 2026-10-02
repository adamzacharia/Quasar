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
import os
import re
import threading
import time
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

# Stuck-probe bookkeeping shared by every detector in the process.
_PROBE_MAX_STUCK = 4
# Hard ceiling on probes in flight at once (all of them, healthy or not). Turns
# (the only callers) run on ui-pro/api/deps.py _chat_executor, sized by
# CHAT_WORKER_THREADS (default 4), so concurrent probes cannot exceed that pool;
# the ceiling sits at 4x the pool (at least 64) and never binds healthy load,
# while bounding a burst that starts before any probe has timed out (guard CX-02/CX-07).
_PROBE_MAX_INFLIGHT = max(64, 4 * int(os.getenv("CHAT_WORKER_THREADS", "4") or 4))
_probe_lock = threading.Lock()
_probe_stuck = 0          # stuck probes still counted (see _stuck_since)
_stuck_since: dict = {}   # probe id -> monotonic time it timed out
_probe_inflight = 0
_probe_timed_out_at = 0.0


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

    def _create_bounded(self, prompt: str):
        """The probe call with a wall-clock cap (QUASAR_COMPLEXITY_TIMEOUT_S,
        default 8 s). A provider that only streams keep-alives (DeepSeek
        outage 2026-10-02) never trips the HTTP read timeout, so the probe hung
        every turn that reached it, on every model. Raises TimeoutError on the
        cap; the caller then falls back to the heuristic score.

        A probe that answers in time behaves exactly as before. The call runs
        on a DAEMON thread (an executor's workers are joined at interpreter
        exit). Only probes that already TIMED OUT count as stuck: once
        _PROBE_MAX_STUCK are stuck, new probes are skipped until one of them
        returns, so a sustained outage cannot pile up threads; healthy
        concurrent probes are never limited. QUASAR_COMPLEXITY_COOLDOWN_S
        (default 0 = off) optionally skips probes for that long after a
        timeout."""
        global _probe_stuck, _probe_timed_out_at, _probe_inflight
        try:
            cap = float(os.getenv("QUASAR_COMPLEXITY_TIMEOUT_S", "8") or 8)
        except ValueError:
            cap = 8.0
        try:
            cooldown = float(os.getenv("QUASAR_COMPLEXITY_COOLDOWN_S", "0") or 0)
        except ValueError:
            cooldown = 0.0
        with _probe_lock:
            if cooldown > 0 and _probe_timed_out_at and time.monotonic() - _probe_timed_out_at < cooldown:
                raise TimeoutError("complexity probe skipped: provider timed out recently")
            # Stuck probes block new ones for at most QUASAR_COMPLEXITY_STUCK_TTL_S
            # (default 120 s): a provider that recovered is tried again even if
            # the old calls never return (guard CX-10). Threads stay bounded by
            # _PROBE_MAX_INFLIGHT.
            try:
                ttl = float(os.getenv("QUASAR_COMPLEXITY_STUCK_TTL_S", "120") or 120)
            except ValueError:
                ttl = 120.0
            now = time.monotonic()
            for pid in [k for k, t0 in _stuck_since.items() if now - t0 >= ttl]:
                _stuck_since.pop(pid, None)
                _probe_stuck -= 1
            if _probe_stuck >= _PROBE_MAX_STUCK:
                raise TimeoutError("complexity probe skipped: earlier probes are still stuck")
            if _probe_inflight >= _PROBE_MAX_INFLIGHT:
                raise TimeoutError("complexity probe skipped: too many probes in flight")
            _probe_inflight += 1
        box: dict = {"stuck": False, "id": object()}
        done = threading.Event()

        def _run():
            global _probe_stuck, _probe_inflight
            try:
                box["resp"] = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    temperature=0,
                    max_output_tokens=300,
                    text={"format": {"type": "json_object"}},
                )
            except BaseException as exc:  # surfaced to the caller below
                box["err"] = exc
            finally:
                with _probe_lock:
                    done.set()
                    _probe_inflight -= 1
                    if box["stuck"] and _stuck_since.pop(box["id"], None) is not None:
                        _probe_stuck -= 1

        try:
            threading.Thread(target=_run, name="complexity-probe", daemon=True).start()
        except BaseException:
            with _probe_lock:
                _probe_inflight -= 1  # no worker will release it
            raise
        if not done.wait(cap):
            # Past the cap the turn always falls back, even if the worker finishes
            # during this bookkeeping (guard CX-08).
            with _probe_lock:
                if not done.is_set():  # still running: count it as stuck until it returns (or the TTL)
                    box["stuck"] = True
                    _probe_stuck += 1
                    _stuck_since[box["id"]] = time.monotonic()
                _probe_timed_out_at = time.monotonic()
            raise TimeoutError(f"complexity probe exceeded {cap:.0f}s")
        if "err" in box:
            raise box["err"]
        return box["resp"]

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
            resp = self._create_bounded(prompt)
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
