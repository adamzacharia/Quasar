# core/token_budget.py
"""
Token Budget — Smart iteration control with diminishing returns detection.

Inspired by OpenClaude's tokenBudget.ts — replaces the hard MAX_TOOL_ROUNDS=12
constant with a dynamic budget that:
  1. Allows complex tasks to run longer (if making progress)
  2. Detects stalls (diminishing returns) and stops early
  3. Prevents infinite loops

Usage:
    budget = TokenBudget(max_tokens=100_000)

    for _round in range(50):  # generous upper bound
        ... do work ...
        budget.record_output(len(new_output))

        if not budget.should_continue():
            break
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Default maximum output tokens per agentic turn
DEFAULT_MAX_BUDGET = 100_000

# Consider the task "complete enough" at this % of budget
COMPLETION_THRESHOLD = 0.90

# If output growth drops below this many chars for 2 consecutive checks,
# we've hit diminishing returns — time to stop
DIMINISHING_THRESHOLD_CHARS = 500

# Hard upper limit on iterations (safety net even if budget isn't exhausted)
HARD_MAX_ITERATIONS = 25


class TokenBudget:
    """
    Tracks cumulative output and detects when the agent should stop.

    Replaces the old MAX_TOOL_ROUNDS = 12 constant with smarter logic
    that adapts to what the agent is actually doing.
    """

    def __init__(self, max_budget: int = DEFAULT_MAX_BUDGET):
        self.max_budget = max_budget
        self._total_output_chars = 0
        self._iteration = 0
        self._last_check_chars = 0
        self._second_last_check_chars = 0
        self._consecutive_low_growth = 0
        self._start_time = time.time()

    def record_output(self, chars: int):
        """Record that the agent produced output in this iteration."""
        self._total_output_chars += chars
        self._iteration += 1

    def should_continue(self) -> bool:
        """
        Determine if the agent should continue running.

        Returns False if:
        1. Token budget is exhausted (>90% used)
        2. Diminishing returns detected (< 500 chars growth, 2x in a row)
        3. Hard iteration limit reached (safety net)
        """
        # Hard safety limit
        if self._iteration >= HARD_MAX_ITERATIONS:
            logger.info(
                "[token_budget] Hard iteration limit reached (%d). Stopping.",
                HARD_MAX_ITERATIONS,
            )
            return False

        # Budget check (rough: 4 chars ≈ 1 token)
        estimated_tokens = self._total_output_chars // 4
        if estimated_tokens >= self.max_budget * COMPLETION_THRESHOLD:
            logger.info(
                "[token_budget] Budget %.0f%% used (%d tokens). Stopping.",
                (estimated_tokens / self.max_budget * 100),
                estimated_tokens,
            )
            return False

        # Diminishing returns detection
        growth_since_last = self._total_output_chars - self._last_check_chars
        self._second_last_check_chars = self._last_check_chars
        self._last_check_chars = self._total_output_chars

        if self._iteration > 2:  # Need at least 2 iterations to check
            if growth_since_last < DIMINISHING_THRESHOLD_CHARS:
                self._consecutive_low_growth += 1
            else:
                self._consecutive_low_growth = 0

            if self._consecutive_low_growth >= 2:
                logger.info(
                    "[token_budget] Diminishing returns detected — "
                    "<%d chars growth for %d consecutive iterations. Stopping.",
                    DIMINISHING_THRESHOLD_CHARS,
                    self._consecutive_low_growth,
                )
                return False

        return True

    def get_stats(self) -> dict:
        """Return budget tracking statistics."""
        elapsed = time.time() - self._start_time
        return {
            "iteration": self._iteration,
            "total_output_chars": self._total_output_chars,
            "estimated_tokens": self._total_output_chars // 4,
            "budget_used_pct": (self._total_output_chars // 4) / self.max_budget * 100,
            "elapsed_seconds": round(elapsed, 1),
            "consecutive_low_growth": self._consecutive_low_growth,
        }

    def reset(self):
        """Reset budget for a new turn."""
        self._total_output_chars = 0
        self._iteration = 0
        self._last_check_chars = 0
        self._second_last_check_chars = 0
        self._consecutive_low_growth = 0
        self._start_time = time.time()
