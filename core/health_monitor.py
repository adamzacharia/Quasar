# core/health_monitor.py
"""
Health Monitor — Track provider health and enable automatic fallback.

Inspired by OpenClaude's SmartRouter — pings providers on startup,
tracks consecutive failures, and auto-switches to healthy alternatives.

Usage:
    monitor = HealthMonitor()
    monitor.record_success("openai")
    monitor.record_failure("openai")
    if not monitor.is_healthy("openai"):
        model = fallback_model
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Mark a provider unhealthy after this many consecutive failures
FAILURE_THRESHOLD = 3

# Recheck unhealthy providers after this many seconds
RECOVERY_CHECK_INTERVAL = 120  # 2 minutes

# Fallback model mapping: if primary model's provider is unhealthy, use this
FALLBACK_MODELS = {
    "openai": "claude-sonnet",       # OpenAI down → use Claude
    "anthropic": "gpt-4o",           # Anthropic down → use GPT-4o
    "google": "gpt-4o",              # Google down → use GPT-4o
    "local": "gpt-4o-mini",          # Local down → use cloud
}


class ProviderHealth:
    """Health state for a single provider."""

    def __init__(self, name: str):
        self.name = name
        self.consecutive_failures = 0
        self.total_successes = 0
        self.total_failures = 0
        self.last_success_time = 0.0
        self.last_failure_time = 0.0
        self.last_error: Optional[str] = None
        self._marked_unhealthy_at = 0.0

    @property
    def is_healthy(self) -> bool:
        """Check if the provider is healthy."""
        if self.consecutive_failures < FAILURE_THRESHOLD:
            return True

        # If marked unhealthy, check if recovery interval has passed
        if self._marked_unhealthy_at > 0:
            elapsed = time.time() - self._marked_unhealthy_at
            if elapsed >= RECOVERY_CHECK_INTERVAL:
                logger.info(
                    "[health] Provider '%s' recovery check — allowing one attempt "
                    "(unhealthy for %.0fs)",
                    self.name, elapsed,
                )
                return True  # Allow one attempt to see if it recovered

        return False

    def record_success(self):
        """Record a successful API call."""
        self.consecutive_failures = 0
        self.total_successes += 1
        self.last_success_time = time.time()
        self._marked_unhealthy_at = 0.0
        self.last_error = None

    def record_failure(self, error: str = ""):
        """Record a failed API call."""
        self.consecutive_failures += 1
        self.total_failures += 1
        self.last_failure_time = time.time()
        self.last_error = error[:200] if error else None

        if self.consecutive_failures >= FAILURE_THRESHOLD and self._marked_unhealthy_at == 0:
            self._marked_unhealthy_at = time.time()
            logger.warning(
                "[health] Provider '%s' marked UNHEALTHY after %d consecutive failures. "
                "Last error: %s. Will recheck in %ds.",
                self.name, self.consecutive_failures,
                self.last_error, RECOVERY_CHECK_INTERVAL,
            )


class HealthMonitor:
    """
    Tracks health across all LLM providers.

    Records successes and failures from API calls, marks providers
    unhealthy after consecutive failures, and provides fallback routing.
    """

    def __init__(self):
        self._providers: Dict[str, ProviderHealth] = {}

    def _get_or_create(self, provider: str) -> ProviderHealth:
        """Get or create health state for a provider."""
        if provider not in self._providers:
            self._providers[provider] = ProviderHealth(provider)
        return self._providers[provider]

    def record_success(self, provider: str):
        """Record a successful API call to a provider."""
        self._get_or_create(provider).record_success()

    def record_failure(self, provider: str, error: str = ""):
        """Record a failed API call to a provider."""
        self._get_or_create(provider).record_failure(error)

    def is_healthy(self, provider: str) -> bool:
        """Check if a provider is healthy."""
        health = self._providers.get(provider)
        if health is None:
            return True  # Unknown provider assumed healthy
        return health.is_healthy

    def get_fallback_model(self, current_model: str, current_provider: str) -> Optional[str]:
        """
        Get a fallback model if the current provider is unhealthy.

        Returns None if no fallback is needed (provider is healthy)
        or if no fallback is configured.
        """
        if self.is_healthy(current_provider):
            return None

        fallback = FALLBACK_MODELS.get(current_provider)
        if fallback:
            logger.info(
                "[health] Provider '%s' unhealthy — falling back from '%s' to '%s'",
                current_provider, current_model, fallback,
            )
            return fallback

        return None

    def get_status(self) -> Dict[str, Any]:
        """Return health status for all known providers."""
        return {
            name: {
                "healthy": health.is_healthy,
                "consecutive_failures": health.consecutive_failures,
                "total_successes": health.total_successes,
                "total_failures": health.total_failures,
                "last_error": health.last_error,
            }
            for name, health in self._providers.items()
        }
