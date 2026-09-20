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
import os
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Mark a provider unhealthy after this many consecutive failures
FAILURE_THRESHOLD = 3

# Recheck unhealthy providers after this many seconds
RECOVERY_CHECK_INTERVAL = 120  # 2 minutes

# Fallback model mapping: if the primary model's provider is unhealthy, use
# this model instead (same-or-adjacent capability on a DIFFERENT provider).
# Every id below is one the app actually deploys today (see
# services/model_pricing.py, ui-pro/api/deps.py visible-model lists, and
# core/llm_client.py TACC_MODEL_IDS) — never invent ids here.
# Per-provider env override: QUASAR_<PROVIDER>_FALLBACK_MODEL
# (value "none"/"disabled" turns that provider's fallback off).
FALLBACK_MODELS = {
    # OpenAI primaries (gpt-5.4-mini / gpt-4.1 / gpt-4o-mini) → DeepSeek's
    # reasoning tier: the closest deployed general-capability model on a
    # different provider (it is already Quasar's QUASAR_REASONING_MODEL).
    "openai": "deepseek-v4-pro",
    # Claude (local-only deployment) → deepseek-v4-pro: the deployed
    # reasoning-tier model on a different provider.
    "anthropic": "deepseek-v4-pro",
    # Gemini (priced but not in the default menu) → deepseek-v4-pro:
    # adjacent general capability, different provider.
    "google": "deepseek-v4-pro",
    # TACC's default chat model is gpt-oss-120b (free tier) → deepseek-v4-flash:
    # the fast/cheap tier on a different provider (Quasar's QUASAR_FAST_MODEL
    # default), capability-adjacent to gpt-oss-120b.
    "tacc": "deepseek-v4-flash",
    # DeepSeek primaries (deepseek-v4-pro / deepseek-v4-flash) → gpt-oss-120b
    # via TACC: free, reasoning-capable, different provider. (Not OpenAI: the
    # platform OpenAI key has been billing-dead — 2026-07 incident — so it is
    # not a dependable failover target.)
    "deepseek": "gpt-oss-120b",
    # Local (Ollama / LM Studio) down → gpt-oss-120b via TACC: the closest
    # free substitute for a local open-weights model.
    "local": "gpt-oss-120b",
}


def health_failure_exempt(error: BaseException) -> bool:
    """True when an exception must NOT count against provider health.

    Provider ill-health means the *service* is failing (5xx, timeouts,
    connection resets, mid-stream deaths). The following are not that:

    - Cancellations: ``GeneratorExit`` (consumer closed the stream),
      ``KeyboardInterrupt``, ``asyncio.CancelledError`` — the user walked
      away; the provider may be perfectly fine.
    - Quasar's own per-user quota trip (``QuotaExceededError`` from
      services.usage_quota_service) — the USER ran out of headroom. Matched
      by class name across the MRO so core never has to import services.*.
    - Provider billing exhaustion (OpenAI's ``insufficient_quota`` rides a
      429) — a dead billing state on the key, not service ill-health, and
      core.retry already treats it as non-retryable for the same reason.
    - Local client-side errors (``ValueError``/``TypeError``/``KeyError`` —
      mirrors core.retry.NON_RETRYABLE_EXCEPTIONS): bad params, missing API
      key config, unknown provider. They say nothing about the remote
      service; provider SDK errors are not subclasses of these.
    """
    if isinstance(error, (GeneratorExit, KeyboardInterrupt)):
        return True
    try:
        import asyncio
        if isinstance(error, asyncio.CancelledError):
            return True
    except ImportError:  # pragma: no cover - asyncio is stdlib
        pass
    for klass in type(error).__mro__:
        if klass.__name__ == "QuotaExceededError":
            return True
    if getattr(error, "code", None) == "insufficient_quota":
        return True
    try:
        if "insufficient_quota" in str(error):
            return True
    except Exception:  # pragma: no cover - defensive str() guard
        pass
    if isinstance(error, (ValueError, TypeError, KeyError)):
        return True
    return False


class ProviderHealth:
    """Health state for a single provider.

    NOT self-locking: every method (and the recovery-probe bookkeeping) must
    be called under the owning HealthMonitor's lock (CX-01).
    """

    def __init__(self, name: str):
        self.name = name
        self.consecutive_failures = 0
        self.total_successes = 0
        self.total_failures = 0
        self.last_success_time = 0.0
        self.last_failure_time = 0.0
        self.last_error: Optional[str] = None
        self._marked_unhealthy_at = 0.0
        # Recovery probing is SINGLE-FLIGHT (CX-05): after the recovery
        # interval, exactly one caller gets a healthy verdict (the probe);
        # everyone else keeps seeing unhealthy (and thus keeps failing over)
        # until the probe's own call reports back.
        self._probe_in_flight = False
        self._probe_started_at = 0.0

    @property
    def is_healthy(self) -> bool:
        """PURE health view — no side effects, safe for status reporting.

        Unhealthy means the failure streak reached FAILURE_THRESHOLD and no
        success has cleared it yet; a provider mid-recovery-probe still
        reports unhealthy here (it IS still degraded until the probe wins).
        Routing decisions must use check_and_claim() instead.
        """
        return self.consecutive_failures < FAILURE_THRESHOLD

    def check_and_claim(self) -> bool:
        """Health check for ROUTING: may claim the single recovery probe.

        Call under the monitor lock. When unhealthy and the recovery interval
        has elapsed, the FIRST caller claims a probe slot and is told
        "healthy" so its real call tests the provider; concurrent callers
        keep getting "unhealthy" until the probe settles (record_success /
        record_failure). A probe whose call never reports back (crash,
        cancellation — cancellations don't record) expires after another
        RECOVERY_CHECK_INTERVAL so recovery can never wedge shut.
        """
        if self.consecutive_failures < FAILURE_THRESHOLD:
            return True

        if self._marked_unhealthy_at <= 0:
            return False
        now = time.time()
        if (now - self._marked_unhealthy_at) < RECOVERY_CHECK_INTERVAL:
            return False
        if self._probe_in_flight and (now - self._probe_started_at) < RECOVERY_CHECK_INTERVAL:
            return False  # someone else is already probing — keep failing over
        self._probe_in_flight = True
        self._probe_started_at = now
        logger.info(
            "[health] Provider '%s' recovery probe — allowing one attempt "
            "(unhealthy for %.0fs)",
            self.name, now - self._marked_unhealthy_at,
        )
        return True

    def record_success(self):
        """Record a successful API call."""
        self.consecutive_failures = 0
        self.total_successes += 1
        self.last_success_time = time.time()
        self._marked_unhealthy_at = 0.0
        self._probe_in_flight = False
        self._probe_started_at = 0.0
        self.last_error = None

    def record_failure(self, error: str = ""):
        """Record a failed API call."""
        self.consecutive_failures += 1
        self.total_failures += 1
        self.last_failure_time = time.time()
        self.last_error = error[:200] if error else None

        if self._probe_in_flight:
            # CX-04: the recovery probe FAILED — back to unhealthy with a
            # FRESH clock, so the next probe waits a full interval instead of
            # the provider drifting permanently "healthy" on a stale stamp.
            self._probe_in_flight = False
            self._probe_started_at = 0.0
            self._marked_unhealthy_at = time.time()
            logger.warning(
                "[health] Provider '%s' recovery probe FAILED (%s). "
                "Still unhealthy; next recheck in %ds.",
                self.name, self.last_error, RECOVERY_CHECK_INTERVAL,
            )
        elif self.consecutive_failures >= FAILURE_THRESHOLD and self._marked_unhealthy_at == 0:
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
        # CX-01/CX-02: the monitor is process-shared and hit from concurrent
        # turn threads — every read/mutation of _providers and of a
        # ProviderHealth's fields happens under this lock.
        self._lock = threading.Lock()

    def _get_or_create(self, provider: str) -> ProviderHealth:
        """Get or create health state for a provider (call under self._lock)."""
        if provider not in self._providers:
            self._providers[provider] = ProviderHealth(provider)
        return self._providers[provider]

    def record_success(self, provider: str):
        """Record a successful API call to a provider."""
        with self._lock:
            self._get_or_create(provider).record_success()

    def record_failure(self, provider: str, error: str = ""):
        """Record a failed API call to a provider."""
        with self._lock:
            self._get_or_create(provider).record_failure(error)

    def is_healthy(self, provider: str) -> bool:
        """Routing health check for a provider.

        NOT a pure read: when an unhealthy provider's recovery interval has
        elapsed this may CLAIM the single recovery probe (telling exactly one
        caller "healthy" so its real call tests the provider — see
        ProviderHealth.check_and_claim). Callers are always about to route a
        call, so the claim is well-placed; pure reporting uses get_status().
        """
        with self._lock:
            health = self._providers.get(provider)
            if health is None:
                return True  # Unknown provider assumed healthy
            return health.check_and_claim()

    def get_fallback_model(self, current_model: str, current_provider: str) -> Optional[str]:
        """
        Get a fallback model if the current provider is unhealthy.

        Returns None if no fallback is needed (provider is healthy)
        or if no fallback is configured. Env override values are stripped
        (CX-08) so stray whitespace can't corrupt provider detection.
        """
        if self.is_healthy(current_provider):
            return None

        env_name = f"QUASAR_{current_provider.upper()}_FALLBACK_MODEL"
        fallback = (
            os.getenv(env_name) or FALLBACK_MODELS.get(current_provider) or ""
        ).strip()
        if fallback.lower() in {"", "none", "disabled", "false"}:
            return None
        logger.info(
            "[health] Provider '%s' unhealthy — falling back from '%s' to '%s'",
            current_provider, current_model, fallback,
        )
        return fallback

    def get_status(self) -> Dict[str, Any]:
        """Return health status for all known providers.

        Snapshot taken under the lock (CX-02) so concurrent provider
        creation/mutation can never blow up or tear the iteration; the
        "healthy" field is the PURE view (no probe side effects).
        """
        with self._lock:
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


# ---------------------------------------------------------------------------
# Process-shared monitor
# ---------------------------------------------------------------------------
# Provider health is a property of THIS process's network path to a provider,
# not of any one agent/client instance — so recording (ResponsesShim, the one
# choke point every LLM call goes through) and consulting (runner turn-start
# failover, ModelRouter, the conductor status panel) all share one instance.

_GLOBAL_MONITOR: Optional[HealthMonitor] = None
_GLOBAL_MONITOR_LOCK = threading.Lock()


def get_health_monitor() -> HealthMonitor:
    """Return the process-wide shared HealthMonitor (lazily created)."""
    global _GLOBAL_MONITOR
    if _GLOBAL_MONITOR is None:
        with _GLOBAL_MONITOR_LOCK:
            if _GLOBAL_MONITOR is None:
                _GLOBAL_MONITOR = HealthMonitor()
    return _GLOBAL_MONITOR


def reset_health_monitor() -> None:
    """Drop the shared monitor (tests only — avoids cross-test pollution)."""
    global _GLOBAL_MONITOR
    with _GLOBAL_MONITOR_LOCK:
        _GLOBAL_MONITOR = None
