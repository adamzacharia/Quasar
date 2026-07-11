"""Basic in-memory login rate limiting (S6).

A small, dependency-free throttle that locks an identity (typically the
lower-cased username) out of login for a cooldown window after too many
*failed* attempts. Successful logins reset the counter, so a legitimate user
is never penalised.

This is process-local (per worker). It is deliberately simple: it raises the
cost of online password guessing without needing Redis/DB state. For a
multi-worker or multi-host deployment, front it with a shared store or an
edge rate limiter — see ``docs/v2/STATUS.md`` handoff notes.

Memory is bounded: the login endpoint is unauthenticated, so an attacker can
submit an unbounded stream of distinct usernames. The tracked state is
therefore capped (LRU eviction) and opportunistically swept of stale entries,
so it can never grow without limit. Very long identities are truncated so a
single entry cannot be inflated.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from typing import Callable, Dict, List, Optional

# Hard ceiling on the identity string length we key on (usernames/emails are
# realistically far shorter); stops per-entry memory amplification.
_MAX_KEY_LEN = 256


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


class LoginRateLimiter:
    """Fixed-threshold sliding-window failure limiter.

    After ``max_attempts`` failures inside ``window_seconds``, the key is
    locked for ``lockout_seconds`` (measured from the last failure). Any
    successful login should call :meth:`reset`.

    Tracked state is bounded by ``max_tracked`` (LRU) so an unauthenticated
    flood of distinct identities cannot exhaust memory.
    """

    def __init__(
        self,
        max_attempts: int = 5,
        window_seconds: float = 300.0,
        lockout_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
        max_tracked: int = 4096,
    ) -> None:
        self.max_attempts = max(1, int(max_attempts))
        self.window_seconds = float(window_seconds)
        self.lockout_seconds = float(lockout_seconds)
        self.max_tracked = max(1, int(max_tracked))
        self._clock = clock
        self._lock = threading.Lock()
        # OrderedDict → cheap LRU: most-recently-touched key at the end.
        self._failures: "OrderedDict[str, List[float]]" = OrderedDict()
        self._locked_until: Dict[str, float] = {}
        # Amortise the global sweep to at most once per window.
        self._sweep_interval = max(1.0, self.window_seconds)
        self._last_sweep = float("-inf")

    @staticmethod
    def _norm(key: str) -> str:
        return (key or "")[:_MAX_KEY_LEN]

    def _sweep(self, now: float) -> None:
        """Drop stale keys: no in-window failures AND no active lockout."""
        cutoff = now - self.window_seconds
        for key in list(self._failures.keys()):
            stamps = [t for t in self._failures[key] if t >= cutoff]
            locked = self._locked_until.get(key)
            if not stamps and (locked is None or locked <= now):
                del self._failures[key]
                self._locked_until.pop(key, None)
            else:
                self._failures[key] = stamps
        for key in list(self._locked_until.keys()):
            if self._locked_until[key] <= now and key not in self._failures:
                del self._locked_until[key]

    def _enforce_cap(self) -> None:
        """Hard bound: evict least-recently-touched keys past the ceiling."""
        while len(self._failures) > self.max_tracked:
            old_key, _ = self._failures.popitem(last=False)
            self._locked_until.pop(old_key, None)

    def seconds_until_unblocked(self, key: str) -> Optional[float]:
        """Return remaining lockout seconds, or ``None`` if not locked."""
        key = self._norm(key)
        if not key:
            return None
        now = self._clock()
        with self._lock:
            locked_until = self._locked_until.get(key)
            if locked_until is None:
                return None
            if locked_until <= now:
                # Lockout expired: clear it and the stale failure history.
                self._locked_until.pop(key, None)
                self._failures.pop(key, None)
                return None
            return locked_until - now

    def record_failure(self, key: str) -> None:
        """Record one failed attempt; may trip a lockout."""
        key = self._norm(key)
        if not key:
            return
        now = self._clock()
        with self._lock:
            if now - self._last_sweep >= self._sweep_interval:
                self._sweep(now)
                self._last_sweep = now
            stamps = [t for t in self._failures.get(key, ()) if t >= now - self.window_seconds]
            stamps.append(now)
            self._failures[key] = stamps
            self._failures.move_to_end(key)
            if len(stamps) >= self.max_attempts:
                self._locked_until[key] = now + self.lockout_seconds
            self._enforce_cap()

    def reset(self, key: str) -> None:
        """Clear failure history and any lockout for *key* (call on success)."""
        key = self._norm(key)
        if not key:
            return
        with self._lock:
            self._failures.pop(key, None)
            self._locked_until.pop(key, None)


_default_limiter: Optional[LoginRateLimiter] = None
_default_lock = threading.Lock()


def get_default_login_rate_limiter() -> LoginRateLimiter:
    """Return the shared, env-configured, process-wide limiter."""
    global _default_limiter
    if _default_limiter is None:
        with _default_lock:
            if _default_limiter is None:
                _default_limiter = LoginRateLimiter(
                    max_attempts=_int_env("QUASAR_LOGIN_MAX_ATTEMPTS", 5),
                    window_seconds=_int_env("QUASAR_LOGIN_WINDOW_SECONDS", 300),
                    lockout_seconds=_int_env("QUASAR_LOGIN_LOCKOUT_SECONDS", 300),
                    max_tracked=_int_env("QUASAR_LOGIN_MAX_TRACKED_KEYS", 4096),
                )
    return _default_limiter
