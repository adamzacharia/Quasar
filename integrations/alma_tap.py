"""Bounded, retrying ALMA TAP transport shared by archive query paths."""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Dict

from integrations.tap import _TimeoutHTTPSession


ALMA_TAP_MIRRORS = (
    "https://almascience.nrao.edu/tap",
    "https://almascience.eso.org/tap",
    "https://almascience.nao.ac.jp/tap",
)


def _retryable(exc: Exception) -> bool:
    """PyVO may wrap the HTTP exception, so inspect the causal chain and text."""
    seen = set()
    stack = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None) or getattr(current, "code", None)
        if isinstance(status, int) and (status == 429 or 500 <= status < 600):
            return True
        text = type(current).__name__.lower() + " " + str(current).lower()
        if re.search(r"\b(?:429|5\d\d)\b", text) and re.search(r"error|status|gateway|unavailable|timeout|too many", text):
            return True
        if any(word in text for word in (
            "timeout", "timed out", "connectionerror", "connection reset", "remote end closed",
            "poolexhausted", "connection pool", "service unavailable", "bad gateway", "gateway time-out",
        )):
            return True
        stack.extend([current.__cause__, current.__context__])
    return False


class AlmaTapService:
    """TAPService-compatible search facade with mirror rotation.

    A mirror that just failed with a transient error is put on a short
    cooldown (process-wide) so the NEXT query starts on a mirror that answered,
    instead of paying the full timeout on the same broken one again (live
    2026-09-16: the NRAO mirror 502'd on every first attempt for hours while
    ESO/NAOJ answered; each turn lost ~60-120 s before rotating).

    Budgets (2026-09-21): the old ``max(120 s)`` per-attempt floor times three
    mirrors was 363 s per query — inverted against a 150 s tool guard, so the
    guard ALWAYS fired first (live: the 12-source Perseus cross-match spent
    310 s failing inside a 150 s guard, three times in one turn). Now each
    attempt gets ``ALMA_TAP_TIMEOUT_SECONDS`` (default 40 s — a healthy mirror
    answers a cone in 2-40 s; one that has not answered by then is not going
    to) and at most ``ALMA_TAP_MAX_ATTEMPTS`` (default 2) mirrors are tried.
    Both are further clamped to the running tool's remaining budget by the
    session (services/tool_budgets.py), and mirrors whose host breaker is
    open (services/host_breaker.py) are skipped without a network call.
    """

    MIRROR_COOLDOWN_S = float(os.getenv("ALMA_TAP_MIRROR_COOLDOWN_SECONDS", "600"))
    DEFAULT_ATTEMPT_TIMEOUT_S = 40.0
    DEFAULT_MAX_ATTEMPTS = 2
    MIN_ATTEMPT_TIMEOUT_S = 5.0
    _cooldown: Dict[str, float] = {}  # mirror url -> monotonic time until which it is tried last
    _cooldown_lock = threading.Lock()

    def __init__(self, *, timeout: float | None = None, mirrors=ALMA_TAP_MIRRORS, max_attempts: int | None = None):
        if timeout is None:
            raw = os.getenv("ALMA_TAP_TIMEOUT_SECONDS", "").strip()
            try:
                timeout = float(raw) if raw else self.DEFAULT_ATTEMPT_TIMEOUT_S
            except ValueError:
                timeout = self.DEFAULT_ATTEMPT_TIMEOUT_S
        self.timeout = max(self.MIN_ATTEMPT_TIMEOUT_S, float(timeout))
        self.mirrors = tuple(mirrors)
        if max_attempts is None:
            raw = os.getenv("ALMA_TAP_MAX_ATTEMPTS", "").strip()
            try:
                max_attempts = int(raw) if raw else self.DEFAULT_MAX_ATTEMPTS
            except ValueError:
                max_attempts = self.DEFAULT_MAX_ATTEMPTS
        self.max_attempts = max(1, min(int(max_attempts), len(self.mirrors)))

    @classmethod
    def worst_case_seconds(cls) -> float:
        """Static worst case one query can cost: attempts x timeout + backoff.
        Declared to the budget hierarchy test (services/tool_budgets.py)."""
        svc = cls()
        return svc.max_attempts * svc.timeout + sum(2 ** i for i in range(max(0, svc.max_attempts - 1)))

    def _ordered_mirrors(self):
        now = time.monotonic()
        with type(self)._cooldown_lock:
            cooling = {m for m in self.mirrors if type(self)._cooldown.get(m, 0.0) > now}
        return [m for m in self.mirrors if m not in cooling] + [m for m in self.mirrors if m in cooling]

    def _mark(self, url: str, *, failed: bool) -> None:
        with type(self)._cooldown_lock:
            if failed:
                type(self)._cooldown[url] = time.monotonic() + self.MIRROR_COOLDOWN_S
            else:
                type(self)._cooldown.pop(url, None)

    def search(self, query: str, *, maxrec: int = 20000):
        import pyvo
        from services.host_breaker import HostBreaker, HostCircuitOpen, circuit_of, host_of
        from services.tool_budgets import BudgetExhausted, remaining_seconds

        failures = []
        order = self._ordered_mirrors()
        # Mirrors whose host breaker (host-level, or the /tap service circuit)
        # is open are skipped outright; when every mirror is open the query
        # fails in milliseconds with the breaker's own error (turned into a
        # structured tool result upstream).
        open_hosts = {h: rec for h, rec in HostBreaker.open_hosts(order)}
        candidates = [m for m in order if host_of(m) not in open_hosts and circuit_of(m) not in open_hosts]
        if not candidates:
            host, rec = next(iter(open_hosts.items()))
            raise HostCircuitOpen(host, rec["retry_after"], rec["reason"])
        attempts = min(self.max_attempts, len(candidates))
        for attempt in range(attempts):
            url = candidates[attempt]
            remaining = remaining_seconds()
            if remaining is not None and remaining < self.MIN_ATTEMPT_TIMEOUT_S:
                failures.append(f"{url}: not attempted — tool budget exhausted ({remaining:.0f} s left)")
                break
            session = _TimeoutHTTPSession(timeout=self.timeout)
            try:
                service = pyvo.dal.TAPService(url, session=session)
                result = service.search(query, maxrec=maxrec)
                self._mark(url, failed=False)
                return _TapResult(result, url, attempt + 1)
            except HostCircuitOpen:
                raise
            except BudgetExhausted as exc:
                failures.append(f"{url}: {exc}")
                break
            except Exception as exc:
                failures.append(f"{url}: {type(exc).__name__}: {exc}")
                if not _retryable(exc):
                    raise RuntimeError("ALMA TAP query failed: " + "; ".join(failures)) from exc
                self._mark(url, failed=True)  # transient: try this mirror last for a while
                if attempt == attempts - 1:
                    break
                print(f"[ALMA] TAP retry {attempt + 1}/{attempts - 1} after {type(exc).__name__}; rotating mirror")
                time.sleep(2 ** attempt)
            finally:
                session.close()
        raise RuntimeError("ALMA TAP query failed: " + "; ".join(failures))


class _TapResult:
    """Carry request provenance without mutating third-party result objects."""
    def __init__(self, result, url, attempts):
        self._result = result
        self.quasar_tap_url = url
        self.quasar_attempts = attempts

    def __getattr__(self, name):
        return getattr(self._result, name)
