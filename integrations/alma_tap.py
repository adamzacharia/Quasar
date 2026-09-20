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
    """

    MIRROR_COOLDOWN_S = float(os.getenv("ALMA_TAP_MIRROR_COOLDOWN_SECONDS", "600"))
    _cooldown: Dict[str, float] = {}  # mirror url -> monotonic time until which it is tried last
    _cooldown_lock = threading.Lock()

    def __init__(self, *, timeout: float = 120.0, mirrors=ALMA_TAP_MIRRORS):
        self.timeout = max(120.0, float(timeout))
        self.mirrors = tuple(mirrors)

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

        failures = []
        order = self._ordered_mirrors()
        for attempt in range(3):
            url = order[attempt % len(order)]
            session = _TimeoutHTTPSession(timeout=self.timeout)
            try:
                service = pyvo.dal.TAPService(url, session=session)
                result = service.search(query, maxrec=maxrec)
                self._mark(url, failed=False)
                return _TapResult(result, url, attempt + 1)
            except Exception as exc:
                failures.append(f"{url}: {type(exc).__name__}: {exc}")
                if not _retryable(exc):
                    raise RuntimeError("ALMA TAP query failed: " + "; ".join(failures)) from exc
                self._mark(url, failed=True)  # transient: try this mirror last for a while
                if attempt == 2:
                    raise RuntimeError("ALMA TAP query failed: " + "; ".join(failures)) from exc
                print(f"[ALMA] TAP retry {attempt + 1}/2 after {type(exc).__name__}; rotating mirror")
                time.sleep(2 ** attempt)
            finally:
                session.close()


class _TapResult:
    """Carry request provenance without mutating third-party result objects."""
    def __init__(self, result, url, attempts):
        self._result = result
        self.quasar_tap_url = url
        self.quasar_attempts = attempts

    def __getattr__(self, name):
        return getattr(self._result, name)
