"""Process-wide safety net: every ``requests`` call made INSIDE a guarded tool
is clamped to the tool's remaining budget and routed through the host breaker.

Why a global hook, on top of the explicit wiring
------------------------------------------------
The 2026-09-21 audit (tmp/latency-repro-2026-09-21/) found ~60 registered
network tools, most with inner timeouts ABOVE their 150 s guard and a dozen
with no bound at all: pyvo services built without a session, astroquery
SkyView (no timeout config exists), SPARCL's 90-minute read timeout,
Splatalogue's 1200 s retry ladder, alminer's keysearch/catalog/download,
lightkurve. Every one of those transports ends in ``requests.Session.request``
— so one hook there enforces the budget hierarchy for all of them:

* **Budget clamp** — when the calling thread carries a tool deadline
  (services/tool_budgets.py; set by ``QuasarAgent._execute_tool_guarded`` on
  its worker), the request timeout becomes ``min(own timeout, remaining)``,
  and a call with too little budget left raises ``BudgetExhausted`` at once.
  A missing timeout (``None`` — pyvo, SkyView) becomes ``remaining``. Outside
  a tool (LLM calls, DB writes, startup) nothing is changed.
* **Host breaker** — transport failures (refused / reset / DNS / TLS /
  timeout) and 502/503/504 open the host in services/host_breaker.py; an
  open host is refused in microseconds. Only inside a tool, and never for the
  platform's own infrastructure hosts (LLM providers, database, telemetry),
  which have their own retry policies and must not be tripped by a tool.

The explicitly wired paths (integrations/tap.py, alma_tap, mast_client,
simbad_resolver, hips_images, datalab clients, cds_xmatch, vo_registry) set a
thread-local marker so the hook does not clamp or record twice for them.

Install once per process with :func:`install`; idempotent. Disable with
``QUASAR_HTTP_BUDGET_HOOK=0``.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Optional, Tuple

__all__ = ["install", "installed", "uninstall", "reset_for_tests", "suppressed", "EXCLUDED_HOST_SUFFIXES"]

_tls = threading.local()
_lock = threading.Lock()
_original_request = None


def _pristine_request():
    """The genuine ``requests.Session.request`` captured on first use, so
    :func:`uninstall`/:func:`reset_for_tests` can always get back to it even
    if a test monkeypatch was in flight when :func:`install` ran."""
    global _PRISTINE
    if _PRISTINE is None:
        import requests

        _PRISTINE = requests.Session.request
    return _PRISTINE


_PRISTINE = None

# Hosts the hook must never clamp or trip: the platform's own dependencies.
# Suffix match on the hostname.
EXCLUDED_HOST_SUFFIXES: Tuple[str, ...] = (
    "localhost", "127.0.0.1", "::1",
    "openai.com", "anthropic.com", "googleapis.com", "deepseek.com", "tacc.utexas.edu",
    "turso.io", "libsql", "langfuse.com", "rollbar.com", "sentry.io",
    "huggingface.co",  # lsdb/HATS parquet reads are fsspec-driven; bounded elsewhere
)


def _excluded(host: str) -> bool:
    if not host:
        return True
    extra = os.getenv("QUASAR_HTTP_BUDGET_HOOK_EXCLUDE", "")
    suffixes = EXCLUDED_HOST_SUFFIXES + tuple(s.strip().lower() for s in extra.split(",") if s.strip())
    # DNS-label bounded: "api.openai.com" matches "openai.com", "notopenai.com"
    # does not (guard CX-07).
    return any(host == s or host.endswith("." + s) for s in suffixes)


class suppressed:
    """Context manager: mark this thread as already inside an explicitly
    budgeted/breaker-guarded call so the hook passes the request through."""

    def __enter__(self):
        self._prior = getattr(_tls, "depth", 0)
        _tls.depth = self._prior + 1
        return self

    def __exit__(self, *exc):
        _tls.depth = self._prior
        return False


def _is_suppressed() -> bool:
    return getattr(_tls, "depth", 0) > 0


def _clamp_timeout(timeout: Any, label: str):
    from services.tool_budgets import bounded_timeout, current_deadline

    if current_deadline() is None:
        return timeout
    if timeout is None:
        # Unbounded client (pyvo without a session, SkyView): the tool
        # deadline is the only bound there is.
        return bounded_timeout(1e9, minimum=1.0, label=label)
    if isinstance(timeout, (tuple, list)):
        # (connect, read): a None element means "unbounded" in requests, so it
        # too becomes the remaining budget (guard CX-05).
        return tuple(bounded_timeout(float(t) if t is not None else 1e9, minimum=1.0, label=label) for t in timeout)
    try:
        return bounded_timeout(float(timeout), minimum=1.0, label=label)
    except (TypeError, ValueError):
        return timeout


def _hooked_request(self, method, url, *args, **kwargs):
    from services.host_breaker import HostBreaker, host_of
    from services.tool_budgets import TurnCancelled, current_deadline

    # If a test teardown left our wrapper installed after uninstall() cleared
    # the captured original, fall back to the genuine method rather than
    # calling None.
    _orig = _original_request or _pristine_request()
    deadline = current_deadline()
    if deadline is None:
        return _orig(self, method, url, *args, **kwargs)
    url_text = str(url)
    host = host_of(url_text)
    label = f"{str(method).upper()} {host}"
    # A cancelled tool call / turn (client gone, Stop pressed, SSE ceiling,
    # guard abandoned this worker) must not start ANY new network request --
    # checked BEFORE the suppressed/excluded pass-throughs (guard CX-04): an
    # integration that clamps its own timeout still runs on a cancellable
    # worker.
    if deadline.cancelled():
        raise TurnCancelled(label, deadline.why_cancelled())
    if _is_suppressed() or _excluded(host):
        return _orig(self, method, url, *args, **kwargs)
    import requests

    # Refuse a dead host (or dead service path) before spending any budget
    # logic on it: the breaker's answer is the more specific one and costs
    # microseconds. The URL, not the bare host, so SOFT failures stay scoped
    # to the service path (datalab.noirlab.edu/sia vs /query).
    HostBreaker.check(url_text)
    kwargs["timeout"] = _clamp_timeout(kwargs.get("timeout"), label)
    try:
        response = _orig(self, method, url, *args, **kwargs)
    except requests.RequestException as exc:
        HostBreaker.record_failure(url_text, exc)
        raise
    status = int(getattr(response, "status_code", 0) or 0)
    if status in (502, 503, 504):
        HostBreaker.record_failure(url_text, status=status)
    else:
        HostBreaker.record_success(url_text)
    return response


def installed() -> bool:
    return _original_request is not None


def install() -> bool:
    """Patch ``requests.Session.request``. Returns True when active."""
    global _original_request
    if os.getenv("QUASAR_HTTP_BUDGET_HOOK", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    with _lock:
        import requests

        _pristine_request()
        if _original_request is not None:
            if requests.Session.request is not _hooked_request:
                # Someone (a test monkeypatch teardown) put another function
                # back over ours: re-arm rather than silently stay inert.
                requests.Session.request = _hooked_request  # type: ignore[assignment]
            return True
        _original_request = requests.Session.request
        requests.Session.request = _hooked_request  # type: ignore[assignment]
        return True


def uninstall() -> None:
    """Restore what :func:`install` wrapped — but only if our hook is still
    the installed attribute. If a monkeypatch teardown already replaced it,
    touching ``Session.request`` here would re-install whatever we captured
    (possibly that test's fake) and poison every later request."""
    global _original_request
    with _lock:
        if _original_request is None:
            return
        import requests

        if requests.Session.request is _hooked_request:
            requests.Session.request = _original_request  # type: ignore[assignment]
        _original_request = None


def reset_for_tests() -> None:
    """Force the genuine ``requests.Session.request`` back and forget any
    captured original, whatever earlier tests left behind."""
    global _original_request
    with _lock:
        import requests

        requests.Session.request = _pristine_request()  # type: ignore[assignment]
        _original_request = None
