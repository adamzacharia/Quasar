"""Process-wide circuit breaker for archive / imaging HTTP calls, keyed on the
remote HOST (hard failures) and on the HOST + SERVICE PATH (soft failures).

Why host, not tool: ``datalab_color_image`` and ``hips_cutout`` both end up at
``alasky.cds.unistra.fr``; the ALMA search family shares three TAP mirrors;
every Data Lab tool shares ``datalab.noirlab.edu``. When one of those hosts is
down (live 2026-09-20: CDS answering only with SSLError / ReadTimeout), each
tool used to pay its full inner timeout again, and the model's habit of
retrying with a *sibling* tool dodged the per-tool timeout breaker entirely.

Why service path for soft failures (2026-09-22, UI benchmark L07 -> L08/L09/L10):
three SIA cutout 502s inside ONE ``datalab_density_vetting`` call were counted
as three consecutive soft failures and opened the whole ``datalab.noirlab.edu``
host for 120 s while its TAP/SQL service was perfectly healthy -- the next
three questions never reached Data Lab and scored zero. Now:

* A **hard** failure (connection refused / reset, DNS, TLS, connect timeout)
  says the HOST is unreachable and opens the host circuit at once
  (``HOST_BREAKER_TRIP_FAILURES``, default 1).
* A **soft** failure (read timeout, 502/503/504 -- what a healthy front proxy
  returns for one heavy request) is scoped to the SERVICE PATH
  (``datalab.noirlab.edu/sia`` is independent of ``datalab.noirlab.edu/query``
  and ``/tap``) and is counted **at most once per tool call**: the current
  tool deadline (services/tool_budgets.py) identifies the call, so a burst of
  sub-requests inside one call is one failure. The circuit opens only when
  ``HOST_BREAKER_SOFT_TRIPS`` (default 3) failures from DISTINCT tool calls
  have accumulated over a span of at least ``HOST_BREAKER_SOFT_SPREAD_SECONDS``
  (default 20 s) inside ``HOST_BREAKER_SOFT_WINDOW_SECONDS`` (default 600 s).
  A success on the same circuit clears the streak.
* An open circuit makes :meth:`HostBreaker.check` raise
  :class:`HostCircuitOpen` immediately -- the second call in a turn fails in
  milliseconds instead of paying another timeout. :func:`structured_error`
  turns that into the tool-result dict the model can answer around; it carries
  ``retry_after_s`` so a SHORT cooldown (<= ``HOST_BREAKER_AUTO_WAIT_SECONDS``,
  default 30) can be waited out inside the turn (the tool guard does that
  automatically; the text tells the model it may retry once afterwards).
* The breaker closes by itself after ``HOST_BREAKER_COOLDOWN_SECONDS``
  (default 120 s): the next call is a live probe; if it fails the same way
  the breaker re-opens with the cooldown doubled, up to a cap.

Everything here is a leaf: no imports from core/ or other services at module
load (the tool-call token is read lazily from services/tool_budgets.py, which
is itself stdlib-only), so it can be used from integrations/ without cycles.
"""
from __future__ import annotations

import os
import re
import socket
import ssl
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit

__all__ = [
    "HostBreaker",
    "HostCircuitOpen",
    "circuit_of",
    "classify_infrastructure_error",
    "guarded_request",
    "host_of",
    "service_path_of",
    "severity_of",
    "structured_error",
]

# Monotonic clock indirection so tests can advance time without sleeping.
_now = time.monotonic


def host_of(url_or_host: str) -> str:
    """Normalise a URL or bare host to a lower-case hostname (no port/path)."""
    text = str(url_or_host or "").strip()
    if not text:
        return ""
    if "://" in text:
        host = urlsplit(text).hostname or ""
    else:
        host = text.split("/", 1)[0].split("?", 1)[0].rsplit("@", 1)[-1]
        if not host.startswith("["):
            host = host.split(":", 1)[0]
    return host.lower().rstrip(".")


def service_path_of(url_or_host: str) -> str:
    """The first path segment of a URL (``/tap/sync`` -> ``tap``), lower-case;
    empty for a bare host. This is the granularity at which SOFT failures are
    scoped: on ``datalab.noirlab.edu`` it separates ``/query`` (SQL), ``/tap``
    and ``/sia`` (cutouts)."""
    text = str(url_or_host or "").strip()
    if not text:
        return ""
    if "://" in text:
        path = urlsplit(text).path or ""
    else:
        rest = text.split("?", 1)[0]
        path = rest.split("/", 1)[1] if "/" in rest else ""
    segment = path.strip("/").split("/", 1)[0].split("?", 1)[0]
    return segment.lower()


def circuit_of(url_or_host: str) -> str:
    """The soft-failure circuit key: ``host`` or ``host/<service path>``."""
    host = host_of(url_or_host)
    if not host:
        return ""
    segment = service_path_of(url_or_host)
    return f"{host}/{segment}" if segment else host


class HostCircuitOpen(RuntimeError):
    """Raised by :meth:`HostBreaker.check` while a circuit is open.

    ``host`` is the circuit label -- the bare hostname for a host-level (hard)
    trip, ``host/<service>`` for a service-path (soft) trip; ``hostname`` is
    always the bare host.
    """

    def __init__(self, host: str, retry_after: float, reason: str):
        self.host = host
        self.hostname = host_of(host)
        self.circuit = host
        self.retry_after = max(0.0, float(retry_after))
        self.reason = reason
        super().__init__(
            f"circuit breaker open for {host} (retry in {int(round(self.retry_after))} s): {reason}"
        )


# Gateway statuses that mean "the host / its front proxy is not serving", as
# opposed to "your request was bad" (4xx) or "this query blew up" (500).
_GATEWAY_STATUSES = frozenset({502, 503, 504})

# Exception class names from requests / urllib3 / httpx / http.client that are
# transport failures. Matched by NAME so this module never imports them.
_INFRA_CLASS_NAMES = frozenset({
    "ConnectionError", "ConnectTimeout", "ReadTimeout", "Timeout", "SSLError",
    "NewConnectionError", "NameResolutionError", "MaxRetryError", "ProtocolError",
    "RemoteDisconnected", "ConnectError", "ReadError", "WriteError", "PoolTimeout",
    "ProxyError", "ChunkedEncodingError", "ReadTimeoutError", "ConnectTimeoutError",
    "IncompleteRead",
})

# Our own budget stops: they describe the CALLER giving up (tool budget spent,
# turn cancelled), never the host -- they must not count as failures.
_OWN_STOP_CLASS_NAMES = frozenset({"BudgetExhausted", "TurnCancelled"})

# Text markers for wrapped errors (pyvo DALServiceError / DALFormatError,
# astroquery, our own RuntimeError("ALMA TAP query failed: ...")).
_INFRA_TEXT_MARKERS = (
    "connection refused", "connection reset", "connection aborted", "remote end closed",
    "name or service not known", "nodename nor servname", "temporary failure in name resolution",
    "getaddrinfo failed", "no address associated", "network is unreachable", "no route to host",
    "max retries exceeded", "newconnectionerror", "nameresolutionerror", "sslerror",
    "ssl: ", "tlsv1", "certificate verify failed", "eof occurred in violation of protocol",
    "read timed out", "connect timeout", "readtimeout", "connecttimeout",
    "read operation timed out", "did not answer within",
    "bad gateway", "service unavailable", "gateway time-out", "gateway timeout", "proxy error",
    "502 server error", "503 server error", "504 server error",
)

# A wrapped message that names a request-level HTTP status is about the
# QUERY (bad ADQL, a heavy aggregate the server aborted, "query execution
# timed out"), not the host -- never trip on it even if it says "timed out"
# (guard CX-04).
_REQUEST_STATUS_RE = re.compile(r"\b(?:http\s*)?(4\d\d|500)\b", re.I)

# Exceptions whose message must NOT be text-matched: they describe the request
# (bad argument, missing key), and a word like "timeout" in them is about a
# parameter, not the network.
_REQUEST_LEVEL_TYPES = (ValueError, TypeError, KeyError, AttributeError, AssertionError, LookupError)


def _status_of(exc: BaseException) -> Optional[int]:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def classify_infrastructure_error(
    exc: Optional[BaseException], *, status: Optional[int] = None
) -> Optional[str]:
    """Return a short reason when ``exc``/``status`` is an infrastructure
    failure (host unreachable, TLS broken, transport timeout, gateway down);
    ``None`` for everything else (bad argument, 4xx, 500, parse error, our own
    budget/cancellation stops, ...).

    Walks ``__cause__``/``__context__`` because pyvo, astroquery and requests
    all wrap the underlying socket/urllib3 error.
    """
    if isinstance(status, int) and status in _GATEWAY_STATUSES:
        return f"HTTP {status}"
    if exc is None:
        return None
    seen: set = set()
    stack: List[BaseException] = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        name = type(current).__name__
        if name in _OWN_STOP_CLASS_NAMES:
            return None
        st = _status_of(current)
        if st in _GATEWAY_STATUSES:
            return f"HTTP {st}"
        if isinstance(current, (ssl.SSLError, socket.gaierror, socket.timeout, TimeoutError,
                                ConnectionError, BrokenPipeError)):
            return name
        if isinstance(current, _REQUEST_LEVEL_TYPES) and not isinstance(current, (ssl.SSLError,)):
            # Do not text-match; but still look at what it wraps.
            stack.extend([getattr(current, "__cause__", None), getattr(current, "__context__", None)])
            continue
        if name in _INFRA_CLASS_NAMES:
            return name
        text = (name + " " + str(current)).lower()
        if any(marker in text for marker in _INFRA_TEXT_MARKERS):
            # Specific transport wording (read timed out, connection reset,
            # proxy error, ...) decides first -- even when the message also
            # carries a number that looks like a status ("read timeout=400.0").
            return name
        if _REQUEST_STATUS_RE.search(text):
            # A request-level HTTP status with no transport wording
            # ("HTTP 500: heavy ADQL query timed out") is about the QUERY,
            # not the host (guard CX-04): keep walking the chain only.
            stack.extend([getattr(current, "__cause__", None), getattr(current, "__context__", None)])
            continue
        stack.extend([getattr(current, "__cause__", None), getattr(current, "__context__", None)])
    return None


# Reasons that mean the HOST is unreachable (trip at once) versus reasons a
# perfectly healthy host produces on one heavy request (read timeout, proxy
# 502/504 on a long-running query): the latter are "soft" and only trip after
# HOST_BREAKER_SOFT_TRIPS distinct tool calls failed over >= the spread window.
# Live 2026-09-22: a single heavy Data Lab aggregate's ReadTimeout opened the
# whole Data Lab host for 120 s and zeroed three DataLabBench questions.
_SOFT_REASON_MARKERS = ("readtimeout", "read timed out", "did not answer within", "http 502", "http 503", "http 504",
                        "timeout limit", "read operation timed out")


def severity_of(reason: str) -> str:
    """'soft' for read-timeout / gateway-status reasons, 'hard' for everything
    else that classify_infrastructure_error returns (refused, reset, DNS, TLS,
    connect timeout)."""
    text = (reason or "").lower()
    if any(m in text for m in _SOFT_REASON_MARKERS):
        return "soft"
    if text in {"timeout", "timeouterror"}:
        return "soft"
    return "hard"


def _current_call_token() -> Optional[object]:
    """Identity of the tool call running on this thread (its inner deadline),
    None outside a guarded tool. Lazy import: tool_budgets is stdlib-only."""
    try:
        from services.tool_budgets import current_call_token

        return current_call_token()
    except Exception:  # pragma: no cover - defensive: never break a request path
        return None


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        value = float(os.getenv(name, "") or default)
    except ValueError:
        value = float(default)
    if value != value:  # NaN
        value = float(default)
    return max(minimum, value)


class HostBreaker:
    """Class-level registry: one circuit per host (hard) or host/service (soft),
    shared by the process."""

    _lock = threading.Lock()
    # circuit -> {"until": monotonic, "reason": str, "opened_at": wall, "trips": int, "cooldown": s,
    #             "circuit": str, "severity": "hard"|"soft"}
    _open: Dict[str, Dict[str, Any]] = {}
    # host -> consecutive HARD infrastructure failures while closed
    _failures: Dict[str, int] = {}
    # circuit -> [(call_token, monotonic ts)] soft failures, at most one per tool call
    _soft: Dict[str, List[Tuple[object, float]]] = {}
    # circuit -> trip count carried across an expired cooldown (probe failing again
    # doubles the next cooldown; a success clears it)
    _last_trip_count: Dict[str, int] = {}

    @staticmethod
    def cooldown_seconds() -> float:
        value = _env_float("HOST_BREAKER_COOLDOWN_SECONDS", 120.0)
        return value if value > 0 else 120.0

    @staticmethod
    def max_cooldown_seconds() -> float:
        return _env_float("HOST_BREAKER_MAX_COOLDOWN_SECONDS", 900.0, minimum=1.0)

    @staticmethod
    def trip_failures() -> int:
        """Consecutive HARD infrastructure failures before the host opens.
        Default 1: a host that refused / reset once is down for the purposes of
        THIS turn -- the point is that the *second* call is instant."""
        try:
            value = int(float(os.getenv("HOST_BREAKER_TRIP_FAILURES", "1") or 1))
        except ValueError:
            value = 1
        return max(1, value)

    @staticmethod
    def soft_trips() -> int:
        """Distinct tool calls with a SOFT failure (read timeout, 502/503/504)
        before the service circuit opens. Default 3: one heavy query timing out
        is not an outage."""
        try:
            value = int(float(os.getenv("HOST_BREAKER_SOFT_TRIPS", "3") or 3))
        except ValueError:
            value = 3
        return max(1, value)

    @staticmethod
    def soft_spread_seconds() -> float:
        """The soft failures must span at least this long (first to last) --
        three tool calls fired in the same second against a slow proxy are one
        incident, not an outage. Default 20 s."""
        return _env_float("HOST_BREAKER_SOFT_SPREAD_SECONDS", 20.0)

    @staticmethod
    def soft_window_seconds() -> float:
        """Soft failures older than this are forgotten. Default 600 s."""
        return _env_float("HOST_BREAKER_SOFT_WINDOW_SECONDS", 600.0, minimum=1.0)

    @staticmethod
    def auto_wait_seconds() -> float:
        """A cooldown at or below this is short enough to wait out INSIDE the
        turn (the tool guard sleeps, then probes) instead of skipping the tool.
        Default 30 s. 0 disables the in-turn wait."""
        return _env_float("HOST_BREAKER_AUTO_WAIT_SECONDS", 30.0)

    @staticmethod
    def enabled() -> bool:
        return os.getenv("HOST_BREAKER_DISABLED", "").strip().lower() not in {"1", "true", "yes", "on"}

    # ── queries ──────────────────────────────────────────────────────────
    @classmethod
    def _state_locked(cls, key: str, now: float) -> Optional[Dict[str, Any]]:
        rec = cls._open.get(key)
        if rec is None:
            return None
        if rec["until"] <= now:
            # Cooldown over: close, but remember the trip count so a failing
            # probe re-opens with a longer cooldown.
            cls._open.pop(key, None)
            cls._failures.pop(key, None)
            cls._soft.pop(key, None)
            cls._last_trip_count[key] = int(rec.get("trips", 1))
            return None
        out = dict(rec)
        out["retry_after"] = rec["until"] - now
        out.setdefault("circuit", key)
        return out

    @classmethod
    def state(cls, url_or_host: str) -> Optional[Dict[str, Any]]:
        """Open-state record for the host-level circuit or, failing that, the
        URL's service-path circuit (with ``retry_after`` and ``circuit``), or
        None when both are closed. Expired records are removed here (lazy close)."""
        host = host_of(url_or_host)
        if not host:
            return None
        circuit = circuit_of(url_or_host)
        now = _now()
        with cls._lock:
            for key in dict.fromkeys((host, circuit)):
                rec = cls._state_locked(key, now)
                if rec is not None:
                    return rec
        return None

    @classmethod
    def is_open(cls, url_or_host: str) -> bool:
        return cls.enabled() and cls.state(url_or_host) is not None

    @classmethod
    def open_hosts(cls, hosts: Iterable[str]) -> List[Tuple[str, Dict[str, Any]]]:
        """``[(circuit label, record)]`` for every entry whose host or service
        circuit is open. Entries may be bare hosts (host-level check only) or
        ``host/service`` labels / URLs (host-level AND that service)."""
        out: List[Tuple[str, Dict[str, Any]]] = []
        if not cls.enabled():
            return out
        for h in hosts:
            rec = cls.state(h)
            if rec is not None:
                out.append((str(rec.get("circuit") or host_of(h)), rec))
        return out

    @classmethod
    def check(cls, url_or_host: str) -> None:
        """Raise :class:`HostCircuitOpen` if the host's (or this URL's service
        path's) circuit is open."""
        if not cls.enabled():
            return
        rec = cls.state(url_or_host)
        if rec is not None:
            raise HostCircuitOpen(str(rec.get("circuit") or host_of(url_or_host)), rec["retry_after"], rec["reason"])

    # ── updates ──────────────────────────────────────────────────────────
    @classmethod
    def record_success(cls, url_or_host: str) -> None:
        """A completed request proves the host reachable and this service
        healthy: close the host circuit and this URL's service circuit, clear
        their streaks. Other services on the same host keep their own state."""
        host = host_of(url_or_host)
        if not host:
            return
        circuit = circuit_of(url_or_host)
        with cls._lock:
            for key in dict.fromkeys((host, circuit)):
                cls._failures.pop(key, None)
                cls._soft.pop(key, None)
                cls._open.pop(key, None)
                cls._last_trip_count.pop(key, None)

    @classmethod
    def _open_locked(cls, key: str, *, why: str, detail: str, severity: str, count_note: str, now: float) -> None:
        prior = cls._open.get(key)
        trips = (int(prior["trips"]) if prior else cls._last_trip_count.get(key, 0)) + 1
        base = cls.cooldown_seconds()
        cooldown = min(base * (2 ** (trips - 1)), cls.max_cooldown_seconds())
        cls._open[key] = {
            "until": now + cooldown,
            "reason": detail,
            "opened_at": time.time(),
            "trips": trips,
            "cooldown": cooldown,
            "circuit": key,
            "severity": severity,
        }
        cls._failures.pop(key, None)
        cls._soft.pop(key, None)
        print(
            f"[HOST BREAKER] {key} circuit OPEN for {int(cooldown)}s "
            f"(trip {trips}, {count_note}) — {why}"
        )

    @classmethod
    def record_failure(
        cls,
        url_or_host: str,
        exc: Optional[BaseException] = None,
        *,
        status: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> Optional[str]:
        """Record an outcome. Returns the infrastructure reason when the failure
        counted (and possibly tripped a circuit), None when it was ignored as a
        request-level error.

        Hard reasons open the HOST circuit; soft reasons are scoped to the
        URL's service path and counted once per tool call."""
        host = host_of(url_or_host)
        if not host or not cls.enabled():
            return None
        why = reason or classify_infrastructure_error(exc, status=status)
        if why is None:
            return None
        detail_for_severity = why if exc is None else f"{why}: {exc}"
        soft = severity_of(detail_for_severity) == "soft"
        detail = f"{why}: {str(exc)[:160]}" if exc is not None else why
        now = _now()
        if not soft:
            needed = cls.trip_failures()
            with cls._lock:
                if cls._state_locked(host, now) is not None:
                    # Already open: an in-flight request that failed after the
                    # trip must not re-open it (each re-open doubles the
                    # cooldown -- guard CX-34 race).
                    return why
                n = cls._failures.get(host, 0) + 1
                cls._failures[host] = n
                if n >= needed:
                    cls._open_locked(host, why=why, detail=detail, severity="hard", count_note="hard", now=now)
            return why

        circuit = circuit_of(url_or_host)
        token = _current_call_token()
        token_note = "tool call" if token is not None else "bare call"
        if token is None:
            token = object()  # outside a guarded tool every record is its own call
        needed = cls.soft_trips()
        spread_needed = cls.soft_spread_seconds()
        window = cls.soft_window_seconds()
        with cls._lock:
            if cls._state_locked(circuit, now) is not None or cls._state_locked(host, now) is not None:
                return why  # already open: never re-trip / escalate from in-flight failures (CX-34)
            entries = [e for e in cls._soft.get(circuit, []) if now - e[1] <= window]
            if any(t is token for t, _ in entries):
                cls._soft[circuit] = entries
                print(
                    f"[HOST BREAKER] {circuit} soft failure ({why}) already counted for this {token_note} "
                    f"— {len(entries)}/{needed}, not tripping"
                )
                return why
            entries.append((token, now))
            cls._soft[circuit] = entries
            n = len(entries)
            spread = entries[-1][1] - entries[0][1]
            if n >= needed and spread >= spread_needed:
                cls._open_locked(
                    circuit, why=why, detail=detail, severity="soft",
                    count_note=f"soft x{n} over {spread:.0f}s", now=now,
                )
            else:
                print(
                    f"[HOST BREAKER] {circuit} soft failure {n}/{needed} ({why}; spread {spread:.0f}s"
                    f"/{spread_needed:.0f}s needed) — not tripping yet"
                )
        return why

    @classmethod
    def snapshot(cls) -> Dict[str, Dict[str, Any]]:
        now = _now()
        with cls._lock:
            return {
                h: {
                    "retry_after": max(0.0, r["until"] - now),
                    "reason": r["reason"],
                    "trips": r["trips"],
                    "opened_at": r["opened_at"],
                    "severity": r.get("severity", "hard"),
                }
                for h, r in cls._open.items()
            }

    @classmethod
    def reset(cls) -> None:
        """Test hook: forget every host and circuit."""
        with cls._lock:
            cls._open.clear()
            cls._failures.clear()
            cls._soft.clear()
            cls._last_trip_count.clear()


def structured_error(
    exc: HostCircuitOpen,
    *,
    tool_name: str = "",
    affected_tools: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """The tool-result dict returned when a call is refused by an open breaker.

    Shape mirrors the tool guard's TIMEOUT / CIRCUIT BREAKER results
    (core/agent.py) so the runner and SSE layer classify it the same way:
    ``success False``, ``circuit_breaker True``, plus ``infrastructure_failure``
    so a retry against the same host is recognisable as pointless, and
    ``retry_after_s`` so a SHORT cooldown can be waited out inside the turn.
    """
    siblings = sorted({t for t in (affected_tools or []) if t and t != tool_name})
    retry = int(round(exc.retry_after))
    auto_wait = HostBreaker.auto_wait_seconds()
    sibling_note = f" or its siblings on the same service ({', '.join(siblings[:6])})" if siblings else ""
    if 0 < retry <= auto_wait:
        text = (
            f"INFRASTRUCTURE FAILURE: {exc.host} is unreachable right now ({exc.reason}); "
            f"this call was NOT executed (failed fast). The circuit re-opens for a live probe in "
            f"~{retry} s (retry_after_s={retry}). You MAY retry this exact call ONCE after that wait "
            "(the platform waits automatically when the turn budget allows); if it fails again, do not "
            f"retry this tool{sibling_note} this turn — answer with the data you already have, say plainly "
            "that the service was unavailable, or use a tool on a DIFFERENT service."
        )
    else:
        text = (
            f"INFRASTRUCTURE FAILURE: {exc.host} is unreachable right now ({exc.reason}); "
            f"this call was NOT executed (failed fast). The service is on cooldown for ~{retry} s "
            f"(retry_after_s={retry}), so retrying this tool{sibling_note} this turn will fail the same way. "
            "Do NOT retry; answer with the data you already have, say plainly that the service is down, "
            "or use a tool on a DIFFERENT service."
        )
    return {
        "success": False,
        "circuit_breaker": True,
        "infrastructure_failure": True,
        "host": exc.host,
        "hostname": exc.hostname,
        "circuit": exc.circuit,
        "retry_after_seconds": retry,
        "retry_after_s": retry,
        "skipped_at_epoch": time.time(),
        "affected_tools": siblings,
        "error": text,
    }


def guarded_request(method: str, url: str, **kwargs):
    """``requests.request`` with the host breaker around it.

    Raises :class:`HostCircuitOpen` in microseconds while the host (or this
    URL's service path) is open; otherwise performs the call, records
    infrastructure failures (transport exceptions, 502/503/504) and successes,
    and returns the response. Callers keep their own timeout policy -- this
    never changes ``kwargs``.
    """
    import requests  # local: keep the module importable without requests

    HostBreaker.check(url)
    # Dispatch through requests.get / requests.post for those verbs so code
    # (and tests) that patch them keep working; other verbs use request().
    verb = str(method or "GET").upper()
    sender = {"GET": requests.get, "POST": requests.post}.get(verb)
    from services.http_budget_hook import suppressed

    try:
        with suppressed():
            response = sender(url, **kwargs) if sender is not None else requests.request(verb, url, **kwargs)
    except requests.RequestException as exc:
        HostBreaker.record_failure(url, exc)
        raise
    status = int(getattr(response, "status_code", 0) or 0)
    if status in _GATEWAY_STATUSES:
        HostBreaker.record_failure(url, status=status)
    else:
        HostBreaker.record_success(url)
    return response
