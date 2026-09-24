"""Wall-clock budget hierarchy for tool execution — ONE rule, enforced twice.

The rule
--------
Every inner network budget (an HTTP timeout, a per-download wall, a TAP
attempt, a MAST query) is STRICTLY BELOW the outer tool guard of every tool
that can pay it, with ``GUARD_HEADROOM_SECONDS`` to spare. The guard
(``QuasarAgent._execute_tool_guarded``) is the last resort — it fires only
when a tool has ignored its own budget — so the normal failure path is the
service returning its own, specific, structured error (partial grid, "TAP
mirror down", "no coverage") that the model can answer around, not the blunt
``timed out after 150s``.

Enforced twice:

1. **Statically** — :data:`INNER_BUDGETS` declares, per tool, the worst-case
   SEQUENTIAL inner budget the tool can pay as a function of the live service
   defaults. ``tests/unit/test_tool_budget_hierarchy.py`` fails if any
   registered tool's declared inner total reaches ``guard - headroom``, or if a
   tool with an elevated guard has no declaration.
2. **Dynamically** — the guard's worker thread opens a :class:`Deadline` for
   the tool (:func:`begin_tool_deadline`), and every inner network call sizes
   its own timeout with :func:`bounded_timeout` = ``min(default, remaining)``.
   Loops (one MAST query per source, one download per band, three TAP mirrors)
   therefore cannot outlast the guard no matter how many iterations they run:
   once the budget is spent the next call raises :class:`BudgetExhausted` at
   once, and the tool returns a partial result BEFORE the guard fires.

This module is a leaf (stdlib only). It lives under services/ (lazy package
init) rather than core/ because ``core/__init__`` eagerly imports the agent —
an integration importing ``core.tool_budgets`` at module load would trigger
that and re-enter its own half-initialised module.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

__all__ = [
    "BudgetExhausted",
    "Deadline",
    "TurnCancellation",
    "TurnCancelled",
    "GUARD_DEFAULT_SECONDS",
    "GUARD_HEADROOM_SECONDS",
    "GUARD_OVERRIDES",
    "DEADLINE_ONLY_TOOLS",
    "INNER_BUDGETS",
    "LOOP_TOOLS",
    "TOOL_HOSTS",
    "adopt_deadline",
    "begin_tool_deadline",
    "bounded_timeout",
    "call_bounded",
    "check_hierarchy",
    "current_call_token",
    "current_deadline",
    "end_tool_deadline",
    "is_cancelled",
    "make_tool_deadline",
    "guard_seconds",
    "hosts_for",
    "inner_ceiling_seconds",
    "remaining_seconds",
    "tools_for_host",
]

# ── outer guards ─────────────────────────────────────────────────────────
# One stuck external service must not hold a whole turn hostage (live
# 2026-07-18: a single CADC TAP search ran 477 s while the UI showed
# "Generating answer"). QUASAR_TOOL_TIMEOUT_SECONDS=0 disables the guard.
GUARD_DEFAULT_SECONDS = 150.0

# Tools that legitimately run long. 2026-09-21: the ALMA search family was
# 390/465 s here — inverted against an archive that answers in ~2 s (live
# probe): a hung mirror cost 6.5 minutes of silence per call, and the
# 12-source cross-match ran 3 x 120 s TAP attempts + 12 x 600 s MAST queries
# under a 150 s guard, so the guard ALWAYS fired and the model re-called the
# tool (then its sibling) for 450 s of dead time. Budgets now go DOWN only.
GUARD_OVERRIDES: Dict[str, float] = {
    # ALMA archive queries: two 45 s TAP attempts on rotating mirrors, a 30 s
    # ALminer fallback and a 15 s SIMBAD resolve fit under the default guard;
    # cycle-wide science queries get a little more for their post-processing.
    "query_alma_science_archive": 180.0,
    "advanced_search": 180.0,
    "get_observation_details": 180.0,
    "search_by_frequency": 180.0,
    "search_by_target": 150.0,
    "search_by_position": 150.0,
    "web_research": 420.0,
    "download_alma_data": 600.0,
    # L9: bulk MAST FITS pulls (e.g. TESS lightcurves) legitimately run
    # long — match download_alma_data; MASTClient's own wall bound
    # (MAST_DOWNLOAD_WALL_SECONDS, default 480 s) fires first with the
    # more specific error.
    "download_mast_data": 600.0,
    # Density vetting = density SQL PLUS an SIA cutout grid in one call;
    # under NSC/archive load it cannot fit the 150 s default, and its
    # abandonment is what tipped the 2026-08-04 production hang into the
    # aggregate + per-peak-cutout grind (density report fix 10).
    "datalab_density_vetting": 300.0,
    # One-shot tools (UI benchmark 2026-09-22): whole-archive server-side
    # scans (redshift ~75 s, BWSW ~150 s live) and the tiled satellite search
    # with per-candidate CMDs need more than the default; all stay <= 300 s so
    # a turn's tool budget (300 s) still ends with a written answer.
    "alma_project_census": 240.0,
    "cross_archive_match": 180.0,
    "archive_overlay": 180.0,
    "datalab_healpix_density_map": 300.0,
    "datalab_satellite_search": 300.0,
}

# Inner totals must stay this far below the guard: enough for the tool to
# assemble and return its OWN structured partial result after its last inner
# call gives up, so the guard (which discards everything) never has to.
GUARD_HEADROOM_SECONDS = 15.0


def guard_seconds(tool_name: str) -> Optional[float]:
    """Resolve the wall-clock guard for one tool; None = guard disabled.

    Precedence: QUASAR_TOOL_TIMEOUT_OVERRIDES="tool=secs,..." (per tool, 0 =
    disable) > GUARD_OVERRIDES > QUASAR_TOOL_TIMEOUT_SECONDS > default.
    """
    raw = os.getenv("QUASAR_TOOL_TIMEOUT_SECONDS", "").strip()
    try:
        default = float(raw) if raw else GUARD_DEFAULT_SECONDS
    except ValueError:
        default = GUARD_DEFAULT_SECONDS
    if default <= 0:
        return None
    budget: Optional[float] = GUARD_OVERRIDES.get(tool_name, default)
    for pair in os.getenv("QUASAR_TOOL_TIMEOUT_OVERRIDES", "").split(","):
        name, sep, secs = pair.partition("=")
        if sep and name.strip() == tool_name:
            try:
                val = float(secs)
            except ValueError:
                continue
            budget = val if val > 0 else None
    return budget


def inner_ceiling_seconds(tool_name: str) -> Optional[float]:
    """The most a tool's inner network work may take in total (guard - headroom)."""
    guard = guard_seconds(tool_name)
    if guard is None:
        return None
    return max(1.0, guard - GUARD_HEADROOM_SECONDS)


# ── dynamic deadline ─────────────────────────────────────────────────────
class BudgetExhausted(TimeoutError):
    """Raised by :func:`bounded_timeout` when the tool's inner budget is spent.

    A ``TimeoutError`` subclass so existing ``_is_timeout_error`` classifiers
    (Data Lab image grid, FITS watchdog) treat it as a timeout, not a crash.
    """

    def __init__(self, label: str, remaining: float, minimum: float):
        self.label = label
        self.remaining = remaining
        self.minimum = minimum
        super().__init__(
            f"tool budget exhausted before {label or 'the next network call'} "
            f"({remaining:.1f} s left, {minimum:.1f} s needed) — returning what was collected"
        )


class TurnCancelled(BudgetExhausted):
    """Raised by :meth:`Deadline.bounded` (and the requests hook) once the
    tool call or its whole turn has been cancelled: the client disconnected,
    the user pressed Stop, the SSE ceiling fired, or the tool guard abandoned
    this worker. No NEW network request may start after this; whatever was
    collected is returned as a partial result.

    Live 2026-09-22 (UI benchmark L06): a turn stopped in the UI at 420 s kept
    its detached Data Lab workers issuing 70 s ReadTimeouts for three more
    minutes and slowed the following questions.
    """

    def __init__(self, label: str = "", reason: str = ""):
        self.label = label
        self.remaining = 0.0
        self.minimum = 0.0
        self.reason = reason
        TimeoutError.__init__(
            self,
            f"turn cancelled before {label or 'the next network call'}"
            + (f" ({reason})" if reason else "")
            + " — no new network requests; returning what was collected",
        )


class TurnCancellation:
    """Per-turn cancellation token shared by every tool deadline of the turn.

    Created by the runner at turn start, handed to :func:`make_tool_deadline`
    by the tool guard, and fired by ``QuasarAgent.cancel_response_run`` (SSE
    disconnect / Stop / hard ceiling) or by the runner's own hard-cap exit.
    :meth:`cancel` marks every LIVE deadline (a worker that has not called
    :func:`end_tool_deadline` yet) cancelled and returns how many there were.
    """

    __slots__ = ("label", "_event", "_lock", "_live", "cancelled_at", "reason")

    def __init__(self, label: str = ""):
        self.label = label
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._live: set = set()
        self.cancelled_at: Optional[float] = None
        self.reason = ""

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def register(self, deadline: "Deadline") -> None:
        with self._lock:
            late = self._event.is_set()
            if not late:
                self._live.add(deadline)
            reason = self.reason
        if late:
            # Registered after cancel (a worker racing Stop): born cancelled,
            # with the turn's reason, and never counted as live (guard CX-34).
            deadline.cancel(reason or "turn already cancelled")

    def release(self, deadline: "Deadline") -> None:
        with self._lock:
            self._live.discard(deadline)

    def live_count(self) -> int:
        with self._lock:
            return len(self._live)

    def cancel(self, reason: str = "") -> int:
        """Cancel the turn. Idempotent; returns the number of tool workers that
        were still live (they stop before their next network request)."""
        with self._lock:
            first = not self._event.is_set()
            self._event.set()
            if first:
                self.cancelled_at = time.monotonic()
                self.reason = reason
            live = list(self._live)
        for d in live:
            d.cancel(reason)
        if not first:
            return 0  # already cancelled: nothing new was stopped
        print(f"[TURN CANCELLED] {self.label or 'turn'} n_workers_stopped={len(live)} reason={reason or 'n/a'}")
        return len(live)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TurnCancellation({self.label!r}, cancelled={self.cancelled}, live={self.live_count()})"


class Deadline:
    """A monotonic deadline with helpers for sizing nested timeouts.

    Also the identity of ONE tool call (``token``: shared with the child
    deadlines that helper threads adopt, so the host breaker can count a burst
    of sub-requests as a single soft failure) and a cancellation flag
    (:meth:`cancel` / :meth:`cancelled`) that stops abandoned workers from
    starting new network requests.
    """

    __slots__ = ("label", "seconds", "_start", "_end", "token", "turn", "_parent", "_cancelled", "cancel_reason")

    def __init__(
        self,
        seconds: float,
        *,
        label: str = "",
        turn: Optional[TurnCancellation] = None,
        _parent: Optional["Deadline"] = None,
        _start: Optional[float] = None,
        _end: Optional[float] = None,
        _token: Optional[object] = None,
    ):
        self.label = label
        self.seconds = float(seconds)
        self._start = time.monotonic() if _start is None else float(_start)
        self._end = (self._start + self.seconds) if _end is None else float(_end)
        self.token = _token if _token is not None else object()
        self.turn = turn
        self._parent = _parent
        self._cancelled = False
        self.cancel_reason = ""

    def remaining(self) -> float:
        return max(0.0, self._end - time.monotonic())

    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def expired(self) -> bool:
        return time.monotonic() >= self._end

    # ── cancellation ────────────────────────────────────────────────
    def cancel(self, reason: str = "") -> None:
        self._cancelled = True
        if reason and not self.cancel_reason:
            self.cancel_reason = reason

    def cancelled(self) -> bool:
        """True once this deadline, any deadline it was derived from, or the
        whole turn has been cancelled."""
        if self._cancelled:
            return True
        parent = self._parent
        while parent is not None:
            if parent._cancelled:
                return True
            parent = parent._parent
        return bool(self.turn is not None and self.turn.cancelled)

    def why_cancelled(self) -> str:
        if self.cancel_reason:
            return self.cancel_reason
        parent = self._parent
        while parent is not None:
            if parent.cancel_reason:
                return parent.cancel_reason
            parent = parent._parent
        return (self.turn.reason if self.turn is not None else "") or "cancelled"

    def child_until(self, seconds: float, *, label: str = "") -> "Deadline":
        """A child that ends after ``seconds`` (or at this deadline's end,
        whichever is first). Every bounded request on the helper then clamps
        its timeout to that end, so an abandoned helper's IN-FLIGHT request
        finishes by then too (guard 12894 CX-02/03), not only its next one."""
        end = min(self._end, time.monotonic() + max(0.0, float(seconds)))
        return Deadline(
            max(0.0, end - self._start), label=label or self.label, turn=self.turn,
            _parent=self, _start=self._start, _end=end, _token=self.token,
        )

    def child(self, *, label: str = "") -> "Deadline":
        """A deadline for a helper worker: same end time, same call token and
        turn, but its OWN cancel flag -- so abandoning the helper stops the
        helper without cancelling the tool that spawned it."""
        return Deadline(
            self.seconds, label=label or self.label, turn=self.turn,
            _parent=self, _start=self._start, _end=self._end, _token=self.token,
        )

    def bounded(self, default: float, *, minimum: float = 1.0, label: str = "") -> float:
        """``min(default, remaining)`` — raises :class:`BudgetExhausted` when
        less than ``minimum`` seconds are left, so a call that could not
        possibly finish is refused instantly instead of started, and
        :class:`TurnCancelled` once the call/turn has been cancelled."""
        if self.cancelled():
            raise TurnCancelled(label or self.label, self.why_cancelled())
        remaining = self.remaining()
        if remaining < minimum:
            raise BudgetExhausted(label or self.label, remaining, minimum)
        return min(float(default), remaining)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Deadline({self.label!r}, remaining={self.remaining():.1f}s, cancelled={self.cancelled()})"


_tls = threading.local()


def make_tool_deadline(
    tool_name: str, budget_seconds: Optional[float], *, turn: Optional[TurnCancellation] = None
) -> Optional[Deadline]:
    """Build (and register with the turn) the inner deadline for a tool call
    WITHOUT making it current: the guard creates it on the parent thread so it
    keeps a handle for cancellation, and the worker adopts it.

    ``budget_seconds`` is the guard budget; the inner deadline is that minus
    the headroom. None (guard disabled) -> None.
    """
    if budget_seconds is None:
        return None
    inner = max(1.0, float(budget_seconds) - GUARD_HEADROOM_SECONDS)
    deadline = Deadline(inner, label=tool_name, turn=turn)
    if turn is not None:
        turn.register(deadline)
    return deadline


UNBOUNDED_DEADLINE_SECONDS = 1e9


def make_identity_deadline(tool_name: str, *, turn: Optional[TurnCancellation] = None) -> Deadline:
    """A deadline with NO time bound (inner calls keep their own defaults) that
    still carries the call identity (breaker soft-failure dedup) and the turn
    cancellation -- for the guard-disabled inline path (guard CX-06)."""
    deadline = Deadline(UNBOUNDED_DEADLINE_SECONDS, label=tool_name, turn=turn)
    if turn is not None:
        turn.register(deadline)
    return deadline


def begin_tool_deadline(
    tool_name: str, budget_seconds: Optional[float], *, turn: Optional[TurnCancellation] = None
) -> Optional[Deadline]:
    """Open the inner deadline for a tool on THIS thread (the guard's worker).

    ``budget_seconds`` is the guard budget; the inner deadline is that minus
    the headroom. None (guard disabled) clears any deadline so inner calls
    fall back to their own defaults.
    """
    deadline = make_tool_deadline(tool_name, budget_seconds, turn=turn)
    _tls.deadline = deadline
    return deadline


def end_tool_deadline() -> None:
    """Close THIS thread's tool deadline (releasing it from its turn)."""
    deadline = getattr(_tls, "deadline", None)
    if deadline is not None and deadline.turn is not None and deadline._parent is None:
        deadline.turn.release(deadline)
    _tls.deadline = None


def adopt_deadline(deadline: Optional[Deadline]) -> None:
    """Make ``deadline`` current on THIS thread. Deadlines are thread-local,
    so a tool that fans work out to helper threads (concurrent archive
    phases) must hand its deadline to each of them or their network calls
    fall back to unbounded defaults."""
    _tls.deadline = deadline


def current_deadline() -> Optional[Deadline]:
    return getattr(_tls, "deadline", None)


def current_call_token() -> Optional[object]:
    """Identity of the tool call running on this thread (shared by its helper
    threads); None outside a guarded tool."""
    deadline = current_deadline()
    return deadline.token if deadline is not None else None


def is_cancelled() -> bool:
    """True when the tool call / turn on this thread has been cancelled."""
    deadline = current_deadline()
    return deadline is not None and deadline.cancelled()


def remaining_seconds() -> Optional[float]:
    """Seconds left in the current tool's inner budget; None outside a tool."""
    deadline = current_deadline()
    return deadline.remaining() if deadline is not None else None


def bounded_timeout(default: float, *, minimum: float = 1.0, label: str = "") -> float:
    """Size an inner network timeout: ``min(default, remaining tool budget)``.

    Outside a guarded tool (tests, scripts, Conductor threads) there is no
    deadline and the default is returned unchanged. Inside one, a call that
    cannot get ``minimum`` seconds raises :class:`BudgetExhausted` immediately.
    """
    deadline = current_deadline()
    if deadline is None:
        return float(default)
    return deadline.bounded(default, minimum=minimum, label=label)


def call_bounded(fn, seconds: float, *, label: str = "", thread_name: str = "quasar-bounded-call"):
    """Run ``fn()`` on a daemon worker and wait at most ``seconds``.

    For third-party clients whose own timeout is not an HTTP timeout at all
    (astroquery's Simbad/MAST ``timeout`` is a server-side execution
    duration). On expiry the worker is abandoned (same contract as the tool
    guard) and a ``TimeoutError`` is raised — an infrastructure-class failure
    for the host breaker. ``seconds`` should already be budget-bounded via
    :func:`bounded_timeout`.
    """
    box: Dict[str, object] = {}
    parent_deadline = current_deadline()
    # The helper gets a CHILD deadline: same budget and call token, its own
    # cancel flag -- abandoning it (below) stops it before its next network
    # request without cancelling the tool that spawned it.
    child = parent_deadline.child(label=label or parent_deadline.label) if parent_deadline is not None else None

    def _run():
        adopt_deadline(child)
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            box["exc"] = exc

    worker = threading.Thread(target=_run, daemon=True, name=thread_name)
    worker.start()
    wall = max(0.0, float(seconds))
    end = time.monotonic() + wall
    while True:
        left = end - time.monotonic()
        if left <= 0:
            break
        worker.join(min(0.25, left))
        if not worker.is_alive():
            break
        if parent_deadline is not None and parent_deadline.cancelled():
            # The tool / turn was cancelled while we waited: stop the helper
            # and stop ourselves -- nothing downstream is listening.
            if child is not None:
                child.cancel(parent_deadline.why_cancelled())
            raise TurnCancelled(label or "call", parent_deadline.why_cancelled())
    if worker.is_alive():
        if child is not None:
            child.cancel(f"{label or 'call'} abandoned after {wall:.0f} s")
        raise TimeoutError(
            f"{label or 'call'} did not answer within {wall:.0f} s (worker abandoned)"
        )
    if "exc" in box:
        raise box["exc"]  # type: ignore[misc]
    return box.get("result")


# ── static declarations (checked by the hierarchy test) ──────────────────
# tool_name -> callable returning {label: seconds}. Each getter reads the LIVE
# service constants (env included) so drift in a constant or an operator
# override is caught, not just this table. Three kinds of declaration:
#
#   declare()       fixed chain — the tool makes these calls in sequence; the
#                   SUM must stay <= guard - headroom.
#   declare_loop()  per-call chain for a tool that loops over an input-sized
#                   list or has several phases; EVERY component must stay
#                   <= guard - headroom (no single inner call may outlast the
#                   guard) and the NUMBER of calls is bounded at runtime by the
#                   tool deadline (services/http_budget_hook.py + explicit
#                   bounded_timeout wiring). Kept in LOOP_TOOLS with a reason.
#   deadline-only   transports with no static timeout constant at all (pyvo
#                   built without a session, astroquery SkyView, psrqpy, lsdb,
#                   third-party SDK defaults we cannot configure): the clamp
#                   is the only bound. Listed in DEADLINE_ONLY_TOOLS with the
#                   reason, so the test can insist the list is deliberate.
INNER_BUDGETS: Dict[str, Callable[[], Dict[str, float]]] = {}
LOOP_TOOLS: Dict[str, str] = {}
DEADLINE_ONLY_TOOLS: Dict[str, str] = {}

# tool_name -> remote hosts it depends on (for the host circuit breaker: an
# open breaker on EVERY declared host fails the call before it starts).
TOOL_HOSTS: Dict[str, Tuple[str, ...]] = {}


def _register_hosts(tool_names: Iterable[str], hosts: Iterable[str]) -> None:
    host_tuple = tuple(h.lower() for h in hosts)
    if not host_tuple:
        return
    for name in tool_names:
        TOOL_HOSTS[name] = tuple(sorted(set(TOOL_HOSTS.get(name, ())) | set(host_tuple)))


def declare(tool_names: Iterable[str], budget: Callable[[], Dict[str, float]], hosts: Iterable[str] = ()) -> None:
    """Fixed sequential chain: the SUM of the getter's values must fit."""
    names = list(tool_names)
    for name in names:
        INNER_BUDGETS[name] = budget
        LOOP_TOOLS.pop(name, None)
        DEADLINE_ONLY_TOOLS.pop(name, None)
    _register_hosts(names, hosts)


def declare_loop(tool_names: Iterable[str], budget: Callable[[], Dict[str, float]], hosts: Iterable[str] = (), *, reason: str) -> None:
    """Per-call chain for a looping / multi-phase tool: EVERY component must
    fit; the call count is bounded by the tool deadline (``reason`` says why)."""
    names = list(tool_names)
    for name in names:
        INNER_BUDGETS[name] = budget
        LOOP_TOOLS[name] = reason
        DEADLINE_ONLY_TOOLS.pop(name, None)
    _register_hosts(names, hosts)


def declare_deadline_only(tool_names: Iterable[str], hosts: Iterable[str] = (), *, reason: str) -> None:
    """A transport with NO static timeout constant: the deadline clamp is the
    only bound. Recorded (with the reason) so the hierarchy test can insist
    the list is deliberate; nothing numeric to check."""
    names = list(tool_names)
    for name in names:
        INNER_BUDGETS[name] = _ceiling(name)
        DEADLINE_ONLY_TOOLS[name] = reason
        LOOP_TOOLS.pop(name, None)
    _register_hosts(names, hosts)


def hosts_for(tool_name: str) -> Tuple[str, ...]:
    return TOOL_HOSTS.get(tool_name, ())


def _split_host_label(label: str) -> Tuple[str, str]:
    """``"datalab.noirlab.edu/sia"`` -> ``("datalab.noirlab.edu", "sia")``;
    a bare host -> ``(host, "")``. Accepts URLs too."""
    text = str(label or "").strip().lower()
    if "://" in text:
        from urllib.parse import urlsplit

        parts = urlsplit(text)
        host = parts.hostname or ""
        path = parts.path or ""
    else:
        host, _, path = text.partition("/")
        host = host.split(":", 1)[0]
    segment = path.strip("/").split("/", 1)[0] if path else ""
    return host, segment


def tools_for_host(host: str) -> List[str]:
    """Tools declared on ``host`` -- a bare hostname matches every declaration
    on that host (a host-level trip affects all of them); a ``host/service``
    circuit label matches the declarations on that service path plus the
    bare-host declarations (which may use any service)."""
    want_host, want_path = _split_host_label(host)
    if not want_host:
        return []
    out = []
    for tool, entries in TOOL_HOSTS.items():
        for entry in entries:
            h, p = _split_host_label(entry)
            if h != want_host:
                continue
            if not want_path or not p or p == want_path:
                out.append(tool)
                break
    return sorted(out)


def check_hierarchy(tool_names: Optional[Iterable[str]] = None) -> List[str]:
    """Return one violation string per tool whose declared inner total exceeds
    its guard minus the headroom — i.e. is not strictly below the guard with
    GUARD_HEADROOM_SECONDS to spare (empty list = rule holds). A tool that is
    driven by the dynamic deadline declares exactly the ceiling."""
    names = list(tool_names) if tool_names is not None else sorted(INNER_BUDGETS)
    violations: List[str] = []
    for name in names:
        getter = INNER_BUDGETS.get(name)
        if getter is None:
            continue
        guard = guard_seconds(name)
        if guard is None:
            continue  # guard disabled by env: nothing to be below
        if name in DEADLINE_ONLY_TOOLS:
            continue  # no static constant exists; the runtime clamp is the bound
        parts = getter()
        ceiling = guard - GUARD_HEADROOM_SECONDS
        detail = " + ".join(f"{k}={v:g}s" for k, v in parts.items())
        if name in LOOP_TOOLS:
            worst = max(parts.values()) if parts else 0.0
            if worst > ceiling:
                violations.append(
                    f"{name}: a single inner call of {worst:g}s ({detail}) exceeds guard {guard:g}s - "
                    f"headroom {GUARD_HEADROOM_SECONDS:g}s = {ceiling:g}s"
                )
            continue
        total = float(sum(parts.values()))
        if total > ceiling:
            violations.append(
                f"{name}: inner {total:g}s ({detail}) exceeds guard {guard:g}s - headroom "
                f"{GUARD_HEADROOM_SECONDS:g}s = {ceiling:g}s"
            )
    return violations

# ── declarations ─────────────────────────────────────────────────────────
# Hosts listed here drive the host circuit breaker's pre-check in
# core/agent.py (a tool whose EVERY host is open is refused before it starts)
# and the sibling list in the structured error. Only the archive the tool
# fundamentally needs is listed — not the optional name resolver.

ALMA_HOSTS = ("almascience.nrao.edu", "almascience.eso.org", "almascience.nao.ac.jp")
# TAP tools are declared on the /tap SERVICE circuit (guard CX-10): the guard
# precheck then sees a soft-open TAP circuit (and auto-waits a short
# cooldown) instead of only host-wide hard failures. DataLink / download /
# QA2 tools use other paths and stay on the bare hosts.
ALMA_TAP_HOSTS = tuple(f"{h}/tap" for h in ALMA_HOSTS)
MAST_HOSTS = ("mast.stsci.edu",)
CDS_IMAGING_HOSTS = ("alasky.cds.unistra.fr",)
CDS_MOC_HOSTS = ("alasky.unistra.fr",)
SIMBAD_HOSTS = ("simbad.cds.unistra.fr",)
# Data Lab services on one host, declared per SERVICE PATH so the host breaker's
# soft (read-timeout / 502) circuits stay independent: an SIA cutout burst must
# never skip the SQL tools (UI benchmark 2026-09-22, L07 -> L08/L09/L10).
DATALAB_SQL_HOSTS = ("datalab.noirlab.edu/query",)
DATALAB_SIA_HOSTS = ("datalab.noirlab.edu/sia",)
DATALAB_HOSTS = DATALAB_SQL_HOSTS
SPARCL_HOSTS = ("astrosparcl.datalab.noirlab.edu",)
CDS_XMATCH_HOSTS = ("cdsxmatch.u-strasbg.fr",)
SPLATALOGUE_HOSTS = ("splatalogue.online",)
ALERCE_HOSTS = ("api.alerce.online",)
VIZIER_HOSTS = ("vizier.cds.unistra.fr",)
TAVILY_HOSTS = ("api.tavily.com",)
# web_search routes across Brave, Exa and Tavily (services/web_search_service):
# declaring all three means an open Tavily circuit no longer refuses the whole
# tool while Brave or Exa could still answer -- the guard refuses a tool only
# when EVERY declared host is open (D12).
WEB_SEARCH_HOSTS = ("api.search.brave.com", "api.exa.ai") + TAVILY_HOSTS
# Every ADS API call is under /v1 -> one service circuit (guard CX-10).
ADS_HOSTS = ("api.adsabs.harvard.edu/v1",)
OPENALEX_HOSTS = ("api.openalex.org",)
CADC_HOSTS = ("ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca",)
ESO_HOSTS = ("archive.eso.org",)
IRSA_HOSTS = ("irsa.ipac.caltech.edu",)
NED_HOSTS = ("ned.ipac.caltech.edu",)
GAIA_HOSTS = ("gea.esac.esa.int",)
SKYVIEW_HOSTS = ("skyview.gsfc.nasa.gov",)
JPL_HOSTS = ("ssd.jpl.nasa.gov",)
SKYBOT_HOSTS = ("ssp.imcce.fr",)
GW_HOSTS = ("gwosc.org", "gcn.gsfc.nasa.gov")
SVO_HOSTS = ("svo2.cab.inta-csic.es",)
ATNF_HOSTS = ("www.atnf.csiro.au",)
HF_HOSTS = ("huggingface.co",)
NRAO_TAP_HOSTS = ("data-query.nrao.edu",)
# Archive catalogue tools (capabilities/catalogs.py, 2026-09).
HEASARC_HOSTS = ("heasarc.gsfc.nasa.gov",)
EXOPLANET_HOSTS = ("exoplanetarchive.ipac.caltech.edu",)
TAPVIZIER_HOSTS = ("tapvizier.cds.unistra.fr",)
TNS_HOSTS = ("www.wis-tns.org",)

# Tools whose transport is NOT requests-based and has no timeout of its own:
# not even the requests-layer clamp applies; the outer guard is the only
# bound. Kept explicit so the hierarchy test can insist every network tool is
# accounted for.
UNBOUNDED_TRANSPORT_TOOLS: Dict[str, str] = {
    "search_mmu_hats_catalog": "lsdb/HATS parquet reads over fsspec (aiohttp), not requests",
    "crossmatch_mmu_hats_catalogs": "lsdb/HATS parquet reads over fsspec (aiohttp), not requests",
    "list_mmu_hats_catalogs": "local dict; lsdb import only",
}


def _env_f(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else float(default)
    except ValueError:
        return float(default)


def _alma_tap_seconds() -> float:
    from integrations.alma_tap import AlmaTapService

    return float(AlmaTapService.worst_case_seconds())


def _alma_tap() -> Dict[str, float]:
    return {"ALMA TAP (2 mirrors x attempt timeout + backoff)": _alma_tap_seconds()}


def _alma_source_search() -> Dict[str, float]:
    from integrations.alminer_client import ALminerClient

    return {
        "ALMA TAP source search": min(_alma_tap_seconds(), float(ALminerClient.SOURCE_TIMEOUT_S)),
        "ALminer fallback": float(ALminerClient.ALMINER_TIMEOUT_S),
    }


def _simbad() -> Dict[str, float]:
    from integrations.simbad_resolver import SIMBAD_TIMEOUT_S

    return {"SIMBAD resolve": float(SIMBAD_TIMEOUT_S)}


def _mast_query() -> Dict[str, float]:
    from integrations.mast_client import MAST_QUERY_TIMEOUT_S

    return {"MAST query": float(MAST_QUERY_TIMEOUT_S)}


def _mast_download() -> Dict[str, float]:
    from integrations.mast_client import MASTClient

    return {**_mast_query(), "MAST download wall (env-capped)": float(MASTClient.DOWNLOAD_WALL_MAX_SECONDS)}


def _hips(n: int) -> Callable[[], Dict[str, float]]:
    return lambda: {f"hips2fits x{n}": _env_f("HIPS2FITS_TIMEOUT", 30.0) * n}


def _hips_one() -> Dict[str, float]:
    return {"hips2fits (one layer)": _env_f("HIPS2FITS_TIMEOUT", 30.0)}


def _fits_download(n: int) -> Callable[[], Dict[str, float]]:
    def get() -> Dict[str, float]:
        from services import fits_service

        return {f"archive FITS download x{n}": float(fits_service.DOWNLOAD_TIMEOUT_S) * n}

    return get


def _fits_one() -> Dict[str, float]:
    from services import fits_service

    return {"archive FITS download (one file)": float(fits_service.DOWNLOAD_TIMEOUT_S)}


def _datalab_sql(n: int = 1) -> Callable[[], Dict[str, float]]:
    return lambda: {f"Data Lab SQL sync wall x{n}": _env_f("DATALAB_TIMEOUT_SECONDS", 60.0) * 1.5 * n}


def _datalab_sia() -> Dict[str, float]:
    return {"Data Lab SIA x2 endpoints": _env_f("DATALAB_SIA_TIMEOUT_SECONDS", 30.0) * 2}


def _datalab_tile() -> Dict[str, float]:
    # services/datalab_image_service.py: wall = 1.5 x DATALAB_IMAGE_DOWNLOAD_TIMEOUT_SECONDS
    return {"Data Lab tile download wall": _env_f("DATALAB_IMAGE_DOWNLOAD_TIMEOUT_SECONDS", 45.0) * 1.5}


def _splatalogue() -> Dict[str, float]:
    import inspect

    from services import splatalogue as _sp

    sig = inspect.signature(_sp.SplatalogueClient.__init__)
    d = {k: v.default for k, v in sig.parameters.items()}
    # One attempt end to end; per-request retries (SPLATALOGUE_MAX_RETRIES)
    # are clamped by the requests-layer hook.
    return {
        "Splatalogue submit": float(d["submit_timeout"]),
        "Splatalogue poll deadline": float(d["timeout_seconds"]),
        "Splatalogue poll request": float(d["poll_timeout"]),
        "Splatalogue non-threaded fallback": float(d["non_threaded_timeout"]),
    }


def _sparcl(n: int) -> Callable[[], Dict[str, float]]:
    return lambda: {f"SPARCL read x{n}": _env_f("SPARCL_READ_TIMEOUT_SECONDS", 60.0) * n}


def _cadc_tap() -> Dict[str, float]:
    return {"CADC TAP": _env_f("CADC_TAP_TIMEOUT", 40.0)}


def _ads() -> Dict[str, float]:
    import inspect

    from integrations.ads_client import ADSService

    d = {k: v.default for k, v in inspect.signature(ADSService.__init__).parameters.items()}
    attempts = int(d.get("retry_attempts", 2))
    return {f"ADS ({attempts} attempts + backoff)": float(d.get("timeout", 10.0)) * attempts + max(0, attempts - 1)}


def _openalex() -> Dict[str, float]:
    import inspect

    from integrations.openalex_client import OpenAlexService as OpenAlexClient

    d = {k: v.default for k, v in inspect.signature(OpenAlexClient.__init__).parameters.items()}
    attempts = int(d.get("retry_attempts", 2))
    return {f"OpenAlex ({attempts} attempts + backoff)": float(d.get("timeout", 12.0)) * attempts + max(0, attempts - 1)}


def _ads_llm() -> Dict[str, float]:
    return {"ADS query-builder LLM": _env_f("ADS_QUERY_BUILDER_TIMEOUT_SECONDS", 30.0)}


def _papers_llm() -> Dict[str, float]:
    return {"literature LLM pass (bounded)": _env_f("PAPERS_LLM_TIMEOUT_SECONDS", 60.0)}


def _web_search() -> Dict[str, float]:
    return {"Exa": 30.0, "Brave x2": 10.0, "Tavily": 60.0, "browser fallback x2": 30.0}


def _sle_chain() -> Dict[str, float]:
    # services/spectral_line_explorer.py: joint resolver future, species cache,
    # Splatalogue (one attempt), one ALMA coverage TAP endpoint (of 3).
    return {
        "SLE resolver future": 30.0,
        "SLE species cache": 20.0,
        **_splatalogue(),
        "SLE ALMA coverage TAP (one endpoint)": 40.0,
    }


def _ceiling(tool: str) -> Callable[[], Dict[str, float]]:
    return lambda: {"deadline-driven (no static constant; bounded by the tool's inner deadline)": float(inner_ceiling_seconds(tool) or 0.0)}


def _fixed(label: str, seconds: float) -> Callable[[], Dict[str, float]]:
    return lambda: {label: float(seconds)}


def _merge(*getters: Callable[[], Dict[str, float]]) -> Callable[[], Dict[str, float]]:
    def get() -> Dict[str, float]:
        out: Dict[str, float] = {}
        for g in getters:
            for k, v in g().items():
                key = k
                i = 2
                while key in out:
                    key = f"{k} ({i})"
                    i += 1
                out[key] = v
        return out

    return get


def _declare_defaults() -> None:
    LOOP = "loops over an input-sized list; call count bounded by the tool deadline"
    PHASES = "several sequential archive phases; each call clamped to the remaining tool budget"
    # ── ALMA ──────────────────────────────────────────────────────────
    declare(["search_by_target"], _merge(_simbad, _alma_source_search), ALMA_TAP_HOSTS)
    declare(["search_by_position"], _alma_source_search, ALMA_TAP_HOSTS)
    declare(
        ["search_by_frequency", "get_observation_details", "advanced_search", "search_alma_co_in_redshift_range"],
        _alma_tap, ALMA_TAP_HOSTS,
    )
    declare_loop(
        ["query_alma_science_archive"], _merge(_simbad, _alma_tap), ALMA_TAP_HOSTS,
        reason="server-side aggregates (services/alma_server_side.py): one bounded TAP query per cycle / per line, "
               "count bounded by the tool deadline; a TOP-capped fallback pull otherwise",
    )
    declare(["list_alma_files"], _fixed("ALMA DataLink (astroquery 60 + 2 x 30 HTTP)", 120.0), ALMA_HOSTS)
    declare(["get_alma_qa2_status"], _fixed("QA2 status fetch", 6.0), ALMA_HOSTS)
    declare_loop(
        ["match_cross_archive_sources", "match_perseus_protostars_alma_jwst"],
        _merge(_alma_tap, _mast_query), ALMA_TAP_HOSTS + MAST_HOSTS,
        reason="ALMA bulk query (or one point cone per source) + one MAST query per source; concurrent phases under the deadline",
    )
    declare_loop(["triage_alma_data_products"], _merge(_simbad, _alma_source_search, _fixed("DataLink per MOUS", 120.0), _fixed("FITS header Range GET", 30.0)),
                 ALMA_TAP_HOSTS, reason=PHASES)
    declare_loop(["download_alma_data"], _fixed("DataLink preflight per MOUS", 120.0), ALMA_HOSTS,
                 reason="DataLink preflight per MOUS, then alminer download (pyvo/requests, clamped by hook)")
    declare_deadline_only(["search_alma_with_keywords", "search_catalog"], ALMA_TAP_HOSTS,
                          reason="alminer keysearch/catalog: pyvo TAPService built without a session — no static timeout")
    declare_loop(["find_alma_line_coverage"], _sle_chain, SPLATALOGUE_HOSTS + ALMA_TAP_HOSTS, reason=PHASES)
    declare_deadline_only(["search_cadc_archive"], CADC_HOSTS,
                          reason="pyvo TAPService built without a session — no static timeout (the 477 s CADC incident)")
    # ── MAST / other archives ─────────────────────────────────────────
    declare(["search_mast"], _merge(_simbad, _mast_query), MAST_HOSTS)
    declare(["search_mast_by_criteria", "get_mast_products"], _mast_query, MAST_HOSTS)
    declare(["download_mast_data"], _mast_download, MAST_HOSTS)
    declare(["search_eso_archive"], _merge(_simbad, _fixed("ESO TAP", 30.0)), ESO_HOSTS)
    declare_deadline_only(["search_irsa"], IRSA_HOSTS,
                          reason="astroquery IRSA builds its TAPService on a bare requests.Session; conf.timeout never applied")
    declare_loop(["cross_match_source"], _merge(_simbad, _fixed("NED", 60.0), _mast_query, _fixed("VizieR", 30.0)),
                 SIMBAD_HOSTS + NED_HOSTS + MAST_HOSTS + VIZIER_HOSTS,
                 reason="5-7 archive legs in a thread pool; each leg clamped by the hook")
    declare_loop(["overlay_archive_images"], _merge(_mast_query, _alma_tap, _fixed("DataLink per MOUS", 120.0), _fits_one),
                 MAST_HOSTS + ALMA_TAP_HOSTS, reason=PHASES)
    declare(["inspect_fits_header"], _fixed("remote FITS header Range GET x2", 60.0))
    # ── Data Lab ──────────────────────────────────────────────────────
    declare(
        ["datalab_describe_table", "datalab_cone_count", "datalab_select_catalog_rows", "datalab_q3c_crossmatch",
         "datalab_sql_query", "datalab_star_lightcurve", "datalab_color_color_diagram",
         "datalab_tiled_search"],
        _datalab_sql(1), DATALAB_HOSTS,
    )
    # The CMD plus its population-feature test (one Hess aggregate, DLB-03).
    declare_loop(["datalab_color_magnitude_diagram"], _datalab_sql(1), DATALAB_HOSTS,
                 reason="the CMD SQL, then one population-test aggregate only when >= 20 s of budget remain (<= 30 s)")
    declare(["datalab_list_catalogs"], _merge(_datalab_sql(1), _simbad), DATALAB_HOSTS)
    declare(["datalab_sia_search"], _datalab_sia, DATALAB_SIA_HOSTS)
    declare_loop(["datalab_variable_candidates", "datalab_job_results", "datalab_density_aggregate"], _datalab_sql(1),
                 DATALAB_HOSTS, reason="two or more SQL round-trips (probe/tiles); each clamped")
    declare_loop(["datalab_image_cutout", "datalab_color_image", "datalab_cutout_grid"], _merge(_datalab_sia, _datalab_tile),
                 DATALAB_SIA_HOSTS, reason="one SIA search + one tile download per band/peak/candidate")
    declare_loop(["datalab_density_vetting"], _merge(_datalab_sql(1), _datalab_sia, _datalab_tile),
                 DATALAB_SQL_HOSTS + DATALAB_SIA_HOSTS,
                 reason="density SQL then a cutout grid (one SIA + tile per peak)")
    declare(["xmatch_user_list"], _fixed("CDS xmatch POST", 90.0), CDS_XMATCH_HOSTS)
    # ── CDS imaging / MOC ─────────────────────────────────────────────
    declare(["hips_cutout", "vlass_cutout"], _hips(1), CDS_IMAGING_HOSTS)
    declare(["hips_contour_overlay", "image_difference"], _hips(2), CDS_IMAGING_HOSTS)
    declare(["hips_rgb_composite"], _hips(3), CDS_IMAGING_HOSTS)
    declare_loop(["hips_multiband_panel", "hips_aperture_photometry"], _hips_one, CDS_IMAGING_HOSTS, reason=LOOP)
    declare(["survey_coverage", "survey_covers_position"], _fixed("MOCServer", 30.0), CDS_MOC_HOSTS)
    declare_loop(["survey_footprint", "moc_operations"], _merge(_fixed("MOCServer per id/order", 30.0), _hips_one), CDS_MOC_HOSTS,
                 reason="one MOCServer fetch per id and order (downsampling loop)")
    declare_loop(["vlass_epoch_comparison"], _merge(_cadc_tap, _cadc_tap, _fits_one), CADC_HOSTS,
                 reason="two CADC TAP queries then one SODA cutout per epoch candidate")
    # ── generic FITS / image analysis (archive URL) ───────────────────
    declare(
        ["render_fits_image", "compute_moment_map", "extract_spectrum", "fit_spectral_line", "pv_slice",
         "image_statistics", "detect_sources", "measure_region", "fit_gaussian_source", "radial_profile"],
        _fits_download(1),
    )
    declare(["overlay_fits_images"], _fits_download(2))
    # ── spectral lines / spectra ──────────────────────────────────────
    declare(["identify_spectral_line", "search_lines_by_molecule", "search_spectral_lines"], _splatalogue, SPLATALOGUE_HOSTS)
    declare(["sparcl_find_spectra", "sparcl_search_spectra"], _sparcl(2), SPARCL_HOSTS)
    declare(["sparcl_get_spectrum", "sparcl_plot_spectrum"], _sparcl(1), SPARCL_HOSTS)
    declare_loop(["sparcl_stack_spectra"], _sparcl(1), SPARCL_HOSTS, reason="one find per bin + one retrieve per batch")
    declare_loop(["svo_filter_wavelength"], _fixed("SVO FPS (astroquery conf) per query shape", 60.0), SVO_HOSTS,
                 reason="up to 3 query shapes per filter, uncapped filter list")
    # ── time domain / misc services ───────────────────────────────────
    declare(["search_ztf_alerts"], _merge(_simbad, _fixed("ALeRCE", 30.0)), ALERCE_HOSTS)
    declare(["ztf_light_curve"], _fixed("ALeRCE", 30.0), ALERCE_HOSTS)
    declare(["ztf_stamps"], _fixed("ALeRCE light curve + 3 stamps", 120.0), ALERCE_HOSTS)
    declare_loop(["monitor_check_now"], _fixed("ALeRCE per watchlist target", 30.0), ALERCE_HOSTS, reason=LOOP)
    declare_loop(["radio_sed"], _fixed("VizieR per survey", 30.0), VIZIER_HOSTS, reason="5 survey queries")
    declare(["galactic_extinction"], _merge(_simbad, _fixed("IRSA dust", 30.0)), IRSA_HOSTS)
    declare(["gaia_distance"], _merge(_simbad, _fixed("Gaia TAP", 60.0)), GAIA_HOSTS)
    declare(["ned_distance", "ned_sed_plot"], _fixed("NED", 60.0), NED_HOSTS)
    declare(["velocity_frame_distance"], _simbad, SIMBAD_HOSTS)
    declare(["resolve_target"], _merge(_fixed("Sesame x2 (astropy remote_timeout)", 20.0), _simbad), SIMBAD_HOSTS)
    declare_deadline_only(["search_space_lightcurves", "plot_space_lightcurve", "period_search"], MAST_HOSTS + ALERCE_HOSTS,
                          reason="lightkurve/astroquery MAST search + download carry no HTTP timeout of their own")
    declare_deadline_only(["get_sky_image", "generate_finding_chart"], SKYVIEW_HOSTS,
                          reason="astroquery SkyView has no timeout configuration item")
    declare_deadline_only(["search_pulsars", "pulsar_lookup"], ATNF_HOSTS, reason="psrqpy QueryATNF has no timeout")
    declare(["solar_system_ephemeris"], _fixed("JPL Horizons", 30.0), JPL_HOSTS)
    declare(["moving_object_check"], _fixed("SkyBoT", 30.0), SKYBOT_HOSTS)
    declare(["get_latest_gw_events", "search_gwtc_catalog", "summarize_gcn_circular"], _fixed("GWOSC/GCN GET", 10.0), GW_HOSTS)
    # ── VO ────────────────────────────────────────────────────────────
    declare_deadline_only(["vo_find_services"], ("reg.g-vo.org",), reason="pyvo.registry.search() uses its own sessionless client")
    declare(["vo_list_tables"], _fixed("VO TAP x2 phases", 90.0))
    declare(["vo_describe_table"], _fixed("VO TAP x3 phases", 135.0))
    declare(["vo_cone_search"], _fixed("VO service", 45.0))
    # mode=auto: a deadline-bounded sync attempt, then (budget permitting)
    # submit POST + job GET + run POST; every request is clamped to the
    # remaining tool budget by the VO session.
    declare_loop(["vo_adql_query"], lambda: {"sync attempt (mode=auto, VO_AUTO_SYNC_TIMEOUT)": 20.0,
                                             "each VO TAP request": 45.0}, reason=PHASES)
    # status = job GET (+ optional UWS WAIT <= 20s); results = job GET + result
    # download (<= VO_ASYNC_MAXREC_CAP rows); abort = job GET + phase POST.
    declare_loop(["vo_tap_job"], _fixed("each VO UWS request", 45.0), reason=PHASES)
    # SIA2 capabilities probe + query (+ SIA1 retry on a failed probe).
    declare_loop(["vo_image_search"], _fixed("each VO SIA request", 45.0), reason=PHASES)
    # ── web ───────────────────────────────────────────────────────────
    declare(["web_search"], _web_search, WEB_SEARCH_HOSTS)
    declare(["web_extract_url"], _fixed("Tavily extract", 60.0), TAVILY_HOSTS)
    declare(["web_research_status"], _fixed("Tavily research status", 30.0), TAVILY_HOSTS)
    declare_deadline_only(["web_map_site", "web_crawl_site", "web_research"], TAVILY_HOSTS,
                          reason="Tavily SDK call defaults (150 s / None) are not configurable per call; clamped by hook")
    declare(["navigate_to_url"], _fixed("Playwright goto", 20.0))
    declare(["read_page", "click_element"], _fixed("Playwright action", 5.0))
    # ── literature ────────────────────────────────────────────────────
    declare(["search_papers"], _merge(_ads_llm, _ads, _openalex), ADS_HOSTS + OPENALEX_HOSTS)
    declare_loop(["search_papers_by_observation_id"], _merge(_ads, _openalex, _alma_tap), ADS_HOSTS + ALMA_TAP_HOSTS,
                 reason="one ADS lookup per derived identifier, then OpenAlex and a reverse ALMA TAP query")
    declare(["evaluate_consensus"], _merge(_ads_llm, _ads, _papers_llm), ADS_HOSTS)
    declare(["extract_paper_details", "reproduce_paper_methods"], _merge(_ads, _fixed("PDF download", 30.0), _papers_llm), ADS_HOSTS)
    declare(["lookup_researcher"], _openalex, OPENALEX_HOSTS)
    declare(["get_research_trends"], _merge(_openalex, _openalex), OPENALEX_HOSTS)
    # ADS author / metrics / export / library tools (integrations/ads_client.py
    # backends, 2026-09-21): one bounded ADS request each; two for the tools
    # that resolve bibcodes before the metrics / listing call.
    declare(
        ["get_author_papers", "get_paper_metrics", "get_paper_abstract", "export_bibtex", "list_ads_libraries",
         "create_ads_library", "add_to_ads_library"],
        _ads, ADS_HOSTS,
    )
    declare(["get_author_metrics", "get_ads_library_papers"], _merge(_ads, _ads), ADS_HOSTS)
    # ── One-shot tools (UI benchmark 2026-09-22, WP3/WP4) ──────────────
    declare_loop(["alma_project_census"], _merge(_simbad, _alma_tap), ALMA_TAP_HOSTS,
                 reason="server-side aggregates: one bounded TAP query per cycle / per line, count bounded by the tool deadline")
    declare_loop(["alma_source_summary"], _merge(_simbad, _alma_tap, _alma_tap), ALMA_TAP_HOSTS,
                 reason="cone query, then a COUNT(*) companion only when the cap hit AND >= 8 s of budget remain")
    declare(["alma_public_band_status"], _alma_tap, ALMA_TAP_HOSTS)
    declare(["alma_archive_link"], _merge(_simbad, _fixed("HEAD probes x2", 12.0)), ALMA_HOSTS)
    declare(["alma_bibliography"], _ads, ADS_HOSTS)
    declare_loop(["cross_archive_match"], _merge(_alma_tap, _mast_query), ALMA_TAP_HOSTS + MAST_HOSTS,
                 reason="one ALMA point cone + one MAST query per source; concurrent phases under the deadline")
    declare_loop(["archive_overlay"], _merge(_mast_query, _alma_tap, _fixed("DataLink per MOUS", 120.0), _fits_one),
                 MAST_HOSTS + ALMA_TAP_HOSTS + CDS_IMAGING_HOSTS, reason=PHASES)
    declare_deadline_only(["alma_reference"], reason="documentation vector store (Qdrant) + embeddings; clamped by the hook")
    declare_loop(["datalab_healpix_density_map"], _datalab_sql(1), DATALAB_HOSTS,
                 reason="tiled HEALPix aggregate: one SQL round-trip per tile, tile count bounded by the tool deadline")
    declare_loop(["datalab_stream_selection"], _datalab_sql(1), DATALAB_HOSTS,
                 reason="core proper-motion probe then the cross-match; each SQL clamped")
    declare(["datalab_selection_diagram"], _datalab_sql(1), DATALAB_HOSTS)
    declare_loop(["datalab_target_class_summary"], _datalab_sql(1), DATALAB_HOSTS,
                 reason="two server-side aggregates (n(z), footprint); each clamped")
    declare_loop(["datalab_sed_sample"], _merge(_datalab_sql(1), _fixed("SVO FPS per filter (cached)", 30.0)),
                 DATALAB_HOSTS + SVO_HOSTS,
                 reason="one bounded LS DR9 cone select, then SVO effective wavelengths for the five filters (cached)")
    declare_loop(["datalab_satellite_search"], _merge(_datalab_sql(1), _datalab_sia, _datalab_tile),
                 DATALAB_SQL_HOSTS + DATALAB_SIA_HOSTS,
                 reason="tiled density scan, then one SIA + tile per peak and one CMD SQL per candidate; all clamped")
    # ── Archive catalogue tools (capabilities/catalogs.py, 2026-09) ────
    # integrations/archive_tap.py sizes every TAP POST with bounded_timeout
    # (default 45 s), so each call is clamped to the remaining tool budget.
    tap = _fixed("TAP query (ArchiveTapClient default, clamped)", 45.0)
    declare_loop(["heasarc_observations"], _merge(_simbad, tap), HEASARC_HOSTS,
                 reason="COUNT/SUM, row and public-count TAP queries in sequence; each clamped")
    declare_loop(["exoplanet_archive"], tap, EXOPLANET_HOSTS,
                 reason="TAP_SCHEMA lookups (cached) then 2 to 4 TAP queries; each clamped")
    declare_loop(["simbad_query"], _merge(_simbad, tap), SIMBAD_HOSTS,
                 reason="one TAP query per identifier (<= 25) or position (<= 50); each clamped")
    declare_loop(["gaia_archive_query"], _merge(_simbad, tap), GAIA_HOSTS + SIMBAD_HOSTS,
                 reason="SIMBAD cross-id queries, then Gaia source / cone / variability queries; each clamped")
    declare_loop(["catalog_find"], _merge(_fixed("VizieR keyword search (astroquery, clamped by hook)", 60.0), tap),
                 VIZIER_HOSTS + TAPVIZIER_HOSTS + HEASARC_HOSTS + IRSA_HOSTS,
                 reason="one keyword search or TAP_SCHEMA query per service")
    declare_loop(["catalog_query", "catalog_crossmatch"], _merge(_simbad, tap),
                 TAPVIZIER_HOSTS + IRSA_HOSTS + HEASARC_HOSTS + GAIA_HOSTS + SIMBAD_HOSTS + EXOPLANET_HOSTS,
                 reason="TAP_SCHEMA lookup, COUNT and row pull per catalogue (two catalogues for a cross-match)")
    declare(["tns_object"], _fixed("TNS object page GET", 30.0), TNS_HOSTS)
    declare(["ztf_object"], _merge(_simbad, _fixed("ALeRCE cone + object + probabilities", 80.0)), ALERCE_HOSTS)
    declare(["ads_search"], _ads, ADS_HOSTS)


_declare_defaults()
