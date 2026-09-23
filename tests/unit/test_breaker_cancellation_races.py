"""Concurrency of the process-shared registries (guard CX-34).

HostBreaker (one lock over _open/_failures/_soft) and TurnCancellation (one
lock over the live set) are hit from many worker threads at once in
production. These tests hammer them concurrently and check the invariants the
sequential tests assume: one soft failure per CALL however many threads of
that call record it; exactly one trip for N distinct calls; no exception or
torn state under interleaved success/failure; every deadline -- live or
registered while/after the turn is cancelled -- ends up cancelled, and none
stays counted as live.
"""
from __future__ import annotations

import random
import threading

import pytest

import services.host_breaker as hb
from services import tool_budgets as tb
from services.host_breaker import HostBreaker


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in ("HOST_BREAKER_DISABLED", "HOST_BREAKER_SOFT_TRIPS", "HOST_BREAKER_SOFT_SPREAD_SECONDS",
                "HOST_BREAKER_SOFT_WINDOW_SECONDS", "HOST_BREAKER_TRIP_FAILURES"):
        monkeypatch.delenv(var, raising=False)
    HostBreaker.reset()
    tb.end_tool_deadline()
    yield
    HostBreaker.reset()
    tb.end_tool_deadline()


def _run_threads(n, target):
    start = threading.Barrier(n)
    errors = []

    def wrapped(i):
        try:
            start.wait(5)
            target(i)
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=wrapped, args=(i,), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not errors, errors
    assert not any(t.is_alive() for t in threads)


URL = "https://archive.example/tap/sync?QUERY=x"


def test_one_call_recorded_from_many_threads_counts_once():
    call = tb.Deadline(60.0, label="one_call")

    def worker(i):
        tb.adopt_deadline(call.child(label=f"helper-{i}"))  # helper threads share the call token
        for _ in range(20):
            HostBreaker.record_failure(URL, status=504)
        tb.end_tool_deadline()

    _run_threads(16, worker)
    assert not HostBreaker.is_open(URL), "one call's soft failures never trip a circuit"
    assert len(HostBreaker._soft.get(hb.circuit_of(URL), [])) == 1


def test_distinct_calls_racing_trip_the_circuit_exactly_once(monkeypatch):
    clock = [1000.0]
    lock = threading.Lock()

    def fake_now():
        with lock:
            clock[0] += 7.0  # every record 7 s later: the spread requirement is met
            return clock[0]

    monkeypatch.setattr(hb, "_now", fake_now)
    opened = []
    real_open = HostBreaker._open_locked.__func__

    def spy(cls, key, **kw):
        opened.append(key)
        return real_open(cls, key, **kw)

    monkeypatch.setattr(HostBreaker, "_open_locked", classmethod(spy))

    def worker(i):
        tb.begin_tool_deadline(f"call-{i}", 60.0)
        HostBreaker.record_failure(URL, status=503)
        tb.end_tool_deadline()

    _run_threads(12, worker)
    assert opened == [hb.circuit_of(URL)], opened
    assert HostBreaker.is_open(URL)


def test_interleaved_success_and_failure_leave_consistent_state():
    urls = [f"https://h{i % 3}.example/svc{i % 2}/x" for i in range(6)]

    def worker(i):
        rnd = random.Random(i)
        for _ in range(200):
            u = rnd.choice(urls)
            if rnd.random() < 0.5:
                HostBreaker.record_failure(u, status=rnd.choice([502, 503, 504]))
            else:
                HostBreaker.record_success(u)
            HostBreaker.state(u)
            HostBreaker.open_hosts(urls)

    _run_threads(10, worker)
    snap = HostBreaker.snapshot()
    assert isinstance(snap, dict)
    for rec in snap.values():
        assert rec.get("retry_after", 0) >= 0 or "until" in rec


def test_cancel_racing_register_leaves_every_deadline_cancelled_and_none_live():
    for _round in range(20):
        turn = tb.TurnCancellation(f"race-{_round}")
        made = []
        made_lock = threading.Lock()
        fire_at = random.randint(0, 15)

        def worker(i):
            d = tb.make_tool_deadline(f"tool-{i}", 30.0, turn=turn)
            with made_lock:
                made.append(d)
            if i == fire_at:
                turn.cancel("stop pressed")

        _run_threads(16, worker)
        turn.cancel("stop pressed")  # idempotent
        assert all(d.cancelled() for d in made)
        assert all(d.why_cancelled() for d in made), "late registrations carry the turn's reason"
        for d in made:
            turn.release(d)
        assert turn.live_count() == 0


def test_register_after_cancel_is_born_cancelled_and_not_live():
    turn = tb.TurnCancellation("late")
    turn.cancel("client gone")
    d = tb.make_tool_deadline("late_tool", 30.0, turn=turn)
    assert d.cancelled() and d.why_cancelled() == "client gone"
    assert turn.live_count() == 0
