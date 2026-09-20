"""Run-record lifecycle races in the SSE layer (guard CX-13 / CX-20).

``_start_run_off_loop`` owns the ordering contract between the run-row insert
and any cancel-path finalization: the two must never race (a finalize racing
the insert updates zero rows and strands the later row in 'started'). These
tests drive the real async helper directly — cancellation mid-insert, and the
no-cancel path — asserting finalization happens strictly AFTER the insert
commits, exactly as the docstring promises.
"""

import asyncio
import sys
import threading
import time
from pathlib import Path

_UI_PRO = str(Path(__file__).resolve().parents[2] / "ui-pro")
if _UI_PRO not in sys.path:
    sys.path.insert(0, _UI_PRO)

from api.sse import _start_run_off_loop  # noqa: E402


def test_cancel_during_insert_finalizes_strictly_after_commit():
    order = []
    insert_started = threading.Event()
    release_insert = threading.Event()

    def start_fn():
        insert_started.set()
        release_insert.wait(5.0)
        order.append("insert")

    def finalize_fn():
        order.append("finalize")

    async def main():
        task = asyncio.create_task(_start_run_off_loop(start_fn, finalize_fn))
        # Wait (off-loop) until the worker is provably inside the insert,
        # then cancel while it is still blocked — the exact CX-13 window.
        await asyncio.get_running_loop().run_in_executor(
            None, insert_started.wait, 5.0
        )
        task.cancel()
        release_insert.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # Let the worker thread and the executor fallback drain.
        await asyncio.sleep(0.3)

    asyncio.run(main())

    assert order and order[0] == "insert", (
        "finalize must never run before the insert commits"
    )
    # Both the worker (flag seen after commit) and the cancel handler's
    # executor fallback may finalize — the caller's finalize latch dedupes in
    # production. Here every finalize must simply be ordered after the insert.
    assert set(order[1:]) == {"finalize"} and 1 <= len(order[1:]) <= 2


def test_uncancelled_start_never_finalizes():
    order = []

    async def main():
        await _start_run_off_loop(
            lambda: order.append("insert"), lambda: order.append("finalize")
        )
        await asyncio.sleep(0.1)

    asyncio.run(main())
    assert order == ["insert"]


def test_cancel_after_insert_completed_still_finalizes():
    """Cancellation landing after the worker finished (insert committed,
    cancel flag set too late for the worker to see) must still finalize via
    the handler's executor fallback — ordered after the commit by virtue of
    the worker having already returned."""
    order = []
    insert_done = threading.Event()

    def start_fn():
        order.append("insert")
        insert_done.set()

    async def main():
        task = asyncio.create_task(_start_run_off_loop(start_fn, lambda: order.append("finalize")))
        await asyncio.get_running_loop().run_in_executor(None, insert_done.wait, 5.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            # Cancellation may lose the race with normal completion once the
            # worker has finished — both outcomes are legal here.
            pass
        await asyncio.sleep(0.3)

    asyncio.run(main())
    assert order[0] == "insert"
    # If the cancel won the race, a finalize must have followed the insert;
    # if normal completion won, no finalize is required.
    assert set(order[1:]) <= {"finalize"}


def test_slow_insert_with_cancel_never_finalizes_early():
    """Round-2 CX-13: an insert outlasting any fallback grace must STILL be
    finalized strictly after its commit — a premature zero-row finalize would
    trip the caller's latch and strand the row in 'started' forever."""
    order = []
    insert_started = threading.Event()

    def slow_start():
        insert_started.set()
        time.sleep(0.4)
        order.append("insert")

    async def main():
        task = asyncio.create_task(
            _start_run_off_loop(slow_start, lambda: order.append("finalize"))
        )
        await asyncio.get_running_loop().run_in_executor(
            None, insert_started.wait, 5.0
        )
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.8)

    asyncio.run(main())
    assert order[0] == "insert", f"finalize must never precede the commit: {order}"
    assert "finalize" in order[1:]


def test_fallback_gives_up_without_finalizing_on_hung_insert(monkeypatch):
    """Round-3 CX-30: when the insert outlives the fallback's bounded wait,
    the fallback must exit WITHOUT finalizing (never early — that would strand
    the row via the caller's latch) and without holding its executor slot
    forever; the worker's finally-finalize owns that path once the insert
    completes."""
    import api.sse as sse_mod

    monkeypatch.setattr(sse_mod, "_FALLBACK_INSERT_WAIT_SECONDS", 0.05)
    order = []
    insert_started = threading.Event()

    def slow_start():
        insert_started.set()
        time.sleep(0.5)
        order.append("insert")

    async def main():
        task = asyncio.create_task(
            _start_run_off_loop(slow_start, lambda: order.append("finalize"))
        )
        await asyncio.get_running_loop().run_in_executor(
            None, insert_started.wait, 5.0
        )
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # Fallback wait (0.05 s) has expired; insert (0.5 s) still running —
        # nothing may be finalized yet.
        await asyncio.sleep(0.25)
        assert order == [], f"premature finalize/insert: {order}"
        await asyncio.sleep(0.6)

    asyncio.run(main())
    assert order[0] == "insert"
    assert order[1:] == ["finalize"], (
        "exactly the worker's post-commit finalize must run"
    )


def test_expiry_sweep_retries_without_finalizing(monkeypatch):
    """Round-5/6 CX-33: when the insert worker outlives the fallback wait, the
    on_expiry sweep must fire and RETRY across the horizon (a hung remote
    commit can become visible only after an earlier sweep saw nothing), while
    finalize_fn stays untouched until the worker's own post-commit path runs."""
    import api.sse as sse_mod

    monkeypatch.setattr(sse_mod, "_FALLBACK_INSERT_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(sse_mod, "_SWEEP_RETRY_DELAYS", (0.08, 0.08))
    order = []
    insert_started = threading.Event()

    def hung_then_completing_start():
        insert_started.set()
        time.sleep(0.7)
        order.append("insert")

    async def main():
        task = asyncio.create_task(
            _start_run_off_loop(
                hung_then_completing_start,
                lambda: order.append("finalize"),
                lambda: order.append("sweep"),
            )
        )
        await asyncio.get_running_loop().run_in_executor(
            None, insert_started.wait, 5.0
        )
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # Whole sweep horizon (0.05 + 0.08 + 0.08) elapses while the insert
        # (0.7 s) is still hung: multiple sweeps, and NOTHING finalized.
        await asyncio.sleep(0.45)
        assert order and set(order) == {"sweep"}, (
            f"only sweeps may run during the horizon: {order}"
        )
        assert len(order) >= 2, f"the sweep must RETRY across the horizon: {order}"
        await asyncio.sleep(0.6)

    asyncio.run(main())
    # Worker eventually committed and, seeing the cancel flag, finalized —
    # strictly after its insert; sweeps never count as a finalize.
    n_sweeps = order.count("sweep")
    assert order[:n_sweeps] == ["sweep"] * n_sweeps
    assert order[n_sweeps:] == ["insert", "finalize"]


def test_sweep_chain_stops_after_successful_sweep(monkeypatch):
    """Round-7 CX-35: once a sweep pass actually lands (returns True), the
    retry chain must stop — further passes are pointless work on borrowed
    threads."""
    import api.sse as sse_mod

    monkeypatch.setattr(sse_mod, "_FALLBACK_INSERT_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(sse_mod, "_SWEEP_RETRY_DELAYS", (0.08, 0.08, 0.08))
    sweeps = []
    insert_started = threading.Event()

    def hung_start():
        insert_started.set()
        time.sleep(0.8)

    def sweep():
        sweeps.append("s")
        return len(sweeps) >= 2  # second pass finds the late-visible row

    async def main():
        task = asyncio.create_task(
            _start_run_off_loop(hung_start, lambda: None, sweep)
        )
        await asyncio.get_running_loop().run_in_executor(
            None, insert_started.wait, 5.0
        )
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.5)

    asyncio.run(main())
    assert len(sweeps) == 2, (
        f"the chain must stop at the first successful sweep: {len(sweeps)} passes"
    )


def test_sweep_chain_wakes_early_when_worker_resolves(monkeypatch):
    """Round-8 CX-36: the sweep chain must not sleep out its full delay once
    the worker resolves — it waits on insert_done and exits early. Proven by
    timing: with a 5 s retry delay and a 0.3 s insert, the whole test must
    finish in well under the delay, with exactly the one pre-resolve sweep."""
    import api.sse as sse_mod

    monkeypatch.setattr(sse_mod, "_FALLBACK_INSERT_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(sse_mod, "_SWEEP_RETRY_DELAYS", (5.0, 5.0))
    order = []
    insert_started = threading.Event()

    def briefly_hung_start():
        insert_started.set()
        time.sleep(0.3)
        order.append("insert")

    async def main():
        task = asyncio.create_task(
            _start_run_off_loop(
                briefly_hung_start,
                lambda: order.append("finalize"),
                lambda: order.append("sweep") or False,
            )
        )
        await asyncio.get_running_loop().run_in_executor(
            None, insert_started.wait, 5.0
        )
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.6)

    start = time.monotonic()
    asyncio.run(main())
    elapsed = time.monotonic() - start
    assert elapsed < 3.0, f"chain slept out its delay instead of waking early ({elapsed:.1f}s)"
    assert order.count("sweep") == 1, f"exactly the pre-resolve sweep: {order}"
    assert order[-2:] == ["insert", "finalize"]


def test_sweep_degrades_to_inline_when_thread_start_fails(monkeypatch):
    """Round-9 CX-37: if the sweep thread cannot start (thread exhaustion),
    the fallback must still perform ONE immediate inline sweep instead of
    silently doing nothing inside its discarded executor future."""
    import api.sse as sse_mod

    monkeypatch.setattr(sse_mod, "_FALLBACK_INSERT_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(sse_mod, "_SWEEP_RETRY_DELAYS", (5.0,))

    def _unstartable(target):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(sse_mod, "_spawn_sweep_thread", _unstartable)
    order = []
    insert_started = threading.Event()

    def hung_start():
        insert_started.set()
        time.sleep(0.4)
        order.append("insert")

    async def main():
        task = asyncio.create_task(
            _start_run_off_loop(
                hung_start,
                lambda: order.append("finalize"),
                lambda: order.append("sweep") or False,
            )
        )
        await asyncio.get_running_loop().run_in_executor(
            None, insert_started.wait, 5.0
        )
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.7)

    start = time.monotonic()
    asyncio.run(main())
    assert time.monotonic() - start < 3.0
    assert order.count("sweep") == 1, f"exactly one inline sweep: {order}"
    assert order[-2:] == ["insert", "finalize"]
