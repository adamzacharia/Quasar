"""Wall-clock bounding of `_download_fits` (guard CX-08 / CX-19).

A scalar requests timeout is connect + per-socket-read, and iter_content
buffers until a full chunk accumulates — so a trickling server used to make
the download wall-clock unbounded (2026-08-04 density hang). Two mechanisms
now bound it: the in-loop deadline check (chunks arriving, too slowly) and a
watchdog timer that force-closes the response (reads blocked so long the loop
body never runs). Both are driven here with fake responses — no sockets.
"""

import threading
import time

import pytest


def _svc():
    from services.datalab_image_service import DatalabImageService

    svc = DatalabImageService.__new__(DatalabImageService)
    svc.download_timeout = 5.0
    return svc


class _FakeResp:
    def __init__(self, iter_impl):
        self.headers = {}
        self.closed = threading.Event()
        self._iter_impl = iter_impl

    def raise_for_status(self):
        return None

    def close(self):
        self.closed.set()

    def iter_content(self, chunk_size):
        return self._iter_impl(self)


def test_deadline_trips_between_slow_chunks(monkeypatch):
    import requests

    monkeypatch.setenv("DATALAB_IMAGE_DOWNLOAD_WALL_SECONDS", "0.15")

    def slow_chunks(resp):
        while True:
            time.sleep(0.06)
            yield b"x" * 8

    monkeypatch.setattr(
        requests, "get", lambda url, timeout, stream: _FakeResp(slow_chunks)
    )
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="wall-clock"):
        _svc()._download_fits("http://fake.example/x.fits")
    assert time.monotonic() - start < 2.0


def test_watchdog_closes_a_blocked_read(monkeypatch):
    """The CX-08 case proper: the response never yields a chunk at all (a
    trickle below iter_content's buffer threshold blocks the read forever).
    The watchdog must force-close the connection at the deadline and the
    resulting error must surface as the TimeoutError contract."""
    import requests

    monkeypatch.setenv("DATALAB_IMAGE_DOWNLOAD_WALL_SECONDS", "0.2")

    def blocked_read(resp):
        # Block until the watchdog closes the response, then fail the way a
        # closed socket does — never yield a chunk.
        resp.closed.wait(5.0)
        raise OSError("connection closed")
        yield b""  # pragma: no cover - makes this a generator

    fake = _FakeResp(blocked_read)
    monkeypatch.setattr(requests, "get", lambda url, timeout, stream: fake)
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="watchdog closed the connection"):
        _svc()._download_fits("http://fake.example/y.fits")
    assert fake.closed.is_set(), "the watchdog must have closed the response"
    assert time.monotonic() - start < 2.0


def test_malformed_wall_env_falls_back(monkeypatch):
    """CX-09 regression: garbage/NaN/negative env values must not disable or
    crash enforcement — the derived default (download_timeout * 1.5) applies,
    which for this fast fake means the download simply succeeds."""
    import requests

    def one_chunk(resp):
        yield b"y" * 16

    for bad in ("not-a-number", "nan", "-5", "inf"):
        monkeypatch.setenv("DATALAB_IMAGE_DOWNLOAD_WALL_SECONDS", bad)
        monkeypatch.setattr(
            requests, "get", lambda url, timeout, stream: _FakeResp(one_chunk)
        )
        path = _svc()._download_fits("http://fake.example/z.fits")
        assert path


def test_header_phase_block_is_bounded(monkeypatch):
    """CX-08 round 2: a server that never returns response HEADERS (or
    trickles them within each read-idle window) cannot be watchdog-closed —
    no response object exists yet. The outer worker-join bound must trip."""
    import requests

    monkeypatch.setenv("DATALAB_IMAGE_DOWNLOAD_WALL_SECONDS", "0.2")

    def hanging_get(url, timeout, stream):
        time.sleep(10.0)
        raise AssertionError("unreachable in this test")

    monkeypatch.setattr(requests, "get", hanging_get)
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="header phase unresponsive|wall-clock"):
        _svc()._download_fits("http://fake.example/hang.fits")
    assert time.monotonic() - start < 5.0


def test_clean_eof_after_watchdog_close_discards_partial_file(monkeypatch):
    """CX-08 round 2: some transports end the iterator CLEANLY when the
    response is closed mid-stream — a partial FITS file must not be returned
    as if complete."""
    import requests

    monkeypatch.setenv("DATALAB_IMAGE_DOWNLOAD_WALL_SECONDS", "0.2")

    def partial_then_clean_eof(resp):
        yield b"x" * 32
        resp.closed.wait(5.0)  # watchdog closes at the deadline
        return  # clean StopIteration — no exception

    fake = _FakeResp(partial_then_clean_eof)
    monkeypatch.setattr(requests, "get", lambda url, timeout, stream: fake)
    with pytest.raises(TimeoutError, match="partial file discarded"):
        _svc()._download_fits("http://fake.example/partial.fits")
    assert fake.closed.is_set()


def test_download_slots_fail_fast_when_exhausted(monkeypatch):
    """Round-3 CX-31: abandoned download workers must not accumulate without
    bound — when every slot is held by an unresponsive transfer, the next
    download fails fast instead of stacking another thread."""
    import requests

    import services.datalab_image_service as dis

    monkeypatch.setattr(dis, "_DOWNLOAD_SLOTS", threading.BoundedSemaphore(1))
    dis._DOWNLOAD_SLOTS.acquire()  # simulate one held-forever worker
    try:
        with pytest.raises(TimeoutError, match="download worker slots"):
            _svc()._download_fits("http://fake.example/slotless.fits")
    finally:
        dis._DOWNLOAD_SLOTS.release()

    # A completed download must give its slot back.
    def one_chunk(resp):
        yield b"z" * 16

    monkeypatch.setenv("DATALAB_IMAGE_DOWNLOAD_WALL_SECONDS", "5")
    monkeypatch.setattr(requests, "get", lambda url, timeout, stream: _FakeResp(one_chunk))
    assert _svc()._download_fits("http://fake.example/ok.fits")
    assert dis._DOWNLOAD_SLOTS.acquire(blocking=False), "slot not released"
    dis._DOWNLOAD_SLOTS.release()


def test_failed_thread_start_returns_the_permit(monkeypatch):
    """Round-4 CX-34: a worker.start() failure must hand its semaphore permit
    back, or repeated failures permanently exhaust every slot."""
    import services.datalab_image_service as dis

    monkeypatch.setattr(dis, "_DOWNLOAD_SLOTS", threading.BoundedSemaphore(1))

    class _Unstartable(threading.Thread):
        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(dis.threading, "Thread", _Unstartable)
    with pytest.raises(RuntimeError, match="can't start new thread"):
        _svc()._download_fits("http://fake.example/nostart.fits")
    assert dis._DOWNLOAD_SLOTS.acquire(blocking=False), "permit leaked"
    dis._DOWNLOAD_SLOTS.release()
