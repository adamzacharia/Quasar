"""L9 (tmp/scan-2026-07-17): MAST product downloads must be timeboxed.

Period-fold/lightcurve turns stalled to the no-progress watchdog (2/2 live:
902 s) while ``download_mast_data`` blocked on TESS FITS pulls with no
timebox. Two layers now bound the transfer:

* the tool guard gives ``download_mast_data`` a 600 s budget
  (core/agent.py ``_TOOL_TIMEOUT_OVERRIDES``), and
* ``MASTClient.download_products`` wall-clock-bounds astroquery's own
  downloader (worker-join, ``MAST_DOWNLOAD_WALL_SECONDS``, default 480 s) so
  a trickling/hung transfer cannot outlive even that budget, returning a
  structured ``timeout: True`` result the runner closes the step on
  (core/runner.py ``_result_indicates_timeout``).
"""

import re
import threading
import time

import pandas as pd
import pytest

from core.agent import QuasarAgent
from integrations.mast_client import MASTClient


# ── tool-guard override ──────────────────────────────────────────────────


def test_download_mast_data_override(monkeypatch):
    """L9: bulk MAST FITS pulls legitimately run long — 600 s budget,
    matching download_alma_data."""
    monkeypatch.delenv("QUASAR_TOOL_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("QUASAR_TOOL_TIMEOUT_OVERRIDES", raising=False)
    assert QuasarAgent._tool_timeout_seconds("download_mast_data") == 600.0


# ── MAST_DOWNLOAD_WALL_SECONDS validation ────────────────────────────────


def test_wall_default_when_unset(monkeypatch):
    monkeypatch.delenv("MAST_DOWNLOAD_WALL_SECONDS", raising=False)
    client = MASTClient()
    assert client._download_wall_seconds() == 480.0
    assert MASTClient.DOWNLOAD_WALL_DEFAULT_SECONDS == 480.0


def test_wall_valid_env_honored(monkeypatch):
    monkeypatch.setenv("MAST_DOWNLOAD_WALL_SECONDS", "12.5")
    assert MASTClient()._download_wall_seconds() == 12.5


@pytest.mark.parametrize(
    "raw", ["", "0", "-30", "nan", "inf", "-inf", "not-a-number"]
)
def test_wall_invalid_env_falls_back(monkeypatch, raw):
    """Malformed/NaN/inf/negative values fall back to the default — never
    disable or crash enforcement (same contract as
    DATALAB_IMAGE_DOWNLOAD_WALL_SECONDS)."""
    monkeypatch.setenv("MAST_DOWNLOAD_WALL_SECONDS", raw)
    assert (
        MASTClient()._download_wall_seconds()
        == MASTClient.DOWNLOAD_WALL_DEFAULT_SECONDS
    )


def test_wall_over_guard_values_clamp_to_540(monkeypatch):
    """Guard CX-03: 601 s would let the generic 600 s tool-guard timeout fire
    before the MAST-specific structured timeout — accepted values clamp to
    540 s (60 s headroom under the download_mast_data guard override)."""
    monkeypatch.setenv("MAST_DOWNLOAD_WALL_SECONDS", "601")
    assert MASTClient()._download_wall_seconds() == 540.0
    assert MASTClient.DOWNLOAD_WALL_MAX_SECONDS == 540.0
    # The ceiling itself is accepted unchanged.
    monkeypatch.setenv("MAST_DOWNLOAD_WALL_SECONDS", "540")
    assert MASTClient()._download_wall_seconds() == 540.0


# ── worker-join bound ────────────────────────────────────────────────────


def test_worker_join_bound_trips_on_hung_downloader(monkeypatch):
    """A downloader that never returns must be abandoned at the wall budget
    with an honest TimeoutError — not waited out."""
    monkeypatch.setenv("MAST_DOWNLOAD_WALL_SECONDS", "0.2")
    client = MASTClient()
    release = threading.Event()
    started = threading.Event()

    def hung_downloader(prod_table, download_dir):
        started.set()
        release.wait(10.0)
        return None

    start = time.monotonic()
    with pytest.raises(TimeoutError, match="wall-clock deadline"):
        client._download_products_bounded(
            None, "unused-dir", downloader=hung_downloader
        )
    elapsed = time.monotonic() - start
    release.set()

    assert started.wait(1.0), "the worker must actually have started"
    assert elapsed < 3.0, "the bound must reclaim control at the wall budget"


def test_bounded_downloader_returns_manifest(monkeypatch):
    monkeypatch.setenv("MAST_DOWNLOAD_WALL_SECONDS", "5")
    client = MASTClient()
    sentinel = object()
    result = client._download_products_bounded(
        "table", "dir", downloader=lambda table, ddir: sentinel
    )
    assert result is sentinel


def test_bounded_downloader_reraises_worker_exception(monkeypatch):
    monkeypatch.setenv("MAST_DOWNLOAD_WALL_SECONDS", "5")
    client = MASTClient()

    def boom(prod_table, download_dir):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        client._download_products_bounded(None, "dir", downloader=boom)


# ── abandoned-worker concurrency cap (guard CX-01) ───────────────────────


def test_download_slots_fail_fast_when_exhausted(monkeypatch):
    """Guard CX-01: abandoned MAST workers must not accumulate without bound
    — when every slot is held by an unresponsive transfer, the next download
    fails fast with a structured-timeout-shaped error instead of stacking
    another thread."""
    import integrations.mast_client as mc

    monkeypatch.setenv("MAST_DOWNLOAD_WALL_SECONDS", "5")
    monkeypatch.setattr(mc, "_DOWNLOAD_SLOTS", threading.BoundedSemaphore(1))
    mc._DOWNLOAD_SLOTS.acquire()  # simulate one held-forever worker
    client = MASTClient()
    start = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="download worker slots"):
            client._download_products_bounded(
                None, "unused-dir", downloader=lambda table, ddir: None
            )
    finally:
        mc._DOWNLOAD_SLOTS.release()
    assert time.monotonic() - start < 1.0, "exhaustion must fail fast"


def test_download_slot_released_after_completion(monkeypatch):
    """A completed download must give its slot back."""
    import integrations.mast_client as mc

    monkeypatch.setenv("MAST_DOWNLOAD_WALL_SECONDS", "5")
    monkeypatch.setattr(mc, "_DOWNLOAD_SLOTS", threading.BoundedSemaphore(1))
    client = MASTClient()
    sentinel = object()
    result = client._download_products_bounded(
        "table", "dir", downloader=lambda table, ddir: sentinel
    )
    assert result is sentinel
    assert mc._DOWNLOAD_SLOTS.acquire(blocking=False), "slot not released"
    mc._DOWNLOAD_SLOTS.release()


def test_failed_thread_start_returns_the_permit(monkeypatch):
    """Guard CX-01 (mirrors datalab CX-34): a worker.start() failure must
    hand its semaphore permit back, or repeated failures permanently exhaust
    every slot."""
    import integrations.mast_client as mc

    monkeypatch.setenv("MAST_DOWNLOAD_WALL_SECONDS", "5")
    monkeypatch.setattr(mc, "_DOWNLOAD_SLOTS", threading.BoundedSemaphore(1))

    class _Unstartable(threading.Thread):
        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(mc.threading, "Thread", _Unstartable)
    with pytest.raises(RuntimeError, match="can't start new thread"):
        MASTClient()._download_products_bounded(
            None, "dir", downloader=lambda table, ddir: None
        )
    assert mc._DOWNLOAD_SLOTS.acquire(blocking=False), "permit leaked"
    mc._DOWNLOAD_SLOTS.release()


def test_slot_timeout_flows_into_structured_timeout_dict(monkeypatch, tmp_path):
    """Slot exhaustion surfaces to the tool as the same structured
    timeout: True dict the wall-clock expiry produces — the runner closes
    the step on it and the model is told not to retry immediately."""
    import integrations.mast_client as mc

    monkeypatch.setattr(mc, "MAST_AVAILABLE", True)
    monkeypatch.setattr(mc, "_DOWNLOAD_SLOTS", threading.BoundedSemaphore(1))
    mc._DOWNLOAD_SLOTS.acquire()  # all slots held
    try:
        client = MASTClient()
        products = pd.DataFrame(
            [{"obsID": "1", "productFilename": "a.fits",
              "productType": "SCIENCE"}]
        )
        result = client.download_products(
            products=products, download_dir=str(tmp_path)
        )
    finally:
        mc._DOWNLOAD_SLOTS.release()

    assert result["success"] is False
    assert result["timeout"] is True
    assert "download worker slots" in result["error"]
    assert "do NOT retry immediately" in result["error"]

    from core.runner import _result_indicates_timeout

    assert _result_indicates_timeout(result)


# ── per-call download directory isolation (guard CX-02) ──────────────────


def test_download_products_uses_unique_per_call_directory(monkeypatch, tmp_path):
    """Guard CX-02: an abandoned timed-out worker keeps writing into ITS
    call's directory — two sequential calls must land in distinct
    uuid-fragment subdirectories under the requested base dir, and the
    returned download_dir must be the per-call one."""
    import os

    import integrations.mast_client as mc

    monkeypatch.setattr(mc, "MAST_AVAILABLE", True)

    seen_dirs = []

    class _Manifest:
        def to_pandas(self):
            return pd.DataFrame(
                [{"Status": "COMPLETE", "Local Path": "a.fits"}]
            )

    def _capture(self, prod_table, download_dir, downloader=None):
        seen_dirs.append(download_dir)
        return _Manifest()

    monkeypatch.setattr(MASTClient, "_download_products_bounded", _capture)

    client = MASTClient()
    products = pd.DataFrame(
        [{"obsID": "1", "productFilename": "a.fits", "productType": "SCIENCE"}]
    )
    first = client.download_products(
        products=products, download_dir=str(tmp_path)
    )
    second = client.download_products(
        products=products, download_dir=str(tmp_path)
    )

    assert first["success"] is True and second["success"] is True
    assert len(seen_dirs) == 2
    assert seen_dirs[0] != seen_dirs[1], "each call must get its own subdir"
    for ddir, result in zip(seen_dirs, (first, second)):
        assert os.path.dirname(ddir) == str(tmp_path)
        # Round 2: the FULL 128-bit uuid4 hex, not a truncated fragment.
        assert re.fullmatch(r"dl-[0-9a-f]{32}", os.path.basename(ddir))
        assert result["download_dir"] == ddir
        assert os.path.isdir(ddir)


def test_no_data_early_returns_create_no_directory(monkeypatch, tmp_path):
    """Round 2 (CX-02): the per-call directory is created only once there is
    actually data to transfer — no-data early returns must not leak empty
    dl-* directories under the base dir."""
    import integrations.mast_client as mc

    monkeypatch.setattr(mc, "MAST_AVAILABLE", True)
    client = MASTClient()

    # (a) nothing to download at all
    result = client.download_products(
        products=pd.DataFrame(), observations=None,
        download_dir=str(tmp_path),
    )
    assert result["success"] is False
    assert "No data to download" in result["error"]

    # (b) the query resolves to zero matching products
    monkeypatch.setattr(
        MASTClient, "get_product_list",
        lambda self, observations, productType=None, extension=None:
            pd.DataFrame(),
    )
    observations = pd.DataFrame([{"obsid": "1"}])
    result = client.download_products(
        products=None, observations=observations,
        download_dir=str(tmp_path),
    )
    assert result["success"] is False
    assert "No matching data products" in result["error"]

    assert list(tmp_path.iterdir()) == [], "no per-call dir may be created"


def test_call_dir_collision_retries_once_with_fresh_uuid(monkeypatch, tmp_path):
    """Round 2 (CX-02): exist_ok=False means a colliding name is never
    silently reused — one retry with a fresh uuid, then a hard error."""
    import os

    import integrations.mast_client as mc

    class _FixedUuid:
        def __init__(self, hex_value):
            self.hex = hex_value

    seq = iter(["a" * 32, "b" * 32])

    class _FakeUuidModule:
        @staticmethod
        def uuid4():
            return _FixedUuid(next(seq))

    monkeypatch.setattr(mc, "uuid", _FakeUuidModule)
    (tmp_path / ("dl-" + "a" * 32)).mkdir()  # force the first candidate to collide

    call_dir = MASTClient._make_call_dir(str(tmp_path))
    assert os.path.basename(call_dir) == "dl-" + "b" * 32
    assert os.path.isdir(call_dir)

    # Both attempts colliding is a hard error, never silent reuse.
    seq = iter(["a" * 32, "b" * 32])
    with pytest.raises(FileExistsError, match="after 2 attempts"):
        MASTClient._make_call_dir(str(tmp_path))


# ── structured timeout result ────────────────────────────────────────────


def test_download_products_timeout_returns_structured_dict(monkeypatch, tmp_path):
    """On timeout the tool result must carry timeout: True and anti-retry
    text — the runner keys step closure and SSE deadline exclusion on it."""
    import integrations.mast_client as mc

    monkeypatch.setattr(mc, "MAST_AVAILABLE", True)

    def _timed_out(self, prod_table, download_dir, downloader=None):
        raise TimeoutError(
            "MAST product download exceeded its 480s wall-clock deadline "
            "while the transfer was still running (hung or trickling; "
            "worker abandoned)."
        )

    monkeypatch.setattr(MASTClient, "_download_products_bounded", _timed_out)

    client = MASTClient()
    products = pd.DataFrame(
        [{"obsID": "1", "productFilename": "a.fits", "productType": "SCIENCE"}]
    )
    result = client.download_products(
        products=products, download_dir=str(tmp_path)
    )

    assert result["success"] is False
    assert result["timeout"] is True
    assert "Do NOT retry this exact call" in result["error"]
    assert "wall-clock deadline" in result["error"]

    # The runner's central classification must treat this as a timeout.
    from core.runner import _result_indicates_timeout

    assert _result_indicates_timeout(result)


def test_download_products_generic_error_is_not_timeout(monkeypatch, tmp_path):
    """A non-timeout failure must keep the legacy shape — no timeout key, so
    the runner never mistakes an ordinary error for a wall-clock expiry."""
    import integrations.mast_client as mc

    monkeypatch.setattr(mc, "MAST_AVAILABLE", True)

    def _broke(self, prod_table, download_dir, downloader=None):
        raise RuntimeError("archive said no")

    monkeypatch.setattr(MASTClient, "_download_products_bounded", _broke)

    client = MASTClient()
    products = pd.DataFrame(
        [{"obsID": "1", "productFilename": "a.fits", "productType": "SCIENCE"}]
    )
    result = client.download_products(
        products=products, download_dir=str(tmp_path)
    )

    assert result["success"] is False
    assert "timeout" not in result

    from core.runner import _result_indicates_timeout

    assert not _result_indicates_timeout(result)
