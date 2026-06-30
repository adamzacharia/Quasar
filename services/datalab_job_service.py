"""Async job runner for long Data Lab catalog-science workflows (e.g. tiled scans).

Modeled on the Spectral Line Explorer job pattern (threaded, status-polled, cancelable),
but a separate service since SpectralLineJobService is operation-locked to its own kinds.
The workflow callable receives a ``cancel_check() -> bool`` it should poll between tiles.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional

# Terminal statuses a job can finish in.
_TERMINAL = {"succeeded", "failed", "canceled"}


class DatalabJobService:
    """In-process threaded job runner with a concurrency cap and cooperative cancellation."""

    def __init__(self, *, max_workers: Optional[int] = None):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        workers = int(max_workers if max_workers is not None else os.getenv("DATALAB_JOB_MAX_WORKERS", "2"))
        self._sem = threading.Semaphore(max(1, workers))

    def start(self, kind: str, fn: Callable[[Callable[[], bool]], Any], *, params: Optional[Dict[str, Any]] = None) -> str:
        """Run ``fn(cancel_check)`` on a background thread; return a job id to poll."""
        job_id = "dlj_" + uuid.uuid4().hex
        cancel_event = threading.Event()
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "kind": str(kind),
                "status": "queued",
                "params": dict(params or {}),
                "result": None,
                "error": None,
                "created_at": time.time(),
                "updated_at": time.time(),
                "_cancel": cancel_event,
            }

        def _run() -> None:
            with self._sem:
                if cancel_event.is_set():
                    self._update(job_id, status="canceled")
                    return
                self._update(job_id, status="running")
                try:
                    result = fn(cancel_event.is_set)
                    if cancel_event.is_set():
                        self._update(job_id, status="canceled", result=result)
                    else:
                        self._update(job_id, status="succeeded", result=result)
                except Exception as exc:  # noqa: BLE001 - surface as a failed job, never crash the worker
                    self._update(job_id, status="failed", error=str(exc))

        thread = threading.Thread(target=_run, name=f"datalab-job-{job_id}", daemon=True)
        with self._lock:
            self._jobs[job_id]["_thread"] = thread
        thread.start()
        return job_id

    def status(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            rec = self._jobs.get(str(job_id))
            if rec is None:
                raise KeyError(f"Unknown Data Lab job: {job_id}")
            return {k: v for k, v in rec.items() if not k.startswith("_")}

    def results(self, job_id: str) -> Dict[str, Any]:
        return self.status(job_id)

    def cancel(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            rec = self._jobs.get(str(job_id))
            if rec is None:
                raise KeyError(f"Unknown Data Lab job: {job_id}")
            rec["_cancel"].set()
            if rec["status"] == "queued":
                rec["status"] = "canceled"
                rec["updated_at"] = time.time()
        return self.status(job_id)

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is not None:
                rec.update(changes)
                rec["updated_at"] = time.time()


_DEFAULT_JOB_SERVICE: Optional[DatalabJobService] = None


def default_job_service() -> DatalabJobService:
    global _DEFAULT_JOB_SERVICE
    if _DEFAULT_JOB_SERVICE is None:
        _DEFAULT_JOB_SERVICE = DatalabJobService()
    return _DEFAULT_JOB_SERVICE


__all__ = ["DatalabJobService", "default_job_service"]
