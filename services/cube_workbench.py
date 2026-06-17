"""Server-side session layer for the first-class FITS cube workbench.

This module owns durable workbench session metadata. Heavy rendering is still
handled by the current `/api/fits/preview` path for now; this service creates
the boundary needed for a dedicated workbench route and interactive APIs.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import requests

from services.fits_processing import FITSProcessingService
from services.splatalogue import SplatalogueTool


DEFAULT_SESSION_DIR = Path(__file__).resolve().parent.parent / "data" / "workbench_sessions"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "workbench_cache"
DEFAULT_MAX_CACHE_BYTES = 500 * 1024 * 1024
DEFAULT_USER_CACHE_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60
DEFAULT_HEADER_RANGE_BYTES = 2 * 1024 * 1024
DEFAULT_PREVIEW_MAX_SPATIAL_PIXELS = 512
DEFAULT_PREVIEW_MAX_CHANNELS = 160
WORKBENCH_JOB_OPERATIONS = {"prepare", "render", "spectrum", "pv_slice", "line_overlays", "export"}
TERMINAL_JOB_STATUSES = {"succeeded", "failed", "canceled", "orphaned"}

COMMON_LINE_PRESETS = [
    {"key": "co_1_0", "label": "CO (1-0)", "rest_frequency_ghz": 115.271202, "family": "CO"},
    {"key": "co_2_1", "label": "CO (2-1)", "rest_frequency_ghz": 230.538000, "family": "CO"},
    {"key": "co_3_2", "label": "CO (3-2)", "rest_frequency_ghz": 345.795990, "family": "CO"},
    {"key": "13co_2_1", "label": "13CO (2-1)", "rest_frequency_ghz": 220.398684, "family": "CO isotopologue"},
    {"key": "c18o_2_1", "label": "C18O (2-1)", "rest_frequency_ghz": 219.560354, "family": "CO isotopologue"},
    {"key": "hco+_3_2", "label": "HCO+ (3-2)", "rest_frequency_ghz": 267.557633, "family": "Dense gas"},
    {"key": "hcn_3_2", "label": "HCN (3-2)", "rest_frequency_ghz": 265.886434, "family": "Dense gas"},
    {"key": "n2h+_3_2", "label": "N2H+ (3-2)", "rest_frequency_ghz": 279.511832, "family": "Dense gas"},
    {"key": "ci_1_0", "label": "[CI] (1-0)", "rest_frequency_ghz": 492.160651, "family": "Fine structure"},
    {"key": "cii_158um", "label": "[CII] 158 um", "rest_frequency_ghz": 1900.536900, "family": "Fine structure"},
]


class CubeWorkbenchError(Exception):
    """Base error for workbench session operations."""


class CubeWorkbenchNotFound(CubeWorkbenchError):
    """Raised when a workbench session cannot be found."""


class CubeWorkbenchForbidden(CubeWorkbenchError):
    """Raised when a session belongs to another user."""


class CubeWorkbenchJobCancelled(CubeWorkbenchError):
    """Raised when a queued/running workbench job is canceled."""


class CubeWorkbenchService:
    """Persistent metadata/session manager for FITS cube workbench workflows."""

    def __init__(
        self,
        session_dir: Path | str = DEFAULT_SESSION_DIR,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
    ):
        self.session_dir = Path(session_dir)
        self.cache_dir = Path(cache_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.splatalogue = SplatalogueTool()
        max_workers = max(1, int(os.getenv("QUASAR_WORKBENCH_WORKERS", "2") or "2"))
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="quasar-workbench")
        self._active_jobs: Dict[str, Optional[Future]] = {}
        self._job_lock = threading.RLock()

    def shutdown(self) -> None:
        """Stop in-process workbench workers for tests or controlled app shutdown."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    def create_session(
        self,
        *,
        user_id: str,
        source_url: str,
        filename: Optional[str] = None,
        project_code: Optional[str] = None,
        mous_uid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a durable workbench session and inspect headers if possible."""
        clean_url = str(source_url or "").strip()
        if not clean_url:
            raise ValueError("source_url is required")

        session_id = uuid.uuid4().hex
        now = int(time.time())
        parsed = urlparse(clean_url)
        inferred_filename = filename or Path(parsed.path).name or "alma_product.fits"
        archive = self._infer_archive(clean_url)

        metadata = self._metadata_for_source(clean_url)
        evidence = self._build_evidence(clean_url, archive, metadata)
        session = {
            "session_id": session_id,
            "user_id": user_id,
            "source_url": clean_url,
            "filename": inferred_filename,
            "archive": archive,
            "project_code": project_code or "",
            "mous_uid": mous_uid or "",
            "created_at": now,
            "last_accessed_at": now,
            "status": "ready" if metadata.get("success") else "metadata_failed",
            "cache_path": "",
            "cache": {
                "status": "not_prepared",
                "cached_bytes": 0,
                "max_bytes": DEFAULT_MAX_CACHE_BYTES,
                "user_cache_limit_bytes": DEFAULT_USER_CACHE_BYTES,
                "ttl_seconds": DEFAULT_CACHE_TTL_SECONDS,
                "prepared_at": None,
                "error": "",
                "preview": {
                    "status": "not_prepared",
                    "path": "",
                },
            },
            "metadata": metadata,
            "evidence": evidence,
            "jobs": [],
            "state": {
                "mode": "image",
                "channel": None,
                "moment": 0,
                "colormap": "inferno",
                "stretch": "asinh",
                "contour_sigma": [3, 5, 10],
                "rms_region": {
                    "x1": None,
                    "y1": None,
                    "x2": None,
                    "y2": None,
                },
                "redshift": 0.0,
                "spectrum": {
                    "x_pixel": None,
                    "y_pixel": None,
                    "aperture_radius_pixels": 3.0,
                    "max_points": 512,
                },
                "pv_slice": {
                    "path": [],
                    "width_pixels": 3.0,
                    "max_points": 512,
                },
            },
        }
        self._write_session(session)
        return session

    def get_session(self, *, session_id: str, user_id: str) -> Dict[str, Any]:
        session = self._read_session(session_id)
        if session.get("user_id") != user_id:
            raise CubeWorkbenchForbidden("Workbench session belongs to another user")
        session["last_accessed_at"] = int(time.time())
        self._write_session(session)
        return session

    def get_metadata(self, *, session_id: str, user_id: str) -> Dict[str, Any]:
        session = self.get_session(session_id=session_id, user_id=user_id)
        cache = dict(session.get("cache") or {})
        cache["user_cache"] = self._cache_summary_for_user(
            user_id=user_id,
            max_user_cache_bytes=self._safe_int(cache.get("user_cache_limit_bytes")) or DEFAULT_USER_CACHE_BYTES,
        )
        return {
            "session_id": session["session_id"],
            "filename": session.get("filename", ""),
            "source_url": session.get("source_url", ""),
            "archive": session.get("archive", ""),
            "project_code": session.get("project_code", ""),
            "mous_uid": session.get("mous_uid", ""),
            "status": session.get("status", ""),
            "metadata": session.get("metadata", {}),
            "state": session.get("state", {}),
            "cache": cache,
            "jobs": session.get("jobs", []),
            "evidence": session.get("evidence", {}),
        }

    @staticmethod
    def line_presets() -> Dict[str, Any]:
        return {
            "presets": COMMON_LINE_PRESETS,
            "evidence": {
                "source": "Quasar curated common radio/mm line list",
                "assumptions": [
                    "Preset frequencies are rest frequencies in GHz.",
                    "Observed frequency is computed with nu_obs = nu_rest / (1 + z).",
                ],
                "confidence": "medium",
            },
        }

    def start_job(
        self,
        *,
        session_id: str,
        user_id: str,
        operation: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Start an asynchronous workbench operation and persist durable progress."""
        clean_operation = str(operation or "").strip().lower()
        if clean_operation not in WORKBENCH_JOB_OPERATIONS:
            raise ValueError(f"Unsupported workbench job operation: {operation}")

        session = self.get_session(session_id=session_id, user_id=user_id)
        now = int(time.time())
        job = {
            "job_id": uuid.uuid4().hex,
            "operation": clean_operation,
            "status": "queued",
            "phase": "queued",
            "progress": 0,
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "cancel_requested": False,
            "request": dict(payload or {}),
            "result": None,
            "error": "",
            "metrics": {},
        }
        session["jobs"] = self._append_job(session.get("jobs"), job)
        self._write_session(session)

        with self._job_lock:
            self._active_jobs[job["job_id"]] = None
        future = self._executor.submit(
            self._run_job,
            session_id,
            user_id,
            job["job_id"],
            clean_operation,
            dict(payload or {}),
        )
        with self._job_lock:
            if job["job_id"] in self._active_jobs:
                self._active_jobs[job["job_id"]] = future
        return {
            "session_id": session_id,
            "job": job,
            "jobs": session["jobs"],
        }

    def get_job(self, *, session_id: str, user_id: str, job_id: str) -> Dict[str, Any]:
        session = self.get_session(session_id=session_id, user_id=user_id)
        job = self._find_job(session, job_id)
        if not job:
            raise CubeWorkbenchNotFound(f"Workbench job not found: {job_id}")
        with self._job_lock:
            active = job_id in self._active_jobs
        if job.get("status") in {"queued", "running"} and not active:
            started = self._safe_int(job.get("started_at")) or self._safe_int(job.get("created_at")) or 0
            if started and int(time.time()) - started > 300:
                job = self._update_job_state(
                    session_id=session_id,
                    user_id=user_id,
                    job_id=job_id,
                    status="orphaned",
                    phase="worker no longer active",
                    progress=100,
                    error="The server worker handling this job is no longer active. Start the operation again.",
                    finished=True,
                )
                session = self.get_session(session_id=session_id, user_id=user_id)
        return {
            "session_id": session_id,
            "job": job,
            "jobs": session.get("jobs", []),
        }

    def cancel_job(self, *, session_id: str, user_id: str, job_id: str) -> Dict[str, Any]:
        session = self.get_session(session_id=session_id, user_id=user_id)
        job = self._find_job(session, job_id)
        if not job:
            raise CubeWorkbenchNotFound(f"Workbench job not found: {job_id}")
        if job.get("status") in TERMINAL_JOB_STATUSES:
            return {
                "session_id": session_id,
                "job": job,
                "jobs": session.get("jobs", []),
            }

        canceled_now = False
        with self._job_lock:
            future = self._active_jobs.get(job_id)
            canceled_now = bool(future and future.cancel())

        job = self._update_job_state(
            session_id=session_id,
            user_id=user_id,
            job_id=job_id,
            status="canceled" if canceled_now else job.get("status", "running"),
            phase="canceled" if canceled_now else "cancel requested",
            progress=100 if canceled_now else job.get("progress", 0),
            cancel_requested=True,
            error="" if canceled_now else "Cancellation requested. The current step will stop at the next safe checkpoint.",
            finished=canceled_now,
        )
        session = self.get_session(session_id=session_id, user_id=user_id)
        return {
            "session_id": session_id,
            "job": job,
            "jobs": session.get("jobs", []),
        }

    def prepare_product(
        self,
        *,
        session_id: str,
        user_id: str,
        max_bytes: int = DEFAULT_MAX_CACHE_BYTES,
        user_cache_bytes: int = DEFAULT_USER_CACHE_BYTES,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        force: bool = False,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
        cancel_callback: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Stage a FITS product into the workbench cache within explicit limits."""
        session = self.get_session(session_id=session_id, user_id=user_id)
        cache = session.setdefault("cache", {})
        raw_existing_path = str(session.get("cache_path") or "")
        existing_path = Path(raw_existing_path) if raw_existing_path else None
        if existing_path and existing_path.exists() and not force:
            cached_bytes = existing_path.stat().st_size
            preview = self._ensure_preview_product(session, existing_path)
            cache.update({
                "status": "cached",
                "cached_bytes": cached_bytes,
                "max_bytes": int(max_bytes),
                "user_cache_limit_bytes": int(user_cache_bytes or DEFAULT_USER_CACHE_BYTES),
                "ttl_seconds": int(cache_ttl_seconds or DEFAULT_CACHE_TTL_SECONDS),
                "prepared_at": cache.get("prepared_at") or int(time.time()),
                "error": "",
                "preview": preview,
            })
            session["status"] = "cached"
            if preview.get("status") == "ready":
                evidence = session.setdefault("evidence", {})
                operations = list(evidence.get("operations") or [])
                if "preview_product" not in operations:
                    operations.append("preview_product")
                evidence["operations"] = operations
            self._write_session(session)
            if progress_callback:
                progress_callback(cached_bytes, cached_bytes)
            return self._cache_response(session, cache)

        source_url = str(session.get("source_url") or "").strip()
        if not source_url:
            raise ValueError("source_url is required")

        max_allowed = max(1, int(max_bytes or DEFAULT_MAX_CACHE_BYTES))
        user_limit = max(1, int(user_cache_bytes or DEFAULT_USER_CACHE_BYTES))
        cache_ttl = max(0, int(cache_ttl_seconds or DEFAULT_CACHE_TTL_SECONDS))
        target_path = self._cache_path_for(session)
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        cache.update({
            "status": "preparing",
            "cached_bytes": 0,
            "max_bytes": max_allowed,
            "user_cache_limit_bytes": user_limit,
            "ttl_seconds": cache_ttl,
            "prepared_at": None,
            "error": "",
        })
        self._write_session(session)
        if cancel_callback and cancel_callback():
            raise CubeWorkbenchJobCancelled("Workbench job was canceled before cache preparation started")

        try:
            cached_bytes = self._copy_or_download(
                source_url,
                tmp_path,
                max_allowed,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
            )
            if cancel_callback and cancel_callback():
                raise CubeWorkbenchJobCancelled("Workbench job was canceled before cache finalization")
            quota_result = self._enforce_cache_limits(
                user_id=user_id,
                protect_session_id=session_id,
                incoming_bytes=cached_bytes,
                max_user_cache_bytes=user_limit,
                ttl_seconds=cache_ttl,
            )
            tmp_path.replace(target_path)
            session["cache_path"] = str(target_path)
            session["status"] = "cached"
            preview = self._ensure_preview_product(session, target_path)
            cache.update({
                "status": "cached",
                "cached_bytes": cached_bytes,
                "max_bytes": max_allowed,
                "user_cache_limit_bytes": user_limit,
                "ttl_seconds": cache_ttl,
                "user_cache": quota_result.get("summary", {}),
                "last_evictions": quota_result.get("evicted", []),
                "prepared_at": int(time.time()),
                "error": "",
                "preview": preview,
            })
            evidence = session.setdefault("evidence", {})
            evidence["downloaded_bytes"] = cached_bytes
            operations = list(evidence.get("operations") or [])
            if "product_cache" not in operations:
                operations.append("product_cache")
            if preview.get("status") == "ready" and "preview_product" not in operations:
                operations.append("preview_product")
            if quota_result.get("evicted") and "cache_eviction" not in operations:
                operations.append("cache_eviction")
            evidence["operations"] = operations
            if preview.get("status") == "failed":
                warnings = list(evidence.get("warnings") or [])
                message = f"Preview product generation failed: {preview.get('error')}"
                if message not in warnings:
                    warnings.append(message)
                evidence["warnings"] = warnings
            if quota_result.get("evicted"):
                evidence["cache_evictions"] = quota_result.get("evicted")
            self._write_session(session)
            if progress_callback:
                progress_callback(cached_bytes, cached_bytes)
            return self._cache_response(session, cache)
        except CubeWorkbenchJobCancelled as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            session["status"] = "cache_canceled"
            cache.update({
                "status": "canceled",
                "error": str(exc),
                "prepared_at": int(time.time()),
            })
            self._write_session(session)
            raise
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            session["status"] = "cache_failed"
            cache.update({
                "status": "failed",
                "error": str(exc),
                "prepared_at": int(time.time()),
            })
            self._write_session(session)
            raise

    def render_plan(
        self,
        *,
        session_id: str,
        user_id: str,
        mode: str = "image",
        channel: Optional[int] = None,
        moment: Optional[int] = None,
        colormap: str = "inferno",
        stretch: str = "asinh",
        contour_sigma: Optional[List[float]] = None,
        rms_region: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Persist requested render state and render from cache when available."""
        session = self.get_session(session_id=session_id, user_id=user_id)
        state = session.setdefault("state", {})
        clean_rms_region = self._clean_region(rms_region) if rms_region is not None else state.get("rms_region")
        state.update({
            "mode": mode,
            "channel": channel,
            "moment": moment,
            "colormap": colormap,
            "stretch": stretch,
            "contour_sigma": contour_sigma or state.get("contour_sigma") or [3, 5, 10],
            "rms_region": clean_rms_region,
        })
        image = {
            "data_url": None,
            "data_status": "pending_cube_cache",
        }
        stats: Dict[str, Any] = {}
        contours: Dict[str, Any] = {
            "sigma": state["contour_sigma"],
            "levels": [],
        }
        status = "planned"
        next_phase = "Prepare the FITS product to compute WCS-aware render pixels from cached data."
        if self._session_cache_path(session).exists():
            image, stats, contours = self._render_from_cache(
                session,
                mode=str(mode or "image"),
                channel=channel,
                moment=moment,
                colormap=colormap,
                stretch=stretch,
                contour_sigma=state["contour_sigma"],
                rms_region=state.get("rms_region"),
            )
            status = "computed"
            next_phase = None
        self._write_session(session)
        return {
            "session_id": session_id,
            "status": status,
            "operation": "render",
            "state": state,
            "image": image,
            "stats": stats,
            "contours": contours,
            "next_phase": next_phase,
            "evidence": self._operation_evidence(
                session,
                "render",
                confidence="high" if status == "computed" else "medium",
                assumptions=[
                    "Render pixels are computed from cached FITS data when the product has been prepared.",
                    "WCS axes are drawn from the celestial part of the FITS WCS when available.",
                ],
            ),
        }

    def spectrum_plan(
        self,
        *,
        session_id: str,
        user_id: str,
        x_pixel: Optional[float] = None,
        y_pixel: Optional[float] = None,
        aperture_radius_pixels: float = 3.0,
        aperture_radius_arcsec: Optional[float] = None,
        max_points: int = 512,
    ) -> Dict[str, Any]:
        """Persist a spectrum extraction request and return spectral-axis metadata."""
        session = self.get_session(session_id=session_id, user_id=user_id)
        metadata = session.get("metadata") or {}
        headers = self._metadata_headers(metadata)
        axis = self._spectral_axis(headers, metadata, max_points=max_points)

        extraction = {
            "x_pixel": self._safe_float(x_pixel),
            "y_pixel": self._safe_float(y_pixel),
            "aperture_radius_pixels": max(0.5, float(aperture_radius_pixels or 3.0)),
            "aperture_radius_arcsec": self._safe_float(aperture_radius_arcsec),
            "method": "circular_aperture_mean",
        }
        session.setdefault("state", {})["spectrum"] = {
            **extraction,
            "max_points": max(16, min(int(max_points or 512), 4096)),
        }
        series = {
            "x": axis.get("values", []),
            "y": [],
            "x_label": axis.get("label", "Channel"),
            "y_label": str(headers.get("BUNIT") or metadata.get("unit") or "Intensity"),
            "data_status": "pending_cube_cache",
        }
        status = "planned"
        next_phase = "Populate y values from cached FITS data after cube cache/range-read support lands."
        if self._session_cache_path(session).exists():
            axis, series = self._extract_spectrum_from_cache(
                session,
                extraction=extraction,
                max_points=max_points,
            )
            status = "computed"
            next_phase = None
        self._write_session(session)

        return {
            "session_id": session_id,
            "status": status,
            "operation": "spectrum",
            "extraction": extraction,
            "spectral_axis": axis,
            "series": series,
            "next_phase": next_phase,
            "evidence": self._operation_evidence(
                session,
                "spectrum_plan",
                confidence="high" if status == "computed" else ("medium" if axis.get("values") else "low"),
                assumptions=[
                    "Spectral x-axis is derived from FITS WCS header cards when present.",
                    "Flux values are computed only after the FITS product is staged in the workbench cache.",
                ],
            ),
        }

    def pv_slice_plan(
        self,
        *,
        session_id: str,
        user_id: str,
        path: Optional[List[Dict[str, float]]] = None,
        width_pixels: float = 3.0,
        max_points: int = 512,
    ) -> Dict[str, Any]:
        """Persist a PV-slice path and return WCS-aware axis planning metadata."""
        session = self.get_session(session_id=session_id, user_id=user_id)
        metadata = session.get("metadata") or {}
        headers = self._metadata_headers(metadata)
        clean_path = self._clean_path(path or [])
        if len(clean_path) < 2:
            shape = self._shape_from_metadata(metadata, headers)
            nx = shape[-1] if len(shape) >= 1 else 100
            ny = shape[-2] if len(shape) >= 2 else 100
            clean_path = [
                {"x": round(nx * 0.25, 2), "y": round(ny * 0.5, 2)},
                {"x": round(nx * 0.75, 2), "y": round(ny * 0.5, 2)},
            ]

        width = max(0.5, float(width_pixels or 3.0))
        spatial_samples = self._estimate_path_samples(clean_path, max_points=max_points)
        spectral_axis = self._spectral_axis(headers, metadata, max_points=max_points)
        session.setdefault("state", {})["pv_slice"] = {
            "path": clean_path,
            "width_pixels": width,
            "max_points": max(16, min(int(max_points or 512), 4096)),
        }
        image = {
            "data_url": None,
            "data_status": "pending_cube_cache",
        }
        status = "planned"
        next_phase = "Populate PV image pixels from cached FITS data after cube cache/range-read support lands."
        if self._session_cache_path(session).exists():
            spectral_axis, image = self._extract_pv_from_cache(
                session,
                path=clean_path,
                width_pixels=width,
                max_points=max_points,
            )
            status = "computed"
            next_phase = None
        self._write_session(session)

        return {
            "session_id": session_id,
            "status": status,
            "operation": "pv_slice",
            "path": clean_path,
            "width_pixels": width,
            "spatial_axis": {
                "label": "Offset (pixel)",
                "unit": "pixel",
                "samples": spatial_samples,
            },
            "spectral_axis": spectral_axis,
            "image": image,
            "next_phase": next_phase,
            "evidence": self._operation_evidence(
                session,
                "pv_slice_plan",
                confidence="high" if status == "computed" else ("medium" if spectral_axis.get("values") else "low"),
                assumptions=[
                    "PV path is stored in pixel coordinates until celestial WCS interaction is implemented.",
                    "PV image values are computed only after the FITS product is staged in the workbench cache.",
                ],
            ),
        }

    def line_overlays(
        self,
        *,
        session_id: str,
        user_id: str,
        observed_frequency_ghz: Optional[float] = None,
        line_preset_key: Optional[str] = None,
        redshift: float = 0.0,
        tolerance_ghz: float = 0.01,
        top_n: int = 8,
    ) -> Dict[str, Any]:
        """Return redshift-aware Splatalogue line candidates for the workbench."""
        session = self.get_session(session_id=session_id, user_id=user_id)
        metadata = session.get("metadata") or {}
        preset = self._line_preset(line_preset_key)
        z = float(redshift or 0.0)
        if preset:
            rest_frequency = float(preset["rest_frequency_ghz"])
            observed = rest_frequency / (1.0 + z)
        else:
            freq = observed_frequency_ghz or metadata.get("rest_freq_ghz")
            if freq is None:
                raise ValueError("observed_frequency_ghz is required when FITS metadata has no rest frequency")
            observed = float(freq)
            rest_frequency = observed * (1.0 + z)
        result = self.splatalogue.identify_spectral_line(
            frequency_ghz=rest_frequency,
            tolerance_ghz=float(tolerance_ghz or 0.01),
            top_n=max(1, min(int(top_n or 8), 25)),
        )
        lines = []
        for line in result.get("lines", []):
            rest = self._safe_float(line.get("frequency_ghz"))
            lines.append({
                **line,
                "rest_frequency_ghz": rest,
                "observed_frequency_ghz": round(rest / (1.0 + z), 9) if rest else None,
                "redshift": z,
            })
        if preset and not any(abs((self._safe_float(line.get("rest_frequency_ghz")) or 0.0) - rest_frequency) <= float(tolerance_ghz or 0.01) for line in lines):
            lines.insert(0, {
                "species": preset["label"],
                "transition": preset["label"],
                "family": preset.get("family", ""),
                "frequency_ghz": rest_frequency,
                "rest_frequency_ghz": rest_frequency,
                "observed_frequency_ghz": round(observed, 9),
                "redshift": z,
                "source": "Quasar preset",
            })

        line_state = {
            "query_observed_frequency_ghz": observed,
            "query_rest_frequency_ghz": rest_frequency,
            "redshift": z,
            "tolerance_ghz": float(tolerance_ghz or 0.01),
            "preset": preset,
            "lines": lines,
            "backend": result.get("backend"),
            "query_note": result.get("note"),
            "query_error": result.get("error"),
        }
        state = session.setdefault("state", {})
        state["redshift"] = z
        state["line_overlays"] = line_state
        self._write_session(session)
        return {
            "session_id": session_id,
            "query_observed_frequency_ghz": observed,
            "query_rest_frequency_ghz": rest_frequency,
            "redshift": z,
            "tolerance_ghz": tolerance_ghz,
            "preset": preset,
            "presets": COMMON_LINE_PRESETS,
            "n_matches": len(lines),
            "lines": lines,
            "backend": result.get("backend"),
            "query_note": result.get("note"),
            "query_error": result.get("error"),
            "evidence": {
                "source": (
                    "Splatalogue via IVOA SLAP"
                    if result.get("backend") == "slap"
                    else (
                        "Splatalogue unavailable"
                        if result.get("backend") == "unavailable"
                        else "Splatalogue via Astroquery"
                    )
                ),
                "assumptions": [
                    "Line search converts observed frequency to rest frequency using nu_rest = nu_obs * (1 + z).",
                    "Candidates are ranked by absolute frequency offset and duplicate catalog entries are merged.",
                ],
                "confidence": "medium" if lines else "low",
            },
        }

    def export_plan(
        self,
        *,
        session_id: str,
        user_id: str,
        formats: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Return reproducible export commands/scripts for the current session."""
        session = self.get_session(session_id=session_id, user_id=user_id)
        requested = {str(fmt).lower() for fmt in (formats or ["casa", "carta", "ds9", "python"])}
        filename = session.get("filename") or "alma_product.fits"
        source_url = session.get("source_url") or ""
        stem = Path(filename).stem
        state = session.get("state") or {}
        spectrum_state = state.get("spectrum") if isinstance(state.get("spectrum"), dict) else {}
        pv_state = state.get("pv_slice") if isinstance(state.get("pv_slice"), dict) else {}
        exports: Dict[str, Any] = {}

        if "casa" in requested:
            exports["casa"] = {
                "filename": "workbench_casa_reduction.py",
                "content": (
                    "from casatasks import importfits, imhead, imstat, immoments, specfit\n\n"
                    f"fitsimage = '{filename}'\n"
                    f"imagename = '{stem}.image'\n"
                    f"moment0 = '{stem}.moment0'\n"
                    f"moment1 = '{stem}.moment1'\n\n"
                    "importfits(fitsimage=fitsimage, imagename=imagename, overwrite=True)\n"
                    "print(imhead(imagename=imagename, mode='summary'))\n"
                    "print(imstat(imagename=imagename))\n"
                    "immoments(imagename=imagename, moments=[0], outfile=moment0)\n"
                    "immoments(imagename=imagename, moments=[1], outfile=moment1)\n\n"
                    "# Spectrum extraction seed from the Quasar workbench.\n"
                    f"# aperture_x_pixel = {spectrum_state.get('x_pixel')}\n"
                    f"# aperture_y_pixel = {spectrum_state.get('y_pixel')}\n"
                    f"# aperture_radius_pixels = {spectrum_state.get('aperture_radius_pixels')}\n"
                    "# Use CASA viewer/region tools to convert this into a region, then run specfit.\n"
                    "# specfit(imagename=imagename, box='', chans='', ngauss=1, multifit=False)\n"
                ),
            }
        if "carta" in requested:
            exports["carta"] = {
                "filename": "open_in_carta.txt",
                "content": (
                    f"Open {filename} in CARTA.\n"
                    f"Source URL: {source_url}\n"
                    f"Colormap: {state.get('colormap', 'inferno')}\n"
                    f"Stretch: {state.get('stretch', 'asinh')}\n"
                    f"Contour sigma: {state.get('contour_sigma', [3, 5, 10])}\n"
                    f"Spectrum aperture: {spectrum_state}\n"
                    f"PV slice: {pv_state}\n"
                ),
            }
        if "ds9" in requested:
            exports["ds9"] = {
                "filename": "open_in_ds9.sh",
                "content": f"ds9 '{filename}' -scale zscale -cmap inferno",
            }
            path = pv_state.get("path") if isinstance(pv_state, dict) else []
            region_lines = ["# Region file format: DS9 version 4.1", "image"]
            if isinstance(path, list) and len(path) >= 2:
                start, end = path[0], path[-1]
                if isinstance(start, dict) and isinstance(end, dict):
                    region_lines.append(
                        "line({x1},{y1},{x2},{y2}) # line=0 0 color=cyan width=2".format(
                            x1=start.get("x", 0),
                            y1=start.get("y", 0),
                            x2=end.get("x", 0),
                            y2=end.get("y", 0),
                        )
                    )
            if spectrum_state.get("x_pixel") is not None and spectrum_state.get("y_pixel") is not None:
                region_lines.append(
                    "circle({x},{y},{r}) # color=green width=2".format(
                        x=spectrum_state.get("x_pixel"),
                        y=spectrum_state.get("y_pixel"),
                        r=spectrum_state.get("aperture_radius_pixels", 3),
                    )
                )
            exports["ds9_region"] = {
                "filename": "workbench_regions.reg",
                "content": "\n".join(region_lines) + "\n",
            }
        if "python" in requested or "notebook" in requested:
            exports["python"] = {
                "filename": "workbench_reproduce.py",
                "content": (
                    "from astropy.io import fits\n"
                    "import numpy as np\n"
                    "import matplotlib.pyplot as plt\n\n"
                    f"url = '{source_url}'\n"
                    f"filename = '{filename}'\n"
                    "with fits.open(filename, memmap=True) as hdul:\n"
                    "    hdu = next(h for h in hdul if getattr(h, 'data', None) is not None)\n"
                    "    data = hdu.data.squeeze()\n"
                    "    image = data.sum(axis=0) if data.ndim == 3 else data\n"
                    "    plt.imshow(image, origin='lower', cmap='inferno')\n"
                    "    plt.colorbar()\n"
                    "    plt.show()\n"
                    "    if data.ndim == 3:\n"
                    f"        x = {spectrum_state.get('x_pixel')!r}\n"
                    f"        y = {spectrum_state.get('y_pixel')!r}\n"
                    "        if x is not None and y is not None:\n"
                    "            spectrum = data[:, int(y), int(x)]\n"
                    "            plt.figure()\n"
                    "            plt.plot(np.arange(spectrum.size), spectrum)\n"
                    "            plt.xlabel('Channel')\n"
                    "            plt.ylabel(hdu.header.get('BUNIT', 'Intensity'))\n"
                    "            plt.show()\n"
                ),
            }
            if "notebook" in requested:
                notebook = {
                    "cells": [
                        {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": [
                                f"# Quasar workbench reproduction: {filename}\n",
                                f"Source URL: {source_url}\n",
                            ],
                        },
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": exports["python"]["content"].splitlines(keepends=True),
                        },
                    ],
                    "metadata": {
                        "kernelspec": {
                            "display_name": "Python 3",
                            "language": "python",
                            "name": "python3",
                        },
                        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
                    },
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
                exports["notebook"] = {
                    "filename": "workbench_reproduce.ipynb",
                    "content": json.dumps(notebook, indent=2),
                    "mime_type": "application/x-ipynb+json",
                    "encoding": "text",
                }
        if "csv" in requested or "spectrum_csv" in requested:
            spectrum_csv = self._spectrum_csv_export(session)
            if spectrum_csv:
                exports["spectrum_csv"] = spectrum_csv
            else:
                exports["spectrum_csv_status"] = {
                    "filename": "spectrum_cache_required.txt",
                    "content": "Prepare the FITS product before exporting a computed spectrum CSV.\n",
                    "mime_type": "text/plain",
                    "encoding": "text",
                }
        if "figure" in requested or "figure_png" in requested or "png" in requested:
            figure_png = self._render_png_export(session)
            if figure_png:
                exports["figure_png"] = figure_png
            else:
                exports["figure_png_status"] = {
                    "filename": "figure_cache_required.txt",
                    "content": "Prepare the FITS product before exporting a computed render PNG.\n",
                    "mime_type": "text/plain",
                    "encoding": "text",
                }

        return {
            "session_id": session_id,
            "formats": sorted(requested),
            "exports": exports,
            "evidence": session.get("evidence", {}),
        }

    def _spectrum_csv_export(self, session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cache_path = self._session_cache_path(session)
        if not cache_path.exists():
            return None
        state = session.get("state") if isinstance(session.get("state"), dict) else {}
        spectrum_state = state.get("spectrum") if isinstance(state.get("spectrum"), dict) else {}
        try:
            _, series = self._extract_spectrum_from_cache(
                session,
                extraction={
                    "x_pixel": spectrum_state.get("x_pixel"),
                    "y_pixel": spectrum_state.get("y_pixel"),
                    "aperture_radius_pixels": spectrum_state.get("aperture_radius_pixels", 3.0),
                },
                max_points=int(spectrum_state.get("max_points") or 4096),
            )
        except Exception:
            return None
        x_values = series.get("x", [])
        y_values = series.get("y", [])
        lines = [
            f"# {series.get('x_label', 'x')},{series.get('y_label', 'intensity')}",
            "x,y",
        ]
        for x_value, y_value in zip(x_values, y_values):
            lines.append(f"{x_value},{'' if y_value is None else y_value}")
        return {
            "filename": "workbench_spectrum.csv",
            "content": "\n".join(lines) + "\n",
            "mime_type": "text/csv",
            "encoding": "text",
        }

    def _render_png_export(self, session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cache_path = self._session_cache_path(session)
        if not cache_path.exists():
            return None
        state = session.get("state") if isinstance(session.get("state"), dict) else {}
        try:
            image, _, _ = self._render_from_cache(
                session,
                mode=str(state.get("mode") or "image"),
                channel=self._safe_int(state.get("channel")),
                moment=self._safe_int(state.get("moment")),
                colormap=str(state.get("colormap") or "inferno"),
                stretch=str(state.get("stretch") or "asinh"),
                contour_sigma=[
                    float(value)
                    for value in (state.get("contour_sigma") if isinstance(state.get("contour_sigma"), list) else [3, 5, 10])
                    if self._safe_float(value) is not None
                ],
                rms_region=state.get("rms_region") if isinstance(state.get("rms_region"), dict) else None,
            )
        except Exception:
            return None
        data_url = image.get("data_url")
        if not data_url:
            return None
        return {
            "filename": "workbench_render.png",
            "content": data_url,
            "mime_type": "image/png",
            "encoding": "data_url",
        }

    def _run_job(
        self,
        session_id: str,
        user_id: str,
        job_id: str,
        operation: str,
        payload: Dict[str, Any],
    ) -> None:
        try:
            if self._job_cancel_requested(session_id=session_id, user_id=user_id, job_id=job_id):
                raise CubeWorkbenchJobCancelled("Workbench job was canceled before it started")
            self._update_job_state(
                session_id=session_id,
                user_id=user_id,
                job_id=job_id,
                status="running",
                phase="starting",
                progress=1,
                started=True,
            )

            result = self._execute_job_operation(
                session_id=session_id,
                user_id=user_id,
                job_id=job_id,
                operation=operation,
                payload=payload,
            )
            self._update_job_state(
                session_id=session_id,
                user_id=user_id,
                job_id=job_id,
                status="succeeded",
                phase="complete",
                progress=100,
                result=result,
                finished=True,
            )
        except CubeWorkbenchJobCancelled as exc:
            self._update_job_state(
                session_id=session_id,
                user_id=user_id,
                job_id=job_id,
                status="canceled",
                phase="canceled",
                progress=100,
                error=str(exc),
                finished=True,
            )
        except Exception as exc:
            self._update_job_state(
                session_id=session_id,
                user_id=user_id,
                job_id=job_id,
                status="failed",
                phase="failed",
                progress=100,
                error=str(exc),
                finished=True,
            )
        finally:
            with self._job_lock:
                self._active_jobs.pop(job_id, None)

    def _execute_job_operation(
        self,
        *,
        session_id: str,
        user_id: str,
        job_id: str,
        operation: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        def cancel_requested() -> bool:
            return self._job_cancel_requested(session_id=session_id, user_id=user_id, job_id=job_id)

        def set_phase(phase: str, progress: int, metrics: Optional[Dict[str, Any]] = None) -> None:
            if cancel_requested():
                raise CubeWorkbenchJobCancelled(f"Workbench {operation} job was canceled")
            self._update_job_state(
                session_id=session_id,
                user_id=user_id,
                job_id=job_id,
                status="running",
                phase=phase,
                progress=progress,
                metrics=metrics,
            )

        if operation == "prepare":
            last_update = {"at": 0.0}

            def copy_progress(done_bytes: int, total_bytes: Optional[int]) -> None:
                now = time.time()
                if total_bytes and done_bytes >= total_bytes:
                    progress = 95
                elif total_bytes:
                    progress = 10 + int(85 * min(1.0, done_bytes / max(1, total_bytes)))
                else:
                    progress = min(90, 10 + int(done_bytes / max(1, 10 * 1024 * 1024)))
                if now - last_update["at"] < 0.75 and progress < 95:
                    return
                last_update["at"] = now
                set_phase(
                    "staging FITS product",
                    max(5, min(95, progress)),
                    {
                        "bytes_done": done_bytes,
                        "bytes_total": total_bytes,
                    },
                )

            set_phase("validating cache target", 5)
            return self.prepare_product(
                session_id=session_id,
                user_id=user_id,
                max_bytes=self._safe_int(payload.get("max_bytes")) or DEFAULT_MAX_CACHE_BYTES,
                user_cache_bytes=self._safe_int(payload.get("user_cache_bytes")) or DEFAULT_USER_CACHE_BYTES,
                cache_ttl_seconds=self._safe_int(payload.get("cache_ttl_seconds")) or DEFAULT_CACHE_TTL_SECONDS,
                force=bool(payload.get("force")),
                progress_callback=copy_progress,
                cancel_callback=cancel_requested,
            )

        if operation == "render":
            set_phase("rendering FITS plane", 15)
            return self.render_plan(
                session_id=session_id,
                user_id=user_id,
                mode=str(payload.get("mode") or "image"),
                channel=self._safe_int(payload.get("channel")),
                moment=self._safe_int(payload.get("moment")),
                colormap=str(payload.get("colormap") or "inferno"),
                stretch=str(payload.get("stretch") or "asinh"),
                contour_sigma=payload.get("contour_sigma") if isinstance(payload.get("contour_sigma"), list) else None,
                rms_region=payload.get("rms_region") if isinstance(payload.get("rms_region"), dict) else None,
            )

        if operation == "spectrum":
            set_phase("extracting aperture spectrum", 15)
            return self.spectrum_plan(
                session_id=session_id,
                user_id=user_id,
                x_pixel=self._safe_float(payload.get("x_pixel")),
                y_pixel=self._safe_float(payload.get("y_pixel")),
                aperture_radius_pixels=float(payload.get("aperture_radius_pixels") or 3.0),
                aperture_radius_arcsec=self._safe_float(payload.get("aperture_radius_arcsec")),
                max_points=self._safe_int(payload.get("max_points")) or 512,
            )

        if operation == "pv_slice":
            set_phase("extracting PV slice", 15)
            return self.pv_slice_plan(
                session_id=session_id,
                user_id=user_id,
                path=payload.get("path") if isinstance(payload.get("path"), list) else None,
                width_pixels=float(payload.get("width_pixels") or 3.0),
                max_points=self._safe_int(payload.get("max_points")) or 512,
            )

        if operation == "line_overlays":
            set_phase("identifying spectral lines", 15)
            return self.line_overlays(
                session_id=session_id,
                user_id=user_id,
                observed_frequency_ghz=self._safe_float(payload.get("observed_frequency_ghz")),
                line_preset_key=str(payload.get("line_preset_key") or "") or None,
                redshift=float(payload.get("redshift") or 0.0),
                tolerance_ghz=float(payload.get("tolerance_ghz") or 0.01),
                top_n=self._safe_int(payload.get("top_n")) or 8,
            )

        if operation == "export":
            set_phase("building reproducible exports", 15)
            formats = payload.get("formats") if isinstance(payload.get("formats"), list) else None
            return self.export_plan(
                session_id=session_id,
                user_id=user_id,
                formats=formats,
            )

        raise ValueError(f"Unsupported workbench job operation: {operation}")

    def _update_job_state(
        self,
        *,
        session_id: str,
        user_id: str,
        job_id: str,
        status: Optional[str] = None,
        phase: Optional[str] = None,
        progress: Optional[int] = None,
        cancel_requested: Optional[bool] = None,
        started: bool = False,
        finished: bool = False,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = self.get_session(session_id=session_id, user_id=user_id)
        job = self._find_job(session, job_id)
        if not job:
            raise CubeWorkbenchNotFound(f"Workbench job not found: {job_id}")
        now = int(time.time())
        if status:
            job["status"] = status
        if phase:
            job["phase"] = phase
        if progress is not None:
            job["progress"] = max(0, min(100, int(progress)))
        if cancel_requested is not None:
            job["cancel_requested"] = bool(cancel_requested)
        if started and not job.get("started_at"):
            job["started_at"] = now
        if finished:
            job["finished_at"] = now
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error
        if metrics is not None:
            job["metrics"] = metrics
        self._write_session(session)
        return dict(job)

    def _job_cancel_requested(self, *, session_id: str, user_id: str, job_id: str) -> bool:
        try:
            session = self._read_session(session_id)
        except CubeWorkbenchNotFound:
            return True
        if session.get("user_id") != user_id:
            return True
        job = self._find_job(session, job_id)
        return bool(job and job.get("cancel_requested"))

    @staticmethod
    def _append_job(jobs: Any, job: Dict[str, Any]) -> List[Dict[str, Any]]:
        clean_jobs = [item for item in jobs if isinstance(item, dict)] if isinstance(jobs, list) else []
        clean_jobs.append(job)
        return clean_jobs[-12:]

    @staticmethod
    def _find_job(session: Dict[str, Any], job_id: str) -> Optional[Dict[str, Any]]:
        jobs = session.get("jobs") if isinstance(session.get("jobs"), list) else []
        for job in jobs:
            if isinstance(job, dict) and str(job.get("job_id")) == str(job_id):
                return job
        return None

    def _read_session(self, session_id: str) -> Dict[str, Any]:
        path = self.session_dir / f"{session_id}.json"
        if not path.exists():
            raise CubeWorkbenchNotFound(f"Workbench session not found: {session_id}")
        with self._job_lock:
            return json.loads(path.read_text(encoding="utf-8"))

    def _write_session(self, session: Dict[str, Any]) -> None:
        path = self.session_dir / f"{session['session_id']}.json"
        with self._job_lock:
            path.write_text(json.dumps(session, indent=2, sort_keys=True), encoding="utf-8")

    def _cache_response(self, session: Dict[str, Any], cache: Dict[str, Any]) -> Dict[str, Any]:
        cache_payload = dict(cache or {})
        cache_payload["user_cache"] = self._cache_summary_for_user(
            user_id=str(session.get("user_id") or ""),
            max_user_cache_bytes=self._safe_int(cache_payload.get("user_cache_limit_bytes")) or DEFAULT_USER_CACHE_BYTES,
        )
        return {
            "session_id": session.get("session_id", ""),
            "status": cache_payload.get("status", "unknown"),
            "cache": cache_payload,
            "cache_path": session.get("cache_path", ""),
            "evidence": session.get("evidence", {}),
        }

    def _ensure_preview_product(self, session: Dict[str, Any], source_path: Path) -> Dict[str, Any]:
        """Create a downsampled FITS preview product when the staged product is large enough."""
        preview_path = self._preview_path_for(session)
        max_spatial = self._preview_max_spatial_pixels()
        max_channels = self._preview_max_channels()
        try:
            preview = self._build_preview_product(
                source_path=source_path,
                preview_path=preview_path,
                max_spatial=max_spatial,
                max_channels=max_channels,
            )
            return preview
        except Exception as exc:
            try:
                if preview_path.exists():
                    preview_path.unlink()
            except OSError:
                pass
            return {
                "status": "failed",
                "path": "",
                "error": str(exc),
                "generated_at": int(time.time()),
                "max_spatial_pixels": max_spatial,
                "max_channels": max_channels,
            }

    def _build_preview_product(
        self,
        *,
        source_path: Path,
        preview_path: Path,
        max_spatial: int,
        max_channels: int,
    ) -> Dict[str, Any]:
        import numpy as np
        from astropy.io import fits

        with fits.open(source_path, memmap=True) as hdul:
            hdu = self._science_hdu(hdul)
            arr = np.asarray(hdu.data).squeeze()
            if arr.ndim > 3:
                arr = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
            if arr.ndim < 2:
                return {
                    "status": "not_applicable",
                    "path": "",
                    "reason": "science HDU is not image-like",
                    "source_shape": list(arr.shape),
                    "generated_at": int(time.time()),
                }

            source_shape = list(arr.shape)
            ny = int(arr.shape[-2])
            nx = int(arr.shape[-1])
            spatial_stride_y = max(1, int(math.ceil(ny / max(1, max_spatial))))
            spatial_stride_x = max(1, int(math.ceil(nx / max(1, max_spatial))))
            spectral_stride = 1
            if arr.ndim >= 3:
                spectral_stride = max(1, int(math.ceil(int(arr.shape[-3]) / max(1, max_channels))))

            if spectral_stride == 1 and spatial_stride_y == 1 and spatial_stride_x == 1:
                if preview_path.exists():
                    preview_path.unlink()
                return {
                    "status": "not_needed",
                    "path": "",
                    "reason": "product is within preview limits",
                    "source_shape": source_shape,
                    "shape": source_shape,
                    "spectral_stride": 1,
                    "spatial_stride_y": 1,
                    "spatial_stride_x": 1,
                    "generated_at": int(time.time()),
                    "max_spatial_pixels": max_spatial,
                    "max_channels": max_channels,
                }

            if arr.ndim >= 3:
                preview_data = arr[::spectral_stride, ::spatial_stride_y, ::spatial_stride_x]
            else:
                preview_data = arr[::spatial_stride_y, ::spatial_stride_x]
            preview_data = np.asarray(preview_data, dtype=np.float32)
            header = hdu.header.copy()
            self._scale_preview_header(
                header,
                spectral_stride=spectral_stride,
                spatial_stride_y=spatial_stride_y,
                spatial_stride_x=spatial_stride_x,
            )
            fits.PrimaryHDU(data=preview_data, header=header).writeto(preview_path, overwrite=True)
            return {
                "status": "ready",
                "path": str(preview_path),
                "bytes": preview_path.stat().st_size,
                "source_shape": source_shape,
                "shape": list(preview_data.shape),
                "spectral_stride": spectral_stride,
                "spatial_stride_y": spatial_stride_y,
                "spatial_stride_x": spatial_stride_x,
                "generated_at": int(time.time()),
                "max_spatial_pixels": max_spatial,
                "max_channels": max_channels,
            }

    @staticmethod
    def _scale_preview_header(
        header: Any,
        *,
        spectral_stride: int,
        spatial_stride_y: int,
        spatial_stride_x: int,
    ) -> None:
        for axis, stride in ((1, spatial_stride_x), (2, spatial_stride_y), (3, spectral_stride)):
            if stride <= 1:
                continue
            cdelt_key = f"CDELT{axis}"
            crpix_key = f"CRPIX{axis}"
            cdelt = CubeWorkbenchService._safe_float(header.get(cdelt_key))
            crpix = CubeWorkbenchService._safe_float(header.get(crpix_key))
            if cdelt is not None:
                header[cdelt_key] = cdelt * stride
            if crpix is not None:
                header[crpix_key] = (crpix - 1.0) / stride + 1.0
        history = (
            "QUASAR preview product: "
            f"spectral stride={spectral_stride}, y stride={spatial_stride_y}, x stride={spatial_stride_x}"
        )
        try:
            header.add_history(history)
        except Exception:
            pass

    @staticmethod
    def _preview_max_spatial_pixels() -> int:
        return max(
            64,
            min(
                4096,
                CubeWorkbenchService._safe_int(os.getenv("QUASAR_WORKBENCH_PREVIEW_MAX_SPATIAL")) or DEFAULT_PREVIEW_MAX_SPATIAL_PIXELS,
            ),
        )

    @staticmethod
    def _preview_max_channels() -> int:
        return max(
            16,
            min(
                2048,
                CubeWorkbenchService._safe_int(os.getenv("QUASAR_WORKBENCH_PREVIEW_MAX_CHANNELS")) or DEFAULT_PREVIEW_MAX_CHANNELS,
            ),
        )

    def _enforce_cache_limits(
        self,
        *,
        user_id: str,
        protect_session_id: str,
        incoming_bytes: int,
        max_user_cache_bytes: int,
        ttl_seconds: int,
    ) -> Dict[str, Any]:
        if incoming_bytes > max_user_cache_bytes:
            raise ValueError(
                f"FITS product is {incoming_bytes / (1024 * 1024):.1f} MB, above the per-user workbench cache limit "
                f"of {max_user_cache_bytes / (1024 * 1024):.1f} MB"
            )

        evicted: List[Dict[str, Any]] = []
        now = int(time.time())
        records = self._cached_session_records(user_id=user_id)
        protected = str(protect_session_id)

        stale_records = []
        if ttl_seconds > 0:
            for record in records:
                session = record["session"]
                if str(session.get("session_id")) == protected:
                    continue
                last_accessed = self._safe_int(session.get("last_accessed_at")) or self._safe_int(session.get("created_at")) or now
                if now - last_accessed > ttl_seconds:
                    stale_records.append(record)
        for record in sorted(stale_records, key=self._eviction_sort_key):
            evicted_item = self._evict_cached_record(record, reason="ttl_expired")
            if evicted_item:
                evicted.append(evicted_item)

        records = self._cached_session_records(user_id=user_id)
        total_bytes = sum(
            int(record["bytes"])
            for record in records
            if str(record["session"].get("session_id")) != protected
        )
        for record in sorted(records, key=self._eviction_sort_key):
            if total_bytes + incoming_bytes <= max_user_cache_bytes:
                break
            if str(record["session"].get("session_id")) == protected:
                continue
            evicted_item = self._evict_cached_record(record, reason="quota_pressure")
            if evicted_item:
                evicted.append(evicted_item)
                total_bytes -= int(record["bytes"])

        summary = self._cache_summary_for_user(
            user_id=user_id,
            max_user_cache_bytes=max_user_cache_bytes,
            include_incoming_bytes=incoming_bytes,
        )
        summary["evicted_count"] = len(evicted)
        summary["evicted_bytes"] = sum(int(item.get("cached_bytes") or 0) for item in evicted)
        return {
            "summary": summary,
            "evicted": evicted,
        }

    def _cache_summary_for_user(
        self,
        *,
        user_id: str,
        max_user_cache_bytes: int,
        include_incoming_bytes: int = 0,
    ) -> Dict[str, Any]:
        records = self._cached_session_records(user_id=user_id)
        cached_bytes = sum(int(record["bytes"]) for record in records)
        projected_bytes = cached_bytes + max(0, int(include_incoming_bytes or 0))
        limit = max(1, int(max_user_cache_bytes or DEFAULT_USER_CACHE_BYTES))
        return {
            "cached_bytes": cached_bytes,
            "projected_cached_bytes": projected_bytes,
            "limit_bytes": limit,
            "file_count": len(records),
            "status": "over_limit" if projected_bytes > limit else "ok",
        }

    def _cached_session_records(self, *, user_id: str) -> List[Dict[str, Any]]:
        records = []
        for session, path in self._iter_session_records():
            if str(session.get("user_id") or "") != str(user_id):
                continue
            cache_path = self._session_cache_path(session)
            if not cache_path.exists():
                continue
            try:
                cached_bytes = cache_path.stat().st_size
            except OSError:
                continue
            preview_info = self._session_preview_info(session)
            preview_path = Path(str(preview_info.get("path"))) if preview_info else None
            preview_bytes = 0
            if preview_path and preview_path.exists():
                try:
                    preview_bytes = preview_path.stat().st_size
                except OSError:
                    preview_bytes = 0
            records.append({
                "session": session,
                "session_path": path,
                "cache_path": cache_path,
                "preview_path": preview_path,
                "bytes": cached_bytes + preview_bytes,
                "product_bytes": cached_bytes,
                "preview_bytes": preview_bytes,
            })
        return records

    def _iter_session_records(self) -> List[tuple[Dict[str, Any], Path]]:
        records: List[tuple[Dict[str, Any], Path]] = []
        with self._job_lock:
            for path in self.session_dir.glob("*.json"):
                try:
                    session = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(session, dict):
                    records.append((session, path))
        return records

    @staticmethod
    def _eviction_sort_key(record: Dict[str, Any]) -> tuple[int, int, str]:
        session = record.get("session") if isinstance(record.get("session"), dict) else {}
        cache = session.get("cache") if isinstance(session.get("cache"), dict) else {}
        prepared_at = CubeWorkbenchService._safe_int(cache.get("prepared_at")) or CubeWorkbenchService._safe_int(session.get("created_at")) or 0
        last_accessed = CubeWorkbenchService._safe_int(session.get("last_accessed_at")) or prepared_at
        return (int(last_accessed), int(prepared_at), str(session.get("session_id") or ""))

    def _evict_cached_record(self, record: Dict[str, Any], *, reason: str) -> Optional[Dict[str, Any]]:
        session = record.get("session") if isinstance(record.get("session"), dict) else {}
        session_id = str(session.get("session_id") or "")
        if not session_id:
            return None
        with self._job_lock:
            if session_id in self._active_jobs:
                return None
        cache_path = record.get("cache_path")
        preview_path = record.get("preview_path")
        cached_bytes = int(record.get("bytes") or 0)
        try:
            if isinstance(cache_path, Path) and cache_path.exists():
                cache_path.unlink()
            if isinstance(preview_path, Path) and preview_path.exists():
                preview_path.unlink()
        except OSError:
            return None

        cache = session.setdefault("cache", {})
        cache.update({
            "status": "evicted",
            "cached_bytes": 0,
            "evicted_at": int(time.time()),
            "eviction_reason": reason,
            "error": "",
        })
        cache["preview"] = {
            "status": "evicted",
            "path": "",
            "bytes": 0,
            "evicted_at": int(time.time()),
        }
        session["cache_path"] = ""
        if str(session.get("status") or "") == "cached":
            session["status"] = "cache_evicted"
        evidence = session.setdefault("evidence", {})
        operations = list(evidence.get("operations") or [])
        if "cache_eviction" not in operations:
            operations.append("cache_eviction")
        evidence["operations"] = operations
        warnings = list(evidence.get("warnings") or [])
        message = f"Cached FITS product was evicted from the workbench cache ({reason})."
        if message not in warnings:
            warnings.append(message)
        evidence["warnings"] = warnings
        self._write_session(session)
        return {
            "session_id": session_id,
            "filename": session.get("filename", ""),
            "cached_bytes": cached_bytes,
            "reason": reason,
        }

    def _cache_path_for(self, session: Dict[str, Any]) -> Path:
        filename = str(session.get("filename") or "alma_product.fits")
        suffix = Path(filename).suffix.lower()
        if suffix not in {".fits", ".fit", ".fz"}:
            suffix = ".fits"
        return self.cache_dir / f"{session['session_id']}{suffix}"

    def _preview_path_for(self, session: Dict[str, Any]) -> Path:
        return self.cache_dir / f"{session['session_id']}.preview.fits"

    @staticmethod
    def _session_cache_path(session: Dict[str, Any]) -> Path:
        raw_path = str(session.get("cache_path") or "")
        if not raw_path:
            return Path("__missing_workbench_cache__")
        return Path(raw_path)

    @staticmethod
    def _session_preview_info(session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cache = session.get("cache") if isinstance(session.get("cache"), dict) else {}
        preview = cache.get("preview") if isinstance(cache.get("preview"), dict) else {}
        preview_path = Path(str(preview.get("path") or ""))
        if preview.get("status") == "ready" and preview_path.exists():
            return dict(preview)
        return None

    @staticmethod
    def _analysis_product_path(session: Dict[str, Any]) -> tuple[Path, Optional[Dict[str, Any]]]:
        preview = CubeWorkbenchService._session_preview_info(session)
        if preview:
            return Path(str(preview["path"])), preview
        return CubeWorkbenchService._session_cache_path(session), None

    @staticmethod
    def _preview_summary(preview: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not preview:
            return None
        return {
            "status": preview.get("status"),
            "bytes": preview.get("bytes"),
            "source_shape": preview.get("source_shape"),
            "shape": preview.get("shape"),
            "spectral_stride": preview.get("spectral_stride"),
            "spatial_stride_y": preview.get("spatial_stride_y"),
            "spatial_stride_x": preview.get("spatial_stride_x"),
        }

    @staticmethod
    def _scale_value_for_preview(value: Optional[float], stride: Any) -> Optional[float]:
        if value is None:
            return None
        divisor = max(1.0, float(stride or 1.0))
        return value / divisor

    @staticmethod
    def _scale_radius_for_preview(value: float, preview: Optional[Dict[str, Any]]) -> float:
        if not preview:
            return value
        stride_x = max(1.0, float(preview.get("spatial_stride_x") or 1.0))
        stride_y = max(1.0, float(preview.get("spatial_stride_y") or 1.0))
        return max(0.5, value / ((stride_x + stride_y) / 2.0))

    @staticmethod
    def _scale_region_for_preview(region: Optional[Dict[str, float]], preview: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        clean = CubeWorkbenchService._clean_region(region)
        if not clean or not preview:
            return clean
        stride_x = max(1.0, float(preview.get("spatial_stride_x") or 1.0))
        stride_y = max(1.0, float(preview.get("spatial_stride_y") or 1.0))
        return {
            "x1": clean["x1"] / stride_x,
            "y1": clean["y1"] / stride_y,
            "x2": clean["x2"] / stride_x,
            "y2": clean["y2"] / stride_y,
        }

    @staticmethod
    def _scale_path_for_preview(path: List[Dict[str, float]], preview: Optional[Dict[str, Any]]) -> List[Dict[str, float]]:
        clean = CubeWorkbenchService._clean_path(path)
        if not preview:
            return clean
        stride_x = max(1.0, float(preview.get("spatial_stride_x") or 1.0))
        stride_y = max(1.0, float(preview.get("spatial_stride_y") or 1.0))
        return [{"x": point["x"] / stride_x, "y": point["y"] / stride_y} for point in clean]

    @staticmethod
    def _scale_channel_for_preview(channel: int, preview: Optional[Dict[str, Any]]) -> int:
        if not preview:
            return channel
        stride = max(1, int(float(preview.get("spectral_stride") or 1)))
        return int(round(channel / stride))

    @staticmethod
    def _copy_or_download(
        source_url: str,
        target_path: Path,
        max_bytes: int,
        *,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
        cancel_callback: Optional[Callable[[], bool]] = None,
    ) -> int:
        parsed = urlparse(source_url)
        if parsed.scheme == "file":
            src_path = parsed.path
            if os.name == "nt" and src_path.startswith("/") and len(src_path) > 2 and src_path[2] == ":":
                src_path = src_path[1:]
            src = Path(src_path)
        elif parsed.scheme in {"", None} and Path(source_url).exists():
            src = Path(source_url)
        else:
            return CubeWorkbenchService._download_to_path(
                source_url,
                target_path,
                max_bytes,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
            )

        size = src.stat().st_size
        if size > max_bytes:
            raise ValueError(
                f"FITS product is {size / (1024 * 1024):.1f} MB, above the workbench cache limit "
                f"of {max_bytes / (1024 * 1024):.1f} MB"
            )
        copied = 0
        with src.open("rb") as input_handle, target_path.open("wb") as output_handle:
            while True:
                if cancel_callback and cancel_callback():
                    raise CubeWorkbenchJobCancelled("Workbench job was canceled during FITS cache copy")
                chunk = input_handle.read(1024 * 1024)
                if not chunk:
                    break
                output_handle.write(chunk)
                copied += len(chunk)
                if progress_callback:
                    progress_callback(copied, size)
        return copied

    @staticmethod
    def _download_to_path(
        source_url: str,
        target_path: Path,
        max_bytes: int,
        *,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
        cancel_callback: Optional[Callable[[], bool]] = None,
    ) -> int:
        try:
            head = requests.head(source_url, allow_redirects=True, timeout=30)
            content_length = head.headers.get("Content-Length") if head.ok else None
            if content_length and int(content_length) > max_bytes:
                raise ValueError(
                    f"FITS product is {int(content_length) / (1024 * 1024):.1f} MB, above the workbench cache limit "
                    f"of {max_bytes / (1024 * 1024):.1f} MB"
                )
        except ValueError:
            raise
        except Exception:
            content_length = None

        size = 0
        with requests.get(source_url, stream=True, timeout=180) as response:
            response.raise_for_status()
            with target_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if cancel_callback and cancel_callback():
                        raise CubeWorkbenchJobCancelled("Workbench job was canceled during FITS download")
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError(
                            f"FITS product exceeded the workbench cache limit of {max_bytes / (1024 * 1024):.1f} MB"
                        )
                    handle.write(chunk)
                    if progress_callback:
                        progress_callback(size, int(content_length) if content_length else None)
        return size or int(content_length or 0)

    def _extract_spectrum_from_cache(
        self,
        session: Dict[str, Any],
        *,
        extraction: Dict[str, Any],
        max_points: int,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        import numpy as np
        from astropy.io import fits

        product_path, preview = self._analysis_product_path(session)
        with fits.open(product_path, memmap=True) as hdul:
            hdu = self._science_hdu(hdul)
            cube = self._cube_from_data(hdu.data)
            if cube is None:
                raise ValueError("Cached FITS product is not a spectral cube")
            nchan, ny, nx = cube.shape
            x = self._safe_float(extraction.get("x_pixel"))
            y = self._safe_float(extraction.get("y_pixel"))
            x = self._scale_value_for_preview(x, preview.get("spatial_stride_x") if preview else 1)
            y = self._scale_value_for_preview(y, preview.get("spatial_stride_y") if preview else 1)
            x_idx = int(round(x if x is not None else nx / 2))
            y_idx = int(round(y if y is not None else ny / 2))
            x_idx = max(0, min(nx - 1, x_idx))
            y_idx = max(0, min(ny - 1, y_idx))
            radius = max(0.5, float(extraction.get("aperture_radius_pixels") or 3.0))
            radius = self._scale_radius_for_preview(radius, preview)
            r_int = int(math.ceil(radius))
            x0, x1 = max(0, x_idx - r_int), min(nx, x_idx + r_int + 1)
            y0, y1 = max(0, y_idx - r_int), min(ny, y_idx + r_int + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            mask = ((xx - x_idx) ** 2 + (yy - y_idx) ** 2) <= radius ** 2
            aperture = np.asarray(cube[:, y0:y1, x0:x1], dtype=float)
            if mask.any():
                values = np.nanmean(aperture[:, mask], axis=1)
            else:
                values = np.asarray(cube[:, y_idx, x_idx], dtype=float)

            metadata = dict(session.get("metadata") or {})
            metadata["channel_count"] = nchan
            axis = self._spectral_axis(dict(hdu.header), metadata, max_points=max_points)
            indices = [idx for idx in axis.get("indices", []) if 0 <= int(idx) < nchan]
            if not indices:
                indices = self._downsample_indices(nchan, max_points=max_points)
                axis["indices"] = indices
            axis["values"] = [axis.get("values", [])[i] for i in range(min(len(axis.get("values", [])), len(indices)))]
            if len(axis["values"]) != len(indices):
                axis["values"] = [float(idx) for idx in indices]
                axis["label"] = "Channel"
                axis["unit"] = "channel"

            return axis, {
                "x": axis["values"],
                "y": [self._json_float(values[int(idx)]) for idx in indices],
                "x_label": axis.get("label", "Channel"),
                "y_label": str(hdu.header.get("BUNIT") or metadata.get("unit") or "Intensity"),
                "data_status": "computed",
                "analysis_product": "preview" if preview else "full",
                "preview": self._preview_summary(preview),
            }

    def _render_from_cache(
        self,
        session: Dict[str, Any],
        *,
        mode: str,
        channel: Optional[int],
        moment: Optional[int],
        colormap: str,
        stretch: str,
        contour_sigma: List[float],
        rms_region: Optional[Dict[str, float]],
    ) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        import numpy as np
        from astropy.io import fits

        product_path, preview = self._analysis_product_path(session)
        with fits.open(product_path, memmap=True) as hdul:
            hdu = self._science_hdu(hdul)
            data = np.array(hdu.data, copy=True).squeeze()
            cube = self._cube_from_data(data)
            if cube is None:
                image = np.asarray(data, dtype=float)
                if image.ndim > 2:
                    image = image.reshape((-1, image.shape[-2], image.shape[-1]))[0]
                label = "Image"
                channel_index = None
            else:
                nchan = cube.shape[0]
                selected_mode = str(mode or "image").lower()
                if selected_mode == "channel":
                    requested_channel = int(channel if channel is not None else nchan // 2)
                    analysis_channel = self._scale_channel_for_preview(requested_channel, preview)
                    channel_index = max(0, min(nchan - 1, analysis_channel))
                    image = np.asarray(cube[channel_index], dtype=float)
                    label = f"Channel {requested_channel}" if not preview else f"Channel {requested_channel} (preview {channel_index})"
                elif selected_mode == "moment":
                    moment_order = int(moment if moment is not None else 0)
                    image = self._moment_image(cube, moment_order)
                    label = f"Moment {moment_order}"
                    channel_index = None
                else:
                    channel_index = nchan // 2
                    image = np.asarray(cube[channel_index], dtype=float)
                    label = f"Image plane, channel {channel_index}"
                line_labels = self._line_labels_for_render(
                    session,
                    header=dict(hdu.header),
                    channel_index=channel_index,
                    nchan=nchan,
                    mode=selected_mode,
                )
            if cube is None:
                line_labels = []

            analysis_rms_region = self._scale_region_for_preview(rms_region, preview)
            rms, rms_payload = self._estimate_rms_with_region(image, analysis_rms_region)
            finite = image[np.isfinite(image)]
            stats = {
                "rms": self._json_float(rms),
                "rms_region": rms_payload.get("region"),
                "rms_method": rms_payload.get("method"),
                "min": self._json_float(np.nanmin(finite)) if finite.size else None,
                "max": self._json_float(np.nanmax(finite)) if finite.size else None,
                "mean": self._json_float(np.nanmean(finite)) if finite.size else None,
                "unit": str(hdu.header.get("BUNIT") or (session.get("metadata") or {}).get("unit") or ""),
                "shape": list(image.shape),
                "analysis_product": "preview" if preview else "full",
                "preview": self._preview_summary(preview),
            }
            levels = [float(sigma) * rms for sigma in contour_sigma if rms and math.isfinite(float(sigma) * rms)]
            image_payload = self._render_image_data_url(
                image,
                header=dict(hdu.header),
                title=label,
                colormap=colormap,
                stretch=stretch,
                contour_levels=levels,
                rms_region=rms_payload.get("region"),
                line_labels=line_labels,
            )
            image_payload["channel"] = channel_index
            image_payload["line_labels"] = line_labels
            image_payload["data_status"] = "computed"
            image_payload["analysis_product"] = "preview" if preview else "full"
            image_payload["preview"] = self._preview_summary(preview)
            return image_payload, stats, {
                "sigma": contour_sigma,
                "levels": [self._json_float(level) for level in levels],
                "unit": stats["unit"],
            }

    @staticmethod
    def _moment_image(cube: Any, moment_order: int):
        import numpy as np

        arr = np.asarray(cube, dtype=float)
        if moment_order == 0:
            return np.nansum(arr, axis=0)
        coords = np.arange(arr.shape[0], dtype=float)[:, None, None]
        weights = np.where(np.isfinite(arr), arr, 0.0)
        denom = np.nansum(weights, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.nansum(weights * coords, axis=0) / denom
        mean = np.where(np.isfinite(mean), mean, np.nan)
        if moment_order == 1:
            return mean
        with np.errstate(invalid="ignore", divide="ignore"):
            variance = np.nansum(weights * (coords - mean) ** 2, axis=0) / denom
        return np.sqrt(np.where(variance >= 0, variance, np.nan))

    @staticmethod
    def _estimate_rms(image: Any) -> float:
        import numpy as np

        arr = np.asarray(image, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return 0.0
        median = np.nanmedian(finite)
        mad = np.nanmedian(np.abs(finite - median))
        rms = 1.4826 * mad
        if not np.isfinite(rms) or rms <= 0:
            rms = float(np.nanstd(finite))
        return float(rms) if np.isfinite(rms) else 0.0

    def _line_labels_for_render(
        self,
        session: Dict[str, Any],
        *,
        header: Dict[str, Any],
        channel_index: Optional[int],
        nchan: int,
        mode: str,
    ) -> List[Dict[str, Any]]:
        state = session.get("state") if isinstance(session.get("state"), dict) else {}
        overlay = state.get("line_overlays") if isinstance(state.get("line_overlays"), dict) else {}
        lines = overlay.get("lines") if isinstance(overlay.get("lines"), list) else []
        if not lines:
            return []

        tolerance = self._safe_float(overlay.get("tolerance_ghz")) or 0.01
        metadata = dict(session.get("metadata") or {})
        metadata["channel_count"] = nchan
        axis = self._spectral_axis(header, metadata, max_points=max(nchan, 512))
        axis_values = axis.get("values") if isinstance(axis.get("values"), list) else []
        axis_indices = axis.get("indices") if isinstance(axis.get("indices"), list) else []
        index_to_freq = {
            int(idx): self._safe_float(value)
            for idx, value in zip(axis_indices, axis_values)
            if self._safe_float(value) is not None
        }
        spectral_values = [value for value in index_to_freq.values() if value is not None]
        spectral_min = min(spectral_values) if spectral_values else None
        spectral_max = max(spectral_values) if spectral_values else None
        channel_freq = index_to_freq.get(int(channel_index)) if channel_index is not None else None
        if len(spectral_values) >= 2:
            diffs = [abs(b - a) for a, b in zip(spectral_values, spectral_values[1:]) if abs(b - a) > 0]
            channel_width = min(diffs) if diffs else 0.0
        else:
            channel_width = 0.0
        match_tolerance = max(tolerance, channel_width / 2.0)

        labels = []
        for line in lines:
            if not isinstance(line, dict):
                continue
            observed = self._safe_float(line.get("observed_frequency_ghz"))
            if observed is None:
                continue
            name = str(line.get("species") or line.get("name") or line.get("transition") or "Line")
            in_range = (
                spectral_min is not None
                and spectral_max is not None
                and min(spectral_min, spectral_max) <= observed <= max(spectral_min, spectral_max)
            )
            delta = abs(observed - channel_freq) if channel_freq is not None else None
            if channel_freq is not None:
                if delta is None or delta > match_tolerance:
                    continue
            elif str(mode).lower() == "moment" and not in_range:
                continue
            elif str(mode).lower() != "moment":
                continue
            labels.append({
                "label": name,
                "observed_frequency_ghz": round(observed, 9),
                "rest_frequency_ghz": self._json_float(line.get("rest_frequency_ghz") or line.get("frequency_ghz")),
                "delta_ghz": self._json_float(delta),
                "channel_frequency_ghz": self._json_float(channel_freq),
                "redshift": self._json_float(line.get("redshift")),
                "source": str(line.get("source") or "Splatalogue"),
            })
        return labels[:4]

    @staticmethod
    def _estimate_rms_with_region(image: Any, region: Optional[Dict[str, float]]) -> tuple[float, Dict[str, Any]]:
        import numpy as np

        arr = np.asarray(image, dtype=float)
        clean = CubeWorkbenchService._clean_region(region)
        if clean:
            ny, nx = arr.shape[-2], arr.shape[-1]
            x1 = max(0, min(nx, int(math.floor(float(clean["x1"])))))
            x2 = max(0, min(nx, int(math.ceil(float(clean["x2"])))))
            y1 = max(0, min(ny, int(math.floor(float(clean["y1"])))))
            y2 = max(0, min(ny, int(math.ceil(float(clean["y2"])))))
            if x2 > x1 and y2 > y1:
                sub = arr[y1:y2, x1:x2]
                finite = sub[np.isfinite(sub)]
                if finite.size >= 4:
                    return CubeWorkbenchService._estimate_rms(sub), {
                        "method": "region_mad",
                        "region": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    }
        return CubeWorkbenchService._estimate_rms(arr), {
            "method": "full_image_mad",
            "region": clean,
        }

    @staticmethod
    def _render_image_data_url(
        image: Any,
        *,
        header: Dict[str, Any],
        title: str,
        colormap: str,
        stretch: str,
        contour_levels: List[float],
        rms_region: Optional[Dict[str, float]],
        line_labels: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.visualization import (
            AsinhStretch,
            ImageNormalize,
            LinearStretch,
            LogStretch,
            SqrtStretch,
            ZScaleInterval,
        )

        wcs = None
        wcs_status = "unavailable"
        try:
            from astropy.wcs import WCS

            candidate = WCS(header)
            celestial = candidate.celestial
            if celestial.pixel_n_dim == 2 and celestial.world_n_dim == 2:
                wcs = celestial
                wcs_status = "present"
        except Exception:
            wcs = None

        stretch_map = {
            "linear": LinearStretch(),
            "sqrt": SqrtStretch(),
            "log": LogStretch(),
            "asinh": AsinhStretch(),
        }
        arr = np.asarray(image, dtype=float)
        finite = arr[np.isfinite(arr)]
        norm = ImageNormalize(
            arr,
            interval=ZScaleInterval() if finite.size else None,
            stretch=stretch_map.get(stretch, AsinhStretch()),
        )
        fig = plt.figure(figsize=(7.5, 6.5), facecolor="#020617")
        ax = fig.add_subplot(111, projection=wcs) if wcs is not None else fig.add_subplot(111)
        im = ax.imshow(arr, origin="lower", cmap=colormap or "inferno", norm=norm)
        valid_levels = [level for level in contour_levels if finite.size and np.nanmin(finite) < level < np.nanmax(finite)]
        if valid_levels:
            ax.contour(arr, levels=valid_levels, colors="cyan", linewidths=0.7, alpha=0.8, origin="lower")
        clean_region = CubeWorkbenchService._clean_region(rms_region)
        if clean_region:
            from matplotlib.patches import Rectangle

            width = float(clean_region["x2"]) - float(clean_region["x1"])
            height = float(clean_region["y2"]) - float(clean_region["y1"])
            if width > 0 and height > 0:
                ax.add_patch(Rectangle(
                    (float(clean_region["x1"]), float(clean_region["y1"])),
                    width,
                    height,
                    fill=False,
                    edgecolor="#fbbf24",
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.9,
                ))
        labels = line_labels or []
        if labels:
            lines_text = []
            for item in labels[:4]:
                label = str(item.get("label") or "Line")
                observed = CubeWorkbenchService._json_float(item.get("observed_frequency_ghz"))
                if observed is not None:
                    lines_text.append(f"{label} @ {observed:.6g} GHz")
                else:
                    lines_text.append(label)
            ax.text(
                0.02,
                0.98,
                "\n".join(lines_text),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7,
                color="#fde68a",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "#020617",
                    "edgecolor": "#fbbf24",
                    "alpha": 0.78,
                    "linewidth": 0.6,
                },
            )
        ax.set_title(title, color="white", fontsize=11)
        if wcs is not None:
            ax.coords[0].set_axislabel("RA (J2000)", color="white", fontsize=9)
            ax.coords[1].set_axislabel("Dec (J2000)", color="white", fontsize=9)
            ax.coords[0].set_ticklabel(color="white", fontsize=7)
            ax.coords[1].set_ticklabel(color="white", fontsize=7)
            axis_labels = {"x": "RA (J2000)", "y": "Dec (J2000)"}
        else:
            ax.set_xlabel("X pixel", color="white", fontsize=9)
            ax.set_ylabel("Y pixel", color="white", fontsize=9)
            axis_labels = {"x": "X pixel", "y": "Y pixel"}
        ax.tick_params(colors="white", labelsize=7)
        ax.set_facecolor("#020617")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="white", labelsize=7)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140, facecolor=fig.get_facecolor())
        plt.close(fig)
        return {
            "data_url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
            "label": title,
            "axis_labels": axis_labels,
            "wcs_status": wcs_status,
            "shape": list(arr.shape),
        }

    def _extract_pv_from_cache(
        self,
        session: Dict[str, Any],
        *,
        path: List[Dict[str, float]],
        width_pixels: float,
        max_points: int,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        import numpy as np
        from astropy.io import fits

        product_path, preview = self._analysis_product_path(session)
        with fits.open(product_path, memmap=True) as hdul:
            hdu = self._science_hdu(hdul)
            cube = self._cube_from_data(hdu.data)
            if cube is None:
                raise ValueError("Cached FITS product is not a spectral cube")
            nchan, ny, nx = cube.shape
            analysis_path = self._scale_path_for_preview(path, preview)
            points = self._sample_path_points(analysis_path, max_points=max_points)
            analysis_width = self._scale_radius_for_preview(float(width_pixels or 3.0), preview)
            radius = max(0, int(math.ceil(analysis_width / 2.0)))
            columns = []
            for point in points:
                x_idx = max(0, min(nx - 1, int(round(point["x"]))))
                y_idx = max(0, min(ny - 1, int(round(point["y"]))))
                x0, x1 = max(0, x_idx - radius), min(nx, x_idx + radius + 1)
                y0, y1 = max(0, y_idx - radius), min(ny, y_idx + radius + 1)
                aperture = np.asarray(cube[:, y0:y1, x0:x1], dtype=float)
                columns.append(np.nanmean(aperture, axis=(1, 2)))
            pv = np.stack(columns, axis=1) if columns else np.empty((nchan, 0))

            metadata = dict(session.get("metadata") or {})
            metadata["channel_count"] = nchan
            axis = self._spectral_axis(dict(hdu.header), metadata, max_points=max_points)
            spectral_indices = [idx for idx in axis.get("indices", []) if 0 <= int(idx) < nchan]
            if not spectral_indices:
                spectral_indices = self._downsample_indices(nchan, max_points=max_points)
                axis["indices"] = spectral_indices
            pv = pv[spectral_indices, :]
            image_url = self._render_pv_data_url(pv, y_label=axis.get("label", "Channel"))
            return axis, {
                "data_url": image_url,
                "data_status": "computed",
                "shape": list(pv.shape),
                "analysis_product": "preview" if preview else "full",
                "preview": self._preview_summary(preview),
            }

    @staticmethod
    def _science_hdu(hdul):
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is not None and getattr(data, "ndim", 0) >= 2:
                return hdu
        raise ValueError("No image or cube HDU found in FITS product")

    @staticmethod
    def _cube_from_data(data: Any):
        import numpy as np

        arr = np.asarray(data).squeeze()
        if arr.ndim < 3:
            return None
        if arr.ndim > 3:
            arr = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
        return np.array(arr, dtype=float, copy=True)

    @staticmethod
    def _sample_path_points(path: List[Dict[str, float]], *, max_points: int) -> List[Dict[str, float]]:
        if len(path) < 2:
            return []
        samples = max(2, min(CubeWorkbenchService._estimate_path_samples(path, max_points=max_points), 512))
        distances = [0.0]
        for start, end in zip(path, path[1:]):
            dx = float(end["x"]) - float(start["x"])
            dy = float(end["y"]) - float(start["y"])
            distances.append(distances[-1] + (dx * dx + dy * dy) ** 0.5)
        total = distances[-1] or 1.0
        result = []
        segment = 0
        for idx in range(samples):
            target = total * idx / max(1, samples - 1)
            while segment < len(distances) - 2 and target > distances[segment + 1]:
                segment += 1
            start = path[segment]
            end = path[segment + 1]
            span = max(1e-9, distances[segment + 1] - distances[segment])
            frac = (target - distances[segment]) / span
            result.append({
                "x": float(start["x"]) + (float(end["x"]) - float(start["x"])) * frac,
                "y": float(start["y"]) + (float(end["y"]) - float(start["y"])) * frac,
            })
        return result

    @staticmethod
    def _render_pv_data_url(pv, *, y_label: str) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        finite = pv[np.isfinite(pv)]
        if finite.size:
            vmin, vmax = np.nanpercentile(finite, [5, 99])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        else:
            vmin, vmax = 0.0, 1.0

        fig, ax = plt.subplots(figsize=(7.5, 4), facecolor="#020617")
        ax.imshow(pv, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Offset (pixel)", color="white")
        ax.set_ylabel(y_label, color="white")
        ax.tick_params(colors="white", labelsize=8)
        ax.set_facecolor("#020617")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140, facecolor=fig.get_facecolor())
        plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _metadata_for_source(self, source_url: str) -> Dict[str, Any]:
        local_path = self._local_path_from_source(source_url)
        if local_path:
            try:
                from astropy.io import fits

                with fits.open(local_path, memmap=True) as hdul:
                    hdu = self._science_hdu(hdul)
                    header = hdu.header
                    result = self._metadata_from_header(header, source_url, local_path.name)
                    result["local_header_read"] = True
                    return result
            except Exception as exc:
                return {
                    "success": False,
                    "url": source_url,
                    "filename": local_path.name,
                    "error": f"Could not read local FITS header: {exc}",
                }

        metadata = self._metadata_for_remote_header_only(source_url)
        if metadata.get("success") or metadata.get("header_only"):
            return metadata

        allow_full_fallback = str(os.getenv("QUASAR_WORKBENCH_ALLOW_FULL_HEADER_FALLBACK", "")).lower() in {"1", "true", "yes"}
        if allow_full_fallback:
            fallback = FITSProcessingService.extract_metadata_from_url(source_url)
            if isinstance(fallback, dict):
                fallback["full_header_fallback"] = True
            return fallback
        return metadata

    def _metadata_for_remote_header_only(self, source_url: str) -> Dict[str, Any]:
        """Read remote FITS metadata through bounded range bytes, never a full download."""
        parsed = urlparse(source_url)
        filename = Path(parsed.path).name or "remote_product.fits"
        max_bytes = self._safe_int(os.getenv("QUASAR_WORKBENCH_HEADER_RANGE_BYTES")) or DEFAULT_HEADER_RANGE_BYTES
        max_bytes = max(2880, min(int(max_bytes), 16 * 1024 * 1024))

        base_payload: Dict[str, Any] = {
            "success": False,
            "url": source_url,
            "filename": filename,
            "header_only": True,
            "range_header_read": True,
            "header_range_limit_bytes": max_bytes,
        }

        try:
            response = requests.get(
                source_url,
                headers={"Range": f"bytes=0-{max_bytes - 1}"},
                timeout=(10, 30),
                stream=True,
            )
            base_payload["range_status_code"] = response.status_code
            if response.headers.get("Content-Length"):
                base_payload["response_content_length"] = self._safe_int(response.headers.get("Content-Length"))
            if response.headers.get("Content-Range"):
                base_payload["content_range"] = response.headers.get("Content-Range")
            if response.headers.get("Content-Length") and response.status_code == 200:
                base_payload["content_length"] = self._safe_int(response.headers.get("Content-Length"))

            if response.status_code not in (200, 206):
                base_payload["error"] = f"Remote FITS header range request returned HTTP {response.status_code}"
                response.close()
                return base_payload

            chunks: List[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                remaining = max_bytes - total
                if remaining <= 0:
                    break
                chunks.append(chunk[:remaining])
                total += min(len(chunk), remaining)
                if total >= max_bytes:
                    break
            response.close()

            raw = b"".join(chunks)
            base_payload["header_bytes_fetched"] = len(raw)
            base_payload["range_truncated"] = response.status_code == 200 and (base_payload.get("content_length") or 0) > len(raw)
            if len(raw) < 2880:
                base_payload["error"] = f"Remote FITS response was too small for a FITS header ({len(raw)} bytes)"
                return base_payload

            parsed_header = self._first_image_header_from_fits_bytes(raw)
            if not parsed_header:
                base_payload["error"] = (
                    "Could not find an image/cube FITS HDU inside the bounded header range. "
                    "Prepare the product or increase QUASAR_WORKBENCH_HEADER_RANGE_BYTES for this source."
                )
                return base_payload

            header = parsed_header["header"]
            result = self._metadata_from_header(header, source_url, filename)
            result.update(base_payload)
            result.update({
                "success": True,
                "data_hdu_index": parsed_header["hdu_index"],
                "header_offset_bytes": parsed_header["header_offset_bytes"],
                "header_block_bytes": parsed_header["header_block_bytes"],
                "data_offset_bytes": parsed_header["data_offset_bytes"],
                "data_bytes_estimated": parsed_header["data_bytes_estimated"],
                "header_strategy": "bounded_remote_range",
            })
            return result
        except Exception as exc:
            base_payload["error"] = f"Could not read remote FITS header via bounded range request: {exc}"
            return base_payload

    @staticmethod
    def _metadata_from_header(header: Any, source_url: str, filename: str) -> Dict[str, Any]:
        result = FITSProcessingService._extract_science_metadata(header, source_url, filename)
        naxis = CubeWorkbenchService._safe_int(header.get("NAXIS")) or 0
        shape = []
        for idx in range(int(naxis), 0, -1):
            axis_len = CubeWorkbenchService._safe_int(header.get(f"NAXIS{idx}"))
            if axis_len:
                shape.append(axis_len)
        if shape:
            result["shape"] = shape
            result["raw_shape"] = shape
            if len(shape) >= 3:
                result["cube_shape"] = shape
                result["channel_count"] = shape[-3]
        result["unit"] = str(header.get("BUNIT") or result.get("bunit") or "")
        result["header_hdu_type"] = str(header.get("XTENSION") or "PRIMARY")
        return result

    @staticmethod
    def _first_image_header_from_fits_bytes(raw: bytes) -> Optional[Dict[str, Any]]:
        from astropy.io import fits

        offset = 0
        hdu_index = 0
        while offset + 2880 <= len(raw) and hdu_index < 32:
            parsed = CubeWorkbenchService._parse_fits_header_block(raw, offset, fits)
            if parsed is None:
                return None
            header, header_end = parsed
            naxis = CubeWorkbenchService._safe_int(header.get("NAXIS")) or 0
            has_image_axes = (
                naxis >= 2
                and CubeWorkbenchService._safe_int(header.get("NAXIS1")) is not None
                and CubeWorkbenchService._safe_int(header.get("NAXIS2")) is not None
            )
            data_bytes = CubeWorkbenchService._fits_data_size_bytes(header)
            if has_image_axes:
                return {
                    "header": header,
                    "hdu_index": hdu_index,
                    "header_offset_bytes": offset,
                    "header_block_bytes": header_end - offset,
                    "data_offset_bytes": header_end,
                    "data_bytes_estimated": data_bytes,
                }
            next_offset = header_end + CubeWorkbenchService._fits_padded_size(data_bytes)
            if next_offset <= offset or next_offset > len(raw):
                return None
            offset = next_offset
            hdu_index += 1
        return None

    @staticmethod
    def _parse_fits_header_block(raw: bytes, offset: int, fits_module: Any) -> Optional[tuple[Any, int]]:
        for cursor in range(offset, len(raw) - 79, 80):
            card = raw[cursor:cursor + 80]
            if card.startswith(b"END"):
                header_end = CubeWorkbenchService._fits_padded_size(cursor + 80 - offset) + offset
                if header_end > len(raw):
                    return None
                header_text = raw[offset:header_end].decode("ascii", errors="ignore")
                return fits_module.Header.fromstring(header_text, sep=""), header_end
        return None

    @staticmethod
    def _fits_data_size_bytes(header: Any) -> int:
        naxis = CubeWorkbenchService._safe_int(header.get("NAXIS")) or 0
        if naxis <= 0:
            return 0
        gcount = CubeWorkbenchService._safe_int(header.get("GCOUNT")) or 1
        pcount = CubeWorkbenchService._safe_int(header.get("PCOUNT")) or 0
        xtension = str(header.get("XTENSION") or "").upper()
        if xtension in {"BINTABLE", "TABLE"}:
            row_bytes = CubeWorkbenchService._safe_int(header.get("NAXIS1")) or 0
            row_count = CubeWorkbenchService._safe_int(header.get("NAXIS2")) or 0
            return max(0, (row_bytes * row_count + pcount) * gcount)
        bitpix = abs(CubeWorkbenchService._safe_int(header.get("BITPIX")) or 0)
        bytes_per_value = max(0, bitpix // 8)
        values = 1
        for idx in range(1, naxis + 1):
            axis = CubeWorkbenchService._safe_int(header.get(f"NAXIS{idx}")) or 0
            values *= max(0, axis)
        return max(0, values * bytes_per_value * gcount + pcount)

    @staticmethod
    def _fits_padded_size(size_bytes: int) -> int:
        size = max(0, int(size_bytes or 0))
        return int(math.ceil(size / 2880.0) * 2880) if size else 0

    @staticmethod
    def _local_path_from_source(source_url: str) -> Optional[Path]:
        parsed = urlparse(source_url)
        if parsed.scheme == "file":
            src_path = parsed.path
            if os.name == "nt" and src_path.startswith("/") and len(src_path) > 2 and src_path[2] == ":":
                src_path = src_path[1:]
            return Path(src_path)
        if parsed.scheme in {"", None}:
            path = Path(source_url)
            if path.exists():
                return path
        return None

    @staticmethod
    def _infer_archive(url: str) -> str:
        host = urlparse(url).hostname or ""
        host = host.lower()
        if "almascience" in host:
            return "ALMA"
        if "cadc" in host or "hia-iha" in host:
            return "CADC"
        if "mast" in host or "stsci" in host:
            return "MAST"
        if "eso" in host:
            return "ESO"
        return host or "unknown"

    @staticmethod
    def _build_evidence(source_url: str, archive: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        headers = metadata.get("all_headers") if isinstance(metadata, dict) else {}
        inspected = sorted(headers.keys())[:80] if isinstance(headers, dict) else []
        success = bool(metadata.get("success")) if isinstance(metadata, dict) else False
        warnings = []
        if not success:
            warnings.append(str(metadata.get("error") or "FITS header metadata could not be read"))
        if metadata.get("range_truncated"):
            warnings.append("The remote server did not honor the range request; Quasar stopped after the bounded header byte limit.")
        operations = ["session_create", "header_inspection"]
        assumptions = [
            "Session metadata is initialized from FITS headers when available.",
            "Full-resolution cube access is deferred until the user prepares the product or a preview product is generated.",
        ]
        if metadata.get("range_header_read"):
            operations.append("range_header_inspection")
            assumptions.append(
                "Remote workbench metadata was read from bounded FITS header bytes; image/cube pixel data was not downloaded."
            )
        if metadata.get("local_header_read"):
            operations.append("local_header_inspection")
        if metadata.get("data_hdu_index") not in (None, 0):
            operations.append("extension_hdu_header_detection")
        return {
            "source_url": source_url,
            "archive": archive,
            "headers_inspected": inspected,
            "wcs_status": "present" if headers and headers.get("CTYPE1") and headers.get("CTYPE2") else "unknown",
            "spectral_axis_status": "present" if headers and headers.get("CTYPE3") else "unknown",
            "header_strategy": metadata.get("header_strategy") or ("local_header" if metadata.get("local_header_read") else "metadata_service"),
            "header_bytes_fetched": metadata.get("header_bytes_fetched"),
            "header_range_limit_bytes": metadata.get("header_range_limit_bytes"),
            "range_status_code": metadata.get("range_status_code"),
            "data_hdu_index": metadata.get("data_hdu_index"),
            "data_bytes_estimated": metadata.get("data_bytes_estimated"),
            "operations": operations,
            "assumptions": assumptions,
            "warnings": warnings,
            "confidence": "high" if success and metadata.get("range_header_read") else ("medium" if success else "low"),
        }

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _metadata_headers(metadata: Dict[str, Any]) -> Dict[str, Any]:
        headers = metadata.get("all_headers") if isinstance(metadata, dict) else {}
        return headers if isinstance(headers, dict) else {}

    def _spectral_axis(
        self,
        headers: Dict[str, Any],
        metadata: Dict[str, Any],
        *,
        max_points: int = 512,
    ) -> Dict[str, Any]:
        nchan = self._safe_int(headers.get("NAXIS3")) or self._safe_int(metadata.get("channel_count"))
        rest_freq = self._safe_float(metadata.get("rest_freq_ghz"))
        ctype = str(headers.get("CTYPE3") or "Channel")
        cunit = str(headers.get("CUNIT3") or "")
        crval = self._safe_float(headers.get("CRVAL3"))
        cdelt = self._safe_float(headers.get("CDELT3"))
        crpix = self._safe_float(headers.get("CRPIX3")) or 1.0

        if not nchan:
            shape = self._shape_from_metadata(metadata, headers)
            if len(shape) >= 3:
                nchan = shape[-3]
        if not nchan:
            nchan = 1

        unit = cunit or ("GHz" if rest_freq is not None else "channel")
        scale = 1.0
        if unit.lower() in {"hz", "hertz"}:
            unit = "GHz"
            scale = 1e-9
        elif unit.lower() in {"mhz"}:
            unit = "GHz"
            scale = 1e-3

        step = max(1, int((nchan + max(1, int(max_points or 512)) - 1) / max(1, int(max_points or 512))))
        indices = list(range(0, int(nchan), step))
        if indices[-1] != int(nchan) - 1:
            indices.append(int(nchan) - 1)

        values: List[float]
        if crval is not None and cdelt is not None:
            values = [round((crval + (idx + 1 - crpix) * cdelt) * scale, 9) for idx in indices]
            label = f"{ctype} ({unit})" if unit else ctype
        elif rest_freq is not None:
            values = [rest_freq for _ in indices]
            label = "RESTFRQ (GHz)"
            unit = "GHz"
        else:
            values = [float(idx) for idx in indices]
            label = "Channel"
            unit = "channel"

        return {
            "label": label,
            "unit": unit,
            "ctype": ctype,
            "channel_count": int(nchan),
            "downsample_step": step,
            "values": values,
            "indices": indices,
            "rest_frequency_ghz": rest_freq,
        }

    @staticmethod
    def _shape_from_metadata(metadata: Dict[str, Any], headers: Dict[str, Any]) -> List[int]:
        for key in ("cube_shape", "raw_shape", "shape"):
            value = metadata.get(key)
            if isinstance(value, list):
                return [int(v) for v in value if isinstance(v, (int, float))]
        naxis = CubeWorkbenchService._safe_int(headers.get("NAXIS"))
        if naxis:
            shape = []
            for idx in range(int(naxis), 0, -1):
                axis_len = CubeWorkbenchService._safe_int(headers.get(f"NAXIS{idx}"))
                if axis_len:
                    shape.append(axis_len)
            if shape:
                return shape
        return []

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _json_float(value: Any) -> Optional[float]:
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _downsample_indices(length: int, *, max_points: int) -> List[int]:
        if length <= 0:
            return []
        step = max(1, int(math.ceil(length / max(1, int(max_points or 512)))))
        indices = list(range(0, length, step))
        if indices[-1] != length - 1:
            indices.append(length - 1)
        return indices

    @staticmethod
    def _clean_path(path: List[Dict[str, float]]) -> List[Dict[str, float]]:
        clean = []
        for point in path:
            if not isinstance(point, dict):
                continue
            x = CubeWorkbenchService._safe_float(point.get("x"))
            y = CubeWorkbenchService._safe_float(point.get("y"))
            if x is None or y is None:
                continue
            clean.append({"x": round(x, 3), "y": round(y, 3)})
        return clean[:16]

    @staticmethod
    def _line_preset(key: Optional[str]) -> Optional[Dict[str, Any]]:
        if not key:
            return None
        normalized = str(key).strip().lower()
        for preset in COMMON_LINE_PRESETS:
            if preset["key"].lower() == normalized:
                return dict(preset)
        return None

    @staticmethod
    def _clean_region(region: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if not isinstance(region, dict):
            return None
        x1 = CubeWorkbenchService._safe_float(region.get("x1"))
        y1 = CubeWorkbenchService._safe_float(region.get("y1"))
        x2 = CubeWorkbenchService._safe_float(region.get("x2"))
        y2 = CubeWorkbenchService._safe_float(region.get("y2"))
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return None
        left, right = sorted((x1, x2))
        bottom, top = sorted((y1, y2))
        if right <= left or top <= bottom:
            return None
        return {
            "x1": round(left, 3),
            "y1": round(bottom, 3),
            "x2": round(right, 3),
            "y2": round(top, 3),
        }

    @staticmethod
    def _estimate_path_samples(path: List[Dict[str, float]], *, max_points: int = 512) -> int:
        if len(path) < 2:
            return 0
        length = 0.0
        for start, end in zip(path, path[1:]):
            dx = float(end["x"]) - float(start["x"])
            dy = float(end["y"]) - float(start["y"])
            length += (dx * dx + dy * dy) ** 0.5
        return max(2, min(int(length) + 1, max(16, min(int(max_points or 512), 4096))))

    @staticmethod
    def _operation_evidence(
        session: Dict[str, Any],
        operation: str,
        *,
        confidence: str,
        assumptions: List[str],
    ) -> Dict[str, Any]:
        base = dict(session.get("evidence") or {})
        operations = list(base.get("operations") or [])
        if operation not in operations:
            operations.append(operation)
        base["operations"] = operations
        base["assumptions"] = list(base.get("assumptions") or []) + assumptions
        base["confidence"] = confidence
        return base
