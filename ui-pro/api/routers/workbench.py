"""Cube/product Workbench endpoints (durable sessions, jobs, render plans)."""

from fastapi import APIRouter, Depends, HTTPException

from api.deps import (
    CubeWorkbenchForbidden,
    CubeWorkbenchNotFound,
    _workbench_user_id,
    cube_workbench_service,
    get_current_user,
)
from api.models import (
    WorkbenchExportRequest,
    WorkbenchJobStartRequest,
    WorkbenchLineOverlayRequest,
    WorkbenchPrepareRequest,
    WorkbenchPvSliceRequest,
    WorkbenchRenderRequest,
    WorkbenchSessionCreate,
    WorkbenchSpectrumRequest,
)

router = APIRouter()


def _raise_workbench_error(exc: Exception):
    if isinstance(exc, CubeWorkbenchNotFound):
        raise HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, CubeWorkbenchForbidden):
        raise HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc))
    raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/workbench/session")
async def create_workbench_session(
    req: WorkbenchSessionCreate,
    current_user: dict = Depends(get_current_user),
):
    """Create a durable first-class cube/product workbench session."""
    try:
        return cube_workbench_service.create_session(
            user_id=_workbench_user_id(current_user),
            source_url=req.source_url,
            filename=req.filename,
            project_code=req.project_code,
            mous_uid=req.mous_uid,
        )
    except Exception as exc:
        _raise_workbench_error(exc)


@router.get("/api/workbench/{session_id}/metadata")
async def get_workbench_metadata(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return FITS/workbench metadata, state, and evidence for a session."""
    try:
        return cube_workbench_service.get_metadata(
            session_id=session_id,
            user_id=_workbench_user_id(current_user),
        )
    except Exception as exc:
        _raise_workbench_error(exc)


@router.post("/api/workbench/{session_id}/jobs")
async def start_workbench_job(
    session_id: str,
    req: WorkbenchJobStartRequest,
    current_user: dict = Depends(get_current_user),
):
    """Start a durable asynchronous workbench operation."""
    try:
        return cube_workbench_service.start_job(
            session_id=session_id,
            user_id=_workbench_user_id(current_user),
            operation=req.operation,
            payload=req.payload,
        )
    except Exception as exc:
        _raise_workbench_error(exc)


@router.get("/api/workbench/{session_id}/jobs/{job_id}")
async def get_workbench_job(
    session_id: str,
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return durable progress/result state for a workbench job."""
    try:
        return cube_workbench_service.get_job(
            session_id=session_id,
            user_id=_workbench_user_id(current_user),
            job_id=job_id,
        )
    except Exception as exc:
        _raise_workbench_error(exc)


@router.delete("/api/workbench/{session_id}/jobs/{job_id}")
async def cancel_workbench_job(
    session_id: str,
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Request cancellation for a queued/running workbench job."""
    try:
        return cube_workbench_service.cancel_job(
            session_id=session_id,
            user_id=_workbench_user_id(current_user),
            job_id=job_id,
        )
    except Exception as exc:
        _raise_workbench_error(exc)


@router.post("/api/workbench/{session_id}/render")
async def plan_workbench_render(
    session_id: str,
    req: WorkbenchRenderRequest,
    current_user: dict = Depends(get_current_user),
):
    """Persist render controls for the future WCS-aware workbench renderer."""
    try:
        return cube_workbench_service.render_plan(
            session_id=session_id,
            user_id=_workbench_user_id(current_user),
            mode=req.mode or "image",
            channel=req.channel,
            moment=req.moment,
            colormap=req.colormap or "inferno",
            stretch=req.stretch or "asinh",
            contour_sigma=req.contour_sigma,
            rms_region=req.rms_region,
        )
    except Exception as exc:
        _raise_workbench_error(exc)


@router.post("/api/workbench/{session_id}/prepare")
async def prepare_workbench_product(
    session_id: str,
    req: WorkbenchPrepareRequest,
    current_user: dict = Depends(get_current_user),
):
    """Stage a FITS product in the bounded workbench cache for real cube operations."""
    try:
        return cube_workbench_service.prepare_product(
            session_id=session_id,
            user_id=_workbench_user_id(current_user),
            max_bytes=req.max_bytes or 500 * 1024 * 1024,
            user_cache_bytes=req.user_cache_bytes or 2 * 1024 * 1024 * 1024,
            cache_ttl_seconds=req.cache_ttl_seconds or 24 * 60 * 60,
            force=bool(req.force),
        )
    except Exception as exc:
        _raise_workbench_error(exc)


@router.post("/api/workbench/{session_id}/spectrum")
async def plan_workbench_spectrum(
    session_id: str,
    req: WorkbenchSpectrumRequest,
    current_user: dict = Depends(get_current_user),
):
    """Persist a spectrum extraction request and return spectral-axis metadata."""
    try:
        return cube_workbench_service.spectrum_plan(
            session_id=session_id,
            user_id=_workbench_user_id(current_user),
            x_pixel=req.x_pixel,
            y_pixel=req.y_pixel,
            aperture_radius_pixels=req.aperture_radius_pixels or 3.0,
            aperture_radius_arcsec=req.aperture_radius_arcsec,
            max_points=req.max_points or 512,
        )
    except Exception as exc:
        _raise_workbench_error(exc)


@router.post("/api/workbench/{session_id}/pv-slice")
async def plan_workbench_pv_slice(
    session_id: str,
    req: WorkbenchPvSliceRequest,
    current_user: dict = Depends(get_current_user),
):
    """Persist a PV-slice path and return axis planning metadata."""
    try:
        return cube_workbench_service.pv_slice_plan(
            session_id=session_id,
            user_id=_workbench_user_id(current_user),
            path=req.path,
            width_pixels=req.width_pixels or 3.0,
            max_points=req.max_points or 512,
        )
    except Exception as exc:
        _raise_workbench_error(exc)


@router.post("/api/workbench/{session_id}/line-overlays")
async def get_workbench_line_overlays(
    session_id: str,
    req: WorkbenchLineOverlayRequest,
    current_user: dict = Depends(get_current_user),
):
    """Return redshift-aware Splatalogue line overlays for a workbench session."""
    try:
        return cube_workbench_service.line_overlays(
            session_id=session_id,
            user_id=_workbench_user_id(current_user),
            observed_frequency_ghz=req.observed_frequency_ghz,
            line_preset_key=req.line_preset_key,
            redshift=req.redshift or 0.0,
            tolerance_ghz=req.tolerance_ghz or 0.01,
            top_n=req.top_n or 8,
        )
    except Exception as exc:
        _raise_workbench_error(exc)


@router.get("/api/workbench/line-presets")
async def get_workbench_line_presets():
    """Return common ALMA/mm spectral-line presets for workbench line ID."""
    return cube_workbench_service.line_presets()


@router.post("/api/workbench/{session_id}/export")
async def get_workbench_exports(
    session_id: str,
    req: WorkbenchExportRequest,
    current_user: dict = Depends(get_current_user),
):
    """Return reproducible export scripts/commands for a workbench session."""
    try:
        return cube_workbench_service.export_plan(
            session_id=session_id,
            user_id=_workbench_user_id(current_user),
            formats=req.formats,
        )
    except Exception as exc:
        _raise_workbench_error(exc)
