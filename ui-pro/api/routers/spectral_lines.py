"""Spectral Line Explorer endpoints (Splatalogue metadata, resolve, jobs)."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from api.deps import (
    _workbench_user_id,
    get_current_user,
    spectral_line_explorer_enabled,
    spectral_line_job_service,
)
from api.models import SpectralLineJobRequest, SpectralTargetResolveRequest

router = APIRouter()


def _ensure_spectral_line_explorer_enabled():
    if not spectral_line_explorer_enabled():
        raise HTTPException(status_code=404, detail="Spectral Line Explorer is disabled")


def _raise_spectral_line_error(exc: Exception):
    if isinstance(exc, KeyError):
        raise HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc))
    raise HTTPException(status_code=500, detail=str(exc))


@router.get("/api/spectral-lines/metadata")
async def get_spectral_line_metadata(
    current_user: dict = Depends(get_current_user),
):
    """Return catalog, unit, ALMA-band, default, and limit metadata."""
    _ensure_spectral_line_explorer_enabled()
    return spectral_line_job_service.metadata()


@router.get("/api/spectral-lines/species")
async def search_spectral_line_species(
    query: str = "",
    limit: int = 25,
    current_user: dict = Depends(get_current_user),
):
    """Return cached Splatalogue species autocomplete records."""
    _ensure_spectral_line_explorer_enabled()
    try:
        return {
            "species": spectral_line_job_service.species.search(query, limit),
            "query": query,
        }
    except Exception as exc:
        _raise_spectral_line_error(exc)


@router.post("/api/spectral-lines/resolve-target")
async def resolve_spectral_line_target(
    req: SpectralTargetResolveRequest,
    current_user: dict = Depends(get_current_user),
):
    """Resolve target coordinates and redshift with SIMBAD/NED provenance.

    With ``include_sparcl`` the response also carries a best-effort NOIRLab
    SPARCL availability probe at the resolved position — the reverse
    SLE→SPARCL "optical spectrum available" chip."""
    _ensure_spectral_line_explorer_enabled()
    from starlette.concurrency import run_in_threadpool

    try:
        # SIMBAD/NED resolution is blocking network I/O — keep it off the event
        # loop (guard CX-10).
        resolution = await run_in_threadpool(
            spectral_line_job_service.resolver.resolve,
            req.target_name,
            explicit_redshift=req.redshift,
            explicit_ra_deg=req.ra_deg,
            explicit_dec_deg=req.dec_deg,
        )
    except Exception as exc:
        _raise_spectral_line_error(exc)
    if req.include_sparcl and isinstance(resolution, dict):
        ra = resolution.get("ra_deg")
        dec = resolution.get("dec_deg")
        if ra is not None and dec is not None:
            # Always set the key so the UI can tell "checked: none found"
            # (a dict with count 0) apart from "probe failed" (null) —
            # find_spectra reports failures as {"success": False}, not raises.
            sparcl_info = None
            try:
                from services.sparcl_spectra import SparclSpectraService

                # limit is applied to the remote BOUNDING-BOX query before the
                # local cone filter, so it must be generous — a tight limit can
                # falsely report "no spectra in the cone" (guard CX-10).
                probe = await run_in_threadpool(
                    SparclSpectraService().find_spectra, ra, dec,
                    radius_arcsec=10, limit=50,
                )
                if probe.get("success"):
                    rows = probe.get("rows") or []
                    sparcl_info = {
                        "count": len(rows),
                        "nearest": ({
                            "sparcl_id": rows[0].get("sparcl_id"),
                            "spectype": rows[0].get("spectype"),
                            "redshift": rows[0].get("redshift"),
                            "data_release": rows[0].get("data_release"),
                            "distance_arcsec": rows[0].get("distance_arcsec"),
                        } if rows else None),
                    }
            except Exception:
                sparcl_info = None  # availability is a nice-to-have; never fail resolution
            resolution["sparcl"] = sparcl_info
    return resolution


@router.post("/api/spectral-lines/jobs")
async def start_spectral_line_job(
    req: SpectralLineJobRequest,
    current_user: dict = Depends(get_current_user),
):
    """Start a catalog, exact ALMA coverage, or line-confusion job."""
    _ensure_spectral_line_explorer_enabled()
    try:
        return spectral_line_job_service.create_job(
            user_id=_workbench_user_id(current_user),
            operation=req.operation,
            payload=req.payload or {},
        )
    except Exception as exc:
        _raise_spectral_line_error(exc)


@router.get("/api/spectral-lines/jobs/{job_id}")
async def get_spectral_line_job(
    job_id: str,
    page: int = 1,
    page_size: int = 100,
    dataset: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    """Return job status, context, warnings, and one stable result page."""
    _ensure_spectral_line_explorer_enabled()
    try:
        return spectral_line_job_service.get_job(
            user_id=_workbench_user_id(current_user),
            job_id=job_id,
            page=page,
            page_size=page_size,
            dataset=dataset,
        )
    except Exception as exc:
        _raise_spectral_line_error(exc)


@router.delete("/api/spectral-lines/jobs/{job_id}")
async def cancel_spectral_line_job(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Cancel queued/running spectral-line work and stop future query segments."""
    _ensure_spectral_line_explorer_enabled()
    try:
        return spectral_line_job_service.cancel_job(
            user_id=_workbench_user_id(current_user),
            job_id=job_id,
        )
    except Exception as exc:
        _raise_spectral_line_error(exc)


@router.get("/api/spectral-lines/jobs/{job_id}/export")
async def export_spectral_line_job(
    job_id: str,
    dataset: str = "lines",
    format: str = "csv",
    current_user: dict = Depends(get_current_user),
):
    """Export the complete job dataset rather than only the current page."""
    _ensure_spectral_line_explorer_enabled()
    try:
        filename, media_type, content = spectral_line_job_service.export(
            user_id=_workbench_user_id(current_user),
            job_id=job_id,
            dataset=dataset,
            format_name=format,
        )
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        _raise_spectral_line_error(exc)
