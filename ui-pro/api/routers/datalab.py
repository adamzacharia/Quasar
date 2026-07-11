"""NOIRLab Astro Data Lab background jobs (tiled catalog-science scans)."""

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_current_user
from api.models import DatalabTiledSearchRequest

router = APIRouter()


@router.post("/api/datalab/jobs")
async def start_datalab_job(
    req: DatalabTiledSearchRequest,
    current_user: dict = Depends(get_current_user),
):
    """Start a background tiled Data Lab overdensity search.

    Returns ``needs_confirmation`` for a wide sky area unless ``confirm=true`` (guardrail:
    confirm the scan footprint before fanning out). Poll/cancel via the GET/DELETE routes.
    """
    from services import datalab_orchestration as _orch
    from services.datalab_job_service import default_job_service as _jobs

    fp = {"ra_min": req.ra_min, "ra_max": req.ra_max, "dec_min": req.dec_min, "dec_max": req.dec_max}
    decision = _orch.confirm_sky_area(fp, req.tile_radius_deg, max_tiles=req.max_tiles)
    if decision["needs_confirmation"] and not req.confirm:
        return {"success": False, "needs_confirmation": True, **decision}

    def _job(cancel_check):
        return _orch.tiled_sky_scan(
            req.catalog, req.table, fp, tile_radius_deg=req.tile_radius_deg, step_deg=req.step_deg,
            color_cut=req.color_cut, value_cuts=req.value_cuts, morphology=req.morphology,
            peak_threshold=req.peak_threshold, max_tiles=req.max_tiles,
            candidate_budget=req.candidate_budget, confirm=True, cancel_check=cancel_check,
        )

    job_id = _jobs().start("tiled_sky_scan", _job, params={"catalog": req.catalog, "table": req.table, **fp})
    return {"success": True, "job_id": job_id, **decision}


@router.get("/api/datalab/jobs/{job_id}")
async def get_datalab_job(job_id: str, current_user: dict = Depends(get_current_user)):
    """Poll a Data Lab background job's status / ranked results."""
    from services.datalab_job_service import default_job_service as _jobs
    try:
        return _jobs().status(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown Data Lab job: {job_id}")


@router.delete("/api/datalab/jobs/{job_id}")
async def cancel_datalab_job(job_id: str, current_user: dict = Depends(get_current_user)):
    """Cancel a running Data Lab background job."""
    from services.datalab_job_service import default_job_service as _jobs
    try:
        return _jobs().cancel(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown Data Lab job: {job_id}")
