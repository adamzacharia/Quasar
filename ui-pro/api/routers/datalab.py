"""NOIRLab Astro Data Lab background jobs (tiled scans + async queries) and
My-tables (durable named tables saved from result_ids)."""

import math
from numbers import Real
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.deps import get_current_user
from api.models import DatalabTiledSearchRequest

router = APIRouter()


def _json_safe_records(frame):
    """Convert DataFrame rows to strict-JSON scalars (no NaN or infinities)."""
    rows = frame.astype(object).where(frame.notna(), None).to_dict("records")
    return [
        {
            key: (
                None
                if isinstance(value, Real) and not math.isfinite(float(value))
                else value
            )
            for key, value in row.items()
        }
        for row in rows
    ]


class SaveMyTableRequest(BaseModel):
    result_id: str
    name: str
    description: Optional[str] = None


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

    job_id = _jobs().start(
        "tiled_sky_scan",
        _job,
        params={"catalog": req.catalog, "table": req.table, **fp},
        owner_id=str(current_user["sub"]),
    )
    return {"success": True, "job_id": job_id, **decision}


def _sync_external_records(service, records, owner_id, *, max_syncs: int = 5) -> None:
    """Live-sync non-terminal SERVER-side job records against the Data Lab job
    API (token-gated, best-effort, bounded) so the Jobs panel's 5s poll shows
    real state instead of the stale submit-time record. (guard CX-06)"""
    pending = [
        rec for rec in records
        if rec.get("external") and rec.get("status") not in {"succeeded", "failed", "canceled"}
    ][:max_syncs]
    if not pending:
        return
    try:
        from capabilities.datalab import _SERVER_STATE_MAP
        from integrations.datalab_client import ANON_TOKEN, DatalabClient
        client = DatalabClient()
        if client.token == ANON_TOKEN:
            return  # anon tokens get HTTP 401 from the server job API
        for rec in pending:
            try:
                state = str(client.status(rec["job_id"])).strip().upper()
                normalized = _SERVER_STATE_MAP.get(state, state.lower() or "unknown")
                service.update_external(rec["job_id"], owner_id=owner_id, status=normalized)
            except Exception:
                continue  # per-record best effort
    except Exception:
        pass


@router.get("/api/datalab/jobs")
async def list_datalab_jobs(current_user: dict = Depends(get_current_user)):
    """List known Data Lab jobs (local threaded + registered server-side), newest
    first. Non-terminal server records are live-synced first (bounded, token-gated)."""
    from starlette.concurrency import run_in_threadpool

    from services.datalab_job_service import default_job_service as _jobs
    service = _jobs()
    owner = str(current_user["sub"])
    await run_in_threadpool(
        _sync_external_records, service, service.list_jobs(owner_id=owner), owner
    )
    jobs = service.list_jobs(owner_id=owner)
    # Trim heavyweight result payloads for the panel — details come from the by-id route.
    slim = []
    for rec in jobs:
        rec = dict(rec)
        result = rec.get("result")
        if isinstance(result, dict):
            rec["result"] = {
                key: result[key]
                for key in ("result_id", "rowcount", "note", "candidates_found")
                if key in result
            } or {"available": True}
        elif result is not None:
            rec["result"] = {"available": True}
        rec.pop("params", None)
        slim.append(rec)
    return {"success": True, "jobs": slim, "count": len(slim)}


@router.get("/api/datalab/jobs/{job_id}")
async def get_datalab_job(job_id: str, current_user: dict = Depends(get_current_user)):
    """Poll a Data Lab background job's status / ranked results.

    Server-side records (kind=server_query) are synced against the live Data Lab
    job API when a real DATALAB_TOKEN is configured; without one the last-known
    local record is returned as-is."""
    from services.datalab_job_service import default_job_service as _jobs
    try:
        record = _jobs().status(job_id, owner_id=str(current_user["sub"]))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown Data Lab job: {job_id}")
    if record.get("external") and record.get("status") not in {"succeeded", "failed", "canceled"}:
        try:
            from capabilities.datalab import _SERVER_STATE_MAP
            from integrations.datalab_client import ANON_TOKEN, DatalabClient
            client = DatalabClient()
            if client.token != ANON_TOKEN:
                state = str(client.status(job_id)).strip().upper()
                normalized = _SERVER_STATE_MAP.get(state, state.lower() or "unknown")
                _jobs().update_external(
                    job_id,
                    owner_id=str(current_user["sub"]),
                    status=normalized,
                )
                record = _jobs().status(job_id, owner_id=str(current_user["sub"]))
        except Exception:
            pass  # live sync is best-effort; the stored record still answers
    return record


@router.delete("/api/datalab/jobs/{job_id}")
async def cancel_datalab_job(job_id: str, current_user: dict = Depends(get_current_user)):
    """Cancel a running Data Lab background job."""
    from services.datalab_job_service import default_job_service as _jobs
    try:
        record = _jobs().status(job_id, owner_id=str(current_user["sub"]))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown Data Lab job: {job_id}")
    if record.get("external"):
        try:
            from integrations.datalab_client import ANON_TOKEN, DatalabClient
            client = DatalabClient()
            if client.token == ANON_TOKEN:
                raise HTTPException(
                    status_code=503,
                    detail="Data Lab job cancellation requires a configured DATALAB_TOKEN",
                )
            client.abort(job_id)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Data Lab job cancellation failed: {exc}",
            ) from exc
        try:
            _jobs().update_external(
                job_id,
                owner_id=str(current_user["sub"]),
                status="canceled",
            )
            return _jobs().status(job_id, owner_id=str(current_user["sub"]))
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown Data Lab job: {job_id}")
    try:
        return _jobs().cancel(job_id, owner_id=str(current_user["sub"]))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown Data Lab job: {job_id}")


# ── My-tables (MyDB-lite) ─────────────────────────────────────────────────────
@router.get("/api/datalab/mytables")
async def list_my_tables(current_user: dict = Depends(get_current_user)):
    """List the saved Data Lab tables for the My tables panel."""
    from services.datalab_result_store import default_result_store
    tables = default_result_store().list_my_tables(user_id=str(current_user["sub"]))
    return {"success": True, "my_tables": tables, "count": len(tables)}


@router.post("/api/datalab/mytables")
async def save_my_table(req: SaveMyTableRequest, current_user: dict = Depends(get_current_user)):
    """Save a result_id as a durable named table."""
    from services.datalab_result_store import default_result_store
    try:
        entry = default_result_store().save_result(
            req.result_id,
            req.name,
            description=req.description,
            user_id=str(current_user["sub"]),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"success": True, "my_table": entry}


@router.get("/api/datalab/mytables/{name}")
async def get_my_table(name: str, max_rows: int = 50, current_user: dict = Depends(get_current_user)):
    """Preview a saved table's rows (capped) for the panel."""
    from services.datalab_result_store import default_result_store
    try:
        result = default_result_store().load_my_table(
            name, user_id=str(current_user["sub"])
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    cap = max(1, min(int(max_rows or 50), 500))
    frame = result.dataframe.head(cap)
    return {
        "success": True,
        "name": name,
        "rowcount": int(len(result.dataframe)),
        "columns": result.columns,
        "rows": _json_safe_records(frame),
        "provenance": result.provenance,
    }


@router.delete("/api/datalab/mytables/{name}")
async def delete_my_table(name: str, current_user: dict = Depends(get_current_user)):
    """Delete a saved table."""
    from services.datalab_result_store import default_result_store
    try:
        entry = default_result_store().delete_my_table(
            name, user_id=str(current_user["sub"])
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"success": True, "deleted": entry}
