"""User-submitted issue-report endpoint (structured, linked to a chat run)."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from api.deps import _storage_executor, get_current_user, issue_report_service
from api.models import IssueReportCreateRequest

router = APIRouter()


@router.post("/api/issue-reports")
async def submit_issue_report(
    req: IssueReportCreateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Store a private structured issue report linked to a chat run."""
    user_id = current_user.get("sub", "")
    try:
        report = await asyncio.get_event_loop().run_in_executor(
            _storage_executor,
            lambda: issue_report_service.create_report(
                user_id=user_id,
                run_id=req.run_id,
                message_id=req.message_id,
                category=req.category,
                description=req.description,
                include_context=req.include_context,
                prompt_excerpt=req.prompt_excerpt or "",
                response_excerpt=req.response_excerpt or "",
                technical_context=req.technical_context or {},
            ),
        )
        return {"success": True, "report": report}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
