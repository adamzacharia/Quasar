"""Admin-only endpoints: issue-report management + analytics/feedback exports."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, Response

from api.deps import (
    _current_user_email,
    _storage_executor,
    analytics_service,
    conversation_service,
    get_current_user,
    is_admin_email,
    issue_report_service,
    usage_quota_service,
)
from api.models import IssueReportUpdateRequest
from services.eval_export_service import build_eval_export, to_jsonl

router = APIRouter()


@router.get("/api/admin/feedback/snapshots/{snapshot_id}")
async def admin_feedback_snapshot(snapshot_id: str, export: bool = False,
                                  current_user: dict = Depends(get_current_user)):
    """Admin-only immutable evidence; retrieval never consults the live chat."""
    if not is_admin_email(_current_user_email(current_user)):
        raise HTTPException(status_code=403, detail="Admin access required")
    snapshot = await asyncio.get_event_loop().run_in_executor(
        _storage_executor, lambda: issue_report_service.snapshots.get(snapshot_id))
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Feedback snapshot not found")
    from services.feedback_snapshot_service import snapshot_json
    headers = {"Cache-Control": "no-store"}
    if export:
        import uuid
        safe_id = str(uuid.UUID(snapshot["id"]))
        headers["Content-Disposition"] = f'attachment; filename="quasar-feedback-{safe_id}.json"'
    return Response(snapshot_json(snapshot), media_type="application/json", headers=headers)


@router.get("/api/admin/issue-reports")
async def admin_issue_reports(
    status: str = "",
    provider: str = "",
    model: str = "",
    category: str = "",
    date_from: str = "",
    date_to: str = "",
    limit: int = 200,
    current_user: dict = Depends(get_current_user),
):
    """List private issue reports with admin filters.

    Each report carries the reporter's question/answer/context excerpts (when
    they consented) and the thumbs vote on the same message, so the admin
    panel can show WHAT was wrong, not just that something was.
    """
    if not is_admin_email(_current_user_email(current_user)):
        raise HTTPException(status_code=403, detail="Admin access required")

    def _load() -> list:
        reports = issue_report_service.list_reports(
            status=status,
            provider=provider,
            model=model,
            category=category,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )
        try:
            # run_id is the stable key (a message's client id changes between
            # the live turn and a reload); message_id is the fallback for old rows.
            votes_by_run = analytics_service.get_feedback_for_runs(
                [r.get("run_id") for r in reports], by_user=True
            )
            votes_by_message = analytics_service.get_feedback_for_messages(
                [r.get("message_id") for r in reports], by_user=True
            )
        except Exception:  # the vote badge is decoration; never fail the listing
            votes_by_run, votes_by_message = {}, {}
        for report in reports:
            report["vote"] = (
                votes_by_run.get((str(report.get("user_id") or ""), str(report.get("run_id") or "")))
                or votes_by_message.get((str(report.get("user_id") or ""), str(report.get("message_id") or "")))
                or ""
            )
        return reports

    # Blocking DB (possibly Turso network) reads stay off the event loop (CX-01).
    reports = await asyncio.get_event_loop().run_in_executor(_storage_executor, _load)
    return {"reports": reports}


@router.patch("/api/admin/issue-reports/{report_id}")
async def admin_update_issue_report(
    report_id: str,
    req: IssueReportUpdateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Update private report triage status or admin notes."""
    if not is_admin_email(_current_user_email(current_user)):
        raise HTTPException(status_code=403, detail="Admin access required")
    try:
        return issue_report_service.update_report(
            report_id,
            status=req.status,
            admin_notes=req.admin_notes,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/api/admin/issue-reports/export")
async def admin_export_issue_reports(
    format: str = "csv",
    status: str = "",
    provider: str = "",
    model: str = "",
    category: str = "",
    date_from: str = "",
    date_to: str = "",
    current_user: dict = Depends(get_current_user),
):
    """Export private issue reports as CSV or JSON."""
    if not is_admin_email(_current_user_email(current_user)):
        raise HTTPException(status_code=403, detail="Admin access required")
    filters = {
        "status": status,
        "provider": provider,
        "model": model,
        "category": category,
        "date_from": date_from,
        "date_to": date_to,
    }
    if format.lower() == "json":
        return issue_report_service.list_reports(limit=1000, **filters)
    csv_data = issue_report_service.export_reports_csv(**filters)
    return StreamingResponse(
        iter([csv_data]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=quasar_issue_reports.csv"},
    )


@router.get("/api/admin/feedback/export")
async def admin_feedback_export(current_user: dict = Depends(get_current_user)):
    """Export all response feedback data. Admin-only."""
    user_email = current_user.get("email", "")
    if not analytics_service.is_admin(user_email):
        raise HTTPException(status_code=403, detail="Admin access required")
    return analytics_service.export_feedback_json()


@router.get("/api/admin/feedback/recent")
async def admin_feedback_recent(
    feedback: str = "dislike",
    limit: int = 50,
    current_user: dict = Depends(get_current_user),
):
    """Recent thumbs votes with their question/answer previews. Admin-only.

    A thumbs-down that never became a full issue report was previously
    invisible except inside the JSON export. ``has_report`` marks votes that
    do have a report, so the panel can point at it instead of repeating it.
    """
    if not is_admin_email(_current_user_email(current_user)):
        raise HTTPException(status_code=403, detail="Admin access required")
    kind = feedback if feedback in ("like", "dislike") else None

    def _load() -> list:
        rows = analytics_service.list_recent_feedback(feedback=kind, limit=limit)
        try:
            reported = issue_report_service.reports_exist_for(
                message_ids=[row.get("message_id") for row in rows],
                run_ids=[row.get("run_id") for row in rows],
                by_user=True,
            )
        except Exception:
            reported = {"message_ids": set(), "run_ids": set()}
        for row in rows:
            row["has_report"] = (
                (str(row.get("user_id") or ""), str(row.get("run_id") or "")) in reported["run_ids"]
                or (str(row.get("user_id") or ""), str(row.get("message_id") or "")) in reported["message_ids"]
            )
        return rows

    rows = await asyncio.get_event_loop().run_in_executor(_storage_executor, _load)
    return {"feedback": rows}


@router.get("/api/admin/eval/export")
async def admin_eval_export(
    format: str = "jsonl",
    user_id: str = "",
    current_user: dict = Depends(get_current_user),
):
    """Labeled eval dataset: block ratings joined to prompt/query/tokens/cost.

    A human-label sidecar for DataLabBench — each row scores one block of one
    real turn, so an automated rubric run can be measured against human
    judgement. Admin-only. `user_id` scopes to one rater.
    """
    user_email = current_user.get("email", "")
    if not analytics_service.is_admin(user_email):
        raise HTTPException(status_code=403, detail="Admin access required")

    rows = build_eval_export(
        analytics_service,
        conversation_service,
        issue_report_service,
        user_id=user_id or None,
    )

    if format == "json":
        return rows

    return StreamingResponse(
        iter([to_jsonl(rows)]),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": "attachment; filename=quasar_eval_labels.jsonl"},
    )


@router.get("/api/admin/analytics/export")
async def admin_analytics_export(
    format: str = "csv",
    current_user: dict = Depends(get_current_user),
):
    """Download all chat analytics as CSV or JSON. Admin-only."""
    user_email = current_user.get("email", "")
    if not analytics_service.is_admin(user_email):
        raise HTTPException(status_code=403, detail="Admin access required")

    if format == "json":
        return analytics_service.export_chat_analytics_json()

    # Default: CSV download
    csv_data = analytics_service.export_chat_analytics_csv()
    return StreamingResponse(
        iter([csv_data]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=quasar_analytics.csv"},
    )


@router.get("/api/admin/analytics/summary")
async def admin_analytics_summary(current_user: dict = Depends(get_current_user)):
    """Aggregated analytics stats. Admin-only."""
    user_email = current_user.get("email", "")
    if not analytics_service.is_admin(user_email):
        raise HTTPException(status_code=403, detail="Admin access required")
    return analytics_service.get_summary()


@router.get("/api/admin/usage")
async def admin_usage_rollup(
    since: str = "",
    group_by: str = "provider",
    source: str = "calls",
    current_user: dict = Depends(get_current_user),
):
    """Token + cost rollup across all users. Admin-only.

    Two sources:
      - `calls` (default): the PER-CALL ledger (llm_usage_events). Attributes
        each LLM call to the model/provider/key_source that actually ran it, so
        an auxiliary call (e.g. a platform OpenAI embedding inside a BYOK turn)
        is counted under its real route, not the turn's selected one. This is
        the authoritative view for platform-vs-BYOK spend.
        `group_by` ∈ provider | model | key_source | user.
      - `turns`: the PER-TURN ledger (chat_runs), one row per turn keyed by the
        selected route — useful for per-question cost-efficiency. Its groups are
        keyed by the turn's MAIN route, so read the cost fields deliberately:
        `cost_usd` is the turn's full cost whoever paid (including the user's
        BYOK spend), while `platform_cost_usd` is the per-call-attributed subset
        Quasar paid and is the one to bill against (CX-28).
        `group_by` ∈ provider | model | key_source | user.

    `since` is an ISO-8601 timestamp.
    """
    if not is_admin_email(_current_user_email(current_user)):
        raise HTTPException(status_code=403, detail="Admin access required")
    if source not in ("calls", "turns"):
        # Allowlist: a typo must be a clear 400, not a silent fall-through to
        # the per-call report (CX-16).
        raise HTTPException(
            status_code=400, detail="source must be 'calls' or 'turns'"
        )
    try:
        if source == "turns":
            return issue_report_service.usage_rollup(
                since=since or None, group_by=group_by
            )
        return usage_quota_service.global_usage_rollup(
            since=since or None, group_by=group_by
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
