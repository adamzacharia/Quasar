"""Public analytics + response-feedback endpoints."""

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from starlette.requests import Request

from services.auth_cookie import read_auth_cookie
from api.deps import (
    _executor,
    _latest_traces,
    _resolve_optional_user,
    _safe_authorization_header,
    _storage_executor,
    analytics_service,
    conversation_service,
    get_current_user,
    issue_report_service,
)

router = APIRouter()


@router.get("/api/analytics/hit")
async def analytics_hit(req: Request):
    """Log a page view and return the current total hit count (public, no auth)."""
    ip = (req.headers.get("x-forwarded-for", "").split(",")[0].strip()
          or (req.client.host if req.client else ""))
    ua = req.headers.get("user-agent", "")
    country = req.headers.get("cf-ipcountry", "") or req.headers.get("x-vercel-ip-country", "")
    loop = asyncio.get_event_loop()
    total = await loop.run_in_executor(
        _executor,
        lambda: analytics_service.log_page_view(ip_address=ip, user_agent=ua, country=country),
    )

    # ── Langfuse: log page hit as a custom event ──────────────────
    from core.langfuse_integration import get_langfuse
    lf_client = get_langfuse()
    if lf_client:
        try:
            await loop.run_in_executor(
                _executor,
                lambda: lf_client.event(
                    name="page_view",
                    user_id="anonymous",
                    metadata={
                        "ip_address": ip,
                        "user_agent": ua,
                        "country": country,
                    }
                )
            )
        except Exception as lf_err:
            print(f"[Langfuse] Failed to log page hit: {lf_err}", flush=True)

    return {"hits": total}


@router.post("/api/feedback")
async def submit_feedback(req: Request, current_user: dict = Depends(get_current_user)):
    """Persist a like/dislike on an assistant response (authenticated).

    UIAPI-12: feedback writes require authentication. Anonymous writers all
    collided on one shared 'anonymous' user_id (overwriting each other's
    votes), and an unauthenticated script could pollute the human-label eval
    dataset behind /api/admin/eval/export with arbitrary rows.
    """
    body = await req.json()
    message_id = body.get("message_id", "")
    feedback = body.get("feedback", "")  # "like" or "dislike"
    if feedback not in ("like", "dislike") or not message_id:
        raise HTTPException(status_code=400, detail="message_id and feedback ('like'/'dislike') required")

    user_id = str(current_user.get("sub") or "")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    loop = asyncio.get_event_loop()
    # The run id is client-supplied. It is stored only when the run belongs to
    # the voter: otherwise anyone could attach a vote to someone else's run and
    # flip that report's admin vote badge or has_report flag (CX-02).
    claimed_run_id = str(body.get("run_id") or "")
    owned_run = None
    if claimed_run_id:
        try:
            owned_run = await loop.run_in_executor(
                _storage_executor,
                lambda: issue_report_service.get_run(claimed_run_id, user_id=user_id),
            )
        except Exception as exc:
            # Run store outage: keep the vote, drop the unverifiable link.
            print(f"[FEEDBACK] run ownership lookup failed; storing vote unlinked: {exc}", flush=True)
            owned_run = None
    run_id = claimed_run_id if owned_run else ""
    # The conversation id is client-supplied too. Take it from the owned run
    # when there is one; otherwise store it (and use it for the Langfuse
    # session fallback) only if the conversation belongs to the voter, so a
    # vote can neither be displayed under nor scored against someone else's
    # conversation.
    claimed_conv_id = str(body.get("conversation_id") or "")
    conv_id = str(owned_run.get("conversation_id") or "") if owned_run else ""
    if not conv_id and claimed_conv_id:
        try:
            owned_conv = await loop.run_in_executor(
                _storage_executor,
                lambda: conversation_service.conversation_belongs_to_user(claimed_conv_id, user_id),
            )
        except Exception:
            owned_conv = False
        conv_id = claimed_conv_id if owned_conv else ""
    # Only a literal affirmative boolean opts into context capture.
    include_context = body.get("include_context") is True

    def _save_feedback():
        snapshot, snapshot_error = (None, "")
        linked_snapshot_id = ""
        report_id = str(body.get("report_id") or "")
        linked_report = issue_report_service.get_report(report_id) if report_id else None
        if linked_report and (linked_report.get("user_id") != user_id
                              or linked_report.get("run_id") != run_id):
            linked_report = None
        if include_context and feedback == "dislike" and linked_report:
            linked_snapshot_id = linked_report.get("snapshot_id") or ""
            snapshot_error = linked_report.get("snapshot_error") or ""
        elif feedback == "dislike" and include_context:
            snapshot, snapshot_error = issue_report_service.build_feedback_snapshot(
                user_id=user_id, conversation_id=conv_id, message_id=message_id,
                run_id=run_id, include_context=True, description=body.get("description", ""))
            if snapshot:
                try:
                    issue_report_service.snapshots.insert(snapshot)
                except Exception:
                    snapshot = None
                    snapshot_error = "Conversation snapshot unavailable: storage write failed"
        analytics_service.log_feedback(
            message_id=message_id,
            feedback=feedback,
            conversation_id=conv_id,
            user_id=user_id,
            model=body.get("model", ""),
            prompt_preview=body.get("prompt_preview", "") if include_context else "",
            response_preview=body.get("response_preview", "") if include_context else "",
            run_id=run_id,
            snapshot_id=snapshot["id"] if snapshot else linked_snapshot_id,
            snapshot_error=snapshot_error,
        )
        return {"snapshot_id": snapshot["id"] if snapshot else linked_snapshot_id, "snapshot_error": snapshot_error}

    capture = await loop.run_in_executor(_storage_executor, _save_feedback)

    # ── Langfuse: ingest user feedback score in real-time ──────────
    from core.langfuse_integration import get_langfuse
    lf_client = get_langfuse()
    if lf_client:
        try:
            # Ownership was already established above; reuse that lookup and
            # the verified conversation id (never the raw client value).
            trace_id = owned_run.get("trace_id") if owned_run else None
            if not trace_id and conv_id:
                trace_id = _latest_traces.get(conv_id)

            # Score target: if trace_id is known, attach directly to the trace, else attach to session
            score_kwargs = {
                "name": "user_feedback",
                "value": 1.0 if feedback == "like" else 0.0,
                "data_type": "BOOLEAN",
                "comment": f"User voted {feedback} on message {message_id}",
            }
            if trace_id:
                score_kwargs["trace_id"] = trace_id
            elif conv_id:
                score_kwargs["session_id"] = conv_id

            if score_kwargs.get("trace_id") or score_kwargs.get("session_id"):
                await loop.run_in_executor(
                    _executor,
                    lambda: lf_client.score(**score_kwargs)
                )
                print(f"[Langfuse] Ingested user feedback score ({feedback}) for trace {score_kwargs.get('trace_id') or score_kwargs.get('session_id')}", flush=True)
        except Exception as lf_err:
            print(f"[Langfuse] Failed to ingest feedback score: {lf_err}", flush=True)

    return {"success": True, "feedback": feedback, **capture}


@router.post("/api/block-feedback")
async def submit_block_feedback(req: Request, current_user: dict = Depends(get_current_user)):
    """Persist a 1-5 star rating on ONE block (card) of an assistant turn.

    Sits alongside /api/feedback rather than replacing it: that endpoint is the
    binary like/dislike on the whole message and still backs the thumbs bar and
    the issue-report flow.

    UIAPI-12: authenticated writers only. The star UI only renders for
    signed-in users anyway, and these rows ARE the human-label eval dataset —
    anonymous writes let anyone stuff attacker-chosen labels into the export
    (and legitimate anonymous raters silently overwrote each other under the
    shared 'anonymous' user_id).
    """
    body = await req.json()
    block_id = str(body.get("block_id") or "")
    if not block_id:
        raise HTTPException(status_code=400, detail="block_id is required")

    # Strict rather than int()-coerced: a 4.9 must not become a 4-star label and
    # JSON `true` must not become a 1-star one. bool subclasses int, so exclude it.
    raw_rating = body.get("rating")
    if isinstance(raw_rating, bool) or not isinstance(raw_rating, int):
        raise HTTPException(status_code=400, detail="rating must be an integer 1-5")
    rating = raw_rating
    if not 1 <= rating <= 5:
        raise HTTPException(status_code=400, detail="rating must be between 1 and 5")

    user_id = str(current_user.get("sub") or "")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    conv_id = str(body.get("conversation_id") or "")
    run_id = str(body.get("run_id") or "")
    block_kind = str(body.get("block_kind") or "")
    comment = str(body.get("comment") or "")

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            _executor,
            lambda: analytics_service.log_block_feedback(
                block_id=block_id,
                rating=rating,
                run_id=run_id,
                conversation_id=conv_id,
                message_db_id=str(body.get("message_db_id") or ""),
                user_id=user_id,
                block_kind=block_kind,
                comment=comment,
                model=str(body.get("model") or ""),
            ),
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))

    # ── Langfuse: same scoring path as /api/feedback, but NUMERIC ──
    from core.langfuse_integration import get_langfuse
    lf_client = get_langfuse()
    if lf_client:
        try:
            trace_id = None
            if run_id and current_user:
                run = await loop.run_in_executor(
                    _storage_executor,
                    lambda: issue_report_service.get_run(run_id, user_id=user_id),
                )
                trace_id = run.get("trace_id") if run else None
            if not trace_id and conv_id:
                trace_id = _latest_traces.get(conv_id)

            score_kwargs = {
                "name": "block_rating",
                "value": float(rating),
                "data_type": "NUMERIC",
                "comment": comment or f"Rated {rating}/5 on {block_kind or 'block'} {block_id[:12]}",
            }
            if trace_id:
                score_kwargs["trace_id"] = trace_id
            elif conv_id:
                score_kwargs["session_id"] = conv_id

            if score_kwargs.get("trace_id") or score_kwargs.get("session_id"):
                await loop.run_in_executor(_executor, lambda: lf_client.score(**score_kwargs))
                print(f"[Langfuse] Ingested block rating ({rating}/5) for block {block_id[:12]}", flush=True)
        except Exception as lf_err:
            print(f"[Langfuse] Failed to ingest block rating: {lf_err}", flush=True)

    return {"success": True, "block_id": block_id, "rating": rating}


@router.get("/api/block-feedback")
async def get_block_feedback(req: Request, authorization: Optional[str] = Header(None)):
    """This user's block ratings for one conversation, so a reloaded page can
    re-light the stars it already earned."""
    conversation_id = req.query_params.get("conversation_id", "")
    if not conversation_id:
        raise HTTPException(status_code=400, detail="conversation_id is required")

    auth_header = _safe_authorization_header(authorization)
    current_user = _resolve_optional_user(auth_header, cookie_token=read_auth_cookie(req))
    user_id = current_user.get("sub") if current_user else "anonymous"

    loop = asyncio.get_event_loop()
    ratings = await loop.run_in_executor(
        _executor,
        lambda: analytics_service.get_block_ratings(conversation_id, user_id),
    )
    return {"ratings": ratings}
