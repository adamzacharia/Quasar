"""Public analytics + response-feedback endpoints."""

import asyncio
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from starlette.requests import Request

from services.auth_cookie import read_auth_cookie
from api.deps import (
    _executor,
    _latest_traces,
    _resolve_optional_user,
    _safe_authorization_header,
    _storage_executor,
    analytics_service,
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
async def submit_feedback(req: Request, authorization: Optional[str] = Header(None)):
    """Persist a like/dislike on an assistant response (public, optional auth)."""
    body = await req.json()
    message_id = body.get("message_id", "")
    feedback = body.get("feedback", "")  # "like" or "dislike"
    if feedback not in ("like", "dislike") or not message_id:
        raise HTTPException(status_code=400, detail="message_id and feedback ('like'/'dislike') required")

    auth_header = _safe_authorization_header(authorization)
    current_user = _resolve_optional_user(auth_header, cookie_token=read_auth_cookie(req))
    user_id = current_user.get("sub") if current_user else "anonymous"

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        _executor,
        lambda: analytics_service.log_feedback(
            message_id=message_id,
            feedback=feedback,
            conversation_id=body.get("conversation_id", ""),
            user_id=user_id,
            model=body.get("model", ""),
            prompt_preview=body.get("prompt_preview", ""),
            response_preview=body.get("response_preview", ""),
        ),
    )

    # ── Langfuse: ingest user feedback score in real-time ──────────
    from core.langfuse_integration import get_langfuse
    lf_client = get_langfuse()
    if lf_client:
        try:
            conv_id = body.get("conversation_id", "")
            run_id = body.get("run_id", "")
            trace_id = None
            if run_id and current_user:
                run = await loop.run_in_executor(
                    _storage_executor,
                    lambda: issue_report_service.get_run(run_id, user_id=user_id),
                )
                trace_id = run.get("trace_id") if run else None
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

    return {"success": True, "feedback": feedback}
