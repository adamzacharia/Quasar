"""General endpoints: model listing + Human-in-the-Loop plan feedback."""

import os

from fastapi import APIRouter, Depends, HTTPException

from api.deps import (
    ANTHROPIC_VISIBLE_MODEL_IDS,
    DEEPSEEK_VISIBLE_MODEL_IDS,
    OPENAI_VISIBLE_MODEL_IDS,
    TACC_MENU_MODEL_IDS,
    _plan_feedback_lock,
    _plan_feedback_queues,
    _unique_models,
    conversation_service,
    get_current_user,
    logger,
)
from api.models import PlanFeedbackRequest

router = APIRouter()


@router.get("/api/models")
async def list_models():
    cloud_models = _unique_models([
        *OPENAI_VISIBLE_MODEL_IDS,
        *DEEPSEEK_VISIBLE_MODEL_IDS,
        *ANTHROPIC_VISIBLE_MODEL_IDS,
        *TACC_MENU_MODEL_IDS,
    ])

    # ── Auto-discover local models (Ollama / LM Studio) ──────────────
    local_models = []
    local_base = os.getenv("LOCAL_LLM_BASE_URL", "")
    if local_base:
        try:
            import httpx
            resp = httpx.get(f"{local_base}/models", timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                for m in data.get("data", []):
                    model_id = m.get("id", "")
                    if model_id:
                        local_models.append(f"local/{model_id}")
        except Exception as e:
            logger.warning(f"[MODELS] Failed to discover local models: {e}")

    return {"models": _unique_models(cloud_models + local_models)}


# ── Plan Feedback (Human-in-the-Loop) ────────────────────────
@router.post("/api/plan-feedback")
async def submit_plan_feedback(req: PlanFeedbackRequest, current_user: dict = Depends(get_current_user)):
    """Receive user approval or feedback for a Conductor execution plan.

    Pushes the response into the plan_feedback_queue for the given
    conversation, unblocking the Conductor's orchestrate() method.

    UIAPI-04: requires authentication AND conversation ownership — every other
    state-changing route takes get_current_user, and without the ownership
    check anyone holding a leaked conversation UUID could approve another
    user's pending plan (spending their quota) or inject arbitrary feedback
    text into their agent's planning prompt.
    """
    user_id = str(current_user.get("sub") or "")
    if not user_id or not conversation_service.conversation_belongs_to_user(req.conversation_id, user_id):
        # Same 404 as "no pending review": do not confirm to a non-owner that
        # the conversation exists.
        raise HTTPException(
            status_code=404,
            detail=f"No pending plan review for conversation {req.conversation_id}",
        )

    # UIAPI-08: the registry is keyed by (conversation_key, run_id) so two
    # concurrent runs in one conversation cannot clobber each other. A caller
    # that names the run gets exactly that run's queue; a legacy caller
    # without run_id gets the conversation's most recent registration (dicts
    # preserve insertion order).
    with _plan_feedback_lock:
        if req.run_id:
            pfq = _plan_feedback_queues.get((req.conversation_id, req.run_id))
        else:
            pfq = None
            for (queue_conv_id, _queue_run_id), queue in _plan_feedback_queues.items():
                if queue_conv_id == req.conversation_id:
                    pfq = queue

    if not pfq:
        raise HTTPException(
            status_code=404,
            detail=f"No pending plan review for conversation {req.conversation_id}",
        )

    pfq.put({
        "approve": req.approve,
        "feedback": req.feedback,
    })

    action = "approved" if req.approve else f"feedback: {req.feedback[:80]}"
    logger.info(f"[HITL] Plan {action} for conversation {req.conversation_id}")
    return {"status": "ok", "action": "approved" if req.approve else "feedback_sent"}
