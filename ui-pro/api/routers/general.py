"""General endpoints: model listing + Human-in-the-Loop plan feedback."""

import os

from fastapi import APIRouter, HTTPException

from api.deps import (
    ANTHROPIC_VISIBLE_MODEL_IDS,
    DEEPSEEK_VISIBLE_MODEL_IDS,
    OPENAI_VISIBLE_MODEL_IDS,
    TACC_MENU_MODEL_IDS,
    _plan_feedback_lock,
    _plan_feedback_queues,
    _unique_models,
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
async def submit_plan_feedback(req: PlanFeedbackRequest):
    """Receive user approval or feedback for a Conductor execution plan.

    Pushes the response into the plan_feedback_queue for the given
    conversation, unblocking the Conductor's orchestrate() method.
    """
    with _plan_feedback_lock:
        pfq = _plan_feedback_queues.get(req.conversation_id)

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
