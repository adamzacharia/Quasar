"""Health check + service root endpoints."""

from fastapi import APIRouter

from api.deps import get_agent

router = APIRouter()


@router.get("/")
async def root():
    return {"status": "ok", "service": "QUASAR API", "version": "2.0.0"}


@router.get("/health")
async def health():
    agent = get_agent()
    return {
        "status": "healthy",
        "agent_loaded": agent is not None,
    }
