"""Per-user token + cost usage reporting."""

from fastapi import APIRouter, Depends

from api.deps import (
    _current_user_email,
    get_current_user,
    provider_key_service,
    usage_quota_service,
)

router = APIRouter()


@router.get("/api/usage/summary")
async def usage_summary(current_user: dict = Depends(get_current_user)):
    """Tokens + estimated cost for the caller: today, this week, cap headroom.

    Canonical usage endpoint. The older `/api/usage-quota` (provider_keys.py)
    returns the same `usage_quota_service.usage_summary()` payload and is kept
    for the existing SettingsModal fetch — both share the one service function,
    so the daily/cost fields land on both rather than drifting apart.
    """
    metadata = provider_key_service.list_keys(current_user["sub"])
    return usage_quota_service.usage_summary(
        user_id=current_user["sub"],
        user_email=_current_user_email(current_user),
        provider_key_metadata=metadata,
    )
