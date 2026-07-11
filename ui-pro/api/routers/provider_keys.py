"""Provider BYOK key management + usage-quota endpoints."""

import os

from fastapi import APIRouter, Depends, HTTPException

from api.deps import (
    LLMClient,
    ProviderKeyError,
    _current_user_email,
    get_current_user,
    provider_key_service,
    redact_secrets,
    usage_quota_service,
)
from api.models import ProviderKeyLimitRequest, ProviderKeySaveRequest

router = APIRouter()


def _test_provider_key(provider: str, api_key: str) -> None:
    """Make a tiny provider call to validate a stored BYOK key."""
    provider = "google" if provider == "gemini" else provider
    test_models = {
        "openai": os.getenv("OPENAI_KEY_TEST_MODEL", "gpt-4o-mini"),
        "deepseek": os.getenv("DEEPSEEK_KEY_TEST_MODEL", "deepseek-chat"),
        "anthropic": os.getenv("ANTHROPIC_KEY_TEST_MODEL", "claude-haiku-4-5"),
        "google": os.getenv("GEMINI_KEY_TEST_MODEL", "gemini-1.5-flash"),
        "tacc": os.getenv("TACC_KEY_TEST_MODEL", "Meta-Llama-3.2-1B-Instruct"),
    }
    model = test_models.get(provider)
    if not model:
        raise ValueError(f"Unsupported provider '{provider}'.")

    client = LLMClient(
        model=model,
        provider_api_keys={provider: api_key},
        key_source_by_provider={provider: "byok"},
    )
    client.responses.create(
        model=model,
        input="Reply with ok.",
        max_output_tokens=2,
        temperature=0,
    )


@router.get("/api/provider-keys")
async def list_provider_keys(current_user: dict = Depends(get_current_user)):
    return {"keys": provider_key_service.list_keys(current_user["sub"])}


@router.post("/api/provider-keys")
async def save_provider_key(req: ProviderKeySaveRequest, current_user: dict = Depends(get_current_user)):
    try:
        metadata = provider_key_service.save_key(
            current_user["sub"],
            req.provider,
            req.api_key,
            token_limit=req.token_limit,
        )
        return {"status": "success", "key": metadata}
    except ProviderKeyError as exc:
        raise HTTPException(status_code=400, detail=redact_secrets(exc))


@router.post("/api/provider-keys/{provider}/test")
async def test_provider_key(provider: str, current_user: dict = Depends(get_current_user)):
    user_id = current_user["sub"]
    try:
        api_key = provider_key_service.decrypt_key(user_id, provider)
        if not api_key:
            raise HTTPException(status_code=404, detail="Provider key not found")
        try:
            _test_provider_key(provider, api_key)
        except Exception as exc:
            provider_key_service.mark_test_result(user_id, provider, False)
            raise HTTPException(
                status_code=400,
                detail=f"Provider key test failed: {redact_secrets(exc)}",
            )
        metadata = provider_key_service.mark_test_result(user_id, provider, True)
        return {"status": "success", "key": metadata}
    except HTTPException:
        raise
    except ProviderKeyError as exc:
        raise HTTPException(status_code=400, detail=redact_secrets(exc))


@router.patch("/api/provider-keys/{provider}/limit")
async def update_provider_key_limit(
    provider: str,
    req: ProviderKeyLimitRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        metadata = provider_key_service.set_token_limit(
            current_user["sub"],
            provider,
            req.token_limit,
        )
        return {"status": "success", "key": metadata}
    except ProviderKeyError as exc:
        raise HTTPException(status_code=400, detail=redact_secrets(exc))


@router.delete("/api/provider-keys/{provider}")
async def delete_provider_key(provider: str, current_user: dict = Depends(get_current_user)):
    try:
        if provider_key_service.delete_key(current_user["sub"], provider):
            return {"status": "success"}
    except ProviderKeyError as exc:
        raise HTTPException(status_code=400, detail=redact_secrets(exc))
    raise HTTPException(status_code=404, detail="Provider key not found")


@router.get("/api/usage-quota")
async def get_usage_quota(current_user: dict = Depends(get_current_user)):
    metadata = provider_key_service.list_keys(current_user["sub"])
    return usage_quota_service.usage_summary(
        user_id=current_user["sub"],
        user_email=_current_user_email(current_user),
        provider_key_metadata=metadata,
    )
