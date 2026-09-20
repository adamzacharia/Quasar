"""Authenticated BYOK management and per-user model catalogs."""
from __future__ import annotations

import os
import httpx
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from api.deps import (
    _current_user_email, get_current_user, provider_catalog_service,
    provider_key_service, usage_quota_service,
)
from api.models import ProviderKeyLimitRequest, ProviderKeySaveRequest
from services.provider_key_service import CatalogRateLimitError, ProviderKeyError, _normalize_provider
from services.provider_models import AvailableModels, CatalogError, ModelInfo, ProviderCatalog

router = APIRouter()


class KeyMetadata(BaseModel):
    provider: str
    key_last4: str
    token_limit: int | None = None
    status: str
    created_at: str
    updated_at: str
    last_tested_at: str | None = None


class KeyResponse(BaseModel):
    status: str = "success"
    key: KeyMetadata
    catalog: ProviderCatalog | None = None


class KeysResponse(BaseModel):
    keys: list[KeyMetadata]


class DeleteResponse(BaseModel):
    status: str = "success"
    removedProvider: str
    defaultModel: str


class _LocalModel(BaseModel):
    id: str


class _LocalModels(BaseModel):
    data: list[_LocalModel]


def _error(exc: ProviderKeyError | CatalogError) -> HTTPException:
    status = 429 if isinstance(exc, CatalogRateLimitError) else (
        503 if isinstance(exc, CatalogError) and not exc.invalid_key else 400
    )
    return HTTPException(status_code=status, detail=str(exc))


@router.get("/api/provider-keys", response_model=KeysResponse)
def list_provider_keys(response: Response, current_user: dict = Depends(get_current_user)) -> KeysResponse:
    response.headers["Cache-Control"] = "no-store"
    return KeysResponse(keys=[KeyMetadata.model_validate(row) for row in provider_key_service.list_keys(current_user["sub"])])


@router.post("/api/provider-keys", response_model=KeyResponse, response_model_exclude_none=True)
def save_provider_key(req: ProviderKeySaveRequest, current_user: dict = Depends(get_current_user)) -> KeyResponse:
    try:
        catalog = provider_catalog_service.save(current_user["sub"], req.provider, req.api_key, req.token_limit)
        meta = provider_key_service.get_metadata(current_user["sub"], req.provider)
        return KeyResponse(key=KeyMetadata.model_validate(meta), catalog=catalog)
    except (ProviderKeyError, CatalogError) as exc:
        raise _error(exc) from None


@router.post("/api/provider-keys/{provider}/test", response_model=KeyResponse, response_model_exclude_none=True)
def test_provider_key(provider: str, current_user: dict = Depends(get_current_user)) -> KeyResponse:
    try:
        provider = _normalize_provider(provider)
        user_id = current_user["sub"]
        snapshot = provider_key_service.key_snapshot(user_id, provider)
        if not snapshot:
            raise HTTPException(status_code=404, detail="Provider key not found")
        provider_key_service.admit_validation(user_id)
        result = provider_catalog_service.adapters[provider].validate_key(snapshot[0])
        if not result.ok:
            raise CatalogError(result.error or "Key validation failed.", result.invalidKey)
        if not provider_key_service.cache_catalog(user_id, provider, snapshot[1], provider_catalog_service.serialize(result.models)):
            raise HTTPException(status_code=409, detail="Provider key changed. Please try again.")
        meta = provider_key_service.mark_test_result(user_id, provider, True)
        catalog = provider_catalog_service.catalog(user_id, provider, cached_only=True)
        return KeyResponse(key=KeyMetadata.model_validate(meta), catalog=catalog)
    except (ProviderKeyError, CatalogError) as exc:
        raise _error(exc) from None


@router.patch("/api/provider-keys/{provider}/limit", response_model=KeyResponse, response_model_exclude_none=True)
def update_provider_key_limit(provider: str, req: ProviderKeyLimitRequest,
                              current_user: dict = Depends(get_current_user)) -> KeyResponse:
    try:
        meta = provider_key_service.set_token_limit(current_user["sub"], provider, req.token_limit)
        return KeyResponse(key=KeyMetadata.model_validate(meta))
    except ProviderKeyError as exc:
        raise _error(exc) from None


@router.delete("/api/provider-keys/{provider}", response_model=DeleteResponse)
def delete_provider_key(provider: str, current_user: dict = Depends(get_current_user)) -> DeleteResponse:
    try:
        provider = _normalize_provider(provider)
        if provider_key_service.delete_key(current_user["sub"], provider):
            return DeleteResponse(removedProvider=provider,
                                  defaultModel=provider_catalog_service.available(current_user["sub"], cached_only=True).defaultModel)
    except ProviderKeyError as exc:
        raise _error(exc) from None
    raise HTTPException(status_code=404, detail="Provider key not found")


@router.post("/api/providers/{provider}/refresh-models", response_model=ProviderCatalog, response_model_exclude_none=True)
def refresh_models(provider: str, current_user: dict = Depends(get_current_user)) -> ProviderCatalog:
    try:
        return provider_catalog_service.catalog(current_user["sub"], _normalize_provider(provider), refresh=True)
    except (ProviderKeyError, CatalogError) as exc:
        raise _error(exc) from None


@router.get("/api/models/available", response_model=AvailableModels, response_model_exclude_none=True)
def available_models(response: Response, current_user: dict = Depends(get_current_user)) -> AvailableModels:
    response.headers["Cache-Control"] = "no-store"
    try:
        available = provider_catalog_service.available(current_user["sub"])
    except ProviderKeyError as exc:
        raise _error(exc) from None
    # Preserve the existing opt-in local LLM menu (operator-configured URL).
    base = os.getenv("LOCAL_LLM_BASE_URL", "").rstrip("/")
    if base:
        try:
            result = httpx.get(f"{base}/models", timeout=3, follow_redirects=False)
            result.raise_for_status()
            local = _LocalModels.model_validate_json(result.content)
            available.providers.append(ProviderCatalog(provider="local", status="included_quota", models=[
                ModelInfo(provider="local", id=f"local/{row.id}", displayName=row.id) for row in local.data if row.id
            ]))
        except (httpx.HTTPError, ValueError):
            pass
    return available


@router.get("/api/usage-quota")
def get_usage_quota(current_user: dict = Depends(get_current_user)):
    metadata = provider_key_service.list_keys(current_user["sub"])
    return usage_quota_service.usage_summary(user_id=current_user["sub"],
        user_email=_current_user_email(current_user), provider_key_metadata=metadata)
