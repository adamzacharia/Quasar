"""Per-user catalogs, with the same persistent storage as encrypted BYOK keys."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Mapping

from pydantic import TypeAdapter, ValidationError

from services.provider_key_service import (
    CatalogRateLimitError, ProviderKeyError, ProviderKeyService, StoredCatalog, _normalize_provider,
)
from services.provider_models import (
    ADAPTERS, AvailableModels, CatalogError, ModelInfo, Provider, ProviderAdapter,
    ProviderCatalog, price_model, sort_models,
)

_MODELS = TypeAdapter(list[ModelInfo])
DEFAULT_MODEL = "gpt-oss-120b"


class ProviderCatalogService:
    def __init__(self, keys: ProviderKeyService, curated: Mapping[Provider, list[str]],
                 adapters: Mapping[str, ProviderAdapter] | None = None):
        self.keys = keys
        self.curated = dict(curated)
        self.adapters = ADAPTERS if adapters is None else adapters

    def static_models(self, provider: Provider) -> list[ModelInfo]:
        return sort_models([price_model(ModelInfo(provider=provider, id=model, displayName=model, source="static"))
                            for model in self.curated.get(provider, [])])

    @staticmethod
    def serialize(models: list[ModelInfo]) -> StoredCatalog:
        return StoredCatalog(_MODELS.dump_json(models, exclude_none=True).decode(),
                             datetime.now(timezone.utc).isoformat())

    def save(self, user_id: str, provider: str, api_key: str, token_limit: int | None) -> ProviderCatalog:
        provider = _normalize_provider(provider)
        self.keys.admit_validation(user_id)
        key = api_key.strip()
        if len(key) < 8 or len(key) > 4096 or any(c in key for c in "\r\n"):
            raise ProviderKeyError("Enter a valid provider API key.")
        if token_limit is not None:
            self.keys._normalize_token_limit(token_limit)
        validation = self.adapters[provider].validate_key(key)
        if not validation.ok:
            raise CatalogError(validation.error or "Could not validate provider key.", validation.invalidKey)
        stored = self.serialize(validation.models)
        self.keys.save_key(user_id, provider, key, token_limit, catalog=stored)
        return ProviderCatalog(provider=provider, status="connected", models=validation.models,
                               fetchedAt=datetime.fromisoformat(stored.fetched_at))

    def catalog(self, user_id: str, provider: Provider, *, refresh: bool = False,
                cached_only: bool = False) -> ProviderCatalog:
        included = provider in {"openai", "deepseek", "tacc", "local"}
        disconnected = ProviderCatalog(provider=provider, status="included_quota" if included else "not_connected",
                                       models=self.static_models(provider) if included else [],
                                       unlockCount=len(self.curated.get(provider, [])))
        if provider in {"tacc", "local"}:
            return disconnected
        snapshot = self.keys.key_snapshot(user_id, provider)
        if snapshot is None:
            if refresh:
                raise ProviderKeyError("Add an API key before refreshing models.")
            return disconnected
        key, revision = snapshot
        cached = self.keys.get_catalog(user_id, provider)
        try:
            models = _MODELS.validate_json(cached.models_json) if cached else self.static_models(provider)
            fetched = datetime.fromisoformat(cached.fetched_at) if cached else None
            if fetched is not None and fetched.tzinfo is None:
                fetched = fetched.replace(tzinfo=timezone.utc)
        except (ValidationError, ValueError):
            models, fetched = self.static_models(provider), None
        fresh = fetched is not None and datetime.now(timezone.utc) - fetched < timedelta(hours=24)
        if (fresh and not refresh) or cached_only:
            return ProviderCatalog(provider=provider, status="connected", models=models,
                                   fetchedAt=fetched, stale=not fresh)
        try:
            if refresh:
                self.keys.admit_validation(user_id)
            elif not self.keys.claim_discovery(user_id, provider, revision):
                return ProviderCatalog(provider=provider, status="connected", models=models, fetchedAt=fetched, stale=True)
            live = self.adapters[provider].list_models(key)
            stored = self.serialize(live)
            if not self.keys.cache_catalog(user_id, provider, revision, stored):
                # Removed or rotated during discovery. Re-read current state,
                # never return models fetched with an obsolete credential.
                return self.catalog(user_id, provider, cached_only=True)
            return ProviderCatalog(provider=provider, status="connected", models=live,
                                   fetchedAt=datetime.fromisoformat(stored.fetched_at))
        except CatalogRateLimitError:
            if refresh:
                raise
        except CatalogError:
            pass
        # Also check the revision on failure: a racing delete must not return a
        # now-disconnected user's previously captured catalog.
        current = self.keys.key_snapshot(user_id, provider)
        if current is None or current[1] != revision:
            return self.catalog(user_id, provider, cached_only=True)
        return ProviderCatalog(provider=provider, status="connected", models=models,
                               fetchedAt=fetched, stale=True)

    def available(self, user_id: str, *, cached_only: bool = False) -> AvailableModels:
        providers = [self.catalog(user_id, provider, cached_only=cached_only) for provider in self.curated]
        ids = [model.id for catalog in providers if catalog.status == "included_quota" for model in catalog.models]
        default = DEFAULT_MODEL if DEFAULT_MODEL in ids else (ids[0] if ids else "")
        return AvailableModels(providers=providers, defaultModel=default)

    def model_providers(self, user_id: str, connected: list[str]) -> dict[str, str]:
        # No provider HTTP calls on the chat hot path. Catalog discovery lives
        # at save/refresh/picker load, with cache fallback on provider outages.
        catalogs = self.keys.list_catalogs(user_id)
        routes: dict[str, str] = {}
        for provider in self.curated:
            if provider not in connected and provider not in {"openai", "deepseek", "tacc", "local"}:
                continue
            models = self.static_models(provider)
            if provider in connected and provider in catalogs:
                try:
                    models = _MODELS.validate_json(catalogs[provider].models_json)
                except ValidationError:
                    pass
            for model in models:
                if model.id in routes and routes[model.id] != provider:
                    raise ProviderKeyError("Ambiguous provider model ID. Refresh your provider catalogs.")
                routes[model.id] = provider
        return routes
