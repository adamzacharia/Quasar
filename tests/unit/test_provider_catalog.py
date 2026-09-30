"""Offline adapter, persistent catalog and FastAPI contract tests."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from services import provider_key_service as key_module
from services.provider_catalog_service import ProviderCatalogService
from services.provider_key_service import CatalogRateLimitError, ProviderKeyService, StoredCatalog
from services.provider_models import (
    AnthropicAdapter, CatalogError, DeepSeekAdapter, GoogleAdapter, ModelInfo,
    OpenAIAdapter, _Page, price_model, sort_models,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "provider_models"
CURATED = {"openai": ["gpt-4.1"], "deepseek": ["deepseek-v4-pro"],
           "tacc": ["gpt-oss-120b"], "anthropic": ["claude-opus-4-8"], "google": ["gemini-2.5-flash"]}


@pytest.mark.parametrize("name,cls,expected", [
    ("openai", OpenAIAdapter, ["model-id-0", "model-id-1", "model-id-2"]),
    ("deepseek", DeepSeekAdapter, ["deepseek-reasoner", "deepseek-chat"]),
    ("anthropic", AnthropicAdapter, ["claude-opus-5"]),
    ("google", GoogleAdapter, ["gemini-2.0-flash"]),
])
def test_adapter_fixture(name, cls, expected):
    payload = (FIXTURES / f"{name}.json").read_bytes()
    seen = []
    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=payload)
    models = cls(httpx.MockTransport(respond)).list_models("unit-secret-key")
    assert [model.id for model in models] == expected
    assert all(model.provider == name and model.source == "live" for model in models)
    assert "unit-secret-key" not in str(seen[0].url)
    if name == "anthropic":
        assert models[0].capabilities.vision is True
        assert models[0].capabilities.reasoning is True
        assert models[0].contextWindow is None
        assert seen[0].headers["anthropic-version"] == "2023-06-01"
    if name == "google":
        assert models[0].contextWindow == 1048576
        assert seen[0].headers["x-goog-api-key"] == "unit-secret-key"


@pytest.mark.parametrize("cls", [OpenAIAdapter, DeepSeekAdapter])
def test_chat_filter_and_snapshot_aliases(cls):
    ids = ["gpt-4.1", "gpt-4.1-2025-04-14", "gpt-4o-mini-2024-07-18", "o3-mini",
           "text-embedding-3-small", "whisper-1", "tts-1", "dall-e-3", "omni-moderation-latest",
           "gpt-realtime", "gpt-audio", "gpt-image-1", "gpt-transcribe", "babbage-002", "davinci-002"]
    page = _Page.model_validate({"data": [{"id": model} for model in ids]})
    models = cls().normalize(page.data)
    assert {m.id for m in models} == {"gpt-4.1", "gpt-4o-mini-2024-07-18", "o3-mini"}


def test_google_filter():
    ids = ["gemini-2.5-flash", "gemini-tts", "aqa", "imagen-3", "veo-2", "embedding-001"]
    page = _Page.model_validate({"models": [{"name": "models/" + id,
        "supportedGenerationMethods": ["generateContent"]} for id in ids]})
    assert [m.id for m in GoogleAdapter().normalize(page.models)] == ["gemini-2.5-flash"]


def test_openai_latest_generations_sort_before_old_models_and_keep_aliases():
    ids = ["gpt-4.1", "gpt-6-sol-2026-09-01", "gpt-6-luna", "gpt-5.6-sol",
           "gpt-6-astra", "gpt-6-sol", "gpt-6.1-sol", "gpt-image-2", "gpt-live-1", "gpt-5.5-pro", "o3-pro-2025-06-10"]
    page = _Page.model_validate({"data": [{"id": model} for model in ids]})
    models = OpenAIAdapter().normalize(page.data)
    assert [m.id for m in models[:4]] == ["gpt-6.1-sol", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna"]
    assert "gpt-6-sol-2026-09-01" not in [m.id for m in models]
    assert "gpt-image-2" not in [m.id for m in models]
    assert "gpt-live-1" not in [m.id for m in models]
    assert "gpt-5.5-pro" not in [m.id for m in models]
    assert "o3-pro-2025-06-10" not in [m.id for m in models]
    assert models[0].inputPricePerM == 2
    assert models[0].outputPricePerM == 10


def test_openai_numeric_versions_and_release_time():
    models = [ModelInfo(provider="openai", id=id, displayName=id, createdAt=date) for id, date in [
        ("gpt-5.9", None), ("gpt-5.10", None),
        ("gpt-7-sol", datetime(2026, 9, 1, tzinfo=timezone.utc)),
        ("gpt-7-luna", datetime(2026, 9, 2, tzinfo=timezone.utc)),
    ]]
    assert [m.id for m in sort_models(models)] == ["gpt-7-luna", "gpt-7-sol", "gpt-5.10", "gpt-5.9"]


def test_openai_defaults_match_frontend_and_include_current_text_models():
    # Parse defaults without importing the API kernel or contacting providers.
    import ast
    import re
    root = Path(__file__).parents[2]
    tree = ast.parse((root / "ui-pro/api/deps.py").read_text(encoding="utf-8"))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "OPENAI_VISIBLE_MODEL_IDS" for target in node.targets))
    backend = ast.literal_eval(assignment.value.args[1])
    source = (root / "ui-pro/src/lib/models.ts").read_text(encoding="utf-8")
    frontend = json.loads("[" + re.search(r"OPENAI_MODELS = \[([\s\S]*?)\];", source)[1].rstrip().rstrip(",") + "]")
    assert backend == frontend
    assert len(backend) == len(set(backend))
    assert {"gpt-6.1-sol", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-5.6-sol",
            "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5", "gpt-5.3-codex"} <= set(backend)
    assert all(price_model(ModelInfo(provider="openai", id=id, displayName=id)).inputPricePerM is not None for id in backend)


def test_openai_stable_alias_precedes_newer_snapshot_in_cache():
    models = [ModelInfo(provider="openai", id=id, displayName=id, createdAt=date) for id, date in [
        ("gpt-6-sol-2026-09-20", datetime(2026, 9, 20, tzinfo=timezone.utc)),
        ("gpt-6-sol", datetime(2026, 9, 1, tzinfo=timezone.utc)),
    ]]
    assert [m.id for m in sort_models(models)] == ["gpt-6-sol", "gpt-6-sol-2026-09-20"]
    assert [m["id"] for m in json.loads(ProviderCatalogService.serialize(models).models_json)] == ["gpt-6-sol", "gpt-6-sol-2026-09-20"]


@pytest.mark.parametrize("cls", [AnthropicAdapter, GoogleAdapter])
def test_pagination(cls):
    seen: list[httpx.Request] = []
    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if cls is AnthropicAdapter:
            return httpx.Response(200, json={"data": [{"id": f"claude-{len(seen)}"}],
                "has_more": len(seen) == 1, "last_id": "claude-1"})
        return httpx.Response(200, json={"models": [{"name": f"models/gemini-{len(seen)}",
            "supportedGenerationMethods": ["generateContent"]}],
            **({"nextPageToken": "next"} if len(seen) == 1 else {})})
    assert len(cls(httpx.MockTransport(respond)).list_models("secret-key")) == 2
    assert seen[1].url.params.get("after_id" if cls is AnthropicAdapter else "pageToken")


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500])
def test_validation_errors_redact_secrets(status, caplog):
    key = "sensitive-do-not-echo"
    adapter = GoogleAdapter(httpx.MockTransport(lambda req: httpx.Response(status,
        json={"error": {"message": key, "reason": "API_KEY_INVALID"}})))
    result = adapter.validate_key(key)
    assert not result.ok
    assert result.invalidKey == (status in {400, 401, 403})
    assert key not in str(result) + caplog.text


def test_invalid_pagination_and_schema():
    repeated = AnthropicAdapter(httpx.MockTransport(lambda req: httpx.Response(200,
        json={"data": [{"id": "claude-one"}], "has_more": True, "last_id": "same"})))
    assert not repeated.validate_key("secret-key").ok
    malformed = OpenAIAdapter(httpx.MockTransport(lambda req: httpx.Response(200, json={})))
    assert not malformed.validate_key("secret-key").ok


def test_prices_are_exact_or_version_prefix_never_family_guesses():
    def price(id: str) -> ModelInfo:
        return price_model(ModelInfo(provider="openai", id=id, displayName=id))
    assert price("gpt-4.1").inputPricePerM == 2
    assert price("gpt-4.1-2025-04-14").inputPricePerM == 2
    assert price("gpt-4.1-unannounced").inputPricePerM is None
    # Verified 2026-09-27, so the catalog now shows it.
    assert price("gpt-5.4-mini").inputPricePerM == 0.75
    assert price("gpt-5.4-mini-ultra").inputPricePerM is None
    assert price("gpt-5.7").inputPricePerM is None
    # Preserve other providers' existing numeric-version suffix behavior.
    google = price_model(ModelInfo(provider="google", id="gemini-2.5-flash.1", displayName="Flash"))
    assert google.inputPricePerM == 0.30
    cached_unknown = ModelInfo(provider="openai", id="gpt-5.4-mini-ultra", displayName="Unknown",
                               inputPricePerM=1.25, outputPricePerM=10)
    refreshed = price_model(cached_unknown)
    assert refreshed.inputPricePerM is None and refreshed.outputPricePerM is None
    assert cached_unknown.inputPricePerM == 1.25


@pytest.fixture
def catalog(monkeypatch, tmp_path):
    monkeypatch.setattr(key_module, "_LOCAL_DB", str(tmp_path / "keys.db"))
    keys = ProviderKeyService()
    adapter = AnthropicAdapter(httpx.MockTransport(lambda req: httpx.Response(200, json={
        "data": [{"id": "claude-opus-test", "display_name": "Test Opus"}], "has_more": False})))
    return ProviderCatalogService(keys, CURATED, {"anthropic": adapter})


def test_fresh_openai_cache_gets_current_sort_without_a_provider_request(catalog):
    ids = ["gpt-4.1", "gpt-5.4-mini", "gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol"]
    models = [ModelInfo(provider="openai", id=id, displayName=id) for id in ids]
    catalog.keys.save_key("alice", "openai", "unit-secret-key", catalog=catalog.serialize(models))
    cached = catalog.catalog("alice", "openai")
    assert not cached.stale
    assert [m.id for m in cached.models] == ["gpt-6.1-sol", "gpt-6-sol", "gpt-6-luna", "gpt-5.4-mini", "gpt-4.1"]
    assert cached.models[0].inputPricePerM == 2


def test_legacy_openai_cache_filters_unsupported_models_on_read_and_routing(catalog):
    ids = ["gpt-5.5-pro", "o3-pro-2025-06-10", "gpt-live-1", "gpt-image-2", "gpt-6-sol"]
    models = [ModelInfo(provider="openai", id=id, displayName=id) for id in ids]
    raw = StoredCatalog(json.dumps([m.model_dump(mode="json") for m in models]), datetime.now(timezone.utc).isoformat())
    catalog.keys.save_key("alice", "openai", "unit-secret-key", catalog=raw)
    assert [m.id for m in catalog.catalog("alice", "openai", cached_only=True).models] == ["gpt-6-sol"]
    assert [m.id for m in catalog.catalog("alice", "openai").models] == ["gpt-6-sol"]
    routes = catalog.model_providers("alice", ["openai"])
    assert routes["gpt-6-sol"] == "openai"
    assert not set(ids[:-1]) & routes.keys()


@pytest.mark.parametrize("provider,good,bad", [
    ("deepseek", "deepseek-v4-pro", "deepseek-embedding-small"),
    ("google", "gemini-2.5-flash", "gemini-tts"),
])
def test_other_provider_legacy_media_cache_is_filtered(catalog, provider, good, bad):
    models = [ModelInfo(provider=provider, id=id, displayName=id) for id in (good, bad)]
    raw = StoredCatalog(json.dumps([m.model_dump(mode="json") for m in models]), datetime.now(timezone.utc).isoformat())
    catalog.keys.save_key("alice", provider, "unit-secret-key", catalog=raw)
    assert [m.id for m in catalog.catalog("alice", provider, cached_only=True).models] == [good]
    assert bad not in catalog.model_providers("alice", [provider])


def test_explicit_curated_openai_overrides_remain_visible_and_routable(catalog):
    configured = ["gpt-5.5-pro", "o3-pro"]
    service = ProviderCatalogService(catalog.keys, {"openai": configured}, {})
    assert [m.id for m in service.catalog("alice", "openai").models] == configured
    assert service.model_providers("alice", []) == {id: "openai" for id in configured}


def test_save_cache_restart_and_isolation(catalog):
    result = catalog.save("alice", "anthropic", "secret-alice", None)
    assert result.models[0].id == "claude-opus-test"
    reopened = ProviderCatalogService(ProviderKeyService(), CURATED, {})
    assert reopened.catalog("alice", "anthropic").models == result.models
    assert reopened.catalog("bob", "anthropic").models == []
    assert reopened.keys.decrypt_key("alice", "anthropic") == "secret-alice"
    assert reopened.keys.get_metadata("alice", "anthropic")["status"] == "valid"


def test_invalid_key_never_overwrites_existing(catalog):
    catalog.save("alice", "anthropic", "old-secret", None)
    before = catalog.keys.get_catalog("alice", "anthropic")
    catalog.adapters["anthropic"] = AnthropicAdapter(httpx.MockTransport(lambda req: httpx.Response(401)))
    with pytest.raises(CatalogError):
        catalog.save("alice", "anthropic", "invalid-secret", None)
    assert catalog.keys.decrypt_key("alice", "anthropic") == "old-secret"
    assert catalog.keys.get_catalog("alice", "anthropic") == before
    with pytest.raises(CatalogError):
        catalog.save("bob", "anthropic", "invalid-secret", None)
    assert catalog.keys.list_keys("bob") == []


def test_stale_fallback_refresh_and_delete(catalog):
    catalog.save("alice", "anthropic", "secret-alice", None)
    snapshot = catalog.keys.key_snapshot("alice", "anthropic")
    cached = catalog.keys.get_catalog("alice", "anthropic")
    old = StoredCatalog(cached.models_json, (datetime.now(timezone.utc) - timedelta(days=2)).isoformat())
    catalog.keys.cache_catalog("alice", "anthropic", snapshot[1], old)
    catalog.adapters["anthropic"] = AnthropicAdapter(httpx.MockTransport(lambda req: httpx.Response(503)))
    assert catalog.catalog("alice", "anthropic").stale
    catalog.keys.save_key("bob", "anthropic", "legacy-key")
    fallback = catalog.catalog("bob", "anthropic")
    assert fallback.models[0].source == "static" and fallback.stale
    catalog.keys.delete_key("alice", "anthropic")
    assert catalog.keys.get_catalog("alice", "anthropic") is None
    assert not catalog.keys.cache_catalog("alice", "anthropic", snapshot[1], cached)
    assert catalog.catalog("alice", "anthropic").models == []
    assert catalog.available("alice").defaultModel == "gpt-oss-120b"


def test_rotated_revision_rejects_inflight_catalog(catalog):
    catalog.save("alice", "anthropic", "old-secret", None)
    revision = catalog.keys.key_snapshot("alice", "anthropic")[1]
    cached = catalog.keys.get_catalog("alice", "anthropic")
    catalog.save("alice", "anthropic", "new-secret", None)
    assert not catalog.keys.cache_catalog("alice", "anthropic", revision, cached)


def test_expired_failure_backoff_and_corrupt_cache(catalog):
    catalog.keys.save_key("alice", "anthropic", "legacy-secret")
    revision = catalog.keys.key_snapshot("alice", "anthropic")[1]
    catalog.keys.cache_catalog("alice", "anthropic", revision, StoredCatalog("broken", "bad-date"))
    calls = []
    def fail(request):
        calls.append(request)
        return httpx.Response(503)
    catalog.adapters["anthropic"] = AnthropicAdapter(httpx.MockTransport(fail))
    for _ in range(5):
        result = catalog.catalog("alice", "anthropic")
        assert result.stale and result.models[0].source == "static"
    assert len(calls) == 1
    # User action bypasses automatic backoff and has a separate rate budget.
    catalog.catalog("alice", "anthropic", refresh=True)
    assert len(calls) == 2


def test_arbitrary_live_ids_use_request_scoped_provider_map():
    from core.llm_client import detect_provider, llm_request_context
    with llm_request_context(model_providers={"custom-experimental": "anthropic"}):
        assert detect_provider("custom-experimental") == "anthropic"
        with llm_request_context(model_providers={"custom-experimental": "google"}):
            assert detect_provider("custom-experimental") == "google"
        assert detect_provider("custom-experimental") == "anthropic"
    assert detect_provider("custom-experimental") == "openai"


def test_rate_limit_is_atomic_persistent_and_per_user(catalog):
    def attempt(_: int) -> bool:
        try:
            catalog.keys.admit_validation("alice")
            return True
        except CatalogRateLimitError:
            return False
    with ThreadPoolExecutor(max_workers=6) as executor:
        assert sum(executor.map(attempt, range(20))) == 10
    with pytest.raises(CatalogRateLimitError):
        ProviderKeyService().admit_validation("alice")
    catalog.keys.admit_validation("bob")


def test_fastapi_save_available_invalid_delete_contract(catalog, monkeypatch):
    # Load the REAL router into a tiny app with only its dependency module
    # substituted: no production singletons, real credentials or science agent.
    def auth(authorization: str | None = Header(None)) -> dict[str, str]:
        if authorization != "Bearer alice":
            raise HTTPException(401)
        return {"sub": "alice"}
    deps = types.ModuleType("api.deps")
    for name, value in {"get_current_user": auth, "provider_key_service": catalog.keys,
        "provider_catalog_service": catalog, "_current_user_email": lambda user: "",
        "usage_quota_service": None}.items():
        setattr(deps, name, value)
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "ui-pro"))
    monkeypatch.setitem(sys.modules, "api.deps", deps)
    path = Path(__file__).parents[2] / "ui-pro/api/routers/provider_keys.py"
    spec = importlib.util.spec_from_file_location("catalog_test_router", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    app = FastAPI()
    app.include_router(module.router)
    with TestClient(app) as client:
        assert client.get("/api/models/available").status_code == 401
        client.headers["Authorization"] = "Bearer alice"
        saved = client.post("/api/provider-keys", json={"provider": "anthropic", "api_key": "secret-alice"})
        assert saved.status_code == 200, saved.text
        assert "secret-alice" not in saved.text
        assert saved.json()["key"]["key_last4"] == "lice"
        assert saved.json()["catalog"]["models"][0]["id"] == "claude-opus-test"
        available = client.get("/api/models/available")
        assert available.headers["cache-control"] == "no-store"
        assert next(p for p in available.json()["providers"] if p["provider"] == "anthropic")["status"] == "connected"
        assert client.post("/api/providers/anthropic/refresh-models").status_code == 200
        deleted = client.delete("/api/provider-keys/anthropic")
        assert deleted.json()["defaultModel"] == "gpt-oss-120b"
        assert catalog.keys.get_catalog("alice", "anthropic") is None
        catalog.adapters["anthropic"] = AnthropicAdapter(httpx.MockTransport(lambda req: httpx.Response(403)))
        rejected = client.post("/api/provider-keys", json={"provider": "anthropic", "api_key": "invalid-secret"})
        assert rejected.status_code == 400
        assert "Invalid API key" in rejected.json()["detail"]
        assert catalog.keys.list_keys("alice") == []
