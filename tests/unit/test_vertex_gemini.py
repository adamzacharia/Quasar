"""Platform Gemini on Vertex AI (VERTEX_API_KEY) and the QUASAR_PLATFORM_GEMINI opt-in."""
from __future__ import annotations

import pytest

from core import llm_client as llm_module
from core.llm_client import (
    LLMClient, google_platform_uses_vertex, llm_request_context, platform_included_providers,
    provider_has_key_path,
)
from services.provider_catalog_service import ProviderCatalogService
from services.usage_quota_service import platform_token_limit

GOOGLE_ENV = ("VERTEX_API_KEY", "GEMINI_API_KEY", "QUASAR_PLATFORM_GEMINI")
CURATED = {"openai": ["gpt-4.1"], "deepseek": ["deepseek-v4-pro"], "tacc": ["gpt-oss-120b"],
           "anthropic": ["claude-opus-4-8"], "google": ["gemini-3.8-flash"]}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in GOOGLE_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def genai_calls(monkeypatch):
    """Record every google.genai.Client(...) construction instead of building one."""
    from google import genai

    calls = []

    def fake_client(**kwargs):
        calls.append(kwargs)
        return ("client", len(calls))

    monkeypatch.setattr(genai, "Client", fake_client)
    return calls


class _NoKeys:
    """ProviderKeyService stand-in for a user with no BYOK keys."""

    def key_snapshot(self, user_id, provider):
        return None

    def list_catalogs(self, user_id):
        return {}


def test_platform_google_uses_vertex_when_vertex_key_set(monkeypatch, genai_calls):
    monkeypatch.setenv("VERTEX_API_KEY", " vertex-key ")
    monkeypatch.setenv("GEMINI_API_KEY", "studio-key")
    client = LLMClient()
    first = client._get_google_client()
    assert client._get_google_client() is first  # cached like the AI Studio client
    assert genai_calls == [{"vertexai": True, "api_key": "vertex-key"}]
    assert google_platform_uses_vertex()


def test_platform_google_falls_back_to_ai_studio(monkeypatch, genai_calls):
    monkeypatch.setenv("GEMINI_API_KEY", "studio-key")
    LLMClient()._get_google_client()
    assert genai_calls == [{"api_key": "studio-key", "vertexai": False}]
    assert not google_platform_uses_vertex()


def test_byok_google_key_stays_on_ai_studio(monkeypatch, genai_calls):
    monkeypatch.setenv("VERTEX_API_KEY", "vertex-key")
    with llm_request_context(provider_api_keys={"google": "user-key"},
                             key_source_by_provider={"google": "byok"}):
        LLMClient()._get_google_client()
    assert genai_calls == [{"api_key": "user-key", "vertexai": False}]


def test_missing_google_key_names_both_env_vars(monkeypatch, genai_calls):
    with pytest.raises(ValueError, match="VERTEX_API_KEY or GEMINI_API_KEY"):
        LLMClient()._get_google_client()


def test_vertex_key_counts_as_google_key_path(monkeypatch):
    assert not provider_has_key_path("google")
    monkeypatch.setenv("VERTEX_API_KEY", "vertex-key")
    assert provider_has_key_path("google")


@pytest.mark.parametrize("flag,key,included", [
    ("", "VERTEX_API_KEY", False),      # opt-in is off by default
    ("1", None, False),                 # flag alone, no platform Google key
    ("1", "VERTEX_API_KEY", True),
    ("true", "GEMINI_API_KEY", True),
    ("0", "VERTEX_API_KEY", False),
])
def test_platform_gemini_opt_in(monkeypatch, flag, key, included):
    monkeypatch.setenv("QUASAR_PLATFORM_GEMINI", flag)
    if key:
        monkeypatch.setenv(key, "k")
    assert ("google" in platform_included_providers()) is included
    assert "anthropic" not in platform_included_providers()
    assert {"openai", "deepseek", "tacc", "local"} <= platform_included_providers()


def test_catalog_offers_platform_gemini_only_when_opted_in(monkeypatch):
    service = ProviderCatalogService(_NoKeys(), CURATED, {})
    monkeypatch.setenv("VERTEX_API_KEY", "vertex-key")

    off = service.catalog("alice", "google")
    assert off.status == "not_connected" and off.models == []
    assert "gemini-3.8-flash" not in service.model_providers("alice", [])

    monkeypatch.setenv("QUASAR_PLATFORM_GEMINI", "1")
    on = service.catalog("alice", "google")
    assert on.status == "included_quota"
    assert [m.id for m in on.models] == ["gemini-3.8-flash"]
    assert service.model_providers("alice", [])["gemini-3.8-flash"] == "google"
    # Anthropic stays BYOK-only.
    assert service.catalog("alice", "anthropic").status == "not_connected"


def test_platform_gemini_has_a_weekly_cap(monkeypatch):
    monkeypatch.delenv("QUASAR_PLATFORM_TOKEN_LIMITS", raising=False)
    assert platform_token_limit("google") == 300_000
    monkeypatch.setenv("QUASAR_PLATFORM_TOKEN_LIMITS", "google=50000")
    assert platform_token_limit("google") == 50_000


def test_platform_image_generation_accepts_vertex_key(monkeypatch):
    from services import image_generation

    monkeypatch.setenv("QUASAR_ALLOW_PLATFORM_IMAGE_GENERATION", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("QUASAR_IMAGE_TOOL_GOOGLE_MODEL", raising=False)
    monkeypatch.setenv("VERTEX_API_KEY", "vertex-key")
    assert image_generation.tool_image_model() == "gemini-3.1-flash-image"


def test_google_contents_never_end_on_a_model_turn():
    # Vertex 400s "Requests ending with a model turn are not supported".
    system, contents = LLMClient()._responses_shim._google_contents([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "How many ALMA observations of M87?"},
        {"role": "assistant", "content": "Let me check the archive."},
    ], model="gemini-3.8-flash")
    assert system == "sys"
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[-1]["parts"] == [{"text": "Continue."}]


def test_google_contents_ending_on_tool_output_unchanged():
    _, contents = LLMClient()._responses_shim._google_contents([
        {"role": "user", "content": "Where is M87?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "function": {"name": "resolve_target", "arguments": "{\"target_name\": \"M87\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "{\"ra\": 187.7}"},
    ], model="gemini-3.8-flash")
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert "function_response" in contents[-1]["parts"][0]


def test_byok_client_ignores_sdk_vertex_env(monkeypatch):
    # The SDK reads GOOGLE_GENAI_USE_VERTEXAI; BYOK must still reach AI Studio.
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    with llm_request_context(provider_api_keys={"google": "user-key"},
                             key_source_by_provider={"google": "byok"}):
        client = LLMClient()._get_google_client()
    assert client._api_client.vertexai is False


def test_whitespace_vertex_key_does_not_claim_platform_images(monkeypatch):
    from services import image_generation

    monkeypatch.setenv("QUASAR_ALLOW_PLATFORM_IMAGE_GENERATION", "1")
    monkeypatch.setenv("VERTEX_API_KEY", "   ")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("QUASAR_IMAGE_TOOL_OPENAI_MODEL", raising=False)
    assert image_generation.tool_image_model() == "gpt-image-2"
