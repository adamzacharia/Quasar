"""Gemini on Vertex AI: the platform VERTEX_API_KEY, users' own Vertex keys, the Gemini quota."""
from __future__ import annotations

import pytest

from core import llm_client as llm_module
from core.llm_client import (
    LLMClient, google_key_routes_to_vertex, google_platform_uses_vertex, llm_request_context,
    platform_included_providers,
    provider_has_key_path,
)
from services.provider_catalog_service import ProviderCatalogService
from services import usage_quota_service as uqs
from services.usage_quota_service import (
    QuotaExceededError, UsageQuotaService, UsageRecord, platform_token_limit, platform_window_days,
)

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
    ("", "VERTEX_API_KEY", True),       # a Vertex key turns platform Gemini on
    ("", "GEMINI_API_KEY", False),      # an AI Studio key alone does not
    ("1", None, False),                 # flag alone, no platform Google key
    ("true", "GEMINI_API_KEY", True),   # explicit opt-in for the AI Studio key
    ("0", "VERTEX_API_KEY", False),     # explicit off switch
])
def test_platform_gemini_switch(monkeypatch, flag, key, included):
    monkeypatch.setenv("QUASAR_PLATFORM_GEMINI", flag)
    if key:
        monkeypatch.setenv(key, "k")
    assert ("google" in platform_included_providers()) is included
    assert "anthropic" not in platform_included_providers()
    assert {"openai", "deepseek", "tacc", "local"} <= platform_included_providers()


def test_catalog_offers_platform_gemini_with_a_vertex_key(monkeypatch):
    service = ProviderCatalogService(_NoKeys(), CURATED, {})
    monkeypatch.setenv("GEMINI_API_KEY", "studio-key")

    off = service.catalog("alice", "google")
    assert off.status == "not_connected" and off.models == []
    assert "gemini-3.8-flash" not in service.model_providers("alice", [])

    monkeypatch.setenv("VERTEX_API_KEY", "AQ.vertex-key")
    on = service.catalog("alice", "google")
    assert on.status == "included_quota"
    assert [m.id for m in on.models] == ["gemini-3.8-flash"]
    assert service.model_providers("alice", [])["gemini-3.8-flash"] == "google"
    # Anthropic stays BYOK-only.
    assert service.catalog("alice", "anthropic").status == "not_connected"


def test_platform_gemini_has_a_monthly_cap(monkeypatch):
    monkeypatch.delenv("QUASAR_PLATFORM_TOKEN_LIMITS", raising=False)
    assert platform_token_limit("google") == 1_000_000
    assert platform_window_days("google") == 30
    assert platform_window_days("deepseek") == 7
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


# -- A user's own Vertex key, the Vertex catalog, quota enforcement ----------

def test_byok_vertex_key_goes_to_vertex(monkeypatch, genai_calls):
    with llm_request_context(provider_api_keys={"google": "AQ.user-vertex"},
                             key_source_by_provider={"google": "byok"}):
        LLMClient()._get_google_client()
    assert genai_calls == [{"api_key": "AQ.user-vertex", "vertexai": True}]
    assert google_key_routes_to_vertex("AQ.user-vertex")
    assert not google_key_routes_to_vertex("AIzaStudio")
    assert not google_key_routes_to_vertex(None)  # platform key, no VERTEX_API_KEY
    monkeypatch.setenv("VERTEX_API_KEY", "AQ.platform")
    assert google_key_routes_to_vertex(None)
    assert not google_key_routes_to_vertex("AIzaStudio")  # the user's key type wins


def _transport(status, seen):
    import httpx

    def respond(request):
        seen.append(request)
        body = {"totalTokens": 1} if status == 200 else {"error": {"code": status}}
        return httpx.Response(status, json=body)
    return httpx.MockTransport(respond)


def test_vertex_key_is_validated_with_count_tokens(monkeypatch):
    from services.provider_models import GoogleAdapter

    monkeypatch.delenv("QUASAR_GOOGLE_MODELS", raising=False)
    seen = []
    result = GoogleAdapter(_transport(200, seen)).validate_key(" AQ.user-vertex ")
    assert result.ok
    assert {m.id for m in result.models} == {"gemini-3.8-flash", "gemini-3.1-pro-preview", "gemini-3.5-flash-lite"}
    assert all(m.provider == "google" for m in result.models)
    assert len(seen) == 1 and seen[0].url.path.endswith("/gemini-3.8-flash:countTokens")
    assert seen[0].headers["x-goog-api-key"] == "AQ.user-vertex"
    assert "AQ.user-vertex" not in str(seen[0].url)


@pytest.mark.parametrize("status", [400, 401, 403])
def test_bad_vertex_key_is_rejected(status):
    from services.provider_models import GoogleAdapter

    result = GoogleAdapter(_transport(status, [])).validate_key("AQ.bad")
    assert not result.ok and result.invalidKey


def test_ai_studio_key_still_lists_models_on_ai_studio():
    from services.provider_models import GoogleAdapter

    seen = []
    GoogleAdapter(_transport(403, seen)).validate_key("AIzaStudio")
    assert seen[0].url.host == "generativelanguage.googleapis.com"


@pytest.fixture
def quota(monkeypatch, tmp_path):
    import services.admin_access as admin_access

    monkeypatch.setattr(uqs, "_LOCAL_DB", str(tmp_path / "usage_quota.db"))
    for name in ("QUASAR_PLATFORM_TOKEN_LIMITS", "QUASAR_TOKEN_LIMIT_EXEMPT_EMAILS", "ADMIN_EMAILS",
                 "QUASAR_DAILY_TOKEN_LIMIT", "QUASAR_GEMINI_UNLIMITED_EMAILS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(admin_access, "_db_role_is_admin", lambda email: False)
    monkeypatch.setenv("VERTEX_API_KEY", "AQ.platform")
    return UsageQuotaService()


def _spend(quota, user, provider, tokens, key_source="platform"):
    quota.record_usage(UsageRecord(user_id=user, provider=provider, model="m", key_source=key_source,
                                   input_tokens=tokens, output_tokens=0))


def test_gemini_cap_and_unlimited_accounts(quota, monkeypatch):
    _spend(quota, "u1", "google", 1_000_000)
    with pytest.raises(QuotaExceededError, match="30-day"):
        quota.ensure_allowed(user_id="u1", user_email="someone@example.test",
                             provider="google", key_source="platform")
    # Their own key still works, other providers keep their own allowance,
    # and other users are unaffected.
    quota.ensure_allowed(user_id="u1", user_email="someone@example.test", provider="google", key_source="byok")
    quota.ensure_allowed(user_id="u1", user_email="someone@example.test", provider="deepseek", key_source="platform")
    quota.ensure_allowed(user_id="u2", user_email="other@example.test", provider="google", key_source="platform")

    monkeypatch.setenv("QUASAR_GEMINI_UNLIMITED_EMAILS", "Friend@Example.test, other@x.org")
    assert quota.ensure_allowed(user_id="u1", user_email="friend@example.test",
                                provider="google", key_source="platform") is None
    _spend(quota, "u1", "deepseek", 500_000)
    with pytest.raises(QuotaExceededError):  # the exemption is Gemini-only
        quota.ensure_allowed(user_id="u1", user_email="friend@example.test",
                             provider="deepseek", key_source="platform")
    summary = quota.usage_summary(user_id="u1", user_email="friend@example.test")
    assert summary["platform"]["google"]["unlimited"] is True
    assert summary["platform"]["deepseek"]["unlimited"] is False


def test_usage_summary_shows_gemini_only_when_offered(quota, monkeypatch):
    summary = quota.usage_summary(user_id="u3", user_email="a@example.test")
    assert summary["platform"]["google"]["limit_tokens"] == 1_000_000
    assert summary["platform"]["google"]["window_days"] == 30
    assert summary["platform"]["deepseek"]["window_days"] == 7
    monkeypatch.setenv("QUASAR_PLATFORM_GEMINI", "0")
    assert "google" not in quota.usage_summary(user_id="u3", user_email="a@example.test")["platform"]


def _interrupted_google_stream(llm):
    """A Gemini stream that reports usage, sends some text, then dies."""
    from types import SimpleNamespace as NS

    meta = NS(prompt_token_count=12_000, tool_use_prompt_token_count=0,
              candidates_token_count=5, thoughts_token_count=8)
    part = NS(text="partial answer", thought=False, function_call=None, thought_signature=None)
    chunk = NS(usage_metadata=meta, candidates=[NS(content=NS(parts=[part]), finish_reason=None)])

    def generate_content_stream(**_kwargs):
        yield chunk
        raise RuntimeError("stream reset")

    llm._get_google_client = lambda: NS(models=NS(generate_content_stream=generate_content_stream))


@pytest.mark.parametrize("close_early", [False, True])
def test_interrupted_gemini_stream_still_settles_usage(monkeypatch, close_early):
    llm = LLMClient()
    shim = llm._responses_shim
    _interrupted_google_stream(llm)
    settled = []
    monkeypatch.setattr(shim, "_record_usage",
                        lambda provider, model, resp, rid: settled.append((resp.usage.input_tokens,
                                                                           resp.usage.output_tokens, rid)))
    monkeypatch.setattr(shim, "_release_quota", lambda rid: settled.append(("released", rid)))
    monkeypatch.setattr(shim, "_log_billed", lambda *a, **k: None)
    monkeypatch.setattr(shim, "_record_provider_health", lambda *a, **k: None)

    raw = shim._stream_google({"model": "gemini-3.8-flash", "input": "hi"})
    stream = shim._wrap_usage_stream(raw, "google", "gemini-3.8-flash", reservation_id="r1")
    if close_early:
        for event in stream:
            if event.type == "response.output_text.delta":
                stream.close()  # the user cancels mid-answer
                break
    else:
        with pytest.raises(RuntimeError, match="stream reset"):
            list(stream)
    assert settled == [(12_000, 13, "r1")]



@pytest.mark.parametrize("env,expected", [(None, False), ("1", True)])
def test_log_diagnose_is_off_unless_enabled(env, expected):
    """Loguru diagnose printed request headers (an API key) in tracebacks.
    A fresh interpreter, so this test never reconfigures the suite's sinks."""
    import os
    import subprocess
    import sys

    child_env = {k: v for k, v in os.environ.items() if k != "QUASAR_LOG_DIAGNOSE"}
    if env is not None:
        child_env["QUASAR_LOG_DIAGNOSE"] = env
    code = ("import core.logger\nfrom loguru import logger\n"
            "print(sorted({h._exception_formatter._diagnose for h in logger._core.handlers.values()}))")
    out = subprocess.run([sys.executable, "-c", code], env=child_env, capture_output=True, text=True,
                         timeout=120, cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    assert out.stdout.strip().splitlines()[-1] == str([expected])
