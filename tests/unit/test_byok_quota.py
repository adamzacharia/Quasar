import sys
import types

import pytest
from cryptography.fernet import Fernet

from core.llm_client import LLMClient, llm_request_context
from services import provider_key_service as pks
from services import usage_quota_service as uqs
from services.provider_key_service import ProviderKeyService
from services.usage_quota_service import QuotaExceededError, UsageQuotaService, UsageRecord


@pytest.fixture
def provider_key_store(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_API_KEY_FERNET_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(pks, "_LOCAL_DB", str(tmp_path / "provider_keys.db"))
    return ProviderKeyService()


@pytest.fixture
def quota_store(monkeypatch, tmp_path):
    monkeypatch.setattr(uqs, "_LOCAL_DB", str(tmp_path / "usage_quota.db"))
    return UsageQuotaService()


def test_provider_key_metadata_never_returns_raw_key(provider_key_store):
    raw_key = "sk-test-secret-value-123456"

    metadata = provider_key_store.save_key("u1", "openai", raw_key)
    listed = provider_key_store.list_keys("u1")

    assert metadata["key_last4"] == "3456"
    assert raw_key not in str(metadata)
    assert raw_key not in str(listed)
    assert provider_key_store.decrypt_key("u1", "openai") == raw_key

    conn = provider_key_store._conn()
    encrypted = conn.execute(
        "SELECT encrypted_key FROM user_provider_keys WHERE user_id=? AND provider=?",
        ("u1", "openai"),
    ).fetchone()[0]
    conn.close()
    assert raw_key not in encrypted


def test_platform_quota_is_weekly_and_blocks_when_exhausted(quota_store):
    quota_store.record_usage(
        UsageRecord(
            user_id="u1",
            provider="openai",
            model="gpt-4o-mini",
            key_source="platform",
            input_tokens=99_999,
            output_tokens=1,
        )
    )

    with pytest.raises(QuotaExceededError):
        quota_store.ensure_allowed(
            user_id="u1",
            user_email="person@example.test",
            provider="openai",
            key_source="platform",
        )


def test_quota_exempt_email_bypasses_platform_quota(monkeypatch, quota_store):
    monkeypatch.setenv("QUASAR_TOKEN_LIMIT_EXEMPT_EMAILS", "quota-exempt@example.test")
    quota_store.record_usage(
        UsageRecord(
            user_id="exempt-user",
            provider="deepseek",
            model="deepseek-v4-pro",
            key_source="platform",
            input_tokens=900_000,
            output_tokens=1,
        )
    )

    quota_store.ensure_allowed(
        user_id="exempt-user",
        user_email="quota-exempt@example.test",
        provider="deepseek",
        key_source="platform",
    )


def test_byok_has_no_limit_unless_user_sets_one(quota_store):
    quota_store.record_usage(
        UsageRecord(
            user_id="u1",
            provider="deepseek",
            model="deepseek-v4-pro",
            key_source="byok",
            input_tokens=9_000_000,
            output_tokens=1,
        )
    )

    quota_store.ensure_allowed(
        user_id="u1",
        user_email="person@example.test",
        provider="deepseek",
        key_source="byok",
        byok_token_limit=None,
    )

    with pytest.raises(QuotaExceededError):
        quota_store.ensure_allowed(
            user_id="u1",
            user_email="person@example.test",
            provider="deepseek",
            key_source="byok",
            byok_token_limit=1_000,
        )


def test_byok_openai_key_wins_over_cached_platform_client(monkeypatch):
    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None, default_headers=None, **kwargs):
            self.api_key = api_key
            self.base_url = base_url
            self.default_headers = default_headers

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "platform-key")
    monkeypatch.delenv("HELICONE_API_KEY", raising=False)

    client = LLMClient(model="gpt-4o-mini")
    platform_client = client._get_openai_client()

    with llm_request_context(
        provider_api_keys={"openai": "byok-key"},
        key_source_by_provider={"openai": "byok"},
    ):
        byok_client = client._get_openai_client()

    assert platform_client.api_key == "platform-key"
    assert byok_client.api_key == "byok-key"
