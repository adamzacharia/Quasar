"""Request-level tests for the usage/admin-usage endpoints.

Covers authentication, per-user scoping, admin gating, and malformed-parameter
rejection — the security-relevant surface of the usage report.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_client(monkeypatch):
    # `api` lives under ui-pro/. Do the path + app import INSIDE the fixture (not
    # at module import time) so this test file has no collection-time side
    # effects on the rest of the suite — mirrors tests/unit/test_s3_s6_security.py.
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for _p in (repo, os.path.join(repo, "ui-pro")):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    import api.main as main_mod
    from api.deps import get_current_user

    client = TestClient(main_mod.app)

    def _override(user):
        main_mod.app.dependency_overrides[get_current_user] = lambda: user

    try:
        yield client, main_mod, _override
    finally:
        main_mod.app.dependency_overrides.pop(get_current_user, None)


def test_usage_summary_requires_auth(app_client):
    client, _, _ = app_client
    # No auth dependency override and no token → not a 200.
    resp = client.get("/api/usage/summary")
    assert resp.status_code in (401, 403, 422)


def test_usage_summary_scopes_to_authenticated_user(app_client, monkeypatch):
    client, main_mod, override = app_client
    override({"sub": "user-abc", "email": "user-abc@example.test"})

    seen = {}

    def fake_summary(*, user_id, user_email, provider_key_metadata=None):
        seen["user_id"] = user_id
        return {"platform": {}, "byok": {}, "is_admin": False,
                "is_quota_exempt": False, "daily": {"unlimited": True}, "cost": {}}

    monkeypatch.setattr(main_mod, "usage", main_mod.usage)  # touch to ensure import
    import services.usage_quota_service as uqs
    monkeypatch.setattr(uqs.UsageQuotaService, "usage_summary",
                        lambda self, **kw: fake_summary(**kw))
    import services.provider_key_service as pks
    monkeypatch.setattr(pks.ProviderKeyService, "list_keys", lambda self, uid: [])

    resp = client.get("/api/usage/summary")
    assert resp.status_code == 200
    # The query is scoped to the authenticated subject, not a client-supplied id.
    assert seen["user_id"] == "user-abc"


def test_admin_usage_rejects_non_admin(app_client, monkeypatch):
    client, main_mod, override = app_client
    override({"sub": "user-abc", "email": "user-abc@example.test"})
    import services.admin_access as admin_access
    monkeypatch.setattr(admin_access, "is_admin_email", lambda e: False)
    import api.routers.admin as admin_router
    monkeypatch.setattr(admin_router, "is_admin_email", lambda e: False)

    resp = client.get("/api/admin/usage")
    assert resp.status_code == 403


def test_admin_usage_allows_admin(app_client, monkeypatch):
    client, main_mod, override = app_client
    override({"sub": "boss", "email": "boss@example.test"})
    import api.routers.admin as admin_router
    monkeypatch.setattr(admin_router, "is_admin_email", lambda e: True)
    # Default source is the per-call ledger (global_usage_rollup).
    import services.usage_quota_service as uqs
    monkeypatch.setattr(
        uqs.UsageQuotaService, "global_usage_rollup",
        lambda self, **kw: {"group_by": kw.get("group_by"), "source": "llm_usage_events",
                            "groups": [], "totals": {}},
    )

    resp = client.get("/api/admin/usage?group_by=provider")
    assert resp.status_code == 200
    assert resp.json()["group_by"] == "provider"
    assert resp.json()["source"] == "llm_usage_events"


def test_admin_usage_rejects_bad_group_by(app_client, monkeypatch):
    client, main_mod, override = app_client
    override({"sub": "boss", "email": "boss@example.test"})
    import api.routers.admin as admin_router
    monkeypatch.setattr(admin_router, "is_admin_email", lambda e: True)

    # A group_by outside the allowlist must be a 400, not an SQL error/500.
    resp = client.get("/api/admin/usage?group_by=user_id;DROP+TABLE+chat_runs")
    assert resp.status_code == 400


def test_admin_usage_rejects_bad_source(app_client, monkeypatch):
    client, main_mod, override = app_client
    override({"sub": "boss", "email": "boss@example.test"})
    import api.routers.admin as admin_router
    monkeypatch.setattr(admin_router, "is_admin_email", lambda e: True)

    # A `source` outside {calls, turns} must be a 400, not a silent fall-through
    # to the per-call report (CX-16).
    resp = client.get("/api/admin/usage?source=bogus")
    assert resp.status_code == 400


def test_every_configured_model_is_priced(app_client):
    """Drift guard (CX-33): a model offered in the picker but absent from
    MODEL_PRICING reports every turn on it as unpriced, which reads as "no
    spend" in the admin rollup (CX-27).

    The configured lists are READ from api.deps rather than restated, so this
    keeps guarding after someone edits deps. It lives in this module, not in
    test_cost_accounting.py, on purpose: importing api.deps calls load_dotenv(),
    which injects the developer's real TURSO_* credentials into os.environ for
    the rest of the process and can route later tests' DBs at the cloud (the
    leak tests/conftest.py's QUASAR_FORCE_LOCAL_DB guard exists for). This file
    already imports the app and pays that cost; the pure pricing module must
    stay free of it.

    TACC is excluded deliberately — grant-funded, no per-token price, so
    "unpriced" is the correct answer there rather than a gap.
    """
    from api import deps
    from services import model_pricing

    configured = (
        [("openai", m) for m in deps.OPENAI_VISIBLE_MODEL_IDS]
        + [("deepseek", m) for m in deps.DEEPSEEK_VISIBLE_MODEL_IDS]
        + [("anthropic", m) for m in deps.ANTHROPIC_VISIBLE_MODEL_IDS]
    )
    assert configured, "no models configured — the guard would pass vacuously"

    unpriced = [
        f"{provider}/{model}"
        for provider, model in configured
        if model_pricing.get_model_pricing(provider, model) is None
    ]
    assert not unpriced, (
        "configured models missing from MODEL_PRICING: " + ", ".join(unpriced)
    )
