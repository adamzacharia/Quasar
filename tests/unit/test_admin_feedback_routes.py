"""Route-level coverage for the feedback/admin endpoints (guard CX-12, CX-02).

The service layer is tested in test_response_feedback_admin.py and
test_issue_report_context.py; these tests pin the HTTP contract: auth and
admin gating, the vote badge joined by the stable run id (message id only as
fallback), the has_report flag, and the ownership check on client-supplied
run ids before a vote is linked to a run.
"""

import os
import sys as _sys

import pytest


@pytest.fixture
def api_client(monkeypatch):
    # Hermetic: importing api.* load_dotenv()s the owner's real env; the Data
    # Lab token must never leak into an api-importing test (repo convention),
    # and neither may Langfuse credentials (a vote would otherwise be scored
    # against the owner's live project from a unit test).
    for name in ("DATALAB_TOKEN", "LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(name, raising=False)

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for _p in (repo, os.path.join(repo, "ui-pro")):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    from fastapi.testclient import TestClient

    import api.main as main_mod
    from api.deps import get_current_user

    client = TestClient(main_mod.app)
    try:
        yield client, main_mod, get_current_user
    finally:
        main_mod.app.dependency_overrides.pop(get_current_user, None)


def _as_user(main_mod, get_current_user, email="nine@example.com", sub="user-9"):
    main_mod.app.dependency_overrides[get_current_user] = lambda: {"sub": sub, "email": email}


def test_admin_feedback_routes_require_auth_and_admin(api_client, monkeypatch):
    client, main_mod, get_current_user = api_client
    assert client.get("/api/admin/feedback/recent").status_code == 401
    assert client.get("/api/admin/issue-reports").status_code == 401

    from api.routers import admin as admin_router

    _as_user(main_mod, get_current_user)
    monkeypatch.setattr(admin_router, "is_admin_email", lambda email: False)
    assert client.get("/api/admin/feedback/recent").status_code == 403
    assert client.get("/api/admin/issue-reports").status_code == 403


def test_recent_feedback_marks_reports_by_run_or_message(api_client, monkeypatch):
    client, main_mod, get_current_user = api_client
    from api.routers import admin as admin_router

    _as_user(main_mod, get_current_user, email="admin@example.com")
    monkeypatch.setattr(admin_router, "is_admin_email", lambda email: email == "admin@example.com")

    rows = [
        {"message_id": "live-1", "run_id": "run-1", "feedback": "dislike", "created_at": "2026-09-02T10:00:00+00:00"},
        {"message_id": "blk-2", "run_id": "", "feedback": "dislike", "created_at": "2026-09-02T09:00:00+00:00"},
        {"message_id": "msg-3", "run_id": "run-3", "feedback": "dislike", "created_at": "2026-09-02T08:00:00+00:00"},
    ]
    seen = {}
    monkeypatch.setattr(
        admin_router.analytics_service, "list_recent_feedback",
        lambda feedback=None, limit=50: seen.update(feedback=feedback, limit=limit) or [dict(r) for r in rows],
    )
    monkeypatch.setattr(
        admin_router.issue_report_service, "reports_exist_for",
        lambda message_ids=(), run_ids=(), by_user=False: (
            seen.update(message_ids=list(message_ids), run_ids=list(run_ids))
            or {"message_ids": {( "", "blk-2")}, "run_ids": {( "", "run-1")}}
        ),
    )

    resp = client.get("/api/admin/feedback/recent?feedback=dislike&limit=7")
    assert resp.status_code == 200
    out = {r["message_id"]: r["has_report"] for r in resp.json()["feedback"]}
    # run id match, message id match (legacy row without run id), neither.
    assert out == {"live-1": True, "blk-2": True, "msg-3": False}
    assert seen["feedback"] == "dislike" and seen["limit"] == 7
    # The join is asked about exactly the returned rows, not the whole table.
    assert seen["message_ids"] == ["live-1", "blk-2", "msg-3"]
    assert seen["run_ids"] == ["run-1", "", "run-3"]

    # An unknown filter value means "both kinds".
    client.get("/api/admin/feedback/recent?feedback=everything")
    assert seen["feedback"] is None


def test_issue_reports_vote_badge_prefers_run_id(api_client, monkeypatch):
    client, main_mod, get_current_user = api_client
    from api.routers import admin as admin_router

    _as_user(main_mod, get_current_user, email="admin@example.com")
    monkeypatch.setattr(admin_router, "is_admin_email", lambda email: True)

    reports = [
        {"id": "r1", "run_id": "run-1", "message_id": "live-1"},   # vote stored under a different message id
        {"id": "r2", "run_id": "run-2", "message_id": "blk-2"},    # legacy vote, message id only
        {"id": "r3", "run_id": "run-3", "message_id": "blk-3"},    # no vote
    ]
    monkeypatch.setattr(admin_router.issue_report_service, "list_reports", lambda **kw: [dict(r) for r in reports])
    monkeypatch.setattr(
        admin_router.analytics_service, "get_feedback_for_runs",
        lambda run_ids, by_user=False: {( "", "run-1"): "dislike"} if "run-1" in list(run_ids) else {},
    )
    monkeypatch.setattr(
        admin_router.analytics_service, "get_feedback_for_messages",
        lambda message_ids, by_user=False: {( "", "blk-2"): "like", ( "", "live-1"): "like"},
    )

    resp = client.get("/api/admin/issue-reports")
    assert resp.status_code == 200
    votes = {r["id"]: r["vote"] for r in resp.json()["reports"]}
    assert votes == {"r1": "dislike", "r2": "like", "r3": ""}


def test_issue_reports_listing_survives_vote_lookup_failure(api_client, monkeypatch):
    client, main_mod, get_current_user = api_client
    from api.routers import admin as admin_router

    _as_user(main_mod, get_current_user, email="admin@example.com")
    monkeypatch.setattr(admin_router, "is_admin_email", lambda email: True)
    monkeypatch.setattr(admin_router.issue_report_service, "list_reports", lambda **kw: [{"id": "r1", "run_id": "run-1", "message_id": "m"}])

    def boom(*a, **k):
        raise RuntimeError("analytics db down")

    monkeypatch.setattr(admin_router.analytics_service, "get_feedback_for_runs", boom)
    resp = client.get("/api/admin/issue-reports")
    assert resp.status_code == 200
    assert resp.json()["reports"][0]["vote"] == ""


def test_feedback_vote_links_run_only_when_owned(api_client, monkeypatch):
    # CX-02: a client-supplied run_id is stored only if the run belongs to the
    # voter; otherwise the vote is kept but unlinked.
    client, main_mod, get_current_user = api_client
    from api.routers import analytics as analytics_router

    _as_user(main_mod, get_current_user, sub="user-9")
    recorded = {}
    monkeypatch.setattr(analytics_router.analytics_service, "log_feedback", lambda **kw: recorded.update(kw))
    _disable_langfuse(monkeypatch)
    monkeypatch.setattr(
        analytics_router.conversation_service, "conversation_belongs_to_user",
        lambda conversation_id, user_id: conversation_id == "conv-mine" and user_id == "user-9",
    )

    looked_up = []

    def fake_get_run(run_id, user_id=None):
        looked_up.append((run_id, user_id))
        return {"id": run_id, "trace_id": "t", "conversation_id": "conv-from-run"} if run_id == "run-mine" else None

    monkeypatch.setattr(analytics_router.issue_report_service, "get_run", fake_get_run)

    resp = client.post("/api/feedback", json={
        "message_id": "m1", "feedback": "dislike", "run_id": "run-theirs", "conversation_id": "conv-theirs",
    })
    assert resp.status_code == 200
    assert recorded["run_id"] == ""
    # A conversation id the voter does not own is not persisted either.
    assert recorded["conversation_id"] == ""
    assert recorded["user_id"] == "user-9"
    assert looked_up[-1] == ("run-theirs", "user-9")

    resp = client.post("/api/feedback", json={
        "message_id": "m1", "feedback": "dislike", "run_id": "run-mine", "conversation_id": "conv-theirs",
    })
    assert resp.status_code == 200
    assert recorded["run_id"] == "run-mine"
    # With an owned run, the conversation comes from the run, not the client.
    assert recorded["conversation_id"] == "conv-from-run"

    # No run id claimed: no run lookup; the conversation is kept only if owned.
    looked_up.clear()
    client.post("/api/feedback", json={"message_id": "m1", "feedback": "like", "conversation_id": "conv-mine"})
    assert recorded["run_id"] == "" and looked_up == []
    assert recorded["conversation_id"] == "conv-mine"


def _disable_langfuse(monkeypatch):
    # The router imports get_langfuse LOCALLY inside submit_feedback, so the
    # patch must land on the source module, not on the router.
    import core.langfuse_integration as langfuse_integration

    monkeypatch.setattr(langfuse_integration, "get_langfuse", lambda: None)


@pytest.mark.parametrize("suffix", ["", "?export=true"])
def test_snapshot_read_and_export_require_admin(api_client, monkeypatch, suffix):
    client, main_mod, get_current_user = api_client
    from api.routers import admin as admin_router
    from services.feedback_snapshot_service import build_snapshot
    snapshot = build_snapshot(source={"messages": [], "total_messages": 0},
        user_id="owner", conversation_id="deleted-chat", message_id="m1")
    path = f"/api/admin/feedback/snapshots/{snapshot['id']}{suffix}"
    assert client.get(path).status_code == 401
    _as_user(main_mod, get_current_user)
    monkeypatch.setattr(admin_router, "is_admin_email", lambda email: email == "admin@example.com")
    assert client.get(path).status_code == 403
    _as_user(main_mod, get_current_user, email="admin@example.com")
    monkeypatch.setattr(admin_router.issue_report_service.snapshots, "get", lambda sid: snapshot if sid == snapshot["id"] else None)
    response = client.get(path)
    assert response.status_code == 200 and response.json() == snapshot
    assert response.headers["cache-control"] == "no-store"
    if suffix:
        assert ".json" in response.headers["content-disposition"]
    assert client.get("/api/admin/feedback/snapshots/missing").status_code == 404


@pytest.mark.parametrize("consent", [False, None, "true", 1, True])
def test_plain_downvote_snapshot_respects_strict_consent(api_client, monkeypatch, tmp_path, consent):
    client, main_mod, get_current_user = api_client
    from api.routers import analytics as analytics_router
    from services.issue_report_service import IssueReportService
    _as_user(main_mod, get_current_user, sub="u1")
    _disable_langfuse(monkeypatch)
    reads = []
    service = IssueReportService(str(tmp_path / "reports.db"), snapshot_lookup=lambda cid, uid:
        reads.append((cid, uid)) or {"messages": [{"id": "1", "role": "user", "content": "private"}], "total_messages": 1})
    service.start_run(run_id="r1", user_id="u1", conversation_id="c1", trace_id="t1",
        model="test", provider="test", key_source="platform")
    monkeypatch.setattr(analytics_router, "issue_report_service", service)
    recorded = {}
    monkeypatch.setattr(analytics_router.analytics_service, "log_feedback", lambda **kw: recorded.update(kw))
    response = client.post("/api/feedback", json={"message_id": "m1", "run_id": "r1", "feedback": "dislike",
        "include_context": consent, "prompt_preview": "private", "response_preview": "answer"})
    assert response.status_code == 200
    if consent is True:
        sid = recorded["snapshot_id"]
        assert reads == [("c1", "u1")] and sid
        assert service.snapshots.get(sid)["messages"][0]["content"] == "private"
    else:
        assert not reads and not recorded["snapshot_id"]
        assert not recorded["prompt_preview"] and not recorded["response_preview"]


def test_downvote_report_link_is_owned_and_reuses_exact_snapshot(api_client, monkeypatch):
    client, main_mod, get_current_user = api_client
    from api.routers import analytics as router
    _as_user(main_mod, get_current_user, sub="u1")
    _disable_langfuse(monkeypatch)
    monkeypatch.setattr(router.issue_report_service, "get_run", lambda *a, **k: {"id": "r1", "conversation_id": "c1"})
    monkeypatch.setattr(router.issue_report_service, "get_report", lambda rid: {"id": rid, "user_id": "u1", "run_id": "r1", "snapshot_id": "snapshot1"})
    monkeypatch.setattr(router.issue_report_service, "build_feedback_snapshot", lambda **kw: pytest.fail("must reuse report snapshot"))
    recorded = {}
    monkeypatch.setattr(router.analytics_service, "log_feedback", lambda **kw: recorded.update(kw))
    response = client.post("/api/feedback", json={"message_id": "m1", "run_id": "r1", "feedback": "dislike", "report_id": "p1", "include_context": True})
    assert response.status_code == 200 and recorded["snapshot_id"] == "snapshot1"
    monkeypatch.setattr(router.issue_report_service, "get_report", lambda rid: {"id": rid, "user_id": "another-user", "run_id": "r1", "snapshot_id": "private-snapshot"})
    monkeypatch.setattr(router.issue_report_service, "build_feedback_snapshot", lambda **kw: (None, "unavailable"))
    response = client.post("/api/feedback", json={"message_id": "m1", "run_id": "r1", "feedback": "dislike", "report_id": "p2", "include_context": True})
    assert response.status_code == 200 and recorded["snapshot_id"] == ""


def test_feedback_vote_survives_run_store_outage(api_client, monkeypatch):
    # The ownership lookup is a hardening step, not a precondition: if the run
    # store is down the vote is still recorded, just unlinked.
    client, main_mod, get_current_user = api_client
    from api.routers import analytics as analytics_router

    _as_user(main_mod, get_current_user, sub="user-9")
    recorded = {}
    monkeypatch.setattr(analytics_router.analytics_service, "log_feedback", lambda **kw: recorded.update(kw))
    _disable_langfuse(monkeypatch)

    def broken(run_id, user_id=None):
        raise RuntimeError("issue_reports db unreachable")

    monkeypatch.setattr(analytics_router.issue_report_service, "get_run", broken)
    resp = client.post("/api/feedback", json={"message_id": "m1", "feedback": "dislike", "run_id": "run-mine"})
    assert resp.status_code == 200
    assert recorded["feedback"] == "dislike"
    assert recorded["run_id"] == ""


@pytest.mark.parametrize("consent", ["true", "false", 1, 0, None, [], {}])
def test_written_report_rejects_non_boolean_consent(api_client, monkeypatch, consent):
    client, main_mod, get_current_user = api_client
    from api.routers import issue_reports
    _as_user(main_mod, get_current_user)
    monkeypatch.setattr(issue_reports.issue_report_service, "create_report", lambda **kw: pytest.fail("invalid consent reached persistence"))
    response = client.post("/api/issue-reports", json={"message_id": "m1", "run_id": "r1", "category": "other", "description": "x", "include_context": consent})
    assert response.status_code == 422


def test_real_report_response_links_same_snapshot_and_scopes_legacy_joins(api_client, monkeypatch, tmp_path):
    from api.routers import issue_reports, analytics, admin
    from services.issue_report_service import IssueReportService
    import services.analytics_service as analytics_module
    service = IssueReportService(str(tmp_path / "reports.db"), snapshot_lookup=lambda *a:
        {"messages": [{"id": "1", "role": "assistant", "content": "evidence"}], "total_messages": 1})
    service.start_run(run_id="r1", user_id="u1", conversation_id="c1", trace_id="t1", model="test", provider="test", key_source="platform")
    monkeypatch.setattr(analytics_module, "_LOCAL_DB", str(tmp_path / "analytics.db"))
    votes = analytics_module.AnalyticsService()
    for router in (issue_reports, analytics, admin):
        monkeypatch.setattr(router, "issue_report_service", service)
    for router in (analytics, admin):
        monkeypatch.setattr(router, "analytics_service", votes)
    _disable_langfuse(monkeypatch)
    client, main_mod, get_current_user = api_client
    _as_user(main_mod, get_current_user, sub="u1")
    response = client.post("/api/issue-reports", json={"run_id": "r1", "message_id": "same-client-id", "category": "other", "description": "written feedback", "include_context": True})
    assert response.status_code == 200
    report = response.json()["report"]
    response = client.post("/api/feedback", json={"message_id": "same-client-id", "run_id": "r1", "feedback": "dislike", "include_context": True, "report_id": report["id"]})
    assert response.json()["snapshot_id"] == report["snapshot_id"]
    conn = service._conn()
    try:
        assert conn.execute("SELECT COUNT(*) FROM feedback_snapshots").fetchone()[0] == 1
    finally:
        conn.close()
    # Another user controls the same legacy client message ID; their newest
    # vote must neither replace the badge nor inherit the report link.
    votes.log_feedback(message_id="same-client-id", feedback="like", user_id="u2")
    monkeypatch.setattr(admin, "is_admin_email", lambda email: True)
    listing = client.get("/api/admin/issue-reports").json()["reports"]
    assert listing[0]["vote"] == "dislike"
    recent = client.get("/api/admin/feedback/recent?feedback=everything").json()["feedback"]
    assert {v["user_id"]: v["has_report"] for v in recent} == {"u1": True, "u2": False}
    # The legacy fallback itself is scoped, even when the report has no stable
    # run match in the feedback table.
    votes.log_feedback(message_id="same-client-id", feedback="dislike", user_id="u1")
    votes.log_feedback(message_id="same-client-id", feedback="like", user_id="u2")
    assert client.get("/api/admin/issue-reports").json()["reports"][0]["vote"] == "dislike"
