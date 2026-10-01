"""One-click MCP connections: OAuth sign-in end to end (Phase 1 + 2).

A fake authorization server and an OAuth-protected FastMCP server
(tests/fixtures/fake_oauth_mcp.py) run in-process on a free port. The real
router (ui-pro/api/routers/tools.py) runs in a FastAPI TestClient with the
auth dependency overridden, and the real per-user pool connects to the fake
server. Covered: detection, discovery, registration, PKCE, the state's
single-use / expiry rules, iss checks, code exchange with encrypted storage,
tool calls with the token, refresh on expiry and on a 401, refresh failure
leading to needs_auth without retry storms, disconnect/revoke, user
isolation, and secrets never reaching responses or logs.
"""

import json
import os
import socket
import sqlite3
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("mcp")
uvicorn = pytest.importorskip("uvicorn")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO, os.path.join(REPO, "ui-pro")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tests.fixtures.fake_oauth_mcp import FakeAuthState, auto_approve, make_app  # noqa: E402

ORIGIN = "http://localhost:3001"
PUBLIC_API = "http://127.0.0.1:8999"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(app, port):
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.05)
    assert srv.started
    return srv, thread


@pytest.fixture()
def fake():
    port = _free_port()
    state = FakeAuthState(base_url=f"http://127.0.0.1:{port}")
    srv, thread = _serve(make_app(state), port)
    yield state
    _close_pool()  # an open MCP session would hold the server's shutdown
    srv.should_exit = True
    thread.join(timeout=5)


def _close_pool():
    from services.user_mcp import get_user_mcp_pool

    for uid in ("alice", "bob"):
        get_user_mcp_pool().forget(uid)
    time.sleep(0.2)


@pytest.fixture()
def keyed_server():
    """A plain 401 server (no OAuth metadata) that accepts one API key."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("keyed")

    @mcp.tool()
    def ping() -> str:
        """Free."""
        return "pong"

    inner = mcp.streamable_http_app()

    async def app(scope, receive, send):
        if scope["type"] == "http":
            auth = dict(scope.get("headers") or []).get(b"authorization", b"")
            if auth != b"Bearer test-key-123456789":
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": b'{"error":"key required"}'})
                return
        return await inner(scope, receive, send)

    port = _free_port()
    srv, thread = _serve(app, port)
    yield f"http://127.0.0.1:{port}/mcp"
    _close_pool()
    srv.should_exit = True
    thread.join(timeout=5)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("QUASAR_MCP_DB_PATH", str(tmp_path / "mcp.db"))
    monkeypatch.setenv("QUASAR_PUBLIC_API_URL", PUBLIC_API)
    monkeypatch.setenv("QUASAR_MCP_TOOL_TIMEOUT_SECONDS", "30")
    monkeypatch.chdir(tmp_path)  # legacy user_tools/ lookups stay inside tmp
    return tmp_path


class _User:
    sub = "alice"


@pytest.fixture()
def api(env):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.deps import get_current_user
    from api.routers import tools
    from services.user_mcp import get_user_mcp_pool

    app = FastAPI()
    app.include_router(tools.router)
    app.dependency_overrides[get_current_user] = lambda: {"sub": _User.sub}
    client = TestClient(app, headers={"Origin": ORIGIN})
    _User.sub = "alice"
    yield client
    for uid in ("alice", "bob"):
        get_user_mcp_pool().forget(uid)
    _User.sub = "alice"


def _connect(api, url, name="fake", **extra):
    resp = api.post("/api/mcp-servers", json={"name": name, "transport": "streamable_http", "url": url, **extra})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _callback(api, redirect_url):
    parts = urlsplit(redirect_url)
    assert f"{parts.scheme}://{parts.netloc}" == PUBLIC_API
    return api.get(f"{parts.path}?{parts.query}")


def _sign_in(api, fake, name="fake"):
    body = _connect(api, fake.resource, name=name)
    assert body["status"] == "needs_auth", body
    page = _callback(api, auto_approve(fake, body["authorize_url"]))
    assert page.status_code == 200
    return body, page


def _db_dump(env) -> str:
    con = sqlite3.connect(str(env / "mcp.db"))
    try:
        rows = []
        for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            rows.append(repr(con.execute(f"SELECT * FROM {table}").fetchall()))
        return "\n".join(rows)
    finally:
        con.close()


def _pool():
    from services.user_mcp import get_user_mcp_pool

    return get_user_mcp_pool()


# ── detection ──────────────────────────────────────────────────────────────

def test_probe_tells_oauth_from_api_key_from_open(fake, keyed_server, env):
    import asyncio

    from services.mcp_oauth import probe

    oauth = asyncio.run(probe(fake.resource))
    assert oauth.kind == "oauth" and oauth.www_scope == "mcp:tools"
    assert str(oauth.prm.resource) == fake.resource
    assert asyncio.run(probe(keyed_server)).kind == "api_key"
    assert asyncio.run(probe("http://127.0.0.1:9/mcp")).kind == "unreachable"


def test_connect_starts_sign_in_with_pkce_resource_and_hashed_state(api, fake, env):
    body = _connect(api, fake.resource)
    assert body["status"] == "needs_auth" and body["server"]["auth"] == "oauth"
    q = {k: v[0] for k, v in parse_qs(urlsplit(body["authorize_url"]).query).items()}
    assert urlsplit(body["authorize_url"]).path == "/authorize"
    assert q["response_type"] == "code" and q["code_challenge_method"] == "S256"
    assert len(q["code_challenge"]) >= 43 and q["client_id"] in fake.clients
    assert q["redirect_uri"] == f"{PUBLIC_API}/api/mcp/oauth/callback"
    assert q["resource"] == fake.resource and q["scope"] == "mcp:tools"
    assert len(q["state"]) >= 43
    assert fake.clients[q["client_id"]]["client_name"] == "Quasar"
    dump = _db_dump(env)
    assert q["state"] not in dump, "the raw state must not be stored"
    # Listed as needing sign-in, no tools, and nothing connects in a chat.
    listed = api.get("/api/mcp-servers").json()
    assert listed[0]["status"]["state"] == "needs_auth"
    assert _pool().tools_for("alice") == []


# ── the full sign-in ───────────────────────────────────────────────────────

def test_sign_in_connects_and_tools_work_with_the_token(api, fake, env, capsys):
    _body, page = _sign_in(api, fake)
    html = page.text
    assert '"type": "quasar-mcp-oauth"' in html and '"ok": true' in html
    assert f'var origin = "{ORIGIN}"' in html and "postMessage(msg, origin)" in html
    assert '"*"' not in html
    assert "Connected to fake: 2 tools" in html
    csp = page.headers["content-security-policy"]
    assert "default-src 'none'" in csp and "frame-ancestors 'none'" in csp
    assert page.headers["referrer-policy"] == "no-referrer" and page.headers["cache-control"] == "no-store"

    listed = api.get("/api/mcp-servers").json()
    assert listed[0]["status"]["state"] == "connected"
    assert sorted(listed[0]["status"]["tools"]) == ["add", "echo"]

    tools = {t.name: t for t in _pool().tools_for("alice")}
    result = tools["fake__add"].function(a=2, b=3)
    assert result["success"] is True and "5" in str(result["data"]), result
    assert any(h.startswith("Bearer at_") for h in fake.mcp_auth_headers)

    # Encrypted at rest; never in responses or logs.
    access = next(iter(fake.access))
    refresh = next(iter(fake.refresh))
    dump = _db_dump(env)
    out = capsys.readouterr().out
    for secret in (access, refresh):
        assert secret not in dump and secret not in html and secret not in out
        assert secret not in json.dumps(listed)
    assert "code_" not in html


def test_state_cannot_be_replayed_forged_or_used_late(api, fake, env):
    body = _connect(api, fake.resource)
    redirect = auto_approve(fake, body["authorize_url"])
    assert '"ok": true' in _callback(api, redirect).text
    replay = _callback(api, redirect)
    assert '"ok": false' in replay.text and "already used" in replay.text

    forged = redirect.replace("state=", "state=x")
    assert "not valid" in _callback(api, forged).text

    body = _connect(api, fake.resource)
    late = auto_approve(fake, body["authorize_url"])
    con = sqlite3.connect(str(env / "mcp.db"))
    con.execute("UPDATE mcp_oauth_states SET created_at = created_at - 3600 WHERE used = 0")
    con.commit()
    con.close()
    assert "expired" in _callback(api, late).text
    assert fake.token_calls == ["authorization_code"], "rejected states never reach the token endpoint"


def test_denied_and_mismatched_issuer_are_rejected(api, fake, env):
    body = _connect(api, fake.resource)
    q = parse_qs(urlsplit(body["authorize_url"]).query)
    denied = api.get(f"/api/mcp/oauth/callback?error=access_denied&state={q['state'][0]}")
    assert '"ok": false' in denied.text and "cancelled" in denied.text

    body = _connect(api, fake.resource)
    redirect = auto_approve(fake, body["authorize_url"]).replace("iss=http", "iss=https%3A%2F%2Fevil.example%2Fx")
    page = _callback(api, redirect)
    assert '"ok": false' in page.text and "unexpected server" in page.text

    body = _connect(api, fake.resource)
    missing = auto_approve(fake, body["authorize_url"]).split("&iss=")[0]
    assert "missing its issuer" in _callback(api, missing).text
    assert fake.token_calls == []


def test_failed_page_still_reports_to_the_right_origin(api, fake, env):
    page = api.get("/api/mcp/oauth/callback?code=x&state=unknown")
    # Unknown state: no stored origin, so the production default is used.
    assert 'var origin = "https://www.quasarassistant.com"' in page.text
    assert "?mcp_oauth=failed" in page.text


# ── tokens over time ───────────────────────────────────────────────────────

def test_expired_token_is_refreshed_before_connecting(api, fake, env):
    from services.mcp_oauth_store import MCPOAuthStore

    _sign_in(api, fake)
    _pool().forget("alice")
    fake.expire_all_access()
    store = MCPOAuthStore()
    rec = store.get("alice", "fake")
    store.set_tokens("alice", "fake", access_token=rec.access_token, refresh_token=rec.refresh_token,
                     expires_in=1, scope=rec.scope)  # due for refresh
    old_access = rec.access_token
    tools = {t.name for t in _pool().tools_for("alice")}
    assert tools == {"fake__add", "fake__echo"}
    assert "refresh_token" in fake.token_calls
    assert store.get("alice", "fake").access_token != old_access
    assert all(r == fake.resource for r in fake.resources_seen), "RFC 8707 resource on every token call"


def test_401_mid_session_refreshes_once_and_replays(api, fake, env):
    _sign_in(api, fake)
    tools = {t.name: t for t in _pool().tools_for("alice")}
    fake.expire_all_access()  # the server now rejects the token we hold
    calls_before = fake.tool_calls
    result = tools["fake__echo"].function(text="hi")
    assert result["success"] is True and "echo: hi" in str(result["data"]), result
    assert fake.token_calls.count("refresh_token") == 1
    assert fake.tool_calls == calls_before + 1


def test_refresh_failure_means_needs_auth_without_retry_storm(api, fake, env):
    _sign_in(api, fake)
    _pool().forget("alice")
    fake.revoke_everything()
    assert _pool().tools_for("alice") == []
    listed = api.get("/api/mcp-servers").json()[0]
    assert listed["status"]["state"] == "needs_auth"
    assert "Reconnect" in listed["status"]["error"]
    seen = (len(fake.mcp_auth_headers), len(fake.token_calls))
    for _ in range(3):
        assert _pool().tools_for("alice") == []
    assert (len(fake.mcp_auth_headers), len(fake.token_calls)) == seen, "no retries after needs_auth"


def test_disconnect_revokes_and_reconnect_starts_over(api, fake, env):
    _sign_in(api, fake)
    refresh = next(iter(fake.refresh))
    resp = api.post("/api/mcp-servers/fake/disconnect").json()
    assert resp == {"status": "needs_auth", "revoked": True}
    assert refresh in fake.revoked
    assert _pool().tools_for("alice") == []
    again = api.post("/api/mcp-servers/fake/reconnect").json()
    assert again["status"] == "needs_auth" and "/authorize?" in again["authorize_url"]
    assert fake.registrations == 1, "the stored client registration is reused"
    page = _callback(api, auto_approve(fake, again["authorize_url"]))
    assert '"ok": true' in page.text


def test_delete_removes_tokens(api, fake, env):
    from services.mcp_oauth_store import MCPOAuthStore

    _sign_in(api, fake)
    assert api.delete("/api/mcp-servers/fake").status_code == 200
    assert MCPOAuthStore().get("alice", "fake") is None
    assert api.get("/api/mcp-servers").json() == []


# ── isolation, keys, policy ────────────────────────────────────────────────

def test_users_never_share_servers_or_tokens(api, fake, env):
    from services.mcp_oauth_store import MCPOAuthStore

    _sign_in(api, fake)
    _User.sub = "bob"
    assert api.get("/api/mcp-servers").json() == []
    assert _pool().tools_for("bob") == []
    assert MCPOAuthStore().get("bob", "fake") is None
    body = _connect(api, fake.resource)  # bob adds the same server: his own sign-in
    assert body["status"] == "needs_auth"
    assert _pool().tools_for("bob") == []
    assert api.post("/api/mcp-servers/fake/disconnect").status_code == 200
    _User.sub = "alice"
    assert {t.name for t in _pool().tools_for("alice")} == {"fake__add", "fake__echo"}


def test_api_key_server_asks_for_a_key_then_connects(api, keyed_server, env):
    body = _connect(api, keyed_server, name="keyed")
    assert body["status"] == "needs_api_key" and "API key" in body["message"]
    assert api.get("/api/mcp-servers").json() == [], "nothing is saved without the key"
    body = _connect(api, keyed_server, name="keyed", headers={"Authorization": "Bearer test-key-123456789"})
    assert body["status"] == "connected", body
    assert body["server"]["headers"] == {"Authorization": "********"}
    wrong = _connect(api, keyed_server, name="keyed", headers={"Authorization": "Bearer wrong-key-000000"})
    assert wrong["status"] == "needs_api_key"
    assert "rejected" in wrong["connection"]["error"]


def test_private_addresses_and_bad_names(api, env):
    resp = api.post("/api/mcp-servers", json={"name": "meta", "transport": "streamable_http",
                                              "url": "http://169.254.169.254/latest/mcp"})
    assert resp.status_code == 400 and "public internet address" in resp.json()["detail"]
    resp = api.post("/api/mcp-servers", json={"name": "", "transport": "streamable_http", "url": "file:///etc/passwd"})
    assert resp.status_code == 400


def test_url_policy_in_production(monkeypatch):
    from cryptography.fernet import Fernet

    from services.mcp_oauth import OAuthSetupError, public_api_base
    from services.mcp_url_policy import check_mcp_url

    monkeypatch.setenv("QUASAR_ENV", "production")
    monkeypatch.setenv("USER_API_KEY_FERNET_KEY", Fernet.generate_key().decode())
    assert check_mcp_url("http://localhost:8765/mcp")[0] is False
    assert check_mcp_url("http://127.0.0.1/mcp")[0] is False
    assert check_mcp_url("http://10.0.0.5/mcp")[0] is False
    assert check_mcp_url("https://user:pw@example.com/mcp")[0] is False
    assert check_mcp_url("http://93.184.215.14/mcp", require_https=True)[0] is False
    assert check_mcp_url("https://93.184.215.14/mcp", require_https=True) == (True, "")
    monkeypatch.delenv("QUASAR_PUBLIC_API_URL", raising=False)
    with pytest.raises(OAuthSetupError, match="QUASAR_PUBLIC_API_URL"):
        public_api_base("http://internal:8000/")
    monkeypatch.setenv("QUASAR_PUBLIC_API_URL", "https://quasar-oi14.onrender.com/")
    assert public_api_base(None) == "https://quasar-oi14.onrender.com"


def test_name_is_derived_from_the_host(api, fake, env):
    from api.routers.tools import _derive_name

    assert _derive_name("https://www.monocrawl.com/mcp") == "Monocrawl"
    assert _derive_name("https://mcp.deepwiki.com/mcp") == "Deepwiki"
    assert _derive_name("https://huggingface.co/mcp") == "Huggingface"


def test_connect_is_rate_limited(api, fake, env, monkeypatch):
    import services.mcp_oauth_store as mos

    monkeypatch.setattr(mos, "CONNECT_ATTEMPTS_PER_WINDOW", 2)
    _connect(api, fake.resource)
    _connect(api, fake.resource)
    resp = api.post("/api/mcp-servers", json={"name": "fake", "transport": "streamable_http", "url": fake.resource})
    assert resp.status_code == 429


def test_scrub_removes_oauth_values():
    from services.mcp_oauth import scrub

    text = scrub("failed: code=abc123&state=zzz client_secret=s3cr3t Bearer eyJabcdefghijklmnop")
    for leaked in ("abc123", "zzz", "s3cr3t", "eyJabcdefghijklmnop"):
        assert leaked not in text


# ── guard round 1 regressions (task-84382ab-7923, CX-01..07) ────────────────

def test_cx01_access_log_drops_the_callback_query(api):
    import logging

    from api.routers.tools import _CallbackQueryFilter

    access = logging.getLogger("uvicorn.access")
    assert any(isinstance(f, _CallbackQueryFilter) for f in access.filters)
    record = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
                               ("1.2.3.4:5", "GET", "/api/mcp/oauth/callback?code=SECRETCODE&state=SECRETSTATE", "1.1", 200),
                               None)
    for f in access.filters:
        f.filter(record)
    line = record.getMessage()
    assert "SECRETCODE" not in line and "SECRETSTATE" not in line and "/api/mcp/oauth/callback?[redacted]" in line
    other = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
                              ("1.2.3.4:5", "GET", "/api/mcp-servers?x=1", "1.1", 200), None)
    for f in access.filters:
        f.filter(other)
    assert "/api/mcp-servers?x=1" in other.getMessage()


def test_cx02_every_request_is_resolved_again(monkeypatch):
    import asyncio

    import httpx

    from services import vo_url_guard
    from services.mcp_url_policy import UnsafeMcpUrl, request_guard

    answers = iter([["93.184.215.14"], ["10.0.0.7"]])  # public, then rebound to private
    monkeypatch.setattr(vo_url_guard, "_resolve", lambda host: next(answers))
    hook = request_guard()
    req = httpx.Request("POST", "https://rebind.example/mcp")
    asyncio.run(hook(req))
    with pytest.raises(UnsafeMcpUrl):
        asyncio.run(hook(req))


def test_cx03_token_rejected_after_refresh_marks_needs_auth_without_redialing(api, fake, env):
    from services.mcp_oauth_store import MCPOAuthStore

    _sign_in(api, fake)
    _pool().forget("alice")
    fake.reject_tokens = True
    assert _pool().tools_for("alice") == []
    rec = MCPOAuthStore().get("alice", "fake")
    assert rec.status == "needs_auth" and rec.access_token is None
    assert fake.token_calls.count("refresh_token") == 1
    seen = len(fake.mcp_auth_headers)
    time.sleep(0.5)  # a background reconnect, if any, would dial now
    for _ in range(3):
        assert _pool().tools_for("alice") == []
    assert len(fake.mcp_auth_headers) == seen


def test_cx04_unreachable_probe_changes_nothing(api, fake, env, monkeypatch):
    from services import mcp_oauth
    from services.mcp_oauth_store import MCPOAuthStore

    _sign_in(api, fake)
    token = MCPOAuthStore().get("alice", "fake").access_token

    async def down(url):
        return mcp_oauth.ProbeResult("unreachable", "The server did not answer within 15 seconds.")

    monkeypatch.setattr(mcp_oauth, "probe", down)
    body = _connect(api, fake.resource)
    assert body["status"] == "error" and "Nothing was changed" in body["message"]
    listed = api.get("/api/mcp-servers").json()
    assert listed[0]["auth"] == "oauth"
    assert MCPOAuthStore().get("alice", "fake").access_token == token
    assert fake.revoked == []
    new = _connect(api, "http://127.0.0.1:9/mcp", name="new-one")
    assert new["status"] == "error"
    assert [s["name"] for s in api.get("/api/mcp-servers").json()] == ["fake"], "a new server is not saved either"


def test_cx05_disconnect_cancels_a_sign_in_in_flight(api, fake, env):
    from services.mcp_oauth_store import MCPOAuthStore

    body = _connect(api, fake.resource)
    pending = auto_approve(fake, body["authorize_url"])
    assert api.post("/api/mcp-servers/fake/disconnect").status_code == 200
    page = _callback(api, pending)
    assert '"ok": false' in page.text and "not valid" in page.text
    assert MCPOAuthStore().get("alice", "fake").access_token is None
    assert fake.token_calls == []


def test_cx05_tokens_are_not_stored_once_the_state_is_gone(tmp_path):
    from services.mcp_oauth_store import MCPOAuthStore

    store = MCPOAuthStore(db_path=str(tmp_path / "o.db"))
    store.save_registration("alice", "srv", server_url="https://s.example/mcp", resource="https://s.example/mcp",
                            issuer="https://as.example", metadata={}, client={"client_id": "c"},
                            redirect_uri="https://api.example/cb")
    raw = store.create_state("alice", "srv", code_verifier="v" * 64, resource="r", redirect_uri="u", return_origin="o")
    store.consume_state(raw)
    store.delete_states("alice", "srv")  # Disconnect while the code was being exchanged
    assert store.set_tokens("alice", "srv", access_token="late", refresh_token=None, expires_in=None,
                            scope=None, require_state=raw) is False
    assert store.get("alice", "srv").access_token is None


def test_cx06_issuer_comparison_is_exact():
    from services.mcp_oauth import _issuer_equal, _issuer_matches_listed_server

    # RFC 9207 iss and client reuse: exact strings only.
    assert _issuer_equal("https://as.example", "https://as.example")
    assert not _issuer_equal("https://as.example", "https://as.example/")
    assert not _issuer_equal("https://AS.example", "https://as.example")
    assert not _issuer_equal(None, "https://as.example")
    # RFC 8414 discovery: exact, plus only the "listed with a bare '/' path"
    # spelling (Semgrep). Host case and other path differences fail.
    match = _issuer_matches_listed_server
    assert match("https://login.semgrep.dev", "https://login.semgrep.dev/")
    assert match("https://as.example/t", "https://as.example/t")
    assert not match("https://as.example/", "https://as.example")
    assert not match("https://AS.example", "https://as.example/")
    assert not match("https://as.example/Tenant", "https://as.example/tenant")
    assert not match("https://as.example/t", "https://as.example/t/")
    assert not match("https://as.example", "http://as.example/")


@pytest.mark.parametrize("status,body,ctype", [
    (503, None, None), (500, None, None), (429, None, None), (408, None, None),
    (404, None, None), (405, None, None), (400, None, None),
    (200, None, None),                                             # 200 text error page
    (200, b'{"error": "maintenance"}', b"application/json"),      # 200 JSON, not JSON-RPC
    (200, b'{"jsonrpc": "2.0", "id": 0, "result": {}}', b"application/json"),  # no protocolVersion
    (200, b'{"jsonrpc": "2.0", "id": 0, "result": {"protocolVersion": "2025-06-18"}}',
     b"application/json"),                                         # no capabilities / serverInfo
    (200, b"event: message\ndata: not json\n\n", b"text/event-stream"),
])
def test_cx04_cx08_no_handshake_never_converts_or_revokes(api, fake, env, status, body, ctype):
    """Only a completed anonymous MCP initialize turns a signed-in server into
    a plain one. Outages (5xx, 408, 429), odd statuses (404, 405, 400) and 2xx
    replies that are not an initialize result change nothing, on a re-save
    or a Reconnect."""
    from services.mcp_oauth_store import MCPOAuthStore

    _sign_in(api, fake)
    token = MCPOAuthStore().get("alice", "fake").access_token
    fake.fail_status = status
    if body is not None:
        fake.fail_body, fake.fail_type = body, ctype
    body = _connect(api, fake.resource)  # re-save while the server misbehaves
    assert body["status"] == "error" and str(status) in body["message"] and "Nothing was changed" in body["message"]
    again = api.post("/api/mcp-servers/fake/reconnect").json()  # Reconnect while it misbehaves
    assert again["status"] == "error" and str(status) in again["message"]
    listed = api.get("/api/mcp-servers").json()[0]
    assert listed["auth"] == "oauth"
    rec = MCPOAuthStore().get("alice", "fake")
    assert rec is not None and rec.access_token == token
    assert fake.revoked == []
    fake.fail_status = None


def test_cx07_reconnect_adopts_a_server_that_stopped_asking_for_sign_in(api, fake, env):
    from services.mcp_oauth_store import MCPOAuthStore

    _sign_in(api, fake)
    api.post("/api/mcp-servers/fake/disconnect")
    fake.open_mode = True
    body = api.post("/api/mcp-servers/fake/reconnect").json()
    assert body["status"] == "connected", body
    listed = api.get("/api/mcp-servers").json()[0]
    assert listed["auth"] == "none" and listed["status"]["state"] == "connected"
    assert MCPOAuthStore().get("alice", "fake") is None


def test_cx04_initialize_detection_matches_the_sdk():
    """Codex's verify-4 cases: a result missing capabilities/serverInfo is
    rejected (the SDK's InitializeResult rejects it too), and a valid result
    split across two SSE data: lines is accepted (lines of one event join)."""
    import asyncio

    import httpx

    from services.mcp_oauth import _read_initialize_result

    full = {"jsonrpc": "2.0", "id": 0, "result": {"protocolVersion": "2025-06-18", "capabilities": {},
                                                  "serverInfo": {"name": "s", "version": "1"}}}
    partial = {"jsonrpc": "2.0", "id": 0, "result": {"protocolVersion": "2025-06-18"}}

    def run(body: bytes, ctype: str) -> bool:
        resp = httpx.Response(200, headers={"content-type": ctype}, content=body)
        return asyncio.run(_read_initialize_result(resp))

    assert run(json.dumps(full).encode(), "application/json") is True
    assert run(json.dumps(partial).encode(), "application/json; charset=utf-8") is False
    text = json.dumps(full, indent=1)
    split = "event: message\n" + "".join(f"data: {line}\n" for line in text.splitlines()) + "\n"
    assert run(split.encode(), "text/event-stream") is True
    assert run(f"event: message\ndata: {json.dumps(partial)}\n\n".encode(), "text/event-stream") is False
    assert run(json.dumps(full).encode(), "text/plain") is False


def test_cx07_cx08_sign_in_is_kept_unless_the_plain_connection_works(api, fake, env, monkeypatch):
    """Even a server that completes an anonymous initialize only replaces a
    signed-in config if a real connection without the sign-in then works;
    otherwise the OAuth config is restored and nothing is revoked."""
    from services.mcp_oauth_store import MCPOAuthStore
    from services.user_mcp import UserMCPPool

    import services.mcp_server_service as mss

    _sign_in(api, fake)
    token = MCPOAuthStore().get("alice", "fake").access_token
    revision = mss.MCPServerService().get_server("alice", "fake")["revision"]
    fake.open_mode = True  # the probe now sees a valid anonymous initialize
    monkeypatch.setattr(UserMCPPool, "test_config", lambda self, uid, cfg: {
        "connected": False, "state": "error", "tools": [], "error": "tools/list failed"})
    body = _connect(api, fake.resource)
    assert body["status"] == "error" and "existing sign-in was kept" in body["message"]
    again = api.post("/api/mcp-servers/fake/reconnect").json()
    assert again["status"] == "error" and "Nothing was changed" in again["message"]
    stored = mss.MCPServerService().get_server("alice", "fake")
    assert stored["auth"] == "oauth" and stored["revision"] == revision, "nothing was written at all"
    assert MCPOAuthStore().get("alice", "fake").access_token == token
    assert fake.revoked == []


def test_cx09_a_save_that_lands_in_between_beats_the_rollback(api, fake, env, monkeypatch):
    """Another worker re-saves the server while this request's plain
    connection test runs: the rollback must not overwrite that newer save,
    and the revoke must not fire on a server this request no longer owns."""
    import services.mcp_server_service as mss
    from services.user_mcp import UserMCPPool

    from services.mcp_oauth_store import MCPOAuthStore

    _sign_in(api, fake)
    fake.open_mode = True

    def interleaved(self, uid, cfg):
        # Another worker re-saves the server while this request verifies.
        mss.MCPServerService().save_server(uid, mss.MCPServerConfig(
            name=cfg["name"], transport="streamable_http", url="http://127.0.0.1:9/other"), auth="oauth")
        return {"connected": True, "state": "connected", "tools": ["echo"], "error": None}

    monkeypatch.setattr(UserMCPPool, "test_config", interleaved)
    body = _connect(api, fake.resource)
    assert body["status"] == "error" and body["saved"] is False
    stored = mss.MCPServerService().get_server("alice", "fake")
    assert stored["url"] == "http://127.0.0.1:9/other", "the newer save was not overwritten"
    assert MCPOAuthStore().get("alice", "fake") is not None and fake.revoked == [], "and nothing was revoked"


def test_cx09_a_sign_in_started_meanwhile_survives_the_switch(api, fake, env, monkeypatch):
    """The switch revokes and deletes only the OAuth row version it read: a
    sign-in another worker starts during the revocation keeps its row and
    its state."""
    from services import mcp_oauth
    from services.mcp_oauth_store import MCPOAuthStore

    _sign_in(api, fake)
    fake.open_mode = True
    real_revoke = mcp_oauth.revoke_tokens
    started = {}

    import services.mcp_server_service as mss

    async def revoke_then_new_sign_in(rec):
        out = await real_revoke(rec)
        # Exactly what the OAuth branch of another worker's connect does:
        # save the server as "oauth", register, create a state.
        time.sleep(0.002)
        mss.MCPServerService().save_server("alice", mss.MCPServerConfig(
            name="fake", transport="streamable_http", url="https://other.example/mcp"), auth="oauth")
        store = MCPOAuthStore()
        started["early"] = store.create_state("alice", "fake", code_verifier="w" * 64, resource="r",
                                              redirect_uri="u", return_origin="o")
        store.save_registration("alice", "fake", server_url="https://other.example/mcp",
                                resource="https://other.example/mcp", issuer="https://as.example",
                                metadata={}, client={"client_id": "c2"}, redirect_uri=f"{PUBLIC_API}/cb")
        started["state"] = store.create_state("alice", "fake", code_verifier="v" * 64, resource="r",
                                              redirect_uri="u", return_origin="o")
        return out

    monkeypatch.setattr(mcp_oauth, "revoke_tokens", revoke_then_new_sign_in)
    body = _connect(api, fake.resource)
    # Our switch was written; the final connection test then sees the other
    # worker's server (OAuth, sign-in not finished yet).
    assert body["saved"] is True and body["status"] == "needs_auth", body
    rec = MCPOAuthStore().get("alice", "fake")
    assert rec is not None and rec.server_url == "https://other.example/mcp", "the newer registration survived"
    for key in ("early", "state"):
        st, why = MCPOAuthStore().consume_state(started[key])
        assert st is not None and why == "", f"its {key} sign-in state is still usable"
    # The server row and the OAuth row agree: the later sign-in owns both.
    stored = mss.MCPServerService().get_server("alice", "fake")
    assert stored["auth"] == "oauth" and stored["url"] == "https://other.example/mcp"


def test_cx09_restore_if_revision_is_compare_and_set(tmp_path):
    import services.mcp_server_service as mss

    svc = mss.MCPServerService(base_dir=str(tmp_path))
    a = svc.save_server("alice", mss.MCPServerConfig(name="s", transport="streamable_http", url="https://a.example/mcp"))
    time.sleep(0.002)
    svc.save_server("alice", mss.MCPServerConfig(name="s", transport="streamable_http", url="https://b.example/mcp"))
    old = mss.MCPServerConfig(name="s", transport="streamable_http", url="https://old.example/mcp")
    assert svc.restore_if_revision("alice", a["revision"], old, auth="oauth") is False
    assert svc.get_server("alice", "s")["url"] == "https://b.example/mcp"
    cur = svc.get_server("alice", "s")["revision"]
    assert svc.restore_if_revision("alice", cur, old, auth="oauth") is True
    assert svc.get_server("alice", "s")["auth"] == "oauth"


def test_cx10_sse_reader_caps_raw_bytes_and_handles_chunk_boundaries():
    import asyncio

    import httpx

    from services import mcp_oauth

    class Chunks(httpx.AsyncByteStream):
        def __init__(self, chunks):
            self.chunks, self.served = chunks, 0

        async def __aiter__(self):
            for c in self.chunks:
                self.served += len(c)
                yield c

    def run(chunks):
        stream = Chunks(chunks)
        resp = httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
        return asyncio.run(mcp_oauth._read_initialize_result(resp)), stream.served

    full = json.dumps({"jsonrpc": "2.0", "id": 0, "result": {"protocolVersion": "2025-06-18", "capabilities": {},
                                                             "serverInfo": {"name": "sé", "version": "1"}}},
                      ensure_ascii=False)
    # One endless event with no terminating blank line (~400 KB offered): the
    # reader gives up on the RAW byte count, before the event completes.
    ok, served = run([b"data: " + b"x" * 8192 + b"\n"] * 50)
    assert ok is False and served <= mcp_oauth._INIT_READ_LIMIT + 8200
    # \r\n split across chunks and a multi-byte character split across chunks.
    raw = ("event: message\r\ndata: " + full + "\r\n\r\n").encode("utf-8")
    cut = raw.index("é".encode("utf-8")) + 1  # inside the two-byte "é"
    crlf = raw.index(b"\r\n") + 1              # between \r and \n
    assert run([raw[:crlf], raw[crlf:cut], raw[cut:]])[0] is True


def test_cx09_a_popup_finishing_during_the_switch_cannot_store_tokens(api, fake, env, monkeypatch):
    """A Reconnect popup is open; the user switches the server to plain; the
    popup's callback completes while the switch is revoking. The callback
    must not store tokens, and the end state is consistent: plain server, no
    OAuth row, no states."""
    import sqlite3 as _sqlite

    import services.mcp_server_service as mss
    from services import mcp_oauth
    from services.mcp_oauth_store import MCPOAuthStore

    _sign_in(api, fake)
    pending = api.post("/api/mcp-servers/fake/reconnect").json()
    assert pending["status"] == "needs_auth"
    redirect = auto_approve(fake, pending["authorize_url"])
    q = {k: v[0] for k, v in parse_qs(urlsplit(redirect).query).items()}
    fake.open_mode = True
    real_revoke = mcp_oauth.revoke_tokens
    outcome = {}

    async def revoke_while_callback_lands(rec):
        out = await real_revoke(rec)
        outcome["cb"] = await mcp_oauth.finish_sign_in(state=q["state"], code=q["code"], iss=q.get("iss"))
        return out

    monkeypatch.setattr(mcp_oauth, "revoke_tokens", revoke_while_callback_lands)
    body = _connect(api, fake.resource)
    assert body["status"] == "connected" and body["saved"] is True, body
    assert outcome["cb"].ok is False and "disconnected or removed" in outcome["cb"].message
    assert mss.MCPServerService().get_server("alice", "fake")["auth"] == "none"
    assert MCPOAuthStore().get("alice", "fake") is None
    con = _sqlite.connect(str(env / "mcp.db"))
    assert con.execute("SELECT COUNT(*) FROM mcp_oauth_states").fetchone()[0] == 0
    con.close()


def test_cx09_reconnect_never_signs_in_onto_a_server_switched_meanwhile(api, fake, env, monkeypatch):
    """Reconnect reads an OAuth server and probes it; another worker switches
    the row to plain before the sign-in starts. The claim fails: no new
    registration or state is written, and the plain row stays as it is."""
    import sqlite3 as _sqlite

    import services.mcp_server_service as mss
    from services import mcp_oauth

    _sign_in(api, fake)
    real_probe = mcp_oauth.probe

    async def probe_then_switch(url):
        result = await real_probe(url)
        mss.MCPServerService().save_server("alice", mss.MCPServerConfig(
            name="fake", transport="streamable_http", url=fake.resource), auth="none")
        return result

    con = _sqlite.connect(str(env / "mcp.db"))
    states_before = con.execute("SELECT COUNT(*) FROM mcp_oauth_states").fetchone()[0]
    oauth_before = con.execute("SELECT updated_at FROM user_mcp_oauth").fetchone()[0]
    con.close()
    monkeypatch.setattr(mcp_oauth, "probe", probe_then_switch)
    body = api.post("/api/mcp-servers/fake/reconnect").json()
    assert body["status"] == "error" and "changed by another request" in body["message"]
    assert mss.MCPServerService().get_server("alice", "fake")["auth"] == "none"
    con = _sqlite.connect(str(env / "mcp.db"))
    assert con.execute("SELECT COUNT(*) FROM mcp_oauth_states").fetchone()[0] == states_before
    assert con.execute("SELECT updated_at FROM user_mcp_oauth").fetchone()[0] == oauth_before
    con.close()


def test_fe_cx03_derived_name_never_replaces_a_different_server(api, keyed_server, env):
    import services.mcp_server_service as mss

    mss.MCPServerService().save_server("alice", mss.MCPServerConfig(
        name="docs", transport="streamable_http", url="https://a.example/mcp"))
    resp = api.post("/api/mcp-servers", json={"name": "docs", "transport": "streamable_http",
                                              "url": "https://b.example/mcp", "replace": False})
    assert resp.status_code == 409
    assert mss.MCPServerService().get_server("alice", "docs")["url"] == "https://a.example/mcp"


def test_fe_cx20_attempt_id_round_trip(api, fake, env):
    import hashlib

    body = _connect(api, fake.resource)
    state = parse_qs(urlsplit(body["authorize_url"]).query)["state"][0]
    assert body["attempt"] == hashlib.sha256(state.encode()).hexdigest()[:16]
    assert state not in json.dumps(body["attempt"])
    listed = api.get("/api/mcp-servers").json()[0]
    assert listed["oauth"]["signed_in"] is False and listed["oauth"]["signed_in_attempt"] is None
    _callback(api, auto_approve(fake, body["authorize_url"]))
    listed = api.get("/api/mcp-servers").json()[0]
    assert listed["oauth"]["signed_in_attempt"] == body["attempt"]
    assert isinstance(listed["oauth"]["signed_in_at"], float)
    assert "revision" not in listed
    # The callback's connection test settled AFTER the sign-in (frontend CX-26).
    assert listed["status"]["state"] == "connected"
    assert listed["status"]["checked_at"] >= listed["oauth"]["signed_in_at"]


def test_q2_signed_in_but_not_connected_is_reported_as_such(api, fake, env, monkeypatch):
    from services.user_mcp import UserMCPPool

    body = _connect(api, fake.resource)
    redirect = auto_approve(fake, body["authorize_url"])
    monkeypatch.setattr(UserMCPPool, "test", lambda self, uid, name: {
        "connected": False, "state": "error", "tools": [], "error": "fake went away"})
    page = _callback(api, redirect)
    assert '"ok": true' in page.text and "fake went away" in page.text
    assert "mcp_oauth=signed_in" in page.text and "Signed in, not connected yet" in page.text


def test_q3_keys_in_url_queries_never_reach_the_browser(api, keyed_server, env):
    from services.mcp_server_service import mask_server

    masked = mask_server({"name": "x", "url": "https://h.example/mcp?api_key=SECRET123&x=1"})
    assert "SECRET123" not in masked["url"] and "x=1" in masked["url"]
    body = _connect(api, keyed_server + "?api_key=SECRET123", name="keyed")
    assert body["status"] == "needs_api_key" and "SECRET123" not in json.dumps(body)
