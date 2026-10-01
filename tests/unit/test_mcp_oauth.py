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
