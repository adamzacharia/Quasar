"""MCP server configs and OAuth rows in the database (Phase 0).

Configs used to be plain JSON on a disk Render wipes on redeploy, with API
keys in clear text. These tests pin the replacement: encrypted at rest,
imported once from the legacy file (renamed, never deleted), isolated per
user, and still on the local SQLite branch under the test guard.
"""

import json
import sqlite3
import time

import pytest

import services.mcp_server_service as mss
from services.mcp_oauth_store import MCPOAuthStore

SECRET = "sk-live-abcdefghijklmnop0123456789"


@pytest.fixture()
def svc(tmp_path):
    return mss.MCPServerService(base_dir=str(tmp_path))


def _raw_rows(db_path):
    con = sqlite3.connect(db_path)
    try:
        return con.execute("SELECT * FROM user_mcp_servers").fetchall()
    finally:
        con.close()


def test_tests_never_touch_turso():
    """tests/conftest.py must keep forcing the local DB (turso-test-leak)."""
    import os

    from services import db

    assert os.environ.get("QUASAR_FORCE_LOCAL_DB") == "1"
    assert db.is_using_turso() is False


def test_headers_and_env_are_encrypted_at_rest(svc, tmp_path):
    svc.save_server("alice", mss.MCPServerConfig(
        name="paid", transport="streamable_http", url="https://example.org/mcp",
        headers={"Authorization": f"Bearer {SECRET}"}))
    raw = repr(_raw_rows(str(tmp_path / "user_mcp.db")))
    assert SECRET not in raw and "Bearer" not in raw
    loaded = svc.load_servers("alice")
    assert loaded[0]["headers"] == {"Authorization": f"Bearer {SECRET}"}
    assert loaded[0]["auth"] == "none"
    masked = mss.mask_server(loaded[0])
    assert masked["headers"] == {"Authorization": "********"} and SECRET not in json.dumps(masked)


def test_update_keeps_auth_kind_and_order(svc):
    svc.save_server("alice", mss.MCPServerConfig(name="a", transport="streamable_http", url="https://a.example/mcp"),
                    auth="oauth")
    svc.save_server("alice", mss.MCPServerConfig(name="b", transport="streamable_http", url="https://b.example/mcp"))
    svc.save_server("alice", mss.MCPServerConfig(name="a", transport="streamable_http", url="https://a2.example/mcp"))
    servers = svc.load_servers("alice")
    assert [s["name"] for s in servers] == ["a", "b"]
    assert servers[0]["url"] == "https://a2.example/mcp" and servers[0]["auth"] == "oauth"
    assert svc.delete_server("alice", "a") is True
    assert svc.delete_server("alice", "a") is False
    assert [s["name"] for s in svc.load_servers("alice")] == ["b"]


def test_users_are_isolated(svc):
    svc.save_server("alice", mss.MCPServerConfig(name="x", transport="streamable_http", url="https://x.example/mcp",
                                                 headers={"x-api-key": SECRET}))
    assert svc.load_servers("bob") == []
    assert svc.delete_server("bob", "x") is False
    assert svc.load_servers("alice")[0]["headers"]["x-api-key"] == SECRET


def test_legacy_json_is_imported_once_encrypted_and_renamed(svc, tmp_path):
    legacy_dir = tmp_path / "alice"
    legacy_dir.mkdir()
    legacy = legacy_dir / "mcp_servers.json"
    legacy.write_text(json.dumps([
        {"name": "deepwiki", "transport": "streamable_http", "url": "https://mcp.deepwiki.com/mcp"},
        {"name": "keyed", "transport": "http", "url": "https://k.example/sse", "headers": {"Authorization": f"Bearer {SECRET}"}},
        {"name": "", "transport": "streamable_http"},  # bad entry: skipped, not fatal
    ]), encoding="utf-8")

    servers = svc.load_servers("alice")
    assert [s["name"] for s in servers] == ["deepwiki", "keyed"]
    assert servers[1]["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert not legacy.exists()
    renamed = list(legacy_dir.glob("mcp_servers.json.migrated-*"))
    assert len(renamed) == 1, "the user's file is renamed, never deleted"
    assert SECRET not in repr(_raw_rows(str(tmp_path / "user_mcp.db")))

    # A second load does not re-import or duplicate anything.
    assert [s["name"] for s in svc.load_servers("alice")] == ["deepwiki", "keyed"]


def test_legacy_import_skips_names_already_in_db(svc, tmp_path):
    svc.save_server("alice", mss.MCPServerConfig(name="deepwiki", transport="streamable_http",
                                                 url="https://new.example/mcp"))
    (tmp_path / "alice").mkdir(exist_ok=True)
    (tmp_path / "alice" / "mcp_servers.json").write_text(json.dumps([
        {"name": "deepwiki", "transport": "streamable_http", "url": "https://old.example/mcp"}]))
    assert svc.load_servers("alice")[0]["url"] == "https://new.example/mcp"


def test_legacy_lookup_refuses_path_like_user_ids(svc):
    for uid in ("..", "a/b", "a\\b", "c:x"):
        assert svc._legacy_file(uid) is None


def test_stdio_gate_still_holds(svc, monkeypatch):
    monkeypatch.delenv("QUASAR_ENABLE_MCP_STDIO", raising=False)
    with pytest.raises(ValueError, match="disabled"):
        svc.save_server("alice", mss.MCPServerConfig(name="local", transport="stdio", command="npx"))


def test_oauth_rows_are_encrypted_and_per_user(tmp_path):
    store = MCPOAuthStore(db_path=str(tmp_path / "o.db"))
    store.save_registration("alice", "srv", server_url="https://s.example/mcp", resource="https://s.example/mcp",
                            issuer="https://auth.example", metadata={"token_endpoint": "https://auth.example/token"},
                            client={"client_id": "cid", "client_secret": "csecret-123456789"},
                            redirect_uri="https://api.example/api/mcp/oauth/callback")
    store.set_tokens("alice", "srv", access_token="at-SECRET-123456", refresh_token="rt-SECRET-654321",
                     expires_in=3600, scope="mcp")
    con = sqlite3.connect(str(tmp_path / "o.db"))
    raw = repr(con.execute("SELECT * FROM user_mcp_oauth").fetchall())
    con.close()
    for secret in ("at-SECRET-123456", "rt-SECRET-654321", "csecret-123456789"):
        assert secret not in raw
    rec = store.get("alice", "srv")
    assert rec.access_token == "at-SECRET-123456" and rec.refresh_token == "rt-SECRET-654321"
    assert rec.client["client_secret"] == "csecret-123456789" and rec.status == "authorized"
    assert store.get("bob", "srv") is None
    store.mark_needs_auth("alice", "srv", "expired")
    rec = store.get("alice", "srv")
    assert rec.access_token is None and rec.status == "needs_auth" and rec.status_reason == "expired"


def test_changed_resource_drops_old_tokens(tmp_path):
    store = MCPOAuthStore(db_path=str(tmp_path / "o.db"))
    kw = dict(issuer="https://auth.example", metadata={}, client={"client_id": "c"},
              redirect_uri="https://api.example/cb")
    store.save_registration("alice", "srv", server_url="https://s.example/mcp", resource="https://s.example/mcp", **kw)
    store.set_tokens("alice", "srv", access_token="t1", refresh_token=None, expires_in=None, scope=None)
    store.save_registration("alice", "srv", server_url="https://s.example/mcp", resource="https://s.example/mcp", **kw)
    assert store.get("alice", "srv").access_token == "t1"  # same resource: kept
    store.save_registration("alice", "srv", server_url="https://evil.example/mcp", resource="https://evil.example/mcp", **kw)
    assert store.get("alice", "srv").access_token is None


def test_state_is_single_use_hashed_and_expires(tmp_path):
    store = MCPOAuthStore(db_path=str(tmp_path / "o.db"))
    raw = store.create_state("alice", "srv", code_verifier="v" * 64, resource="https://s.example/mcp",
                             redirect_uri="https://api.example/cb", return_origin="http://localhost:3001")
    assert len(raw) >= 43
    con = sqlite3.connect(str(tmp_path / "o.db"))
    dump = repr(con.execute("SELECT * FROM mcp_oauth_states").fetchall())
    con.close()
    assert raw not in dump and "v" * 64 not in dump
    assert store.consume_state("nope") == (None, "unknown")
    rec, why = store.consume_state(raw)
    assert why == "" and rec.user_id == "alice" and rec.code_verifier == "v" * 64
    assert store.consume_state(raw) == (None, "used")

    old = store.create_state("alice", "srv", code_verifier="w" * 64, resource="r", redirect_uri="u",
                             return_origin="o")
    con = sqlite3.connect(str(tmp_path / "o.db"))
    con.execute("UPDATE mcp_oauth_states SET created_at = ?", (time.time() - 3600,))
    con.commit()
    con.close()
    assert store.consume_state(old) == (None, "expired")


def test_connect_rate_limit(tmp_path, monkeypatch):
    import services.mcp_oauth_store as mos

    monkeypatch.setattr(mos, "CONNECT_ATTEMPTS_PER_WINDOW", 3)
    store = MCPOAuthStore(db_path=str(tmp_path / "o.db"))
    for _ in range(3):
        store.admit_connect("alice")
    with pytest.raises(mos.ConnectRateLimited):
        store.admit_connect("alice")
    store.admit_connect("bob")  # per user
