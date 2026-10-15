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


# ── default servers (MANNA in every account) ──────────────────────────────

DEFAULT_URL = "https://manna.example/manna/mcp"


@pytest.fixture()
def manna_default(monkeypatch):
    monkeypatch.setenv("QUASAR_DEFAULT_MANNA_URL", DEFAULT_URL)
    monkeypatch.delenv("QUASAR_ENABLE_MANNA", raising=False)
    mss._DEFAULTS_SEEDED.clear()
    mss._MARKS_READY.clear()
    yield
    mss._DEFAULTS_SEEDED.clear()
    mss._MARKS_READY.clear()


def _urls(svc, user):
    return [(s["name"], s["url"]) for s in svc.load_servers(user)]


def test_every_account_starts_with_manna(svc, manna_default):
    for user in ("alice", "bob"):
        servers = svc.load_servers(user)
        assert [(s["name"], s["transport"], s["url"], s["auth"]) for s in servers] == [
            ("manna", "streamable_http", DEFAULT_URL, "none")]
    mss._DEFAULTS_SEEDED.clear()  # a new process adds nothing twice
    assert len(svc.load_servers("alice")) == 1


def test_removed_default_stays_removed(svc, manna_default):
    svc.load_servers("alice")
    assert svc.delete_server("alice", "manna") is True
    mss._DEFAULTS_SEEDED.clear()  # e.g. after a restart
    assert svc.load_servers("alice") == []


def test_stale_seed_after_removal_does_not_resurrect(svc, manna_default, tmp_path):
    """CX-02: a request that had not seen the seed (another worker, empty
    memo) runs after the user removed MANNA: the mark still blocks it."""
    svc.load_servers("alice")
    svc.delete_server("alice", "manna")
    other_worker = mss.MCPServerService(base_dir=str(tmp_path))
    mss._DEFAULTS_SEEDED.clear()
    other_worker._seed_defaults("alice")
    assert other_worker.load_servers("alice") == []


def test_users_own_manna_entry_is_left_alone(svc, manna_default, monkeypatch):
    svc.save_server("alice", mss.MCPServerConfig(name="manna", transport="streamable_http",
                                                 url="https://mine.example/mcp"))
    mss._DEFAULTS_SEEDED.clear()
    assert _urls(svc, "alice") == [("manna", "https://mine.example/mcp")]
    # Even when the default later happens to equal, then leave, that URL.
    monkeypatch.setenv("QUASAR_DEFAULT_MANNA_URL", "https://mine.example/mcp")
    svc.load_servers("alice")
    monkeypatch.setenv("QUASAR_DEFAULT_MANNA_URL", "https://moved.example/mcp")
    assert _urls(svc, "alice") == [("manna", "https://mine.example/mcp")]


def test_preexisting_row_at_the_default_url_is_not_taken_over(svc, manna_default, monkeypatch):
    """CX-01: same URL as the default, but saved by the user: never follows."""
    svc.save_server("alice", mss.MCPServerConfig(name="manna", transport="streamable_http", url=DEFAULT_URL))
    mss._DEFAULTS_SEEDED.clear()
    svc.load_servers("alice")
    monkeypatch.setenv("QUASAR_DEFAULT_MANNA_URL", "https://moved.example/mcp")
    assert _urls(svc, "alice") == [("manna", DEFAULT_URL)]


def test_untouched_default_follows_url_changes_both_ways(svc, manna_default, monkeypatch):
    """CX-04/05: the URL is resolved per load, so A -> B -> A works inside one
    process and nothing is written back for workers to fight over."""
    svc.load_servers("alice")
    for url in ("https://b.example/mcp", DEFAULT_URL):
        monkeypatch.setenv("QUASAR_DEFAULT_MANNA_URL", url)
        assert _urls(svc, "alice") == [("manna", url)]
    con = sqlite3.connect(str(svc._db_path))
    try:
        assert con.execute("SELECT url FROM user_mcp_servers WHERE name = 'manna'").fetchone()[0] == DEFAULT_URL
    finally:
        con.close()


def test_any_edit_stops_following(svc, manna_default, monkeypatch):
    """CX-01: a header-only edit (same URL) makes the row the user's own."""
    svc.load_servers("alice")
    svc.save_server("alice", mss.MCPServerConfig(name="manna", transport="streamable_http", url=DEFAULT_URL,
                                                 headers={"Authorization": f"Bearer {SECRET}"}))
    monkeypatch.setenv("QUASAR_DEFAULT_MANNA_URL", "https://elsewhere.example/mcp")
    [srv] = svc.load_servers("alice")
    assert srv["url"] == DEFAULT_URL and srv["headers"]["Authorization"] == f"Bearer {SECRET}"


@pytest.mark.parametrize("value", ["off", "0", "", "none"])
def test_default_can_be_turned_off(svc, monkeypatch, value):
    monkeypatch.setenv("QUASAR_DEFAULT_MANNA_URL", value)
    mss._DEFAULTS_SEEDED.clear()
    assert svc.load_servers("alice") == []


def test_no_default_when_platform_already_mounts_manna(svc, manna_default, monkeypatch):
    monkeypatch.setenv("QUASAR_ENABLE_MANNA", "1")
    assert svc.load_servers("alice") == []


def test_marks_table_failure_never_hides_saved_servers(svc, manna_default, monkeypatch):
    """CX-06: the optional marks table cannot be created -> no default, but
    the constructor works and saved servers still load."""
    monkeypatch.setenv("QUASAR_DEFAULT_MANNA_URL", "off")  # no marks table yet
    svc.save_server("alice", mss.MCPServerConfig(name="x", transport="streamable_http", url="https://x.example/mcp"))
    monkeypatch.setenv("QUASAR_DEFAULT_MANNA_URL", DEFAULT_URL)
    real_conn = svc._conn

    def deny_marks():
        con = real_conn()

        def authorizer(action, arg1, *_):
            if action == sqlite3.SQLITE_CREATE_TABLE and arg1 == "user_mcp_default_marks":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        con.set_authorizer(authorizer)
        return con

    monkeypatch.setattr(mss, "_SCHEMA_READY", set())
    monkeypatch.setattr(mss.MCPServerService, "_conn", lambda self: deny_marks())
    fresh = mss.MCPServerService(base_dir=str(svc.base_dir))
    assert _urls(fresh, "alice") == [("x", "https://x.example/mcp")]


def test_stock_default_is_the_hosted_manna(monkeypatch):
    monkeypatch.delenv("QUASAR_DEFAULT_MANNA_URL", raising=False)
    monkeypatch.delenv("QUASAR_ENABLE_MANNA", raising=False)
    [cfg] = mss.default_mcp_servers()
    assert cfg.name == "manna" and cfg.transport == "streamable_http"
    assert cfg.url.startswith("https://") and cfg.url.endswith("/manna/mcp")


def test_platform_tool_wins_over_a_same_named_user_tool():
    """CX-03: platform MANNA plus an account's seeded MANNA offer each
    manna__* name once, and the platform copy is the one kept."""
    from types import SimpleNamespace

    from core.runner import _merge_user_mcp_tools

    tools = [{"type": "function", "name": "manna__run_adql_query", "description": "platform"}]
    user_tools: dict = {}
    offered = [SimpleNamespace(name=n, description="user", parameters={})
               for n in ("manna__run_adql_query", "manna__list_archives", "manna__list_archives")]
    _merge_user_mcp_tools(tools, user_tools, offered)
    names = [t["name"] for t in tools]
    assert names == ["manna__run_adql_query", "manna__list_archives"]
    assert tools[0]["description"] == "platform" and set(user_tools) == {"manna__list_archives"}


def test_unreadable_marks_table_never_hides_saved_servers(svc, manna_default, monkeypatch):
    """CX-07: the marks table exists but cannot be read -> saved servers
    (the seeded row included) still load, just without URL following."""
    svc.save_server("alice", mss.MCPServerConfig(name="x", transport="streamable_http", url="https://x.example/mcp"))
    real_conn = svc._conn

    def deny_reads():
        con = real_conn()

        def authorizer(action, arg1, *_):
            if action == sqlite3.SQLITE_READ and arg1 == "user_mcp_default_marks":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        con.set_authorizer(authorizer)
        return con

    monkeypatch.setattr(mss.MCPServerService, "_conn", lambda self: deny_reads())
    mss._DEFAULTS_SEEDED.clear()
    assert sorted(n for n, _ in _urls(svc, "alice")) == ["manna", "x"]
