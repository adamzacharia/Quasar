"""Per-user MCP pool (services/user_mcp.py).

The web app's shared agent never loaded servers saved in Settings > MCP
Servers (verified live 2026-09-27). These tests pin the replacement: a real
MCP server (FastMCP over streamable HTTP, started in-process on a free port)
is connected per user, its tools are callable, a reconnect never leaves a
tool pointing at a closed session, other users never see the tools, and the
stdio RCE gate still holds.
"""

import socket
import threading
import time

import pytest

mcp = pytest.importorskip("mcp")
uvicorn = pytest.importorskip("uvicorn")

import services.mcp_server_service as mss
from services import user_mcp


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def mcp_url():
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("adder")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    port = _free_port()
    config = uvicorn.Config(server.streamable_http_app(), host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    assert srv.started, "test MCP server did not start"
    yield f"http://127.0.0.1:{port}/mcp"
    srv.should_exit = True
    thread.join(timeout=5)


@pytest.fixture()
def pool(tmp_path, monkeypatch):
    orig = mss.MCPServerService.__init__
    monkeypatch.setattr(mss.MCPServerService, "__init__", lambda self, base_dir=str(tmp_path): orig(self, base_dir))
    p = user_mcp.UserMCPPool()
    yield p
    p.forget("alice")


def _save(name, **kw):
    mss.MCPServerService().save_server("alice", mss.MCPServerConfig(name=name, **kw))


def test_saved_server_tools_reach_only_that_user(pool, mcp_url):
    _save("calc", transport="streamable_http", url=mcp_url)
    tools = {t.name: t for t in pool.tools_for("alice")}
    assert "calc__add" in tools
    result = tools["calc__add"].function(a=2, b=40)
    assert result["success"] is True, result
    assert "42" in str(result.get("data"))
    assert pool.tools_for("bob") == []
    assert pool.tools_for("anonymous") == []


def test_legacy_http_setting_falls_back_to_streamable(pool, mcp_url):
    # The old form stored modern /mcp URLs under the SSE-only "http" transport.
    _save("calc", transport="http", url=mcp_url)
    status = pool.test("alice", "calc")
    assert status["connected"] is True, status
    assert status["transport"] == "streamable_http"
    assert status["tools"] == ["add"]


def test_tool_survives_a_reconnect(pool, mcp_url):
    _save("calc", transport="streamable_http", url=mcp_url)
    add = {t.name: t for t in pool.tools_for("alice")}["calc__add"]
    assert pool.test("alice", "calc")["connected"] is True  # replaces the session
    result = add.function(a=1, b=1)
    assert result["success"] is True, result


def test_unreachable_server_reports_error_and_is_skipped(pool, mcp_url):
    _save("calc", transport="streamable_http", url=mcp_url)
    _save("dead", transport="streamable_http", url="http://127.0.0.1:9/mcp")
    names = [t.name for t in pool.tools_for("alice")]
    assert names == ["calc__add"]
    status = pool.test("alice", "dead")
    assert status["connected"] is False and status["error"]


def test_stdio_configs_are_never_spawned_when_gated(pool, monkeypatch, tmp_path):
    monkeypatch.setenv("QUASAR_ENABLE_MCP_STDIO", "1")
    _save("local", transport="stdio", command="definitely-not-a-real-binary")
    monkeypatch.delenv("QUASAR_ENABLE_MCP_STDIO")
    assert pool.tools_for("alice") == []
    assert pool.test("alice", "local")["connected"] is False


def test_function_names_are_provider_safe():
    assert user_mcp.tool_function_name("my server", "do.thing/x") == "my_server__do_thing_x"
    assert len(user_mcp.tool_function_name("s" * 40, "t" * 40)) == 64
