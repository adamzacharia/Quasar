"""Per-user MCP pool (services/user_mcp.py).

The web app's shared agent never loaded servers saved in Settings > MCP
Servers (verified live 2026-09-27). These tests pin the replacement: a real
MCP server (FastMCP over streamable HTTP, started in-process on a free port)
is connected per user, its tools are callable, a reconnect never leaves a
tool pointing at a closed session, other users never see the tools, and the
stdio RCE gate still holds.
"""

import json
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


def _serve(app):
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    assert srv.started, "test MCP server did not start"
    return srv, thread, f"http://127.0.0.1:{port}/mcp"


@pytest.fixture(scope="module")
def mcp_url():
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("adder")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @server.tool()
    def boom() -> str:
        """Always fails."""
        raise ValueError("upstream exploded")

    srv, thread, url = _serve(server.streamable_http_app())
    yield url
    srv.should_exit = True
    thread.join(timeout=5)


def _reject_paid_tool_calls(app, tool="get_balance", status=401):
    """ASGI middleware: answer `status` to tools/call POSTs for `tool`, like
    www.monocrawl.com/mcp does for credit-gated tools without an API key
    (initialize and tools/list stay public)."""
    async def gated(scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await app(scope, receive, send)
        chunks = []
        while True:
            msg = await receive()
            chunks.append(msg.get("body", b""))
            if not msg.get("more_body"):
                break
        body = b"".join(chunks)
        try:
            payload = json.loads(body)
        except ValueError:
            payload = {}
        if payload.get("method") == "tools/call" and (payload.get("params") or {}).get("name") == tool:
            await send({"type": "http.response.start", "status": status,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": b'{"error":"API key required"}'})
            return
        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        return await app(scope, replay, send)

    return gated


@pytest.fixture(scope="module")
def paid_mcp_url():
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("paid")

    @server.tool()
    def get_balance() -> str:
        """Credit-gated."""
        return "100 credits"

    @server.tool()
    def ping() -> str:
        """Free."""
        return "pong"

    srv, thread, url = _serve(_reject_paid_tool_calls(server.streamable_http_app()))
    yield url
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


def test_tools_carry_server_identity_and_failures_are_detectable(pool, mcp_url):
    """The Research timeline badges MCP calls by server and marks failed ones
    (core/runner.py _user_mcp_step_meta + tool_result_failure_reason)."""
    from core.runner import _user_mcp_step_meta
    from core.turn_recovery import tool_result_failure_reason

    _save("calc", transport="streamable_http", url=mcp_url)
    tools = {t.name: t for t in pool.tools_for("alice")}
    add, boom = tools["calc__add"], tools["calc__boom"]
    assert (add.mcp_server, add.mcp_tool) == ("calc", "add")
    assert tool_result_failure_reason(add.function(a=1, b=2)) is None
    reason = tool_result_failure_reason(boom.function())
    assert reason and "upstream exploded" in reason
    meta = _user_mcp_step_meta(boom, "calc__boom", "Running calc  boom")
    assert meta == {"step": "calc: boom", "server": "calc", "tool": "boom"}


def test_legacy_http_setting_falls_back_to_streamable(pool, mcp_url):
    # The old form stored modern /mcp URLs under the SSE-only "http" transport.
    _save("calc", transport="http", url=mcp_url)
    status = pool.test("alice", "calc")
    assert status["connected"] is True, status
    assert status["transport"] == "streamable_http"
    assert status["tools"] == ["add", "boom"]


def test_tool_survives_a_reconnect(pool, mcp_url):
    _save("calc", transport="streamable_http", url=mcp_url)
    add = {t.name: t for t in pool.tools_for("alice")}["calc__add"]
    assert pool.test("alice", "calc")["connected"] is True  # replaces the session
    result = add.function(a=1, b=1)
    assert result["success"] is True, result


def test_http_401_on_tool_call_fails_fast_and_server_recovers(pool, paid_mcp_url, monkeypatch):
    """A 401 on the tools/call POST kills the SDK transport's task group and
    the pending call_tool never resolves; it used to wait out the whole MCP
    call timeout (120 s) and then say only "timed out" (seen live 2026-10-01
    against www.monocrawl.com/mcp)."""
    monkeypatch.setenv("QUASAR_MCP_TOOL_TIMEOUT_SECONDS", "30")
    _save("monocrawl", transport="streamable_http", url=paid_mcp_url)
    tools = {t.name: t for t in pool.tools_for("alice")}
    assert set(tools) == {"monocrawl__get_balance", "monocrawl__ping"}

    started = time.monotonic()
    result = tools["monocrawl__get_balance"].function()
    elapsed = time.monotonic() - started
    assert elapsed < 5, f"401 took {elapsed:.1f}s to surface"
    assert result["success"] is False, result
    assert "monocrawl rejected the call (401 Unauthorized)" in result["error"]
    assert "Authorization: Bearer <key>" in result["error"]

    # The drop reconnects in the background; the free tool keeps working
    # through the same Tool object, and the 401 repeats fast, not slow.
    assert tools["monocrawl__ping"].function()["success"] is True
    started = time.monotonic()
    again = tools["monocrawl__get_balance"].function()
    assert time.monotonic() - started < 5 and "401" in again["error"]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not pool.status("alice").get("monocrawl", {}).get("connected"):
        time.sleep(0.05)
    assert pool.status("alice")["monocrawl"]["connected"] is True


def test_http_error_messages():
    import httpx

    def status_error(code):
        req = httpx.Request("POST", "https://example.test/mcp")
        return httpx.HTTPStatusError("x", request=req, response=httpx.Response(code, request=req))

    cfg = {"name": "srv"}
    wrapped = BaseExceptionGroup("tg", [BaseExceptionGroup("inner", [status_error(401)])])
    assert user_mcp._http_error_message(wrapped, cfg, "call") == (
        "srv rejected the call (401 Unauthorized): this server needs an API key, "
        "add it under Request headers as Authorization: Bearer <key>")
    keyed = {"name": "srv", "headers": {"Authorization": "Bearer abc"}}
    assert "was rejected" in user_mcp._http_error_message(status_error(401), keyed, "call")
    assert "403 Forbidden" in user_mcp._http_error_message(status_error(403), cfg, "connection")
    assert "server error (502 Bad Gateway)" in user_mcp._http_error_message(status_error(502), cfg, "call")
    assert user_mcp._http_error_message(ValueError("nope"), cfg, "call") is None


def test_unreachable_server_reports_error_and_is_skipped(pool, mcp_url):
    _save("calc", transport="streamable_http", url=mcp_url)
    _save("dead", transport="streamable_http", url="http://127.0.0.1:9/mcp")
    names = [t.name for t in pool.tools_for("alice")]
    assert names == ["calc__add", "calc__boom"]
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
