"""Platform-level MCP mount (MANNA) + MCP result normalization.

Covers:
- services.mcp_server_service.platform_mcp_servers env parsing + validation
- adapters.mcp normalization (success, failure, provenance, content shapes)
- QuasarAgent._load_platform_mcp_servers wiring (stub self, no network)
- per-call MCP timeout env override
- stdio RCE gate: applies to per-user configs (any non-URL transport), not
  to platform configs
- full bridge lifecycle over a fake streamable-HTTP transport (mount →
  register → call → shutdown teardown) and setup-failure retry/give-up
- the loud missing-SDK diagnostic
"""

import asyncio
import os
import threading
import time
import types

import pytest

import core.agent as agent_module
from capabilities.base import PROVENANCE_SIDECAR_KEY
from core.agent import QuasarAgent, _build_mcp_tool_wrapper
from core.tools import ToolRegistry
from adapters.mcp import normalize_mcp_failure, normalize_mcp_result
from services.mcp_server_service import (
    MCPServerConfig,
    MCPServerService,
    manna_enabled,
    platform_mcp_servers,
    validate_server_config,
)


_MCP_ENV_KEYS = (
    "QUASAR_ENABLE_MANNA",
    "MANNA_MCP_URL",
    "MANNA_MCP_COMMAND",
    "QUASAR_PLATFORM_MCP_SERVERS",
    "QUASAR_MCP_TOOL_TIMEOUT_SECONDS",
    "QUASAR_ENABLE_MCP_STDIO",
)


@pytest.fixture(autouse=True)
def _clean_mcp_env(monkeypatch):
    for key in _MCP_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


# ── platform_mcp_servers ─────────────────────────────────────────────────


def test_platform_servers_default_empty():
    assert platform_mcp_servers() == []
    assert manna_enabled() is False


def test_manna_stdio_default_pinned(monkeypatch):
    monkeypatch.setenv("QUASAR_ENABLE_MANNA", "1")
    servers = platform_mcp_servers()
    assert len(servers) == 1
    cfg = servers[0]
    assert cfg["name"] == "manna"
    assert cfg["transport"] == "stdio"
    assert cfg["command"] == "uvx"
    # pinned to the verified release, not a floating latest
    assert cfg["args"] == ["--from", "manna-mcp==0.7.0", "manna", "--stdio"]


def test_manna_url_selects_streamable_http(monkeypatch):
    monkeypatch.setenv("QUASAR_ENABLE_MANNA", "true")
    monkeypatch.setenv("MANNA_MCP_URL", "http://mannahost:8000/mcp")
    servers = platform_mcp_servers()
    assert len(servers) == 1
    assert servers[0]["transport"] == "streamable_http"
    assert servers[0]["url"] == "http://mannahost:8000/mcp"


def test_manna_command_override(monkeypatch):
    monkeypatch.setenv("QUASAR_ENABLE_MANNA", "yes")
    monkeypatch.setenv("MANNA_MCP_COMMAND", "python -m manna --stdio")
    cfg = platform_mcp_servers()[0]
    assert cfg["command"] == "python"
    assert cfg["args"] == ["-m", "manna", "--stdio"]


def test_manna_windows_path_command_survives_split(monkeypatch):
    monkeypatch.setenv("QUASAR_ENABLE_MANNA", "1")
    monkeypatch.setenv(
        "MANNA_MCP_COMMAND", r"C:\tools\uv\uvx.exe --from manna-mcp==0.7.0 manna --stdio"
    )
    import os as _os
    cfg = platform_mcp_servers()[0]
    if _os.name == "nt":
        assert cfg["command"] == r"C:\tools\uv\uvx.exe"
    else:  # POSIX split still yields a usable head token
        assert cfg["command"]


def test_manna_bad_command_does_not_suppress_other_servers(monkeypatch, capsys):
    monkeypatch.setenv("QUASAR_ENABLE_MANNA", "1")
    monkeypatch.setenv("MANNA_MCP_COMMAND", 'uvx "unclosed quote')
    monkeypatch.setenv(
        "QUASAR_PLATFORM_MCP_SERVERS",
        '[{"name": "jwst", "transport": "streamable_http", "url": "http://x/mcp"}]',
    )
    servers = platform_mcp_servers()
    assert [s["name"] for s in servers] == ["jwst"]
    assert "Skipping MANNA config" in capsys.readouterr().out


def test_platform_servers_json_list(monkeypatch):
    monkeypatch.setenv(
        "QUASAR_PLATFORM_MCP_SERVERS",
        '[{"name": "jwst", "transport": "streamable_http", "url": "http://x/mcp"}]',
    )
    servers = platform_mcp_servers()
    assert [s["name"] for s in servers] == ["jwst"]


def test_platform_servers_bad_json_ignored(monkeypatch):
    monkeypatch.setenv("QUASAR_ENABLE_MANNA", "1")
    monkeypatch.setenv("QUASAR_PLATFORM_MCP_SERVERS", "{not json")
    servers = platform_mcp_servers()
    # malformed extra config never blocks the MANNA mount
    assert [s["name"] for s in servers] == ["manna"]


def test_platform_invalid_middle_entry_keeps_valid_ones(monkeypatch, capsys):
    monkeypatch.setenv(
        "QUASAR_PLATFORM_MCP_SERVERS",
        '[{"name": "a", "transport": "streamable_http", "url": "http://a/mcp"},'
        ' {"name": "bad", "transport": "http"},'
        ' {"name": "c", "transport": "stdio", "command": "srv"}]',
    )
    servers = platform_mcp_servers()
    assert [s["name"] for s in servers] == ["a", "c"]
    assert "Skipping QUASAR_PLATFORM_MCP_SERVERS" in capsys.readouterr().out


def test_platform_unknown_transport_skipped(monkeypatch):
    monkeypatch.setenv(
        "QUASAR_PLATFORM_MCP_SERVERS",
        '[{"name": "x", "transport": "websocket", "url": "http://x"}]',
    )
    assert platform_mcp_servers() == []


def test_platform_duplicate_names_first_wins(monkeypatch, capsys):
    monkeypatch.setenv("QUASAR_ENABLE_MANNA", "1")
    monkeypatch.setenv(
        "QUASAR_PLATFORM_MCP_SERVERS",
        '[{"name": "manna", "transport": "streamable_http", "url": "http://other/mcp"}]',
    )
    servers = platform_mcp_servers()
    assert len(servers) == 1
    assert servers[0]["transport"] == "stdio"  # the built-in MANNA entry won
    assert "duplicate" in capsys.readouterr().out


# ── validate_server_config / save_server ─────────────────────────────────


def test_validate_rejects_unknown_transport():
    with pytest.raises(ValueError, match="Unknown MCP transport"):
        validate_server_config(
            MCPServerConfig(name="x", transport="carrier-pigeon", url="http://x")
        )


def test_validate_requires_url_for_streamable_http():
    with pytest.raises(ValueError, match="URL is required"):
        validate_server_config(MCPServerConfig(name="x", transport="streamable_http"))


def test_save_server_unknown_transport_rejected(tmp_path):
    svc = MCPServerService(base_dir=str(tmp_path))
    with pytest.raises(ValueError, match="Unknown MCP transport"):
        svc.save_server("u1", MCPServerConfig(name="x", transport="Sneaky", command="cmd"))


def test_validate_normalizes_whitespace_fields():
    """' manna ' / ' uvx ' must come out canonical, or the tool namespace and
    the spawned executable are corrupted downstream (CX-25)."""
    cfg = validate_server_config(
        MCPServerConfig(name=" manna ", transport=" STDIO ", command=" uvx ")
    )
    assert cfg.name == "manna"
    assert cfg.transport == "stdio"
    assert cfg.command == "uvx"
    cfg2 = validate_server_config(
        MCPServerConfig(name="m", transport="streamable_http", url=" http://x/mcp ")
    )
    assert cfg2.url == "http://x/mcp"


def test_save_server_cased_stdio_still_gated(tmp_path):
    """'Stdio' must not slip past the RCE gate via case difference (CX-01)."""
    svc = MCPServerService(base_dir=str(tmp_path))
    with pytest.raises(ValueError, match="disabled"):
        svc.save_server("u1", MCPServerConfig(name="x", transport="Stdio", command="cmd"))


# ── normalize_mcp_result / normalize_mcp_failure ─────────────────────────


def _result(content=None, structured=None, is_error=False):
    blocks = []
    for item in content or []:
        if isinstance(item, tuple):  # (type, text)
            blocks.append(types.SimpleNamespace(type=item[0], text=item[1]))
        else:
            blocks.append(types.SimpleNamespace(type="text", text=item))
    return types.SimpleNamespace(
        content=blocks, structuredContent=structured, isError=is_error
    )


def test_normalize_error_result_keeps_provenance():
    args = {"adql": "SELECT 1", "endpoint": "https://t/tap"}
    out = normalize_mcp_result("manna", "vo_tap_query", _result(["boom"], is_error=True), args)
    assert out["success"] is False
    assert out["error"] == "boom"
    prov = out[PROVENANCE_SIDECAR_KEY]["provenance"]
    assert prov["query"] == "SELECT 1"
    assert prov["endpoint"] == "https://t/tap"


def test_normalize_failure_helper():
    out = normalize_mcp_failure("manna", "t", "timed out", {"endpoint": "https://e"})
    assert out["success"] is False and out["error"] == "timed out"
    assert out[PROVENANCE_SIDECAR_KEY]["provenance"]["endpoint"] == "https://e"


def test_normalize_structured_content_wrapped_with_provenance():
    res = _result(structured={"archives": [1, 2]})
    args = {"adql": "SELECT TOP 5 * FROM ivoa.obscore", "endpoint": "https://t/tap"}
    out = normalize_mcp_result("manna", "vo_tap_query", res, args)
    assert out["success"] is True
    assert out["data"] == {"archives": [1, 2]}
    prov = out[PROVENANCE_SIDECAR_KEY]["provenance"]
    assert prov["service"] == "manna"
    assert prov["tool"] == "vo_tap_query"
    assert prov["query"] == "SELECT TOP 5 * FROM ivoa.obscore"
    assert prov["endpoint"] == "https://t/tap"


def test_normalize_empty_structured_content_is_data():
    out = normalize_mcp_result("manna", "t", _result(["ignored"], structured={}))
    assert out["success"] is True
    assert out["data"] == {}


def test_normalize_native_shaped_structured_content_passes_through():
    res = _result(structured={"success": False, "error": "no rows"})
    out = normalize_mcp_result("manna", "t", res)
    assert out["success"] is False
    assert out["error"] == "no rows"


def test_normalize_single_json_text_block():
    out = normalize_mcp_result("manna", "t", _result(['{"ra": 187.7}']))
    assert out["success"] is True
    assert out["data"] == {"ra": 187.7}


def test_normalize_json_text_native_failure_passes_through():
    out = normalize_mcp_result("manna", "t", _result(['{"success": false, "error": "bad"}']))
    assert out["success"] is False
    assert out["error"] == "bad"


def test_normalize_plain_text_and_multi_block_joined():
    out = normalize_mcp_result("manna", "t", _result(["hello"]))
    assert out["data"] == "hello"
    out2 = normalize_mcp_result("manna", "t", _result(["a", "b"]))
    assert out2["data"] == "a\nb"


def test_normalize_non_text_only_content_is_typed_note():
    out = normalize_mcp_result("manna", "t", _result([("image", None)]))
    assert out["success"] is True
    assert out["data"]["content_types"] == ["image"]
    assert "non-text" in out["data"]["note"]


def test_normalize_never_raises_and_keeps_provenance():
    class _Evil:
        @property
        def content(self):
            raise RuntimeError("bad server")

    out = normalize_mcp_result("manna", "t", _Evil(), {"endpoint": "https://e"})
    assert out["success"] is False
    assert "normalization failed" in out["error"]
    # even the last-resort failure path is auditable (CX-23)
    prov = out[PROVENANCE_SIDECAR_KEY]["provenance"]
    assert prov["service"] == "manna" and prov["endpoint"] == "https://e"


# ── _load_platform_mcp_servers wiring ────────────────────────────────────


class _StubAgent:
    def __init__(self):
        self.mounted = []

    def _mount_mcp_bridges(self, configs, enforce_stdio_gate):
        self.mounted.append((configs, enforce_stdio_gate))


def test_platform_load_disabled_by_default():
    stub = _StubAgent()
    QuasarAgent._load_platform_mcp_servers(stub)
    assert stub.mounted == []


def test_platform_load_mounts_manna_without_stdio_gate(monkeypatch):
    monkeypatch.setenv("QUASAR_ENABLE_MANNA", "on")
    stub = _StubAgent()
    QuasarAgent._load_platform_mcp_servers(stub)
    assert len(stub.mounted) == 1
    configs, enforce = stub.mounted[0]
    assert enforce is False
    assert configs[0]["name"] == "manna"


def test_platform_load_called_in_init():
    import inspect

    src = inspect.getsource(QuasarAgent.__init__)
    assert "_load_platform_mcp_servers()" in src


def test_agent_init_mounts_platform_manna_behaviorally(monkeypatch):
    """End-to-end init wiring: a real QuasarAgent constructed with the MANNA
    flag on (and a fake MCP transport) must expose the bridged tool — this
    is the observable contract behind the source-inspection test above."""
    _install_fake_mcp(monkeypatch)
    # offline pins, mirroring scripts/gen_tool_registry_doc.py
    monkeypatch.setenv("QDRANT_URL", "")
    monkeypatch.setenv("QDRANT_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY") or "sk-test-dummy")
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "gpt-4.1")
    monkeypatch.setenv("QUASAR_ENABLE_MANNA", "1")
    monkeypatch.setenv("MANNA_MCP_URL", "http://fake/mcp")

    from core.agent import AgentConfig

    agent = QuasarAgent(AgentConfig())
    try:
        assert _wait_for(lambda: agent.tool_registry.has("manna__ping"))
        out = agent.tool_registry.get_tool("manna__ping").execute()
        assert out["success"] is True and out["data"] == {"pong": True}
    finally:
        agent.shutdown_mcp_servers()


# ── per-call timeout env override ────────────────────────────────────────


def _run_loop_in_thread():
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    assert ready.wait(timeout=5)
    return loop, t


def test_mcp_timeout_env_override(monkeypatch):
    monkeypatch.setenv("QUASAR_MCP_TOOL_TIMEOUT_SECONDS", "0.01")

    class _SlowSession:
        async def call_tool(self, name, arguments=None):
            await asyncio.Future()

    loop, t = _run_loop_in_thread()
    try:
        wrapper = _build_mcp_tool_wrapper(_SlowSession(), "slow", loop, "srv")
        out = wrapper()
        assert out["success"] is False
        assert out["error"] == "MCP tool call timed out after 0.01 seconds"
        assert out[PROVENANCE_SIDECAR_KEY]["provenance"]["service"] == "srv"
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=5)
        loop.close()


def test_mcp_timeout_env_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("QUASAR_MCP_TOOL_TIMEOUT_SECONDS", "not-a-number")
    assert agent_module._mcp_call_timeout_seconds() == (
        agent_module._MCP_TOOL_CALL_TIMEOUT_SECONDS
    )
    monkeypatch.setenv("QUASAR_MCP_TOOL_TIMEOUT_SECONDS", "-5")
    assert agent_module._mcp_call_timeout_seconds() == (
        agent_module._MCP_TOOL_CALL_TIMEOUT_SECONDS
    )


def test_wrapper_dead_loop_is_normalized_failure():
    loop = asyncio.new_event_loop()
    loop.close()
    wrapper = _build_mcp_tool_wrapper(object(), "t", loop, "srv")
    out = wrapper()
    assert out["success"] is False
    assert "not running" in out["error"]
    assert out[PROVENANCE_SIDECAR_KEY]["provenance"]["tool"] == "t"


# ── fake MCP transport plumbing (no real SDK server, no network) ─────────


class _FakeCM:
    """Async context manager handing out a fixed value; records exit."""

    exits = []  # class-level: every __aexit__ appends

    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, tb):
        _FakeCM.exits.append(self)
        return False


class _FakeBridgeSession:
    def __init__(self, read, write):
        self._entered = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        _FakeCM.exits.append(self)
        return False

    async def initialize(self):
        return None

    async def list_tools(self):
        tool = types.SimpleNamespace(
            name="ping",
            description="fake ping",
            inputSchema={"type": "object", "properties": {}},
        )
        return types.SimpleNamespace(tools=[tool])

    async def call_tool(self, name, arguments=None):
        return types.SimpleNamespace(
            content=[], structuredContent={"pong": True}, isError=False
        )


def _install_fake_mcp(monkeypatch, streamable_factory=None):
    """Shadow the mcp.client.* modules the bridge imports with fakes."""
    import sys

    def _streamable(url):
        return _FakeCM((object(), object(), lambda: "sid"))

    stdio_mod = types.ModuleType("mcp.client.stdio")
    stdio_mod.stdio_client = lambda params: _FakeCM((object(), object()))
    stdio_mod.StdioServerParameters = lambda **kw: types.SimpleNamespace(**kw)
    sse_mod = types.ModuleType("mcp.client.sse")
    sse_mod.sse_client = lambda url: _FakeCM((object(), object()))
    session_mod = types.ModuleType("mcp.client.session")
    session_mod.ClientSession = _FakeBridgeSession
    http_mod = types.ModuleType("mcp.client.streamable_http")
    http_mod.streamablehttp_client = streamable_factory or _streamable

    monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.sse", sse_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.session", session_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", http_mod)


def _bridge_stub():
    stub = types.SimpleNamespace()
    stub.tool_registry = ToolRegistry()
    return stub


def _wait_for(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_streamable_http_bridge_full_lifecycle(monkeypatch):
    """Mount over the (fake) streamable-HTTP transport: 3-tuple unpack,
    registration, a live call through the wrapper, and exit-stack teardown."""
    _FakeCM.exits = []
    _install_fake_mcp(monkeypatch)
    stub = _bridge_stub()
    cfg = {"name": "fake", "transport": "streamable_http", "url": "http://x/mcp"}

    QuasarAgent._mount_mcp_bridges(stub, [cfg], enforce_stdio_gate=False)
    assert _wait_for(lambda: stub.tool_registry.has("fake__ping"))

    tool = stub.tool_registry.get_tool("fake__ping")
    assert tool.category == "mcp"
    out = tool.execute()
    assert out["success"] is True
    assert out["data"] == {"pong": True}
    assert out[PROVENANCE_SIDECAR_KEY]["provenance"]["service"] == "fake"

    QuasarAgent.shutdown_mcp_servers(stub)
    # session + transport context managers must both be exited
    assert _wait_for(lambda: len(_FakeCM.exits) >= 2)


def test_bridge_setup_failure_retries_then_gives_up(monkeypatch, capsys):
    _FakeCM.exits = []

    calls = {"n": 0}

    def _always_fail(url):
        calls["n"] += 1
        raise ConnectionError("refused")

    _install_fake_mcp(monkeypatch, streamable_factory=_always_fail)
    monkeypatch.setattr(agent_module, "_MCP_BRIDGE_RETRY_BACKOFF_SECONDS", 0.01)
    stub = _bridge_stub()
    cfg = {"name": "down", "transport": "streamable_http", "url": "http://x/mcp"}

    QuasarAgent._mount_mcp_bridges(stub, [cfg], enforce_stdio_gate=False)
    assert _wait_for(lambda: calls["n"] >= agent_module._MCP_BRIDGE_MAX_ATTEMPTS)
    assert calls["n"] == agent_module._MCP_BRIDGE_MAX_ATTEMPTS
    # the shutdown event was registered even though setup never succeeded,
    # so shutdown is a no-op rather than an error
    assert len(stub._mcp_shutdown_events) == 1
    QuasarAgent.shutdown_mcp_servers(stub)


def test_shutdown_cancels_hung_connect(monkeypatch):
    """A transport that never finishes connecting must not outlive
    shutdown_mcp_servers(): the setup task is cancelled and the bridge
    thread exits, closing its loop (CX-08)."""
    connect_started = threading.Event()

    class _HangCM:
        async def __aenter__(self):
            connect_started.set()
            await asyncio.Future()  # hangs until cancelled

        async def __aexit__(self, exc_type, exc, tb):
            return False

    _install_fake_mcp(monkeypatch, streamable_factory=lambda url: _HangCM())
    stub = _bridge_stub()
    cfg = {"name": "hung", "transport": "streamable_http", "url": "http://x/mcp"}

    QuasarAgent._mount_mcp_bridges(stub, [cfg], enforce_stdio_gate=False)
    # the shutdown event is registered synchronously, before the thread runs
    assert len(stub._mcp_shutdown_events) == 1
    assert connect_started.wait(timeout=5)

    loop = stub._mcp_loops[0]
    QuasarAgent.shutdown_mcp_servers(stub)
    assert _wait_for(lambda: loop.is_closed())
    assert not stub.tool_registry.has("hung__ping")


def test_missing_sdk_is_loud_not_silent(monkeypatch, capsys):
    import sys

    # Simulate a missing SDK: None in sys.modules makes the import raise.
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", None)
    stub = _bridge_stub()
    cfg = {"name": "x", "transport": "streamable_http", "url": "http://x/mcp"}
    QuasarAgent._mount_mcp_bridges(stub, [cfg], enforce_stdio_gate=False)
    out = capsys.readouterr().out
    assert "'mcp' SDK is not installed" in out
    assert not getattr(stub, "_mcp_loops", [])


# ── stdio gate semantics in _mount_mcp_bridges ───────────────────────────


def test_user_stdio_config_gated_off_by_default(monkeypatch):
    _install_fake_mcp(monkeypatch)
    stub = _bridge_stub()
    cfg = {"name": "x", "transport": "stdio", "command": "definitely-not-a-cmd"}
    QuasarAgent._mount_mcp_bridges(stub, [cfg], enforce_stdio_gate=True)
    # gate refused the spawn: no bridge loop/thread was created
    assert not getattr(stub, "_mcp_loops", [])


def test_user_unknown_transport_gated_like_stdio(monkeypatch):
    """A non-URL transport string reaches the stdio spawn path, so the gate
    must cover it (CX-01)."""
    _install_fake_mcp(monkeypatch)
    stub = _bridge_stub()
    cfg = {"name": "x", "transport": "mystery", "command": "definitely-not-a-cmd"}
    QuasarAgent._mount_mcp_bridges(stub, [cfg], enforce_stdio_gate=True)
    assert not getattr(stub, "_mcp_loops", [])


def test_platform_stdio_config_bypasses_gate(monkeypatch):
    _install_fake_mcp(monkeypatch)
    stub = _bridge_stub()
    cfg = {"name": "plat", "transport": "stdio", "command": "fake-cmd"}
    QuasarAgent._mount_mcp_bridges(stub, [cfg], enforce_stdio_gate=False)
    # bridge started and (with the fake transport) registered its tool
    assert len(getattr(stub, "_mcp_loops", [])) == 1
    assert _wait_for(lambda: stub.tool_registry.has("plat__ping"))
    QuasarAgent.shutdown_mcp_servers(stub)
