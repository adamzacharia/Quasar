"""The benchmark-only tool allowlist (core/bench_toolset.py): a no-op when
unset; when set, the model sees and can run only matching tools."""

from __future__ import annotations

import threading
import types

import pytest

from core import bench_toolset
from core.tools import Tool, ToolRegistry


@pytest.fixture
def allow_manna(monkeypatch):
    monkeypatch.setenv(bench_toolset.ENV, "manna__*")


def test_unset_is_a_no_op(monkeypatch):
    monkeypatch.delenv(bench_toolset.ENV, raising=False)
    assert not bench_toolset.active()
    assert bench_toolset.allowed("datalab_sql_query")
    assert bench_toolset.filter_names(["a", "b"]) == ["a", "b"]


def test_whitespace_only_value_is_unset(monkeypatch):
    monkeypatch.setenv(bench_toolset.ENV, "   ")
    assert not bench_toolset.active()


def test_malformed_value_fails_closed(monkeypatch):
    """ " , " is an empty allowlist: nothing is exposed (guard CX-05)."""
    monkeypatch.setenv(bench_toolset.ENV, " , ")
    assert bench_toolset.active()
    assert not bench_toolset.allowed("datalab_sql_query")
    assert not bench_toolset.allowed("manna__run_adql_query")


def test_patterns_match_case_sensitively(allow_manna, monkeypatch):
    assert bench_toolset.allowed("manna__run_adql_query")
    assert not bench_toolset.allowed("vo_adql_query")
    assert not bench_toolset.allowed("MANNA__run_adql_query")
    monkeypatch.setenv(bench_toolset.ENV, "manna__*, resolve_target")
    assert bench_toolset.filter_names(["manna__x", "resolve_target", "search_mast"]) == [
        "manna__x", "resolve_target"]


def _agent_with_tools(*names):
    from tests.integration.test_agent_archive_tools import _load_agent_module

    module = _load_agent_module()
    agent = module.QuasarAgent.__new__(module.QuasarAgent)
    agent._tls = threading.local()
    agent.tool_registry = ToolRegistry()
    agent.config = types.SimpleNamespace(enable_mcp=True, mcp_server_url="https://mcp.example/alma",
                                         user_id="")
    for name in names:
        agent.tool_registry.register(Tool(
            name=name, description=name, function=lambda **kw: {"success": True, "ran": True},
            parameters={"type": "object", "properties": {}, "required": []},
            category="mcp" if name.startswith("manna__") else "archive"))
    return agent


def test_llm_tool_list_is_filtered_and_hosted_mcp_dropped(allow_manna):
    agent = _agent_with_tools("vo_adql_query", "manna__run_adql_query", "search_mast")
    tools = agent._build_tools_for_responses_api()
    assert [t.get("name") for t in tools] == ["manna__run_adql_query"]
    assert not any(t.get("type") == "mcp" for t in tools)


def test_llm_tool_list_unchanged_when_unset(monkeypatch):
    monkeypatch.delenv(bench_toolset.ENV, raising=False)
    agent = _agent_with_tools("vo_adql_query", "manna__run_adql_query")
    tools = agent._build_tools_for_responses_api()
    assert {t.get("name") for t in tools if t.get("type") == "function"} == {
        "vo_adql_query", "manna__run_adql_query"}
    assert any(t.get("type") == "mcp" for t in tools)


def test_no_match_allowlist_yields_empty_tool_list(monkeypatch):
    monkeypatch.setenv(bench_toolset.ENV, "nothing_matches_*")
    agent = _agent_with_tools("vo_adql_query")
    assert not agent._build_tools_for_responses_api()  # None/[]: the runner coerces to []


def test_sandbox_unknown_tool_lists_only_allowed_names(allow_manna):
    agent = _agent_with_tools("vo_adql_query", "manna__run_adql_query")
    out = agent._sandbox_tool_bridge("vo_adql_query", {})
    assert "manna__run_adql_query" in out["error"] and "'vo_adql_query'" not in out["error"].split("Available:")[1]


def test_guarded_worker_thread_inherits_the_turn_user(monkeypatch):
    """CX-01: the budgeted path runs the tool on a fresh thread; it must see
    the submitting user, or per-user job ownership collapses to ''."""
    monkeypatch.delenv(bench_toolset.ENV, raising=False)
    agent = _agent_with_tools()
    seen = {}

    def fn(**kw):
        seen["user"] = getattr(agent._tls, "current_user_id", None)
        return {"success": True}

    agent.tool_registry.register(Tool(name="probe_user", description="d", function=fn,
                                      parameters={"type": "object", "properties": {}}, category="archive"))
    agent._tls.current_user_id = "alice"
    for attr in ("_accumulated_run_results", "_accumulated_tool_trace"):
        if not hasattr(type(agent), attr):
            setattr(agent, attr, [])
    agent._execute_tool_guarded(agent.tool_registry.get_tool("probe_user"), {},
                                tool_name="probe_user", timeout_seconds=5)
    assert seen["user"] == "alice"


def test_guarded_execution_refuses_tools_outside_the_list(allow_manna):
    agent = _agent_with_tools("vo_adql_query")
    tool = agent.tool_registry.get_tool("vo_adql_query")
    out = agent._execute_tool_guarded(tool, {}, tool_name="vo_adql_query")
    assert out["success"] is False and "not available" in out["error"]
