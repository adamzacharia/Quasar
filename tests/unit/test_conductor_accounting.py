"""R9 — Conductor cost-accounting propagation gate.

The historical CX-01 leak: the LLM accounting context (usage recorder, quota
checker, BYOK key routing) lives in a threading.local installed on the SSE
worker thread. The runner spawns a dedicated Conductor thread
(core/runner.py) and the Conductor fans subtasks into further executor
threads (core/conductor.py _execute_node) — none of which inherit a
threading.local. The fix chain is:

  sse worker (context installed)
    → runner captures get_llm_request_context() and wraps the conductor
      thread body in reinstall_llm_request_context (A2 CX-01)
      → _execute_node captures the reinstalled context and wraps every
        run_in_executor callable in _with_request_ctx

These tests gate that chain end-to-end with real Conductor code and stub
executors, plus the negative control documenting the original leak shape.
"""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from core.conductor import Conductor
from core.llm_client import (
    get_llm_request_context,
    llm_request_context,
    reinstall_llm_request_context,
)


class _Node:
    def __init__(self, agent_type="archive"):
        self.id = "t1"
        self.description = "count ALMA observations of M87"
        self.agent_type = agent_type
        self.depends_on = []
        self.sla_seconds = 30
        self.model_used = None


def _mini_run():
    memory = SimpleNamespace(get_dependency_context=lambda deps, max_chars=8000: "")
    dag = SimpleNamespace(nodes={})
    return SimpleNamespace(dag=dag, workflow_memory=memory)


def _make_conductor(tool_executor=None, sandbox_executor=None):
    return Conductor(
        client=None,
        model="gpt-4.1",
        tool_executor=tool_executor,
        sandbox_executor=sandbox_executor,
        verbose=False,
    )


def test_negative_control_spawned_thread_has_no_context_without_reinstall():
    """Documents the original leak: a bare thread sees no context."""
    seen = {}

    def worker():
        seen["ctx"] = get_llm_request_context()

    with llm_request_context(user_id="u1"):
        t = threading.Thread(target=worker)
        t.start()
        t.join()
    assert seen["ctx"] is None


def test_reinstalled_context_recorder_works_across_threads():
    recorded = []

    def recorder(**kwargs):
        recorded.append(kwargs)

    with llm_request_context(user_id="u1", usage_recorder=recorder):
        captured = get_llm_request_context()

        def worker():
            with reinstall_llm_request_context(captured):
                ctx = get_llm_request_context()
                assert ctx is captured
                ctx.usage_recorder(provider="openai", model="gpt-4.1",
                                   key_source="platform",
                                   input_tokens=10, output_tokens=5,
                                   reservation_id=None)

        t = threading.Thread(target=worker)
        t.start()
        t.join()

    assert len(recorded) == 1
    assert recorded[0]["input_tokens"] == 10 and recorded[0]["output_tokens"] == 5


def test_execute_node_propagates_context_into_tool_executor_thread():
    """The full two-hop chain: sse worker → conductor thread → executor thread.

    The stub tool_executor records the context and thread it runs on; the
    assertion is IDENTITY with the sse-side context object on a genuinely
    different thread.
    """
    seen = {}

    def tool_executor(task, dep_context, model, *extra):
        seen["ctx"] = get_llm_request_context()
        seen["thread"] = threading.get_ident()
        return "42 observations"

    conductor = _make_conductor(tool_executor=tool_executor)
    run = _mini_run()
    node = _Node()
    result_holder = {}

    with llm_request_context(user_id="u1", usage_recorder=lambda **kw: None):
        parent_ctx = get_llm_request_context()
        outer_thread = threading.get_ident()

        def conductor_thread_body():
            # Exactly the runner's A2 pattern (core/runner.py).
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                with reinstall_llm_request_context(parent_ctx):
                    result_holder["result"] = loop.run_until_complete(
                        conductor._execute_node(run, node)
                    )
            finally:
                loop.close()

        t = threading.Thread(target=conductor_thread_body)
        t.start()
        t.join(timeout=30)

    assert result_holder["result"] == "42 observations"
    assert seen["ctx"] is parent_ctx, (
        "tool_executor ran without the request accounting context — "
        "the CX-01 leak is back"
    )
    assert seen["thread"] != outer_thread, (
        "test invalid: the executor did not run on a separate thread"
    )


def test_execute_node_propagates_context_into_sandbox_thread():
    seen = {}

    class _Sandbox:
        def run(self, task, dep_context, user_id=None):
            seen["ctx"] = get_llm_request_context()
            return "sandbox done"

    conductor = _make_conductor(sandbox_executor=_Sandbox())
    run = _mini_run()
    node = _Node(agent_type="compute")

    with llm_request_context(user_id="u1"):
        parent_ctx = get_llm_request_context()

        def conductor_thread_body():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                with reinstall_llm_request_context(parent_ctx):
                    loop.run_until_complete(conductor._execute_node(run, node))
            finally:
                loop.close()

        t = threading.Thread(target=conductor_thread_body)
        t.start()
        t.join(timeout=30)

    assert seen["ctx"] is parent_ctx


def test_execute_node_with_user_id_still_propagates():
    seen = {}

    def tool_executor(task, dep_context, model, user_id=None):
        seen["ctx"] = get_llm_request_context()
        seen["user_id"] = user_id
        return "ok"

    conductor = _make_conductor(tool_executor=tool_executor)
    run = _mini_run()
    node = _Node()

    with llm_request_context(user_id="u2"):
        parent_ctx = get_llm_request_context()

        def body():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                with reinstall_llm_request_context(parent_ctx):
                    loop.run_until_complete(
                        conductor._execute_node(run, node, user_id="u2")
                    )
            finally:
                loop.close()

        t = threading.Thread(target=body)
        t.start()
        t.join(timeout=30)

    assert seen["ctx"] is parent_ctx
    assert seen["user_id"] == "u2"


def test_runner_source_carries_the_reinstall_pattern():
    """Cheap tripwire: the runner must keep capturing + reinstalling the
    context around the conductor thread (the exact site of the original
    leak, core/runner.py:948-region). A refactor that drops it should fail
    here even though the sibling tests construct the chain manually."""
    import inspect

    import core.runner as runner_mod

    src = inspect.getsource(runner_mod)
    assert "get_llm_request_context" in src
    assert "reinstall_llm_request_context" in src
    idx_capture = src.find("_parent_llm_ctx = get_llm_request_context()")
    idx_thread = src.find("def _run_conductor()")
    idx_reinstall = src.find("with reinstall_llm_request_context(_parent_llm_ctx)")
    assert -1 < idx_capture < idx_thread < idx_reinstall, (
        "runner no longer captures/reinstalls the LLM context around the "
        "conductor thread"
    )


# ─────────────────────────────────────────────────────────────────────────────
# R9 guard round (task-d3c8362-11851) — integration + isolation gates
# ─────────────────────────────────────────────────────────────────────────────
def _make_llm_shim(tokens=(11, 7)):
    """A REAL LLMClient responses shim with only the wire stubbed."""
    import types

    from core.llm_client import LLMClient

    client = LLMClient(model="gpt-4o-mini")
    shim = client.responses
    shim._call_openai = lambda kwargs, attachments=None: types.SimpleNamespace(
        usage=types.SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1]),
        output_text="ok",
    )
    return shim


def test_cx01_full_chain_real_llmclient_accounting_through_executor_thread():
    """The complete R9 chain with the REAL accounting client: sse-style
    context (quota checker + recorder + BYOK key source) → conductor thread
    (reinstall) → _execute_node → tool-executor thread (_with_request_ctx)
    → LLMClient.responses.create() with only the provider wire stubbed.
    Gates quota admission, BYOK key-source selection, usage recording, and
    reservation settlement across BOTH thread hops."""
    shim = _make_llm_shim()
    recorded, granted, released = [], [], []

    def tool_executor(task, dep_context, model, *extra):
        out = shim.create(model="gpt-4o-mini", input=task, stream=False)
        return out.output_text

    conductor = _make_conductor(tool_executor=tool_executor)
    run = _mini_run()
    node = _Node()
    holder = {}

    with llm_request_context(
        user_id="u-chain",
        provider_api_keys={"openai": "sk-byok-test"},
        key_source_by_provider={"openai": "user"},
        usage_recorder=lambda **kw: recorded.append(kw),
        quota_checker=lambda **kw: (granted.append(kw), f"res-{len(granted)}")[1],
        quota_releaser=lambda rid: released.append(rid),
    ):
        parent_ctx = get_llm_request_context()

        def conductor_thread_body():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                with reinstall_llm_request_context(parent_ctx):
                    holder["result"] = loop.run_until_complete(
                        conductor._execute_node(run, node)
                    )
            finally:
                loop.close()

        t = threading.Thread(target=conductor_thread_body)
        t.start()
        t.join(timeout=30)

    assert holder["result"] == "ok"
    # Quota admission happened INSIDE the executor thread.
    assert granted and granted[0]["provider"] == "openai"
    # BYOK routing: the key source resolved from the reinstalled context.
    assert granted[0]["key_source"] == "user"
    # Usage recorded with the reservation settled, not leaked.
    assert recorded and recorded[0]["input_tokens"] == 11
    assert recorded[0]["output_tokens"] == 7
    assert recorded[0]["key_source"] == "user"
    assert recorded[0]["reservation_id"] == "res-1"
    assert released == []


def test_cx03_two_concurrent_requests_do_not_cross_contaminate():
    """Two simultaneous request contexts (distinct users, recorders, key
    sources) each drive their own conductor-thread chain concurrently; every
    executor call must see ITS OWN context object and record into ITS OWN
    recorder with its own key source."""
    barrier = threading.Barrier(2, timeout=20)
    outcomes = {}

    def run_request(tag, key_source):
        shim = _make_llm_shim()
        recorded = []
        seen = {}

        def tool_executor(task, dep_context, model, *extra):
            seen["ctx"] = get_llm_request_context()
            barrier.wait()  # both executors in flight at the same time
            out = shim.create(model="gpt-4o-mini", input=task, stream=False)
            return f"done-{tag}"

        conductor = _make_conductor(tool_executor=tool_executor)
        run = _mini_run()
        node = _Node()

        with llm_request_context(
            user_id=f"u-{tag}",
            key_source_by_provider={"openai": key_source},
            usage_recorder=lambda **kw: recorded.append(kw),
        ):
            parent_ctx = get_llm_request_context()

            def body():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    with reinstall_llm_request_context(parent_ctx):
                        loop.run_until_complete(conductor._execute_node(run, node))
                finally:
                    loop.close()

            t = threading.Thread(target=body)
            t.start()
            t.join(timeout=30)

        outcomes[tag] = {"ctx": seen["ctx"], "parent": parent_ctx,
                         "recorded": recorded}

    ta = threading.Thread(target=run_request, args=("A", "user"))
    tb = threading.Thread(target=run_request, args=("B", "platform"))
    ta.start(); tb.start()
    ta.join(timeout=60); tb.join(timeout=60)

    a, b = outcomes["A"], outcomes["B"]
    # Each executor saw its own request's context object — never the other's.
    assert a["ctx"] is a["parent"] and b["ctx"] is b["parent"]
    assert a["ctx"] is not b["ctx"]
    assert a["ctx"].user_id == "u-A" and b["ctx"].user_id == "u-B"
    # Each recorder got exactly its own call, with its own key source.
    assert len(a["recorded"]) == 1 and len(b["recorded"]) == 1
    assert a["recorded"][0]["key_source"] == "user"
    assert b["recorded"][0]["key_source"] == "platform"


# ─────────────────────────────────────────────────────────────────────────────
# R9 verify reopen (CX-01) — the PRODUCTION conductor tool executor
# (QuasarAgent._conductor_tool_executor) driven through the full two-hop
# chain with the real LLMClient facade, stubbed only at the OpenAI SDK
# constructor — so BYOK key ROUTING (which api_key the SDK client is built
# with) is proven, not just accounting metadata.
# ─────────────────────────────────────────────────────────────────────────────
def _fake_openai_factory(captured):
    from types import SimpleNamespace

    class _FakeResponses:
        def create(self, **kwargs):
            captured.setdefault("create_calls", []).append(kwargs)
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=23, output_tokens=9),
                output_text="production executor ok",
                output=[],
                id="resp-fake-1",
            )

    class _FakeOpenAI:
        def __init__(self, *args, **kwargs):
            captured.setdefault("client_kwargs", []).append(kwargs)
            self.responses = _FakeResponses()

    return _FakeOpenAI


def _production_agent():
    """A partially-constructed QuasarAgent carrying exactly the attributes
    _conductor_tool_executor needs: the REAL recording LLMClient facade, a
    (empty) tool registry, per-request TLS, and a no-MCP config."""
    from types import SimpleNamespace

    from core.agent import QuasarAgent
    from core.llm_client import LLMClient
    from core.tools import ToolRegistry

    agent = QuasarAgent.__new__(QuasarAgent)
    agent._tls = threading.local()
    agent.tool_registry = ToolRegistry()
    agent.client = LLMClient(model="gpt-4o-mini")
    agent.config = SimpleNamespace(enable_mcp=False, mcp_server_url="", model="gpt-4o-mini")
    return agent


def test_cx01_reopen_production_executor_real_byok_routing(monkeypatch):
    import openai

    captured = {}
    monkeypatch.setattr(openai, "OpenAI", _fake_openai_factory(captured))
    # The platform env key must NOT be what reaches the SDK client.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-platform-env")
    monkeypatch.delenv("HELICONE_API_KEY", raising=False)

    agent = _production_agent()
    recorded, granted, released = [], [], []

    conductor = _make_conductor(tool_executor=agent._conductor_tool_executor)
    run = _mini_run()
    node = _Node()
    holder = {}

    with llm_request_context(
        user_id="u-prod",
        provider_api_keys={"openai": "sk-byok-context"},
        key_source_by_provider={"openai": "user"},
        usage_recorder=lambda **kw: recorded.append(kw),
        quota_checker=lambda **kw: (granted.append(kw), f"res-{len(granted)}")[1],
        quota_releaser=lambda rid: released.append(rid),
    ):
        parent_ctx = get_llm_request_context()

        def conductor_thread_body():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                with reinstall_llm_request_context(parent_ctx):
                    holder["result"] = loop.run_until_complete(
                        conductor._execute_node(run, node)
                    )
            finally:
                loop.close()

        t = threading.Thread(target=conductor_thread_body)
        t.start()
        t.join(timeout=60)

    # The PRODUCTION executor ran end-to-end and returned the model text.
    assert holder["result"] == "production executor ok"
    # REAL BYOK routing: the OpenAI SDK client was constructed with the
    # context's BYOK key (not the platform env key) INSIDE the executor
    # thread — this is _get_openai_client resolving the reinstalled context.
    assert captured["client_kwargs"], "the SDK client was never constructed"
    used_keys = [kw.get("api_key") for kw in captured["client_kwargs"]]
    assert used_keys == ["sk-byok-context"]
    # Quota admission + usage recording + reservation settlement all fired
    # through the real facade on the executor thread.
    assert granted and granted[0]["provider"] == "openai"
    assert granted[0]["key_source"] == "user"
    assert recorded and recorded[0]["input_tokens"] == 23
    assert recorded[0]["output_tokens"] == 9
    assert recorded[0]["reservation_id"] == "res-1"
    assert released == []


def test_cx01_reopen_production_executor_platform_fallback(monkeypatch):
    """Without a BYOK context key the SAME production path uses the platform
    env key — proving the routing decision really is context-driven."""
    import openai

    captured = {}
    monkeypatch.setattr(openai, "OpenAI", _fake_openai_factory(captured))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-platform-env")
    monkeypatch.delenv("HELICONE_API_KEY", raising=False)

    agent = _production_agent()
    recorded = []
    conductor = _make_conductor(tool_executor=agent._conductor_tool_executor)
    run = _mini_run()
    node = _Node()

    with llm_request_context(
        user_id="u-plat",
        usage_recorder=lambda **kw: recorded.append(kw),
    ):
        parent_ctx = get_llm_request_context()

        def body():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                with reinstall_llm_request_context(parent_ctx):
                    loop.run_until_complete(conductor._execute_node(run, node))
            finally:
                loop.close()

        t = threading.Thread(target=body)
        t.start()
        t.join(timeout=60)

    assert captured["client_kwargs"]
    assert captured["client_kwargs"][0].get("api_key") == "sk-platform-env"
    assert recorded and recorded[0]["key_source"] == "platform"
