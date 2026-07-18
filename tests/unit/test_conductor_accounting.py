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
