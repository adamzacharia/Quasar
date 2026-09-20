import asyncio
import inspect
import threading

import core.agent as agent_module
from core.agent import QuasarAgent, _build_mcp_tool_wrapper


class _FakeResult:
    content = []
    isError = False


class _FakeSession:
    def __init__(self):
        self.calls = []
        self.loops = []

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments))
        self.loops.append(asyncio.get_running_loop())
        return _FakeResult()


class _SlowSession:
    def __init__(self):
        self.cancelled = threading.Event()

    async def call_tool(self, name, arguments=None):
        try:
            await asyncio.Future()
        finally:
            self.cancelled.set()


def test_wrapper_reuses_one_loop_across_calls(monkeypatch):
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run_loop():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    t = threading.Thread(target=_run_loop, daemon=True)
    t.start()
    try:
        assert ready.wait(timeout=5)
        sess = _FakeSession()
        wrapper = _build_mcp_tool_wrapper(sess, "toolx", loop, "srv")
        r1 = wrapper(a=1)
        r2 = wrapper(a=2)
        # empty content + no structuredContent normalizes to success/data=None
        assert r1["success"] is True and r1["data"] is None
        assert r2["success"] is True and r2["data"] is None
        assert sess.calls == [("toolx", {"a": 1}), ("toolx", {"a": 2})]
        assert len(sess.loops) == 2
        assert sess.loops[0] is loop and sess.loops[1] is loop

        monkeypatch.setattr(agent_module, "_MCP_TOOL_CALL_TIMEOUT_SECONDS", 0.01)
        slow_sess = _SlowSession()
        timeout_wrapper = _build_mcp_tool_wrapper(slow_sess, "slow_tool", loop)
        timeout_result = timeout_wrapper()
        assert timeout_result["success"] is False
        assert timeout_result["error"] == "MCP tool call timed out after 0.01 seconds"
        assert slow_sess.cancelled.wait(timeout=5)

        bridge_source = inspect.getsource(QuasarAgent._mount_mcp_bridges)
        assert bridge_source.count("asyncio.new_event_loop()") == 1
        assert "_build_mcp_tool_wrapper(" in bridge_source
        assert "loop.run_forever()" in bridge_source
        assert "run_until_complete(_do_call())" not in bridge_source
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=5)
        assert not t.is_alive()
        loop.close()


def test_bridge_setup_failure_stops_thread_closes_stack_and_loop():
    """A failed bridge task must not leave its run_forever thread alive."""
    from contextlib import AsyncExitStack

    loop = asyncio.new_event_loop()
    setup_started = threading.Event()
    stack_closed = threading.Event()
    outcomes = []

    class _TrackedContext:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            stack_closed.set()

    async def _failing_setup():
        stack = AsyncExitStack()
        try:
            await stack.enter_async_context(_TrackedContext())
            setup_started.set()
            raise RuntimeError("bridge setup failed")
        finally:
            await stack.aclose()

    def _thread_main():
        asyncio.set_event_loop(loop)
        task = loop.create_task(_failing_setup())

        def _stop_loop(_):
            loop.stop()

        task.add_done_callback(_stop_loop)
        try:
            loop.run_forever()
        finally:
            task.remove_done_callback(_stop_loop)
            if not task.done():
                task.cancel()
            outcomes.extend(
                loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
            )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
            loop.close()

    t = threading.Thread(target=_thread_main, daemon=True)
    t.start()

    assert setup_started.wait(timeout=5)
    t.join(timeout=5)

    assert not t.is_alive()
    assert stack_closed.is_set()
    assert loop.is_closed()
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], RuntimeError)
    assert str(outcomes[0]) == "bridge setup failed"
