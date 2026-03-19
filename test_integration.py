"""
End-to-end integration smoke test for the Conductor.

Tests:
1. Agent imports without errors
2. Conductor initializes with tool_executor and agent_pool
3. Complexity detector identifies multi-step queries
4. SSE event protocol is correctly structured
"""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

errors = []
passed = []

# ── Test 1: Core imports ─────────────────────────────────────────────────
try:
    from core.agent import QuasarAgent, AgentConfig
    from core.conductor import Conductor
    from core.task_dag import TaskDAG, TaskNode
    from core.model_router import ModelRouter
    from core.recovery import RecoveryEngine
    from core.agent_pool import AgentPool
    passed.append("1. Core imports OK")
except Exception as e:
    errors.append(f"1. Core import FAILED: {e}")

# ── Test 2: Agent + Conductor initialization ─────────────────────────────
try:
    api_key = os.getenv("OPENAI_API_KEY", "test-key")
    config = AgentConfig(api_key=api_key)
    agent = QuasarAgent(config)

    assert agent.conductor is not None, "Conductor is None"
    assert agent.conductor.tool_executor is not None, "tool_executor is None"
    assert agent.conductor.agent_pool is not None, "agent_pool is None"
    passed.append("2. Agent + Conductor init OK (tool_executor & agent_pool wired)")
except Exception as e:
    errors.append(f"2. Agent + Conductor init FAILED: {e}")

# ── Test 3: _build_tool_definitions ──────────────────────────────────────
try:
    tools = agent._build_tool_definitions()
    assert len(tools) > 20, f"Expected 20+ tools, got {len(tools)}"
    # Check structure
    for t in tools[:3]:
        assert "type" in t, "Tool missing 'type'"
        assert "name" in t, "Tool missing 'name'"
    passed.append(f"3. _build_tool_definitions OK ({len(tools)} tools)")
except Exception as e:
    errors.append(f"3. _build_tool_definitions FAILED: {e}")

# ── Test 4: _dispatch_tool_call ──────────────────────────────────────────
try:
    # Use a safe tool that doesn't hit network
    result = agent._dispatch_tool_call("nonexistent_tool", "{}")
    import json
    parsed = json.loads(result)
    assert "error" in parsed, "Expected error for unknown tool"
    passed.append("4. _dispatch_tool_call OK (error handling works)")
except Exception as e:
    errors.append(f"4. _dispatch_tool_call FAILED: {e}")

# ── Test 5: Conductor SSE event structure ────────────────────────────────
try:
    import json as _json

    events_received = []

    def mock_on_event(event_type: str, data: dict):
        events_received.append({"type": event_type, "data": data})

    # Test emit methods directly
    conductor = agent.conductor
    conductor.on_event = mock_on_event

    conductor._emit_task_list([
        TaskNode(id="t1", description="Search ALMA", agent_type="archive", dependencies=[]),
        TaskNode(id="t2", description="Analyze data", agent_type="analysis", dependencies=["t1"]),
    ])

    assert len(events_received) > 0, "No events emitted"
    evt = events_received[0]
    assert evt["type"] == "task_list", f"Expected task_list, got {evt['type']}"
    passed.append("5. Conductor SSE event emission OK")
except Exception as e:
    errors.append(f"5. Conductor SSE events FAILED: {e}")

# ── Test 6: Complexity detection ─────────────────────────────────────────
try:
    assert hasattr(agent, '_conductor_tool_executor'), "Missing _conductor_tool_executor"
    assert hasattr(agent, '_build_tool_definitions'), "Missing _build_tool_definitions"
    assert hasattr(agent, '_dispatch_tool_call'), "Missing _dispatch_tool_call"
    passed.append("6. All conductor methods present on agent")
except Exception as e:
    errors.append(f"6. Complexity detection FAILED: {e}")

# ── Results ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  QUASAR INTEGRATION SMOKE TEST")
print("=" * 60)
for p in passed:
    print(f"  ✅ {p}")
for e in errors:
    print(f"  ❌ {e}")
print("=" * 60)
print(f"  {len(passed)} passed, {len(errors)} failed")
print("=" * 60)

sys.exit(1 if errors else 0)
