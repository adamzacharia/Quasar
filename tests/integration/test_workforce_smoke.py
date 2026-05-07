"""Smoke test for Phase 0-4 multi-agent workforce modules.

Uses importlib to import each module directly, bypassing core/__init__.py
which triggers the full QuasarAgent → pyvo import chain (broken on Py3.13).
"""
import sys
import os
import importlib.util

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def import_module_directly(module_path, module_name):
    """Import a single module from a file path, bypassing package __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

# Import modules directly from file paths
task_dag = import_module_directly(
    os.path.join(PROJECT_ROOT, "core", "task_dag.py"), "core.task_dag"
)
workflow_memory = import_module_directly(
    os.path.join(PROJECT_ROOT, "core", "workflow_memory.py"), "core.workflow_memory"
)
model_router = import_module_directly(
    os.path.join(PROJECT_ROOT, "core", "model_router.py"), "core.model_router"
)
recovery = import_module_directly(
    os.path.join(PROJECT_ROOT, "core", "recovery.py"), "core.recovery"
)
observability = import_module_directly(
    os.path.join(PROJECT_ROOT, "core", "observability.py"), "core.observability"
)
agent_pool = import_module_directly(
    os.path.join(PROJECT_ROOT, "core", "agent_pool.py"), "core.agent_pool"
)
workflow_store = import_module_directly(
    os.path.join(PROJECT_ROOT, "core", "workflow_store.py"), "core.workflow_store"
)
datalink = import_module_directly(
    os.path.join(PROJECT_ROOT, "integrations", "datalink.py"), "integrations.datalink"
)

# Classes
TaskDAG = task_dag.TaskDAG
TaskStatus = task_dag.TaskStatus
WorkflowMemory = workflow_memory.WorkflowMemory
ModelRouter = model_router.ModelRouter
RecoveryEngine = recovery.RecoveryEngine
QueryTracer = observability.QueryTracer
AgentPool = agent_pool.AgentPool
WorkflowStore = workflow_store.WorkflowStore
DataLinkClient = datalink.DataLinkClient

# ─── Test WorkflowMemory ─────────────────────────────────────────────
wm = WorkflowMemory()
wm.write("test_agent", "key1", {"data": 42})
assert wm.read("key1") == {"data": 42}
assert len(wm) == 1
assert "key1" in wm
assert wm.get_agent_contributions("test_agent") == {"key1": {"data": 42}}
summary = wm.get_context_summary()
assert "key1" in summary
wm.clear()
assert len(wm) == 0
print("[PASS] WorkflowMemory")

# ─── Test TaskDAG ───────────────────────────────────────────────────
dag = TaskDAG()
dag.build_from_subtasks([
    {"id": "t1", "description": "Search ALMA for Sz65", "depends_on": [], "agent_type": "archive"},
    {"id": "t2", "description": "List files for found UIDs", "depends_on": ["t1"], "agent_type": "archive"},
    {"id": "t3", "description": "Inspect FITS headers", "depends_on": ["t2"], "agent_type": "analysis"},
    {"id": "t4", "description": "Build comparison table", "depends_on": ["t3"], "agent_type": "synthesis"},
])
assert len(dag.nodes) == 4

ready = dag.get_ready_tasks()
assert len(ready) == 1 and ready[0].id == "t1"

dag.mark_running("t1")
assert dag.nodes["t1"].status == TaskStatus.RUNNING
dag.mark_completed("t1", {"results": 15})
assert dag.nodes["t1"].status == TaskStatus.COMPLETED

ready2 = dag.get_ready_tasks()
assert len(ready2) == 1 and ready2[0].id == "t2"

dag.mark_completed("t2", {"files": ["a.fits", "b.fits"]})
ready3 = dag.get_ready_tasks()
assert len(ready3) == 1 and ready3[0].id == "t3"

serialized = dag.to_dict()
assert len(serialized) == 4

summary = dag.get_execution_summary()
assert summary["completed"] == 2
assert summary["total_tasks"] == 4

# Test parallel readiness
dag2 = TaskDAG()
dag2.build_from_subtasks([
    {"id": "a", "description": "Search archive 1", "depends_on": [], "agent_type": "archive"},
    {"id": "b", "description": "Search archive 2", "depends_on": [], "agent_type": "archive"},
    {"id": "c", "description": "Search papers", "depends_on": [], "agent_type": "literature"},
    {"id": "d", "description": "Synthesize", "depends_on": ["a", "b", "c"], "agent_type": "synthesis"},
])
parallel_ready = dag2.get_ready_tasks()
assert len(parallel_ready) == 3, f"Expected 3 parallel tasks, got {len(parallel_ready)}"
print("[PASS] TaskDAG (including parallel readiness)")

# ─── Test ModelRouter ──────────────────────────────────────────────
mr = ModelRouter()
assert mr.route("Search ALMA for Sz65", "archive") == "gpt-4o"
assert mr.route("Find recent papers on AGN jets", "literature") == "gemini-2.0-pro"
assert mr.route("What is the speed of light?", "general") == "gpt-4o-mini"
assert mr.classify_task_type("Search ALMA archive for target M87") == "archive_search"
assert mr.classify_task_type("Recent papers on black hole imaging") == "literature_review"
assert mr.classify_task_type("Plot the sky distribution") == "visualization"
info = mr.get_routing_info("Search for AGN observations", "archive")
assert info["model"] == "gpt-4o"
print("[PASS] ModelRouter")

# ─── Test QueryTracer ──────────────────────────────────────────────
tracer = QueryTracer()
tid = tracer.new_trace("Test query", user_id="user1")
tracer.log_step(tid, "decompose", "completed", 150, {"subtasks": 4})
tracer.log_step(tid, "archive_agent", "completed", 2300, {"results": 15})
tracer.end_trace(tid, "completed")
trace = tracer.get_trace(tid)
assert trace["status"] == "completed"
assert len(trace["steps"]) == 2
metrics = tracer.export_metrics()
assert metrics["total_traces"] == 1
assert metrics["completed"] == 1
print("[PASS] QueryTracer")

# ─── Test AgentPool ────────────────────────────────────────────────
pool = AgentPool()
agent = pool.get_agent("test", factory_fn=lambda: "mock_agent")
assert agent == "mock_agent"
agent2 = pool.get_agent("test")
assert agent2 == "mock_agent"
pool.cleanup_idle()
assert pool.get_agent("test") is not None
pool.clear()
assert pool.get_agent("test") is None
print("[PASS] AgentPool")

# ─── Test WorkflowStore ───────────────────────────────────────────
import tempfile
store_dir = os.path.join(tempfile.gettempdir(), "quasar_test_workflows")
store = WorkflowStore(store_dir=store_dir)
qid = store.generate_query_id()
store.save_checkpoint(qid, dag.to_dict(), {"t1": {"results": 15}}, query="Test query")
loaded = store.load_checkpoint(qid)
assert loaded is not None
assert loaded["query"] == "Test query"
assert loaded["status"] == "active"
store.pause(qid)
assert store.load_checkpoint(qid)["status"] == "paused"
store.resume(qid)
assert store.load_checkpoint(qid)["status"] == "active"
snap_id = store.snapshot(qid)
assert snap_id is not None
store.delete(qid)
store.delete(snap_id)
assert store.load_checkpoint(qid) is None
print("[PASS] WorkflowStore")

# ─── Test DataLinkClient ──────────────────────────────────────────
dl = DataLinkClient()
assert hasattr(dl, 'list_files')
assert hasattr(dl, 'get_file_access_url')
# The actual on-disk version uses list_files_batch instead of list_files_for_project
has_batch = hasattr(dl, 'list_files_batch') or hasattr(dl, 'list_files_for_project')
assert has_batch, "Expected either list_files_batch or list_files_for_project"
print("[PASS] DataLinkClient")

# ─── Test RecoveryEngine ──────────────────────────────────────────
re = RecoveryEngine(max_retries=3, verbose=True)
assert re._is_empty_result(None) == True
assert re._is_empty_result("") == True
assert re._is_empty_result({"total_results": 0}) == True
assert re._is_empty_result({"total_results": 5}) == False
assert re._is_empty_result("some text") == False
assert re._is_empty_result([]) == True
assert re._is_empty_result([1, 2]) == False
print("[PASS] RecoveryEngine")

print("")
print("=" * 50)
print("  ALL PHASE 0-4 SMOKE TESTS PASSED")
print("=" * 50)
