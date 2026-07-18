"""Runtime unit tests for the post-S42 Conductor OrchestrationRun interface
(T4 correctness-debt / C12).

S42 moves per-request orchestration state off the shared Conductor singleton onto
a per-call ``OrchestrationRun`` dataclass threaded through the five orchestration
methods. S42's own verification is static greps + a boot smoke; this file exercises
the same interface AT RUNTIME so slips a grep cannot see (e.g. ``run._lf_trace``
instead of ``run.lf_trace`` — an AttributeError only when the line runs) fail loudly.
Written against the post-S42 target state (precedent: S44 inlines a full test for
code that does not exist yet).

S42 amendments reflected here (see LEDGER 2026-07-13 S42 rows): the recovery
hand-off is an async ``_execute_scoped`` closure (not ``functools.partial``)
because the SX batch already threads ``user_id`` through it, and
``_execute_dag_with_events``/``_execute_node`` accept ``user_id`` — so the fake
executors accept ``user_id=None``.

Offline: no network, no real LLM. The OpenAI client, the node executor, the
synthesizer, and the DAG cache are faked/monkeypatched. Async methods are driven
with ``asyncio.run(...)`` (no pytest-asyncio locally — CONVENTIONS section 2).
"""

import asyncio
import dataclasses
import inspect

from core.conductor import Conductor, OrchestrationRun
from core.task_dag import TaskDAG, TaskStatus
from core.workflow_memory import WorkflowMemory


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
class _FakeClient:
    """Placeholder OpenAI client — never called in these offline tests."""


def _conductor():
    """A real Conductor, built offline. __init__ only stores config and loads the
    on-disk DAG cache (no network/LLM), so full construction is safe — no __new__
    skeleton needed (contrast test_viz_capability.py::_wiring_agent, which must
    skeleton QuasarAgent). Post-S51 __init__ has no agent_pool arg; we pass only
    client, so this is robust either way."""
    return Conductor(client=_FakeClient())


def _one_node_dag(node_id="t1", description="A", agent_type="general"):
    dag = TaskDAG()
    dag.build_from_subtasks(
        [{"id": node_id, "description": description,
          "agent_type": agent_type, "depends_on": []}]
    )
    return dag


class _FakeRecovery:
    """Mimics RecoveryEngine.execute_with_recovery's calling convention
    (core/recovery.py:109-114): if the executor is a coroutine function, call it
    with the node; else fall back to the legacy sync (description, "") path.
    Post-S42 (as amended) the conductor hands us the async ``_execute_scoped``
    closure, which asyncio.iscoroutinefunction detects directly, so the
    coroutine branch must be taken."""

    def __init__(self):
        self.saw_coroutine_fn = None

    async def execute_with_recovery(self, node, executor_fn,
                                    on_status=None, dep_context=None):
        self.saw_coroutine_fn = asyncio.iscoroutinefunction(executor_fn)
        if self.saw_coroutine_fn:
            return await executor_fn(node)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, executor_fn, node.description, "")


# ---------------------------------------------------------------------------
# (a) OrchestrationRun dataclass field contract — the underscore-slip catcher
# ---------------------------------------------------------------------------
def test_orchestration_run_field_contract():
    names = [f.name for f in dataclasses.fields(OrchestrationRun)]
    assert names == ["dag", "workflow_memory", "lf_trace", "notebook"]

    run = OrchestrationRun(dag=TaskDAG(), workflow_memory=WorkflowMemory())
    assert run.lf_trace is None            # default, no underscore
    assert run.notebook is None            # default, no underscore
    # The singleton attrs were self._lf_trace / self._notebook; S42 drops the
    # leading underscore. If a slip kept it, run.lf_trace would not exist.
    assert not hasattr(run, "_lf_trace")
    assert not hasattr(run, "_notebook")


# ---------------------------------------------------------------------------
# (b) __init__ drops request-scoped state but keeps the cross-run cache
# ---------------------------------------------------------------------------
def test_conductor_init_drops_request_state_keeps_cache():
    c = _conductor()
    assert not hasattr(c, "dag")              # was self.dag = TaskDAG()
    assert not hasattr(c, "workflow_memory")  # was self.workflow_memory = WorkflowMemory()
    assert not hasattr(c, "_lf_trace")
    assert not hasattr(c, "_notebook")
    assert hasattr(c, "dag_cache")            # legit cross-run cache — KEPT


# ---------------------------------------------------------------------------
# (c) the orphan Conductor.get_execution_summary is deleted
# ---------------------------------------------------------------------------
def test_conductor_get_execution_summary_removed():
    assert not hasattr(Conductor, "get_execution_summary")
    # the DAG keeps its own; the agent now reads conductor_run.dag.<this>().
    assert hasattr(TaskDAG, "get_execution_summary")


# ---------------------------------------------------------------------------
# (d) the four run-threaded methods take `run` as the first arg after self
# ---------------------------------------------------------------------------
def test_orchestration_methods_take_run_first():
    for name in ("_emit_task_list", "_execute_dag_with_events",
                 "_execute_node", "_synthesize"):
        params = list(inspect.signature(getattr(Conductor, name)).parameters)
        assert params[0] == "self", name
        assert params[1] == "run", name


# ---------------------------------------------------------------------------
# (e) a real OrchestrationRun threaded through _execute_dag_with_events
# ---------------------------------------------------------------------------
def test_execute_dag_with_events_threads_run():
    conductor = _conductor()
    conductor.recovery = None                 # take the direct _execute_node branch

    seen = []

    async def fake_execute_node(run, node, user_id=None):
        seen.append((run, node.id))
        return {"ok": node.id}

    conductor._execute_node = fake_execute_node

    dag = TaskDAG()
    dag.build_from_subtasks([
        {"id": "t1", "description": "A", "agent_type": "general", "depends_on": []},
        {"id": "t2", "description": "B", "agent_type": "general", "depends_on": []},
    ])
    run = OrchestrationRun(dag=dag, workflow_memory=WorkflowMemory())

    results = asyncio.run(conductor._execute_dag_with_events(run))

    assert results == {"t1": {"ok": "t1"}, "t2": {"ok": "t2"}}
    assert all(passed is run for passed, _ in seen)     # SAME run object each call
    # state landed on run.*, not conductor.*
    assert not hasattr(conductor, "dag")
    assert not hasattr(conductor, "workflow_memory")
    assert len(run.workflow_memory) == 2                # _run_one wrote both results
    assert run.dag.nodes["t1"].status == TaskStatus.COMPLETED
    assert run.dag.nodes["t2"].status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# (e cont.) recovery hand-off keeps async-closure coroutine detection
# ---------------------------------------------------------------------------
def test_recovery_handoff_uses_coroutine_closure():
    conductor = _conductor()
    rec = _FakeRecovery()
    conductor.recovery = rec

    captured = {}

    async def fake_execute_node(run, node, user_id=None):
        captured["run"] = run
        captured["node_id"] = node.id
        return {"ok": True}

    conductor._execute_node = fake_execute_node
    run = OrchestrationRun(dag=_one_node_dag(), workflow_memory=WorkflowMemory())

    results = asyncio.run(conductor._execute_dag_with_events(run))

    # The async _execute_scoped closure is detected as a coroutine function
    # (a sync lambda would not be — it would wrongly take recovery's sync branch).
    assert rec.saw_coroutine_fn is True
    assert captured["run"] is run           # run threaded through the closure
    assert captured["node_id"] == "t1"
    assert results == {"t1": {"ok": True}}


# ---------------------------------------------------------------------------
# (f) two sequential runs on the SAME singleton keep independent state (C12)
# ---------------------------------------------------------------------------
def test_two_sequential_runs_independent_state():
    conductor = _conductor()
    conductor.recovery = None

    async def fake_execute_node(run, node, user_id=None):
        return {"tag": node.description}

    conductor._execute_node = fake_execute_node

    def _run(tag):
        r = OrchestrationRun(dag=_one_node_dag(description=tag),
                             workflow_memory=WorkflowMemory())
        return r, asyncio.run(conductor._execute_dag_with_events(r))

    run_a, results_a = _run("ALPHA")
    run_b, results_b = _run("BETA")

    assert results_a == {"t1": {"tag": "ALPHA"}}
    assert results_b == {"t1": {"tag": "BETA"}}
    assert run_a.dag is not run_b.dag
    assert run_a.workflow_memory is not run_b.workflow_memory
    assert run_a.dag.nodes["t1"].result == {"tag": "ALPHA"}
    assert run_b.dag.nodes["t1"].result == {"tag": "BETA"}
    # the singleton never accreted per-request state between runs (C12 fix)
    assert not hasattr(conductor, "dag")
    assert not hasattr(conductor, "workflow_memory")


# ---------------------------------------------------------------------------
# orchestrate() return contract: (None, None) early exit and (answer, run)
# ---------------------------------------------------------------------------
def test_orchestrate_early_exit_returns_none_none():
    conductor = _conductor()
    conductor.dag_cache.find_similar = lambda q: None
    conductor._decompose = lambda query, context, max_subtasks=None: []

    result = asyncio.run(conductor.orchestrate("too simple"))
    assert result == (None, None)


def test_orchestrate_returns_answer_and_run_tuple():
    conductor = _conductor()
    conductor.recovery = None
    conductor.dag_cache.find_similar = lambda q: None
    conductor.dag_cache.store = lambda *a, **k: None
    conductor._decompose = lambda query, context, max_subtasks=None: [
        {"id": "t1", "description": "do X", "agent_type": "general", "depends_on": []}
    ]

    async def fake_execute_node(run, node, user_id=None):
        return {"ok": node.id}

    conductor._execute_node = fake_execute_node
    conductor._synthesize = lambda run, query, results, on_token=None: "FINAL ANSWER"

    answer, run = asyncio.run(conductor.orchestrate("Compare several things in depth"))

    assert answer == "FINAL ANSWER"
    assert isinstance(run, OrchestrationRun)
    # the agent's post-run consumer path reads run.dag.get_execution_summary()
    summary = run.dag.get_execution_summary()
    assert summary["total_tasks"] == 1
    assert summary["completed"] == 1
    assert not hasattr(conductor, "dag")
    assert not hasattr(conductor, "workflow_memory")


def test_parse_plan_json_is_tolerant_of_provider_output_shapes():
    """L4 (live 2026-07-18): gpt-oss/TACC does not honor json_object — the
    plan can arrive fenced, prose-wrapped, or empty (reasoning ate the token
    budget). Raw json.loads killed EVERY Conductor activation; the tolerant
    parser must recover the object or return {} without raising."""
    from core.conductor import Conductor

    parse = Conductor._parse_plan_json
    plan = '{"subtasks": [{"id": "t1"}], "reasoning": "r"}'

    assert parse(plan)["subtasks"] == [{"id": "t1"}]
    assert parse(f"```json\n{plan}\n```")["subtasks"] == [{"id": "t1"}]
    assert parse(f"```\n{plan}\n```")["subtasks"] == [{"id": "t1"}]
    assert parse(f"Here is the decomposition:\n{plan}\nDone.")["subtasks"] == [{"id": "t1"}]
    assert parse("") == {}
    assert parse(None) == {}
    assert parse("The model rambled with no JSON at all.") == {}
    assert parse('["a", "list", "not", "object"]') == {}
