import asyncio
from types import SimpleNamespace

from core.conductor import Conductor


def test_conductor_passes_user_id_into_worker_tool_executor():
    calls = []

    def tool_executor(task, context, model, user_id):
        calls.append((task, context, model, user_id))
        return "done"

    conductor = Conductor.__new__(Conductor)
    conductor.model_router = None
    conductor.conductor_model = "test-model"
    conductor.verbose = False
    conductor.sandbox_executor = None
    conductor.tool_executor = tool_executor

    # Post-S42 the per-request state lives on the OrchestrationRun threaded
    # through _execute_node, not on the Conductor singleton.
    run = SimpleNamespace(
        workflow_memory=SimpleNamespace(
            get_dependency_context=lambda *args, **kwargs: ""
        ),
        dag=SimpleNamespace(nodes={}),
    )

    node = SimpleNamespace(
        id="task-1",
        agent_type="archive",
        description="find data",
        depends_on=[],
        model_used=None,
    )
    result = asyncio.run(conductor._execute_node(run, node, user_id="alice"))

    assert result == "done"
    assert calls == [("find data", "", "test-model", "alice")]
