# Conductor Documentation Rule

Whenever you make changes to any file related to the Conductor orchestration system, you MUST also update `docs/CONDUCTOR_ARCHITECTURE.md` to reflect those changes.

**Conductor-related files include:**
- `core/conductor.py`
- `core/task_dag.py`
- `core/workflow_memory.py`
- `core/recovery.py`
- `core/model_router.py`
- `core/observability.py`
- `core/result_cache.py`
- `core/dag_cache.py`
- `core/agent_pool.py`
- `core/health_monitor.py`
- `services/notebook_gen.py`
- Any new files added to the Conductor pipeline

**What to update in the doc:**
- Diagrams (mermaid) if the flow changed
- File map table if files were added/removed
- Relevant section text if behavior changed (e.g. model used, new parameters, new SSE events)
