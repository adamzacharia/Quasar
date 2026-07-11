"""Conductor Dashboard API — execution metrics, traces, health, DAG cache.

S5: the ``/api/conductor/traces*`` routes now require authentication (they
expose per-query orchestration internals). ``/health`` and ``/dag-cache`` were
outside the S5 scope and are left unchanged.
"""

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_agent, get_current_user

router = APIRouter()


@router.get("/api/conductor/traces")
async def get_conductor_traces(current_user: dict = Depends(get_current_user)):
    """Get aggregated Conductor execution metrics and recent traces."""
    agent = get_agent()
    if not agent:
        return {"error": "Agent not loaded"}
    return agent.query_tracer.export_metrics()


@router.get("/api/conductor/traces/{trace_id}")
async def get_conductor_trace(trace_id: str, current_user: dict = Depends(get_current_user)):
    """Get detailed trace for a specific Conductor execution."""
    agent = get_agent()
    if not agent:
        return {"error": "Agent not loaded"}
    trace = agent.query_tracer.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace


@router.get("/api/conductor/health")
async def get_conductor_health():
    """Get Conductor system health: router, recovery, cache, pool."""
    agent = get_agent()
    if not agent:
        return {"error": "Agent not loaded"}
    health_data = {
        "conductor_model": getattr(agent.conductor, "conductor_model", "unknown"),
        "result_cache": agent.conductor.result_cache.get_stats() if hasattr(agent.conductor, "result_cache") else None,
        "dag_cache": agent.conductor.dag_cache.get_stats() if hasattr(agent.conductor, "dag_cache") else None,
        "workflow_memory": agent.conductor.workflow_memory.get_stats() if hasattr(agent.conductor, "workflow_memory") else None,
    }
    if hasattr(agent, "model_router") and agent.model_router:
        health_data["model_router"] = {
            "default_model": agent.model_router.default_model,
            "routing_table": {k: v["model"] for k, v in agent.model_router.ROUTING_TABLE.items()},
        }
    if hasattr(agent, "recovery_engine") and agent.recovery_engine:
        health_data["recovery_engine"] = agent.recovery_engine.get_stats()
    if hasattr(agent, "health_monitor") and agent.health_monitor:
        health_data["health_monitor"] = agent.health_monitor.get_status()
    return health_data


@router.get("/api/conductor/dag-cache")
async def get_dag_cache():
    """Get cached DAG decomposition patterns."""
    agent = get_agent()
    if not agent:
        return {"error": "Agent not loaded"}
    if hasattr(agent.conductor, "dag_cache"):
        return agent.conductor.dag_cache.get_stats()
    return {"entries": 0}
