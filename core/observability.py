# core/observability.py
"""
Observability — Structured trace IDs and execution metrics.

Every tool call, sub-agent execution, model routing decision, and
recovery attempt gets logged with a trace_id for debugging and
dashboard display.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TraceStep:
    """A single step in an execution trace."""
    step_name: str
    status: str             # "running", "completed", "failed"
    duration_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class QueryTracer:
    """
    Structured tracer for observability of multi-agent query execution.

    Usage
    -----
    tracer = QueryTracer()
    trace_id = tracer.new_trace("Search Sz65 data", user_id="user1")
    tracer.log_step(trace_id, "Decomposition", "completed", 150, {"subtasks": 4})
    tracer.log_step(trace_id, "ArchiveAgent", "completed", 2300, {"results": 15})
    trace = tracer.get_trace(trace_id)
    """

    def __init__(self):
        self._traces: Dict[str, Dict[str, Any]] = {}

    def new_trace(self, query: str, user_id: str = "anonymous") -> str:
        """Start a new trace, returns trace_id."""
        trace_id = str(uuid.uuid4())[:12]
        self._traces[trace_id] = {
            "trace_id": trace_id,
            "query": query,
            "user_id": user_id,
            "start_time": time.time(),
            "end_time": None,
            "steps": [],
            "status": "running",
        }
        logger.info("Trace started: %s (%s)", trace_id, query[:60])
        return trace_id

    def log_step(
        self,
        trace_id: str,
        step_name: str,
        status: str,
        duration_ms: float = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a step in an existing trace."""
        trace = self._traces.get(trace_id)
        if not trace:
            logger.warning("Unknown trace_id: %s", trace_id)
            return

        step = TraceStep(
            step_name=step_name,
            status=status,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        trace["steps"].append({
            "step_name": step.step_name,
            "status": step.status,
            "duration_ms": step.duration_ms,
            "metadata": step.metadata,
            "timestamp": step.timestamp,
        })

    def end_trace(self, trace_id: str, status: str = "completed") -> None:
        """Mark a trace as complete."""
        trace = self._traces.get(trace_id)
        if trace:
            trace["end_time"] = time.time()
            trace["status"] = status
            total_ms = (trace["end_time"] - trace["start_time"]) * 1000
            logger.info(
                "Trace %s %s in %.0fms (%d steps)",
                trace_id, status, total_ms, len(trace["steps"]),
            )

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get the full trace for a given trace_id."""
        return self._traces.get(trace_id)

    def export_metrics(self) -> Dict[str, Any]:
        """Export aggregated metrics for dashboard display."""
        if not self._traces:
            return {"total_traces": 0}

        total = len(self._traces)
        completed = sum(1 for t in self._traces.values() if t["status"] == "completed")
        failed = sum(1 for t in self._traces.values() if t["status"] == "failed")

        durations = []
        for t in self._traces.values():
            if t.get("end_time") and t.get("start_time"):
                durations.append((t["end_time"] - t["start_time"]) * 1000)

        return {
            "total_traces": total,
            "completed": completed,
            "failed": failed,
            "avg_duration_ms": round(sum(durations) / len(durations), 0) if durations else 0,
            "recent_traces": [
                {
                    "trace_id": t["trace_id"],
                    "query": t["query"][:60],
                    "status": t["status"],
                    "steps": len(t["steps"]),
                }
                for t in sorted(
                    self._traces.values(),
                    key=lambda x: x.get("start_time", 0),
                    reverse=True,
                )[:10]
            ],
        }

    def clear(self) -> None:
        """Clear all traces."""
        self._traces.clear()
