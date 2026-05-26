# core/observability.py
"""
Observability — Structured trace IDs, execution metrics, and cost tracking.

Every tool call, sub-agent execution, model routing decision, and
recovery attempt gets logged with a trace_id for debugging and
dashboard display.

Includes (#6) per-step token counting and cost estimation.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Cost estimation per model (#6) ────────────────────────────────────────
# Prices in USD per 1K tokens (approximate, as of early 2026).
# Update these when pricing changes.
MODEL_PRICING = {
    # model_name: (input_per_1k, output_per_1k)
    "gpt-5.4":       (0.010, 0.030),
    "gpt-5.4-mini":  (0.00075, 0.0045),
    "gpt-4.1":       (0.002, 0.008),
    "gpt-4.1-mini":  (0.0004, 0.0016),
    "gpt-4.1-nano":  (0.0001, 0.0004),
    "gpt-4o":        (0.0025, 0.010),
    "gpt-4o-mini":   (0.00015, 0.0006),
    # DeepSeek
    "deepseek-v4-pro":   (0.000435, 0.00087),
    "deepseek-v4-flash": (0.00014, 0.00028),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate the cost of an API call in USD.

    Falls back to gpt-5.4-mini pricing if the model is unknown.
    """
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        # Try prefix match (e.g. "gpt-5.4-mini-2026-04-14" → "gpt-5.4-mini")
        for known_model, p in MODEL_PRICING.items():
            if model.startswith(known_model):
                pricing = p
                break
    if not pricing:
        pricing = MODEL_PRICING.get("gpt-5.4-mini", (0.00075, 0.0045))

    input_cost = (input_tokens / 1000) * pricing[0]
    output_cost = (output_tokens / 1000) * pricing[1]
    return round(input_cost + output_cost, 6)


@dataclass
class TraceStep:
    """A single step in an execution trace."""
    step_name: str
    status: str             # "running", "completed", "failed"
    duration_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    # (#6) Cost & token tracking
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


class QueryTracer:
    """
    Structured tracer for observability of multi-agent query execution.

    Tracks per-step timing, token usage, cost estimates, and model routing
    decisions across the entire Conductor DAG lifecycle.

    Usage
    -----
    tracer = QueryTracer()
    trace_id = tracer.new_trace("Search Sz65 data", user_id="user1")
    tracer.log_step(trace_id, "Decomposition", "completed", 150, {"subtasks": 4})
    tracer.log_step(trace_id, "t1_archive", "completed", 2300, {
        "model": "gpt-4.1", "input_tokens": 1200, "output_tokens": 400,
    })
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
            # (#6) Aggregate cost tracking
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
            "models_used": [],
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
        """
        Log a step in an existing trace.

        If metadata contains 'model', 'input_tokens', 'output_tokens',
        they will be extracted for cost tracking.
        """
        trace = self._traces.get(trace_id)
        if not trace:
            logger.warning("Unknown trace_id: %s", trace_id)
            return

        meta = metadata or {}

        # (#6) Extract token/cost info from metadata
        model = meta.get("model", "")
        input_tokens = meta.get("input_tokens", 0)
        output_tokens = meta.get("output_tokens", 0)
        cost = 0.0
        if model and (input_tokens or output_tokens):
            cost = estimate_cost(model, input_tokens, output_tokens)

        step = TraceStep(
            step_name=step_name,
            status=status,
            duration_ms=duration_ms,
            metadata=meta,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
        )
        trace["steps"].append({
            "step_name": step.step_name,
            "status": step.status,
            "duration_ms": step.duration_ms,
            "metadata": step.metadata,
            "timestamp": step.timestamp,
            "model": step.model,
            "input_tokens": step.input_tokens,
            "output_tokens": step.output_tokens,
            "estimated_cost_usd": step.estimated_cost_usd,
        })

        # (#6) Update aggregate totals
        trace["total_input_tokens"] += input_tokens
        trace["total_output_tokens"] += output_tokens
        trace["total_cost_usd"] += cost
        if model and model not in trace["models_used"]:
            trace["models_used"].append(model)

    def end_trace(self, trace_id: str, status: str = "completed") -> None:
        """Mark a trace as complete."""
        trace = self._traces.get(trace_id)
        if trace:
            trace["end_time"] = time.time()
            trace["status"] = status
            total_ms = (trace["end_time"] - trace["start_time"]) * 1000
            logger.info(
                "Trace %s %s in %.0fms (%d steps, $%.4f, %d tokens in, %d tokens out)",
                trace_id, status, total_ms, len(trace["steps"]),
                trace["total_cost_usd"],
                trace["total_input_tokens"],
                trace["total_output_tokens"],
            )

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get the full trace for a given trace_id."""
        return self._traces.get(trace_id)

    def get_recent_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get the most recent traces (for dashboard API)."""
        return sorted(
            self._traces.values(),
            key=lambda x: x.get("start_time", 0),
            reverse=True,
        )[:limit]

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

        total_cost = sum(t.get("total_cost_usd", 0) for t in self._traces.values())
        total_input = sum(t.get("total_input_tokens", 0) for t in self._traces.values())
        total_output = sum(t.get("total_output_tokens", 0) for t in self._traces.values())

        return {
            "total_traces": total,
            "completed": completed,
            "failed": failed,
            "avg_duration_ms": round(sum(durations) / len(durations), 0) if durations else 0,
            "total_cost_usd": round(total_cost, 4),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "recent_traces": [
                {
                    "trace_id": t["trace_id"],
                    "query": t["query"][:60],
                    "status": t["status"],
                    "steps": len(t["steps"]),
                    "cost_usd": round(t.get("total_cost_usd", 0), 4),
                    "models": t.get("models_used", []),
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
