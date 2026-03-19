# core/task_dag.py
"""
Task DAG — Dependency-aware task graph with parallel execution.

Replaces the sequential subtask loop in rlm.py with a DAG that automatically
runs independent tasks in parallel via asyncio.gather, while respecting
data dependencies between tasks.

Key insight: if a query requires searching 7 sources, all 7 searches can run
in parallel instead of sequentially — 7× speedup with the same tools.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskNode:
    """Single node in the task execution graph."""

    id: str
    description: str
    depends_on: List[str] = field(default_factory=list)
    agent_type: str = "general"        # archive, literature, analysis, viz, web, synthesis
    model_hint: Optional[str] = None   # optional model preference override
    sla_seconds: int = 120             # max time allowed per task
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retries: int = 0

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return round(self.end_time - self.start_time, 2)
        return None


class TaskDAG:
    """
    Dependency-aware task graph with parallel execution.

    Usage
    -----
    dag = TaskDAG()
    dag.build_from_subtasks([
        {"id": "t1", "description": "Search ALMA for Sz65", "depends_on": [], "agent_type": "archive"},
        {"id": "t2", "description": "List files for MOUS UIDs", "depends_on": ["t1"], "agent_type": "archive"},
        {"id": "t3", "description": "Read FITS headers", "depends_on": ["t2"], "agent_type": "analysis"},
        {"id": "t4", "description": "Build comparison table", "depends_on": ["t3"], "agent_type": "synthesis"},
    ])
    results = await dag.execute(my_executor_fn)
    """

    def __init__(self):
        self.nodes: Dict[str, TaskNode] = {}
        self._adjacency: Dict[str, List[str]] = {}  # id → ids that depend on it

    def build_from_subtasks(self, subtasks: List[dict]) -> "TaskDAG":
        """
        Build the DAG from a list of structured subtask dicts.

        Each dict should have:
          - id: str
          - description: str
          - depends_on: List[str]       (optional, defaults to [])
          - agent_type: str             (optional, defaults to "general")
          - model_hint: str             (optional)
          - sla_seconds: int            (optional, defaults to 120)
        """
        self.nodes.clear()
        self._adjacency.clear()

        for st in subtasks:
            task_id = st.get("id") or f"t{len(self.nodes) + 1}"
            node = TaskNode(
                id=task_id,
                description=st.get("description", ""),
                depends_on=st.get("depends_on", []),
                agent_type=st.get("agent_type", "general"),
                model_hint=st.get("model_hint"),
                sla_seconds=st.get("sla_seconds", 120),
            )
            self.nodes[task_id] = node

            # Build reverse adjacency (who depends on me)
            for dep_id in node.depends_on:
                self._adjacency.setdefault(dep_id, []).append(task_id)

        # Validate: all dependencies must exist
        for node in self.nodes.values():
            for dep_id in node.depends_on:
                if dep_id not in self.nodes:
                    logger.warning(
                        "Task %s depends on unknown task %s — removing dependency",
                        node.id, dep_id,
                    )
                    node.depends_on.remove(dep_id)

        return self

    def get_ready_tasks(self) -> List[TaskNode]:
        """
        Return all tasks that are ready to execute:
        - Status is PENDING
        - All dependencies are COMPLETED
        """
        ready = []
        for node in self.nodes.values():
            if node.status != TaskStatus.PENDING:
                continue
            # Check all dependencies are completed
            deps_met = all(
                self.nodes[dep_id].status == TaskStatus.COMPLETED
                for dep_id in node.depends_on
                if dep_id in self.nodes
            )
            if deps_met:
                ready.append(node)
        return ready

    def mark_completed(self, task_id: str, result: Any) -> None:
        """Mark a task as completed with its result."""
        node = self.nodes.get(task_id)
        if node:
            node.status = TaskStatus.COMPLETED
            node.result = result
            node.end_time = time.time()
            logger.info("Task %s completed in %.1fs", task_id, node.duration or 0)

    def mark_failed(self, task_id: str, error: str) -> None:
        """Mark a task as failed with an error message."""
        node = self.nodes.get(task_id)
        if node:
            node.status = TaskStatus.FAILED
            node.error = error
            node.end_time = time.time()
            logger.warning("Task %s failed: %s", task_id, error)

    def mark_running(self, task_id: str) -> None:
        """Mark a task as running."""
        node = self.nodes.get(task_id)
        if node:
            node.status = TaskStatus.RUNNING
            node.start_time = time.time()

    async def execute(
        self,
        executor_fn: Callable[["TaskNode"], Coroutine[Any, Any, Any]],
        on_status: Optional[Callable[[str, str], None]] = None,
        max_rounds: int = 20,
    ) -> Dict[str, Any]:
        """
        Execute the full DAG, running independent tasks in parallel.

        Parameters
        ----------
        executor_fn : async callable
            Called with each TaskNode, should return the result.
        on_status : callable, optional
            Status callback: (step_label, status) for UI updates.
        max_rounds : int
            Safety limit to prevent infinite loops.

        Returns
        -------
        dict
            Mapping of task_id → result for all completed tasks.
        """
        total = len(self.nodes)
        completed = 0

        for _round in range(max_rounds):
            ready = self.get_ready_tasks()
            if not ready:
                # Check if we're fully done or stuck
                pending = [n for n in self.nodes.values() if n.status == TaskStatus.PENDING]
                if not pending:
                    break  # All done
                else:
                    # Stuck — remaining tasks have failed dependencies
                    logger.warning(
                        "DAG stuck: %d pending tasks with unmet dependencies",
                        len(pending),
                    )
                    for p in pending:
                        self.mark_failed(p.id, "Unmet dependencies — predecessor task failed")
                    break

            # Run all ready tasks in parallel
            if on_status:
                task_names = ", ".join(t.description[:40] for t in ready)
                on_status(f"Running {len(ready)} parallel task(s): {task_names}", "running")

            async def _run_one(node: TaskNode) -> None:
                self.mark_running(node.id)
                try:
                    result = await asyncio.wait_for(
                        executor_fn(node),
                        timeout=node.sla_seconds,
                    )
                    self.mark_completed(node.id, result)
                except asyncio.TimeoutError:
                    self.mark_failed(node.id, f"Timeout after {node.sla_seconds}s")
                except Exception as e:
                    self.mark_failed(node.id, str(e))

            await asyncio.gather(*[_run_one(node) for node in ready])

            completed = sum(1 for n in self.nodes.values() if n.status == TaskStatus.COMPLETED)
            if on_status:
                on_status(f"Completed {completed}/{total} tasks", "completed")

        # Collect results
        return {
            tid: node.result
            for tid, node in self.nodes.items()
            if node.status == TaskStatus.COMPLETED
        }

    def get_execution_summary(self) -> Dict[str, Any]:
        """Return a human-readable summary of the DAG execution."""
        return {
            "total_tasks": len(self.nodes),
            "completed": sum(1 for n in self.nodes.values() if n.status == TaskStatus.COMPLETED),
            "failed": sum(1 for n in self.nodes.values() if n.status == TaskStatus.FAILED),
            "tasks": [
                {
                    "id": n.id,
                    "description": n.description,
                    "status": n.status.value,
                    "agent_type": n.agent_type,
                    "duration_sec": n.duration,
                    "error": n.error,
                }
                for n in self.nodes.values()
            ],
        }

    def to_dict(self) -> List[dict]:
        """Serialize the DAG nodes for storage/inspection."""
        return [
            {
                "id": n.id,
                "description": n.description,
                "depends_on": n.depends_on,
                "agent_type": n.agent_type,
                "model_hint": n.model_hint,
                "sla_seconds": n.sla_seconds,
                "status": n.status.value,
            }
            for n in self.nodes.values()
        ]
