# core/task_dag.py
"""
Task DAG — Dependency-aware task graph with parallel execution.

Replaces the sequential subtask loop in rlm.py with a DAG that automatically
runs independent tasks in parallel via asyncio.gather, while respecting
data dependencies between tasks.

Key insight: if a query requires searching 7 sources, all 7 searches can run
in parallel instead of sequentially — 7× speedup with the same tools.

Features:
  - Cycle detection (Kahn's algorithm) — prevents infinite loops
  - Adaptive SLA per agent type — tighter timeouts for fast tasks
  - Speculative execution — pre-warm downstream tasks for deep DAGs
  - Duplicate ID detection — warns and deduplicates
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ── Adaptive SLA defaults per agent type (#8) ─────────────────────────────
# These replace the flat 120s timeout.  Each is tuned to the expected
# execution time for that agent type:
#   - archive:    ALMA/CADC HTTP calls = 5-30s typically, 60s generous
#   - literature:  ADS API call = 5-15s, 45s generous
#   - analysis:   Splatalogue + computation = 10-30s, 90s generous
#   - viz:        FITS download + rendering = 30-90s, 180s generous
#   - web:        Tavily search = 3-10s, 30s generous
#   - synthesis:  LLM generation = 10-30s, 120s generous
DEFAULT_SLA_BY_AGENT_TYPE = {
    "archive":    60,
    "literature": 45,
    "analysis":   90,
    "viz":        180,
    "web":        30,
    "synthesis":  120,
    "general":    120,
}


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    SPECULATIVE = "speculative"  # (#13) pre-warmed but not yet needed


@dataclass
class TaskNode:
    """Single node in the task execution graph."""

    id: str
    description: str
    depends_on: List[str] = field(default_factory=list)
    agent_type: str = "general"        # archive, literature, analysis, viz, web, synthesis
    model_hint: Optional[str] = None   # optional model preference override
    sla_seconds: int = 120             # max time allowed per task (overridden by adaptive SLA)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retries: int = 0
    model_used: Optional[str] = None   # (#6) track which model was actually used
    input_tokens: int = 0              # (#6) token tracking
    output_tokens: int = 0             # (#6) token tracking
    estimated_cost_usd: float = 0.0    # (#6) cost tracking

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return round(self.end_time - self.start_time, 2)
        return None


class TaskDAG:
    """
    Dependency-aware task graph with parallel execution.

    Features:
      - Cycle detection via Kahn's algorithm
      - Adaptive SLA per agent type
      - Speculative execution for deep DAGs
      - Duplicate ID and self-loop detection

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
        self._validation_warnings: List[str] = []

    def build_from_subtasks(self, subtasks: List[dict]) -> "TaskDAG":
        """
        Build the DAG from a list of structured subtask dicts.

        Each dict should have:
          - id: str
          - description: str
          - depends_on: List[str]       (optional, defaults to [])
          - agent_type: str             (optional, defaults to "general")
          - model_hint: str             (optional)
          - sla_seconds: int            (optional, auto-set by agent_type)
        """
        self.nodes.clear()
        self._adjacency.clear()
        self._validation_warnings.clear()

        seen_ids: Set[str] = set()

        for st in subtasks:
            task_id = st.get("id") or f"t{len(self.nodes) + 1}"

            # (#3) Duplicate ID detection
            if task_id in seen_ids:
                new_id = f"{task_id}_{len(self.nodes)}"
                self._validation_warnings.append(
                    f"Duplicate task ID '{task_id}' — renamed to '{new_id}'"
                )
                logger.warning("Duplicate task ID '%s' — renamed to '%s'", task_id, new_id)
                task_id = new_id
            seen_ids.add(task_id)

            # (#8) Adaptive SLA: use agent_type-specific timeout if not explicitly set
            agent_type = st.get("agent_type", "general")
            if "sla_seconds" in st:
                sla = st["sla_seconds"]
            else:
                sla = DEFAULT_SLA_BY_AGENT_TYPE.get(agent_type, 120)

            node = TaskNode(
                id=task_id,
                description=st.get("description", ""),
                depends_on=st.get("depends_on", []),
                agent_type=agent_type,
                model_hint=st.get("model_hint"),
                sla_seconds=sla,
            )
            self.nodes[task_id] = node

            # Build reverse adjacency (who depends on me)
            for dep_id in node.depends_on:
                self._adjacency.setdefault(dep_id, []).append(task_id)

        # Validate the DAG
        self._validate()

        return self

    # ── Validation (#3) ────────────────────────────────────────────────────

    def _validate(self) -> None:
        """
        Validate the DAG:
          1. Remove references to unknown dependencies
          2. Remove self-loops
          3. Detect and break cycles via Kahn's algorithm
        """
        # 1. Remove unknown dependencies
        for node in self.nodes.values():
            bad_deps = [d for d in node.depends_on if d not in self.nodes]
            for d in bad_deps:
                self._validation_warnings.append(
                    f"Task '{node.id}' depends on unknown task '{d}' — removed"
                )
                logger.warning(
                    "Task %s depends on unknown task %s — removing dependency",
                    node.id, d,
                )
            node.depends_on = [d for d in node.depends_on if d in self.nodes]

        # 2. Remove self-loops
        for node in self.nodes.values():
            if node.id in node.depends_on:
                self._validation_warnings.append(
                    f"Task '{node.id}' depends on itself — removed self-loop"
                )
                logger.warning("Task %s has self-loop — removing", node.id)
                node.depends_on = [d for d in node.depends_on if d != node.id]

        # 3. Cycle detection via Kahn's algorithm
        cycle_members = self._detect_cycles()
        if cycle_members:
            self._validation_warnings.append(
                f"Cycle detected involving tasks: {cycle_members} — breaking edges"
            )
            logger.warning("DAG cycle detected: %s — breaking edges", cycle_members)
            self._break_cycles(cycle_members)

    def _detect_cycles(self) -> List[str]:
        """
        Detect cycles using Kahn's algorithm (topological sort).

        Returns a list of task IDs involved in cycles, or empty list if acyclic.
        """
        # Build in-degree map
        in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes}
        for node in self.nodes.values():
            for dep_id in node.depends_on:
                if dep_id in self.nodes:
                    # dep_id is depended ON, but in_degree counts incoming
                    # node.id has an incoming edge from dep_id
                    pass
            # Actually: depends_on means "I need these to finish first"
            # So in_degree[node.id] = len(node.depends_on)
        for node in self.nodes.values():
            in_degree[node.id] = len([d for d in node.depends_on if d in self.nodes])

        # Kahn's: start with nodes that have in_degree 0
        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        visited: Set[str] = set()

        while queue:
            nid = queue.popleft()
            visited.add(nid)

            # For each node that depends on nid, decrease its in-degree
            for dependent_id in self._adjacency.get(nid, []):
                if dependent_id in in_degree:
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        queue.append(dependent_id)

        # Any node NOT visited is part of a cycle
        cycle_members = [nid for nid in self.nodes if nid not in visited]
        return cycle_members

    def _break_cycles(self, cycle_members: List[str]) -> None:
        """
        Break cycles by removing the last dependency edge from cycle members.

        Strategy: for each cycle member, remove its last depends_on entry.
        This is a heuristic — it breaks the cycle with minimal disruption.
        """
        for nid in cycle_members:
            node = self.nodes[nid]
            if node.depends_on:
                removed_dep = node.depends_on.pop()
                logger.warning(
                    "Breaking cycle: removed dependency %s → %s",
                    node.id, removed_dep,
                )
                # Also clean up the reverse adjacency
                if removed_dep in self._adjacency:
                    self._adjacency[removed_dep] = [
                        d for d in self._adjacency[removed_dep] if d != nid
                    ]

        # Verify cycle is broken
        remaining = self._detect_cycles()
        if remaining:
            # Nuclear option: remove ALL dependencies from cycle members
            logger.error("Cycle still exists after breaking — removing all deps for: %s", remaining)
            for nid in remaining:
                self.nodes[nid].depends_on.clear()

    def get_validation_warnings(self) -> List[str]:
        """Return any validation warnings from the last build_from_subtasks() call."""
        return self._validation_warnings.copy()

    # ── Speculative Execution (#13) ────────────────────────────────────────

    def get_speculatable_tasks(self) -> List[TaskNode]:
        """
        Return tasks that could be speculatively started.

        A task is speculatable when:
          - It is PENDING
          - At least one (but not all) of its dependencies is COMPLETED
          - Its remaining dependencies are currently RUNNING
          - It is an independent computation (analysis, viz) not a synthesis

        This enables pre-warming downstream tasks while upstream is still running.
        """
        speculatable = []
        for node in self.nodes.values():
            if node.status != TaskStatus.PENDING:
                continue
            if node.agent_type == "synthesis":
                continue  # Never speculate on synthesis — it needs ALL results

            deps = [d for d in node.depends_on if d in self.nodes]
            if not deps:
                continue  # No dependencies = ready, not speculative

            completed_deps = [d for d in deps if self.nodes[d].status == TaskStatus.COMPLETED]
            running_deps = [d for d in deps if self.nodes[d].status == TaskStatus.RUNNING]

            # At least one done, rest are running (not failed/pending)
            if completed_deps and running_deps:
                all_active = len(completed_deps) + len(running_deps) == len(deps)
                if all_active:
                    speculatable.append(node)

        return speculatable

    def mark_speculative(self, task_id: str) -> None:
        """Mark a task as speculatively started."""
        node = self.nodes.get(task_id)
        if node:
            node.status = TaskStatus.SPECULATIVE
            node.start_time = time.time()
            logger.info("Task %s marked as speculative", task_id)

    # ── Core Operations ───────────────────────────────────────────────────

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

    # ── DAG Depth Analysis (#13) ──────────────────────────────────────────

    def get_critical_path_depth(self) -> int:
        """
        Return the longest chain of sequential dependencies (critical path).
        Used to decide whether speculative execution is worth the overhead.
        """
        if not self.nodes:
            return 0

        memo: Dict[str, int] = {}

        def _depth(nid: str) -> int:
            if nid in memo:
                return memo[nid]
            node = self.nodes.get(nid)
            if not node or not node.depends_on:
                memo[nid] = 1
                return 1
            max_dep = max(
                _depth(d) for d in node.depends_on if d in self.nodes
            ) if node.depends_on else 0
            memo[nid] = max_dep + 1
            return memo[nid]

        return max(_depth(nid) for nid in self.nodes)

    # ── Execution ──────────────────────────────────────────────────────────

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
                        # Propagate the actual predecessor error for diagnostics
                        failed_deps = [
                            (dep_id, self.nodes[dep_id].error or "unknown error")
                            for dep_id in p.depends_on
                            if dep_id in self.nodes
                            and self.nodes[dep_id].status == TaskStatus.FAILED
                        ]
                        if failed_deps:
                            cause = "; ".join(f"{did}: {err[:100]}" for did, err in failed_deps)
                            self.mark_failed(p.id, f"Predecessor failed — {cause}")
                        else:
                            self.mark_failed(p.id, "Unmet dependencies — predecessor did not complete")
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

    # ── Summaries ──────────────────────────────────────────────────────────

    def get_execution_summary(self) -> Dict[str, Any]:
        """Return a human-readable summary of the DAG execution."""
        total_cost = sum(n.estimated_cost_usd for n in self.nodes.values())
        total_input_tokens = sum(n.input_tokens for n in self.nodes.values())
        total_output_tokens = sum(n.output_tokens for n in self.nodes.values())

        return {
            "total_tasks": len(self.nodes),
            "completed": sum(1 for n in self.nodes.values() if n.status == TaskStatus.COMPLETED),
            "failed": sum(1 for n in self.nodes.values() if n.status == TaskStatus.FAILED),
            "critical_path_depth": self.get_critical_path_depth(),
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "validation_warnings": self._validation_warnings,
            "tasks": [
                {
                    "id": n.id,
                    "description": n.description,
                    "status": n.status.value,
                    "agent_type": n.agent_type,
                    "model_used": n.model_used,
                    "duration_sec": n.duration,
                    "sla_seconds": n.sla_seconds,
                    "input_tokens": n.input_tokens,
                    "output_tokens": n.output_tokens,
                    "estimated_cost_usd": round(n.estimated_cost_usd, 6),
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

    def to_plan_dict(self) -> Dict[str, Any]:
        """
        Serialize the DAG for human-in-the-loop approval (#12).

        Returns a structured dict that the frontend can render as an
        interactive plan approval UI.
        """
        return {
            "total_tasks": len(self.nodes),
            "critical_path_depth": self.get_critical_path_depth(),
            "estimated_parallel_rounds": self._estimate_rounds(),
            "tasks": [
                {
                    "id": n.id,
                    "description": n.description,
                    "depends_on": n.depends_on,
                    "agent_type": n.agent_type,
                    "sla_seconds": n.sla_seconds,
                }
                for n in self.nodes.values()
            ],
            "validation_warnings": self._validation_warnings,
        }

    def _estimate_rounds(self) -> int:
        """Estimate the number of parallel execution rounds."""
        # Simulate the execution to count rounds
        simulated_completed: Set[str] = set()
        rounds = 0

        for _ in range(len(self.nodes) + 1):  # safety cap
            ready = [
                n for n in self.nodes.values()
                if n.id not in simulated_completed
                and all(d in simulated_completed for d in n.depends_on if d in self.nodes)
            ]
            if not ready:
                break
            simulated_completed.update(n.id for n in ready)
            rounds += 1

        return rounds
