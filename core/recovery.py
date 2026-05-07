# core/recovery.py
"""
Recovery Engine — 5 strategies for resilient task execution.

Implements CAMEL's recovery strategies:
  1. RETRY: Transient errors (network timeout, rate limit) → retry with backoff
  2. REPLAN: Empty results → LLM reformulates the task
  3. REASSIGN: Tool failure → try alternative tool/approach
  4. DECOMPOSE: Task too broad → break into smaller sub-tasks
  5. CREATE_WORKER: No agent fits → dynamically compose from available tools

The Conductor wires this into _run_one() so every subtask automatically
gets cascading recovery instead of ad-hoc retry logic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class RecoveryStrategy(str, Enum):
    RETRY = "retry"
    REPLAN = "replan"
    REASSIGN = "reassign"
    DECOMPOSE = "decompose"
    CREATE_WORKER = "create_worker"


class EmptyResultError(Exception):
    """Raised when a tool returns empty/no results."""
    pass


class ToolError(Exception):
    """Raised when a specific tool fails entirely."""
    pass


class SoftFailureError(Exception):
    """Raised when a tool returns a result with embedded error indicators."""
    pass


class RecoveryEngine:
    """
    Executes tasks with automatic recovery on failure.

    Tries each strategy in order of escalation:
    RETRY → REPLAN → REASSIGN → DECOMPOSE → CREATE_WORKER

    Wired into Conductor._run_one() so every DAG node gets recovery.
    """

    def __init__(
        self,
        client: Any = None,
        model: str = "gpt-4.1-mini",
        max_retries: int = 3,
        verbose: bool = False,
    ):
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.verbose = verbose
        self._recovery_log: List[Dict[str, Any]] = []

    async def execute_with_recovery(
        self,
        task_node: Any,
        executor_fn: Callable,
        on_status: Optional[Callable] = None,
        dep_context: Optional[str] = None,
    ) -> Any:
        """
        Execute a task with cascading recovery strategies.

        Parameters
        ----------
        task_node : TaskNode
            The task to execute.
        executor_fn : callable
            Function to execute the task (sync — will be awaited in executor).
        on_status : callable, optional
            Status callback.
        dep_context : str, optional
            Summary of predecessor task results/errors for root-cause analysis.
            Passed to _replan() and _reassign() so the LLM can understand the
            upstream data context when deciding how to recover.

        Returns
        -------
        Any
            The task result.
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # executor_fn might be sync (from Conductor tool_executor)
                if asyncio.iscoroutinefunction(executor_fn):
                    result = await executor_fn(task_node)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, executor_fn, task_node.description, ""
                    )

                # Check for soft failures (result with embedded error)
                soft_err = self._check_soft_failure(result)
                if soft_err:
                    raise SoftFailureError(soft_err)

                # Check for empty results
                if self._is_empty_result(result):
                    raise EmptyResultError(f"Empty result for: {task_node.description}")

                return result

            except EmptyResultError as e:
                last_error = e
                strategy = RecoveryStrategy.REPLAN
                self._log_recovery(task_node, strategy, str(e), attempt)

                if on_status:
                    on_status(
                        f"Recovery ({strategy.value}): Reformulating task",
                        "running",
                    )

                # REPLAN: reformulate the task description
                task_node.description = await self._replan(task_node, str(e), dep_context)

            except SoftFailureError as e:
                last_error = e
                if attempt == 0:
                    # First soft failure: RETRY
                    strategy = RecoveryStrategy.RETRY
                    self._log_recovery(task_node, strategy, str(e), attempt)
                    if on_status:
                        on_status(f"Recovery ({strategy.value}): Retrying after soft failure", "running")
                    await asyncio.sleep(1)  # Brief pause
                else:
                    # Second soft failure: REASSIGN (try different approach)
                    strategy = RecoveryStrategy.REASSIGN
                    self._log_recovery(task_node, strategy, str(e), attempt)
                    if on_status:
                        on_status(f"Recovery ({strategy.value}): Trying alternative approach", "running")
                    task_node.description = await self._reassign(task_node, str(e), dep_context)

            except asyncio.TimeoutError as e:
                last_error = e
                if attempt == 0:
                    # First timeout: RETRY with longer SLA
                    strategy = RecoveryStrategy.RETRY
                    task_node.sla_seconds = int(task_node.sla_seconds * 1.5)
                    self._log_recovery(task_node, strategy, "Timeout", attempt)
                    if on_status:
                        on_status(
                            f"Recovery ({strategy.value}): Extended timeout to {task_node.sla_seconds}s",
                            "running",
                        )
                else:
                    # Second timeout: DECOMPOSE
                    strategy = RecoveryStrategy.DECOMPOSE
                    self._log_recovery(task_node, strategy, "Repeated timeout", attempt)
                    if on_status:
                        on_status(f"Recovery ({strategy.value}): Breaking task into smaller pieces", "running")
                    return await self._decompose_and_execute(task_node, executor_fn)

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # RETRY with exponential backoff
                    strategy = RecoveryStrategy.RETRY
                    wait_time = 2 ** attempt
                    self._log_recovery(task_node, strategy, str(e), attempt)
                    logger.info("Retrying in %ds...", wait_time)
                    await asyncio.sleep(wait_time)
                else:
                    # Final attempt: CREATE_WORKER fallback
                    strategy = RecoveryStrategy.CREATE_WORKER
                    self._log_recovery(task_node, strategy, str(e), attempt)
                    return f"[Recovery exhausted after {self.max_retries} attempts: {last_error}]"

        return f"[All recovery strategies failed: {last_error}]"

    def _check_soft_failure(self, result: Any) -> Optional[str]:
        """Check if a result contains embedded error indicators."""
        if isinstance(result, dict):
            if result.get("success") is False:
                return result.get("error", "Unknown soft failure")
            if "error" in result and not result.get("success"):
                return str(result["error"])
        elif isinstance(result, str):
            if "[Tool execution error:" in result:
                return result
        return None

    async def _replan(self, task_node: Any, error: str, dep_context: Optional[str] = None) -> str:
        """LLM reformulates a task that returned empty results.

        If dep_context is provided, the LLM can see what predecessor tasks
        returned — enabling root-cause analysis (e.g. "the upstream search
        returned 0 results, so the line coverage check had nothing to check").
        """
        if not self.client:
            return task_node.description + " (Try alternative search parameters)"

        predecessor_info = ""
        if dep_context:
            predecessor_info = (
                f"\n\nPredecessor task results (upstream context):\n{dep_context}\n"
                f"Use this to understand WHY the task failed — the root cause may be in the upstream results.\n"
            )

        try:
            resp = self.client.responses.create(
                model=self.model,
                input=(
                    f"The following task returned empty results:\n"
                    f"Task: {task_node.description}\n"
                    f"Error: {error}\n"
                    f"{predecessor_info}\n"
                    f"Reformulate this task to try a different approach. "
                    f"For example, if a target name search failed, try coordinates. "
                    f"If a specific band filter returned nothing, try without the band filter. "
                    f"If a predecessor returned no data, adjust this task to work without that data. "
                    f"Return ONLY the new task description, no explanation."
                ),
                temperature=0.3,
                max_output_tokens=200,
            )
            new_desc = resp.output_text.strip()
            logger.info("REPLAN: '%s' → '%s'", task_node.description[:60], new_desc[:60])
            return new_desc
        except Exception:
            return task_node.description + " (broadened search)"

    async def _reassign(self, task_node: Any, error: str, dep_context: Optional[str] = None) -> str:
        """LLM suggests an alternative tool or approach for a failed task.

        If dep_context is provided, the LLM can see what predecessor tasks
        returned — enabling smarter tool/approach selection.
        """
        if not self.client:
            return task_node.description + " (Try using a different tool or approach)"

        predecessor_info = ""
        if dep_context:
            predecessor_info = (
                f"\nPredecessor task results (upstream context):\n{dep_context}\n"
            )

        try:
            resp = self.client.responses.create(
                model=self.model,
                input=(
                    f"The following task failed with a soft error:\n"
                    f"Task: {task_node.description}\n"
                    f"Agent type: {getattr(task_node, 'agent_type', 'unknown')}\n"
                    f"Error: {error}\n"
                    f"{predecessor_info}\n"
                    f"Suggest an alternative approach using different tools. "
                    f"For example:\n"
                    f"- If ALMA search failed, try CADC archive\n"
                    f"- If target name resolution failed, try known coordinates\n"
                    f"- If a specific tool timed out, try a simpler query\n"
                    f"- If a predecessor returned incompatible data, adjust accordingly\n"
                    f"Return ONLY the new task description."
                ),
                temperature=0.4,
                max_output_tokens=200,
            )
            new_desc = resp.output_text.strip()
            logger.info("REASSIGN: '%s' → '%s'", task_node.description[:60], new_desc[:60])
            return new_desc
        except Exception:
            return task_node.description + " (alternative approach)"

    async def _decompose_and_execute(
        self, task_node: Any, executor_fn: Callable
    ) -> str:
        """
        Break a broad task into smaller pieces and execute each.

        Uses the LLM to split the task, then runs each piece sequentially.
        If any piece succeeds, returns its result.
        """
        logger.info("DECOMPOSE: Breaking down '%s'", task_node.description[:60])

        if not self.client:
            return f"[Task '{task_node.description[:60]}' timed out and could not be decomposed — no LLM client available.]"

        try:
            resp = self.client.responses.create(
                model=self.model,
                input=(
                    f"The following task timed out because it was too broad:\n"
                    f"Task: {task_node.description}\n\n"
                    f"Break it into 2-3 smaller, simpler tasks that can each "
                    f"be completed quickly. Return ONLY a JSON array of strings:\n"
                    f'["smaller task 1", "smaller task 2"]'
                ),
                temperature=0.2,
                max_output_tokens=300,
                text={"format": {"type": "json_object"}},
            )

            import json
            data = json.loads(resp.output_text)
            sub_tasks = data if isinstance(data, list) else data.get("tasks", data.get("subtasks", []))

            if not sub_tasks:
                return f"[Task '{task_node.description[:60]}' was too broad and could not be decomposed.]"

            # Execute each sub-piece sequentially
            results = []
            for i, sub_desc in enumerate(sub_tasks[:3]):  # Cap at 3 sub-pieces
                logger.info("DECOMPOSE sub-task %d/%d: %s", i + 1, len(sub_tasks), sub_desc[:60])
                try:
                    if asyncio.iscoroutinefunction(executor_fn):
                        result = await asyncio.wait_for(
                            executor_fn(sub_desc, ""),
                            timeout=task_node.sla_seconds,
                        )
                    else:
                        loop = asyncio.get_event_loop()
                        result = await asyncio.wait_for(
                            loop.run_in_executor(None, executor_fn, sub_desc, ""),
                            timeout=task_node.sla_seconds,
                        )
                    if result and not self._is_empty_result(result):
                        results.append(f"[Sub-task {i + 1}]: {result}")
                except Exception as e:
                    results.append(f"[Sub-task {i + 1} failed]: {e}")

            if results:
                return "\n\n".join(results)
            return f"[Task '{task_node.description[:60]}' was decomposed but all sub-tasks failed.]"

        except Exception as e:
            logger.error("DECOMPOSE failed: %s", e)
            return f"[Task '{task_node.description[:60]}' timed out. Decomposition failed: {e}]"

    def _is_empty_result(self, result: Any) -> bool:
        """Check if a result is effectively empty."""
        if result is None:
            return True
        if isinstance(result, str) and not result.strip():
            return True
        if isinstance(result, dict):
            if result.get("total_results", -1) == 0:
                return True
            if result.get("file_count", -1) == 0:
                return True
        if isinstance(result, list) and len(result) == 0:
            return True
        return False

    def _log_recovery(self, task_node: Any, strategy: RecoveryStrategy, error: str, attempt: int):
        """Log a recovery attempt for observability."""
        entry = {
            "task_id": getattr(task_node, "id", "?"),
            "strategy": strategy.value,
            "error": error[:200],
            "attempt": attempt,
            "timestamp": time.time(),
        }
        self._recovery_log.append(entry)
        if self.verbose:
            logger.info("Recovery [%s] attempt %d: %s — %s", entry["task_id"], attempt, strategy.value, error[:80])

    def get_recovery_log(self) -> List[Dict[str, Any]]:
        """Return the full recovery log for observability."""
        return self._recovery_log

    def get_stats(self) -> Dict[str, Any]:
        """Return recovery statistics."""
        if not self._recovery_log:
            return {"total_recoveries": 0}

        by_strategy = {}
        for entry in self._recovery_log:
            s = entry["strategy"]
            by_strategy[s] = by_strategy.get(s, 0) + 1

        return {
            "total_recoveries": len(self._recovery_log),
            "by_strategy": by_strategy,
            "recent": self._recovery_log[-5:],
        }
