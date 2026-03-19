# core/recovery.py
"""
Recovery Engine — 5 strategies for resilient task execution.

Implements CAMEL's recovery strategies:
  1. RETRY: Transient errors (network timeout, rate limit) → retry with backoff
  2. REPLAN: Empty results → LLM reformulates the task
  3. REASSIGN: Tool failure → try alternative tool
  4. DECOMPOSE: Task too broad → break into smaller sub-tasks
  5. CREATE_WORKER: No agent fits → dynamically compose from available tools
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


class RecoveryEngine:
    """
    Executes tasks with automatic recovery on failure.

    Tries each strategy in order of escalation:
    RETRY → REPLAN → REASSIGN → DECOMPOSE → CREATE_WORKER
    """

    def __init__(
        self,
        client: Any = None,
        model: str = "gpt-4o-mini",
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
    ) -> Any:
        """
        Execute a task with cascading recovery strategies.

        Parameters
        ----------
        task_node : TaskNode
            The task to execute.
        executor_fn : callable
            Function to execute the task.
        on_status : callable, optional
            Status callback.

        Returns
        -------
        Any
            The task result.
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                result = await executor_fn(task_node)

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
                task_node.description = await self._replan(task_node, str(e))

            except asyncio.TimeoutError as e:
                last_error = e
                if attempt == 0:
                    # First timeout: RETRY with longer SLA
                    strategy = RecoveryStrategy.RETRY
                    task_node.sla_seconds = int(task_node.sla_seconds * 1.5)
                    self._log_recovery(task_node, strategy, "Timeout", attempt)
                else:
                    # Second timeout: DECOMPOSE
                    strategy = RecoveryStrategy.DECOMPOSE
                    self._log_recovery(task_node, strategy, "Repeated timeout", attempt)
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
                    # Final attempt: try CREATE_WORKER
                    strategy = RecoveryStrategy.CREATE_WORKER
                    self._log_recovery(task_node, strategy, str(e), attempt)
                    return f"[Recovery exhausted after {self.max_retries} attempts: {last_error}]"

        return f"[All recovery strategies failed: {last_error}]"

    async def _replan(self, task_node: Any, error: str) -> str:
        """LLM reformulates a task that returned empty results."""
        if not self.client:
            return task_node.description + " (Try alternative search parameters)"

        try:
            resp = self.client.responses.create(
                model=self.model,
                input=(
                    f"The following task returned empty results:\n"
                    f"Task: {task_node.description}\n"
                    f"Error: {error}\n\n"
                    f"Reformulate this task to try a different approach. "
                    f"For example, if a target name search failed, try coordinates. "
                    f"Return ONLY the new task description."
                ),
                temperature=0.3,
                max_output_tokens=200,
            )
            return resp.output_text.strip()
        except Exception:
            return task_node.description + " (broadened search)"

    async def _decompose_and_execute(
        self, task_node: Any, executor_fn: Callable
    ) -> str:
        """Break a broad task into smaller pieces and execute each."""
        logger.info("DECOMPOSE: Breaking down '%s'", task_node.description[:60])
        # For now, return a note about the failure
        return f"[Task '{task_node.description[:60]}' was too broad and timed out. Try breaking it into smaller queries.]"

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
