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
import copy
import logging
import re
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# R7 — ReplicationBench failure mode #1: agents prematurely giving up while
# falsely citing compute/resource limits ("cannot be completed due to
# computational constraints") instead of attempting the work with the tools
# they actually have. Matched against textual task results; a hit triggers
# ONE replan with an explicit budget statement (capped — a false positive
# must never re-run an expensive DAG more than once).
_GIVE_UP_RE = re.compile(
    r"(?i)(?:"
    r"(?:cannot|can't|can\s+not|couldn't|could\s+not|won't|will\s+not|"
    r"wasn't|was\s+not|isn't|is\s+not|unable\s+to)\s+(?:be\s+)?"
    r"(?:\w+ly\s+)?"
    r"(?:complet|perform|execut|run|do|carr)\w*\s+"
    r"(?:this|the|such)?\s*\w{0,12}\s*"
    r"(?:due\s+to|because\s+of|given|owing\s+to)\s+"
    r"(?:my|the|our|available\s+)?\s*"
    r"(?:computational|compute|resource|time|memory|processing)"
    r"|(?:computational|compute|resource|memory|processing)\s+"
    r"(?:limit|constraint|restriction|budget)s?\s+"
    r"(?:prevent|preclude|do(?:es)?\s+not\s+allow|don't\s+allow|make\s+it\s+impossible|prohibit)"
    r"|limited\s+(?:compute|computational\s+\w+|resources|memory|time)\s+prevents?"
    r"|(?:infeasible|not\s+feasible|impossible)\s+"
    r"(?:due\s+to|because\s+of|given)\s+"
    r"(?:my|the|our)?\s*(?:computational|compute|resource|memory|processing)"
    r"|too\s+computationally\s+(?:expensive|intensive|demanding)"
    r"|(?:exceeds?|beyond)\s+(?:my|the|available)\s+"
    r"(?:computational\s+|compute\s+|processing\s+)?(?:capabilit|capacit|resource)\w*"
    r"|as\s+an\s+ai(?:\s+(?:language\s+)?model)?\s*,?\s+i\s+(?:cannot|can't|am\s+unable)"
    r"|would\s+(?:require|take|need)\s+(?:excessive|prohibitive|too\s+much)\s+"
    r"(?:time|compute|computation|memory|resources)"
    r")"
)

# Suppressor (guard CX-12): text that ALSO reports successful completion is an
# explanation, not a refusal — "would require excessive memory, but the query
# completed successfully" must not trigger a re-run.
_COMPLETION_RE = re.compile(
    r"(?i)\b(?:"
    r"completed\s+successfully|successfully\s+(?:completed|ran|executed|finished)|"
    r"query\s+(?:completed|succeeded|ran)|"
    r"results?\s+(?:are|is)\s+(?:below|as\s+follows|shown)|"
    r"here\s+(?:are|is)\s+the\s+results?"
    r")\b"
)

# A completion phrase DIRECTLY preceded by a negator ("this canNOT be
# completed successfully...") is a REFUSAL, not a completion report — it must
# not feed the suppressor (verify-round CX-12 regression). Adjacent-only:
# only an auxiliary ("be"/"been"/"being"/"get") may sit between the negator
# and the phrase, so an unrelated "not"/"without" earlier in the sentence
# ("did not use the naive scan and completed successfully") does not negate it.
_NEG_CORE = (
    r"\b(?:cannot|can't|can\s+not|couldn't|could\s+not|won't|will\s+not|"
    r"never|unable\s+to|not|isn't|wasn't|doesn't|didn't|hasn't|haven't|"
    r"fail(?:s|ed)?\s+to)\s+"
    r"(?:(?:have|has|had)\s+been\s+|been\s+|be\s+|being\s+|get\s+|to\s+be\s+)?"
)
_NEGATION_ADJACENT_RE = re.compile(r"(?i)" + _NEG_CORE + r"$")
_NEGATION_WITH_ADVERB_RE = re.compile(r"(?i)" + _NEG_CORE + r"(?:\w+ly\s+){1,2}$")

# Structural disambiguation for "not <adverb> completed successfully":
# focusing-adverb AFFIRMATIVES ("not only/merely/uniquely completed
# successfully, but also… / ; the others also succeeded") always continue
# with a correlative or additional-success clause; manner-adverb NEGATIONS
# ("not fully completed successfully; it would require…") do not. Checking
# the continuation instead of enumerating adverbs handles the whole class.
# The continuation must carry ADDITIVE-SUCCESS content: "(but) also
# <something that is not a failure verb>" or a "succeeded" that is not
# preceded by a negative quantifier/failure word anywhere in the window.
# Bare "but"/"additionally"/"as well" can introduce the refusal itself, and
# "also failed" / "No fallback succeeded" are failures — none of those count.
_AFFIRMATIVE_ALSO_RE = re.compile(
    r"(?i)^[^.!?\n]{0,80}?\b(?:but\s+)?also\s+"
    r"(?!fail|not\b|never\b|couldn|didn|doesn|wasn|isn|cannot\b|can't|won't"
    r"|(?:did|could|does|do|can|will|would|may|might|must|is|was)\s+not\b)"
    r"\w+"
)
_AFFIRMATIVE_SUCCEED_RE = re.compile(
    r"(?i)^(?:(?!\b(?:no|none|neither|nothing|never|fail\w*|not)\b)[^.!?\n]){0,90}?"
    r"\bsucceed(?:ed|s)?\b"
)


def _affirmative_continuation(tail: str) -> bool:
    return bool(_AFFIRMATIVE_ALSO_RE.search(tail)
                or _AFFIRMATIVE_SUCCEED_RE.search(tail))


def _reports_completion(text: str) -> bool:
    """True when the text affirmatively reports successful completion.

    Misclassification here degrades GRACEFULLY, never corrupts an answer:
    a false "completion" verdict merely skips the challenge — the refusal
    passes through unchallenged, which is exactly the pre-R7 baseline
    behavior — and a false "refusal" verdict costs at most ONE duplicate
    subtask execution (the challenge is hard-capped at a single retry).
    """
    for match in _COMPLETION_RE.finditer(text):
        prefix = text[max(0, match.start() - 80):match.start()]
        if _NEGATION_ADJACENT_RE.search(prefix):
            continue  # hard-negated completion ("cannot be completed…")
        if _NEGATION_WITH_ADVERB_RE.search(prefix):
            tail = text[match.end():match.end() + 90]
            if _affirmative_continuation(tail):
                return True  # "not only/merely/… , but also/succeeded…"
            continue  # adverbial negation without continuation ("not fully…")
        return True
    return False

# Appended to the task description on the single give-up replan.
_BUDGET_STATEMENT = (
    " [EXECUTION BUDGET NOTE: You have an adequate budget for this task — "
    "several minutes of wall-clock and the full archive/catalog/analysis "
    "tool set. Do NOT refuse based on claimed computational or resource "
    "limits: attempt the task with the available tools. If a specific step "
    "genuinely fails, report exactly which tool call failed and with what "
    "error, instead of declaring the task infeasible.]"
)


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


class PrematureGiveUpError(Exception):
    """Raised when a textual result declares infeasibility on claimed
    compute/resource limits (ReplicationBench failure mode #1, R7)."""
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
                # executor_fn may be a coroutine (preferred: the full node
                # executor, called with the node) or sync (legacy: called with
                # description/context).
                if asyncio.iscoroutinefunction(executor_fn):
                    coro = executor_fn(task_node)
                else:
                    loop = asyncio.get_event_loop()
                    coro = loop.run_in_executor(
                        None, executor_fn, task_node.description, ""
                    )

                # Enforce the task's SLA on the attempt so the TimeoutError
                # handler below (RETRY → DECOMPOSE) can actually fire — without
                # this wrapper nothing imposed a deadline and that branch was
                # dead code. (C7)
                sla = getattr(task_node, "sla_seconds", None)
                if isinstance(sla, (int, float)) and sla > 0:
                    result = await asyncio.wait_for(coro, timeout=sla)
                else:
                    result = await coro

                # Check for soft failures (result with embedded error)
                soft_err = self._check_soft_failure(result)
                if soft_err:
                    raise SoftFailureError(soft_err)

                # Check for empty results
                if self._is_empty_result(result):
                    raise EmptyResultError(f"Empty result for: {task_node.description}")

                # R7: premature give-up on claimed compute limits — challenge
                # it ONCE (attempt 0 only, and only when a retry attempt
                # actually exists — CX-13: with max_retries=1 the raise would
                # end as "recovery exhausted" instead of returning the answer).
                # On the retried attempt any give-up text is accepted as the
                # honest answer; a false-positive match must never re-run an
                # expensive task more than once.
                give_up = self._check_premature_give_up(result)
                if give_up and attempt == 0 and self.max_retries > 1:
                    raise PrematureGiveUpError(give_up)

                return result

            except PrematureGiveUpError as e:
                last_error = e
                strategy = RecoveryStrategy.REPLAN
                self._log_recovery(task_node, strategy, f"premature give-up: {e}", attempt)
                if on_status:
                    on_status(
                        "Recovery (replan): Result declared infeasibility on claimed "
                        "compute limits — retrying once with an explicit budget statement",
                        "running",
                    )
                if _BUDGET_STATEMENT not in task_node.description:
                    task_node.description = task_node.description + _BUDGET_STATEMENT

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

    def _check_premature_give_up(self, result: Any) -> Optional[str]:
        """R7: detect give-up-on-claimed-compute-limits answers.

        Only textual results are checked (LLM answers); dict tool results with
        real errors are already handled by _check_soft_failure. Returns the
        matched snippet for the recovery log, or None.
        """
        if isinstance(result, str):
            text = result
        elif isinstance(result, dict):
            candidate = result.get("answer") or result.get("text") or result.get("content")
            text = candidate if isinstance(candidate, str) else ""
        else:
            return None
        if not text:
            return None
        # Normalize typographic apostrophes so "couldn’t"/"can’t" hit the
        # same contraction alternatives as their ASCII forms.
        text = text.replace("’", "'").replace("ʼ", "'")
        match = _GIVE_UP_RE.search(text)
        if match and not _reports_completion(text):
            start = max(0, match.start() - 40)
            return text[start:match.end() + 60].strip()
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
                        # executor_fn is the full node executor and takes a
                        # TaskNode, not (desc, ctx). Run each sub-piece through a
                        # shallow-cloned node with the sub-description so routing/
                        # SLA metadata is preserved. (C7 — DECOMPOSE arity fix)
                        sub_node = copy.copy(task_node)
                        try:
                            sub_node.description = sub_desc
                        except Exception:
                            sub_node = task_node
                        result = await asyncio.wait_for(
                            executor_fn(sub_node),
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
            # C8: a SUCCESSFUL tool result with an explicit zero count is a
            # meaningful scientific answer ("confirmed: no data found"), not a
            # failure to recover from. C3 made genuine archive outages typed
            # errors (success=False + error), which _check_soft_failure catches
            # BEFORE this check — so recovering here only rewrote valid
            # "none found" answers via RETRY/REPLAN.
            if result.get("success") is True:
                return False
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
