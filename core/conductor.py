# core/conductor.py
"""
Conductor — Central reasoning engine for complex multi-step queries.

Inspired by Perplexity Computer architecture: a single "brain" model holds
the full query context, decomposes it into a dependency DAG, routes each
subtask to the right sub-agent, runs them in parallel where possible,
handles failures with the Recovery Engine, and synthesizes the final answer.

Integration point:
  agent.py → stream_response_api() checks complexity →
  if complex, delegates to Conductor.orchestrate()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

from core.task_dag import TaskDAG, TaskNode, TaskStatus
from core.workflow_memory import WorkflowMemory

logger = logging.getLogger(__name__)

# ── DAG decomposition prompt ─────────────────────────────────────────────

DAG_DECOMPOSITION_PROMPT = """\
You are a task-decomposition engine for Quasar, a radio astronomy research assistant.

Given a complex user query, break it into ordered sub-tasks with EXPLICIT DEPENDENCIES.
Tasks with no dependencies can run IN PARALLEL — this is critical for speed.

Rules:
1. Each sub-task must be a SINGLE, concrete action.
2. Assign each task to ONE agent type:
   - "archive": ALMA/VLA search, target resolution, file listing, downloads
   - "literature": NASA ADS paper search, author metrics, BibTeX export
   - "analysis": Spectral line ID, FITS header inspection, CASA scripting, cross-match
   - "viz": Plotting, sky maps, Jupyter notebook generation
   - "web": Web search, page navigation
   - "synthesis": Final answer assembly, comparison tables (runs LAST, reads all results)
3. If tasks are independent, they should NOT depend on each other (enables parallel execution).
4. Return at most 10 sub-tasks.
5. If the query is simple enough to answer directly, return an empty subtasks list.

Prior context (from conversation):
{context}

User query: "{query}"

Respond with ONLY valid JSON:
{{
    "subtasks": [
        {{"id": "t1", "description": "...", "depends_on": [], "agent_type": "archive"}},
        {{"id": "t2", "description": "...", "depends_on": ["t1"], "agent_type": "analysis"}},
        ...
    ],
    "reasoning": "Brief explanation of the decomposition"
}}
"""

CONDUCTOR_SYNTHESIS_PROMPT = """\
You are a senior astronomer synthesizing results from multiple sub-agents.

The user asked: "{query}"

Below are all the sub-agent results (each one tackled a piece of the question):

{results}

Instructions:
- Combine into a single, coherent, expert-level answer.
- Cite specific values (frequencies, beam sizes, RMS, observation counts) from the results.
- If any step failed, note what data is missing and suggest alternatives.
- Use markdown formatting: tables for comparisons, bullet points for lists.
- Be concise but thorough — this is the final answer the user sees.
"""


class Conductor:
    """
    Central orchestrator — decomposes complex queries into a DAG,
    routes subtasks to sub-agents, runs them in parallel, and
    synthesizes the final answer.

    For queries that aren't complex enough, returns None so the caller
    falls back to the standard tool-calling loop.
    """

    MAX_SUBTASKS = 10
    COMPLEXITY_THRESHOLD = 0.55

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o",
        tool_executor: Optional[Callable] = None,
        model_router: Optional[Any] = None,
        recovery_engine: Optional[Any] = None,
        on_status: Optional[Callable[[str, str], None]] = None,
        verbose: bool = False,
    ):
        self.client = client
        self.model = model
        self.tool_executor = tool_executor
        self.model_router = model_router
        self.recovery = recovery_engine
        self.on_status = on_status
        self.verbose = verbose
        self.workflow_memory = WorkflowMemory()
        self.dag = TaskDAG()

    async def orchestrate(
        self,
        query: str,
        context: str = "",
        on_status: Optional[Callable[[str, str], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Optional[str]:
        """
        Main entry point for complex query orchestration.

        Returns
        -------
        str or None
            The final synthesized answer, or None if the query isn't
            complex enough for DAG orchestration.
        """
        status_fn = on_status or self.on_status
        self.workflow_memory.clear()

        # Step 1: Decompose query into DAG
        if status_fn:
            status_fn("Analyzing query complexity and building execution plan", "running")

        subtasks = self._decompose(query, context)

        if not subtasks:
            if status_fn:
                status_fn("Query is simple — using direct response", "completed")
            return None  # Caller falls back to standard path

        # Build the DAG
        self.dag.build_from_subtasks(subtasks)

        if self.verbose or status_fn:
            task_summary = ", ".join(
                f"{s['id']}:{s['agent_type']}" for s in subtasks
            )
            msg = f"Execution plan: {len(subtasks)} tasks ({task_summary})"
            logger.info(msg)
            if status_fn:
                status_fn(msg, "completed")

        # Step 2: Execute DAG
        if status_fn:
            status_fn(f"Executing {len(subtasks)} sub-tasks (parallel where possible)", "running")

        results = await self.dag.execute(
            executor_fn=self._execute_node,
            on_status=status_fn,
        )

        # Step 3: Write all results to workflow memory
        for task_id, result in results.items():
            node = self.dag.nodes.get(task_id)
            agent_type = node.agent_type if node else "unknown"
            self.workflow_memory.write(agent_type, task_id, result)

        # Step 4: Synthesize final answer
        if status_fn:
            status_fn("Synthesizing final answer from all results", "running")

        final_answer = self._synthesize(query, results)

        if status_fn:
            status_fn("Answer ready", "completed")

        if on_token:
            on_token(final_answer)

        return final_answer

    # ── Internals ──────────────────────────────────────────────────────────

    def _decompose(self, query: str, context: str) -> List[dict]:
        """Decompose a query into structured subtasks with dependencies."""
        prompt = DAG_DECOMPOSITION_PROMPT.format(
            query=query,
            context=context or "(no prior context)",
        )

        try:
            resp = self.client.responses.create(
                model=self.model,
                input=prompt,
                temperature=0.1,
                max_output_tokens=800,
                text={"format": {"type": "json_object"}},
            )
            data = json.loads(resp.output_text)
            subtasks = data.get("subtasks", [])

            if self.verbose:
                reasoning = data.get("reasoning", "")
                logger.info("Decomposition: %s", reasoning)

            return subtasks[:self.MAX_SUBTASKS]

        except Exception as e:
            logger.error("Conductor decomposition failed: %s", e)
            return []

    async def _execute_node(self, node: TaskNode) -> Any:
        """
        Execute a single task node.

        Uses the tool_executor callback which ultimately calls the same
        tool functions as the main agent — so all 27+ tools are available.
        """
        if self.verbose:
            logger.info("Executing %s (%s): %s", node.id, node.agent_type, node.description)

        # Gather context from completed dependencies
        dep_context = ""
        for dep_id in node.depends_on:
            dep_node = self.dag.nodes.get(dep_id)
            if dep_node and dep_node.result:
                result_str = json.dumps(dep_node.result, default=str)[:3000]
                dep_context += f"\n[Result from {dep_id}]: {result_str}"

        # Build the task description with dependency context
        full_task = node.description
        if dep_context:
            full_task += f"\n\nContext from prior steps:{dep_context}"

        # Use the tool executor (wired to agent's tool-calling loop)
        if self.tool_executor:
            try:
                result = self.tool_executor(full_task, dep_context)
                if result:
                    return result
            except Exception as e:
                logger.error("Tool executor failed for %s: %s", node.id, e)
                # If we have a recovery engine, try recovery
                if self.recovery:
                    try:
                        return await self.recovery.execute_with_recovery(
                            node, self.tool_executor
                        )
                    except Exception as re:
                        logger.error("Recovery also failed for %s: %s", node.id, re)

        # Fallback: direct LLM answer
        return self._direct_answer(full_task, dep_context)

    def _direct_answer(self, task: str, context: str) -> str:
        """Answer a task directly via LLM (fallback when no tool applies)."""
        user_input = f"Context:\n{context}\n\nTask: {task}" if context else task
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=user_input,
                instructions=(
                    "You are an expert radio astronomer. Answer concisely "
                    "using the provided context."
                ),
                temperature=0.3,
                max_output_tokens=500,
            )
            return resp.output_text.strip()
        except Exception as e:
            return f"[Error: {e}]"

    def _synthesize(self, query: str, results: Dict[str, Any]) -> str:
        """Synthesize all sub-task results into a final coherent answer."""
        # Build results text
        results_parts = []
        for task_id, result in results.items():
            node = self.dag.nodes.get(task_id)
            desc = node.description if node else task_id
            result_str = json.dumps(result, default=str)[:2000]
            results_parts.append(f"### {task_id}: {desc}\n{result_str}")

        # Include failed tasks
        for node in self.dag.nodes.values():
            if node.status == TaskStatus.FAILED:
                results_parts.append(
                    f"### {node.id}: {node.description}\n⚠️ FAILED: {node.error}"
                )

        results_text = "\n\n".join(results_parts)

        prompt = CONDUCTOR_SYNTHESIS_PROMPT.format(
            query=query,
            results=results_text,
        )

        try:
            resp = self.client.responses.create(
                model=self.model,
                input=prompt,
                temperature=0.3,
                max_output_tokens=2000,
            )
            return resp.output_text.strip()
        except Exception as e:
            # Fallback: return raw results
            return (
                f"**Results for:** {query}\n\n"
                + results_text
                + f"\n\n(Synthesis failed: {e})"
            )

    def get_execution_summary(self) -> Dict[str, Any]:
        """Return DAG execution metrics for observability."""
        return self.dag.get_execution_summary()
