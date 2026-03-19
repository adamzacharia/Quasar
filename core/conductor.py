# core/conductor.py
"""
Conductor — Central reasoning engine for complex multi-step queries.

Inspired by Perplexity Computer architecture: a single "brain" model holds
the full query context, decomposes it into a dependency DAG, routes each
subtask to the right sub-agent, runs them in parallel where possible,
handles failures with the Recovery Engine, and synthesizes the final answer.

SSE Protocol:
  The Conductor emits structured events for the Perplexity-style task UI:
    - task_group:  parallel batch header
    - task_update: individual task progress
    - task_list:   full checklist view

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

# ── Agent type → Icon + Display name mapping ────────────────────────────
AGENT_ICONS = {
    "archive":    ("🗄️", "Archive Search"),
    "literature": ("📚", "Literature Review"),
    "analysis":   ("🔬", "Data Analysis"),
    "viz":        ("📊", "Visualization"),
    "web":        ("🌐", "Web Search"),
    "synthesis":  ("📋", "Result Synthesis"),
    "general":    ("⚡", "General Task"),
}

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

    Emits Perplexity Computer-style SSE events for real-time task UI.
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
        agent_pool: Optional[Any] = None,
        on_status: Optional[Callable[[str, str], None]] = None,
        on_event: Optional[Callable[[dict], None]] = None,
        verbose: bool = False,
    ):
        self.client = client
        self.model = model
        self.tool_executor = tool_executor
        self.model_router = model_router
        self.recovery = recovery_engine
        self.agent_pool = agent_pool
        self.on_status = on_status
        self.on_event = on_event   # Structured SSE event emitter
        self.verbose = verbose
        self.workflow_memory = WorkflowMemory()
        self.dag = TaskDAG()

    # ── SSE Event Helpers ──────────────────────────────────────────────────

    def _emit(self, event: dict, on_event: Optional[Callable] = None):
        """Emit a structured SSE event for the frontend."""
        fn = on_event or self.on_event
        if fn:
            fn(event)

    def _emit_status(self, step: str, state: str,
                     on_status: Optional[Callable] = None):
        fn = on_status or self.on_status
        if fn:
            fn(step, state)

    def _emit_task_list(self, title: str,
                        on_event: Optional[Callable] = None):
        """Emit a task_list event showing DAG progress as a checklist."""
        tasks = []
        for n in self.dag.nodes.values():
            icon, _ = AGENT_ICONS.get(n.agent_type, ("⚡", "Task"))
            tasks.append({
                "id": n.id,
                "description": f"{icon} {n.description}",
                "status": n.status.value,
                "agentType": n.agent_type,
            })
        self._emit({
            "type": "task_list",
            "title": title,
            "tasks": tasks,
        }, on_event)

    def _emit_task_group(self, group_id: str, title: str,
                         ready_nodes: List[TaskNode],
                         on_event: Optional[Callable] = None):
        """Emit a task_group event for parallel execution."""
        self._emit({
            "type": "task_group",
            "groupId": group_id,
            "title": title,
            "taskIds": [n.id for n in ready_nodes],
            "tasks": [
                {
                    "id": n.id,
                    "description": n.description,
                    "agentType": n.agent_type,
                    "icon": AGENT_ICONS.get(n.agent_type, ("⚡",))[0],
                }
                for n in ready_nodes
            ],
        }, on_event)

    def _emit_task_update(self, node: TaskNode, status: str,
                          detail: str = "",
                          group_id: str = "",
                          on_event: Optional[Callable] = None):
        """Emit a task_update event for a single task."""
        icon, agent_label = AGENT_ICONS.get(node.agent_type, ("⚡", "Task"))
        self._emit({
            "type": "task_update",
            "taskId": node.id,
            "groupId": group_id,
            "title": node.description,
            "status": status,
            "agentType": node.agent_type,
            "agentLabel": agent_label,
            "icon": icon,
            "detail": detail,
        }, on_event)

    # ── Main Orchestration ─────────────────────────────────────────────────

    async def orchestrate(
        self,
        query: str,
        context: str = "",
        on_status: Optional[Callable[[str, str], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_event: Optional[Callable[[dict], None]] = None,
    ) -> Optional[str]:
        """
        Main entry point for complex query orchestration.

        Returns the final synthesized answer, or None if the query
        isn't complex enough for DAG orchestration.
        """
        status_fn = on_status or self.on_status
        event_fn = on_event or self.on_event
        self.workflow_memory.clear()

        # Step 1: Decompose query into DAG
        self._emit_status("Analyzing query complexity and building execution plan", "running", status_fn)

        subtasks = self._decompose(query, context)

        if not subtasks:
            self._emit_status("Query is simple — using direct response", "completed", status_fn)
            return None  # Caller falls back to standard path

        # Build the DAG
        self.dag.build_from_subtasks(subtasks)

        task_summary = ", ".join(f"{s['id']}:{s['agent_type']}" for s in subtasks)
        msg = f"Execution plan: {len(subtasks)} tasks ({task_summary})"
        logger.info(msg)
        self._emit_status(msg, "completed", status_fn)

        # Emit task_list showing the full checklist
        self._emit_task_list(f"Multi-step analysis: {query[:60]}...", event_fn)

        # Step 2: Execute DAG with structured SSE events
        self._emit_status(f"Executing {len(subtasks)} sub-tasks (parallel where possible)", "running", status_fn)

        results = await self._execute_dag_with_events(event_fn, status_fn)

        # Step 3: Write all results to workflow memory
        for task_id, result in results.items():
            node = self.dag.nodes.get(task_id)
            agent_type = node.agent_type if node else "unknown"
            self.workflow_memory.write(agent_type, task_id, result)

        # Step 4: Emit final task_list showing all completed
        self._emit_task_list(f"Multi-step analysis: {query[:60]}...", event_fn)

        # Step 5: Synthesize final answer
        self._emit_status("Synthesizing final answer from all results", "running", status_fn)
        self._emit_task_update(
            TaskNode(id="synthesis", description="Synthesizing results", agent_type="synthesis"),
            "running", "Combining results from all agents...", "", event_fn,
        )

        final_answer = self._synthesize(query, results)

        self._emit_status("Answer ready", "completed", status_fn)

        if on_token:
            on_token(final_answer)

        return final_answer

    async def _execute_dag_with_events(
        self,
        on_event: Optional[Callable] = None,
        on_status: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Execute the full DAG, emitting Perplexity-style parallel group events."""
        total = len(self.dag.nodes)
        round_num = 0

        for _round in range(20):
            ready = self.dag.get_ready_tasks()
            if not ready:
                pending = [n for n in self.dag.nodes.values() if n.status == TaskStatus.PENDING]
                if not pending:
                    break
                for p in pending:
                    self.dag.mark_failed(p.id, "Unmet dependencies — predecessor failed")
                    self._emit_task_update(p, "error", "Predecessor task failed", f"g{round_num}", on_event)
                break

            round_num += 1
            group_id = f"g{round_num}"

            # Emit parallel group header
            if len(ready) > 1:
                self._emit_task_group(group_id, "Running tasks in parallel", ready, on_event)
            else:
                self._emit_task_group(group_id, f"Running: {ready[0].description[:50]}", ready, on_event)

            # Emit individual task_update: running
            for node in ready:
                icon, label = AGENT_ICONS.get(node.agent_type, ("⚡", "Task"))
                self._emit_task_update(node, "running", f"Starting {label}...", group_id, on_event)

            # Execute all ready tasks in parallel
            async def _run_one(node: TaskNode):
                self.dag.mark_running(node.id)
                try:
                    result = await asyncio.wait_for(
                        self._execute_node(node),
                        timeout=node.sla_seconds,
                    )
                    self.dag.mark_completed(node.id, result)

                    # Emit success update
                    detail = self._summarize_result(result)
                    self._emit_task_update(node, "completed", detail, group_id, on_event)

                except asyncio.TimeoutError:
                    self.dag.mark_failed(node.id, f"Timeout after {node.sla_seconds}s")
                    self._emit_task_update(node, "error", f"Timed out after {node.sla_seconds}s", group_id, on_event)
                except Exception as e:
                    self.dag.mark_failed(node.id, str(e))
                    self._emit_task_update(node, "error", str(e)[:100], group_id, on_event)

            await asyncio.gather(*[_run_one(node) for node in ready])

            completed = sum(1 for n in self.dag.nodes.values() if n.status == TaskStatus.COMPLETED)
            self._emit_status(f"Completed {completed}/{total} tasks", "completed", on_status)

        # Collect results
        return {
            tid: node.result
            for tid, node in self.dag.nodes.items()
            if node.status == TaskStatus.COMPLETED
        }

    def _summarize_result(self, result: Any) -> str:
        """Create a short human-readable summary of a task result."""
        if result is None:
            return "No results"
        if isinstance(result, str):
            return result[:80] + ("..." if len(result) > 80 else "")
        if isinstance(result, dict):
            if "total_results" in result:
                return f"Found {result['total_results']} results"
            if "error" in result:
                return f"Error: {result['error']}"
            return f"{len(result)} fields returned"
        if isinstance(result, list):
            return f"Found {len(result)} items"
        return str(result)[:80]

    # ── Node Execution ─────────────────────────────────────────────────────

    async def _execute_node(self, node: TaskNode) -> Any:
        """
        Execute a single task node.

        Uses the tool_executor callback which ultimately calls the same
        tool functions as the main agent — so all 28+ tools are available.
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

    def _synthesize(self, query: str, results: Dict[str, Any]) -> str:
        """Synthesize all sub-task results into a final coherent answer."""
        results_parts = []
        for task_id, result in results.items():
            node = self.dag.nodes.get(task_id)
            desc = node.description if node else task_id
            result_str = json.dumps(result, default=str)[:2000]
            results_parts.append(f"### {task_id}: {desc}\n{result_str}")

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
            return (
                f"**Results for:** {query}\n\n"
                + results_text
                + f"\n\n(Synthesis failed: {e})"
            )

    def get_execution_summary(self) -> Dict[str, Any]:
        """Return DAG execution metrics for observability."""
        return self.dag.get_execution_summary()
