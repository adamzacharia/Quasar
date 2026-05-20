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
import queue as stdlib_queue
import time
from typing import Any, Callable, Dict, List, Optional, Union

from openai import OpenAI

from core.task_dag import TaskDAG, TaskNode, TaskStatus
from core.workflow_memory import WorkflowMemory
from core.dag_cache import DAGCache
from core.result_cache import ResultCache
from core.observability import estimate_cost
from core.langfuse_integration import get_langfuse, langfuse_trace, langfuse_generation
from services.notebook_gen import generate_conductor_notebook

logger = logging.getLogger(__name__)

# ── Rollbar helpers (graceful no-op if Rollbar is not initialised) ────────────
try:
    import rollbar as _rollbar
    _ROLLBAR_AVAILABLE = True
except ImportError:
    _ROLLBAR_AVAILABLE = False

# ── Agent type → Display name mapping ───────────────────────────────────
# Icons are rendered as custom SVG components on the frontend.
AGENT_ICONS = {
    "archive":    ("archive",    "Archive Search"),
    "literature": ("literature", "Literature Review"),
    "analysis":   ("analysis",   "Data Analysis"),
    "viz":        ("viz",        "Visualization"),
    "web":        ("web",        "Web Search"),
    "compute":    ("compute",    "Python Computation"),
    "synthesis":  ("synthesis",  "Result Synthesis"),
    "general":    ("general",    "General Task"),
}

# ── DAG decomposition prompt ─────────────────────────────────────────────

DAG_DECOMPOSITION_PROMPT = """\
You are a task-decomposition engine for Quasar, a radio astronomy research assistant.

Given a complex user query, break it into ordered sub-tasks with EXPLICIT DEPENDENCIES.

## Available Tools (ONLY use these — do NOT invent tools or capabilities)

**Archive search tools (agent_type: "archive")**:
- search_by_target: Search ALMA archive by target name. This is the ONLY radio archive we have.
- search_by_position: Search ALMA archive by RA/Dec coordinates.
- search_cadc_archive: Search CADC for JWST, HST, JCMT, Gemini data. Use ONLY when user asks for non-ALMA data.
- resolve_target: Resolve target name to RA/Dec via SIMBAD.

**Analysis tools (agent_type: "analysis")**:
- check_co_lines: Check CO/13CO/C18O line coverage in the LAST search results. Requires a prior search_by_target call.
- check_line_coverage: Check if a specific frequency falls in LAST search results. Requires a prior search_by_target call.
- search_lines_by_molecule: Search Splatalogue for spectral lines by molecule name.
- filter_results: Apply numeric filters to the LAST search results.

**Literature tools (agent_type: "literature")**:
- search_papers: Search NASA ADS for papers. ONLY use when user asks for papers/publications.

**Web tools (agent_type: "web")**:
- web_search: Web search for real-time info, news, schedules. NEVER for papers — use search_papers instead.

**CRITICAL**: We do NOT have VLA, VLBA, or GBT archive search. Do NOT create tasks to search VLA or any non-ALMA radio archive.

## Key Principles

1. **PARALLELISM IS YOUR SUPERPOWER**: Tasks with no dependencies MUST be independent.
   - Independent archive searches → run in parallel
   - Literature + archive searches → run in parallel
   - Only synthesis depends on all prior results

2. **SEQUENTIAL WHEN REQUIRED**: If tool B needs the results from tool A (e.g. check_co_lines needs search_by_target results), they MUST be sequential with explicit depends_on.

3. Each sub-task must be a SINGLE, concrete, actionable step — not a vague directive.
   BAD:  "Analyze the data"
   GOOD: "Search ALMA archive for NGC 1068 Band 6 observations with resolution < 0.5 arcsec"

4. Assign each task to ONE agent type:
   - "archive": ALMA archive searches (search_by_target, search_by_position, search_cadc_archive)
   - "literature": NASA ADS paper search (ONLY when user asks for papers)
   - "analysis": Line coverage checks, spectral line ID, filtering (check_co_lines, check_line_coverage, filter_results)
   - "compute": Python calculations (frequency conversions, sensitivity estimates, unit conversions, data filtering with numpy/scipy/astropy). Use when math or data processing is needed.
   - "web": Web search, real-time info
   - "synthesis": Final answer assembly (runs LAST)

5. Return at most 10 sub-tasks. If the query needs fewer, use fewer.
6. If the query is simple enough to answer directly, return an empty subtasks list.

## Common Workflows

**Line coverage check** (e.g. "Check CO(2-1) coverage for M87"):
  t1 (archive): search_by_target for M87 → t2 (analysis): check_co_lines or check_line_coverage → t3 (synthesis)
  That's 3 tasks maximum. Do NOT search VLA. Do NOT add extra analysis.

**Multi-target search** (e.g. "ALMA data on M87 and NGC 1068"):
  t1 (archive): search_by_target("M87, NGC 1068") → t2 (synthesis)
  Both targets in ONE call, not separate tasks.

## Anti-Patterns (NEVER do these)
- NEVER search VLA, VLBA, or GBT — we don't have those archives.
- Never create a task that says "Based on your findings" — each task gets
  explicit dependency context injected automatically.
- Never create "verify" or "check" tasks unless the user explicitly asked.
- Never chain archive searches sequentially if they are for DIFFERENT targets.
- Never search papers unless the user explicitly asked for papers/articles/publications.

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
    "reasoning": "Brief explanation of the decomposition and parallelism strategy"
}}
"""

DAG_REPLAN_PROMPT = """\
You are a task-decomposition engine for Quasar, a radio astronomy research assistant.

The user has reviewed your execution plan and is requesting changes.
Revise the plan based on their feedback while respecting all the same tool constraints
as the original decomposition.

=== ORIGINAL QUERY ===
{query}

=== CURRENT PLAN ===
{current_plan}

=== USER FEEDBACK ===
{feedback}

Revise the plan above based on the user's feedback. You may add, remove, reorder,
or modify sub-tasks. Keep the same JSON format. Ensure dependencies remain valid
(no circular deps, every depends_on ID must exist). Maximum {max_subtasks} sub-tasks.

Available agent types: "archive", "literature", "analysis", "compute", "web", "synthesis".
Available tools: search_by_target, search_by_position, search_cadc_archive, resolve_target,
check_co_lines, check_line_coverage, search_lines_by_molecule, filter_results,
search_papers, web_search.

Respond with ONLY valid JSON:
{{
    "subtasks": [
        {{"id": "t1", "description": "...", "depends_on": [], "agent_type": "archive"}},
        ...
    ],
    "reasoning": "Brief explanation of what changed and why"
}}
"""

CONDUCTOR_SYNTHESIS_PROMPT = """\
You are a senior astronomer synthesizing results from multiple parallel sub-agents.

The user asked: "{query}"

Below are all the sub-agent results (each tackled a piece of the question):

{results}

## Synthesis Instructions

1. **Combine** all results into a single, coherent, expert-level answer.
2. **Cite specific values** from the results: frequencies, beam sizes, RMS, observation counts,
   paper titles, and author names. Never invent or hallucinate data.
3. **Use data-driven comparisons**: If multiple targets/papers were queried, present a
   comparison table with actual values from the results.
4. **Handle failures HONESTLY**: If a sub-task failed, say EXACTLY what failed and why
   (e.g. "FITS download timed out" or "CADC returned 0 imaging products").
   Do NOT dress up failures as "recommendations" or "next steps for the user."
5. **Format for the web UI**:
   - Use ## headings for major sections
   - Use Markdown tables for comparisons (ALWAYS use pipe syntax)
   - Bold key findings and observatory names
   - Bullet points for lists
6. **Be concise but thorough** — this is the final answer the user sees.
   Don't repeat raw tool output; synthesize it into insight.
7. **If images were rendered**: If a result mentions "image_path" or "success: True",
   that image is ALREADY displayed inline in the chat above your text. Reference it
   naturally (e.g. "As shown in the ALMA+JWST overlay above..."). Do NOT tell the
   user to download data or follow steps. The visualization is DONE.

## CRITICAL ANTI-PATTERNS — NEVER DO THESE:

8. NEVER write "Recommendations for Data Access" or "Steps to Overlay".
9. NEVER tell the user to "re-run the search", "provide the URL", or "identify the product".
   The sub-agents had all the tools. If they failed, explain the error.
10. NEVER write generic advice like "filter by band" or "consider filtering" unless the
    user explicitly asked for advice. They asked for RESULTS.
11. NEVER write "Current Blockers" — if something blocked, say what error occurred.
12. If a sub-agent returned an error string, quote that exact error.
"""



class Conductor:
    """
    Central orchestrator — decomposes complex queries into a DAG,
    routes subtasks to sub-agents, runs them in parallel, and
    synthesizes the final answer.

    Emits Perplexity Computer-style SSE events for real-time task UI.
    """

    MAX_SUBTASKS = 10  # absolute maximum (expert tier fallback)
    COMPLEXITY_THRESHOLD = 0.7  # minimum score to trigger Conductor at all

    # Tiered complexity: maps score ranges to subtask budgets.
    # This prevents over-decomposition of moderate queries (which hurts speed
    # without helping accuracy) while preserving full DAG power for hard queries.
    COMPLEXITY_TIERS = {
        # (min_score, max_score): (tier_name, max_subtasks, description)
        (0.70, 0.80): ("moderate", 3, "Focused analysis with minimal decomposition"),
        (0.80, 0.90): ("complex",  6, "Multi-step analysis with parallel execution"),
        (0.90, 1.01): ("expert",  10, "Full DAG orchestration for deeply complex queries"),
    }

    @classmethod
    def classify_tier(cls, score: float) -> tuple:
        """Map a complexity score to a (tier_name, max_subtasks) tuple.

        Returns ('moderate', 3), ('complex', 6), or ('expert', 10).
        Falls back to ('expert', 10) if score doesn't match any tier.
        """
        for (lo, hi), (name, max_sub, _desc) in cls.COMPLEXITY_TIERS.items():
            if lo <= score < hi:
                return (name, max_sub)
        return ("expert", cls.MAX_SUBTASKS)

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o",
        conductor_model: Optional[str] = None,
        synthesis_model: Optional[str] = None,
        tool_executor: Optional[Callable] = None,
        model_router: Optional[Any] = None,
        recovery_engine: Optional[Any] = None,
        agent_pool: Optional[Any] = None,
        sandbox_executor: Optional[Any] = None,
        on_status: Optional[Callable[[str, str], None]] = None,
        on_event: Optional[Callable[[dict], None]] = None,
        verbose: bool = False,
    ):
        self.client = client
        self.model = model
        # Planning model: expensive, only used for DAG decomposition
        self.conductor_model = conductor_model or model
        # Synthesis model: cheaper, used for combining results into final answer
        self.synthesis_model = synthesis_model or "gpt-5.4-mini"
        self.tool_executor = tool_executor
        self.model_router = model_router
        self.recovery = recovery_engine
        self.agent_pool = agent_pool
        self.sandbox_executor = sandbox_executor
        self.on_status = on_status
        self.on_event = on_event
        self.verbose = verbose
        self.workflow_memory = WorkflowMemory()
        self.dag = TaskDAG()
        # (#14) Cross-session DAG learning
        self.dag_cache = DAGCache(cache_path="data/dag_cache.json")
        # (#11) Result deduplication cache
        self.result_cache = ResultCache(ttl_seconds=300)

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
        max_subtasks: Optional[int] = None,
        complexity_tier: Optional[str] = None,
        on_status: Optional[Callable[[str, str], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_event: Optional[Callable[[dict], None]] = None,
        plan_feedback_queue: Optional[stdlib_queue.Queue] = None,
    ) -> Optional[str]:
        """
        Main entry point for complex query orchestration.

        Returns the final synthesized answer, or None if the query
        isn't complex enough for DAG orchestration.
        """
        status_fn = on_status or self.on_status
        event_fn = on_event or self.on_event
        self.workflow_memory.clear()

        # Step 1: Decompose query into DAG (with cache lookup #14)
        self._emit_status("Analyzing query complexity and building execution plan", "running", status_fn)

        # (#14) Check DAG cache for a similar previous decomposition
        cached_plan = self.dag_cache.find_similar(query)
        if cached_plan:
            subtasks = cached_plan["subtasks"]
            logger.info("DAG cache hit — reusing decomposition with %d tasks", len(subtasks))
            self._emit_status(
                f"Reusing proven execution plan ({len(subtasks)} tasks)",
                "completed", status_fn,
            )
        else:
            subtasks = self._decompose(query, context, max_subtasks=max_subtasks or self.MAX_SUBTASKS)

        # Apply max_subtasks cap from tier (if provided)
        effective_max = max_subtasks or self.MAX_SUBTASKS
        if len(subtasks) > effective_max:
            logger.info(
                "Tier '%s' caps subtasks at %d — trimming %d → %d",
                complexity_tier or "default", effective_max, len(subtasks), effective_max,
            )
            subtasks = subtasks[:effective_max]

        if not subtasks:
            self._emit_status("Query is simple — using direct response", "completed", status_fn)
            return None  # Caller falls back to standard path

        # Build the DAG
        self.dag.build_from_subtasks(subtasks)

        # Log validation warnings (#3)
        warnings = self.dag.get_validation_warnings()
        if warnings:
            for w in warnings:
                logger.warning("DAG validation: %s", w)

        task_summary = ", ".join(f"{s['id']}:{s['agent_type']}" for s in subtasks)
        msg = f"Execution plan: {len(subtasks)} tasks ({task_summary})"
        logger.info(msg)
        self._emit_status(msg, "completed", status_fn)

        # ── Rollbar: attach trace context so any crash in this orchestration
        # is labelled with trace_id and query info ────────────────────────────
        _trace_id = f"conductor-{id(self.dag)}"
        if _ROLLBAR_AVAILABLE:
            try:
                _rollbar.report_message(
                    f"[conductor] Orchestration started: {query[:100]}",
                    level="debug",
                    extra_data={
                        "trace_id": _trace_id,
                        "subtask_count": len(subtasks),
                        "complexity_tier": complexity_tier or "unknown",
                        "task_ids": [s["id"] for s in subtasks],
                    },
                )
            except Exception:
                pass

        # ── Langfuse: open a trace that spans the full orchestration ─────
        lf_client = get_langfuse()
        lf_trace = None
        if lf_client:
            try:
                lf_trace = lf_client.trace(
                    name=f"conductor: {query[:80]}",
                    user_id="anonymous",
                    metadata={
                        "complexity_tier": complexity_tier or "unknown",
                        "subtask_count": len(subtasks),
                        "task_summary": task_summary,
                    },
                    tags=["conductor", complexity_tier or "unknown"],
                )
            except Exception as e:
                logger.debug("[Langfuse] trace creation failed: %s", e)
        # Store on self so _execute_dag_with_events can create child spans
        self._lf_trace = lf_trace

        # (#12) Human-in-the-loop: emit plan for review and await approval
        effective_max = max_subtasks or self.MAX_SUBTASKS
        MAX_REPLAN_ITERATIONS = 3

        if plan_feedback_queue is not None:
            for iteration in range(MAX_REPLAN_ITERATIONS + 1):
                # Emit plan_review event with full subtask details
                plan_dict = self.dag.to_plan_dict()
                self._emit({
                    "type": "plan_review",
                    "title": f"Execution Plan: {len(subtasks)} tasks, {plan_dict.get('estimated_parallel_rounds', '?')} rounds",
                    "subtasks": [
                        {
                            "id": s["id"],
                            "description": s["description"],
                            "agentType": s.get("agent_type", "general"),
                            "dependsOn": s.get("depends_on", []),
                        }
                        for s in subtasks
                    ],
                    "reasoning": "",
                    "query": query[:200],
                    "iteration": iteration + 1,
                    "maxIterations": MAX_REPLAN_ITERATIONS,
                }, event_fn)

                self._emit_status(
                    "Waiting for plan approval...",
                    "running", status_fn,
                )

                # Block until user responds via /api/plan-feedback
                # Uses stdlib queue.Queue (thread-safe) bridged to async via run_in_executor
                try:
                    loop = asyncio.get_event_loop()
                    feedback = await loop.run_in_executor(None, plan_feedback_queue.get)
                except Exception as e:
                    logger.warning("Plan feedback queue error: %s — auto-approving", e)
                    break

                if feedback.get("approve", False):
                    logger.info("Plan approved by user (iteration %d)", iteration + 1)
                    self._emit_status("Plan approved — executing", "completed", status_fn)
                    break

                # User provided feedback — re-plan
                user_feedback = feedback.get("feedback", "")
                if not user_feedback.strip():
                    logger.info("Empty feedback — treating as approval")
                    self._emit_status("Plan approved — executing", "completed", status_fn)
                    break

                if iteration >= MAX_REPLAN_ITERATIONS:
                    logger.info("Max re-plan iterations reached — executing last plan")
                    self._emit_status("Max revisions reached — executing current plan", "completed", status_fn)
                    break

                logger.info("Re-planning with user feedback (iteration %d): %s", iteration + 1, user_feedback[:100])
                self._emit_status(
                    f"Revising plan based on your feedback (revision {iteration + 1})...",
                    "running", status_fn,
                )

                subtasks = self._replan(
                    query, subtasks, user_feedback,
                    max_subtasks=effective_max,
                )
                if not subtasks:
                    self._emit_status("Re-planning failed — executing original plan", "completed", status_fn)
                    break

                # Rebuild DAG with revised subtasks
                self.dag = TaskDAG()
                self.dag.build_from_subtasks(subtasks)

                task_summary = ", ".join(f"{s['id']}:{s.get('agent_type', '?')}" for s in subtasks)
                msg = f"Revised plan: {len(subtasks)} tasks ({task_summary})"
                logger.info(msg)
                self._emit_status(msg, "completed", status_fn)
                # Loop back to emit updated plan_review

        # Emit task_list showing the full checklist
        self._emit_task_list(f"Multi-step analysis: {query[:60]}...", event_fn)

        # Step 2: Execute DAG with structured SSE events
        self._emit_status(f"Executing {len(subtasks)} sub-tasks (parallel where possible)", "running", status_fn)

        results = await self._execute_dag_with_events(event_fn, status_fn)

        # (#7) Write results to workflow memory (already done per-task in _run_one,
        # but do a final pass to ensure all are captured)
        for task_id, result in results.items():
            node = self.dag.nodes.get(task_id)
            agent_type = node.agent_type if node else "unknown"
            self.workflow_memory.write(agent_type, task_id, result, task_id=task_id)

        # Step 4: Emit final task_list showing all completed
        self._emit_task_list(f"Multi-step analysis: {query[:60]}...", event_fn)

        # Step 5: Synthesize final answer (#4 — streaming)
        self._emit_status("Synthesizing final answer from all results", "running", status_fn)
        self._emit_task_update(
            TaskNode(id="synthesis", description="Synthesizing results", agent_type="synthesis"),
            "running", "Combining results from all agents...", "", event_fn,
        )

        final_answer = self._synthesize(query, results, on_token=on_token)

        self._emit_status("Answer ready", "completed", status_fn)

        # ── Langfuse: close the orchestration trace ──────────────────────
        if lf_trace:
            try:
                lf_trace.update(metadata={
                    "status": "completed",
                    "subtask_count": len(subtasks),
                    "result_length": len(final_answer) if final_answer else 0,
                })
            except Exception:
                pass

        # ── Rollbar: no teardown needed (context is per-request in rollbar) ─

        # (#14) Store successful decomposition for future reuse
        dag_summary = self.dag.get_execution_summary()
        self.dag_cache.store(query, subtasks, execution_summary=dag_summary)

        # Step 6: Generate companion Jupyter notebook
        try:
            notebook_dict = generate_conductor_notebook(
                query=query,
                subtasks=subtasks,
                results=results,
                dag_summary=dag_summary,
            )
            self._notebook = notebook_dict
        except Exception as nb_err:
            logger.warning("Notebook generation failed: %s", nb_err)
            self._notebook = None

        # Don't re-emit via on_token — streaming synthesis already did that
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
                # Propagate actual predecessor errors instead of a generic message
                for p in pending:
                    failed_deps = [
                        (dep_id, self.dag.nodes[dep_id].error or "unknown error")
                        for dep_id in p.depends_on
                        if dep_id in self.dag.nodes
                        and self.dag.nodes[dep_id].status == TaskStatus.FAILED
                    ]
                    if failed_deps:
                        cause = "; ".join(f"{did}: {err[:100]}" for did, err in failed_deps)
                        self.dag.mark_failed(p.id, f"Predecessor failed — {cause}")
                    else:
                        self.dag.mark_failed(p.id, "Unmet dependencies — predecessor did not complete")
                    self._emit_task_update(p, "error", f"Blocked: predecessor failed", f"g{round_num}", on_event)
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

                # ── Langfuse: create a child span for this DAG node ──
                lf_span = None
                if getattr(self, '_lf_trace', None):
                    try:
                        lf_span = self._lf_trace.span(
                            name=f"{node.id} ({node.agent_type})",
                            metadata={
                                "agent_type": node.agent_type,
                                "description": node.description[:200],
                                "depends_on": node.depends_on,
                            },
                        )
                        # Set as thread-local parent so LLM calls inside
                        # the tool executor are parented to this span
                        from core.llm_client import set_langfuse_parent
                        set_langfuse_parent(lf_span)
                    except Exception:
                        pass

                # ── Rollbar: log breadcrumb for this task start ──
                if _ROLLBAR_AVAILABLE:
                    try:
                        _rollbar.report_message(
                            f"[conductor] Starting {node.id} ({node.agent_type}): {node.description[:80]}",
                            level="debug",
                            extra_data={"trace_id": _trace_id},
                        )
                    except Exception:
                        pass

                try:
                    # Build dependency context summary for the RecoveryEngine.
                    # This lets _replan() and _reassign() see what predecessors
                    # returned, enabling root-cause analysis.
                    dep_context = None
                    if node.depends_on:
                        dep_parts = []
                        for dep_id in node.depends_on:
                            dep_node = self.dag.nodes.get(dep_id)
                            if dep_node:
                                if dep_node.status == TaskStatus.COMPLETED:
                                    summary = self._summarize_result(dep_node.result)
                                    dep_parts.append(f"[{dep_id}] ({dep_node.description[:60]}): {summary}")
                                elif dep_node.status == TaskStatus.FAILED:
                                    dep_parts.append(f"[{dep_id}] ({dep_node.description[:60]}): FAILED — {dep_node.error}")
                        if dep_parts:
                            dep_context = "\n".join(dep_parts)

                    # (#2) Use RecoveryEngine for resilient execution
                    if self.recovery:
                        result = await self.recovery.execute_with_recovery(
                            node,
                            lambda desc, ctx="", model="": self.tool_executor(desc, ctx, model) if self.tool_executor else self._direct_answer(desc, ctx),
                            on_status=on_status,
                            dep_context=dep_context,
                        )
                    else:
                        result = await asyncio.wait_for(
                            self._execute_node(node),
                            timeout=node.sla_seconds,
                        )

                    # (#7) Write result to WorkflowMemory immediately
                    self.workflow_memory.write(
                        node.agent_type, node.id, result, task_id=node.id
                    )

                    # Soft-failure detection for UI display
                    is_soft_failure = False
                    if isinstance(result, dict):
                        if result.get("success") is False or "error" in result:
                            is_soft_failure = True
                    elif isinstance(result, str) and "[Tool execution error:" in result:
                        is_soft_failure = True

                    if is_soft_failure:
                        detail = self._summarize_result(result)
                        self.dag.mark_completed(node.id, result)
                        self._emit_task_update(node, "error", detail, group_id, on_event)
                        # ── Langfuse: end span with soft-failure marker ──
                        if lf_span:
                            try:
                                lf_span.end(output=detail, metadata={"soft_failure": True})
                            except Exception:
                                pass
                    else:
                        self.dag.mark_completed(node.id, result)
                        detail = self._summarize_result(result)
                        self._emit_task_update(node, "completed", detail, group_id, on_event)
                        # ── Langfuse: end span with success ──
                        if lf_span:
                            try:
                                lf_span.end(output=detail)
                            except Exception:
                                pass

                except asyncio.TimeoutError:
                    self.dag.mark_failed(node.id, f"Timeout after {node.sla_seconds}s")
                    self._emit_task_update(node, "error", f"Timed out after {node.sla_seconds}s", group_id, on_event)
                    # ── Langfuse: record timeout on span ──
                    if lf_span:
                        try:
                            lf_span.end(output=f"TIMEOUT after {node.sla_seconds}s")
                        except Exception:
                            pass
                except Exception as e:
                    self.dag.mark_failed(node.id, str(e))
                    self._emit_task_update(node, "error", str(e)[:100], group_id, on_event)
                    # ── Rollbar: report actual failure with task context ──
                    if _ROLLBAR_AVAILABLE:
                        try:
                            _rollbar.report_exc_info(
                                extra_data={
                                    "task_id": node.id,
                                    "agent_type": node.agent_type,
                                    "description": node.description[:100],
                                    "trace_id": _trace_id,
                                }
                            )
                        except Exception:
                            pass
                    # ── Langfuse: record error on span ──
                    if lf_span:
                        try:
                            lf_span.end(output=f"ERROR: {e}")
                        except Exception:
                            pass
                finally:
                    # Clear thread-local Langfuse parent
                    try:
                        from core.llm_client import set_langfuse_parent
                        set_langfuse_parent(None)
                    except Exception:
                        pass

            await asyncio.gather(*[_run_one(node) for node in ready])

            completed = sum(1 for n in self.dag.nodes.values() if n.status == TaskStatus.COMPLETED)
            errored = sum(
                1 for n in self.dag.nodes.values()
                if n.status == TaskStatus.COMPLETED and isinstance(n.result, dict)
                and (n.result.get("success") is False or "error" in n.result)
            )
            true_ok = completed - errored
            failed_hard = sum(1 for n in self.dag.nodes.values() if n.status == TaskStatus.FAILED)
            self._emit_status(
                f"Completed {true_ok}/{total} tasks"
                + (f" ({errored + failed_hard} errors)" if errored + failed_hard else ""),
                "completed", on_status,
            )

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

        Model routing: each subtask is routed to the optimal OpenAI model
        via ModelRouter (GPT-5.4 for synthesis/reasoning, GPT-4.1 for
        archive/analysis, GPT-4.1-mini for simple tasks).
        """
        # ── Route to the optimal model for this subtask ──────────────────
        if self.model_router:
            subtask_model = self.model_router.route(node.description, node.agent_type)
            routing_info = self.model_router.get_routing_info(node.description, node.agent_type)
            logger.info(
                "Routed %s (%s) → %s (%s)",
                node.id, node.agent_type,
                subtask_model, routing_info.get("reason", ""),
            )
        else:
            subtask_model = self.conductor_model  # fallback: use conductor_model for everything

        if self.verbose:
            logger.info("Executing %s (%s) on %s: %s", node.id, node.agent_type, subtask_model, node.description)

        # (#6) Track which model is used on this node
        node.model_used = subtask_model

        # (#7) Gather context from WorkflowMemory (structured) instead of raw JSON
        dep_context = self.workflow_memory.get_dependency_context(
            node.depends_on, max_chars=8000
        )
        # Fallback: if WorkflowMemory is empty, use raw node results
        if not dep_context:
            for dep_id in node.depends_on:
                dep_node = self.dag.nodes.get(dep_id)
                if dep_node and dep_node.result:
                    result_str = json.dumps(dep_node.result, default=str)[:6000]
                    dep_context += f"\n[Result from {dep_id}]: {result_str}"

        # Build the task description with dependency context
        full_task = node.description
        if dep_context:
            full_task += f"\n\nContext from prior steps:{dep_context}"

        # ── Route "compute" tasks to the sandbox executor ────────────────
        if node.agent_type == "compute" and self.sandbox_executor:
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self.sandbox_executor.run,
                    full_task,  # query
                    dep_context or "",  # context
                )
                if result:
                    return result
            except Exception as e:
                logger.error("Sandbox executor failed for %s: %s", node.id, e)
                # Fall through to tool_executor as backup

        # Use the tool executor (wired to agent's tool-calling loop)
        # IMPORTANT: tool_executor is SYNCHRONOUS (OpenAI API + HTTP archive
        # calls).  Running it directly would block this event loop and make
        # asyncio.gather() tasks execute sequentially instead of in parallel.
        # Wrapping in run_in_executor() puts each subtask in a real thread.
        if self.tool_executor:
            try:
                loop = asyncio.get_event_loop()
                # Pass subtask_model as 3rd arg so the executor uses the
                # routed model instead of the default conductor_model.
                result = await loop.run_in_executor(
                    None, self.tool_executor, full_task, dep_context, subtask_model
                )
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

    def _decompose(self, query: str, context: str, max_subtasks: Optional[int] = None) -> List[dict]:
        """Decompose a query into structured subtasks with dependencies."""
        effective_max = max_subtasks or self.MAX_SUBTASKS
        prompt = DAG_DECOMPOSITION_PROMPT.format(
            query=query,
            context=context or "(no prior context)",
        )
        # Inject subtask budget guidance into the prompt so the LLM
        # knows how many subtasks to generate for this complexity tier.
        if effective_max < self.MAX_SUBTASKS:
            prompt += (
                f"\n\nIMPORTANT: This query has moderate complexity. "
                f"Use at most {effective_max} sub-tasks. Prefer fewer, focused tasks "
                f"over many granular ones. Combine related operations into single tasks."
            )

        try:
            resp = self.client.responses.create(
                model=self.conductor_model,
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

    def _replan(
        self,
        query: str,
        current_subtasks: List[dict],
        feedback: str,
        max_subtasks: Optional[int] = None,
    ) -> List[dict]:
        """Re-decompose a query based on user feedback about the current plan.

        Uses a clearly labeled prompt with three sections:
        - ORIGINAL QUERY: what the user asked
        - CURRENT PLAN: the existing subtask list
        - USER FEEDBACK: what the user wants changed
        """
        effective_max = max_subtasks or self.MAX_SUBTASKS

        # Format current plan as readable JSON for the LLM
        plan_json = json.dumps(current_subtasks, indent=2)

        prompt = DAG_REPLAN_PROMPT.format(
            query=query,
            current_plan=plan_json,
            feedback=feedback,
            max_subtasks=effective_max,
        )

        try:
            resp = self.client.responses.create(
                model=self.conductor_model,
                input=prompt,
                temperature=0.1,
                max_output_tokens=800,
                text={"format": {"type": "json_object"}},
            )
            data = json.loads(resp.output_text)
            subtasks = data.get("subtasks", [])

            reasoning = data.get("reasoning", "")
            logger.info("Re-plan reasoning: %s", reasoning)

            return subtasks[:effective_max]

        except Exception as e:
            logger.error("Conductor re-planning failed: %s", e)
            return current_subtasks  # Fall back to current plan on failure

    def _synthesize(
        self, query: str, results: Dict[str, Any],
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Synthesize all sub-task results into a final coherent answer.

        (#4) Streams tokens via on_token for instant UX — the user sees
        the synthesis appear word-by-word instead of waiting 3-5s.
        """
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
            # (#4) Streaming synthesis -- uses cheaper model (not the planning model)
            resp = self.client.responses.create(
                model=self.synthesis_model,
                input=prompt,
                temperature=0.3,
                max_output_tokens=2000,
                stream=True,
            )
            answer = ""
            for event in resp:
                if event.type == "response.output_text.delta":
                    answer += event.delta
                    if on_token:
                        on_token(event.delta)

            if not answer.strip():
                logger.warning("Synthesis model returned empty — falling back to raw results")
                fallback = (
                    f"**Results for:** {query}\n\n"
                    + results_text
                    + "\n\n*(Synthesis returned empty — showing raw sub-agent results)*"
                )
                if on_token:
                    on_token(fallback)
                return fallback
            return answer
        except Exception as e:
            fallback = (
                f"**Results for:** {query}\n\n"
                + results_text
                + f"\n\n(Synthesis failed: {e})"
            )
            if on_token:
                on_token(fallback)
            return fallback

    def get_execution_summary(self) -> Dict[str, Any]:
        """Return DAG execution metrics for observability."""
        return self.dag.get_execution_summary()
