# core/rlm.py
"""
Recursive Language Model (RLM) Module

Implements the RLM paradigm from "Transforming Long Prompts into Short Programs"
(MIT CSAIL). Complex queries are decomposed into sub-tasks that are recursively
solved via LLM calls and then aggregated into a final answer.

This is tailored for Quasar's radio astronomy domain: multi-hop questions like
"Do any M87 observations cover the CO(2-1) line?" are decomposed into:
  1. Look up rest frequency of CO(2-1)
  2. Look up redshift of M87
  3. Calculate observed frequency
  4. Search archive for matching observations
  5. Aggregate and answer
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from core.rlm_environment import RLMREPLExecutor


# ---------------------------------------------------------------------------
# Complexity Detector
# ---------------------------------------------------------------------------

# Keywords / patterns that suggest multi-step reasoning is needed
_MULTI_HOP_INDICATORS = [
    "cover",
    "overlap",
    "redshift",
    "observed frequency",
    "rest frequency",
    "line coverage",
    "compare",
    "correlate",
    "combine",
    "both",
    "relationship between",
    "across multiple",
    "step by step",
]


@dataclass
class ComplexityResult:
    """Output of the complexity detector."""
    is_complex: bool
    score: float  # 0.0 (trivial) – 1.0 (very complex)
    reasoning: str = ""
    suggested_subtasks: List[str] = field(default_factory=list)


class ComplexityDetector:
    """
    Determines whether a query requires recursive decomposition.

    Uses a two-stage approach:
      1. Fast heuristic check (keyword scan) — cheap, no LLM call.
      2. LLM-based assessment (only triggered when heuristic is ambiguous).
    """

    THRESHOLD = 0.55  # queries scoring above this are treated as complex

    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    # --- public API ---------------------------------------------------------

    def assess(self, query: str) -> ComplexityResult:
        """Return a ComplexityResult for *query*."""
        # Stage 1: fast heuristic
        heuristic_score = self._heuristic_score(query)
        if heuristic_score >= 0.8:
            return ComplexityResult(
                is_complex=True,
                score=heuristic_score,
                reasoning="Strong multi-hop indicators detected",
            )
        if heuristic_score <= 0.2:
            return ComplexityResult(
                is_complex=False,
                score=heuristic_score,
                reasoning="Simple single-step query",
            )

        # Stage 2: LLM probe
        return self._llm_assess(query, heuristic_score)

    # --- internals ----------------------------------------------------------

    @staticmethod
    def _heuristic_score(query: str) -> float:
        q = query.lower()
        hits = sum(1 for kw in _MULTI_HOP_INDICATORS if kw in q)
        # Normalize: 3+ hits → 1.0
        return min(hits / 3.0, 1.0)

    def _llm_assess(self, query: str, heuristic_score: float) -> ComplexityResult:
        prompt = (
            "You are a complexity analyst for an astronomy research assistant.\n"
            "Given the user's query, decide whether it requires MULTIPLE sequential\n"
            "reasoning steps (e.g. look up a value, then compute something, then search).\n\n"
            "Respond with JSON only:\n"
            '{"is_complex": bool, "score": float 0-1, "reasoning": "...", '
            '"subtasks": ["task1", "task2", ...]}\n\n'
            f"User query: \"{query}\"\n"
        )
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=prompt,
                temperature=0,
                max_output_tokens=300,
                text={"format": {"type": "json_object"}},
            )
            data = json.loads(resp.output_text)
            score = float(data.get("score", heuristic_score))
            return ComplexityResult(
                is_complex=score >= self.THRESHOLD,
                score=score,
                reasoning=data.get("reasoning", ""),
                suggested_subtasks=data.get("subtasks", []),
            )
        except Exception:
            # Fallback to heuristic
            return ComplexityResult(
                is_complex=heuristic_score >= self.THRESHOLD,
                score=heuristic_score,
                reasoning="LLM assessment failed; using heuristic",
            )


# ---------------------------------------------------------------------------
# Sub-task execution
# ---------------------------------------------------------------------------

@dataclass
class SubTaskResult:
    """Result of executing one sub-task."""
    task: str
    answer: str
    success: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Recursive Language Model
# ---------------------------------------------------------------------------

class RecursiveLanguageModel:
    """
    Core RLM engine.  Given a complex query it:
      1. Decomposes it into ordered sub-tasks (via LLM).
      2. Executes each sub-task sequentially, feeding prior context forward.
      3. Aggregates sub-results into a coherent final answer.

    The *tool_executor* callback lets the agent inject its own search /
    computation tools so the RLM can call archive searches, frequency lookups,
    etc.
    """

    MAX_SUBTASKS = 8
    MAX_RECURSION_DEPTH = 3

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o",
        tool_executor=None,
        verbose: bool = False,
        use_repl: bool = True,
    ):
        self.client = client
        self.model = model
        self.tool_executor = tool_executor  # callback(subtask_str) -> str
        self.verbose = verbose
        self.detector = ComplexityDetector(client, model="gpt-4o-mini")
        self.use_repl = use_repl
        # tool_executor will be set after agent initialises (see QuasarAgent._register_rlm_tools)
        self.tool_executor = None
        self.repl_executor = RLMREPLExecutor(
            client=client,
            model=model,
            sub_model="gpt-4o-mini",
            verbose=verbose,
            tool_executor=None,  # wired up later via set_tool_executor()
        )

    # --- public API ---------------------------------------------------------

    def should_use_rlm(self, query: str) -> Tuple[bool, ComplexityResult]:
        """Check whether *query* warrants RLM decomposition."""
        result = self.detector.assess(query)
        return result.is_complex, result

    def execute(
        self,
        query: str,
        context: str = "",
        depth: int = 0,
        use_repl: bool | None = None,
        status_callback: Callable[[str, str], None] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """
        Recursively execute *query*.

        If *use_repl* is True (or self.use_repl), the REPL environment is used
        to offload the context and process it via code execution — the core
        RLM insight.  Otherwise falls back to LLM-only decomposition.

        Returns a final aggregated answer string.
        """
        repl_mode = use_repl if use_repl is not None else self.use_repl

        # Use REPL mode when context is genuinely massive (>80k chars ≈ 20k tokens).
        # This is the paper's intended use case — processing datasets far beyond the
        # LLM's context window, not for normal multi-step queries.
        REPL_CONTEXT_THRESHOLD = 80_000
        if repl_mode and context and len(context) > REPL_CONTEXT_THRESHOLD:
            if self.verbose:
                print(f"[RLM] Using REPL mode for {len(context):,} char context (threshold: {REPL_CONTEXT_THRESHOLD:,})")
            if status_callback:
                status_callback("Initializing Python REPL environment", "running")
                status_callback("Initializing Python REPL environment", "completed")
            return self.repl_executor.run(query, context, status_callback=status_callback, on_token=on_token)

        # Fallback: LLM-only decomposition (rarely used — main path is stream_response_api)
        return self._execute_decompose(query, context, depth, status_callback=status_callback, on_token=on_token)

    def _execute_decompose(self, query: str, context: str, depth: int = 0, status_callback=None, on_token=None) -> str:
        """LLM-only decomposition fallback (original RLM approach)."""
        if depth >= self.MAX_RECURSION_DEPTH:
            return self._direct_answer(query, context)

        # Step 1: Decompose
        subtasks = self._decompose(query, context)

        if not subtasks or len(subtasks) <= 1:
            # Not worth decomposing — answer directly
            if status_callback:
                status_callback("Decomposing query into subtasks", "completed")
                status_callback("Answering directly without subtasks", "running")
            
            ans = self._direct_answer(query, context)
            
            if status_callback:
                status_callback("Answering directly without subtasks", "completed")
            if on_token:
                on_token(ans)
                
            return ans

        if self.verbose:
            print(f"[RLM depth={depth}] Decomposed into {len(subtasks)} sub-tasks")
            
        if status_callback:
            status_callback("Decomposing query into subtasks", "completed")

        # Step 2: Execute sub-tasks sequentially
        results: List[SubTaskResult] = []
        accumulated_context = context

        for idx, task in enumerate(subtasks):
            if self.verbose:
                print(f"  [{idx+1}/{len(subtasks)}] {task}")
            
            if status_callback:
                status_callback(f"Executing subtask {idx+1}/{len(subtasks)}: {task}", "running")

            result = self._execute_subtask(task, accumulated_context, depth, status_callback=status_callback, on_token=on_token)
            
            if status_callback:
                status_callback(f"Executing subtask {idx+1}/{len(subtasks)}: {task}", "completed")

            results.append(result)

            # Feed result forward as context
            accumulated_context += f"\n\n[Sub-task {idx+1}] {task}\n→ {result.answer}"

        # Step 3: Aggregate
        
        if status_callback:
            status_callback("Aggregating sub-task results", "running")
            
        final_answer = self._aggregate(query, results)
        
        if status_callback:
            status_callback("Aggregating sub-task results", "completed")
            
        # Manually stream the final answer here if requested
        if on_token:
            on_token(final_answer)
            
        return final_answer

    # --- internals ----------------------------------------------------------

    def _decompose(self, query: str, context: str) -> List[str]:
        """Break *query* into ordered sub-tasks via LLM."""
        from core.prompts import RLM_DECOMPOSITION_PROMPT

        prompt = RLM_DECOMPOSITION_PROMPT.format(
            query=query,
            context=context or "(no prior context)",
        )

        try:
            resp = self.client.responses.create(
                model=self.model,
                input=prompt,
                temperature=0.1,
                max_output_tokens=600,
                text={"format": {"type": "json_object"}},
            )
            data = json.loads(resp.output_text)
            subtasks = data.get("subtasks", [])
            return subtasks[: self.MAX_SUBTASKS]
        except Exception as e:
            if self.verbose:
                print(f"[RLM] Decomposition failed: {e}")
            return []

    def _execute_subtask(
        self, task: str, context: str, depth: int, status_callback=None, on_token=None
    ) -> SubTaskResult:
        """Execute a single sub-task.  Uses tool_executor if provided."""

        # Check if the sub-task itself is complex enough to recurse
        complexity = self.detector.assess(task)
        if complexity.is_complex and depth + 1 < self.MAX_RECURSION_DEPTH:
            answer = self.execute(task, context, depth + 1, status_callback=status_callback, on_token=on_token)
            return SubTaskResult(task=task, answer=answer)

        # Try the tool executor first (e.g., archive search, frequency lookup)
        if self.tool_executor:
            try:
                tool_answer = self.tool_executor(task, context)
                if tool_answer:
                    return SubTaskResult(task=task, answer=str(tool_answer))
            except Exception as e:
                if self.verbose:
                    print(f"[RLM] Tool executor error: {e}")

        # Fallback: LLM direct answer
        answer = self._direct_answer(task, context)
        return SubTaskResult(task=task, answer=answer)

    def _direct_answer(self, query: str, context: str) -> str:
        """Answer *query* directly via a single LLM call."""
        user_input = (
            f"Context:\n{context}\n\nQuestion: {query}"
            if context
            else query
        )
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=user_input,
                instructions=(
                    "You are an expert radio astronomer. Answer concisely "
                    "using the provided context. If you don't have enough "
                    "information, say so clearly."
                ),
                temperature=0.3,
                max_output_tokens=500,
            )
            return resp.output_text.strip()
        except Exception as e:
            return f"[Error answering sub-task: {e}]"

    def _aggregate(self, original_query: str, results: List[SubTaskResult]) -> str:
        """Aggregate sub-task results into a final answer."""
        from core.prompts import RLM_AGGREGATION_PROMPT

        sub_results_text = "\n".join(
            f"Step {i+1}: {r.task}\n  → {r.answer}"
            for i, r in enumerate(results)
        )

        prompt = RLM_AGGREGATION_PROMPT.format(
            query=original_query,
            sub_results=sub_results_text,
        )

        try:
            resp = self.client.responses.create(
                model=self.model,
                input=prompt,
                temperature=0.3,
                max_output_tokens=800,
            )
            return resp.output_text.strip()
        except Exception as e:
            # Fallback: concatenate raw results
            return (
                f"**Results for:** {original_query}\n\n"
                + sub_results_text
                + f"\n\n(Aggregation failed: {e})"
            )
