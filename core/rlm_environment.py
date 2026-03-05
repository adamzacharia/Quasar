# core/rlm_environment.py
"""
RLM REPL Environment

Implements the key RLM insight: treat the prompt/context as a variable in an
external Python REPL environment, NOT as tokens in the LLM context window.

The LLM writes code to interact with the context (slice, search, filter)
rather than reading the entire context directly.  This enables processing
documents far beyond the model's context window.

Reference: "Recursive Language Models" (MIT CSAIL, 2025)
"""

from __future__ import annotations

import io
import re
import sys
import json
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI


# ---------------------------------------------------------------------------
# Safe execution sandbox
# ---------------------------------------------------------------------------

# Modules that are allowed inside the REPL
_ALLOWED_MODULES = {
    # --- Python stdlib ---
    "re", "math", "json", "collections", "itertools", "functools",
    "statistics", "textwrap", "string", "datetime", "operator",
    # --- Scientific computing (astronomy essentials) ---
    "numpy", "np",
    "scipy", "scipy.constants", "scipy.interpolate", "scipy.optimize",
    "scipy.signal", "scipy.fft",
    "pandas", "pd",
    # --- Astronomy-specific ---
    "astropy", "astropy.units", "astropy.constants", "astropy.coordinates",
    "astropy.cosmology", "astropy.io", "astropy.io.fits",
}

# Builtins that are blocked
_BLOCKED_BUILTINS = {
    "exec", "eval", "compile", "__import__", "globals", "locals",
    "exit", "quit", "breakpoint", "input", "help",
}

# Patterns that are outright rejected before execution
_DANGEROUS_PATTERNS = [
    r"\bimport\s+(?:os|sys|subprocess|shutil|socket|requests|urllib|pathlib)\b",
    r"\bopen\s*\(",
    r"\b__\w+__\b",        # dunder access
    r"\.system\s*\(",
    r"\.popen\s*\(",
    r"\brmtree\b",
]


def _is_safe(code: str) -> Tuple[bool, str]:
    """Check whether *code* looks safe to execute."""
    for pattern in _DANGEROUS_PATTERNS:
        match = re.search(pattern, code)
        if match:
            return False, f"Blocked: dangerous pattern '{match.group()}'"
    return True, ""


def _make_safe_builtins() -> dict:
    """Return a restricted builtins dict."""
    import builtins
    safe = {k: v for k, v in vars(builtins).items() if k not in _BLOCKED_BUILTINS}
    # Allow safe imports only
    def _safe_import(name, *args, **kwargs):
        if name not in _ALLOWED_MODULES:
            raise ImportError(f"Module '{name}' is not allowed in the RLM REPL")
        return __import__(name, *args, **kwargs)
    safe["__import__"] = _safe_import
    return safe


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Result of one code execution step."""
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    success: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# RLM Environment
# ---------------------------------------------------------------------------

class RLMEnvironment:
    """
    Sandboxed Python REPL that holds a large context as a variable.

    The LLM generates code to interact with the context.  The environment
    persists its namespace across multiple execution steps, so the LLM can
    build up intermediate variables, filter the context, and accumulate
    results over the course of an RLM trajectory.
    """

    MAX_OUTPUT_CHARS = 4_000  # Truncate stdout to avoid flooding the LLM

    def __init__(self, context: str):
        self.context = context
        self.lines = context.splitlines()
        self.trajectory: List[Dict[str, Any]] = []  # Full execution trace

        # Persistent namespace shared across steps
        self.namespace: Dict[str, Any] = {
            "__builtins__": _make_safe_builtins(),
            # --- Context helpers ---
            "context": context,
            "context_lines": self.lines,
            "num_lines": len(self.lines),
            "num_chars": len(context),
            # --- Utility functions ---
            "search": self._search,
            "slice_lines": self._slice_lines,
            "slice_chars": self._slice_chars,
            "count_matches": self._count_matches,
            "head": lambda n=20: "\n".join(self.lines[:n]),
            "tail": lambda n=20: "\n".join(self.lines[-n:]),
            # --- Results accumulator ---
            "results": [],
        }

    # --- Public API ---------------------------------------------------------

    def execute(self, code: str) -> ExecutionResult:
        """
        Execute *code* in the sandboxed namespace.

        Returns an ExecutionResult with stdout, possible return value, and
        any error information.
        """
        safe, reason = _is_safe(code)
        if not safe:
            result = ExecutionResult(success=False, error=reason)
            self._record_step(code, result)
            return result

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                # Try exec first; if the code is a single expression, eval it
                try:
                    exec(code, self.namespace)
                    return_val = self.namespace.get("_", None)
                except SyntaxError:
                    return_val = eval(code, self.namespace)

            stdout_text = stdout_buf.getvalue()
            if len(stdout_text) > self.MAX_OUTPUT_CHARS:
                stdout_text = (
                    stdout_text[: self.MAX_OUTPUT_CHARS]
                    + f"\n... [truncated to {self.MAX_OUTPUT_CHARS} chars]"
                )

            result = ExecutionResult(
                stdout=stdout_text,
                stderr=stderr_buf.getvalue(),
                return_value=return_val,
                success=True,
            )
        except Exception as e:
            result = ExecutionResult(
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
                success=False,
                error=f"{type(e).__name__}: {e}",
            )

        self._record_step(code, result)
        return result

    def get_state_summary(self) -> str:
        """Return a compact summary of the current environment state."""
        user_vars = {
            k: _summarize_value(v)
            for k, v in self.namespace.items()
            if not k.startswith("_") and k not in (
                "context", "context_lines", "num_lines", "num_chars",
                "search", "slice_lines", "slice_chars", "count_matches",
                "head", "tail", "results",
            )
        }
        return (
            f"Context: {self.namespace['num_chars']:,} chars, "
            f"{self.namespace['num_lines']:,} lines\n"
            f"Results collected: {len(self.namespace['results'])}\n"
            f"User variables: {json.dumps(user_vars, default=str)}\n"
            f"Steps executed: {len(self.trajectory)}"
        )

    # --- Built-in helper functions exposed to the REPL ----------------------

    def _search(self, pattern: str, flags: int = 0) -> List[Tuple[int, str]]:
        """Search context lines by regex. Returns [(line_num, line), ...]."""
        compiled = re.compile(pattern, flags)
        matches = []
        for i, line in enumerate(self.lines):
            if compiled.search(line):
                matches.append((i + 1, line))
        return matches

    def _slice_lines(self, start: int, end: int) -> str:
        """Return lines[start:end] (1-indexed, inclusive)."""
        return "\n".join(self.lines[max(0, start - 1): end])

    def _slice_chars(self, start: int, end: int) -> str:
        """Return context[start:end]."""
        return self.context[start:end]

    def _count_matches(self, pattern: str) -> int:
        """Count regex matches across all lines."""
        return sum(1 for line in self.lines if re.search(pattern, line))

    # --- Internal -----------------------------------------------------------

    def _record_step(self, code: str, result: ExecutionResult):
        self.trajectory.append({
            "code": code,
            "stdout": result.stdout,
            "error": result.error,
            "success": result.success,
        })


# ---------------------------------------------------------------------------
# RLM Executor  (REPL-based)
# ---------------------------------------------------------------------------

RLM_REPL_SYSTEM_PROMPT = """\
You are an RLM (Recursive Language Model) agent operating inside a Python REPL \
environment.

The user's input context has been loaded as:
- `context`       — the full text as a string ({num_chars:,} chars)
- `context_lines` — list of lines ({num_lines:,} lines)
- `num_lines`, `num_chars` — dimensions

You also have these helper functions:
- search(pattern)           → [(line_num, line), ...]  regex search
- slice_lines(start, end)   → str  (1-indexed, inclusive)
- slice_chars(start, end)   → str
- count_matches(pattern)    → int
- head(n=20) / tail(n=20)   → first/last n lines
- sub_query(prompt)          → str  (make a recursive LLM call)

Store your intermediate findings in the `results` list.
When ready, call `answer(your_final_answer_string)`.

⚠️  Do NOT try to print the entire context. It is too large.
Instead, use code to filter and extract relevant sections.

Always explain your plan BEFORE writing code blocks.
Separate code from explanation with ```python ... ``` markers.\
"""

# Sentinel used to detect when the LLM calls answer()
_ANSWER_SENTINEL = object()


class RLMREPLExecutor:
    """
    Runs the full RLM REPL loop:
      1. Show the LLM the environment state
      2. LLM generates a plan + code
      3. Execute code, capture output
      4. Feed output back to LLM
      5. Repeat until answer() is called or max iterations hit
    """

    DEFAULT_MAX_ITERATIONS = 12

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o",
        sub_model: str = "gpt-4o-mini",
        verbose: bool = False,
    ):
        self.client = client
        self.model = model
        self.sub_model = sub_model
        self.verbose = verbose

    def run(
        self,
        query: str,
        context: str,
        max_iterations: int | None = None,
        status_callback: Callable[[str, str], None] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """
        Execute the full RLM loop for *query* over *context*.
        Returns the final answer string.
        """
        max_iter = max_iterations or self.DEFAULT_MAX_ITERATIONS

        # Set up environment
        env = RLMEnvironment(context)
        final_answer: List[str] = []  # mutable container for the answer() call

        # Inject answer() and sub_query() into namespace
        def _answer(result: str):
            final_answer.append(str(result))

        def _sub_query(prompt: str) -> str:
            return self._sub_query_llm(prompt)

        env.namespace["answer"] = _answer
        env.namespace["sub_query"] = _sub_query

        # Build initial system prompt (used as instructions for Responses API)
        system = RLM_REPL_SYSTEM_PROMPT.format(
            num_chars=env.namespace["num_chars"],
            num_lines=env.namespace["num_lines"],
        )

        # Initial user input for the first turn
        current_input = f"Question: {query}\n\nEnvironment state:\n{env.get_state_summary()}"
        last_response_id = None  # Track conversation continuity

        for iteration in range(max_iter):
            if self.verbose:
                print(f"  [REPL iter {iteration+1}/{max_iter}]")
                
            if status_callback:
                status_callback(f"REPL Step {iteration+1}: Analyzing context", "running")

            # Ask LLM for next action using Responses API
            resp = self.client.responses.create(
                model=self.model,
                input=current_input,
                instructions=system,
                previous_response_id=last_response_id,
                temperature=0.2,
                max_output_tokens=1500,
            )
            last_response_id = resp.id
            assistant_msg = resp.output_text or ""
            
            if status_callback:
                status_callback(f"REPL Step {iteration+1}: Analyzing context", "completed")

            # Extract code blocks
            code_blocks = re.findall(r"```python\s*\n(.*?)```", assistant_msg, re.DOTALL)

            if not code_blocks:
                # No code — maybe the LLM answered directly
                if final_answer:
                    if on_token:
                        on_token(final_answer[0])
                    return final_answer[0]
                # Treat the whole message as the answer if no code
                if iteration > 0:
                    if on_token:
                        on_token(assistant_msg)
                    return assistant_msg
                # First iteration with no code — ask for code
                current_input = "Please write Python code to work with the context. Use ```python ... ``` code blocks."
                continue

            # Execute each code block
            if status_callback:
                status_callback(f"REPL Step {iteration+1}: Executing Python code", "running")
            all_outputs = []
            for code in code_blocks:
                result = env.execute(code.strip())
                output = ""
                if result.stdout:
                    output += f"stdout:\n{result.stdout}\n"
                if result.error:
                    output += f"error: {result.error}\n"
                if result.return_value is not None:
                    output += f"return: {result.return_value}\n"
                if not output:
                    output = "(no output)\n"
                all_outputs.append(output)

            if status_callback:
                status_callback(f"REPL Step {iteration+1}: Found execution results", "completed")

            # Check if answer was called
            if final_answer:
                if on_token:
                    on_token(final_answer[0])
                return final_answer[0]

            # Feed results back as the next user input
            current_input = (
                "Execution results:\n"
                + "\n---\n".join(all_outputs)
                + f"\n\nEnvironment state:\n{env.get_state_summary()}"
            )

        # Max iterations reached — ask for final answer
        current_input = "Max iterations reached. Please call answer() with your best answer now."
        resp = self.client.responses.create(
            model=self.model,
            input=current_input,
            instructions=system,
            previous_response_id=last_response_id,
            temperature=0.2,
            max_output_tokens=800,
        )
        final_msg = resp.output_text or ""

        # Try to extract answer
        ans = final_answer[0] if final_answer else final_msg
        if on_token:
            on_token(ans)
        return ans

    def _sub_query_llm(self, prompt: str) -> str:
        """Make a recursive LLM sub-call using the cheaper sub-model."""
        try:
            resp = self.client.responses.create(
                model=self.sub_model,
                input=prompt,
                instructions="You are a helpful research assistant. Be concise.",
                temperature=0.2,
                max_output_tokens=500,
            )
            return resp.output_text.strip()
        except Exception as e:
            return f"[Sub-query error: {e}]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarize_value(v: Any) -> str:
    """Return a compact string representation of a value."""
    if isinstance(v, str):
        return f"str({len(v)} chars)" if len(v) > 100 else repr(v)
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}({len(v)} items)"
    if isinstance(v, dict):
        return f"dict({len(v)} keys)"
    if callable(v):
        return "<function>"
    return repr(v)[:80]
