"""Benchmark-only tool allowlist (``QUASAR_BENCH_TOOL_ALLOWLIST``).

Unset (the default and every deployment): no effect anywhere.

Set to comma-separated fnmatch patterns, e.g. ``manna__*``, it turns a
Quasar backend into a controlled benchmark arm that exposes ONLY the matching
tools. It exists for the three-arm comparison of Quasar's native archive
tools against the MANNA MCP server (tmp/manna-port-2026-09-24/), where the
"MANNA only" arm must not reach a native archive tool by any route:

* the per-turn tool list sent to the model is filtered;
* execution refuses a tool outside the list (main loop, Conductor, sandbox
  bridge), and the unknown-tool auto-correct / "did you mean" lists only
  offer allowed names;
* the paths that steer toward or force native tools are skipped: one-shot
  routing directives, forced first-round tool_choice, the Conductor, the web
  pre-pass, and the native-tool-heavy system prompt (replaced by a short
  neutral one, below).
"""

from __future__ import annotations

import fnmatch
import os
from typing import Iterable, List, Optional, Tuple

ENV = "QUASAR_BENCH_TOOL_ALLOWLIST"

# Neutral instructions for an allowlisted arm: no tool names, so the prompt
# does not favour any toolset. Same honesty rules the product prompt enforces.
BENCH_SYSTEM_PROMPT = (
    "You are an astronomy research assistant. Answer the user's question using the tools "
    "available to you in this session; do not assume any other tools exist. Query the "
    "archives rather than answering from memory, report the numbers the tools return, say "
    "which archive and query produced each result, and state plainly when data are not "
    "available or a premise is false. Never invent values, identifiers or sources."
)


def patterns() -> Optional[Tuple[str, ...]]:
    """None when unset/blank. A non-blank value that yields no patterns
    (" , ") is an EMPTY allowlist, so it exposes nothing rather than every
    tool (guard CX-05)."""
    raw = os.getenv(ENV, "")
    if not raw.strip():
        return None
    return tuple(p.strip() for p in raw.split(",") if p.strip())


def active() -> bool:
    return patterns() is not None


def allowed(name: str) -> bool:
    pats = patterns()
    if pats is None:
        return True
    return any(fnmatch.fnmatchcase(str(name or ""), p) for p in pats)


def filter_names(names: Iterable[str]) -> List[str]:
    return [n for n in names if allowed(n)]


def refusal(name: str) -> str:
    return (f"Tool '{name}' is not available in this session (benchmark tool allowlist "
            f"{os.getenv(ENV, '')!r}).")


__all__ = ["ENV", "BENCH_SYSTEM_PROMPT", "active", "allowed", "filter_names", "patterns", "refusal"]
