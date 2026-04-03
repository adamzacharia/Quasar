# core/tool_budget.py
"""
Tool Result Budget — Truncate oversized tool outputs before sending to the API.

Inspired by OpenClaude's toolResultStorage.ts — prevents large ALMA search
results, paper lists, or FITS headers from consuming the entire context window.

Strategy:
  - Most recent tool results are preserved at full size
  - Older tool results are truncated to stay within budget
  - A marker is added so the LLM knows data was truncated

Usage:
    messages = apply_tool_result_budget(messages, max_chars_per_result=50_000)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Default per-result character limit (50KB — leaves room for ~10 results in 128K context)
DEFAULT_MAX_CHARS_PER_RESULT = 50_000

# Results from these tools are never truncated (small by nature)
EXEMPT_TOOLS = {
    "web_search",
    "get_current_time",
    "check_observation_status",
}


def _estimate_result_chars(result: Any) -> int:
    """Estimate character count of a tool result."""
    if isinstance(result, str):
        return len(result)
    try:
        return len(json.dumps(result, default=str))
    except (TypeError, ValueError):
        return len(str(result))


def truncate_result(result_str: str, max_chars: int) -> str:
    """Truncate a single tool result string with a marker."""
    if len(result_str) <= max_chars:
        return result_str

    # Keep first 80% and last 10% of the budget
    head_size = int(max_chars * 0.80)
    tail_size = int(max_chars * 0.10)

    truncated = result_str[:head_size]
    truncated += f"\n\n[... TRUNCATED — showing {head_size:,} of {len(result_str):,} chars. "
    truncated += f"Last {tail_size:,} chars below ...]\n\n"
    truncated += result_str[-tail_size:]

    return truncated


def apply_tool_result_budget(
    tool_results: List[Dict[str, Any]],
    max_chars_per_result: int = DEFAULT_MAX_CHARS_PER_RESULT,
    protect_last_n: int = 2,
) -> List[Dict[str, Any]]:
    """
    Enforce a per-result character budget on tool outputs.

    Parameters
    ----------
    tool_results : list
        List of tool result dicts with format:
        [{"type": "function_call_output", "call_id": "...", "output": "..."}]
    max_chars_per_result : int
        Maximum characters per tool result.
    protect_last_n : int
        Number of most recent results to protect from truncation.

    Returns
    -------
    list
        Tool results with oversized entries truncated.
    """
    if not tool_results:
        return tool_results

    total_before = 0
    total_after = 0
    truncated_count = 0

    processed = []
    n = len(tool_results)

    for i, result in enumerate(tool_results):
        # Shallow copy to avoid mutating the original
        result = dict(result)

        output = result.get("output", "")
        if not isinstance(output, str):
            output = json.dumps(output, default=str)

        total_before += len(output)

        # Protect the last N results (most recent, most relevant)
        is_protected = (n - i) <= protect_last_n

        if not is_protected and len(output) > max_chars_per_result:
            output = truncate_result(output, max_chars_per_result)
            truncated_count += 1
            logger.info(
                "[tool_budget] Truncated result %d/%d from %s to %s chars",
                i + 1, n,
                f"{len(result.get('output', '')):,}",
                f"{len(output):,}",
            )

        result["output"] = output
        total_after += len(output)
        processed.append(result)

    if truncated_count > 0:
        savings = total_before - total_after
        logger.info(
            "[tool_budget] Truncated %d/%d results, saved %s chars (%.0f%%)",
            truncated_count, n,
            f"{savings:,}",
            (savings / total_before * 100) if total_before > 0 else 0,
        )

    return processed


def apply_message_tool_budget(
    messages: List[Any],
    max_chars_per_result: int = DEFAULT_MAX_CHARS_PER_RESULT,
) -> List[Any]:
    """
    Scan a list of conversation messages and truncate tool_result content
    in older messages. Preserves the most recent tool results at full size.

    Works with both Responses API format (function_call_output) and
    Chat Completions format (tool role messages).
    """
    if not messages:
        return messages

    # Find all tool result indices
    tool_indices = []
    for i, msg in enumerate(messages):
        if isinstance(msg, dict):
            # Responses API format
            if msg.get("type") == "function_call_output":
                tool_indices.append(i)
            # Chat Completions format
            elif msg.get("role") == "tool":
                tool_indices.append(i)

    if not tool_indices:
        return messages

    # Protect the last 3 tool results
    protect_set = set(tool_indices[-3:])

    total_saved = 0
    result = list(messages)

    for idx in tool_indices:
        if idx in protect_set:
            continue

        msg = dict(result[idx])
        output = msg.get("output") or msg.get("content") or ""
        if not isinstance(output, str):
            output = json.dumps(output, default=str)

        if len(output) > max_chars_per_result:
            original_len = len(output)
            output = truncate_result(output, max_chars_per_result)
            total_saved += original_len - len(output)

            # Write back to the appropriate key
            if "output" in msg:
                msg["output"] = output
            else:
                msg["content"] = output
            result[idx] = msg

    if total_saved > 0:
        logger.info("[tool_budget] Message-level truncation saved %s chars", f"{total_saved:,}")

    return result
