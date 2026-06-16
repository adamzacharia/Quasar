# core/context_manager.py
"""
Context Manager — Smart conversation compaction for long sessions.

Inspired by OpenClaude's autoCompact.ts + microCompact.ts — replaces
the naive session truncation in agent.py with:
  1. Token-aware threshold detection (per-model context windows)
  2. LLM-powered summarization of old turns (preserves key info)
  3. Tool result truncation for old tool outputs
  4. Circuit breaker to prevent compaction loops on failure

Instead of nuking the entire session (old _prune_session_if_needed),
this module summarizes old turns while keeping recent context intact.

Usage:
    ctx_mgr = ContextManager(client, model="gpt-4o")
    messages = ctx_mgr.compact_if_needed(messages)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Per-model context window limits ──────────────────────────────────────

MODEL_CONTEXT_WINDOWS = {
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    # Anthropic
    "claude-sonnet": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-haiku": 200_000,
    # Google
    "gemini-2.0-pro": 2_000_000,
    "gemini-2.0-flash": 1_000_000,
    "gemini-1.5-pro": 2_000_000,
    "gemini-1.5-flash": 1_000_000,
}

# Trigger compaction when context reaches this % of the model's window
COMPACTION_THRESHOLD_RATIO = 0.70

# Keep this many recent messages uncompacted (most relevant context)
PROTECTED_TAIL_MESSAGES = 8

# Circuit breaker — stop trying after this many consecutive failures
MAX_CONSECUTIVE_FAILURES = 3

# Maximum chars for the compact summary
MAX_SUMMARY_CHARS = 3000

# Model used for summarization (cheap + fast)
SUMMARY_MODEL = os.getenv("QUASAR_SUMMARY_MODEL") or os.getenv("QUASAR_FAST_MODEL", "gpt-4o-mini")

# System prompt for the summarization agent
COMPACTION_SYSTEM_PROMPT = """\
You are a conversation summarizer for Quasar, a radio astronomy AI assistant.

Your job is to distill a conversation history into a compact summary that 
preserves ALL essential information:

1. **Astronomical targets** mentioned (names, coordinates, frequencies)
2. **Search results**: what was found, counts, key fields
3. **Tool outputs**: key data from ALMA searches, paper lookups, spectral analyses
4. **User preferences**: what they want to do, format preferences, ongoing goals
5. **Decisions made**: which data to use, filtering choices, analysis results
6. **Errors encountered**: what failed and why (so we don't repeat mistakes)

Rules:
- Be concise but COMPLETE — missing a target name or frequency is unacceptable
- Use structured format: bullet points, not prose
- Include specific numbers (beam sizes, RMS values, observation counts)
- Do NOT include pleasantries or meta-commentary
- Output a single markdown summary block

The summary will replace the old conversation turns, so if you miss something,
the agent will lose that context permanently.
"""


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4


def _estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate total token count across all messages."""
    total = 0
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content", "") or msg.get("output", "")
            if isinstance(content, str):
                total += _estimate_tokens(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        total += _estimate_tokens(
                            item.get("text", "") or item.get("content", "") or ""
                        )
                    elif isinstance(item, str):
                        total += _estimate_tokens(item)
            # Count tool arguments too
            args = msg.get("arguments", "")
            if args:
                total += _estimate_tokens(args)
    return total


def _get_context_window(model: str) -> int:
    """Get context window size for a model, with fuzzy matching."""
    # Exact match
    if model in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model]

    # Prefix match (e.g., "gpt-4o-2024-..." → gpt-4o)
    for key, window in MODEL_CONTEXT_WINDOWS.items():
        if model.startswith(key):
            return window

    # Fallback: assume 128K (conservative for modern models)
    logger.warning("[context] Unknown model '%s', assuming 128K context window", model)
    return 128_000


class ContextManager:
    """
    Manages conversation context to prevent context window overflow.

    Replaces the old _prune_session_if_needed() which nuked the entire session.
    Instead, this summarizes old turns while keeping recent context intact.
    """

    def __init__(
        self,
        client: Any,
        model: str = "gpt-4o",
        on_status: Any = None,
    ):
        self.client = client
        self.model = model
        self.on_status = on_status
        self._consecutive_failures = 0
        self._last_compact_time = 0.0
        self._total_compactions = 0
        self._tokens_saved = 0

    def compact_if_needed(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Check if messages exceed the context threshold and compact if needed.

        Returns the (possibly compacted) message list. If compaction fails
        or isn't needed, returns the original messages unchanged.
        """
        if not messages:
            return messages

        active_model = model or self.model
        context_window = _get_context_window(active_model)
        threshold = int(context_window * COMPACTION_THRESHOLD_RATIO)

        current_tokens = _estimate_messages_tokens(messages)

        if current_tokens < threshold:
            return messages

        # Circuit breaker — don't keep trying if compaction keeps failing
        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logger.warning(
                "[context] Circuit breaker open — %d consecutive compaction failures. "
                "Skipping until next session reset.",
                self._consecutive_failures,
            )
            return messages

        logger.info(
            "[context] Token count %d exceeds threshold %d (%.0f%% of %d). "
            "Starting compaction...",
            current_tokens, threshold,
            (current_tokens / context_window * 100), context_window,
        )

        if self.on_status:
            self.on_status("Compacting conversation context", "running")

        try:
            compacted = self._do_compact(messages, active_model)
            new_tokens = _estimate_messages_tokens(compacted)
            saved = current_tokens - new_tokens

            self._consecutive_failures = 0
            self._total_compactions += 1
            self._tokens_saved += saved
            self._last_compact_time = time.time()

            logger.info(
                "[context] Compaction #%d complete: %d → %d tokens (saved %d, %.0f%%)",
                self._total_compactions, current_tokens, new_tokens,
                saved, (saved / current_tokens * 100) if current_tokens > 0 else 0,
            )

            if self.on_status:
                self.on_status("Compacting conversation context", "completed")

            return compacted

        except Exception as e:
            self._consecutive_failures += 1
            logger.error(
                "[context] Compaction failed (attempt %d/%d): %s",
                self._consecutive_failures, MAX_CONSECUTIVE_FAILURES, e,
            )

            if self.on_status:
                self.on_status("Compacting conversation context", "completed")

            # Return original messages — don't lose anything on failure
            return messages

    def _do_compact(
        self,
        messages: List[Dict[str, Any]],
        model: str,
    ) -> List[Dict[str, Any]]:
        """
        Perform the actual compaction:
        1. Split messages into old (to be summarized) and recent (to keep)
        2. Summarize old messages using a cheap LLM
        3. Return [summary_message] + recent_messages
        """
        if len(messages) <= PROTECTED_TAIL_MESSAGES:
            # Not enough messages to compact — return as-is
            return messages

        # Split: keep the last N messages as-is
        old_messages = messages[:-PROTECTED_TAIL_MESSAGES]
        recent_messages = messages[-PROTECTED_TAIL_MESSAGES:]

        # Build text representation of old messages for summarization
        old_text = self._messages_to_text(old_messages)

        if not old_text.strip():
            return messages

        # Summarize using a fast/cheap model
        summary = self._generate_summary(old_text, model)

        if not summary or len(summary.strip()) < 50:
            raise ValueError("Compaction produced empty or too-short summary")

        # Build the compacted message list
        summary_message = {
            "role": "user",
            "content": (
                f"[CONTEXT SUMMARY — compacted from {len(old_messages)} earlier messages]\n\n"
                f"{summary}\n\n"
                f"[END OF SUMMARY — the conversation continues below with full detail]"
            ),
        }

        return [summary_message] + list(recent_messages)

    def _messages_to_text(self, messages: List[Dict[str, Any]]) -> str:
        """Convert messages to a text representation for summarization."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "") or msg.get("output", "")

            if isinstance(content, list):
                # Handle structured content (tool results, etc.)
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        text_parts.append(
                            item.get("text", "") or item.get("content", "") or ""
                        )
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = "\n".join(text_parts)

            if isinstance(content, str) and content.strip():
                # Truncate very long individual messages to avoid blowing summary context
                if len(content) > 5000:
                    content = content[:4500] + f"\n[... truncated {len(content) - 4500} chars ...]"
                parts.append(f"[{role}]: {content}")

            # Handle tool call results
            if msg.get("type") == "function_call_output":
                output = msg.get("output", "")
                if isinstance(output, str) and len(output) > 3000:
                    output = output[:2500] + f"\n[... truncated {len(output) - 2500} chars ...]"
                parts.append(f"[tool_result]: {output}")

        return "\n\n".join(parts)

    def _generate_summary(self, conversation_text: str, model: str) -> str:
        """Use LLM to generate a compact summary of old conversation turns."""
        # Use the cheap model for summarization
        summary_input = (
            f"Summarize the following Quasar conversation history. "
            f"Preserve ALL astronomical data, search results, targets, "
            f"frequencies, and user decisions:\n\n"
            f"---\n{conversation_text}\n---"
        )

        try:
            resp = self.client.responses.create(
                model=SUMMARY_MODEL,
                input=summary_input,
                instructions=COMPACTION_SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=1500,
            )
            summary = resp.output_text.strip()

            # Enforce maximum length
            if len(summary) > MAX_SUMMARY_CHARS:
                summary = summary[:MAX_SUMMARY_CHARS] + "\n[... summary truncated ...]"

            return summary

        except Exception as e:
            logger.error("[context] Summary generation failed: %s", e)
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Return compaction statistics for observability."""
        return {
            "total_compactions": self._total_compactions,
            "tokens_saved": self._tokens_saved,
            "consecutive_failures": self._consecutive_failures,
            "last_compact_time": self._last_compact_time,
        }
