# core/session_memory.py
"""
Session Memory — Background note-taking agent for long conversations.

Inspired by OpenClaude's sessionMemory.ts — maintains a running markdown
summary of the conversation that persists across context compaction.

Unlike mem0 (which stores/retrieves memories), Session Memory:
  - Runs in the background after each agent turn
  - Extracts key facts, decisions, and tool results
  - Injects the summary into the system prompt
  - Uses dual thresholds (tokens + tool calls) to trigger extraction

Usage:
    sm = SessionMemory(client)
    sm.extract_if_needed(messages, tool_call_count=5)
    context = sm.get_context()  # inject into system prompt
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Minimum token count before first extraction
MIN_TOKENS_TO_INIT = 5_000

# Minimum new tokens between extractions
MIN_TOKENS_BETWEEN_UPDATES = 8_000

# Minimum tool calls between extractions
MIN_TOOL_CALLS_BETWEEN_UPDATES = 4

# Model used for extraction (cheap + fast)
EXTRACTION_MODEL = os.getenv("QUASAR_MEMORY_EXTRACTION_MODEL") or os.getenv("QUASAR_FAST_MODEL", "deepseek-v4-flash")

# Maximum size of the running memory file
MAX_MEMORY_CHARS = 4_000

EXTRACTION_PROMPT = """\
You are a background note-taking assistant for Quasar, a radio astronomy AI.

Below is the current running session memory, followed by recent conversation.
Update the session memory by:

1. ADDING any new targets, frequencies, search results, decisions, or errors
2. UPDATING any information that has changed (e.g., refined search results)
3. KEEPING everything that is still relevant
4. REMOVING nothing unless it's clearly obsolete

Format: Use concise bullet points grouped by topic.

Current session memory:
---
{current_memory}
---

Recent conversation to extract from:
---
{recent_conversation}
---

Return the COMPLETE updated session memory (not just the additions).
Keep it under 3000 characters. Use this structure:

## Targets & Objects
- (target names, coordinates, redshifts mentioned)

## Search History
- (what was searched, what was found, result counts)

## Key Data
- (frequencies, beam sizes, RMS values, important numbers)

## Decisions & Goals
- (what the user wants to do, choices made)

## Errors & Issues
- (what failed, workarounds tried)
"""


def _estimate_tokens(text: str) -> int:
    """Rough token estimate."""
    return len(text) // 4


def _messages_to_text(messages: List[Any], last_n: int = 10) -> str:
    """Convert recent messages to text for extraction."""
    parts = []
    for msg in messages[-last_n:]:
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", "") or msg.get("output", "")
            if isinstance(content, list):
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
                # Truncate long messages
                if len(content) > 2000:
                    content = content[:1800] + "..."
                parts.append(f"[{role}]: {content}")
    return "\n\n".join(parts)


class SessionMemory:
    """
    Maintains a running markdown summary of the conversation.

    Runs extraction in a background thread to avoid blocking responses.
    The summary is injected into the system prompt on each turn.
    """

    def __init__(self, client: Any):
        self.client = client
        self._memory: str = ""
        self._initialized = False
        self._last_extraction_tokens = 0
        self._last_extraction_tool_calls = 0
        self._total_tool_calls = 0
        self._extraction_count = 0
        self._lock = threading.Lock()
        self._extracting = False

    def record_tool_calls(self, count: int = 1):
        """Record that tool calls were made (called by the agent after tool execution)."""
        self._total_tool_calls += count

    def extract_if_needed(
        self,
        messages: List[Any],
        on_status: Optional[Any] = None,
    ):
        """
        Check if extraction thresholds are met and run extraction in background.

        Thresholds (both must be met for non-initial extraction):
        1. Token count has grown by MIN_TOKENS_BETWEEN_UPDATES since last extraction
        2. Tool calls have increased by MIN_TOOL_CALLS_BETWEEN_UPDATES
        """
        if self._extracting:
            return  # Don't overlap extractions

        current_tokens = sum(
            _estimate_tokens(
                m.get("content", "") if isinstance(m, dict) and isinstance(m.get("content"), str)
                else str(m.get("content", ""))[:500] if isinstance(m, dict) else ""
            )
            for m in messages
        )

        # First extraction: wait for minimum tokens
        if not self._initialized:
            if current_tokens < MIN_TOKENS_TO_INIT:
                return
            self._initialized = True

        # Subsequent extractions: check both thresholds
        token_growth = current_tokens - self._last_extraction_tokens
        tool_growth = self._total_tool_calls - self._last_extraction_tool_calls

        has_met_token_threshold = token_growth >= MIN_TOKENS_BETWEEN_UPDATES
        has_met_tool_threshold = tool_growth >= MIN_TOOL_CALLS_BETWEEN_UPDATES

        if not (has_met_token_threshold and has_met_tool_threshold):
            return

        # Run extraction in background thread
        self._extracting = True
        self._last_extraction_tokens = current_tokens
        self._last_extraction_tool_calls = self._total_tool_calls

        def _bg_extract():
            try:
                recent_text = _messages_to_text(messages, last_n=12)
                self._run_extraction(recent_text)
                self._extraction_count += 1
                logger.info(
                    "[session_memory] Extraction #%d complete (%d chars)",
                    self._extraction_count, len(self._memory),
                )
            except Exception as e:
                logger.error("[session_memory] Extraction failed: %s", e)
            finally:
                self._extracting = False

        thread = threading.Thread(target=_bg_extract, daemon=True)
        thread.start()

    def _run_extraction(self, recent_conversation: str):
        """Run the LLM extraction to update session memory."""
        prompt = EXTRACTION_PROMPT.format(
            current_memory=self._memory or "(empty — first extraction)",
            recent_conversation=recent_conversation,
        )

        try:
            resp = self.client.responses.create(
                model=EXTRACTION_MODEL,
                input=prompt,
                instructions="You are a concise note-taking assistant. Return only the updated session memory.",
                temperature=0.1,
                max_output_tokens=1200,
            )
            new_memory = resp.output_text.strip()

            if new_memory and len(new_memory) > 50:
                with self._lock:
                    # Enforce maximum size
                    if len(new_memory) > MAX_MEMORY_CHARS:
                        new_memory = new_memory[:MAX_MEMORY_CHARS] + "\n[... truncated ...]"
                    self._memory = new_memory
        except Exception as e:
            logger.error("[session_memory] LLM extraction failed: %s", e)

    def get_context(self) -> str:
        """
        Get the current session memory for injection into system prompt.

        Returns empty string if no memory has been extracted yet.
        """
        with self._lock:
            if not self._memory:
                return ""
            return (
                "\n\n## Session Memory (auto-extracted key context)\n"
                f"{self._memory}\n"
                "## End Session Memory\n"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Return session memory statistics."""
        return {
            "extraction_count": self._extraction_count,
            "memory_chars": len(self._memory),
            "total_tool_calls_tracked": self._total_tool_calls,
            "is_extracting": self._extracting,
        }

    def clear(self):
        """Clear session memory (on conversation reset)."""
        with self._lock:
            self._memory = ""
            self._initialized = False
            self._last_extraction_tokens = 0
            self._last_extraction_tool_calls = 0
            self._total_tool_calls = 0
            self._extraction_count = 0
