# core/workflow_memory.py
"""
Workflow Memory — shared structured context across sub-agents.

During orchestration of a complex query, each sub-agent writes its results
to WorkflowMemory.  Later agents (especially SynthesisAgent) read from it
to build aggregated answers, comparison tables, etc.

Implements CAMEL Workforce "Workflow Memory / Context Sharing".
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single write to workflow memory."""
    agent_id: str
    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)


class WorkflowMemory:
    """
    Structured key-value store shared across all sub-agents during one query
    execution.  Thread-safe for asyncio (not for multi-threaded).

    Usage
    -----
    wm = WorkflowMemory()
    wm.write("archive_agent", "search_results_sz65", {...})
    wm.write("analysis_agent", "beam_sizes", [0.3, 0.5, ...])
    results = wm.read("search_results_sz65")
    all_data = wm.read_all()
    """

    def __init__(self):
        self._store: Dict[str, MemoryEntry] = {}
        self._history: List[MemoryEntry] = []

    def write(self, agent_id: str, key: str, value: Any) -> None:
        """Write a value to memory, overwriting any existing entry for this key."""
        entry = MemoryEntry(agent_id=agent_id, key=key, value=value)
        self._store[key] = entry
        self._history.append(entry)
        logger.debug("WorkflowMemory: %s wrote key '%s'", agent_id, key)

    def read(self, key: str, default: Any = None) -> Any:
        """Read a value from memory by key."""
        entry = self._store.get(key)
        return entry.value if entry else default

    def read_all(self) -> Dict[str, Any]:
        """Return all stored key-value pairs (keys → values, no metadata)."""
        return {k: e.value for k, e in self._store.items()}

    def get_agent_contributions(self, agent_id: str) -> Dict[str, Any]:
        """Return all key-value pairs written by a specific agent."""
        return {
            k: e.value
            for k, e in self._store.items()
            if e.agent_id == agent_id
        }

    def get_context_summary(self, max_chars: int = 8000) -> str:
        """
        Build a concise text summary of all stored data for use as context
        in LLM prompts.  Truncates large values.
        """
        lines = []
        for key, entry in self._store.items():
            val_str = json.dumps(entry.value, default=str)
            if len(val_str) > 500:
                val_str = val_str[:500] + "… (truncated)"
            lines.append(f"[{entry.agent_id}] {key}: {val_str}")
        summary = "\n".join(lines)
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "\n… (truncated)"
        return summary

    def keys(self) -> List[str]:
        """List all stored keys."""
        return list(self._store.keys())

    def clear(self) -> None:
        """Clear all stored data (between queries)."""
        self._store.clear()
        self._history.clear()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return key in self._store
