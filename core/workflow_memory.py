# core/workflow_memory.py
"""
Workflow Memory — shared structured context across sub-agents.

During orchestration of a complex query, each sub-agent writes its results
to WorkflowMemory.  Later agents (especially SynthesisAgent) read from it
to build aggregated answers, comparison tables, etc.

Implements CAMEL Workforce "Workflow Memory / Context Sharing".

Thread-safe: uses threading.Lock for safe concurrent writes from
run_in_executor() threads in the Conductor DAG.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
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
    task_id: Optional[str] = None  # (#7) link to the DAG task that produced this


class WorkflowMemory:
    """
    Structured key-value store shared across all sub-agents during one query
    execution.  Thread-safe via threading.Lock.

    Usage
    -----
    wm = WorkflowMemory()
    wm.write("archive_agent", "search_results_sz65", {...}, task_id="t1")
    wm.write("analysis_agent", "beam_sizes", [0.3, 0.5, ...], task_id="t3")
    results = wm.read("search_results_sz65")
    all_data = wm.read_all()
    """

    def __init__(self):
        self._store: Dict[str, MemoryEntry] = {}
        self._history: List[MemoryEntry] = []
        self._lock = threading.Lock()  # (#5) Thread-safe for Conductor's parallel threads

    def write(self, agent_id: str, key: str, value: Any, task_id: Optional[str] = None) -> None:
        """Write a value to memory, overwriting any existing entry for this key."""
        with self._lock:
            entry = MemoryEntry(agent_id=agent_id, key=key, value=value, task_id=task_id)
            self._store[key] = entry
            self._history.append(entry)
            logger.debug("WorkflowMemory: %s wrote key '%s' (task=%s)", agent_id, key, task_id)

    def read(self, key: str, default: Any = None) -> Any:
        """Read a value from memory by key."""
        with self._lock:
            entry = self._store.get(key)
            return entry.value if entry else default

    def read_all(self) -> Dict[str, Any]:
        """Return all stored key-value pairs (keys → values, no metadata)."""
        with self._lock:
            return {k: e.value for k, e in self._store.items()}

    def get_agent_contributions(self, agent_id: str) -> Dict[str, Any]:
        """Return all key-value pairs written by a specific agent."""
        with self._lock:
            return {
                k: e.value
                for k, e in self._store.items()
                if e.agent_id == agent_id
            }

    def get_task_result(self, task_id: str) -> Optional[Any]:
        """
        (#7) Read the result for a specific DAG task.

        This allows downstream tasks to access structured results from
        upstream tasks via WorkflowMemory instead of raw JSON strings.
        """
        with self._lock:
            for entry in self._store.values():
                if entry.task_id == task_id:
                    return entry.value
            return None

    def get_dependency_context(self, task_ids: List[str], max_chars: int = 8000) -> str:
        """
        (#7) Build a structured context string from specific task results.

        Used by the Conductor to pass dependency context to downstream tasks
        in a more organized way than raw JSON dumps.
        """
        with self._lock:
            parts = []
            for task_id in task_ids:
                for entry in self._store.values():
                    if entry.task_id == task_id:
                        val_str = json.dumps(entry.value, default=str)
                        if len(val_str) > 2000:
                            val_str = val_str[:2000] + "… (truncated)"
                        parts.append(f"[{entry.agent_id} — {task_id}]: {val_str}")
                        break

            context = "\n\n".join(parts)
            if len(context) > max_chars:
                context = context[:max_chars] + "\n… (truncated)"
            return context

    def get_context_summary(self, max_chars: int = 8000) -> str:
        """
        Build a concise text summary of all stored data for use as context
        in LLM prompts.  Truncates large values.
        """
        with self._lock:
            lines = []
            for key, entry in self._store.items():
                val_str = json.dumps(entry.value, default=str)
                if len(val_str) > 500:
                    val_str = val_str[:500] + "… (truncated)"
                task_label = f" (task={entry.task_id})" if entry.task_id else ""
                lines.append(f"[{entry.agent_id}{task_label}] {key}: {val_str}")
            summary = "\n".join(lines)
            if len(summary) > max_chars:
                summary = summary[:max_chars] + "\n… (truncated)"
            return summary

    def get_stats(self) -> Dict[str, Any]:
        """Return memory usage statistics."""
        with self._lock:
            total_size = sum(
                len(json.dumps(e.value, default=str)) for e in self._store.values()
            )
            return {
                "entries": len(self._store),
                "history_writes": len(self._history),
                "total_size_chars": total_size,
                "keys": list(self._store.keys()),
            }

    def keys(self) -> List[str]:
        """List all stored keys."""
        with self._lock:
            return list(self._store.keys())

    def clear(self) -> None:
        """Clear all stored data (between queries)."""
        with self._lock:
            self._store.clear()
            self._history.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._store
