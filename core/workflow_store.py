# core/workflow_store.py
"""
Workflow Store — Persistent execution state for task DAGs.

Enables checkpoint/resume and implements CAMEL's "Controls" layer
(pause, resume, snapshot).  Uses JSON file storage for simplicity.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "workflows")


class WorkflowStore:
    """Persists task DAGs, execution state, and results for resumability."""

    def __init__(self, store_dir: str = _DEFAULT_STORE_DIR):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        query_id: str,
        dag_data: List[dict],
        results: Dict[str, Any],
        query: str = "",
        user_id: str = "",
    ) -> str:
        """
        Save a checkpoint of the current DAG execution state.

        Returns the checkpoint filepath.
        """
        checkpoint = {
            "query_id": query_id,
            "query": query,
            "user_id": user_id,
            "timestamp": time.time(),
            "dag": dag_data,
            "results": {k: _safe_serialize(v) for k, v in results.items()},
            "status": "active",
        }

        filepath = self.store_dir / f"{query_id}.json"
        with open(filepath, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)

        logger.info("Checkpoint saved: %s", filepath)
        return str(filepath)

    def load_checkpoint(
        self, query_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load a previously saved checkpoint."""
        filepath = self.store_dir / f"{query_id}.json"
        if not filepath.exists():
            return None

        with open(filepath) as f:
            return json.load(f)

    def list_workflows(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all saved workflows, optionally filtered by user_id."""
        workflows = []
        for fp in self.store_dir.glob("*.json"):
            try:
                with open(fp) as f:
                    data = json.load(f)
                if user_id and data.get("user_id") != user_id:
                    continue
                workflows.append({
                    "query_id": data.get("query_id"),
                    "query": data.get("query", "")[:100],
                    "timestamp": data.get("timestamp"),
                    "status": data.get("status", "unknown"),
                    "task_count": len(data.get("dag", [])),
                })
            except Exception:
                continue
        return sorted(workflows, key=lambda w: w.get("timestamp", 0), reverse=True)

    def pause(self, query_id: str) -> bool:
        """Mark a workflow as paused."""
        return self._update_status(query_id, "paused")

    def resume(self, query_id: str) -> bool:
        """Mark a workflow as active (resumed)."""
        return self._update_status(query_id, "active")

    def snapshot(self, query_id: str) -> Optional[str]:
        """Create a timestamped snapshot copy of the current state."""
        data = self.load_checkpoint(query_id)
        if not data:
            return None

        snapshot_id = f"{query_id}_snap_{int(time.time())}"
        data["query_id"] = snapshot_id
        data["status"] = "snapshot"

        filepath = self.store_dir / f"{snapshot_id}.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return snapshot_id

    def delete(self, query_id: str) -> bool:
        """Delete a stored workflow."""
        filepath = self.store_dir / f"{query_id}.json"
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def _update_status(self, query_id: str, status: str) -> bool:
        """Update the status of a stored workflow."""
        data = self.load_checkpoint(query_id)
        if not data:
            return False
        data["status"] = status
        filepath = self.store_dir / f"{query_id}.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return True

    @staticmethod
    def generate_query_id() -> str:
        """Generate a unique query ID."""
        return str(uuid.uuid4())[:12]


def _safe_serialize(obj: Any) -> Any:
    """Convert an object to a JSON-serializable form."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)
