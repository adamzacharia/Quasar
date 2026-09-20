"""Immutable, consented feedback evidence, independent of live chat retention.

Snapshots contain the persisted conversation, not a reconstruction from client
previews. No foreign keys or cleanup hooks connect them to the conversation DB.
"""
from __future__ import annotations

import json
import math
import re
import uuid
import threading
from datetime import datetime, timezone
from typing import Any

from services.db import get_connection
from services.secret_redaction import redact_secrets, redact_url

MAX_SNAPSHOT_MESSAGES = 400
MAX_SNAPSHOT_BYTES = 4 * 1024 * 1024
_SECRET_KEY = re.compile(r"(?i)^(?:.*[_-])?(?:api[_-]?key|authorization|password|passwd|secret|token|cookie|credentials?|access_token|refresh_token|auth|session|session_?id|sid|signature|sig)$")
_URL = re.compile(r"https?://[^\s<>\"']+")


def safe_snapshot_value(value: Any, depth: int = 0) -> Any:
    """Redact nested keys/values, including short credentials and URL tokens.

    Depth overflow is explicit; ordinary messages and tool JSON are preserved
    without per-string clipping. The enclosing snapshot enforces its byte cap.
    """
    if depth > 40:
        return {"omitted": "nested value exceeds 40 levels"}
    if isinstance(value, str):
        text = value.encode("utf-8", "replace").decode("utf-8")
        return redact_secrets(_URL.sub(lambda m: redact_url(m.group()), text))
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [safe_snapshot_value(v, depth + 1) for v in value]
    if isinstance(value, dict):
        return {
            str(safe_snapshot_value(str(k), depth + 1)):
                "[REDACTED]" if _SECRET_KEY.match(str(k)) else safe_snapshot_value(v, depth + 1)
            for k, v in value.items()
        }
    return safe_snapshot_value(str(value), depth + 1)


def snapshot_json(snapshot: dict) -> str:
    return json.dumps(snapshot, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class FeedbackToolTrace:
    """Request-owned, thread-safe evidence collector, upstream of display caps."""
    def __init__(self):
        self._lock = threading.Lock()
        self._calls = []
        self._total = 0
        self._bytes = 0
        self._truncated = False
        self._terminal_status = None

    def record(self, name, arguments, output, provenance=None):
        with self._lock:
            if self._terminal_status is not None:
                return
            if self._truncated:
                self._total += 1
                return
        # Serialization/redaction may be expensive. It must not hold the lock
        # used by cancellation on the event loop. Evidence still being prepared
        # when the stream closes is excluded from that frozen partial record.
        call = safe_snapshot_value({"call_id": uuid.uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "name": name, "arguments": arguments, "output": output,
            "provenance": provenance})
        size = len(snapshot_json(call).encode("utf-8")) + 1
        with self._lock:
            if self._terminal_status is not None:
                return
            self._total += 1
            if self._truncated:
                return
            if self._bytes + size > 2 * 1024 * 1024 - 1024:
                self._truncated = True
                return
            self._calls.append(call)
            self._bytes += size

    def snapshot(self):
        with self._lock:
            return self._snapshot_locked()

    def freeze(self, status: str):
        """Close the evidence at run termination; late workers cannot change it."""
        with self._lock:
            if self._terminal_status is None:
                self._terminal_status = status
            return self._snapshot_locked()

    def _snapshot_locked(self):
        result = {"calls": list(self._calls), "total_calls": self._total,
                  "truncated": self._truncated, "max_bytes": 2 * 1024 * 1024}
        if self._terminal_status is not None:
            result["run_status"] = self._terminal_status
            result["partial"] = self._terminal_status != "completed"
        return result


def persist_feedback_tool_trace(trace: list) -> dict:
    """Keep redacted calls/results with the live turn for later consented capture.

    The existing display projection intentionally drops results. This separate
    evidence field preserves them with an explicit 2 MiB per-turn limit.
    """
    calls, used = [], 0
    for call in trace or []:
        clean = safe_snapshot_value(call)
        size = len(snapshot_json(clean).encode("utf-8")) + 1
        if used + size > 2 * 1024 * 1024:
            break
        calls.append(clean)
        used += size
    return {"calls": calls, "total_calls": len(trace or []),
            "truncated": len(calls) < len(trace or []), "max_bytes": 2 * 1024 * 1024}


def build_snapshot(*, source: dict, user_id: str, conversation_id: str,
                   message_id: str, run: dict | None = None, vote: str = "dislike",
                   description: str = "", max_messages: int = MAX_SNAPSHOT_MESSAGES,
                   max_bytes: int = MAX_SNAPSHOT_BYTES) -> dict:
    """Copy complete messages in chronological order; never silently slice JSON."""
    max_messages = min(MAX_SNAPSHOT_MESSAGES, max(0, max_messages))
    max_bytes = min(MAX_SNAPSHOT_BYTES, max(4096, max_bytes))
    run = run or {}
    messages = source.get("messages") or []
    result = safe_snapshot_value({
        "schema_version": 1, "id": str(uuid.uuid4()),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id, "conversation_id": conversation_id,
        "message_id": message_id, "run_id": run.get("id") or "",
        "model": run.get("model") or source.get("model") or "",
        "provider": run.get("provider") or "", "vote": vote,
        "feedback_text": description,
        "run": {k: run.get(k) for k in (
            "id", "trace_id", "model", "provider", "started_at", "completed_at",
            "status", "tools_called", "error_code", "error_message")},
        "tool_evidence": {
            "source": "messages.metadata.toolTrace and copied run metadata",
            "note": "Redacted calls/results are in feedbackToolTrace for new turns. Older turns may contain only toolTrace requests and run/trace references; external traces may have their own retention period.",
        },
        "limits": {"max_messages": max_messages, "max_bytes": max_bytes},
        "messages": [],
        "truncation": {"truncated": False, "captured_messages": 0,
                       "total_messages": source.get("total_messages", len(messages)),
                       "reasons": []},
    })
    # Metadata about a report is small by contract, but fail explicitly if a
    # malformed source supplies a huge run/header rather than exceeding the cap.
    used = len(snapshot_json(result).encode("utf-8"))
    if used > max_bytes - 1024:
        raise ValueError("Snapshot metadata exceeds the size limit")
    reasons = result["truncation"]["reasons"]
    for message in messages[:max_messages]:
        entry = safe_snapshot_value(message)
        if entry.get("source_omitted"):
            reasons.append("source_byte_limit")
            break
        meta = entry.get("metadata") or {}
        turn_run = meta.get("runMeta") or {}
        entry["is_reported_answer"] = entry.get("role") == "assistant" and bool(
            (run.get("id") and turn_run.get("run_id") == run["id"])
            or str(entry.get("id") or "") == message_id
            or turn_run.get("text_block_id") == message_id)
        encoded_size = len(snapshot_json(entry).encode("utf-8")) + 1
        if used + encoded_size > max_bytes - 1024:
            reasons.append("snapshot_byte_limit")
            break
        result["messages"].append(entry)
        used += encoded_size
    count = len(result["messages"])
    total = result["truncation"]["total_messages"]
    if total > max_messages:
        reasons.append("message_limit")
    if source.get("source_truncated") and "source_byte_limit" not in reasons:
        reasons.append("source_byte_limit")
    result["truncation"].update(
        truncated=count < total or bool(reasons), captured_messages=count,
        marker=f"Truncated at {count} of {total} messages" if count < total or reasons else "",
    )
    # This also protects callers passing a nonstandard source shape.
    if len(snapshot_json(result).encode("utf-8")) > max_bytes:
        raise ValueError("Snapshot exceeds the size limit")
    return result


class FeedbackSnapshotService:
    def __init__(self, db_path: str):
        self.db_path = db_path
        conn = get_connection(db_path)
        try:
            conn.execute("""CREATE TABLE IF NOT EXISTS feedback_snapshots (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                captured_at TEXT NOT NULL, snapshot_json TEXT NOT NULL
            )""")
            conn.commit()
        finally:
            conn.close()

    def insert(self, snapshot: dict, *, conn=None) -> None:
        own_connection = conn is None
        conn = conn if conn is not None else get_connection(self.db_path)
        try:
            conn.execute("INSERT INTO feedback_snapshots (id,user_id,captured_at,snapshot_json) VALUES (?,?,?,?)",
                         (snapshot["id"], snapshot["user_id"], snapshot["captured_at"], snapshot_json(snapshot)))
            if own_connection:
                conn.commit()
        finally:
            if own_connection:
                conn.close()

    def get(self, snapshot_id: str) -> dict | None:
        conn = get_connection(self.db_path)
        try:
            row = conn.execute("SELECT snapshot_json FROM feedback_snapshots WHERE id = ?", (snapshot_id,)).fetchone()
            return json.loads(row[0]) if row else None
        finally:
            conn.close()
