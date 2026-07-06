"""Private chat-run telemetry and structured issue reporting."""

from __future__ import annotations

import csv
import io
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.db import get_connection
from services.secret_redaction import redact_secrets


RUN_STATUSES = {"started", "completed", "failed", "timed_out", "cancelled"}
REPORT_STATUSES = {"new", "investigating", "resolved", "dismissed"}
REPORT_CATEGORIES = {"stuck_slow", "wrong_answer", "incorrect_data", "ui_problem", "other"}
MAX_CONTEXT_CHARS = 2000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any, limit: int = MAX_CONTEXT_CHARS) -> str:
    return redact_secrets(str(value or "")).strip()[:limit]


class ChatDeadline:
    """Track meaningful stream inactivity and total turn deadlines."""

    def __init__(
        self,
        *,
        inactivity_seconds: int,
        standard_seconds: int,
        conductor_seconds: int,
        started_at: Optional[float] = None,
        hard_max_seconds: Optional[int] = None,
    ):
        self.inactivity_seconds = max(1, int(inactivity_seconds))
        self.standard_seconds = max(1, int(standard_seconds))
        self.conductor_seconds = max(self.standard_seconds, int(conductor_seconds))
        # Ceiling for progress-aware extensions (see extend_for_progress).
        self.hard_max_seconds = max(
            self.conductor_seconds,
            int(hard_max_seconds) if hard_max_seconds is not None else int(self.standard_seconds * 2.5),
        )
        self.started_at = time.monotonic() if started_at is None else float(started_at)
        self.last_activity_at = self.started_at
        self.total_seconds = self.standard_seconds

    def mark_activity(self, now: Optional[float] = None) -> None:
        self.last_activity_at = time.monotonic() if now is None else float(now)

    def extend_for_progress(self, now: Optional[float] = None, *, extension_seconds: int = 90) -> None:
        """Push the total-turn deadline out when real work just completed.

        A turn that keeps finishing tool calls is progressing, not stuck — the
        fixed total cap was killing legitimate long multi-tool workflows
        (2026-07 live test). Guarantees at least `extension_seconds` of total
        budget after each completed tool call, capped at hard_max_seconds; the
        inactivity watchdog still ends genuinely stalled runs.
        """
        current = time.monotonic() if now is None else float(now)
        elapsed = current - self.started_at
        if self.total_seconds - elapsed < extension_seconds:
            self.total_seconds = min(float(self.hard_max_seconds), elapsed + float(extension_seconds))

    def enable_conductor(self) -> None:
        self.total_seconds = self.conductor_seconds

    def remaining(self, now: Optional[float] = None) -> float:
        current = time.monotonic() if now is None else float(now)
        return min(
            self.inactivity_seconds - (current - self.last_activity_at),
            self.total_seconds - (current - self.started_at),
        )

    def timeout_code(self, now: Optional[float] = None) -> str:
        current = time.monotonic() if now is None else float(now)
        if current - self.last_activity_at >= self.inactivity_seconds:
            return "chat_inactivity_timeout"
        if current - self.started_at >= self.total_seconds:
            return "chat_turn_timeout"
        return ""


class IssueReportService:
    """Persist private run diagnostics and user-submitted issue reports."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            root = Path(__file__).resolve().parent.parent
            db_path = str(root / "data" / "issue_reports.db")
        self._local_db_path = db_path
        self._init_db()

    def _conn(self):
        return get_connection(self._local_db_path)

    def _init_db(self) -> None:
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_runs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                conversation_id TEXT,
                trace_id TEXT,
                model TEXT NOT NULL,
                provider TEXT NOT NULL,
                key_source TEXT,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                duration_ms INTEGER,
                tools_called TEXT,
                last_status TEXT,
                error_code TEXT,
                error_message TEXT,
                client_ip TEXT,
                first_token_ms INTEGER,
                provider_chunk_count INTEGER
            )
            """
        )
        for migration in (
            "ALTER TABLE chat_runs ADD COLUMN first_token_ms INTEGER",
            "ALTER TABLE chat_runs ADD COLUMN provider_chunk_count INTEGER",
        ):
            try:
                cur.execute(migration)
            except Exception:
                pass
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_runs_user_created
            ON chat_runs(user_id, started_at DESC)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_runs_provider_status
            ON chat_runs(provider, status, started_at DESC)
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS issue_reports (
                id TEXT PRIMARY KEY,
                run_id TEXT,
                user_id TEXT NOT NULL,
                conversation_id TEXT,
                message_id TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                include_context INTEGER NOT NULL DEFAULT 0,
                prompt_excerpt TEXT,
                response_excerpt TEXT,
                model TEXT,
                provider TEXT,
                trace_id TEXT,
                technical_context TEXT,
                status TEXT NOT NULL,
                admin_notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_issue_reports_status_created
            ON issue_reports(status, created_at DESC)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_issue_reports_provider_model
            ON issue_reports(provider, model, created_at DESC)
            """
        )
        conn.commit()
        conn.close()

    def start_run(
        self,
        *,
        run_id: str,
        user_id: str,
        conversation_id: str,
        trace_id: str,
        model: str,
        provider: str,
        key_source: str,
        client_ip: str = "",
    ) -> Dict[str, Any]:
        now = _utc_now()
        conn = self._conn()
        conn.execute(
            """
            INSERT INTO chat_runs (
                id, user_id, conversation_id, trace_id, model, provider,
                key_source, status, started_at, updated_at, tools_called, client_ip
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                user_id,
                conversation_id or "",
                trace_id or "",
                model,
                provider,
                key_source or "platform",
                "started",
                now,
                now,
                "[]",
                client_ip or "",
            ),
        )
        conn.commit()
        conn.close()
        return self.get_run(run_id, user_id=user_id) or {}

    def update_run_activity(
        self,
        run_id: str,
        *,
        last_status: Optional[str] = None,
        tools_called: Optional[List[str]] = None,
    ) -> None:
        assignments = ["updated_at = ?"]
        params: List[Any] = [_utc_now()]
        if last_status is not None:
            assignments.append("last_status = ?")
            params.append(_clean_text(last_status, 500))
        if tools_called is not None:
            assignments.append("tools_called = ?")
            params.append(json.dumps(list(dict.fromkeys(tools_called))))
        params.append(run_id)
        conn = self._conn()
        conn.execute(
            f"UPDATE chat_runs SET {', '.join(assignments)} WHERE id = ?",
            tuple(params),
        )
        conn.commit()
        conn.close()

    def finalize_run(
        self,
        run_id: str,
        *,
        status: str,
        duration_ms: int,
        tools_called: Optional[List[str]] = None,
        last_status: str = "",
        error_code: str = "",
        error_message: str = "",
        first_token_ms: Optional[int] = None,
        provider_chunk_count: int = 0,
    ) -> None:
        if status not in RUN_STATUSES:
            raise ValueError(f"Unsupported run status: {status}")
        now = _utc_now()
        conn = self._conn()
        conn.execute(
            """
            UPDATE chat_runs
            SET status = ?, updated_at = ?, completed_at = ?, duration_ms = ?,
                tools_called = ?, last_status = ?, error_code = ?, error_message = ?,
                first_token_ms = ?, provider_chunk_count = ?
            WHERE id = ?
            """,
            (
                status,
                now,
                now,
                max(0, int(duration_ms)),
                json.dumps(list(dict.fromkeys(tools_called or []))),
                _clean_text(last_status, 500),
                _clean_text(error_code, 100),
                _clean_text(error_message),
                first_token_ms,
                max(0, int(provider_chunk_count)),
                run_id,
            ),
        )
        conn.commit()
        conn.close()

    def get_run(self, run_id: str, *, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        conn = self._conn()
        if user_id is None:
            row = conn.execute(
                """
                SELECT id, user_id, conversation_id, trace_id, model, provider,
                       key_source, status, started_at, updated_at, completed_at,
                       duration_ms, tools_called, last_status, error_code,
                       error_message, client_ip, first_token_ms, provider_chunk_count
                FROM chat_runs WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT id, user_id, conversation_id, trace_id, model, provider,
                       key_source, status, started_at, updated_at, completed_at,
                       duration_ms, tools_called, last_status, error_code,
                       error_message, client_ip, first_token_ms, provider_chunk_count
                FROM chat_runs WHERE id = ? AND user_id = ?
                """,
                (run_id, user_id),
            ).fetchone()
        conn.close()
        if not row:
            return None
        keys = [
            "id", "user_id", "conversation_id", "trace_id", "model", "provider",
            "key_source", "status", "started_at", "updated_at", "completed_at",
            "duration_ms", "tools_called", "last_status", "error_code",
            "error_message", "client_ip", "first_token_ms", "provider_chunk_count",
        ]
        result = dict(zip(keys, row))
        try:
            result["tools_called"] = json.loads(result.get("tools_called") or "[]")
        except (TypeError, json.JSONDecodeError):
            result["tools_called"] = []
        return result

    def create_report(
        self,
        *,
        user_id: str,
        run_id: str,
        message_id: str,
        category: str,
        description: str,
        include_context: bool = False,
        prompt_excerpt: str = "",
        response_excerpt: str = "",
        technical_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if category not in REPORT_CATEGORIES:
            raise ValueError("Unsupported issue category")
        description = _clean_text(description)
        if not description:
            raise ValueError("Issue description is required")
        run = self.get_run(run_id, user_id=user_id)
        if not run:
            raise LookupError("Chat run not found")

        report_id = str(uuid.uuid4())
        now = _utc_now()
        safe_prompt = _clean_text(prompt_excerpt) if include_context else ""
        safe_response = _clean_text(response_excerpt) if include_context else ""
        diagnostics = {
            "run_status": run.get("status"),
            "key_source": run.get("key_source"),
            "duration_ms": run.get("duration_ms"),
            "tools_called": run.get("tools_called") or [],
            "last_status": run.get("last_status"),
            "error_code": run.get("error_code"),
            "error_message": run.get("error_message"),
            "first_token_ms": run.get("first_token_ms"),
            "provider_chunk_count": run.get("provider_chunk_count"),
            "client": technical_context or {},
        }
        conn = self._conn()
        conn.execute(
            """
            INSERT INTO issue_reports (
                id, run_id, user_id, conversation_id, message_id, category,
                description, include_context, prompt_excerpt, response_excerpt,
                model, provider, trace_id, technical_context, status,
                admin_notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                run_id,
                user_id,
                run.get("conversation_id") or "",
                message_id,
                category,
                description,
                1 if include_context else 0,
                safe_prompt,
                safe_response,
                run.get("model") or "",
                run.get("provider") or "",
                run.get("trace_id") or "",
                _clean_text(json.dumps(diagnostics, default=str), 8000),
                "new",
                "",
                now,
                now,
            ),
        )
        conn.commit()
        conn.close()
        return self.get_report(report_id) or {}

    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        rows = self.list_reports(report_id=report_id, limit=1)
        return rows[0] if rows else None

    def list_reports(
        self,
        *,
        report_id: str = "",
        status: str = "",
        provider: str = "",
        model: str = "",
        category: str = "",
        date_from: str = "",
        date_to: str = "",
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        for column, value in (
            ("id", report_id),
            ("status", status),
            ("provider", provider),
            ("model", model),
            ("category", category),
        ):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        if date_from:
            clauses.append("created_at >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("created_at <= ?")
            params.append(f"{date_to}T23:59:59.999999+00:00" if len(date_to) == 10 else date_to)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit), 1000)))
        conn = self._conn()
        rows = conn.execute(
            f"""
            SELECT id, run_id, user_id, conversation_id, message_id, category,
                   description, include_context, prompt_excerpt, response_excerpt,
                   model, provider, trace_id, technical_context, status,
                   admin_notes, created_at, updated_at
            FROM issue_reports
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        conn.close()
        keys = [
            "id", "run_id", "user_id", "conversation_id", "message_id",
            "category", "description", "include_context", "prompt_excerpt",
            "response_excerpt", "model", "provider", "trace_id",
            "technical_context", "status", "admin_notes", "created_at", "updated_at",
        ]
        results = []
        for row in rows:
            item = dict(zip(keys, row))
            item["include_context"] = bool(item["include_context"])
            try:
                item["technical_context"] = json.loads(item["technical_context"] or "{}")
            except (TypeError, json.JSONDecodeError):
                item["technical_context"] = {}
            results.append(item)
        return results

    def update_report(
        self,
        report_id: str,
        *,
        status: Optional[str] = None,
        admin_notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        assignments = ["updated_at = ?"]
        params: List[Any] = [_utc_now()]
        if status is not None:
            if status not in REPORT_STATUSES:
                raise ValueError("Unsupported report status")
            assignments.append("status = ?")
            params.append(status)
        if admin_notes is not None:
            assignments.append("admin_notes = ?")
            params.append(_clean_text(admin_notes, 4000))
        params.append(report_id)
        conn = self._conn()
        cursor = conn.execute(
            f"UPDATE issue_reports SET {', '.join(assignments)} WHERE id = ?",
            tuple(params),
        )
        conn.commit()
        conn.close()
        if getattr(cursor, "rowcount", 0) == 0:
            raise LookupError("Issue report not found")
        return self.get_report(report_id) or {}

    def export_reports_csv(self, **filters: Any) -> str:
        rows = self.list_reports(limit=1000, **filters)
        output = io.StringIO()
        fields = [
            "id", "created_at", "status", "category", "description", "provider",
            "model", "run_id", "trace_id", "conversation_id", "message_id",
            "include_context", "prompt_excerpt", "response_excerpt",
            "technical_context", "admin_notes",
        ]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            record = dict(row)
            record["technical_context"] = json.dumps(record.get("technical_context") or {})
            writer.writerow(record)
        return output.getvalue()
