"""Private chat-run telemetry and structured issue reporting."""

from __future__ import annotations

import csv
import io
import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.db import get_connection
import math
from typing import Callable, Iterable, Set, Tuple  # noqa: F401  (complements the typing imports above)

from services.secret_redaction import redact_secrets
from services.feedback_snapshot_service import FeedbackSnapshotService, build_snapshot


RUN_STATUSES = {"started", "completed", "failed", "timed_out", "cancelled"}
REPORT_STATUSES = {"new", "investigating", "resolved", "dismissed"}
REPORT_CATEGORIES = {"stuck_slow", "wrong_answer", "incorrect_data", "ui_problem", "other"}
MAX_CONTEXT_CHARS = 2000
# Server-captured conversation context (stored only when the reporter consents).
# The reported ANSWER may be longer than the prompt/description cap: a 2,000-char
# excerpt cut most Quasar answers in half, which is exactly the part a
# maintainer needs to read. The excerpt window is the reported answer plus the
# turns leading up to it, so "this answer is wrong" arrives with the question,
# the answer, and what came before.
MAX_RESPONSE_CHARS = 6000
MAX_EXCERPT_MESSAGES = 6
MAX_EXCERPT_TOTAL_CHARS = 24000
MAX_EXCERPT_TOOLS = 12


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utf8_safe(value: str) -> str:
    """Drop lone UTF-16 surrogates a JSON client can send (a string clipped in
    the middle of an emoji). Python accepts them; SQLite cannot bind them, so
    an otherwise valid report would fail at INSERT (verify follow-up to CX-07)."""
    try:
        value.encode("utf-8")
        return value
    except UnicodeEncodeError:
        return value.encode("utf-8", "replace").decode("utf-8")


def _clean_text(value: Any, limit: int = MAX_CONTEXT_CHARS) -> str:
    return _utf8_safe(redact_secrets(str(value or ""))).strip()[:limit]


MAX_DIAGNOSTICS_CHARS = 8000
MAX_CLIENT_CONTEXT_KEYS = 24
MAX_CLIENT_CONTEXT_VALUE_CHARS = 400


def _bounded_client_context(technical_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Client-supplied diagnostics, bounded BEFORE serialization.

    The client dict is untrusted input; capping keys and value lengths here
    keeps the serialized diagnostics small enough that the row never has to be
    cut mid-JSON (CX-05).
    """
    if not isinstance(technical_context, dict):
        return {}
    bounded: Dict[str, Any] = {}
    for key in list(technical_context)[:MAX_CLIENT_CONTEXT_KEYS]:
        value = technical_context[key]
        # Keys are client strings too: redact and cap them, drop empties.
        clean_key = _clean_text(key, 64)
        if not clean_key:
            continue
        # Python's json parser accepts the non-standard NaN/Infinity tokens;
        # a stored NaN cannot be re-serialised by the admin listing and would
        # 500 it for every admin until the row expires. Treat as absent.
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        if isinstance(value, (int, float, bool)) or value is None:
            bounded[clean_key] = value
        else:
            bounded[clean_key] = _clean_text(
                value if isinstance(value, str) else json.dumps(value, default=str),
                MAX_CLIENT_CONTEXT_VALUE_CHARS,
            )
    return bounded


def _serialize_diagnostics(diagnostics: Dict[str, Any]) -> str:
    """JSON for technical_context that always parses.

    Never slices the serialized string (that produced unparsable JSON, and
    list_reports then dropped the whole object, including the conversation
    lookup error). Over budget, the client block is replaced by a marker and
    long server strings are shortened; the mandatory keys survive (CX-05).
    """
    text = json.dumps(diagnostics, default=str)
    if len(text) <= MAX_DIAGNOSTICS_CHARS:
        return text
    trimmed = dict(diagnostics)
    trimmed["client"] = {"truncated": True}
    for key in ("error_message", "last_status", "conversation_excerpt_error"):
        if isinstance(trimmed.get(key), str):
            trimmed[key] = trimmed[key][:500]
    text = json.dumps(trimmed, default=str)
    if len(text) <= MAX_DIAGNOSTICS_CHARS:
        return text
    tools = trimmed.get("tools_called")
    if isinstance(tools, list):
        trimmed["tools_called"] = tools[:50]
        trimmed["tools_called_truncated"] = len(tools) > 50
    return json.dumps(trimmed, default=str)


def _message_run_id(message: Dict[str, Any]) -> str:
    """run_id persisted on an assistant message (sse.py stores it under
    metadata.runMeta.run_id); '' when absent or malformed."""
    meta = message.get("metadata")
    if not isinstance(meta, dict):
        return ""
    run_meta = meta.get("runMeta")
    if not isinstance(run_meta, dict):
        return ""
    return str(run_meta.get("run_id") or "")


def _message_tool_names(message: Dict[str, Any]) -> List[str]:
    """Distinct tool names from a persisted toolTrace, in call order."""
    meta = message.get("metadata")
    if not isinstance(meta, dict):
        return []
    trace = meta.get("toolTrace")
    if not isinstance(trace, list):
        return []
    names: List[str] = []
    for item in trace:
        if not isinstance(item, dict):
            continue
        name = str(item.get("tool") or item.get("tool_name") or item.get("name") or "").strip()
        if name and name not in names:
            names.append(name)
        if len(names) >= MAX_EXCERPT_TOOLS:
            break
    return names


def _anchor_index(
    messages: List[Dict[str, Any]], run_id: str, response_excerpt: str
) -> Tuple[Optional[int], str]:
    """(index, method) of the assistant message the report is about.

    Preference order, with the method recorded so the admin view can show how
    sure the match is: ``run_id`` (the persisted runMeta.run_id matches the
    report's run; exact), ``answer_text`` (the latest assistant message whose
    redacted text starts with the whole redacted client excerpt; legacy
    history without runMeta), ``last_assistant`` (nothing matched; the latest
    assistant message is the best guess). (None, "") when the conversation has
    no assistant message.
    """
    if run_id:
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if msg.get("role") == "assistant" and _message_run_id(msg) == run_id:
                return idx, "run_id"
    # The client excerpt arrives already secret-redacted (create_report cleans
    # it), so compare against the persisted text redacted the same way, or a
    # secret near the start would defeat the match (round-1 CX-03). The WHOLE
    # excerpt is compared, not a 200-char prefix, so answers that share
    # boilerplate openings stay distinguishable wherever they differ (CX-03).
    probe = redact_secrets(str(response_excerpt or "")).strip()
    if len(probe) >= 8:
        matches: List[int] = []
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if msg.get("role") != "assistant":
                continue
            persisted = redact_secrets(str(msg.get("content") or "")).strip()
            if persisted.startswith(probe):
                matches.append(idx)
        if len(matches) == 1:
            return matches[0], "answer_text"
        if matches:
            # Identical repeated answers ("Yes, done.", canned refusals) cannot
            # be told apart by text: take the latest but say so, so the admin
            # card labels it a guess rather than an exact match.
            return matches[0], "answer_text_ambiguous"
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            return idx, "last_assistant"
    return None, ""


def build_conversation_excerpt(
    messages: List[Dict[str, Any]],
    *,
    run_id: str,
    response_excerpt: str = "",
    max_messages: int = MAX_EXCERPT_MESSAGES,
    max_total_chars: int = MAX_EXCERPT_TOTAL_CHARS,
) -> List[Dict[str, Any]]:
    """The reported answer plus the turns leading up to it, redacted and capped.

    Pure function over the conversation store's message dicts
    (role/content/metadata[/created_at]); the reporter's consent is checked by
    the caller. Each entry carries the role, redacted text, a ``truncated``
    flag, ``is_reported_answer`` on the anchor, and the tool names the
    assistant called when the persisted trace has them.
    """
    if not messages or max_messages < 1:
        return []
    anchor, anchor_method = _anchor_index(messages, run_id, response_excerpt)
    if anchor is None:
        return []
    start = max(0, anchor - (max_messages - 1))
    window = messages[start:anchor + 1]
    # Redact first, then cap: ``truncated`` must mean "the cap cut text", not
    # "a secret was redacted" (redaction also shortens the string).
    prepared = [
        (start + offset, msg, redact_secrets(str(msg.get("content") or "")).strip())
        for offset, msg in enumerate(window)
    ]
    # Allocate the shared character budget NEWEST-FIRST: the reported answer,
    # then the question that produced it, then earlier turns. Oldest-first
    # allocation let four long earlier turns consume the whole budget and
    # leave the question and the answer empty (CX-01).
    budget = max(0, int(max_total_chars))
    allocated: Dict[int, str] = {}
    for idx, _msg, redacted in reversed(prepared):
        limit = min(MAX_RESPONSE_CHARS, budget)
        text = redacted[:limit] if limit > 0 else ""
        budget -= len(text)
        allocated[idx] = text
    excerpt: List[Dict[str, Any]] = []
    for idx, msg, redacted in prepared:
        text = allocated[idx]
        is_anchor = idx == anchor
        if not text and redacted and not is_anchor:
            # An earlier turn that received no budget is dropped rather than
            # rendered as an empty block; the answer is always emitted.
            continue
        entry: Dict[str, Any] = {
            "role": str(msg.get("role") or ""),
            "content": text,
            "truncated": len(redacted) > len(text),
            "is_reported_answer": is_anchor,
        }
        if is_anchor:
            # How the reported answer was identified; anything but "run_id"
            # is a best-effort guess on legacy history (CX-03, CX-04).
            entry["anchor_method"] = anchor_method
        created_at = str(msg.get("created_at") or "")
        if created_at:
            entry["created_at"] = created_at
        tools = _message_tool_names(msg)
        if tools:
            entry["tools"] = tools
        excerpt.append(entry)
    return excerpt


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

    def __init__(
        self,
        db_path: Optional[str] = None,
        conversation_lookup: Optional[Callable[[str, str], Optional[List[Dict[str, Any]]]]] = None,
        snapshot_lookup: Optional[Callable[[str, str], Optional[Dict[str, Any]]]] = None,
    ):
        if db_path is None:
            root = Path(__file__).resolve().parent.parent
            db_path = str(root / "data" / "issue_reports.db")
        self._local_db_path = db_path
        # Read-only accessor (conversation_id, user_id) -> messages, or None
        # when the conversation does not belong to that user (api.deps injects
        # ConversationService.get_conversation_messages_for_user). With it, a
        # consenting report carries the real question, answer and surrounding
        # turns from the server's own store instead of only the client's
        # truncated excerpt, and only ever from the reporter's own
        # conversation. Without it, reports degrade to the client excerpt and
        # say so in technical_context.
        self._conversation_lookup = conversation_lookup
        self._snapshot_lookup = snapshot_lookup
        self._init_db()
        self.snapshots = FeedbackSnapshotService(self._local_db_path)

    def _conn(self):
        return get_connection(self._local_db_path)

    def _init_db(self) -> None:
        conn = self._conn()
        try:
            self._create_schema(conn)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _create_schema(conn) -> None:
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
                provider_chunk_count INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                cost_usd REAL,
                platform_cost_usd REAL,
                unpriced_tokens INTEGER
            )
            """
        )
        for migration in (
            "ALTER TABLE chat_runs ADD COLUMN first_token_ms INTEGER",
            "ALTER TABLE chat_runs ADD COLUMN provider_chunk_count INTEGER",
            # Per-turn usage. cost_usd is a derived estimate (services/model_pricing.py)
            # and stays NULL for models we cannot price — NULL means "unknown",
            # not "free". unpriced_tokens records HOW MANY of this turn's tokens
            # cost_usd could not cover, so a mixed turn (e.g. TACC main model +
            # a priced OpenAI embedding) is not silently reported as fully
            # priced just because cost_usd is non-null.
            "ALTER TABLE chat_runs ADD COLUMN input_tokens INTEGER",
            "ALTER TABLE chat_runs ADD COLUMN output_tokens INTEGER",
            "ALTER TABLE chat_runs ADD COLUMN total_tokens INTEGER",
            "ALTER TABLE chat_runs ADD COLUMN cost_usd REAL",
            # cost_usd is the WHOLE turn's cost regardless of who paid, which is
            # the right number for cost-efficiency (what did this answer cost to
            # produce). It is the wrong number for "what does Quasar owe": one
            # turn can mix routes, spending the user's BYOK key on the main loop
            # and Quasar's platform key on an embedding. platform_cost_usd is
            # the per-call-attributed subset paid by the platform, so admin
            # totals can exclude user-paid BYOK spend (CX-28).
            "ALTER TABLE chat_runs ADD COLUMN platform_cost_usd REAL",
            "ALTER TABLE chat_runs ADD COLUMN unpriced_tokens INTEGER",
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
                conversation_excerpt TEXT,
                status TEXT NOT NULL,
                admin_notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        # Databases created before the server-side context capture lack the
        # column; the ALTER is a no-op (swallowed) on fresh schemas. The
        # follow-up SELECT distinguishes "already present" from a swallowed
        # real failure (permissions, unsupported DDL) so a missing column is
        # reported at startup instead of at the first INSERT (CX-08).
        for migration in (
            "ALTER TABLE issue_reports ADD COLUMN conversation_excerpt TEXT",
        ):
            try:
                cur.execute(migration)
            except Exception:
                pass
        try:
            cur.execute("SELECT conversation_excerpt FROM issue_reports LIMIT 0")
        except Exception as exc:
            print(
                "[ISSUE REPORTS] WARNING: issue_reports.conversation_excerpt is missing after "
                f"migration; consenting reports will fail to store context: {exc}",
                flush=True,
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
        # Join keys for the admin vote/report linkage (reports_exist_for).
        cur.execute("CREATE INDEX IF NOT EXISTS idx_issue_reports_message ON issue_reports(message_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_issue_reports_run ON issue_reports(run_id)")

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
        return_row: bool = True,
    ) -> Dict[str, Any]:
        now = _utc_now()
        conn = self._conn()
        try:
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
        finally:
            conn.close()
        if not return_row:
            # Insert-only mode (SSE chat path — it ignores the row): the
            # cancel-safe start_run wrapper's guarantee "an unfinished
            # start_fn committed no row" requires that NOTHING follows the
            # commit — a post-commit get_run() hanging past the fallback
            # window stranded the committed row in 'started' (guard CX-33).
            return {}
        return self.get_run(run_id, user_id=user_id) or {}

    def mark_run_cancelled_if_started(self, run_id: str) -> bool:
        """Conditionally sweep ONE possibly-orphaned run row (guard CX-33).

        The SSE cancel fallback calls this when the run insert's worker is
        still unresponsive after the bounded wait: on Turso the server can
        commit while the client hangs awaiting the HTTP response, so the row
        may exist even though the inserting worker never reached its
        finalize. This runs on its OWN connection (sees any server-side
        commit), is a no-op when the commit never landed, and the
        status='started' guard means it can never clobber a proper finalize
        that got there first — which is why it may safely bypass the
        finalize latch."""
        now = _utc_now()
        conn = self._conn()
        try:
            cur = conn.execute(
                """
                UPDATE chat_runs
                SET status = 'cancelled', error_code = 'client_cancelled',
                    error_message = 'cancelled while the run record was being created',
                    updated_at = ?
                WHERE id = ? AND status = 'started'
                """,
                (now, run_id),
            )
            conn.commit()
            return bool(getattr(cur, "rowcount", 0))
        finally:
            conn.close()

    def sweep_stale_started_runs(self, *, max_age_minutes: int = 30) -> int:
        """Mark long-stale 'started' runs as failed (taskboard RUN-SWEEPER).

        Deploy-level backstop behind the in-request lifecycle: process death,
        a commit that became visible after the in-request sweep horizon
        (~11 min — guard CX-33), or any other orphaning leaves rows stuck in
        'started' forever, polluting run analytics. Runs at startup (and may
        be called periodically); the age floor (20 min) must comfortably
        exceed both the SSE hard cap (900 s) and the in-request sweep horizon
        so it can never race a LIVE run. A stale started_at alone is NOT
        proof of orphaning: update_run_activity heartbeats updated_at on
        every tool step, so a run with a FRESH updated_at is live and must
        never be swept (guard CX-10). Timestamps are ISO-8601 UTC strings,
        so the cutoff comparisons are lexicographic; a NULL updated_at
        (legacy rows) falls back to the started_at check alone."""
        max_age_minutes = max(int(max_age_minutes), 20)
        cutoff = (
            datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
        ).isoformat()
        now = _utc_now()
        conn = self._conn()
        try:
            cur = conn.execute(
                """
                UPDATE chat_runs
                SET status = 'failed', error_code = 'orphaned',
                    error_message = 'run never finalized (orphaned by a process death or a hung run-record insert)',
                    updated_at = ?
                WHERE status = 'started' AND started_at < ?
                  AND (updated_at IS NULL OR updated_at < ?)
                """,
                (now, cutoff, cutoff),
            )
            conn.commit()
            return int(getattr(cur, "rowcount", 0) or 0)
        finally:
            conn.close()

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
        try:
            conn.execute(
                f"UPDATE chat_runs SET {', '.join(assignments)} WHERE id = ?",
                tuple(params),
            )
            conn.commit()
        finally:
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
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        platform_cost_usd: Optional[float] = None,
        unpriced_tokens: Optional[int] = None,
    ) -> None:
        if status not in RUN_STATUSES:
            raise ValueError(f"Unsupported run status: {status}")
        now = _utc_now()
        # total_tokens is derived so callers cannot report a total that
        # disagrees with its own parts.
        safe_input = None if input_tokens is None else max(0, int(input_tokens))
        safe_output = None if output_tokens is None else max(0, int(output_tokens))
        if safe_input is None and safe_output is None:
            total_tokens = None
        else:
            total_tokens = (safe_input or 0) + (safe_output or 0)
        safe_cost = None if cost_usd is None else max(0.0, float(cost_usd))
        safe_platform_cost = (
            None if platform_cost_usd is None else max(0.0, float(platform_cost_usd))
        )
        safe_unpriced = None if unpriced_tokens is None else max(0, int(unpriced_tokens))
        conn = self._conn()
        try:
            conn.execute(
                """
                UPDATE chat_runs
                SET status = ?, updated_at = ?, completed_at = ?, duration_ms = ?,
                    tools_called = ?, last_status = ?, error_code = ?, error_message = ?,
                    first_token_ms = ?, provider_chunk_count = ?,
                    input_tokens = ?, output_tokens = ?, total_tokens = ?, cost_usd = ?,
                    platform_cost_usd = ?, unpriced_tokens = ?
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
                    safe_input,
                    safe_output,
                    total_tokens,
                    safe_cost,
                    safe_platform_cost,
                    safe_unpriced,
                    run_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_run(self, run_id: str, *, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        conn = self._conn()
        try:
            if user_id is None:
                row = conn.execute(
                    """
                    SELECT id, user_id, conversation_id, trace_id, model, provider,
                           key_source, status, started_at, updated_at, completed_at,
                           duration_ms, tools_called, last_status, error_code,
                           error_message, client_ip, first_token_ms, provider_chunk_count,
                           input_tokens, output_tokens, total_tokens, cost_usd,
                           platform_cost_usd, unpriced_tokens
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
                           error_message, client_ip, first_token_ms, provider_chunk_count,
                           input_tokens, output_tokens, total_tokens, cost_usd,
                           platform_cost_usd, unpriced_tokens
                    FROM chat_runs WHERE id = ? AND user_id = ?
                    """,
                    (run_id, user_id),
                ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        keys = [
            "id", "user_id", "conversation_id", "trace_id", "model", "provider",
            "key_source", "status", "started_at", "updated_at", "completed_at",
            "duration_ms", "tools_called", "last_status", "error_code",
            "error_message", "client_ip", "first_token_ms", "provider_chunk_count",
            "input_tokens", "output_tokens", "total_tokens", "cost_usd",
            "platform_cost_usd", "unpriced_tokens",
        ]
        result = dict(zip(keys, row))
        try:
            result["tools_called"] = json.loads(result.get("tools_called") or "[]")
        except (TypeError, json.JSONDecodeError):
            result["tools_called"] = []
        return result

    def usage_rollup(
        self,
        *,
        since: Optional[str] = None,
        group_by: str = "model",
    ) -> Dict[str, Any]:
        """Aggregate per-turn tokens + cost across chat_runs, for admin cost
        reporting and benchmark cost-efficiency numbers.

        `cost_usd` covers only the priced portion, so the unpriced counters
        must be read alongside it. `unpriced_tokens` (token-level) is the honest
        measure: a turn can be partly priced (a TACC main model plus a priced
        OpenAI embedding) and still carry unpriced tokens, so cost_usd being
        non-zero does NOT mean the whole turn was priced. `unpriced_runs` counts
        runs that carry any unpriced tokens — this both flags those mixed turns
        and excludes zero-token failures (which spent nothing to price).

        `cost_usd` is every turn's full cost regardless of who paid, so it
        INCLUDES user-funded BYOK spend — do not read it as Quasar's bill.
        `platform_cost_usd` is the per-call-attributed subset the platform paid
        and is the number to bill against (CX-28). The two differ whenever a
        turn used a BYOK key, and a single turn can contribute to both (a BYOK
        main loop plus a platform embedding).
        """
        column = {
            "model": "model",
            "provider": "provider",
            "user": "user_id",
            "key_source": "key_source",
        }.get(group_by)
        if column is None:
            raise ValueError(f"Unsupported group_by: {group_by}")

        params: List[Any] = []
        where = "WHERE total_tokens IS NOT NULL"
        if since:
            # started_at is stored as UTC isoformat, so the comparison string
            # must also be UTC — a raw offset timestamp (e.g. "...-05:00")
            # compared lexically would select the wrong rows. Parse, normalize
            # to UTC, and reject malformed input instead of silently accepting
            # it. A bare date/naive timestamp is treated as UTC.
            try:
                parsed = datetime.fromisoformat(since.replace("Z", "+00:00"))
            except (TypeError, ValueError):
                raise ValueError(f"Invalid ISO-8601 timestamp: {since!r}")
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            where += " AND started_at >= ?"
            params.append(parsed.astimezone(timezone.utc).isoformat())

        conn = self._conn()
        try:
            rows = conn.execute(
                f"""
                SELECT {column},
                       COUNT(*),
                       COALESCE(SUM(input_tokens), 0),
                       COALESCE(SUM(output_tokens), 0),
                       COALESCE(SUM(total_tokens), 0),
                       COALESCE(SUM(cost_usd), 0),
                       COALESCE(SUM(platform_cost_usd), 0),
                       COALESCE(SUM(unpriced_tokens), 0),
                       SUM(CASE WHEN COALESCE(unpriced_tokens, 0) > 0 THEN 1 ELSE 0 END)
                FROM chat_runs
                {where}
                GROUP BY {column}
                ORDER BY SUM(total_tokens) DESC
                """,
                tuple(params),
            ).fetchall()
        finally:
            conn.close()

        # Keep each group's RAW cost for the total; round only per-group for
        # display. Summing already-rounded groups would zero a total made of
        # many sub-cent groups (CX-08) — e.g. 20 one-token turns across 20
        # user groups each round to $0 but really total ~$0.000003.
        raw_costs = [float(row[5] or 0.0) for row in rows or []]
        raw_platform_costs = [float(row[6] or 0.0) for row in rows or []]
        groups = [
            {
                group_by: row[0],
                "runs": int(row[1] or 0),
                "input_tokens": int(row[2] or 0),
                "output_tokens": int(row[3] or 0),
                "total_tokens": int(row[4] or 0),
                "cost_usd": round(float(row[5] or 0.0), 6),
                "platform_cost_usd": round(float(row[6] or 0.0), 6),
                "unpriced_tokens": int(row[7] or 0),
                "unpriced_runs": int(row[8] or 0),
            }
            for row in rows or []
        ]
        return {
            "group_by": group_by,
            "since": since or "",
            "groups": groups,
            "totals": {
                "runs": sum(g["runs"] for g in groups),
                "total_tokens": sum(g["total_tokens"] for g in groups),
                "cost_usd": round(sum(raw_costs), 6),
                "platform_cost_usd": round(sum(raw_platform_costs), 6),
                "unpriced_tokens": sum(g["unpriced_tokens"] for g in groups),
                "unpriced_runs": sum(g["unpriced_runs"] for g in groups),
            },
            "cost_is_estimate": True,
        }

    def build_feedback_snapshot(self, *, user_id: str, conversation_id: str,
                                message_id: str, run_id: str = "",
                                include_context: bool = False, description: str = ""):
        """Fail closed on consent/ownership; keep filing feedback on store outages."""
        if include_context is not True:
            return None, ""
        if self._snapshot_lookup is None:
            return None, "Full conversation capture is not configured"
        try:
            run = self.get_run(run_id, user_id=user_id) if run_id else None
            if run_id and not run:
                return None, "Chat run does not belong to reporter"
            conv_id = str(run.get("conversation_id") or "") if run else conversation_id
            if not conv_id:
                return None, "No owned conversation is linked to this feedback"
            source = self._snapshot_lookup(conv_id, user_id)
            if source is None:
                return None, "Conversation does not belong to reporter or was deleted"
            return build_snapshot(source=source, user_id=user_id, conversation_id=conv_id,
                                  message_id=_clean_text(message_id, 128), run=run,
                                  description=_clean_text(description)), ""
        except Exception:
            # Never put raw storage errors (which may contain credentials) in
            # client responses or the capture-status field.
            return None, "Conversation snapshot unavailable: storage read failed"

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
        include_context = include_context is True
        if category not in REPORT_CATEGORIES:
            raise ValueError("Unsupported issue category")
        description = _clean_text(description)
        if not description:
            raise ValueError("Issue description is required")
        # Identifiers are client strings as well: cap, redact, make bindable.
        message_id = _clean_text(message_id, 128)
        run = self.get_run(run_id, user_id=user_id)
        if not run:
            raise LookupError("Chat run not found")

        report_id = str(uuid.uuid4())
        now = _utc_now()
        safe_prompt = _clean_text(prompt_excerpt) if include_context else ""
        safe_response = _clean_text(response_excerpt, MAX_RESPONSE_CHARS) if include_context else ""
        excerpt: List[Dict[str, Any]] = []
        excerpt_error = ""
        if include_context:
            excerpt, excerpt_error = self._capture_conversation_excerpt(
                conversation_id=run.get("conversation_id") or "",
                run_id=run_id,
                user_id=user_id,
                response_excerpt=safe_response,
            )
        diagnostics = {
            "run_status": run.get("status"),
            "key_source": run.get("key_source"),
            "duration_ms": run.get("duration_ms"),
            "tools_called": run.get("tools_called") or [],
            "error_code": run.get("error_code"),
            "error_message": run.get("error_message"),
            "first_token_ms": run.get("first_token_ms"),
            "provider_chunk_count": run.get("provider_chunk_count"),
            "client": _bounded_client_context(technical_context),
        }
        if include_context:
            # The last SSE status label can embed tool arguments taken from the
            # user's request (e.g. the target name), so it counts as
            # conversation text and is stored only with consent.
            diagnostics["last_status"] = run.get("last_status")
        if excerpt_error:
            diagnostics["conversation_excerpt_error"] = excerpt_error
        snapshot, snapshot_error = self.build_feedback_snapshot(
            user_id=user_id, conversation_id=run.get("conversation_id") or "",
            message_id=message_id, run_id=run_id, include_context=include_context,
            description=description,
        )
        if snapshot:
            diagnostics["snapshot_id"] = snapshot["id"]
        if snapshot_error:
            diagnostics["snapshot_error"] = snapshot_error
        conn = self._conn()
        try:
            if snapshot:
                self.snapshots.insert(snapshot, conn=conn)
            conn.execute(
                """
                INSERT INTO issue_reports (
                    id, run_id, user_id, conversation_id, message_id, category,
                    description, include_context, prompt_excerpt, response_excerpt,
                    model, provider, trace_id, technical_context, status,
                    admin_notes, created_at, updated_at, conversation_excerpt
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    _serialize_diagnostics(diagnostics),
                    "new",
                    "",
                    now,
                    now,
                    json.dumps(excerpt, default=str) if excerpt else "",
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return self.get_report(report_id) or {}

    def _capture_conversation_excerpt(
        self,
        *,
        conversation_id: str,
        run_id: str,
        user_id: str,
        response_excerpt: str = "",
    ) -> Tuple[List[Dict[str, Any]], str]:
        """(excerpt, error) from the injected conversation store.

        Never raises: a report must still be filed when the store is
        unavailable, but the reason is recorded so the admin view can say
        "context unavailable" instead of showing an empty box. The lookup is
        ownership-scoped: the run row's conversation_id is trusted only as far
        as the store confirms it belongs to the reporter.
        """
        if self._conversation_lookup is None:
            return [], "conversation lookup not configured"
        if not conversation_id:
            return [], "run has no conversation_id"
        try:
            fetched = self._conversation_lookup(conversation_id, user_id)
        except Exception as exc:  # store outage must not block the report
            return [], _clean_text(f"conversation lookup failed: {exc}", 300)
        if fetched is None:
            return [], "conversation does not belong to reporter"
        messages = list(fetched or [])
        excerpt = build_conversation_excerpt(
            messages, run_id=run_id, response_excerpt=response_excerpt
        )
        if not excerpt:
            return [], "conversation has no persisted assistant message"
        return excerpt, ""

    def reports_exist_for(
        self,
        *,
        message_ids: Iterable[Any] = (),
        run_ids: Iterable[Any] = (),
        by_user: bool = False,
    ) -> Dict[str, Set[str]]:
        """Which of the given message ids / run ids already have an issue report.

        Exact lookups over the caller's ids (chunked IN queries on indexed
        columns) instead of a capped global DISTINCT scan, which produced
        false negatives once more than the cap had reports (CX-04). The
        run id is the stable key: a message's client id changes between the
        live turn and a reload (CX-02).
        """
        found: Dict[str, Set[str]] = {"message_ids": set(), "run_ids": set()}
        conn = self._conn()
        try:
            for column, values, key in (
                ("message_id", message_ids, "message_ids"),
                ("run_id", run_ids, "run_ids"),
            ):
                ids = [str(v) for v in (values or []) if v]
                for start in range(0, len(ids), 500):
                    chunk = ids[start:start + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    rows = conn.execute(
                        f"SELECT DISTINCT {column}, user_id FROM issue_reports WHERE {column} IN ({placeholders})",
                        tuple(chunk),
                    ).fetchall()
                    found[key].update((str(row[1] or ""), str(row[0])) if by_user else str(row[0])
                                      for row in rows if row and row[0])
        finally:
            conn.close()
        return found

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
        try:
            rows = conn.execute(
                f"""
                SELECT id, run_id, user_id, conversation_id, message_id, category,
                       description, include_context, prompt_excerpt, response_excerpt,
                       model, provider, trace_id, technical_context, status,
                       admin_notes, created_at, updated_at, conversation_excerpt
                FROM issue_reports
                {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        finally:
            conn.close()
        keys = [
            "id", "run_id", "user_id", "conversation_id", "message_id",
            "category", "description", "include_context", "prompt_excerpt",
            "response_excerpt", "model", "provider", "trace_id",
            "technical_context", "status", "admin_notes", "created_at", "updated_at",
            "conversation_excerpt",
        ]
        results = []
        for row in rows:
            item = dict(zip(keys, row))
            item["include_context"] = bool(item["include_context"])
            # parse_constant only fires for NaN/Infinity/-Infinity: a row
            # poisoned before the write-side guard existed must not take the
            # whole listing down (JSONResponse refuses non-finite floats).
            try:
                item["technical_context"] = json.loads(
                    item["technical_context"] or "{}", parse_constant=lambda _c: None
                )
            except (TypeError, json.JSONDecodeError):
                item["technical_context"] = {}
            if not isinstance(item["technical_context"], dict):
                item["technical_context"] = {}
            item["snapshot_id"] = item["technical_context"].get("snapshot_id", "")
            item["snapshot_error"] = item["technical_context"].get("snapshot_error", "")
            try:
                parsed_excerpt = json.loads(
                    item.get("conversation_excerpt") or "[]", parse_constant=lambda _c: None
                )
            except (TypeError, json.JSONDecodeError):
                parsed_excerpt = []
            item["conversation_excerpt"] = parsed_excerpt if isinstance(parsed_excerpt, list) else []
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
        try:
            cursor = conn.execute(
                f"UPDATE issue_reports SET {', '.join(assignments)} WHERE id = ?",
                tuple(params),
            )
            conn.commit()
        finally:
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
            "conversation_excerpt", "technical_context", "admin_notes",
        ]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            record = dict(row)
            record["technical_context"] = json.dumps(record.get("technical_context") or {})
            record["conversation_excerpt"] = json.dumps(record.get("conversation_excerpt") or [])
            writer.writerow(record)
        return output.getvalue()
