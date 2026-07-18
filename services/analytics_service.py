# services/analytics_service.py
"""
Analytics Service
Tracks page views (hit counter) and chat usage analytics.
Uses Turso (cloud) when TURSO_DATABASE_URL is set, else local SQLite.
"""

import csv
import io
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from services.admin_access import configured_admin_emails, is_admin_email
from services.db import get_connection
from pathlib import Path


# Admin email(s) allowed to access the analytics export
ADMIN_EMAILS = sorted(configured_admin_emails())

_LOCAL_DB = str(Path(__file__).resolve().parent.parent / "data" / "analytics.db")


class AnalyticsService:
    """
    Manages page-view counting and per-chat analytics logging.
    All data is persisted in Turso (production) or local SQLite (dev).
    """

    def __init__(self):
        self._init_db()
        # Cache the total page-view count in memory to avoid
        # a slow SELECT COUNT(*) on every /api/analytics/hit request.
        self._cached_total: int = self._load_total_views()

    # ── helpers ──────────────────────────────────────────────────

    def _conn(self):
        return get_connection(_LOCAL_DB)

    def _load_total_views(self) -> int:
        """One-time load of total page views from DB."""
        try:
            conn = self._conn()
            try:
                total = conn.execute("SELECT COUNT(*) FROM page_views").fetchone()[0]
            finally:
                conn.close()
            return total
        except Exception:
            return 0

    def _init_db(self):
        conn = self._conn()
        try:
            cur = conn.cursor()

            cur.execute("""
                CREATE TABLE IF NOT EXISTS page_views (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    ip_address TEXT,
                    user_agent TEXT,
                    country    TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_analytics (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id          TEXT,
                    username         TEXT,
                    email            TEXT,
                    display_name     TEXT,
                    ip_address       TEXT,
                    conversation_id  TEXT,
                    prompt           TEXT,
                    response_preview TEXT,
                    model            TEXT,
                    tools_called     TEXT,
                    response_time_ms INTEGER,
                    created_at       TEXT NOT NULL
                )
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_analytics_ts
                ON chat_analytics(created_at DESC)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_page_views_ts
                ON page_views(created_at DESC)
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS response_feedback (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id       TEXT NOT NULL,
                    conversation_id  TEXT,
                    user_id          TEXT,
                    feedback         TEXT NOT NULL,
                    model            TEXT,
                    prompt_preview   TEXT,
                    response_preview TEXT,
                    created_at       TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_ts
                ON response_feedback(created_at DESC)
            """)

            # Per-block 1-5 star ratings (Feature 4). Separate from
            # response_feedback: that one is a binary like/dislike keyed on the
            # whole message and stays in service for the thumbs bar.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS block_feedback (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    block_id         TEXT NOT NULL,
                    run_id           TEXT,
                    conversation_id  TEXT,
                    message_db_id    TEXT,
                    user_id          TEXT,
                    block_kind       TEXT,
                    rating           INTEGER NOT NULL,
                    comment          TEXT,
                    model            TEXT,
                    created_at       TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_block_feedback_block
                ON block_feedback(block_id)
            """)
            # One rating per user per block; re-rating replaces (delete+insert,
            # same idiom as log_feedback).
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_block_feedback_user_block
                ON block_feedback(user_id, block_id)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_block_feedback_run
                ON block_feedback(run_id)
            """)

            conn.commit()
        finally:
            conn.close()

    # ── page views / hit counter ─────────────────────────────────

    def log_page_view(
        self,
        ip_address: str = "",
        user_agent: str = "",
        country: str = "",
    ) -> int:
        """Log a page view and return the new total count (from memory cache)."""
        now = datetime.utcnow().isoformat()
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO page_views (ip_address, user_agent, country, created_at) "
                "VALUES (?, ?, ?, ?)",
                (ip_address, user_agent, country, now),
            )
            conn.commit()
        finally:
            conn.close()
        self._cached_total += 1
        return self._cached_total

    def get_total_views(self) -> int:
        conn = self._conn()
        try:
            total = conn.execute("SELECT COUNT(*) FROM page_views").fetchone()[0]
        finally:
            conn.close()
        return total

    def get_unique_visitors(self) -> int:
        conn = self._conn()
        try:
            total = conn.execute(
                "SELECT COUNT(DISTINCT ip_address) FROM page_views WHERE ip_address != ''"
            ).fetchone()[0]
        finally:
            conn.close()
        return total

    # ── chat analytics ───────────────────────────────────────────

    def log_chat(
        self,
        user_id: str = "anonymous",
        username: str = "",
        email: str = "",
        display_name: str = "",
        ip_address: str = "",
        conversation_id: str = "",
        prompt: str = "",
        response_preview: str = "",
        model: str = "",
        tools_called: Optional[List[str]] = None,
        response_time_ms: int = 0,
    ):
        """Persist one chat interaction for analytics."""
        now = datetime.utcnow().isoformat()
        conn = self._conn()
        try:
            conn.execute(
                """INSERT INTO chat_analytics
                   (user_id, username, email, display_name, ip_address,
                    conversation_id, prompt, response_preview, model,
                    tools_called, response_time_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id,
                    username,
                    email,
                    display_name,
                    ip_address,
                    conversation_id,
                    prompt,
                    response_preview[:500] if response_preview else "",
                    model,
                    json.dumps(tools_called or []),
                    response_time_ms,
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # ── admin export ─────────────────────────────────────────────

    def export_chat_analytics_csv(self) -> str:
        """Return all chat analytics as a CSV string."""
        conn = self._conn()
        try:
            rows = conn.execute(
                """SELECT created_at, user_id, username, email, display_name,
                          ip_address, conversation_id, prompt, response_preview,
                          model, tools_called, response_time_ms
                   FROM chat_analytics
                   ORDER BY created_at DESC"""
            ).fetchall()
        finally:
            conn.close()

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "timestamp", "user_id", "username", "email", "display_name",
            "ip_address", "conversation_id", "prompt", "response_preview",
            "model", "tools_called", "response_time_ms",
        ])
        for r in rows:
            writer.writerow(list(r))
        return buf.getvalue()

    def export_chat_analytics_json(self) -> List[Dict]:
        """Return all chat analytics as a list of dicts."""
        conn = self._conn()
        try:
            rows = conn.execute(
                """SELECT created_at, user_id, username, email, display_name,
                          ip_address, conversation_id, prompt, response_preview,
                          model, tools_called, response_time_ms
                   FROM chat_analytics
                   ORDER BY created_at DESC"""
            ).fetchall()
        finally:
            conn.close()

        keys = [
            "timestamp", "user_id", "username", "email", "display_name",
            "ip_address", "conversation_id", "prompt", "response_preview",
            "model", "tools_called", "response_time_ms",
        ]
        return [dict(zip(keys, r)) for r in rows]

    def get_summary(self) -> Dict:
        """Aggregated analytics for the admin dashboard."""
        conn = self._conn()
        try:
            cur = conn.cursor()

            total_chats = cur.execute("SELECT COUNT(*) FROM chat_analytics").fetchone()[0]
            unique_users = cur.execute(
                "SELECT COUNT(DISTINCT user_id) FROM chat_analytics WHERE user_id != 'anonymous'"
            ).fetchone()[0]
            anon_chats = cur.execute(
                "SELECT COUNT(*) FROM chat_analytics WHERE user_id = 'anonymous'"
            ).fetchone()[0]

            # Top models
            top_models = cur.execute(
                "SELECT model, COUNT(*) as cnt FROM chat_analytics "
                "WHERE model != '' GROUP BY model ORDER BY cnt DESC LIMIT 5"
            ).fetchall()

            # Top tools
            # tools_called is a JSON array — we count occurrences across all rows
            all_tools_rows = cur.execute(
                "SELECT tools_called FROM chat_analytics WHERE tools_called != '[]'"
            ).fetchall()
            tool_counts: Dict[str, int] = {}
            for (tools_json,) in all_tools_rows:
                try:
                    for t in json.loads(tools_json):
                        tool_counts[t] = tool_counts.get(t, 0) + 1
                except (json.JSONDecodeError, TypeError):
                    pass
            top_tools = sorted(tool_counts.items(), key=lambda x: -x[1])[:10]

            # Page views
            total_views = cur.execute("SELECT COUNT(*) FROM page_views").fetchone()[0]
            unique_ips = cur.execute(
                "SELECT COUNT(DISTINCT ip_address) FROM page_views WHERE ip_address != ''"
            ).fetchone()[0]

        finally:
            conn.close()

        return {
            "total_chats": total_chats,
            "unique_users": unique_users,
            "anonymous_chats": anon_chats,
            "top_models": [{"model": m, "count": c} for m, c in top_models],
            "top_tools": [{"tool": t, "count": c} for t, c in top_tools],
            "total_page_views": total_views,
            "unique_visitors": unique_ips,
            "feedback": self._get_feedback_summary(),
        }

    # ── admin gate ───────────────────────────────────────────────

    @staticmethod
    def is_admin(email: str) -> bool:
        """Check if the given email is an admin."""
        return is_admin_email(email)

    # ── response feedback (like/dislike) ─────────────────────────

    def log_feedback(
        self,
        message_id: str,
        feedback: str,
        conversation_id: str = "",
        user_id: str = "anonymous",
        model: str = "",
        prompt_preview: str = "",
        response_preview: str = "",
    ):
        """Persist a like/dislike on a specific assistant response."""
        now = datetime.utcnow().isoformat()
        conn = self._conn()
        try:
            # Upsert: if user already voted on this message, update it
            conn.execute(
                "DELETE FROM response_feedback WHERE message_id = ? AND user_id = ?",
                (message_id, user_id),
            )
            conn.execute(
                """INSERT INTO response_feedback
                   (message_id, conversation_id, user_id, feedback, model,
                    prompt_preview, response_preview, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    message_id,
                    conversation_id,
                    user_id,
                    feedback,
                    model,
                    prompt_preview[:200] if prompt_preview else "",
                    response_preview[:500] if response_preview else "",
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _get_feedback_summary(self) -> Dict:
        """Return aggregated feedback counts."""
        try:
            conn = self._conn()
            try:
                likes = conn.execute(
                    "SELECT COUNT(*) FROM response_feedback WHERE feedback = 'like'"
                ).fetchone()[0]
                dislikes = conn.execute(
                    "SELECT COUNT(*) FROM response_feedback WHERE feedback = 'dislike'"
                ).fetchone()[0]
            finally:
                conn.close()
            return {"likes": likes, "dislikes": dislikes, "total": likes + dislikes}
        except Exception:
            return {"likes": 0, "dislikes": 0, "total": 0}

    # ── per-block star ratings (Feature 4) ───────────────────────

    def log_block_feedback(
        self,
        block_id: str,
        rating: int,
        run_id: str = "",
        conversation_id: str = "",
        message_db_id: str = "",
        user_id: str = "anonymous",
        block_kind: str = "",
        comment: str = "",
        model: str = "",
    ) -> None:
        """Persist a 1-5 star rating on one block. Re-rating replaces.

        Raises ValueError on an out-of-range rating so a bad client can't
        poison the label set with a 0 or an 11.
        """
        # Strict: int() would quietly coerce 4.9 -> 4 and JSON true -> 1, both of
        # which land in the label set as a real human judgement that nobody made.
        # bool is an int subclass, hence the explicit exclusion.
        if isinstance(rating, bool) or not isinstance(rating, int):
            raise ValueError("rating must be an integer 1-5")
        safe_rating = rating
        if not 1 <= safe_rating <= 5:
            raise ValueError("rating must be between 1 and 5")
        if not block_id:
            raise ValueError("block_id is required")

        now = datetime.utcnow().isoformat()
        conn = self._conn()
        try:
            # Upsert: same user re-rating the same block overwrites their vote.
            conn.execute(
                "DELETE FROM block_feedback WHERE block_id = ? AND user_id = ?",
                (block_id, user_id),
            )
            conn.execute(
                """INSERT INTO block_feedback
                   (block_id, run_id, conversation_id, message_db_id, user_id,
                    block_kind, rating, comment, model, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    block_id,
                    run_id or "",
                    conversation_id or "",
                    message_db_id or "",
                    user_id,
                    block_kind or "",
                    safe_rating,
                    (comment or "")[:2000],
                    model or "",
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def export_block_feedback(self, user_id: Optional[str] = None) -> List[Dict]:
        """Return block ratings, newest first. `user_id` scopes to one rater."""
        conn = self._conn()
        try:
            sql = (
                """SELECT created_at, block_id, run_id, conversation_id,
                          message_db_id, user_id, block_kind, rating, comment, model
                   FROM block_feedback"""
            )
            params: tuple = ()
            if user_id:
                sql += " WHERE user_id = ?"
                params = (user_id,)
            sql += " ORDER BY created_at DESC"
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        keys = [
            "timestamp", "block_id", "run_id", "conversation_id",
            "message_db_id", "user_id", "block_kind", "rating", "comment", "model",
        ]
        return [dict(zip(keys, r)) for r in rows]

    def get_block_ratings(self, conversation_id: str, user_id: str) -> Dict[str, Dict]:
        """block_id -> this user's rating, for one whole conversation.

        Conversation-scoped rather than run-scoped because that is what a reload
        needs: the client rehydrates every turn at once and has to re-light the
        stars on all of them.
        """
        if not conversation_id:
            return {}
        conn = self._conn()
        try:
            rows = conn.execute(
                """SELECT block_id, rating, comment, block_kind
                   FROM block_feedback WHERE conversation_id = ? AND user_id = ?""",
                (conversation_id, user_id),
            ).fetchall()
        finally:
            conn.close()
        return {
            r[0]: {"block_id": r[0], "rating": r[1], "comment": r[2], "block_kind": r[3]}
            for r in rows
        }

    def export_feedback_json(self) -> List[Dict]:
        """Return all feedback entries as a list of dicts."""
        conn = self._conn()
        try:
            rows = conn.execute(
                """SELECT created_at, message_id, conversation_id, user_id,
                          feedback, model, prompt_preview, response_preview
                   FROM response_feedback
                   ORDER BY created_at DESC"""
            ).fetchall()
        finally:
            conn.close()
        keys = [
            "timestamp", "message_id", "conversation_id", "user_id",
            "feedback", "model", "prompt_preview", "response_preview",
        ]
        return [dict(zip(keys, r)) for r in rows]
