"""Weekly LLM token usage accounting and quota enforcement."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from services.admin_access import is_admin_email, is_quota_exempt_email
from services.db import get_connection


_LOCAL_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "usage_quota.db")

PLATFORM_TOKEN_LIMITS = {
    "deepseek": 500_000,
    "openai": 100_000,
    "tacc": 1_000_000,
}

PLATFORM_QUOTA_WINDOW_DAYS = 7


class QuotaExceededError(RuntimeError):
    """Raised when a user cannot spend more tokens for the selected key source."""


@dataclass
class UsageRecord:
    user_id: str
    provider: str
    model: str
    key_source: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return max(0, int(self.input_tokens or 0)) + max(0, int(self.output_tokens or 0))


class UsageQuotaService:
    """Tracks usage and enforces rolling weekly platform-key quotas."""

    def __init__(self):
        self._init_db()

    def _conn(self):
        return get_connection(_LOCAL_DB)

    def _init_db(self) -> None:
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_usage_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                key_source TEXT NOT NULL,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_llm_usage_user_provider_source
            ON llm_usage_events(user_id, provider, key_source)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_llm_usage_user_provider_source_created
            ON llm_usage_events(user_id, provider, key_source, created_at)
            """
        )
        conn.commit()
        conn.close()

    @staticmethod
    def normalize_provider(provider: str) -> str:
        value = (provider or "").strip().lower()
        if value == "gemini":
            return "google"
        if value in {"tejas", "texas", "texas_ai"}:
            return "tacc"
        return value

    @staticmethod
    def normalize_key_source(key_source: str) -> str:
        value = (key_source or "platform").strip().lower()
        return "byok" if value == "byok" else "platform"

    @staticmethod
    def weekly_window_start() -> datetime:
        return datetime.now(timezone.utc) - timedelta(days=PLATFORM_QUOTA_WINDOW_DAYS)

    def get_used_tokens(
        self,
        user_id: str,
        provider: str,
        key_source: str,
        *,
        since: Optional[datetime] = None,
    ) -> int:
        provider = self.normalize_provider(provider)
        key_source = self.normalize_key_source(key_source)
        conn = self._conn()
        params = [user_id, provider, key_source]
        since_clause = ""
        if since is not None:
            since_clause = " AND created_at >= ?"
            params.append(since.astimezone(timezone.utc).isoformat())
        row = conn.execute(
            f"""
            SELECT COALESCE(SUM(total_tokens), 0)
            FROM llm_usage_events
            WHERE user_id = ? AND provider = ? AND key_source = ?{since_clause}
            """,
            tuple(params),
        ).fetchone()
        conn.close()
        return int(row[0] or 0) if row else 0

    def ensure_allowed(
        self,
        *,
        user_id: str,
        user_email: str = "",
        provider: str,
        key_source: str,
        byok_token_limit: Optional[int] = None,
    ) -> None:
        provider = self.normalize_provider(provider)
        key_source = self.normalize_key_source(key_source)

        if is_quota_exempt_email(user_email):
            return

        if key_source == "platform":
            limit = PLATFORM_TOKEN_LIMITS.get(provider)
            if limit is None:
                return
            used = self.get_used_tokens(
                user_id,
                provider,
                "platform",
                since=self.weekly_window_start(),
            )
            if used >= limit:
                provider_label = {
                    "deepseek": "DeepSeek",
                    "openai": "OpenAI",
                    "tacc": "TACC",
                }.get(provider, provider)
                raise QuotaExceededError(
                    "You have exhausted your included Quasar token allowance. "
                    f"Your {provider_label} platform-key allowance is {limit:,} tokens per week. "
                    "Please try again after your weekly quota window resets."
                )
            return

        if byok_token_limit is not None:
            used = self.get_used_tokens(user_id, provider, "byok")
            if used >= int(byok_token_limit):
                raise QuotaExceededError(
                    "Your self-imposed API key token limit has been reached. "
                    "Raise or clear the limit in Provider Keys to continue."
                )

    def record_usage(self, record: UsageRecord) -> None:
        provider = self.normalize_provider(record.provider)
        key_source = self.normalize_key_source(record.key_source)
        total = record.total_tokens
        if total <= 0:
            return
        conn = self._conn()
        conn.execute(
            """
            INSERT INTO llm_usage_events
            (user_id, provider, model, key_source, input_tokens, output_tokens, total_tokens, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.user_id,
                provider,
                record.model or "",
                key_source,
                max(0, int(record.input_tokens or 0)),
                max(0, int(record.output_tokens or 0)),
                total,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        conn.close()

    def usage_summary(
        self,
        *,
        user_id: str,
        user_email: str = "",
        provider_key_metadata: Optional[list[dict]] = None,
    ) -> Dict:
        byok_limits = {
            item.get("provider"): item.get("token_limit")
            for item in (provider_key_metadata or [])
            if item.get("provider")
        }
        providers = sorted(set(PLATFORM_TOKEN_LIMITS) | set(byok_limits))
        platform: Dict[str, Dict] = {}
        byok: Dict[str, Dict] = {}
        quota_exempt = is_quota_exempt_email(user_email)
        admin = is_admin_email(user_email)
        window_start = self.weekly_window_start()
        for provider in providers:
            platform_limit = PLATFORM_TOKEN_LIMITS.get(provider)
            platform_used = self.get_used_tokens(user_id, provider, "platform", since=window_start)
            platform[provider] = {
                "used_tokens": platform_used,
                "limit_tokens": None if quota_exempt else platform_limit,
                "unlimited": quota_exempt or platform_limit is None,
                "exhausted": False if quota_exempt or platform_limit is None else platform_used >= platform_limit,
            }
            byok_limit = byok_limits.get(provider)
            byok_used = self.get_used_tokens(user_id, provider, "byok")
            byok[provider] = {
                "used_tokens": byok_used,
                "limit_tokens": byok_limit,
                "unlimited": byok_limit is None,
                "exhausted": False if byok_limit is None else byok_used >= int(byok_limit),
            }
        return {
            "platform": platform,
            "byok": byok,
            "is_admin": admin,
            "is_quota_exempt": quota_exempt,
            "platform_quota_window_days": PLATFORM_QUOTA_WINDOW_DAYS,
        }
