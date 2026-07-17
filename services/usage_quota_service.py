"""LLM token usage accounting and quota enforcement (rolling weekly + daily)."""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from services.admin_access import is_admin_email, is_quota_exempt_email
from services.db import get_connection
from services.model_pricing import PRICING_LAST_VERIFIED, estimate_cost

logger = logging.getLogger(__name__)


_LOCAL_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "usage_quota.db")

PLATFORM_TOKEN_LIMITS = {
    "deepseek": 500_000,
    "openai": 100_000,
    "tacc": 1_000_000,
}

PLATFORM_QUOTA_WINDOW_DAYS = 7

# Optional per-provider daily caps, mirroring PLATFORM_TOKEN_LIMITS' shape.
# Intentionally EMPTY: the daily cap ships disabled and is opt-in via
# QUASAR_DAILY_TOKEN_LIMIT, so merging this changes no deployment's behaviour.
# Populating a provider here adds a daily cap for that provider on top of both
# the weekly quota and the aggregate cap.
PLATFORM_DAILY_TOKEN_LIMITS: Dict[str, int] = {}

PLATFORM_DAILY_QUOTA_WINDOW_HOURS = 24

# Tokens held per in-flight platform LLM call so concurrent calls can see each
# other's pending spend. See _try_reserve for why this exists and why the
# default is a flat estimate rather than the call's true worst case.
DEFAULT_CALL_TOKEN_RESERVATION = 4_000

# How long a reservation counts before it is treated as abandoned. A crashed or
# disconnected turn never settles its reservation, so without an expiry the
# holder's own headroom would be consumed until the window rolled. Comfortably
# longer than any real LLM call.
CALL_RESERVATION_TTL_SECONDS = 600


def call_token_reservation() -> int:
    """Tokens to hold per in-flight platform call. <=0 disables reservations."""
    raw = os.getenv("QUASAR_CALL_TOKEN_RESERVATION")
    if raw is None or not str(raw).strip():
        return DEFAULT_CALL_TOKEN_RESERVATION
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        # Same rationale as daily_token_limit: an operator typo must not lock
        # every user out, so fall back to the default rather than fail closed.
        logger.warning(
            "QUASAR_CALL_TOKEN_RESERVATION=%r is not an integer; using default %d.",
            raw,
            DEFAULT_CALL_TOKEN_RESERVATION,
        )
        return DEFAULT_CALL_TOKEN_RESERVATION
    return max(0, value)


def daily_token_limit() -> Optional[int]:
    """Aggregate platform-key daily token cap across all providers.

    Absent, empty, 0, or negative -> None (disabled). Env-only by design: no
    in-code default, so this cannot start throttling an existing deployment
    without an explicit operator action.
    """
    raw = os.getenv("QUASAR_DAILY_TOKEN_LIMIT")
    if raw is None or not str(raw).strip():
        return None
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        # Operator typo. Warn loudly rather than fail closed and lock every
        # user out of a working deployment; the value is operator config, not
        # attacker-reachable input.
        logger.warning(
            "QUASAR_DAILY_TOKEN_LIMIT=%r is not an integer; daily cap disabled.", raw
        )
        return None
    return value if value > 0 else None


class QuotaExceededError(RuntimeError):
    """Raised when a user cannot spend more tokens for the selected key source."""


def _provider_label(provider: str) -> str:
    return {
        "deepseek": "DeepSeek",
        "openai": "OpenAI",
        "tacc": "TACC",
        "anthropic": "Anthropic",
        "google": "Google",
    }.get(provider, provider)


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
        try:
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
            # The aggregate daily cap sums across ALL providers, so it filters on
            # (user_id, key_source, created_at) with no provider — the index
            # above (provider between user and key_source) can't serve it. This
            # one keeps the per-request cap check from scanning a user's whole
            # historical ledger as it grows.
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_llm_usage_user_source_created
                ON llm_usage_events(user_id, key_source, created_at)
                """
            )
            # Tokens held by calls that are in flight but not yet recorded. Kept
            # in its OWN table, deliberately not as a flag on llm_usage_events:
            # reservations are estimates, and every rollup/cost report reads that
            # ledger as a record of tokens actually spent. A reservation landing
            # there would inflate spend and bill an estimate as a fact.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_usage_reservations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    key_source TEXT NOT NULL,
                    tokens INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )
            # Serves both reservation sums: the aggregate cap (user_id,
            # key_source, expires_at) and the per-provider caps, which use the
            # same prefix plus provider.
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_llm_reservations_user_source_expires
                ON llm_usage_reservations(user_id, key_source, expires_at, provider)
                """
            )
            conn.commit()
        finally:
            conn.close()
        self._purge_expired_reservations()

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

    @staticmethod
    def daily_window_start() -> datetime:
        """Start of the rolling daily window (matches the weekly window's
        rolling semantics — not a calendar day, so there is no midnight reset
        to burst against)."""
        return datetime.now(timezone.utc) - timedelta(
            hours=PLATFORM_DAILY_QUOTA_WINDOW_HOURS
        )

    def get_used_tokens_all_providers(
        self,
        user_id: str,
        key_source: str,
        *,
        since: Optional[datetime] = None,
    ) -> int:
        """Total tokens for a user across every provider for one key source."""
        key_source = self.normalize_key_source(key_source)
        conn = self._conn()
        try:
            params = [user_id, key_source]
            since_clause = ""
            if since is not None:
                since_clause = " AND created_at >= ?"
                params.append(since.astimezone(timezone.utc).isoformat())
            row = conn.execute(
                f"""
                SELECT COALESCE(SUM(total_tokens), 0)
                FROM llm_usage_events
                WHERE user_id = ? AND key_source = ?{since_clause}
                """,
                tuple(params),
            ).fetchone()
        finally:
            conn.close()
        return int(row[0] or 0) if row else 0

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
        try:
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
        finally:
            conn.close()
        return int(row[0] or 0) if row else 0

    def _purge_expired_reservations(self) -> None:
        """Drop abandoned reservations. Housekeeping only — every read filters on
        `expires_at`, so correctness never depends on this having run."""
        conn = self._conn()
        try:
            conn.execute(
                "DELETE FROM llm_usage_reservations WHERE expires_at <= ?",
                (datetime.now(timezone.utc).isoformat(),),
            )
            conn.commit()
        except Exception as exc:  # pragma: no cover - housekeeping must not fail a turn
            logger.warning("[quota] Failed to purge expired reservations: %s", exc)
        finally:
            conn.close()

    def _platform_caps(self, provider: str) -> List[Tuple[Optional[str], datetime, int]]:
        """Every platform cap that applies to `provider`, as
        (provider_scope, window_start, limit). `provider_scope` None means the
        cap sums across all providers.

        Order matters: it is the order _raise_for_exceeded_platform_cap reports
        in, and mirrors the original daily-before-weekly check order.
        """
        caps: List[Tuple[Optional[str], datetime, int]] = []
        aggregate_limit = daily_token_limit()
        if aggregate_limit is not None:
            caps.append((None, self.daily_window_start(), aggregate_limit))
        provider_daily = PLATFORM_DAILY_TOKEN_LIMITS.get(provider)
        if provider_daily is not None:
            caps.append((provider, self.daily_window_start(), provider_daily))
        weekly = PLATFORM_TOKEN_LIMITS.get(provider)
        if weekly is not None:
            caps.append((provider, self.weekly_window_start(), weekly))
        return caps

    def _try_reserve(
        self,
        user_id: str,
        provider: str,
        tokens: int,
        caps: List[Tuple[Optional[str], datetime, int]],
    ) -> Optional[str]:
        """Atomically admit one platform call and hold `tokens` against it.

        Returns a reservation id, or None if admitting would breach a cap.

        This is ONE conditional INSERT rather than a read-then-insert because
        read-then-insert is the CX-29 bug: concurrent callers each read the same
        under-limit total and every one of them passes. As a single statement the
        admission decision and the record of it cannot be split — SQLite
        serializes it under the write lock, and Turso's HTTP transport has no
        interactive transaction, so a single statement is the only atomic
        primitive available on both.

        The predicate is `used + outstanding < limit`, NOT
        `used + outstanding + tokens <= limit`. That keeps the sequential
        admission rule byte-for-byte identical to the pre-reservation behaviour,
        so nobody is cut off earlier than before; reservations only make
        concurrent in-flight calls visible to each other. The effect is that
        allowed concurrency scales with remaining headroom (~headroom/tokens
        calls), so a user at 0 usage is never falsely rejected while a user with
        one call's worth left admits exactly one.
        """
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        expires_iso = (
            now + timedelta(seconds=CALL_RESERVATION_TTL_SECONDS)
        ).isoformat()
        reservation_id = str(uuid.uuid4())

        conditions: List[str] = []
        params: List = [
            reservation_id, user_id, provider, "platform", int(tokens),
            now_iso, expires_iso,
        ]
        for provider_scope, window_start, limit in caps:
            window_iso = window_start.astimezone(timezone.utc).isoformat()
            provider_filter = " AND provider = ?" if provider_scope else ""
            conditions.append(
                f"""
                (COALESCE((SELECT SUM(total_tokens) FROM llm_usage_events
                           WHERE user_id = ? AND key_source = 'platform'
                                 {provider_filter} AND created_at >= ?), 0)
                 + COALESCE((SELECT SUM(tokens) FROM llm_usage_reservations
                             WHERE user_id = ? AND key_source = 'platform'
                                   {provider_filter} AND expires_at > ?), 0)) < ?
                """
            )
            params.append(user_id)
            if provider_scope:
                params.append(provider_scope)
            params.append(window_iso)
            params.append(user_id)
            if provider_scope:
                params.append(provider_scope)
            params.append(now_iso)
            params.append(int(limit))

        where = " AND ".join(conditions)
        conn = self._conn()
        try:
            cur = conn.execute(
                f"""
                INSERT INTO llm_usage_reservations
                (id, user_id, provider, key_source, tokens, created_at, expires_at)
                SELECT ?, ?, ?, ?, ?, ?, ?
                WHERE {where}
                """,
                tuple(params),
            )
            # Turso buffers DML and only reports rows_affected once commit()
            # flushes the batch, so rowcount must be read after the commit.
            conn.commit()
            admitted = cur.rowcount == 1
        finally:
            conn.close()
        return reservation_id if admitted else None

    def release_reservation(self, reservation_id: Optional[str]) -> None:
        """Free a reservation whose call never recorded usage (it failed, was
        cancelled, or reported no tokens). Safe to call twice."""
        if not reservation_id:
            return
        conn = self._conn()
        try:
            conn.execute(
                "DELETE FROM llm_usage_reservations WHERE id = ?", (reservation_id,)
            )
            conn.commit()
        except Exception as exc:
            # A leaked reservation self-heals at the TTL, so this must never
            # propagate into the caller's request.
            logger.warning("[quota] Failed to release reservation: %s", exc)
        finally:
            conn.close()

    def outstanding_reservation_tokens(
        self, user_id: str, key_source: str = "platform", *, provider: Optional[str] = None
    ) -> int:
        """Unexpired reserved tokens for a user. Diagnostic/testing helper."""
        key_source = self.normalize_key_source(key_source)
        params = [user_id, key_source, datetime.now(timezone.utc).isoformat()]
        provider_filter = ""
        if provider is not None:
            provider_filter = " AND provider = ?"
            params.append(self.normalize_provider(provider))
        conn = self._conn()
        try:
            row = conn.execute(
                f"""
                SELECT COALESCE(SUM(tokens), 0) FROM llm_usage_reservations
                WHERE user_id = ? AND key_source = ? AND expires_at > ?{provider_filter}
                """,
                tuple(params),
            ).fetchone()
        finally:
            conn.close()
        return int(row[0] or 0) if row else 0

    def _enforce_daily_caps(self, user_id: str, provider: str) -> None:
        """Raise QuotaExceededError if the user is over a daily platform cap.

        Callers MUST already have cleared the quota-exempt and BYOK checks —
        this method knows nothing about either, and enforces unconditionally.
        """
        window_start = self.daily_window_start()

        aggregate_limit = daily_token_limit()
        if aggregate_limit is not None:
            used = self.get_used_tokens_all_providers(
                user_id, "platform", since=window_start
            )
            if used >= aggregate_limit:
                raise QuotaExceededError(
                    "You have reached your daily Quasar token limit of "
                    f"{aggregate_limit:,} tokens. This limit resets on a rolling "
                    "24-hour basis. Please try again later, or add your own "
                    "provider API key in Settings to continue right away."
                )

        provider_limit = PLATFORM_DAILY_TOKEN_LIMITS.get(provider)
        if provider_limit is not None:
            used = self.get_used_tokens(
                user_id, provider, "platform", since=window_start
            )
            if used >= provider_limit:
                raise QuotaExceededError(
                    "You have reached your daily Quasar token limit for "
                    f"{_provider_label(provider)} ({provider_limit:,} tokens). "
                    "This limit resets on a rolling 24-hour basis. Please try "
                    "again later, or add your own provider API key in Settings "
                    "to continue right away."
                )

    def _raise_for_exceeded_platform_cap(self, user_id: str, provider: str) -> None:
        """Raise the specific error for whichever platform cap is exhausted.

        Read-only diagnosis. It runs only when reservations are disabled or when
        admission has already been refused, so its extra queries stay off the
        admitted path.
        """
        # Daily cap is checked FIRST, and deliberately ahead of the
        # `limit is None` early return below: providers absent from
        # PLATFORM_TOKEN_LIMITS (anthropic, google, local) have no weekly
        # quota and would otherwise return before any daily check ran,
        # making the aggregate cap trivially bypassable by picking such a
        # provider. The daily cap stacks with the weekly quota — either can
        # trip, and neither loosens the other.
        self._enforce_daily_caps(user_id, provider)

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
            raise QuotaExceededError(
                "You have exhausted your included Quasar token allowance. "
                f"Your {_provider_label(provider)} platform-key allowance is "
                f"{limit:,} tokens per week. "
                "Please try again after your weekly quota window resets."
            )

    def ensure_allowed(
        self,
        *,
        user_id: str,
        user_email: str = "",
        provider: str,
        key_source: str,
        byok_token_limit: Optional[int] = None,
        reserve: bool = False,
    ) -> Optional[str]:
        """Admit one unit of work, or raise QuotaExceededError.

        With `reserve=True` the admission is atomic and holds tokens against the
        caller until it settles — pass the returned id to `record_usage`, or to
        `release_reservation` if the call never records. Callers that only
        pre-flight a request (and have no settle point) leave it False and get
        the historical read-only check.

        Returns a reservation id when one was taken, else None.
        """
        provider = self.normalize_provider(provider)
        key_source = self.normalize_key_source(key_source)

        if is_quota_exempt_email(user_email):
            return None

        if key_source == "platform":
            caps = self._platform_caps(provider)
            if not caps:
                # No cap applies (e.g. anthropic/google while the daily cap is
                # disabled). Nothing to enforce, so nothing to reserve against —
                # uncapped providers keep their zero-write fast path.
                return None

            tokens = call_token_reservation() if reserve else 0
            if tokens <= 0:
                self._raise_for_exceeded_platform_cap(user_id, provider)
                return None

            handle = self._try_reserve(user_id, provider, tokens, caps)
            if handle is not None:
                return handle

            # Refused. Name the cap that is actually exhausted so the user gets
            # the same specific message as before.
            self._raise_for_exceeded_platform_cap(user_id, provider)
            # No cap is exhausted on the ledger, so the refusal came from other
            # calls of this user's own holding the remaining headroom. That is
            # transient, and says so — telling them their allowance is gone
            # would be wrong, and they may well be far from it.
            raise QuotaExceededError(
                "Too many requests are in flight against your remaining Quasar "
                "token allowance. Please retry in a moment, or add your own "
                "provider API key in Settings to continue right away."
            )

        # BYOK from here down. The daily platform cap intentionally does not
        # reach this branch: the user is paying their own provider, so Quasar
        # has no cost to cap. Only their own self-imposed limit applies, and it
        # is left as a plain read: overshooting a self-imposed limit spends the
        # user's own money, so it does not warrant holding tokens.
        if byok_token_limit is not None:
            used = self.get_used_tokens(user_id, provider, "byok")
            if used >= int(byok_token_limit):
                raise QuotaExceededError(
                    "Your self-imposed API key token limit has been reached. "
                    "Raise or clear the limit in Provider Keys to continue."
                )

    def record_usage(
        self, record: UsageRecord, *, reservation_id: Optional[str] = None
    ) -> None:
        """Record actual spend, settling `reservation_id` if one was held.

        The reservation is dropped and the real event written in a SINGLE
        commit — one atomic batch on Turso, one transaction on SQLite. Doing it
        as two commits would open a window where the estimate no longer counts
        but the actual spend does not yet, letting a concurrent call be admitted
        against headroom that was already gone.
        """
        provider = self.normalize_provider(record.provider)
        key_source = self.normalize_key_source(record.key_source)
        total = record.total_tokens
        if total <= 0:
            # Nothing spent, but the hold must still come off.
            self.release_reservation(reservation_id)
            return
        conn = self._conn()
        try:
            if reservation_id:
                conn.execute(
                    "DELETE FROM llm_usage_reservations WHERE id = ?",
                    (reservation_id,),
                )
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
        finally:
            conn.close()

    def cost_breakdown(
        self,
        user_id: str,
        key_source: str,
        *,
        since: Optional[datetime] = None,
    ) -> Dict:
        """Estimated spend for a user/key-source since `since`.

        `cost_usd` sums only the tokens we can price; `unpriced_tokens` reports
        what was left out (TACC, local, unknown models). Callers must surface
        both — reporting the cost alone reads as a complete figure when it may
        cover a fraction of the tokens actually spent.
        """
        key_source = self.normalize_key_source(key_source)
        conn = self._conn()
        try:
            params = [user_id, key_source]
            since_clause = ""
            if since is not None:
                since_clause = " AND created_at >= ?"
                params.append(since.astimezone(timezone.utc).isoformat())
            rows = conn.execute(
                f"""
                SELECT provider, model,
                       COALESCE(SUM(input_tokens), 0),
                       COALESCE(SUM(output_tokens), 0),
                       COALESCE(SUM(total_tokens), 0)
                FROM llm_usage_events
                WHERE user_id = ? AND key_source = ?{since_clause}
                GROUP BY provider, model
                """,
                tuple(params),
            ).fetchall()
        finally:
            conn.close()

        total_cost = 0.0
        unpriced_tokens = 0
        total_tokens = 0
        by_provider: Dict[str, float] = {}
        for provider, model, tokens_in, tokens_out, tokens_total in rows or []:
            total_tokens += int(tokens_total or 0)
            # estimate_cost is unrounded; accumulate raw, round once below, so
            # many sub-cent groups don't each round to 0.
            cost = estimate_cost(provider, model, tokens_in, tokens_out)
            if cost is None:
                unpriced_tokens += int(tokens_total or 0)
                continue
            total_cost += cost
            by_provider[provider] = by_provider.get(provider, 0.0) + cost

        return {
            "cost_usd": round(total_cost, 6),
            "by_provider_usd": {p: round(c, 6) for p, c in by_provider.items()},
            "total_tokens": total_tokens,
            "unpriced_tokens": unpriced_tokens,
        }

    def global_usage_rollup(
        self,
        *,
        since: Optional[str] = None,
        group_by: str = "provider",
    ) -> Dict:
        """All-user usage rollup sourced from the PER-CALL ledger.

        The admin cost view must attribute each LLM call to the model/provider/
        key_source that actually ran it. A `chat_runs`-based rollup can't: it
        stores one route per turn, so a turn's auxiliary calls (an OpenAI
        embedding inside an Anthropic-BYOK turn) get lumped under the selected
        route — hiding platform-paid spend under a BYOK label. `llm_usage_events`
        records every call individually, so grouping it is honest. Cost is
        priced per (provider, model) sub-group so unpriced models (TACC/local)
        surface as unpriced tokens, never as $0.
        """
        column = {
            "provider": "provider",
            "model": "model",
            "key_source": "key_source",
            "user": "user_id",
        }.get(group_by)
        if column is None:
            raise ValueError(f"Unsupported group_by: {group_by}")

        params: list = []
        since_clause = ""
        if since:
            try:
                parsed = datetime.fromisoformat(since.replace("Z", "+00:00"))
            except (TypeError, ValueError):
                raise ValueError(f"Invalid ISO-8601 timestamp: {since!r}")
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            since_clause = " AND created_at >= ?"
            params.append(parsed.astimezone(timezone.utc).isoformat())

        conn = self._conn()
        try:
            # Group by column + (provider, model) so each sub-row can be priced,
            # then fold up to the column level.
            rows = conn.execute(
                f"""
                SELECT {column}, provider, model,
                       COALESCE(SUM(input_tokens), 0),
                       COALESCE(SUM(output_tokens), 0),
                       COALESCE(SUM(total_tokens), 0)
                FROM llm_usage_events
                WHERE 1 = 1{since_clause}
                GROUP BY {column}, provider, model
                """,
                tuple(params),
            ).fetchall()
        finally:
            conn.close()

        acc: Dict = {}
        for key, provider, model, tokens_in, tokens_out, tokens_total in rows or []:
            g = acc.setdefault(
                key,
                {group_by: key, "total_tokens": 0, "cost_usd": 0.0,
                 "unpriced_tokens": 0},
            )
            g["total_tokens"] += int(tokens_total or 0)
            cost = estimate_cost(provider, model, tokens_in, tokens_out)
            if cost is None:
                g["unpriced_tokens"] += int(tokens_total or 0)
            else:
                g["cost_usd"] += cost

        groups = sorted(acc.values(), key=lambda x: -x["total_tokens"])
        # Sum RAW group costs for the total, then round each group for display.
        # Rounding per group first would zero a total made of many sub-cent
        # groups (CX-08).
        raw_total = sum(g["cost_usd"] for g in groups)
        for g in groups:
            g["cost_usd"] = round(g["cost_usd"], 6)
        return {
            "group_by": group_by,
            "since": since or "",
            "source": "llm_usage_events",  # per-call, exact route attribution
            "groups": groups,
            "totals": {
                "total_tokens": sum(g["total_tokens"] for g in groups),
                "cost_usd": round(raw_total, 6),
                "unpriced_tokens": sum(g["unpriced_tokens"] for g in groups),
            },
            "cost_is_estimate": True,
        }

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
        daily_start = self.daily_window_start()
        daily_limit = daily_token_limit()
        daily_used = self.get_used_tokens_all_providers(
            user_id, "platform", since=daily_start
        )
        # An exempt user's cap is None/unlimited here for the same reason it is
        # in the platform block above: ensure_allowed returns before any cap
        # runs for them, so reporting a limit would be a lie.
        daily_capped = daily_limit is not None and not quota_exempt

        return {
            "platform": platform,
            "byok": byok,
            "is_admin": admin,
            "is_quota_exempt": quota_exempt,
            "platform_quota_window_days": PLATFORM_QUOTA_WINDOW_DAYS,
            "daily": {
                "used_tokens": daily_used,
                "limit_tokens": daily_limit if daily_capped else None,
                "unlimited": not daily_capped,
                "exhausted": bool(daily_capped and daily_used >= daily_limit),
                "remaining_tokens": (
                    max(0, daily_limit - daily_used) if daily_capped else None
                ),
                "window_hours": PLATFORM_DAILY_QUOTA_WINDOW_HOURS,
            },
            "cost": {
                # Split by key source: platform spend is Quasar's bill, BYOK
                # spend is the user's own. Never add them together.
                "platform_today": self.cost_breakdown(
                    user_id, "platform", since=daily_start
                ),
                "platform_week": self.cost_breakdown(
                    user_id, "platform", since=window_start
                ),
                "byok_today": self.cost_breakdown(user_id, "byok", since=daily_start),
                "byok_week": self.cost_breakdown(user_id, "byok", since=window_start),
                "is_estimate": True,
                "pricing_last_verified": PRICING_LAST_VERIFIED,
            },
        }
