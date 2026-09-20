"""Encrypted per-user LLM provider key storage."""

from __future__ import annotations

import base64
import hashlib
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from cryptography.fernet import Fernet, InvalidToken

from services.db import get_connection


ALLOWED_PROVIDERS = {"openai", "deepseek", "anthropic", "google"}
_LOCAL_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "provider_keys.db")


class ProviderKeyError(ValueError):
    """Raised when provider-key storage cannot complete safely."""


@dataclass(frozen=True)
class StoredCatalog:
    models_json: str
    fetched_at: str


class CatalogRateLimitError(ProviderKeyError):
    """Too many provider validation/discovery attempts for one account."""


def _current_environment() -> str:
    return (
        os.environ.get("QUASAR_ENV")
        or os.environ.get("APP_ENV")
        or os.environ.get("ENVIRONMENT")
        or "development"
    ).strip().lower()


def _dev_fernet_key() -> bytes:
    digest = hashlib.sha256(b"quasar-local-development-provider-key").digest()
    return base64.urlsafe_b64encode(digest)


def _resolve_fernet() -> Fernet:
    raw_key = os.environ.get("USER_API_KEY_FERNET_KEY", "").strip()
    if raw_key:
        try:
            return Fernet(raw_key.encode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ProviderKeyError("USER_API_KEY_FERNET_KEY is not a valid Fernet key.") from exc

    if _current_environment() in {"production", "prod"}:
        raise ProviderKeyError(
            "USER_API_KEY_FERNET_KEY is required in production to encrypt user API keys."
        )

    return Fernet(_dev_fernet_key())


def _normalize_provider(provider: str) -> str:
    value = (provider or "").strip().lower()
    if value == "gemini":
        value = "google"
    if value in {"tejas", "texas", "texas_ai"}:
        value = "tacc"
    if value not in ALLOWED_PROVIDERS:
        raise ProviderKeyError(f"Unsupported provider '{provider}'.")
    return value


class ProviderKeyService:
    """CRUD for encrypted user-owned provider API keys."""

    def __init__(self):
        self._fernet = _resolve_fernet()
        self._init_db()

    def _conn(self):
        return get_connection(_LOCAL_DB)

    def _init_db(self) -> None:
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_provider_keys (
                    user_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    encrypted_key TEXT NOT NULL,
                    key_last4 TEXT NOT NULL,
                    token_limit INTEGER,
                    status TEXT NOT NULL DEFAULT 'untested',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_tested_at TEXT,
                    PRIMARY KEY (user_id, provider)
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_provider_keys_user
                ON user_provider_keys(user_id)
                """
            )
            cur.execute("""CREATE TABLE IF NOT EXISTS user_provider_models (
                user_id TEXT NOT NULL, provider TEXT NOT NULL,
                key_revision TEXT NOT NULL, models_json TEXT NOT NULL,
                fetched_at TEXT NOT NULL, PRIMARY KEY (user_id, provider)
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS provider_validation_limits (
                user_id TEXT PRIMARY KEY, window_id INTEGER NOT NULL, attempts INTEGER NOT NULL
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS provider_model_attempts (
                user_id TEXT NOT NULL, provider TEXT NOT NULL, revision TEXT NOT NULL,
                attempted_at REAL NOT NULL, PRIMARY KEY (user_id, provider)
            )""")
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _metadata_from_row(row) -> Dict:
        return {
            "provider": row[1],
            "key_last4": row[3],
            "token_limit": row[4],
            "status": row[5],
            "created_at": row[6],
            "updated_at": row[7],
            "last_tested_at": row[8],
        }

    def list_keys(self, user_id: str) -> List[Dict]:
        conn = self._conn()
        try:
            rows = conn.execute(
                """
                SELECT user_id, provider, encrypted_key, key_last4, token_limit,
                       status, created_at, updated_at, last_tested_at
                FROM user_provider_keys
                WHERE user_id = ?
                ORDER BY provider
                """,
                (user_id,),
            ).fetchall()
        finally:
            conn.close()
        return [self._metadata_from_row(row) for row in rows]

    def get_metadata(self, user_id: str, provider: str) -> Optional[Dict]:
        provider = _normalize_provider(provider)
        conn = self._conn()
        try:
            row = conn.execute(
                """
                SELECT user_id, provider, encrypted_key, key_last4, token_limit,
                       status, created_at, updated_at, last_tested_at
                FROM user_provider_keys
                WHERE user_id = ? AND provider = ?
                """,
                (user_id, provider),
            ).fetchone()
        finally:
            conn.close()
        return self._metadata_from_row(row) if row else None

    def save_key(self, user_id: str, provider: str, raw_key: str, token_limit: Optional[int] = None,
                 *, catalog: StoredCatalog | None = None) -> Dict:
        provider = _normalize_provider(provider)
        key = (raw_key or "").strip()
        if len(key) < 8:
            raise ProviderKeyError("API key is too short.")
        if token_limit is not None:
            token_limit = self._normalize_token_limit(token_limit)

        encrypted = self._fernet.encrypt(key.encode("utf-8")).decode("utf-8")
        last4 = key[-4:]
        now = datetime.now(timezone.utc).isoformat()

        conn = self._conn()
        try:
            existing = conn.execute(
                "SELECT created_at, token_limit FROM user_provider_keys WHERE user_id = ? AND provider = ?",
                (user_id, provider),
            ).fetchone()
            if existing:
                created_at = existing[0]
                if token_limit is None:
                    token_limit = existing[1]
                conn.execute(
                    """
                    UPDATE user_provider_keys
                    SET encrypted_key = ?, key_last4 = ?, token_limit = ?, status = 'untested',
                        updated_at = ?, last_tested_at = NULL
                    WHERE user_id = ? AND provider = ?
                    """,
                    (encrypted, last4, token_limit, now, user_id, provider),
                )
            else:
                created_at = now
                conn.execute(
                    """
                    INSERT INTO user_provider_keys
                    (user_id, provider, encrypted_key, key_last4, token_limit, status,
                     created_at, updated_at, last_tested_at)
                    VALUES (?, ?, ?, ?, ?, 'untested', ?, ?, NULL)
                    """,
                    (user_id, provider, encrypted, last4, token_limit, created_at, now),
                )
            # Rotate key and catalog together. No stale catalog may survive a
            # different credential, even if its last four characters match.
            conn.execute("DELETE FROM user_provider_models WHERE user_id=? AND provider=?", (user_id, provider))
            if catalog is not None:
                conn.execute("""INSERT INTO user_provider_models
                    (user_id, provider, key_revision, models_json, fetched_at) VALUES (?, ?, ?, ?, ?)""",
                    (user_id, provider, encrypted, catalog.models_json, catalog.fetched_at))
                conn.execute("""UPDATE user_provider_keys SET status='valid', last_tested_at=?
                    WHERE user_id=? AND provider=?""", (now, user_id, provider))
            conn.commit()
        finally:
            conn.close()
        return self.get_metadata(user_id, provider) or {}

    def decrypt_key(self, user_id: str, provider: str) -> Optional[str]:
        provider = _normalize_provider(provider)
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT encrypted_key FROM user_provider_keys WHERE user_id = ? AND provider = ?",
                (user_id, provider),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        try:
            return self._fernet.decrypt(str(row[0]).encode("utf-8")).decode("utf-8")
        except InvalidToken as exc:
            raise ProviderKeyError("Stored API key could not be decrypted. Rotate the key.") from exc

    def decrypt_all_keys(self, user_id: str) -> Dict[str, str]:
        keys: Dict[str, str] = {}
        for metadata in self.list_keys(user_id):
            provider = metadata["provider"]
            raw = self.decrypt_key(user_id, provider)
            if raw:
                keys[provider] = raw
        return keys

    def set_token_limit(self, user_id: str, provider: str, token_limit: Optional[int]) -> Dict:
        provider = _normalize_provider(provider)
        normalized = self._normalize_token_limit(token_limit) if token_limit is not None else None
        now = datetime.now(timezone.utc).isoformat()
        conn = self._conn()
        try:
            cur = conn.execute(
                """
                UPDATE user_provider_keys
                SET token_limit = ?, updated_at = ?
                WHERE user_id = ? AND provider = ?
                """,
                (normalized, now, user_id, provider),
            )
            conn.commit()
        finally:
            conn.close()
        if getattr(cur, "rowcount", 0) == 0:
            raise ProviderKeyError("Provider key not found.")
        return self.get_metadata(user_id, provider) or {}

    def mark_test_result(self, user_id: str, provider: str, ok: bool) -> Dict:
        provider = _normalize_provider(provider)
        now = datetime.now(timezone.utc).isoformat()
        conn = self._conn()
        try:
            conn.execute(
                """
                UPDATE user_provider_keys
                SET status = ?, last_tested_at = ?, updated_at = ?
                WHERE user_id = ? AND provider = ?
                """,
                ("valid" if ok else "invalid", now, now, user_id, provider),
            )
            conn.commit()
        finally:
            conn.close()
        return self.get_metadata(user_id, provider) or {}

    def delete_key(self, user_id: str, provider: str) -> bool:
        provider = _normalize_provider(provider)
        conn = self._conn()
        try:
            conn.execute("DELETE FROM user_provider_models WHERE user_id=? AND provider=?", (user_id, provider))
            conn.execute("DELETE FROM provider_model_attempts WHERE user_id=? AND provider=?", (user_id, provider))
            cur = conn.execute(
                "DELETE FROM user_provider_keys WHERE user_id = ? AND provider = ?",
                (user_id, provider),
            )
            conn.commit()
        finally:
            conn.close()
        return getattr(cur, "rowcount", 0) > 0

    def key_snapshot(self, user_id: str, provider: str) -> tuple[str, str] | None:
        """Internal decrypted credential + opaque encrypted revision; never serialize."""
        provider = _normalize_provider(provider)
        conn = self._conn()
        try:
            row = conn.execute("SELECT encrypted_key FROM user_provider_keys WHERE user_id=? AND provider=?",
                               (user_id, provider)).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        revision = str(row[0])
        try:
            return self._fernet.decrypt(revision.encode()).decode(), revision
        except InvalidToken:
            raise ProviderKeyError("Stored API key could not be decrypted. Rotate the key.") from None

    def get_catalog(self, user_id: str, provider: str) -> StoredCatalog | None:
        conn = self._conn()
        try:
            row = conn.execute("""SELECT c.models_json, c.fetched_at FROM user_provider_models c
                JOIN user_provider_keys k ON k.user_id=c.user_id AND k.provider=c.provider
                    AND k.encrypted_key=c.key_revision
                WHERE c.user_id=? AND c.provider=?""", (user_id, provider)).fetchone()
        finally:
            conn.close()
        return StoredCatalog(str(row[0]), str(row[1])) if row else None

    def cache_catalog(self, user_id: str, provider: str, revision: str, catalog: StoredCatalog) -> bool:
        """Compare-and-set: in-flight discovery cannot resurrect a removed/rotated key."""
        conn = self._conn()
        try:
            cur = conn.execute("""INSERT INTO user_provider_models
                (user_id, provider, key_revision, models_json, fetched_at)
                SELECT user_id, provider, encrypted_key, ?, ? FROM user_provider_keys
                WHERE user_id=? AND provider=? AND encrypted_key=?
                ON CONFLICT(user_id, provider) DO UPDATE SET key_revision=excluded.key_revision,
                    models_json=excluded.models_json, fetched_at=excluded.fetched_at""",
                (catalog.models_json, catalog.fetched_at, user_id, provider, revision))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def admit_validation(self, user_id: str) -> None:
        """Ten attempts/account/10 minutes, atomically shared by all API workers.

        Save, test, manual refresh, and automatic discovery share this budget;
        failed validations count too. One row per user keeps storage bounded.
        """
        window = int(time.time()) // 600
        conn = self._conn()
        try:
            cur = conn.execute("""INSERT INTO provider_validation_limits(user_id, window_id, attempts)
                VALUES (?, ?, 1) ON CONFLICT(user_id) DO UPDATE SET window_id=excluded.window_id,
                attempts=CASE WHEN provider_validation_limits.window_id=excluded.window_id
                    THEN provider_validation_limits.attempts+1 ELSE 1 END
                WHERE provider_validation_limits.window_id<>excluded.window_id
                    OR provider_validation_limits.attempts<10""", (user_id, window))
            conn.commit()
            if cur.rowcount == 0:
                raise CatalogRateLimitError("Too many provider key checks. Try again in 10 minutes.")
        finally:
            conn.close()

    def claim_discovery(self, user_id: str, provider: str, revision: str) -> bool:
        """Shared five-minute automatic-refresh lease/backoff, separate from user checks."""
        now = time.time()
        conn = self._conn()
        try:
            cur = conn.execute("""INSERT INTO provider_model_attempts(user_id, provider, revision, attempted_at)
                SELECT user_id, provider, encrypted_key, ? FROM user_provider_keys
                WHERE user_id=? AND provider=? AND encrypted_key=?
                ON CONFLICT(user_id, provider) DO UPDATE SET revision=excluded.revision,
                    attempted_at=excluded.attempted_at
                WHERE provider_model_attempts.revision<>excluded.revision
                    OR provider_model_attempts.attempted_at < ?""", (now, user_id, provider, revision, now - 300))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def list_catalogs(self, user_id: str) -> dict[str, StoredCatalog]:
        conn = self._conn()
        try:
            rows = conn.execute("""SELECT c.provider, c.models_json, c.fetched_at FROM user_provider_models c
                JOIN user_provider_keys k ON k.user_id=c.user_id AND k.provider=c.provider
                    AND k.encrypted_key=c.key_revision WHERE c.user_id=?""", (user_id,)).fetchall()
            return {str(row[0]): StoredCatalog(str(row[1]), str(row[2])) for row in rows}
        finally:
            conn.close()

    @staticmethod
    def _normalize_token_limit(value: int) -> int:
        try:
            normalized = int(value)
        except (TypeError, ValueError) as exc:
            raise ProviderKeyError("Token limit must be a whole number.") from exc
        if normalized <= 0:
            raise ProviderKeyError("Token limit must be greater than zero, or clear it for no limit.")
        return normalized
