"""Database storage for MCP OAuth: client registrations, tokens, sign-in states.

Three tables, all keyed by user id first so one user can never read another
user's rows (every query below filters on ``user_id``):

* ``user_mcp_oauth``: one row per (user, server). Authorization server
  metadata (public), the dynamically registered client (Fernet-encrypted, it
  can carry a client_secret), the tokens (Fernet-encrypted), expiry, scope and
  a status: ``authorized`` or ``needs_auth`` with a readable reason.
* ``mcp_oauth_states``: pending sign-ins. The browser only ever sees the raw
  ``state``; the table stores its SHA-256, the PKCE verifier (encrypted), the
  user, server, resource, redirect URI and the frontend origin to report back
  to. Single use, 10 minute lifetime, claimed with one conditional UPDATE.
* ``mcp_connect_limits``: per-user connect attempts per 10 minute window.

Secrets never leave this module except through ``OAuthRecord`` (decrypted, in
memory, for the token calls in services/mcp_oauth.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from services.mcp_server_service import mcp_db_path

STATE_TTL_SECONDS = 10 * 60
CONNECT_WINDOW_SECONDS = 10 * 60
CONNECT_ATTEMPTS_PER_WINDOW = 20

_SCHEMA_READY: set = set()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_state(raw: str) -> str:
    return hashlib.sha256((raw or "").encode("utf-8")).hexdigest()


class ConnectRateLimited(Exception):
    """Too many connect / sign-in attempts for one account."""


@dataclass
class OAuthRecord:
    user_id: str
    server: str
    server_url: str
    resource: str
    issuer: str
    metadata: Dict[str, Any]
    client: Dict[str, Any]
    redirect_uri: str
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[float] = None
    scope: Optional[str] = None
    status: str = "needs_auth"
    status_reason: Optional[str] = None

    @property
    def has_tokens(self) -> bool:
        return bool(self.access_token)

    def expires_within(self, seconds: float) -> bool:
        return self.expires_at is not None and time.time() + seconds >= self.expires_at


@dataclass
class StateRecord:
    user_id: str
    server: str
    code_verifier: str
    resource: str
    redirect_uri: str
    return_origin: str
    created_at: float
    extra: Dict[str, Any] = field(default_factory=dict)


class MCPOAuthStore:
    def __init__(self, db_path: Optional[str] = None):
        self._db_path = mcp_db_path(db_path)
        self._init_db()

    def _conn(self):
        from services.db import get_connection, is_using_turso

        if not is_using_turso():
            os.makedirs(os.path.dirname(os.path.abspath(self._db_path)), exist_ok=True)
        return get_connection(self._db_path)

    def _init_db(self) -> None:
        from services.db import is_using_turso

        key = ("turso",) if is_using_turso() else ("sqlite", os.path.abspath(self._db_path))
        if key in _SCHEMA_READY and (key[0] == "turso" or os.path.exists(key[1])):
            return
        conn = self._conn()
        try:
            conn.execute("""CREATE TABLE IF NOT EXISTS user_mcp_oauth (
                user_id TEXT NOT NULL,
                server TEXT NOT NULL,
                server_url TEXT NOT NULL,
                resource TEXT NOT NULL,
                issuer TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                client_enc TEXT,
                redirect_uri TEXT NOT NULL,
                tokens_enc TEXT,
                expires_at REAL,
                scope TEXT,
                status TEXT NOT NULL DEFAULT 'needs_auth',
                status_reason TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, server)
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS mcp_oauth_states (
                state_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                server TEXT NOT NULL,
                verifier_enc TEXT NOT NULL,
                resource TEXT NOT NULL,
                redirect_uri TEXT NOT NULL,
                return_origin TEXT NOT NULL,
                extra_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                used INTEGER NOT NULL DEFAULT 0
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS mcp_connect_limits (
                user_id TEXT PRIMARY KEY,
                window_id INTEGER NOT NULL,
                attempts INTEGER NOT NULL
            )""")
            conn.commit()
        finally:
            conn.close()
        _SCHEMA_READY.add(key)

    # ── per-server OAuth rows ────────────────────────────────────────────
    def get(self, user_id: str, server: str) -> Optional[OAuthRecord]:
        from services.provider_key_service import decrypt_secret

        conn = self._conn()
        try:
            row = conn.execute(
                """SELECT server_url, resource, issuer, metadata_json, client_enc, redirect_uri,
                          tokens_enc, expires_at, scope, status, status_reason
                   FROM user_mcp_oauth WHERE user_id = ? AND server = ?""",
                (str(user_id), server),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        (server_url, resource, issuer, metadata_json, client_enc, redirect_uri,
         tokens_enc, expires_at, scope, status, status_reason) = row
        rec = OAuthRecord(
            user_id=str(user_id), server=server, server_url=server_url, resource=resource,
            issuer=issuer, metadata=json.loads(metadata_json or "{}"), client={},
            redirect_uri=redirect_uri, expires_at=expires_at, scope=scope,
            status=status, status_reason=status_reason,
        )
        try:
            if client_enc:
                rec.client = json.loads(decrypt_secret(client_enc))
            if tokens_enc:
                tokens = json.loads(decrypt_secret(tokens_enc))
                rec.access_token = tokens.get("access_token") or None
                rec.refresh_token = tokens.get("refresh_token") or None
        except Exception as e:  # noqa: BLE001 - a key rotation makes the row unusable, not fatal
            print(f"[MCPOAuth] stored OAuth data for {server!r} could not be decrypted: {type(e).__name__}")
            rec.client, rec.access_token, rec.refresh_token = {}, None, None
            rec.status, rec.status_reason = "needs_auth", "Saved sign-in could not be read; sign in again."
        return rec

    def save_registration(self, user_id: str, server: str, *, server_url: str, resource: str,
                          issuer: str, metadata: Dict[str, Any], client: Dict[str, Any],
                          redirect_uri: str) -> None:
        """Store (or refresh) discovery + registration for a server. Existing
        tokens survive only if they were issued for the same resource and
        authorization server; anything else starts from a clean sign-in."""
        from services.provider_key_service import encrypt_secret

        client_enc = encrypt_secret(json.dumps(client)) if client else None
        now = _now_iso()
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT resource, issuer, server_url FROM user_mcp_oauth WHERE user_id = ? AND server = ?",
                (str(user_id), server),
            ).fetchone()
            if row:
                same = (row[0], row[1], row[2]) == (resource, issuer, server_url)
                if same:
                    conn.execute(
                        """UPDATE user_mcp_oauth SET metadata_json = ?, client_enc = ?, redirect_uri = ?,
                           updated_at = ? WHERE user_id = ? AND server = ?""",
                        (json.dumps(metadata), client_enc, redirect_uri, now, str(user_id), server),
                    )
                else:
                    conn.execute(
                        """UPDATE user_mcp_oauth SET server_url = ?, resource = ?, issuer = ?,
                           metadata_json = ?, client_enc = ?, redirect_uri = ?, tokens_enc = NULL,
                           expires_at = NULL, scope = NULL, status = 'needs_auth',
                           status_reason = 'Sign in to finish connecting.', updated_at = ?
                           WHERE user_id = ? AND server = ?""",
                        (server_url, resource, issuer, json.dumps(metadata), client_enc,
                         redirect_uri, now, str(user_id), server),
                    )
            else:
                conn.execute(
                    """INSERT INTO user_mcp_oauth (user_id, server, server_url, resource, issuer,
                       metadata_json, client_enc, redirect_uri, status, status_reason, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'needs_auth', 'Sign in to finish connecting.', ?)""",
                    (str(user_id), server, server_url, resource, issuer, json.dumps(metadata),
                     client_enc, redirect_uri, now),
                )
            conn.commit()
        finally:
            conn.close()

    def set_tokens(self, user_id: str, server: str, *, access_token: str,
                   refresh_token: Optional[str], expires_in: Optional[int],
                   scope: Optional[str]) -> None:
        from services.provider_key_service import encrypt_secret

        tokens_enc = encrypt_secret(json.dumps({"access_token": access_token,
                                                "refresh_token": refresh_token or None}))
        expires_at = time.time() + int(expires_in) if expires_in is not None else None
        conn = self._conn()
        try:
            conn.execute(
                """UPDATE user_mcp_oauth SET tokens_enc = ?, expires_at = ?, scope = ?,
                   status = 'authorized', status_reason = NULL, updated_at = ?
                   WHERE user_id = ? AND server = ?""",
                (tokens_enc, expires_at, scope, _now_iso(), str(user_id), server),
            )
            conn.commit()
        finally:
            conn.close()

    def mark_needs_auth(self, user_id: str, server: str, reason: str) -> None:
        """Drop the tokens (they no longer work) and record why."""
        conn = self._conn()
        try:
            conn.execute(
                """UPDATE user_mcp_oauth SET tokens_enc = NULL, expires_at = NULL,
                   status = 'needs_auth', status_reason = ?, updated_at = ?
                   WHERE user_id = ? AND server = ?""",
                (reason[:300], _now_iso(), str(user_id), server),
            )
            conn.commit()
        finally:
            conn.close()

    def delete(self, user_id: str, server: str) -> None:
        conn = self._conn()
        try:
            conn.execute("DELETE FROM user_mcp_oauth WHERE user_id = ? AND server = ?", (str(user_id), server))
            conn.execute("DELETE FROM mcp_oauth_states WHERE user_id = ? AND server = ?", (str(user_id), server))
            conn.commit()
        finally:
            conn.close()

    # ── sign-in states ───────────────────────────────────────────────────
    def create_state(self, user_id: str, server: str, *, code_verifier: str, resource: str,
                     redirect_uri: str, return_origin: str,
                     extra: Optional[Dict[str, Any]] = None) -> str:
        from services.provider_key_service import encrypt_secret

        raw = secrets.token_urlsafe(32)  # 256 bits
        now = time.time()
        conn = self._conn()
        try:
            # Opportunistic cleanup: old rows carry nothing reusable.
            conn.execute("DELETE FROM mcp_oauth_states WHERE created_at < ?", (now - 24 * 3600,))
            conn.execute(
                """INSERT INTO mcp_oauth_states (state_hash, user_id, server, verifier_enc, resource,
                   redirect_uri, return_origin, extra_json, created_at, used)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                (_hash_state(raw), str(user_id), server, encrypt_secret(code_verifier), resource,
                 redirect_uri, return_origin, json.dumps(extra or {}), now),
            )
            conn.commit()
        finally:
            conn.close()
        return raw

    def peek_state_origin(self, raw_state: str) -> Optional[str]:
        """The frontend origin a state reports back to, without claiming it
        (used to address the result page even when the state is rejected)."""
        if not raw_state:
            return None
        conn = self._conn()
        try:
            row = conn.execute("SELECT return_origin FROM mcp_oauth_states WHERE state_hash = ?",
                               (_hash_state(raw_state),)).fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    def consume_state(self, raw_state: str) -> "tuple[Optional[StateRecord], str]":
        """Claim a state exactly once. Returns (record, "") or (None, reason)
        where reason is "unknown", "expired" or "used"."""
        from services.provider_key_service import decrypt_secret

        if not raw_state or len(raw_state) > 512:
            return None, "unknown"
        digest = _hash_state(raw_state)
        cutoff = time.time() - STATE_TTL_SECONDS
        conn = self._conn()
        try:
            row = conn.execute(
                """SELECT user_id, server, verifier_enc, resource, redirect_uri, return_origin,
                          extra_json, created_at, used
                   FROM mcp_oauth_states WHERE state_hash = ?""",
                (digest,),
            ).fetchone()
            if not row:
                return None, "unknown"
            if row[8]:
                return None, "used"
            if row[7] < cutoff:
                return None, "expired"
            # The claim: only one caller can flip used 0 -> 1.
            cur = conn.execute(
                "UPDATE mcp_oauth_states SET used = 1 WHERE state_hash = ? AND used = 0 AND created_at >= ?",
                (digest, cutoff),
            )
            conn.commit()
            if getattr(cur, "rowcount", 0) != 1:
                return None, "used"
        finally:
            conn.close()
        try:
            verifier = decrypt_secret(row[2])
        except Exception:  # noqa: BLE001
            return None, "unknown"
        return StateRecord(
            user_id=row[0], server=row[1], code_verifier=verifier, resource=row[3],
            redirect_uri=row[4], return_origin=row[5], created_at=row[7],
            extra=json.loads(row[6] or "{}"),
        ), ""

    # ── rate limit ───────────────────────────────────────────────────────
    def admit_connect(self, user_id: str) -> None:
        """At most CONNECT_ATTEMPTS_PER_WINDOW connect / sign-in starts per
        user per window, shared by every API worker (atomic upsert)."""
        window = int(time.time()) // CONNECT_WINDOW_SECONDS
        conn = self._conn()
        try:
            cur = conn.execute(
                """INSERT INTO mcp_connect_limits(user_id, window_id, attempts) VALUES (?, ?, 1)
                   ON CONFLICT(user_id) DO UPDATE SET window_id = excluded.window_id,
                   attempts = CASE WHEN mcp_connect_limits.window_id = excluded.window_id
                       THEN mcp_connect_limits.attempts + 1 ELSE 1 END
                   WHERE mcp_connect_limits.window_id <> excluded.window_id
                       OR mcp_connect_limits.attempts < ?""",
                (str(user_id), window, CONNECT_ATTEMPTS_PER_WINDOW),
            )
            conn.commit()
            if getattr(cur, "rowcount", 0) == 0:
                raise ConnectRateLimited(
                    "Too many connection attempts. Wait a few minutes and try again.")
        finally:
            conn.close()
