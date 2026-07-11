"""Shared admin access helpers.

Admin privilege is granted by either:

1. the ``ADMIN_EMAILS`` environment allowlist (comma-separated), or
2. a per-user ``role = 'admin'`` value in the ``users`` table.

There is intentionally **no** hard-coded admin email (S3). The old
``DEFAULT_ADMIN_EMAILS = {"1@1"}`` compiled a permanent admin into the binary;
local development now gets its admin from the seeded test user's ``role``
column instead (see ``services/auth.py``).

The DB lookup is wrapped in a small TTL cache because ``is_quota_exempt_email``
runs on the per-request LLM-quota path; without it every request would issue a
``users`` query (a network round-trip under Turso).
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple


# No compiled-in admins. Populate ADMIN_EMAILS or set role='admin' in the DB.
DEFAULT_ADMIN_EMAILS: set[str] = set()


def _users_db_path() -> Optional[str]:
    """Path to the local users SQLite DB (ignored when Turso is configured).

    Mirrors ``services.auth._default_db_path`` so both read the same file.
    """
    override = os.environ.get("QUASAR_USERS_DB_PATH")
    if override and override.strip():
        return override.strip()
    root = Path(__file__).resolve().parent.parent
    return str(root / "data" / "users.db")


def configured_admin_emails() -> set[str]:
    """Return the environment-configured admin email allowlist."""
    env_emails = {
        email.strip().lower()
        for email in os.getenv("ADMIN_EMAILS", "").split(",")
        if email.strip()
    }
    return set(DEFAULT_ADMIN_EMAILS) | env_emails


def configured_quota_exempt_emails() -> set[str]:
    """Return emails allowed to bypass platform token quotas (env only)."""
    env_emails = {
        email.strip().lower()
        for email in os.getenv("QUASAR_TOKEN_LIMIT_EXEMPT_EMAILS", "").split(",")
        if email.strip()
    }
    return env_emails | configured_admin_emails()


# ── DB role lookup (TTL-cached) ──────────────────────────────────────────────

_role_cache: Dict[str, Tuple[float, bool]] = {}
_role_cache_lock = threading.Lock()


def _role_cache_ttl() -> float:
    raw = os.environ.get("QUASAR_ADMIN_ROLE_CACHE_TTL")
    if raw is None or not str(raw).strip():
        return 60.0
    try:
        return max(0.0, float(str(raw).strip()))
    except (TypeError, ValueError):
        return 60.0


def clear_admin_role_cache() -> None:
    """Drop the cached DB role lookups (used by tests and after role edits)."""
    with _role_cache_lock:
        _role_cache.clear()


def _query_db_role_is_admin(email: str) -> bool:
    try:
        from services.db import get_connection

        with get_connection(_users_db_path()) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM users "
                "WHERE role = 'admin' AND (lower(username) = ? OR lower(email) = ?) "
                "LIMIT 1",
                (email, email),
            )
            return cursor.fetchone() is not None
    except Exception:
        # Missing column / unreachable DB → fail closed (not admin).
        return False


def _db_role_is_admin(email: Optional[str]) -> bool:
    """Return True when *email* has ``role = 'admin'`` in the users table."""
    normalized = (email or "").strip().lower()
    if not normalized:
        return False

    ttl = _role_cache_ttl()
    now = time.monotonic()
    if ttl > 0:
        with _role_cache_lock:
            cached = _role_cache.get(normalized)
            if cached is not None and (now - cached[0]) < ttl:
                return cached[1]

    result = _query_db_role_is_admin(normalized)

    if ttl > 0:
        with _role_cache_lock:
            _role_cache[normalized] = (now, result)
    return result


# ── Public predicates ────────────────────────────────────────────────────────


def is_admin_email(email: Optional[str]) -> bool:
    """Return True when the email is allowed through admin gates."""
    if not email:
        return False
    normalized = email.strip().lower()
    if normalized in configured_admin_emails():
        return True
    return _db_role_is_admin(normalized)


def is_quota_exempt_email(email: Optional[str]) -> bool:
    """Return True when the email is allowed to bypass token quotas."""
    if not email:
        return False
    normalized = email.strip().lower()
    if normalized in configured_quota_exempt_emails():
        return True
    # DB-role admins inherit quota exemption (matches prior admin behaviour).
    return _db_role_is_admin(normalized)


def is_admin_user(user: Optional[Mapping[str, object]]) -> bool:
    """Return True for JWT payloads or user dicts with an admin email."""
    if not user:
        return False
    email = str(user.get("email") or user.get("username") or "").strip()
    return is_admin_email(email)
