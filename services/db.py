# services/db.py
"""
Centralized Database Connection Service — Turso (libsql) + SQLite fallback.

CALLED BY: services/auth.py, services/conversation_service.py, ui-pro/api/main.py
CALLS:     libsql_client (cloud Turso DB) OR sqlite3 (local fallback)

Provides a sqlite3-compatible connection wrapper so existing SQL code
works unchanged. When TURSO_DATABASE_URL is set, queries go to the
cloud Turso DB (persistent). Otherwise, falls back to local SQLite
(for development).
"""

import os
import sqlite3
from typing import Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TURSO_DATABASE_URL = os.environ.get("TURSO_DATABASE_URL")
TURSO_AUTH_TOKEN = os.environ.get("TURSO_AUTH_TOKEN")


def _truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _current_environment() -> str:
    return (
        os.environ.get("QUASAR_ENV")
        or os.environ.get("APP_ENV")
        or os.environ.get("ENVIRONMENT")
        or "development"
    ).strip().lower()


_FORCE_LOCAL_DB = _truthy(os.environ.get("QUASAR_FORCE_LOCAL_DB"))
_IS_PRODUCTION = _current_environment() in {"production", "prod"}
if _FORCE_LOCAL_DB and _IS_PRODUCTION:
    raise RuntimeError(
        "QUASAR_FORCE_LOCAL_DB cannot be used when QUASAR_ENV/ENVIRONMENT is production."
    )

_USE_TURSO = bool(TURSO_DATABASE_URL and TURSO_AUTH_TOKEN and not _FORCE_LOCAL_DB)


# ---------------------------------------------------------------------------
# Turso wrapper that mimics sqlite3's connection/cursor pattern
# ---------------------------------------------------------------------------

class _TursoCursor:
    """Minimal cursor-like wrapper around libsql_client result sets."""

    def __init__(self, client):
        self._client = client
        self._rows: list = []
        self._description = None
        self.lastrowid: Optional[int] = None
        self.rowcount: int = -1

    def execute(self, sql: str, params: tuple = ()) -> "_TursoCursor":
        # libsql_client expects list params
        rs = self._client.execute(sql, list(params))
        self._rows = list(rs.rows) if rs.rows else []
        self._description = rs.columns if hasattr(rs, "columns") else None
        self.lastrowid = rs.last_insert_rowid if hasattr(rs, "last_insert_rowid") else None
        self.rowcount = rs.rows_affected if hasattr(rs, "rows_affected") else len(self._rows)
        return self

    def fetchone(self) -> Optional[tuple]:
        if self._rows:
            row = self._rows.pop(0)
            return tuple(row)
        return None

    def fetchall(self) -> List[tuple]:
        rows = [tuple(r) for r in self._rows]
        self._rows = []
        return rows

    def close(self):
        pass


class _TursoConnection:
    """Minimal connection-like wrapper around the libsql_client sync client."""

    def __init__(self):
        import libsql_client
        # Convert libsql:// to https:// for HTTP transport
        # (wss transport can fail with 505 on some platforms)
        url = TURSO_DATABASE_URL
        if url.startswith("libsql://"):
            url = url.replace("libsql://", "https://", 1)
        self._client = libsql_client.create_client_sync(
            url=url,
            auth_token=TURSO_AUTH_TOKEN,
        )

    def cursor(self) -> _TursoCursor:
        return _TursoCursor(self._client)

    def execute(self, sql: str, params: tuple = ()):
        return _TursoCursor(self._client).execute(sql, params)

    def commit(self):
        # Turso auto-commits each statement executed over HTTP
        pass

    def close(self):
        try:
            self._client.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_connection(local_db_path: str = None) -> Any:
    """
    Return a database connection.

    If TURSO_DATABASE_URL and TURSO_AUTH_TOKEN are set → Turso cloud DB.
    Otherwise → local SQLite at *local_db_path* (for development).

    The returned object quacks like a sqlite3.Connection: it has
    .cursor(), .execute(), .commit(), .close(), and works as a
    context manager.
    """
    if _USE_TURSO:
        return _TursoConnection()
    else:
        # Local SQLite fallback for development
        if local_db_path is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(root_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            local_db_path = os.path.join(data_dir, "quasar.db")
        return sqlite3.connect(local_db_path)


def is_using_turso() -> bool:
    """Return True when running against the cloud Turso database."""
    return _USE_TURSO
