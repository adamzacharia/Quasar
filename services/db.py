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

import atexit
import os
import sqlite3
import threading
from typing import Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
#
# Everything here is resolved per call rather than snapshotted at import.
# ``config`` and ``core.agent`` call ``load_dotenv()`` when they are imported,
# so whether TURSO_* is present in ``os.environ`` depends on *when* this module
# happens to be imported. An import-time snapshot froze that answer for the
# life of the process, which made the routing depend on module import order.
# ---------------------------------------------------------------------------


def _truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _current_environment() -> str:
    return (
        os.environ.get("QUASAR_ENV")
        or os.environ.get("APP_ENV")
        or os.environ.get("ENVIRONMENT")
        or "development"
    ).strip().lower()


def _force_local_db() -> bool:
    return _truthy(os.environ.get("QUASAR_FORCE_LOCAL_DB"))


def _is_production() -> bool:
    return _current_environment() in {"production", "prod"}


def _turso_credentials() -> Tuple[Optional[str], Optional[str]]:
    return os.environ.get("TURSO_DATABASE_URL"), os.environ.get("TURSO_AUTH_TOKEN")


def _check_force_local_allowed() -> None:
    if _force_local_db() and _is_production():
        raise RuntimeError(
            "QUASAR_FORCE_LOCAL_DB cannot be used when QUASAR_ENV/ENVIRONMENT is production."
        )


def _use_turso() -> bool:
    _check_force_local_allowed()
    url, token = _turso_credentials()
    return bool(url and token and not _force_local_db())


# Fail fast on a misconfigured process rather than only on the first query.
_check_force_local_allowed()


# ---------------------------------------------------------------------------
# Turso wrapper that mimics sqlite3's connection/cursor pattern
# ---------------------------------------------------------------------------

# Every live _TursoConnection, so shutdown can close the ones a caller stranded.
_open_connections = set()
_open_connections_lock = threading.Lock()


def _close_open_connections() -> None:
    """Close any _TursoConnection a caller never closed.

    ``create_client_sync`` runs its event loop on a NON-daemon thread that stops
    only on ``client.close()``, and the thread is already started by the time we
    get the client, so it cannot be re-flagged as a daemon. Python joins
    non-daemon threads during interpreter shutdown, so one connection stranded by
    a raising statement hangs the process forever — the symptom being a test run
    that prints its summary and then never exits.

    This must run BEFORE that join. ``threading._register_atexit`` fires inside
    ``threading._shutdown()``, ahead of the join; a plain ``atexit`` handler runs
    after it and would never get the chance. (``concurrent.futures`` reaches for
    the same private hook for the same reason.)
    """
    with _open_connections_lock:
        stranded = list(_open_connections)
    for conn in stranded:
        try:
            conn.close()
        except Exception:
            pass


_register_shutdown = getattr(threading, "_register_atexit", None)
if _register_shutdown is not None:
    _register_shutdown(_close_open_connections)
else:  # pragma: no cover - only on interpreters without the private hook
    atexit.register(_close_open_connections)


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

        url, auth_token = _turso_credentials()
        # Convert libsql:// to https:// for HTTP transport
        # (wss transport can fail with 505 on some platforms)
        if url.startswith("libsql://"):
            url = url.replace("libsql://", "https://", 1)
        self._client = libsql_client.create_client_sync(
            url=url,
            auth_token=auth_token,
        )
        with _open_connections_lock:
            _open_connections.add(self)

    def cursor(self) -> _TursoCursor:
        return _TursoCursor(self._client)

    def execute(self, sql: str, params: tuple = ()):
        return _TursoCursor(self._client).execute(sql, params)

    def commit(self):
        # Turso auto-commits each statement executed over HTTP
        pass

    def close(self):
        with _open_connections_lock:
            _open_connections.discard(self)
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

    NOTE: *local_db_path* is the local-fallback location, NOT an override —
    when Turso is configured every caller shares the one cloud database, and
    the path is ignored. Set QUASAR_FORCE_LOCAL_DB=1 to pin the SQLite branch.

    The returned object quacks like a sqlite3.Connection: it has
    .cursor(), .execute(), .commit(), .close(), and works as a
    context manager.
    """
    if _use_turso():
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
    return _use_turso()
