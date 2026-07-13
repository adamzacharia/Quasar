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
    """Cursor wrapper: reads/DDL execute immediately; DML is buffered on the
    parent connection and flushed atomically by _TursoConnection.commit().
    A connection opened with autocommit=True keeps the legacy behavior:
    every statement (DML included) executes immediately."""

    _WRITE_PREFIXES = ("INSERT", "UPDATE", "DELETE", "REPLACE")

    @staticmethod
    def _top_level_sql_words(sql: str):
        """Yield unquoted SQL words outside parenthesized expressions."""
        i = 0
        depth = 0
        while i < len(sql):
            ch = sql[i]
            if ch.isspace():
                i += 1
            elif sql.startswith("--", i):
                i += 2
                while i < len(sql) and sql[i] not in "\r\n":
                    i += 1
            elif sql.startswith("/*", i):
                end = sql.find("*/", i + 2)
                if end < 0:
                    return
                i = end + 2
            elif ch in "'\"`":
                quote = ch
                i += 1
                while i < len(sql):
                    if sql[i] == quote:
                        if i + 1 < len(sql) and sql[i + 1] == quote:
                            i += 2
                        else:
                            i += 1
                            break
                    else:
                        i += 1
            elif ch == "[":
                end = sql.find("]", i + 1)
                if end < 0:
                    return
                i = end + 1
            elif ch == "(":
                depth += 1
                i += 1
            elif ch == ")":
                depth = max(0, depth - 1)
                i += 1
            elif ch.isalpha() or ch == "_":
                start = i
                i += 1
                while i < len(sql) and (sql[i].isalnum() or sql[i] in "_$"):
                    i += 1
                if depth == 0:
                    yield sql[start:i].upper()
            else:
                i += 1

    @classmethod
    def _is_write_statement(cls, sql: str) -> bool:
        words = iter(cls._top_level_sql_words(sql))
        first = next(words, None)
        if first in cls._WRITE_PREFIXES:
            return True
        if first != "WITH":
            return False

        # CTE bodies are parenthesized, so the next statement keyword at the
        # top level distinguishes writable WITH statements from SELECT/VALUES.
        for word in words:
            if word in cls._WRITE_PREFIXES:
                return True
            if word in {"SELECT", "VALUES"}:
                return False
        return False

    def __init__(self, connection):
        self._conn = connection
        self._client = connection._client
        self._rows: list = []
        self._description = None
        self.lastrowid: Optional[int] = None
        self.rowcount: int = -1

    def execute(self, sql: str, params: tuple = ()) -> "_TursoCursor":
        if self._is_write_statement(sql) and not self._conn._autocommit:
            # Buffer DML; flushed as one atomic batch on commit().
            self._conn._pending.append((sql, tuple(params)))
            self._conn._last_write_cursor = self
            self._rows = []
            self._description = None
            self.lastrowid = None
            self.rowcount = -1
            return self
        # Reads and DDL run immediately (matches prior behavior).
        rs = self._client.execute(sql, list(params))
        self._rows = list(rs.rows) if rs.rows else []
        self._description = rs.columns if hasattr(rs, "columns") else None
        self.lastrowid = rs.last_insert_rowid if hasattr(rs, "last_insert_rowid") else None
        self.rowcount = rs.rows_affected if hasattr(rs, "rows_affected") else len(self._rows)
        return self

    def fetchone(self) -> Optional[tuple]:
        if self._rows:
            return tuple(self._rows.pop(0))
        return None

    def fetchall(self) -> List[tuple]:
        rows = [tuple(r) for r in self._rows]
        self._rows = []
        return rows

    def close(self):
        pass


class _TursoConnection:
    """Connection wrapper with real atomic commit() via libsql batch().
    autocommit=True opts out of DML buffering (legacy immediate-execute) for
    callers that read per-statement cursor state before commit."""

    def __init__(self, autocommit: bool = False):
        import libsql_client

        url, auth_token = _turso_credentials()
        if url.startswith("libsql://"):
            url = url.replace("libsql://", "https://", 1)
        self._client = libsql_client.create_client_sync(url=url, auth_token=auth_token)
        self._autocommit = bool(autocommit)
        self._pending: list = []
        self._last_write_cursor: Optional["_TursoCursor"] = None
        with _open_connections_lock:
            _open_connections.add(self)

    def cursor(self) -> _TursoCursor:
        return _TursoCursor(self)

    def execute(self, sql: str, params: tuple = ()):
        return _TursoCursor(self).execute(sql, params)

    def commit(self):
        # Flush buffered DML as ONE atomic batch (BEGIN…COMMIT…ROLLBACK).
        # HTTP transport has no interactive transaction(); batch() is the only
        # atomic primitive. Reads/DDL already executed immediately.
        if not self._pending:
            return
        stmts = [(sql, list(params)) for sql, params in self._pending]
        results = self._client.batch(stmts)
        self._pending = []
        if results and self._last_write_cursor is not None:
            last = results[-1]
            self._last_write_cursor.lastrowid = getattr(last, "last_insert_rowid", None)
            self._last_write_cursor.rowcount = getattr(last, "rows_affected", -1)
        self._last_write_cursor = None

    def rollback(self):
        # Discard buffered-but-unflushed DML (nothing was sent to the server).
        self._pending = []
        self._last_write_cursor = None

    def close(self):
        # Unflushed writes are discarded (implicit rollback).
        self._pending = []
        self._last_write_cursor = None
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

def get_connection(local_db_path: str = None, autocommit: bool = False) -> Any:
    """
    Return a database connection.

    If TURSO_DATABASE_URL and TURSO_AUTH_TOKEN are set → Turso cloud DB.
    Otherwise → local SQLite at *local_db_path* (for development).

    NOTE: *local_db_path* is the local-fallback location, NOT an override —
    when Turso is configured every caller shares the one cloud database, and
    the path is ignored. Set QUASAR_FORCE_LOCAL_DB=1 to pin the SQLite branch.
    autocommit=True keeps legacy per-statement auto-commit on the Turso path.

    The returned object quacks like a sqlite3.Connection: it has
    .cursor(), .execute(), .commit(), .close(), and works as a
    context manager.
    """
    if _use_turso():
        return _TursoConnection(autocommit=autocommit)
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
