import pytest

import services.db as db


class _FakeResult:
    def __init__(self, rowid=None, affected=0, rows=None, columns=None):
        self.last_insert_rowid = rowid
        self.rows_affected = affected
        self.rows = rows or []
        self.columns = columns


class _FakeClient:
    def __init__(self):
        self.executed = []   # immediate execute() calls
        self.batches = []    # successful batch() flushes
        self.batch_attempts = 0
        self.fail_next_batch = False
        self.closed = False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return _FakeResult(rowid=1, affected=1, rows=[])

    def batch(self, stmts):
        self.batch_attempts += 1
        if self.fail_next_batch:
            self.fail_next_batch = False
            raise RuntimeError("batch failed")
        self.batches.append(list(stmts))
        return [_FakeResult(rowid=i + 1, affected=i + 1) for i, _ in enumerate(stmts)]

    def close(self):
        self.closed = True


def _make_conn(fake, autocommit=False):
    conn = db._TursoConnection.__new__(db._TursoConnection)
    conn._client = fake
    conn._autocommit = autocommit
    conn._pending = []
    conn._last_write_cursor = None
    return conn


def test_dml_is_buffered_until_commit():
    fake = _FakeClient()
    conn = _make_conn(fake)
    cur = conn.cursor()
    writes = [
        ("/* fake UPDATE */ -- fake DELETE\n  insert INTO t (a) VALUES (?)", (1,)),
        ("WITH source AS (SELECT ?) INSERT INTO t SELECT * FROM source", (2,)),
        ("WITH source AS (SELECT ?) UPDATE t SET a = ?", (3, 4)),
        ("WITH source AS (SELECT ?) DELETE FROM t WHERE a = ?", (5, 6)),
        ("WITH source AS (SELECT ?) REPLACE INTO t SELECT * FROM source", (7,)),
    ]
    for sql, params in writes:
        cur.execute(sql, params)
    assert fake.batches == []
    assert fake.executed == []
    assert conn._pending == writes


def test_commit_flushes_one_atomic_batch():
    fake = _FakeClient()
    conn = _make_conn(fake)
    cur = conn.cursor()
    cur.execute("INSERT INTO t (a) VALUES (?)", (1,))
    cur.execute("DELETE FROM t WHERE a = ?", (1,))
    fake.fail_next_batch = True
    with pytest.raises(RuntimeError, match="batch failed"):
        conn.commit()
    assert len(conn._pending) == 2
    assert conn._last_write_cursor is cur

    conn.commit()
    assert fake.batch_attempts == 2
    assert fake.batches == [[
        ("INSERT INTO t (a) VALUES (?)", [1]),
        ("DELETE FROM t WHERE a = ?", [1]),
    ]]


def test_commit_populates_cursor_lastrowid():
    fake = _FakeClient()
    conn = _make_conn(fake)
    cur = conn.cursor()
    cur.execute("INSERT INTO t (a) VALUES (?)", (5,))
    cur.execute("INSERT INTO t (a) VALUES (?)", (6,))
    assert cur.lastrowid is None
    conn.commit()
    assert cur.lastrowid == 2


def test_commit_populates_cursor_rowcount():
    # Covers the update_conversation_title_for_user pattern (rowcount read
    # after commit) and proves the last batch result is used.
    fake = _FakeClient()
    conn = _make_conn(fake)
    cur = conn.cursor()
    cur.execute("UPDATE t SET a = ? WHERE a = ?", (2, 1))
    cur.execute("UPDATE t SET a = ? WHERE a = ?", (3, 2))
    assert cur.rowcount == -1
    conn.commit()
    assert cur.rowcount == 2


def test_rollback_discards_buffer():
    fake = _FakeClient()
    conn = _make_conn(fake)
    cur = conn.cursor()
    cur.execute("INSERT INTO t (a) VALUES (?)", (9,))
    conn.rollback()
    assert conn._pending == []
    assert conn._last_write_cursor is None
    conn.commit()
    assert fake.batches == []


def test_close_without_commit_discards():
    fake = _FakeClient()
    conn = _make_conn(fake)
    cur = conn.cursor()
    cur.execute("INSERT INTO t (a) VALUES (?)", (7,))
    conn.close()
    assert conn._pending == []
    assert conn._last_write_cursor is None
    assert fake.closed is True
    assert fake.batches == []


def test_reads_and_ddl_execute_immediately():
    fake = _FakeClient()
    conn = _make_conn(fake)
    cur = conn.cursor()
    immediate = [
        "-- fake INSERT\n/* fake DELETE */ SELECT * FROM t",
        "/* fake UPDATE */ CREATE TABLE t (a INTEGER)",
        "WITH source AS (SELECT 1) SELECT * FROM source",
        "WITH RECURSIVE source(a) AS (VALUES (1)) VALUES (2)",
        "WITH source AS (SELECT ') DELETE FROM t' /* ) UPDATE t */) SELECT * FROM source",
        'WITH "DELETE" AS (SELECT 1) SELECT * FROM "DELETE"',
    ]
    for sql in immediate:
        cur.execute(sql)
    assert fake.executed == [(sql, []) for sql in immediate]
    conn.commit()
    assert fake.batches == []


def test_autocommit_connection_executes_dml_immediately(monkeypatch):
    # sky_monitor opts out of buffering: per-row rowcount must be available
    # at execute time, exactly as before S44.
    fake = _FakeClient()
    conn = _make_conn(fake, autocommit=True)
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO t (a) VALUES (?)", (3,))
    assert len(fake.executed) == 1
    assert cur.rowcount == 1
    assert cur.lastrowid == 1
    conn.commit()
    assert fake.batches == []

    forwarded = []
    monkeypatch.setattr(db, "_use_turso", lambda: True)
    monkeypatch.setattr(
        db,
        "_TursoConnection",
        lambda autocommit=False: forwarded.append(autocommit) or object(),
    )
    db.get_connection(autocommit=True)
    assert forwarded == [True]
