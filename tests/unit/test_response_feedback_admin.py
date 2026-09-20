"""Thumbs votes carry their run id and are reviewable with context.

A thumbs-down that never became a full issue report used to be visible only
inside the admin JSON export, as a 200-char prompt stub. These tests pin the
run_id linkage, the larger previews, the recent-votes listing the admin panel
reads, and the message_id join used to badge issue reports with their vote.
"""

import pytest

from services import analytics_service as analytics_module
from services.analytics_service import AnalyticsService


@pytest.fixture
def analytics(monkeypatch, tmp_path):
    monkeypatch.setattr(analytics_module, "_LOCAL_DB", str(tmp_path / "analytics.db"))
    return AnalyticsService()


def test_vote_stores_run_id_and_full_previews(analytics):
    question = "Q" * 700
    answer = "A" * 2500
    analytics.log_feedback(
        message_id="msg-1",
        feedback="dislike",
        conversation_id="conv-1",
        user_id="user-1",
        model="gpt-oss-120b",
        prompt_preview=question,
        response_preview=answer,
        run_id="run-1",
    )

    rows = analytics.list_recent_feedback(feedback="dislike")
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == "run-1"
    assert row["conversation_id"] == "conv-1"
    assert row["feedback"] == "dislike"
    # Whole question (600) and a readable slice of the answer (2000), not 200/500.
    assert len(row["prompt_preview"]) == 600
    assert len(row["response_preview"]) == 2000

    exported = analytics.export_feedback_json()
    assert exported[0]["run_id"] == "run-1"


def test_recent_listing_filters_and_orders(analytics):
    analytics.log_feedback(message_id="m-like", feedback="like", user_id="u", run_id="r1")
    analytics.log_feedback(message_id="m-dislike", feedback="dislike", user_id="u", run_id="r2")

    dislikes = analytics.list_recent_feedback(feedback="dislike")
    assert [r["message_id"] for r in dislikes] == ["m-dislike"]
    everything = analytics.list_recent_feedback(feedback=None, limit=10)
    assert {r["message_id"] for r in everything} == {"m-like", "m-dislike"}
    # Newest first.
    assert everything[0]["message_id"] == "m-dislike"


def test_revote_replaces_and_keeps_latest_run_id(analytics):
    analytics.log_feedback(message_id="msg-1", feedback="like", user_id="user-1", run_id="run-1")
    analytics.log_feedback(message_id="msg-1", feedback="dislike", user_id="user-1", run_id="run-1")

    rows = analytics.list_recent_feedback()
    assert len(rows) == 1
    assert rows[0]["feedback"] == "dislike"


def test_feedback_join_by_message_id(analytics):
    analytics.log_feedback(message_id="msg-1", feedback="dislike", user_id="u1")
    analytics.log_feedback(message_id="msg-2", feedback="like", user_id="u1")

    votes = analytics.get_feedback_for_messages(["msg-1", "msg-2", "msg-none", "", None])
    assert votes == {"msg-1": "dislike", "msg-2": "like"}
    assert analytics.get_feedback_for_messages([]) == {}


def test_revote_after_reload_replaces_by_run_id(analytics):
    # CX-02: a live answer carries a random client id; after a reload the same
    # answer is identified by its text_block_id. One run, two message ids,
    # still one vote.
    analytics.log_feedback(message_id="live-3f9a", feedback="like", user_id="u1", run_id="run-1")
    analytics.log_feedback(message_id="blk-1", feedback="dislike", user_id="u1", run_id="run-1")

    rows = analytics.list_recent_feedback()
    assert [(r["message_id"], r["feedback"]) for r in rows] == [("blk-1", "dislike")]
    assert analytics.get_feedback_for_runs(["run-1", "run-none", ""]) == {"run-1": "dislike"}
    # Another user's vote on the same run is untouched.
    analytics.log_feedback(message_id="blk-1", feedback="like", user_id="u2", run_id="run-1")
    assert len(analytics.list_recent_feedback()) == 2


def test_existing_feedback_table_gains_run_id_column(monkeypatch, tmp_path):
    import sqlite3

    db = tmp_path / "old-analytics.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """CREATE TABLE response_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT, message_id TEXT NOT NULL,
            conversation_id TEXT, user_id TEXT, feedback TEXT NOT NULL, model TEXT,
            prompt_preview TEXT, response_preview TEXT, created_at TEXT NOT NULL)"""
    )
    conn.execute(
        "INSERT INTO response_feedback (message_id, feedback, created_at) VALUES ('old', 'dislike', '2026-01-01T00:00:00')"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(analytics_module, "_LOCAL_DB", str(db))

    service = AnalyticsService()
    service.log_feedback(message_id="new", feedback="dislike", user_id="u", run_id="run-9")
    rows = service.list_recent_feedback(feedback="dislike", limit=10)
    by_id = {r["message_id"]: r for r in rows}
    assert by_id["old"]["run_id"] is None
    assert by_id["new"]["run_id"] == "run-9"
    # Timestamps are presented tz-aware so the admin UI does not render the
    # naive legacy rows as local time: old rows gain +00:00, new rows are
    # written with an offset in the first place.
    assert by_id["old"]["created_at"] == "2026-01-01T00:00:00+00:00"
    assert by_id["new"]["created_at"].endswith("+00:00")
    # The export keeps stored values verbatim (existing API contract, CX-11);
    # only the browser-facing listing normalises.
    exported = {r["message_id"]: r for r in service.export_feedback_json()}
    assert exported["old"]["timestamp"] == "2026-01-01T00:00:00"
    assert exported["new"]["run_id"] == "run-9"


def test_vote_lookups_chunk_large_id_lists(analytics):
    # CX-16: the IN queries are chunked at 500 ids; cross the boundary, include
    # duplicates and blanks, and aggregate across chunks.
    for i in range(3):
        analytics.log_feedback(message_id=f"msg-{i}", feedback="dislike", user_id="u", run_id=f"run-{i}")
    ids = [f"x{i}" for i in range(1200)] + ["msg-0", "msg-2", "msg-0", "", None]
    assert analytics.get_feedback_for_messages(ids) == {"msg-0": "dislike", "msg-2": "dislike"}
    runs = [f"y{i}" for i in range(600)] + ["run-1"]
    assert analytics.get_feedback_for_runs(runs) == {"run-1": "dislike"}
    assert analytics.get_feedback_for_runs([]) == {}


def test_lone_surrogate_previews_are_sanitized(analytics):
    analytics.log_feedback(
        message_id="m", feedback="dislike", user_id="u",
        prompt_preview="question \ud83d", response_preview="answer \ude00 tail",
    )
    row = analytics.list_recent_feedback()[0]
    row["prompt_preview"].encode("utf-8")
    row["response_preview"].encode("utf-8")
    assert row["prompt_preview"].startswith("question ")


def test_identifiers_are_capped_redacted_and_bindable(analytics):
    secret = "sk-" + ("e" * 48)
    analytics.log_feedback(
        message_id="m-" + ("x" * 10_000), feedback="dislike", user_id="u",
        conversation_id=f"conv {secret}", model="gpt \ud83d", run_id="run-1",
    )
    row = analytics.list_recent_feedback()[0]
    assert len(row["message_id"]) == 128
    assert secret not in row["conversation_id"]
    row["model"].encode("utf-8")
    # The capped id is the join key from now on.
    assert analytics.get_feedback_for_messages([row["message_id"]]) == {row["message_id"]: "dislike"}


def test_init_survives_a_genuinely_failed_run_id_migration(monkeypatch, tmp_path, capsys):
    # CX-08/CX-14: when the ALTER fails for a non-duplicate reason on a legacy
    # database, the service must warn and keep booting (the run_id index is
    # skipped) instead of raising out of the module-level constructor.
    import sqlite3

    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """CREATE TABLE response_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT, message_id TEXT NOT NULL,
            conversation_id TEXT, user_id TEXT, feedback TEXT NOT NULL, model TEXT,
            prompt_preview TEXT, response_preview TEXT, created_at TEXT NOT NULL)"""
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(analytics_module, "_LOCAL_DB", str(db))
    real_get_connection = analytics_module.get_connection

    class _Cursor:
        def __init__(self, cur):
            self._cur = cur

        def execute(self, sql, *args, **kwargs):
            if sql.strip().upper().startswith("ALTER TABLE RESPONSE_FEEDBACK"):
                raise sqlite3.OperationalError("simulated: DDL not permitted")
            return self._cur.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._cur, name)

    class _Conn:
        def __init__(self, inner):
            self._inner = inner

        def cursor(self):
            return _Cursor(self._inner.cursor())

        def execute(self, sql, *args, **kwargs):
            return _Cursor(self._inner.cursor()).execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(analytics_module, "get_connection", lambda *a, **k: _Conn(real_get_connection(*a, **k)))

    service = AnalyticsService()  # must not raise
    out = capsys.readouterr().out
    assert "response_feedback.run_id is missing" in out
    # Votes still record (unlinked) on the legacy schema? No: the column is
    # absent, so the INSERT would fail; that is the loud, expected failure the
    # warning announces. The point here is that boot succeeded.
    assert service is not None


def test_previews_are_secret_redacted_before_storage(analytics):
    # CX-09: previews are shown in the admin panel and are sent BEFORE the
    # consent checkbox appears, so secrets must never be persisted in them.
    secret = "sk-" + ("d" * 48)
    analytics.log_feedback(
        message_id="m", feedback="dislike", user_id="u",
        prompt_preview=f"my key is {secret}", response_preview=f"do not paste {secret} anywhere",
    )
    row = analytics.list_recent_feedback()[0]
    assert secret not in row["prompt_preview"]
    assert secret not in row["response_preview"]
    assert row["prompt_preview"].startswith("my key is ")
