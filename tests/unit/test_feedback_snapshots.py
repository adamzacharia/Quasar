import json
import sqlite3

import pytest

from services.conversation_service import ConversationService
from services.issue_report_service import IssueReportService
from services.feedback_snapshot_service import (
    MAX_SNAPSHOT_BYTES, MAX_SNAPSHOT_MESSAGES, FeedbackSnapshotService, FeedbackToolTrace,
    build_snapshot, persist_feedback_tool_trace, safe_snapshot_value, snapshot_json,
)


def message(i, content=None):
    return {"id": str(i), "role": "user" if i % 2 == 0 else "assistant",
            "content": content if content is not None else f"Turn {i}",
            "created_at": "2026-09-05T10:00:00+00:00",
            "metadata": {"runMeta": {"run_id": f"run-{i}", "model": "test", "provider": "test"}}}


def snapshot(messages, **kwargs):
    return build_snapshot(source={"messages": messages, "total_messages": len(messages)},
                          user_id="u1", conversation_id="c1", message_id="1", **kwargs)


def test_snapshot_keeps_every_turn_and_long_answer_including_after_reported_turn():
    messages = [message(i) for i in range(20)]
    messages[1]["content"] = "answer " * 4000
    snap = snapshot(messages, run={"id": "run-1", "model": "test", "provider": "test"}, description="wrong")
    assert snap["messages"] == [dict(m, is_reported_answer=i == 1) for i, m in enumerate(messages)]
    assert snap["feedback_text"] == "wrong" and snap["vote"] == "dislike"
    assert snap["truncation"]["truncated"] is False
    assert snap["captured_at"] and snap["provider"] == "test"
    messages[1]["content"] = "changed"
    assert len(snap["messages"][1]["content"]) == 28000


def test_redacts_keys_values_url_tokens_and_non_finite_or_surrogate_values():
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    value = {secret: secret, "password": "short", "headers": {"Authorization": "anything"},
             "url": "https://example.org?q=x&TOKEN=short", "nested": [float("nan"), "x\ud800", secret]}
    clean = safe_snapshot_value(value)
    text = json.dumps(clean, allow_nan=False)
    assert secret not in text and "short" not in text
    assert clean["password"] == "[REDACTED]"
    assert clean["nested"][0] is None
    assert "TOKEN=[REDACTED]" in clean["url"]
    text.encode("utf-8")


def test_message_limit_reports_exact_count_and_marker():
    snap = snapshot([message(i) for i in range(403)])
    assert len(snap["messages"]) == MAX_SNAPSHOT_MESSAGES
    assert snap["truncation"]["marker"] == "Truncated at 400 of 403 messages"
    assert snap["truncation"]["reasons"] == ["message_limit"]


def test_byte_limit_drops_whole_messages_and_keeps_valid_json():
    snap = snapshot([message(0), message(1, "😀" * 20000), message(2)], max_bytes=8192)
    assert len(snap["messages"]) == 1
    assert snap["truncation"]["marker"] == "Truncated at 1 of 3 messages"
    assert "snapshot_byte_limit" in snap["truncation"]["reasons"]
    assert len(snapshot_json(snap).encode("utf-8")) <= 8192
    assert json.loads(snapshot_json(snap)) == snap


def test_tool_calls_and_results_are_copied_redacted_and_bounded():
    calls = [{"name": "lookup", "arguments": {"api_key": "short", "target": "M87"},
              "output": {"rows": [{"flux": 1.2}]}}]
    trace = persist_feedback_tool_trace(calls)
    assert trace["calls"][0]["arguments"]["api_key"] == "[REDACTED]"
    assert trace["calls"][0]["output"] == calls[0]["output"]
    assert trace["truncated"] is False
    calls.append({"name": "huge", "output": "x" * MAX_SNAPSHOT_BYTES})
    trace = persist_feedback_tool_trace(calls)
    assert trace["truncated"] and trace["total_calls"] == 2 and len(trace["calls"]) == 1


@pytest.fixture
def stores(tmp_path):
    conv = ConversationService(str(tmp_path / "conversations.db"))
    cid = conv.create_conversation("u1")
    conv.save_full_conversation(cid, [message(i) for i in range(12)])
    reports = IssueReportService(str(tmp_path / "reports.db"),
        conversation_lookup=conv.get_conversation_messages_for_user,
        snapshot_lookup=conv.get_feedback_snapshot_source)
    reports.start_run(run_id="run-1", user_id="u1", conversation_id=cid,
                      trace_id="trace-1", model="test", provider="test", key_source="platform")
    return conv, reports, cid


def test_report_snapshot_survives_edits_deletion_and_service_restart(stores):
    conv, reports, cid = stores
    report = reports.create_report(user_id="u1", run_id="run-1", message_id="1",
        category="wrong_answer", description="test evidence", include_context=True)
    snap = reports.snapshots.get(report["snapshot_id"])
    assert len(snap["messages"]) == 12
    assert all(m["id"] and m["created_at"] for m in snap["messages"])
    conv.save_full_conversation(cid, [{"role": "user", "content": "replacement"}])
    conv.delete_conversation_for_user(cid, "u1")
    assert FeedbackSnapshotService(reports._local_db_path).get(snap["id"]) == snap


@pytest.mark.parametrize("consent", [False, None, "true", 1])
def test_consent_off_never_reads_or_stores_snapshot(stores, consent):
    conv, reports, cid = stores
    reports._snapshot_lookup = lambda *a: pytest.fail("context lookup without consent")
    snap, error = reports.build_feedback_snapshot(user_id="u1", conversation_id=cid,
        message_id="1", run_id="run-1", include_context=consent)
    assert snap is None and not error
    if consent is False:
        report = reports.create_report(user_id="u1", run_id="run-1", message_id="1",
            category="other", description="no context", include_context=False)
        assert not report["snapshot_id"] and not report["conversation_excerpt"]
        with sqlite3.connect(reports._local_db_path) as db:
            assert db.execute("SELECT COUNT(*) FROM feedback_snapshots").fetchone()[0] == 0


def test_ownership_checks_are_in_source_read_and_run_capture(stores):
    conv, reports, cid = stores
    assert conv.get_feedback_snapshot_source(cid, "intruder") is None
    snap, error = reports.build_feedback_snapshot(user_id="intruder", conversation_id=cid,
        message_id="1", run_id="run-1", include_context=True)
    assert snap is None and "belong" in error
    snap, error = reports.build_feedback_snapshot(user_id="intruder", conversation_id=cid,
        message_id="1", include_context=True)
    assert snap is None and "belong" in error


def test_source_reads_only_400_messages_with_explicit_total(stores):
    conv, reports, cid = stores
    conv.save_full_conversation(cid, [message(i) for i in range(405)])
    source = conv.get_feedback_snapshot_source(cid, "u1")
    assert source["total_messages"] == 405 and len(source["messages"]) == 400
    assert source["messages"][0]["content"] == "Turn 0"
    assert source["messages"][-1]["content"] == "Turn 399"


def test_oversize_source_never_returns_cut_json_or_cut_text(stores):
    conv, reports, cid = stores
    conv.save_full_conversation(cid, [message(0), message(1, "x" * (MAX_SNAPSHOT_BYTES + 1)), message(2)])
    source = conv.get_feedback_snapshot_source(cid, "u1")
    assert source["source_truncated"] and source["messages"][1]["source_omitted"]
    snap, error = reports.build_feedback_snapshot(user_id="u1", conversation_id=cid,
        message_id="1", run_id="run-1", include_context=True)
    assert not error and snap["truncation"]["marker"] == "Truncated at 1 of 3 messages"


def test_snapshot_and_report_write_are_atomic(stores, monkeypatch):
    conv, reports, cid = stores
    real_insert = reports.snapshots.insert
    def failing(snapshot, conn=None):
        real_insert(snapshot, conn=conn)
        raise RuntimeError("failure after snapshot insert")
    monkeypatch.setattr(reports.snapshots, "insert", failing)
    with pytest.raises(RuntimeError):
        reports.create_report(user_id="u1", run_id="run-1", message_id="1",
            category="other", description="atomic", include_context=True)
    with sqlite3.connect(reports._local_db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM feedback_snapshots").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM issue_reports").fetchone()[0] == 0


@pytest.mark.parametrize("key", ["auth", "session", "session_id", "sessionid", "sid", "signature", "sig", "credentials", "refresh_token"])
def test_nested_credential_keys_are_redacted_even_for_short_values(key):
    assert safe_snapshot_value({"metadata": {key: "secret"}})["metadata"][key] == "[REDACTED]"


def test_agent_trace_collects_full_results_beyond_compact_trace_limits():
    import threading
    from core.agent import QuasarAgent
    from core.llm_client import llm_request_context, get_llm_request_context, reinstall_llm_request_context
    class Agent:
        _tls = threading.local()
        _accumulated_tool_trace = []
    agent = Agent()
    result = {"data": "z" * 5000}
    with llm_request_context(user_id="u1"):
        ctx = get_llm_request_context()
        for i in range(205):
            QuasarAgent._record_tool_trace(agent, f"call-{i}", {"i": i}, "clipped", result_obj=result)
        def worker():
            with reinstall_llm_request_context(ctx):
                QuasarAgent._record_tool_trace(agent, "worker-call", {}, "short", result_obj=["full", "result"])
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        trace = ctx.feedback_tool_trace.snapshot()
    assert len(agent._accumulated_tool_trace) == 200  # existing display contract
    assert len(trace["calls"]) == trace["total_calls"] == 206
    assert trace["calls"][204]["output"] == result
    assert trace["calls"][-1]["output"] == ["full", "result"]
    assert trace["truncated"] is False
    # The captured evidence is a snapshot source, not the compact display trace.
    turn = message(1)
    turn["metadata"]["feedbackToolTrace"] = trace
    snap = snapshot([turn])
    assert len(snap["messages"][0]["metadata"]["feedbackToolTrace"]["calls"]) == 206
    with llm_request_context(user_id="u2"):
        assert get_llm_request_context().feedback_tool_trace.snapshot()["calls"] == []


def test_tool_trace_byte_bound_counts_omitted_calls():
    trace = FeedbackToolTrace()
    trace.record("small", {}, {"ok": True})
    trace.record("huge", {}, "x" * MAX_SNAPSHOT_BYTES)
    trace.record("later", {}, "later result")
    value = trace.snapshot()
    assert len(value["calls"]) == 1 and value["total_calls"] == 3 and value["truncated"]
    assert len(snapshot_json(value).encode("utf-8")) < value["max_bytes"]
