"""Exercise the real SSE generator through completed and interrupted turns."""
import asyncio
import threading
import time
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ui-pro"))

from services.conversation_service import ConversationService
from services.issue_report_service import IssueReportService
from core.llm_client import get_llm_request_context


@pytest.mark.parametrize("outcome,emit_text", [
    ("completed", True), ("failed", True), ("timed_out", True),
    ("cancelled", True), ("cancelled", False), ("closed", True),
])
def test_sse_saves_tool_evidence_even_without_done(monkeypatch, tmp_path, outcome, emit_text):
    from api import sse
    from api.models import ChatRequest
    import core.langfuse_integration as langfuse

    conv = ConversationService(str(tmp_path / "chat.db"))
    cid = conv.create_conversation("u1")
    reports = IssueReportService(str(tmp_path / "reports.db"),
        snapshot_lookup=conv.get_feedback_snapshot_source,
        conversation_lookup=conv.get_conversation_messages_for_user)
    ready, release, worker_done, persisted = (threading.Event() for _ in range(4))
    captured = {}
    real_save = conv.save_message

    def save(*args, **kwargs):
        real_save(*args, **kwargs)
        if args[1] == "assistant":
            captured["metadata"] = kwargs["metadata"]
            captured["text"] = args[2]
            persisted.set()

    def stream(*args, **kwargs):
        collector = get_llm_request_context().feedback_tool_trace
        captured["collector"] = collector
        collector.record("lookup", {"api_key": "short-secret"}, {"result": "x" * 7000})
        if emit_text:
            kwargs["on_token"]("Partial answer")
        ready.set()
        try:
            if outcome in {"timed_out", "cancelled", "closed"}:
                assert release.wait(4), "test did not release fake worker"
                collector.record("late_call", {}, {"late": True})
            if outcome == "failed":
                raise RuntimeError("simulated worker failure")
            return "Partial answer"
        finally:
            worker_done.set()

    agent = SimpleNamespace(
        config=SimpleNamespace(model="gpt-oss-120b"),
        memory=SimpleNamespace(clear=lambda: None, sync_from_ui=lambda h: None),
        stream_response_api=stream, cancel_response_run=lambda *a: None,
        _accumulated_tool_trace=[], _accumulated_run_results=[],
        last_run_result=None, _tls=threading.local(),
    )
    monkeypatch.setattr(sse, "get_agent", lambda: agent)
    monkeypatch.setattr(sse, "_resolve_optional_user", lambda *a, **k: {"sub": "u1", "email": "test@example.org"})
    monkeypatch.setattr(sse, "_build_llm_context_for_user", lambda uid: {
        "provider_api_keys": {}, "key_source_by_provider": {}, "byok_token_limits": {}})
    monkeypatch.setattr(sse, "_make_usage_recorder", lambda *a: lambda **k: None)
    monkeypatch.setattr(sse, "_make_quota_checker", lambda *a: lambda **k: None)
    monkeypatch.setattr(sse, "_make_quota_releaser", lambda: lambda *a, **k: None)
    monkeypatch.setattr(sse.usage_quota_service, "ensure_allowed", lambda **k: None)
    monkeypatch.setattr(sse.analytics_service, "log_chat", lambda **k: None)
    monkeypatch.setattr(langfuse, "get_langfuse", lambda: None)
    monkeypatch.setattr(sse, "conversation_service", conv)
    monkeypatch.setattr(sse, "issue_report_service", reports)
    monkeypatch.setattr(conv, "save_message", save)
    if outcome == "timed_out":
        class Deadline:
            def __init__(self, **k): pass
            def remaining(self, now): return 0.02
            def timeout_code(self, now): return "test_timeout" if ready.is_set() else None
            def mark_activity(self, now): pass
        monkeypatch.setattr(sse, "ChatDeadline", Deadline)

    async def run():
        response = await sse._stream_chat_response(ChatRequest(
            message="capture this tool result", conversation_id=cid,
            model="gpt-oss-120b", grounded_summary=True))
        async def consume():
            async for event in response.body_iterator:
                if outcome == "closed" and '"type": "token"' in event:
                    await response.body_iterator.aclose()
                    break
        task = asyncio.create_task(consume())
        try:
            assert await asyncio.to_thread(ready.wait, 2)
            if outcome == "cancelled":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                await asyncio.wait_for(task, 2)
            assert await asyncio.to_thread(persisted.wait, 2), "assistant evidence was not persisted"
            trace = captured["metadata"]["feedbackToolTrace"]
            assert trace["total_calls"] == 1
            assert trace["calls"][0]["output"] == {"result": "x" * 7000}
            assert trace["calls"][0]["arguments"]["api_key"] == "[REDACTED]"
            assert trace["run_status"] == ("cancelled" if outcome == "closed" else outcome)
            assert trace["partial"] == (outcome != "completed")
            if not emit_text:
                assert captured["text"] == ""
            run_id = captured["metadata"]["runMeta"]["run_id"]
            report = reports.create_report(user_id="u1", run_id=run_id, message_id="m1",
                category="stuck_slow", description="verify interruption evidence", include_context=True)
            snap = reports.snapshots.get(report["snapshot_id"])
            assert snap["messages"][-1]["metadata"]["feedbackToolTrace"] == trace
            release.set()
            assert await asyncio.to_thread(worker_done.wait, 2)
            await asyncio.sleep(0.05)  # drain the fake worker's final queue callback
            assert captured["collector"].snapshot() == trace
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_freezing_does_not_wait_for_large_result_serialization(monkeypatch):
    import services.feedback_snapshot_service as snapshots
    preparing, release = threading.Event(), threading.Event()
    real_clean = snapshots.safe_snapshot_value
    def blocked_clean(value, depth=0):
        if depth == 0:
            preparing.set()
            assert release.wait(2)
        return real_clean(value, depth)
    monkeypatch.setattr(snapshots, "safe_snapshot_value", blocked_clean)
    collector = snapshots.FeedbackToolTrace()
    worker = threading.Thread(target=collector.record, args=("tool", {}, "result"))
    worker.start()
    try:
        assert preparing.wait(1)
        # This would block behind serialization if it held the collector lock.
        freeze_started = time.monotonic()
        frozen = collector.freeze("cancelled")
        assert time.monotonic() - freeze_started < 0.5
        assert frozen["partial"] and frozen["calls"] == []
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert collector.snapshot() == frozen
