import pytest

from services.issue_report_service import ChatDeadline, IssueReportService


def _started_run(service: IssueReportService, run_id: str = "run-1"):
    service.start_run(
        run_id=run_id,
        user_id="user-1",
        conversation_id="conv-1",
        trace_id="trace-1",
        model="gpt-oss-120b",
        provider="tacc",
        key_source="platform",
    )
    service.finalize_run(
        run_id,
        status="timed_out",
        duration_ms=91_000,
        tools_called=["search_papers"],
        last_status="Searching papers",
        error_code="chat_inactivity_timeout",
        error_message="provider stalled",
        first_token_ms=None,
        provider_chunk_count=0,
    )


def test_report_without_consent_excludes_conversation_text(tmp_path):
    service = IssueReportService(str(tmp_path / "reports.db"))
    _started_run(service)

    report = service.create_report(
        user_id="user-1",
        run_id="run-1",
        message_id="message-1",
        category="stuck_slow",
        description="The second literature search stayed in Thinking.",
        include_context=False,
        prompt_excerpt="private prompt",
        response_excerpt="private response",
        technical_context={"user_agent": "test"},
    )

    assert report["prompt_excerpt"] == ""
    assert report["response_excerpt"] == ""
    assert report["provider"] == "tacc"
    assert report["technical_context"]["tools_called"] == ["search_papers"]
    assert report["technical_context"]["provider_chunk_count"] == 0


def test_report_with_consent_redacts_and_truncates_context(tmp_path):
    service = IssueReportService(str(tmp_path / "reports.db"))
    _started_run(service)
    secret = "sk-" + ("a" * 48)

    report = service.create_report(
        user_id="user-1",
        run_id="run-1",
        message_id="message-1",
        category="wrong_answer",
        description="The answer was unrelated.",
        include_context=True,
        prompt_excerpt=f"{secret} " + ("p" * 3000),
        response_excerpt="r" * 3000,
    )

    assert secret not in report["prompt_excerpt"]
    assert len(report["prompt_excerpt"]) <= 2000
    assert len(report["response_excerpt"]) == 2000


def test_report_must_reference_users_own_run(tmp_path):
    service = IssueReportService(str(tmp_path / "reports.db"))
    _started_run(service)

    with pytest.raises(LookupError):
        service.create_report(
            user_id="another-user",
            run_id="run-1",
            message_id="message-1",
            category="other",
            description="Cannot attach to another user's run.",
        )


def test_admin_filters_updates_and_csv_export(tmp_path):
    service = IssueReportService(str(tmp_path / "reports.db"))
    _started_run(service)
    report = service.create_report(
        user_id="user-1",
        run_id="run-1",
        message_id="message-1",
        category="stuck_slow",
        description="It got stuck.",
    )

    updated = service.update_report(
        report["id"],
        status="investigating",
        admin_notes="Reproducing against TACC.",
    )
    assert updated["status"] == "investigating"
    assert service.list_reports(provider="tacc", status="investigating")

    exported = service.export_reports_csv(provider="tacc")
    assert "Reproducing against TACC." in exported
    assert "private prompt" not in exported


def test_never_ending_stream_hits_inactivity_timeout():
    deadline = ChatDeadline(
        inactivity_seconds=90,
        standard_seconds=240,
        conductor_seconds=360,
        started_at=1000,
    )

    assert deadline.timeout_code(1089) == ""
    assert deadline.timeout_code(1090) == "chat_inactivity_timeout"


def test_conductor_mode_extends_only_total_deadline():
    deadline = ChatDeadline(
        inactivity_seconds=90,
        standard_seconds=240,
        conductor_seconds=360,
        started_at=1000,
    )
    deadline.mark_activity(1230)
    deadline.enable_conductor()

    assert deadline.timeout_code(1241) == ""
    assert deadline.timeout_code(1360) == "chat_inactivity_timeout"
