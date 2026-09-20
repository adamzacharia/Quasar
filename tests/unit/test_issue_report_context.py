"""Server-side conversation context on issue reports.

Before this, a thumbs-down report reached the maintainer as a bare
description ("this answer is wrong") with no question, no answer and no
surrounding turns: the consent box defaulted to off, the client excerpt was
capped at 2000 chars, and the admin panel never rendered excerpts anyway.
These tests pin the server-side capture: when the reporter consents, the
report carries the reported answer (anchored on the persisted run id), the
question before it, and the turns leading up to it, redacted and capped; when
they do not consent, nothing of the conversation text is stored.
"""

import json

import pytest

from services.issue_report_service import (
    MAX_CLIENT_CONTEXT_KEYS,
    MAX_CLIENT_CONTEXT_VALUE_CHARS,
    MAX_DIAGNOSTICS_CHARS,
    MAX_EXCERPT_MESSAGES,
    MAX_EXCERPT_TOOLS,
    MAX_RESPONSE_CHARS,
    IssueReportService,
    build_conversation_excerpt,
)


def _run(service: IssueReportService, run_id: str = "run-1", conversation_id: str = "conv-1"):
    service.start_run(
        run_id=run_id,
        user_id="user-1",
        conversation_id=conversation_id,
        trace_id="trace-1",
        model="gpt-oss-120b",
        provider="tacc",
        key_source="platform",
    )
    service.finalize_run(
        run_id,
        status="completed",
        duration_ms=4_000,
        tools_called=["search_by_target"],
        last_status="done",
        error_code=None,
        error_message=None,
        first_token_ms=800,
        provider_chunk_count=12,
    )


def _assistant(content: str, run_id: str, tools=None):
    meta = {"runMeta": {"run_id": run_id, "model": "gpt-oss-120b"}}
    if tools:
        meta["toolTrace"] = [{"tool": name} for name in tools]
    return {"role": "assistant", "content": content, "metadata": meta}


def _conversation():
    return [
        {"role": "user", "content": "Find ALMA data of M87"},
        _assistant("Here are 12 observations of M87.", "run-0", tools=["search_by_target"]),
        {"role": "user", "content": "How many Cycle 7 projects used 12m+7m+TP?"},
        _assistant("Zero projects used all three arrays in Cycle 7.", "run-1", tools=["query_alma_science_archive"]),
        {"role": "user", "content": "Are you sure?"},
        _assistant("Yes.", "run-2"),
    ]


def test_consenting_report_captures_question_answer_and_prior_turns(tmp_path):
    service = IssueReportService(str(tmp_path / "reports.db"), conversation_lookup=lambda cid, uid: _conversation())
    _run(service)

    report = service.create_report(
        user_id="user-1",
        run_id="run-1",
        message_id="message-1",
        category="wrong_answer",
        description="There are definitely projects that used all three arrays.",
        include_context=True,
        prompt_excerpt="How many Cycle 7 projects used 12m+7m+TP?",
        response_excerpt="Zero projects used all three arrays in Cycle 7.",
    )

    excerpt = report["conversation_excerpt"]
    # Anchored on the persisted run id, NOT on the last assistant message.
    assert excerpt[-1]["is_reported_answer"] is True
    assert excerpt[-1]["content"] == "Zero projects used all three arrays in Cycle 7."
    assert excerpt[-1]["tools"] == ["query_alma_science_archive"]
    assert excerpt[-2] == {
        "role": "user",
        "content": "How many Cycle 7 projects used 12m+7m+TP?",
        "truncated": False,
        "is_reported_answer": False,
    }
    # The turns before the question come along (the "context of the whole thing").
    assert [m["role"] for m in excerpt] == ["user", "assistant", "user", "assistant"]
    assert excerpt[0]["content"] == "Find ALMA data of M87"
    # Turns AFTER the reported answer are not part of what the reporter saw.
    assert all(m["content"] != "Yes." for m in excerpt)
    assert "conversation_excerpt_error" not in report["technical_context"]
    # Round-trips through list_reports and the CSV export.
    listed = service.list_reports(category="wrong_answer")[0]
    assert listed["conversation_excerpt"] == excerpt
    exported = service.export_reports_csv()
    assert "conversation_excerpt" in exported.splitlines()[0]
    assert "Zero projects used all three arrays" in exported
    found = service.reports_exist_for(message_ids=["message-1", "other"], run_ids=["run-1", "run-x", ""])
    assert found == {"message_ids": {"message-1"}, "run_ids": {"run-1"}}


def test_report_without_consent_stores_no_conversation_text(tmp_path):
    calls = []

    def lookup(cid, uid):
        calls.append((cid, uid))
        return _conversation()

    service = IssueReportService(str(tmp_path / "reports.db"), conversation_lookup=lookup)
    _run(service)

    report = service.create_report(
        user_id="user-1",
        run_id="run-1",
        message_id="message-1",
        category="wrong_answer",
        description="Wrong.",
        include_context=False,
        prompt_excerpt="private question",
        response_excerpt="private answer",
    )

    assert report["conversation_excerpt"] == []
    assert report["prompt_excerpt"] == ""
    assert report["response_excerpt"] == ""
    # Without consent the conversation store is never even read.
    assert calls == []
    assert "private" not in service.export_reports_csv()
    # The last SSE status label can embed the user's target name, so it is
    # withheld without consent; the tool list and timings stay.
    assert "last_status" not in report["technical_context"]
    assert report["technical_context"]["tools_called"] == ["search_by_target"]


def test_consenting_report_keeps_the_status_label_and_scopes_the_lookup_to_the_reporter(tmp_path):
    seen = {}

    def lookup(cid, uid):
        seen.update(cid=cid, uid=uid)
        return _conversation()

    service = IssueReportService(str(tmp_path / "reports.db"), conversation_lookup=lookup)
    _run(service)
    report = service.create_report(
        user_id="user-1", run_id="run-1", message_id="m", category="other",
        description="x", include_context=True,
    )
    assert report["technical_context"]["last_status"] == "done"
    # The store is asked for the REPORTER's view of the run's conversation.
    assert seen == {"cid": "conv-1", "uid": "user-1"}


def test_excerpt_is_refused_when_the_conversation_is_not_the_reporters(tmp_path):
    # The run row's conversation_id is only trusted as far as the store confirms
    # ownership: an ownership-scoped lookup returns None for someone else's
    # conversation, and no text is stored.
    service = IssueReportService(str(tmp_path / "reports.db"), conversation_lookup=lambda cid, uid: None)
    _run(service)
    report = service.create_report(
        user_id="user-1", run_id="run-1", message_id="m", category="other",
        description="x", include_context=True, prompt_excerpt="q", response_excerpt="a",
    )
    assert report["conversation_excerpt"] == []
    assert report["technical_context"]["conversation_excerpt_error"] == "conversation does not belong to reporter"


def test_excerpt_is_redacted_and_capped(tmp_path):
    secret = "sk-" + ("b" * 48)
    long_answer = "A" * (MAX_RESPONSE_CHARS + 500)
    convo = [
        {"role": "user", "content": f"my key is {secret} please use it"},
        _assistant(long_answer, "run-1"),
    ]
    service = IssueReportService(str(tmp_path / "reports.db"), conversation_lookup=lambda cid, uid: convo)
    _run(service)

    report = service.create_report(
        user_id="user-1",
        run_id="run-1",
        message_id="message-1",
        category="incorrect_data",
        description="Numbers look off.",
        include_context=True,
        response_excerpt=long_answer,
    )

    excerpt = report["conversation_excerpt"]
    assert secret not in excerpt[0]["content"]
    assert len(excerpt[1]["content"]) == MAX_RESPONSE_CHARS
    assert excerpt[1]["truncated"] is True
    assert excerpt[0]["truncated"] is False
    # The client excerpt honours the same, larger answer cap.
    assert len(report["response_excerpt"]) == MAX_RESPONSE_CHARS


def test_anchor_falls_back_to_answer_prefix_then_last_assistant():
    # Older persisted messages carry no runMeta: fall back to the reported
    # answer's text, then to the latest assistant message.
    convo = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "answer one is long enough to match"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "answer two"},
    ]
    by_prefix = build_conversation_excerpt(convo, run_id="missing", response_excerpt="answer one is long")
    assert by_prefix[-1]["content"] == "answer one is long enough to match"
    assert by_prefix[-1]["is_reported_answer"] is True
    assert [m["content"] for m in by_prefix] == ["q1", "answer one is long enough to match"]

    last = build_conversation_excerpt(convo, run_id="missing", response_excerpt="")
    assert last[-1]["content"] == "answer two"
    assert len(last) == 4

    assert build_conversation_excerpt([{"role": "user", "content": "only me"}], run_id="r") == []
    assert build_conversation_excerpt([], run_id="r") == []


def test_excerpt_window_and_total_budget():
    convo = []
    for turn in range(10):
        convo.append({"role": "user", "content": f"question {turn} " + ("x" * 300)})
        convo.append(_assistant(f"answer {turn} " + ("y" * 300), f"run-{turn}"))

    window = build_conversation_excerpt(convo, run_id="run-9")
    assert len(window) == MAX_EXCERPT_MESSAGES
    assert window[-1]["content"].startswith("answer 9")
    assert window[0]["content"].startswith("question 7")

    tight = build_conversation_excerpt(convo, run_id="run-9", max_total_chars=700)
    assert sum(len(m["content"]) for m in tight) <= 700
    # The budget is allocated newest-first: the reported answer and its
    # question arrive whole, one earlier turn is truncated, the rest dropped.
    assert tight[-1]["is_reported_answer"] is True
    assert tight[-1]["content"] == "answer 9 " + ("y" * 300)
    assert tight[-2]["content"] == "question 9 " + ("x" * 300)
    assert len(tight) == 3
    assert tight[0]["truncated"] is True and tight[0]["content"]


def test_long_earlier_turns_cannot_starve_the_question_and_answer():
    # CX-01: four 6000-char earlier messages used to consume the whole budget
    # oldest-first, leaving the question and the reported answer empty.
    convo = [
        {"role": "user", "content": "E" * 6000},
        _assistant("F" * 6000, "run-a"),
        {"role": "user", "content": "G" * 6000},
        _assistant("H" * 6000, "run-b"),
        {"role": "user", "content": "the actual question"},
        _assistant("the reported answer", "run-c"),
    ]
    excerpt = build_conversation_excerpt(convo, run_id="run-c")
    assert excerpt[-1]["content"] == "the reported answer"
    assert excerpt[-1]["is_reported_answer"] is True
    assert excerpt[-2]["content"] == "the actual question"
    assert sum(len(m["content"]) for m in excerpt) <= 24000
    # The oldest turn absorbs the shortfall.
    assert excerpt[0]["truncated"] is True


def test_prefix_fallback_matches_when_the_answer_starts_with_a_secret():
    # CX-03: the client excerpt reaches the anchor finder already redacted;
    # comparing it against the raw persisted text failed and anchored on the
    # wrong (later) assistant message.
    secret = "sk-" + ("c" * 48)
    convo = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": f"Your key {secret} is invalid; use the Data Lab token instead."},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "answer two"},
    ]
    from services.secret_redaction import redact_secrets

    client_excerpt = redact_secrets(convo[1]["content"])
    excerpt = build_conversation_excerpt(convo, run_id="missing", response_excerpt=client_excerpt)
    assert excerpt[-1]["is_reported_answer"] is True
    assert excerpt[-1]["content"].startswith("Your key ")
    assert secret not in excerpt[-1]["content"]
    assert all(m["content"] != "answer two" for m in excerpt)


def test_lookup_failure_still_files_the_report_and_records_why(tmp_path):
    def broken(cid, uid):
        raise RuntimeError("turso unreachable")

    service = IssueReportService(str(tmp_path / "reports.db"), conversation_lookup=broken)
    _run(service)

    report = service.create_report(
        user_id="user-1",
        run_id="run-1",
        message_id="message-1",
        category="stuck_slow",
        description="Hung.",
        include_context=True,
        prompt_excerpt="the question",
        response_excerpt="the partial answer",
    )

    assert report["conversation_excerpt"] == []
    assert "turso unreachable" in report["technical_context"]["conversation_excerpt_error"]
    # The client excerpts still provide something to read.
    assert report["prompt_excerpt"] == "the question"
    assert report["response_excerpt"] == "the partial answer"


def test_no_lookup_configured_is_recorded_not_fatal(tmp_path):
    service = IssueReportService(str(tmp_path / "reports.db"))
    _run(service)
    report = service.create_report(
        user_id="user-1",
        run_id="run-1",
        message_id="message-1",
        category="other",
        description="Hm.",
        include_context=True,
        prompt_excerpt="q",
        response_excerpt="a",
    )
    assert report["conversation_excerpt"] == []
    assert report["technical_context"]["conversation_excerpt_error"] == "conversation lookup not configured"


def test_anchor_compares_the_whole_excerpt_and_reports_its_method():
    # CX-03: two legacy answers sharing a long boilerplate opening must stay
    # distinguishable, and the admin view is told how the answer was found.
    shared = "Summary of the archive search: " + ("s" * 250)
    convo = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": shared + " FIRST distinct tail"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": shared + " SECOND distinct tail"},
    ]
    first = build_conversation_excerpt(convo, run_id="missing", response_excerpt=shared + " FIRST distinct tail")
    assert first[-1]["content"].endswith("FIRST distinct tail")
    assert first[-1]["anchor_method"] == "answer_text"
    assert "anchor_method" not in first[-2]

    exact = build_conversation_excerpt([_assistant("a", "run-1")], run_id="run-1")
    assert exact[-1]["anchor_method"] == "run_id"

    guess = build_conversation_excerpt(convo, run_id="missing", response_excerpt="")
    assert guess[-1]["anchor_method"] == "last_assistant"
    # Boilerplate this short is never used for matching.
    assert build_conversation_excerpt(convo, run_id="missing", response_excerpt="Yes.")[-1]["anchor_method"] == "last_assistant"


def test_tool_names_use_the_persisted_trace_shape_dedup_and_cap():
    # CX-15: core.provenance.persistable_trace stores {"name", "ok", ...}.
    # Duplicates and malformed entries are interleaved BEFORE the cap is
    # reached, so the dedup and the isinstance/empty-name guards are actually
    # exercised (a verifier showed the previous ordering never reached them).
    trace = [
        {"name": "tool_0", "ok": True},
        {"name": "tool_0", "ok": False},   # duplicate: must not repeat
        {"bogus": 1},                       # malformed: skipped
        "not-a-dict",                       # malformed: skipped
        {"name": ""},                       # empty name: skipped
        {"name": "tool_1", "ok": True},
        {"name": "tool_0"},                 # duplicate again
    ] + [{"name": f"tool_{i}", "ok": True} for i in range(2, 20)]
    msg = {"role": "assistant", "content": "a", "metadata": {"runMeta": {"run_id": "run-1"}, "toolTrace": trace}}
    excerpt = build_conversation_excerpt([msg], run_id="run-1")
    assert excerpt[-1]["tools"] == [f"tool_{i}" for i in range(MAX_EXCERPT_TOOLS)]
    assert len(excerpt[-1]["tools"]) == 12


def test_oversized_client_context_keeps_diagnostics_parsable(tmp_path):
    # CX-05: never slice serialized JSON; bound the client dict first, and
    # keep the mandatory keys (incl. the conversation lookup error).
    def broken(cid, uid):
        raise RuntimeError("store down")

    service = IssueReportService(str(tmp_path / "reports.db"), conversation_lookup=broken)
    _run(service)

    moderate = service.create_report(
        user_id="user-1", run_id="run-1", message_id="m-1", category="other", description="x",
        include_context=True, technical_context={"user_agent": "U" * 5000, "viewport": "1x1", "n": 3},
    )
    tc = moderate["technical_context"]
    assert tc["conversation_excerpt_error"] == "conversation lookup failed: store down"
    assert len(tc["client"]["user_agent"]) == MAX_CLIENT_CONTEXT_VALUE_CHARS
    assert tc["client"]["n"] == 3

    huge = service.create_report(
        user_id="user-1", run_id="run-1", message_id="m-2", category="other", description="x",
        include_context=True, technical_context={f"k{i}": "v" * 5000 for i in range(60)},
    )
    tc = huge["technical_context"]
    assert tc  # parsed, not replaced by {}
    assert tc["run_status"] == "completed"
    assert tc["conversation_excerpt_error"] == "conversation lookup failed: store down"
    assert tc["client"] == {"truncated": True}
    assert len(json.dumps(tc)) <= MAX_DIAGNOSTICS_CHARS
    assert MAX_CLIENT_CONTEXT_KEYS == 24


def test_reports_exist_for_chunks_large_id_lists(tmp_path):
    # CX-16: chunk boundary, duplicates and blanks across both key columns.
    service = IssueReportService(str(tmp_path / "reports.db"), conversation_lookup=lambda cid, uid: _conversation())
    _run(service)
    service.create_report(user_id="user-1", run_id="run-1", message_id="message-1", category="other", description="x")

    message_ids = [f"m{i}" for i in range(1200)] + ["message-1", "message-1", "", None]
    run_ids = [f"r{i}" for i in range(1100)] + ["run-1"]
    assert service.reports_exist_for(message_ids=message_ids, run_ids=run_ids) == {
        "message_ids": {"message-1"},
        "run_ids": {"run-1"},
    }
    assert service.reports_exist_for() == {"message_ids": set(), "run_ids": set()}


def test_conversation_lookup_can_be_bounded_to_the_tail(tmp_path):
    # CX-06: api.deps asks the store for the newest 400 messages only.
    from services.conversation_service import ConversationService

    store = ConversationService(str(tmp_path / "conv.db"))
    conv = store.create_conversation("user-1", "bounded", "gpt-oss-120b")
    for i in range(10):
        store.save_message(conv, "user" if i % 2 == 0 else "assistant", f"m{i}")

    tail = store.get_conversation_messages(conv, last_n=4)
    assert [m["content"] for m in tail] == ["m6", "m7", "m8", "m9"]
    everything = store.get_conversation_messages(conv)
    assert [m["content"] for m in everything] == [f"m{i}" for i in range(10)]
    # The excerpt still anchors correctly on the bounded tail.
    excerpt = build_conversation_excerpt(tail, run_id="missing", response_excerpt="m9")
    assert excerpt[-1]["content"] == "m9"


def test_non_finite_floats_are_dropped_and_poisoned_rows_still_list(tmp_path):
    # Python's json parser accepts NaN/Infinity tokens from a raw request body.
    # A stored NaN cannot be re-serialised by the admin listing (JSONResponse
    # refuses non-finite floats), which would 500 the panel for every admin.
    service = IssueReportService(str(tmp_path / "reports.db"), conversation_lookup=lambda cid, uid: _conversation())
    _run(service)
    report = service.create_report(
        user_id="user-1", run_id="run-1", message_id="m-1", category="other", description="x",
        include_context=True,
        technical_context={"viewport": float("nan"), "dpr": float("inf"), "ok": 2.5, "flag": True},
    )
    client = report["technical_context"]["client"]
    assert client == {"viewport": None, "dpr": None, "ok": 2.5, "flag": True}
    # Every listed row must survive json.dumps with allow_nan=False.
    for row in service.list_reports():
        json.dumps(row, allow_nan=False)

    # Rows poisoned BEFORE the write-side guard existed must not take the
    # listing down either.
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "reports.db"))
    conn.execute(
        "UPDATE issue_reports SET technical_context = ? WHERE id = ?",
        ('{"run_status": "completed", "client": {"x": NaN, "y": Infinity}}', report["id"]),
    )
    conn.commit()
    conn.close()
    listed = service.list_reports()[0]
    assert listed["technical_context"]["client"] == {"x": None, "y": None}
    json.dumps(listed, allow_nan=False)


def test_client_context_keys_are_redacted_and_empty_keys_dropped(tmp_path):
    service = IssueReportService(str(tmp_path / "reports.db"), conversation_lookup=lambda cid, uid: _conversation())
    _run(service)
    secret_key = "sk-" + ("k" * 40)
    report = service.create_report(
        user_id="user-1", run_id="run-1", message_id="m-1", category="other", description="x",
        include_context=True, technical_context={secret_key: "v", "": "dropped", "   ": "dropped", "page": "/"},
    )
    client = report["technical_context"]["client"]
    assert secret_key not in json.dumps(client)
    assert "" not in client
    assert client["page"] == "/"
    assert not any(k.startswith("sk-") for k in client)
    assert len(client) == 2  # the redacted key and "page"; blank keys dropped


def test_identical_legacy_answers_are_labelled_ambiguous():
    convo = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "Yes, done."},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "Yes, done."},
    ]
    excerpt = build_conversation_excerpt(convo, run_id="missing", response_excerpt="Yes, done.")
    assert excerpt[-1]["anchor_method"] == "answer_text_ambiguous"
    assert excerpt[-1]["is_reported_answer"] is True
    # A unique match keeps the confident label.
    convo[1]["content"] = "No, not yet."
    assert build_conversation_excerpt(convo, run_id="missing", response_excerpt="Yes, done.")[-1]["anchor_method"] == "answer_text"


def test_identifiers_are_capped_and_redacted(tmp_path):
    service = IssueReportService(str(tmp_path / "reports.db"), conversation_lookup=lambda cid, uid: _conversation())
    _run(service)
    huge = "m-" + ("x" * 10_000)
    report = service.create_report(
        user_id="user-1", run_id="run-1", message_id=huge, category="other", description="x",
    )
    assert len(report["message_id"]) == 128
    assert report["message_id"].startswith("m-x")


def test_lone_surrogates_from_the_client_do_not_break_the_insert(tmp_path):
    # Verify follow-up to CX-07: a client string clipped in the middle of an
    # emoji arrives as a lone UTF-16 surrogate; SQLite cannot bind it, so the
    # server sanitises every client-supplied field before the INSERT.
    service = IssueReportService(str(tmp_path / "reports.db"), conversation_lookup=lambda cid, uid: _conversation())
    _run(service)
    cut = "The answer was wrong \ud83d"
    report = service.create_report(
        user_id="user-1", run_id="run-1", message_id="m", category="other",
        description=cut, include_context=True, prompt_excerpt=cut, response_excerpt=cut,
        technical_context={"user_agent": cut},
    )
    for text in (
        report["description"], report["prompt_excerpt"], report["response_excerpt"],
        report["technical_context"]["client"]["user_agent"],
    ):
        text.encode("utf-8")  # raises if a lone surrogate survived
        assert text.startswith("The answer was wrong")


def test_existing_database_gains_the_excerpt_column(tmp_path):
    # A pre-existing issue_reports table without the column must be migrated
    # in place rather than crash the INSERT.
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """CREATE TABLE issue_reports (
            id TEXT PRIMARY KEY, run_id TEXT, user_id TEXT NOT NULL, conversation_id TEXT,
            message_id TEXT NOT NULL, category TEXT NOT NULL, description TEXT NOT NULL,
            include_context INTEGER NOT NULL DEFAULT 0, prompt_excerpt TEXT, response_excerpt TEXT,
            model TEXT, provider TEXT, trace_id TEXT, technical_context TEXT, status TEXT NOT NULL,
            admin_notes TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"""
    )
    conn.commit()
    conn.close()

    service = IssueReportService(str(db), conversation_lookup=lambda cid, uid: _conversation())
    _run(service)
    report = service.create_report(
        user_id="user-1", run_id="run-1", message_id="message-1", category="other",
        description="migrated", include_context=True,
    )
    assert report["conversation_excerpt"][-1]["is_reported_answer"] is True
