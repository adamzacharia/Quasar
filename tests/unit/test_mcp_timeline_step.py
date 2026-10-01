"""Research-timeline metadata for user MCP tool calls (core/runner.py).

A failed MCP call (isError, exception, timeout) comes back as a
{"success": False} dict and used to close its timeline step as a plain
"completed", so the UI could not tell success from failure. The runner now
tags MCP calls with their server and outcome over the status channel.
"""

import json

from core.runner import MCP_STEP_SENTINEL, _emit_mcp_step, _user_mcp_step_meta
from core.tools import Tool


def _tool(server=None, mcp_tool=None):
    return Tool(name="x", description="", function=lambda **_: None, parameters={},
                category="mcp", mcp_server=server, mcp_tool=mcp_tool)


def test_non_mcp_tools_get_no_metadata():
    assert _user_mcp_step_meta(None, "alma_query", "Querying ALMA") is None
    assert _user_mcp_step_meta(_tool(), "x", "Running x") is None


def test_label_names_server_and_keeps_argument_hint():
    tool = _tool("Astropy", "search_astropy_documentation")
    generic = 'Running Astropy  search astropy documentation ("SkyCoord")'
    meta = _user_mcp_step_meta(tool, "Astropy__search_astropy_documentation", generic)
    assert meta == {
        "step": 'Astropy: search astropy documentation ("SkyCoord")',
        "server": "Astropy",
        "tool": "search_astropy_documentation",
    }


def test_emitted_sentinel_round_trips():
    seen = []
    _emit_mcp_step(lambda step, state: seen.append((step, state)),
                   {"step": "Astropy: fetch", "server": "Astropy", "ok": False, "error": "401"}, "error")
    (step, state), = seen
    assert state == "error" and step.startswith(MCP_STEP_SENTINEL)
    assert json.loads(step[len(MCP_STEP_SENTINEL):])["error"] == "401"


def test_emit_never_raises():
    def broken(step, state):
        raise RuntimeError("queue closed")

    _emit_mcp_step(broken, {"step": "s", "server": "a"}, "running")


def test_skipped_repeat_of_mcp_call_is_one_badged_failure():
    from core.runner import _report_skipped_repeat

    seen = []
    meta = {"step": "Monocrawl: get balance", "server": "Monocrawl", "tool": "get_balance"}
    _report_skipped_repeat(lambda s, st: seen.append((s, st)), "Running Monocrawl  get balance", meta,
                           "identical call already failed this turn", failed=True, reason="timed out")
    assert seen[0] == ("Monocrawl: get balance (repeat skipped)", "error")
    sent = json.loads(seen[1][0][len(MCP_STEP_SENTINEL):])
    assert sent["step"] == "Monocrawl: get balance (repeat skipped)"
    assert sent["ok"] is False and "timed out" in sent["error"]
    # No plain "running" label left open for the UI to show as a stray chip.
    assert all(not s.startswith("Running ") for s, _ in seen)


def test_skipped_repeat_of_builtin_tool_is_unchanged():
    from core.runner import _report_skipped_repeat

    seen = []
    _report_skipped_repeat(lambda s, st: seen.append((s, st)), "Querying ALMA", None,
                           "identical call already ran this turn", failed=False)
    assert seen == [("Querying ALMA", "running"),
                    ("Querying ALMA skipped — identical call already ran this turn", "completed")]


# ── guard task-84382ab-5601 fixes ─────────────────────────────────────────

def test_close_meta_marks_failure_and_scrubs_credential_urls():  # CX-01, CX-06
    from core.runner import _mcp_close_meta

    meta = {"step": "Hub: search", "server": "Hub", "tool": "search"}
    failed = {"success": False, "error": "Client error '401' for url 'https://h.io/mcp?token=abc123&q=1'"}
    state, payload = _mcp_close_meta(meta, json.dumps(failed), False, 1.25)
    assert state == "error" and payload["ok"] is False and payload["ms"] == 1250
    assert "abc123" not in payload["error"] and "token=[REDACTED]" in payload["error"]

    state, payload = _mcp_close_meta(meta, {"success": True, "data": "x"}, False, 0.2)
    assert state == "completed" and payload["ok"] is True and payload["error"] is None

    state, payload = _mcp_close_meta(meta, {"success": True}, True, 150)
    assert state == "error" and payload["error"] == "timed out"


def test_repeat_calls_get_distinct_labels():  # CX-02
    tool = _tool("Hub", "search")
    used = {}
    labels = [_user_mcp_step_meta(tool, "Hub__search", "Running Hub  search", used)["step"] for _ in range(3)]
    assert labels == ["Hub: search", "Hub: search · call 2", "Hub: search · call 3"]


def test_sse_attaches_outcome_to_persisted_steps_and_emits_event():  # CX-06
    import sys
    from pathlib import Path

    ui_pro = str(Path(__file__).resolve().parents[2] / "ui-pro")
    if ui_pro not in sys.path:
        sys.path.insert(0, ui_pro)
    from api.sse import _apply_mcp_step

    rich = [{"step": "Hub: search", "state": "running"}, {"step": "Other", "state": "completed"},
            {"step": "Hub: search", "state": "error"}]
    sentinel = MCP_STEP_SENTINEL + json.dumps(
        {"step": "Hub: search", "server": "Hub", "tool": "search", "ok": False, "error": "401"})
    line = _apply_mcp_step(sentinel, "error", rich)
    event = json.loads(line[len("data: "):].strip())
    assert event["type"] == "mcp_step" and event["state"] == "error" and event["error"] == "401"
    assert rich[0]["mcp"]["ok"] is False and rich[2]["mcp"]["state"] == "error"
    assert "mcp" not in rich[1]
    assert _apply_mcp_step(MCP_STEP_SENTINEL + "{not json", "error", rich) is None


# ── full runner turn -> SSE persistence (guard verify CX-06, CX-08) ─────────

def _run_mcp_turn(monkeypatch, rounds, result):
    """Drive a real runner turn whose model calls the user MCP tool
    Hub__search, feeding every status through the SSE layer's own handling
    (sentinel -> _apply_mcp_step, else appended to the saved thinkingSteps)."""
    import sys
    from pathlib import Path
    from types import SimpleNamespace as NS

    ui_pro = str(Path(__file__).resolve().parents[2] / "ui-pro")
    if ui_pro not in sys.path:
        sys.path.insert(0, ui_pro)
    from api.sse import _apply_mcp_step
    from services import user_mcp
    from tests.unit.test_discovery_recovery import _Responses, _tool_agent

    tool = Tool(name="Hub__search", description="Search the hub", function=lambda **_: None,
                parameters={"type": "object", "properties": {}}, category="mcp",
                mcp_server="Hub", mcp_tool="search")
    monkeypatch.setattr(user_mcp, "get_user_mcp_pool", lambda: NS(tools_for=lambda uid: [tool]))
    agent, executed = _tool_agent(_Responses(rounds), result=result)
    saved, events = [], []

    def on_status(step, state):
        if step.startswith(MCP_STEP_SENTINEL):
            line = _apply_mcp_step(step, state, saved)
            if line:
                events.append(json.loads(line[len("data: "):]))
        else:
            saved.append({"step": step, "state": state})

    agent.stream_response_api("hello there", conversation_id="mcp-timeline", on_status=on_status)
    return saved, events, executed


def test_failing_mcp_call_is_saved_as_a_failed_badged_step(monkeypatch):
    from tests.unit.test_discovery_recovery import _events

    failure = {"success": False, "error": "Client error '401' for url 'https://hub.example/mcp?token=s3cret'"}
    saved, events, executed = _run_mcp_turn(
        monkeypatch, [_events(1, tool="Hub__search"), _events(2, text="It failed.")], failure)
    assert len(executed) == 1
    steps = [s for s in saved if s["step"] == "Hub: search"]
    assert [s["state"] for s in steps] == ["running", "error"], saved
    assert all(s["mcp"]["ok"] is False and s["mcp"]["state"] == "error" for s in steps)
    assert "s3cret" not in json.dumps(saved) + json.dumps(events)
    assert [e["state"] for e in events] == ["running", "error"]


def test_cached_repeat_of_successful_mcp_search_gets_a_badged_chip(monkeypatch):
    from tests.unit.test_discovery_recovery import _events

    saved, events, executed = _run_mcp_turn(
        monkeypatch,
        [_events(1, tool="Hub__search"), _events(2, tool="Hub__search"), _events(3, text="Done.")],
        {"success": True, "results": [1]},
    )
    assert len(executed) == 1, "the identical repeat is served from cache"
    repeat = [s for s in saved if "repeat skipped" in s["step"]]
    # "skipped", not "completed": sse.py extends the deadline only for completed steps (CX-09/CX-10).
    assert repeat and repeat[-1]["state"] == "skipped" and repeat[-1]["mcp"]["ok"] is True
    assert repeat[-1]["step"].startswith("Hub: search")


def test_repeat_skipped_chip_never_extends_the_turn_deadline():  # guard verify CX-09, CX-10
    """A cache-served MCP repeat closes as "skipped", which sse.py's progress
    rule (state == "completed") ignores; no label text is involved, so a real
    tool whose name ends in "(repeat skipped)" still counts as progress."""
    import inspect
    import sys
    from pathlib import Path

    from core.runner import MCP_SKIPPED_STATE, _report_skipped_repeat

    seen = []
    meta = {"step": "Hub: search", "server": "Hub", "tool": "search"}
    _report_skipped_repeat(lambda s, st: seen.append((s, st)), "", meta,
                           "identical call already ran this turn", failed=False)
    assert MCP_SKIPPED_STATE != "completed"
    assert {st for _, st in seen} == {MCP_SKIPPED_STATE}

    ui_pro = str(Path(__file__).resolve().parents[2] / "ui-pro")
    if ui_pro not in sys.path:
        sys.path.insert(0, ui_pro)
    import api.sse as sse

    src = inspect.getsource(sse)
    guard = src[src.index('step.startswith("__tool_heartbeat__")'):src.index("deadline.extend_for_progress(")]
    assert 'state == "completed"' in guard and "repeat skipped" not in guard
