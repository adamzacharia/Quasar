"""Follow-ups from guard task-dd87861-1630 (CX-01, CX-02, CX-25): defects in
earlier uncommitted prompt-v2 work that were out of scope for that task."""
from types import SimpleNamespace as NS
import threading

import pytest

from core.prompts import playbooks as pb
from tests.unit.test_discovery_recovery import _Responses, _events, _tool_agent

pytestmark = pytest.mark.unit


def _truncated_round(round_id, text):
    return [
        NS(type="response.created", response=NS(id=f"resp-{round_id}")),
        NS(type="response.output_text.delta", delta=text),
        NS(type="response.completed", response=NS(finish_reason="length", output=[])),
    ]


def test_cx01_cutoff_continuation_wins_over_a_stale_playbook_followup(monkeypatch):
    """Round 1 calls a tool and activates a follow-up playbook (_next_input);
    round 2 is cut off by the provider. Round 3 must carry the cutoff
    instruction, not re-send round 1's tool outputs and playbook item."""
    monkeypatch.delenv("QUASAR_BENCH_TOOLSET", raising=False)
    monkeypatch.setattr(pb, "select_tool_followups",
                        lambda names, flags, state, budget=pb.FOLLOWUP_BUDGET: [pb.get_playbook("density_maps")])
    responses = _Responses([
        _events(1, tool="query_archive"),
        _truncated_round(2, "The density map shows the clump near"),
        _events(3, text=" RA 10.5, Dec -3.2."),
    ])
    agent, executed = _tool_agent(responses)
    agent.prompt_bundle = "v2"
    agent.stream_response_api("hello there", conversation_id="cx01-trunc")
    assert len(executed) == 1 and len(responses.calls) == 3
    round2_input = responses.calls[1]["input"]
    assert isinstance(round2_input, list) and any(
        isinstance(i, dict) and i.get("role") == "user" and "Turn context update" in str(i.get("content"))
        for i in round2_input
    ), "precondition: the follow-up playbook item was sent after the tool round"
    round3_input = responses.calls[2]["input"]
    assert isinstance(round3_input, str) and round3_input.startswith("[SYSTEM CONTINUATION]")
    assert "cut off" in round3_input


def _shim(prune=True):
    from core.llm_client import ResponsesShim

    fake = object.__new__(ResponsesShim)
    fake._history_lock = threading.Lock()
    fake._prune_turn_context = prune
    ctx = pb.render_turn_context("2026-09-26 (CDT)", "", [pb.get_playbook("open_regions")])
    fake._history_cache = {"r1": [{"role": "system", "content": "S"},
                                  {"role": "user", "content": "q1" + ctx},
                                  {"role": "assistant", "content": "partial"}]}
    fake._build_chat_messages = lambda *a, **k: []
    return fake, ctx


def test_cx02_chat_shim_keeps_current_turn_context_on_a_string_continuation():
    fake, ctx = _shim()
    cont = "[SYSTEM CONTINUATION] Your previous message was cut off before it finished."
    msgs = fake._chat_messages_for_input("r1", "IGNORED", cont)
    assert msgs[1]["content"] == "q1" + ctx          # same turn: date + playbooks kept
    assert msgs[-1]["content"] == cont
    # a genuine new user turn still prunes
    msgs2 = fake._chat_messages_for_input("r1", "IGNORED", "q2" + ctx)
    assert msgs2[1]["content"] == "q1"


def test_cx02_anthropic_shim_keeps_current_turn_context_on_a_string_continuation():
    fake, ctx = _shim()
    fake._build_anthropic_messages = lambda input_data, attachments=None: [{"role": "user", "content": input_data}]
    cont = "[SYSTEM CONTINUATION] The tool budget for this turn is spent (x). Answer the user NOW."
    msgs = fake._anthropic_messages_for_input("r1", cont)
    assert msgs[1]["content"] == "q1" + ctx
    msgs2 = fake._anthropic_messages_for_input("r1", "q2" + ctx)
    assert msgs2[1]["content"] == "q1"


def _out(text):
    return {"type": "function_call_output", "call_id": "c", "output": text}


@pytest.mark.parametrize(("output", "expected"), [
    ('{"success": true, "coverage_gap": false, "row_cap": false}', set()),
    ("{'coverage_gap': False, 'row_cap_hit': False}", set()),
    ('{"row_cap": 500, "truncated": false}', set()),                 # a configured limit, not a hit
    ('{"panels": [{"coverage_gap": false}], "coverage_gap": true}', {"coverage_gap"}),
    ("{'coverage_gap': True}", {"coverage_gap"}),
    ('{"row_cap_hit": true}', {"row_cap"}),
    ('{"warnings": ["Result hit its row cap (LIMIT 5000)"]}', {"row_cap"}),
])
def test_cx25_result_flags_are_read_by_value(output, expected):
    assert pb.result_flags_from_outputs([_out(output)]) == expected
