"""gpt-oss harmony control-token leak — filtered out of user-visible text.

Live 2026-09-21 23:56 (gpt-oss-120b via TACC, the round after a tool result):
the chat-completions stream carried a proper structured tool call
(tool_calls=1) AND, as ``content`` (output_chars=188), the raw harmony text of
that same call::

    <|channel|>commentary to=functions.match_perseus_protostars_alma_jwst
    <|constrain|>json<|message|>{"max_alma_rows":5000,...,"radius_arcsec":5}

The UI rendered it above the answer. These tests pin: the TACC stream
translator strips the span (tokens, header and tool payload) from the
``response.output_text.delta`` events and from the cached history, never drops
the structured call, forwards reasoning as before; the filter is
split-boundary safe; the runner scrubs any residue from the final text.
"""

from types import SimpleNamespace as NS

import pytest

from core.harmony_filter import HarmonyStreamFilter, strip_harmony_markup
from core.llm_client import LLMClient
from tests.unit.test_discovery_recovery import _Responses, _events, _tool_agent

TOOL = "match_perseus_protostars_alma_jwst"
ARGS = '{"max_alma_rows":5000,"max_mast_results_per_source":80,"max_sources":12,"radius_arcsec":5}'
LEAK = f"<|channel|>commentary to=functions.{TOOL} <|constrain|>json<|message|>{ARGS}"

_TOOLS = [{
    "type": "function",
    "name": TOOL,
    "description": "cross-match",
    "parameters": {"type": "object", "properties": {}},
}]


def _chunk(content=None, tool_calls=None, reasoning=None, finish_reason=None, usage=None):
    if usage is not None:
        return NS(choices=[], usage=usage)
    return NS(
        choices=[NS(
            delta=NS(content=content, tool_calls=tool_calls, reasoning_content=reasoning),
            finish_reason=finish_reason,
        )],
        usage=None,
    )


def _observed_stream():
    """The chunk sequence observed live: reasoning, the leaked harmony text as
    token-sized content deltas, then the structured call, then the usage tail."""
    yield _chunk(reasoning="The first pass returned nothing within 2 arcsec.")
    yield _chunk(reasoning=" I should widen the radius and re-run the matcher.")
    for piece in (
        "<|channel|>", "commentary", " to=functions.", TOOL, " ", "<|constrain|>", "json",
        "<|message|>", '{"max_alma_rows":', "5000,", '"max_mast_results_per_source":80,',
        '"max_sources":12,', '"radius_arcsec":5}',
    ):
        yield _chunk(content=piece)
    yield _chunk(tool_calls=[NS(index=0, id="call_live_1", function=NS(name=TOOL, arguments=""))])
    yield _chunk(tool_calls=[NS(index=0, id=None, function=NS(name=None, arguments=ARGS))])
    yield _chunk(finish_reason="tool_calls")
    yield _chunk(usage=NS(prompt_tokens=1200, completion_tokens=90))


def _tacc_client(stream_factory):
    client = LLMClient(model="gpt-oss-120b")
    engine = NS(create=lambda **kw: stream_factory())
    client._get_tacc_client = lambda: NS(chat=NS(completions=engine))
    return client


def test_observed_leak_is_188_chars():
    # Matches the [PROVIDER] log line of the live event (output_chars=188).
    assert len(LEAK) == 188


def test_tacc_stream_strips_leaked_tool_call_text_but_keeps_the_structured_call(capsys):
    client = _tacc_client(_observed_stream)
    events = list(client.responses.create(
        model="gpt-oss-120b", instructions="sys", input="widen the match", tools=_TOOLS, stream=True,
    ))

    text_deltas = [e.delta for e in events if e.type == "response.output_text.delta"]
    assert text_deltas == []  # nothing user-visible: the whole content was the leaked call
    assert not any("<|" in e.delta for e in events if e.delta)

    completed = next(e.response for e in events if e.type == "response.completed")
    assert [fc.name for fc in completed.output] == [TOOL]
    assert completed.output[0].arguments == ARGS
    assert completed.output[0].call_id == "call_live_1"
    assert completed.finish_reason == "tool_calls"
    assert completed.usage.output_tokens == 90

    # Reasoning is forwarded untouched and closes exactly once, before the call.
    reasoning = [e.delta for e in events if e.type == "response.reasoning_summary_text.delta"]
    assert reasoning == [
        "The first pass returned nothing within 2 arcsec.",
        " I should widen the radius and re-run the matcher.",
    ]
    types = [e.type for e in events]
    assert types.count("response.reasoning_summary_text.done") == 1
    assert types.index("response.reasoning_summary_text.done") < types.index("response.output_item.added")

    # The cached history never feeds the markup back to the model.
    cached = client.responses._history_cache[completed.id]
    assert cached[-1]["role"] == "assistant"
    assert cached[-1]["content"] is None
    assert cached[-1]["tool_calls"][0]["function"] == {"name": TOOL, "arguments": ARGS}

    out = capsys.readouterr().out
    assert f"leaked harmony tool-call markup for functions.{TOOL}" in out
    assert "also issued structurally this round" in out
    # 3 control tokens: <|channel|>, <|constrain|>, <|message|> (the <|call|>
    # stop token never reaches `content`).
    assert "output_chars=0" in out and "tool_calls=1" in out and "harmony_tokens_stripped=3" in out


def test_tacc_stream_keeps_prose_around_a_leaked_span_and_flags_a_call_without_structural_twin(capsys):
    def _stream():
        yield _chunk(content="Here is what I found.\n\n")
        yield _chunk(content=LEAK)
        yield _chunk(content="<|start|>assistant<|channel|>final<|message|>")
        yield _chunk(content="The catalog has 12 sources.")
        yield _chunk(content="<|return|>", finish_reason="stop")

    client = _tacc_client(_stream)
    events = list(client.responses.create(model="gpt-oss-120b", input="q", tools=_TOOLS, stream=True))
    content = "".join(e.delta for e in events if e.type == "response.output_text.delta")
    assert content == "Here is what I found.\n\nThe catalog has 12 sources."
    completed = next(e.response for e in events if e.type == "response.completed")
    assert completed.output == []  # nothing is synthesised from prose
    assert client.responses._history_cache[completed.id][-1]["content"] == content
    out = capsys.readouterr().out
    assert f"functions.{TOOL}" in out and "NOT issued structurally this round" in out


def test_tacc_stream_without_markup_is_byte_identical():
    def _stream():
        yield _chunk(content="Angles: a < b | c <| d and 3 <5.")
        yield _chunk(content=" End <", finish_reason="stop")

    client = _tacc_client(_stream)
    events = list(client.responses.create(model="gpt-oss-120b", input="q", stream=True))
    content = "".join(e.delta for e in events if e.type == "response.output_text.delta")
    assert content == "Angles: a < b | c <| d and 3 <5. End <"


# ── the filter itself ────────────────────────────────────────────────────────

def _run(pieces):
    f = HarmonyStreamFilter()
    visible, reasoning = "", ""
    for p in pieces:
        v, r = f.feed(p)
        visible += v
        reasoning += r
    v, r = f.flush()
    return visible + v, reasoning + r, f


@pytest.mark.parametrize("split", range(len(LEAK) + 1))
def test_leaked_call_is_dropped_at_every_split_point(split):
    visible, reasoning, f = _run([LEAK[:split], LEAK[split:]])
    assert visible == "" and reasoning == ""
    assert f.dropped_calls == [{"name": TOOL, "payload": ARGS}]


def test_leaked_call_is_dropped_when_fed_one_character_at_a_time():
    visible, _, f = _run(list("Before. " + LEAK + " After."))
    assert visible == "Before.  After."
    assert f.dropped_calls == [{"name": TOOL, "payload": ARGS}]


def test_analysis_channel_becomes_reasoning_and_final_channel_stays_visible():
    visible, reasoning, _ = _run([
        "<|channel|>analysis<|message|>Let me think about the radius.<|end|>",
        "<|start|>assistant<|channel|>final<|message|>Use 5 arcsec.<|return|>",
    ])
    assert visible == "Use 5 arcsec."
    assert reasoning == "Let me think about the radius."


def test_json_payload_with_braces_in_strings_ends_the_span_without_a_call_token():
    payload = '{"q":"a}b{","o":{"k":[1,{"z":"\\"}"}]}}'
    visible, _, f = _run([f"<|channel|>commentary to=functions.f<|constrain|>json<|message|>{payload} then prose"])
    assert visible == " then prose"
    assert f.dropped_calls == [{"name": "f", "payload": payload}]


def test_commentary_without_recipient_is_a_visible_preamble():
    visible, reasoning, f = _run(["<|channel|>commentary<|message|>I'll query ALMA next.<|end|>"])
    assert visible == "I'll query ALMA next." and reasoning == "" and f.dropped_calls == []


def test_unknown_control_tokens_are_dropped_and_prose_is_untouched():
    visible, _, _ = _run(["done<|endoftext|>", " a < b | c <| d", " tail <"])
    assert visible == "done a < b | c <| d tail <"


def test_partial_token_at_end_of_stream_is_released_or_dropped_correctly():
    assert _run(["see <|b"])[0] == "see <|b"       # never a harmony token → kept
    assert _run(["cut <|chan"])[0] == "cut "        # truncated harmony token → dropped
    assert _run(["x <"])[0] == "x <"


def test_fast_path_and_empty_delta():
    f = HarmonyStreamFilter()
    assert f.feed("") == ("", "")
    assert f.feed("plain prose") == ("plain prose", "")
    assert f.flush() == ("", "")
    assert f.saw_markup is False and f.dropped_tokens == 0


def test_strip_harmony_markup_one_shot():
    assert strip_harmony_markup("Hello " + LEAK) == "Hello "
    assert strip_harmony_markup(LEAK + "\n\nAnswer.") == "\n\nAnswer."
    plain = "no markup here"
    assert strip_harmony_markup(plain) is plain
    assert strip_harmony_markup("") == ""


# ── runner: final-text residue scrub ─────────────────────────────────────────

def test_runner_scrubs_harmony_residue_from_the_final_answer(capsys):
    leak = '<|channel|>commentary to=functions.query_archive <|constrain|>json<|message|>{"x": 1}'
    responses = _Responses([
        _events(1, tool="query_archive", text=leak),   # a non-TACC path let the markup through
        _events(2, text="Found the data."),
    ])
    agent, executed = _tool_agent(responses)
    result = agent.stream_response_api("hello there", conversation_id="harmony-residue")
    assert len(executed) == 1
    assert "Found the data." in result
    assert "<|" not in result and "functions.query_archive" not in result
    out = capsys.readouterr().out
    assert "harmony control markup from the final text" in out
