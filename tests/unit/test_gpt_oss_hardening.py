"""Tests for the gpt-oss-120b agent hardening.

Covers the observed failure modes: malformed tool-call JSON (arithmetic inside
arguments) surfacing as a raw provider dump, silent tool_choice="required"
downgrades on the TACC path, missing fix hints on Data Lab SQL errors, and
live-data queries (SMASH/DECam) leaking to the web-search path.
"""

from __future__ import annotations

import pytest

from core.llm_client import ResponsesShim
from tests.integration.test_agent_archive_tools import _load_agent_module


# ── ResponsesShim: gpt-oss instruction hardening ─────────────────────────────

def test_gpt_oss_instructions_include_strict_tool_json_rule():
    out = ResponsesShim._configure_gpt_oss_instructions("You are QUASAR.", "gpt-oss-120b")
    assert out.startswith("Reasoning: ")
    assert "strictly valid JSON" in out
    assert "10.0/60.0" in out


def test_gpt_oss_instructions_idempotent():
    once = ResponsesShim._configure_gpt_oss_instructions("Base prompt.", "gpt-oss-120b")
    twice = ResponsesShim._configure_gpt_oss_instructions(once, "gpt-oss-120b")
    assert twice.count("strictly valid JSON") == 1
    assert twice.count("Reasoning:") == 1


def test_non_gpt_oss_instructions_untouched():
    assert ResponsesShim._configure_gpt_oss_instructions("Base.", "Qwen3-32B") == "Base."


# ── ResponsesShim: tool_choice="required" emulation ──────────────────────────

def test_required_tool_choice_nudge_appends_to_system_message():
    messages = [
        {"role": "system", "content": "You are QUASAR."},
        {"role": "user", "content": "Show me an image of M31."},
    ]
    nudged = ResponsesShim._apply_required_tool_choice_nudge(messages)
    assert nudged[0]["content"].startswith("You are QUASAR.")
    assert "Tool use is REQUIRED" in nudged[0]["content"]
    # Call-only copy: the original history must stay pristine.
    assert "REQUIRED" not in messages[0]["content"]
    assert nudged[1] == messages[1]


def test_required_tool_choice_nudge_without_system_message():
    messages = [{"role": "user", "content": "hi"}]
    nudged = ResponsesShim._apply_required_tool_choice_nudge(messages)
    assert nudged[0]["role"] == "system"
    assert "Tool use is REQUIRED" in nudged[0]["content"]
    assert len(messages) == 1


# ── TACC provider path: required → auto downgrade + nudge actually sent ──────

from types import SimpleNamespace  # noqa: E402

from core.llm_client import LLMClient  # noqa: E402


class _FakeTaccCompletions:
    def __init__(self):
        self.calls = []
        self.count = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        self.count += 1
        if kwargs.get("stream"):
            return self._stream(f"answer {self.count}")
        return SimpleNamespace(
            id=f"resp_{self.count}",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=f"answer {self.count}", tool_calls=None)
                )
            ],
            usage=None,
        )

    @staticmethod
    def _stream(content):
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=None))]
        )


def _client_with_fake_tacc():
    client = LLMClient(model="gpt-oss-120b")
    completions = _FakeTaccCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client._get_tacc_client = lambda: fake_client
    return client, completions


_TOOLS = [{
    "type": "function",
    "name": "hips_cutout",
    "description": "Fetch a cutout.",
    "parameters": {"type": "object", "properties": {}},
}]


def test_tacc_required_tool_choice_downgraded_with_nudge_nonstreaming():
    client, completions = _client_with_fake_tacc()
    client.responses.create(
        model="gpt-oss-120b",
        instructions="system prompt",
        input="show me an image of M31",
        tools=_TOOLS,
        tool_choice="required",
    )
    sent = completions.calls[-1]
    assert sent["tool_choice"] == "auto"
    assert sent["messages"][0]["role"] == "system"
    assert "Tool use is REQUIRED" in sent["messages"][0]["content"]


def test_tacc_required_tool_choice_downgraded_with_nudge_streaming():
    client, completions = _client_with_fake_tacc()
    list(client.responses.create(
        model="gpt-oss-120b",
        instructions="system prompt",
        input="show me an image of M31",
        tools=_TOOLS,
        tool_choice="required",
        stream=True,
    ))
    sent = completions.calls[-1]
    assert sent["tool_choice"] == "auto"
    assert "Tool use is REQUIRED" in sent["messages"][0]["content"]


def test_tacc_required_nudge_does_not_leak_into_history():
    client, completions = _client_with_fake_tacc()
    first = client.responses.create(
        model="gpt-oss-120b",
        instructions="system prompt",
        input="show me an image of M31",
        tools=_TOOLS,
        tool_choice="required",
    )
    # Follow-up round without forcing: cached history must be nudge-free.
    client.responses.create(
        model="gpt-oss-120b",
        instructions="system prompt",
        previous_response_id=first.id,
        input="thanks",
        tools=_TOOLS,
        tool_choice="auto",
    )
    followup_messages = completions.calls[-1]["messages"]
    assert all("Tool use is REQUIRED" not in str(m.get("content")) for m in followup_messages)


def test_tacc_auto_tool_choice_gets_no_nudge():
    client, completions = _client_with_fake_tacc()
    client.responses.create(
        model="gpt-oss-120b",
        instructions="system prompt",
        input="hello",
        tools=_TOOLS,
        tool_choice="auto",
    )
    sent = completions.calls[-1]
    assert sent["tool_choice"] == "auto"
    assert "Tool use is REQUIRED" not in sent["messages"][0]["content"]


# ── Agent: provider errors must never reach the chat verbatim ────────────────

@pytest.fixture(scope="module")
def agent_module():
    return _load_agent_module()


def test_tool_call_parse_failure_gets_friendly_message(agent_module):
    err = Exception(
        "Error code: 400 - {'error': \"Failed to parse tool call from GPT OSS output: "
        "Expecting ',' delimiter: line 6 column 21 (char 107)\"}"
    )
    msg = agent_module.QuasarAgent._user_facing_provider_error(err)
    assert "Failed to parse" not in msg
    assert "400" not in msg
    assert "resend" in msg or "retry" in msg.lower()


def test_generic_provider_error_sanitized(agent_module):
    class _FakeApiError(Exception):
        status_code = 502

    msg = agent_module.QuasarAgent._user_facing_provider_error(_FakeApiError("upstream blew up: secret-payload"))
    assert "secret-payload" not in msg
    assert "502" in msg


def test_parse_failure_matches_recovery_substrings():
    provider_error = (
        "litellm.BadRequestError: OpenAIException - Error code: 400 - "
        "{'error': \"Failed to parse tool call from GPT OSS output: Expecting ',' delimiter\", "
        "'error_type': 'Invalid function calling output.'}"
    )
    recovery_substrings = [
        "No tool output found for function call",
        "must be followed by tool messages",
        "insufficient tool messages",
        "tool_calls",
        "Failed to parse tool call",
        "Invalid function calling output",
    ]
    assert any(s in provider_error for s in recovery_substrings)


# ── Agent: Data Lab error fix hints ──────────────────────────────────────────

def test_datalab_error_column_hint(agent_module):
    agent = agent_module.QuasarAgent.__new__(agent_module.QuasarAgent)
    payload = agent._datalab_error(Exception(
        'Data Lab /query returned HTTP 400: Error: column "gmagmag" does not exist '
        'HINT: Perhaps you meant to reference the column "object.gmag".'
    ))
    assert payload["success"] is False
    assert "datalab_describe_table" in payload.get("fix_hint", "")


def test_datalab_error_expression_hint(agent_module):
    agent = agent_module.QuasarAgent.__new__(agent_module.QuasarAgent)
    payload = agent._datalab_error(Exception("Column 'M_G' not found for expression 'M_G'"))
    assert "log10" in payload.get("fix_hint", "")


# ── Agent: live-data routing ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "query",
    [
        "Show me a color image of the center of M31 from the DECam Legacy Surveys.",
        "Help me look for a stellar overdensity - a possible dwarf companion - in SMASH DR1 "
        "field 169. Select blue main-sequence stars and find where they clump on the sky.",
        "Get g and r magnitudes for point sources within 0.4 deg of the Draco dwarf from NSC DR2 "
        "and plot a g vs (g-r) CMD.",
        "How many Gaia DR3 sources lie within 10 arcminutes of Palomar 5?",
        "Find high-proper-motion white-dwarf candidates in Gaia DR3 and give me the HR diagram.",
    ],
)
def test_data_queries_detected_as_live_data(agent_module, query):
    agent = agent_module.QuasarAgent.__new__(agent_module.QuasarAgent)
    assert agent._is_live_data_query(query) is True


def test_plain_knowledge_query_not_live_data(agent_module):
    agent = agent_module.QuasarAgent.__new__(agent_module.QuasarAgent)
    assert agent._is_live_data_query("What is the ALMA proprietary period?") is False
