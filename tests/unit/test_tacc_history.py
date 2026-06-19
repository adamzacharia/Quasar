from types import SimpleNamespace

from core.llm_client import LLMClient


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
                    message=SimpleNamespace(
                        content=f"answer {self.count}",
                        tool_calls=None,
                    )
                )
            ],
            usage=None,
        )

    @staticmethod
    def _stream(content):
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=content, tool_calls=None)
                )
            ]
        )


def _client_with_fake_tacc():
    client = LLMClient(model="gpt-oss-120b")
    completions = _FakeTaccCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client._get_tacc_client = lambda: fake_client
    return client, completions


def test_tacc_previous_response_appends_second_user_turn():
    client, completions = _client_with_fake_tacc()

    first = client.responses.create(
        model="gpt-oss-120b",
        instructions="system prompt",
        input="first literature question",
    )
    client.responses.create(
        model="gpt-oss-120b",
        instructions="system prompt",
        previous_response_id=first.id,
        input="second literature question",
    )

    messages = completions.calls[-1]["messages"]
    assert [message["content"] for message in messages if message["role"] == "user"] == [
        "first literature question",
        "second literature question",
    ]
    assert messages[0] == {
        "role": "system",
        "content": "Reasoning: high\n\nsystem prompt",
    }


def test_streaming_tacc_previous_response_appends_second_user_turn():
    client, completions = _client_with_fake_tacc()

    first_events = list(
        client.responses.create(
            model="gpt-oss-120b",
            instructions="system prompt",
            input="first literature question",
            stream=True,
        )
    )
    first_response_id = first_events[0].response.id
    list(
        client.responses.create(
            model="gpt-oss-120b",
            instructions="system prompt",
            previous_response_id=first_response_id,
            input="second literature question",
            stream=True,
        )
    )

    messages = completions.calls[-1]["messages"]
    assert messages[-1] == {"role": "user", "content": "second literature question"}


def test_clearing_tacc_history_starts_a_fresh_chain():
    client, completions = _client_with_fake_tacc()
    first_events = list(
        client.responses.create(
            model="gpt-oss-120b",
            instructions="system prompt",
            input="first question",
            stream=True,
        )
    )
    response_id = first_events[0].response.id
    client.responses.clear_history(response_id)

    list(
        client.responses.create(
            model="gpt-oss-120b",
            instructions="system prompt",
            previous_response_id=response_id,
            input="retry question",
            stream=True,
        )
    )

    assert completions.calls[-1]["messages"] == [
        {"role": "system", "content": "Reasoning: high\n\nsystem prompt"},
        {"role": "user", "content": "retry question"},
    ]


def test_tacc_gpt_oss_reasoning_effort_can_be_overridden(monkeypatch):
    monkeypatch.setenv("QUASAR_GPT_OSS_REASONING", "low")
    client, completions = _client_with_fake_tacc()

    client.responses.create(
        model="gpt-oss-120b",
        instructions="system prompt",
        input="question",
    )

    assert completions.calls[-1]["messages"][0]["content"] == (
        "Reasoning: low\n\nsystem prompt"
    )


def test_tacc_non_gpt_oss_model_does_not_receive_reasoning_instruction():
    client, completions = _client_with_fake_tacc()

    client.responses.create(
        model="Qwen3-32B",
        instructions="system prompt",
        input="question",
    )

    assert completions.calls[-1]["messages"][0]["content"] == "system prompt"


def test_invalid_gpt_oss_reasoning_effort_defaults_to_high(monkeypatch):
    monkeypatch.setenv("QUASAR_GPT_OSS_REASONING", "maximum")
    client, completions = _client_with_fake_tacc()

    client.responses.create(
        model="gpt-oss-120b",
        instructions="Reasoning: medium\n\nsystem prompt",
        input="question",
    )

    assert completions.calls[-1]["messages"][0]["content"] == (
        "Reasoning: high\n\nsystem prompt"
    )
