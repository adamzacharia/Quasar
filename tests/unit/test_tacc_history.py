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
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "retry question"},
    ]
