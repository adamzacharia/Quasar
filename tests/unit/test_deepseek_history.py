from types import SimpleNamespace

from core.llm_client import LLMClient


class _FakeDeepSeekCompletions:
    def __init__(self):
        self.calls = []
        self.count = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        self.count += 1
        if kwargs.get("stream"):
            return self._stream(f"stream answer {self.count}")
        return SimpleNamespace(
            id=f"resp_{self.count}",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=f"answer {self.count}",
                        tool_calls=None,
                        reasoning_content=None,
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
                    delta=SimpleNamespace(
                        content=content,
                        tool_calls=None,
                        reasoning_content=None,
                    )
                )
            ]
        )


def _client_with_fake_deepseek():
    client = LLMClient(model="deepseek-v4-pro")
    completions = _FakeDeepSeekCompletions()
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    client._get_deepseek_client = lambda: fake_client
    return client, completions


def test_deepseek_previous_response_appends_new_user_turn():
    client, completions = _client_with_fake_deepseek()

    first = client.responses.create(
        model="deepseek-v4-pro",
        instructions="system prompt",
        input="first question",
    )
    client.responses.create(
        model="deepseek-v4-pro",
        instructions="system prompt",
        previous_response_id=first.id,
        input="second question",
    )

    messages = completions.calls[-1]["messages"]

    assert [m["content"] for m in messages if m["role"] == "user"] == [
        "first question",
        "second question",
    ]
    assert messages[-1] == {"role": "user", "content": "second question"}


def test_streaming_deepseek_previous_response_appends_new_user_turn():
    client, completions = _client_with_fake_deepseek()

    first_events = list(client.responses.create(
        model="deepseek-v4-pro",
        instructions="system prompt",
        input="first question",
        stream=True,
    ))
    first_response_id = first_events[0].response.id

    list(client.responses.create(
        model="deepseek-v4-pro",
        instructions="system prompt",
        previous_response_id=first_response_id,
        input="second question",
        stream=True,
    ))

    messages = completions.calls[-1]["messages"]

    assert [m["content"] for m in messages if m["role"] == "user"] == [
        "first question",
        "second question",
    ]
    assert messages[-1] == {"role": "user", "content": "second question"}
