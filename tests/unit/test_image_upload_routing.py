import json
from types import SimpleNamespace

from core.llm_client import LLMClient, model_accepts_direct_image_input


class _FakeOpenAIResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(id="resp_openai", output_text="ok")


class _FakeDeepSeekCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="resp_deepseek",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="ok",
                        tool_calls=None,
                        reasoning_content=None,
                    )
                )
            ],
            usage=None,
        )


def test_openai_image_attachment_uses_responses_input_image():
    client = LLMClient(model="gpt-5.4-mini")
    responses = _FakeOpenAIResponses()
    client._get_openai_client = lambda: SimpleNamespace(responses=responses)

    client.responses.create(
        model="gpt-5.4-mini",
        input="what is this?",
        attachments=[
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,abc123",
                    "detail": "auto",
                },
            }
        ],
    )

    content = responses.calls[-1]["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "what is this?"}
    assert content[1] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,abc123",
        "detail": "auto",
    }
    assert all(item["type"] != "image_url" for item in content)


def test_deepseek_does_not_receive_raw_image_url_parts():
    client = LLMClient(model="deepseek-v4-pro")
    completions = _FakeDeepSeekCompletions()
    client._get_deepseek_client = lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    client.responses.create(
        model="deepseek-v4-pro",
        input="what is this?",
        attachments=[
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,abc123",
                    "detail": "auto",
                },
            }
        ],
    )

    messages = completions.calls[-1]["messages"]
    assert all(isinstance(message["content"], str) for message in messages)
    assert "image_url" not in json.dumps(messages)


def test_image_direct_input_capability_matches_current_provider_paths():
    assert model_accepts_direct_image_input("gpt-5.4-mini")
    assert model_accepts_direct_image_input("gpt-4o-mini")
    assert not model_accepts_direct_image_input("deepseek-v4-pro")
    assert not model_accepts_direct_image_input("deepseek-v4-flash")
