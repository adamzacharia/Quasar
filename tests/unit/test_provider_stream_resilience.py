"""Provider stream resilience — per-provider read timeouts and reasoning
delta forwarding.

The 90 s flat httpx read timeout killed long-thinking gpt-oss rounds via TACC
mid-stream (live 2026-07: Gaia white-dwarf turn, overnight run), and
_stream_tacc dropped reasoning deltas so the SSE inactivity watchdog starved
during genuine thinking. These tests pin the new defaults and the forwarded
reasoning events.
"""

from types import SimpleNamespace

from core.llm_client import LLMClient


def _clear_timeout_env(monkeypatch):
    for name in (
        "LLM_READ_TIMEOUT_SECONDS",
        "LLM_READ_TIMEOUT_SECONDS_TACC",
        "LLM_READ_TIMEOUT_SECONDS_DEEPSEEK",
        "LLM_READ_TIMEOUT_SECONDS_OPENAI",
    ):
        monkeypatch.delenv(name, raising=False)


def test_read_timeout_defaults_are_per_provider(monkeypatch):
    _clear_timeout_env(monkeypatch)
    assert LLMClient._http_timeout("tacc").read == 600.0
    assert LLMClient._http_timeout("deepseek").read == 600.0
    assert LLMClient._http_timeout("openai").read == 300.0
    assert LLMClient._http_timeout("local").read == 300.0
    assert LLMClient._http_timeout().read == 300.0


def test_read_timeout_global_env_override_applies_to_all_providers(monkeypatch):
    _clear_timeout_env(monkeypatch)
    monkeypatch.setenv("LLM_READ_TIMEOUT_SECONDS", "120")
    assert LLMClient._http_timeout("tacc").read == 120.0
    assert LLMClient._http_timeout("openai").read == 120.0


def test_read_timeout_per_provider_env_wins_over_global(monkeypatch):
    _clear_timeout_env(monkeypatch)
    monkeypatch.setenv("LLM_READ_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("LLM_READ_TIMEOUT_SECONDS_TACC", "720")
    assert LLMClient._http_timeout("tacc").read == 720.0
    assert LLMClient._http_timeout("deepseek").read == 120.0


class _ReasoningTaccCompletions:
    """Streams gpt-oss-style reasoning deltas before the content delta."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._stream()

    @staticmethod
    def _stream():
        for thought in ("thinking about white dwarfs", " ...still thinking"):
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=None,
                            reasoning_content=thought,
                        )
                    )
                ]
            )
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="final answer", tool_calls=None)
                )
            ]
        )


def _client_with_reasoning_tacc():
    client = LLMClient(model="gpt-oss-120b")
    completions = _ReasoningTaccCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client._get_tacc_client = lambda: fake_client
    return client


def test_tacc_stream_forwards_reasoning_deltas_as_activity():
    client = _client_with_reasoning_tacc()

    events = list(
        client.responses.create(
            model="gpt-oss-120b",
            instructions="system prompt",
            input="question",
            stream=True,
        )
    )

    reasoning_deltas = [
        e.delta for e in events if e.type == "response.reasoning_summary_text.delta"
    ]
    assert reasoning_deltas == ["thinking about white dwarfs", " ...still thinking"]
    # Reasoning closes before the content starts, mirroring the DeepSeek shape.
    types = [e.type for e in events]
    assert types.index("response.reasoning_summary_text.done") < types.index(
        "response.output_text.delta"
    )
    # Content output is unchanged — reasoning never leaks into the answer text.
    content = "".join(
        e.delta for e in events if e.type == "response.output_text.delta"
    )
    assert content == "final answer"


class _CoRidingTaccCompletions:
    """Streams a chunk carrying BOTH a reasoning delta and content — some
    providers flush the last thought and the first answer token together."""

    def create(self, **kwargs):
        return self._stream()

    @staticmethod
    def _stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=None,
                        reasoning_content="thinking",
                    )
                )
            ]
        )
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="co-riding answer",
                        tool_calls=None,
                        reasoning_content=" ...final thought",
                    )
                )
            ]
        )


def test_tacc_stream_co_riding_reasoning_and_content_keeps_content():
    client = LLMClient(model="gpt-oss-120b")
    client._get_tacc_client = lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=_CoRidingTaccCompletions())
    )

    events = list(
        client.responses.create(
            model="gpt-oss-120b",
            instructions="system prompt",
            input="question",
            stream=True,
        )
    )

    # The reasoning delta on the mixed chunk is forwarded...
    reasoning_deltas = [
        e.delta for e in events if e.type == "response.reasoning_summary_text.delta"
    ]
    assert reasoning_deltas == ["thinking", " ...final thought"]
    # ...and the content co-riding the same chunk is NOT discarded.
    content = "".join(
        e.delta for e in events if e.type == "response.output_text.delta"
    )
    assert content == "co-riding answer"
    types = [e.type for e in events]
    assert types.index("response.reasoning_summary_text.done") < types.index(
        "response.output_text.delta"
    )


def test_deepseek_stream_co_riding_reasoning_and_content_keeps_content():
    client = LLMClient(model="deepseek-v4-pro")
    client._get_deepseek_client = lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=_CoRidingTaccCompletions())
    )

    events = list(
        client.responses.create(
            model="deepseek-v4-pro",
            instructions="system prompt",
            input="question",
            stream=True,
        )
    )

    reasoning_deltas = [
        e.delta for e in events if e.type == "response.reasoning_summary_text.delta"
    ]
    assert reasoning_deltas == ["thinking", " ...final thought"]
    content = "".join(
        e.delta for e in events if e.type == "response.output_text.delta"
    )
    assert content == "co-riding answer"


def test_tacc_stream_without_reasoning_is_unchanged():
    client = LLMClient(model="gpt-oss-120b")

    def _plain_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="plain answer", tool_calls=None)
                )
            ]
        )

    completions = SimpleNamespace(create=lambda **kwargs: _plain_stream())
    client._get_tacc_client = lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    events = list(
        client.responses.create(
            model="gpt-oss-120b",
            instructions="system prompt",
            input="question",
            stream=True,
        )
    )

    assert not [
        e for e in events if e.type == "response.reasoning_summary_text.delta"
    ]
    content = "".join(
        e.delta for e in events if e.type == "response.output_text.delta"
    )
    assert content == "plain answer"
