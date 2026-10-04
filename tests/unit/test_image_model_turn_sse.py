"""The image-model chat turn in ui-pro/api/sse.py, driven end to end with fakes.

A selected image model (Nano Banana / gpt-image) skips the agent: the SSE
pipeline calls services.image_generation.generate_image, streams image cards,
persists them with the turn and finalizes the run row on every path.
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

_UI_PRO = str(Path(__file__).resolve().parents[2] / "ui-pro")
if _UI_PRO not in sys.path:
    sys.path.insert(0, _UI_PRO)

import api.sse as sse  # noqa: E402
from api.models import ChatRequest  # noqa: E402
from services import image_generation as ig  # noqa: E402


class _Conversations:
    def __init__(self, history=None):
        self.history = list(history or [])
        self.saved = []

    def generate_title_from_message(self, message):
        return "title"

    def create_conversation(self, user_id, title, model):
        return "conv-1"

    def conversation_belongs_to_user(self, conv_id, user_id):
        return True

    def update_conversation_model(self, conv_id, model):
        pass

    def save_message(self, conv_id, role, content, message_type="general", metadata=None):
        self.saved.append({"role": role, "content": content, "metadata": metadata})

    def get_conversation_messages(self, conv_id, last_n=None):
        return list(self.history)


class _Runs:
    def __init__(self):
        self.finalized = []

    def start_run(self, **kwargs):
        pass

    def finalize_run(self, run_id, **kwargs):
        self.finalized.append(kwargs)

    def mark_run_cancelled_if_started(self, run_id):
        return False


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(ig, "RENDERED_DIR", str(tmp_path))
    conversations, runs, calls = _Conversations(), _Runs(), []
    monkeypatch.setattr(sse, "get_agent", lambda: None)
    monkeypatch.setattr(sse, "_resolve_optional_user", lambda *a, **k: {"sub": "u1"})
    monkeypatch.setattr(sse, "_current_user_email", lambda user: "u1@example.org")
    monkeypatch.setattr(sse, "_build_llm_context_for_user", lambda uid: {
        "provider_api_keys": {"openai": "k"}, "model_providers": {"gpt-image-2": "openai"},
        "key_source_by_provider": {"openai": "byok"}, "byok_token_limits": {}})
    monkeypatch.setattr(sse, "usage_quota_service", NS(ensure_allowed=lambda **k: None))
    monkeypatch.setattr(sse, "_make_usage_recorder", lambda uid: (lambda **kw: None))
    monkeypatch.setattr(sse, "_make_quota_checker", lambda *a: (lambda **kw: None))
    monkeypatch.setattr(sse, "_make_quota_releaser", lambda: (lambda rid: None))
    monkeypatch.setattr(sse, "conversation_service", conversations)
    monkeypatch.setattr(sse, "provider_file_service", NS(get_active_files=lambda **kw: {}))
    monkeypatch.setattr(sse, "issue_report_service", runs)
    import core.langfuse_integration as lf
    monkeypatch.setattr(lf, "get_langfuse", lambda: None)

    def fake_generate(model, prompt, **kwargs):
        calls.append({"model": model, "prompt": prompt, **kwargs})
        image = ig.save_image(b"png-bytes", "image/png")
        return ig.ImageGenerationResult(provider="openai", model=model, images=[image], text="")

    monkeypatch.setattr(sse, "generate_image", fake_generate)
    return NS(conversations=conversations, runs=runs, calls=calls, monkeypatch=monkeypatch)


def _events(chunks):
    out = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if line.startswith("data: "):
                payload = line[6:]
                out.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return out


def _run(message="an apple", conversation_id=None, attachment_context=None):
    async def main():
        resp = await sse._stream_chat_response(
            ChatRequest(message=message, model="gpt-image-2", conversation_id=conversation_id),
            authorization="Bearer t", attachment_context=attachment_context)
        return [c if isinstance(c, str) else c.decode() for c in [x async for x in resp.body_iterator]]
    return _events(asyncio.run(main()))


def test_success_streams_the_card_persists_it_and_completes_the_run(env):
    events = _run()
    kinds = [e if e == "[DONE]" else e["type"] for e in events]
    assert kinds[0] == "run_meta" and kinds[-1] == "[DONE]"
    assert kinds.index("conversation_meta") < kinds.index("image")
    image = next(e for e in events if e != "[DONE]" and e["type"] == "image")
    assert image["meta"] == {"kind": "generated", "model": "gpt-image-2"}
    assert image["url"].startswith("/api/images/gen_") and image["blockKind"] == "image"
    saved = env.conversations.saved[-1]
    assert saved["role"] == "assistant"
    assert saved["metadata"]["images"][0]["blockId"] == image["blockId"]
    assert saved["metadata"]["runMeta"]["model"] == "gpt-image-2"
    assert env.runs.finalized[-1]["status"] == "completed"
    assert env.calls[0]["previous_image"] is None and env.calls[0]["references"] == []


def _generated_turn(url):
    return {"role": "assistant", "content": "",
            "metadata": {"images": [{"url": url, "meta": {"kind": "generated", "model": "gpt-image-2"}}]}}


def test_follow_up_edits_the_last_generated_image_until_a_text_turn_intervenes(env):
    first = ig.save_image(b"old-image", "image/png")
    env.conversations.history = [{"role": "user", "content": "an apple"}, _generated_turn(first.url),
                                 {"role": "user", "content": "make it green"}]
    _run("make it green", conversation_id="conv-1")
    assert env.calls[-1]["previous_image"] == (b"old-image", "image/png")

    env.conversations.history.append({"role": "assistant", "content": "A text answer from a chat model."})
    _run("draw a pear", conversation_id="conv-1")
    assert env.calls[-1]["previous_image"] is None


def test_uploaded_images_override_the_previous_image(env):
    first = ig.save_image(b"old-image", "image/png")
    env.conversations.history = [_generated_turn(first.url)]
    upload = base64.b64encode(b"uploaded").decode()
    _run("put a hat on it", conversation_id="conv-1", attachment_context={"attachments": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{upload}"}}]})
    assert env.calls[-1]["references"] == [(b"uploaded", "image/jpeg")]
    assert env.calls[-1]["previous_image"] is None


def test_text_only_answer_is_a_failed_run_with_the_explanation(env):
    env.monkeypatch.setattr(sse, "generate_image", lambda *a, **k: ig.ImageGenerationResult(
        provider="google", model="gpt-image-2", text="Which apple variety would you like?"))
    events = _run()
    assert not any(e != "[DONE]" and e["type"] == "image" for e in events)
    assert any(e != "[DONE]" and e["type"] == "token" and "variety" in e["content"] for e in events)
    terminal = [e for e in events if e != "[DONE]" and e["type"] == "run_meta"][-1]
    assert terminal["status"] == "failed" and terminal["errorCode"] == "image_not_generated"
    assert env.runs.finalized[-1]["status"] == "failed"
    assert env.conversations.saved[-1]["metadata"]["runMeta"]["status"] == "failed"


def test_provider_error_message_reaches_the_chat(env):
    def refuse(*a, **k):
        raise ig.ImageGenerationError("OpenAI declined to generate this image under its content policy.")

    env.monkeypatch.setattr(sse, "generate_image", refuse)
    events = _run()
    assert any(e != "[DONE]" and e["type"] == "token" and "content policy" in e["content"] for e in events)
    assert env.runs.finalized[-1]["error_code"] == "image_generation_failed"


def test_quota_denial_keeps_its_message(env):
    def deny(*a, **k):
        raise sse.QuotaExceededError("Weekly token limit reached for gpt-image-2.")

    env.monkeypatch.setattr(sse, "generate_image", deny)
    events = _run()
    assert any(e != "[DONE]" and e["type"] == "token" and "Weekly token limit" in e["content"] for e in events)
    assert env.runs.finalized[-1]["error_code"] == "quota_exceeded"


def test_client_disconnect_mid_generation_still_finalizes_the_run(env):
    release = threading.Event()

    def slow(*a, **k):
        release.wait(5)
        return ig.ImageGenerationResult(provider="openai", model="gpt-image-2")

    env.monkeypatch.setattr(sse, "generate_image", slow)

    async def main():
        resp = await sse._stream_chat_response(ChatRequest(message="x", model="gpt-image-2"),
                                               authorization="Bearer t")
        gen = resp.body_iterator
        seen = []
        async for chunk in gen:
            seen.append(chunk)
            if "Generating image" in chunk:
                break
        await gen.aclose()  # the client went away
        release.set()
        await asyncio.sleep(0.3)
        return seen

    asyncio.run(main())
    deadline = time.time() + 3
    while not env.runs.finalized and time.time() < deadline:
        time.sleep(0.05)
    assert env.runs.finalized and env.runs.finalized[-1]["status"] == "cancelled"


# ── Real service underneath: usage, in-flight cancellation, close points ────


@pytest.fixture
def real_service(env, monkeypatch):
    """generate_image itself runs; only the OpenAI client is fake."""
    from core.llm_client import LLMClient

    usage = []
    state = {"started": threading.Event(), "release": threading.Event(), "calls": 0, "block": False}

    def generate(**kwargs):
        state["calls"] += 1
        state["started"].set()
        if state["block"]:
            state["release"].wait(5)
        return NS(data=[NS(b64_json=base64.b64encode(b"png").decode(), revised_prompt=None)],
                  output_format="png", usage=NS(input_tokens=12, output_tokens=272))

    monkeypatch.setattr(LLMClient, "_get_openai_client", lambda self: NS(images=NS(generate=generate)))
    monkeypatch.setattr(sse, "generate_image", ig.generate_image)
    monkeypatch.setattr(sse, "_make_usage_recorder", lambda uid: (lambda **kw: usage.append(kw)))
    return NS(usage=usage, state=state)


def _wait(predicate, seconds=3):
    deadline = time.time() + seconds
    while not predicate() and time.time() < deadline:
        time.sleep(0.02)
    return predicate()


def test_real_generation_records_usage_and_reports_cost(env, real_service):
    events = _run()
    usage_event = next(e for e in events if e != "[DONE]" and e["type"] == "usage")
    assert usage_event["inputTokens"] == 12 and usage_event["outputTokens"] == 272
    assert usage_event["costUsd"] > 0  # priced at the gpt-image-2 rates
    assert real_service.usage[0]["model"] == "gpt-image-2" and real_service.usage[0]["key_source"] == "byok"
    assert env.runs.finalized[-1]["status"] == "completed"


def _open(message="x"):
    async def opener():
        resp = await sse._stream_chat_response(ChatRequest(message=message, model="gpt-image-2"),
                                               authorization="Bearer t")
        return resp.body_iterator
    return opener


def test_cancel_while_the_provider_runs_finalizes_and_still_settles_late_usage(env, real_service):
    real_service.state["block"] = True

    async def main():
        gen = await _open()()
        async for chunk in gen:
            if "Generating image" in chunk:
                break
        pending = asyncio.ensure_future(gen.__anext__())  # resumes into the provider wait
        await asyncio.get_running_loop().run_in_executor(None, real_service.state["started"].wait, 5)
        pending.cancel()
        try:
            await pending
        except (asyncio.CancelledError, StopAsyncIteration):
            pass
        real_service.state["release"].set()

    asyncio.run(main())
    assert _wait(lambda: env.runs.finalized)
    assert env.runs.finalized[-1]["status"] == "cancelled"
    assert _wait(lambda: real_service.usage), "the provider billed: its usage must still reach the ledger"
    assert real_service.usage[0]["output_tokens"] == 272


def test_close_at_the_first_run_meta_finalizes_the_run(env, real_service):
    async def main():
        gen = await _open()()
        first = await gen.__anext__()
        assert '"run_meta"' in first
        await gen.aclose()

    asyncio.run(main())
    assert _wait(lambda: env.runs.finalized)
    assert env.runs.finalized[-1]["status"] == "cancelled"
    assert real_service.state["calls"] == 0


def test_close_after_done_keeps_the_run_completed(env, real_service):
    async def main():
        gen = await _open()()
        async for chunk in gen:
            if "[DONE]" in chunk:
                break
        await gen.aclose()

    asyncio.run(main())
    assert _wait(lambda: env.runs.finalized)
    assert [f["status"] for f in env.runs.finalized] == ["completed"]
    assert len([m for m in env.conversations.saved if m["role"] == "assistant"]) == 1


def test_a_queued_generation_is_cancelled_with_the_turn(env, real_service, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=1)
    blocker = threading.Event()
    pool.submit(blocker.wait, 5)  # saturate the chat executor
    monkeypatch.setattr(sse, "_chat_executor", pool)

    async def main():
        gen = await _open()()
        async for chunk in gen:
            if "Generating image" in chunk:
                break
        pending = asyncio.ensure_future(gen.__anext__())
        await asyncio.sleep(0.2)  # the generation is now queued behind the blocker
        pending.cancel()
        try:
            await pending
        except (asyncio.CancelledError, StopAsyncIteration):
            pass

    asyncio.run(main())
    blocker.set()
    pool.shutdown(wait=True)
    assert real_service.state["calls"] == 0, "a cancelled turn must not start its queued provider call"
    assert _wait(lambda: env.runs.finalized) and env.runs.finalized[-1]["status"] == "cancelled"
