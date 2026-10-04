"""Gemini chat adapter, image-model catalog flags, and AI image generation.

Offline: provider clients are replaced by fakes that capture the request.
"""
from __future__ import annotations

import base64
import json
import os
from types import SimpleNamespace as NS

import pytest

from core.llm_client import LLMClient, llm_request_context
from core.retry import _extract_status_code, _is_retryable
from services import image_generation as ig
from services.provider_models import ModelInfo, chat_models

# Real Quasar schema features the old `parameters=` path rejected client-side.
REF_TOOL = {
    "type": "function",
    "name": "alma_project_census",
    "description": "census",
    "parameters": {
        "type": "object",
        "properties": {
            "cycle": {"type": ["integer", "null"]},
            "constraint": {"$ref": "#/$defs/Constraint"},
        },
        "$defs": {"Constraint": {"type": "object", "properties": {"op": {"type": "string"}}}},
    },
}


def _part(text=None, thought=False, function_call=None, signature=None, inline=None):
    return NS(text=text, thought=thought, function_call=function_call, thought_signature=signature,
              inline_data=inline)


def _chunk(parts, finish=None, usage=None):
    return NS(candidates=[NS(content=NS(parts=parts), finish_reason=finish)], usage_metadata=usage, text=None)


class _FakeGoogle:
    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.calls = []
        self.models = NS(generate_content_stream=self._stream, generate_content=self._call)

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.rounds.pop(0))

    def _call(self, **kwargs):
        self.calls.append(kwargs)
        return self.rounds.pop(0)


def _client(fake):
    client = LLMClient(model="gemini-3.8-flash")
    client._get_google_client = lambda: fake
    return client


# ── Gemini chat adapter ─────────────────────────────────────────────────


def test_tool_schemas_with_refs_and_type_arrays_translate():
    decls = LLMClient(model="gemini-3.8-flash").responses._translate_tools_for_google([REF_TOOL])
    fd = decls[0].function_declarations[0]
    assert fd.name == "alma_project_census"
    assert fd.parameters is None
    assert fd.parameters_json_schema["$defs"]["Constraint"]["type"] == "object"


def test_streaming_tool_loop_replays_history_and_thought_signature():
    sig = b"\x01sig-bytes"
    fc = NS(name="simbad_query", args={"identifier": "NGC 1068"}, id=None)
    usage = NS(prompt_token_count=100, candidates_token_count=10, thoughts_token_count=40,
               tool_use_prompt_token_count=0)
    fake = _FakeGoogle([
        [_chunk([_part(text="planning", thought=True)]),
         _chunk([_part(function_call=fc, signature=sig)], finish=NS(value="STOP"), usage=usage)],
        [_chunk([_part(text="z = 0.0035")], finish=NS(value="STOP"))],
    ])
    client = _client(fake)
    events = list(client.responses._stream_google({
        "model": "gemini-3.8-flash", "instructions": "SYS", "input": "redshift of NGC 1068?",
        "tools": [REF_TOOL], "max_output_tokens": 1000,
    }))
    types = [e.type for e in events]
    assert "response.reasoning_summary_text.delta" in types
    done = events[-1].response
    assert done.finish_reason == "tool_calls"
    call = done.output[0]
    assert call.name == "simbad_query" and json.loads(call.arguments) == {"identifier": "NGC 1068"}
    # Thinking tokens bill as output.
    assert done.usage.output_tokens == 50 and done.usage.input_tokens == 100
    first = fake.calls[0]
    assert first["config"]["system_instruction"] == "SYS"
    assert "temperature" not in first["config"]  # Gemini 3 keeps its default 1.0
    assert first["config"]["thinking_config"] == {"include_thoughts": True}

    list(client.responses._stream_google({
        "model": "gemini-3.8-flash", "instructions": "SYS", "previous_response_id": done.id,
        "input": [{"type": "function_call_output", "call_id": call.call_id, "output": "{\"z\": 0.0035}"}],
        "tools": [REF_TOOL],
    }))
    contents = fake.calls[1]["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[0]["parts"][0]["text"] == "redshift of NGC 1068?"
    model_part = contents[1]["parts"][0]
    assert model_part["function_call"]["name"] == "simbad_query"
    assert model_part["thought_signature"] == sig
    response_part = contents[2]["parts"][0]["function_response"]
    assert response_part["name"] == "simbad_query"
    assert response_part["response"] == {"output": "{\"z\": 0.0035}"}


def test_follow_up_turn_keeps_the_conversation():
    fake = _FakeGoogle([
        NS(candidates=[NS(content=NS(parts=[_part(text="Hello!")]), finish_reason=NS(value="STOP"))],
           usage_metadata=None, text=None),
        NS(candidates=[NS(content=NS(parts=[_part(text="A Seyfert 2.")]), finish_reason=NS(value="STOP"))],
           usage_metadata=None, text=None),
    ])
    client = _client(fake)
    first = client.responses._call_google({"model": "gemini-3.8-flash", "input": "hi"})
    client.responses._call_google({"model": "gemini-3.8-flash", "input": "what is NGC 1068?",
                                    "previous_response_id": first.id})
    roles = [c["role"] for c in fake.calls[1]["contents"]]
    assert roles == ["user", "model", "user"]


def test_finish_reasons_and_required_tool_choice():
    fake = _FakeGoogle([NS(candidates=[NS(content=NS(parts=[_part(text="cut")]),
                                          finish_reason=NS(value="MAX_TOKENS"))],
                           usage_metadata=None, text=None)])
    result = _client(fake).responses._call_google({
        "model": "gemini-2.5-flash", "input": "x", "tools": [REF_TOOL], "tool_choice": "required",
        "temperature": 0.2})
    assert result.finish_reason == "length"
    config = fake.calls[0]["config"]
    assert config["tool_config"] == {"function_calling_config": {"mode": "ANY"}}
    assert config["temperature"] == 0.2  # 2.x still takes the agent's temperature


def test_gemma_folds_system_prompt_and_images_ride_on_the_user_turn():
    shim = LLMClient(model="gemma-4-31b-it").responses
    png = base64.b64encode(b"\x89PNG fake").decode()
    system, contents = shim._google_contents(
        [{"role": "system", "content": "SYS"}, {"role": "user", "content": "describe"}],
        attachments=[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}}],
        model="gemma-4-31b-it")
    assert system == ""
    parts = contents[0]["parts"]
    assert parts[0] == {"text": "SYS"}
    assert parts[1]["inline_data"]["mime_type"] == "image/png"
    assert parts[2] == {"text": "describe"}


def _signed_call_round(name="describe_upload", call_id=None):
    return [_chunk([_part(function_call=NS(name=name, args={}, id=call_id), signature=b"s")],
                   finish=NS(value="STOP"))]


def test_uploads_are_replayed_on_later_tool_rounds_and_dropped_on_a_new_turn():
    png = base64.b64encode(b"\x89PNG fake").decode()
    image = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}}
    doc = {"kind": "gemini_file", "file_uri": "https://files/abc", "mime_type": "application/pdf"}
    fake = _FakeGoogle([_signed_call_round(), [_chunk([_part(text="done")], finish=NS(value="STOP"))],
                        [_chunk([_part(text="hi")], finish=NS(value="STOP"))]])
    client = _client(fake)
    first = list(client.responses._stream_google(
        {"model": "gemini-3.8-flash", "input": "what is in these?", "tools": [REF_TOOL]},
        attachments=[image, doc]))[-1].response
    call = first.output[0]
    second = list(client.responses._stream_google({  # round 1: the runner passes no attachments
        "model": "gemini-3.8-flash", "previous_response_id": first.id, "tools": [REF_TOOL],
        "input": [{"type": "function_call_output", "call_id": call.call_id, "output": "ok"}]}))[-1].response
    round_two_user = fake.calls[1]["contents"][0]["parts"]
    assert round_two_user[0]["inline_data"]["mime_type"] == "image/png"
    assert round_two_user[1]["file_data"]["file_uri"] == "https://files/abc"
    assert round_two_user[2] == {"text": "what is in these?"}

    list(client.responses._stream_google({"model": "gemini-3.8-flash", "previous_response_id": second.id,
                                          "input": "thanks"}))
    assert not any("inline_data" in p or "file_data" in p
                   for c in fake.calls[2]["contents"] for p in c["parts"])


def test_parallel_calls_to_one_function_pair_with_their_responses():
    history = [
        {"role": "user", "content": "two lookups"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "g-1", "type": "function", "function": {"name": "simbad_query", "arguments": "{\"identifier\": \"M31\"}"},
             "_gemini_thought_signature": base64.b64encode(b"sig").decode()},
            {"id": "g-2", "type": "function", "function": {"name": "simbad_query", "arguments": "{\"identifier\": \"M33\"}"}},
        ]},
        {"role": "tool", "tool_call_id": "g-1", "content": "a"},
        {"role": "tool", "tool_call_id": "g-2", "content": "b"},
    ]
    _, contents = LLMClient(model="gemini-3.8-flash").responses._google_contents(history, model="gemini-3.8-flash")
    calls = contents[1]["parts"]
    assert [c["function_call"]["id"] for c in calls] == ["g-1", "g-2"]
    assert calls[0]["thought_signature"] == b"sig"
    responses = [p["function_response"] for p in contents[2]["parts"]]
    assert [(r["id"], r["name"], r["response"]["output"]) for r in responses] == [
        ("g-1", "simbad_query", "a"), ("g-2", "simbad_query", "b")]


def test_broken_chain_tool_outputs_become_a_user_note_for_gemini():
    fake = _FakeGoogle([[_chunk([_part(text="continuing")], finish=NS(value="STOP"))]])
    list(_client(fake).responses._stream_google({
        "model": "gemini-3.8-flash", "previous_response_id": "resp_gone", "instructions": "SYS",
        "input": [{"type": "function_call_output", "call_id": "call_x", "output": "{\"rows\": 3}"}]}))
    contents = fake.calls[0]["contents"]
    assert len(contents) == 1 and contents[0]["role"] == "user"
    assert "tool-call history was reset" in contents[0]["parts"][0]["text"]
    assert "{\"rows\": 3}" in contents[0]["parts"][0]["text"]


def test_google_errors_have_an_integer_status_and_a_clear_message():
    from core.agent import QuasarAgent

    class GoogleError(Exception):
        code = 404
        status = "NOT_FOUND"

    err = GoogleError("404 NOT_FOUND. This model models/gemini-2.5-flash is no longer available to new users.")
    assert _extract_status_code(err) == 404
    assert _is_retryable(err) is False  # used to raise TypeError comparing "NOT_FOUND" with ints
    msg = QuasarAgent._user_facing_provider_error(err)
    assert "not available to your API key" in msg and "HTTP 404" in msg


# ── Catalog ─────────────────────────────────────────────────────────────


def _m(provider, model_id):
    return ModelInfo(provider=provider, id=model_id, displayName=model_id)


def test_catalog_keeps_image_models_flagged_and_drops_non_chat_gemini_models():
    models = chat_models([_m("google", i) for i in (
        "gemini-3.8-flash", "gemini-3.1-flash-image", "nano-banana-pro-preview", "gemini-3.8-live",
        "gemini-omni-flash-preview", "lyria-3.5", "gemini-robotics-er-2-preview",
        "gemini-2.5-computer-use-preview-10-2025", "deep-research-preview-04-2026", "gemini-3.5-transcribe",
    )]) + chat_models([_m("openai", i) for i in ("gpt-5.4", "gpt-image-2", "dall-e-3")])
    by_id = {m.id: m for m in models}
    assert set(by_id) == {"gemini-3.8-flash", "gemini-3.1-flash-image", "nano-banana-pro-preview",
                          "gpt-5.4", "gpt-image-2"}
    for image_id in ("gemini-3.1-flash-image", "nano-banana-pro-preview", "gpt-image-2"):
        assert by_id[image_id].capabilities.imageGeneration is True
    assert by_id["gemini-3.8-flash"].capabilities.imageGeneration is None


def test_openai_discovery_keeps_gpt_image_but_not_its_dated_snapshot():
    from services.provider_models import OpenAIAdapter, _RawModel

    rows = [_RawModel(id=i) for i in ("gpt-image-2", "gpt-image-2-2026-04-21", "gpt-image-1-mini",
                                       "chatgpt-image-latest", "whisper-1")]
    ids = [m.id for m in OpenAIAdapter().normalize(rows)]
    assert sorted(ids) == ["chatgpt-image-latest", "gpt-image-1-mini", "gpt-image-2"]


def test_image_model_prices_resolve():
    from services.model_pricing import get_model_pricing

    assert get_model_pricing("google", "gemini-3.1-flash-image-preview")["output_per_mtok"] == 60.0
    assert get_model_pricing("google", "nano-banana-pro-preview")["output_per_mtok"] == 120.0
    assert get_model_pricing("openai", "gpt-image-2-2026-04-21")["output_per_mtok"] == 30.0
    # The text model rows must not swallow the image variants.
    assert get_model_pricing("google", "gemini-2.5-flash-image")["output_per_mtok"] == 30.0


# ── Image generation service ─────────────────────────────────────────────


@pytest.mark.parametrize("model,provider", [
    ("gemini-3.1-flash-image", "google"), ("gemini-3-pro-image-preview", "google"),
    ("gemini-2.5-flash-image-preview-05-20", "google"), ("nano-banana-pro-preview", "google"),
    ("gpt-image-1-mini", "openai"), ("chatgpt-image-latest", "openai"),
    ("gemini-3.8-flash", None), ("gemini-3.1-flash-lite", None), ("gpt-5.4", None),
])
def test_image_model_detection(model, provider):
    assert ig.image_model_provider(model) == provider


@pytest.fixture
def rendered_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ig, "RENDERED_DIR", str(tmp_path))
    return tmp_path


def _recorder(log):
    return lambda **kw: log.append(kw)


def test_google_image_generation_saves_the_image_and_records_usage(rendered_dir):
    resp = NS(candidates=[NS(finish_reason=NS(value="STOP"), content=NS(parts=[
        _part(text="", thought=True, inline=NS(data=b"thinking-draft", mime_type="image/png")),
        _part(text="Here is your apple."),
        _part(inline=NS(data=b"\xff\xd8jpeg", mime_type="image/jpeg")),
    ]))], usage_metadata=NS(prompt_token_count=11, candidates_token_count=1493, thoughts_token_count=0),
        prompt_feedback=None)
    fake = _FakeGoogle([resp])
    llm = LLMClient(model="gemini-3.1-flash-image")
    llm._get_google_client = lambda: fake
    usage, released = [], []
    with llm_request_context(provider_api_keys={"google": "k"}, key_source_by_provider={"google": "byok"},
                             usage_recorder=_recorder(usage), quota_checker=lambda **kw: "res-1",
                             quota_releaser=released.append):
        result = ig.generate_image("gemini-3.1-flash-image", "an apple", aspect_ratio="16:9", client=llm)
    assert len(result.images) == 1  # the interim thought image is skipped
    image = result.images[0]
    assert image.url.startswith("/api/images/gen_") and image.url.endswith(".jpg")
    assert (rendered_dir / os.path.basename(image.path)).read_bytes() == b"\xff\xd8jpeg"
    assert result.text == "Here is your apple."
    assert usage == [dict(provider="google", model="gemini-3.1-flash-image", key_source="byok",
                          input_tokens=11, output_tokens=1493, reservation_id="res-1")]
    assert released == []
    config = fake.calls[0]["config"]
    assert config["response_modalities"] == ["TEXT", "IMAGE"]
    assert config["image_config"] == {"aspect_ratio": "16:9"}


def test_blocked_image_is_a_clear_error_but_still_billed(rendered_dir):
    resp = NS(candidates=[NS(finish_reason=NS(value="IMAGE_SAFETY"), content=NS(parts=[]))],
              usage_metadata=NS(prompt_token_count=9, candidates_token_count=0, thoughts_token_count=0),
              prompt_feedback=None)
    llm = LLMClient(model="gemini-3.1-flash-image")
    llm._get_google_client = lambda: _FakeGoogle([resp])
    usage = []
    with llm_request_context(provider_api_keys={"google": "k"}, key_source_by_provider={"google": "byok"},
                             usage_recorder=_recorder(usage)):
        with pytest.raises(ig.ImageGenerationError, match="safety"):
            ig.generate_image("gemini-3.1-flash-image", "x", client=llm)
    assert usage and usage[0]["input_tokens"] == 9


def test_platform_key_is_refused_unless_enabled(monkeypatch):
    monkeypatch.delenv("QUASAR_ALLOW_PLATFORM_IMAGE_GENERATION", raising=False)
    with llm_request_context(key_source_by_provider={"openai": "platform"}):
        with pytest.raises(ig.ImageGenerationError, match="your own API key"):
            ig.generate_image("gpt-image-2", "an apple")


def test_openai_edits_the_previous_image(rendered_dir):
    captured = {}

    def edit(**kwargs):
        captured["edit"] = kwargs
        return NS(data=[NS(b64_json=base64.b64encode(b"png!").decode(), revised_prompt=None)],
                  output_format="png", usage=NS(input_tokens=300, output_tokens=272))

    llm = LLMClient(model="gpt-image-2")
    llm._get_openai_client = lambda: NS(images=NS(edit=edit, generate=None))
    with llm_request_context(provider_api_keys={"openai": "k"}, key_source_by_provider={"openai": "byok"}):
        result = ig.generate_image("gpt-image-2", "make it green", previous_image=(b"old", "image/png"),
                                   aspect_ratio="9:16", client=llm)
    assert result.images[0].url.endswith(".png")
    assert captured["edit"]["image"][0][1] == b"old"
    assert captured["edit"]["size"] == "1024x1536"
    assert "make it green" in captured["edit"]["prompt"]


def test_only_our_own_generated_files_can_be_loaded(rendered_dir):
    saved = ig.save_image(b"data", "image/png")
    assert ig.load_generated_image(saved.url) == (b"data", "image/png")
    assert ig.load_generated_image("http://localhost:8000" + saved.url) == (b"data", "image/png")
    for bad in ("/api/images/../../.env", "/api/images/cmd_abc.png", "/plots/gen_x.png",
                "/api/images/gen_" + "0" * 32 + ".png"):
        assert ig.load_generated_image(bad) is None


def test_generate_image_tool_without_a_key_tells_the_model_what_to_say(monkeypatch):
    monkeypatch.delenv("QUASAR_ALLOW_PLATFORM_IMAGE_GENERATION", raising=False)
    with llm_request_context():
        out = ig.run_generate_image_tool(prompt="an apple")
    assert out["success"] is False and "Provider Keys" in out["error"]


def test_generate_image_tool_returns_cards(monkeypatch):
    def fake_generate(model, prompt, **kwargs):
        return ig.ImageGenerationResult(provider="google", model=model, images=[
            ig.GeneratedImage(url="/api/images/gen_" + "a" * 32 + ".png", path="p", mime_type="image/png")])

    monkeypatch.setattr(ig, "generate_image", fake_generate)
    with llm_request_context(provider_api_keys={"google": "k"}, key_source_by_provider={"google": "byok"}):
        out = ig.run_generate_image_tool(prompt="an apple")
    assert out["success"] is True and out["model"] == "gemini-3.1-flash-image"
    card = out["_cards"][0]
    assert card["type"] == "image" and card["meta"] == {"kind": "generated", "model": "gemini-3.1-flash-image"}


def test_image_pack_triggers_only_on_generation_requests():
    from core.tool_packs import packs_from_text

    assert "image_generation" in packs_from_text("generate an image of an apple")
    assert "image_generation" in packs_from_text("Draw me an artist's impression of a protoplanetary disk")
    assert "image_generation" not in packs_from_text("show me an image of M31 from DSS")
    assert "image_generation" not in packs_from_text("Find ALMA images of HL Tau")


def test_injected_unsigned_calls_get_the_documented_placeholder_on_gemini_3():
    history = [
        {"role": "user", "content": "summarize my notes"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_prefetch1", "type": "function",
             "function": {"name": "search_my_documents", "arguments": "{\"query\": \"notes\"}"}}]},
        {"role": "tool", "tool_call_id": "call_prefetch1", "content": "excerpt"},
    ]
    shim = LLMClient(model="gemini-3.8-flash").responses
    _, contents = shim._google_contents(history, model="gemini-3.8-flash")
    call = contents[1]["parts"][0]
    assert call["thought_signature"] == b"skip_thought_signature_validator"
    assert "id" not in call["function_call"]  # our own call_ ids are never sent as Gemini ids
    assert contents[2]["parts"][0]["function_response"]["name"] == "search_my_documents"
    # Gemma never sees a signature it did not produce.
    _, gemma = shim._google_contents(history, model="gemma-4-31b-it")
    assert "thought_signature" not in gemma[1]["parts"][0]


def _google_llm(resp):
    llm = LLMClient(model="gemini-3.1-flash-image")
    llm._get_google_client = lambda: _FakeGoogle([resp])
    return llm


def _image_resp(parts, usage=(11, 1400)):
    return NS(candidates=[NS(finish_reason=NS(value="STOP"), content=NS(parts=parts))],
              usage_metadata=NS(prompt_token_count=usage[0], candidates_token_count=usage[1],
                                thoughts_token_count=0), prompt_feedback=None)


def test_a_failed_save_still_records_the_billed_usage(rendered_dir, monkeypatch):
    def disk_full(data, mime):
        raise OSError("No space left on device")

    monkeypatch.setattr(ig, "save_image", disk_full)
    usage, released = [], []
    with llm_request_context(provider_api_keys={"google": "k"}, key_source_by_provider={"google": "byok"},
                             usage_recorder=_recorder(usage), quota_checker=lambda **kw: "r1",
                             quota_releaser=released.append):
        with pytest.raises(ig.ImageGenerationError, match="could not be saved"):
            ig.generate_image("gemini-3.1-flash-image", "x", client=_google_llm(
                _image_resp([_part(inline=NS(data=b"img", mime_type="image/png"))])))
    assert usage and usage[0]["output_tokens"] == 1400 and usage[0]["reservation_id"] == "r1"
    assert released == []


def test_malformed_openai_payload_is_reported_and_billed(rendered_dir):
    llm = LLMClient(model="gpt-image-2")
    llm._get_openai_client = lambda: NS(images=NS(generate=lambda **kw: NS(
        data=[NS(b64_json="!!not base64!!", revised_prompt=None)], output_format="png",
        usage=NS(input_tokens=5, output_tokens=272))))
    usage = []
    with llm_request_context(provider_api_keys={"openai": "k"}, key_source_by_provider={"openai": "byok"},
                             usage_recorder=_recorder(usage)):
        with pytest.raises(ig.ImageGenerationError, match="could not be saved"):
            ig.generate_image("gpt-image-2", "x", client=llm)
    assert usage[0]["output_tokens"] == 272


def test_a_provider_that_never_answered_releases_the_reservation():
    def boom(**kwargs):
        raise RuntimeError("connection reset")

    llm = LLMClient(model="gpt-image-2")
    llm._get_openai_client = lambda: NS(images=NS(generate=boom))
    released = []
    with llm_request_context(provider_api_keys={"openai": "k"}, key_source_by_provider={"openai": "byok"},
                             quota_checker=lambda **kw: "r9", quota_releaser=released.append):
        with pytest.raises(ig.ImageGenerationError):
            ig.generate_image("gpt-image-2", "x", client=llm)
    assert released == ["r9"]


def test_text_only_gemini_answer_returns_no_images_with_the_text(rendered_dir):
    with llm_request_context(provider_api_keys={"google": "k"}, key_source_by_provider={"google": "byok"}):
        result = ig.generate_image("gemini-3.1-flash-image", "x", client=_google_llm(
            _image_resp([_part(text="Which style would you like?")])))
    assert result.images == [] and result.text == "Which style would you like?"


@pytest.mark.parametrize("refs,match", [
    ([(b"x" * (ig.MAX_REFERENCE_BYTES + 1), "image/png")], "larger than"),
    ([(b"x", "image/png")] * (ig.MAX_REFERENCE_IMAGES + 1), "at most"),
    ([(b"%PDF", "application/pdf")], "must be images"),
])
def test_unusable_references_are_rejected_before_any_spend(refs, match):
    checked = []
    with llm_request_context(provider_api_keys={"openai": "k"}, key_source_by_provider={"openai": "byok"},
                             quota_checker=lambda **kw: checked.append(kw)):
        with pytest.raises(ig.ImageGenerationError, match=match):
            ig.generate_image("gpt-image-2", "edit it", references=refs)
    assert checked == []


def test_provider_calls_carry_a_bounded_timeout(rendered_dir, monkeypatch):
    seen = {}

    def generate(**kwargs):
        seen.update(kwargs)
        return NS(data=[NS(b64_json=base64.b64encode(b"p").decode(), revised_prompt=None)],
                  output_format="png", usage=NS(input_tokens=1, output_tokens=1))

    monkeypatch.setenv("QUASAR_IMAGE_TIMEOUT_SECONDS", "42")
    llm = LLMClient(model="gpt-image-2")
    llm._get_openai_client = lambda: NS(images=NS(generate=generate))
    with llm_request_context(provider_api_keys={"openai": "k"}, key_source_by_provider={"openai": "byok"}):
        ig.generate_image("gpt-image-2", "x", client=llm)
    assert seen["timeout"] == 42.0


def test_an_empty_upload_is_rejected_not_silently_dropped():
    checked = []
    with llm_request_context(provider_api_keys={"openai": "k"}, key_source_by_provider={"openai": "byok"},
                             quota_checker=lambda **kw: checked.append(kw)):
        with pytest.raises(ig.ImageGenerationError, match="empty"):
            ig.generate_image("gpt-image-2", "make it green", references=[(b"", "image/png")],
                              previous_image=(b"old", "image/png"))
    assert checked == []


def test_sdk_retries_are_disabled_so_a_call_cannot_outlive_its_deadline(rendered_dir):
    seen = {}

    class Client:
        def with_options(self, **kwargs):
            seen["options"] = kwargs
            return NS(images=NS(generate=lambda **kw: NS(
                data=[NS(b64_json=base64.b64encode(b"p").decode(), revised_prompt=None)],
                output_format="png", usage=NS(input_tokens=1, output_tokens=1))))

    llm = LLMClient(model="gpt-image-2")
    llm._get_openai_client = lambda: Client()
    with llm_request_context(provider_api_keys={"openai": "k"}, key_source_by_provider={"openai": "byok"}):
        ig.generate_image("gpt-image-2", "x", client=llm)
    assert seen["options"]["max_retries"] == 0 and seen["options"]["timeout"] > 0

    fake = _FakeGoogle([_image_resp([_part(inline=NS(data=b"i", mime_type="image/png"))])])
    gllm = LLMClient(model="gemini-3.1-flash-image")
    gllm._get_google_client = lambda: fake
    with llm_request_context(provider_api_keys={"google": "k"}, key_source_by_provider={"google": "byok"}):
        ig.generate_image("gemini-3.1-flash-image", "x", client=gllm)
    http = fake.calls[0]["config"]["http_options"]
    assert http["retry_options"] == {"attempts": 1} and http["timeout"] > 0
