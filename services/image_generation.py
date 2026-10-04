"""
AI image generation: Gemini image models ("Nano Banana") and OpenAI gpt-image.

CALLED BY: ui-pro/api/sse.py (a chat turn whose selected model is an image
           model), services/provider_models.py (catalog capability flag)
CALLS:     core.llm_client.LLMClient client getters (BYOK key resolution),
           the request's quota checker / usage recorder

An image model never runs the agent loop: the user's message is the prompt,
the provider returns one or more images, and they are saved under
data/rendered_images (served at /api/images, like every other rendered image)
and shown as image cards. Uploaded images, or the image this chat generated
last, go along as references so "make it green" edits the previous picture.

Image generation is billed to the user's own key. A platform key is refused
unless QUASAR_ALLOW_PLATFORM_IMAGE_GENERATION=1: images cost 10-100x a text
turn and the platform quota counts tokens, not dollars. The flag only permits
the SPEND; it does not list any model. Platform image models appear only when
an operator adds them to a curated platform list (e.g. QUASAR_OPENAI_MODELS);
Google image models stay BYOK-only in the picker, like every Gemini model, and
with the flag the generate_image tool may use the platform Gemini key.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

RENDERED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "rendered_images")
URL_PREFIX = "/api/images/"
GENERATED_PREFIX = "gen_"
MAX_REFERENCE_IMAGES = 4
MAX_REFERENCE_BYTES = 15 * 1024 * 1024

_GOOGLE_IMAGE_RE = re.compile(r"^(?:gemini-[\w.-]+-image(?:-preview)?(?:-\d+)*|nano-banana[\w.-]*)$")
_OPENAI_IMAGE_RE = re.compile(r"^(?:gpt-image-[\w.-]+|chatgpt-image-[\w.-]+)$")
_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}


def image_model_provider(model_id: str, provider: Optional[str] = None) -> Optional[str]:
    """'google' or 'openai' when ``model_id`` is an image-generation model, else None."""
    model = (model_id or "").strip().lower()
    if provider in (None, "google") and _GOOGLE_IMAGE_RE.match(model):
        return "google"
    if provider in (None, "openai") and _OPENAI_IMAGE_RE.match(model):
        return "openai"
    return None


def is_image_generation_model(model_id: str, provider: Optional[str] = None) -> bool:
    return image_model_provider(model_id, provider) is not None


class ImageGenerationError(Exception):
    """A failure whose message is safe to show in the chat."""


@dataclass
class GeneratedImage:
    url: str
    path: str
    mime_type: str


@dataclass
class ImageGenerationResult:
    provider: str
    model: str
    images: List[GeneratedImage] = field(default_factory=list)
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


def platform_images_allowed() -> bool:
    return os.getenv("QUASAR_ALLOW_PLATFORM_IMAGE_GENERATION", "").strip().lower() in ("1", "true", "yes", "on")


def save_image(data: bytes, mime_type: str) -> GeneratedImage:
    os.makedirs(RENDERED_DIR, exist_ok=True)
    ext = _EXT.get((mime_type or "").lower(), "png")
    name = f"{GENERATED_PREFIX}{uuid.uuid4().hex}.{ext}"
    path = os.path.join(RENDERED_DIR, name)
    with open(path, "wb") as fh:
        fh.write(data)
    return GeneratedImage(url=URL_PREFIX + name, path=path, mime_type=mime_type or "image/png")


def load_generated_image(url: str) -> Optional[Tuple[bytes, str]]:
    """Bytes + mime of an image THIS service saved, from its /api/images URL.

    Only our own gen_* files resolve; anything else (other paths, ../, remote
    URLs) returns None, so a saved message cannot make the server read an
    arbitrary file."""
    match = re.match(r"^(?:https?://[^/]+)?/api/images/(gen_[0-9a-f]{32}\.(png|jpg|webp|gif))$", str(url or ""))
    if not match:
        return None
    path = os.path.join(RENDERED_DIR, match.group(1))
    if not os.path.isfile(path):
        return None
    mime = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp", "gif": "image/gif"}[match.group(2)]
    with open(path, "rb") as fh:
        return fh.read(), mime


def decode_data_url(url: str) -> Optional[Tuple[bytes, str]]:
    match = re.match(r"^data:(image/[\w+.-]+);base64,(.*)$", str(url or ""), re.S)
    if not match:
        return None
    try:
        return base64.b64decode(match.group(2)), match.group(1)
    except (ValueError, TypeError):
        return None


def _usable_references(references: Sequence[Tuple[bytes, str]]) -> List[Tuple[bytes, str]]:
    """Validate uploaded references BEFORE anything is spent: an edit must
    never silently run without the image it was asked to edit."""
    out = []
    for data, mime in references or []:
        if not data:
            raise ImageGenerationError("An attached image is empty. Attach the image again.")
        if not str(mime).startswith("image/"):
            raise ImageGenerationError("Reference files for an image model must be images.")
        if len(data) > MAX_REFERENCE_BYTES:
            raise ImageGenerationError(
                f"A reference image is larger than {MAX_REFERENCE_BYTES // (1024 * 1024)} MB. "
                "Attach a smaller image."
            )
        out.append((data, mime))
    if len(out) > MAX_REFERENCE_IMAGES:
        raise ImageGenerationError(
            f"Attach at most {MAX_REFERENCE_IMAGES} reference images to an image model (got {len(out)})."
        )
    return out


# Ceiling for one provider image request. The generate_image tool runs inside
# its guarded deadline (services/tool_budgets.py), which can only shorten it.
PROVIDER_TIMEOUT_SECONDS = 120.0


def _provider_timeout() -> float:
    from services.tool_budgets import bounded_timeout

    try:
        default = float(os.getenv("QUASAR_IMAGE_TIMEOUT_SECONDS", "") or PROVIDER_TIMEOUT_SECONDS)
    except ValueError:
        default = PROVIDER_TIMEOUT_SECONDS
    return bounded_timeout(default, minimum=5.0, label="image generation")


def generate_image(
    model: str,
    prompt: str,
    *,
    references: Sequence[Tuple[bytes, str]] = (),
    previous_image: Optional[Tuple[bytes, str]] = None,
    aspect_ratio: Optional[str] = None,
    client: Any = None,
) -> ImageGenerationResult:
    """Generate (or edit) an image with ``model``; settles quota and usage.

    ``references`` are images the user uploaded this turn. ``previous_image``
    is the last image this chat generated, offered so a follow-up can refine
    it; it is only used when the user uploaded nothing.

    Every path settles the quota reservation: usage is recorded whenever the
    provider answered (it billed the call, even when no image could be used or
    saved); the reservation is released only when it never answered.
    """
    from core.llm_client import LLMClient, get_llm_request_context

    provider = image_model_provider(model)
    if provider is None:
        raise ImageGenerationError(f"{model} is not an image-generation model.")
    prompt = (prompt or "").strip()
    if not prompt:
        raise ImageGenerationError("Describe the image you want.")
    llm = client or LLMClient(model=model)
    key_source = llm._resolve_key_source(provider)
    if key_source != "byok" and not platform_images_allowed():
        label = "Google Gemini" if provider == "google" else "OpenAI"
        raise ImageGenerationError(
            f"Image generation uses your own API key. Add a {label} key in Settings > Provider Keys."
        )

    refs = _usable_references(references)
    use_previous = previous_image is not None and not refs
    timeout = _provider_timeout()
    context = get_llm_request_context()
    reservation_id = None
    if context and context.quota_checker:
        reservation_id = context.quota_checker(provider=provider, model=model, key_source=key_source)
    settled = False

    def record(res: ImageGenerationResult) -> bool:
        recorder = context.usage_recorder if context else None
        if recorder is None or not (res.input_tokens or res.output_tokens):
            return False
        try:
            recorder(
                provider=provider, model=model, key_source=key_source,
                input_tokens=res.input_tokens, output_tokens=res.output_tokens,
                reservation_id=reservation_id,
            )
            return True
        except Exception as exc:
            logger.warning("[image] could not record usage: %s", type(exc).__name__)
            return False

    try:
        call = _call_google if provider == "google" else _call_openai
        resp = call(llm, model, prompt, refs, previous_image if use_previous else None,
                    _normalize_aspect(aspect_ratio), timeout)
        # The provider answered and billed: take its usage before anything
        # that can fail (decoding, disk), so the spend is never lost.
        result = ImageGenerationResult(provider=provider, model=model)
        result.input_tokens, result.output_tokens = _response_usage(provider, resp)
        settled = record(result)
        try:
            (_finish_google if provider == "google" else _finish_openai)(resp, result)
        except ImageGenerationError:
            raise
        except Exception as exc:  # malformed payload, disk full, ...
            logger.warning("[image] could not decode or save the image: %s", type(exc).__name__)
            raise ImageGenerationError(
                "The image was generated but could not be saved on the server. Please try again."
            ) from exc
        return result
    finally:
        if not settled and reservation_id and context and context.quota_releaser:
            try:
                context.quota_releaser(reservation_id)
            except Exception as exc:  # the reservation expires on its own
                logger.warning("[image] could not release quota reservation: %s", type(exc).__name__)


_PREVIOUS_IMAGE_NOTE = (
    "The attached image is the one you generated earlier in this conversation. "
    "If the request below asks to change, refine or continue it, edit that image; "
    "if it asks for something new, ignore it and create a new image."
)


_ASPECTS = ("1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9")
# gpt-image accepts three sizes; map each aspect to the nearest.
_OPENAI_SIZE = {"1:1": "1024x1024", "3:2": "1536x1024", "4:3": "1536x1024", "16:9": "1536x1024",
                "21:9": "1536x1024", "2:3": "1024x1536", "3:4": "1024x1536", "9:16": "1024x1536"}


def _normalize_aspect(value: Optional[str]) -> Optional[str]:
    value = str(value or "").strip().replace(" ", "")
    return value if value in _ASPECTS else None


def _response_usage(provider: str, resp) -> Tuple[int, int]:
    """(input, output) tokens the provider billed. Gemini thinking tokens bill as output."""
    if provider == "google":
        meta = getattr(resp, "usage_metadata", None)
        if meta is None:
            return 0, 0
        return (int(getattr(meta, "prompt_token_count", 0) or 0),
                int(getattr(meta, "candidates_token_count", 0) or 0)
                + int(getattr(meta, "thoughts_token_count", 0) or 0))
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)


def _call_google(llm, model, prompt, refs, previous, aspect, timeout):
    client = llm._get_google_client()
    parts: List[Dict[str, Any]] = []
    for data, mime in refs:
        parts.append({"inline_data": {"mime_type": mime, "data": data}})
    if previous is not None:
        parts.append({"inline_data": {"mime_type": previous[1], "data": previous[0]}})
        parts.append({"text": _PREVIOUS_IMAGE_NOTE})
    parts.append({"text": prompt})
    # One attempt: a retry could start after the caller's deadline and bill
    # an image nobody receives (the timeout bounds the whole call).
    config: Dict[str, Any] = {"response_modalities": ["TEXT", "IMAGE"],
                              "http_options": {"timeout": int(timeout * 1000),
                                               "retry_options": {"attempts": 1}}}
    if aspect:
        config["image_config"] = {"aspect_ratio": aspect}
    try:
        return client.models.generate_content(
            model=model, contents=[{"role": "user", "parts": parts}], config=config,
        )
    except Exception as exc:
        raise ImageGenerationError(_provider_error_message(exc, "Gemini")) from exc


def _finish_google(resp, result: ImageGenerationResult) -> None:
    candidates = getattr(resp, "candidates", None) or []
    finish = ""
    texts: List[str] = []
    for candidate in candidates[:1]:
        reason = getattr(candidate, "finish_reason", None)
        finish = str(getattr(reason, "value", None) or reason or "").upper()
        content = getattr(candidate, "content", None)
        for part in (getattr(content, "parts", None) or []):
            if getattr(part, "thought", False):
                continue  # Gemini 3 Pro Image streams interim "thinking" images
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                result.images.append(save_image(bytes(inline.data), getattr(inline, "mime_type", "") or "image/png"))
            elif getattr(part, "text", None):
                texts.append(part.text)
    result.text = "\n".join(t.strip() for t in texts if t.strip())
    if not result.images:
        feedback = getattr(resp, "prompt_feedback", None)
        blocked = getattr(feedback, "block_reason", None) if feedback else None
        if blocked or "SAFETY" in finish or "PROHIBITED" in finish or "BLOCKLIST" in finish:
            raise ImageGenerationError("Gemini declined to generate this image under its safety policy. "
                                       "Try rephrasing the request.")
        if not result.text:
            raise ImageGenerationError("Gemini returned no image for this request. Try rephrasing it.")
        # Text only (a clarifying question or refusal): returned with no
        # images; callers report it as an unsuccessful generation.


def _call_openai(llm, model, prompt, refs, previous, aspect, timeout):
    client = llm._get_openai_client()
    # No SDK retries (default 2): each would restart the full timeout, so the
    # call could outlive the tool deadline and bill an abandoned image.
    if hasattr(client, "with_options"):
        client = client.with_options(max_retries=0, timeout=timeout)
    images_in = list(refs) or ([previous] if previous is not None else [])
    extra: Dict[str, Any] = {"timeout": timeout}
    if aspect:
        extra["size"] = _OPENAI_SIZE[aspect]
    try:
        if images_in:
            files = [(f"reference_{i + 1}.{_EXT.get(mime, 'png')}", data, mime)
                     for i, (data, mime) in enumerate(images_in)]
            edit_prompt = prompt if refs else f"{_PREVIOUS_IMAGE_NOTE}\n\n{prompt}"
            return client.images.edit(model=model, image=files, prompt=edit_prompt, **extra)
        return client.images.generate(model=model, prompt=prompt, **extra)
    except Exception as exc:
        raise ImageGenerationError(_provider_error_message(exc, "OpenAI")) from exc


def _finish_openai(resp, result: ImageGenerationResult) -> None:
    fmt = str(getattr(resp, "output_format", None) or "png").lower()
    mime = {"jpeg": "image/jpeg", "jpg": "image/jpeg", "webp": "image/webp"}.get(fmt, "image/png")
    for item in getattr(resp, "data", None) or []:
        b64 = getattr(item, "b64_json", None)
        if b64:
            result.images.append(save_image(base64.b64decode(b64, validate=True), mime))
        revised = getattr(item, "revised_prompt", None)
        if revised:
            result.text = f"Prompt as revised by the model: {revised}"
    if not result.images:
        raise ImageGenerationError("OpenAI returned no image for this request. Try rephrasing it.")


def _provider_error_message(exc: Exception, label: str) -> str:
    from core.retry import _extract_status_code

    status = _extract_status_code(exc)
    text = str(exc).lower()
    if "moderation" in text or "safety" in text or "content_policy" in text:
        return f"{label} declined to generate this image under its content policy. Try rephrasing the request."
    if status in (401, 403) or "api key not valid" in text or "invalid api key" in text or "incorrect api key" in text:
        return f"{label} rejected the API key. Check it in Settings > Provider Keys."
    if status == 404:
        return f"{label} reports this image model is not available to your API key. Pick another image model."
    if status == 429:
        return f"{label} rate-limited the request or the key is out of credit. Wait a moment, or check billing."
    if status == 400:
        return f"{label} rejected the request (HTTP 400). Try a shorter prompt or a different reference image."
    detail = f" (HTTP {status})" if status else ""
    return f"{label} could not generate the image{detail}. Please try again."


# ── generate_image tool (chat models) ─────────────────────────────────────

TOOL_NAME = "generate_image"
TOOL_DESCRIPTION = (
    "Generate an AI image (illustration, diagram-style artwork, artist's impression, picture) from a text "
    "description and show it to the user. Use it ONLY when the user explicitly asks you to generate, create, "
    "draw or illustrate an image. The result is AI-generated artwork, NOT data: never use it to show what a "
    "real object looks like in observations (use hips_cutout or archive imagery for real sky images), and "
    "never present it as an observation. Runs on the user's own Google Gemini or OpenAI key; if it reports "
    "that no key is available, tell the user how to add one instead of describing an image."
)
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string", "description": "Detailed description of the image to generate, in English."},
        "aspect_ratio": {"type": "string", "enum": list(_ASPECTS),
                         "description": "Optional aspect ratio. Default 1:1."},
    },
    "required": ["prompt"],
}


def tool_image_model() -> Optional[str]:
    """Image model the tool uses for this request: the user's own Google key
    first (Nano Banana 2), then their OpenAI key. Env overrides the models."""
    from core.llm_client import get_llm_request_context

    context = get_llm_request_context()
    keys = (context.provider_api_keys or {}) if context else {}
    sources = (context.key_source_by_provider or {}) if context else {}
    for provider, env, default in (("google", "QUASAR_IMAGE_TOOL_GOOGLE_MODEL", "gemini-3.1-flash-image"),
                                   ("openai", "QUASAR_IMAGE_TOOL_OPENAI_MODEL", "gpt-image-2")):
        byok = bool(keys.get(provider)) and sources.get(provider, "byok") == "byok"
        if byok or (platform_images_allowed() and provider == "openai" and os.getenv("OPENAI_API_KEY")) or (
                platform_images_allowed() and provider == "google" and os.getenv("GEMINI_API_KEY")):
            model = os.getenv(env, "").strip() or default
            if image_model_provider(model, provider):
                return model
    return None


def run_generate_image_tool(prompt: str = "", aspect_ratio: Optional[str] = None, **_ignored) -> Dict[str, Any]:
    """Tool body. Returns the model-facing result; image cards ride in "_cards"."""
    model = tool_image_model()
    if model is None:
        return {
            "success": False,
            "error": ("Image generation needs the user's own Google Gemini or OpenAI API key. Tell the user to "
                      "add one in Settings > Provider Keys, or to pick an image model (for example a Gemini "
                      "'image' model) in the model selector. Do not describe or invent an image."),
        }
    try:
        result = generate_image(model, prompt, aspect_ratio=aspect_ratio)
    except ImageGenerationError as exc:
        return {"success": False, "model": model, "error": str(exc)}
    if not result.images:
        return {"success": False, "model": model,
                "error": result.text or "The image model returned no image."}
    caption = f"AI-generated by {model}"
    return {
        "success": True,
        "model": model,
        "images_generated": len(result.images),
        "model_comment": result.text[:500],
        "note": ("The image is already displayed to the user as a card. Do not embed it, link it or repeat "
                 "its URL; say in one or two sentences what was generated and that it is AI-generated."),
        "_cards": [{"type": "image", "image_url": img.url, "caption": caption,
                    "meta": {"kind": "generated", "model": model}} for img in result.images],
    }
