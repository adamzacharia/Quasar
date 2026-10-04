"""Typed, server-only provider model discovery. Provider payloads are untrusted."""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError

from services.image_generation import is_image_generation_model
from services.model_pricing import CATALOG_UNVERIFIED_PRICES, get_model_pricing

Provider = Literal["openai", "deepseek", "anthropic", "google", "tacc", "local"]
KeyProvider = Literal["openai", "deepseek", "anthropic", "google"]
logger = logging.getLogger(__name__)


class Capabilities(BaseModel):
    vision: bool | None = None
    tools: bool | None = None
    reasoning: bool | None = None
    # Image-generation model (Gemini "Nano Banana", OpenAI gpt-image): a chat
    # turn on it returns images instead of running the agent.
    imageGeneration: bool | None = None


class ModelInfo(BaseModel):
    provider: Provider
    id: str
    displayName: str
    contextWindow: int | None = None
    maxOutput: int | None = None
    inputPricePerM: float | None = None
    outputPricePerM: float | None = None
    capabilities: Capabilities = Field(default_factory=Capabilities)
    createdAt: datetime | None = None
    source: Literal["live", "static"] = "live"


class ProviderCatalog(BaseModel):
    provider: Provider
    status: Literal["connected", "included_quota", "not_connected"]
    models: list[ModelInfo] = Field(default_factory=list)
    stale: bool = False
    fetchedAt: datetime | None = None
    # Curated fallback count, not a claim about inaccessible account entitlements.
    unlockCount: int = 0


class AvailableModels(BaseModel):
    providers: list[ProviderCatalog]
    defaultModel: str = "gpt-oss-120b"


class CatalogError(Exception):
    def __init__(self, message: str, invalid_key: bool = False):
        super().__init__(message)
        self.invalid_key = invalid_key


class ValidationResult(BaseModel):
    ok: bool
    error: str | None = None
    invalidKey: bool = False
    models: list[ModelInfo] = Field(default_factory=list)


class ProviderAdapter(Protocol):
    def list_models(self, api_key: str) -> list[ModelInfo]: ...
    def validate_key(self, api_key: str) -> ValidationResult: ...


class _Capability(BaseModel):
    supported: bool | None = None


class _AnthropicCapabilities(BaseModel):
    image_input: _Capability | None = None
    tool_use: _Capability | None = None
    thinking: _Capability | None = None


class _RawModel(BaseModel):
    id: str = ""
    name: str = ""
    display_name: str = ""
    displayName: str = ""
    created: int | None = None
    created_at: datetime | None = None
    max_input_tokens: int | None = None
    max_tokens: int | None = None
    inputTokenLimit: int | None = None
    outputTokenLimit: int | None = None
    supportedGenerationMethods: list[str] = Field(default_factory=list)
    capabilities: _AnthropicCapabilities | None = None


class _Page(BaseModel):
    data: list[_RawModel] | None = None
    models: list[_RawModel] | None = None
    has_more: bool = False
    last_id: str | None = None
    nextPageToken: str | None = None


EXCLUDED = ("embedding", "whisper", "tts", "dall-e", "moderation", "realtime",
            "audio", "image", "transcribe", "babbage", "davinci", "sora", "gpt-live-")
SNAPSHOT = re.compile(r"-(?:\d{4}-\d{2}-\d{2}|\d{8}|\d{4})$")
# Official model pages list no streaming support for these text models.
# Quasar's chat/tool loop requires streamed Responses events.
NON_STREAMING_OPENAI = ("gpt-5.5-pro", "o3-pro")


def is_openai_chat_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return not (any(word in lowered for word in EXCLUDED) or any(
        lowered == name or lowered.startswith(name + "-") for name in NON_STREAMING_OPENAI
    ))


# Gemini models that cannot serve a Quasar chat turn: embeddings, media and
# speech generation, Live/Interactions-API-only and agent products, and the
# computer-use/robotics models that need their own tools. Image models
# ("-image", "nano-banana") stay: they serve image-generation turns.
GOOGLE_EXCLUDED = ("embedding", "aqa", "imagen", "veo", "tts", "lyria", "live", "native-audio",
                   "transcribe", "computer-use", "robotics", "antigravity", "deep-research", "omni")


def is_google_selectable_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(word in lowered for word in GOOGLE_EXCLUDED)


def with_capability_flags(model: ModelInfo) -> ModelInfo:
    """Stamp capabilities that follow from the model id alone (old caches and
    curated static entries carry none)."""
    if is_image_generation_model(model.id, model.provider) and not model.capabilities.imageGeneration:
        caps = model.capabilities.model_copy(update={"imageGeneration": True, "tools": False})
        return model.model_copy(update={"capabilities": caps})
    return model
FAMILY_ORDER: dict[str, tuple[str, ...]] = {
    "anthropic": ("claude-opus", "claude-sonnet", "claude-haiku"),
    "openai": ("gpt-", "o4", "o3", "o1"),
    "deepseek": ("deepseek-v4-pro", "deepseek-v4-flash", "deepseek-reasoner", "deepseek-chat"),
    "google": ("gemini-3", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0", "gemini-1.5"),
}
# These existing accounting entries are explicitly labelled inferred, so do not
# present them as known catalog prices. Accounting behaviour stays unchanged.


def price_model(model: ModelInfo) -> ModelInfo:
    if any(model.id == key or model.id.startswith(key + "-") for key in CATALOG_UNVERIFIED_PRICES):
        return model.model_copy(update={"inputPricePerM": None, "outputPricePerM": None})
    rates = get_model_pricing(model.provider, model.id)
    if rates:
        return model.model_copy(update={"inputPricePerM": rates["input_per_mtok"],
                                        "outputPricePerM": rates["output_per_mtok"]})
    # Cached estimates from older rate tables must not survive a now-unmatched ID.
    return model.model_copy(update={"inputPricePerM": None, "outputPricePerM": None})


def sort_models(models: list[ModelInfo]) -> list[ModelInfo]:
    def rank(model: ModelInfo) -> tuple:
        families = FAMILY_ORDER.get(model.provider, ())
        index = next((i for i, family in enumerate(families) if model.id.startswith(family)), len(families))
        created = model.createdAt
        timestamp = created.replace(tzinfo=created.tzinfo or timezone.utc).timestamp() if created else 0
        if model.provider == "openai":
            generation = re.match(r"^gpt-(\d+)(?:\.(\d+))?", model.id)
            major, minor = (int(generation[1]), int(generation[2] or 0)) if generation else (0, 0)
            suffix = model.id[generation.end():].lstrip("-") if generation else ""
            tiers = ("", "astra", "sol", "terra", "luna", "mini", "nano", "pro", "codex")
            tier = next((i for i, name in enumerate(tiers) if suffix == name or (name and suffix.startswith(name + "-"))), len(tiers))
            return index, -major, -minor, bool(SNAPSHOT.search(model.id)), -timestamp, tier, model.id.casefold()
        return index, -timestamp, model.id.casefold()
    return sorted({model.id: model for model in models}.values(), key=rank)


def chat_models(models: list[ModelInfo]) -> list[ModelInfo]:
    """Apply current chat compatibility to old caches as well as discovery."""
    def supported(model: ModelInfo) -> bool:
        if model.provider == "openai":
            return is_openai_chat_model(model.id) or is_image_generation_model(model.id, "openai")
        if model.provider == "deepseek":
            return not any(word in model.id.lower() for word in EXCLUDED)
        if model.provider == "google":
            return is_google_selectable_model(model.id)
        return True
    return sort_models([with_capability_flags(model) for model in models if supported(model)])


class _Adapter:
    provider: KeyProvider
    url: str

    def __init__(self, transport: httpx.BaseTransport | None = None):
        self.transport = transport

    def normalize(self, rows: list[_RawModel]) -> list[ModelInfo]:
        result: list[ModelInfo] = []
        ids = {row.id for row in rows}
        for row in rows:
            model_id = row.id
            capabilities = Capabilities()
            if self.provider == "google":
                model_id = row.name.removeprefix("models/")
                if "generateContent" not in row.supportedGenerationMethods or not is_google_selectable_model(model_id):
                    continue
            elif self.provider in {"openai", "deepseek"}:
                image_model = self.provider == "openai" and is_image_generation_model(model_id, "openai")
                if any(word in model_id.lower() for word in EXCLUDED) and not image_model:
                    continue
                if self.provider == "openai" and not (is_openai_chat_model(model_id) or image_model):
                    continue
                alias = SNAPSHOT.sub("", model_id)
                if alias != model_id and alias in ids:
                    continue
            elif row.capabilities:
                caps = row.capabilities
                capabilities = Capabilities(
                    vision=caps.image_input.supported if caps.image_input else None,
                    tools=caps.tool_use.supported if caps.tool_use else None,
                    reasoning=caps.thinking.supported if caps.thinking else None,
                )
            if not model_id:
                continue
            created = row.created_at
            if created is None and row.created is not None:
                created = datetime.fromtimestamp(row.created, timezone.utc)
            result.append(with_capability_flags(price_model(ModelInfo(
                provider=self.provider, id=model_id,
                displayName=row.display_name or row.displayName or model_id,
                contextWindow=row.max_input_tokens or row.inputTokenLimit,
                maxOutput=row.max_tokens or row.outputTokenLimit,
                capabilities=capabilities, createdAt=created,
            ))))
        return sort_models(result)

    def list_models(self, api_key: str) -> list[ModelInfo]:
        headers = {"Authorization": f"Bearer {api_key}"}
        params: dict[str, str | int] = {}
        if self.provider == "anthropic":
            headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
            params["limit"] = 100
        elif self.provider == "google":
            # Equivalent Google API-key header avoids putting credentials in
            # request URLs, access logs, tracebacks, or httpx INFO logs.
            headers = {"x-goog-api-key": api_key}
            params["pageSize"] = 200
        rows: list[_RawModel] = []
        cursors: set[str] = set()
        deadline = time.monotonic() + 30
        try:
            with httpx.Client(timeout=10, follow_redirects=False, transport=self.transport) as client:
                for _ in range(100):
                    if time.monotonic() >= deadline:
                        raise CatalogError("Provider model discovery timed out. Please try again.")
                    with client.stream("GET", self.url, headers=headers, params=params) as response:
                        chunks: list[bytes] = []
                        size = 0
                        for chunk in response.iter_bytes():
                            size += len(chunk)
                            if size > 2_000_000 or time.monotonic() >= deadline:
                                raise CatalogError("Provider model response exceeded its safety limit.")
                            chunks.append(chunk)
                        body = b"".join(chunks)
                    if response.status_code in (401, 403):
                        raise CatalogError("Invalid API key, or key lacks permission to list models.", True)
                    if self.provider == "google" and response.status_code == 400:
                        # Gemini also returns 400 API_KEY_INVALID. Never echo its body.
                        if b"API_KEY_INVALID" in body:
                            raise CatalogError("Invalid API key.", True)
                    response.raise_for_status()
                    page = _Page.model_validate_json(body)
                    batch = page.models if self.provider == "google" else page.data
                    if batch is None:
                        raise CatalogError("Provider returned an invalid model catalog.")
                    rows.extend(batch)
                    if len(rows) > 20_000:
                        raise CatalogError("Provider model catalog exceeded its safety limit.")
                    cursor = page.nextPageToken if self.provider == "google" else (
                        page.last_id or (batch[-1].id if batch else None)
                    ) if page.has_more else None
                    if not cursor:
                        if page.has_more:
                            raise CatalogError("Provider returned incomplete model pagination.")
                        return self.normalize(rows)
                    if cursor in cursors:
                        raise CatalogError("Provider returned repeated model pagination.")
                    cursors.add(cursor)
                    params["pageToken" if self.provider == "google" else "after_id"] = cursor
            raise CatalogError("Provider model pagination exceeded its safety limit.")
        except CatalogError as exc:
            logger.warning("Model discovery failed: provider=%s invalid_key=%s", self.provider, exc.invalid_key)
            raise
        except (httpx.HTTPError, ValidationError, ValueError, OverflowError) as exc:
            # Exception messages/URLs/bodies may contain credentials. Only log type.
            logger.warning("Model discovery failed: provider=%s error_type=%s", self.provider, type(exc).__name__)
            raise CatalogError("Provider model service unavailable. Please try again.") from None

    def validate_key(self, api_key: str) -> ValidationResult:
        try:
            return ValidationResult(ok=True, models=self.list_models(api_key))
        except CatalogError as exc:
            return ValidationResult(ok=False, error=str(exc), invalidKey=exc.invalid_key)


class OpenAIAdapter(_Adapter):
    provider: KeyProvider = "openai"
    url = "https://api.openai.com/v1/models"


class DeepSeekAdapter(_Adapter):
    provider: KeyProvider = "deepseek"
    url = "https://api.deepseek.com/models"


class AnthropicAdapter(_Adapter):
    provider: KeyProvider = "anthropic"
    url = "https://api.anthropic.com/v1/models"


class GoogleAdapter(_Adapter):
    provider: KeyProvider = "google"
    url = "https://generativelanguage.googleapis.com/v1beta/models"


ADAPTERS: dict[str, ProviderAdapter] = {
    "openai": OpenAIAdapter(), "deepseek": DeepSeekAdapter(),
    "anthropic": AnthropicAdapter(), "google": GoogleAdapter(),
}
