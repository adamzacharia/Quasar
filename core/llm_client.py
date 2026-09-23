# core/llm_client.py
"""
LLM Client — Unified, provider-agnostic interface for all LLM calls.

Every Quasar module (agent, Conductor, RLM, Recovery, sub-agents) calls
`client.responses.create(...)`.  This module provides a drop-in replacement
that routes to the correct provider based on the model name:

    gpt-* / o1* / o3* / o4*  →  OpenAI Responses API  (native)
    claude-*                  →  Anthropic Messages API (tool_use)
    gemini-*                  →  Google GenAI API       (function_calling)
    local/*                   →  OpenAI-compat Chat Completions
                                 (Ollama / LM Studio via base_url)

The key insight is the `ResponsesShim` — a property on `LLMClient` named
`responses` that exposes `.create(**kwargs)`.  This means existing code like
`self.client.responses.create(...)` works unchanged.

Usage:
    # In agent.py (replaces `self.client = OpenAI(...)`)
    from core.llm_client import LLMClient
    self.client = LLMClient(model="gpt-4o")       # OpenAI
    self.client = LLMClient(model="local/deepseek-r1:7b")  # Ollama

    # All existing calls work:
    resp = self.client.responses.create(
        model="gpt-4o", input="Hello", instructions="...",
    )
    print(resp.output_text)
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from core.retry import _is_retryable as _retry_is_retryable
from core.retry import with_retry
from core.harmony_filter import HarmonyStreamFilter
from core.langfuse_integration import get_langfuse, _safe_serialize
from services.secret_redaction import redact_secrets

logger = logging.getLogger(__name__)

# ── Thread-local Langfuse trace context ──────────────────────────────────
# The Conductor sets this before executing DAG nodes so that LLM calls
# made inside tool executors are automatically parented to the right trace.
import threading as _threading
_langfuse_tls = _threading.local()
_request_tls = _threading.local()


def _feedback_trace_factory():
    from services.feedback_snapshot_service import FeedbackToolTrace
    return FeedbackToolTrace()


@dataclass
class LLMRequestContext:
    """Request-scoped provider key and usage accounting context."""

    provider_api_keys: Dict[str, str] = field(default_factory=dict)
    model_providers: Dict[str, str] = field(default_factory=dict)
    key_source_by_provider: Dict[str, str] = field(default_factory=dict)
    user_id: str = "anonymous"
    user_email: str = ""
    usage_recorder: Optional[Callable[..., None]] = None
    quota_checker: Optional[Callable[..., None]] = None
    # Frees a reservation taken by quota_checker when the call never records
    # usage (it raised, or the provider reported none). Without it that call's
    # reserved tokens would hold the user's own headroom until the TTL.
    quota_releaser: Optional[Callable[..., None]] = None
    byok_token_limits: Dict[str, Optional[int]] = field(default_factory=dict)
    # Conductor tool-trace collector (CX-11/CX-39): {"calls": [], "lock": Lock,
    # "owner": <request thread ident>}. Rides the REQUEST-scoped context object
    # (propagated to conductor/executor threads via reinstall) instead of a
    # singleton agent attribute, so concurrent conductor turns can never
    # overwrite or cross-append each other's collectors.
    tool_trace_collector: Optional[Dict[str, Any]] = None
    # R3: mechanical citation recall/precision computed at synthesis
    # finalization (services/citation_metrics.py). Rides the request-scoped
    # context so both the simple-path (runner) and Conductor threads write to
    # the same per-request slot; the SSE layer surfaces it in run_meta.
    citation_metrics: Optional[Dict[str, Any]] = None
    # Full redacted feedback evidence is independent of the compact UI/benchmark
    # trace. The same context follows tools into worker/conductor threads.
    feedback_tool_trace: Any = field(default_factory=_feedback_trace_factory)


@contextmanager
def llm_request_context(
    *,
    provider_api_keys: Optional[Dict[str, str]] = None,
    model_providers: Optional[Dict[str, str]] = None,
    key_source_by_provider: Optional[Dict[str, str]] = None,
    user_id: str = "anonymous",
    user_email: str = "",
    usage_recorder: Optional[Callable[..., None]] = None,
    quota_checker: Optional[Callable[..., None]] = None,
    quota_releaser: Optional[Callable[..., None]] = None,
    byok_token_limits: Optional[Dict[str, Optional[int]]] = None,
):
    """Install request-local LLM key routing and usage accounting."""
    previous = getattr(_request_tls, "context", None)
    _request_tls.context = LLMRequestContext(
        provider_api_keys=dict(provider_api_keys or {}),
        model_providers=dict(model_providers or {}),
        key_source_by_provider=dict(key_source_by_provider or {}),
        user_id=user_id or "anonymous",
        user_email=user_email or "",
        usage_recorder=usage_recorder,
        quota_checker=quota_checker,
        quota_releaser=quota_releaser,
        byok_token_limits=dict(byok_token_limits or {}),
    )
    try:
        yield
    finally:
        if previous is None:
            try:
                delattr(_request_tls, "context")
            except AttributeError:
                pass
        else:
            _request_tls.context = previous


def get_llm_request_context() -> Optional[LLMRequestContext]:
    """Return current request-local LLM context, if any."""
    return getattr(_request_tls, "context", None)


@contextmanager
def reinstall_llm_request_context(ctx: Optional[LLMRequestContext]):
    """Re-install a captured request context on a DIFFERENT thread (A2 CX-01).

    The context lives in a threading.local, so worker threads spawned by the
    runner/conductor never inherit it — LLM calls made there used to skip both
    usage recording and quota admission. Capture ``get_llm_request_context()``
    on the request thread, hand the object to the worker, and run the worker's
    body inside this. No-op when ``ctx`` is None.
    """
    if ctx is None:
        yield
        return
    previous = getattr(_request_tls, "context", None)
    _request_tls.context = ctx
    try:
        yield
    finally:
        if previous is None:
            try:
                delattr(_request_tls, "context")
            except AttributeError:
                pass
        else:
            _request_tls.context = previous

def set_langfuse_parent(parent):
    """Set the current thread's Langfuse trace/span parent for LLM calls."""
    _langfuse_tls.parent = parent

def get_langfuse_parent():
    """Get the current thread's Langfuse parent (trace/span), or None."""
    return getattr(_langfuse_tls, 'parent', None)


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

TACC_DEFAULT_BASE_URL = "https://ai.tejas.tacc.utexas.edu/v1"

TACC_MODEL_IDS = [
    "gpt-oss-120b",
    "Llama-4-Maverick-17B-128E-Instruct",
    "gemma-4-31B-it",
    "MiniMax-M2.7",
    "Qwen3-32B",
    "Meta-Llama-3.2-1B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "Meta-Llama-3.3-70B-Instruct",
    "Mistral-Large-3-675B-Instruct-2512",
    "E5-Mistral-7B-Instruct",
]

TACC_VISIBLE_MODEL_IDS = [
    "gpt-oss-120b",
    "Qwen3-32B",
    "gemma-4-31B-it",
    "MiniMax-M2.7",
]

TACC_MODEL_ID_SET = frozenset(TACC_MODEL_IDS)

def detect_provider(model: str) -> str:
    """Detect the LLM provider from the model name."""
    context = get_llm_request_context()
    if context and model in context.model_providers:
        return context.model_providers[model]
    if model in TACC_MODEL_ID_SET or model.startswith("tacc/"):
        return "tacc"
    if model.startswith("local/"):
        return "local"
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gemini-") or model.startswith("gemma-"):
        return "google"
    if "deepseek" in model.lower():
        return "deepseek"
    # Default: OpenAI (gpt-*, o1*, o3*, o4*, etc.)
    return "openai"


def model_accepts_direct_image_input(model: str) -> bool:
    """Return True when Quasar can pass uploaded images directly to the model."""
    return detect_provider(model) == "openai"


# Attachment support is KIND-specific, not provider-wide (CX-21). Uploaded
# document references carry a per-provider `kind` stamped by
# services/provider_file_service.py `_ref_to_attachment` (:475-488), and each
# builder in this module accepts ONLY its own kind, silently `continue`-ing
# past the rest:
#   "openai_input_file"       — _build_openai_input     (kind check at :805)
#   "anthropic_document_file" — _build_anthropic_messages (kind check at :1082)
#   "gemini_file"             — _build_google_contents  (kind check at :1353)
# Inline image attachments ("image_url" key or type=="image_url") are built in
# by _build_openai_input's image branch and _inject_images_into_messages
# (tacc); anthropic/google builders ignore them, deepseek explicitly DROPS
# them with a warning, local raises. Turn-start failover must not switch an
# attachment-carrying turn onto a provider whose builder would drop any of the
# turn's attachment kinds — the user's uploads would silently vanish.
ATTACHMENT_KIND_PROVIDERS: Dict[str, frozenset] = {
    "openai_input_file": frozenset({"openai"}),
    "anthropic_document_file": frozenset({"anthropic"}),
    "gemini_file": frozenset({"google"}),
    "image_url": frozenset({"openai", "tacc"}),
}


def attachment_kind(attachment: Any) -> str:
    """Classify one attachment dict the way the provider builders do:
    an explicit provider-file `kind` wins; otherwise the image detection used
    by _build_openai_input/_inject_images_into_messages ("image_url" key or
    type=="image_url"); anything else is "unknown"."""
    if not isinstance(attachment, dict):
        return "unknown"
    kind = attachment.get("kind")
    if kind:
        return str(kind)
    if attachment.get("type") == "image_url" or "image_url" in attachment:
        return "image_url"
    return "unknown"


def providers_accepting_attachments(attachments) -> frozenset:
    """Providers whose builders accept EVERY attachment in ``attachments``.

    The intersection over per-kind acceptance sets (ATTACHMENT_KIND_PROVIDERS);
    an unknown kind maps to the empty set — no provider is known to forward
    it, so failover conservatively stays put. An empty/None list intersects
    nothing and returns the union of all known-capable providers (no
    constraint), but callers gate on ``if attachments:`` first anyway.
    """
    compatible: Optional[frozenset] = None
    for attachment in attachments or []:
        accepts = ATTACHMENT_KIND_PROVIDERS.get(attachment_kind(attachment), frozenset())
        compatible = accepts if compatible is None else (compatible & accepts)
        if not compatible:
            return frozenset()
    if compatible is None:
        return frozenset().union(*ATTACHMENT_KIND_PROVIDERS.values())
    return compatible


# Environment variables that establish a platform credential path per provider
# (mirrors the _get_*_client getters below — keep in sync). "local" needs a
# base URL rather than a key; TACC accepts three historical env spellings.
PROVIDER_KEY_ENV_VARS: Dict[str, tuple] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GEMINI_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "tacc": ("TACC_API_KEY", "TEJAS_API_KEY", "TEXAS_AI_API_KEY"),
    "local": ("LOCAL_LLM_BASE_URL",),
}


def provider_has_key_path(provider: str, client: Optional["LLMClient"] = None) -> bool:
    """True when a call to ``provider`` has SOME usable credential path:
    a BYOK key (client-scoped or request-context) or a platform env key.

    Used by turn-start health failover (core/runner.py): switching to a
    fallback whose provider has no key path would just trade one error for
    another, so the failover only engages when this returns True. Unknown
    providers return False.
    """
    provider = LLMClient._normalize_provider_key(provider)
    try:
        if client is not None and (client.provider_api_keys or {}).get(provider):
            return True
    except Exception:
        pass
    context = get_llm_request_context()
    if context and (context.provider_api_keys or {}).get(provider):
        return True
    return any(
        (os.getenv(name, "") or "").strip()
        for name in PROVIDER_KEY_ENV_VARS.get(provider, ())
    )


# ---------------------------------------------------------------------------
# Response objects — mimic OpenAI Responses API output shapes
# ---------------------------------------------------------------------------

@dataclass
class FunctionCallItem:
    """Mimics an OpenAI function_call output item."""
    type: str = "function_call"
    name: str = ""
    arguments: str = ""
    call_id: str = ""
    id: str = ""

    def __post_init__(self):
        if not self.call_id:
            self.call_id = f"call_{uuid.uuid4().hex[:16]}"
        if not self.id:
            self.id = self.call_id


@dataclass
class TextContentItem:
    """Mimics an output_text content block."""
    type: str = "output_text"
    text: str = ""


@dataclass
class MessageOutputItem:
    """Mimics a message output item containing content blocks."""
    type: str = "message"
    content: List[TextContentItem] = field(default_factory=list)


@dataclass
class LLMUsage:
    """Standardized usage token counting for all LLM providers.

    For DeepSeek: the API returns prompt_cache_hit_tokens and
    prompt_cache_miss_tokens which are priced at vastly different rates
    (cache hits are 50-120× cheaper).  We carry both so Langfuse can
    compute accurate costs instead of pricing everything at the
    expensive cache-miss rate.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    # DeepSeek cache-aware breakdown (None = not applicable / unknown)
    cache_hit_tokens: Optional[int] = None
    cache_miss_tokens: Optional[int] = None


@dataclass
class LLMResponse:
    """
    Mimics OpenAI's Response object so all existing code works.

    Key attributes used by existing code:
      - resp.output_text  → combined text
      - resp.id           → response ID (for previous_response_id chaining)
      - resp.output       → list of items (function_call / message items)
    """
    output_text: str = ""
    id: str = ""
    output: List[Any] = field(default_factory=list)
    usage: Optional[LLMUsage] = None
    # Chat Completions finish_reason ("stop" | "length" | "tool_calls" | ...).
    # "length" means the provider truncated the response at its output-token
    # cap — the agent loop uses this to auto-continue instead of treating the
    # cut-off text as the final answer. None = provider did not report one.
    finish_reason: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = f"resp_{uuid.uuid4().hex[:16]}"


# ---------------------------------------------------------------------------
# Streaming event objects — mimic OpenAI SSE events
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    """Mimics a single SSE event from OpenAI's streaming Responses API."""
    type: str = ""
    # For response.created
    response: Optional[LLMResponse] = None
    # For response.output_text.delta
    delta: str = ""
    # For response.output_item.added
    item: Optional[Any] = None


# ---------------------------------------------------------------------------
# Langfuse usage builder — cache-aware cost tracking
# ---------------------------------------------------------------------------

def _build_langfuse_usage(result: Any, provider: str) -> Optional[Dict[str, Any]]:
    """Build a Langfuse-compatible usage dict from an LLM response.

    For DeepSeek, the API returns prompt_cache_hit_tokens and
    prompt_cache_miss_tokens which are priced at 50-120× lower rates
    than cache misses.  Langfuse supports custom usage types in its
    model pricing configuration, so we pass these as separate keys
    so that costs are computed at the correct tier.

    To make this work, create a custom model definition in Langfuse
    (Settings → Models) for each DeepSeek model with usage types:
        - prompt_cache_hit_tokens   → e.g. $0.003625 / 1M tokens
        - prompt_cache_miss_tokens  → e.g. $0.435 / 1M tokens
        - completion_tokens         → e.g. $0.87 / 1M tokens

    For other providers, we pass standard input/output counts which
    Langfuse prices using its built-in model definitions.
    """
    if result is None:
        return None

    usage_obj = getattr(result, 'usage', None)
    if usage_obj is None:
        return None

    input_tokens = (
        getattr(usage_obj, 'input_tokens', 0)
        or getattr(usage_obj, 'prompt_tokens', 0)
        or 0
    )
    output_tokens = (
        getattr(usage_obj, 'output_tokens', 0)
        or getattr(usage_obj, 'completion_tokens', 0)
        or 0
    )

    # DeepSeek: use cache-aware breakdown for accurate Langfuse pricing
    if provider == "deepseek":
        cache_hit = getattr(usage_obj, 'cache_hit_tokens', None) or getattr(usage_obj, 'prompt_cache_hit_tokens', None)
        cache_miss = getattr(usage_obj, 'cache_miss_tokens', None) or getattr(usage_obj, 'prompt_cache_miss_tokens', None)

        if cache_hit is not None or cache_miss is not None:
            return {
                "prompt_cache_hit_tokens": int(cache_hit or 0),
                "prompt_cache_miss_tokens": int(cache_miss or 0),
                "completion_tokens": int(output_tokens),
                "total": int(input_tokens) + int(output_tokens),
                "unit": "TOKENS",
            }

    # All other providers: standard input/output
    if input_tokens or output_tokens:
        return {
            "input": int(input_tokens),
            "output": int(output_tokens),
        }

    return None


# ---------------------------------------------------------------------------
# Responses Shim — makes `client.responses.create(...)` work
# ---------------------------------------------------------------------------

class ResponsesShim:
    """
    Provides `client.responses.create(...)` interface.
    Routes to the correct provider implementation.
    """

    def __init__(self, llm_client: "LLMClient"):
        self._llm = llm_client
        self._history_cache = {}  # response_id -> list of chat messages
        self._history_lock = _threading.Lock()

    def clear_history(self, response_id: Optional[str] = None) -> None:
        """Clear one compatibility-history chain or all cached chains."""
        with self._history_lock:
            if response_id:
                self._history_cache.pop(response_id, None)
            else:
                self._history_cache.clear()

    def _release_quota(self, reservation_id: Optional[str]) -> None:
        """Hand a reservation back unused. Idempotent and never raises."""
        if not reservation_id:
            return
        context = get_llm_request_context()
        releaser = context.quota_releaser if context else None
        if releaser is None:
            return
        try:
            releaser(reservation_id)
        except Exception as exc:
            # The reservation expires on its own, so a failure here costs the
            # user some headroom for the TTL but must not break their request.
            logger.warning("[usage] Failed to release quota reservation: %s", redact_secrets(exc))

    def _record_usage(
        self, provider: str, model: str, result: Any, reservation_id: Optional[str] = None
    ) -> None:
        """Record provider-reported token usage, if a request recorder is installed.

        Settles `reservation_id` in the same step. Every path out of here either
        settles or releases it — a reservation that is silently dropped keeps
        holding the caller's own headroom until it expires.
        """
        context = get_llm_request_context()
        recorder = context.usage_recorder if context else None
        usage = getattr(result, "usage", None)
        if recorder is None or usage is None:
            self._release_quota(reservation_id)
            return
        try:
            recorder(
                provider=provider,
                model=model,
                key_source=self._llm._resolve_key_source(provider),
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                reservation_id=reservation_id,
            )
        except Exception as exc:
            logger.warning("[usage] Failed to record LLM usage: %s", redact_secrets(exc))
            self._release_quota(reservation_id)

    def _byok_call(self, provider: str) -> bool:
        """True when the CURRENT call to ``provider`` rides a user-supplied
        (BYOK) key — a client-scoped or request-context key, or an explicit
        key-source declaration. If the key source can't be determined (e.g.
        a stubbed client in tests), assume platform."""
        try:
            if self._llm._resolve_context_api_key(provider):
                return True
            return self._llm._resolve_key_source(provider) == "byok"
        except Exception:
            return False

    def _record_provider_health(
        self, provider: str, model: str, error: Optional[BaseException] = None
    ) -> None:
        """Feed the process-shared HealthMonitor from THE one choke point every
        LLM call flows through (RE-A2: record_success/record_failure were never
        called, so is_healthy() was always True and failover was dead code).

        ``error=None`` records a success. Failures are recorded per PROVIDER
        (the monitor's key and FALLBACK_MODELS' key) with the model noted in
        the error text; cancellations, quota trips, and local client-side
        errors are exempt (see core.health_monitor.health_failure_exempt).

        CX-03 key fairness: failures count against GLOBAL provider health only
        for PLATFORM-key calls. A failure on a user's own (BYOK) key is that
        user's key problem — one bad/rate-limited user key must never flip the
        provider unhealthy for everyone else. Successes record for both key
        sources (a success is evidence the provider is up regardless of whose
        key proved it). Never raises — health accounting must not break a
        live call.
        """
        try:
            from core.health_monitor import get_health_monitor, health_failure_exempt

            monitor = get_health_monitor()
            if error is None:
                monitor.record_success(provider)
            elif not health_failure_exempt(error) and not self._byok_call(provider):
                monitor.record_failure(
                    provider,
                    f"{model}: {type(error).__name__}: {redact_secrets(error)}",
                )
        except Exception:  # pragma: no cover - accounting must never bite
            logger.debug("[health] recording failed", exc_info=True)

    def _wrap_usage_stream(
        self, stream: Any, provider: str, model: str, reservation_id: Optional[str] = None
    ):
        """Wrap a stream and record usage from the completed response event.

        Also the streaming half of provider-health recording: only exceptions
        raised by the UNDERLYING stream (``next(it)``) count as provider
        failures — the consumer's own ``close()``/``throw()`` surfaces at the
        ``yield`` below, so a user cancellation is never miscounted.
        """
        last_response = None
        it = iter(stream)
        try:
            while True:
                try:
                    event = next(it)
                except StopIteration:
                    # Clean exhaustion — the provider delivered a whole stream.
                    self._record_provider_health(provider, model)
                    break
                except Exception as exc:
                    # Provider-side stream death (mid-stream timeout, reset,
                    # RemoteProtocolError...) — record ill-health unless exempt.
                    self._record_provider_health(provider, model, error=exc)
                    raise
                response = getattr(event, "response", None)
                if response is not None and getattr(response, "usage", None):
                    last_response = response
                yield event
        finally:
            # Runs on exhaustion, on close(), and when the consumer throws, so a
            # stream abandoned mid-flight still settles rather than leaking.
            if last_response is not None:
                self._record_usage(provider, model, last_response, reservation_id)
            else:
                self._release_quota(reservation_id)

    def create(self, **kwargs) -> Any:
        """
        Drop-in replacement for OpenAI's `client.responses.create(...)`.

        Supported kwargs (translated per provider):
          - model: str
          - input: str | list  (user message or tool results)
          - instructions: str  (system prompt)
          - temperature: float
          - max_output_tokens: int
          - tools: list        (function tool definitions)
          - stream: bool
          - text: dict         (e.g. {"format": {"type": "json_object"}})
          - previous_response_id: str  (conversation continuity — OpenAI only)
        """
        import time as _time

        model = kwargs.get("model", self._llm.default_model)
        provider = detect_provider(model)
        stream = kwargs.get("stream", False)
        attachments = kwargs.pop("attachments", None)
        user_id = kwargs.pop("user_id", None)
        session_id = kwargs.pop("session_id", None) or kwargs.pop("conversation_id", None)
        context = get_llm_request_context()
        # Admission for THIS call. The checker may hold a reservation against the
        # user's remaining allowance, which every path below must settle (via
        # _record_usage) or release.
        reservation_id = None
        if context and context.quota_checker:
            reservation_id = context.quota_checker(
                provider=provider,
                model=model,
                key_source=self._llm._resolve_key_source(provider),
            )

        # ── Langfuse: create a generation span if a parent trace exists ──
        lf_gen = None
        lf_parent = get_langfuse_parent()
        lf_client = get_langfuse()
        if lf_parent or lf_client:
            try:
                parent = lf_parent or (lf_client.trace(
                    name="llm_call",
                    user_id=user_id or "anonymous",
                    session_id=session_id,
                ) if lf_client else None)
                if parent:
                    from core.langfuse_integration import langfuse_generation
                    lf_gen = langfuse_generation(
                        parent,
                        name=f"{provider}/{model}",
                        model=model,
                        input_data=kwargs.get("input", ""),
                        metadata={"provider": provider, "stream": stream},
                    )
            except Exception:
                pass  # Never let tracing break the LLM call

        t0 = _time.perf_counter()

        try:
            # Runner-only flag: escalate tool_choice="required" from an emulated
            # nudge to server-enforced decoding (TACC) — set on re-samples after a
            # reasoning-only stop. Never forwarded to a provider SDK.
            strict_required = bool(kwargs.pop("tool_choice_strict", False))
            if provider == "openai":
                result = self._call_openai(kwargs, attachments=attachments)
            elif provider == "tacc":
                if stream:
                    kwargs["_tool_choice_strict"] = strict_required  # read by the TACC shims only
                    result = self._stream_tacc(kwargs, attachments=attachments)
                else:
                    kwargs["_tool_choice_strict"] = strict_required
                    result = self._call_tacc(kwargs, attachments=attachments)
            elif provider == "deepseek":
                if stream:
                    result = self._stream_deepseek(kwargs, attachments=attachments)
                else:
                    result = self._call_deepseek(kwargs, attachments=attachments)
            elif provider == "anthropic":
                if stream:
                    result = self._stream_anthropic(kwargs, attachments=attachments)
                else:
                    result = self._call_anthropic(kwargs, attachments=attachments)
            elif provider == "google":
                if stream:
                    result = self._stream_google(kwargs, attachments=attachments)
                else:
                    result = self._call_google(kwargs, attachments=attachments)
            elif provider == "local":
                if attachments:
                    raise ValueError(
                        "Document attachments are not supported for local models in this path."
                    )
                if stream:
                    result = self._stream_local(kwargs)
                else:
                    result = self._call_local(kwargs)
            else:
                raise ValueError(f"Unknown provider for model: {model}")

            # ── Langfuse: end generation with output/usage ──
            if lf_gen:
                if not stream:
                    elapsed_ms = (_time.perf_counter() - t0) * 1000
                    try:
                        output_text = getattr(result, 'output_text', '') or ''
                        usage = _build_langfuse_usage(result, provider)
                        lf_gen.end(
                            output=output_text[:2000],
                            usage=usage if usage else None,
                            metadata={"latency_ms": round(elapsed_ms)},
                        )
                    except Exception:
                        pass
                else:
                    # For streaming, wrap the generator/iterator to end the Langfuse generation when it completes!
                    def wrap_generator(gen, gen_span, start_time, _provider=provider):
                        accumulated_text = []
                        last_usage_obj = None
                        try:
                            for event in gen:
                                yield event
                                try:
                                    event_type = getattr(event, 'type', '')
                                    delta = getattr(event, 'delta', '') or ''
                                    if delta and (
                                        "delta" in event_type 
                                        or event_type in ("response.output_text.delta", "response.reasoning_summary_text.delta", "text_delta")
                                    ):
                                        accumulated_text.append(delta)
                                    
                                    # Extract usage from completed/done event response
                                    resp = getattr(event, 'response', None)
                                    if resp and hasattr(resp, 'usage') and resp.usage:
                                        last_usage_obj = resp
                                except Exception:
                                    pass
                        except Exception as e_stream:
                            try:
                                gen_span.update(metadata={"error": redact_secrets(e_stream)[:500]})
                            except Exception:
                                pass
                            raise e_stream
                        finally:
                            elapsed_ms = (_time.perf_counter() - start_time) * 1000
                            try:
                                full_output = "".join(accumulated_text)
                                usage = _build_langfuse_usage(last_usage_obj, _provider) if last_usage_obj else None
                                gen_span.end(
                                    output=full_output[:2000],
                                    usage=usage,
                                    metadata={"latency_ms": round(elapsed_ms), "streamed": True},
                                )
                            except Exception:
                                pass

                    result = wrap_generator(result, lf_gen, t0)

            if stream:
                # Ownership of the reservation passes to the generator, which
                # settles or releases it in its finally once consumed — health
                # for streams is recorded inside _wrap_usage_stream, where the
                # outcome is actually known.
                result = self._wrap_usage_stream(result, provider, model, reservation_id)
            else:
                self._record_usage(provider, model, result, reservation_id)
                self._record_provider_health(provider, model)

            return result

        except Exception as e:
            # The call never produced usage to settle against, so hand the
            # reservation back rather than making the user wait out its TTL.
            self._release_quota(reservation_id)
            # Health: covers non-streaming failures (after @with_retry gave up)
            # and eager stream-creation failures. Quota trips / cancellations /
            # local client errors are exempt inside the recorder.
            self._record_provider_health(provider, model, error=e)
            # ── Langfuse: record the error on the generation ──
            if lf_gen:
                try:
                    lf_gen.update(metadata={"error": redact_secrets(e)[:500]})
                    lf_gen.end(output=f"ERROR: {redact_secrets(e)}")
                except Exception:
                    pass
            raise

    # ── OpenAI (passthrough — native Responses API) ──────────────────────

    # Models that do NOT support the 'temperature' parameter
    _NO_TEMPERATURE_MODELS = frozenset({
        "o1", "o1-mini", "o1-pro",
        "o3", "o3-mini", "o3-pro",
        "o4-mini",
        "gpt-5-nano", "gpt-5-mini", "gpt-5.4-mini",
    })

    def _strip_unsupported_params(self, kwargs: dict) -> dict:
        """Remove params unsupported by certain OpenAI models (e.g. temperature)."""
        model = kwargs.get("model", "")
        if model in self._NO_TEMPERATURE_MODELS:
            kwargs = {k: v for k, v in kwargs.items() if k != "temperature"}
        return kwargs

    @with_retry(max_retries=3, backoff_base=1.0)
    def _call_openai(self, kwargs: dict, attachments: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Direct passthrough to OpenAI Responses API."""
        client = self._llm._get_openai_client()
        kwargs = self._strip_unsupported_params(kwargs)
        if attachments:
            kwargs = dict(kwargs)
            kwargs["input"] = self._build_openai_input(kwargs.get("input", ""), attachments)
        return client.responses.create(**kwargs)

    def _build_openai_input(
        self,
        input_data,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ):
        """Build a Responses API input payload with native input_file items."""
        if isinstance(input_data, list):
            return input_data

        content = []
        prompt_text = input_data if isinstance(input_data, str) else json.dumps(input_data, default=str)
        if prompt_text:
            content.append({"type": "input_text", "text": prompt_text})

        for attachment in attachments or []:
            if "image_url" in attachment or attachment.get("type") == "image_url":
                image_payload = attachment.get("image_url") or attachment
                if isinstance(image_payload, dict):
                    image_url = image_payload.get("url") or image_payload.get("image_url")
                    detail = image_payload.get("detail") or attachment.get("detail") or "auto"
                else:
                    image_url = str(image_payload)
                    detail = attachment.get("detail") or "auto"
                if not image_url:
                    raise ValueError("OpenAI image attachment is missing image_url.")
                content.append({
                    "type": "input_image",
                    "image_url": image_url,
                    "detail": detail,
                })
                continue
            if attachment.get("kind") != "openai_input_file":
                continue
            file_id = attachment.get("file_id")
            if not file_id:
                raise ValueError("OpenAI attachment is missing file_id.")
            content.append({"type": "input_file", "file_id": file_id})

        if not content:
            content.append({"type": "input_text", "text": ""})

        return [{"role": "user", "content": content}]

    # ── Anthropic (Claude) ───────────────────────────────────────────────

    # Claude Opus 4.7+, Sonnet 5, Fable 5, and Mythos 5 removed the sampling
    # parameters — sending temperature / top_p / top_k returns a 400
    # invalid_request_error. Older Claude models (Haiku 4.5, Opus 4.6 /
    # Sonnet 4.6 and earlier) still accept them, so strip only for the
    # newer families rather than dropping temperature everywhere.
    _ANTHROPIC_NO_SAMPLING_PREFIXES = (
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-mythos-5",
    )

    _ANTHROPIC_SAMPLING_PARAMS = ("temperature", "top_p", "top_k")

    # A static prefix list cannot stay correct now that the BYOK catalog
    # surfaces every model a user's key can see — a new Claude family reaches
    # the picker the day it ships, long before this tuple is updated. So the
    # list is only a fast path; the authority is the provider's own rejection,
    # learned once per process and applied to every later call for that model.
    _ANTHROPIC_NO_SAMPLING_LEARNED: set[str] = set()

    @classmethod
    def _anthropic_accepts_sampling(cls, model: str) -> bool:
        """Return False for Claude models that reject temperature/top_p/top_k."""
        model_id = (model or "").strip().lower()
        if model_id in cls._ANTHROPIC_NO_SAMPLING_LEARNED:
            return False
        return not any(model_id.startswith(p) for p in cls._ANTHROPIC_NO_SAMPLING_PREFIXES)

    @classmethod
    def _strip_sampling_after_rejection(cls, call_kwargs: dict, exc: Exception) -> bool:
        """Drop sampling params when `exc` is the provider rejecting them.

        Returns True when the call is worth retrying. Matched on the provider's
        400 body (e.g. "`temperature` is deprecated for this model.") rather
        than an exception class, because `anthropic` is imported lazily here.
        """
        if getattr(exc, "status_code", None) != 400:
            return False
        message = str(exc).lower()
        if not any(param in message for param in cls._ANTHROPIC_SAMPLING_PARAMS):
            return False
        if not any(word in message for word in ("deprecated", "unsupported", "not supported", "unexpected")):
            return False
        removed = [p for p in cls._ANTHROPIC_SAMPLING_PARAMS if call_kwargs.pop(p, None) is not None]
        if not removed:
            return False
        model_id = str(call_kwargs.get("model", "")).strip().lower()
        if model_id:
            cls._ANTHROPIC_NO_SAMPLING_LEARNED.add(model_id)
        logger.info(
            "Anthropic model %s rejects %s; retrying without them and remembering for this process.",
            model_id or "<unknown>", ", ".join(removed),
        )
        return True

    # Referencing an uploaded file (Files API) in a document block requires this
    # beta header on the messages request as well as on the upload.
    _ANTHROPIC_FILES_BETA_HEADER = {"anthropic-beta": "files-api-2025-04-14"}

    @staticmethod
    def _anthropic_has_document(attachments) -> bool:
        return any(
            isinstance(a, dict) and a.get("kind") == "anthropic_document_file"
            for a in (attachments or [])
        )

    def _anthropic_messages_for_input(self, prev_id, input_data, attachments=None) -> list:
        """Resolve the Anthropic message list, replaying cached history.

        Mirrors _chat_messages_for_input: the Anthropic Messages API is
        stateless and rejects a tool_result turn that is not preceded by the
        assistant tool_use block it answers (HTTP 400). So on a continuation
        round we start from the cached prior turns (user -> assistant tool_use)
        and append the new turn (tool_result blocks, or a follow-up user
        message). On a broken chain (a deadline kill cleared the cache mid-turn)
        an orphaned tool_result would still 400, so fold the tool outputs into a
        plain user message instead.
        """
        with self._history_lock:
            cached = list(self._history_cache.get(prev_id, [])) if prev_id else []
        if cached:
            cached.extend(self._build_anthropic_messages(input_data, attachments=attachments))
            return cached
        is_tool_results = isinstance(input_data, list) and any(
            isinstance(i, dict) and i.get("type") == "function_call_output" for i in input_data
        )
        if prev_id and is_tool_results:
            parts = [
                str(i.get("output", ""))[:4000]
                for i in input_data
                if isinstance(i, dict) and i.get("type") == "function_call_output"
            ]
            print(
                f"[PROVIDER] anthropic history chain broken for {prev_id} — converting "
                f"{len(parts)} orphaned tool output(s) into a user message"
            )
            recovery_text = (
                "[SYSTEM NOTE] The tool-call history was reset mid-turn. These are "
                "the results of the tool calls you just made:\n\n"
                + "\n\n---\n\n".join(parts)
                + "\n\nContinue the user's request from these results."
            )
            return self._build_anthropic_messages(recovery_text, attachments=attachments)
        return self._build_anthropic_messages(input_data, attachments=attachments)

    def _cache_anthropic_turn(self, response_id: str, messages: list, assistant_content) -> None:
        """Store the conversation so the next round can replay tool_use blocks."""
        if not response_id or assistant_content is None:
            return
        new_messages = list(messages)
        new_messages.append({"role": "assistant", "content": assistant_content})
        with self._history_lock:
            self._history_cache[response_id] = new_messages

    @with_retry(max_retries=3, backoff_base=1.0)
    def _call_anthropic(
        self,
        kwargs: dict,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Translate responses.create() to Anthropic Messages API."""
        client = self._llm._get_anthropic_client()
        model = kwargs.get("model", self._llm.default_model)
        instructions = kwargs.get("instructions", "")
        input_data = kwargs.get("input", "")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_output_tokens", 2000)
        tools_raw = kwargs.get("tools", None)
        json_mode = False
        text_opt = kwargs.get("text", None)
        if text_opt and isinstance(text_opt, dict):
            fmt = text_opt.get("format", {})
            if fmt.get("type") == "json_object":
                json_mode = True

        # Build messages, replaying cached history so tool_result turns follow
        # the assistant tool_use blocks that produced them (Anthropic 400s on an
        # orphaned tool_result).
        prev_id = kwargs.get("previous_response_id", None)
        messages = self._anthropic_messages_for_input(prev_id, input_data, attachments=attachments)

        # Translate tool schemas
        anthropic_tools = None
        if tools_raw:
            anthropic_tools = self._translate_tools_for_anthropic(tools_raw)

        # System prompt
        system_text = instructions or ""
        if json_mode:
            system_text += "\n\nYou MUST respond with valid JSON only. No extra text."

        call_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if self._anthropic_accepts_sampling(model):
            call_kwargs["temperature"] = temperature
        if system_text:
            call_kwargs["system"] = system_text
        if anthropic_tools:
            call_kwargs["tools"] = anthropic_tools
            # A forced-final round keeps the schemas (the Messages API rejects
            # tool_use/tool_result history without `tools`) and disables new
            # calls via tool_choice none.
            if kwargs.get("tool_choice") == "none":
                call_kwargs["tool_choice"] = {"type": "none"}
        if self._anthropic_has_document(attachments):
            call_kwargs["extra_headers"] = dict(self._ANTHROPIC_FILES_BETA_HEADER)

        try:
            resp = client.messages.create(**call_kwargs)
        except Exception as exc:
            if not self._strip_sampling_after_rejection(call_kwargs, exc):
                raise
            resp = client.messages.create(**call_kwargs)

        # Convert to LLMResponse and cache the turn for the next round.
        result = self._anthropic_to_llm_response(resp)
        self._cache_anthropic_turn(result.id, messages, getattr(resp, "content", None))
        return result

    def _stream_anthropic(
        self,
        kwargs: dict,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ):
        """Streaming Anthropic call — returns an iterator of StreamEvents."""
        client = self._llm._get_anthropic_client()
        model = kwargs.get("model", self._llm.default_model)
        instructions = kwargs.get("instructions", "")
        input_data = kwargs.get("input", "")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_output_tokens", 2000)
        tools_raw = kwargs.get("tools", None)

        prev_id = kwargs.get("previous_response_id", None)
        messages = self._anthropic_messages_for_input(prev_id, input_data, attachments=attachments)
        anthropic_tools = self._translate_tools_for_anthropic(tools_raw) if tools_raw else None

        call_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if self._anthropic_accepts_sampling(model):
            call_kwargs["temperature"] = temperature
        if instructions:
            call_kwargs["system"] = instructions
        if anthropic_tools:
            call_kwargs["tools"] = anthropic_tools
            # Forced-final round: keep schemas (tool_use/tool_result history
            # requires `tools`), forbid new calls.
            if kwargs.get("tool_choice") == "none":
                call_kwargs["tool_choice"] = {"type": "none"}
        if self._anthropic_has_document(attachments):
            call_kwargs["extra_headers"] = dict(self._ANTHROPIC_FILES_BETA_HEADER)

        # Return a generator that yields StreamEvent objects
        return self._anthropic_stream_generator(client, call_kwargs)

    @contextmanager
    def _anthropic_stream_cm(self, client, call_kwargs):
        """Open a Messages stream, retrying once without sampling params.

        The rejection is raised when the stream is opened, before any event is
        produced. `opened` guards the retry so a failure *during* iteration is
        never replayed — that would duplicate already-streamed output.
        """
        opened = False
        try:
            with client.messages.stream(**call_kwargs) as stream:
                opened = True
                yield stream
            return
        except Exception as exc:
            if opened or not self._strip_sampling_after_rejection(call_kwargs, exc):
                raise
        with client.messages.stream(**call_kwargs) as stream:
            yield stream

    @staticmethod
    def _log_served_model(requested, event) -> None:
        """Record the model Anthropic reports for a turn, warning on a mismatch.

        An alias legitimately resolves to a dated snapshot
        (claude-haiku-4-5 -> claude-haiku-4-5-20251001), so only a served model
        that is not the requested one nor a snapshot of it is worth a warning.
        """
        served = getattr(getattr(event, "message", None), "model", None)
        if not served or not requested:
            return
        if served == requested or served.startswith(f"{requested}-"):
            logger.info("Anthropic served model=%s (requested %s)", served, requested)
            return
        logger.warning("Anthropic served model=%s but %s was requested.", served, requested)

    def _anthropic_stream_generator(self, client, call_kwargs):
        """Generate StreamEvent objects from Anthropic streaming."""
        resp_id = f"resp_{uuid.uuid4().hex[:16]}"

        # Emit response.created
        yield StreamEvent(
            type="response.created",
            response=LLMResponse(id=resp_id),
        )

        function_calls = {}  # index -> {name, arguments}
        current_tool_index = None

        with self._anthropic_stream_cm(client, call_kwargs) as stream:
            for event in stream:
                event_type = getattr(event, 'type', '')

                if event_type == 'message_start':
                    # The provider echoes the model that actually served the
                    # turn. Logged so a silent substitution (or an alias
                    # resolving elsewhere) is visible rather than assumed.
                    self._log_served_model(call_kwargs.get("model"), event)

                if event_type == 'content_block_start':
                    block = getattr(event, 'content_block', None)
                    if block and getattr(block, 'type', '') == 'tool_use':
                        idx = getattr(event, 'index', 0)
                        current_tool_index = idx
                        fc = FunctionCallItem(
                            name=block.name,
                            call_id=block.id or f"call_{uuid.uuid4().hex[:16]}",
                        )
                        function_calls[idx] = fc
                        yield StreamEvent(
                            type="response.output_item.added",
                            item=fc,
                        )

                elif event_type == 'content_block_delta':
                    delta = getattr(event, 'delta', None)
                    if delta:
                        delta_type = getattr(delta, 'type', '')
                        if delta_type == 'text_delta':
                            yield StreamEvent(
                                type="response.output_text.delta",
                                delta=delta.text,
                            )
                        elif delta_type == 'input_json_delta':
                            idx = getattr(event, 'index', current_tool_index or 0)
                            fc = function_calls.get(idx)
                            if fc:
                                fc.arguments += delta.partial_json
                                yield StreamEvent(
                                    type="response.function_call_arguments.delta",
                                    delta=delta.partial_json,
                                    item=fc,
                                )

        # Extract final message usage
        usage_obj = None
        final_msg = None
        try:
            final_msg = stream.get_final_message()
            if final_msg and hasattr(final_msg, 'usage') and final_msg.usage:
                usage_obj = LLMUsage(
                    input_tokens=getattr(final_msg.usage, 'input_tokens', 0),
                    output_tokens=getattr(final_msg.usage, 'output_tokens', 0),
                )
        except Exception:
            pass

        completed_items = [fc for fc in function_calls.values()]
        yield StreamEvent(
            type="response.completed",
            response=LLMResponse(
                id=resp_id,
                output=completed_items,
                usage=usage_obj,
            )
        )

        # Cache the assistant turn (text + tool_use blocks) so a follow-up
        # tool_result / user round can replay it — otherwise Anthropic 400s on
        # an orphaned tool_result and multi-turn context is lost.
        try:
            if final_msg is not None:
                self._cache_anthropic_turn(
                    resp_id, call_kwargs.get("messages", []), getattr(final_msg, "content", None)
                )
        except Exception:
            pass

    def _build_anthropic_messages(self, input_data, attachments: Optional[List[Dict[str, Any]]] = None) -> list:
        """Convert responses.create() input to Anthropic messages format."""
        if isinstance(input_data, str):
            if attachments:
                content_blocks = []
                for attachment in attachments:
                    if attachment.get("kind") != "anthropic_document_file":
                        continue
                    file_id = attachment.get("file_id")
                    if not file_id:
                        raise ValueError("Anthropic attachment is missing file_id.")
                    content_blocks.append({
                        "type": "document",
                        "title": attachment.get("filename") or "Attached document",
                        "source": {
                            "type": "file",
                            "file_id": file_id,
                        },
                    })
                if input_data:
                    content_blocks.append({"type": "text", "text": input_data})
                if not content_blocks:
                    content_blocks.append({"type": "text", "text": input_data or ""})
                return [{"role": "user", "content": content_blocks}]
            return [{"role": "user", "content": input_data}]
        elif isinstance(input_data, list):
            # Tool results format: [{"type": "function_call_output", "call_id": ..., "output": ...}]
            messages = []
            # Build assistant message with tool_use blocks (required by Anthropic)
            tool_use_blocks = []
            tool_results = []
            for item in input_data:
                if isinstance(item, dict) and item.get("type") == "function_call_output":
                    call_id = item.get("call_id", "")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": item.get("output", ""),
                    })
                elif isinstance(item, dict) and item.get("role") == "user":
                    tool_results.append({"type": "text", "text": item.get("content", "")})
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            return messages
        return [{"role": "user", "content": str(input_data)}]

    def _translate_tools_for_anthropic(self, tools: list) -> list:
        """Convert OpenAI tool format to Anthropic tool format."""
        anthropic_tools = []
        for t in tools:
            if t.get("type") == "function" or "name" in t:
                anthropic_tools.append({
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
                })
        return anthropic_tools

    def _anthropic_to_llm_response(self, resp) -> LLMResponse:
        """Convert Anthropic response to LLMResponse."""
        output_text = ""
        output_items = []

        for block in resp.content:
            if block.type == "text":
                output_text += block.text
            elif block.type == "tool_use":
                fc = FunctionCallItem(
                    name=block.name,
                    arguments=json.dumps(block.input) if isinstance(block.input, dict) else str(block.input),
                    call_id=block.id,
                )
                output_items.append(fc)

        usage = None
        if hasattr(resp, 'usage') and resp.usage:
            usage = LLMUsage(
                input_tokens=getattr(resp.usage, 'input_tokens', 0),
                output_tokens=getattr(resp.usage, 'output_tokens', 0),
            )

        return LLMResponse(
            output_text=output_text,
            id=resp.id if hasattr(resp, 'id') else f"resp_{uuid.uuid4().hex[:16]}",
            output=output_items or [MessageOutputItem(content=[TextContentItem(text=output_text)])],
            usage=usage,
        )

    # ── Google Gemini ────────────────────────────────────────────────────

    @with_retry(max_retries=3, backoff_base=1.0)
    def _call_google(
        self,
        kwargs: dict,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Translate responses.create() to Google GenAI API."""
        client = self._llm._get_google_client()
        model = kwargs.get("model", self._llm.default_model)
        instructions = kwargs.get("instructions", "")
        input_data = kwargs.get("input", "")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_output_tokens", 2000)
        tools_raw = kwargs.get("tools", None)
        json_mode = False
        text_opt = kwargs.get("text", None)
        if text_opt and isinstance(text_opt, dict):
            fmt = text_opt.get("format", {})
            if fmt.get("type") == "json_object":
                json_mode = True

        # Build content
        user_text = input_data if isinstance(input_data, str) else json.dumps(input_data, default=str)
        full_prompt = f"{instructions}\n\n{user_text}" if instructions else user_text
        if json_mode:
            full_prompt += "\n\nRespond with valid JSON only."

        # Translate tools for Gemini
        gemini_tools = None
        if tools_raw:
            gemini_tools = self._translate_tools_for_google(tools_raw)

        gen_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if json_mode:
            gen_config["response_mime_type"] = "application/json"

        call_kwargs = {
            "model": model,
            "contents": self._build_google_contents(full_prompt, attachments=attachments),
            "config": gen_config,
        }
        if gemini_tools:
            call_kwargs["config"]["tools"] = gemini_tools
        if kwargs.get("tool_choice") == "none":
            call_kwargs["config"]["tool_config"] = {"function_calling_config": {"mode": "NONE"}}

        resp = client.models.generate_content(**call_kwargs)

        return self._google_to_llm_response(resp)

    def _stream_google(
        self,
        kwargs: dict,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ):
        """Streaming Google GenAI call — returns an iterator of StreamEvents."""
        client = self._llm._get_google_client()
        model = kwargs.get("model", self._llm.default_model)
        instructions = kwargs.get("instructions", "")
        input_data = kwargs.get("input", "")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_output_tokens", 2000)
        tools_raw = kwargs.get("tools", None)

        user_text = input_data if isinstance(input_data, str) else json.dumps(input_data, default=str)
        full_prompt = f"{instructions}\n\n{user_text}" if instructions else user_text

        gemini_tools = self._translate_tools_for_google(tools_raw) if tools_raw else None

        gen_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        call_kwargs = {
            "model": model,
            "contents": self._build_google_contents(full_prompt, attachments=attachments),
            "config": gen_config,
        }
        if gemini_tools:
            call_kwargs["config"]["tools"] = gemini_tools
        if kwargs.get("tool_choice") == "none":
            call_kwargs["config"]["tool_config"] = {"function_calling_config": {"mode": "NONE"}}

        # Emit response.created
        resp_id = f"resp_{uuid.uuid4().hex[:16]}"
        yield StreamEvent(type="response.created", response=LLMResponse(id=resp_id))

        usage_obj = None
        function_calls = []

        stream = client.models.generate_content_stream(**call_kwargs)
        for chunk in stream:
            if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                usage_obj = LLMUsage(
                    input_tokens=getattr(chunk.usage_metadata, 'prompt_token_count', 0),
                    output_tokens=getattr(chunk.usage_metadata, 'candidates_token_count', 0),
                )

            text = chunk.text if hasattr(chunk, "text") and chunk.text else ""
            if text:
                yield StreamEvent(type="response.output_text.delta", delta=text)

            # Handle function calls in streaming
            if hasattr(chunk, 'candidates'):
                for candidate in chunk.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                fc = FunctionCallItem(
                                    name=part.function_call.name,
                                    arguments=json.dumps(dict(part.function_call.args)) if part.function_call.args else "{}",
                                )
                                function_calls.append(fc)
                                yield StreamEvent(type="response.output_item.added", item=fc)

        yield StreamEvent(
            type="response.completed",
            response=LLMResponse(
                id=resp_id,
                output=function_calls,
                usage=usage_obj,
            )
        )

    def _translate_tools_for_google(self, tools: list) -> list:
        """Convert OpenAI tool format to Google GenAI function declarations."""
        from google.genai import types as genai_types

        function_declarations = []
        for t in tools:
            if t.get("type") == "function" or "name" in t:
                params = t.get("parameters", {})
                function_declarations.append(
                    genai_types.FunctionDeclaration(
                        name=t.get("name", ""),
                        description=t.get("description", ""),
                        parameters=params,
                    )
                )
        if function_declarations:
            return [genai_types.Tool(function_declarations=function_declarations)]
        return []

    def _google_to_llm_response(self, resp) -> LLMResponse:
        """Convert Google GenAI response to LLMResponse."""
        output_text = ""
        output_items = []

        if hasattr(resp, 'text') and resp.text:
            output_text = resp.text

        # Handle function calls
        if hasattr(resp, 'candidates'):
            for candidate in resp.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            if not output_text:
                                output_text = part.text
                        if hasattr(part, 'function_call') and part.function_call:
                            fc = FunctionCallItem(
                                name=part.function_call.name,
                                arguments=json.dumps(dict(part.function_call.args)) if part.function_call.args else "{}",
                            )
                            output_items.append(fc)

        usage = None
        if hasattr(resp, 'usage_metadata') and resp.usage_metadata:
            usage = LLMUsage(
                input_tokens=getattr(resp.usage_metadata, 'prompt_token_count', 0),
                output_tokens=getattr(resp.usage_metadata, 'candidates_token_count', 0),
            )

        return LLMResponse(
            output_text=output_text,
            output=output_items or [MessageOutputItem(content=[TextContentItem(text=output_text)])],
            usage=usage,
        )

    def _build_google_contents(
        self,
        prompt_text: str,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ):
        """Build Gemini contents with file_data parts for uploaded documents."""
        if not attachments:
            return prompt_text

        parts = []
        for attachment in attachments:
            if attachment.get("kind") != "gemini_file":
                continue
            file_uri = attachment.get("file_uri")
            mime_type = attachment.get("mime_type")
            if not file_uri or not mime_type:
                raise ValueError("Gemini attachment is missing file_uri or mime_type.")
            parts.append({
                "file_data": {
                    "mime_type": mime_type,
                    "file_uri": file_uri,
                }
            })

        if prompt_text:
            parts.append({"text": prompt_text})

        if not parts:
            return prompt_text

        return [{"role": "user", "parts": parts}]

    # ── Local LLM (Ollama / LM Studio via OpenAI-compat API) ────────────

    @with_retry(max_retries=3, backoff_base=1.0)
    def _call_local(self, kwargs: dict) -> LLMResponse:
        """Translate responses.create() to OpenAI Chat Completions (local)."""
        client = self._llm._get_local_client()
        model = kwargs.get("model", self._llm.default_model)
        actual_model = model.removeprefix("local/")

        instructions = kwargs.get("instructions", "")
        input_data = kwargs.get("input", "")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_output_tokens", 2000)
        tools_raw = kwargs.get("tools", None)
        json_mode = False
        text_opt = kwargs.get("text", None)
        if text_opt and isinstance(text_opt, dict):
            fmt = text_opt.get("format", {})
            if fmt.get("type") == "json_object":
                json_mode = True

        # Build messages
        messages = self._build_chat_messages(instructions, input_data, json_mode)

        # Translate tools
        openai_tools = None
        if tools_raw:
            openai_tools = self._translate_tools_for_chat_completions(tools_raw)

        call_kwargs = {
            "model": actual_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if openai_tools:
            call_kwargs["tools"] = openai_tools
        if json_mode:
            call_kwargs["response_format"] = {"type": "json_object"}

        completions_engine = getattr(getattr(client, "chat"), "completions")
        if kwargs.get("tool_choice") == "none":
            call_kwargs["tool_choice"] = "none"
        resp = completions_engine.create(**call_kwargs)
        return self._chat_completion_to_llm_response(resp)

    def _stream_local(self, kwargs: dict):
        """Streaming local LLM call — returns an iterator of StreamEvents."""
        client = self._llm._get_local_client()
        model = kwargs.get("model", self._llm.default_model)
        actual_model = model.removeprefix("local/")

        instructions = kwargs.get("instructions", "")
        input_data = kwargs.get("input", "")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_output_tokens", 2000)
        tools_raw = kwargs.get("tools", None)

        messages = self._build_chat_messages(instructions, input_data)
        openai_tools = self._translate_tools_for_chat_completions(tools_raw) if tools_raw else None

        call_kwargs = {
            "model": actual_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if openai_tools:
            call_kwargs["tools"] = openai_tools

        # Emit response.created
        resp_id = f"resp_{uuid.uuid4().hex[:16]}"
        yield StreamEvent(type="response.created", response=LLMResponse(id=resp_id))

        # Track function calls across chunks
        function_calls = {}  # index -> FunctionCallItem
        usage_obj = None

        completions_engine = getattr(getattr(client, "chat"), "completions")
        if kwargs.get("tool_choice") == "none":
            call_kwargs["tool_choice"] = "none"
        stream = completions_engine.create(**call_kwargs)
        for chunk in stream:
            if hasattr(chunk, 'usage') and chunk.usage:
                usage_obj = LLMUsage(
                    input_tokens=getattr(chunk.usage, 'prompt_tokens', 0),
                    output_tokens=getattr(chunk.usage, 'completion_tokens', 0),
                )

            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue

            delta = choice.delta

            # Text content
            if delta.content:
                yield StreamEvent(type="response.output_text.delta", delta=delta.content)

            # Tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in function_calls:
                        fc = FunctionCallItem(
                            name=tc.function.name or "",
                            call_id=tc.id or f"call_{uuid.uuid4().hex[:16]}",
                        )
                        function_calls[idx] = fc
                        yield StreamEvent(type="response.output_item.added", item=fc)

                    fc = function_calls[idx]
                    if tc.function.name:
                        fc.name = tc.function.name
                    if tc.function.arguments:
                        fc.arguments += tc.function.arguments
                        yield StreamEvent(
                            type="response.function_call_arguments.delta",
                            delta=tc.function.arguments,
                            item=fc,
                        )

        completed_items = [fc for fc in function_calls.values()]
        yield StreamEvent(
            type="response.completed",
            response=LLMResponse(
                id=resp_id,
                output=completed_items,
                usage=usage_obj,
            )
        )

    @with_retry(max_retries=3, backoff_base=1.0)
    def _call_tacc(self, kwargs: dict, attachments: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        """Translate responses.create() to TACC's OpenAI-compatible Chat Completions API."""
        client = self._llm._get_tacc_client()
        model = self._llm._normalize_tacc_model(kwargs.get("model", self._llm.default_model))
        instructions = self._configure_gpt_oss_instructions(
            kwargs.get("instructions", ""),
            model,
        )
        input_data = kwargs.get("input", "")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_output_tokens", 2000)
        tools_raw = kwargs.get("tools", None)
        prev_id = kwargs.get("previous_response_id", None)
        json_mode = False
        text_opt = kwargs.get("text", None)
        if text_opt and isinstance(text_opt, dict):
            fmt = text_opt.get("format", {})
            if fmt.get("type") == "json_object":
                json_mode = True

        messages = self._chat_messages_for_input(prev_id, instructions, input_data, json_mode)

        messages = self._inject_images_into_messages(messages, attachments)
        openai_tools = self._translate_tools_for_chat_completions(tools_raw) if tools_raw else None
        tool_choice = kwargs.get("tool_choice", None)

        call_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if openai_tools:
            call_kwargs["tools"] = openai_tools
            if tool_choice:
                call_kwargs["tool_choice"] = self._tacc_tool_choice(tool_choice, bool(kwargs.get("_tool_choice_strict")))
                if tool_choice == "required":
                    call_kwargs["messages"] = self._apply_required_tool_choice_nudge(call_kwargs["messages"])
        if json_mode and os.getenv("TACC_ENABLE_RESPONSE_FORMAT", "").lower() in {"1", "true", "yes"}:
            call_kwargs["response_format"] = {"type": "json_object"}

        completions_engine = getattr(getattr(client, "chat"), "completions")
        resp = self._tacc_create_with_tool_choice_fallback(completions_engine, call_kwargs)
        result = self._chat_completion_to_llm_response(resp)

        new_messages = list(messages)
        tool_calls = [out for out in result.output if getattr(out, "type", None) == "function_call"]
        if tool_calls:
            new_messages.append({
                "role": "assistant",
                "content": result.output_text or None,
                "tool_calls": [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })
        else:
            new_messages.append({"role": "assistant", "content": result.output_text})
        with self._history_lock:
            self._history_cache[result.id] = new_messages
        return result

    def _stream_tacc(self, kwargs: dict, attachments: Optional[List[Dict[str, Any]]] = None):
        """Streaming TACC call via OpenAI-compatible Chat Completions."""
        client = self._llm._get_tacc_client()
        model = self._llm._normalize_tacc_model(kwargs.get("model", self._llm.default_model))
        instructions = self._configure_gpt_oss_instructions(
            kwargs.get("instructions", ""),
            model,
        )
        input_data = kwargs.get("input", "")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_output_tokens", 2000)
        tools_raw = kwargs.get("tools", None)
        prev_id = kwargs.get("previous_response_id", None)

        messages = self._chat_messages_for_input(prev_id, instructions, input_data)

        messages = self._inject_images_into_messages(messages, attachments)
        openai_tools = self._translate_tools_for_chat_completions(tools_raw) if tools_raw else None
        tool_choice = kwargs.get("tool_choice", None)

        call_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if os.getenv("TACC_STREAM_INCLUDE_USAGE", "true").lower() in {"1", "true", "yes"}:
            call_kwargs["stream_options"] = {"include_usage": True}
        if openai_tools:
            call_kwargs["tools"] = openai_tools
            if tool_choice:
                call_kwargs["tool_choice"] = self._tacc_tool_choice(tool_choice, bool(kwargs.get("_tool_choice_strict")))
                if tool_choice == "required":
                    call_kwargs["messages"] = self._apply_required_tool_choice_nudge(call_kwargs["messages"])

        resp_id = f"resp_{uuid.uuid4().hex[:16]}"
        yield StreamEvent(type="response.created", response=LLMResponse(id=resp_id))

        st = self._new_tacc_stream_state()

        completions_engine = getattr(getattr(client, "chat"), "completions")
        stream = self._tacc_create_with_tool_choice_fallback(completions_engine, call_kwargs)
        try:
            yield from self._iter_tacc_stream(stream, st)
        except Exception as exc:
            # A server-ENFORCED tool call (tool_choice="required") that dies
            # mid-stream before producing any content or call is retried ONCE
            # with the emulated form (auto + the nudge already in the system
            # message), and "required" is cooled down for this process. Live
            # 2026-09-17: every one of the 10 provider-killed benchmark trials
            # died on exactly this call, seconds after the same prompt had
            # streamed fine under "auto"; the request-time 400 fallback above
            # never sees an in-band stream error. Reasoning deltas already
            # forwarded stay visible — they are commentary, not the answer.
            if (
                call_kwargs.get("tool_choice") == "required"
                and not st.function_calls
                and not st.output_text
                and _retry_is_retryable(exc)
            ):
                cooldown = self._tacc_required_cooldown_seconds()
                type(self)._tacc_required_cooldown_until = time.monotonic() + cooldown
                print(
                    f"[PROVIDER] tacc stream died under tool_choice=required "
                    f"({type(exc).__name__}: {str(exc)[:160]}) — retrying this round once with "
                    f"auto (+nudge); required is cooled down for {cooldown:.0f}s"
                )
                st = self._new_tacc_stream_state()
                stream = completions_engine.create(**{**call_kwargs, "tool_choice": "auto"})
                yield from self._iter_tacc_stream(stream, st)
            else:
                raise

        function_calls = st.function_calls
        reasoning_done_emitted = st.reasoning_done_emitted
        output_text = st.output_text
        reasoning_content = st.reasoning_content
        usage_obj = st.usage_obj
        finish_reason = st.finish_reason

        if not reasoning_done_emitted:
            yield StreamEvent(type="response.reasoning_summary_text.done")

        for fc in function_calls.values():
            yield StreamEvent(type="response.output_item.done", item=fc)

        completed_items = [fc for fc in function_calls.values()]
        if finish_reason:
            print(
                f"[PROVIDER] tacc stream finish_reason={finish_reason!r} "
                f"output_chars={len(output_text)} reasoning_chars={len(reasoning_content)} "
                f"tool_calls={len(completed_items)}"
                + (
                    f" harmony_tokens_stripped={st.harmony.dropped_tokens}"
                    if st.harmony.dropped_tokens
                    else ""
                )
            )
        yield StreamEvent(
            type="response.completed",
            response=LLMResponse(
                id=resp_id,
                output=completed_items,
                usage=usage_obj,
                finish_reason=finish_reason,
            ),
        )

        new_messages = list(messages)
        if completed_items:
            new_messages.append({
                "role": "assistant",
                "content": output_text or None,
                "tool_calls": [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in completed_items
                ],
            })
        else:
            new_messages.append({"role": "assistant", "content": output_text})
        with self._history_lock:
            self._history_cache[resp_id] = new_messages

    @staticmethod
    def _new_tacc_stream_state():
        """Mutable per-attempt accumulator for one TACC chat-completions stream."""
        from types import SimpleNamespace
        return SimpleNamespace(
            function_calls={},
            reasoning_done_emitted=False,
            channel_call_index={},  # id -> slot for index-less channel deltas
            output_text="",
            reasoning_content="",
            usage_obj=None,
            finish_reason=None,
            # Strips gpt-oss harmony control markup that leaks into `content`
            # (live 2026-09-21: the raw text of a tool call the same round also
            # issued structurally). output_text only ever holds the cleaned text.
            harmony=HarmonyStreamFilter(),
        )

    def _iter_tacc_stream(self, stream, st):
        """Translate one chat-completions chunk stream into Responses-style
        events, accumulating into ``st`` so a caller can restart the attempt."""
        for chunk in stream:
            if hasattr(chunk, "usage") and chunk.usage:
                st.usage_obj = LLMUsage(
                    input_tokens=getattr(chunk.usage, "prompt_tokens", 0),
                    output_tokens=getattr(chunk.usage, "completion_tokens", 0),
                )

            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue
            if getattr(choice, "finish_reason", None):
                st.finish_reason = choice.finish_reason

            delta = choice.delta
            function_calls = st.function_calls
            channel_call_index = st.channel_call_index

            # gpt-oss serves its chain of thought as reasoning deltas. Forward
            # them (same event type as DeepSeek) so a long-thinking round keeps
            # the SSE inactivity watchdog fed and the Thinking box live instead
            # of looking byte-dead for minutes. Content output is unchanged.
            reasoning_delta = (
                getattr(delta, "reasoning_content", None)
                or getattr(delta, "reasoning", None)
            )
            if isinstance(reasoning_delta, dict):
                reasoning_delta = reasoning_delta.get("content") or reasoning_delta.get("text")
            if isinstance(reasoning_delta, str) and reasoning_delta:
                st.reasoning_content += reasoning_delta
                yield StreamEvent(type="response.reasoning_summary_text.delta", delta=reasoning_delta)
                # No continue: content/tool_calls co-riding the same chunk
                # must still be processed below or they'd be silently lost.

            # Some proxies put structured calls under a channel object. Only
            # consume explicit tool_calls fields; never interpret reasoning prose.
            tool_deltas = getattr(delta, "tool_calls", None)
            if not tool_deltas:
                for channel in ("commentary", "reasoning", "reasoning_content"):
                    payload = getattr(delta, channel, None)
                    channel_calls = payload.get("tool_calls") if isinstance(payload, dict) else getattr(payload, "tool_calls", None)
                    if channel_calls:
                        tool_deltas = channel_calls
                        break
            if isinstance(tool_deltas, dict):
                tool_deltas = [tool_deltas]

            # Content may carry leaked harmony markup — the `<|channel|>commentary
            # to=functions.X <|constrain|>json<|message|>{...}` text of a tool call
            # (observed live 2026-09-21 alongside the structured tool_calls delta
            # of the same round). Only the cleaned text is user-visible; a leaked
            # `analysis` body is chain of thought and goes to the Thinking box.
            visible_text = ""
            if delta.content:
                visible_text, leaked_reasoning = st.harmony.feed(delta.content)
                if leaked_reasoning:
                    st.reasoning_content += leaked_reasoning
                    yield StreamEvent(type="response.reasoning_summary_text.delta", delta=leaked_reasoning)

            # If we were streaming reasoning but it has now stopped, emit done event
            if not st.reasoning_done_emitted:
                if visible_text or tool_deltas:
                    yield StreamEvent(type="response.reasoning_summary_text.done")
                    st.reasoning_done_emitted = True

            if visible_text:
                st.output_text += visible_text
                yield StreamEvent(type="response.output_text.delta", delta=visible_text)

            if tool_deltas:
                for tc in tool_deltas:
                    if isinstance(tc, dict):
                        from types import SimpleNamespace
                        function = tc.get("function") or {}
                        call_id = tc.get("id")
                        raw_index = tc.get("index")
                        if raw_index is None:
                            # No stream index: a known id continues its call, a new
                            # id opens the next slot, an id-less delta continues the
                            # most recent call (argument fragments).
                            if call_id and call_id in channel_call_index:
                                raw_index = channel_call_index[call_id]
                            elif call_id:
                                raw_index = max(function_calls.keys(), default=-1) + 1
                                channel_call_index[call_id] = raw_index
                            else:
                                raw_index = max(function_calls.keys(), default=0)
                        elif call_id:
                            channel_call_index.setdefault(call_id, raw_index)
                        tc = SimpleNamespace(index=raw_index, id=call_id,
                                             function=SimpleNamespace(name=function.get("name"),
                                                                      arguments=function.get("arguments")))
                    idx = tc.index
                    if idx not in function_calls:
                        fc = FunctionCallItem(
                            name=tc.function.name or "" if tc.function else "",
                            call_id=tc.id or f"call_{uuid.uuid4().hex[:16]}",
                        )
                        function_calls[idx] = fc
                        yield StreamEvent(type="response.output_item.added", item=fc)

                    fc = function_calls[idx]
                    if tc.function and tc.function.name and not fc.name:
                        fc.name = tc.function.name
                    if tc.function and tc.function.arguments:
                        fc.arguments += tc.function.arguments
                        yield StreamEvent(
                            type="response.function_call_arguments.delta",
                            delta=tc.function.arguments,
                            item=fc,
                        )

        # End of stream: release a held-back partial token / close an unterminated
        # harmony span (the leaked tool-call text ends at the JSON — its <|call|>
        # stop token never reaches `content`).
        visible_text, leaked_reasoning = st.harmony.flush()
        if leaked_reasoning:
            st.reasoning_content += leaked_reasoning
            yield StreamEvent(type="response.reasoning_summary_text.delta", delta=leaked_reasoning)
        if visible_text:
            if not st.reasoning_done_emitted:
                yield StreamEvent(type="response.reasoning_summary_text.done")
                st.reasoning_done_emitted = True
            st.output_text += visible_text
            yield StreamEvent(type="response.output_text.delta", delta=visible_text)
        if st.harmony.dropped_calls:
            structural = sorted({fc.name for fc in st.function_calls.values() if fc.name})
            for dropped in st.harmony.dropped_calls:
                name = dropped.get("name") or "?"
                verdict = (
                    "also issued structurally this round"
                    if name in structural
                    else f"NOT issued structurally this round (structural calls: {structural or 'none'})"
                )
                print(
                    f"[PROVIDER] tacc content leaked harmony tool-call markup for functions.{name} "
                    f"({len(dropped.get('payload', ''))} payload chars) — stripped from the visible text; "
                    f"{verdict}"
                )

    def _chat_messages_for_input(self, prev_id, instructions: str, input_data, json_mode: bool = False) -> list:
        """Resolve the message list for a chat-completions round.

        Normal path: extend the cached history for prev_id. Healing path: when
        the chain is broken (a deadline kill clears the history cache while the
        agent thread is still inside a long tool call — live DS-P8 2026-07-05),
        a tool-results input would otherwise become [system, tool, tool], which
        providers reject with 400 "role 'tool' must be a response to ...".
        Convert the orphaned tool outputs into a user message so the round
        stays valid and the model can continue from the results it produced.
        """
        with self._history_lock:
            cached = list(self._history_cache.get(prev_id, [])) if prev_id else []
        if cached:
            self._append_chat_input(cached, input_data)
            return cached
        is_tool_results = isinstance(input_data, list) and any(
            isinstance(i, dict) and i.get("type") == "function_call_output" for i in input_data
        )
        if prev_id and is_tool_results:
            parts = [
                str(i.get("output", ""))[:4000]
                for i in input_data
                if isinstance(i, dict) and i.get("type") == "function_call_output"
            ]
            print(
                f"[PROVIDER] history chain broken for {prev_id} — converting "
                f"{len(parts)} orphaned tool output(s) into a user message"
            )
            recovery_text = (
                "[SYSTEM NOTE] The tool-call history was reset mid-turn. These are "
                "the results of the tool calls you just made:\n\n"
                + "\n\n---\n\n".join(parts)
                + "\n\nContinue the user's request from these results."
            )
            return self._build_chat_messages(instructions, recovery_text, json_mode)
        return self._build_chat_messages(instructions, input_data, json_mode)

    def _build_chat_messages(self, instructions: str, input_data, json_mode: bool = False) -> list:
        """Build Chat Completions messages from responses.create() args."""
        messages = []

        if instructions:
            sys_content = instructions
            if json_mode:
                sys_content += "\n\nYou MUST respond with valid JSON only. No extra text."
            messages.append({"role": "system", "content": sys_content})

        if isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})
        elif isinstance(input_data, list):
            # Tool results: [{"type": "function_call_output", "call_id": ..., "output": ...}]
            # In Chat Completions, these are "tool" role messages
            for item in input_data:
                if isinstance(item, dict) and item.get("type") == "function_call_output":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": item.get("output", ""),
                    })
                elif isinstance(item, dict) and item.get("role") == "user":
                    messages.append({"role": "user", "content": item.get("content", "")})
        else:
            messages.append({"role": "user", "content": str(input_data)})

        return messages

    # GPT-OSS models occasionally emit arithmetic ("radius_deg": 10.0/60.0) or
    # comments inside tool-call argument JSON; the serving side then rejects the
    # whole turn with a 400 "Failed to parse tool call from GPT OSS output".
    _GPT_OSS_TOOL_JSON_RULE = (
        "Tool-call arguments MUST be strictly valid JSON: every number must be a plain "
        "literal (NEVER arithmetic such as 10.0/60.0 — compute the value yourself and "
        "write 0.1667), with no comments, no trailing commas, and no expressions."
    )

    @staticmethod
    def _configure_gpt_oss_instructions(instructions: str, model: str) -> str:
        """Set GPT-OSS reasoning effort using its supported system-message format."""
        if "gpt-oss" not in str(model or "").lower():
            return instructions

        effort = os.getenv("QUASAR_GPT_OSS_REASONING", "high").strip().lower()
        if effort not in {"low", "medium", "high"}:
            logger.warning(
                "Invalid QUASAR_GPT_OSS_REASONING=%r; defaulting to high.",
                effort,
            )
            effort = "high"

        reasoning_instruction = f"Reasoning: {effort}"
        existing_pattern = re.compile(
            r"(?im)^[ \t]*Reasoning:[ \t]*(?:low|medium|high)[ \t]*$"
        )
        if existing_pattern.search(instructions):
            updated = existing_pattern.sub(reasoning_instruction, instructions, count=1)
        elif instructions:
            updated = f"{reasoning_instruction}\n\n{instructions}"
        else:
            updated = reasoning_instruction
        if ResponsesShim._GPT_OSS_TOOL_JSON_RULE not in updated:
            updated = f"{updated}\n\n{ResponsesShim._GPT_OSS_TOOL_JSON_RULE}"
        return updated

    # tool_choice="required" pass-through for the TACC OpenAI-compatible server.
    # Live-verified 2026-09-16: vLLM behind ai.tejas.tacc.utexas.edu honours
    # "required" (200 + a tool call) even with 160 tool schemas. The earlier
    # blanket downgrade to "auto" let gpt-oss end forced data-fetch rounds with
    # reasoning and no call. A server that rejects "required" gets one 400,
    # after which this process downgrades to "auto" (+nudge).
    _tacc_required_supported = True
    # A server-enforced call that died MID-STREAM (in-band error event on a
    # 200 stream; live 2026-09-17, ~60% of enforced re-samples) puts "required"
    # on a process-wide cooldown: until it expires, escalations use the
    # emulated form. Distinct from the permanent 400 downgrade above — a
    # mid-stream death is a serving-side fault that may clear.
    _tacc_required_cooldown_until = 0.0

    @staticmethod
    def _tacc_required_cooldown_seconds() -> float:
        try:
            return max(0.0, float(os.getenv("TACC_REQUIRED_TOOL_CHOICE_COOLDOWN_SECONDS", "600")))
        except ValueError:
            return 600.0

    def _tacc_tool_choice(self, tool_choice, strict_required=False):
        """"required" is emulated (auto + nudge) on a first attempt — grammar-
        constrained decoding degrades argument quality (live 2026-09-16: 3 of 5
        first calls stuffed constraints into a free-text field) — and enforced
        server-side only when the runner escalates after a reasoning-only stop,
        the server has not rejected it, and no mid-stream death cooldown is active."""
        if not isinstance(tool_choice, str):
            return "auto"
        if tool_choice == "required":
            cls = type(self)
            enforce = (
                strict_required
                and cls._tacc_required_supported
                and time.monotonic() >= cls._tacc_required_cooldown_until
            )
            return "required" if enforce else "auto"
        return tool_choice

    def _tacc_create_with_tool_choice_fallback(self, completions_engine, call_kwargs):
        try:
            return completions_engine.create(**call_kwargs)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status is None:
                status = getattr(getattr(exc, "response", None), "status_code", None)
            if (call_kwargs.get("tool_choice") == "required" and status == 400
                    and "tool_choice" in str(exc).lower()):
                print("[PROVIDER] tacc rejected tool_choice=required; downgrading to auto (+nudge) for this process")
                type(self)._tacc_required_supported = False
                return completions_engine.create(**{**call_kwargs, "tool_choice": "auto"})
            raise

    @staticmethod
    def _apply_required_tool_choice_nudge(messages: list) -> list:
        """Emulate tool_choice="required" for providers that only accept "auto".

        TACC/vLLM rejects "required", so the shim downgrades it to "auto" — which
        silently lets weaker models answer in plain text instead of calling a
        tool. Compensate by appending an explicit instruction to the system
        message on a call-only copy (the conversation history cache keeps the
        original messages so the pressure does not leak into later rounds).
        """
        nudge = (
            "Tool use is REQUIRED for this turn: call the most appropriate tool "
            "now instead of answering in plain text."
        )
        call_messages = [dict(m) if isinstance(m, dict) else m for m in messages]
        first = call_messages[0] if call_messages else None
        if isinstance(first, dict) and first.get("role") == "system" and isinstance(first.get("content"), str):
            first["content"] = f"{first['content']}\n\n{nudge}"
        else:
            call_messages.insert(0, {"role": "system", "content": nudge})
        return call_messages

    def _append_chat_input(self, messages: list, input_data) -> None:
        """Append Responses-style input to an existing Chat Completions history."""
        if isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})
        elif isinstance(input_data, list):
            for item in input_data:
                if isinstance(item, dict) and item.get("type") == "function_call_output":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": item.get("output", ""),
                    })
                elif isinstance(item, dict) and item.get("role") == "user":
                    messages.append({"role": "user", "content": item.get("content", "")})
        else:
            messages.append({"role": "user", "content": str(input_data)})

    def _translate_tools_for_chat_completions(self, tools: list) -> list:
        """Convert Responses API flat tool format to Chat Completions nested format."""
        cc_tools = []
        for t in tools:
            cc_tools.append({
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                }
            })
        return cc_tools

    def _chat_completion_to_llm_response(self, resp) -> LLMResponse:
        """Convert Chat Completions response to LLMResponse."""
        choice = resp.choices[0] if resp.choices else None
        if not choice:
            return LLMResponse(output_text="")

        msg = choice.message
        output_text = msg.content or ""
        output_items = []

        # Handle tool calls
        if msg.tool_calls:
            for tc in msg.tool_calls:
                fc = FunctionCallItem(
                    name=tc.function.name,
                    arguments=tc.function.arguments or "",
                    call_id=tc.id,
                )
                output_items.append(fc)

        if not output_items:
            output_items = [MessageOutputItem(content=[TextContentItem(text=output_text)])]

        usage = None
        if hasattr(resp, 'usage') and resp.usage:
            usage = LLMUsage(
                input_tokens=getattr(resp.usage, 'prompt_tokens', 0),
                output_tokens=getattr(resp.usage, 'completion_tokens', 0),
                # DeepSeek cache-aware fields (present in DeepSeek API responses)
                cache_hit_tokens=getattr(resp.usage, 'prompt_cache_hit_tokens', None),
                cache_miss_tokens=getattr(resp.usage, 'prompt_cache_miss_tokens', None),
            )

        return LLMResponse(
            output_text=output_text,
            id=resp.id if hasattr(resp, 'id') else f"resp_{uuid.uuid4().hex[:16]}",
            output=output_items,
            usage=usage,
            finish_reason=getattr(choice, 'finish_reason', None),
        )

    @staticmethod
    def _inject_images_into_messages(
        messages: list,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> list:
        """Inject image attachments into the last user message for compatible chat APIs."""
        if not attachments:
            return messages

        # Collect image_url attachments
        image_parts = []
        for att in attachments:
            if att.get("type") == "image_url" or "image_url" in att:
                image_parts.append({
                    "type": "image_url",
                    "image_url": att.get("image_url") or att,
                })

        if not image_parts:
            return messages

        # Find the last user message and convert to multimodal
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                existing = messages[i]["content"]
                if isinstance(existing, str):
                    content_parts = [{"type": "text", "text": existing}]
                elif isinstance(existing, list):
                    content_parts = list(existing)
                else:
                    content_parts = [{"type": "text", "text": str(existing)}]
                content_parts.extend(image_parts)
                messages[i] = dict(messages[i], content=content_parts)
                break

        return messages

    @staticmethod
    def _deepseek_max_tokens(requested) -> int:
        """Clamp DeepSeek max_tokens to a predictable ceiling.

        DeepSeek truncates output at its provider-side cap regardless of an
        oversized request (live 2026-07-05: MAX_TOKENS=200000 turns died
        mid-sentence with no error). A bounded request keeps truncation
        detectable via finish_reason=length so the agent loop can auto-continue.
        """
        cap = int(os.getenv("DEEPSEEK_MAX_OUTPUT_TOKENS", "32768"))
        try:
            req = int(requested)
        except (TypeError, ValueError):
            req = cap
        return max(1, min(req, cap))

    @with_retry(max_retries=3, backoff_base=1.0)
    def _call_deepseek(self, kwargs: dict, attachments: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Translate responses.create() to DeepSeek Chat Completions API."""
        client = self._llm._get_deepseek_client()
        model = kwargs.get("model", self._llm.default_model)
        instructions = kwargs.get("instructions", "")
        input_data = kwargs.get("input", "")
        max_tokens = self._deepseek_max_tokens(kwargs.get("max_output_tokens", 2000))
        tools_raw = kwargs.get("tools", None)
        prev_id = kwargs.get("previous_response_id", None)
        json_mode = False
        text_opt = kwargs.get("text", None)
        if text_opt and isinstance(text_opt, dict):
            fmt = text_opt.get("format", {})
            if fmt.get("type") == "json_object":
                json_mode = True

        messages = self._chat_messages_for_input(prev_id, instructions, input_data, json_mode)

        if attachments and any(att.get("type") == "image_url" or "image_url" in att for att in attachments):
            logger.warning("Dropping raw image attachments for DeepSeek; use the image prepass upstream.")

        openai_tools = self._translate_tools_for_chat_completions(tools_raw) if tools_raw else None
        tool_choice = kwargs.get("tool_choice", None)

        call_kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if openai_tools:
            call_kwargs["tools"] = openai_tools
            if tool_choice:
                # DeepSeek thinking mode only supports "auto" or "none".
                # Omit/translate forced choices (like "required" or dict structures) to "auto" to avoid 400 error.
                call_kwargs["tool_choice"] = "auto" if tool_choice == "required" or not isinstance(tool_choice, str) else tool_choice
        if json_mode:
            call_kwargs["response_format"] = {"type": "json_object"}

        # highest thinking settings as requested
        call_kwargs["reasoning_effort"] = "max"
        call_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

        completions_engine = getattr(getattr(client, "chat"), "completions")
        resp = completions_engine.create(**call_kwargs)
        result = self._chat_completion_to_llm_response(resp)
        print(
            f"[PROVIDER] deepseek finish_reason={result.finish_reason!r} "
            f"max_tokens={max_tokens} output_chars={len(result.output_text or '')}"
        )

        # Extract reasoning_content from the response message
        msg = resp.choices[0].message if resp.choices else None
        reasoning_content = getattr(msg, "reasoning_content", None) if msg else None

        # Cache the generated response history
        new_messages = list(messages)
        tool_calls = [out for out in result.output if getattr(out, 'type', None) == 'function_call']
        if tool_calls:
            new_messages.append({
                "role": "assistant",
                "content": result.output_text or None,
                "reasoning_content": reasoning_content or None,
                "tool_calls": [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    }
                    for tc in tool_calls
                ]
            })
        else:
            new_messages.append({
                "role": "assistant",
                "content": result.output_text,
                "reasoning_content": reasoning_content or None,
            })
        
        with self._history_lock:
            self._history_cache[result.id] = new_messages
        return result

    def _stream_deepseek(self, kwargs: dict, attachments: Optional[List[Dict[str, Any]]] = None):
        """Streaming DeepSeek call — returns an iterator of StreamEvents."""
        client = self._llm._get_deepseek_client()
        model = kwargs.get("model", self._llm.default_model)
        instructions = kwargs.get("instructions", "")
        input_data = kwargs.get("input", "")
        max_tokens = self._deepseek_max_tokens(kwargs.get("max_output_tokens", 2000))
        tools_raw = kwargs.get("tools", None)
        prev_id = kwargs.get("previous_response_id", None)

        messages = self._chat_messages_for_input(prev_id, instructions, input_data)

        if attachments and any(att.get("type") == "image_url" or "image_url" in att for att in attachments):
            logger.warning("Dropping raw image attachments for DeepSeek streaming; use the image prepass upstream.")

        openai_tools = self._translate_tools_for_chat_completions(tools_raw) if tools_raw else None
        tool_choice = kwargs.get("tool_choice", None)

        call_kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if openai_tools:
            call_kwargs["tools"] = openai_tools
            if tool_choice:
                # DeepSeek thinking mode only supports "auto" or "none".
                # Omit/translate forced choices (like "required" or dict structures) to "auto" to avoid 400 error.
                call_kwargs["tool_choice"] = "auto" if tool_choice == "required" or not isinstance(tool_choice, str) else tool_choice

        # highest thinking settings as requested
        call_kwargs["reasoning_effort"] = "max"
        call_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        call_kwargs["stream_options"] = {"include_usage": True}

        resp_id = f"resp_{uuid.uuid4().hex[:16]}"
        yield StreamEvent(type="response.created", response=LLMResponse(id=resp_id))

        function_calls = {}
        reasoning_done_emitted = False
        output_text = ""
        reasoning_content = ""
        usage_obj = None
        finish_reason = None

        completions_engine = getattr(getattr(client, "chat"), "completions")
        stream = completions_engine.create(**call_kwargs)
        for chunk in stream:
            if hasattr(chunk, 'usage') and chunk.usage:
                usage_obj = LLMUsage(
                    input_tokens=getattr(chunk.usage, 'prompt_tokens', 0),
                    output_tokens=getattr(chunk.usage, 'completion_tokens', 0),
                    cache_hit_tokens=getattr(chunk.usage, 'prompt_cache_hit_tokens', None),
                    cache_miss_tokens=getattr(chunk.usage, 'prompt_cache_miss_tokens', None),
                )

            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue
            if getattr(choice, 'finish_reason', None):
                finish_reason = choice.finish_reason

            delta = choice.delta

            # 1. Real-time Chain of Thought (CoT) reasoning content
            reasoning_delta = getattr(delta, "reasoning_content", None)
            if reasoning_delta:
                reasoning_content += reasoning_delta
                yield StreamEvent(type="response.reasoning_summary_text.delta", delta=reasoning_delta)
                # No continue: content/tool_calls co-riding the same chunk
                # must still be processed below or they'd be silently lost.

            # If we were streaming reasoning but it has now stopped, emit done event
            if not reasoning_done_emitted:
                if delta.content or delta.tool_calls:
                    yield StreamEvent(type="response.reasoning_summary_text.done")
                    reasoning_done_emitted = True

            # 2. Text content
            if delta.content:
                output_text += delta.content
                yield StreamEvent(type="response.output_text.delta", delta=delta.content)

            # 3. Tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in function_calls:
                        fc = FunctionCallItem(
                            name=tc.function.name or "" if tc.function else "",
                            call_id=tc.id or f"call_{uuid.uuid4().hex[:16]}",
                        )
                        function_calls[idx] = fc
                        yield StreamEvent(type="response.output_item.added", item=fc)
                    
                    fc = function_calls[idx]
                    # Update name if it arrives in a later chunk
                    if tc.function and tc.function.name and not fc.name:
                        fc.name = tc.function.name
                    if tc.function and tc.function.arguments:
                        fc.arguments += tc.function.arguments
                        yield StreamEvent(
                            type="response.function_call_arguments.delta",
                            delta=tc.function.arguments,
                            item=fc,
                            # NOTE: We don't set call_id on StreamEvent — the
                            # agent must read it from item.call_id instead.
                        )

        if not reasoning_done_emitted:
            yield StreamEvent(type="response.reasoning_summary_text.done")

        for fc in function_calls.values():
            yield StreamEvent(type="response.output_item.done", item=fc)

        completed_items = [fc for fc in function_calls.values()]
        print(
            f"[PROVIDER] deepseek stream finish_reason={finish_reason!r} "
            f"max_tokens={max_tokens} output_chars={len(output_text)} "
            f"reasoning_chars={len(reasoning_content)} tool_calls={len(completed_items)}"
        )
        yield StreamEvent(
            type="response.completed",
            response=LLMResponse(
                id=resp_id,
                output=completed_items,
                usage=usage_obj,
                finish_reason=finish_reason,
            )
        )

        # Cache the generated response history at the end of streaming
        new_messages = list(messages)
        tool_calls = completed_items
        if tool_calls:
            new_messages.append({
                "role": "assistant",
                "content": output_text or None,
                "reasoning_content": reasoning_content or None,
                "tool_calls": [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    }
                    for tc in tool_calls
                ]
            })
        else:
            new_messages.append({
                "role": "assistant",
                "content": output_text,
                "reasoning_content": reasoning_content or None,
            })
        
        with self._history_lock:
            self._history_cache[resp_id] = new_messages



# ---------------------------------------------------------------------------
# Main LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Unified LLM client. Drop-in replacement for OpenAI() client.

    Exposes `client.responses.create(...)` via the ResponsesShim so all
    existing code works unchanged.  Routes to the correct provider based
    on model name.

    Usage:
        client = LLMClient(model="gpt-4o")
        resp = client.responses.create(model="gpt-4o", input="Hello", ...)
        print(resp.output_text)
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        provider_api_keys: Optional[Dict[str, str]] = None,
        key_source_by_provider: Optional[Dict[str, str]] = None,
    ):
        self.default_model = model
        self.provider_api_keys = dict(provider_api_keys or {})
        self.key_source_by_provider = dict(key_source_by_provider or {})
        self._responses_shim = ResponsesShim(self)

        # Lazy-loaded provider clients
        self._openai_client = None
        self._deepseek_client = None
        self._anthropic_client = None
        self._google_client = None
        self._tacc_client = None
        self._local_client = None

    @staticmethod
    def _http_timeout(provider: str = ""):
        """Shared bounded timeout policy for OpenAI-compatible providers.

        The read timeout is per-provider: reasoning providers (gpt-oss via
        TACC, DeepSeek thinking) legitimately go byte-silent for minutes during
        prompt prefill / long thinking, and the old flat 90 s read timeout
        killed those streams mid-turn with httpx.ReadTimeout (live 2026-07:
        a long-thinking Gaia white-dwarf turn and an overnight run). Precedence:
        LLM_READ_TIMEOUT_SECONDS_<PROVIDER> > LLM_READ_TIMEOUT_SECONDS >
        per-provider default (600 s for tacc/deepseek, 300 s otherwise).
        """
        import httpx

        provider = (provider or "").strip().lower()
        default_read = "600" if provider in {"tacc", "deepseek"} else "300"
        read_timeout = (
            (os.getenv(f"LLM_READ_TIMEOUT_SECONDS_{provider.upper()}") if provider else None)
            or os.getenv("LLM_READ_TIMEOUT_SECONDS")
            or default_read
        )
        return httpx.Timeout(
            connect=float(os.getenv("LLM_CONNECT_TIMEOUT_SECONDS", "15")),
            read=float(read_timeout),
            write=float(os.getenv("LLM_WRITE_TIMEOUT_SECONDS", "30")),
            pool=float(os.getenv("LLM_POOL_TIMEOUT_SECONDS", "15")),
        )

    @staticmethod
    def _normalize_provider_key(provider: str) -> str:
        provider = (provider or "").strip().lower()
        aliases = {
            "gemini": "google",
            "tejas": "tacc",
            "texas": "tacc",
            "texas_ai": "tacc",
        }
        return aliases.get(provider, provider)

    @staticmethod
    def _normalize_tacc_model(model: str) -> str:
        return (model or "").removeprefix("tacc/")

    def _resolve_context_api_key(self, provider: str) -> Optional[str]:
        provider = self._normalize_provider_key(provider)
        if provider in self.provider_api_keys and self.provider_api_keys[provider]:
            return self.provider_api_keys[provider]
        context = get_llm_request_context()
        if context and provider in context.provider_api_keys and context.provider_api_keys[provider]:
            return context.provider_api_keys[provider]
        return None

    def _resolve_key_source(self, provider: str) -> str:
        provider = self._normalize_provider_key(provider)
        if provider in self.key_source_by_provider:
            return self.key_source_by_provider.get(provider) or "platform"
        context = get_llm_request_context()
        if context and provider in context.key_source_by_provider:
            return context.key_source_by_provider.get(provider) or "platform"
        return "platform"

    def _resolve_api_key(self, provider: str, env_name: str) -> tuple[str, str]:
        provider = self._normalize_provider_key(provider)
        context_key = self._resolve_context_api_key(provider)
        if context_key:
            return context_key, "byok"
        api_key = os.getenv(env_name, "")
        if not api_key:
            raise ValueError(
                f"{env_name} is required for {provider} models. "
                "Set it in .env or add your own provider API key in Quasar settings."
            )
        return api_key, "platform"

    @property
    def responses(self) -> ResponsesShim:
        """Provides `client.responses.create(...)` interface."""
        return self._responses_shim

    # ── Provider client getters (lazy init) ─────────────────────────────

    def _get_openai_client(self):
        """Get or create OpenAI client.

        If HELICONE_API_KEY is set, routes through Helicone's proxy for
        automatic cost analytics, rate limiting, and request logging.
        Sign up free at https://helicone.ai (100k requests/month free).
        """
        from openai import OpenAI
        context_key = self._resolve_context_api_key("openai")
        if context_key:
            return OpenAI(api_key=context_key, timeout=self._http_timeout("openai"))
        if self._openai_client is None:
            api_key, key_source = self._resolve_api_key("openai", "OPENAI_API_KEY")

            helicone_key = os.getenv("HELICONE_API_KEY", "")
            if key_source == "byok":
                return OpenAI(api_key=api_key, timeout=self._http_timeout("openai"))
            if helicone_key:
                self._openai_client = OpenAI(
                    api_key=api_key,
                    base_url="https://oai.helicone.ai/v1",
                    timeout=self._http_timeout("openai"),
                    default_headers={
                        "Helicone-Auth": f"Bearer {helicone_key}",
                    },
                )
                logger.info(
                    "[Helicone] OpenAI calls routed through Helicone proxy. "
                    "Dashboard: https://helicone.ai/dashboard"
                )
            else:
                self._openai_client = OpenAI(api_key=api_key, timeout=self._http_timeout("openai"))
        return self._openai_client

    def _get_anthropic_client(self):
        """Get or create Anthropic client."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install 'anthropic' package: pip install anthropic")
        context_key = self._resolve_context_api_key("anthropic")
        if context_key:
            return anthropic.Anthropic(api_key=context_key)
        if self._anthropic_client is None:
            api_key, key_source = self._resolve_api_key("anthropic", "ANTHROPIC_API_KEY")
            if key_source == "byok":
                return anthropic.Anthropic(api_key=api_key)
            self._anthropic_client = anthropic.Anthropic(api_key=api_key)
        return self._anthropic_client

    def _get_google_client(self):
        """Get or create Google GenAI client."""
        try:
            from google import genai as ggenai
        except ImportError:
            raise ImportError("Install 'google-genai' package: pip install google-genai")
        context_key = self._resolve_context_api_key("google")
        if context_key:
            return ggenai.Client(api_key=context_key)
        if self._google_client is None:
            api_key, key_source = self._resolve_api_key("google", "GEMINI_API_KEY")
            if key_source == "byok":
                return ggenai.Client(api_key=api_key)
            self._google_client = ggenai.Client(api_key=api_key)
        return self._google_client

    def _get_local_client(self):
        """Get or create local LLM client (Ollama / LM Studio)."""
        if self._local_client is None:
            from openai import OpenAI
            base_url = os.getenv("LOCAL_LLM_BASE_URL", "")
            if not base_url:
                raise ValueError(
                    "LOCAL_LLM_BASE_URL is required for local models. "
                    "Set it in .env (e.g. http://localhost:11434/v1 for Ollama)."
                )
            self._local_client = OpenAI(
                base_url=base_url,
                api_key="not-needed",  # Local servers don't require API keys
                timeout=self._http_timeout("local"),
            )
        return self._local_client

    def _get_tacc_client(self):
        """Get or create Texas Advanced Computing Center OpenAI-compatible client."""
        from openai import OpenAI
        context_key = self._resolve_context_api_key("tacc")
        base_url = (
            os.getenv("TACC_BASE_URL", "")
            or os.getenv("TEJAS_BASE_URL", "")
            or os.getenv("TEXAS_AI_BASE_URL", "")
            or TACC_DEFAULT_BASE_URL
        )
        if context_key:
            return OpenAI(api_key=context_key, base_url=base_url, timeout=self._http_timeout("tacc"))
        if self._tacc_client is None:
            api_key = (
                os.getenv("TACC_API_KEY", "")
                or os.getenv("TEJAS_API_KEY", "")
                or os.getenv("TEXAS_AI_API_KEY", "")
            )
            if not api_key:
                raise ValueError(
                    "TACC_API_KEY is required for TACC models. "
                    "Set it in .env locally or in the Render service environment."
                )
            self._tacc_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self._http_timeout("tacc"),
            )
        return self._tacc_client

    def _get_deepseek_client(self):
        """Get or create DeepSeek client."""
        from openai import OpenAI
        context_key = self._resolve_context_api_key("deepseek")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        if context_key:
            return OpenAI(api_key=context_key, base_url=base_url, timeout=self._http_timeout("deepseek"))
        if self._deepseek_client is None:
            api_key, key_source = self._resolve_api_key("deepseek", "DEEPSEEK_API_KEY")
            if key_source == "byok":
                return OpenAI(api_key=api_key, base_url=base_url, timeout=self._http_timeout("deepseek"))
            self._deepseek_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self._http_timeout("deepseek"),
            )
        return self._deepseek_client
