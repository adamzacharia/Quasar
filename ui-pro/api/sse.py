"""SSE chat pipeline + streaming helpers, extracted from ``api/main.py``.

``_stream_chat_response`` is the shared Server-Sent-Events state machine behind
both ``POST /api/chat`` and ``POST /api/chat/upload``: it builds the per-request
LLM context, runs the agent on a worker thread, and multiplexes tokens /
thoughts / tool cards / web sources / usage into one SSE stream, persisting the
rich result to conversation history at the end. The web-source merge helpers,
document/image pre-pass helpers, and low-level SSE framing helpers live here too.

Kept importable on its own (``import api.sse``) for unit testing: it depends only
on ``api.deps`` (shared singletons/helpers), ``api.models`` (ChatRequest), and
``api.serializers.data_card`` (the data-card builder) — never on the routers.
"""

from api import bootstrap  # noqa: F401  (sys.path + shims, run before service imports)

import asyncio
import json
import queue as stdlib_queue
import re
import uuid
import time as _time
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from services.evidence_quality import (
    annotate_web_source_evidence,
    choose_better_evidence_quality,
    rank_web_sources,
)
from services.block_identity import BlockIdAllocator
from services.content_safety import is_safe_web_image, is_safe_web_source
from services.model_pricing import TurnCostAccumulator, estimate_cost

from api import deps
from api.deps import (
    ChatDeadline,
    LLMClient,
    ProviderFileService,
    ProviderKeyError,
    QuotaExceededError,
    _anthropic_models_enabled,
    _build_llm_context_for_user,
    _chat_executor,
    _current_user_email,
    _executor,
    _get_pers_db,
    _latest_traces,
    _make_quota_checker,
    _make_quota_releaser,
    _make_usage_recorder,
    _plan_feedback_lock,
    _plan_feedback_queues,
    _resolve_optional_user,
    _safe_authorization_header,
    analytics_service,
    conversation_service,
    detect_provider,
    get_agent,
    get_agent_error,
    issue_report_service,
    llm_request_context,
    logger,
    provider_file_service,
    redact_secrets,
    usage_quota_service,
)
from api.models import ChatRequest
from api.serializers.data_card import _build_data_card_event

# os is used by the image pre-pass model default lookup.
import os

# ── Assistant-text sanitation ─────────────────────────────────────────────────
_LAB_SCIENCE_EMOJI_RE = re.compile(
    f"(?:{chr(0x1F9D1)}\u200d)?{chr(0x1F52C)}\\s*"
)


def _sanitize_assistant_text(text: str) -> str:
    return _LAB_SCIENCE_EMOJI_RE.sub("", text or "")


# ── Low-level SSE framing + web-source merge helpers ──────────────────────────
def _stream_headers() -> Dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


def _sse_status(step: str, state: str = "running") -> str:
    return f"data: {json.dumps({'type': 'status', 'step': step, 'state': state})}\n\n"


def _normalize_web_url(value: Any) -> str:
    url = str(value or "").strip().strip("<>")
    url = url.rstrip(".,;:)]}'\"")
    if not url:
        return ""
    if url.startswith(("http://", "https://")):
        return url
    if url.startswith("www."):
        return f"https://{url}"
    if re.match(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}/\S+$", url):
        return f"https://{url}"
    return ""


def _merge_web_items(existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    by_url: Dict[str, int] = {}

    for item in [*(existing or []), *(incoming or [])]:
        if not isinstance(item, dict):
            continue
        if not is_safe_web_image(item):
            continue
        url = _normalize_web_url(item.get("url") or item.get("link") or item.get("href") or item.get("image_url") or item.get("src"))
        if not url:
            continue
        key = url.lower().rstrip("/")
        clean = dict(item)
        clean["url"] = url
        source_url = _normalize_web_url(
            item.get("sourceUrl")
            or item.get("source_url")
            or item.get("sourcePageUrl")
            or item.get("source_page_url")
            or item.get("pageUrl")
            or item.get("page_url")
            or item.get("source")
            or ""
        )
        if source_url:
            clean["sourceUrl"] = source_url
        source_title = str(
            item.get("sourceTitle")
            or item.get("source_title")
            or item.get("sourcePageTitle")
            or item.get("source_page_title")
            or item.get("pageTitle")
            or item.get("page_title")
            or ""
        ).strip()
        if source_title:
            clean["sourceTitle"] = source_title
        if key in by_url:
            current = merged[by_url[key]]
            for field in ("title", "snippet", "description", "sourceUrl", "sourceTitle"):
                if not current.get(field) and clean.get(field):
                    current[field] = clean[field]
            continue
        by_url[key] = len(merged)
        merged.append(clean)

    return merged


def _merge_web_sources(existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    by_url: Dict[str, int] = {}

    for item in [*(existing or []), *(incoming or [])]:
        if not isinstance(item, dict):
            continue
        if not is_safe_web_source(item):
            continue
        url = _normalize_web_url(item.get("url") or item.get("link") or item.get("href") or item.get("source_url"))
        if not url:
            continue
        key = url.lower().rstrip("/")
        clean = annotate_web_source_evidence({**item, "url": url})
        if key in by_url:
            current = merged[by_url[key]]
            for field in ("title", "snippet", "description"):
                if not current.get(field) and clean.get(field):
                    current[field] = clean[field]
            current["evidenceQuality"] = choose_better_evidence_quality(
                current.get("evidenceQuality"),
                clean.get("evidenceQuality"),
            )
            continue
        by_url[key] = len(merged)
        merged.append(clean)

    return rank_web_sources(merged)


def _merge_web_label(existing: str, incoming: Any) -> str:
    label = str(incoming or "").strip()
    if not label:
        return existing
    if not existing:
        return label
    parts = [part.strip() for part in existing.split(" + ") if part.strip()]
    if label in parts:
        return existing
    return f"{existing} + {label}"


def _request_from_trace(
    trace: Any, tool_name: str, call_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """The `request` recorded for the most recent call to ``tool_name``.

    Fallback for cards whose result dict was not stamped by the runner (e.g. a
    tool that set its card through a path the stamper does not see). Returns
    None when the tool made no traced call. Feature 1.

    When ``call_id`` is given, the entry whose ``call_id`` matches exactly wins
    regardless of name — two conductor calls to the SAME tool must each resolve
    their OWN request, never both the latest by name. Latest-by-name remains
    the fallback when the id is absent or unmatched. (A1 CX-14 sliver)
    """
    if not isinstance(trace, list):
        return None
    if call_id:
        for call in trace:
            if not isinstance(call, dict) or call.get("call_id") != call_id:
                continue
            request = call.get("request")
            # The producing call is KNOWN — if it recorded no request, show
            # none rather than substituting another call's. (A1 CX-14 sliver)
            return request if isinstance(request, dict) and request else None
    if not tool_name:
        return None
    for call in reversed(trace):
        if not isinstance(call, dict) or call.get("name") != tool_name:
            continue
        request = call.get("request")
        if isinstance(request, dict) and request:
            return request
    return None


def _sse_error_response(message: str) -> StreamingResponse:
    async def generate():
        yield f"data: {json.dumps({'type': 'error', 'content': redact_secrets(message)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=_stream_headers(),
    )


# ── Attachment / image pre-pass helpers ───────────────────────────────────────
def _extract_document_preview_text(filename: str, content_type: str, raw: bytes) -> str:
    """Best-effort text extraction used only when documents accompany images/FITS."""

    # Detect PDFs by content_type OR filename extension.
    # Browsers sometimes report application/octet-stream for PDFs, so the extension
    # check is the reliable fallback that ensures DeepSeek always receives the text.
    is_pdf = (
        content_type == "application/pdf"
        or (filename or "").lower().endswith(".pdf")
    )
    if is_pdf:
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(stream=raw, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            if not text.strip():
                return f"\n\n[PDF: {filename} - no extractable text (possibly a scanned image PDF)]"
            return f"\n\n### Attached PDF: {filename}\n{text[:8000]}"
        except ImportError:
            return f"\n\n[PDF: {filename} - install pymupdf to extract text]"
        except Exception as e:
            return f"\n\n[PDF: {filename} - extraction failed: {e}]"

    if content_type in ("text/plain", "text/csv", "text/markdown", "application/json") or \
       any((filename or "").lower().endswith(ext) for ext in [".csv", ".txt", ".md", ".json", ".tsv"]):
        try:
            text = raw.decode("utf-8", errors="replace")[:8000]
            return f"\n\n### Attached file: {filename}\n```\n{text}\n```"
        except Exception:
            return f"\n\n[Binary file: {filename} ({len(raw)/1024:.1f} KB)]"

    return f"\n\n[Attached file: {filename} ({len(raw)/1024:.1f} KB)]"


def _image_attachment_url_and_detail(image: Dict[str, Any]) -> tuple[str, str]:
    payload = image.get("image_url") or image
    if isinstance(payload, dict):
        image_url = str(payload.get("url") or payload.get("image_url") or "")
        detail = str(payload.get("detail") or "auto")
        return image_url, detail
    return str(payload or ""), str(image.get("detail") or "auto")


def _analyze_uploaded_images_for_text(
    images: List[Dict[str, Any]],
    user_prompt: str,
    *,
    model: Optional[str] = None,
) -> str:
    """Summarize/OCR uploaded images for models that cannot accept image inputs."""
    if not images:
        return ""

    prepass_model = model or os.getenv("QUASAR_IMAGE_PREPASS_MODEL", "gpt-4o-mini")
    instruction = (
        "Analyze the uploaded image(s) so a text-only model can answer the user's request.\n"
        "Return concise but complete context with these sections:\n"
        "- Visual summary\n"
        "- Visible/OCR text and UI labels\n"
        "- Tables, charts, numbers, or screen state\n"
        "- Uncertainties or unreadable regions\n\n"
        "If a person's name or identity appears as visible on-screen text, transcribe it. "
        "Do not infer a person's identity from their face alone.\n\n"
        f"User request: {user_prompt or '(no text prompt provided)'}"
    )
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": instruction}]
    for image in images:
        image_url, detail = _image_attachment_url_and_detail(image)
        if not image_url:
            continue
        content.append({
            "type": "input_image",
            "image_url": image_url,
            "detail": detail or "auto",
        })

    if len(content) == 1:
        raise RuntimeError("No readable image payload was available for image analysis.")

    client = LLMClient(model=prepass_model)
    response = client.responses.create(
        model=prepass_model,
        input=[{"role": "user", "content": content}],
        max_output_tokens=1200,
    )
    analysis = (getattr(response, "output_text", "") or "").strip()
    if not analysis:
        raise RuntimeError("The image analysis model returned no usable text.")
    return analysis


def _run_with_llm_context(
    llm_context: Dict[str, Any],
    user_id: str,
    user_email: str,
    usage_recorder,
    quota_checker,
    fn,
    quota_releaser=None,
):
    with llm_request_context(
        provider_api_keys=llm_context["provider_api_keys"],
        key_source_by_provider=llm_context["key_source_by_provider"],
        byok_token_limits=llm_context["byok_token_limits"],
        user_id=user_id,
        user_email=user_email,
        usage_recorder=usage_recorder,
        quota_checker=quota_checker,
        quota_releaser=quota_releaser,
    ):
        return fn()


# ── Shared SSE chat pipeline ──────────────────────────────────────────────────
def _stream_chat_response(
    request: ChatRequest,
    authorization: Optional[str] = None,
    attachment_context: Optional[Dict[str, Any]] = None,
    client_ip: str = "",
    cookie_token: Optional[str] = None,
) -> StreamingResponse:
    """Shared SSE chat pipeline used by both /api/chat and /api/chat/upload."""
    agent = get_agent()
    auth_header = _safe_authorization_header(authorization)
    current_user = _resolve_optional_user(auth_header, cookie_token=cookie_token)

    requested_model = request.model or (getattr(agent.config, "model", None) if agent else None) or os.getenv("DEFAULT_LLM_MODEL", "gpt-oss-120b")
    provider = detect_provider(requested_model)
    # Enforce the local-only gate on the request path too — the /api/models
    # filter only hides Claude from the picker, so without this a BYOK user or a
    # synced conversation whose stored model is claude-* could still invoke it in
    # production. Keep "hidden" and "unavailable" in sync.
    if provider == "anthropic" and not _anthropic_models_enabled():
        raise HTTPException(
            status_code=403,
            detail=(
                "Anthropic (Claude) models are not enabled on this deployment. "
                "They are available locally; set QUASAR_ENABLE_ANTHROPIC=1 to enable them here."
            ),
        )
    current_user_id = current_user.get("sub") if current_user else None
    current_user_email = _current_user_email(current_user)
    
    if not current_user_id:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please sign in to your Quasar account to access the system."
        )

    try:
        llm_context = _build_llm_context_for_user(current_user_id)
        selected_key_source = llm_context["key_source_by_provider"].get(provider, "platform")
        usage_quota_service.ensure_allowed(
            user_id=current_user_id,
            user_email=current_user_email,
            provider=provider,
            key_source=selected_key_source,
            byok_token_limit=llm_context["byok_token_limits"].get(provider),
        )
    except QuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=redact_secrets(exc))
    except ProviderKeyError as exc:
        raise HTTPException(status_code=400, detail=redact_secrets(exc))

    _quota_usage_recorder = _make_usage_recorder(current_user_id)
    # Per-turn accounting: every LLM call in this request (main loop + auxiliary
    # calls) reports here. Tokens are emitted as a final SSE "usage" event so the
    # UI can show what a response spent; cost is accumulated per CALL and
    # attributed to the key source that paid for it — see TurnCostAccumulator for
    # why the turn's selected model/route is not a safe proxy for either.
    usage = TurnCostAccumulator()

    def usage_recorder(provider: str, model: str, key_source: str, input_tokens: int,
                       output_tokens: int, reservation_id: Optional[str] = None):
        usage.record(provider, model, key_source, input_tokens, output_tokens)
        _quota_usage_recorder(
            provider=provider,
            model=model,
            key_source=key_source,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reservation_id=reservation_id,
        )

    quota_checker = _make_quota_checker(
        current_user_id,
        current_user_email,
        llm_context["byok_token_limits"],
    )
    quota_releaser = _make_quota_releaser()
    selected_provider_api_key = llm_context["provider_api_keys"].get(provider)
    selected_provider_key_scope = ProviderFileService._key_scope(selected_provider_api_key)

    # ── Conversation persistence: auto-create + save user message ──────
    conv_id = request.conversation_id
    if current_user_id:
        try:
            if not conv_id:
                # No conversation_id from frontend — create a new one
                title = conversation_service.generate_title_from_message(request.message)
                conv_id = conversation_service.create_conversation(current_user_id, title, requested_model)
                logger.info(f"[CHAT] Auto-created conversation {conv_id} with model {requested_model} for user {current_user_id}")
            else:
                # Frontend sent a conversation_id — verify it exists in the DB.
                # UIAPI-06: a direct id+owner lookup, NOT a scan of the 100 most
                # recently updated conversations — that scan silently forked any
                # OLDER conversation into a brand-new thread (reply misfiled,
                # history lost) for users with >100 conversations.
                if not conversation_service.conversation_belongs_to_user(conv_id, current_user_id):
                    # ID doesn't exist in DB (orphan/client-generated) — create a proper one
                    title = conversation_service.generate_title_from_message(request.message)
                    conv_id = conversation_service.create_conversation(current_user_id, title, requested_model)
                    logger.info(f"[CHAT] Client conv_id not found in DB, created new {conv_id} with model {requested_model}")
                else:
                    # Update model in DB to ensure it matches current selected model
                    conversation_service.update_conversation_model(conv_id, requested_model)
            conversation_service.save_message(conv_id, "user", request.message)
        except Exception as e:
            logger.warning(f"[CHAT] Failed to persist user message: {e}")

    if attachment_context is None and provider in {"openai", "anthropic", "google"}:
        attachment_context = provider_file_service.get_active_files(
            provider=provider,
            conversation_id=request.conversation_id,
            user_id=current_user_id,
            key_scope=selected_provider_key_scope,
        )
    attachment_context = attachment_context or {}
    active_attachments = attachment_context.get("attachments") or []

    # ── Langfuse parent trace creation ──────────────────────────────────────────
    from core.langfuse_integration import get_langfuse
    lf_client = get_langfuse()
    lf_trace = None
    if lf_client:
        try:
            lf_trace = lf_client.trace(
                name=f"chat: {request.message[:80]}",
                user_id=current_user_id or "anonymous",
                session_id=conv_id or request.conversation_id or "anonymous_session",
                metadata={
                    "model": requested_model,
                    "client_ip": client_ip,
                    "attachment_count": len(active_attachments),
                },
                tags=["chat"],
            )
            # Store the latest trace_id for this conversation and globally.
            # ``_last_trace_id`` is rebound on the deps module so any future
            # reader sees it (the plain name would only rebind a local here).
            deps._last_trace_id = lf_trace.id
            if conv_id or request.conversation_id:
                _latest_traces[conv_id or request.conversation_id] = lf_trace.id
        except Exception as lf_err:
            logger.warning(f"[Langfuse] Failed to create parent trace: {lf_err}")

    run_id = str(uuid.uuid4())
    trace_id = str(getattr(lf_trace, "id", "") or "")
    try:
        issue_report_service.start_run(
            run_id=run_id,
            user_id=current_user_id,
            conversation_id=conv_id or request.conversation_id or "",
            trace_id=trace_id,
            model=requested_model,
            provider=provider,
            key_source=selected_key_source,
            client_ip=client_ip,
        )
    except Exception as run_err:
        logger.warning(f"[RUN] Failed to create run record {run_id}: {run_err}")

    async def generate():
        inactivity_timeout = int(os.getenv("CHAT_INACTIVITY_TIMEOUT_SECONDS", "90"))
        standard_timeout = int(os.getenv("CHAT_STANDARD_TIMEOUT_SECONDS", "240"))
        conductor_timeout = int(os.getenv("CHAT_CONDUCTOR_TIMEOUT_SECONDS", "360"))
        # Methodical reasoning providers legitimately need more than 240s of
        # multi-round tool work (2026-07-04 live test: four deepseek turns were
        # killed mid-analysis by this cap while actively calling tools). The
        # inactivity watchdog still catches genuinely stuck runs.
        if provider == "deepseek":
            standard_timeout = int(os.getenv("CHAT_DEEPSEEK_TIMEOUT_SECONDS", "480"))
            conductor_timeout = max(conductor_timeout, standard_timeout + 120)
        # Ceiling for progress-aware deadline extensions (tool completions and
        # heartbeats push the turn deadline out, but never past this).
        hard_max_timeout = int(os.getenv("CHAT_HARD_MAX_TIMEOUT_SECONDS", "900"))
        run_started_at = _time.perf_counter()
        run_status = "started"
        run_error_code = ""
        run_error_message = ""
        run_last_status = ""
        run_tools: List[str] = []
        first_token_ms: Optional[int] = None
        provider_chunk_count = 0
        run_finalized = False
        usage_emitted = False

        def usage_sse_line() -> Optional[str]:
            """The `usage` SSE line for this turn, or None if no tokens spent.

            Emitted on BOTH the normal and the error path: a question that
            errors after one or more LLM calls has still spent tokens, and
            DataLabBench learns per-turn cost only from this event — without it,
            an expensive failed question is reported costless. Guarded by
            `usage_emitted` so the two paths never double-emit.
            """
            nonlocal usage_emitted
            # Snapshot point: close the accumulator so late conductor-thread
            # calls can't mutate totals AFTER this event / the chat_runs row
            # (CX-17); their spend stays in the per-call ledger.
            usage.freeze()
            tokens_total = usage.total_tokens
            if usage_emitted or tokens_total <= 0:
                return None
            usage_emitted = True
            return "data: " + json.dumps({
                "type": "usage",
                "inputTokens": usage.input_tokens,
                "outputTokens": usage.output_tokens,
                "totalTokens": tokens_total,
                # null when nothing in the turn could be priced (TACC/local).
                # unpricedTokens lets the UI say "≈$X, N tokens unpriced"
                # instead of implying costUsd covers the whole turn.
                "costUsd": usage.turn_cost_usd(),
                "unpricedTokens": usage.unpriced_tokens,
                "costIsEstimate": True,
                # Authoritative backend compute time for the thought chip — the
                # frontend's own clock measures stream LIFETIME, which drip
                # throttling inflated to 15+ minutes (live P15).
                "durationMs": int((_time.perf_counter() - run_started_at) * 1000),
            }) + "\n\n"

        def finalize_run_record() -> None:
            nonlocal run_finalized
            if run_finalized:
                return
            # Freeze at the snapshot funnel (CX-17): EVERY finalize path —
            # including CancelledError, which never reaches usage_sse_line() —
            # must close the accumulator before reading it, or a surviving
            # conductor executor thread can mutate totals after the chat_runs
            # row is written. Surface any late-dropped calls for observability.
            try:
                usage.freeze()
                if usage.late_calls_dropped:
                    logger.warning(
                        f"[usage] {usage.late_calls_dropped} late LLM call(s) arrived "
                        f"after run {run_id}'s usage snapshot; turn totals exclude them "
                        "(per-call ledger has the full record)."
                    )
            except Exception:  # noqa: BLE001 - accounting must never block finalize
                pass
            try:
                issue_report_service.finalize_run(
                    run_id,
                    status=run_status if run_status != "started" else "failed",
                    duration_ms=int((_time.perf_counter() - run_started_at) * 1000),
                    tools_called=run_tools,
                    last_status=run_last_status,
                    error_code=run_error_code,
                    error_message=run_error_message,
                    first_token_ms=first_token_ms,
                    provider_chunk_count=provider_chunk_count,
                    # Per-turn usage. cost_usd covers only the priced portion of
                    # the turn, so SUM(cost_usd) across chat_runs is priced spend,
                    # not total spend — total_tokens is the complete figure, and
                    # unpriced_tokens says how much of it cost_usd could not
                    # cover (so a mixed TACC + priced-embedding turn is not
                    # mistaken for fully priced).
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cost_usd=usage.turn_cost_usd(),
                    # ...and cost_usd spans every key source this turn used, so
                    # it is NOT Quasar's bill. platform_cost_usd is the subset
                    # attributed per-call to the platform key (CX-28).
                    platform_cost_usd=usage.platform_cost_usd(),
                    unpriced_tokens=usage.unpriced_tokens,
                )
                run_finalized = True
            except Exception as run_err:
                logger.warning(f"[RUN] Failed to finalize run {run_id}: {run_err}")

        # Stable per-block identity for this turn (Feature 4). conv_id is
        # already resolved above (create_conversation runs before this point),
        # so every block of the turn hashes against the same conversation.
        # Ids are minted here at emission and persisted into rich_meta; replay
        # reads them back rather than recomputing, so the eager-vs-done
        # emission order in this file can never orphan an existing rating.
        _blocks = BlockIdAllocator(conv_id or request.conversation_id or "", run_id)
        # The assistant's prose is always the turn's first text block.
        _text_block_id = _blocks.next("text")

        run_meta = {
            "type": "run_meta",
            "run_id": run_id,
            "trace_id": trace_id,
            "model": requested_model,
            "provider": provider,
            "conversation_id": conv_id or request.conversation_id or "",
            # Carries the text block's id to the client: the assistant message
            # is created before any card event arrives, so it has nowhere else
            # to learn its identity from.
            "text_block_id": _text_block_id,
            "inactivity_timeout_seconds": inactivity_timeout,
            # Advertise the HARD ceiling: the effective turn deadline extends while
            # tools are completing, so the frontend watchdog must not pre-empt an
            # actively-progressing backend at the base value.
            "turn_timeout_seconds": hard_max_timeout,
        }
        yield f"data: {json.dumps(run_meta)}\n\n"

        def _terminal_run_meta_line() -> str:
            """Re-emit run_meta with the turn's FINAL status (Feature 4).

            The first run_meta goes out before any work starts, when the outcome
            is necessarily unknown. Most failure paths then emit an `error` event
            and the client marks the turn failed from that — but not all: the
            agent-is-None path reports itself as `token` TEXT followed by an
            ordinary [DONE], which the client reads as a normal completion. That
            turn would then look successful to the eval gate, which would demand a
            star on a backend crash the user cannot judge and cannot rate.

            Re-emitting the same event with the settled status makes the outcome
            authoritative for every path, without touching the rendered text the
            way an `error` event would. No-op on healthy turns.
            """
            if run_status in ("started", "completed"):
                return ""
            final = dict(run_meta)
            final["status"] = run_status
            final["errorCode"] = run_error_code
            return f"data: {json.dumps(final)}\n\n"

        if agent is None:
            err = get_agent_error() or "Unknown initialization error"
            run_status = "failed"
            run_error_code = "agent_unavailable"
            run_error_message = err
            mock_response = (
                "Backend initialization failed.\n\n"
                "QuasarAgent could not be loaded. Python traceback:\n\n"
                f"```python\n{err}\n```\n"
            )
            for chunk in mock_response.split("\n"):
                data = json.dumps({"type": "token", "content": chunk + "\n"})
                yield f"data: {data}\n\n"
                await asyncio.sleep(0.02)
            _agent_none_meta = _terminal_run_meta_line()
            if _agent_none_meta:
                yield _agent_none_meta
            yield "data: [DONE]\n\n"
            finalize_run_record()
            return

        for note in attachment_context.get("messages", []):
            yield _sse_status(note, "completed")

        # Personal RAG retrieval for authenticated users only.
        # Searches ONLY the user's personal Qdrant collection (not the shared ALMA docs).
        enriched_message = request.message
        image_prepass = attachment_context.get("image_prepass") or {}
        image_prepass_images = image_prepass.get("images") or []
        if image_prepass_images:
            yield _sse_status("Analyzing uploaded image", "running")
            try:
                loop_img = asyncio.get_event_loop()
                image_analysis = await loop_img.run_in_executor(
                    _executor,
                    lambda: _run_with_llm_context(
                        llm_context,
                        current_user_id,
                        current_user_email,
                        usage_recorder,
                        quota_checker,
                        lambda: _analyze_uploaded_images_for_text(
                            image_prepass_images,
                            request.message,
                        ),
                        quota_releaser=quota_releaser,
                    ),
                )
            except Exception as e:
                msg = (
                    "I could not analyze the uploaded image before sending it to "
                    f"{requested_model}. The selected model cannot accept raw image inputs. "
                    f"Image analysis error: {redact_secrets(e)}"
                )
                run_status = "failed"
                run_error_code = "image_prepass_failed"
                run_error_message = msg
                yield f"data: {json.dumps({'type': 'error', 'content': msg})}\n\n"
                yield "data: [DONE]\n\n"
                finalize_run_record()
                return
            yield _sse_status("Analyzing uploaded image", "completed")
            enriched_message = (
                f"{enriched_message}\n\n"
                "### Image analysis context\n"
                "The selected model cannot receive the raw uploaded image directly, "
                "so this context was generated from the image before the final answer:\n\n"
                f"{image_analysis}"
            )
            yield _sse_status(f"Passing image context to {requested_model}", "completed")
        if request.grounded_summary:
            yield _sse_status("Grounded summary mode enabled", "completed")
        if current_user and not request.grounded_summary:
            user_id = current_user.get("sub")
            if user_id:
                try:
                    loop2 = asyncio.get_event_loop()

                    def _personal_rag_search():
                        from langchain_openai import OpenAIEmbeddings
                        from langchain_core.documents import Document as LCDocument
                        from services.vector_db import search_vectors, ensure_collection

                        # Fast check 1: Skip short greetings / simple conversational chatter
                        msg_clean = request.message.strip().lower().rstrip("?.! ")
                        if msg_clean in {
                            "hi", "hello", "hey", "hola", "thanks", "thank you", "ok", "okay", 
                            "yes", "no", "cool", "great", "awesome", "perfect", "clear", 
                            "how are you", "what's up", "good morning", "good afternoon", "good evening"
                        } or len(msg_clean) < 4:
                            return []

                        # Fast check 2: Check if user has uploaded any personalization documents
                        try:
                            conn = _get_pers_db()
                            row = conn.execute(
                                "SELECT 1 FROM documents WHERE user_id=? LIMIT 1",
                                (user_id,)
                            ).fetchone()
                            conn.close()
                            if not row:
                                return []
                        except Exception as db_err:
                            print(f"[PERSONAL_RAG] DB check failed: {db_err}")

                        collection_name = f"user_{user_id}_personal"
                        try:
                            ensure_collection(collection_name)
                        except Exception:
                            return []

                        openai_byok_key = llm_context["provider_api_keys"].get("openai")
                        openai_key_source = "byok" if openai_byok_key else "platform"
                        # reserve=True: the embed call is a real spend point
                        # with a settle below, so it takes an ATOMIC reservation
                        # like every LLM call — a bare read-then-check let
                        # concurrent embedding requests overshoot the daily cap
                        # (A2 guard CX-02 reopen).
                        _embed_reservation = usage_quota_service.ensure_allowed(
                            user_id=user_id,
                            user_email=current_user_email,
                            provider="openai",
                            key_source=openai_key_source,
                            byok_token_limit=llm_context["byok_token_limits"].get("openai"),
                            reserve=True,
                        )
                        try:
                            embeddings = (
                                OpenAIEmbeddings(openai_api_key=openai_byok_key)
                                if openai_byok_key
                                else OpenAIEmbeddings()
                            )
                            query_vector = embeddings.embed_query(request.message)
                        except Exception:
                            # The call never spent — free the held tokens.
                            usage_quota_service.release_reservation(_embed_reservation)
                            raise
                        # This call is quota-CHECKED above but was never
                        # RECORDED: it goes through langchain_openai, not
                        # LLMClient, so it never reaches LLMClient._record_usage
                        # and its tokens were missing from both the per-turn
                        # total and the quota ledger. Count them here.
                        #
                        # cl100k_base is OpenAI's real tokenizer for the
                        # text-embedding-* family, so this is the exact billed
                        # prompt-token count, not an approximation. Embeddings
                        # have no output tokens.
                        try:
                            import tiktoken

                            _embed_tokens = len(
                                tiktoken.get_encoding("cl100k_base").encode(
                                    request.message or ""
                                )
                            )
                            usage_recorder(
                                provider="openai",
                                model=getattr(
                                    embeddings, "model", "text-embedding-ada-002"
                                ),
                                key_source=openai_key_source,
                                input_tokens=_embed_tokens,
                                output_tokens=0,
                                reservation_id=_embed_reservation,
                            )
                        except Exception as _embed_usage_err:
                            # Accounting must never break retrieval, but a
                            # silent drop hides undercounted spend — log it so a
                            # persistent recording failure is visible. The held
                            # reservation must still come off.
                            usage_quota_service.release_reservation(_embed_reservation)
                            logger.warning(
                                f"[PERSONAL_RAG] embedding usage not recorded: {_embed_usage_err}"
                            )
                        hits = search_vectors(collection_name, query_vector, limit=6)
                        print(f"[PERSONAL_RAG] User={user_id}, Query='{request.message[:60]}', Hits={len(hits)}")
                        if hits:
                            for h in hits[:3]:
                                print(f"  → score={h['score']:.4f}, file={h['payload'].get('source_file','?')}")

                        # Filter by semantic score ≥ 0.25 (lower threshold for personal docs)
                        results = []
                        for hit in hits:
                            if hit["score"] < 0.25:
                                continue
                            meta = {k: v for k, v in hit["payload"].items() if k != "text"}
                            meta["_semantic_score"] = round(hit["score"], 4)
                            results.append(LCDocument(
                                page_content=hit["payload"].get("text", ""),
                                metadata=meta,
                            ))
                        return results[:4]

                    personal_docs = await loop2.run_in_executor(_executor, _personal_rag_search)
                    if personal_docs:
                        yield _sse_status("Searching personal knowledge base", "running")
                        ctx_lines = []
                        for d in personal_docs:
                            src = d.metadata.get("source_file", "personal doc")
                            ctx_lines.append(f"[From: {src}]\n{d.page_content.strip()}")
                        context_block = "\n\n---\n".join(ctx_lines)
                        enriched_message = (
                            "The user has the following relevant documents in their personal "
                            f"knowledge base:\n\n{context_block}\n\n---\nUser's question: {enriched_message}"
                        )
                        yield _sse_status("Searching personal knowledge base", "completed")
                except Exception as e:
                    print(f"[WARN] Personal RAG search failed: {e}")

        if request.grounded_summary:
            enriched_message = (
                "[GROUNDED_SUMMARY_MODE]\n"
                "For this response, use only rows, counts, identifiers, coordinates, links, "
                "and errors returned by tools in this current run. Do not add background "
                "facts, likely interpretations, unstated targets, or archive counts that "
                "are not present in the retrieved result rows. If tool results are partial "
                "or one archive fails, state that explicitly and summarize only the rows "
                "that were returned.\n\n"
                f"User request: {enriched_message}"
            )

        effective_request = ChatRequest(
            message=enriched_message,
            conversation_id=request.conversation_id,
            model=requested_model,
            grounded_summary=request.grounded_summary,
            web_search=request.web_search,
        )

        _chat_start_time = _time.perf_counter()
        try:
            loop = asyncio.get_event_loop()

            print("[INFO] Routing query to standard Response API (tool-calling loop)")

            queue = asyncio.Queue()

            # ── Plan Feedback Queue (HITL) ────────────────────────────
            # Created at generator scope so the SSE event loop can re-key
            # the queue when conversation_meta arrives with the server UUID.
            # UIAPI-08: keyed by (conversation_key, run_id), NOT conversation
            # alone — two concurrent runs in one conversation used to share a
            # single key, so the second registration clobbered the first and
            # the first run's cleanup then deleted the second's live queue,
            # making its Approve click 404.
            _pfq = stdlib_queue.Queue()
            _pfq_key = [(conv_id or f"_anon_{id(_pfq)}", run_id)]  # mutable for re-keying
            with _plan_feedback_lock:
                _plan_feedback_queues[_pfq_key[0]] = _pfq

            # Emit conversation_meta EARLY so the frontend has the server UUID
            # (or the anon queue key) BEFORE plan_review events arrive.
            # This is critical for HITL feedback routing — the frontend must
            # know the exact key used in _plan_feedback_queues.
            _meta_cid = _pfq_key[0][0]
            early_meta = json.dumps({"type": "conversation_meta", "conversation_id": _meta_cid})
            yield f"data: {early_meta}\n\n"

            def on_token(token: str):
                if token:
                    clean_token = _sanitize_assistant_text(token)
                    if clean_token:
                        asyncio.run_coroutine_threadsafe(queue.put(("token", clean_token)), loop)

            def on_thought(thought: str):
                if thought:
                    clean_thought = _sanitize_assistant_text(thought)
                    if clean_thought:
                        asyncio.run_coroutine_threadsafe(queue.put(("thought", clean_thought)), loop)

            def _run_agent():
                if lf_trace:
                    from core.llm_client import set_langfuse_parent
                    set_langfuse_parent(lf_trace)
                try:
                    try:
                        effective_user_id = (current_user.get("sub") if current_user else None) or "anonymous"

                        # Sync conversation memory to agent. agent.memory is
                        # request-thread-local (UIAPI-02), and worker threads
                        # are POOLED — clear unconditionally so a previous
                        # request's history (possibly another user's) can
                        # never survive into this run's prompts.
                        try:
                            agent.memory.clear()
                        except Exception as e:  # noqa: BLE001 - never block the turn
                            print(f"[WARN] Failed to reset agent memory: {e}")
                        if conv_id:
                            try:
                                history = conversation_service.get_conversation_messages_for_user(conv_id, effective_user_id)
                                if history:
                                    agent.memory.sync_from_ui(history)
                                    print(f"[MEMORY] Synced {len(history)} messages to agent memory.")
                            except Exception as e:
                                print(f"[WARN] Failed to sync conversation memory: {e}")

                        def _on_status(step: str, state: str):
                            asyncio.run_coroutine_threadsafe(queue.put(("status", step, state)), loop)

                        try:
                            res = agent.stream_response_api(
                                effective_request.message,
                                message_placeholder=None,
                                user_id=effective_user_id,
                                on_token=on_token,
                                on_status=_on_status,
                                attachments=active_attachments,
                                raw_query=request.message,
                                conversation_id=conv_id,
                                plan_feedback_queue=_pfq,
                                on_thought=on_thought,
                                web_search=request.web_search,
                                model=requested_model,
                                run_token=run_id,
                            )
                        finally:
                            # Clean up the plan feedback queue. UIAPI-08:
                            # identity-guarded — only remove the entry if it is
                            # still OUR queue, never a successor run's.
                            with _plan_feedback_lock:
                                if _plan_feedback_queues.get(_pfq_key[0]) is _pfq:
                                    _plan_feedback_queues.pop(_pfq_key[0], None)

                        # Snapshot thread-local results BEFORE leaving this thread.
                        # The async generator runs on the event-loop thread where
                        # these thread-local values would be invisible.
                        # R3: citation metrics ride the request-scoped LLM
                        # context (still installed on this thread here).
                        from core.llm_client import get_llm_request_context as _get_llm_ctx
                        _ctx_obj = _get_llm_ctx()
                        done_payload = {
                            "text": res,
                            "all_results": list(getattr(agent, '_accumulated_run_results', []) or []),
                            "last_result": agent.last_run_result,
                            "tool_trace": list(getattr(agent, '_accumulated_tool_trace', []) or []),
                            "citation_metrics": getattr(_ctx_obj, "citation_metrics", None),
                        }
                        asyncio.run_coroutine_threadsafe(queue.put(("done", done_payload)), loop)
                    except Exception as e:
                        print(f"Agent error: {e}")
                        asyncio.run_coroutine_threadsafe(queue.put(("error", redact_secrets(e))), loop)
                finally:
                    if lf_trace:
                        from core.llm_client import set_langfuse_parent
                        set_langfuse_parent(None)

            agent_future = loop.run_in_executor(
                _chat_executor,
                lambda: _run_with_llm_context(
                    llm_context,
                    current_user_id,
                    current_user_email,
                    usage_recorder,
                    quota_checker,
                    _run_agent,
                    quota_releaser=quota_releaser,
                ),
            )

            first_token = True
            response_text = ""
            _rich_thinking_text = ""
            # Accumulators for rich UI events — persisted to Turso for history replay
            _rich_data_tables = []  # list of data tables (multi-target support)
            _rich_data_table = None  # last data table (backward compat)
            _rich_papers = None
            _rich_papers_request = None   # the ADS query behind the papers grid (Feature 1)
            _rich_papers_block_id = None  # stable id for the grid (Feature 4)
            # A turn can emit more than one papers grid / notebook (the agent can
            # search twice; Conductor adds its own notebook alongside the normal
            # path). The singular fields above are the back-compat primary; these
            # lists keep EVERY block, so a rating on an earlier grid isn't
            # orphaned when the next one overwrites the singular field.
            _rich_papers_groups = []
            _rich_notebooks = []
            _rich_notebook = None
            _rich_image = None
            _rich_images = []
            _rich_image_urls = set()
            _emitted_image_urls = set()
            _rich_web_sources = []
            _rich_web_images = []
            _rich_web_provider = ""
            _rich_web_image_provider = ""
            _rich_web_search_type = ""
            _rich_web_query = ""
            _rich_thinking = []
            _eagerly_emitted = set()  # indices of data cards already emitted during streaming
            _pending_eager_data = []
            # UIAPI-07: initialized HERE (not only after the loop) so the
            # timeout/cancel persistence paths always have a defined trace.
            _tool_trace: List[Any] = []

            def _record_rich_image(img_url, caption, meta=None, request=None,
                                   block_id=None, block_kind=None):
                nonlocal _rich_image
                if not img_url or img_url in _rich_image_urls:
                    return None
                entry = {"url": img_url, "caption": caption}
                if meta is not None:
                    entry["meta"] = meta
                # The exact request behind this image/figure (Feature 1) — rides
                # into messages.metadata so the card keeps it across a reload.
                if isinstance(request, dict) and request:
                    entry["request"] = request
                # Stable identity (Feature 4) — the reloaded figure must resolve
                # to the same rating row as the one the user starred live.
                if block_id:
                    entry["blockId"] = block_id
                    entry["blockKind"] = block_kind or "image"
                _rich_image_urls.add(img_url)
                _rich_images.append(entry)
                _rich_image = entry
                return entry

            # UIAPI-07: persistence lives in a helper so the timeout path (via
            # the shared tail) AND the CancelledError handler can both save
            # what the turn already streamed. Guarded so a cancellation that
            # lands after the normal save cannot double-persist the message.
            turn_persisted = False

            def _persist_assistant_turn() -> None:
                nonlocal turn_persisted
                if turn_persisted:
                    return
                if not (current_user_id and conv_id and (
                    response_text or _rich_data_tables or _rich_data_table or
                    _rich_papers or _rich_notebook or _rich_images or _rich_image or
                    _rich_web_sources or _rich_web_images or
                    _rich_thinking or _rich_thinking_text
                )):
                    return
                turn_persisted = True
                try:
                    rich_meta = {}
                    if _rich_data_tables:
                        # Store all data tables for multi-target support
                        if len(_rich_data_tables) == 1:
                            rich_meta["dataTable"] = _rich_data_tables[0]
                        else:
                            rich_meta["dataTable"] = _rich_data_tables[0]  # primary (backward compat)
                            rich_meta["dataTables"] = _rich_data_tables    # all tables
                    elif _rich_data_table:
                        rich_meta["dataTable"] = _rich_data_table
                    if _rich_papers:
                        rich_meta["papers"] = _rich_papers
                        if _rich_papers_request:
                            rich_meta["papersRequest"] = _rich_papers_request
                        # Sibling of papersRequest: `papers` is the raw tile list,
                        # so the grid's own id has nowhere else to ride.
                        if _rich_papers_block_id:
                            rich_meta["papersBlockId"] = _rich_papers_block_id
                        # Same shape as dataTable/dataTables: the singular field
                        # stays the back-compat primary and the list carries every
                        # grid, so a second search can't erase the first one's
                        # rated block on replay.
                        if len(_rich_papers_groups) > 1:
                            rich_meta["papersGroups"] = _rich_papers_groups
                    if _rich_notebook:
                        rich_meta["notebook"] = _rich_notebook
                        if len(_rich_notebooks) > 1:
                            rich_meta["notebooks"] = _rich_notebooks
                    if _rich_images:
                        rich_meta["images"] = _rich_images
                        rich_meta["image"] = _rich_images[-1]
                    elif _rich_image:
                        rich_meta["image"] = _rich_image
                    if _rich_web_sources:
                        rich_meta["webSources"] = _rich_web_sources
                    if _rich_web_images:
                        rich_meta["webImages"] = _rich_web_images
                    if _rich_web_provider:
                        rich_meta["webProvider"] = _rich_web_provider
                    if _rich_web_image_provider:
                        rich_meta["webImageProvider"] = _rich_web_image_provider
                    if _rich_web_search_type:
                        rich_meta["webSearchType"] = _rich_web_search_type
                    if _rich_web_query:
                        rich_meta["webQuery"] = _rich_web_query
                    if _rich_thinking:
                        rich_meta["thinkingSteps"] = _rich_thinking
                    if _rich_thinking_text:
                        rich_meta["thinking"] = _rich_thinking_text
                    # Raw request provenance (Feature 1). Without this the exact
                    # queries are dropped on history replay. Persist a REDACTED,
                    # reload-safe projection — never the raw arguments/output,
                    # which can carry credential-bearing URLs into the DB (CX-01).
                    if _tool_trace:
                        from core.provenance import persistable_trace
                        _persist_trace = persistable_trace(_tool_trace)
                        if _persist_trace:
                            rich_meta["toolTrace"] = _persist_trace
                    rich_meta["runMeta"] = {
                        "run_id": run_id,
                        "trace_id": trace_id,
                        "model": requested_model,
                        "provider": provider,
                        "conversation_id": conv_id or request.conversation_id or "",
                        # The text block's id (Feature 4) — restored onto the
                        # base message, which is the block the prose lives on.
                        "text_block_id": _text_block_id,
                    }
                    # R3: mechanical citation recall/precision — persisted so
                    # eval mode can read them on history replay.
                    if _citation_metrics:
                        rich_meta["runMeta"]["citationMetrics"] = _citation_metrics
                    # UIAPI-07: failed AND timed_out/cancelled turns persist
                    # their real status, so a reloaded partial answer replays
                    # as degraded rather than masquerading as a clean turn.
                    if run_status not in ("started", "completed"):
                        rich_meta["runMeta"]["status"] = run_status
                        rich_meta["runMeta"]["errorCode"] = run_error_code
                    conversation_service.save_message(
                        conv_id, "assistant", response_text or "",
                        metadata=rich_meta if rich_meta else None,
                    )
                except Exception as e:
                    logger.warning(f"[CHAT] Failed to persist assistant message: {e}")

            deadline = ChatDeadline(
                inactivity_seconds=inactivity_timeout,
                standard_seconds=standard_timeout,
                conductor_seconds=conductor_timeout,
                started_at=run_started_at,
                hard_max_seconds=hard_max_timeout,
            )

            while True:
                # Use a timeout so we can send SSE keepalive comments.
                # Render's reverse proxy kills idle connections after ~30s.
                # Sending `:keepalive\n\n` (an SSE comment) every 15s prevents this.
                try:
                    now = _time.perf_counter()
                    wait_seconds = min(15.0, deadline.remaining(now))
                    if wait_seconds <= 0:
                        raise asyncio.TimeoutError
                    msg = await asyncio.wait_for(queue.get(), timeout=wait_seconds)
                except asyncio.TimeoutError:
                    now = _time.perf_counter()
                    timeout_code = deadline.timeout_code(now)
                    if timeout_code:
                        run_status = "timed_out"
                        run_error_code = timeout_code
                        run_error_message = (
                            f"The {provider} model stopped producing progress. "
                            "The run was ended so the chat would not remain stuck."
                        )
                        try:
                            agent.cancel_response_run(
                                conv_id or request.conversation_id or "",
                                requested_model,
                                run_id,
                            )
                        except Exception as clear_err:
                            logger.warning(f"[RUN] Failed to clear response state: {clear_err}")
                        agent_future.cancel()
                        # Usage before the error event, so both the UI (which
                        # stops at error) and the benchmark capture the tokens
                        # spent before the timeout.
                        _to_usage_line = usage_sse_line()
                        if _to_usage_line:
                            yield _to_usage_line
                        yield f"data: {json.dumps({'type': 'error', 'code': timeout_code, 'content': run_error_message, 'run_id': run_id})}\n\n"
                        # UIAPI-07: BREAK instead of returning — the shared
                        # tail below persists everything this turn already
                        # streamed (accumulated tokens + eagerly emitted
                        # figures) with runMeta.status="timed_out", then emits
                        # the terminal run_meta and [DONE]. Returning here
                        # skipped persistence entirely, so a half-answered turn
                        # vanished from history on reload.
                        break
                    yield ": keepalive\n\n"
                    continue
                deadline.mark_activity(_time.perf_counter())
                if isinstance(msg, tuple) and len(msg) == 3:
                    msg_type, step, state = msg
                    if msg_type == "status" and isinstance(step, str) and (
                        step.startswith("__tool_heartbeat__")
                        or (state == "completed" and not step.startswith("__"))
                    ):
                        # Completed tool steps and live tool heartbeats are real
                        # progress — push the total-turn deadline out so long
                        # multi-tool workflows aren't killed mid-analysis while
                        # actively working (2026-07 live test P6/P9/P14/P15).
                        deadline.extend_for_progress(_time.perf_counter())
                    if msg_type == "status":
                        if step == "__run_mode__:conductor":
                            deadline.enable_conductor()
                            continue
                        if isinstance(step, str) and step.startswith("__tool_heartbeat__"):
                            heartbeat_detail = step[len("__tool_heartbeat__"):]
                            heartbeat_tool, _, heartbeat_label = heartbeat_detail.partition("::")
                            run_last_status = heartbeat_label or heartbeat_tool or run_last_status
                            yield f"data: {json.dumps({'type': 'run_progress', 'phase': run_last_status, 'tool': heartbeat_tool})}\n\n"
                            continue
                        if isinstance(step, str) and step.startswith("__eager_data__"):
                            # Agent sent inline result data — stash it for
                            # the next __data_ready__ to consume.
                            try:
                                import pandas as pd
                                _inline_json = step[len("__eager_data__"):]
                                _inline_result = json.loads(_inline_json)
                                # Reconstruct DataFrame if serialized
                                if "data" in _inline_result and isinstance(_inline_result["data"], (dict, list)):
                                    try:
                                        _inline_result["data"] = pd.DataFrame(_inline_result["data"])
                                    except Exception:
                                        pass
                                _pending_eager_data.append(_inline_result)
                            except Exception as _e:
                                print(f"[WARN] Failed to parse eager data: {_e}")
                        elif isinstance(step, str) and step.startswith("__data_ready__"):
                            # ── Eager data card emission ──
                            # The agent finished a tool call — build the card
                            # from the inline data stashed above.
                            try:
                                _payload_str = step[len("__data_ready__"):]
                                _payload_meta = json.loads(_payload_str)
                                _eager_idx = _payload_meta.get("_idx", -1)
                                # Use inline data if available
                                _eager_result = _pending_eager_data.pop(0) if _pending_eager_data else None
                                if _eager_result:
                                    if _eager_result.get("type") != "image":
                                        # UIAPI-14: images are the ONLY eagerly
                                        # emitted card kind. "data" results are
                                        # deliberately bypassed (intermediate
                                        # tables must not flash mid-stream; the
                                        # done-path emits the final one), and
                                        # for every other type the old
                                        # else-branch called
                                        # _build_data_card_event, which returns
                                        # None for all non-"data" types — that
                                        # branch was confirmed-unreachable dead
                                        # code and has been removed. As a
                                        # consequence _eagerly_emitted stays
                                        # empty today; the done-path guard on it
                                        # is kept for a future re-enable.
                                        print(f"[EAGER] Bypassed eager emission for idx={_eager_idx} (type={_eager_result.get('type')}) during streaming")
                                    else:
                                        # Figures render the moment their tool completes; a
                                        # deadline-killed turn no longer loses already-built
                                        # plots. _emitted_image_urls keeps the done-path from
                                        # re-emitting these.
                                        img_url = _eager_result.get("image_url", "")
                                        caption = _eager_result.get("caption", "")
                                        meta = _eager_result.get("meta")
                                        plotly_spec = _eager_result.get("plotly_spec")
                                        if img_url and img_url not in _emitted_image_urls:
                                            _emitted_image_urls.add(img_url)
                                            # Request stamped on the result by runner.py (the
                                            # trace is thread-local and unreachable here).
                                            _eimg_req = _eager_result.get("request")
                                            _eimg_req = _eimg_req if isinstance(_eimg_req, dict) else None
                                            # Ordinal is burned only now, after the
                                            # dedup guard — a re-emitted URL must not
                                            # renumber the blocks behind it.
                                            _eimg_kind = "plotly" if plotly_spec else "image"
                                            _eimg_block = _blocks.next(_eimg_kind)
                                            if plotly_spec:
                                                yield f"data: {json.dumps({'type': 'plotly', 'spec': plotly_spec, 'title': caption, 'png_fallback': img_url, 'meta': meta, 'request': _eimg_req, 'blockId': _eimg_block, 'blockKind': 'plotly'})}\n\n"
                                            else:
                                                yield f"data: {json.dumps({'type': 'image', 'url': img_url, 'caption': caption, 'meta': meta, 'request': _eimg_req, 'blockId': _eimg_block, 'blockKind': 'image'})}\n\n"
                                            _record_rich_image(img_url, caption, meta, request=_eimg_req, block_id=_eimg_block, block_kind=_eimg_kind)
                                            print(f"[EAGER] Image emitted during streaming: {str(img_url)[:100]}")
                            except Exception as _eager_err:
                                print(f"[WARN] Eager data card emission failed: {_eager_err}")
                        elif isinstance(step, str) and step.startswith("__event__"):
                            try:
                                event_json = step[len("__event__"):]
                                event_parsed = json.loads(event_json)
                                # Re-key plan feedback queue when server assigns conversation_id
                                if event_parsed.get("type") == "conversation_meta":
                                    new_cid = event_parsed.get("conversation_id")
                                    if new_cid and new_cid != _pfq_key[0][0]:
                                        old_key = _pfq_key[0]
                                        new_key = (new_cid, run_id)  # UIAPI-08: stays run-scoped
                                        with _plan_feedback_lock:
                                            if _plan_feedback_queues.get(old_key) is _pfq:
                                                _plan_feedback_queues.pop(old_key, None)
                                            _plan_feedback_queues[new_key] = _pfq
                                        _pfq_key[0] = new_key
                                        print(f"[HITL] Re-keyed plan feedback queue: {str(old_key[0])[:20]}... → {new_cid[:20]}...")
                                if event_parsed.get("type") == "web_sources":
                                    _rich_web_sources = _merge_web_sources(
                                        _rich_web_sources,
                                        event_parsed.get("sources") or [],
                                    )
                                    _rich_web_images = _merge_web_items(
                                        _rich_web_images,
                                        event_parsed.get("images") or [],
                                    )
                                    _rich_web_provider = _merge_web_label(
                                        _rich_web_provider,
                                        event_parsed.get("provider"),
                                    )
                                    _rich_web_image_provider = _merge_web_label(
                                        _rich_web_image_provider,
                                        event_parsed.get("image_provider"),
                                    )
                                    _rich_web_search_type = _merge_web_label(
                                        _rich_web_search_type,
                                        event_parsed.get("search_type"),
                                    )
                                    _rich_web_query = _rich_web_query or str(event_parsed.get("query") or "")
                                    event_parsed["sources"] = _rich_web_sources
                                    event_parsed["images"] = _rich_web_images
                                    event_parsed["provider"] = _rich_web_provider
                                    event_parsed["image_provider"] = _rich_web_image_provider
                                    event_parsed["search_type"] = _rich_web_search_type
                                    event_parsed["query"] = _rich_web_query
                                    event_json = json.dumps(event_parsed)
                                yield f"data: {event_json}\n\n"
                            except Exception:
                                yield _sse_status(step, state)
                        else:
                            yield _sse_status(step, state)
                            # Capture thinking steps for history
                            _rich_thinking.append({"step": step, "state": state})
                            run_last_status = str(step)
                        continue

                msg_type, payload = msg[0], msg[1]
                if msg_type == "done":
                    # payload is a dict with text + snapshotted thread-local results
                    response_text = _sanitize_assistant_text(payload["text"] if isinstance(payload, dict) else payload)
                    _snapshot_all = payload.get("all_results", []) if isinstance(payload, dict) else []
                    _snapshot_last = payload.get("last_result") if isinstance(payload, dict) else None
                    _snapshot_tool_trace = payload.get("tool_trace", []) if isinstance(payload, dict) else []
                    _snapshot_citation_metrics = payload.get("citation_metrics") if isinstance(payload, dict) else None
                    break
                if msg_type == "error":
                    run_status = "failed"
                    run_error_code = "agent_error"
                    run_error_message = str(payload)
                    response_text = f"An error occurred: {payload}"
                    _snapshot_all = []
                    _snapshot_last = None
                    # Emit usage BEFORE the error event: the web client stops
                    # consuming at the error, so tokens spent before this
                    # worker-reported failure must precede it (CX-07). Guarded by
                    # usage_emitted, so the post-loop emission won't double-send.
                    _werr_usage_line = usage_sse_line()
                    if _werr_usage_line:
                        yield _werr_usage_line
                    yield f"data: {json.dumps({'type': 'error', 'code': run_error_code, 'content': run_error_message, 'run_id': run_id})}\n\n"
                    break
                if msg_type == "token":
                    provider_chunk_count += 1
                    if first_token_ms is None:
                        first_token_ms = int((_time.perf_counter() - run_started_at) * 1000)
                    first_token = False
                    # UIAPI-07: accumulate as we stream — a timed-out or
                    # cancelled turn persists exactly what the user already saw
                    # instead of vanishing on reload. The "done" payload
                    # replaces this with the agent's full sanitized text.
                    response_text += payload
                    data = json.dumps({"type": "token", "content": payload})
                    yield f"data: {data}\n\n"
                if msg_type == "thought":
                    _rich_thinking_text += payload
                    data = json.dumps({"type": "thought", "content": payload})
                    yield f"data: {data}\n\n"

            # ── Collect all accumulated results (multi-target support) ──
            # Results were snapshotted inside _run_agent (same thread as
            # the agent) so they survive thread-local cleanup.
            _all_results = _snapshot_all if '_snapshot_all' in dir() else []
            _last_result = _snapshot_last if '_snapshot_last' in dir() else None
            _tool_trace = _snapshot_tool_trace if '_snapshot_tool_trace' in dir() else []
            _citation_metrics = _snapshot_citation_metrics if '_snapshot_citation_metrics' in dir() else None
            # Deduplicate: if last_run_result isn't already in the list, add it
            if _last_result and not _all_results:
                _all_results = [_last_result]
            elif _last_result and _all_results:
                # Check by data/papers identity — the same object means same result
                _last_data_id = id(_last_result.get("data")) if _last_result.get("data") is not None else None
                _last_papers_id = id(_last_result.get("papers")) if _last_result.get("papers") is not None else None
                _last_result_marker = _last_result.get("_result_id")
                _already_present = False
                for r in _all_results:
                    r_data_id = id(r.get("data")) if r.get("data") is not None else None
                    r_papers_id = id(r.get("papers")) if r.get("papers") is not None else None
                    r_result_marker = r.get("_result_id")
                    if r is _last_result:
                        _already_present = True
                        break
                    if _last_data_id is not None and r_data_id == _last_data_id:
                        _already_present = True
                        break
                    if _last_papers_id is not None and r_papers_id == _last_papers_id:
                        _already_present = True
                        break
                    # Match by _result_id marker set during accumulation
                    if _last_result_marker is not None and r_result_marker == _last_result_marker:
                        _already_present = True
                        break
                if not _already_present:
                    _all_results.append(_last_result)

            # If _all_results contains multiple items of type "data", keep only the last one.
            # To avoid shifting indices of other results (which would break _eagerly_emitted check),
            # we replace intermediate "data" results with None.
            data_indices = [i for i, r in enumerate(_all_results) if r and r.get("type") == "data"]
            if len(data_indices) > 1:
                for idx in data_indices[:-1]:
                    _all_results[idx] = None

            # If _all_results contains multiple items of type "conductor_result", keep only the last one.
            # To avoid shifting indices, we replace intermediate "conductor_result" results with None.
            conductor_indices = [i for i, r in enumerate(_all_results) if r and r.get("type") == "conductor_result"]
            if len(conductor_indices) > 1:
                for idx in conductor_indices[:-1]:
                    _all_results[idx] = None

            # ── Process each accumulated result ──────────────────────
            _seen_result_ids = set()  # avoid duplicate emissions
            for _result_idx, _run_result in enumerate(_all_results):
                if not _run_result:
                    continue
                # Skip results already emitted eagerly during streaming
                if _result_idx in _eagerly_emitted:
                    continue
                # Deduplicate by stable payload identity; images use URL so copied stale dicts collapse.
                _image_url_for_dedup = str(_run_result.get("image_url") or "").strip()
                if _run_result.get("type") == "image" and _image_url_for_dedup:
                    _dedup_key = ("image", _image_url_for_dedup)
                else:
                    _dedup_key = (
                        _run_result.get("type", ""),
                        _run_result.get("tool_name", ""),
                        id(_run_result.get("data")) if _run_result.get("data") is not None else id(_run_result),
                    )
                if _dedup_key in _seen_result_ids:
                    continue
                _seen_result_ids.add(_dedup_key)

                result_type = _run_result.get("type", "")
                tool_name_raw = _run_result.get("tool_name") or (
                    "search_papers" if result_type == "papers" else
                    "search_alma_archive" if result_type == "data" else
                    "quasar_tool"
                )
                tool_display = tool_name_raw.replace("_", " ").title()
                if tool_name_raw not in run_tools:
                    run_tools.append(tool_name_raw)
                # The exact request behind this card (Feature 1): prefer the one
                # runner.py stamped on the result, else match this tool's most
                # recent trace record. Already secret-redacted at both sources.
                _rr_request = _run_result.get("request")
                if not isinstance(_rr_request, dict) or not _rr_request:
                    _rr_request = _request_from_trace(_tool_trace, tool_name_raw)
                tool_event = json.dumps({
                    "type": "tool_call",
                    "name": tool_name_raw,
                    "displayName": tool_display,
                    "status": "completed",
                    "input": _run_result.get("params", {}),
                    "output": "Found results",
                    "request": _rr_request or None,
                })
                yield f"data: {tool_event}\n\n"
                await asyncio.sleep(0.05)

                if result_type == "data":
                    # Same allocator as the eager path above: a turn whose first
                    # table went out eagerly numbers this one "data-1", and the
                    # id is persisted either way.
                    # UIAPI-11: built on a worker thread — the card build makes
                    # a blocking CADC DataLink call (urllib, timeout=8s) and can
                    # DiskCache-write a large frame; on the event loop a CADC
                    # outage would stall the whole backend (and every other
                    # user's SSE keepalive) for up to 8s per card.
                    _card_result = _run_result
                    card = await loop.run_in_executor(
                        _executor,
                        lambda: _build_data_card_event(
                            _card_result,
                            current_user_id,
                            block_id_factory=lambda: _blocks.next("data"),
                        ),
                    )
                    if card:
                        _event_str, _rich_dt = card
                        yield _event_str
                        _rich_data_tables.append(_rich_dt)
                        _rich_data_table = _rich_dt
                        await asyncio.sleep(0.05)

                elif result_type == "papers":
                    papers = _run_result.get("papers", [])
                    if papers:
                        # The executed ADS query behind this result set (Feature 1).
                        # ONE shared request for the whole grid — repeating the same
                        # query on every paper card would be noise.
                        # One block id for the grid too, for the same reason: the
                        # user rates "these papers", not each tile.
                        _papers_block = _blocks.next("papers")
                        papers_event = json.dumps({
                            "type": "papers", "papers": papers,
                            "request": _rr_request or None,
                            "blockId": _papers_block, "blockKind": "papers",
                        })
                        yield f"data: {papers_event}\n\n"
                        _rich_papers = papers  # Capture for history
                        _rich_papers_request = _rr_request or None
                        _rich_papers_block_id = _papers_block
                        _rich_papers_groups.append({
                            "papers": papers,
                            "request": _rr_request or None,
                            "blockId": _papers_block,
                        })
                        await asyncio.sleep(0.05)

                elif result_type == "notebook":
                    nb_data = _run_result.get("notebook_data", {})
                    nb_title = _run_result.get("title", "Analysis Notebook")
                    if nb_data:
                        _nb_block = _blocks.next("notebook")
                        notebook_event = json.dumps({
                            "type": "notebook",
                            "title": nb_title,
                            "data": nb_data,
                            "blockId": _nb_block, "blockKind": "notebook",
                        })
                        yield f"data: {notebook_event}\n\n"
                        _rich_notebook = {"title": nb_title, "data": nb_data,
                                          "blockId": _nb_block}
                        _rich_notebooks.append(_rich_notebook)
                        await asyncio.sleep(0.05)

                elif result_type == "image":
                    img_url = _run_result.get("image_url", "")
                    caption = _run_result.get("caption", "")
                    meta = _run_result.get("meta")
                    plotly_spec = _run_result.get("plotly_spec")
                    if img_url and img_url not in _emitted_image_urls:
                        _emitted_image_urls.add(img_url)
                        _img_kind = "plotly" if plotly_spec else "image"
                        _img_block = _blocks.next(_img_kind)
                        if plotly_spec:
                            # Interactive card; the PNG stays as fallback + history record.
                            plotly_event = json.dumps({
                                "type": "plotly", "spec": plotly_spec, "title": caption,
                                "png_fallback": img_url, "meta": meta,
                                "request": _rr_request or None,
                                "blockId": _img_block, "blockKind": "plotly",
                            })
                            yield f"data: {plotly_event}\n\n"
                        else:
                            image_event = json.dumps({
                                "type": "image", "url": img_url, "caption": caption, "meta": meta,
                                "request": _rr_request or None,
                                "blockId": _img_block, "blockKind": "image",
                            })
                            yield f"data: {image_event}\n\n"
                        _record_rich_image(img_url, caption, meta, request=_rr_request,
                                           block_id=_img_block, block_kind=_img_kind)
                        await asyncio.sleep(0.05)

                elif result_type == "conductor_result":
                    # Multiple images accumulated during Conductor orchestration
                    images = _run_result.get("images", [])
                    for img in images:
                        img_url = img.get("image_url", "")
                        caption = img.get("caption", "")
                        meta = img.get("meta")
                        if img_url:
                            if img_url in _emitted_image_urls:
                                continue
                            _emitted_image_urls.add(img_url)
                            # CX-14-residual: normal image events carry the exact
                            # `request` behind the figure (Feature 1); conductor
                            # figures must not silently lose theirs. Prefer the
                            # request runner.py stamped on the image dict, else
                            # the most recent traced call of the tool the dict
                            # names. When neither exists the event carries NO
                            # request — never a fabricated one — and the
                            # turn-level tool_trace persisted with the message
                            # is the provenance fallback surface.
                            _cimg_req = img.get("request")
                            if not isinstance(_cimg_req, dict) or not _cimg_req:
                                # The stamped id pins the exact producing call —
                                # name-only matching gave BOTH cards the latest
                                # call's request when one tool rendered two
                                # figures. (A1 CX-14 sliver)
                                _cimg_req = _request_from_trace(
                                    _tool_trace, str(img.get("tool_name") or ""),
                                    call_id=img.get("trace_call_id"),
                                )
                            # Conductor figures share the "image" counter with the
                            # single-image path above — they are the same kind of
                            # card to the reader, and to the rater.
                            _cimg_block = _blocks.next("image")
                            image_event = json.dumps({
                                "type": "image", "url": img_url, "caption": caption, "meta": meta,
                                "request": _cimg_req or None,
                                "blockId": _cimg_block, "blockKind": "image",
                            })
                            yield f"data: {image_event}\n\n"
                            _record_rich_image(img_url, caption, meta, request=_cimg_req,
                                               block_id=_cimg_block, block_kind="image")
                            await asyncio.sleep(0.05)
                            
                    # Companion notebook
                    nb_data = _run_result.get("notebook_data", {})
                    nb_title = _run_result.get("title", "Research Notebook")
                    if nb_data:
                        _cnb_block = _blocks.next("notebook")
                        nb_event = json.dumps({
                            "type": "notebook",
                            "title": nb_title,
                            "data": nb_data,
                            "blockId": _cnb_block, "blockKind": "notebook",
                        })
                        yield f"data: {nb_event}\n\n"
                        _rich_notebook = {"title": nb_title, "data": nb_data,
                                          "blockId": _cnb_block}
                        _rich_notebooks.append(_rich_notebook)
                        await asyncio.sleep(0.05)

            if response_text and first_token:
                data = json.dumps({"type": "token", "content": response_text})
                yield f"data: {data}\n\n"

            # ── Persist assistant response + rich UI data to DB ─────
            # (UIAPI-07: shared with the timeout path — see _persist_assistant_turn.)
            _persist_assistant_turn()

            # ── Log chat analytics ─────────────────────────────────
            try:
                _elapsed_ms = int((_time.perf_counter() - _chat_start_time) * 1000)
                _tool_names = [s.get("step", "").replace("Calling tool: ", "") for s in _rich_thinking if "Calling tool:" in s.get("step", "")]
                for tool_name in _tool_names:
                    if tool_name and tool_name not in run_tools:
                        run_tools.append(tool_name)
                _user_email = current_user.get("email", "") if current_user else ""
                _user_name = current_user.get("name", "") if current_user else ""
                analytics_service.log_chat(
                    user_id=current_user_id or "anonymous",
                    username=_user_email,
                    email=_user_email,
                    display_name=_user_name,
                    ip_address=client_ip,
                    conversation_id=conv_id or "",
                    prompt=request.message,
                    response_preview=response_text,
                    model=requested_model,
                    tools_called=_tool_names,
                    response_time_ms=_elapsed_ms,
                )
            except Exception as e:
                logger.warning(f"[ANALYTICS] Failed to log chat: {e}")

            # Send back the conversation_id so the frontend can track it
            if conv_id:
                meta_event = json.dumps({"type": "conversation_meta", "conversation_id": conv_id})
                yield f"data: {meta_event}\n\n"

            # Full tool-call trace (names, arguments, truncated outputs) —
            # consumed by debug tooling and Benchmark/datalabbench scoring.
            if _tool_trace:
                try:
                    _TRACE_CAP = 131072  # hard cap for one SSE event
                    trace_event = json.dumps(
                        {"type": "tool_trace", "calls": _tool_trace}, default=str
                    )
                    if len(trace_event) > _TRACE_CAP:
                        # Degrade in stages until it fits: (1) drop raw outputs,
                        # (2) replace oversized argument dicts, (3) drop tail
                        # calls — always recording how much was omitted.
                        slim_calls = [
                            {k: v for k, v in c.items() if k != "output"}
                            for c in _tool_trace if isinstance(c, dict)
                        ]
                        for c in slim_calls:
                            try:
                                if len(json.dumps(c.get("arguments", {}), default=str)) > 4000:
                                    c["arguments"] = {"_truncated": True}
                            except Exception:
                                c["arguments"] = {"_truncated": True}
                        omitted = 0
                        trace_event = json.dumps(
                            {"type": "tool_trace", "calls": slim_calls,
                             "truncated": True, "omitted_calls": omitted}, default=str
                        )
                        while len(trace_event) > _TRACE_CAP and slim_calls:
                            slim_calls.pop()
                            omitted += 1
                            trace_event = json.dumps(
                                {"type": "tool_trace", "calls": slim_calls,
                                 "truncated": True, "omitted_calls": omitted}, default=str
                            )
                    yield f"data: {trace_event}\n\n"
                except Exception as _tt_err:
                    logger.warning(f"[CHAT] Failed to emit tool_trace event: {_tt_err}")

            # R3: mechanical citation recall/precision for this turn — consumed
            # by eval mode and DataLabBench (informational; never affects
            # question scoring).
            if _citation_metrics:
                try:
                    yield f"data: {json.dumps({'type': 'citation_metrics', 'metrics': _citation_metrics, 'run_id': run_id})}\n\n"
                except Exception as _cm_err:
                    logger.warning(f"[CHAT] Failed to emit citation_metrics event: {_cm_err}")

            _usage_line = usage_sse_line()
            if _usage_line:
                yield _usage_line

            if run_status == "started":
                run_status = "completed"
            _final_meta = _terminal_run_meta_line()
            if _final_meta:
                yield _final_meta
            yield "data: [DONE]\n\n"

        except asyncio.CancelledError:
            run_status = "cancelled"
            run_error_code = "client_cancelled"
            run_error_message = "The client cancelled the chat run."
            try:
                if "agent_future" in locals():
                    agent_future.cancel()
                agent.cancel_response_run(
                    conv_id or request.conversation_id or "",
                    requested_model,
                    run_id,
                )
            except Exception:
                pass
            # UIAPI-07: a mid-turn page close must not erase what already
            # streamed — persist the accumulated tokens + eager cards with
            # runMeta.status="cancelled". Best-effort: cancellation can land
            # before the accumulators (or the helper) even exist, in which
            # case there is nothing to save.
            try:
                _persist_assistant_turn()
            except Exception:
                pass
            raise
        except Exception as e:
            run_status = "failed"
            run_error_code = "chat_error"
            run_error_message = redact_secrets(e)
            print(f"Chat route error: {redact_secrets(e)}")
            # Report tokens/cost already spent BEFORE the error event: the web
            # client stops consuming the stream at the error event, so usage
            # emitted after it would reach the benchmark (which drains to
            # [DONE]) but never the UI. Emitting first covers both.
            _err_usage_line = usage_sse_line()
            if _err_usage_line:
                yield _err_usage_line
            error_data = json.dumps({
                "type": "error",
                "code": run_error_code,
                "content": run_error_message,
                "run_id": run_id,
            })
            yield f"data: {error_data}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            # UIAPI-08 leak guard: if the executor cancelled _run_agent before
            # it ever started (saturated pool), its finally-block cleanup never
            # ran and the registered queue would sit in the module-level dict
            # forever. Identity-compare so a successor run's queue is untouched.
            if "_pfq" in locals():
                with _plan_feedback_lock:
                    if _plan_feedback_queues.get(_pfq_key[0]) is _pfq:
                        _plan_feedback_queues.pop(_pfq_key[0], None)
            finalize_run_record()
            if lf_trace:
                try:
                    lf_trace.update(
                        metadata={
                            "response_length": len(response_text) if response_text else 0,
                            "response_preview": response_text[:200] if response_text else "",
                            "run_id": run_id,
                            "run_status": run_status,
                            "first_token_ms": first_token_ms,
                            "provider_chunk_count": provider_chunk_count,
                        }
                    )
                except Exception:
                    pass
            # Free memory between requests — critical on 2GB instances
            import gc
            gc.collect()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=_stream_headers(),
    )
