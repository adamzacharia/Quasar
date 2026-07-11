"""Chat endpoints: streaming SSE chat and file-upload chat."""

import os
from typing import List as PyList, Optional

from fastapi import APIRouter, File, Form, Header, UploadFile
from starlette.requests import Request

from services.auth_cookie import read_auth_cookie
from api.deps import (
    ProviderFileError,
    ProviderFileService,
    ProviderKeyError,
    QuotaExceededError,
    UploadedChatFile,
    _build_llm_context_for_user,
    _current_user_email,
    _resolve_optional_user,
    _safe_authorization_header,
    detect_provider,
    logger,
    model_accepts_direct_image_input,
    provider_file_service,
    redact_secrets,
    usage_quota_service,
)
from api.models import ChatRequest
from api.sse import (
    _extract_document_preview_text,
    _sse_error_response,
    _stream_chat_response,
)

router = APIRouter()


@router.post("/api/chat")
async def chat(request: ChatRequest, req: Request = None, authorization: Optional[str] = Header(None)):
    """Stream a chat response via SSE using the shared chat pipeline."""
    _ip = ""
    cookie_token = None
    if req:
        _ip = (req.headers.get("x-forwarded-for", "").split(",")[0].strip()
               or (req.client.host if req.client else ""))
        cookie_token = read_auth_cookie(req)
    return _stream_chat_response(
        request, authorization=authorization, client_ip=_ip, cookie_token=cookie_token
    )


@router.post("/api/chat/upload")
async def chat_with_files(
    message: str = Form(""),
    conversation_id: Optional[str] = Form(None),
    model: Optional[str] = Form("gpt-oss-120b"),
    grounded_summary: bool = Form(False),
    web_search: bool = Form(True),
    files: PyList[UploadFile] = File(default=[]),
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """Stream a chat response with attached files (images/documents) via SSE."""
    auth_header = _safe_authorization_header(authorization)
    cookie_token = read_auth_cookie(http_request) if http_request else None
    current_user = _resolve_optional_user(auth_header, cookie_token=cookie_token)
    user_id = current_user.get("sub") if current_user else None
    if not user_id:
        return _sse_error_response(
            "Authentication required. Please sign in to your Quasar account to upload files."
        )
    selected_model = model or os.getenv("DEFAULT_LLM_MODEL", "gpt-oss-120b")
    provider = detect_provider(selected_model)
    try:
        upload_llm_context = _build_llm_context_for_user(user_id)
        upload_user_email = _current_user_email(current_user)
        upload_key_source = upload_llm_context["key_source_by_provider"].get(provider, "platform")
        usage_quota_service.ensure_allowed(
            user_id=user_id,
            user_email=upload_user_email,
            provider=provider,
            key_source=upload_key_source,
            byok_token_limit=upload_llm_context["byok_token_limits"].get(provider),
        )
    except QuotaExceededError as exc:
        return _sse_error_response(redact_secrets(exc))
    except ProviderKeyError as exc:
        return _sse_error_response(redact_secrets(exc))
    selected_provider_api_key = upload_llm_context["provider_api_keys"].get(provider)
    selected_provider_key_scope = ProviderFileService._key_scope(selected_provider_api_key)

    import base64

    logger.info(f"[UPLOAD] Received {len(files)} file(s), message={message[:80]!r}")

    image_contents = []
    enriched_text = message.strip()
    document_uploads: PyList[UploadedChatFile] = []
    mixed_document_previews: PyList[str] = []

    for f in files:
        filename = f.filename or "upload"
        content_type = (f.content_type or "").lower()
        raw = await f.read()

        if content_type.startswith("image/"):
            b64 = base64.b64encode(raw).decode("utf-8")
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:{content_type};base64,{b64}", "detail": "auto"},
            })
            continue

        if filename.lower().endswith(".fits") or filename.lower().endswith(".fit"):
            try:
                from services.fits_processing import FITSProcessingService

                metadata = FITSProcessingService.extract_metadata(raw)
                header_str = "\n".join([f"{k}: {v}" for k, v in metadata.items() if k != "error"])
                err = metadata.get("error", "")

                if hdrs := header_str.strip():
                    enriched_text += (
                        f"\n\n### Attached FITS: {filename}\n**Header Metadata:**\n"
                        f"```yaml\n{hdrs}\n```\n"
                    )
                if err:
                    enriched_text += f"\n[FITS Metadata Error: {err}]"

                b64_img = FITSProcessingService.generate_preview(raw)
                if b64_img:
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_img}", "detail": "high"},
                    })
                    enriched_text += (
                        "*(A 2D visual representation of this FITS file has been attached "
                        "as an image for analysis.)*\n"
                    )
                else:
                    enriched_text += "*(Could not generate a 2D preview image for this FITS data.)*\n"
            except Exception as e:
                enriched_text += f"\n\n[FITS: {filename} - extraction failed: {e}]"
            continue

        document_uploads.append(
            UploadedChatFile(
                filename=filename,
                mime_type=content_type or "application/octet-stream",
                data=raw,
            )
        )
        mixed_document_previews.append(_extract_document_preview_text(filename, content_type, raw))

    attachment_context = None
    if document_uploads:
        if provider in {"openai", "anthropic", "google"}:
            try:
                attachment_context = provider_file_service.prepare_files(
                    provider=provider,
                    model=selected_model,
                    files=document_uploads,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    provider_api_key=selected_provider_api_key,
                    provider_key_scope=selected_provider_key_scope,
                )
            except ProviderFileError as exc:
                return _sse_error_response(redact_secrets(exc))
        else:
            # Fallback for providers without native document upload support (e.g. DeepSeek)
            # Extract document text server-side and append it directly to the message prompt
            for preview in mixed_document_previews:
                enriched_text += preview

    if image_contents and model_accepts_direct_image_input(selected_model):
        attachment_context = attachment_context or {
            "provider": provider,
            "attachments": [],
            "messages": []
        }
        for img in image_contents:
            attachment_context["attachments"].append({
                "provider": provider,
                "type": "image_url",
                "image_url": img["image_url"]
            })
    elif image_contents:
        attachment_context = attachment_context or {
            "provider": provider,
            "attachments": [],
            "messages": []
        }
        attachment_context["image_prepass"] = {"images": image_contents}

    req = ChatRequest(
        message=enriched_text or message,
        conversation_id=conversation_id,
        model=selected_model,
        grounded_summary=grounded_summary,
        web_search=web_search,
    )
    return _stream_chat_response(
        req, authorization=auth_header, attachment_context=attachment_context,
        cookie_token=cookie_token,
    )
