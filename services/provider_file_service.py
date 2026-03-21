"""
Provider-native document file handling for conversation-scoped uploads.

This service is intentionally separate from Qdrant, mem0, and personal RAG.
Uploaded chat attachments are runtime/model inputs, not long-term memory.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx

from core.llm_client import detect_provider
from services.conversation_service import ConversationService


OPENAI_EXTRA_MIME_TYPES = {
    "application/pdf",
    "application/json",
    "application/xml",
    "application/rtf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.oasis.opendocument.text",
    "text/csv",
    "text/tsv",
    "text/markdown",
    "text/html",
    "text/plain",
    "text/xml",
}

OPENAI_EXTRA_EXTENSIONS = {
    ".pdf", ".txt", ".text", ".md", ".markdown", ".json", ".html", ".htm",
    ".xml", ".csv", ".tsv", ".doc", ".docx", ".rtf", ".odt",
    ".ppt", ".pptx", ".xls", ".xlsx",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cc", ".cpp",
    ".cs", ".go", ".rs", ".scala", ".sql", ".sh", ".yaml", ".yml", ".toml",
}

ANTHROPIC_SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "text/plain",
}

GEMINI_SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "application/json",
    "application/xml",
    "text/plain",
    "text/markdown",
    "text/html",
    "text/xml",
    "text/csv",
    "text/tsv",
}

GEMINI_TEXT_ONLY_MIME_TYPES = GEMINI_SUPPORTED_MIME_TYPES - {"application/pdf"}


class ProviderFileError(Exception):
    """Raised when provider-native file preparation fails."""


@dataclass
class UploadedChatFile:
    filename: str
    mime_type: str
    data: bytes
    content_hash: str = ""


@dataclass
class PreparedAttachmentRef:
    filename: str
    mime_type: str
    content_hash: str
    provider_file_id: Optional[str] = None
    provider_file_name: Optional[str] = None
    provider_file_uri: Optional[str] = None


@dataclass
class PreparedAttachmentContext:
    provider: str
    attachments: List[Dict] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    provider_switch_required: bool = False
    provider_switch_message: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "provider": self.provider,
            "attachments": list(self.attachments),
            "messages": list(self.messages),
            "provider_switch_required": self.provider_switch_required,
            "provider_switch_message": self.provider_switch_message,
        }


class ProviderFileService:
    """Uploads supported documents to provider file APIs and manages reuse."""

    def __init__(self):
        self._conversation_service = ConversationService()
        self._anonymous_refs: Dict[str, Dict[str, List[Dict]]] = {}
        self._lock = threading.Lock()

    def prepare_files(
        self,
        provider: str,
        model: str,
        files: List[UploadedChatFile],
        conversation_id: Optional[str],
        user_id: Optional[str],
    ) -> Dict:
        provider = provider or detect_provider(model or "")
        if provider not in {"openai", "anthropic", "google"}:
            raise ProviderFileError(
                f"Document uploads are not supported for provider '{provider}'."
            )

        if not files:
            return self.get_active_files(provider, conversation_id, user_id)

        normalized_files = [self._normalize_file(file) for file in files]
        existing_refs = self._load_refs(conversation_id, provider, user_id)
        existing_by_hash = {ref["content_hash"]: ref for ref in existing_refs}

        notes: List[str] = []
        reused_count = 0
        uploaded_count = 0

        for upload in normalized_files:
            self._validate_provider_support(provider, upload)
            if upload.content_hash in existing_by_hash:
                reused_count += 1
                continue

            uploaded_ref = self._upload_file(provider, upload)
            record = {
                "provider": provider,
                "model_family": self._model_family(model),
                "filename": upload.filename,
                "mime_type": upload.mime_type,
                "content_hash": upload.content_hash,
                "provider_file_id": uploaded_ref.provider_file_id,
                "provider_file_name": uploaded_ref.provider_file_name,
                "provider_file_uri": uploaded_ref.provider_file_uri,
            }
            self._save_ref(conversation_id, user_id, record)
            existing_by_hash[upload.content_hash] = record
            uploaded_count += 1

        if reused_count:
            notes.append(f"Reusing {reused_count} previously uploaded document(s).")
        if uploaded_count:
            scope_label = "conversation" if conversation_id else "request"
            notes.append(f"Attached {uploaded_count} document(s) to this {scope_label}.")

        if not conversation_id:
            current_refs = list(existing_by_hash.values())
            attachment_dicts = [self._ref_to_attachment(provider, ref) for ref in current_refs]
            if provider == "google" and any(
                ref["mime_type"] in GEMINI_TEXT_ONLY_MIME_TYPES for ref in current_refs
            ):
                notes.append("Gemini will read non-PDF documents as text only in this request.")
            return PreparedAttachmentContext(
                provider=provider,
                attachments=attachment_dicts,
                messages=notes,
            ).to_dict()

        current_context = self.get_active_files(provider, conversation_id, user_id)
        current_context["messages"] = notes + current_context.get("messages", [])
        return current_context

    def get_active_files(
        self,
        provider: str,
        conversation_id: Optional[str],
        user_id: Optional[str],
    ) -> Dict:
        provider = provider or "openai"
        if not conversation_id:
            return PreparedAttachmentContext(provider=provider).to_dict()

        refs = self._load_refs(conversation_id, provider, user_id)
        other_refs = [
            ref for ref in self._load_refs(conversation_id, None, user_id)
            if ref.get("provider") != provider
        ]

        messages: List[str] = []
        if refs:
            attachment_dicts = [self._ref_to_attachment(provider, ref) for ref in refs]
            if provider == "google" and any(
                ref["mime_type"] in GEMINI_TEXT_ONLY_MIME_TYPES for ref in refs
            ):
                messages.append(
                    "Gemini will read non-PDF documents as text only in this conversation."
                )
            return PreparedAttachmentContext(
                provider=provider,
                attachments=attachment_dicts,
                messages=messages,
            ).to_dict()

        if other_refs:
            other_provider = other_refs[0]["provider"]
            message = (
                f"This conversation has active uploaded documents for provider "
                f"'{other_provider}'. Re-upload the document(s) after switching to '{provider}'."
            )
            return PreparedAttachmentContext(
                provider=provider,
                messages=[message],
                provider_switch_required=True,
                provider_switch_message=message,
            ).to_dict()

        return PreparedAttachmentContext(provider=provider).to_dict()

    def clear_active_files(
        self,
        provider: str,
        conversation_id: Optional[str],
        user_id: Optional[str],
    ):
        if not conversation_id:
            return
        if user_id:
            self._conversation_service.clear_conversation_file_refs(user_id, conversation_id, provider)
            return
        with self._lock:
            provider_map = self._anonymous_refs.get(conversation_id, {})
            provider_map.pop(provider, None)
            if provider_map:
                self._anonymous_refs[conversation_id] = provider_map
            else:
                self._anonymous_refs.pop(conversation_id, None)

    def _normalize_file(self, file: UploadedChatFile) -> UploadedChatFile:
        filename = file.filename or "document"
        mime_type = (file.mime_type or "").strip().lower()
        if not mime_type or mime_type == "application/octet-stream":
            guessed, _ = mimetypes.guess_type(filename)
            mime_type = (guessed or "application/octet-stream").lower()

        data = file.data or b""
        if not data:
            raise ProviderFileError(f"Uploaded file '{filename}' is empty.")

        return UploadedChatFile(
            filename=filename,
            mime_type=mime_type,
            data=data,
            content_hash=self._content_hash(data),
        )

    def _validate_provider_support(self, provider: str, file: UploadedChatFile):
        ext = os.path.splitext(file.filename.lower())[1]
        mime_type = file.mime_type

        if provider == "openai":
            if mime_type.startswith("text/"):
                return
            if mime_type in OPENAI_EXTRA_MIME_TYPES or ext in OPENAI_EXTRA_EXTENSIONS:
                return
            raise ProviderFileError(
                f"OpenAI native document upload does not support '{file.filename}' ({mime_type})."
            )

        if provider == "anthropic":
            if mime_type in ANTHROPIC_SUPPORTED_MIME_TYPES:
                return
            raise ProviderFileError(
                f"Claude native document upload currently supports PDFs and plain text only. "
                f"Received '{file.filename}' ({mime_type})."
            )

        if provider == "google":
            if mime_type in GEMINI_SUPPORTED_MIME_TYPES or mime_type.startswith("text/"):
                return
            raise ProviderFileError(
                f"Gemini native document upload does not support '{file.filename}' ({mime_type}) "
                f"in this path. Convert it to PDF or plain text and try again."
            )

        raise ProviderFileError(f"Unknown provider '{provider}'.")

    def _upload_file(self, provider: str, file: UploadedChatFile) -> PreparedAttachmentRef:
        if provider == "openai":
            return self._upload_openai_file(file)
        if provider == "anthropic":
            return self._upload_anthropic_file(file)
        if provider == "google":
            return self._upload_google_file(file)
        raise ProviderFileError(f"Unknown provider '{provider}'.")

    def _upload_openai_file(self, file: UploadedChatFile) -> PreparedAttachmentRef:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ProviderFileError("OPENAI_API_KEY is not configured on the server.")

        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                "https://api.openai.com/v1/files",
                headers={"Authorization": f"Bearer {api_key}"},
                data={"purpose": "user_data"},
                files={"file": (file.filename, file.data, file.mime_type)},
            )
        payload = self._raise_for_provider_error("OpenAI", response)
        return PreparedAttachmentRef(
            filename=file.filename,
            mime_type=file.mime_type,
            content_hash=file.content_hash,
            provider_file_id=payload.get("id"),
            provider_file_name=payload.get("filename") or payload.get("id"),
        )

    def _upload_anthropic_file(self, file: UploadedChatFile) -> PreparedAttachmentRef:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ProviderFileError("ANTHROPIC_API_KEY is not configured on the server.")

        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                "https://api.anthropic.com/v1/files",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "files-api-2025-04-14",
                },
                data={"purpose": "user_data"},
                files={"file": (file.filename, file.data, file.mime_type)},
            )
        payload = self._raise_for_provider_error("Anthropic", response)
        return PreparedAttachmentRef(
            filename=file.filename,
            mime_type=file.mime_type,
            content_hash=file.content_hash,
            provider_file_id=payload.get("id"),
            provider_file_name=payload.get("filename") or payload.get("id"),
        )

    def _upload_google_file(self, file: UploadedChatFile) -> PreparedAttachmentRef:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ProviderFileError("GEMINI_API_KEY is not configured on the server.")

        start_headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(file.data)),
            "X-Goog-Upload-Header-Content-Type": file.mime_type,
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=120.0) as client:
            start_response = client.post(
                f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={api_key}",
                headers=start_headers,
                json={"file": {"display_name": file.filename}},
            )
            self._raise_for_provider_error("Gemini", start_response)
            upload_url = start_response.headers.get("x-goog-upload-url")
            if not upload_url:
                raise ProviderFileError("Gemini file upload did not return an upload URL.")

            finalize_response = client.post(
                upload_url,
                headers={
                    "Content-Length": str(len(file.data)),
                    "X-Goog-Upload-Offset": "0",
                    "X-Goog-Upload-Command": "upload, finalize",
                },
                content=file.data,
            )
            finalize_payload = self._raise_for_provider_error("Gemini", finalize_response)
            file_payload = finalize_payload.get("file", finalize_payload)

            file_name = file_payload.get("name")
            if file_name:
                file_payload = self._wait_for_google_file(client, api_key, file_name)

        return PreparedAttachmentRef(
            filename=file.filename,
            mime_type=file.mime_type,
            content_hash=file.content_hash,
            provider_file_name=file_payload.get("name"),
            provider_file_uri=file_payload.get("uri"),
        )

    def _wait_for_google_file(self, client: httpx.Client, api_key: str, file_name: str) -> Dict:
        file_payload: Dict = {}
        for _ in range(10):
            response = client.get(
                f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={api_key}"
            )
            payload = self._raise_for_provider_error("Gemini", response)
            file_payload = payload.get("file", payload)
            state = file_payload.get("state")
            state_name = state.get("name") if isinstance(state, dict) else str(state or "")
            if not state_name or state_name == "ACTIVE":
                return file_payload
            if state_name == "FAILED":
                raise ProviderFileError(
                    f"Gemini rejected '{file_payload.get('displayName') or file_name}'."
                )
            time.sleep(1.0)
        return file_payload

    def _raise_for_provider_error(self, provider_name: str, response: httpx.Response) -> Dict:
        if response.is_success:
            if response.content:
                return response.json()
            return {}
        detail = response.text
        try:
            payload = response.json()
            detail = (
                payload.get("error", {}).get("message")
                or payload.get("message")
                or payload.get("detail")
                or detail
            )
        except Exception:
            pass
        raise ProviderFileError(f"{provider_name} file upload failed: {detail}")

    def _ref_to_attachment(self, provider: str, ref: Dict) -> Dict:
        attachment = {
            "provider": provider,
            "filename": ref["filename"],
            "mime_type": ref["mime_type"],
            "content_hash": ref["content_hash"],
        }

        if provider == "openai":
            attachment["kind"] = "openai_input_file"
            attachment["file_id"] = ref.get("provider_file_id")
            return attachment

        if provider == "anthropic":
            attachment["kind"] = "anthropic_document_file"
            attachment["file_id"] = ref.get("provider_file_id")
            return attachment

        if provider == "google":
            attachment["kind"] = "gemini_file"
            attachment["file_name"] = ref.get("provider_file_name")
            attachment["file_uri"] = ref.get("provider_file_uri")
            return attachment

        return attachment

    def _load_refs(
        self,
        conversation_id: Optional[str],
        provider: Optional[str],
        user_id: Optional[str],
    ) -> List[Dict]:
        if not conversation_id:
            return []

        if user_id:
            return self._conversation_service.list_conversation_file_refs(
                user_id=user_id,
                conversation_id=conversation_id,
                provider=provider,
            )

        with self._lock:
            provider_map = self._anonymous_refs.get(conversation_id, {})
            if provider:
                return [dict(ref) for ref in provider_map.get(provider, [])]

            refs: List[Dict] = []
            for provider_refs in provider_map.values():
                refs.extend(dict(ref) for ref in provider_refs)
            return refs

    def _save_ref(self, conversation_id: Optional[str], user_id: Optional[str], ref: Dict):
        if not conversation_id:
            return

        if user_id:
            self._conversation_service.save_conversation_file_ref(
                user_id=user_id,
                conversation_id=conversation_id,
                provider=ref["provider"],
                model_family=ref["model_family"],
                filename=ref["filename"],
                mime_type=ref["mime_type"],
                content_hash=ref["content_hash"],
                provider_file_id=ref.get("provider_file_id"),
                provider_file_name=ref.get("provider_file_name"),
                provider_file_uri=ref.get("provider_file_uri"),
            )
            return

        with self._lock:
            provider_map = self._anonymous_refs.setdefault(conversation_id, {})
            refs = provider_map.setdefault(ref["provider"], [])
            existing_index = next(
                (index for index, item in enumerate(refs) if item["content_hash"] == ref["content_hash"]),
                None,
            )
            stored_ref = dict(ref)
            if existing_index is None:
                refs.append(stored_ref)
            else:
                refs[existing_index] = stored_ref

    def _content_hash(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _model_family(self, model: str) -> str:
        if not model:
            return "unknown"
        if model.startswith("gpt-") or model.startswith("o"):
            return model.split("-")[0]
        if model.startswith("claude-") or model.startswith("gemini-"):
            return model.split("-")[0]
        return detect_provider(model)
