"""Shared runtime dependencies for the QUASAR API package.

This is the central kernel that every router and the SSE pipeline import from:
the service singletons, auth/agent helpers, the executors, the human-in-the-loop
plan-feedback registry, the Langfuse trace registry, the model-visibility lists,
and the personalization DB accessor. It deliberately imports nothing from
``api.sse`` / ``api.routers`` / ``api.serializers`` so the import graph stays a
DAG (no circular imports).

Extracted from ``api/main.py`` during the P1 monolith split; behaviour identical.
"""

from api import bootstrap  # noqa: F401  (sys.path + Windows/py3.13 shims, run first)

import os
import threading as _threading
import queue as stdlib_queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path as _Path
from typing import Any, Dict, List, Optional

from fastapi import Header, HTTPException, Request

# ── Observability: Loguru + Sentry ────────────────────────────────────────────
from core.logger import logger

# ── Service layer imports (re-exported for routers / sse) ─────────────────────
from services.auth import AuthService
from services.conversation_service import ConversationService
from services.cube_workbench import (
    CubeWorkbenchForbidden,
    CubeWorkbenchNotFound,
    CubeWorkbenchService,
)
from services.provider_file_service import (
    ProviderFileError,
    ProviderFileService,
    UploadedChatFile,
)
from services.provider_key_service import ProviderKeyError, ProviderKeyService
from services.issue_report_service import ChatDeadline, IssueReportService
from services.spectral_line_explorer import (
    SpectralLineJobService,
    spectral_line_explorer_enabled,
)
from services.secret_redaction import redact_secrets
from services.usage_quota_service import QuotaExceededError, UsageQuotaService, UsageRecord
from services.admin_access import is_admin_email
from services.auth_cookie import read_auth_cookie
from services.analytics_service import AnalyticsService
from core.llm_client import (
    LLMClient,
    TACC_VISIBLE_MODEL_IDS,
    detect_provider,
    llm_request_context,
    model_accepts_direct_image_input,
)

# ── Service singletons ────────────────────────────────────────────────────────
auth_service = AuthService()
conversation_service = ConversationService()
cube_workbench_service = CubeWorkbenchService()
spectral_line_job_service = SpectralLineJobService()
provider_file_service = ProviderFileService()
provider_key_service = ProviderKeyService()
usage_quota_service = UsageQuotaService()
issue_report_service = IssueReportService()
analytics_service = AnalyticsService()


# ── Thread pools for running synchronous agent/storage calls ──────────────────
_executor = ThreadPoolExecutor(max_workers=2)  # Keep low to avoid OOM on 2GB instances
_chat_executor = ThreadPoolExecutor(
    max_workers=max(2, int(os.getenv("CHAT_WORKER_THREADS", "4")))
)
_storage_executor = ThreadPoolExecutor(max_workers=2)


# ── Plan Feedback Registry (Human-in-the-Loop) ────────────────────────────────
# Maps (conversation_key, run_id) → stdlib_queue.Queue for plan review blocking.
# When the Conductor emits a plan_review event, it blocks on the queue.
# The POST /api/plan-feedback endpoint pushes user responses into it.
# UIAPI-08: run-scoped keys — one conversation with two concurrent runs used to
# share a single conversation_id key, so registrations/cleanups clobbered each
# other and Approve clicks went undeliverable.
_plan_feedback_queues: Dict[tuple, stdlib_queue.Queue] = {}
_plan_feedback_lock = _threading.Lock()

# Map conversation_id -> latest_trace_id for score ingestion
_latest_traces: Dict[str, str] = {}
_last_trace_id: Optional[str] = None


# ── Model-visibility helpers + lists ──────────────────────────────────────────
def _visible_model_list(env_name: str, default_models: List[str]) -> List[str]:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return list(default_models)
    models = [part.strip() for part in raw.split(",") if part.strip()]
    return models or list(default_models)


def _unique_models(models: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for model in models:
        if model and model not in seen:
            seen.add(model)
            ordered.append(model)
    return ordered


OPENAI_VISIBLE_MODEL_IDS = _visible_model_list(
    "QUASAR_OPENAI_MODELS",
    ["gpt-5.4-mini", "gpt-4.1", "gpt-4o-mini"],
)
DEEPSEEK_VISIBLE_MODEL_IDS = _visible_model_list(
    "QUASAR_DEEPSEEK_MODELS",
    ["deepseek-v4-pro", "deepseek-v4-flash"],
)
TACC_MENU_MODEL_IDS = _visible_model_list(
    "QUASAR_TACC_MODELS",
    list(TACC_VISIBLE_MODEL_IDS),
)


def _anthropic_models_enabled() -> bool:
    """Whether Anthropic (Claude) models are surfaced in the model picker.

    Default: visible locally (development/testing) and hidden on the deployed
    production service, so Claude only shows up where it's intended to be used.
    Override explicitly with QUASAR_ENABLE_ANTHROPIC (1/0, true/false, yes/no,
    on/off) — e.g. set it to 1 in production once an ANTHROPIC_API_KEY is added.
    """
    override = os.getenv("QUASAR_ENABLE_ANTHROPIC", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    environment = (
        os.getenv("QUASAR_ENV")
        or os.getenv("APP_ENV")
        or os.getenv("ENVIRONMENT")
        or "development"
    ).strip().lower()
    return environment not in {"production", "prod"}


ANTHROPIC_DEFAULT_MODEL_IDS = [
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-haiku-4-5",
]
ANTHROPIC_VISIBLE_MODEL_IDS = (
    _visible_model_list("QUASAR_ANTHROPIC_MODELS", ANTHROPIC_DEFAULT_MODEL_IDS)
    if _anthropic_models_enabled()
    else []
)


# ── Auth / user helpers ───────────────────────────────────────────────────────
def _bearer_token(authorization: Any) -> Optional[str]:
    """Extract the token from an ``Authorization: Bearer <token>`` header."""
    if isinstance(authorization, str) and authorization.startswith("Bearer "):
        return authorization.split(" ", 1)[1]
    return None


def get_current_user(request: Request, authorization: Optional[str] = Header(None)):
    # Accept the JWT from the Authorization header (legacy Bearer clients) OR
    # the httpOnly auth cookie (S6 — token no longer lives in localStorage).
    #
    # An EXPLICIT credential must fail closed: if a Bearer header is present but
    # invalid we do NOT fall back to the ambient cookie. Otherwise a caller
    # presenting a low-privilege (or expired) Bearer would be silently executed
    # as whatever identity the browser's cookie happens to carry — a
    # confused-deputy / ambient-authority escalation.
    token = _bearer_token(authorization) or read_auth_cookie(request)
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized")
    payload = auth_service.verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload


def _current_user_email(current_user: Optional[dict]) -> str:
    if not current_user:
        return ""
    return str(current_user.get("email") or current_user.get("username") or "").strip().lower()


def _workbench_user_id(current_user: dict) -> str:
    return str(current_user.get("sub") or current_user.get("id") or "anonymous")


def _safe_authorization_header(authorization: Any) -> Optional[str]:
    return authorization if isinstance(authorization, str) else None


def _resolve_optional_user(
    authorization: Any, cookie_token: Optional[str] = None
) -> Optional[dict]:
    # Bearer header first, then the httpOnly cookie token (S6). An explicit but
    # invalid Bearer fails closed rather than falling back to the ambient cookie
    # (see get_current_user).
    token = _bearer_token(_safe_authorization_header(authorization)) or (cookie_token or None)
    if not token:
        return None
    try:
        return auth_service.verify_token(token)
    except Exception:
        return None


def _build_llm_context_for_user(user_id: str) -> Dict[str, Any]:
    metadata = provider_key_service.list_keys(user_id)
    api_keys = provider_key_service.decrypt_all_keys(user_id)
    providers = {"openai", "deepseek", "anthropic", "google", "tacc"} | set(api_keys.keys())
    key_sources = {provider: ("byok" if api_keys.get(provider) else "platform") for provider in providers}
    byok_limits = {
        item.get("provider"): item.get("token_limit")
        for item in metadata
        if item.get("provider")
    }
    return {
        "provider_api_keys": api_keys,
        "key_source_by_provider": key_sources,
        "byok_token_limits": byok_limits,
        "metadata": metadata,
    }


def _make_usage_recorder(user_id: str):
    def _record_usage(provider: str, model: str, key_source: str, input_tokens: int,
                      output_tokens: int, reservation_id: Optional[str] = None):
        usage_quota_service.record_usage(
            UsageRecord(
                user_id=user_id,
                provider=provider,
                model=model,
                key_source=key_source,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            reservation_id=reservation_id,
        )
    return _record_usage


def _make_quota_checker(user_id: str, user_email: str, byok_token_limits: Dict[str, Optional[int]]):
    """Per-LLM-call admission. Reserves, because this is the check with a settle
    point: the same call records its usage, so the tokens it holds are handed
    back. Pre-flight callers with no settle point must not reserve."""
    def _check_quota(provider: str, model: str, key_source: str) -> Optional[str]:
        return usage_quota_service.ensure_allowed(
            user_id=user_id,
            user_email=user_email,
            provider=provider,
            key_source=key_source,
            byok_token_limit=byok_token_limits.get(provider),
            reserve=True,
        )
    return _check_quota


def _make_quota_releaser():
    def _release_quota(reservation_id: str):
        usage_quota_service.release_reservation(reservation_id)
    return _release_quota


# ── Lazy-load the Quasar agent ────────────────────────────────────────────────
_agent = None
_agent_error = None


def get_agent():
    """Lazy-load QuasarAgent to avoid import errors during dev."""
    global _agent, _agent_error
    if _agent is None and _agent_error is None:
        try:
            from core.agent import QuasarAgent, AgentConfig
            config = AgentConfig()
            _agent = QuasarAgent(config)
            print("[INFO] QuasarAgent loaded successfully.")
        except Exception as e:
            import traceback
            _agent_error = traceback.format_exc()
            print(f"[WARN] Could not load QuasarAgent: {e}")
            traceback.print_exc()
            return None
    return _agent


def get_agent_error():
    return _agent_error


# ── Personalization metadata DB ───────────────────────────────────────────────
# Fallback path (used only in local dev mode). ``deps`` lives at
# ``<repo>/ui-pro/api/deps.py`` so ``parent.parent`` is ``<repo>/ui-pro``.
_PERS_DB_LOCAL = _Path(__file__).resolve().parent.parent / "data" / "personalization.db"


def _get_pers_db():
    """Return a connection to the personalization metadata DB (Turso cloud or local SQLite)."""
    from services.db import get_connection
    return get_connection(str(_PERS_DB_LOCAL))
