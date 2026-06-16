"""
QUASAR Professional Chat Interface — FastAPI Backend

Provides RESTful + SSE endpoints wrapping the existing QuasarAgent.
Run with:  uvicorn api.main:app --reload --port 8000
"""

import sys, os, json, asyncio, uuid, re
import queue as stdlib_queue

# ── Windows fix: langchain_community.document_loaders.pebblo imports 'pwd' (Unix-only) ──
if sys.platform == "win32" and "pwd" not in sys.modules:
    import types
    _pwd_stub = types.ModuleType("pwd")
    _pwd_stub.getpwuid = lambda uid: type("pw", (), {"pw_name": "user"})()
    sys.modules["pwd"] = _pwd_stub

# ── Python 3.13 fix for 'cgi' module missing (needed by older pyvo) ──
if "cgi" not in sys.modules:
    import email.message
    import types
    _cgi_stub = types.ModuleType("cgi")
    def _parse_header(line):
        m = email.message.EmailMessage()
        m['content-type'] = line
        return m.get_content_type(), m.get_params() or {}
    _cgi_stub.parse_header = _parse_header
    sys.modules["cgi"] = _cgi_stub
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List as PyList
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

# Add project root to path so we can import Quasar modules
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load .env from project root so OPENAI_API_KEY etc. are available
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from utils.archive_links import build_archive_link, infer_archive_kind
from services.evidence_quality import annotate_web_source_evidence, choose_better_evidence_quality, rank_web_sources

# ── Observability: Loguru + Sentry ────────────────────────────────────────────
from core.logger import logger, init_rollbar
init_rollbar()  # no-op if ROLLBAR_ACCESS_TOKEN env var is not set
logger.info("[QUASAR API] Starting up")

_LAB_SCIENCE_EMOJI_RE = re.compile(
    f"(?:{chr(0x1F9D1)}\u200d)?{chr(0x1F52C)}\\s*"
)


def _sanitize_assistant_text(text: str) -> str:
    return _LAB_SCIENCE_EMOJI_RE.sub("", text or "")


def _normalize_qa2_value(value: Any) -> str:
    """Normalize archive-table QA2 values for the existing UI badge."""
    if isinstance(value, bool):
        return "Pass" if value else "Fail"
    if value is None:
        return "Unknown"

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>", "-", "--", "unknown"}:
        return "Unknown"

    compact = re.sub(r"[\s_-]+", "", text).lower()
    if compact in {"t", "true", "y", "yes", "1", "pass", "passed", "qa2pass", "qa2passed"}:
        return "Pass"
    if compact in {"f", "false", "n", "no", "0", "fail", "failed", "qa2fail", "qa2failed"}:
        return "Fail"
    if compact in {"semipass", "semipassed", "qa2semipass", "qa2semipassed"}:
        return "SemiPass"
    if "semipass" in compact:
        return "SemiPass"
    if "fail" in compact:
        return "Fail"
    if "pass" in compact:
        return "Pass"
    return text


def _qa2_status_from_table(df: Any) -> Optional[Any]:
    """Return QA2 status from table columns; do not derive it from report PDFs."""
    for column in ("qa2_status", "QA2", "qa2_passed", "qa2Passed", "qa2"):
        if hasattr(df, "columns") and column in df.columns:
            statuses = df[column].map(_normalize_qa2_value)
            if statuses.astype(str).str.lower().ne("unknown").any():
                return statuses
    return None

app = FastAPI(
    title="QUASAR API",
    description="Backend API for the QUASAR Professional Chat Interface",
    version="2.0.0",
)

try:
    from rollbar.contrib.fastapi import add_to as rollbar_add_to
    rollbar_add_to(app)
except Exception as e:
    logger.warning(f"Could not attach Rollbar to FastAPI: {e}")

# NOTE: CORSMiddleware is added AFTER LoggingMiddleware below (line ~115)
# so it becomes the OUTERMOST middleware (Starlette uses LIFO order).
# This ensures CORS headers are always present, even on error responses.

# ── Startup: ensure all DB tables/collections exist ────────────────────────────
@app.on_event("startup")
def _init_all_tables():
    """Create SQL tables that may not exist yet (runs once on boot)."""
    try:
        from services.db import get_connection
        conn = get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                uploaded_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        logger.info("[STARTUP] Personalization documents table ready")
    except Exception as e:
        logger.warning(f"[STARTUP] Personalization table init failed: {e}")

# ── Request / response logging middleware ─────────────────────────────────────
import time as _time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        t0 = _time.perf_counter()
        response = await call_next(request)
        elapsed = _time.perf_counter() - t0
        logger.info(
            f"{request.method} {request.url.path} → {response.status_code} "
            f"({elapsed*1000:.0f}ms)"
        )
        return response

app.add_middleware(LoggingMiddleware)

# CORS — added LAST so it is the OUTERMOST middleware (Starlette LIFO).
# This guarantees every response (including errors) carries CORS headers.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.vercel\.app|https://.*\.onrender\.com|https://(www\.)?quasarassistant\.com|http://localhost:300[0-9]|http://127\.0\.0\.1:300[0-9]",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static file serving for rendered FITS images ─────────────────
import os as _os
from fastapi.staticfiles import StaticFiles
_RENDERED_IMAGES_DIR = _os.path.join(_os.path.dirname(__file__), "..", "..", "data", "rendered_images")
_os.makedirs(_RENDERED_IMAGES_DIR, exist_ok=True)
app.mount("/api/images", StaticFiles(directory=_RENDERED_IMAGES_DIR), name="rendered_images")

# ── Plan Feedback Registry (Human-in-the-Loop) ────────────────────────────────
# Maps conversation_id → stdlib_queue.Queue for plan review blocking.
# When the Conductor emits a plan_review event, it blocks on the queue.
# The POST /api/plan-feedback endpoint pushes user responses into it.
import threading as _threading
_plan_feedback_queues: Dict[str, stdlib_queue.Queue] = {}
_plan_feedback_lock = _threading.Lock()

# Map conversation_id -> latest_trace_id for score ingestion
_latest_traces: Dict[str, str] = {}
_last_trace_id: Optional[str] = None

# ── Catch-all exception handler — ensures a proper JSON 500 with CORS headers ──
from starlette.responses import JSONResponse

@app.exception_handler(Exception)
async def _global_exception_handler(request, exc):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {redact_secrets(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": redact_secrets(exc)},
    )


# ── Demographics & FITS estimation helper ────────────────────
def _compute_demographics(df) -> tuple:
    """Compute distribution data, FITS estimate, and sky coordinates from a search DataFrame.
    Returns (demographics_dict, fits_estimate_int).
    demographics_dict may include a 'skyCoords' key with [{ra, dec}, ...] entries.
    """
    import math
    demographics: Dict[str, Any] = {}

    if "band_list" in df.columns:
        band_counts = df["band_list"].astype(str).value_counts().head(8)
        demographics["bands"] = {str(k): int(v) for k, v in band_counts.items()}

    if "project_code" in df.columns:
        proj_counts = df["project_code"].value_counts().head(6)
        demographics["projects"] = {str(k): int(v) for k, v in proj_counts.items()}
    elif "proposal_id" in df.columns:
        proj_counts = df["proposal_id"].value_counts().head(6)
        demographics["projects"] = {str(k): int(v) for k, v in proj_counts.items()}

    # Multi-telescope support (CADC results have obs_collection)
    if "obs_collection" in df.columns:
        tel_counts = df["obs_collection"].value_counts().head(8)
        demographics["telescopes"] = {str(k): int(v) for k, v in tel_counts.items()}

    if "instrument_name" in df.columns:
        inst_counts = df["instrument_name"].value_counts().head(6)
        demographics["instruments"] = {str(k): int(v) for k, v in inst_counts.items()}

    # ── Sky coordinates for sky-map widget ──────────────────────
    ra_col = next((c for c in ["s_ra", "ra"] if c in df.columns), None)
    dec_col = next((c for c in ["s_dec", "dec"] if c in df.columns), None)
    if ra_col and dec_col:
        coords = []
        for _, row in df.head(10000).iterrows():
            try:
                ra_v = float(row[ra_col])
                dec_v = float(row[dec_col])
                # Filter out NaN and zero-coordinate rows (invalid positions)
                if math.isnan(ra_v) or math.isnan(dec_v):
                    continue
                if abs(ra_v) < 0.001 and abs(dec_v) < 0.001:
                    continue
                coords.append({"ra": round(ra_v, 4), "dec": round(dec_v, 4)})
            except (ValueError, TypeError):
                pass
        if coords:
            demographics["skyCoords"] = coords

    # ── Observation year timeline ────────────────────────────────
    if "obs_release_date" in df.columns:
        try:
            years = pd.to_datetime(df["obs_release_date"], errors="coerce").dt.year.dropna()
            if not years.empty:
                year_counts = years.astype(int).value_counts().sort_index()
                demographics["observationYears"] = {str(k): int(v) for k, v in year_counts.items()}
        except Exception:
            pass

    # ── Science categories ───────────────────────────────────────
    if "scientific_category" in df.columns:
        try:
            cat_counts = df["scientific_category"].dropna().value_counts().head(6)
            if not cat_counts.empty and len(cat_counts) > 1:
                demographics["scienceCategories"] = {str(k): int(v) for k, v in cat_counts.items()}
        except Exception:
            pass

    # ── Angular resolution distribution ──────────────────────────
    res_col = next((c for c in ["spatial_resolution", "s_resolution"] if c in df.columns), None)
    if res_col:
        try:
            resolutions = pd.to_numeric(df[res_col], errors="coerce").dropna()
            if not resolutions.empty and len(resolutions) > 2:
                bins = [0, 0.1, 0.5, 1.0, 5.0, float("inf")]
                labels = ['<0.1"', '0.1-0.5"', '0.5-1"', '1-5"', '>5"']
                binned = pd.cut(resolutions, bins=bins, labels=labels)
                res_counts = binned.value_counts()
                res_dict = {str(k): int(v) for k, v in res_counts.items() if v > 0}
                if len(res_dict) > 1:
                    demographics["resolutionBins"] = res_dict
        except Exception:
            pass

    # FITS file estimation
    fits_estimate = 0
    if "member_ous_uid" in df.columns:
        unique_mous = df["member_ous_uid"].dropna().nunique()
        fits_estimate = unique_mous * 5  # ~5 FITS products per MOUS (conservative)
    elif "obs_publisher_did" in df.columns:
        fits_estimate = int(df["obs_publisher_did"].dropna().nunique())
    elif len(df) > 0:
        fits_estimate = len(df) * 3  # Generic estimate for non-ALMA archives

    return demographics, fits_estimate


def _default_archive_source(df, source_hint: str = "", filter_label: str = "") -> str:
    """Return the label shown above a data card when the tool did not provide one."""
    if filter_label:
        return filter_label
    if source_hint:
        return source_hint

    archive_kind = infer_archive_kind(df, source_hint=source_hint, filter_label=filter_label)
    if archive_kind == "cadc":
        if hasattr(df, "columns") and "obs_collection" in df.columns:
            collections = []
            for value in df["obs_collection"].dropna().astype(str):
                text = value.strip()
                if text and text not in collections:
                    collections.append(text)
                if len(collections) >= 3:
                    break
            if collections:
                return " / ".join(collections)
        return "CADC"
    if archive_kind == "alma":
        return "ALMA Archive"
    if archive_kind == "mast":
        return "MAST"
    if archive_kind == "eso":
        return "ESO"
    if archive_kind == "irsa":
        return "IRSA"
    return "Archive"


# ── CADC DataLink preview URL fetcher ─────────────────────────
_CADC_DATALINK_URL = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/caom2ops/datalink"

def _fetch_cadc_preview_urls(obs_publisher_dids: List[str], max_ids: int = 200) -> Dict[str, str]:
    """Batch-query CADC DataLink for preview image URLs.

    Sends a single HTTP request with multiple IDs and parses the VOTable
    response to extract rows with semantics=#preview or #thumbnail.

    Returns {obs_publisher_did: preview_access_url} mapping.
    Falls back to empty dict on any error (non-blocking).
    """
    import urllib.request
    import urllib.parse
    import xml.etree.ElementTree as ET

    ids = obs_publisher_dids[:max_ids]
    if not ids:
        return {}

    try:
        params = "&".join(f"ID={urllib.parse.quote(oid, safe='')}" for oid in ids)
        url = f"{_CADC_DATALINK_URL}?{params}"
        req = urllib.request.Request(url, headers={"Accept": "application/x-votable+xml"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = resp.read()

        # Parse VOTable XML — extract ID, semantics, access_url columns
        root = ET.fromstring(data)
        ns = {"vo": "http://www.ivoa.net/xml/VOTable/v1.3"}

        # Find FIELD column indices
        table = root.find(".//vo:TABLE", ns) or root.find(".//TABLE")
        if table is None:
            # Try without namespace
            ns = {}
            table = root.find(".//TABLE")
        if table is None:
            return {}

        fields = table.findall("vo:FIELD", ns) if ns else table.findall("FIELD")
        col_names = [f.get("name", "").lower() for f in fields]

        id_idx = next((i for i, n in enumerate(col_names) if n == "id"), None)
        sem_idx = next((i for i, n in enumerate(col_names) if n == "semantics"), None)
        url_idx = next((i for i, n in enumerate(col_names) if n == "access_url"), None)

        if id_idx is None or sem_idx is None or url_idx is None:
            return {}

        preview_map: Dict[str, str] = {}
        data_el = table.find("vo:DATA", ns) if ns else table.find("DATA")
        if data_el is None:
            return {}
        tabledata = data_el.find("vo:TABLEDATA", ns) if ns else data_el.find("TABLEDATA")
        if tabledata is None:
            return {}

        for tr in (tabledata.findall("vo:TR", ns) if ns else tabledata.findall("TR")):
            tds = tr.findall("vo:TD", ns) if ns else tr.findall("TD")
            if len(tds) <= max(id_idx, sem_idx, url_idx):
                continue
            obs_id = (tds[id_idx].text or "").strip()
            semantics = (tds[sem_idx].text or "").strip().lower()
            access = (tds[url_idx].text or "").strip()

            if semantics in ("#preview", "#thumbnail") and access:
                # Prefer #thumbnail (smaller), but #preview is fine too
                if obs_id not in preview_map or semantics == "#thumbnail":
                    preview_map[obs_id] = access

        return preview_map

    except Exception as e:
        print(f"[INFO] CADC DataLink preview fetch skipped: {e}")
        return {}


# Thread pool for running synchronous agent calls
_executor = ThreadPoolExecutor(max_workers=2)  # Keep low to avoid OOM on 2GB instances

# ── Channel Routers ──────────────────────────────────────────
try:
    from api.channels.telegram import router as telegram_router
    app.include_router(telegram_router)
    print("[INFO] Telegram channel router registered at /channels/telegram")
except ImportError as e:
    print(f"[WARNING] Telegram channel not loaded: {e}")

try:
    from api.channels.whatsapp import router as whatsapp_router
    app.include_router(whatsapp_router)
    print("[INFO] WhatsApp channel router registered at /channels/whatsapp")
except ImportError as e:
    print(f"[WARNING] WhatsApp channel not loaded: {e}")


# ── Models ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    model: Optional[str] = "gpt-4o"
    grounded_summary: bool = False
    web_search: bool = True

class RegisterRequest(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    display_name: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class GoogleLoginRequest(BaseModel):
    credential: str

class FitsPreviewRequest(BaseModel):
    url: str
    filename: Optional[str] = None

class WorkbenchSessionCreate(BaseModel):
    source_url: str
    filename: Optional[str] = None
    project_code: Optional[str] = None
    mous_uid: Optional[str] = None

class WorkbenchRenderRequest(BaseModel):
    mode: Optional[str] = "image"
    channel: Optional[int] = None
    moment: Optional[int] = None
    colormap: Optional[str] = "inferno"
    stretch: Optional[str] = "asinh"
    contour_sigma: Optional[List[float]] = None
    rms_region: Optional[Dict[str, float]] = None

class WorkbenchPrepareRequest(BaseModel):
    max_bytes: Optional[int] = None
    user_cache_bytes: Optional[int] = None
    cache_ttl_seconds: Optional[int] = None
    force: Optional[bool] = False

class WorkbenchSpectrumRequest(BaseModel):
    x_pixel: Optional[float] = None
    y_pixel: Optional[float] = None
    aperture_radius_pixels: Optional[float] = 3.0
    aperture_radius_arcsec: Optional[float] = None
    max_points: Optional[int] = 512

class WorkbenchPvSliceRequest(BaseModel):
    path: Optional[List[Dict[str, float]]] = None
    width_pixels: Optional[float] = 3.0
    max_points: Optional[int] = 512

class WorkbenchLineOverlayRequest(BaseModel):
    observed_frequency_ghz: Optional[float] = None
    line_preset_key: Optional[str] = None
    redshift: Optional[float] = 0.0
    tolerance_ghz: Optional[float] = 0.01
    top_n: Optional[int] = 8

class WorkbenchExportRequest(BaseModel):
    formats: Optional[List[str]] = None

class WorkbenchJobStartRequest(BaseModel):
    operation: str
    payload: Optional[Dict[str, Any]] = None

class ProviderKeySaveRequest(BaseModel):
    provider: str
    api_key: str
    token_limit: Optional[int] = None

class ProviderKeyLimitRequest(BaseModel):
    token_limit: Optional[int] = None

# ── Auth Service Instance ──
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
from services.secret_redaction import redact_secrets
from services.usage_quota_service import QuotaExceededError, UsageQuotaService, UsageRecord
from core.llm_client import LLMClient, detect_provider, llm_request_context, model_accepts_direct_image_input
auth_service = AuthService()
conversation_service = ConversationService()
cube_workbench_service = CubeWorkbenchService()
provider_file_service = ProviderFileService()
provider_key_service = ProviderKeyService()
usage_quota_service = UsageQuotaService()

from services.analytics_service import AnalyticsService
analytics_service = AnalyticsService()

def get_current_user(authorization: Optional[str] = Header(None)):
    if not isinstance(authorization, str) or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ")[1]
    payload = auth_service.verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload


def _current_user_email(current_user: Optional[dict]) -> str:
    if not current_user:
        return ""
    return str(current_user.get("email") or current_user.get("username") or "").strip().lower()


def _build_llm_context_for_user(user_id: str) -> Dict[str, Any]:
    metadata = provider_key_service.list_keys(user_id)
    api_keys = provider_key_service.decrypt_all_keys(user_id)
    providers = {"openai", "deepseek", "anthropic", "google"} | set(api_keys.keys())
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
    def _record_usage(provider: str, model: str, key_source: str, input_tokens: int, output_tokens: int):
        usage_quota_service.record_usage(
            UsageRecord(
                user_id=user_id,
                provider=provider,
                model=model,
                key_source=key_source,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )
    return _record_usage


def _make_quota_checker(user_id: str, user_email: str, byok_token_limits: Dict[str, Optional[int]]):
    def _check_quota(provider: str, model: str, key_source: str):
        usage_quota_service.ensure_allowed(
            user_id=user_id,
            user_email=user_email,
            provider=provider,
            key_source=key_source,
            byok_token_limit=byok_token_limits.get(provider),
        )
    return _check_quota


class ConversationCreate(BaseModel):
    title: Optional[str] = "New Chat"
    model: Optional[str] = None


# ── Lazy-load the Quasar agent ───────────────────────────────

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


def _safe_authorization_header(authorization: Any) -> Optional[str]:
    return authorization if isinstance(authorization, str) else None


def _resolve_optional_user(authorization: Any) -> Optional[dict]:
    auth_header = _safe_authorization_header(authorization)
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    try:
        return auth_service.verify_token(auth_header.split(" ", 1)[1])
    except Exception:
        return None


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
        url = _normalize_web_url(item.get("url") or item.get("link") or item.get("href") or item.get("source_url"))
        if not url:
            continue
        key = url.lower().rstrip("/")
        clean = dict(item)
        clean["url"] = url
        if key in by_url:
            current = merged[by_url[key]]
            for field in ("title", "snippet", "description"):
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


def _build_data_card_event(_run_result: dict) -> Optional[tuple]:
    """Build a data card SSE event from a run_result dict.
    
    Returns (sse_event_str, rich_dt_dict) or None if the result can't be serialized.
    This is extracted so it can be called both during streaming (eager)
    and after streaming (fallback), avoiding the 2-4s delay.
    """
    import math
    import pandas as pd

    result_type = _run_result.get("type", "")
    if result_type != "data":
        return None

    df = _run_result.get("data")
    if df is None or not hasattr(df, "to_dict"):
        return None

    try:
        table_kind = str(_run_result.get("table_kind", "") or "")
        _source = _run_result.get("source", "")
        _filter_label = _run_result.get("filter_label", "")
        archive_kind = infer_archive_kind(df, source_hint=_source, filter_label=_filter_label)
        if archive_kind == "alma" and table_kind != "alma_project_picker":
            qa2_status = _qa2_status_from_table(df)
            if qa2_status is not None:
                # Copy before adding UI-only normalization so we do not mutate agent-owned results.
                df = df.copy()
                df["qa2_status"] = qa2_status
        product_display_cols = [
            ("filename", "File"),
            ("product_kind", "Product"),
            ("size_mb", "Size (MB)"),
            ("proposal_id", "Proposal ID"),
            ("target_name", "Target"),
            ("scan_intent", "Scan Intent"),
            ("member_ous_uid", "MOUS ID"),
            ("qa2_status", "QA2"),
            ("qa2_passed", "QA2"),
            ("triage_status", "Triage"),
            ("readiness_score", "Readiness"),
            ("warnings", "Warnings"),
        ]
        project_picker_cols = [
            ("proposal_id", "Proposal ID"),
            ("target_name", "Target"),
            ("band_list", "Band"),
            ("observations", "Observations"),
            ("member_ous_count", "MOUS Count"),
            ("dataproduct_type", "Type"),
            ("pi_name", "PI"),
            ("obs_title", "Project Title"),
            ("obs_release_date", "Release Date"),
        ]
        alma_display_cols = [
            ("obs_publisher_did", "Project"),
            ("target_name", "Target"),
            ("scan_intent", "Scan Intent"),
            ("qa2_status", "QA2"),
            ("qa2_passed", "QA2"),
            ("obs_collection", "Telescope"),
            ("instrument_name", "Instrument"),
            ("band_list", "Band"),
            ("frequency", "Freq (GHz)"),
            ("frequency_support", "Freq Support"),
            ("cont_sensitivity_bandwidth", "Cont. Sens. (mJy/beam)"),
            ("dataproduct_type", "Type"),
            ("calib_level", "Cal Level"),
            ("spatial_resolution", "Ang. Res. (arcsec)"),
            ("s_resolution", "Ang. Res. (arcsec)"),
            ("velocity_resolution", "Vel. Res. (km/s)"),
            ("spatial_scale_max", "Max Recov. Scale (arcsec)"),
            ("t_exptime", "Int. Time (s)"),
            ("antenna_arrays", "Array"),
            ("is_mosaic", "Mosaic"),
            ("s_fov", "FOV (arcsec)"),
            ("scientific_category", "Science Category"),
            ("science_keyword", "Science Keyword"),
            ("pol_states", "Polarization"),
            ("pwv", "PWV (mm)"),
            ("pi_name", "PI"),
            ("proposal_authors", "Authors"),
            ("obs_release_date", "Release Date"),
            ("obs_title", "Project Title"),
            ("schedblock_name", "SB Name"),
            ("proposal_id", "Proposal ID"),
            ("member_ous_uid", "MOUS ID"),
            ("group_ous_uid", "Group OUS ID"),
            ("asdm_uid", "ASDM UID"),
        ]
        if table_kind == "alma_products":
            alma_display_cols = product_display_cols
        elif table_kind == "alma_project_picker":
            alma_display_cols = project_picker_cols
        seen_display = set()
        sel_cols, display_cols = [], []
        for raw, nice in alma_display_cols:
            if raw in df.columns and nice not in seen_display:
                sel_cols.append(raw)
                display_cols.append(nice)
                seen_display.add(nice)

        if not sel_cols:
            sel_cols = list(df.columns[:8])
            display_cols = sel_cols

        _MAX_TABLE_ROWS = 10000
        sub = df[sel_cols].head(_MAX_TABLE_ROWS).copy()
        sub.columns = display_cols

        per_row_links = []
        if table_kind == "alma_products" and "access_url" in df.columns:
            per_row_links = df["access_url"].head(_MAX_TABLE_ROWS).fillna("").tolist()
        elif archive_kind == "cadc" and "obs_id" in df.columns:
            # Build browsable CADC archive links (not raw DataLink URLs)
            _collections = df["obs_collection"].head(_MAX_TABLE_ROWS).fillna("").tolist() if "obs_collection" in df.columns else [""] * min(len(df), _MAX_TABLE_ROWS)
            _obs_ids = df["obs_id"].head(_MAX_TABLE_ROWS).fillna("").tolist()
            import urllib.parse as _urlparse
            per_row_links = [
                f"https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/search/?Observation.observationID={_urlparse.quote(str(oid), safe='')}"
                if oid else ""
                for oid, col in zip(_obs_ids, _collections)
            ]
        elif "member_ous_uid" in df.columns:
            # ALMA archive links via member OUS UID
            per_row_links = [
                f"https://almascience.nrao.edu/aq/?member_ous_id={v}"
                if pd.notna(v) and str(v).strip() else ""
                for v in df["member_ous_uid"].head(_MAX_TABLE_ROWS)
            ]
        elif "access_url" in df.columns:
            # Fallback: use access_url directly (non-DataLink sources)
            per_row_links = df["access_url"].head(_MAX_TABLE_ROWS).fillna("").tolist()

        def _fmt(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return ""
            if isinstance(v, float):
                return f"{v:.3f}".rstrip("0").rstrip(".")
            return str(v)[:60]

        for col in sub.columns:
            sub[col] = sub[col].map(_fmt)

        rows = sub.to_dict("records")
        for i, link in enumerate(per_row_links):
            if link and i < len(rows):
                rows[i]["_link"] = link

        # ── Inject sky preview thumbnail URLs ──────────
        ra_col = next((c for c in ["s_ra", "ra"] if c in df.columns), None)
        dec_col = next((c for c in ["s_dec", "dec"] if c in df.columns), None)
        has_preview = False

        is_cadc = archive_kind == "cadc"
        cadc_collections = set()
        if is_cadc and "obs_collection" in df.columns:
            cadc_collections = set(df["obs_collection"].dropna().astype(str).unique())

        cadc_preview_map = {}
        if is_cadc and "obs_publisher_did" in df.columns:
            pub_ids = df["obs_publisher_did"].head(_MAX_TABLE_ROWS).dropna().astype(str).tolist()
            try:
                cadc_preview_map = _fetch_cadc_preview_urls(pub_ids)
            except Exception:
                pass

        if cadc_preview_map:
            for i, (_, orig_row) in enumerate(df.head(_MAX_TABLE_ROWS).iterrows()):
                if i >= len(rows):
                    break
                pub_id = str(orig_row.get("obs_publisher_did", "")).strip()
                if pub_id in cadc_preview_map:
                    rows[i]["_preview"] = cadc_preview_map[pub_id]
                    has_preview = True
                elif ra_col and dec_col:
                    try:
                        ra_v = float(orig_row[ra_col])
                        dec_v = float(orig_row[dec_col])
                        if not (math.isnan(ra_v) or math.isnan(dec_v)):
                            rows[i]["_preview"] = (
                                f"https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
                                f"?hips=CDS%2FP%2FDSS2%2Fcolor&width=120&height=120"
                                f"&fov=0.033&projection=TAN&coordsys=icrs"
                                f"&ra={ra_v:.6f}&dec={dec_v:.6f}&format=jpg"
                            )
                            has_preview = True
                    except (ValueError, TypeError):
                        pass
        elif ra_col and dec_col:
            for i, (_, orig_row) in enumerate(df.head(_MAX_TABLE_ROWS).iterrows()):
                if i >= len(rows):
                    break
                try:
                    ra_v = float(orig_row[ra_col])
                    dec_v = float(orig_row[dec_col])
                    if not (math.isnan(ra_v) or math.isnan(dec_v)):
                        rows[i]["_preview"] = (
                            f"https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
                            f"?hips=CDS%2FP%2FDSS2%2Fcolor&width=120&height=120"
                            f"&fov=0.033&projection=TAN&coordsys=icrs"
                            f"&ra={ra_v:.6f}&dec={dec_v:.6f}&format=jpg"
                        )
                        has_preview = True
                except (ValueError, TypeError):
                    pass

        # ── Demographics & FITS estimation ─────────────
        demographics, fits_estimate = _compute_demographics(df)
        if table_kind == "alma_products":
            demographics = {}
            if "filename" in df.columns:
                fits_estimate = int(
                    df["filename"].astype(str).str.contains(r"\.fits?(\.gz)?$", case=False, regex=True, na=False).sum()
                )
            else:
                fits_estimate = len(df)
        elif table_kind == "alma_project_picker":
            fits_estimate = 0

        # ── Detect archive source dynamically ─────────
        _detected_source = _default_archive_source(
            df,
            source_hint=_source,
            filter_label=_filter_label,
        )

        metrics = [{"label": "Results", "value": len(df), "color": "blue"}]
        if "band_list" in df.columns:
            metrics.append({
                "label": "Bands",
                "value": int(df["band_list"].astype(str).nunique()),
                "color": "purple",
            })
        if fits_estimate > 0:
            metrics.append({
                "label": "FITS" if table_kind == "alma_products" else "Est. FITS",
                "value": f"{fits_estimate}" if table_kind == "alma_products" else f"~{fits_estimate}",
                "color": "amber",
            })
        if "obs_collection" in df.columns:
            metrics.append({
                "label": "Telescopes",
                "value": int(df["obs_collection"].nunique()),
                "color": "emerald",
            })

        # ── Build archive link (per-archive) ──────────
        archive_link = build_archive_link(
            df,
            source_hint=_source,
            filter_label=_filter_label,
        )

        warnings = list(_run_result.get("warnings") or [])
        table_payload = {
            "type": "data",
            "metrics": metrics,
            "columns": list(sub.columns),
            "rows": rows,
            "sourceName": _detected_source,
            "warnings": warnings,
            "partial": bool(_run_result.get("partial") or warnings),
            "archiveLink": archive_link,
            "hasRowLinks": any(bool(r.get("_link")) for r in rows),
            "hasPreview": has_preview,
            "demographics": demographics if demographics else None,
            "fitsEstimate": fits_estimate if fits_estimate > 0 else None,
            "tableKind": table_kind or None,
        }
        table_event_str = f"data: {json.dumps(table_payload)}\n\n"
        rich_dt = table_payload.copy()
        rich_dt.pop("type", None)
        return (table_event_str, rich_dt)
    except Exception as e:
        print(f"[WARN] Could not build data card: {e}")
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
):
    with llm_request_context(
        provider_api_keys=llm_context["provider_api_keys"],
        key_source_by_provider=llm_context["key_source_by_provider"],
        byok_token_limits=llm_context["byok_token_limits"],
        user_id=user_id,
        user_email=user_email,
        usage_recorder=usage_recorder,
        quota_checker=quota_checker,
    ):
        return fn()


def _stream_chat_response(
    request: ChatRequest,
    authorization: Optional[str] = None,
    attachment_context: Optional[Dict[str, Any]] = None,
    client_ip: str = "",
) -> StreamingResponse:
    """Shared SSE chat pipeline used by both /api/chat and /api/chat/upload."""
    agent = get_agent()
    auth_header = _safe_authorization_header(authorization)
    current_user = _resolve_optional_user(auth_header)

    requested_model = request.model or (getattr(agent.config, "model", None) if agent else None) or "gpt-4o"
    provider = detect_provider(requested_model)
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

    usage_recorder = _make_usage_recorder(current_user_id)
    quota_checker = _make_quota_checker(
        current_user_id,
        current_user_email,
        llm_context["byok_token_limits"],
    )
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
                # Frontend sent a conversation_id — verify it exists in the DB
                existing = conversation_service.get_user_conversations(current_user_id, limit=100)
                if not any(c["id"] == conv_id for c in existing):
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
            # Store the latest trace_id for this conversation and globally
            global _last_trace_id
            _last_trace_id = lf_trace.id
            if conv_id or request.conversation_id:
                _latest_traces[conv_id or request.conversation_id] = lf_trace.id
        except Exception as lf_err:
            logger.warning(f"[Langfuse] Failed to create parent trace: {lf_err}")

    async def generate():
        if agent is None:
            err = get_agent_error() or "Unknown initialization error"
            mock_response = (
                "Backend initialization failed.\n\n"
                "QuasarAgent could not be loaded. Python traceback:\n\n"
                f"```python\n{err}\n```\n"
            )
            for chunk in mock_response.split("\n"):
                data = json.dumps({"type": "token", "content": chunk + "\n"})
                yield f"data: {data}\n\n"
                await asyncio.sleep(0.02)
            yield "data: [DONE]\n\n"
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
                    ),
                )
            except Exception as e:
                msg = (
                    "I could not analyze the uploaded image before sending it to "
                    f"{requested_model}. The selected model cannot accept raw image inputs. "
                    f"Image analysis error: {redact_secrets(e)}"
                )
                yield f"data: {json.dumps({'type': 'error', 'content': msg})}\n\n"
                yield "data: [DONE]\n\n"
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
                        usage_quota_service.ensure_allowed(
                            user_id=user_id,
                            user_email=current_user_email,
                            provider="openai",
                            key_source=openai_key_source,
                            byok_token_limit=llm_context["byok_token_limits"].get("openai"),
                        )
                        embeddings = (
                            OpenAIEmbeddings(openai_api_key=openai_byok_key)
                            if openai_byok_key
                            else OpenAIEmbeddings()
                        )
                        query_vector = embeddings.embed_query(request.message)
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

        if requested_model != agent.config.model:
            agent.set_model(requested_model)

        _chat_start_time = _time.perf_counter()
        try:
            loop = asyncio.get_event_loop()

            print("[INFO] Routing query to standard Response API (tool-calling loop)")

            queue = asyncio.Queue()

            # ── Plan Feedback Queue (HITL) ────────────────────────────
            # Created at generator scope so the SSE event loop can re-key
            # the queue when conversation_meta arrives with the server UUID.
            _pfq = stdlib_queue.Queue()
            _pfq_key = [conv_id or f"_anon_{id(_pfq)}"]  # mutable for re-keying
            with _plan_feedback_lock:
                _plan_feedback_queues[_pfq_key[0]] = _pfq

            # Emit conversation_meta EARLY so the frontend has the server UUID
            # (or the anon queue key) BEFORE plan_review events arrive.
            # This is critical for HITL feedback routing — the frontend must
            # know the exact key used in _plan_feedback_queues.
            _meta_cid = _pfq_key[0]
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

                        # Sync conversation memory to agent
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
                            )
                        finally:
                            # Clean up the plan feedback queue
                            with _plan_feedback_lock:
                                _plan_feedback_queues.pop(_pfq_key[0], None)

                        # Snapshot thread-local results BEFORE leaving this thread.
                        # The async generator runs on the event-loop thread where
                        # these thread-local values would be invisible.
                        done_payload = {
                            "text": res,
                            "all_results": list(getattr(agent, '_accumulated_run_results', []) or []),
                            "last_result": agent.last_run_result,
                        }
                        asyncio.run_coroutine_threadsafe(queue.put(("done", done_payload)), loop)
                    except Exception as e:
                        print(f"Agent error: {e}")
                        asyncio.run_coroutine_threadsafe(queue.put(("error", redact_secrets(e))), loop)
                finally:
                    if lf_trace:
                        from core.llm_client import set_langfuse_parent
                        set_langfuse_parent(None)

            loop.run_in_executor(
                _executor,
                lambda: _run_with_llm_context(
                    llm_context,
                    current_user_id,
                    current_user_email,
                    usage_recorder,
                    quota_checker,
                    _run_agent,
                ),
            )

            first_token = True
            response_text = ""
            _rich_thinking_text = ""
            # Accumulators for rich UI events — persisted to Turso for history replay
            _rich_data_tables = []  # list of data tables (multi-target support)
            _rich_data_table = None  # last data table (backward compat)
            _rich_papers = None
            _rich_notebook = None
            _rich_image = None
            _rich_web_sources = []
            _rich_web_images = []
            _rich_web_provider = ""
            _rich_web_image_provider = ""
            _rich_web_search_type = ""
            _rich_web_query = ""
            _rich_thinking = []
            _eagerly_emitted = set()  # indices of data cards already emitted during streaming
            _pending_eager_data = []

            while True:
                # Use a timeout so we can send SSE keepalive comments.
                # Render's reverse proxy kills idle connections after ~30s.
                # Sending `:keepalive\n\n` (an SSE comment) every 15s prevents this.
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if isinstance(msg, tuple) and len(msg) == 3:
                    msg_type, step, state = msg
                    if msg_type == "status":
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
                                    if _eager_result.get("type") == "data":
                                        # Do not eagerly emit data cards during streaming to avoid showing intermediate results
                                        print(f"[EAGER] Bypassed eager emission for data card idx={_eager_idx} during streaming")
                                    else:
                                        card = _build_data_card_event(_eager_result)
                                        if card:
                                            _event_str, _rich = card
                                            _tn = _eager_result.get("tool_name", "search_alma_archive")
                                            yield f"data: {json.dumps({'type': 'tool_call', 'name': _tn, 'displayName': _tn.replace('_',' ').title(), 'status': 'completed', 'input': {}, 'output': 'Found results'})}\n\n"
                                            yield _event_str
                                            _rich_data_tables.append(_rich)
                                            _rich_data_table = _rich
                                            _eagerly_emitted.add(_eager_idx)
                                            print(f"[EAGER] Data card emitted for idx={_eager_idx} during streaming")
                            except Exception as _eager_err:
                                print(f"[WARN] Eager data card emission failed: {_eager_err}")
                        elif isinstance(step, str) and step.startswith("__event__"):
                            try:
                                event_json = step[len("__event__"):]
                                event_parsed = json.loads(event_json)
                                # Re-key plan feedback queue when server assigns conversation_id
                                if event_parsed.get("type") == "conversation_meta":
                                    new_cid = event_parsed.get("conversation_id")
                                    if new_cid and new_cid != _pfq_key[0]:
                                        old_key = _pfq_key[0]
                                        with _plan_feedback_lock:
                                            _plan_feedback_queues.pop(old_key, None)
                                            _plan_feedback_queues[new_cid] = _pfq
                                        _pfq_key[0] = new_cid
                                        print(f"[HITL] Re-keyed plan feedback queue: {old_key[:20]}... → {new_cid[:20]}...")
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
                        continue

                msg_type, payload = msg[0], msg[1]
                if msg_type == "done":
                    # payload is a dict with text + snapshotted thread-local results
                    response_text = _sanitize_assistant_text(payload["text"] if isinstance(payload, dict) else payload)
                    _snapshot_all = payload.get("all_results", []) if isinstance(payload, dict) else []
                    _snapshot_last = payload.get("last_result") if isinstance(payload, dict) else None
                    break
                if msg_type == "error":
                    response_text = f"An error occurred: {payload}"
                    _snapshot_all = []
                    _snapshot_last = None
                    break
                if msg_type == "token":
                    first_token = False
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
                # Deduplicate by (type, tool_name, data-id) to avoid emitting same result twice
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
                tool_event = json.dumps({
                    "type": "tool_call",
                    "name": tool_name_raw,
                    "displayName": tool_display,
                    "status": "completed",
                    "input": _run_result.get("params", {}),
                    "output": "Found results",
                })
                yield f"data: {tool_event}\n\n"
                await asyncio.sleep(0.05)

                if result_type == "data":
                    card = _build_data_card_event(_run_result)
                    if card:
                        _event_str, _rich_dt = card
                        yield _event_str
                        _rich_data_tables.append(_rich_dt)
                        _rich_data_table = _rich_dt
                        await asyncio.sleep(0.05)

                elif result_type == "papers":
                    papers = _run_result.get("papers", [])
                    if papers:
                        papers_event = json.dumps({"type": "papers", "papers": papers})
                        yield f"data: {papers_event}\n\n"
                        _rich_papers = papers  # Capture for history
                        await asyncio.sleep(0.05)

                elif result_type == "notebook":
                    nb_data = _run_result.get("notebook_data", {})
                    nb_title = _run_result.get("title", "Analysis Notebook")
                    if nb_data:
                        notebook_event = json.dumps({
                            "type": "notebook",
                            "title": nb_title,
                            "data": nb_data,
                        })
                        yield f"data: {notebook_event}\n\n"
                        _rich_notebook = {"title": nb_title, "data": nb_data}
                        await asyncio.sleep(0.05)

                elif result_type == "image":
                    img_url = _run_result.get("image_url", "")
                    caption = _run_result.get("caption", "")
                    if img_url:
                        image_event = json.dumps({
                            "type": "image", "url": img_url, "caption": caption,
                        })
                        yield f"data: {image_event}\n\n"
                        _rich_image = {"url": img_url, "caption": caption}
                        await asyncio.sleep(0.05)

                elif result_type == "conductor_result":
                    # Multiple images accumulated during Conductor orchestration
                    images = _run_result.get("images", [])
                    _local_seen_img = set()
                    for img in images:
                        img_url = img.get("image_url", "")
                        caption = img.get("caption", "")
                        if img_url:
                            if img_url in _local_seen_img:
                                continue
                            _local_seen_img.add(img_url)
                            image_event = json.dumps({
                                "type": "image", "url": img_url, "caption": caption,
                            })
                            yield f"data: {image_event}\n\n"
                            _rich_image = {"url": img_url, "caption": caption}
                            await asyncio.sleep(0.05)
                            
                    # Companion notebook
                    nb_data = _run_result.get("notebook_data", {})
                    nb_title = _run_result.get("title", "Research Notebook")
                    if nb_data:
                        nb_event = json.dumps({
                            "type": "notebook",
                            "title": nb_title,
                            "data": nb_data,
                        })
                        yield f"data: {nb_event}\n\n"
                        _rich_notebook = {"title": nb_title, "data": nb_data}
                        await asyncio.sleep(0.05)

            if response_text and first_token:
                data = json.dumps({"type": "token", "content": response_text})
                yield f"data: {data}\n\n"

            # ── Persist assistant response + rich UI data to DB ─────
            if current_user_id and conv_id and (
                response_text or _rich_data_tables or _rich_data_table or
                _rich_papers or _rich_notebook or _rich_image or
                _rich_web_sources or _rich_web_images or
                _rich_thinking or _rich_thinking_text
            ):
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
                    if _rich_notebook:
                        rich_meta["notebook"] = _rich_notebook
                    if _rich_image:
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
                    conversation_service.save_message(
                        conv_id, "assistant", response_text or "",
                        metadata=rich_meta if rich_meta else None,
                    )
                except Exception as e:
                    logger.warning(f"[CHAT] Failed to persist assistant message: {e}")

            # ── Log chat analytics ─────────────────────────────────
            try:
                _elapsed_ms = int((_time.perf_counter() - _chat_start_time) * 1000)
                _tool_names = [s.get("step", "").replace("Calling tool: ", "") for s in _rich_thinking if "Calling tool:" in s.get("step", "")]
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

            yield "data: [DONE]\n\n"

        except Exception as e:
            print(f"Chat route error: {redact_secrets(e)}")
            error_data = json.dumps({"type": "error", "content": redact_secrets(e)})
            yield f"data: {error_data}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if lf_trace:
                try:
                    lf_trace.update(
                        metadata={
                            "response_length": len(response_text) if response_text else 0,
                            "response_preview": response_text[:200] if response_text else "",
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


# ── Endpoints ────────────────────────────────────────────────

# ── Plan Feedback (Human-in-the-Loop) ────────────────────────

class PlanFeedbackRequest(BaseModel):
    conversation_id: str
    approve: bool = False
    feedback: str = ""

@app.post("/api/plan-feedback")
async def submit_plan_feedback(req: PlanFeedbackRequest):
    """Receive user approval or feedback for a Conductor execution plan.

    Pushes the response into the plan_feedback_queue for the given
    conversation, unblocking the Conductor's orchestrate() method.
    """
    with _plan_feedback_lock:
        pfq = _plan_feedback_queues.get(req.conversation_id)

    if not pfq:
        raise HTTPException(
            status_code=404,
            detail=f"No pending plan review for conversation {req.conversation_id}",
        )

    pfq.put({
        "approve": req.approve,
        "feedback": req.feedback,
    })

    action = "approved" if req.approve else f"feedback: {req.feedback[:80]}"
    logger.info(f"[HITL] Plan {action} for conversation {req.conversation_id}")
    return {"status": "ok", "action": "approved" if req.approve else "feedback_sent"}

# ── General Endpoints ────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "service": "QUASAR API", "version": "2.0.0"}


@app.get("/api/models")
async def list_models():
    cloud_models = [
            "gpt-5.4-mini",             # GPT-5.4 Mini (default — adaptive reasoning, cost-effective)
            "gpt-4.1",                 # GPT-4.1
            "gpt-4o-mini",             # GPT-4o Mini
            "deepseek-v4-pro",          # DeepSeek V4 Pro (thinking reasoning model)
            "deepseek-v4-flash",        # DeepSeek V4 Flash (fast thinking, lowest cost)
    ]

    # ── Auto-discover local models (Ollama / LM Studio) ──────────────
    local_models = []
    local_base = os.getenv("LOCAL_LLM_BASE_URL", "")
    if local_base:
        try:
            import httpx
            resp = httpx.get(f"{local_base}/models", timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                for m in data.get("data", []):
                    model_id = m.get("id", "")
                    if model_id:
                        local_models.append(f"local/{model_id}")
        except Exception as e:
            logger.warning(f"[MODELS] Failed to discover local models: {e}")

    return {"models": cloud_models + local_models}


# Provider Key / Quota Endpoints

def _test_provider_key(provider: str, api_key: str) -> None:
    """Make a tiny provider call to validate a stored BYOK key."""
    provider = "google" if provider == "gemini" else provider
    test_models = {
        "openai": os.getenv("OPENAI_KEY_TEST_MODEL", "gpt-4o-mini"),
        "deepseek": os.getenv("DEEPSEEK_KEY_TEST_MODEL", "deepseek-chat"),
        "anthropic": os.getenv("ANTHROPIC_KEY_TEST_MODEL", "claude-3-5-haiku-latest"),
        "google": os.getenv("GEMINI_KEY_TEST_MODEL", "gemini-1.5-flash"),
    }
    model = test_models.get(provider)
    if not model:
        raise ValueError(f"Unsupported provider '{provider}'.")

    client = LLMClient(
        model=model,
        provider_api_keys={provider: api_key},
        key_source_by_provider={provider: "byok"},
    )
    client.responses.create(
        model=model,
        input="Reply with ok.",
        max_output_tokens=2,
        temperature=0,
    )


@app.get("/api/provider-keys")
async def list_provider_keys(current_user: dict = Depends(get_current_user)):
    return {"keys": provider_key_service.list_keys(current_user["sub"])}


@app.post("/api/provider-keys")
async def save_provider_key(req: ProviderKeySaveRequest, current_user: dict = Depends(get_current_user)):
    try:
        metadata = provider_key_service.save_key(
            current_user["sub"],
            req.provider,
            req.api_key,
            token_limit=req.token_limit,
        )
        return {"status": "success", "key": metadata}
    except ProviderKeyError as exc:
        raise HTTPException(status_code=400, detail=redact_secrets(exc))


@app.post("/api/provider-keys/{provider}/test")
async def test_provider_key(provider: str, current_user: dict = Depends(get_current_user)):
    user_id = current_user["sub"]
    try:
        api_key = provider_key_service.decrypt_key(user_id, provider)
        if not api_key:
            raise HTTPException(status_code=404, detail="Provider key not found")
        try:
            _test_provider_key(provider, api_key)
        except Exception as exc:
            provider_key_service.mark_test_result(user_id, provider, False)
            raise HTTPException(
                status_code=400,
                detail=f"Provider key test failed: {redact_secrets(exc)}",
            )
        metadata = provider_key_service.mark_test_result(user_id, provider, True)
        return {"status": "success", "key": metadata}
    except HTTPException:
        raise
    except ProviderKeyError as exc:
        raise HTTPException(status_code=400, detail=redact_secrets(exc))


@app.patch("/api/provider-keys/{provider}/limit")
async def update_provider_key_limit(
    provider: str,
    req: ProviderKeyLimitRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        metadata = provider_key_service.set_token_limit(
            current_user["sub"],
            provider,
            req.token_limit,
        )
        return {"status": "success", "key": metadata}
    except ProviderKeyError as exc:
        raise HTTPException(status_code=400, detail=redact_secrets(exc))


@app.delete("/api/provider-keys/{provider}")
async def delete_provider_key(provider: str, current_user: dict = Depends(get_current_user)):
    try:
        if provider_key_service.delete_key(current_user["sub"], provider):
            return {"status": "success"}
    except ProviderKeyError as exc:
        raise HTTPException(status_code=400, detail=redact_secrets(exc))
    raise HTTPException(status_code=404, detail="Provider key not found")


@app.get("/api/usage-quota")
async def get_usage_quota(current_user: dict = Depends(get_current_user)):
    metadata = provider_key_service.list_keys(current_user["sub"])
    return usage_quota_service.usage_summary(
        user_id=current_user["sub"],
        user_email=_current_user_email(current_user),
        provider_key_metadata=metadata,
    )

# ── Custom Tool Models ────────────────────────────────────────

class UserToolRequest(BaseModel):
    name: str
    description: str
    code: str
    api_key_name: Optional[str] = None
    api_key_value: Optional[str] = None

class MCPServerRequest(BaseModel):
    name: str
    transport: str = "stdio"
    command: Optional[str] = None
    args: List[str] = []
    url: Optional[str] = None
    env: Dict[str, str] = {}

# ── Custom Tool Endpoints ────────────────────────────────────

@app.get("/api/user-tools")
async def list_user_tools(current_user: dict = Depends(get_current_user)):
    from services.user_tools_service import UserToolsService
    svc = UserToolsService()
    return svc.load_tools(current_user["sub"])

@app.post("/api/user-tools")
async def save_user_tool(req: UserToolRequest, current_user: dict = Depends(get_current_user)):
    from services.user_tools_service import UserToolsService
    svc = UserToolsService()
    user_id = current_user["sub"]

    # Save secret if provided
    if req.api_key_name and req.api_key_value:
        svc.save_secret(user_id, req.api_key_name.strip(), req.api_key_value.strip())

    try:
        tool = svc.save_tool(
            user_id,
            name=req.name.strip(),
            description=req.description.strip(),
            code=req.code,
            api_key_name=req.api_key_name.strip() if req.api_key_name else None
        )
        return {"status": "success", "tool": tool}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/user-tools/{tool_name}")
async def delete_user_tool(tool_name: str, current_user: dict = Depends(get_current_user)):
    from services.user_tools_service import UserToolsService
    svc = UserToolsService()
    user_id = current_user["sub"]

    if svc.delete_tool(user_id, tool_name):
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Tool not found")


# ── MCP Server Endpoints ─────────────────────────────────────

@app.get("/api/mcp-servers")
async def list_mcp_servers(current_user: dict = Depends(get_current_user)):
    from services.mcp_server_service import MCPServerService
    svc = MCPServerService()
    return svc.load_servers(current_user["sub"])

@app.post("/api/mcp-servers")
async def save_mcp_server(req: MCPServerRequest, current_user: dict = Depends(get_current_user)):
    from services.mcp_server_service import MCPServerService, MCPServerConfig
    svc = MCPServerService()
    user_id = current_user["sub"]
    
    try:
        config = MCPServerConfig(**req.model_dump())
        srv = svc.save_server(user_id, config)
        return {"status": "success", "server": srv}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/mcp-servers/{name}")
async def delete_mcp_server(name: str, current_user: dict = Depends(get_current_user)):
    from services.mcp_server_service import MCPServerService
    svc = MCPServerService()
    user_id = current_user["sub"]
    
    if svc.delete_server(user_id, name):
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Server not found")

# ── Auth Endpoints ──────────────────────────────────────────

@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    success, msg, token = auth_service.register_user(
        req.username, req.password, req.email, req.display_name
    )
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    
    # We verify token to get full payload for frontend
    payload = auth_service.verify_token(token)
    return {
        "token": token,
        "user": {
            "id": payload["sub"],
            "username": payload.get("email") or req.username,
            "display_name": payload.get("name") or req.username,
            "auth_provider": "local"
        }
    }

@app.post("/api/auth/login")
async def login(req: LoginRequest):
    success, user_id, email, display_name, msg = auth_service.login_user(req.username, req.password)
    if not success:
        raise HTTPException(status_code=401, detail=msg)
        
    token = auth_service.generate_token(user_id, email, display_name)
    return {
        "token": token,
        "user": {
            "id": user_id,
            "username": email or req.username,
            "display_name": display_name or req.username,
            "auth_provider": "local"
        }
    }

@app.post("/api/auth/google")
async def google_login(req: GoogleLoginRequest):
    try:
        # Verify the Google ID token
        client_id = os.environ.get("GOOGLE_CLIENT_ID")
        if not client_id:
            raise HTTPException(status_code=500, detail="Google authentication is not configured on the server")
            
        idinfo = id_token.verify_oauth2_token(
            req.credential, google_requests.Request(), client_id
        )
        
        email = idinfo.get("email")
        name = idinfo.get("name")
        picture = idinfo.get("picture")  # Google profile picture URL
        
        if not email:
            raise HTTPException(status_code=400, detail="Google account has no email")
            
        success, user_id, msg, token = auth_service.register_or_login_google_user(email, name, picture)
        
        if not success:
            raise HTTPException(status_code=400, detail=msg)
            
        return {
            "token": token,
            "user": {
                "id": user_id,
                "username": email,
                "display_name": name,
                "auth_provider": "google",
                "picture_url": picture,
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"Invalid Google token: {str(e)}")

@app.get("/api/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return {
        "user": {
            "id": current_user["sub"],
            "username": current_user.get("email"),
            "display_name": current_user.get("name")
        }
    }


# ── Conversation History Endpoints ──────────────────────────────

@app.post("/api/fits/preview")
async def preview_fits(req: FitsPreviewRequest, current_user: dict = Depends(get_current_user)):
    """Download a small ALMA FITS product and return workbench-ready previews."""
    from urllib.parse import urlparse
    import base64
    import io
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import numpy as np
    import requests
    from astropy.io import fits
    from astropy.visualization import simple_norm

    parsed = urlparse(str(req.url or "").strip())
    allowed_hosts = {"almascience.nrao.edu", "almascience.eso.org", "almascience.nao.ac.jp"}
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
        raise HTTPException(status_code=400, detail="Only HTTPS ALMA data product URLs can be previewed.")

    max_bytes = 50 * 1024 * 1024
    try:
        with requests.get(req.url, stream=True, timeout=90) as response:
            response.raise_for_status()
            length = response.headers.get("content-length")
            if length and int(length) > max_bytes:
                raise HTTPException(status_code=413, detail="This FITS file is too large for in-browser preview. Open or download it from ALMA instead.")

            chunks = []
            total = 0
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail="This FITS file is too large for in-browser preview. Open or download it from ALMA instead.")
                chunks.append(chunk)
        fits_bytes = b"".join(chunks)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not fetch FITS product: {exc}")

    try:
        def _finite_float(value):
            try:
                number = float(value)
                return number if math.isfinite(number) else None
            except (TypeError, ValueError):
                return None

        def _figure_data_url(fig) -> str:
            png_buf = io.BytesIO()
            fig.savefig(png_buf, format="png", bbox_inches="tight")
            plt.close(fig)
            return "data:image/png;base64," + base64.b64encode(png_buf.getvalue()).decode("ascii")

        def _robust_rms(values) -> Optional[float]:
            arr = np.asarray(values, dtype=float).ravel()
            if arr.size > 250000:
                step = int(math.ceil(arr.size / 250000))
                arr = arr[::step]
            arr = arr[np.isfinite(arr)]
            if arr.size < 10:
                return None
            median = float(np.nanmedian(arr))
            mad = float(np.nanmedian(np.abs(arr - median)))
            if mad > 0:
                return 1.4826 * mad
            std = float(np.nanstd(arr))
            return std if math.isfinite(std) and std > 0 else None

        def _safe_minmax(values) -> tuple[Optional[float], Optional[float]]:
            arr = np.asarray(values, dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return None, None
            return float(np.nanmin(finite)), float(np.nanmax(finite))

        def _spectral_axis(header, nchan: int):
            crval = _finite_float(header.get("CRVAL3"))
            cdelt = _finite_float(header.get("CDELT3"))
            crpix = _finite_float(header.get("CRPIX3")) or 1.0
            ctype = str(header.get("CTYPE3") or "Channel").strip()
            cunit = str(header.get("CUNIT3") or "").strip()
            if crval is None or cdelt is None:
                return np.arange(nchan, dtype=float), "Channel", "channel", None

            axis = crval + ((np.arange(nchan, dtype=float) + 1.0 - crpix) * cdelt)
            unit_lower = cunit.lower()
            ctype_upper = ctype.upper()
            overlay_unit = cunit or ""
            if "hz" in unit_lower or ("freq" in ctype_upper and np.nanmedian(np.abs(axis)) > 1e5):
                if unit_lower == "ghz":
                    converted = axis
                elif unit_lower == "mhz":
                    converted = axis / 1000.0
                elif unit_lower == "khz":
                    converted = axis / 1e6
                else:
                    converted = axis / 1e9
                return converted, f"{ctype or 'Frequency'} (GHz)", "GHz", "frequency"
            if unit_lower in {"m/s", "ms-1", "meter/s", "metre/s"}:
                return axis / 1000.0, f"{ctype or 'Velocity'} (km/s)", "km/s", "velocity"
            if unit_lower in {"km/s", "kms-1"}:
                return axis, f"{ctype or 'Velocity'} (km/s)", "km/s", "velocity"
            return axis, f"{ctype or 'Axis 3'}{f' ({overlay_unit})' if overlay_unit else ''}", overlay_unit, None

        def _render_image(image, title: str, unit: str, header, rms_value: Optional[float], add_contours: bool):
            fig, ax = plt.subplots(figsize=(6.2, 6.0), dpi=150)
            try:
                norm = simple_norm(image, "asinh", percent=99.5)
            except Exception:
                norm = None
            im = ax.imshow(image, origin="lower", cmap="inferno", norm=norm)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Pixel")
            ax.set_ylabel("Pixel")
            contour_levels = []
            if add_contours and rms_value and rms_value > 0:
                image_max = _safe_minmax(image)[1]
                contour_levels = [
                    level for level in (3 * rms_value, 5 * rms_value, 10 * rms_value)
                    if image_max is not None and image_max > level
                ]
                if contour_levels:
                    ax.contour(image, levels=contour_levels, colors="white", linewidths=0.45, alpha=0.6)

            bmaj = _finite_float(header.get("BMAJ"))
            bmin = _finite_float(header.get("BMIN"))
            bpa = _finite_float(header.get("BPA")) or 0.0
            cdelt1 = abs(_finite_float(header.get("CDELT1")) or 0)
            cdelt2 = abs(_finite_float(header.get("CDELT2")) or 0)
            if bmaj and bmin and cdelt1 and cdelt2:
                ny, nx = image.shape[-2], image.shape[-1]
                width_px = max(1.0, (bmaj * 3600.0) / (cdelt1 * 3600.0))
                height_px = max(1.0, (bmin * 3600.0) / (cdelt2 * 3600.0))
                beam = Ellipse(
                    (max(10, nx * 0.12), max(10, ny * 0.1)),
                    width=width_px,
                    height=height_px,
                    angle=bpa,
                    facecolor="none",
                    edgecolor="white",
                    linewidth=1.0,
                    alpha=0.9,
                )
                ax.add_patch(beam)
                ax.text(max(10, nx * 0.12), max(6, ny * 0.1 - height_px * 0.8), "beam",
                        color="white", fontsize=6, ha="center", alpha=0.85)

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(unit or "value", fontsize=8)
            fig.tight_layout()
            return _figure_data_url(fig), [float(level) for level in contour_levels]

        def _render_channel_maps(cube, spectral_values, spectral_label: str, unit: str):
            nchan = int(cube.shape[0])
            sample_count = min(9, nchan)
            indices = np.unique(np.linspace(0, nchan - 1, sample_count, dtype=int))
            cols = 3 if len(indices) > 2 else len(indices)
            rows_n = int(math.ceil(len(indices) / cols))
            fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.4, rows_n * 2.25), dpi=145)
            axes_arr = np.asarray(axes).reshape(-1)
            for ax, idx in zip(axes_arr, indices):
                plane = cube[int(idx)]
                try:
                    norm = simple_norm(plane, "asinh", percent=99.3)
                except Exception:
                    norm = None
                ax.imshow(plane, origin="lower", cmap="inferno", norm=norm)
                if spectral_values is not None and len(spectral_values) > int(idx):
                    ax.set_title(f"ch {int(idx)} | {float(spectral_values[int(idx)]):.4g}", fontsize=7)
                else:
                    ax.set_title(f"ch {int(idx)}", fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
            for ax in axes_arr[len(indices):]:
                ax.axis("off")
            fig.suptitle(f"Channel maps | {spectral_label}", fontsize=9)
            fig.text(0.5, 0.02, unit or "image value", ha="center", fontsize=7)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            return _figure_data_url(fig)

        def _render_pv_slice(cube, y_index: int, spectral_label: str, unit: str):
            pv = cube[:, int(y_index), :]
            fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
            try:
                norm = simple_norm(pv, "asinh", percent=99.5)
            except Exception:
                norm = None
            im = ax.imshow(pv, origin="lower", aspect="auto", cmap="magma", norm=norm)
            ax.set_title("Central PV slice through peak row", fontsize=10)
            ax.set_xlabel("Spatial pixel X")
            ax.set_ylabel(spectral_label)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(unit or "value", fontsize=8)
            fig.tight_layout()
            return _figure_data_url(fig)

        with fits.open(io.BytesIO(fits_bytes), memmap=False) as hdul:
            hdu = next((item for item in hdul if getattr(item, "data", None) is not None), None)
            if hdu is None:
                raise ValueError("No image data found in FITS file.")
            raw_data = np.asarray(hdu.data, dtype=float)
            header = hdu.header

            squeezed = np.squeeze(raw_data)
            if squeezed.ndim < 2:
                raise ValueError("FITS data is not image-like.")
            cube = None
            cube_shape = None
            if squeezed.ndim == 2:
                image = squeezed
                primary_label = "Image preview with beam/contour overlay"
            else:
                if squeezed.ndim > 3:
                    squeezed = squeezed.reshape((-1, squeezed.shape[-2], squeezed.shape[-1]))
                cube = squeezed
                cube_shape = list(cube.shape)
                image = np.nansum(cube, axis=0)
                primary_label = "Moment-0 style cube collapse with beam/contour overlay"

            if not np.isfinite(image).any():
                raise ValueError("Image data contains no finite values.")

            object_name = str(header.get("OBJECT") or req.filename or "ALMA FITS")
            unit = str(header.get("BUNIT") or "")
            rms_value = _robust_rms(cube if cube is not None else image)
            image_data_url, contour_levels = _render_image(
                image,
                f"{object_name} | {unit}".strip(" |"),
                unit,
                header,
                rms_value,
                add_contours=True,
            )

            bmaj = _finite_float(header.get("BMAJ"))
            bmin = _finite_float(header.get("BMIN"))
            bpa = _finite_float(header.get("BPA"))
            restfreq = _finite_float(header.get("RESTFRQ"))
            cdelt1 = abs(_finite_float(header.get("CDELT1")) or 0) or None
            cdelt2 = abs(_finite_float(header.get("CDELT2")) or 0) or None
            min_value, max_value = _safe_minmax(image)

            spectral_values = None
            spectral_label = "Channel"
            spectral_unit = "channel"
            spectral_kind = None
            spectrum_payload = None
            channel_maps_data_url = None
            pv_slice_data_url = None
            line_overlays = []
            assumptions = [
                "Preview renders a downloaded copy limited to 50 MB.",
                "RMS is a robust MAD estimate from previewed pixel values.",
            ]
            confidence = "medium"

            if cube is not None:
                nchan = int(cube.shape[0])
                spectral_values, spectral_label, spectral_unit, spectral_kind = _spectral_axis(header, nchan)
                assumptions.append("For cube-like FITS files, the first non-spatial axis after squeezing is treated as the channel/spectral axis.")

                peak_source = np.asarray(image, dtype=float)
                if np.isfinite(peak_source).any():
                    peak_y, peak_x = np.unravel_index(np.nanargmax(np.abs(peak_source)), peak_source.shape)
                    spectrum_values = cube[:, int(peak_y), int(peak_x)]
                    extraction = f"peak pixel x={int(peak_x)}, y={int(peak_y)}"
                else:
                    peak_y = cube.shape[1] // 2
                    peak_x = cube.shape[2] // 2
                    spectrum_values = np.nanmedian(cube, axis=(1, 2))
                    extraction = "spatial median"

                finite_spec = np.isfinite(spectrum_values)
                if finite_spec.any():
                    max_points = 360
                    step = max(1, int(math.ceil(len(spectrum_values) / max_points)))
                    xs = spectral_values[::step]
                    ys = spectrum_values[::step]
                    spec_pairs = [
                        (float(x_val), float(y_val))
                        for x_val, y_val in zip(xs, ys)
                        if math.isfinite(float(x_val)) and math.isfinite(float(y_val))
                    ]
                    spectrum_payload = {
                        "x": [round(x_val, 8) for x_val, _ in spec_pairs],
                        "y": [round(y_val, 8) for _, y_val in spec_pairs],
                        "xLabel": spectral_label,
                        "yLabel": unit or "Value",
                        "extraction": extraction,
                    }
                    confidence = "high" if restfreq or str(header.get("CTYPE3") or "").strip() else "medium"

                channel_maps_data_url = _render_channel_maps(cube, spectral_values, spectral_label, unit)
                pv_slice_data_url = _render_pv_slice(cube, int(peak_y), spectral_label, unit)

                if restfreq and spectral_kind == "frequency":
                    rest_ghz = restfreq / 1e9
                    axis_min, axis_max = float(np.nanmin(spectral_values)), float(np.nanmax(spectral_values))
                    line_overlays.append({
                        "label": "Header RESTFRQ",
                        "value": round(rest_ghz, 8),
                        "unit": "GHz",
                        "source": "FITS header",
                        "inRange": bool(min(axis_min, axis_max) <= rest_ghz <= max(axis_min, axis_max)),
                    })

            if bmaj and bmin and cdelt1 and cdelt2:
                confidence = "high" if confidence == "medium" else confidence

            filename = req.filename or Path(parsed.path).name or "alma_product.fits"
            casa_script = (
                f"from casatasks import importfits\n"
                f"importfits(fitsimage='{filename}', imagename='{Path(filename).stem}.image', overwrite=True)\n"
                f"# Then inspect in CASA viewer or run imstat/imhead on {Path(filename).stem}.image"
            )
            if cube is not None:
                casa_script += "\n# For cube products, use immoments or specfit after checking the spectral axis."

            metadata = {
                "object": object_name,
                "unit": unit,
                "shape": list(image.shape),
                "rawShape": list(raw_data.shape),
                "cubeShape": cube_shape,
                "isCube": cube is not None,
                "channelCount": int(cube.shape[0]) if cube is not None else None,
                "beamMajorArcsec": bmaj * 3600 if bmaj is not None else None,
                "beamMinorArcsec": bmin * 3600 if bmin is not None else None,
                "beamPaDeg": bpa,
                "restFreqGhz": restfreq / 1e9 if restfreq is not None else None,
                "pixelScaleArcsec": {
                    "x": cdelt1 * 3600 if cdelt1 is not None else None,
                    "y": cdelt2 * 3600 if cdelt2 is not None else None,
                },
                "spectralAxisLabel": spectral_label if cube is not None else None,
                "rms": rms_value,
                "rmsUnit": unit or None,
                "min": min_value,
                "max": max_value,
                "sizeBytes": len(fits_bytes),
            }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not render FITS preview: {exc}")

    return {
        "imageDataUrl": image_data_url,
        "metadata": metadata,
        "downloadUrl": req.url,
        "filename": filename,
        "workbench": {
            "kind": "cube" if metadata.get("isCube") else "image",
            "primaryImageLabel": primary_label,
            "channelMapsDataUrl": channel_maps_data_url,
            "pvSliceDataUrl": pv_slice_data_url,
            "spectrum": spectrum_payload,
            "rms": rms_value,
            "rmsUnit": unit or None,
            "contourLevels": [round(float(level), 8) for level in contour_levels],
            "lineOverlays": line_overlays,
            "exports": [
                {"label": "CASA", "command": casa_script},
                {"label": "CARTA", "command": f"carta '{filename}'"},
                {"label": "DS9", "command": f"ds9 '{filename}' -scale zscale -cmap inferno"},
            ],
            "evidence": {
                "dataAccess": f"Fetched {round(len(fits_bytes) / (1024 * 1024), 2)} MB from {parsed.hostname}",
                "headersInspected": [
                    key for key in ["OBJECT", "BUNIT", "BMAJ", "BMIN", "BPA", "RESTFRQ", "CTYPE3", "CUNIT3", "CRVAL3", "CDELT3"]
                    if header.get(key) is not None
                ],
                "assumptions": assumptions,
                "confidence": confidence,
            },
        },
        "suggestedActions": [
            "Measure peak flux and RMS noise",
            "Inspect channel maps and the spectrum before choosing line windows",
            "Export/open the FITS in CASA, CARTA, DS9, or Python",
        ],
    }


def _workbench_user_id(current_user: dict) -> str:
    return str(current_user.get("sub") or current_user.get("id") or "anonymous")


def _raise_workbench_error(exc: Exception):
    if isinstance(exc, CubeWorkbenchNotFound):
        raise HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, CubeWorkbenchForbidden):
        raise HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc))
    raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/workbench/session")
async def create_workbench_session(
    req: WorkbenchSessionCreate,
    current_user: dict = Depends(get_current_user),
):
    """Create a durable first-class cube/product workbench session."""
    try:
        return cube_workbench_service.create_session(
            user_id=_workbench_user_id(current_user),
            source_url=req.source_url,
            filename=req.filename,
            project_code=req.project_code,
            mous_uid=req.mous_uid,
        )
    except Exception as exc:
        _raise_workbench_error(exc)


@app.get("/api/workbench/{session_id}/metadata")
async def get_workbench_metadata(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return FITS/workbench metadata, state, and evidence for a session."""
    try:
        return cube_workbench_service.get_metadata(
            session_id=session_id,
            user_id=_workbench_user_id(current_user),
        )
    except Exception as exc:
        _raise_workbench_error(exc)


@app.post("/api/workbench/{session_id}/jobs")
async def start_workbench_job(
    session_id: str,
    req: WorkbenchJobStartRequest,
    current_user: dict = Depends(get_current_user),
):
    """Start a durable asynchronous workbench operation."""
    try:
        return cube_workbench_service.start_job(
            session_id=session_id,
            user_id=_workbench_user_id(current_user),
            operation=req.operation,
            payload=req.payload,
        )
    except Exception as exc:
        _raise_workbench_error(exc)


@app.get("/api/workbench/{session_id}/jobs/{job_id}")
async def get_workbench_job(
    session_id: str,
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return durable progress/result state for a workbench job."""
    try:
        return cube_workbench_service.get_job(
            session_id=session_id,
            user_id=_workbench_user_id(current_user),
            job_id=job_id,
        )
    except Exception as exc:
        _raise_workbench_error(exc)


@app.delete("/api/workbench/{session_id}/jobs/{job_id}")
async def cancel_workbench_job(
    session_id: str,
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Request cancellation for a queued/running workbench job."""
    try:
        return cube_workbench_service.cancel_job(
            session_id=session_id,
            user_id=_workbench_user_id(current_user),
            job_id=job_id,
        )
    except Exception as exc:
        _raise_workbench_error(exc)


@app.post("/api/workbench/{session_id}/render")
async def plan_workbench_render(
    session_id: str,
    req: WorkbenchRenderRequest,
    current_user: dict = Depends(get_current_user),
):
    """Persist render controls for the future WCS-aware workbench renderer."""
    try:
        return cube_workbench_service.render_plan(
            session_id=session_id,
            user_id=_workbench_user_id(current_user),
            mode=req.mode or "image",
            channel=req.channel,
            moment=req.moment,
            colormap=req.colormap or "inferno",
            stretch=req.stretch or "asinh",
            contour_sigma=req.contour_sigma,
            rms_region=req.rms_region,
        )
    except Exception as exc:
        _raise_workbench_error(exc)


@app.post("/api/workbench/{session_id}/prepare")
async def prepare_workbench_product(
    session_id: str,
    req: WorkbenchPrepareRequest,
    current_user: dict = Depends(get_current_user),
):
    """Stage a FITS product in the bounded workbench cache for real cube operations."""
    try:
        return cube_workbench_service.prepare_product(
            session_id=session_id,
            user_id=_workbench_user_id(current_user),
            max_bytes=req.max_bytes or 500 * 1024 * 1024,
            user_cache_bytes=req.user_cache_bytes or 2 * 1024 * 1024 * 1024,
            cache_ttl_seconds=req.cache_ttl_seconds or 24 * 60 * 60,
            force=bool(req.force),
        )
    except Exception as exc:
        _raise_workbench_error(exc)


@app.post("/api/workbench/{session_id}/spectrum")
async def plan_workbench_spectrum(
    session_id: str,
    req: WorkbenchSpectrumRequest,
    current_user: dict = Depends(get_current_user),
):
    """Persist a spectrum extraction request and return spectral-axis metadata."""
    try:
        return cube_workbench_service.spectrum_plan(
            session_id=session_id,
            user_id=_workbench_user_id(current_user),
            x_pixel=req.x_pixel,
            y_pixel=req.y_pixel,
            aperture_radius_pixels=req.aperture_radius_pixels or 3.0,
            aperture_radius_arcsec=req.aperture_radius_arcsec,
            max_points=req.max_points or 512,
        )
    except Exception as exc:
        _raise_workbench_error(exc)


@app.post("/api/workbench/{session_id}/pv-slice")
async def plan_workbench_pv_slice(
    session_id: str,
    req: WorkbenchPvSliceRequest,
    current_user: dict = Depends(get_current_user),
):
    """Persist a PV-slice path and return axis planning metadata."""
    try:
        return cube_workbench_service.pv_slice_plan(
            session_id=session_id,
            user_id=_workbench_user_id(current_user),
            path=req.path,
            width_pixels=req.width_pixels or 3.0,
            max_points=req.max_points or 512,
        )
    except Exception as exc:
        _raise_workbench_error(exc)


@app.post("/api/workbench/{session_id}/line-overlays")
async def get_workbench_line_overlays(
    session_id: str,
    req: WorkbenchLineOverlayRequest,
    current_user: dict = Depends(get_current_user),
):
    """Return redshift-aware Splatalogue line overlays for a workbench session."""
    try:
        return cube_workbench_service.line_overlays(
            session_id=session_id,
            user_id=_workbench_user_id(current_user),
            observed_frequency_ghz=req.observed_frequency_ghz,
            line_preset_key=req.line_preset_key,
            redshift=req.redshift or 0.0,
            tolerance_ghz=req.tolerance_ghz or 0.01,
            top_n=req.top_n or 8,
        )
    except Exception as exc:
        _raise_workbench_error(exc)


@app.get("/api/workbench/line-presets")
async def get_workbench_line_presets():
    """Return common ALMA/mm spectral-line presets for workbench line ID."""
    return cube_workbench_service.line_presets()


@app.post("/api/workbench/{session_id}/export")
async def get_workbench_exports(
    session_id: str,
    req: WorkbenchExportRequest,
    current_user: dict = Depends(get_current_user),
):
    """Return reproducible export scripts/commands for a workbench session."""
    try:
        return cube_workbench_service.export_plan(
            session_id=session_id,
            user_id=_workbench_user_id(current_user),
            formats=req.formats,
        )
    except Exception as exc:
        _raise_workbench_error(exc)

@app.get("/api/conversations")
async def list_conversations(current_user: dict = Depends(get_current_user)):
    """List a user's conversations, most recent first."""
    user_id = current_user["sub"]
    convos = conversation_service.get_user_conversations(user_id, limit=50)
    return {"conversations": convos}


@app.post("/api/conversations")
async def create_conversation_endpoint(req: ConversationCreate, current_user: dict = Depends(get_current_user)):
    """Create a new empty conversation."""
    user_id = current_user["sub"]
    conv_id = conversation_service.create_conversation(user_id, req.title, req.model)
    return {"id": conv_id, "title": req.title, "model": req.model}


@app.get("/api/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str, current_user: dict = Depends(get_current_user)):
    """Fetch all messages for a conversation."""
    user_id = current_user["sub"]
    messages = conversation_service.get_conversation_messages_for_user(conversation_id, user_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"messages": messages}


class ConversationTitleUpdate(BaseModel):
    title: str

@app.put("/api/conversations/{conversation_id}/title")
async def update_conversation_title_endpoint(
    conversation_id: str,
    req: ConversationTitleUpdate,
    current_user: dict = Depends(get_current_user),
):
    """Update a conversation's title."""
    user_id = current_user["sub"]
    updated = conversation_service.update_conversation_title_for_user(
        conversation_id,
        user_id,
        req.title,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "ok"}


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a conversation and all its messages."""
    try:
        user_id = current_user["sub"]
        deleted = conversation_service.delete_conversation_for_user(conversation_id, user_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
        print(f"[INFO] Deleted conversation {conversation_id} for user {current_user.get('sub', 'unknown')}")
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Failed to delete conversation {conversation_id}: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@app.post("/api/chat")
async def chat(request: ChatRequest, req: Request = None, authorization: Optional[str] = Header(None)):
    """Stream a chat response via SSE using the shared chat pipeline."""
    _ip = ""
    if req:
        _ip = (req.headers.get("x-forwarded-for", "").split(",")[0].strip()
               or (req.client.host if req.client else ""))
    return _stream_chat_response(request, authorization=authorization, client_ip=_ip)



@app.post("/api/chat/upload")
async def chat_with_files(
    message: str = Form(""),
    conversation_id: Optional[str] = Form(None),
    model: Optional[str] = Form("gpt-4o"),
    grounded_summary: bool = Form(False),
    web_search: bool = Form(True),
    files: PyList[UploadFile] = File(default=[]),
    authorization: Optional[str] = Header(None),
):
    """Stream a chat response with attached files (images/documents) via SSE."""
    auth_header = _safe_authorization_header(authorization)
    current_user = _resolve_optional_user(auth_header)
    user_id = current_user.get("sub") if current_user else None
    if not user_id:
        return _sse_error_response(
            "Authentication required. Please sign in to your Quasar account to upload files."
        )
    selected_model = model or "gpt-4o"
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
    return _stream_chat_response(req, authorization=auth_header, attachment_context=attachment_context)

# ── Personalization Endpoints ────────────────────────────────────────────────

import sqlite3, tempfile, datetime as _dt
from pathlib import Path as _Path

# Personalization DB fallback path (used only in local dev mode)
_PERS_DB_LOCAL = _Path(__file__).resolve().parent.parent / "data" / "personalization.db"

def _get_pers_db():
    """Return a connection to the personalization metadata DB (Turso cloud or local SQLite)."""
    from services.db import get_connection
    return get_connection(str(_PERS_DB_LOCAL))


@app.post("/api/personalization/upload")
async def personalization_upload(
    files: PyList[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Upload and index documents into the user's personal RAG collection."""
    user_id = current_user["sub"]
    results = []

    for f in files:
        raw = await f.read()
        ext = (_Path(f.filename or "file").suffix or ".txt").lower()

        # Write to a temp file so RAGService can read it
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        try:
            loop = asyncio.get_event_loop()

            def _ingest():
                from services.rag_service import RAGService
                svc = RAGService(user_id=user_id)
                return svc.ingest_document(tmp_path, personal=True, original_filename=f.filename)

            result = await loop.run_in_executor(_executor, _ingest)

            if result.get("success"):
                doc_id = str(uuid.uuid4())
                conn = _get_pers_db()
                conn.execute(
                    "INSERT INTO documents VALUES (?,?,?,?,?,?)",
                    (doc_id, user_id, f.filename, len(raw),
                     result.get("chunks", 0), _dt.datetime.utcnow().isoformat())
                )
                conn.commit()
                conn.close()
                results.append({"filename": f.filename, "success": True, "chunks": result.get("chunks", 0)})
            else:
                results.append({"filename": f.filename, "success": False, "error": result.get("error", "Unknown error")})
        except Exception as e:
            results.append({"filename": f.filename, "success": False, "error": str(e)})
        finally:
            try: _Path(tmp_path).unlink()
            except: pass

    successes = sum(1 for r in results if r["success"])
    return {"message": f"{successes}/{len(results)} documents indexed successfully.", "results": results}


@app.get("/api/personalization/documents")
async def personalization_list(current_user: dict = Depends(get_current_user)):
    """List all documents in the user's personal knowledge base."""
    user_id = current_user["sub"]
    conn = _get_pers_db()
    rows = conn.execute(
        "SELECT id, filename, size_bytes, uploaded_at, chunk_count FROM documents WHERE user_id=? ORDER BY uploaded_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "filename": r[1], "size_bytes": r[2], "uploaded_at": r[3], "chunk_count": r[4]}
        for r in rows
    ]


@app.delete("/api/personalization/document/{doc_id}")
async def personalization_delete(doc_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a document from the user's personal knowledge base."""
    user_id = current_user["sub"]
    conn = _get_pers_db()
    row = conn.execute("SELECT filename FROM documents WHERE id=? AND user_id=?", (doc_id, user_id)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    filename = row[0]

    # Remove from vector store
    try:
        loop = asyncio.get_event_loop()
        def _delete():
            from services.rag_service import RAGService
            svc = RAGService(user_id=user_id)
            svc.delete_personal_document(filename)
        await loop.run_in_executor(_executor, _delete)
    except Exception as e:
        print(f"[WARN] Vector delete failed: {e}")

    conn.execute("DELETE FROM documents WHERE id=? AND user_id=?", (doc_id, user_id))
    conn.commit()
    conn.close()
    return {"success": True}







# ── Red Team TAC (Proposal Critic) Endpoint ────────────────

@app.post("/api/proposals/review")
async def review_proposal(file: UploadFile = File(...)):
    """Upload a proposal PDF and stream a Red Team TAC critique."""
    import tempfile
    from pathlib import Path

    async def generate():
        def _status(step: str, state: str = "running"):
            return f"data: {json.dumps({'type': 'status', 'step': step, 'state': state})}\n\n"
            
        tmp_path = None
        try:
            raw = await file.read()
            ext = (Path(file.filename or "file.pdf").suffix or ".pdf").lower()
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name

            loop = asyncio.get_event_loop()
            queue = asyncio.Queue()
            
            def progress_callback(msg: str, pct: int):
                asyncio.run_coroutine_threadsafe(
                    queue.put(("status", msg, "running" if pct < 100 else "completed")), loop
                )
            
            def _run_critic():
                try:
                    from services.proposal_critic import ProposalCriticService
                    from services.rag_service import RAGService
                    critic = ProposalCriticService()
                    rag = RAGService()
                    res = critic.review_proposal(tmp_path, rag, progress_callback=progress_callback)
                    asyncio.run_coroutine_threadsafe(queue.put(("done", res)), loop)
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(queue.put(("error", str(e))), loop)
            
            # Start execution in thread
            loop.run_in_executor(_executor, _run_critic)
            
            yield _status("Initializing Proposal Critic...", "running")
            
            while True:
                msg = await queue.get()
                if msg[0] == "status":
                    yield _status(msg[1], msg[2])
                elif msg[0] == "done":
                    res = msg[1]
                    if res.get("success"):
                        yield _status("Critique generated", "completed")
                        # Emitting the critique text as tokens so it can render immediately
                        critique_text = res.get('critique', '')
                        data_event = json.dumps({'type': 'critique', 'content': critique_text})
                        yield f"data: {data_event}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'error', 'content': res.get('error', 'Unknown Error')})}\n\n"
                    break
                elif msg[0] == "error":
                    yield f"data: {json.dumps({'type': 'error', 'content': msg[1]})}\n\n"
                    break
                    
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if tmp_path:
                try: Path(tmp_path).unlink()
                except: pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


# ── Health check ─────────────────────────────────────────────

@app.get("/health")
async def health():
    agent = get_agent()
    return {
        "status": "healthy",
        "agent_loaded": agent is not None,
    }

# ── Analytics API ────────────────────────────────────────────

@app.get("/api/analytics/hit")
async def analytics_hit(req: Request):
    """Log a page view and return the current total hit count (public, no auth)."""
    ip = (req.headers.get("x-forwarded-for", "").split(",")[0].strip()
          or (req.client.host if req.client else ""))
    ua = req.headers.get("user-agent", "")
    country = req.headers.get("cf-ipcountry", "") or req.headers.get("x-vercel-ip-country", "")
    loop = asyncio.get_event_loop()
    total = await loop.run_in_executor(
        _executor,
        lambda: analytics_service.log_page_view(ip_address=ip, user_agent=ua, country=country),
    )

    # ── Langfuse: log page hit as a custom event ──────────────────
    from core.langfuse_integration import get_langfuse
    lf_client = get_langfuse()
    if lf_client:
        try:
            await loop.run_in_executor(
                _executor,
                lambda: lf_client.event(
                    name="page_view",
                    user_id="anonymous",
                    metadata={
                        "ip_address": ip,
                        "user_agent": ua,
                        "country": country,
                    }
                )
            )
        except Exception as lf_err:
            print(f"[Langfuse] Failed to log page hit: {lf_err}", flush=True)

    return {"hits": total}


@app.post("/api/feedback")
async def submit_feedback(req: Request, authorization: Optional[str] = Header(None)):
    """Persist a like/dislike on an assistant response (public, optional auth)."""
    body = await req.json()
    message_id = body.get("message_id", "")
    feedback = body.get("feedback", "")  # "like" or "dislike"
    if feedback not in ("like", "dislike") or not message_id:
        raise HTTPException(status_code=400, detail="message_id and feedback ('like'/'dislike') required")

    auth_header = _safe_authorization_header(authorization)
    current_user = _resolve_optional_user(auth_header)
    user_id = current_user.get("sub") if current_user else "anonymous"

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        _executor,
        lambda: analytics_service.log_feedback(
            message_id=message_id,
            feedback=feedback,
            conversation_id=body.get("conversation_id", ""),
            user_id=user_id,
            model=body.get("model", ""),
            prompt_preview=body.get("prompt_preview", ""),
            response_preview=body.get("response_preview", ""),
        ),
    )

    # ── Langfuse: ingest user feedback score in real-time ──────────
    from core.langfuse_integration import get_langfuse
    lf_client = get_langfuse()
    if lf_client:
        try:
            conv_id = body.get("conversation_id", "")
            # Look up trace_id from latest traces, or fall back to _last_trace_id globally
            trace_id = (_latest_traces.get(conv_id) if conv_id else None) or _last_trace_id
            
            # Score target: if trace_id is known, attach directly to the trace, else attach to session
            score_kwargs = {
                "name": "user_feedback",
                "value": 1.0 if feedback == "like" else 0.0,
                "data_type": "BOOLEAN",
                "comment": f"User voted {feedback} on message {message_id}",
            }
            if trace_id:
                score_kwargs["trace_id"] = trace_id
            elif conv_id:
                score_kwargs["session_id"] = conv_id
            else:
                score_kwargs["trace_id"] = _last_trace_id

            if score_kwargs.get("trace_id") or score_kwargs.get("session_id"):
                await loop.run_in_executor(
                    _executor,
                    lambda: lf_client.score(**score_kwargs)
                )
                print(f"[Langfuse] Ingested user feedback score ({feedback}) for trace {score_kwargs.get('trace_id') or score_kwargs.get('session_id')}", flush=True)
        except Exception as lf_err:
            print(f"[Langfuse] Failed to ingest feedback score: {lf_err}", flush=True)

    return {"success": True, "feedback": feedback}


@app.get("/api/admin/feedback/export")
async def admin_feedback_export(current_user: dict = Depends(get_current_user)):
    """Export all response feedback data. Admin-only."""
    user_email = current_user.get("email", "")
    if not analytics_service.is_admin(user_email):
        raise HTTPException(status_code=403, detail="Admin access required")
    return analytics_service.export_feedback_json()


@app.get("/api/admin/analytics/export")
async def admin_analytics_export(
    format: str = "csv",
    current_user: dict = Depends(get_current_user),
):
    """Download all chat analytics as CSV or JSON. Admin-only."""
    user_email = current_user.get("email", "")
    if not analytics_service.is_admin(user_email):
        raise HTTPException(status_code=403, detail="Admin access required")

    if format == "json":
        return analytics_service.export_chat_analytics_json()

    # Default: CSV download
    csv_data = analytics_service.export_chat_analytics_csv()
    return StreamingResponse(
        iter([csv_data]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=quasar_analytics.csv"},
    )


@app.get("/api/admin/analytics/summary")
async def admin_analytics_summary(current_user: dict = Depends(get_current_user)):
    """Aggregated analytics stats. Admin-only."""
    user_email = current_user.get("email", "")
    if not analytics_service.is_admin(user_email):
        raise HTTPException(status_code=403, detail="Admin access required")
    return analytics_service.get_summary()


# ── Conductor Dashboard API (#9) ─────────────────────────────

@app.get("/api/conductor/traces")
async def get_conductor_traces():
    """Get aggregated Conductor execution metrics and recent traces."""
    agent = get_agent()
    if not agent:
        return {"error": "Agent not loaded"}
    return agent.query_tracer.export_metrics()


@app.get("/api/conductor/traces/{trace_id}")
async def get_conductor_trace(trace_id: str):
    """Get detailed trace for a specific Conductor execution."""
    agent = get_agent()
    if not agent:
        return {"error": "Agent not loaded"}
    trace = agent.query_tracer.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace


@app.get("/api/conductor/health")
async def get_conductor_health():
    """Get Conductor system health: router, recovery, cache, pool."""
    agent = get_agent()
    if not agent:
        return {"error": "Agent not loaded"}
    health_data = {
        "conductor_model": getattr(agent.conductor, "conductor_model", "unknown"),
        "result_cache": agent.conductor.result_cache.get_stats() if hasattr(agent.conductor, "result_cache") else None,
        "dag_cache": agent.conductor.dag_cache.get_stats() if hasattr(agent.conductor, "dag_cache") else None,
        "workflow_memory": agent.conductor.workflow_memory.get_stats() if hasattr(agent.conductor, "workflow_memory") else None,
    }
    if hasattr(agent, "model_router") and agent.model_router:
        health_data["model_router"] = {
            "default_model": agent.model_router.default_model,
            "routing_table": {k: v["model"] for k, v in agent.model_router.ROUTING_TABLE.items()},
        }
    if hasattr(agent, "recovery_engine") and agent.recovery_engine:
        health_data["recovery_engine"] = agent.recovery_engine.get_stats()
    if hasattr(agent, "health_monitor") and agent.health_monitor:
        health_data["health_monitor"] = agent.health_monitor.get_status()
    return health_data


@app.get("/api/conductor/dag-cache")
async def get_dag_cache():
    """Get cached DAG decomposition patterns."""
    agent = get_agent()
    if not agent:
        return {"error": "Agent not loaded"}
    if hasattr(agent.conductor, "dag_cache"):
        return agent.conductor.dag_cache.get_stats()
    return {"entries": 0}
