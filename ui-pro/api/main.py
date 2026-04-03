"""
QUASAR Professional Chat Interface — FastAPI Backend

Provides RESTful + SSE endpoints wrapping the existing QuasarAgent.
Run with:  uvicorn api.main:app --reload --port 8000
"""

import sys, os, json, asyncio, uuid

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

# ── Observability: Loguru + Sentry ────────────────────────────────────────────
from core.logger import logger, init_sentry
init_sentry()  # no-op if SENTRY_DSN env var is not set
logger.info("[QUASAR API] Starting up")

app = FastAPI(
    title="QUASAR API",
    description="Backend API for the QUASAR Professional Chat Interface",
    version="2.0.0",
)

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
    allow_origin_regex=r"https://.*\.vercel\.app|https://.*\.onrender\.com|https://(www\.)?quasarassistant\.com|http://localhost:3000|http://127\.0\.0\.1:3000",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Catch-all exception handler — ensures a proper JSON 500 with CORS headers ──
from starlette.responses import JSONResponse

@app.exception_handler(Exception)
async def _global_exception_handler(request, exc):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

# Thread pool for running synchronous agent calls
_executor = ThreadPoolExecutor(max_workers=4)

# ── Channel Routers ──────────────────────────────────────────
try:
    from api.channels.telegram import router as telegram_router
    app.include_router(telegram_router)
    print("[INFO] Telegram channel router registered at /channels/telegram")
except ImportError as e:
    print(f"[WARNING] Telegram channel not loaded: {e}")


# ── Models ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    model: Optional[str] = "gpt-4o"

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

# ── Auth Service Instance ──
from services.auth import AuthService
from services.conversation_service import ConversationService
from services.provider_file_service import (
    ProviderFileError,
    ProviderFileService,
    UploadedChatFile,
)
from core.llm_client import detect_provider
auth_service = AuthService()
conversation_service = ConversationService()
provider_file_service = ProviderFileService()

def get_current_user(authorization: Optional[str] = Header(None)):
    if not isinstance(authorization, str) or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ")[1]
    payload = auth_service.verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload


class ConversationCreate(BaseModel):
    title: Optional[str] = "New Chat"


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


def _sse_error_response(message: str) -> StreamingResponse:
    async def generate():
        yield f"data: {json.dumps({'type': 'error', 'content': message})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=_stream_headers(),
    )


def _extract_document_preview_text(filename: str, content_type: str, raw: bytes) -> str:
    """Best-effort text extraction used only when documents accompany images/FITS."""
    import io

    if content_type == "application/pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(raw))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return f"\n\n### Attached PDF: {filename}\n{text[:8000]}"
        except ImportError:
            return f"\n\n[PDF: {filename} - install pypdf to extract text]"
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


def _stream_chat_response(
    request: ChatRequest,
    authorization: Optional[str] = None,
    attachment_context: Optional[Dict[str, Any]] = None,
) -> StreamingResponse:
    """Shared SSE chat pipeline used by both /api/chat and /api/chat/upload."""
    agent = get_agent()
    auth_header = _safe_authorization_header(authorization)
    current_user = _resolve_optional_user(auth_header)

    requested_model = request.model or (getattr(agent.config, "model", None) if agent else None) or "gpt-4o"
    provider = detect_provider(requested_model)
    current_user_id = current_user.get("sub") if current_user else None

    # ── Conversation persistence: auto-create + save user message ──────
    conv_id = request.conversation_id
    if current_user_id:
        try:
            if not conv_id:
                title = conversation_service.generate_title_from_message(request.message)
                conv_id = conversation_service.create_conversation(current_user_id, title)
                logger.info(f"[CHAT] Auto-created conversation {conv_id} for user {current_user_id}")
            conversation_service.save_message(conv_id, "user", request.message)
        except Exception as e:
            logger.warning(f"[CHAT] Failed to persist user message: {e}")

    if attachment_context is None and provider in {"openai", "anthropic", "google"}:
        attachment_context = provider_file_service.get_active_files(
            provider=provider,
            conversation_id=request.conversation_id,
            user_id=current_user_id,
        )
    attachment_context = attachment_context or {}
    active_attachments = attachment_context.get("attachments") or []

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
        enriched_message = request.message
        if current_user:
            user_id = current_user.get("sub")
            if user_id:
                try:
                    yield _sse_status("Searching personal knowledge base", "running")
                    loop2 = asyncio.get_event_loop()

                    def _rag_search():
                        from services.rag_service import RAGService

                        svc = RAGService(user_id=user_id)
                        return svc.search(request.message, k=4, include_personal=True)

                    rag_docs = await loop2.run_in_executor(_executor, _rag_search)
                    if rag_docs:
                        ctx_lines = []
                        for d in rag_docs:
                            src = d.metadata.get("source_file", "personal doc")
                            ctx_lines.append(f"[From: {src}]\n{d.page_content.strip()}")
                        context_block = "\n\n---\n".join(ctx_lines)
                        enriched_message = (
                            "The user has the following relevant documents in their personal "
                            f"knowledge base:\n\n{context_block}\n\n---\nUser's question: {request.message}"
                        )
                    yield _sse_status("Searching personal knowledge base", "completed")
                except Exception as e:
                    print(f"[WARN] Personal RAG search failed: {e}")
                    yield _sse_status("Searching personal knowledge base", "completed")

        effective_request = ChatRequest(
            message=enriched_message,
            conversation_id=request.conversation_id,
            model=requested_model,
        )

        if requested_model != agent.config.model:
            agent.set_model(requested_model)

        try:
            loop = asyncio.get_event_loop()

            print("[INFO] Routing query to standard Response API (tool-calling loop)")

            queue = asyncio.Queue()

            def on_token(token: str):
                if token:
                    asyncio.run_coroutine_threadsafe(queue.put(("token", token)), loop)

            def _run_agent():
                try:
                    # Clear stale results from previous conversations
                    agent.last_run_result = None

                    effective_user_id = (current_user.get("sub") if current_user else None) or "anonymous"

                    def _on_status(step: str, state: str):
                        asyncio.run_coroutine_threadsafe(queue.put(("status", step, state)), loop)

                    res = agent.stream_response_api(
                        effective_request.message,
                        message_placeholder=None,
                        user_id=effective_user_id,
                        on_token=on_token,
                        on_status=_on_status,
                        attachments=active_attachments,
                    )
                    asyncio.run_coroutine_threadsafe(queue.put(("done", res)), loop)
                except Exception as e:
                    print(f"Agent error: {e}")
                    asyncio.run_coroutine_threadsafe(queue.put(("error", str(e))), loop)

            loop.run_in_executor(_executor, _run_agent)

            first_token = True
            response_text = ""

            while True:
                msg = await queue.get()
                if isinstance(msg, tuple) and len(msg) == 3:
                    msg_type, step, state = msg
                    if msg_type == "status":
                        if isinstance(step, str) and step.startswith("__event__"):
                            try:
                                event_json = step[len("__event__"):]
                                yield f"data: {event_json}\n\n"
                            except Exception:
                                yield _sse_status(step, state)
                        else:
                            yield _sse_status(step, state)
                        continue

                msg_type, payload = msg[0], msg[1]
                if msg_type == "done":
                    response_text = payload
                    break
                if msg_type == "error":
                    response_text = f"An error occurred: {payload}"
                    break
                if msg_type == "token":
                    first_token = False
                    data = json.dumps({"type": "token", "content": payload})
                    yield f"data: {data}\n\n"

            last_run_result = getattr(agent, "last_run_result", None)
            if last_run_result:
                result_type = last_run_result.get("type", "")
                tool_name_raw = last_run_result.get("tool_name") or (
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
                    "input": last_run_result.get("params", {}),
                    "output": "Found results",
                })
                yield f"data: {tool_event}\n\n"
                await asyncio.sleep(0.05)

                if result_type == "data":
                    df = last_run_result.get("data")
                    if df is not None and hasattr(df, "to_dict"):
                        try:
                            import math
                            import pandas as pd

                            alma_display_cols = [
                                ("project_code", "Project"),
                                ("target_name", "Target"),
                                ("band_list", "Band"),
                                ("frequency", "Freq (GHz)"),
                                ("min_frequency", "Min Freq (GHz)"),
                                ("max_frequency", "Max Freq (GHz)"),
                                ("spatial_resolution", "Res (arcsec)"),
                                ("s_resolution", "Res (arcsec)"),
                                ("t_exptime", "Exp (s)"),
                                ("pi_name", "PI"),
                                ("obs_release_date", "Release"),
                                ("member_ous_uid", "MOUS ID"),
                            ]
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

                            sub = df[sel_cols].head(50).copy()
                            sub.columns = display_cols

                            per_row_links = []
                            if "access_url" in df.columns:
                                per_row_links = df["access_url"].head(50).fillna("").tolist()
                            elif "member_ous_uid" in df.columns:
                                per_row_links = [
                                    f"https://almascience.nrao.edu/aq/?member_ous_id={v}"
                                    if pd.notna(v) and str(v).strip() else ""
                                    for v in df["member_ous_uid"].head(50)
                                ]

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
                            #    Uses CDS HiPS2FITS to generate DSS2 color cutouts
                            ra_col = next((c for c in ["s_ra", "ra"] if c in df.columns), None)
                            dec_col = next((c for c in ["s_dec", "dec"] if c in df.columns), None)
                            has_preview = False
                            if ra_col and dec_col:
                                for i, (_, orig_row) in enumerate(df.head(50).iterrows()):
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

                            metrics = [{"label": "Results", "value": len(df), "color": "blue"}]
                            if "band_list" in df.columns:
                                metrics.append({
                                    "label": "Bands",
                                    "value": int(df["band_list"].astype(str).nunique()),
                                    "color": "purple",
                                })

                            archive_link = ""
                            if "access_url" in df.columns and not df["access_url"].isna().all():
                                archive_link = str(df["access_url"].dropna().iloc[0])
                            elif "member_ous_uid" in df.columns:
                                mous = next((str(v) for v in df["member_ous_uid"] if pd.notna(v) and str(v).strip()), "")
                                if mous:
                                    archive_link = f"https://almascience.nrao.edu/aq/?member_ous_id={mous}"

                            table_event = json.dumps({
                                "type": "data",
                                "metrics": metrics,
                                "columns": list(sub.columns),
                                "rows": rows,
                                "sourceName": last_run_result.get("filter_label")
                                or last_run_result.get("source", "ALMA Archive"),
                                "archiveLink": archive_link,
                                "hasRowLinks": any(bool(r.get("_link")) for r in rows),
                                "hasPreview": has_preview,
                            })
                            yield f"data: {table_event}\n\n"
                            await asyncio.sleep(0.05)
                        except Exception as e:
                            print(f"[WARN] Could not serialize data table for UI: {e}")

                elif result_type == "papers":
                    papers = last_run_result.get("papers", [])
                    if papers:
                        papers_event = json.dumps({"type": "papers", "papers": papers})
                        yield f"data: {papers_event}\n\n"
                        await asyncio.sleep(0.05)

                elif result_type == "notebook":
                    nb = last_run_result.get("notebook", {})
                    if nb:
                        notebook_event = json.dumps({"type": "notebook", **nb})
                        yield f"data: {notebook_event}\n\n"
                        await asyncio.sleep(0.05)

            if response_text and first_token:
                data = json.dumps({"type": "token", "content": response_text})
                yield f"data: {data}\n\n"

            # ── Persist assistant response to DB ──────────────────────
            if current_user_id and conv_id and response_text:
                try:
                    conversation_service.save_message(conv_id, "assistant", response_text)
                except Exception as e:
                    logger.warning(f"[CHAT] Failed to persist assistant message: {e}")

            # Send back the conversation_id so the frontend can track it
            if conv_id:
                meta_event = json.dumps({"type": "conversation_meta", "conversation_id": conv_id})
                yield f"data: {meta_event}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            print(f"Chat route error: {e}")
            error_data = json.dumps({"type": "error", "content": str(e)})
            yield f"data: {error_data}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=_stream_headers(),
    )


# ── Endpoints ────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "service": "QUASAR API", "version": "2.0.0"}


@app.get("/api/models")
async def list_models():
    cloud_models = [
            # ── Anthropic Claude (Latest) ──────────────────────────
            "claude-sonnet-4-6",       # Latest Sonnet — best coding, default free/pro
            "claude-opus-4-6",         # Latest Opus — smartest, 1M context (beta)
            "claude-sonnet-4-5",       # Sonnet 4.5 — coding/computer use
            "claude-opus-4-5",         # Opus 4.5 — best coding & agents
            "claude-haiku-4-5",        # Haiku 4.5 — fastest & cheapest
            "claude-opus-4-1",         # Opus 4.1 — agentic tasks
            "claude-sonnet-4",         # Sonnet 4
            "claude-opus-4",           # Opus 4
            "claude-3-7-sonnet-20250219",  # Claude 3.7 Sonnet — hybrid reasoning
            "claude-3-5-sonnet-20241022",  # Claude 3.5 Sonnet (Oct 2024)
            "claude-3-5-haiku-20241022",   # Claude 3.5 Haiku
            # ── OpenAI GPT (Latest) ───────────────────────────────
            "gpt-5.4",                 # GPT-5.4 — latest flagship (Mar 2026)
            "gpt-5.4-2026-03-05",      # GPT-5.4 snapshot
            "gpt-5",                   # GPT-5
            "gpt-5-mini",              # GPT-5 Mini — lower latency
            "gpt-5-nano",              # GPT-5 Nano — cheapest
            "gpt-4o",                  # GPT-4o — omni flagship
            "gpt-4o-mini",             # GPT-4o Mini — fast & cheap
            "gpt-4.1",                 # GPT-4.1 — coding & long-context (1M)
            "gpt-4.1-mini",            # GPT-4.1 Mini — balanced
            "gpt-4.1-nano",            # GPT-4.1 Nano — fastest/cheapest
            "o3",                      # o3 — advanced reasoning
            "o4-mini",                 # o4-mini — reasoning, lower cost
            # ── Google Gemini (Latest) ────────────────────────────
            "gemini-3.1-pro",          # Gemini 3.1 Pro — best reasoning
            "gemini-3.1-flash",        # Gemini 3.1 Flash
            "gemini-3.1-flash-lite",   # Gemini 3.1 Flash Lite
            "gemini-3-flash",          # Gemini 3 Flash (default app)
            "gemini-3-pro",            # Gemini 3 Pro
            "gemini-2.5-pro",          # Gemini 2.5 Pro
            "gemini-2.5-flash",        # Gemini 2.5 Flash
            "gemini-2.5-flash-lite",   # Gemini 2.5 Flash Lite
            "gemini-2.0-flash",        # Gemini 2.0 Flash
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
        
        if not email:
            raise HTTPException(status_code=400, detail="Google account has no email")
            
        success, user_id, msg, token = auth_service.register_or_login_google_user(email, name)
        
        if not success:
            raise HTTPException(status_code=400, detail=msg)
            
        return {
            "token": token,
            "user": {
                "id": user_id,
                "username": email,
                "display_name": name,
                "auth_provider": "google"
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
    conv_id = conversation_service.create_conversation(user_id, req.title)
    return {"id": conv_id, "title": req.title}


@app.get("/api/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str, current_user: dict = Depends(get_current_user)):
    """Fetch all messages for a conversation."""
    messages = conversation_service.get_conversation_messages(conversation_id)
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
    conversation_service.update_conversation_title(conversation_id, req.title)
    return {"status": "ok"}


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a conversation and all its messages."""
    conversation_service.delete_conversation(conversation_id)
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(request: ChatRequest, authorization: Optional[str] = Header(None)):
    """Stream a chat response via SSE using the shared chat pipeline."""
    return _stream_chat_response(request, authorization=authorization)


    # ── Resolve user from optional token (for personal RAG) ──
    _current_user = None
    if isinstance(authorization, str) and authorization.startswith("Bearer "):
        try:
            _current_user = auth_service.verify_token(authorization.split(" ")[1])
        except Exception:
            pass

    async def generate():
        if agent is None:
            err = get_agent_error() or "Unknown initialization error"
            mock_response = (
                f"🚨 **BACKEND CRASH TRACE** 🚨\n\n"
                f"QuasarAgent failed to load. Here is the exact Python error:\n\n"
                f"```python\n{err}\n```\n"
            )
            # Stream chunk by chunk so it looks like typing and renders markdown correctly
            for chunk in mock_response.split("\n"):
                data = json.dumps({"type": "token", "content": chunk + "\n"})
                yield f"data: {data}\n\n"
                await asyncio.sleep(0.02)
            yield "data: [DONE]\n\n"
            return

        # Helper to emit status events
        def _status(step: str, state: str = "running"):
            return f"data: {json.dumps({'type': 'status', 'step': step, 'state': state})}\n\n"

        # ── Personal RAG retrieval (authenticated users only) ──────────────
        enriched_message = request.message
        if _current_user:
            user_id = _current_user.get("sub")
            if user_id:
                try:
                    yield _status("Searching personal knowledge base", "running")
                    loop2 = asyncio.get_event_loop()
                    def _rag_search():
                        from services.rag_service import RAGService
                        svc = RAGService(user_id=user_id)
                        docs = svc.search(request.message, k=4, include_personal=True)
                        return docs
                    rag_docs = await loop2.run_in_executor(_executor, _rag_search)
                    if rag_docs:
                        ctx_lines = []
                        for d in rag_docs:
                            src = d.metadata.get("source_file", "personal doc")
                            ctx_lines.append(f"[From: {src}]\n{d.page_content.strip()}")
                        context_block = "\n\n---\n".join(ctx_lines)
                        enriched_message = (
                            f"The user has the following relevant documents in their personal knowledge base:\n\n"
                            f"{context_block}\n\n"
                            f"---\nUser's question: {request.message}"
                        )
                        yield _status("Searching personal knowledge base", "completed")
                    else:
                        yield _status("Searching personal knowledge base", "completed")
                except Exception as e:
                    print(f"[WARN] Personal RAG search failed: {e}")
                    yield _status("Searching personal knowledge base", "completed")

        # Use enriched message for the rest of the pipeline
        effective_request = ChatRequest(
            message=enriched_message,
            conversation_id=request.conversation_id,
            model=request.model
        )

        # ── QUASAR system prompt for non-OpenAI models ──────────────────────
        # OpenAI models go through agent.stream_response_api() which uses the
        # full 27-tool system prompt.  Gemini & Claude skip that path but still
        # need to know they are QUASAR so responses are domain-specific.
        from datetime import datetime as _dt
        QUASAR_SYSTEM_PROMPT = (
            "You are Quasar, an expert AI assistant for radio astronomy.\n\n"
            "You have deep knowledge of the ALMA Science Archive, radio interferometry, "
            "spectral line identification, UV coverage analysis, CASA calibration, "
            "and astronomical data reduction.\n\n"
            "GUIDELINES:\n"
            "- You specialize in ALMA, VLA, VLBA, and GBT data.\n"
            "- Help users find, analyze, and visualize radio astronomy data.\n"
            "- When users ask what you can do, describe your radio astronomy capabilities: "
            "searching the ALMA archive, spectral line ID via Splatalogue, publication "
            "search via NASA ADS, CASA script generation, UV coverage analysis, "
            "multi-archive cross-matching, and web search for real-time info.\n"
            "- If asked about observations, cite ALMA bands, frequencies, resolutions.\n"
            "- Be concise, scientific, and action-oriented.\n"
            "- FORMATTING: Always use proper Markdown. For tabular data, ALWAYS use "
            "Markdown table syntax with | pipes and | --- | header separators. "
            "Never use space-aligned text for tables. Use ## headings, **bold** for "
            "key values, and bullet lists for structure.\n"
            f"- Current date: {_dt.now().strftime('%Y-%m-%d')}\n"
        )

        # ── Unified model routing ─────────────────────────────────────────
        # All models (OpenAI, Claude, Gemini, Local) now go through the
        # same full tool-calling pipeline via LLMClient.
        model_name = request.model or "gpt-4o"
        if model_name != agent.config.model:
            agent.set_model(model_name)

        # ── Real OpenAI agent execution ────────────────────────────────────

        try:
            loop = asyncio.get_event_loop()

            # ── Complexity check (informational only — does NOT change routing) ──
            # All queries go through stream_response_api which has the full 27-tool
            # calling loop. The RLM path was removed because rlm.execute() uses a
            # pure LLM with no tools, causing hallucinated step descriptions instead
            # of actual archive searches, plots, and script generation.
            yield _status("Analyzing prompt complexity")
            await asyncio.sleep(0.05)
            yield _status("Analyzing prompt complexity", "completed")
            # Paper standard: RLM REPL fires ONLY for massive data contexts
            # (>80k chars ≈ 20k tokens — e.g. large FITS headers, multi-MB CSVs).
            # All normal queries, including multi-step research, always use the
            # 27-tool calling loop via stream_response_api.
            REPL_CONTEXT_THRESHOLD = 80_000

            if len(enriched_message) > REPL_CONTEXT_THRESHOLD:
                # ── RLM REPL Path (beyond-context massive data) ──
                print(f"[INFO] Message size {len(enriched_message):,} chars > {REPL_CONTEXT_THRESHOLD:,} threshold → RLM REPL")
                yield _status("Launching RLM REPL (beyond-context mode)")
                await asyncio.sleep(0.05)
                yield _status("Launching RLM REPL (beyond-context mode)", "completed")

                repl_queue = asyncio.Queue()

                def repl_status(step: str, state: str):
                    asyncio.run_coroutine_threadsafe(repl_queue.put(("status", step, state)), loop)

                def repl_token(tok: str):
                    if tok:
                        asyncio.run_coroutine_threadsafe(repl_queue.put(("token", tok)), loop)

                def _run_repl():
                    try:
                        res = agent.rlm.execute(
                            request.message,       # just the query — not the giant blob
                            context=enriched_message,  # giant data goes into REPL as variable
                            use_repl=True,
                            status_callback=repl_status,
                            on_token=repl_token,
                        )
                        asyncio.run_coroutine_threadsafe(repl_queue.put(("done", res)), loop)
                    except Exception as e:
                        print(f"[RLM REPL error] {e}")
                        asyncio.run_coroutine_threadsafe(repl_queue.put(("error", str(e))), loop)

                loop.run_in_executor(_executor, _run_repl)

                response_text = ""
                has_repl_tokens = False
                while True:
                    msg = await repl_queue.get()
                    mtype = msg[0]
                    if mtype == "done":
                        response_text = msg[1]
                        if not has_repl_tokens and response_text:
                            data = json.dumps({"type": "token", "content": response_text})
                            yield f"data: {data}\n\n"
                        break
                    elif mtype == "error":
                        response_text = f"RLM REPL error: {msg[1]}"
                        data = json.dumps({"type": "token", "content": response_text})
                        yield f"data: {data}\n\n"
                        break
                    elif mtype == "status":
                        yield _status(msg[1], msg[2])
                    elif mtype == "token":
                        has_repl_tokens = True
                        data = json.dumps({"type": "token", "content": msg[1]})
                        yield f"data: {data}\n\n"

                agent.memory.add_message("user", request.message)
                agent.memory.add_message("assistant", response_text)
                agent.last_run_result = None
                yield "data: [DONE]\n\n"
                return

            # ── Standard Agent Path (all normal queries) ──
            print("[INFO] Routing query to standard Response API (tool-calling loop)")

            queue = asyncio.Queue()
            
            def on_token(token: str):
                if token:
                    asyncio.run_coroutine_threadsafe(queue.put(("token", token)), loop)
                    
            def _run_agent():
                try:
                    # Clear stale results from previous conversations
                    agent.last_run_result = None

                    # Use authenticated user_id for mem0 isolation.
                    # Anonymous users get a sentinel — no stored long-term memory.
                    _uid = (_current_user.get("sub") if _current_user else None) or "anonymous"

                    def _on_status(step: str, state: str):
                        asyncio.run_coroutine_threadsafe(
                            queue.put(("status", step, state)), loop
                        )

                    res = agent.stream_response_api(
                        effective_request.message,
                        message_placeholder=None,
                        user_id=_uid,
                        on_token=on_token,
                        on_status=_on_status,
                    )
                    asyncio.run_coroutine_threadsafe(queue.put(("done", res)), loop)
                except Exception as e:
                    print(f"Agent error: {e}")
                    asyncio.run_coroutine_threadsafe(queue.put(("error", str(e))), loop)

            agent_task = loop.run_in_executor(_executor, _run_agent)

            first_token = True
            response_text = ""
            
            while True:
                msg = await queue.get()
                if isinstance(msg, tuple) and len(msg) == 3:
                    msg_type, step, state = msg
                    if msg_type == "status":
                        # Check for structured Conductor events (prefixed with __event__)
                        if isinstance(step, str) and step.startswith("__event__"):
                            try:
                                event_json = step[len("__event__"):]
                                yield f"data: {event_json}\n\n"
                            except Exception:
                                yield _status(step, state)
                        else:
                            yield _status(step, state)
                        continue
                msg_type, payload = msg[0], msg[1]
                if msg_type == "done":
                    response_text = payload
                    break
                elif msg_type == "error":
                    response_text = f"An error occurred: {payload}"
                    break
                elif msg_type == "token":
                    first_token = False
                    data = json.dumps({"type": "token", "content": payload})
                    yield f"data: {data}\n\n"

            # ── Tool call results ──
            last_run_result = getattr(agent, 'last_run_result', None)

            if last_run_result:
                result_type = last_run_result.get("type", "")
                # Use actual tool name stored by each tool
                tool_name_raw = last_run_result.get("tool_name") or (
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
                    "input": last_run_result.get("params", {}),
                    "output": "Found results"
                })
                yield f"data: {tool_event}\n\n"
                await asyncio.sleep(0.05)

                if result_type == "data":
                    df = last_run_result.get("data")
                    if df is not None and hasattr(df, 'to_dict'):
                        try:
                            import math

                            # ── Key display columns — ordered for readability ──
                            ALMA_DISPLAY_COLS = [
                                ("project_code",    "Project"),
                                ("target_name",     "Target"),
                                ("band_list",       "Band"),
                                ("frequency",       "Freq (GHz)"),
                                ("min_frequency",   "Min Freq (GHz)"),
                                ("max_frequency",   "Max Freq (GHz)"),
                                ("spatial_resolution", "Res (arcsec)"),
                                ("s_resolution",    "Res (arcsec)"),
                                ("t_exptime",       "Exp (s)"),
                                ("pi_name",         "PI"),
                                ("obs_release_date","Release"),
                                ("member_ous_uid",  "MOUS ID"),
                            ]
                            # Only keep first match per display name (avoid duplicate Res column)
                            seen_display = set()
                            sel_cols, display_cols = [], []
                            for raw, nice in ALMA_DISPLAY_COLS:
                                if raw in df.columns and nice not in seen_display:
                                    sel_cols.append(raw)
                                    display_cols.append(nice)
                                    seen_display.add(nice)

                            if not sel_cols:
                                sel_cols = list(df.columns[:8])
                                display_cols = sel_cols

                            sub = df[sel_cols].head(50).copy()
                            sub.columns = display_cols

                            # ── Per-row archive links from access_url ──
                            # access_url is built by _standardize_columns from member_ous_uid
                            per_row_links = []
                            if "access_url" in df.columns:
                                per_row_links = df["access_url"].head(50).fillna("").tolist()
                            elif "member_ous_uid" in df.columns:
                                per_row_links = [
                                    f"https://almascience.nrao.edu/aq/?member_ous_id={v}"
                                    if pd.notna(v) and str(v).strip() else ""
                                    for v in df["member_ous_uid"].head(50)
                                ]

                            def _fmt(v):
                                """Format a cell value cleanly."""
                                if v is None or (isinstance(v, float) and math.isnan(v)):
                                    return ""
                                if isinstance(v, float):
                                    # Round long decimals to 3 significant figures
                                    return f"{v:.3f}".rstrip("0").rstrip(".")
                                return str(v)[:60]

                            for col in sub.columns:
                                sub[col] = sub[col].apply(_fmt)

                            rows = sub.to_dict("records")
                            # Attach per-row link
                            for i, link in enumerate(per_row_links):
                                if i < len(rows):
                                    rows[i]["_link"] = link

                            # ── Inject sky preview thumbnail URLs ──────────
                            ra_col = next((c for c in ["s_ra", "ra"] if c in df.columns), None)
                            dec_col = next((c for c in ["s_dec", "dec"] if c in df.columns), None)
                            has_preview = False
                            if ra_col and dec_col:
                                for i, (_, orig_row) in enumerate(df.head(50).iterrows()):
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

                            source = last_run_result.get("source", "ALMA")
                            filter_label = last_run_result.get("filter_label", source)

                            # Build footer ALMA archive link
                            alma_link = None
                            if "target_name" in df.columns and not df.empty:
                                _tgt = str(df["target_name"].iloc[0]).replace(" ", "+")
                                alma_link = f"https://almascience.eso.org/aq/?target={_tgt}"

                            metrics = [
                                {"label": "Observations",  "value": str(len(df)), "color": "text-primary"},
                                {"label": "Showing",       "value": str(len(rows)), "color": "text-indigo-400"},
                                {"label": "Source",        "value": filter_label, "color": "text-emerald-400"},
                            ]
                            data_event = json.dumps({
                                "type": "data",
                                "content": {
                                    "sourceName": source,
                                    "metrics": metrics,
                                    "columns": display_cols,
                                    "rows": rows,
                                    "archiveLink": alma_link,
                                    "hasRowLinks": bool(per_row_links),
                                    "hasPreview": has_preview,
                                }
                            })
                            yield f"data: {data_event}\n\n"
                            await asyncio.sleep(0.05)
                        except Exception as e:
                            import traceback
                            print(f"[WARN] Failed to serialize data table: {e}")
                            traceback.print_exc()

                elif result_type == "papers":
                    papers = last_run_result.get("papers")
                    
                    # Backwards compatibility: if papers is missing, try to read from 'data' DataFrame
                    if papers is None and "data" in last_run_result:
                        df = last_run_result["data"]
                        if hasattr(df, "to_dict"):
                            papers = df.to_dict("records")
                            
                    papers = papers or []
                    
                    if papers:
                        papers_event = json.dumps({
                            "type": "papers",
                            "content": papers
                        })
                        yield f"data: {papers_event}\n\n"
                        await asyncio.sleep(0.05)

                elif result_type == "notebook":
                    notebook_data = last_run_result.get("notebook_data")
                    title = last_run_result.get("title", "notebook")
                    if notebook_data:
                        nb_event = json.dumps({
                            "type": "notebook",
                            "content": {
                                "title": title,
                                "data": notebook_data
                            }
                        })
                        yield f"data: {nb_event}\n\n"
                        await asyncio.sleep(0.05)

            # Clear last_run_result
            agent.last_run_result = None

            yield "data: [DONE]\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_data = json.dumps({"type": "error", "content": str(e)})
            yield f"data: {error_data}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat/upload")
async def chat_with_files(
    message: str = Form(""),
    conversation_id: Optional[str] = Form(None),
    model: Optional[str] = Form("gpt-4o"),
    files: PyList[UploadFile] = File(default=[]),
    authorization: Optional[str] = Header(None),
):
    """Stream a chat response with attached files (images/documents) via SSE."""
    auth_header = _safe_authorization_header(authorization)
    current_user = _resolve_optional_user(auth_header)
    user_id = current_user.get("sub") if current_user else None
    selected_model = model or "gpt-4o"
    provider = detect_provider(selected_model)

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

    if image_contents:
        import openai as _openai

        if mixed_document_previews:
            enriched_text += "".join(preview for preview in mixed_document_previews if preview)

        client = _openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        content_parts = []
        if enriched_text:
            content_parts.append({"type": "text", "text": enriched_text})
        content_parts.extend(image_contents)

        async def vision_stream():
            try:
                yield _sse_status("Analyzing image with vision model", "running")
                loop = asyncio.get_event_loop()

                def _call():
                    return client.chat.completions.create(
                        model=selected_model,
                        messages=[{"role": "user", "content": content_parts}],
                        stream=True,
                        max_tokens=1024,
                    )

                stream = await loop.run_in_executor(_executor, _call)
                yield _sse_status("Analyzing image with vision model", "completed")

                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        data = json.dumps({"type": "token", "content": delta})
                        yield f"data: {data}\n\n"

                yield "data: [DONE]\n\n"
            except Exception as e:
                err = json.dumps({"type": "error", "content": str(e)})
                yield f"data: {err}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            vision_stream(),
            media_type="text/event-stream",
            headers=_stream_headers(),
        )

    attachment_context = None
    if document_uploads:
        try:
            attachment_context = provider_file_service.prepare_files(
                provider=provider,
                model=selected_model,
                files=document_uploads,
                conversation_id=conversation_id,
                user_id=user_id,
            )
        except ProviderFileError as exc:
            return _sse_error_response(str(exc))

    req = ChatRequest(
        message=message.strip() or message,
        conversation_id=conversation_id,
        model=selected_model,
    )
    return _stream_chat_response(req, authorization=auth_header, attachment_context=attachment_context)

    import base64, io
    logger.info(f"[UPLOAD] Received {len(files)} file(s), message={message[:80]!r}")

    # Split files into images vs documents
    image_contents = []   # OpenAI vision content dicts
    enriched_text = message.strip()

    for f in files:
        content_type = f.content_type or ""
        raw = await f.read()

        if content_type.startswith("image/"):
            # Build vision content block
            b64 = base64.b64encode(raw).decode("utf-8")
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:{content_type};base64,{b64}", "detail": "auto"},
            })

        elif content_type == "application/pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(raw))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                enriched_text += f"\n\n### Attached PDF: {f.filename}\n{text[:8000]}"
            except ImportError:
                enriched_text += f"\n\n[PDF: {f.filename} — install pypdf to extract text]"
            except Exception as e:
                enriched_text += f"\n\n[PDF: {f.filename} — extraction failed: {e}]"

        elif f.filename and (f.filename.endswith(".fits") or f.filename.endswith(".fit")):
            try:
                from services.fits_processing import FITSProcessingService
                
                # Extract header text
                metadata = FITSProcessingService.extract_metadata(raw)
                header_str = "\n".join([f"{k}: {v}" for k, v in metadata.items() if k != "error"])
                err = metadata.get("error", "")
                
                if hdrs := header_str.strip():
                    enriched_text += f"\n\n### Attached FITS: {f.filename}\n**Header Metadata:**\n```yaml\n{hdrs}\n```\n"
                if err:
                    enriched_text += f"\n[FITS Metadata Error: {err}]"
                    
                # Render visual preview
                b64_img = FITSProcessingService.generate_preview(raw)
                if b64_img:
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_img}", "detail": "high"}
                    })
                    enriched_text += f"*(A 2D visual representation of this FITS file has been attached as an image for your analysis.)*\n"
                else:
                    enriched_text += f"*(Could not generate a 2D preview image for this FITS data.)*\n"
            except Exception as e:
                enriched_text += f"\n\n[FITS: {f.filename} — extraction failed: {e}]"

        elif content_type in ("text/plain", "text/csv", "text/markdown", "application/json") or \
             (f.filename and any(f.filename.endswith(ext) for ext in [".csv", ".txt", ".md", ".json"])):
            try:
                text = raw.decode("utf-8", errors="replace")[:8000]
                enriched_text += f"\n\n### Attached file: {f.filename}\n```\n{text}\n```"
            except Exception:
                enriched_text += f"\n\n[Binary file: {f.filename} ({len(raw)/1024:.1f} KB)]"
        else:
            enriched_text += f"\n\n[Attached file: {f.filename} ({len(raw)/1024:.1f} KB)]"

    # ── If there are images, use OpenAI vision directly (streaming) ─────────
    if image_contents:
        import openai as _openai
        client = _openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Build the multimodal content array
        content_parts = []
        if enriched_text:
            content_parts.append({"type": "text", "text": enriched_text})
        content_parts.extend(image_contents)

        async def vision_stream():
            def _vstatus(step: str, state: str = "running") -> str:
                return f"data: {json.dumps({'type': 'status', 'step': step, 'state': state})}\n\n"
            try:
                yield _vstatus("Analyzing image with vision model", "running")
                # Run the blocking OpenAI call in a thread
                loop = asyncio.get_event_loop()

                def _call():
                    return client.chat.completions.create(
                        model=model or "gpt-4o",
                        messages=[{"role": "user", "content": content_parts}],
                        stream=True,
                        max_tokens=1024,
                    )

                stream = await loop.run_in_executor(_executor, _call)
                yield _vstatus("Analyzing image with vision model", "completed")

                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        data = json.dumps({"type": "token", "content": delta})
                        yield f"data: {data}\n\n"

                yield "data: [DONE]\n\n"
            except Exception as e:
                err = json.dumps({"type": "error", "content": str(e)})
                yield f"data: {err}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            vision_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    # ── No images — delegate to regular chat endpoint ────────────────────────
    req = ChatRequest(message=enriched_text or message, conversation_id=conversation_id, model=model)
    return _stream_chat_response(req, authorization=authorization)


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
                return svc.ingest_document(tmp_path, personal=True)

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


@app.get("/api/conversations")

async def list_conversations():
    """Return conversation list — placeholder for future DB integration."""
    return {"conversations": []}


@app.post("/api/conversations")
async def create_conversation(req: ConversationCreate):
    """Create a new conversation."""
    return {"id": str(uuid.uuid4()), "title": req.title}


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
