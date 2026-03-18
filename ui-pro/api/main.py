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
from typing import Optional, List, Dict
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

# CORS — allow the Next.js dev server and deployed domains
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.vercel\.app|https://.*\.onrender\.com|http://localhost:3000|http://127\.0\.0\.1:3000",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
auth_service = AuthService()

def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
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


# ── Endpoints ────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "service": "QUASAR API", "version": "2.0.0"}


@app.get("/api/models")
async def list_models():
    return {
        "models": [
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
    }

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


@app.post("/api/chat")
async def chat(request: ChatRequest, authorization: str = Header(None)):
    """Stream a chat response via SSE using the real QuasarAgent."""
    agent = get_agent()

    # ── Resolve user from optional token (for personal RAG) ──
    _current_user = None
    if authorization and authorization.startswith("Bearer "):
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
            f"- Current date: {_dt.now().strftime('%Y-%m-%d')}\n"
        )

        # ── Gemini model routing ──────────────────────────────────────────
        model_name = request.model or "gpt-4o"
        if model_name.startswith("gemini-"):
            try:
                from google import genai as ggenai
                gemini_client = ggenai.Client(api_key=os.getenv("GEMINI_API_KEY"))

                yield _status(f"Routing to {model_name}", "running")

                def _gemini_call():
                    # Prepend QUASAR system prompt so Gemini knows its role
                    full_prompt = QUASAR_SYSTEM_PROMPT + "\n\nUser query: " + effective_request.message
                    return gemini_client.models.generate_content_stream(
                        model=model_name,
                        contents=full_prompt,
                    )

                loop = asyncio.get_event_loop()
                stream = await loop.run_in_executor(_executor, _gemini_call)

                yield _status(f"Routing to {model_name}", "completed")

                for chunk in stream:
                    text = chunk.text if hasattr(chunk, "text") and chunk.text else ""
                    if text:
                        data = json.dumps({"type": "token", "content": text})
                        yield f"data: {data}\n\n"

                yield "data: [DONE]\n\n"
            except Exception as e:
                err = json.dumps({"type": "error", "content": f"Gemini error: {e}"})
                yield f"data: {err}\n\n"
                yield "data: [DONE]\n\n"
            return

        # ── Claude (Anthropic) model routing ──────────────────────────────────
        if model_name.startswith("claude-"):
            try:
                import anthropic
                claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

                yield _status(f"Routing to {model_name}", "running")

                def _claude_call():
                    return claude_client.messages.stream(
                        model=model_name,
                        max_tokens=int(os.getenv("MAX_TOKENS", "8096")),
                        system=QUASAR_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": effective_request.message}],
                    )

                loop = asyncio.get_event_loop()
                yield _status(f"Routing to {model_name}", "completed")

                def _run_claude_stream(q):
                    with claude_client.messages.stream(
                        model=model_name,
                        max_tokens=int(os.getenv("MAX_TOKENS", "8096")),
                        system=QUASAR_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": effective_request.message}],
                    ) as stream:
                        for text in stream.text_stream:
                            asyncio.run_coroutine_threadsafe(q.put(("token", text)), loop)
                    asyncio.run_coroutine_threadsafe(q.put(("done", "")), loop)

                claude_queue = asyncio.Queue()
                loop.run_in_executor(_executor, _run_claude_stream, claude_queue)

                while True:
                    msg = await claude_queue.get()
                    if msg[0] == "done":
                        break
                    elif msg[0] == "token" and msg[1]:
                        data = json.dumps({"type": "token", "content": msg[1]})
                        yield f"data: {data}\n\n"

                yield "data: [DONE]\n\n"
            except Exception as e:
                err = json.dumps({"type": "error", "content": f"Claude error: {e}"})
                yield f"data: {err}\n\n"
                yield "data: [DONE]\n\n"
            return

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

            # ── Route based on DATA SIZE, not query complexity ──────────────────────
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
            yield _status("Routing to standard agent")
            await asyncio.sleep(0.05)
            yield _status("Routing to standard agent", "completed")

            # Prefer the model name requested by the user, fallback to config
            model_name = request.model or getattr(agent.config, 'model', 'GPT-4o') or 'GPT-4o'
            yield _status(f"Calling {model_name}")

            queue = asyncio.Queue()
            
            def on_token(token: str):
                if token:
                    asyncio.run_coroutine_threadsafe(queue.put(("token", token)), loop)
                    
            def _run_agent():
                try:
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

            yield _status(f"Calling {model_name}", "completed")

            first_token = True
            response_text = ""
            
            while True:
                msg = await queue.get()
                if isinstance(msg, tuple) and len(msg) == 3:
                    msg_type, step, state = msg
                    if msg_type == "status":
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
                    if first_token:
                        yield _status("Generating response", "completed")
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
):
    """Stream a chat response with attached files (images/documents) via SSE."""
    import base64, io

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
    return await chat(req)


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
