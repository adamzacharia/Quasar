"""
QUASAR Professional Chat Interface — FastAPI Backend

Provides RESTful + SSE endpoints wrapping the existing QuasarAgent.
Run with:  uvicorn api.main:app --reload --port 8000

This module is intentionally thin after the P1 monolith split: it owns process
setup (app instance, middleware, CORS, static mounts, startup/shutdown, the
catch-all exception handler) and mounts the domain routers from ``api.routers``.
The route handlers themselves live in ``api/routers/*``; the SSE chat pipeline in
``api/sse.py``; the UI serialization (column maps, CADC XML, matplotlib FITS) in
``api/serializers/*``; shared singletons/helpers in ``api/deps.py``.
"""

# ── Make the ``api`` package importable before importing from it ──────────────
# ``bootstrap`` runs the Windows/py3.13 shims and puts the repo root + ui-pro on
# sys.path. It must be importable first, so ensure ui-pro is on the path inline
# (works under uvicorn AND when a test loads this file via importlib).
import sys
from pathlib import Path as _PathForBootstrap

_UI_PRO_DIR = str(_PathForBootstrap(__file__).resolve().parent.parent)
if _UI_PRO_DIR not in sys.path:
    sys.path.append(_UI_PRO_DIR)

from api import bootstrap  # noqa: E402,F401  (shims + sys.path + dotenv, run first)

import os  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

# ── Observability: Loguru + Sentry ────────────────────────────────────────────
from core.logger import logger, init_rollbar  # noqa: E402
from services.secret_redaction import redact_secrets  # noqa: E402

init_rollbar()  # no-op if ROLLBAR_ACCESS_TOKEN env var is not set
logger.info("[QUASAR API] Starting up")

# ── Backward-compatible re-exports ────────────────────────────────────────────
# Historically these lived in ``main.py``; a few tests load this module by path
# (importlib) and call ``main._build_data_card_event(...)``. Keep the names
# resolvable from ``api.main`` after the split.
from api.serializers.data_card import (  # noqa: E402,F401
    _build_data_card_event,
    _compute_demographics,
    _default_archive_source,
    _normalize_qa2_value,
    _qa2_status_from_table,
)
from api.serializers.cadc import _fetch_cadc_preview_urls  # noqa: E402,F401

# ── Shared singletons used directly by app-level hooks ────────────────────────
from api.deps import spectral_line_job_service  # noqa: E402

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


@app.on_event("shutdown")
def _shutdown_spectral_line_jobs():
    spectral_line_job_service.shutdown()


@app.on_event("shutdown")
def _shutdown_mcp_bridges():
    # Release MCP bridge threads (they otherwise park forever on their
    # keep-alive event and hold child stdio processes open).
    from . import deps

    if deps._agent is not None:
        try:
            deps._agent.shutdown_mcp_servers()
        except Exception as e:
            print(f"[MCPServers] Shutdown signalling failed: {e}")


# ── Request / response logging middleware ─────────────────────────────────────
import re  # noqa: E402
import time as _time  # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse as _JSONResponse  # noqa: E402

from services.auth_cookie import AUTH_COOKIE_NAME  # noqa: E402


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

# ── CORS — added LAST so it is the OUTERMOST middleware (Starlette LIFO). ──────
# This guarantees every response (including errors) carries CORS headers.
#
# S4 fix: replace the open ``*.vercel.app`` / ``*.onrender.com`` regex (which,
# combined with allow_credentials=True, let ANY attacker-registered subdomain on
# those hosts make credentialed cross-origin requests) with an EXPLICIT origin
# allowlist. Production origins are fixed; localhost dev is matched by a tight
# regex (localhost/127.0.0.1:300x is not attacker-registerable). Operators can
# add specific deploy origins (e.g. a Vercel preview or a raw onrender URL) via
# the comma-separated ``QUASAR_CORS_ORIGINS`` env var — never a wildcard.
_DEFAULT_CORS_ORIGINS = [
    "https://quasarassistant.com",
    "https://www.quasarassistant.com",
]
_cors_env = os.getenv("QUASAR_CORS_ORIGINS", "").strip()
_extra_cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
CORS_ALLOWED_ORIGINS = list(dict.fromkeys(_DEFAULT_CORS_ORIGINS + _extra_cors_origins))
CORS_ALLOWED_ORIGIN_REGEX = r"http://localhost:300[0-9]|http://127\.0\.0\.1:300[0-9]"


# ── CSRF: Origin check for cookie-authenticated writes (S6) ───────────────────
# Moving the JWT into an httpOnly cookie (S6) removed the XSS-exfiltration risk
# but ADDED a CSRF surface that Bearer auth never had: the browser attaches the
# cookie to cross-site requests on its own. CORS does not help — it governs
# reading the *response*, not sending the *request* — and several write
# endpoints are "simple requests" that skip the preflight entirely:
#   * multipart/form-data uploads: /api/chat/upload, /api/personalization/upload,
#     /api/proposals/review  (Form(...)/UploadFile)
#   * /api/feedback, which parses the body with `await req.json()` and therefore
#     accepts a preflight-free `text/plain` Content-Type.
# In production the cookie is SameSite=None (frontend and API are cross-site),
# so SameSite alone does not close this.
#
# Rule: for unsafe methods, if the request carries the auth cookie AND declares
# an Origin that is not allow-listed, reject it. Requests with no Origin are
# left alone — every browser sends Origin on unsafe methods, so an absent Origin
# means a non-browser client (curl, server-to-server), which cannot be a CSRF
# victim. Bearer-only clients are untouched.
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})


def _is_allowed_origin(origin: str) -> bool:
    if origin in CORS_ALLOWED_ORIGINS:
        return True
    return re.fullmatch(CORS_ALLOWED_ORIGIN_REGEX, origin) is not None


class CookieCsrfMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method not in _SAFE_METHODS and AUTH_COOKIE_NAME in request.cookies:
            origin = request.headers.get("origin")
            if origin and not _is_allowed_origin(origin):
                logger.warning(
                    f"[CSRF] blocked cookie-authenticated {request.method} "
                    f"{request.url.path} from origin {origin}"
                )
                return _JSONResponse(
                    {"detail": "Cross-site request blocked."}, status_code=403
                )
        return await call_next(request)


# Added BEFORE CORSMiddleware so CORS stays outermost (Starlette LIFO) and the
# 403 still carries CORS headers.
app.add_middleware(CookieCsrfMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_origin_regex=CORS_ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # None of these are CORS-safelisted response headers, so a cross-origin
    # browser cannot read them unless explicitly exposed. Retry-After: the
    # login-throttle 429 (S6). The X-Quasar-* trio: the export route's
    # truncation verdict — without them a cross-site browser reads null and
    # DataTableCard would treat a ceiling-clipped CSV as full, the exact
    # silent-partial Feature 2 forbids (f2-CX-20). Content-Disposition: the
    # export filename the client parses from the same response.
    expose_headers=[
        "Retry-After",
        "X-Quasar-Truncated",
        "X-Quasar-Rowcount",
        "X-Quasar-Total-Rows",
        "Content-Disposition",
    ],
)

# ── Static file serving for rendered FITS images ─────────────────
from fastapi.staticfiles import StaticFiles  # noqa: E402

_RENDERED_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "rendered_images")
os.makedirs(_RENDERED_IMAGES_DIR, exist_ok=True)
app.mount("/api/images", StaticFiles(directory=_RENDERED_IMAGES_DIR), name="rendered_images")

# ── Static file serving for matplotlib-rendered plots ─────────────
# PlottingService writes PNGs here and returns web_url "/plots/<file>.png" (CMDs, sky maps,
# Data Lab analysis plots). Mount it so those URLs resolve — previously only /api/images was
# served, so every /plots image 404'd and never displayed in chat.
try:
    from services.plotting import PLOT_OUTPUT_DIR as _PLOT_OUTPUT_DIR
    os.makedirs(_PLOT_OUTPUT_DIR, exist_ok=True)
    app.mount("/plots", StaticFiles(directory=_PLOT_OUTPUT_DIR), name="plots")
except Exception as _plots_mount_err:  # pragma: no cover - best-effort mount
    print(f"[startup] Could not mount /plots static dir: {_plots_mount_err}")


# ── Catch-all exception handler — ensures a proper JSON 500 with CORS headers ──
from starlette.responses import JSONResponse  # noqa: E402


@app.exception_handler(Exception)
async def _global_exception_handler(request, exc):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {redact_secrets(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": redact_secrets(exc)},
    )


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


# ── Domain routers (extracted from the former monolithic main.py) ─────────────
from api.routers import (  # noqa: E402
    admin,
    analytics,
    auth,
    chat,
    conductor,
    conversations,
    datalab,
    fits,
    general,
    health,
    issue_reports,
    personalization,
    proposals,
    provider_keys,
    results,
    spectral_lines,
    tools,
    usage,
    workbench,
)

app.include_router(general.router)
app.include_router(provider_keys.router)
app.include_router(usage.router)
app.include_router(tools.router)
app.include_router(auth.router)
app.include_router(fits.router)
app.include_router(spectral_lines.router)
app.include_router(datalab.router)
app.include_router(results.router)
app.include_router(workbench.router)
app.include_router(conversations.router)
app.include_router(chat.router)
app.include_router(personalization.router)
app.include_router(proposals.router)
app.include_router(analytics.router)
app.include_router(issue_reports.router)
app.include_router(admin.router)
app.include_router(conductor.router)
app.include_router(health.router)
