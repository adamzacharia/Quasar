"""
core/logger.py — Quasar Observability Layer

Provides:
  - Loguru-based structured logging (file + console)
  - @log_tool decorator: auto-logs every agent tool call (entry, exit, timing, errors)
  - Rollbar integration (graceful no-op if ROLLBAR_ACCESS_TOKEN is not set)

Usage:
    from core.logger import logger, log_tool

    @log_tool
    def my_tool_function(self, query: str) -> dict:
        ...
"""

import os
import sys
import time
import functools
from pathlib import Path
from loguru import logger as _loguru_logger

# ── Log directory ──────────────────────────────────────────────────────────────
_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

# ── Remove default loguru sink so we control formatting ───────────────────────
_loguru_logger.remove()

# ── Console sink (colored, human-readable) ────────────────────────────────────
_loguru_logger.add(
    sys.stderr,
    level="DEBUG",
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    ),
    colorize=True,
)

# ── File sink (structured, rotating, 7-day retention) ─────────────────────────
_loguru_logger.add(
    _LOG_DIR / "quasar_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} — {message}",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    enqueue=True,   # thread-safe async writes
)

# ── Error-only sink (separate file for quick triage) ─────────────────────────
_loguru_logger.add(
    _LOG_DIR / "quasar_errors.log",
    level="ERROR",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} — {message}\n{exception}",
    rotation="5 MB",
    retention="30 days",
    compression="zip",
    enqueue=True,
)

# Public logger instance
logger = _loguru_logger


# ── Rollbar integration (graceful no-op if token not set) ────────────────────
def init_rollbar():
    """
    Initialise Rollbar if ROLLBAR_ACCESS_TOKEN is set.
    Safe to call multiple times — subsequent calls are no-ops.
    Signup: https://rollbar.com (free tier: 5,000 errors/month)
    """
    token = os.getenv("ROLLBAR_ACCESS_TOKEN", "")
    if not token:
        logger.info("[Rollbar] ROLLBAR_ACCESS_TOKEN not set — Rollbar disabled")
        return
    try:
        import rollbar
        rollbar.init(
            access_token=token,
            environment=os.getenv("ENVIRONMENT", "production"),
            code_version="3.3.0",
            suppress_reinit_warning=True,
        )
        logger.success(f"[Rollbar] Initialised (env={os.getenv('ENVIRONMENT','production')})")
    except Exception as e:
        logger.warning(f"[Rollbar] Initialisation failed: {e}")


# ── @log_tool decorator ────────────────────────────────────────────────────────
def log_tool(fn):
    """
    Decorator that wraps any agent tool method with:
      - Entry log  (DEBUG): function name + kwargs
      - Exit log   (SUCCESS): execution time
      - Error log  (ERROR): full exception with Rollbar capture
      - Langfuse Span logging: creates a nested span under the active parent trace.

    Usage:
        @log_tool
        def _my_tool(self, param: str) -> dict:
            ...
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Build a readable kwarg summary (skip 'self')
        kw_str = ", ".join(f"{k}={repr(v)[:60]}" for k, v in kwargs.items())
        logger.debug(f"[TOOL →] {fn.__name__}({kw_str})")
        t0 = time.perf_counter()

        # ── Langfuse: create a child span for the tool execution ──
        from core.llm_client import get_langfuse_parent
        parent = get_langfuse_parent()
        lf_span = None
        if parent:
            try:
                # Clean up tool name for display (e.g. _search_by_target -> search_by_target)
                display_name = fn.__name__.lstrip('_')
                lf_span = parent.span(
                    name=f"tool: {display_name}",
                    input={k: str(v)[:500] for k, v in kwargs.items()},
                    metadata={"tool_function": fn.__name__}
                )
            except Exception:
                pass

        try:
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            # Log success/failure based on result dict
            if isinstance(result, dict) and not result.get("success", True):
                logger.warning(
                    f"[TOOL ✗] {fn.__name__} returned error in {elapsed:.2f}s "
                    f"— {result.get('error', '?')}"
                )
                if lf_span:
                    try:
                        lf_span.end(output={"success": False, "error": result.get('error')})
                    except Exception:
                        pass
            else:
                logger.success(f"[TOOL ✓] {fn.__name__} completed in {elapsed:.2f}s")
                if lf_span:
                    try:
                        # Safely serialize output snippet to avoid giant payloads
                        from core.langfuse_integration import _safe_serialize
                        lf_span.end(output=_safe_serialize(result, max_len=2000))
                    except Exception:
                        pass
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.exception(f"[TOOL ✗] {fn.__name__} raised after {elapsed:.2f}s: {exc}")
            
            if lf_span:
                try:
                    lf_span.end(output={"success": False, "error": str(exc)})
                except Exception:
                    pass

            # Report to Rollbar with tool name as extra context
            try:
                import rollbar
                rollbar.report_exc_info(
                    extra_data={"tool": fn.__name__, "kwargs": kw_str[:200]}
                )
            except Exception:
                pass
            raise
    return wrapper
