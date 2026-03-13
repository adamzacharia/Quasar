"""
core/logger.py — Quasar Observability Layer

Provides:
  - Loguru-based structured logging (file + console)
  - @log_tool decorator: auto-logs every agent tool call (entry, exit, timing, errors)
  - Sentry integration (graceful no-op if SENTRY_DSN is not set)

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


# ── Sentry integration (graceful no-op if DSN not set) ────────────────────────
def init_sentry():
    """
    Initialise Sentry SDK if SENTRY_DSN is set.
    Safe to call multiple times — subsequent calls are no-ops.
    """
    dsn = os.getenv("SENTRY_DSN", "")
    if not dsn:
        logger.info("[Sentry] SENTRY_DSN not set — Sentry disabled")
        return
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_RATE", "0.1")),
            profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_RATE", "0.1")),
            environment=os.getenv("ENVIRONMENT", "production"),
            integrations=[
                StarletteIntegration(transaction_style="endpoint"),
                FastApiIntegration(transaction_style="endpoint"),
            ],
            send_default_pii=False,
        )
        logger.success(f"[Sentry] Initialised (env={os.getenv('ENVIRONMENT','production')})")
    except Exception as e:
        logger.warning(f"[Sentry] Initialisation failed: {e}")


# ── @log_tool decorator ────────────────────────────────────────────────────────
def log_tool(fn):
    """
    Decorator that wraps any agent tool method with:
      - Entry log  (DEBUG): function name + kwargs
      - Exit log   (SUCCESS): execution time
      - Error log  (ERROR): full exception with Sentry capture
    
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
        try:
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            # Log success/failure based on result dict
            if isinstance(result, dict) and not result.get("success", True):
                logger.warning(
                    f"[TOOL ✗] {fn.__name__} returned error in {elapsed:.2f}s "
                    f"— {result.get('error', '?')}"
                )
            else:
                logger.success(f"[TOOL ✓] {fn.__name__} completed in {elapsed:.2f}s")
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.exception(f"[TOOL ✗] {fn.__name__} raised after {elapsed:.2f}s: {exc}")
            # Report to Sentry if available
            try:
                import sentry_sdk
                sentry_sdk.capture_exception(exc)
            except Exception:
                pass
            raise
    return wrapper
