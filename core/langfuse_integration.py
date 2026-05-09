# core/langfuse_integration.py
"""
Langfuse Integration — Full agent tracing & cost analytics.

Provides a thin, safe wrapper around the Langfuse Python SDK.
Everything is a graceful no-op when LANGFUSE_SECRET_KEY is not set,
so this module can be imported unconditionally.

Usage (generation logging — wraps any LLM call):
    from core.langfuse_integration import get_langfuse, langfuse_generation

    lf = get_langfuse()
    if lf:
        trace = lf.trace(name="my_query", user_id="u1")
        gen = langfuse_generation(trace, model="gpt-4o", input="Hello")
        # ... make LLM call ...
        gen.end(output="Hi!", usage={"input": 100, "output": 50})

Usage (Conductor orchestration trace):
    from core.langfuse_integration import langfuse_trace

    with langfuse_trace("Search M87", user_id="u1") as trace:
        span = trace.span(name="t1_archive")
        # ... execute task ...
        span.end(output={"results": 42})

Signup: https://cloud.langfuse.com (free tier: 50k observations/month)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Module-level singleton — initialized lazily
_langfuse_client = None
_langfuse_init_attempted = False


def get_langfuse():
    """
    Get the Langfuse client singleton.

    Returns the client if LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY are set,
    otherwise returns None (all downstream code must handle None gracefully).
    """
    global _langfuse_client, _langfuse_init_attempted

    if _langfuse_init_attempted:
        return _langfuse_client

    _langfuse_init_attempted = True

    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")

    if not secret_key or not public_key:
        logger.info(
            "[Langfuse] LANGFUSE_SECRET_KEY/PUBLIC_KEY not set — tracing disabled. "
            "Sign up free at https://cloud.langfuse.com"
        )
        return None

    try:
        from langfuse import Langfuse

        host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"
        _langfuse_client = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=host,
        )
        logger.info("[Langfuse] Initialised (host=%s)", host)
        return _langfuse_client

    except ImportError:
        logger.warning(
            "[Langfuse] 'langfuse' package not installed. "
            "Run: pip install langfuse"
        )
        return None
    except Exception as e:
        logger.warning("[Langfuse] Initialisation failed: %s", e)
        return None


def flush_langfuse():
    """Flush pending Langfuse events (call on shutdown)."""
    client = get_langfuse()
    if client:
        try:
            client.flush()
        except Exception:
            pass


# ── Trace context manager (for Conductor orchestration) ──────────────────────

@contextmanager
def langfuse_trace(
    name: str,
    user_id: str = "anonymous",
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
):
    """
    Context manager that creates a Langfuse trace for the duration of a block.

    Yields the trace object (or a no-op stub if Langfuse is disabled).
    Automatically ends the trace and sets status on exit.

    Usage:
        with langfuse_trace("Search M87", user_id="u1") as trace:
            span = trace.span(name="decompose")
            ...
    """
    client = get_langfuse()
    if client is None:
        yield _NoOpTrace()
        return

    try:
        trace = client.trace(
            name=name,
            user_id=user_id,
            metadata=metadata or {},
            tags=tags or ["quasar"],
        )
        yield trace
        trace.update(metadata={**(metadata or {}), "status": "completed"})
    except Exception as e:
        logger.warning("[Langfuse] Trace error: %s", e)
        yield _NoOpTrace()


def langfuse_generation(
    parent,  # trace or span
    name: str = "llm_call",
    model: str = "",
    input_data: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Create a Langfuse generation (LLM call) on a parent trace/span.

    Returns a generation object with .end() method, or a no-op stub.
    The caller is responsible for calling .end(output=..., usage=...).
    """
    if parent is None or isinstance(parent, _NoOpTrace):
        return _NoOpGeneration()

    try:
        gen = parent.generation(
            name=name,
            model=model,
            input=_safe_serialize(input_data),
            metadata=metadata or {},
        )
        return gen
    except Exception as e:
        logger.debug("[Langfuse] Generation creation failed: %s", e)
        return _NoOpGeneration()


# ── No-op stubs (used when Langfuse is disabled) ─────────────────────────────

class _NoOpTrace:
    """Stub that silently ignores all calls when Langfuse is disabled."""

    def span(self, **kwargs):
        return _NoOpSpan()

    def generation(self, **kwargs):
        return _NoOpGeneration()

    def update(self, **kwargs):
        pass

    def event(self, **kwargs):
        pass


class _NoOpSpan:
    """Stub span that silently ignores all calls."""

    def span(self, **kwargs):
        return _NoOpSpan()

    def generation(self, **kwargs):
        return _NoOpGeneration()

    def end(self, **kwargs):
        pass

    def update(self, **kwargs):
        pass

    def event(self, **kwargs):
        pass


class _NoOpGeneration:
    """Stub generation that silently ignores all calls."""

    def end(self, **kwargs):
        pass

    def update(self, **kwargs):
        pass


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_serialize(data: Any, max_len: int = 5000) -> Any:
    """Safely serialize data for Langfuse, truncating large payloads."""
    if data is None:
        return None
    if isinstance(data, str):
        return data[:max_len]
    if isinstance(data, (int, float, bool)):
        return data
    if isinstance(data, dict):
        return {k: _safe_serialize(v, max_len=1000) for k, v in list(data.items())[:50]}
    if isinstance(data, list):
        return [_safe_serialize(item, max_len=1000) for item in data[:20]]
    return str(data)[:max_len]
