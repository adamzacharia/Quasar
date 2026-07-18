# core/retry.py
"""
Retry Engine — Exponential backoff with jitter for LLM API calls.

Inspired by OpenClaude's withRetry.ts — wraps any API call with:
  - Exponential backoff (2^attempt * base_delay + random jitter)
  - Rate-limit detection (429, 529)
  - Server error retry (500, 502, 503)
  - Non-retryable error pass-through (401, 403, 404)
  - Maximum retry count with configurable limit

Usage:
    @with_retry(max_retries=3, backoff_base=1.0)
    def call_api():
        return client.responses.create(...)
"""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Any, Callable, Optional, Set, Tuple, Type

logger = logging.getLogger(__name__)

# HTTP status codes that should be retried
RETRYABLE_STATUS_CODES: Set[int] = {429, 500, 502, 503, 529}

# HTTP status codes that should NEVER be retried (auth, not-found, etc.)
NON_RETRYABLE_STATUS_CODES: Set[int] = {401, 403, 404, 422}

# Exception types that should never be retried
NON_RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ValueError,
    TypeError,
    KeyError,
)


def _extract_status_code(error: Exception) -> Optional[int]:
    """Extract HTTP status code from various API error types."""
    # OpenAI errors
    if hasattr(error, 'status_code'):
        return getattr(error, 'status_code')
    # Anthropic errors
    if hasattr(error, 'status'):
        return getattr(error, 'status')
    # httpx / requests errors
    if hasattr(error, 'response') and hasattr(error.response, 'status_code'):
        return error.response.status_code
    return None


def _extract_retry_after(error: Exception) -> Optional[float]:
    """Extract Retry-After header value from API errors (seconds)."""
    if hasattr(error, 'response') and hasattr(error.response, 'headers'):
        retry_after = error.response.headers.get('retry-after')
        if retry_after:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                pass
    return None


def _is_retryable(error: Exception) -> bool:
    """Determine whether an error is retryable."""
    # Never retry these exception types
    if isinstance(error, NON_RETRYABLE_EXCEPTIONS):
        return False

    # OpenAI's "insufficient_quota" rides a 429 but is a BILLING state, not a
    # transient rate limit — retrying burns ~a minute of backoff per call and
    # stalled every web-search turn for 80 s live (2026-07-18).
    if (
        getattr(error, "code", None) == "insufficient_quota"
        or "insufficient_quota" in str(error)
    ):
        return False

    status_code = _extract_status_code(error)
    if status_code is not None:
        if status_code in NON_RETRYABLE_STATUS_CODES:
            return False
        if status_code in RETRYABLE_STATUS_CODES:
            return True
        # Unknown status — don't retry 4xx, do retry 5xx
        if 400 <= status_code < 500:
            return False
        if status_code >= 500:
            return True

    # Connection errors, timeouts → retryable
    error_name = type(error).__name__.lower()
    retryable_patterns = ['timeout', 'connection', 'network', 'reset', 'eof']
    return any(p in error_name for p in retryable_patterns)


def _compute_delay(attempt: int, backoff_base: float, max_delay: float) -> float:
    """Compute delay with exponential backoff + jitter."""
    delay = min(backoff_base * (2 ** attempt), max_delay)
    # Add jitter: ±25% randomization
    jitter = delay * 0.25 * (2 * random.random() - 1)
    return max(0.1, delay + jitter)


def with_retry(
    max_retries: int = 3,
    backoff_base: float = 1.0,
    max_delay: float = 30.0,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
):
    """
    Decorator that wraps a function with retry logic.

    Parameters
    ----------
    max_retries : int
        Maximum number of retry attempts (default: 3).
    backoff_base : float
        Base delay in seconds for exponential backoff (default: 1.0).
    max_delay : float
        Maximum delay between retries in seconds (default: 30.0).
    on_retry : callable, optional
        Called before each retry with (attempt, error, delay).

    Example
    -------
    @with_retry(max_retries=3, backoff_base=1.0)
    def call_openai():
        return client.responses.create(...)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    # Don't retry on final attempt
                    if attempt >= max_retries:
                        break

                    # Don't retry non-retryable errors
                    if not _is_retryable(e):
                        logger.warning(
                            "[retry] Non-retryable error on %s: %s (%s)",
                            func.__name__, type(e).__name__, e
                        )
                        break

                    # Compute delay
                    retry_after = _extract_retry_after(e)
                    delay = retry_after if retry_after else _compute_delay(
                        attempt, backoff_base, max_delay
                    )

                    status = _extract_status_code(e) or "?"
                    logger.info(
                        "[retry] %s attempt %d/%d failed (status=%s): %s. "
                        "Retrying in %.1fs...",
                        func.__name__, attempt + 1, max_retries,
                        status, type(e).__name__, delay,
                    )

                    if on_retry:
                        on_retry(attempt, e, delay)

                    time.sleep(delay)

            # All retries exhausted — raise the last error
            raise last_error

        return wrapper
    return decorator


def with_retry_async(
    max_retries: int = 3,
    backoff_base: float = 1.0,
    max_delay: float = 30.0,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
):
    """Async version of with_retry for async API calls."""
    import asyncio

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    if attempt >= max_retries:
                        break

                    if not _is_retryable(e):
                        logger.warning(
                            "[retry] Non-retryable error on %s: %s (%s)",
                            func.__name__, type(e).__name__, e
                        )
                        break

                    retry_after = _extract_retry_after(e)
                    delay = retry_after if retry_after else _compute_delay(
                        attempt, backoff_base, max_delay
                    )

                    status = _extract_status_code(e) or "?"
                    logger.info(
                        "[retry] %s attempt %d/%d failed (status=%s): %s. "
                        "Retrying in %.1fs...",
                        func.__name__, attempt + 1, max_retries,
                        status, type(e).__name__, delay,
                    )

                    if on_retry:
                        on_retry(attempt, e, delay)

                    await asyncio.sleep(delay)

            raise last_error

        return wrapper
    return decorator
