"""insufficient_quota must never be retried.

OpenAI reports an exhausted billing quota as HTTP 429 — the same status as a
transient rate limit — but it is a PERMANENT state. The retry engine treating
it as retryable stalled every web-search turn ~80 s live (2026-07-18: the
post-answer summary synthesis retried a quota-dead key through the full
backoff ladder).
"""

import time

import pytest

from core.retry import _is_retryable, with_retry


class QuotaError(Exception):
    """Mimics openai.RateLimitError for a billing-exhausted account."""

    status_code = 429
    code = "insufficient_quota"

    def __init__(self):
        super().__init__(
            "Error code: 429 - {'error': {'message': 'You exceeded your current "
            "quota', 'type': 'insufficient_quota', 'code': 'insufficient_quota'}}"
        )


class PlainRateLimitError(Exception):
    """A genuine transient 429 — still retryable."""

    status_code = 429

    def __init__(self):
        super().__init__("Error code: 429 - rate limit, slow down")


def test_insufficient_quota_is_not_retryable():
    assert _is_retryable(QuotaError()) is False


def test_insufficient_quota_detected_from_message_alone():
    err = Exception("429 ... 'type': 'insufficient_quota' ...")
    assert _is_retryable(err) is False


def test_plain_429_stays_retryable():
    assert _is_retryable(PlainRateLimitError()) is True


def test_with_retry_fails_fast_on_quota_error():
    calls = {"n": 0}

    @with_retry(max_retries=3, backoff_base=1.0)
    def call():
        calls["n"] += 1
        raise QuotaError()

    start = time.monotonic()
    with pytest.raises(QuotaError):
        call()
    elapsed = time.monotonic() - start

    assert calls["n"] == 1, "a billing-dead key must not be re-attempted"
    assert elapsed < 0.5, "no backoff sleeps for a permanent error"
