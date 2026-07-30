"""Mid-stream connection deaths must be retryable.

openai does NOT wrap iteration errors, so a TCP reset during the body read
(httpx.ReadError) or a peer closing mid-body (httpx.RemoteProtocolError)
surfaces raw from the stream. The name-substring classifier missed both —
'readerror' and 'remoteprotocol' matched no pattern — so the round-level
retry in core/runner.py never fired for exactly the failures it was built for.
"""

import httpx

from core.retry import _is_retryable


def test_httpx_read_error_is_retryable():
    assert _is_retryable(httpx.ReadError("")) is True


def test_httpx_remote_protocol_error_is_retryable():
    assert (
        _is_retryable(
            httpx.RemoteProtocolError(
                "peer closed connection without sending complete message body"
            )
        )
        is True
    )


def test_httpx_read_timeout_with_empty_message_is_retryable():
    assert _is_retryable(httpx.ReadTimeout("")) is True


def test_httpx_connect_error_is_retryable():
    assert _is_retryable(httpx.ConnectError("connection refused")) is True


def test_4xx_stays_non_retryable():
    class BadRequestError(Exception):
        status_code = 400

    assert _is_retryable(BadRequestError("bad request")) is False


def test_value_error_stays_non_retryable():
    assert _is_retryable(ValueError("nope")) is False
