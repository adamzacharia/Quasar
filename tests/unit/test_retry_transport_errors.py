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


# ── in-band stream errors (openai's generic APIError, no HTTP status) ────────
#
# A litellm/vLLM gateway that fails AFTER the SSE stream is already 200
# delivers its failure as an in-band {"error": ...} event, which the openai SDK
# raises as the bare APIError with status_code=None. Its type name matches none
# of the name patterns, so the stream-round retry in core/runner.py never fired
# (live 2026-09-17: 10 of 27 benchmark trials lost to
# "litellm.MidStreamFallbackError: ... An error occurred during streaming").

def _stream_api_error(message, body=None):
    import openai

    return openai.APIError(
        message=message,
        request=httpx.Request("POST", "http://tacc.test/v1/chat/completions"),
        body=body if body is not None else {"message": message, "type": "internal_server_error"},
    )


def test_generic_api_error_from_a_dead_stream_is_retryable():
    err = _stream_api_error(
        "litellm.MidStreamFallbackError: litellm.APIConnectionError: APIConnectionError: "
        "OpenAIException - An error occurred during streaming. Received Model Group=gpt-oss-120b"
    )
    assert getattr(err, "status_code", None) is None
    assert _is_retryable(err) is True


def test_statusless_api_error_without_a_known_signature_is_still_retryable():
    # The SDK raises the bare APIError ONLY for an in-band error event on an
    # open stream; rejected requests are APIStatusError subclasses.
    assert _is_retryable(_stream_api_error("something odd happened")) is True


def test_request_fault_reported_mid_stream_is_not_retryable():
    err = _stream_api_error(
        "This model's maximum context length is 131072 tokens",
        body={"message": "maximum context length exceeded", "code": "context_length_exceeded"},
    )
    assert _is_retryable(err) is False


def test_insufficient_quota_reported_mid_stream_stays_non_retryable():
    err = _stream_api_error(
        "You exceeded your current quota",
        body={"message": "You exceeded your current quota", "code": "insufficient_quota"},
    )
    assert _is_retryable(err) is False


def test_transport_signature_in_a_plain_exception_message_is_retryable():
    class ProxyError(Exception):
        pass

    assert _is_retryable(ProxyError("upstream connection reset by peer")) is True
    assert _is_retryable(ProxyError("An error occurred during streaming")) is True


def test_4xx_with_a_transport_looking_message_stays_non_retryable():
    class BadRequestError(Exception):
        status_code = 400

    assert _is_retryable(BadRequestError("connection reset (but the request was rejected)")) is False


def test_status_bearing_openai_errors_keep_their_status_classification():
    import openai

    req = httpx.Request("POST", "http://tacc.test/v1/chat/completions")
    bad = openai.BadRequestError("bad", response=httpx.Response(400, request=req), body=None)
    assert _is_retryable(bad) is False
    ise = openai.InternalServerError("ise", response=httpx.Response(500, request=req), body=None)
    assert _is_retryable(ise) is True
