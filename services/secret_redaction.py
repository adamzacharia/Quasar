"""Utilities for removing API-key shaped values from user-visible text and logs."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_\-]{20,}\b"),
    # The canonical HTTP header form has NO ':'/'=' between the scheme and the
    # token ("Authorization: Bearer eyJ...") — the key[:=]value pattern below
    # missed it entirely, so live JWTs survived into user-visible errors and
    # admin exports (scan SR-01). Value class includes +/=/. so standard-base64
    # and JWT segments redact fully (SR-02).
    re.compile(r"(?i)\bbearer\s+([A-Za-z0-9_\-\.+/=]{12,})"),
    re.compile(r"(?i)(api[_-]?key|authorization|x-api-key|bearer)\s*[:=]\s*['\"]?([A-Za-z0-9_\-\.+/=]{12,})"),
]


def redact_secrets(value: Any) -> str:
    """Return a string with likely provider keys and auth headers redacted."""
    text = str(value or "")
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            text = pattern.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    return text


# Query-string parameters that carry credentials. Data Lab passes `TOKEN=`, which
# is shorter than the generic key patterns above and survives them — a raw
# endpoint URL is only safe to show a user after this pass.
_SECRET_QUERY_PARAMS = {
    "token", "api_key", "apikey", "access_token", "refresh_token", "auth",
    "authorization", "password", "passwd", "secret", "client_secret",
    "session", "session_id", "sessionid", "sid", "signature", "sig",
}


def redact_url(value: Any) -> str:
    """Redact credential-bearing query parameters from a URL, then its text.

    Preserves the URL's structure (and the parameter NAME) so the result stays
    a recognizable, reproducible request:
    ``https://x/q?TOKEN=abc&ra=1`` → ``https://x/q?TOKEN=[REDACTED]&ra=1``.
    A value that does not parse as a URL is returned text-redacted only.
    """
    raw = str(value or "")
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
        if not parts.query:
            return redact_secrets(raw)
        pairs = parse_qsl(parts.query, keep_blank_values=True)
        cleaned = [
            (k, "[REDACTED]" if k.strip().lower() in _SECRET_QUERY_PARAMS else v)
            for k, v in pairs
        ]
        # safe="[]" keeps the marker readable as [REDACTED] rather than
        # percent-encoding it to %5BREDACTED%5D.
        rebuilt = urlunsplit(
            (parts.scheme, parts.netloc, parts.path,
             urlencode(cleaned, safe="[]"), parts.fragment)
        )
        return redact_secrets(rebuilt)
    except Exception:
        return redact_secrets(raw)
