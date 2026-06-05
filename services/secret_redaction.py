"""Utilities for removing API-key shaped values from user-visible text and logs."""

from __future__ import annotations

import re
from typing import Any


_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"(?i)(api[_-]?key|authorization|x-api-key|bearer)\s*[:=]\s*['\"]?([A-Za-z0-9_\-\.]{12,})"),
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
