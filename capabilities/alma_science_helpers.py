"""Small network helpers shared by the one-shot ALMA tools (leaf module)."""
from __future__ import annotations

from typing import Tuple


def head_verify(url: str, timeout: float = 6.0) -> Tuple[bool, str]:
    """Bounded liveness probe: 2xx/3xx (or 405/403 on HEAD followed by a GET
    that answers 2xx/3xx) counts as verified."""
    if not url:
        return False, "no url"
    try:
        import requests

        from services.http_budget_hook import suppressed

        with suppressed():
            resp = requests.head(url, timeout=timeout, allow_redirects=True)
            status = int(resp.status_code)
            if status in (405, 403, 501):
                resp = requests.get(url, timeout=timeout, allow_redirects=True, stream=True)
                status = int(resp.status_code)
                resp.close()
        return (200 <= status < 400), f"HTTP {status}"
    except Exception as exc:  # noqa: BLE001 - a probe never breaks the tool
        return False, type(exc).__name__
