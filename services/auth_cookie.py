"""httpOnly auth-cookie helper (S6 — WIRED).

Server-side half of moving the JWT out of ``localStorage`` (XSS-exfiltratable)
and into an ``httpOnly`` cookie. Wired call sites:

- ``ui-pro/api/routers/auth.py`` — ``set_auth_cookie`` on /api/auth/login,
  /register, /google; ``clear_auth_cookie`` on POST /api/auth/logout.
- ``ui-pro/api/deps.py`` — ``get_current_user`` / ``_resolve_optional_user``
  fall back to ``read_auth_cookie(request)`` when no Bearer header is present.
- ``ui-pro/api/sse.py`` + routers/chat.py + routers/analytics.py thread the
  cookie token through the optional-user chat paths.
- Frontend: every backend fetch sends ``credentials: "include"``; the auth
  store persists only the user object (never the JWT) and bootstraps from
  GET /api/auth/me.

CORS (S4) already uses ``allow_credentials=True`` with an explicit origin
allowlist. Dev (localhost:3001 → localhost:8000) is same-site, so SameSite=Lax
suffices; production (Vercel frontend → Render API) is cross-site →
``SameSite=None; Secure`` (auto-resolved below).

CSRF: because the browser attaches this cookie automatically, cookie auth has a
CSRF surface that Bearer auth did not, and ``SameSite=None`` does not close it.
``CookieCsrfMiddleware`` in ``ui-pro/api/main.py`` therefore rejects unsafe-method
requests that carry this cookie from a non-allow-listed ``Origin`` (403). Do not
remove it without replacing it — several write endpoints are preflight-free
(multipart uploads; ``/api/feedback`` parses the body via ``await req.json()``,
which accepts ``text/plain``).
"""

from __future__ import annotations

import os
from typing import Any, Optional

# Cookie name the browser stores the JWT under.
AUTH_COOKIE_NAME = os.environ.get("QUASAR_AUTH_COOKIE_NAME", "quasar_auth")

# Keep in sync with services.auth.JWT_EXPIRATION_HOURS (1 week) so the cookie
# lifetime matches the token lifetime.
_DEFAULT_MAX_AGE_SECONDS = 24 * 7 * 3600


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _is_production() -> bool:
    env = (
        os.environ.get("QUASAR_ENV")
        or os.environ.get("APP_ENV")
        or os.environ.get("ENVIRONMENT")
        or "development"
    ).strip().lower()
    return env in {"production", "prod"}


def cookie_samesite() -> str:
    """Resolve the SameSite policy.

    Cross-site deployments (frontend and API on different hosts, the Quasar
    production topology) need ``none``; same-site/local dev defaults to ``lax``.
    Override with ``QUASAR_AUTH_COOKIE_SAMESITE``.
    """
    override = os.environ.get("QUASAR_AUTH_COOKIE_SAMESITE")
    if override and override.strip().lower() in {"lax", "strict", "none"}:
        return override.strip().lower()
    return "none" if _is_production() else "lax"


def cookie_secure() -> bool:
    """Whether to mark the cookie ``Secure`` (HTTPS-only).

    ``Secure`` is *forced on* (non-overridable) in production and whenever
    SameSite=None, because browsers reject a ``SameSite=None`` cookie without
    ``Secure`` and a production auth cookie must never travel in cleartext.
    Only outside those cases does ``QUASAR_AUTH_COOKIE_SECURE`` apply.
    """
    # Non-overridable safety conditions first, so an explicit `=0` can never
    # produce the browser-rejected SameSite=None-without-Secure combination.
    if _is_production() or cookie_samesite() == "none":
        return True
    override = os.environ.get("QUASAR_AUTH_COOKIE_SECURE")
    if override is not None and str(override).strip() != "":
        return _truthy(override)
    return False


def cookie_max_age() -> int:
    return _DEFAULT_MAX_AGE_SECONDS


def cookie_params() -> dict:
    """Return the kwargs used for ``response.set_cookie`` (minus key/value).

    Exposed separately so it can be unit-tested without a real Response.
    """
    params = {
        "max_age": cookie_max_age(),
        "path": "/",
        "httponly": True,
        "secure": cookie_secure(),
        "samesite": cookie_samesite(),
    }
    domain = os.environ.get("QUASAR_AUTH_COOKIE_DOMAIN")
    if domain and domain.strip():
        params["domain"] = domain.strip()
    return params


def set_auth_cookie(response: Any, token: str) -> None:
    """Attach the JWT to *response* as an httpOnly cookie."""
    response.set_cookie(key=AUTH_COOKIE_NAME, value=token, **cookie_params())


def clear_auth_cookie(response: Any) -> None:
    """Expire the auth cookie (logout)."""
    params = cookie_params()
    # delete_cookie only accepts a subset of set_cookie kwargs.
    response.delete_cookie(
        key=AUTH_COOKIE_NAME,
        path=params["path"],
        domain=params.get("domain"),
        secure=params["secure"],
        httponly=params["httponly"],
        samesite=params["samesite"],
    )


def read_auth_cookie(request: Any) -> Optional[str]:
    """Return the JWT stored in the request's auth cookie, if present."""
    try:
        cookies = request.cookies
    except AttributeError:
        return None
    if not cookies:
        return None
    token = cookies.get(AUTH_COOKIE_NAME)
    return token or None
