"""Authentication endpoints: local register/login, Google OAuth, and /me."""

import ipaddress
import os

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from api.deps import auth_service, get_current_user, is_admin_email
from api.models import GoogleLoginRequest, LoginRequest, RegisterRequest
from services.auth_cookie import clear_auth_cookie, set_auth_cookie
from services.login_rate_limit import LoginRateLimiter, _int_env

router = APIRouter()

# ── Per-IP login throttle (S6) ────────────────────────────────────────────────
# Second dimension next to the per-username limiter inside AuthService: catches
# one source spraying MANY usernames. Deliberately looser (default 20 vs 5) so
# a shared NAT/proxy egress is not locked out by one bad actor. Per-process
# defense-in-depth — a distributed attack still needs an edge/WAF limiter.
_ip_limiter = LoginRateLimiter(
    max_attempts=_int_env("QUASAR_LOGIN_IP_MAX_ATTEMPTS", 20),
    window_seconds=_int_env("QUASAR_LOGIN_IP_WINDOW_SECONDS", 300),
    lockout_seconds=_int_env("QUASAR_LOGIN_IP_LOCKOUT_SECONDS", 300),
    max_tracked=_int_env("QUASAR_LOGIN_IP_MAX_TRACKED_KEYS", 4096),
)


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_ip(value: str) -> str:
    """Canonicalize an IP string (IPv6 zones/brackets, v4-mapped forms)."""
    raw = (value or "").strip().strip("[]")
    if not raw:
        return ""
    raw = raw.split("%", 1)[0]  # drop an IPv6 zone id
    try:
        return str(ipaddress.ip_address(raw))
    except ValueError:
        return raw


def _client_ip(request: Request) -> str:
    """Client key for the IP limiter.

    X-Forwarded-For is honored ONLY when the deployment declares a trusted
    ingress via ``QUASAR_TRUST_XFF=1`` (e.g. Render, where the socket peer is
    always the proxy). Default: the socket peer.

    **Read from the RIGHT, never the left.** Each proxy *appends* the address it
    received the request from, so the header is ``<client-supplied…>, <real peer>``.
    The leftmost element is whatever the client sent and is fully forgeable:
    trusting it would let an attacker (a) mint a fresh limiter key per request,
    bypassing the per-IP bucket entirely, and (b) lock out any victim IP by
    spending the bucket under a forged key. The entry written by our own
    outermost trusted proxy is ``parts[-hops]``, where ``hops`` is the number of
    trusted proxies in front of the app (``QUASAR_TRUSTED_PROXY_HOPS``, default 1).
    If the header is shorter than the declared hop count it has been tampered
    with (or the config is wrong) — fall back to the socket peer.
    """
    if _truthy(os.environ.get("QUASAR_TRUST_XFF")):
        parts = [
            part.strip()
            for part in request.headers.get("x-forwarded-for", "").split(",")
            if part.strip()
        ]
        hops = max(1, _int_env("QUASAR_TRUSTED_PROXY_HOPS", 1))
        if len(parts) >= hops:
            normalized = _normalize_ip(parts[-hops])
            if normalized:
                return normalized
    host = request.client.host if request.client else ""
    return _normalize_ip(host) or "unknown"


def _throttle_429(retry_after: float) -> HTTPException:
    seconds = max(1, int(retry_after) + 1)
    return HTTPException(
        status_code=429,
        detail="Too many failed login attempts. Please try again later.",
        headers={"Retry-After": str(seconds)},
    )


@router.post("/api/auth/register")
async def register(req: RegisterRequest, response: Response):
    success, msg, token = auth_service.register_user(
        req.username, req.password, req.email, req.display_name
    )
    if not success:
        raise HTTPException(status_code=400, detail=msg)

    # We verify token to get full payload for frontend
    payload = auth_service.verify_token(token)
    # S6: also deliver the JWT as an httpOnly cookie (kept in the body too during
    # the Bearer→cookie transition; the frontend stops persisting it).
    set_auth_cookie(response, token)
    return {
        "token": token,
        "user": {
            "id": payload["sub"],
            "username": payload.get("email") or req.username,
            "display_name": payload.get("name") or req.username,
            "auth_provider": "local",
            "is_admin": is_admin_email(payload.get("email") or req.email or req.username),
        }
    }


@router.post("/api/auth/login")
async def login(req: LoginRequest, response: Response, request: Request):
    # S6 throttle semantics: an attempt made while ALREADY locked → 429 with
    # Retry-After; the attempt that trips a limiter still returns 401 (the
    # credentials were wrong). Checked before any hashing happens.
    retry_after = auth_service.login_retry_after(req.username)
    if retry_after is not None:
        raise _throttle_429(retry_after)
    client_ip = _client_ip(request)
    ip_retry_after = _ip_limiter.seconds_until_unblocked(client_ip)
    if ip_retry_after is not None:
        raise _throttle_429(ip_retry_after)

    success, user_id, email, display_name, msg = auth_service.login_user(req.username, req.password)
    if not success:
        # Only invalid-credential failures consume IP budget (never Google-
        # provider redirects or 429 short-circuits).
        if msg == "Invalid username or password":
            _ip_limiter.record_failure(client_ip)
        # Defensive: a concurrent request can trip the per-user lock between
        # the pre-check above and login_user's internal check — map the
        # service-level throttle message to 429 rather than 401.
        if msg.startswith("Too many failed login attempts"):
            late_retry = auth_service.login_retry_after(req.username)
            raise _throttle_429(late_retry if late_retry is not None else 1.0)
        raise HTTPException(status_code=401, detail=msg)

    # NOTE: the IP bucket deliberately does NOT reset on success — a sprayer
    # must not be able to erase its own history with one valid credential.
    # (The per-username limiter resets inside login_user, which is correct:
    # it protects a single account, and its owner just proved themselves.)
    token = auth_service.generate_token(user_id, email, display_name)
    set_auth_cookie(response, token)  # S6: httpOnly cookie
    return {
        "token": token,
        "user": {
            "id": user_id,
            "username": email or req.username,
            "display_name": display_name or req.username,
            "auth_provider": "local",
            "is_admin": is_admin_email(email or req.username),
        }
    }


@router.post("/api/auth/google")
async def google_login(req: GoogleLoginRequest, response: Response):
    try:
        # Verify the Google ID token
        client_id = os.environ.get("GOOGLE_CLIENT_ID")
        if not client_id:
            raise HTTPException(status_code=500, detail="Google authentication is not configured on the server")

        idinfo = id_token.verify_oauth2_token(
            req.credential, google_requests.Request(), client_id
        )

        email = idinfo.get("email")
        name = idinfo.get("name")
        picture = idinfo.get("picture")  # Google profile picture URL

        if not email:
            raise HTTPException(status_code=400, detail="Google account has no email")

        success, user_id, msg, token = auth_service.register_or_login_google_user(email, name, picture)

        if not success:
            raise HTTPException(status_code=400, detail=msg)

        set_auth_cookie(response, token)  # S6: httpOnly cookie
        return {
            "token": token,
            "user": {
                "id": user_id,
                "username": email,
                "display_name": name,
                "auth_provider": "google",
                "picture_url": picture,
                "is_admin": is_admin_email(email),
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"Invalid Google token: {str(e)}")


@router.post("/api/auth/logout")
async def logout(response: Response):
    """Clear the httpOnly auth cookie (S6). Safe to call when not logged in."""
    clear_auth_cookie(response)
    return {"status": "ok"}


@router.get("/api/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return {
        "user": {
            "id": current_user["sub"],
            "username": current_user.get("email"),
            "display_name": current_user.get("name"),
            "is_admin": is_admin_email(current_user.get("email")),
        }
    }
