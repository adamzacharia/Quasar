"""OAuth sign-in for user MCP servers (MCP authorization spec 2025-06-18+).

Why not the SDK's ``OAuthClientProvider`` end to end: it assumes the browser
redirect and the callback happen inside one awaiting coroutine, which does
not fit a web app (the callback is a separate HTTP request, maybe minutes
later, maybe on another worker). So the sign-in is split in two requests:

1. ``probe`` + ``start_sign_in`` (from POST /api/mcp-servers): detect OAuth
   from a 401 (RFC 9728 ``resource_metadata`` / well-known protected resource
   metadata), discover the authorization server (RFC 8414, OIDC fallback,
   path-aware), register a client (RFC 7591), and build the authorize URL
   with PKCE S256, a server-side single-use ``state`` and the RFC 8707
   ``resource``.
2. ``finish_sign_in`` (from GET /api/mcp/oauth/callback): claim the state,
   check RFC 9207 ``iss``, exchange the code and store the tokens encrypted.

At chat time ``OAuthBearer`` (an ``httpx.Auth``) attaches the access token,
refreshes it shortly before expiry, and on a 401 refreshes once and replays
the request. A refresh the server rejects marks the server ``needs_auth``;
nothing retries until the user signs in again.

Every URL fetched here passes services/mcp_url_policy.py (public address,
https for OAuth endpoints, re-checked on each request). No token, verifier,
code or client secret is logged or put in a user-facing message.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import httpx

from services.mcp_oauth_store import MCPOAuthStore, OAuthRecord
from services.mcp_url_policy import (
    UnsafeMcpUrl,
    check_mcp_url,
    guarded_async_client,
    is_production,
)

CLIENT_NAME = "Quasar"
REFRESH_MARGIN_SECONDS = 60
CALLBACK_PATH = "/api/mcp/oauth/callback"
_INIT_BODY = {
    "jsonrpc": "2.0", "id": 0, "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {},
               "clientInfo": {"name": "Quasar", "version": "1.0"}},
}

# OAuth-specific values our generic redactor does not know about.
_OAUTH_PARAM_RE = re.compile(r"(?i)(?<![A-Za-z0-9_])(code|code_verifier|state|client_secret|refresh_token|access_token|id_token)=([^&\s'\"<>]+)")


def scrub(text: Any, limit: int = 300) -> str:
    """User-facing error text: no credentials, bounded length."""
    from services.secret_redaction import redact_error_text

    cleaned = _OAUTH_PARAM_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", str(text or ""))
    return redact_error_text(cleaned).strip()[:limit]


class OAuthSetupError(Exception):
    """A sign-in step failed; ``str(exc)`` is safe to show the user."""


class TransientOAuthError(Exception):
    """The token endpoint could not be reached or failed with 5xx; retry later."""


# ── configuration ──────────────────────────────────────────────────────────

def public_api_base(request_base_url: Optional[str] = None) -> str:
    """The backend's public origin, used to build the OAuth redirect URI.

    ``QUASAR_PUBLIC_API_URL`` (e.g. https://quasar-oi14.onrender.com) is
    required in production: behind Render's proxy the request URL may not be
    the public one. In development the request's own base URL is used."""
    env = os.getenv("QUASAR_PUBLIC_API_URL", "").strip().rstrip("/")
    if env:
        return env
    if not is_production() and request_base_url:
        return str(request_base_url).rstrip("/")
    raise OAuthSetupError(
        "Sign-in with OAuth is not set up on this Quasar server yet "
        "(QUASAR_PUBLIC_API_URL is missing). Ask the administrator to set it.")


def redirect_uri_for(base: str) -> str:
    return base.rstrip("/") + CALLBACK_PATH


def _preconfigured_client(issuer: str) -> Optional[Dict[str, Any]]:
    """Operator-provided client for authorization servers without dynamic
    registration: ``QUASAR_MCP_OAUTH_CLIENTS`` = JSON object mapping an
    issuer URL to {"client_id": ..., "client_secret": ...?,
    "token_endpoint_auth_method": ...?}."""
    raw = os.getenv("QUASAR_MCP_OAUTH_CLIENTS", "").strip()
    if not raw:
        return None
    try:
        table = json.loads(raw)
    except ValueError:
        print("[MCPOAuth] QUASAR_MCP_OAUTH_CLIENTS is not valid JSON; ignored")
        return None
    entry = table.get(issuer) or table.get(issuer.rstrip("/")) or table.get(issuer.rstrip("/") + "/")
    if not isinstance(entry, dict) or not entry.get("client_id"):
        return None
    method = entry.get("token_endpoint_auth_method") or (
        "client_secret_post" if entry.get("client_secret") else "none")
    return {"client_id": str(entry["client_id"]), "client_secret": entry.get("client_secret"),
            "token_endpoint_auth_method": method, "preconfigured": True}


# ── probe ──────────────────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    kind: str  # "open" | "oauth" | "api_key" | "unreachable" | "blocked"
    message: str = ""
    status_code: Optional[int] = None
    www_scope: Optional[str] = None
    prm: Any = None  # mcp.shared.auth.ProtectedResourceMetadata


def _same_url(a: str, b: str) -> bool:
    return str(a or "").rstrip("/").lower() == str(b or "").rstrip("/").lower()


async def _fetch_prm(client: httpx.AsyncClient, url: str, server_url: str):
    from pydantic import ValidationError

    from mcp.client.auth.utils import create_oauth_metadata_request
    from mcp.shared.auth import ProtectedResourceMetadata
    from mcp.shared.auth_utils import check_resource_allowed, resource_url_from_server_url

    ok, _ = check_mcp_url(url, label="The server's sign-in metadata URL", require_https=True)
    if not ok:
        return None
    try:
        resp = await client.send(create_oauth_metadata_request(url))
    except (httpx.HTTPError, UnsafeMcpUrl):
        return None
    if resp.status_code != 200:
        return None
    try:
        prm = ProtectedResourceMetadata.model_validate_json(resp.content)
    except ValidationError:
        return None
    # RFC 8707 / 9728: the metadata must describe this server (or a parent
    # of it), or a hostile server could send us to someone else's resource.
    if not check_resource_allowed(requested_resource=resource_url_from_server_url(server_url),
                                  configured_resource=str(prm.resource)):
        return None
    return prm


async def probe(url: str) -> ProbeResult:
    """Classify a server before saving it: no auth needed, OAuth, or an API key."""
    from mcp.client.auth.utils import (
        build_protected_resource_metadata_discovery_urls,
        extract_field_from_www_auth,
        extract_scope_from_www_auth,
    )

    ok, reason = check_mcp_url(url)
    if not ok:
        return ProbeResult("blocked", reason)
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json",
               "MCP-Protocol-Version": "2025-06-18"}
    try:
        async with guarded_async_client(follow_redirects=True, timeout=15.0) as client:
            # stream(): never read a body, an SSE reply may stay open.
            async with client.stream("POST", url, json=_INIT_BODY, headers=headers) as resp:
                status = resp.status_code
                www_meta = extract_field_from_www_auth(resp, "resource_metadata") if status == 401 else None
                www_scope = extract_scope_from_www_auth(resp) if status == 401 else None
            if status == 401:
                for prm_url in build_protected_resource_metadata_discovery_urls(www_meta, url):
                    prm = await _fetch_prm(client, prm_url, url)
                    if prm is not None:
                        return ProbeResult("oauth", status_code=401, www_scope=www_scope, prm=prm)
                return ProbeResult("api_key", "This server needs an API key.", status_code=401)
            if status == 403:
                return ProbeResult("api_key", "This server refused the request without an API key.",
                                   status_code=403)
            return ProbeResult("open", status_code=status)
    except UnsafeMcpUrl as exc:
        return ProbeResult("blocked", str(exc))
    except httpx.TimeoutException:
        return ProbeResult("unreachable", "The server did not answer within 15 seconds.")
    except httpx.HTTPError as exc:
        return ProbeResult("unreachable", f"Could not reach the server ({type(exc).__name__}).")


# ── discovery + registration ───────────────────────────────────────────────

def _endpoint_ok(url: Optional[str], what: str) -> None:
    if url is None:
        return
    ok, reason = check_mcp_url(str(url), label=f"The {what}", require_https=True, resolve=False)
    if not ok:
        raise OAuthSetupError(reason)


async def discover_authorization_server(prm, server_url: str) -> Tuple[Any, Dict[str, Any]]:
    """(OAuthMetadata, raw metadata dict) for the first usable authorization
    server the protected resource lists."""
    from pydantic import ValidationError

    from mcp.client.auth.utils import (
        build_oauth_authorization_server_metadata_discovery_urls,
        create_oauth_metadata_request,
    )
    from mcp.shared.auth import OAuthMetadata

    errors = []
    async with guarded_async_client(require_https=True) as client:
        for as_url in [str(u) for u in prm.authorization_servers][:3]:
            ok, reason = check_mcp_url(as_url, label="The sign-in server", require_https=True)
            if not ok:
                errors.append(reason)
                continue
            for meta_url in build_oauth_authorization_server_metadata_discovery_urls(as_url, server_url):
                try:
                    resp = await client.send(create_oauth_metadata_request(meta_url))
                except (httpx.HTTPError, UnsafeMcpUrl) as exc:
                    errors.append(type(exc).__name__)
                    continue
                if resp.status_code >= 500:
                    break
                if resp.status_code != 200:
                    continue
                try:
                    raw = resp.json()
                    asm = OAuthMetadata.model_validate(raw)
                except (ValueError, ValidationError):
                    errors.append("unreadable metadata")
                    continue
                # RFC 8414 section 3.3: the issuer must be the server we asked.
                if not _same_url(str(asm.issuer), as_url):
                    errors.append("issuer mismatch")
                    continue
                return asm, raw
    detail = f" ({', '.join(dict.fromkeys(errors))})" if errors else ""
    raise OAuthSetupError(f"Could not read the sign-in server's settings{detail}.")


def _validate_metadata(asm) -> None:
    _endpoint_ok(str(asm.authorization_endpoint), "sign-in page address")
    _endpoint_ok(str(asm.token_endpoint), "token endpoint")
    _endpoint_ok(str(asm.registration_endpoint) if asm.registration_endpoint else None, "registration endpoint")
    _endpoint_ok(str(asm.revocation_endpoint) if asm.revocation_endpoint else None, "revocation endpoint")
    methods = asm.code_challenge_methods_supported
    # PKCE S256 is mandatory in MCP. An absent list is tolerated (common on
    # OIDC providers that do support it); an explicit list without S256 is not.
    if methods is not None and "S256" not in methods:
        raise OAuthSetupError("This sign-in server does not support PKCE (S256), which Quasar requires.")
    grants = asm.grant_types_supported
    if grants is not None and "authorization_code" not in grants:
        raise OAuthSetupError("This sign-in server does not support the authorization code flow.")
    if "code" not in (asm.response_types_supported or ["code"]):
        raise OAuthSetupError("This sign-in server does not support the authorization code flow.")


def _pick_auth_method(asm) -> str:
    supported = asm.token_endpoint_auth_methods_supported
    if not supported or "none" in supported:
        return "none"
    for method in ("client_secret_post", "client_secret_basic"):
        if method in supported:
            return method
    raise OAuthSetupError("This sign-in server needs a client authentication method Quasar does not support.")


async def register_client(asm, redirect_uri: str, scope: Optional[str]) -> Dict[str, Any]:
    from pydantic import ValidationError

    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata

    pre = _preconfigured_client(str(asm.issuer))
    if pre:
        return pre
    if not asm.registration_endpoint:
        raise OAuthSetupError(
            "This server does not let apps register for sign-in automatically, so it needs a "
            "client ID set up for Quasar in advance. Ask the administrator, or use an API key if "
            "the server offers one.")
    metadata = OAuthClientMetadata(
        redirect_uris=[redirect_uri],
        token_endpoint_auth_method=_pick_auth_method(asm),
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope=scope,
        client_name=CLIENT_NAME,
    )
    async with guarded_async_client(require_https=True) as client:
        try:
            resp = await client.post(str(asm.registration_endpoint),
                                     json=metadata.model_dump(by_alias=True, mode="json", exclude_none=True))
        except (httpx.HTTPError, UnsafeMcpUrl) as exc:
            raise OAuthSetupError(f"Could not register Quasar with the sign-in server ({type(exc).__name__}).")
    if resp.status_code not in (200, 201):
        raise OAuthSetupError(
            f"The sign-in server refused to register Quasar ({resp.status_code}{_oauth_error_code(resp)}).")
    try:
        info = OAuthClientInformationFull.model_validate_json(resp.content)
    except ValidationError:
        raise OAuthSetupError("The sign-in server sent an unreadable registration reply.")
    if not info.client_id:
        raise OAuthSetupError("The sign-in server did not issue a client ID.")
    out = info.model_dump(mode="json", exclude_none=True)
    out["token_endpoint_auth_method"] = info.token_endpoint_auth_method or metadata.token_endpoint_auth_method
    return out


def _oauth_error_code(resp: httpx.Response) -> str:
    """", invalid_grant" style suffix from an OAuth error body (code only,
    never the free-text description, which some servers fill with input)."""
    try:
        code = (resp.json() or {}).get("error")
    except ValueError:
        return ""
    if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.\-]{1,64}", code):
        return f", {code}"
    return ""


# ── sign-in ────────────────────────────────────────────────────────────────

def _scope_for(probe_result: ProbeResult, asm) -> Optional[str]:
    from mcp.client.auth.utils import get_client_metadata_scopes

    scope = get_client_metadata_scopes(probe_result.www_scope, probe_result.prm, asm)
    scope = " ".join((scope or "").split())
    return scope or None


def _resource_for(server_url: str, prm) -> str:
    from mcp.shared.auth_utils import check_resource_allowed, resource_url_from_server_url

    resource = resource_url_from_server_url(server_url)
    if prm is not None and prm.resource:
        candidate = str(prm.resource)
        if check_resource_allowed(requested_resource=resource, configured_resource=candidate):
            resource = candidate
    return resource


def _with_query(url: str, params: Dict[str, str]) -> str:
    parts = urlsplit(url)
    query = parse_qsl(parts.query, keep_blank_values=True) + list(params.items())
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def _client_usable(client: Dict[str, Any], redirect_uri: str) -> bool:
    if not client.get("client_id"):
        return False
    if client.get("preconfigured"):
        return True
    uris = [str(u) for u in client.get("redirect_uris") or []]
    if redirect_uri not in uris:
        return False
    expires = client.get("client_secret_expires_at")
    import time

    return not (expires and expires < time.time() + 60)


async def start_sign_in(*, user_id: str, server: str, server_url: str, probe_result: ProbeResult,
                        redirect_uri: str, return_origin: str,
                        store: Optional[MCPOAuthStore] = None) -> str:
    """Discover, register if needed, and return the provider's sign-in URL."""
    from mcp.client.auth.oauth2 import PKCEParameters

    if probe_result.prm is None:
        raise OAuthSetupError("This server did not describe how to sign in.")
    ok, reason = check_mcp_url(server_url, require_https=True)
    if not ok:
        raise OAuthSetupError(reason)
    store = store or MCPOAuthStore()
    asm, raw = await discover_authorization_server(probe_result.prm, server_url)
    _validate_metadata(asm)
    issuer = str(asm.issuer)
    scope = _scope_for(probe_result, asm)
    resource = _resource_for(server_url, probe_result.prm)

    existing = await asyncio.to_thread(store.get, user_id, server)
    client = None
    if existing and _same_url(existing.issuer, issuer) and _client_usable(existing.client, redirect_uri):
        client = existing.client
    if client is None:
        client = await register_client(asm, redirect_uri, scope)
    metadata = {k: raw.get(k) for k in (
        "issuer", "authorization_endpoint", "token_endpoint", "registration_endpoint",
        "revocation_endpoint", "authorization_response_iss_parameter_supported", "scopes_supported",
    ) if raw.get(k) is not None}
    await asyncio.to_thread(
        store.save_registration, user_id, server, server_url=server_url, resource=resource,
        issuer=issuer, metadata=metadata, client=client, redirect_uri=redirect_uri)

    pkce = PKCEParameters.generate()
    state = await asyncio.to_thread(
        store.create_state, user_id, server, code_verifier=pkce.code_verifier, resource=resource,
        redirect_uri=redirect_uri, return_origin=return_origin)
    params = {
        "response_type": "code",
        "client_id": str(client["client_id"]),
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": pkce.code_challenge,
        "code_challenge_method": "S256",
        "resource": resource,
    }
    if scope:
        params["scope"] = scope
    return _with_query(str(asm.authorization_endpoint), params)


def _client_auth(client: Dict[str, Any], data: Dict[str, str], headers: Dict[str, str]) -> None:
    method = client.get("token_endpoint_auth_method") or "none"
    secret = client.get("client_secret")
    if method == "client_secret_basic" and secret:
        pair = f"{quote(str(client['client_id']), safe='')}:{quote(str(secret), safe='')}"
        headers["Authorization"] = "Basic " + base64.b64encode(pair.encode()).decode()
    elif method == "client_secret_post" and secret:
        data["client_secret"] = str(secret)


async def _token_request(rec: OAuthRecord, data: Dict[str, str]) -> httpx.Response:
    token_url = str(rec.metadata.get("token_endpoint") or "")
    _endpoint_ok(token_url or None, "token endpoint")
    if not token_url:
        raise OAuthSetupError("The sign-in server has no token endpoint.")
    data = {**data, "client_id": str(rec.client.get("client_id") or "")}
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    _client_auth(rec.client, data, headers)
    async with guarded_async_client(require_https=True) as client:
        try:
            return await client.post(token_url, data=data, headers=headers)
        except UnsafeMcpUrl as exc:
            raise OAuthSetupError(str(exc))
        except httpx.HTTPError as exc:
            raise TransientOAuthError(f"Could not reach the sign-in server ({type(exc).__name__}).")


def _parse_token(resp: httpx.Response):
    from pydantic import ValidationError

    from mcp.shared.auth import OAuthToken

    try:
        return OAuthToken.model_validate_json(resp.content)
    except ValidationError:
        raise OAuthSetupError("The sign-in server sent a token Quasar cannot use (only Bearer tokens are supported).")


@dataclass
class SignInResult:
    ok: bool
    message: str
    return_origin: Optional[str] = None
    user_id: Optional[str] = None
    server: Optional[str] = None


_STATE_REASONS = {
    "unknown": "This sign-in link is not valid. Start again from Settings.",
    "expired": "This sign-in took longer than 10 minutes and expired. Start again from Settings.",
    "used": "This sign-in link was already used. Start again from Settings if the server is not connected.",
}


async def finish_sign_in(*, state: Optional[str], code: Optional[str], error: Optional[str] = None,
                         error_description: Optional[str] = None, iss: Optional[str] = None,
                         store: Optional[MCPOAuthStore] = None) -> SignInResult:
    store = store or MCPOAuthStore()
    st, why = await asyncio.to_thread(store.consume_state, state or "")
    if st is None:
        return SignInResult(False, _STATE_REASONS.get(why, _STATE_REASONS["unknown"]),
                            return_origin=await asyncio.to_thread(store.peek_state_origin, state or ""))
    base = SignInResult(False, "", return_origin=st.return_origin, user_id=st.user_id, server=st.server)
    if error:
        base.message = ("Sign-in was cancelled." if error == "access_denied"
                        else f"The sign-in page reported an error ({scrub(error, 64)}).")
        return base
    if not code:
        base.message = "The sign-in page did not return an authorization code."
        return base
    rec = await asyncio.to_thread(store.get, st.user_id, st.server)
    if rec is None or not rec.client.get("client_id"):
        base.message = "This server was removed or changed while signing in. Start again from Settings."
        return base
    # RFC 9207: reject a code minted by a different authorization server.
    if iss is not None and not _same_url(iss, rec.issuer):
        base.message = "The sign-in reply came from an unexpected server and was rejected."
        return base
    if iss is None and rec.metadata.get("authorization_response_iss_parameter_supported"):
        base.message = "The sign-in reply was missing its issuer and was rejected."
        return base
    try:
        resp = await _token_request(rec, {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": st.redirect_uri,
            "code_verifier": st.code_verifier,
            "resource": st.resource,
        })
        if resp.status_code != 200:
            base.message = f"The sign-in server did not accept the sign-in ({resp.status_code}{_oauth_error_code(resp)})."
            return base
        token = _parse_token(resp)
    except (OAuthSetupError, TransientOAuthError) as exc:
        base.message = scrub(exc)
        return base
    await asyncio.to_thread(
        store.set_tokens, st.user_id, st.server, access_token=token.access_token,
        refresh_token=token.refresh_token, expires_in=token.expires_in, scope=token.scope or rec.scope)
    base.ok, base.message = True, "Signed in."
    return base


# ── tokens at chat time ────────────────────────────────────────────────────

_REFRESH_LOCKS: Dict[Tuple[int, str, str], asyncio.Lock] = {}
_REFRESH_LOCKS_GUARD = threading.Lock()


def _refresh_lock(user_id: str, server: str) -> asyncio.Lock:
    key = (id(asyncio.get_running_loop()), user_id, server)
    with _REFRESH_LOCKS_GUARD:
        lock = _REFRESH_LOCKS.get(key)
        if lock is None:
            lock = _REFRESH_LOCKS[key] = asyncio.Lock()
        return lock


SIGN_IN_AGAIN = "Sign-in expired. Open Settings, MCP servers, and press Reconnect."


async def current_access_token(user_id: str, server: str, *, failed_token: Optional[str] = None,
                               store: Optional[MCPOAuthStore] = None) -> Optional[str]:
    """A usable access token, refreshing first when it is about to expire or
    when ``failed_token`` was just rejected with a 401. None means the server
    needs a new sign-in (already recorded in the store). Raises
    TransientOAuthError when the sign-in server is unreachable."""
    store = store or MCPOAuthStore()
    async with _refresh_lock(user_id, server):
        rec = await asyncio.to_thread(store.get, user_id, server)
        if rec is None or not rec.access_token:
            return None
        rejected = failed_token is not None and rec.access_token == failed_token
        if not rejected and not rec.expires_within(REFRESH_MARGIN_SECONDS):
            # Valid, or another request already refreshed past the failure.
            return rec.access_token
        if not rec.refresh_token:
            await asyncio.to_thread(store.mark_needs_auth, user_id, server, SIGN_IN_AGAIN)
            return None
        resp = await _token_request(rec, {
            "grant_type": "refresh_token",
            "refresh_token": rec.refresh_token,
            "resource": rec.resource,
        })
        if resp.status_code in (400, 401, 403):
            print(f"[MCPOAuth] refresh rejected for {server!r} ({resp.status_code}{_oauth_error_code(resp)}); needs sign-in")
            await asyncio.to_thread(store.mark_needs_auth, user_id, server, SIGN_IN_AGAIN)
            return None
        if resp.status_code != 200:
            raise TransientOAuthError(f"The sign-in server failed to refresh the token ({resp.status_code}).")
        try:
            token = _parse_token(resp)
        except OAuthSetupError:
            await asyncio.to_thread(store.mark_needs_auth, user_id, server, SIGN_IN_AGAIN)
            return None
        await asyncio.to_thread(
            store.set_tokens, user_id, server, access_token=token.access_token,
            # Servers that do not rotate refresh tokens omit it; keep ours.
            refresh_token=token.refresh_token or rec.refresh_token,
            expires_in=token.expires_in, scope=token.scope or rec.scope)
        return token.access_token


class OAuthBearer(httpx.Auth):
    """Bearer auth for one (user, server), with refresh-and-replay on 401."""

    requires_request_body = True

    def __init__(self, user_id: str, server: str, store: Optional[MCPOAuthStore] = None):
        self.user_id, self.server = user_id, server
        self._store = store

    def sync_auth_flow(self, request):  # pragma: no cover - the MCP SDK is async only
        raise RuntimeError("OAuthBearer is async only")

    async def async_auth_flow(self, request: httpx.Request):
        token = await current_access_token(self.user_id, self.server, store=self._store)
        if token is None:
            raise OAuthSignInRequired(SIGN_IN_AGAIN)
        request.headers["Authorization"] = f"Bearer {token}"
        response = yield request
        if response.status_code != 401:
            return
        fresh = await current_access_token(self.user_id, self.server, failed_token=token, store=self._store)
        if fresh is None or fresh == token:
            if fresh == token:
                # The server rejects a token the sign-in server still calls
                # valid: stop here instead of looping on it.
                await asyncio.to_thread((self._store or MCPOAuthStore()).mark_needs_auth,
                                        self.user_id, self.server, SIGN_IN_AGAIN)
            return
        request.headers["Authorization"] = f"Bearer {fresh}"
        yield request


class OAuthSignInRequired(Exception):
    """No usable token: the user has to sign in again."""


# ── disconnect ─────────────────────────────────────────────────────────────

async def revoke_and_forget(user_id: str, server: str, *, keep_registration: bool = False,
                            store: Optional[MCPOAuthStore] = None) -> bool:
    """Best-effort RFC 7009 revocation, then drop the tokens (and, unless
    ``keep_registration``, the whole OAuth row). Returns True if the server
    confirmed a revocation."""
    store = store or MCPOAuthStore()
    rec = await asyncio.to_thread(store.get, user_id, server)
    revoked = False
    if rec is not None and rec.metadata.get("revocation_endpoint") and (rec.refresh_token or rec.access_token):
        url = str(rec.metadata["revocation_endpoint"])
        ok, _ = check_mcp_url(url, require_https=True, resolve=False)
        if ok:
            async with guarded_async_client(require_https=True, timeout=10.0) as client:
                for token, hint in ((rec.refresh_token, "refresh_token"), (rec.access_token, "access_token")):
                    if not token:
                        continue
                    data = {"token": token, "token_type_hint": hint,
                            "client_id": str(rec.client.get("client_id") or "")}
                    headers = {"Content-Type": "application/x-www-form-urlencoded"}
                    _client_auth(rec.client, data, headers)
                    try:
                        resp = await client.post(url, data=data, headers=headers)
                        revoked = revoked or resp.status_code == 200
                    except (httpx.HTTPError, UnsafeMcpUrl):
                        pass
    if keep_registration:
        if rec is not None:
            await asyncio.to_thread(store.mark_needs_auth, user_id, server, "Disconnected. Press Reconnect to sign in again.")
    else:
        await asyncio.to_thread(store.delete, user_id, server)
    return revoked
