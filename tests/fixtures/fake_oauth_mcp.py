"""A fake OAuth authorization server + OAuth-protected MCP server, in one ASGI app.

Used by tests/unit/test_mcp_oauth.py in-process, and runnable standalone for
the live UI check:

    .venv/Scripts/python tests/fixtures/fake_oauth_mcp.py --port 8765

Implements just enough of the specs Quasar's connector relies on:
RFC 9728 protected resource metadata (and a 401 WWW-Authenticate that points
at it), RFC 8414 authorization server metadata, RFC 7591 dynamic client
registration, the authorization code grant with PKCE S256 (an "Allow / Deny"
page), refresh tokens with rotation, RFC 8707 resource checks, RFC 7009
revocation, and RFC 9207 ``iss`` on the redirect. The MCP endpoint is a real
FastMCP streamable-HTTP app behind a bearer-token check.

Test knobs live on ``FakeAuthState`` (token lifetime, revoke everything,
auto-approve without the HTML page, counters).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlencode, urlsplit


@dataclass
class FakeAuthState:
    base_url: str = "http://127.0.0.1:8765"
    access_ttl: int = 3600
    clients: Dict[str, dict] = field(default_factory=dict)
    codes: Dict[str, dict] = field(default_factory=dict)
    access: Dict[str, float] = field(default_factory=dict)      # token -> expiry
    refresh: Dict[str, str] = field(default_factory=dict)       # token -> client_id
    revoked: List[str] = field(default_factory=list)
    registrations: int = 0
    token_calls: List[str] = field(default_factory=list)        # grant types seen
    resources_seen: List[str] = field(default_factory=list)
    mcp_auth_headers: List[str] = field(default_factory=list)
    send_iss: bool = True
    tool_calls: int = 0

    @property
    def resource(self) -> str:
        return self.base_url.rstrip("/") + "/mcp"

    def expire_all_access(self) -> None:
        for tok in list(self.access):
            self.access[tok] = 0.0

    def revoke_everything(self) -> None:
        self.access.clear()
        self.refresh.clear()

    def issue(self, client_id: str) -> dict:
        at, rt = "at_" + secrets.token_urlsafe(24), "rt_" + secrets.token_urlsafe(24)
        self.access[at] = time.time() + self.access_ttl
        self.refresh[rt] = client_id
        return {"access_token": at, "token_type": "Bearer", "expires_in": self.access_ttl,
                "refresh_token": rt, "scope": "mcp:tools"}


def _json(send_status, body, headers=None):
    from starlette.responses import JSONResponse

    return JSONResponse(body, status_code=send_status, headers=headers or {})


def make_app(state: FakeAuthState):
    """(asgi_app). ``state.base_url`` must be the URL the app is reachable at."""
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
    from starlette.routing import Route

    base = state.base_url.rstrip("/")

    mcp = FastMCP("fake-oauth-tools",
                  transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))

    @mcp.tool()
    def echo(text: str) -> str:
        """Repeat the text back (from the fake OAuth MCP server)."""
        state.tool_calls += 1
        return f"echo: {text}"

    @mcp.tool()
    def add(a: int, b: int) -> int:
        """Add two integers (from the fake OAuth MCP server)."""
        state.tool_calls += 1
        return a + b

    mcp_app = mcp.streamable_http_app()

    async def prm(request: Request):
        return JSONResponse({"resource": state.resource, "authorization_servers": [base],
                             "scopes_supported": ["mcp:tools"], "bearer_methods_supported": ["header"],
                             "resource_name": "Fake OAuth Tools"})

    async def asm(request: Request):
        return JSONResponse({
            "issuer": base,
            "authorization_endpoint": f"{base}/authorize",
            "token_endpoint": f"{base}/token",
            "registration_endpoint": f"{base}/register",
            "revocation_endpoint": f"{base}/revoke",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "token_endpoint_auth_methods_supported": ["none"],
            "code_challenge_methods_supported": ["S256"],
            "scopes_supported": ["mcp:tools"],
            "authorization_response_iss_parameter_supported": state.send_iss,
        })

    async def register(request: Request):
        body = await request.json()
        uris = body.get("redirect_uris") or []
        if not uris:
            return JSONResponse({"error": "invalid_redirect_uri"}, status_code=400)
        client_id = "client_" + secrets.token_urlsafe(8)
        state.registrations += 1
        state.clients[client_id] = {"redirect_uris": uris, "client_name": body.get("client_name")}
        return JSONResponse({"client_id": client_id, "client_id_issued_at": int(time.time()),
                             "redirect_uris": uris, "token_endpoint_auth_method": "none",
                             "grant_types": ["authorization_code", "refresh_token"],
                             "response_types": ["code"], "client_name": body.get("client_name")},
                            status_code=201)

    def _check_authorize(params) -> Optional[str]:
        client = state.clients.get(params.get("client_id", ""))
        if client is None:
            return "unknown client"
        if params.get("redirect_uri") not in client["redirect_uris"]:
            return "redirect_uri not registered"
        if params.get("code_challenge_method") != "S256" or not params.get("code_challenge"):
            return "PKCE S256 required"
        if params.get("resource") != state.resource:
            return f"resource must be {state.resource}"
        return None

    def _approve(params) -> str:
        code = "code_" + secrets.token_urlsafe(16)
        state.codes[code] = {"client_id": params["client_id"], "redirect_uri": params["redirect_uri"],
                             "challenge": params["code_challenge"], "resource": params["resource"],
                             "exp": time.time() + 120}
        query = {"code": code, "state": params.get("state", "")}
        if state.send_iss:
            query["iss"] = base
        return f"{params['redirect_uri']}?{urlencode(query)}"

    async def authorize(request: Request):
        params = dict(request.query_params)
        problem = _check_authorize(params)
        if problem:
            return HTMLResponse(f"<h1>Bad request</h1><p>{html.escape(problem)}</p>", status_code=400)
        name = html.escape(state.clients[params["client_id"]].get("client_name") or "An app")
        hidden = "".join(f'<input type="hidden" name="{html.escape(k)}" value="{html.escape(v)}">'
                         for k, v in params.items())
        page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Fake OAuth sign-in</title>
<style>body{{font:15px system-ui;background:#f4f4f2;display:flex;justify-content:center;padding-top:80px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:14px;padding:28px 32px;max-width:380px}}
button{{font:inherit;padding:8px 18px;border-radius:999px;border:1px solid #222;margin-right:8px;cursor:pointer}}
.allow{{background:#222;color:#fff}}</style></head><body><div class="card">
<h2>Fake OAuth Tools</h2><p><b>{name}</b> wants to use your Fake OAuth Tools account (scope mcp:tools).</p>
<form method="post" action="/authorize/decide">{hidden}
<button class="allow" name="decision" value="allow" type="submit">Allow</button>
<button name="decision" value="deny" type="submit">Deny</button></form></div></body></html>"""
        return HTMLResponse(page)

    async def decide(request: Request):
        form = await request.form()
        params = {k: str(v) for k, v in form.items()}
        decision = params.pop("decision", "deny")
        problem = _check_authorize(params)
        if problem:
            return HTMLResponse(f"<h1>Bad request</h1><p>{html.escape(problem)}</p>", status_code=400)
        if decision != "allow":
            query = {"error": "access_denied", "state": params.get("state", "")}
            if state.send_iss:
                query["iss"] = base
            return RedirectResponse(f"{params['redirect_uri']}?{urlencode(query)}", status_code=302)
        return RedirectResponse(_approve(params), status_code=302)

    async def token(request: Request):
        form = {k: str(v) for k, v in (await request.form()).items()}
        grant = form.get("grant_type", "")
        state.token_calls.append(grant)
        state.resources_seen.append(form.get("resource", ""))
        if form.get("resource") != state.resource:
            return JSONResponse({"error": "invalid_target"}, status_code=400)
        if grant == "authorization_code":
            entry = state.codes.pop(form.get("code", ""), None)
            if entry is None or entry["exp"] < time.time():
                return JSONResponse({"error": "invalid_grant"}, status_code=400)
            if entry["client_id"] != form.get("client_id") or entry["redirect_uri"] != form.get("redirect_uri"):
                return JSONResponse({"error": "invalid_grant"}, status_code=400)
            digest = hashlib.sha256(form.get("code_verifier", "").encode()).digest()
            if base64.urlsafe_b64encode(digest).decode().rstrip("=") != entry["challenge"]:
                return JSONResponse({"error": "invalid_grant", "error_description": "PKCE failed"}, status_code=400)
            return JSONResponse(state.issue(entry["client_id"]))
        if grant == "refresh_token":
            client_id = state.refresh.pop(form.get("refresh_token", ""), None)
            if client_id is None or client_id != form.get("client_id"):
                return JSONResponse({"error": "invalid_grant"}, status_code=400)
            return JSONResponse(state.issue(client_id))  # rotates the refresh token
        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

    async def revoke(request: Request):
        form = {k: str(v) for k, v in (await request.form()).items()}
        tok = form.get("token", "")
        state.revoked.append(tok)
        state.access.pop(tok, None)
        state.refresh.pop(tok, None)
        return Response(status_code=200)

    oauth_app = Starlette(routes=[
        Route("/.well-known/oauth-protected-resource", prm),
        Route("/.well-known/oauth-protected-resource/mcp", prm),
        Route("/.well-known/oauth-authorization-server", asm),
        Route("/register", register, methods=["POST"]),
        Route("/authorize", authorize),
        Route("/authorize/decide", decide, methods=["POST"]),
        Route("/token", token, methods=["POST"]),
        Route("/revoke", revoke, methods=["POST"]),
    ])

    unauthorized = (b'Bearer error="invalid_token", resource_metadata="'
                    + f"{base}/.well-known/oauth-protected-resource/mcp".encode() + b'", scope="mcp:tools"')

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            return await mcp_app(scope, receive, send)
        if scope["type"] == "http" and scope["path"].startswith("/mcp"):
            auth = dict(scope.get("headers") or []).get(b"authorization", b"").decode()
            state.mcp_auth_headers.append(auth)
            tok = auth[7:] if auth.lower().startswith("bearer ") else ""
            if not tok or state.access.get(tok, 0) < time.time():
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"www-authenticate", unauthorized),
                                        (b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": b'{"error":"invalid_token"}'})
                return
            return await mcp_app(scope, receive, send)
        return await oauth_app(scope, receive, send)

    return app


def auto_approve(state: FakeAuthState, authorize_url: str) -> str:
    """Simulate the user pressing Allow: returns the redirect (callback) URL."""
    params = {k: v[0] for k, v in parse_qs(urlsplit(authorize_url).query).items()}
    client = state.clients.get(params.get("client_id", ""))
    assert client is not None, "authorize URL names an unregistered client"
    assert params.get("redirect_uri") in client["redirect_uris"]
    assert params.get("code_challenge_method") == "S256"
    assert params.get("resource") == state.resource, params.get("resource")
    code = "code_" + secrets.token_urlsafe(16)
    state.codes[code] = {"client_id": params["client_id"], "redirect_uri": params["redirect_uri"],
                         "challenge": params["code_challenge"], "resource": params["resource"],
                         "exp": time.time() + 120}
    query = {"code": code, "state": params.get("state", "")}
    if state.send_iss:
        query["iss"] = state.base_url.rstrip("/")
    return f"{params['redirect_uri']}?{urlencode(query)}"


def main() -> None:  # pragma: no cover - manual demo server
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--access-ttl", type=int, default=3600)
    args = parser.parse_args()
    state = FakeAuthState(base_url=f"http://127.0.0.1:{args.port}", access_ttl=args.access_ttl)
    print(f"Fake OAuth MCP server: {state.resource}")
    uvicorn.run(make_app(state), host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()
