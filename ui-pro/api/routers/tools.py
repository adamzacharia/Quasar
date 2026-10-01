"""Per-user MCP-server registry endpoints, plus the OAuth sign-in callback.

Connect flow (POST /api/mcp-servers): the URL is probed first. A server that
uses OAuth answers ``{"status": "needs_auth", "authorize_url": ...}`` and the
browser opens that URL in a popup; the provider redirects to
GET /api/mcp/oauth/callback, which identifies the user ONLY from the
single-use server-side ``state`` (never the Quasar session cookie: the
frontend and API are different sites in production). A server that wants an
API key answers ``needs_api_key``; anything else is saved and connected.
"""

import asyncio
import html
import json
import secrets
from typing import Optional
from urllib.parse import quote, urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from api.cors_origins import default_frontend_origin, is_allowed_origin
from api.deps import get_current_user
from api.models import MCPServerRequest

router = APIRouter()

_KEY_HEADERS = ("authorization", "x-api-key", "api-key")


def _pool():
    from services.user_mcp import get_user_mcp_pool

    return get_user_mcp_pool()


def _scrub(text) -> str:
    from services.mcp_oauth import scrub

    return scrub(text)


def _return_origin(request: Request) -> str:
    origin = (request.headers.get("origin") or "").rstrip("/")
    return origin if origin and is_allowed_origin(origin) else default_frontend_origin()


def _derive_name(url: str) -> str:
    """"https://www.monocrawl.com/mcp" -> "Monocrawl"."""
    host = (urlsplit(url).hostname or "").lower()
    labels = [p for p in host.split(".") if p and p not in ("www", "mcp", "api")]
    core = labels[-2] if len(labels) >= 2 else (labels[0] if labels else "server")
    return core[:1].upper() + core[1:]


def _admit(user_id: str) -> None:
    from services.mcp_oauth_store import ConnectRateLimited, MCPOAuthStore

    try:
        MCPOAuthStore().admit_connect(user_id)
    except ConnectRateLimited as exc:
        raise HTTPException(status_code=429, detail=str(exc))


def _oauth_summary(user_id: str, name: str) -> dict:
    from services.mcp_oauth_store import MCPOAuthStore

    rec = MCPOAuthStore().get(user_id, name)
    return {"signed_in": bool(rec and rec.access_token),
            "reason": rec.status_reason if rec else "Sign in to finish connecting this server."}


def _public_view(srv: dict, live: Optional[dict], oauth: Optional[dict]) -> dict:
    from services.mcp_server_service import mask_server

    out = mask_server(srv)
    out["auth"] = srv.get("auth") or "none"
    if live is not None:
        out["status"] = live
    elif oauth is not None and not oauth["signed_in"]:
        out["status"] = {"connected": False, "state": "needs_auth", "tools": [],
                         "error": oauth["reason"], "transport": None}
    return out


async def _start_oauth(request: Request, user_id: str, name: str, url: str, probe_result) -> str:
    from services.mcp_oauth import public_api_base, redirect_uri_for, start_sign_in

    base = public_api_base(str(request.base_url))
    return await start_sign_in(
        user_id=user_id, server=name, server_url=url, probe_result=probe_result,
        redirect_uri=redirect_uri_for(base), return_origin=_return_origin(request))


# ── MCP Server Endpoints ─────────────────────────────────────
@router.get("/api/mcp-servers")
async def list_mcp_servers(current_user: dict = Depends(get_current_user)):
    from services.mcp_server_service import MCPServerService

    user_id = current_user["sub"]
    servers = await asyncio.to_thread(MCPServerService().load_servers, user_id)
    # Last known connection state (no new connects here). Secret values
    # (tokens in env / headers) never travel back to the browser.
    status = _pool().status(user_id)
    out = []
    for srv in servers:
        if not isinstance(srv, dict):
            continue
        oauth = None
        if (srv.get("auth") or "none") == "oauth":
            oauth = await asyncio.to_thread(_oauth_summary, user_id, srv["name"])
        out.append(_public_view(srv, status.get(srv.get("name")), oauth))
    return out


@router.post("/api/mcp-servers/{name}/test")
async def test_mcp_server(name: str, current_user: dict = Depends(get_current_user)):
    """Connect to a saved server now and report tools or the real error."""
    user_id = current_user["sub"]
    _admit(user_id)
    return await asyncio.to_thread(_pool().test, user_id, name)


@router.post("/api/mcp-servers")
async def save_mcp_server(req: MCPServerRequest, request: Request,
                          current_user: dict = Depends(get_current_user)):
    """Detect what the server needs, then save and connect (or start sign-in)."""
    from services.mcp_oauth import OAuthSetupError, probe, revoke_and_forget
    from services.mcp_server_service import (MCPServerConfig, MCPServerService, mask_server,
                                             validate_server_config)
    from services.mcp_url_policy import check_mcp_url

    user_id = current_user["sub"]
    svc = MCPServerService()
    payload = req.model_dump()
    if not (payload.get("name") or "").strip() and payload.get("url"):
        payload["name"] = _derive_name(payload["url"])
    try:
        config = validate_server_config(MCPServerConfig(**payload))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    _admit(user_id)

    previous = await asyncio.to_thread(svc.get_server, user_id, config.name)
    is_http = config.transport in ("http", "streamable_http")
    if is_http:
        ok, reason = await asyncio.to_thread(check_mcp_url, config.url)
        if not ok:
            raise HTTPException(status_code=400, detail=reason)
    has_key = any(str(k).strip().lower() in _KEY_HEADERS for k in (config.headers or {}))

    if is_http and not has_key:
        result = await probe(config.url)
        if result.kind == "blocked":
            raise HTTPException(status_code=400, detail=result.message)
        if result.kind == "api_key":
            # Nothing is saved: the form asks for the key and submits again.
            return {"status": "needs_api_key", "message": result.message,
                    "server": {"name": config.name, "url": config.url, "transport": config.transport}}
        if result.kind == "oauth":
            try:
                srv = await asyncio.to_thread(
                    lambda: svc.save_server(user_id, config, auth="oauth"))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            _pool().forget(user_id, config.name)
            view = {**mask_server(srv), "auth": "oauth"}
            try:
                authorize_url = await _start_oauth(request, user_id, config.name, config.url, result)
            except OAuthSetupError as exc:
                return {"status": "error", "message": _scrub(exc), "server": view}
            return {"status": "needs_auth", "authorize_url": authorize_url, "server": view,
                    "message": "Sign in to finish connecting."}

    try:
        srv = await asyncio.to_thread(lambda: svc.save_server(user_id, config, auth="none"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    if previous and previous.get("auth") == "oauth":
        # Switched from OAuth to a key / no auth: the old sign-in is dead weight.
        await revoke_and_forget(user_id, config.name)
    # "Save & Connect" really connects: the result tells the user at once
    # whether the server answered and which tools it offers.
    result = await asyncio.to_thread(_pool().test, user_id, srv["name"])
    state = result.get("state") or ("connected" if result.get("connected") else "error")
    return {"status": state, "server": {**mask_server(srv), "auth": "none"}, "connection": result}


@router.post("/api/mcp-servers/{name}/reconnect")
async def reconnect_mcp_server(name: str, request: Request,
                               current_user: dict = Depends(get_current_user)):
    """OAuth servers: start a fresh sign-in. Others: reconnect now."""
    from services.mcp_oauth import OAuthSetupError, probe
    from services.mcp_server_service import MCPServerService

    user_id = current_user["sub"]
    _admit(user_id)
    srv = await asyncio.to_thread(MCPServerService().get_server, user_id, name)
    if srv is None:
        raise HTTPException(status_code=404, detail="Server not found")
    if srv.get("auth") != "oauth":
        result = await asyncio.to_thread(_pool().test, user_id, name)
        return {"status": result.get("state") or "error", "connection": result}
    result = await probe(srv["url"])
    if result.kind != "oauth":
        if result.kind in ("blocked", "unreachable"):
            return {"status": "error", "message": result.message}
        # The server stopped asking for sign-in; just connect.
        conn = await asyncio.to_thread(_pool().test, user_id, name)
        return {"status": conn.get("state") or "error", "connection": conn}
    try:
        authorize_url = await _start_oauth(request, user_id, name, srv["url"], result)
    except OAuthSetupError as exc:
        return {"status": "error", "message": _scrub(exc)}
    return {"status": "needs_auth", "authorize_url": authorize_url}


@router.post("/api/mcp-servers/{name}/disconnect")
async def disconnect_mcp_server(name: str, current_user: dict = Depends(get_current_user)):
    """Sign out of an OAuth server (revoking the tokens when the server
    supports it) but keep it in the list for a later Reconnect."""
    from services.mcp_oauth import revoke_and_forget
    from services.mcp_server_service import MCPServerService

    user_id = current_user["sub"]
    srv = await asyncio.to_thread(MCPServerService().get_server, user_id, name)
    if srv is None:
        raise HTTPException(status_code=404, detail="Server not found")
    if srv.get("auth") != "oauth":
        raise HTTPException(status_code=400, detail="This server does not use sign-in.")
    revoked = await revoke_and_forget(user_id, name, keep_registration=True)
    _pool().forget(user_id, name)
    return {"status": "needs_auth", "revoked": revoked}


@router.delete("/api/mcp-servers/{name}")
async def delete_mcp_server(name: str, current_user: dict = Depends(get_current_user)):
    from services.mcp_oauth import revoke_and_forget
    from services.mcp_server_service import MCPServerService

    user_id = current_user["sub"]
    srv = await asyncio.to_thread(MCPServerService().get_server, user_id, name)
    if srv is None:
        raise HTTPException(status_code=404, detail="Server not found")
    if srv.get("auth") == "oauth":
        await revoke_and_forget(user_id, name)
    await asyncio.to_thread(MCPServerService().delete_server, user_id, name)
    _pool().forget(user_id, name)
    return {"status": "success"}


# ── OAuth callback (no Quasar auth: the state row identifies the user) ──────

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quasar sign-in</title>
<style>
:root{{color-scheme:light dark;--bg:#f7f7f5;--fg:#1c1c1a;--muted:#6b6b66;--ok:#15803d;--err:#b91c1c}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161615;--fg:#ececea;--muted:#9a9a94;--ok:#4ade80;--err:#f87171}}}}
body{{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:420px;padding:24px;text-align:center}} h1{{font-size:17px;margin:0 0 8px;color:var(--{tone})}}
p{{margin:0 0 16px;color:var(--muted)}} a{{color:inherit}}
</style></head><body><main>
<h1>{title}</h1><p>{body}</p><p><a id="back" href="{back}">Return to Quasar</a></p>
</main>
<script nonce="{nonce}">
(function () {{
  var msg = {payload};
  var origin = {origin};
  var back = {back_js};
  try {{ history.replaceState(null, "", location.pathname); }} catch (e) {{}}
  var sent = false;
  try {{
    if (window.opener && !window.opener.closed) {{ window.opener.postMessage(msg, origin); sent = true; }}
  }} catch (e) {{}}
  if (sent) {{ setTimeout(function () {{ window.close(); }}, 400); return; }}
  setTimeout(function () {{ try {{ window.close(); }} catch (e) {{}} location.replace(back); }}, 1500);
}})();
</script></body></html>"""


def _js(value) -> str:
    # JSON inside a <script>: escape "<" so no value can close the tag.
    return json.dumps(value).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


@router.get("/api/mcp/oauth/callback", include_in_schema=False)
async def mcp_oauth_callback(request: Request, code: Optional[str] = None, state: Optional[str] = None,
                             error: Optional[str] = None, error_description: Optional[str] = None,
                             iss: Optional[str] = None):
    from services.mcp_oauth import finish_sign_in

    result = await finish_sign_in(state=state, code=code, error=error,
                                  error_description=error_description, iss=iss)
    message = result.message
    if result.ok:
        conn = await asyncio.to_thread(_pool().test, result.user_id, result.server)
        if conn.get("connected"):
            n = len(conn.get("tools") or [])
            message = f"Connected to {result.server}: {n} tool{'s' if n != 1 else ''} ready for your chats."
        else:
            message = f"Signed in, but {result.server} did not connect yet: {_scrub(conn.get('error'))}"
    origin = result.return_origin if result.return_origin and is_allowed_origin(result.return_origin) \
        else default_frontend_origin()
    flag = "connected" if result.ok else "failed"
    back = f"{origin}/?mcp_oauth={flag}" + (f"&mcp_server={quote(result.server)}" if result.server else "")
    payload = {"type": "quasar-mcp-oauth", "server": result.server, "ok": result.ok,
               "error": None if result.ok else _scrub(message)}
    nonce = secrets.token_urlsafe(16)
    page = _PAGE.format(
        tone="ok" if result.ok else "err",
        title=html.escape("You're connected" if result.ok else "Sign-in did not finish"),
        body=html.escape(_scrub(message)), back=html.escape(back, quote=True), nonce=nonce,
        payload=_js(payload), origin=_js(origin), back_js=_js(back),
    )
    return HTMLResponse(page, status_code=200, headers={
        "Content-Security-Policy": f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'unsafe-inline'; "
                                   "base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
        "Cache-Control": "no-store",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
    })
