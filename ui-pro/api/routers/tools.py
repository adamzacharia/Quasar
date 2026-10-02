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
import logging
import secrets
from typing import Optional
from urllib.parse import quote, urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from api.cors_origins import default_frontend_origin, is_allowed_origin
from api.deps import get_current_user
from api.models import MCPServerRequest
from services.secret_redaction import redact_url

router = APIRouter()

_KEY_HEADERS = ("authorization", "x-api-key", "api-key")
CALLBACK_ROUTE = "/api/mcp/oauth/callback"


class _CallbackQueryFilter(logging.Filter):
    """Uvicorn's access log prints the path WITH its query string, which for
    the OAuth callback carries the authorization code and state (guard
    CX-01). Drop the query for that one route; every other line is untouched."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str) \
                and args[2].startswith(CALLBACK_ROUTE + "?"):
            record.args = args[:2] + (CALLBACK_ROUTE + "?[redacted]",) + args[3:]
        return True


def _install_access_log_filter() -> None:
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, _CallbackQueryFilter) for f in access.filters):
        access.addFilter(_CallbackQueryFilter())


_install_access_log_filter()


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
    signed_in = bool(rec and rec.access_token)
    return {"signed_in": signed_in,
            # Change only when a sign-in completes (not on refresh): the
            # Settings popup waiter watches them (frontend guard CX-06/20).
            "signed_in_at": rec.signed_in_at if signed_in else None,
            "signed_in_attempt": rec.signed_in_attempt if signed_in else None,
            "reason": rec.status_reason if rec else "Sign in to finish connecting this server."}


def _attempt_of(authorize_url: str) -> Optional[str]:
    """The public attempt id for the state inside an authorize URL."""
    from urllib.parse import parse_qs

    from services.mcp_oauth_store import attempt_id

    state = (parse_qs(urlsplit(authorize_url).query).get("state") or [""])[0]
    return attempt_id(state) if state else None


def _public_view(srv: dict, live: Optional[dict], oauth: Optional[dict]) -> dict:
    from services.mcp_server_service import mask_server

    out = mask_server(srv)
    out.pop("revision", None)
    out["auth"] = srv.get("auth") or "none"
    if oauth is not None:
        out["oauth"] = {"signed_in": oauth["signed_in"], "signed_in_at": oauth["signed_in_at"],
                        "signed_in_attempt": oauth["signed_in_attempt"]}
    if live is not None:
        out["status"] = live
    elif oauth is not None and not oauth["signed_in"]:
        out["status"] = {"connected": False, "state": "needs_auth", "tools": [],
                         "error": oauth["reason"], "transport": None}
    return out


# Mutations of one (user, server) run one at a time in this process, so two
# overlapping saves / reconnects cannot interleave their save-test-revoke
# steps (backend guard CX-09). Across processes the row revision check in
# _save_plain still keeps an older snapshot from overwriting a newer save.
_SERVER_LOCKS: dict = {}


def _server_lock(user_id: str, name: str) -> asyncio.Lock:
    key = (user_id, name)
    lock = _SERVER_LOCKS.get(key)
    if lock is None:
        if len(_SERVER_LOCKS) > 2048:
            for k in [k for k, v in _SERVER_LOCKS.items() if not v.locked()][:1024]:
                _SERVER_LOCKS.pop(k, None)
        lock = _SERVER_LOCKS[key] = asyncio.Lock()
    return lock


def _not_saved(name: str, url: Optional[str], transport: str, message: str, status: str = "error") -> dict:
    return {"status": status, "saved": False, "message": message,
            "server": {"name": name, "url": redact_url(url) if url else url, "transport": transport}}


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


@router.get("/api/mcp-registry/search")
async def search_mcp_registry(q: Optional[str] = None, topic: Optional[str] = None, cursor: Optional[str] = None,
                              current_user: dict = Depends(get_current_user)):
    """Hosted servers from the official MCP Registry (unvetted listings).

    ``q`` is a free-text name search (with ``cursor`` for the next page);
    ``topic=astronomy`` merges several astronomy searches. Results are data
    only: connecting one goes through POST /api/mcp-servers like a pasted URL.
    """
    from services import mcp_registry as reg

    query = (q or "").strip()
    if topic is not None:
        if topic not in reg.TOPICS:
            raise HTTPException(status_code=400, detail=f"Unknown topic. Use one of: {', '.join(reg.TOPICS)}.")
        if query or cursor:
            raise HTTPException(status_code=400, detail="Send either q or topic, not both.")
    elif not query:
        raise HTTPException(status_code=400, detail="Type something to search for.")
    if len(query) > reg.MAX_QUERY_LEN:
        raise HTTPException(status_code=400, detail=f"Keep the search under {reg.MAX_QUERY_LEN} characters.")
    if cursor is not None and (not cursor or len(cursor) > reg.MAX_CURSOR_LEN):
        raise HTTPException(status_code=400, detail="That page link is not valid. Search again.")
    try:
        reg.admit_search(current_user["sub"])
    except reg.SearchRateLimited as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    try:
        if topic is not None:
            return await reg.search_topic(topic)
        return await reg.search_registry(query, cursor)
    except reg.RegistryUnavailable as exc:
        logging.getLogger(__name__).warning("MCP registry search failed: %s", exc)
        raise HTTPException(status_code=502, detail="The MCP Registry did not answer. Try again in a minute.")


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
    from services.mcp_server_service import MCPServerConfig, MCPServerService, validate_server_config

    user_id = current_user["sub"]
    svc = MCPServerService()
    payload = req.model_dump()
    replace = bool(payload.pop("replace", True))
    if not (payload.get("name") or "").strip() and payload.get("url"):
        payload["name"] = _derive_name(payload["url"])
    try:
        config = validate_server_config(MCPServerConfig(**payload))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    _admit(user_id)
    async with _server_lock(user_id, config.name):
        return await _save_locked(request, user_id, svc, config, replace)


async def _save_locked(request: Request, user_id: str, svc, config, replace: bool = True):
    from services.mcp_oauth import OAuthSetupError, probe
    from services.mcp_server_service import mask_server
    from services.mcp_url_policy import check_mcp_url

    previous = await asyncio.to_thread(svc.get_server, user_id, config.name)
    if previous and not replace and (previous.get("url") or "") != (config.url or ""):
        # A name the form derived must never overwrite a different server
        # (frontend guard CX-03); the form picks another name and retries.
        raise HTTPException(status_code=409, detail=f"A different server is already saved as {config.name}.")
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
        if result.kind == "unreachable":
            # Never save (or overwrite and revoke a working sign-in) on the
            # strength of a failed probe: an outage says nothing about what
            # the server needs (guard CX-04).
            return _not_saved(config.name, config.url, config.transport,
                              f"Could not reach the server: {result.message} Nothing was changed.")
        if result.kind == "api_key":
            # Nothing is saved: the form asks for the key and submits again.
            return _not_saved(config.name, config.url, config.transport, result.message, status="needs_api_key")
        if result.kind == "open" and previous and previous.get("auth") == "oauth" and not result.answered_ok:
            # Replacing a signed-in server with a plain one revokes its
            # sign-in: only on a completed anonymous MCP handshake.
            return _not_saved(config.name, config.url, config.transport,
                              f"The server answered HTTP {result.status_code} without a working MCP handshake, "
                              "so it is unclear whether it still needs sign-in. Nothing was changed.")
        if result.kind == "oauth":
            try:
                srv = await asyncio.to_thread(
                    lambda: svc.save_server(user_id, config, auth="oauth"))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            _pool().forget(user_id, config.name)
            view = {**mask_server(srv), "auth": "oauth"}
            view.pop("revision", None)
            try:
                authorize_url = await _start_oauth(request, user_id, config.name, config.url, result)
            except OAuthSetupError as exc:
                return {"status": "error", "saved": True, "message": _scrub(exc), "server": view}
            return {"status": "needs_auth", "saved": True, "authorize_url": authorize_url,
                    "attempt": _attempt_of(authorize_url), "server": view,
                    "message": "Sign in to finish connecting."}

    try:
        srv, result, kept = await _save_plain(user_id, config, previous)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    if kept:
        view = {**mask_server(previous), "auth": "oauth"}
        view.pop("revision", None)
        return {"status": "error", "saved": False, "server": view, "connection": result,
                "message": f"{config.name} did not connect that way ({_scrub(result.get('error'))}), "
                           "so the existing sign-in was kept. Nothing was changed."}
    # "Save & Connect" really connects: the result tells the user at once
    # whether the server answered and which tools it offers.
    state = result.get("state") or ("connected" if result.get("connected") else "error")
    view = {**mask_server(srv), "auth": "none"}
    view.pop("revision", None)
    return {"status": state, "saved": True, "server": view, "connection": result}


async def _save_plain(user_id: str, config, previous: Optional[dict]):
    """Save ``config`` as a plain (key or no-auth) server and connect it.

    When it replaces a signed-in OAuth server, the new config must first
    connect on a throwaway connection; only then is it written (compare-and-
    set) and the old sign-in revoked. If it does not connect, nothing is
    written at all (guard CX-04/07/08/09: no probe heuristic or overlapping
    request can destroy a working sign-in). Returns
    (saved server, connection result, kept_previous)."""
    from services.mcp_oauth import revoke_tokens
    from services.mcp_oauth_store import MCPOAuthStore
    from services.mcp_server_service import MCPServerService

    svc = MCPServerService()
    if not (previous and previous.get("auth") == "oauth"):
        srv = await asyncio.to_thread(lambda: svc.save_server(user_id, config, auth="none"))
        _pool().forget(user_id, config.name)
        result = await asyncio.to_thread(_pool().test, user_id, srv["name"])
        return srv, result, False
    # Replacing a signed-in server (guard CX-09). Nothing is written until the
    # plain config has connected on a throwaway, unregistered connection, so
    # chats never see an unverified config and a failure changes nothing.
    result = await asyncio.to_thread(_pool().test_config, user_id, config.model_dump())
    if not result.get("connected"):
        return previous, result, True
    store = MCPOAuthStore()
    snapshot = await asyncio.to_thread(store.get, user_id, config.name)
    # One compare-and-set write: only if the row is still the version this
    # request started from. A save from another worker in between wins.
    if not await asyncio.to_thread(lambda: svc.restore_if_revision(user_id, previous.get("revision"), config,
                                                                   auth="none")):
        return previous, {"connected": False, "state": "error", "tools": [],
                          "error": "the server was changed by another request meanwhile"}, True
    _pool().forget(user_id, config.name)
    srv = await asyncio.to_thread(svc.get_server, user_id, config.name) or {**config.model_dump(), "auth": "none"}
    # Revoke the OLD tokens (a snapshot), then drop this server's OAuth data
    # only while the server row is still plain: a sign-in begun meanwhile has
    # already flipped it back to "oauth" and keeps its row and states, and an
    # older popup's callback can no longer store tokens (set_tokens checks
    # the row too).
    await revoke_tokens(snapshot)
    await asyncio.to_thread(store.delete_if_server_plain, user_id, config.name)
    result = await asyncio.to_thread(_pool().test, user_id, config.name)
    return srv, result, False


@router.post("/api/mcp-servers/{name}/reconnect")
async def reconnect_mcp_server(name: str, request: Request,
                               current_user: dict = Depends(get_current_user)):
    """OAuth servers: start a fresh sign-in. Others: reconnect now."""
    user_id = current_user["sub"]
    _admit(user_id)
    async with _server_lock(user_id, name):
        return await _reconnect_locked(request, user_id, name)


async def _reconnect_locked(request: Request, user_id: str, name: str):
    from services.mcp_oauth import OAuthSetupError, probe
    from services.mcp_server_service import MCPServerService

    srv = await asyncio.to_thread(MCPServerService().get_server, user_id, name)
    if srv is None:
        raise HTTPException(status_code=404, detail="Server not found")
    if srv.get("auth") != "oauth":
        result = await asyncio.to_thread(_pool().test, user_id, name)
        return {"status": result.get("state") or "error", "connection": result}
    result = await probe(srv["url"])
    if result.kind in ("blocked", "unreachable"):
        return {"status": "error", "message": result.message}
    if result.kind == "api_key":
        return {"status": "needs_api_key", "message": f"{result.message} It no longer offers sign-in; "
                "remove it and add it again with a key."}
    if result.kind == "open" and not result.answered_ok:
        return {"status": "error", "message": f"The server answered HTTP {result.status_code} without a working "
                "MCP handshake instead of asking for sign-in. Nothing was changed; try again later."}
    if result.kind == "open":
        # The server stopped asking for sign-in (it completed an anonymous
        # initialize): try it as a plain server; the old sign-in is dropped
        # only if that connection works (guard CX-07/08).
        from services.mcp_server_service import MCPServerConfig

        cfg = MCPServerConfig(**{k: srv[k] for k in ("name", "transport", "url", "command", "args", "env", "headers")
                                 if k in srv})
        _, conn, kept = await _save_plain(user_id, cfg, srv)
        if kept:
            return {"status": "error", "connection": conn,
                    "message": "The server no longer asks for sign-in but did not connect without it either. "
                               "Nothing was changed."}
        return {"status": conn.get("state") or "error", "connection": conn}
    # Claim the server row (compare-and-set at the version read above, still
    # auth="oauth") BEFORE any registration or state is written: if another
    # worker switched it to plain meanwhile, the claim fails and no sign-in
    # starts; if ours lands first, that worker's own CAS fails instead. The
    # server row and the OAuth row can never disagree (backend guard CX-09).
    from services.mcp_server_service import MCPServerConfig

    cfg = MCPServerConfig(**{k: srv[k] for k in ("name", "transport", "url", "command", "args", "env", "headers")
                             if k in srv})
    claimed = await asyncio.to_thread(
        lambda: MCPServerService().restore_if_revision(user_id, srv.get("revision"), cfg, auth="oauth"))
    if not claimed:
        return {"status": "error", "message": "This server was changed by another request meanwhile. "
                "Nothing was started; reload the list and try again."}
    try:
        authorize_url = await _start_oauth(request, user_id, name, srv["url"], result)
    except OAuthSetupError as exc:
        return {"status": "error", "message": _scrub(exc)}
    return {"status": "needs_auth", "authorize_url": authorize_url, "attempt": _attempt_of(authorize_url)}


@router.post("/api/mcp-servers/{name}/disconnect")
async def disconnect_mcp_server(name: str, current_user: dict = Depends(get_current_user)):
    """Sign out of an OAuth server (revoking the tokens when the server
    supports it) but keep it in the list for a later Reconnect."""
    from services.mcp_oauth import revoke_and_forget
    from services.mcp_server_service import MCPServerService

    user_id = current_user["sub"]
    async with _server_lock(user_id, name):
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
    async with _server_lock(user_id, name):
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
    # ok = the sign-in finished and the tokens are stored. Whether the MCP
    # server then connected is reported separately: `error` is set (and the
    # redirect flag says signed_in, not connected) when it did not.
    connected = False
    if result.ok:
        conn = await asyncio.to_thread(_pool().test, result.user_id, result.server)
        connected = bool(conn.get("connected"))
        if connected:
            n = len(conn.get("tools") or [])
            message = f"Connected to {result.server}: {n} tool{'s' if n != 1 else ''} ready for your chats."
        else:
            message = f"Signed in, but {result.server} did not connect yet: {_scrub(conn.get('error'))}"
    origin = result.return_origin if result.return_origin and is_allowed_origin(result.return_origin) \
        else default_frontend_origin()
    flag = "connected" if connected else ("signed_in" if result.ok else "failed")
    back = f"{origin}/?mcp_oauth={flag}" + (f"&mcp_server={quote(result.server)}" if result.server else "")
    payload = {"type": "quasar-mcp-oauth", "server": result.server, "ok": result.ok,
               "error": None if connected else _scrub(message)}
    nonce = secrets.token_urlsafe(16)
    page = _PAGE.format(
        tone="ok" if connected else "err",
        title=html.escape("You're connected" if connected else
                          ("Signed in, not connected yet" if result.ok else "Sign-in did not finish")),
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
