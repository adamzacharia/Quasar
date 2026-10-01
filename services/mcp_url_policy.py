"""Which URLs the MCP connector may fetch (SSRF guard + https policy).

User MCP servers and OAuth discovery make the backend fetch URLs a user (or
a remote server's metadata) chose. Without a check a user could point the
hosted backend at loopback, a private range or the cloud metadata address
and read the reply back through the connect error. Policy:

* http or https only, a host is required, no credentials in the URL.
* Loopback (localhost, 127.0.0.0/8, ::1) only outside production, so local
  development and the in-process test servers keep working.
* Everything else must resolve only to globally routable addresses
  (services/vo_url_guard.py, without its VO-only host allowlist).
* ``require_https``: OAuth endpoints (authorization server, token,
  registration, revocation) and OAuth-protected MCP servers must use https
  unless they are loopback in development.

The check runs again on every outgoing request of a connector client
(``guarded_client_factory``), so redirect hops are covered too. Known
residual risk, shared with the VO guard: DNS rebinding between the check and
the connect.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
from typing import Optional, Tuple
from urllib.parse import urlsplit

import httpx


class UnsafeMcpUrl(ValueError):
    """The URL is not an allowed target for the MCP connector."""


def _environment() -> str:
    return (os.environ.get("QUASAR_ENV") or os.environ.get("APP_ENV")
            or os.environ.get("ENVIRONMENT") or "development").strip().lower()


def is_production() -> bool:
    return _environment() in ("production", "prod")


def _is_loopback(host: str) -> bool:
    host = host.strip("[]").lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_mcp_url(url: str, *, label: str = "The server URL",
                  require_https: bool = False, resolve: bool = True) -> Tuple[bool, str]:
    """(ok, user-facing reason)."""
    try:
        parts = urlsplit(str(url or "").strip())
    except ValueError:
        return False, f"{label} is not a valid URL."
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        return False, f"{label} must start with https://"
    if parts.username or parts.password:
        return False, f"{label} must not contain a user name or password."
    host = (parts.hostname or "").strip("[]").lower()
    if not host:
        return False, f"{label} has no host name."
    if _is_loopback(host):
        if is_production():
            return False, f"{label} points at this server itself, which is not allowed here."
        return True, ""
    if require_https and scheme != "https":
        return False, f"{label} must use https."
    from services.vo_url_guard import check_public_url

    ok, _reason = check_public_url(url, "url", resolve=resolve, use_allowlist=False)
    if not ok:
        # Same text for "private address" and "does not resolve", like the
        # VO guard, so the connector cannot be used to map a private network.
        return False, f"{label} does not point at a public internet address."
    return True, ""


def ensure_mcp_url(url: str, **kw) -> str:
    ok, reason = check_mcp_url(url, **kw)
    if not ok:
        raise UnsafeMcpUrl(reason)
    return str(url).strip()


def request_guard(require_https: bool = False):
    """An httpx async ``request`` event hook that refuses unsafe targets. It
    runs for every request the client sends, redirect hops included, and
    resolves the host every time: a cached "public" verdict would let a
    rebinding host point later requests at a private address (guard CX-02).
    The OS resolver cache keeps this cheap."""

    async def hook(request: httpx.Request) -> None:
        ok, reason = await asyncio.to_thread(check_mcp_url, str(request.url), require_https=require_https)
        if not ok:
            raise UnsafeMcpUrl(reason)

    return hook


def guarded_client_factory(require_https: bool = False):
    """An MCP SDK ``httpx_client_factory`` with the guard on every request."""
    from mcp.shared._httpx_utils import MCP_DEFAULT_SSE_READ_TIMEOUT, MCP_DEFAULT_TIMEOUT

    def factory(headers: Optional[dict] = None, timeout: Optional[httpx.Timeout] = None,
                auth: Optional[httpx.Auth] = None) -> httpx.AsyncClient:
        kwargs = {
            "follow_redirects": True,
            "max_redirects": 5,
            "timeout": timeout or httpx.Timeout(MCP_DEFAULT_TIMEOUT, read=MCP_DEFAULT_SSE_READ_TIMEOUT),
            "event_hooks": {"request": [request_guard(require_https)]},
        }
        if headers is not None:
            kwargs["headers"] = headers
        if auth is not None:
            kwargs["auth"] = auth
        return httpx.AsyncClient(**kwargs)

    return factory


def guarded_async_client(*, require_https: bool = False, timeout: float = 15.0,
                         follow_redirects: bool = False) -> httpx.AsyncClient:
    """A plain client for OAuth discovery / token calls. Redirects are off by
    default: metadata and token endpoints answer directly."""
    return httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=follow_redirects,
        max_redirects=3,
        event_hooks={"request": [request_guard(require_https)]},
        headers={"User-Agent": "Quasar-MCP-Connector/1.0"},
    )


__all__ = [
    "UnsafeMcpUrl", "check_mcp_url", "ensure_mcp_url", "guarded_async_client",
    "guarded_client_factory", "is_production", "request_guard",
]
