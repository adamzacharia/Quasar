"""Outbound URL guard for the generic VO tools (SSRF protection).

The ``vo_*`` tools fetch URLs the model supplies (``access_url``) or that a
remote service hands back (async job URLs, redirect targets). Those requests
leave from inside Quasar's network, so an unguarded fetch could reach
loopback, private ranges or the cloud metadata address. Idea taken from
MANNA's ``_url_guard.py`` (NSF-Simons CosmicAI, MIT); reimplemented here.

Policy, in order:

1. Scheme is http or https; a host is present; no credentials in the URL.
2. If ``VO_ALLOWED_HOSTS`` is set (comma-separated), the host must equal an
   entry or be a dot-subdomain of one. Unset allows any public host, because
   ``vo_find_services`` exists to reach archives Quasar has no profile for.
3. Every address the host resolves to must be globally routable. An IP
   literal is checked directly. DNS failure fails closed.

Any port is allowed: real TAP services run on :8080 and friends.

The guard is enforced per HTTP request in ``vo_registry._TimeoutHTTPSession``
(``send``), so every redirect hop and every server-supplied job URL is
checked, not only the first URL.

Known residual risk: DNS rebinding between this check and the connect. Closing
it needs a pinned-address transport, which is out of scope here.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from typing import List, Tuple
from urllib.parse import urlsplit

# Deliberately generic, and the SAME text for "resolves to a private
# address" and "does not resolve": distinguishing them (or echoing the
# address) would turn the guard into an internal-network scanner.
_BLOCKED = "{param} does not point at a public network address and will not be fetched."

_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_IPV4_COMPATIBLE = ipaddress.ip_network("::/96")
_SITE_LOCAL = ipaddress.ip_network("fec0::/10")


class UnsafeUrlError(ValueError):
    """The URL is not a safe outbound target for a VO request."""


def _resolve(host: str) -> List[str]:
    """All addresses ``host`` maps to. Module-level so tests can stub it."""
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return [str(info[4][0]).split("%")[0] for info in infos]


def _allowed_hosts() -> Tuple[str, ...]:
    raw = os.getenv("VO_ALLOWED_HOSTS", "")
    return tuple(h.strip().lower().rstrip(".") for h in raw.split(",") if h.strip())


def _allowlist_configured() -> bool:
    """VO_ALLOWED_HOSTS is set to something non-blank. A value that parses to
    no hostnames (" , ") then means "allow nothing", never "allow all"
    (guard CX-04)."""
    return bool(os.getenv("VO_ALLOWED_HOSTS", "").strip())


def _host_matches(host: str, allowed: Tuple[str, ...]) -> bool:
    # The leading dot matters: "eso.org" must not admit "notaneso.org".
    return any(host == entry or host.endswith("." + entry) for entry in allowed)


def _is_public(ip: "ipaddress.IPv4Address | ipaddress.IPv6Address") -> bool:
    if isinstance(ip, ipaddress.IPv6Address):
        # IPv6 forms that carry an IPv4 address: judge the embedded address.
        # (Python reports several of these as is_global, e.g. the NAT64 form
        # of the metadata address 64:ff9b::a9fe:a9fe.)
        if ip in _NAT64:
            return _is_public(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
        if ip.teredo:
            # (server, client): both halves are routing targets (guard CX-03).
            server, client = ip.teredo
            return _is_public(server) and _is_public(client)
        embedded = ip.ipv4_mapped or ip.sixtofour
        if embedded is not None:
            return _is_public(embedded)
        if ip in _IPV4_COMPATIBLE or ip in _SITE_LOCAL:
            return False
    return ip.is_global and not ip.is_multicast


def check_public_url(
    url: str, param: str = "access_url", *, resolve: bool = True
) -> Tuple[bool, str]:
    """(ok, reason). ``reason`` is model-facing and names ``param``.

    ``resolve=False`` skips DNS (syntax, allowlist, IP literals and local
    names only): the cheap early check used before a service object is
    built. The HTTP-layer check always resolves."""
    try:
        parts = urlsplit(str(url or "").strip())
    except ValueError:
        return False, f"{param} is not a valid URL."
    if parts.scheme.lower() not in ("http", "https"):
        return False, f"{param} must be an http(s) URL."
    if parts.username or parts.password:
        return False, f"{param} must not carry credentials."
    host = (parts.hostname or "").strip("[]").lower().rstrip(".")
    if not host:
        return False, f"{param} has no host."
    try:
        parts.port
    except ValueError:
        return False, f"{param} has an invalid port."

    allowed = _allowed_hosts()
    if _allowlist_configured() and not _host_matches(host, allowed):
        return False, f"{param} host {host!r} is not in this deployment's VO_ALLOWED_HOSTS."

    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        return False, _BLOCKED.format(param=param)
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        if not resolve:
            return True, ""
        try:
            addresses = [ipaddress.ip_address(a) for a in _resolve(host)]
        except (OSError, UnicodeError, ValueError):
            return False, _BLOCKED.format(param=param)
    # Every address must be public: a split-horizon name with one public and
    # one private record is still a pivot.
    if not addresses or not all(_is_public(ip) for ip in addresses):
        return False, _BLOCKED.format(param=param)
    return True, ""


def ensure_public_url(url: str, param: str = "access_url", *, resolve: bool = True) -> str:
    """Return the stripped URL, or raise :class:`UnsafeUrlError`."""
    ok, reason = check_public_url(url, param, resolve=resolve)
    if not ok:
        raise UnsafeUrlError(reason)
    return str(url).strip()


__all__ = ["UnsafeUrlError", "check_public_url", "ensure_public_url"]
