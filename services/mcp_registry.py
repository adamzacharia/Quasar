"""Search the official MCP Registry for hosted servers (Settings > MCP servers).

The registry (https://registry.modelcontextprotocol.io, API v0) lists servers
that their authors published; nobody vets them. Quasar can only connect to a
server that has a URL (``remotes``), not to npm / PyPI packages that run on
your own computer, so entries without a usable remote are dropped here.

Two kinds of search:

* free text: the registry's own ``search`` (it matches server NAMES only);
* the "astronomy" topic: several astronomy queries merged, then kept only
  when the title or description is actually about astronomy. A plain "astro"
  search is mostly astrology, so the topic also drops astrology on purpose.

Every result says when Quasar already has the same archive built in
(``BUILTIN_OVERLAPS``), so people do not add a third-party wrapper that
duplicates a hardened built-in tool. The registry is fetched from a fixed
https host only; result URLs are just data until the normal Connect flow
(with its SSRF guard) runs on one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

REGISTRY_BASE = "https://registry.modelcontextprotocol.io"
REGISTRY_SERVERS = f"{REGISTRY_BASE}/v0/servers"
SOURCE_LABEL = "registry.modelcontextprotocol.io"

PAGE_LIMIT = 100            # the registry's maximum page size
MAX_QUERY_LEN = 80
MAX_CURSOR_LEN = 512
MAX_RESPONSE_BYTES = 4_000_000
# Registry name searches took 0.3-15 s each on 2026-10-02 (it scans names).
FETCH_TIMEOUT_S = 25.0
# The astronomy topic: all its searches run together (25 parallel requests
# drew no rate limiting on 2026-10-02) under one overall deadline; searches
# still running then count as failed and the result is marked partial.
TOPIC_CONCURRENCY = 24
TOPIC_DEADLINE_S = 30.0
CACHE_TTL_S = 15 * 60
TOPIC_CACHE_TTL_S = 60 * 60  # a topic is ~22 registry requests; it changes slowly
PARTIAL_TOPIC_TTL_S = 2 * 60  # some searches failed: retry soon, but not on every click
CACHE_MAX = 256
SEARCHES_PER_MINUTE = 30

TOPICS = ("astronomy",)

# Name searches that find astronomy servers in the registry (checked against
# the live registry on 2026-10-02; "jwst", "hubble" and "astrophysics"
# matched nothing then). Results still pass ASTRONOMY_RE below.
ASTRONOMY_QUERIES = (
    "astronomy", "astro", "nasa", "arxiv", "simbad", "exoplanet", "horizons",
    "telescope", "ephemeris", "celestial", "eclipse", "asteroid", "comet", "meteor",
    "gaia", "spaceweather", "space", "sky", "star", "moon", "solar", "satellite",
)

# Matched against title + description + the last part of the name (never the
# publisher namespace: "io.github.AvatarGaia/..." is not about Gaia).
ASTRONOMY_RE = re.compile(
    r"astronom|astrophys|telescope|exoplanet|ephemer|celestial|eclipse|asteroid|\bcomets?\b|"
    r"fireball|near[- ]earth|solar system|night[- ]sky|dark[- ]sky|sky darkness|observing spot|"
    r"space weather|spaceweather|geomagnetic|\baurora|heliophys|solar (wind|flare|activity)|"
    r"moon ?phase|moonrise|\bplanets\b.*\b(rise|set|position)|rise/set|\bstar (catalog|position|map)|"
    r"\barxiv\b|\bsimbad\b|\bvizier\b|\bnasa\b|\bjpl\b|\besa gaia\b|\bgaia (star|catalog)|"
    r"hubble|\bjwst\b|james webb|\bmast\b|satellite (pass|orbit|track)|\btle\b|"
    r"astrophysics data system|extragalactic|galax(y|ies) (survey|catalog)|spectroscop|"
    r"\bastropy\b|radio astronomy|planetary (data|science)",
    re.IGNORECASE,
)

# Never an astronomy result, whatever else the text says.
ASTRONOMY_DENY_RE = re.compile(
    r"astrolog|horoscop|vedic|kundl|zodiac|natal chart|\bnatal\b|tarot|panchang|jyotish|\bbazi\b|"
    r"numerolog|synastry|tzolkin|muhurta|manglik|star wars|brawl stars|bitcoin|crypto|\bxrpl\b|"
    r"jewel|engraved|checkout|demand intelligence|competitor share",
    re.IGNORECASE,
)

# Archives and services Quasar already has as built-in tools. ``tools`` names
# real tools (tests/unit/test_mcp_registry.py checks each is defined in
# core/tool_registrations.py or capabilities/), so a label is never claimed
# without the tool behind it.
BUILTIN_OVERLAPS = (
    {"label": "SIMBAD", "pattern": r"\bsimbad\b", "tools": ("simbad_query", "resolve_target")},
    {"label": "NASA ADS", "pattern": r"nasa[- ]ads\b|astrophysics data system|adsabs",
     "tools": ("ads_search", "search_papers")},
    {"label": "arXiv search", "pattern": r"\barxiv\b", "tools": ("search_papers",)},
    {"label": "MAST", "pattern": r"\bmast\b", "tools": ("search_mast", "download_mast_data")},
    {"label": "NASA Exoplanet Archive", "pattern": r"exoplanet[- ]archive|exoplanetarchive",
     "tools": ("exoplanet_archive",)},
    {"label": "VizieR", "pattern": r"\bvizier\b", "tools": ("catalog_query", "catalog_find")},
    {"label": "NED", "pattern": r"extragalactic database|\bnasa[/ -]ned\b|\bned (database|archive)\b",
     "tools": ("ned_distance",)},
    {"label": "Gaia", "pattern": r"\besa[- ]gaia\b|\bgaia (dr\d|star|catalog|archive)",
     "tools": ("gaia_archive_query", "gaia_distance")},
    {"label": "IRSA", "pattern": r"\birsa\b", "tools": ("search_irsa",)},
    {"label": "HEASARC", "pattern": r"\bheasarc\b", "tools": ("heasarc_observations",)},
    {"label": "ESO archive", "pattern": r"\beso (archive|tap)\b|european southern observatory",
     "tools": ("search_eso_archive",)},
    {"label": "CADC", "pattern": r"\bcadc\b", "tools": ("search_cadc_archive",)},
    {"label": "ALMA archive", "pattern": r"\balma\b", "tools": ("query_alma_science_archive",)},
    {"label": "Astro Data Lab (SDSS, DESI, Legacy Survey)", "pattern": r"\bsdss\b|astro data lab|noirlab",
     "tools": ("datalab_sql_query",)},
    {"label": "ZTF", "pattern": r"\bztf\b", "tools": ("ztf_light_curve", "ztf_object")},
    {"label": "JPL Horizons", "pattern": r"jpl horizons|horizons[- ]nasa|\bhorizons\b.*ephemer|ephemer.*\bhorizons\b",
     "tools": ("solar_system_ephemeris",)},
    {"label": "Fermi", "pattern": r"\bfermi\b", "tools": ("fermi_lcr_lightcurve",)},
    {"label": "Splatalogue (spectral lines)", "pattern": r"splatalogue", "tools": ("search_spectral_lines",)},
)
_OVERLAP_RES = tuple((o["label"], re.compile(o["pattern"], re.IGNORECASE)) for o in BUILTIN_OVERLAPS)


class RegistryUnavailable(RuntimeError):
    """The registry did not answer usefully (network, status, or body)."""


class SearchRateLimited(RuntimeError):
    """Too many registry searches from one user in the last minute."""


# ── normalising one registry entry ───────────────────────────────────────────

def _short_name(name: str) -> str:
    return name.rsplit("/", 1)[-1] if name else ""


def _pretty_title(name: str) -> str:
    """"io.github.cyanheads/astronomy-mcp-server" -> "Astronomy" (fallback title)."""
    core = _short_name(name)
    core = re.sub(r"(^|[-_])(mcp|server)(?=$|[-_])", " ", core, flags=re.IGNORECASE)
    words = [w for w in re.split(r"[-_\s]+", core) if w]
    return " ".join(w[:1].upper() + w[1:] for w in words) or name


def _safe_remote_url(url) -> Optional[str]:
    if not isinstance(url, str):
        return None
    url = url.strip()
    if len(url) > 2048 or any(c.isspace() for c in url):
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        return None
    try:
        parts.port  # noqa: B018 - raises ValueError for ":bad" or ":99999" (guard CX-14)
    except ValueError:
        return None
    return url


# Anything the frontend paste parser (mcp-paste.js hasPlaceholder) would call
# a placeholder, so such an address is never offered for a direct connect.
# Same as looksLikeKey in ui-pro/src/lib/mcp-registry.js.
_KEY_NAME_RE = re.compile(r"authorization|api[-_]?key|token|secret|password", re.IGNORECASE)

# Kept identical to hasPlaceholder (both test suites run the same cases).
_PLACEHOLDER_RE = re.compile(r"\$\{[^}]*\}|\$[A-Z_][A-Z0-9_]*|<[^>]+>|\{[^{}]*\}|YOUR_|\.\.\.", re.IGNORECASE)


def _is_templated(url: str) -> bool:
    return bool(_PLACEHOLDER_RE.search(url.split("://", 1)[-1]))


def endpoint_key(url: str) -> str:
    """Scheme and host case-insensitive; path and query exact (they can be
    case-sensitive tenant ids); a trailing slash does not count."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    path = parts.path.rstrip("/")
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}{path}" + (f"?{parts.query}" if parts.query else "")


def _headers_of(remote: dict) -> list:
    out = []
    for h in remote.get("headers") or []:
        if not isinstance(h, dict) or not isinstance(h.get("name"), str) or not h["name"].strip():
            continue
        value = h.get("value") if isinstance(h.get("value"), str) else None
        out.append({
            "name": h["name"].strip()[:100],
            "required": bool(h.get("isRequired")),
            "secret": bool(h.get("isSecret")),
            "description": str(h.get("description") or "")[:300],
            "value_template": value[:200] if value else None,
        })
    return out


def overlaps_for(text: str) -> list:
    """Built-in Quasar archives this text describes (labels, in table order)."""
    return [label for label, rx in _OVERLAP_RES if rx.search(text or "")]


def _match_text(name: str, title: str, description: str) -> str:
    return f"{_short_name(name)} {title} {description}"


def normalize_entry(raw) -> Optional[dict]:
    """One registry list item -> a result Quasar can show and connect, or None.

    A malformed entry is skipped, never an error for the whole search
    (guard CX-08)."""
    try:
        return _normalize(raw)
    except Exception:  # noqa: BLE001 - unvetted third-party data
        logger.debug("skipping malformed MCP registry entry", exc_info=True)
        return None


def _normalize(raw) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    srv = raw.get("server") if isinstance(raw.get("server"), dict) else raw
    meta_root = raw.get("_meta") if isinstance(raw.get("_meta"), dict) else {}
    meta = meta_root.get("io.modelcontextprotocol.registry/official")
    meta = meta if isinstance(meta, dict) else {}
    name = srv.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    if str(meta.get("status") or "active").lower() not in ("active", ""):
        return None  # deprecated / deleted
    remote = None
    for r in srv.get("remotes") or []:
        if not isinstance(r, dict) or r.get("type") not in ("streamable-http", "sse"):
            continue
        url = _safe_remote_url(r.get("url"))
        if url:
            remote = dict(r, url=url)
            break
    if not remote:
        return None
    title = str(srv.get("title") or "").strip()[:120] or _pretty_title(name)
    description = str(srv.get("description") or "").strip()[:400]
    headers = _headers_of(remote)
    templated = _is_templated(remote["url"])
    repo = srv.get("repository") if isinstance(srv.get("repository"), dict) else {}
    website = _safe_remote_url(srv.get("websiteUrl"))
    repo_url = _safe_remote_url(repo.get("url"))
    return {
        "id": name.strip()[:200],
        "title": title,
        "description": description,
        "version": str(srv.get("version") or "")[:40],
        "url": remote["url"],
        "transport": "sse" if remote.get("type") == "sse" else "streamable-http",
        "headers": headers,
        # A required secret (or key-named) header; a fixed non-secret value
        # such as a version needs nothing from the user (guard CX-03).
        "needs_key": any(h["required"] and (h["secret"] or _KEY_NAME_RE.search(h["name"])) for h in headers),
        "templated": templated,
        "website": website,
        "repository": repo_url,
        "overlaps": overlaps_for(_match_text(name, title, description)),
        "updated_at": str(meta.get("updatedAt") or "")[:40] or None,
    }


def is_astronomy(item: dict) -> bool:
    text = _match_text(item.get("id", ""), item.get("title", ""), item.get("description", ""))
    return bool(ASTRONOMY_RE.search(text)) and not ASTRONOMY_DENY_RE.search(text)


def dedupe(items: list) -> list:
    """Keep the first entry per registry name and per URL."""
    seen_ids, seen_urls, out = set(), set(), []
    for it in items:
        url_key = endpoint_key(it["url"])
        if it["id"] in seen_ids or url_key in seen_urls:
            continue
        seen_ids.add(it["id"])
        seen_urls.add(url_key)
        out.append(it)
    return out


# ── fetching, cache, rate limit ──────────────────────────────────────────────

_cache: dict = {}
_cache_lock = threading.Lock()
_rate: dict = {}
_rate_lock = threading.Lock()


def _cache_get(key, allow_stale: bool = False):
    with _cache_lock:
        hit = _cache.get(key)
    if not hit:
        return None
    expires, value = hit
    if expires >= time.monotonic() or allow_stale:
        return value
    return None


def _cache_put(key, value, ttl: Optional[float] = None) -> None:
    ttl = CACHE_TTL_S if ttl is None else ttl
    with _cache_lock:
        if len(_cache) >= CACHE_MAX:
            for k in sorted(_cache, key=lambda k: _cache[k][0])[: CACHE_MAX // 4]:
                _cache.pop(k, None)
        _cache[key] = (time.monotonic() + ttl, value)


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
    with _rate_lock:
        _rate.clear()


def admit_search(user_id: str) -> None:
    """At most SEARCHES_PER_MINUTE searches per user per rolling minute."""
    now = time.monotonic()
    with _rate_lock:
        recent = [t for t in _rate.get(user_id, []) if now - t < 60]
        if len(recent) >= SEARCHES_PER_MINUTE:
            _rate[user_id] = recent
            raise SearchRateLimited("Too many registry searches. Wait a minute and try again.")
        recent.append(now)
        _rate[user_id] = recent
        if len(_rate) > 4096:
            for k in [k for k, v in _rate.items() if not v or now - v[-1] >= 60][:2048]:
                _rate.pop(k, None)


async def _fetch_page(client: httpx.AsyncClient, query: str, cursor: Optional[str]) -> dict:
    params = {"search": query, "limit": str(PAGE_LIMIT), "version": "latest"}
    if cursor:
        params["cursor"] = cursor
    try:
        # Streamed so the size cap holds while reading, not after buffering
        # the whole reply (guard CX-07).
        async with client.stream("GET", REGISTRY_SERVERS, params=params) as resp:
            if resp.status_code != 200:
                raise RegistryUnavailable(f"HTTP {resp.status_code}")
            declared = resp.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > MAX_RESPONSE_BYTES:
                raise RegistryUnavailable("response too large")
            body = bytearray()
            async for chunk in resp.aiter_bytes():
                body.extend(chunk)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise RegistryUnavailable("response too large")
    except httpx.HTTPError as exc:
        raise RegistryUnavailable(f"network error: {type(exc).__name__}") from exc
    try:
        data = json.loads(bytes(body))
    except ValueError as exc:
        raise RegistryUnavailable("response was not JSON") from exc
    if not isinstance(data, dict) or not isinstance(data.get("servers"), list):
        raise RegistryUnavailable("unexpected response shape")
    return data


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=FETCH_TIMEOUT_S, follow_redirects=False,
                             headers={"User-Agent": "Quasar MCP registry search", "Accept": "application/json"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


_inflight: dict = {}


async def _singleflight(key, factory):
    """Concurrent identical cold searches share one upstream run (guard
    CX-09). Keyed per event loop; a caller that goes away does not cancel
    the shared run for the others."""
    loop = asyncio.get_running_loop()
    k = (id(loop), key)
    task = _inflight.get(k)
    if task is None or task.done():
        task = loop.create_task(factory())
        _inflight[k] = task

        def _forget(t, k=k):
            if _inflight.get(k) is t:
                _inflight.pop(k, None)

        task.add_done_callback(_forget)
    return await asyncio.shield(task)


async def search_registry(query: str, cursor: Optional[str] = None) -> dict:
    """Free-text search: one registry page, hosted servers only."""
    key = ("q", query.lower(), cursor or "")
    cached = _cache_get(key)
    if cached is not None:
        return cached
    return await _singleflight(key, lambda: _search_registry_uncached(key, query, cursor))


async def _search_registry_uncached(key, query: str, cursor: Optional[str]) -> dict:
    try:
        async with _client() as client:
            data = await _fetch_page(client, query, cursor)
    except RegistryUnavailable:
        stale = _cache_get(key, allow_stale=True)
        if stale is not None:
            return dict(stale, stale=True)
        raise
    items = dedupe([it for it in (normalize_entry(r) for r in data["servers"]) if it])
    meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    next_cursor = meta.get("nextCursor")
    # Only a cursor the router will accept back is offered (guard CX-15).
    usable = isinstance(next_cursor, str) and 0 < len(next_cursor) <= MAX_CURSOR_LEN
    out = {"items": items, "next_cursor": next_cursor if usable else None,
           "scanned": len(data["servers"]), "source": SOURCE_LABEL, "fetched_at": _now_iso(), "stale": False}
    _cache_put(key, out)
    return out


async def search_topic(topic: str) -> dict:
    """The astronomy topic: several name searches, merged and filtered."""
    if topic not in TOPICS:
        raise ValueError(f"unknown topic {topic!r}")
    key = ("topic", topic)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    return await _singleflight(key, lambda: _search_topic_uncached(key, topic))


async def _search_topic_uncached(key, topic: str) -> dict:
    sem = asyncio.Semaphore(TOPIC_CONCURRENCY)
    failures = []

    async def one(client, q):
        async with sem:
            try:
                return (await _fetch_page(client, q, None))["servers"]
            except Exception as exc:  # noqa: BLE001 - one bad search never sinks the topic
                failures.append(f"{q}: {exc if isinstance(exc, RegistryUnavailable) else type(exc).__name__}")
                return []

    async with _client() as client:
        tasks = {asyncio.ensure_future(one(client, q)): q for q in ASTRONOMY_QUERIES}
        # One overall deadline, so the wait the UI promises holds (guard CX-13).
        done, pending = await asyncio.wait(tasks, timeout=TOPIC_DEADLINE_S)
        for t in pending:
            t.cancel()
            failures.append(f"{tasks[t]}: no answer within {TOPIC_DEADLINE_S:.0f} s")
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    pages = [t.result() for t in done]
    if len(failures) == len(ASTRONOMY_QUERIES):
        stale = _cache_get(key, allow_stale=True)
        if stale is not None:
            return dict(stale, stale=True)
        raise RegistryUnavailable("; ".join(failures[:3]))
    raw = [r for page in pages for r in page]
    items = dedupe([it for it in (normalize_entry(r) for r in raw) if it and is_astronomy(it)])
    # Servers that add something Quasar lacks first, then the duplicates.
    items.sort(key=lambda it: (bool(it["overlaps"]), it["needs_key"] or it["templated"], it["title"].lower()))
    out = {"items": items, "next_cursor": None, "scanned": len(raw), "source": SOURCE_LABEL,
           "fetched_at": _now_iso(), "stale": False, "partial": bool(failures)}
    if failures:
        logger.warning("MCP registry topic %s: %d of %d searches failed: %s",
                       topic, len(failures), len(ASTRONOMY_QUERIES), "; ".join(failures[:5]))
    _cache_put(key, out, PARTIAL_TOPIC_TTL_S if failures else TOPIC_CACHE_TTL_S)
    return out
