"""MCP Registry search for Settings > MCP servers (services/mcp_registry.py).

The registry is replaced by an httpx MockTransport, so nothing leaves the
machine. Covered: normalising entries (hosted only, https only, deprecated
dropped, key headers, templated URLs), the astronomy topic filter (astrology
and lookalikes out), built-in overlap labels backed by real tools, caching,
stale-on-error, the per-user rate limit, and the router's input checks.
"""

import json
import os
import re
import sys
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO, os.path.join(REPO, "ui-pro")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from services import mcp_registry as reg  # noqa: E402


def entry(name, description="", title=None, url=None, rtype="streamable-http", headers=None,
          status="active", packages_only=False, repo=None):
    srv = {"name": name, "description": description, "version": "1.0.0"}
    if title:
        srv["title"] = title
    if repo:
        srv["repository"] = {"url": repo, "source": "github"}
    if packages_only:
        srv["packages"] = [{"registryType": "npm", "identifier": name}]
    else:
        remote = {"type": rtype, "url": url or f"https://{name.split('/')[-1]}.example.com/mcp"}
        if headers:
            remote["headers"] = headers
        srv["remotes"] = [remote]
    return {"server": srv, "_meta": {"io.modelcontextprotocol.registry/official": {
        "status": status, "isLatest": True, "updatedAt": "2026-09-01T00:00:00Z"}}}


ASTRO_FIXTURE = [
    entry("io.github.cyanheads/astronomy-mcp-server",
          "Offline observational astronomy: positions, rise/set, moon phases, eclipses, and seasons.",
          url="https://astronomy.caseyjhand.com/mcp"),
    entry("io.github.pipeworx-io/simbad", "SIMBAD MCP - CDS astronomical object database.",
          url="https://gateway.pipeworx.io/simbad/mcp"),
    entry("com.kundlit/astro", "Free Vedic astrology: panchang, kundali, transits, matching."),
    entry("com.roxyapi/astrology", "Western natal charts, horoscopes, transits, verified vs NASA JPL."),
    entry("ai.astrofabric/mcp", "Agentic growth platform: SEO, ads, outbound and CRM ops tools."),
    entry("com.asteroidxrpl/asteroid-xrpl-hub", "Read-only XRPL tools for ASTEROID: identity, market."),
    entry("io.github.AvatarGaia/canvas-mcp", "TeamAgent Canvas MCP: 13 tools."),
    entry("io.github.mrfentmen/exoplanets-mcp", "Confirmed exoplanets from the NASA Exoplanet Archive.",
          packages_only=True),
    entry("io.github.nasa/earthdata-mcp", "Discover NASA Earth science datasets via the CMR.",
          url="https://cmr.earthdata.nasa.gov/mcp/v1"),
    entry("com.thenightsky/store", "Design custom star maps and engraved jewellery in chat; checkout finishes on site."),
]


class FakeRegistry:
    """Serves ASTRO_FIXTURE filtered by the registry's name-substring search."""

    def __init__(self, servers=None, fail=False, status=200, body=None):
        self.servers = servers if servers is not None else ASTRO_FIXTURE
        self.fail = fail
        self.status = status
        self.body = body
        self.requests = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.url.host == "registry.modelcontextprotocol.io"
        assert request.url.scheme == "https"
        if self.fail:
            raise httpx.ConnectError("down", request=request)
        if self.body is not None:
            return httpx.Response(self.status, content=self.body)
        q = parse_qs(urlsplit(str(request.url)).query)
        assert q["version"] == ["latest"]
        assert q["limit"] == [str(reg.PAGE_LIMIT)]
        term = q.get("search", [""])[0].lower()
        hits = [s for s in self.servers if term in s["server"]["name"].lower()]
        meta = {"count": len(hits)}
        if q.get("cursor") is None and term == "paged":
            meta["nextCursor"] = "next-1"
        return httpx.Response(self.status, json={"servers": hits, "metadata": meta})


@pytest.fixture()
def fake(monkeypatch):
    reg.clear_cache()
    registry = FakeRegistry()

    def client():
        return httpx.AsyncClient(transport=httpx.MockTransport(registry.handler), timeout=5)

    monkeypatch.setattr(reg, "_client", client)
    yield registry
    reg.clear_cache()


def run(coro):
    import asyncio
    return asyncio.run(coro)


# ── normalising ──────────────────────────────────────────────────────────────

def test_hosted_https_entries_only():
    assert reg.normalize_entry(entry("a/pkg", packages_only=True)) is None
    assert reg.normalize_entry(entry("a/http", url="http://plain.example.com/mcp")) is None
    assert reg.normalize_entry(entry("a/creds", url="https://u:p@x.example.com/mcp")) is None
    assert reg.normalize_entry(entry("a/old", status="deprecated")) is None
    assert reg.normalize_entry(entry("a/ws", rtype="websocket")) is None
    assert reg.normalize_entry({"server": {"description": "no name"}}) is None
    sse = reg.normalize_entry(entry("a/sse", rtype="sse"))
    assert sse["transport"] == "sse"


def test_key_headers_templates_and_titles():
    it = reg.normalize_entry(entry(
        "ai.rockmoon/financial-data", "Financials.", title="Rockmoon Financial Data",
        headers=[{"name": "X-API-Key", "isRequired": True, "isSecret": True, "description": "Get a key"}],
        repo="https://github.com/rockmoon/mcp"))
    assert it["needs_key"] is True
    assert it["headers"][0] == {"name": "X-API-Key", "required": True, "secret": True,
                                "description": "Get a key", "value_template": None}
    assert it["title"] == "Rockmoon Financial Data"
    assert it["repository"] == "https://github.com/rockmoon/mcp"
    tmpl = reg.normalize_entry(entry("io.gitmcp/x", url="https://gitmcp.io/{owner}/{repo}"))
    assert tmpl["templated"] is True
    # No registry title: a readable one from the name, without "mcp server".
    assert reg.normalize_entry(entry("io.github.cyanheads/astronomy-mcp-server"))["title"] == "Astronomy"


def test_dedupe_by_name_and_url():
    a = reg.normalize_entry(entry("x/a", url="https://same.example.com/mcp"))
    b = reg.normalize_entry(entry("x/b", url="https://SAME.example.com/mcp/"))
    c = reg.normalize_entry(entry("x/a", url="https://other.example.com/mcp"))
    assert [i["id"] for i in reg.dedupe([a, b, c])] == ["x/a"]


# ── astronomy filter and built-in overlaps ───────────────────────────────────

def test_astronomy_topic_keeps_astronomy_and_drops_astrology(fake):
    res = run(reg.search_topic("astronomy"))
    ids = [i["id"] for i in res["items"]]
    assert "io.github.cyanheads/astronomy-mcp-server" in ids
    assert "io.github.nasa/earthdata-mcp" in ids
    assert "io.github.pipeworx-io/simbad" in ids
    for gone in ("com.kundlit/astro", "com.roxyapi/astrology", "ai.astrofabric/mcp",
                 "com.asteroidxrpl/asteroid-xrpl-hub", "io.github.AvatarGaia/canvas-mcp",
                 "io.github.mrfentmen/exoplanets-mcp", "com.thenightsky/store"):
        assert gone not in ids, gone
    # Servers that duplicate a built-in come after the ones that add something.
    assert ids[-1] == "io.github.pipeworx-io/simbad"
    simbad = res["items"][-1]
    assert simbad["overlaps"] == ["SIMBAD"]
    assert res["partial"] is False
    assert len(fake.requests) == len(reg.ASTRONOMY_QUERIES)


def test_overlap_labels():
    assert reg.overlaps_for("NASA ADS - the Astrophysics Data System") == ["NASA ADS"]
    assert reg.overlaps_for("JPL Horizons MCP.") == ["JPL Horizons"]
    assert reg.overlaps_for("ESA Gaia MCP - the Gaia star catalogue") == ["Gaia"]
    assert reg.overlaps_for("MAST MCP - Space Telescopes archive.") == ["MAST"]
    assert reg.overlaps_for("Search arXiv papers") == ["arXiv search"]
    # Lookalikes are not overlaps.
    assert reg.overlaps_for("Buyer-side fair-price checks (horizon-shield)") == []
    assert reg.overlaps_for("downloads and uploads for your ads account") == []
    assert reg.overlaps_for("Ned's notes app") == []


def _defined_tool_names():
    names = set()
    sources = [os.path.join(REPO, "core", "tool_registrations.py")]
    cap_dir = os.path.join(REPO, "capabilities")
    sources += [os.path.join(cap_dir, f) for f in os.listdir(cap_dir) if f.endswith(".py")]
    for path in sources:
        with open(path, encoding="utf-8") as fh:
            names.update(re.findall(r'\bname\s*=\s*"([a-z_][a-z0-9_]*)"', fh.read()))
    return names


def test_every_overlap_label_is_backed_by_a_real_tool():
    defined = _defined_tool_names()
    for o in reg.BUILTIN_OVERLAPS:
        missing = [t for t in o["tools"] if t not in defined]
        assert not missing, f"{o['label']}: {missing} not defined"


# ── fetching: pages, cache, failures, limits ────────────────────────────────

def test_free_text_search_pages_and_caches(fake):
    fake.servers = [entry("x/paged-one"), entry("x/paged-two", packages_only=True)]
    first = run(reg.search_registry("paged"))
    assert [i["id"] for i in first["items"]] == ["x/paged-one"]  # package-only dropped
    assert first["next_cursor"] == "next-1"
    assert first["scanned"] == 2
    run(reg.search_registry("PAGED"))
    assert len(fake.requests) == 1  # cached, case-insensitive key
    nxt = run(reg.search_registry("paged", "next-1"))
    assert nxt["next_cursor"] is None
    assert parse_qs(urlsplit(str(fake.requests[-1].url)).query)["cursor"] == ["next-1"]


def test_stale_results_when_the_registry_goes_down(fake, monkeypatch):
    fresh = run(reg.search_registry("astronomy"))
    assert fresh["stale"] is False
    monkeypatch.setattr(reg, "CACHE_TTL_S", -1)
    reg.clear_cache()
    run(reg.search_registry("astronomy"))  # cached with an already-expired TTL
    fake.fail = True
    stale = run(reg.search_registry("astronomy"))
    assert stale["stale"] is True and stale["items"]
    with pytest.raises(reg.RegistryUnavailable):
        run(reg.search_registry("nothing-cached"))


@pytest.mark.parametrize("status,body", [(500, b"{}"), (200, b"not json"), (200, b'{"servers": "x"}')])
def test_bad_registry_answers_raise(fake, status, body):
    fake.status, fake.body = status, body
    with pytest.raises(reg.RegistryUnavailable):
        run(reg.search_registry("anything"))


def test_topic_fails_only_when_every_query_fails(fake):
    fake.fail = True
    with pytest.raises(reg.RegistryUnavailable):
        run(reg.search_topic("astronomy"))
    with pytest.raises(ValueError):
        run(reg.search_topic("astrology"))


def test_rate_limit_per_user(monkeypatch):
    reg.clear_cache()
    monkeypatch.setattr(reg, "SEARCHES_PER_MINUTE", 3)
    for _ in range(3):
        reg.admit_search("alice")
    with pytest.raises(reg.SearchRateLimited):
        reg.admit_search("alice")
    reg.admit_search("bob")  # other users are unaffected
    reg.clear_cache()


# ── router ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def api(fake):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.deps import get_current_user
    from api.routers import tools

    app = FastAPI()
    app.include_router(tools.router)
    app.dependency_overrides[get_current_user] = lambda: {"sub": "alice"}
    return TestClient(app)


def test_router_search_and_topic(api):
    res = api.get("/api/mcp-registry/search", params={"q": "astronomy"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["source"] == "registry.modelcontextprotocol.io"
    assert [i["id"] for i in body["items"]] == ["io.github.cyanheads/astronomy-mcp-server"]
    topic = api.get("/api/mcp-registry/search", params={"topic": "astronomy"})
    assert topic.status_code == 200
    assert any(i["overlaps"] == ["SIMBAD"] for i in topic.json()["items"])


@pytest.mark.parametrize("params,detail", [
    ({}, "Type something"),
    ({"q": "   "}, "Type something"),
    ({"q": "x" * 81}, "under 80"),
    ({"topic": "astrology"}, "Unknown topic"),
    ({"topic": "astronomy", "q": "x"}, "not both"),
    ({"q": "x", "cursor": "c" * 600}, "page link"),
])
def test_router_rejects_bad_input(api, params, detail):
    res = api.get("/api/mcp-registry/search", params=params)
    assert res.status_code == 400
    assert detail in res.json()["detail"]


def test_router_maps_failures(api, fake, monkeypatch):
    fake.fail = True
    res = api.get("/api/mcp-registry/search", params={"q": "down"})
    assert res.status_code == 502
    assert "did not answer" in res.json()["detail"]
    monkeypatch.setattr(reg, "SEARCHES_PER_MINUTE", 0)
    assert api.get("/api/mcp-registry/search", params={"q": "x"}).status_code == 429


def test_router_requires_sign_in():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routers import tools

    app = FastAPI()
    app.include_router(tools.router)
    res = TestClient(app).get("/api/mcp-registry/search", params={"q": "x"})
    assert res.status_code in (401, 403)


# ── guard task-4558f74-244 regressions ──────────────────────────────────────

def test_cx05_any_placeholder_form_marks_the_url_templated():
    for url in ("https://vendor.example/<tenant>/mcp", "https://x.example.com/{tenant}/mcp",
                "https://x.example.com/${TENANT}/mcp", "https://x.example.com/YOUR_TENANT/mcp"):
        assert reg.normalize_entry(entry("a/t", url=url))["templated"] is True, url
    assert reg.normalize_entry(entry("a/plain", url="https://x.example.com/mcp"))["templated"] is False


def test_cx07_size_cap_holds_while_streaming(fake, monkeypatch):
    monkeypatch.setattr(reg, "MAX_RESPONSE_BYTES", 1000)
    fake.servers = [entry(f"x/big-{i}", "d" * 300) for i in range(20)]
    with pytest.raises(reg.RegistryUnavailable, match="too large"):
        run(reg.search_registry("big"))


def test_cx08_a_malformed_entry_is_skipped_not_a_500(fake):
    bad = entry("x/astronomy-bad")
    bad["_meta"] = {"io.modelcontextprotocol.registry/official": "not-a-dict"}
    worse = {"server": {"name": "x/astronomy-worse", "remotes": "nope"}}
    weird = {"server": {"name": "x/astronomy-weird", "remotes": [{"type": "sse", "url": "https://w.example.com/mcp",
                                                                   "headers": "nope"}]}}
    fake.servers = [bad, worse, weird, entry("x/astronomy-good")]
    ids = [i["id"] for i in run(reg.search_registry("astronomy"))["items"]]
    assert "x/astronomy-good" in ids
    assert "x/astronomy-bad" in ids  # a bad _meta just means "no status"
    assert reg.normalize_entry(5) is None


def test_cx09_concurrent_cold_topic_searches_share_one_run(fake):
    import asyncio

    async def both():
        return await asyncio.gather(reg.search_topic("astronomy"), reg.search_topic("astronomy"))

    a, b = run(both())
    assert a["items"] == b["items"]
    assert len(fake.requests) == len(reg.ASTRONOMY_QUERIES)


def test_cx10_dedupe_keeps_case_sensitive_paths_apart():
    a = reg.normalize_entry(entry("x/a", url="https://h.example.com/TenantA/mcp"))
    b = reg.normalize_entry(entry("x/b", url="https://h.example.com/tenanta/mcp"))
    c = reg.normalize_entry(entry("x/c", url="https://H.EXAMPLE.com/TenantA/mcp/"))
    assert [i["id"] for i in reg.dedupe([a, b, c])] == ["x/a", "x/b"]


def test_cx13_topic_has_one_overall_deadline(monkeypatch):
    import asyncio

    reg.clear_cache()
    monkeypatch.setattr(reg, "TOPIC_DEADLINE_S", 0.5)

    async def handler(request):
        term = parse_qs(urlsplit(str(request.url)).query)["search"][0]
        if term == "astronomy":
            await asyncio.sleep(5)  # never answers in time
        hits = [s for s in ASTRO_FIXTURE if term in s["server"]["name"].lower()]
        return httpx.Response(200, json={"servers": hits, "metadata": {}})

    monkeypatch.setattr(reg, "_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    import time as _time
    t0 = _time.monotonic()
    res = run(reg.search_topic("astronomy"))
    assert _time.monotonic() - t0 < 3
    assert res["partial"] is True
    reg.clear_cache()


def test_cx14_bad_ports_are_not_connectable():
    for url in ("https://example.com:bad/mcp", "https://example.com:99999/mcp"):
        assert reg.normalize_entry(entry("a/p", url=url)) is None, url
    assert reg.normalize_entry(entry("a/ok", url="https://example.com:8443/mcp")) is not None


def test_cx15_an_unusable_long_cursor_is_not_offered(fake, monkeypatch):
    long_cursor = "c" * (reg.MAX_CURSOR_LEN + 1)

    def handler(request):
        return httpx.Response(200, json={"servers": [entry("x/long-one")], "metadata": {"nextCursor": long_cursor}})

    monkeypatch.setattr(reg, "_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert run(reg.search_registry("long"))["next_cursor"] is None


# ── guard verify round 1 reopen ─────────────────────────────────────────────

def test_cx03_needs_key_only_for_secret_or_key_headers():
    fixed = reg.normalize_entry(entry("a/v", headers=[
        {"name": "X-Client-Version", "isRequired": True, "isSecret": False, "value": "2026-01"}]))
    assert fixed["needs_key"] is False
    secret = reg.normalize_entry(entry("a/s", headers=[{"name": "X-Auth", "isRequired": True, "isSecret": True}]))
    assert secret["needs_key"] is True
    named = reg.normalize_entry(entry("a/n", headers=[{"name": "X-Api-Key", "isRequired": True}]))
    assert named["needs_key"] is True


def test_cx17_placeholder_cases_match_the_frontend():
    path = os.path.join(REPO, "ui-pro", "tests", "fixtures", "mcp-placeholder-cases.json")
    with open(path, encoding="utf-8") as fh:
        cases = json.load(fh)
    for u in cases["placeholder"]:
        assert reg._is_templated("https://" + u), u
    for u in cases["plain"]:
        assert not reg._is_templated("https://" + u), u
