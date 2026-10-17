"""Web search fallbacks when Tavily runs out (2026-10-16).

1. An out-of-credits / rejected / rate-limited Tavily account opens a circuit
   breaker instead of being retried (SDK then REST) on every search.
2. Exa is the general fallback after Tavily, including for domain-pack
   searches, carrying the domain list, freshness and news hints.
3. A failed BrowserService scrape comes back as ``success: False`` instead of
   a success with zero results.
"""
import sys
import time
import types

import pytest

from services import web_search_service
from services.web_search_service import WebSearchService


class DummyResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


# tavily-python's own class names: the breaker matches on the name
class ForbiddenError(Exception):
    pass


class UsageLimitExceededError(Exception):
    pass


EXA_PAYLOAD = {
    "results": [
        {"title": "Exa hit", "url": "https://exa.example/page", "highlights": ["exa snippet"]},
    ]
}


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(web_search_service, "USAGE_FILE", str(tmp_path / "search_usage.json"))
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    monkeypatch.setenv("EXA_API_KEY", "exa-test-key")
    monkeypatch.setenv("QUASAR_WEB_IMAGES_ENABLED", "true")
    return monkeypatch


def install_raising_tavily(monkeypatch, exc):
    calls = {"sdk": 0}

    class FakeClient:
        def __init__(self, api_key):
            pass

        def search(self, **kwargs):
            calls["sdk"] += 1
            raise exc

    monkeypatch.setitem(sys.modules, "tavily", types.SimpleNamespace(TavilyClient=FakeClient))
    return calls


def capture_exa(monkeypatch, response=None):
    sent = []

    def fake_post(url, headers=None, json=None, timeout=None):
        assert "exa.ai" in url, f"unexpected POST to {url}"
        sent.append(json)
        return response or DummyResponse(200, EXA_PAYLOAD)

    monkeypatch.setattr(web_search_service.requests, "post", fake_post)
    return sent


# ── 1. Tavily circuit breaker ──────────────────────────────────────────────


def test_sdk_out_of_credits_opens_breaker_without_a_rest_retry(env):
    calls = install_raising_tavily(env, ForbiddenError("This request exceeds your plan's set usage limit."))

    def no_rest(*a, **k):
        raise AssertionError("an account refusal must not be retried over REST")

    env.setattr(WebSearchService, "_tavily_post", no_rest)

    res = WebSearchService().search_tavily("current policy")
    assert res["success"] is False
    assert "usage limit" in res["error"]
    assert WebSearchService._tavily_cooling_down() is True
    remaining = WebSearchService._tavily_down_until - time.time()
    assert remaining > 1700, "account refusals cool down for 30 minutes"

    # While open, Tavily fast-fails without touching the SDK.
    res2 = WebSearchService().search_tavily("current policy")
    assert res2 == {"success": False, "error": "Tavily circuit breaker open"}
    assert calls["sdk"] == 1


@pytest.mark.parametrize("status", [401, 403, 432, 433])
def test_rest_account_refusal_opens_breaker(env, status):
    env.setitem(sys.modules, "tavily", None)  # SDK import fails -> REST path
    env.setattr(
        web_search_service.requests, "post",
        lambda *a, **k: DummyResponse(status, text="refused"),
    )
    res = WebSearchService().search_tavily("current policy")
    assert res["success"] is False
    assert res["status_code"] == status
    assert WebSearchService._tavily_cooling_down() is True


def test_rate_limit_cools_down_briefly_and_never_shortens_an_account_cooldown(env):
    install_raising_tavily(env, UsageLimitExceededError("rate limited"))
    WebSearchService().search_tavily("q")
    remaining = WebSearchService._tavily_down_until - time.time()
    assert 0 < remaining <= 61

    WebSearchService._note_tavily_failure({"status_code": 432})
    long_until = WebSearchService._tavily_down_until
    WebSearchService._note_tavily_failure({"status_code": 429})
    assert WebSearchService._tavily_down_until == long_until


def test_server_error_does_not_open_breaker(env):
    env.setitem(sys.modules, "tavily", None)
    env.setattr(web_search_service.requests, "post", lambda *a, **k: DummyResponse(502, text="bad gateway"))
    res = WebSearchService().search_tavily("q")
    assert res["success"] is False
    assert WebSearchService._tavily_cooling_down() is False


def test_open_breaker_blocks_every_tavily_entry_point(env):
    WebSearchService._note_tavily_failure({"status_code": 432})

    def no_network(*a, **k):
        raise AssertionError("Tavily must not be called while its breaker is open")

    env.setattr(web_search_service.requests, "post", no_network)
    env.setattr(WebSearchService, "_get_tavily_client", no_network)
    svc = WebSearchService()
    breaker = WebSearchService._tavily_breaker_result()
    assert svc.extract_tavily("https://example.com/page") == breaker
    assert svc.map_tavily("https://example.com") == breaker
    assert svc.crawl_tavily("https://example.com") == breaker
    assert svc.research_tavily("ALMA band 3 sensitivity") == breaker
    assert svc._fetch_tavily_images("M87 jet") == []
    assert svc._start_image_prefetch("M87 jet") == (None, None)


# ── 2. Exa as the general fallback ─────────────────────────────────────────


def test_router_falls_back_to_exa_when_tavily_is_out_of_credits(env):
    install_raising_tavily(env, ForbiddenError("plan usage limit exceeded"))
    sent = capture_exa(env)

    res = WebSearchService().route_and_search(
        "latest ALMA cycle 13 news", freshness="week", topic="news", want_images=False,
    )

    assert res["success"] is True
    assert res["provider"] == "Exa"
    assert res["results"][0]["url"] == "https://exa.example/page"
    assert len(sent) == 1
    assert sent[0]["type"] == "auto"
    assert sent[0]["category"] == "news"
    assert sent[0]["startPublishedDate"].endswith("Z")
    assert "includeDomains" not in sent[0]


def test_router_skips_tavily_while_breaker_open_and_uses_exa(env):
    WebSearchService._note_tavily_failure({"status_code": 433})

    def no_tavily(self, *a, **k):
        raise AssertionError("Tavily must be skipped while its breaker is open")

    env.setattr(WebSearchService, "search_tavily", no_tavily)
    sent = capture_exa(env)

    res = WebSearchService().route_and_search("current observatory policy", want_images=False)
    assert res["provider"] == "Exa"
    assert len(sent) == 1


def test_domain_restricted_search_falls_back_to_exa_with_the_domain_list(env):
    env.setattr(
        WebSearchService, "search_tavily",
        lambda self, *a, **k: {"success": False, "error": "Tavily search returned status code 432", "status_code": 432},
    )
    sent = capture_exa(env)

    res = WebSearchService().route_and_search(
        "JWST Cycle 5 deadline", include_domains=["stsci.edu", "nasa.gov"],
    )
    assert res["success"] is True
    assert res["provider"] == "Exa"
    assert sent[0]["includeDomains"] == ["stsci.edu", "nasa.gov"]


def test_domain_restricted_search_without_any_key_keeps_its_old_error(env):
    env.delenv("TAVILY_API_KEY")
    env.delenv("EXA_API_KEY")
    res = WebSearchService().route_and_search("JWST deadline", include_domains=["stsci.edu"])
    assert res == {"success": False, "error": "Tavily API key missing (domain-restricted search)"}


def test_literature_query_does_not_retry_a_failed_exa(env):
    sent = capture_exa(env, response=DummyResponse(500))
    env.setattr(
        WebSearchService, "search_tavily",
        lambda self, *a, **k: {"success": False, "error": "Tavily down"},
    )
    res = WebSearchService().route_and_search("papers on protoplanetary disk substructure", want_images=False)
    assert res["success"] is False
    assert len(sent) == 1, "Exa already failed as the specialist; no second Exa call"
    assert "Exa: Exa returned status code 500" in res["error"]
    assert "Tavily: Tavily down" in res["error"]


# ── 3. Scraper failures are failures ───────────────────────────────────────


class FakeBrowser:
    def __init__(self, payload):
        self.payload = payload

    def web_search(self, query):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def _all_apis_down(env):
    env.delenv("TAVILY_API_KEY")
    env.delenv("EXA_API_KEY")


@pytest.mark.parametrize(
    "payload, reason",
    [
        ({"error": "Playwright is not installed", "query": "q"}, "BrowserService: Playwright is not installed"),
        (RuntimeError("browser crashed"), "BrowserService: browser crashed"),
        ({"query": "q", "results": [], "raw_text": ""}, "BrowserService: no results"),
        ({"query": "q", "results": [], "raw_text": "[Error extracting text: boom]"}, "BrowserService: no results"),
    ],
)
def test_scraper_failure_is_reported_as_failure(env, payload, reason):
    _all_apis_down(env)
    res = WebSearchService(browser_service=FakeBrowser(payload)).route_and_search("q", want_images=False)
    assert res["success"] is False
    assert reason in res["error"]
    assert res["error"].startswith("No search providers available or all quotas exceeded")


def _agent_search(browser, monkeypatch, router_result=None, router_raises=False):
    """Run QuasarAgent._tavily_web_search against a stub self and router."""
    from core.agent import QuasarAgent

    def fake_route(self, **kw):
        if router_raises:
            raise RuntimeError("router crashed")
        return router_result

    monkeypatch.setattr(WebSearchService, "route_and_search", fake_route)
    stub = types.SimpleNamespace(browser_service=browser, _expand_astro_query=lambda q: q)
    return QuasarAgent._tavily_web_search(stub, query="q")


def test_agent_keeps_the_router_failure_instead_of_scraping_again(env):
    browser = FakeBrowser(AssertionError("the router already tried the scraper"))
    failure = {"success": False, "error": "No search providers available or all quotas exceeded (Tavily: x)"}
    res = _agent_search(browser, env, router_result=failure)
    assert res["success"] is False
    assert "Tavily: x" in res["error"]


@pytest.mark.parametrize(
    "payload, success",
    [
        ({"error": "Playwright is not installed"}, False),
        ({"query": "q", "results": [], "raw_text": ""}, False),
        ({"query": "q", "results": [{"title": "t", "url": "https://ddg.example/a", "snippet": "s"}]}, True),
    ],
)
def test_agent_scraper_fallback_after_a_router_crash(env, payload, success):
    res = _agent_search(FakeBrowser(payload), env, router_raises=True)
    assert res["success"] is success
    if success:
        assert res["provider"] == "BrowserService (fallback)"
    else:
        assert res["error"].startswith("Search failed")


def test_scraper_results_still_succeed(env):
    _all_apis_down(env)
    hit = {"title": "DDG hit", "url": "https://ddg.example/a", "snippet": "s"}
    res = WebSearchService(
        browser_service=FakeBrowser({"query": "q", "results": [hit]})
    ).route_and_search("q", want_images=False)
    assert res["success"] is True
    assert res["provider"] == "BrowserService (fallback)"
    assert res["results"][0]["url"] == "https://ddg.example/a"
