import sys
import types

import pytest

from services import web_search_service
from services.web_search_service import WebSearchService


class DummyResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture
def isolated_usage_file(monkeypatch, tmp_path):
    usage_file = tmp_path / "search_usage.json"
    monkeypatch.setattr(web_search_service, "USAGE_FILE", str(usage_file))
    return usage_file


def install_fake_tavily(monkeypatch, fake_client_cls):
    monkeypatch.setitem(
        sys.modules,
        "tavily",
        types.SimpleNamespace(TavilyClient=fake_client_cls),
    )


def test_brave_llm_context_parses_grounding_response(monkeypatch, isolated_usage_file):
    monkeypatch.setenv("BRAVE_API_KEY", "brave-test-key")

    payload = {
        "grounding": {
            "generic": [
                {
                    "url": "https://observatory.example/status",
                    "title": "Observatory Status",
                    "snippets": ["First relevant chunk.", "Second relevant chunk."],
                }
            ],
            "map": [],
        },
        "sources": {
            "https://observatory.example/status": {
                "title": "Source Metadata Title",
                "hostname": "observatory.example",
            }
        },
    }

    def fake_get(url, headers=None, params=None, timeout=None):
        return DummyResponse(200, payload)

    monkeypatch.setattr(web_search_service.requests, "get", fake_get)

    result = WebSearchService().search_brave("observatory status", max_results=3)

    assert result["success"] is True
    assert result["provider"] == "Brave LLM Context"
    assert result["results"] == [
        {
            "title": "Observatory Status",
            "url": "https://observatory.example/status",
            "snippet": "First relevant chunk.\n\nSecond relevant chunk.",
        }
    ]


def test_brave_llm_context_empty_response_falls_back_to_web_search(
    monkeypatch,
    isolated_usage_file,
):
    monkeypatch.setenv("BRAVE_API_KEY", "brave-test-key")
    responses = [
        DummyResponse(200, {"grounding": {"generic": [], "map": []}, "sources": {}}),
        DummyResponse(
            200,
            {
                "web": {
                    "results": [
                        {
                            "title": "Fallback Result",
                            "url": "https://example.com/fallback",
                            "snippet": "Fallback snippet",
                        }
                    ]
                }
            },
        ),
    ]

    def fake_get(url, headers=None, params=None, timeout=None):
        return responses.pop(0)

    monkeypatch.setattr(web_search_service.requests, "get", fake_get)

    result = WebSearchService().search_brave("fallback query", max_results=1)

    assert result["success"] is True
    assert result["provider"] == "Brave Web Search"
    assert result["results"][0]["url"] == "https://example.com/fallback"


def test_advanced_research_routes_to_exa_deep(monkeypatch, isolated_usage_file):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("EXA_API_KEY", "exa-test-key")
    calls = {}

    def fake_search_exa(self, query, num_results=5, search_type="auto"):
        calls["query"] = query
        calls["num_results"] = num_results
        calls["search_type"] = search_type
        return {
            "success": True,
            "provider": "Exa",
            "search_type": search_type,
            "results": [],
            "images": [],
        }

    monkeypatch.setattr(WebSearchService, "search_exa", fake_search_exa)

    result = WebSearchService().route_and_search(
        "compare current AI research architectures",
        max_results=4,
        search_depth="advanced",
    )

    assert result["success"] is True
    assert result["provider"] == "Exa"
    assert calls == {
        "query": "compare current AI research architectures",
        "num_results": 4,
        "search_type": "deep",
    }


def test_tavily_search_uses_rest_fallback_and_returns_images(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    monkeypatch.delitem(sys.modules, "tavily", raising=False)
    calls = {}

    def fake_tavily_post(self, endpoint, payload, timeout=60):
        calls["endpoint"] = endpoint
        calls["payload"] = payload
        return {
            "answer": "Current answer",
            "results": [
                {
                    "title": "Policy Page",
                    "url": "https://example.com/policy",
                    "content": "Policy snippet",
                }
            ],
            "images": [
                {
                    "url": "https://example.com/image.jpg",
                    "description": "Image description",
                }
            ],
        }

    monkeypatch.setattr(WebSearchService, "_tavily_post", fake_tavily_post)

    result = WebSearchService().search_tavily("current policy", max_results=3)

    assert result["success"] is True
    assert result["provider"] == "Tavily"
    assert result["results"][0]["url"] == "https://example.com/policy"
    assert result["images"] == [
        {
            "url": "https://example.com/image.jpg",
            "description": "Image description",
        }
    ]
    assert calls["endpoint"] == "search"
    assert calls["payload"]["include_images"] is True
    assert calls["payload"]["include_image_descriptions"] is True


def test_route_enriches_brave_results_with_tavily_images(monkeypatch, isolated_usage_file):
    monkeypatch.setenv("BRAVE_API_KEY", "brave-test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")

    def fake_search_brave(self, query, max_results=5):
        return {
            "success": True,
            "provider": "Brave LLM Context",
            "query": query,
            "results": [{"title": "Result", "url": "https://example.com", "snippet": ""}],
            "images": [],
        }

    def fake_fetch_images(self, query, max_images=6):
        return [{"url": "https://example.com/img.jpg", "description": "Result image"}]

    monkeypatch.setattr(WebSearchService, "search_brave", fake_search_brave)
    monkeypatch.setattr(WebSearchService, "_fetch_tavily_images", fake_fetch_images)

    result = WebSearchService().route_and_search("current observatory policy", max_results=5)

    assert result["success"] is True
    assert result["provider"] == "Brave LLM Context"
    assert result["image_provider"] == "Tavily Images"
    assert result["images"][0]["url"] == "https://example.com/img.jpg"


def test_tavily_extract_urls_uses_sdk_and_trims_content(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    calls = {}

    class FakeTavilyClient:
        def __init__(self, api_key):
            calls["api_key"] = api_key

        def extract(self, urls=None, **kwargs):
            calls["extract"] = {"urls": urls, "kwargs": kwargs}
            return {
                "results": [
                    {
                        "url": "https://example.com/a",
                        "raw_content": "x" * 60,
                    }
                ]
            }

    install_fake_tavily(monkeypatch, FakeTavilyClient)

    result = WebSearchService().extract_tavily(
        "https://example.com/a, https://example.com/b",
        query="authentication",
        chunks_per_source=9,
        max_content_chars=20,
    )

    assert result["success"] is True
    assert result["provider"] == "Tavily Extract"
    assert calls["api_key"] == "tvly-test-key"
    assert calls["extract"]["urls"] == ["https://example.com/a", "https://example.com/b"]
    assert calls["extract"]["kwargs"]["query"] == "authentication"
    assert calls["extract"]["kwargs"]["chunks_per_source"] == 5
    assert "truncated" in result["results"][0]["raw_content"]


def test_tavily_map_site_normalizes_filters_and_caps_limit(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    calls = {}

    class FakeTavilyClient:
        def __init__(self, api_key):
            pass

        def map(self, url=None, **kwargs):
            calls["map"] = {"url": url, "kwargs": kwargs}
            return {"results": ["https://docs.example.com/api/auth"]}

    install_fake_tavily(monkeypatch, FakeTavilyClient)

    result = WebSearchService().map_tavily(
        "https://docs.example.com",
        instructions="authentication docs",
        limit=900,
        select_paths="/api/.*,/guides/.*",
        allow_external=True,
    )

    assert result["success"] is True
    assert result["provider"] == "Tavily Map"
    assert calls["map"]["url"] == "https://docs.example.com"
    assert calls["map"]["kwargs"]["limit"] == 500
    assert calls["map"]["kwargs"]["select_paths"] == ["/api/.*", "/guides/.*"]
    assert calls["map"]["kwargs"]["allow_external"] is True


def test_tavily_crawl_site_uses_bounded_defaults(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    calls = {}

    class FakeTavilyClient:
        def __init__(self, api_key):
            pass

        def crawl(self, url=None, **kwargs):
            calls["crawl"] = {"url": url, "kwargs": kwargs}
            return {
                "results": [
                    {
                        "url": "https://docs.example.com/api/auth",
                        "content": "y" * 80,
                    }
                ]
            }

    install_fake_tavily(monkeypatch, FakeTavilyClient)

    result = WebSearchService().crawl_tavily(
        "https://docs.example.com",
        instructions="API authentication",
        limit=200,
        chunks_per_source=0,
        max_content_chars=25,
    )

    assert result["success"] is True
    assert result["provider"] == "Tavily Crawl"
    assert calls["crawl"]["kwargs"]["limit"] == 50
    assert calls["crawl"]["kwargs"]["chunks_per_source"] == 1
    assert calls["crawl"]["kwargs"]["allow_external"] is False
    assert "truncated" in result["results"][0]["content"]


def test_tavily_research_polls_until_completed(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    monkeypatch.setattr(web_search_service.time, "sleep", lambda seconds: None)
    calls = {"status_count": 0}

    class FakeTavilyClient:
        def __init__(self, api_key):
            pass

        def research(self, input=None, **kwargs):
            calls["research"] = {"input": input, "kwargs": kwargs}
            return {"request_id": "research-123", "status": "pending"}

        def get_research(self, request_id):
            calls["status_count"] += 1
            if calls["status_count"] == 1:
                return {"request_id": request_id, "status": "running"}
            return {
                "request_id": request_id,
                "status": "completed",
                "content": "Cited research report",
                "sources": [{"url": "https://example.com/source"}],
            }

    install_fake_tavily(monkeypatch, FakeTavilyClient)

    result = WebSearchService().research_tavily(
        "compare observatory operations",
        model="pro",
        timeout_seconds=10,
    )

    assert result["success"] is True
    assert result["provider"] == "Tavily Research"
    assert result["status"] == "completed"
    assert result["request_id"] == "research-123"
    assert result["content"] == "Cited research report"
    assert calls["research"]["kwargs"]["model"] == "pro"
    assert calls["status_count"] == 2
