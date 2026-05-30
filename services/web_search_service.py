# services/web_search_service.py
"""
Web Search Router — Unified search service that routes queries to the optimal provider.

- Brave Search API (Primary): For general info, real-time facts, and context.
- Brave LLM Context API: Optimized machine-readable grounding context.
- Tavily: Fallback for clean LLM summaries and images with descriptions.
- Exa: Specialist for deep technical documentation and literature lookup.
- BrowserService: Ultimate local/scraping fallback.

USAGE TRACKING & RATE LIMITS:
- Tracks requests locally in ./data/search_usage.json.
- Resets counts automatically at the beginning of each calendar month.
- Caps Brave and Exa at 1,000 monthly free requests, auto-falling back to Tavily or local scraping.
"""

import os
import re
import json
import time
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional

# Constants
USAGE_FILE = "./data/search_usage.json"
MAX_FREE_LIMIT = 1000

class WebSearchService:
    """Intelligent router and rate-limiter for web search providers."""

    def __init__(self, browser_service: Optional[Any] = None):
        self.browser_service = browser_service
        self.brave_key = os.getenv("BRAVE_API_KEY", "").strip()
        self.tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
        self.exa_key = os.getenv("EXA_API_KEY", "").strip()
        
        # Ensure data directory exists
        os.makedirs("./data", exist_ok=True)

    @staticmethod
    def has_any_provider_key() -> bool:
        """Return True when any web provider is configured."""
        return any(
            os.getenv(key, "").strip()
            for key in ("BRAVE_API_KEY", "TAVILY_API_KEY", "EXA_API_KEY")
        )

    def _get_tavily_client(self) -> Any:
        """Create a Tavily SDK client or raise a clear error."""
        if not self.tavily_key:
            raise RuntimeError("Tavily API key missing")

        from tavily import TavilyClient

        return TavilyClient(api_key=self.tavily_key)

    def _tavily_post(self, endpoint: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        """Call Tavily REST endpoints directly when an older SDK lacks a method."""
        response = requests.post(
            f"https://api.tavily.com/{endpoint.lstrip('/')}",
            headers={
                "Authorization": f"Bearer {self.tavily_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        if response.status_code not in (200, 201, 202):
            return {
                "success": False,
                "error": f"Tavily {endpoint} returned status code {response.status_code}",
                "details": response.text[:1000],
            }
        return response.json()

    def _tavily_get(self, endpoint: str, timeout: int = 30) -> Dict[str, Any]:
        """Call Tavily REST GET endpoints directly when an older SDK lacks a method."""
        response = requests.get(
            f"https://api.tavily.com/{endpoint.lstrip('/')}",
            headers={"Authorization": f"Bearer {self.tavily_key}"},
            timeout=timeout,
        )
        if response.status_code not in (200, 202):
            return {
                "success": False,
                "error": f"Tavily {endpoint} returned status code {response.status_code}",
                "details": response.text[:1000],
            }
        return response.json()

    @staticmethod
    def _normalize_string_list(value: Any) -> Optional[List[str]]:
        """Accept comma/newline strings or lists and return a clean list."""
        if value is None or value == "":
            return None
        if isinstance(value, str):
            items = re.split(r"[\n,]+", value)
        elif isinstance(value, list):
            items = value
        else:
            items = [value]
        cleaned = [str(item).strip() for item in items if str(item).strip()]
        return cleaned or None

    @classmethod
    def _normalize_urls(cls, urls: Any) -> List[str]:
        """Normalize URL input from tool calls into a bounded URL list."""
        normalized = cls._normalize_string_list(urls) or []
        valid_urls = []
        for url in normalized:
            if re.match(r"^https?://[^\s/$.?#].[^\s]*$", url):
                valid_urls.append(url)
        return valid_urls[:20]

    @staticmethod
    def _truncate_text(value: Any, max_chars: int) -> Any:
        """Trim oversized web content while keeping provider metadata intact."""
        if not isinstance(value, str) or len(value) <= max_chars:
            return value
        return (
            value[:max_chars]
            + f"\n\n[... truncated {len(value) - max_chars:,} chars from provider result ...]"
        )

    @classmethod
    def _truncate_result_content(cls, response: Any, max_content_chars: int) -> Any:
        """Trim raw content fields inside Tavily-style responses."""
        if not isinstance(response, dict):
            return response

        trimmed = dict(response)
        if isinstance(trimmed.get("content"), str):
            trimmed["content"] = cls._truncate_text(trimmed["content"], max_content_chars)

        results = trimmed.get("results")
        if isinstance(results, list):
            trimmed_results = []
            for item in results:
                if not isinstance(item, dict):
                    trimmed_results.append(item)
                    continue
                cloned = dict(item)
                for key in ("raw_content", "content", "text"):
                    if key in cloned:
                        cloned[key] = cls._truncate_text(cloned[key], max_content_chars)
                trimmed_results.append(cloned)
            trimmed["results"] = trimmed_results

        return trimmed

    def _get_usage(self) -> Dict[str, Any]:
        """Load search usage from disk, resetting counts if a new month has started."""
        current_month = datetime.now().strftime("%Y-%m")
        default_usage = {
            "month": current_month,
            "brave_count": 0,
            "exa_count": 0
        }
        
        if not os.path.exists(USAGE_FILE):
            return default_usage
            
        try:
            with open(USAGE_FILE, "r", encoding="utf-8") as f:
                usage = json.load(f)
            
            # Reset monthly quota if calendar month has changed
            if usage.get("month") != current_month:
                usage = default_usage
                self._save_usage(usage)
            return usage
        except Exception as e:
            print(f"[SEARCH ROUTER] Failed to load usage file: {e}")
            return default_usage

    def _save_usage(self, usage: Dict[str, Any]) -> None:
        """Persist current monthly search usage to disk."""
        try:
            with open(USAGE_FILE, "w", encoding="utf-8") as f:
                json.dump(usage, f, indent=4)
        except Exception as e:
            print(f"[SEARCH ROUTER] Failed to save usage file: {e}")

    def _increment_usage(self, provider: str) -> None:
        """Increment count for the given provider (brave or exa)."""
        usage = self._get_usage()
        if provider == "brave":
            usage["brave_count"] = usage.get("brave_count", 0) + 1
        elif provider == "exa":
            usage["exa_count"] = usage.get("exa_count", 0) + 1
        self._save_usage(usage)

    def _determine_exa_type(self, query: str) -> str:
        """Determine if Exa should use 'deep' or 'deep-reasoning' based on complexity."""
        q_lower = query.lower()
        word_count = len(query.split())
        
        # Keywords indicating high-value, complex comparison/research questions
        reasoning_keywords = [
            "tradeoff", "trade-off", "trade off",
            "disagreement", "disagree", "methodological",
            "competing", "risk", "ranking", "rank",
            "parameter tradeoff", "parameter trade-off"
        ]
        
        # Trigger deep-reasoning for strong reasoning indicators
        if any(kw in q_lower for kw in reasoning_keywords):
            return "deep-reasoning"
            
        # Trigger deep-reasoning for complex comparison questions (at least 10 words + comparison verbs)
        if any(kw in q_lower for kw in ["compare", "versus", "vs"]) and word_count >= 10:
            return "deep-reasoning"
            
        # Normal Exa specialist searches get deep search
        return "deep"

    def route_and_search(
        self, 
        query: str, 
        max_results: int = 5, 
        search_depth: str = "basic"
    ) -> Dict[str, Any]:
        """Classify search intent and route to the optimal provider with fallbacks."""
        q_lower = query.lower()
        
        # 1. Image or Plot intent -> route to Image Search
        if any(kw in q_lower for kw in ["image", "photo", "chart", "map", "plot", "spectrum"]):
            print(f"[SEARCH ROUTER] Routed to Image Search for: {query!r}")
            return self.search_images(query, max_results=max_results)
            
        # 2. Deep Technical / Research intent -> route to Exa
        if any(kw in q_lower for kw in ["compare", "versus", "vs", "formula", "equations", "papers", "documentation", "handbook", "innovations", "architecture", "literature", "review", "research", "academic"]):
            usage = self._get_usage()
            if self.exa_key and usage.get("exa_count", 0) < MAX_FREE_LIMIT:
                print(f"[SEARCH ROUTER] Routed to Exa (Specialist) for: {query!r}")
                exa_type = self._determine_exa_type(query)
                res = self.search_exa(query, num_results=max_results, search_type=exa_type)
                if res.get("success"):
                    return self._enrich_with_tavily_images(query, res)
                print("[SEARCH ROUTER] Exa failed, falling back to Brave/Tavily")
            elif self.exa_key:
                print("[SEARCH ROUTER] Exa monthly free limit (1000) reached. Falling back.")

        # 3. Fresh / Current / RAG fresh info -> route to Brave Search
        usage = self._get_usage()
        if self.brave_key and usage.get("brave_count", 0) < MAX_FREE_LIMIT:
            print(f"[SEARCH ROUTER] Routed to Brave (Primary) for: {query!r}")
            res = self.search_brave(query, max_results=max_results)
            if res.get("success"):
                return self._enrich_with_tavily_images(query, res)
            print("[SEARCH ROUTER] Brave failed, falling back to Tavily")
        elif self.brave_key:
            print("[SEARCH ROUTER] Brave monthly free limit (1000) reached. Falling back.")

        # 4. Fallback: Tavily Search
        if self.tavily_key:
            print(f"[SEARCH ROUTER] Routing to Tavily for: {query!r}")
            res = self.search_tavily(query, max_results=max_results, search_depth=search_depth)
            if res.get("success"):
                return res
                
        # 5. Ultimate Fallback: Local Scraping / BrowserService
        if self.browser_service:
            print(f"[SEARCH ROUTER] All APIs failed/limited. Falling back to BrowserService scraping for: {query!r}")
            try:
                fallback = self.browser_service.web_search(query=query)
                fallback_results = (
                    fallback.get("results", [])
                    if isinstance(fallback, dict)
                    else fallback
                )
                if not isinstance(fallback_results, list):
                    fallback_results = []
                return {
                    "success": True,
                    "provider": "BrowserService (fallback)",
                    "query": query,
                    "results": fallback_results,
                    "raw_text": fallback.get("raw_text", "") if isinstance(fallback, dict) else "",
                    "images": self._fetch_tavily_images(query) if self.tavily_key else []
                }
            except Exception as e:
                return {"success": False, "error": f"Scraping fallback failed: {e}"}
                
        return {"success": False, "error": "No search providers available or all quotas exceeded"}

    @staticmethod
    def _coerce_snippet(snippets: Any) -> str:
        """Normalize provider snippet fields into a readable text block."""
        if isinstance(snippets, list):
            return "\n\n".join(str(s).strip() for s in snippets if str(s).strip())
        if snippets is None:
            return ""
        return str(snippets).strip()

    def _parse_brave_llm_context(self, data: Any, max_results: int) -> List[Dict[str, str]]:
        """Parse Brave LLM Context responses into Quasar's common result shape."""
        candidates: List[Dict[str, Any]] = []
        sources: Dict[str, Any] = {}

        if isinstance(data, list):
            candidates = [item for item in data if isinstance(item, dict)]
        elif isinstance(data, dict):
            sources = data.get("sources", {}) if isinstance(data.get("sources"), dict) else {}
            grounding = data.get("grounding", {}) if isinstance(data.get("grounding"), dict) else {}

            generic = grounding.get("generic", [])
            if isinstance(generic, list):
                candidates.extend(item for item in generic if isinstance(item, dict))

            poi = grounding.get("poi")
            if isinstance(poi, dict):
                candidates.append(poi)

            map_items = grounding.get("map", [])
            if isinstance(map_items, list):
                candidates.extend(item for item in map_items if isinstance(item, dict))

            # Backward-compatible parsing for any older/alternate wrappers.
            legacy_context = data.get("context", [])
            if isinstance(legacy_context, list):
                candidates.extend(item for item in legacy_context if isinstance(item, dict))

        results: List[Dict[str, str]] = []
        seen_urls = set()
        for item in candidates:
            url = item.get("url", "") or ""
            if url in seen_urls:
                continue

            source_meta = sources.get(url, {}) if isinstance(sources, dict) else {}
            title = (
                item.get("title")
                or item.get("name")
                or source_meta.get("title")
                or "Brave Result"
            )
            snippet = self._coerce_snippet(
                item.get("snippets", item.get("snippet", item.get("description", "")))
            )

            if not url and not snippet:
                continue

            seen_urls.add(url)
            results.append({
                "title": str(title),
                "url": url,
                "snippet": snippet,
            })
            if len(results) >= max_results:
                break

        return results

    @staticmethod
    def _normalize_image_items(items: Any) -> List[Dict[str, str]]:
        """Normalize provider image payloads into Quasar's web image shape."""
        if not isinstance(items, list):
            return []

        images: List[Dict[str, str]] = []
        seen_urls = set()
        for item in items:
            if isinstance(item, str):
                url = item.strip()
                description = ""
            elif isinstance(item, dict):
                url = str(item.get("url") or item.get("src") or item.get("image_url") or "").strip()
                description = str(
                    item.get("description")
                    or item.get("title")
                    or item.get("alt")
                    or ""
                ).strip()
            else:
                continue

            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            images.append({"url": url, "description": description})

        return images

    def _fetch_tavily_images(self, query: str, max_images: int = 6) -> List[Dict[str, str]]:
        """Fetch image tiles for a search result using Tavily's REST API."""
        if not self.tavily_key:
            return []

        payload = {
            "query": query,
            "max_results": min(max(max_images, 1), 10),
            "include_answer": False,
            "include_images": True,
            "include_image_descriptions": True,
        }
        response = self._tavily_post("search", payload, timeout=30)
        if isinstance(response, dict) and response.get("success") is False:
            return []
        return self._normalize_image_items(response.get("images", []))[:max_images]

    def _enrich_with_tavily_images(
        self,
        query: str,
        result: Dict[str, Any],
        max_images: int = 6,
    ) -> Dict[str, Any]:
        """Attach image tiles to otherwise text-only Brave/Exa results."""
        if (
            not isinstance(result, dict)
            or not result.get("success")
            or result.get("images")
            or not self.tavily_key
        ):
            return result

        try:
            images = self._fetch_tavily_images(query, max_images=max_images)
        except Exception as e:
            print(f"[SEARCH ROUTER] Tavily image enrichment failed: {e}")
            images = []

        if not images:
            return result

        enriched = dict(result)
        enriched["images"] = images
        enriched["image_provider"] = "Tavily Images"
        return enriched

    def search_brave(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Perform search using Brave LLM Context (primary RAG) or Web Search."""
        if not self.brave_key:
            return {"success": False, "error": "Brave API key missing"}

        # Attempt Brave LLM Context endpoint (highly pre-summarized and optimized)
        url = "https://api.search.brave.com/res/v1/llm/context"
        headers = {
            "X-Subscription-Token": self.brave_key,
            "Accept": "application/json"
        }
        params = {
            "q": query,
            "maximum_number_of_urls": min(max_results, 10)
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                self._increment_usage("brave")
                data = response.json()
                results = self._parse_brave_llm_context(data, max_results=max_results)
                if results:
                    return {
                        "success": True,
                        "provider": "Brave LLM Context",
                        "query": query,
                        "results": results,
                        "images": []
                    }
        except Exception as e:
            print(f"[SEARCH ROUTER] Brave LLM Context API call failed: {e}")

        # Fallback to standard Brave Web Search
        web_url = "https://api.search.brave.com/res/v1/web/search"
        try:
            web_response = requests.get(web_url, headers=headers, params={"q": query, "count": min(max_results, 10)}, timeout=10)
            if web_response.status_code == 200:
                self._increment_usage("brave")
                data = web_response.json()
                results = []
                for item in data.get("web", {}).get("results", []):
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", item.get("link", "")),
                        "snippet": item.get("snippet", "")
                    })
                return {
                    "success": True,
                    "provider": "Brave Web Search",
                    "query": query,
                    "results": results[:max_results],
                    "images": []
                }
        except Exception as e:
            print(f"[SEARCH ROUTER] Brave Web Search API call failed: {e}")

        return {"success": False, "error": "Brave API request failed"}

    def search_tavily(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_images: bool = True,
    ) -> Dict[str, Any]:
        """Perform Tavily search with source and optional image metadata."""
        if not self.tavily_key:
            return {"success": False, "error": "Tavily API key missing"}

        payload = {
            "query": query,
            "max_results": min(int(max_results), 10),
            "search_depth": search_depth,
            "include_answer": True,
            "include_images": include_images,
            "include_image_descriptions": include_images,
        }

        try:
            try:
                client = self._get_tavily_client()
                response = client.search(**payload)
            except Exception as sdk_error:
                print(f"[SEARCH ROUTER] Tavily SDK search unavailable, using REST: {sdk_error}")
                response = self._tavily_post("search", payload, timeout=60)

            if isinstance(response, dict) and response.get("success") is False:
                return response

            results = []
            for r in response.get("results", []) if isinstance(response, dict) else []:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", "")
                })
            images = self._normalize_image_items(response.get("images", [])) if isinstance(response, dict) else []
            if not images and isinstance(response, dict):
                for r in response.get("results", []):
                    images.extend(self._normalize_image_items(r.get("images", [])))
            return {
                "success": True,
                "provider": "Tavily",
                "query": query,
                "answer": response.get("answer", "") if isinstance(response, dict) else "",
                "results": results,
                "images": images[:6]
            }
        except Exception as e:
            return {"success": False, "error": f"Tavily search failed: {e}"}

    def extract_tavily(
        self,
        urls: Any,
        query: Optional[str] = None,
        chunks_per_source: int = 3,
        extract_depth: str = "basic",
        include_images: bool = False,
        content_format: str = "markdown",
        max_content_chars: int = 12000,
    ) -> Dict[str, Any]:
        """Extract clean content from one or more URLs using Tavily Extract."""
        clean_urls = self._normalize_urls(urls)
        if not clean_urls:
            return {
                "success": False,
                "error": "web_extract_url requires at least one full http(s) URL. Use web_search for keyword queries.",
            }
        if not self.tavily_key:
            return {"success": False, "error": "Tavily API key missing"}

        kwargs: Dict[str, Any] = {
            "extract_depth": extract_depth,
            "include_images": include_images,
            "format": content_format,
        }
        if query:
            kwargs["query"] = query
            kwargs["chunks_per_source"] = max(1, min(int(chunks_per_source), 5))

        try:
            client = self._get_tavily_client()
            url_arg: Any = clean_urls[0] if len(clean_urls) == 1 else clean_urls
            try:
                response = client.extract(urls=url_arg, **kwargs)
            except (AttributeError, TypeError):
                payload = {"urls": url_arg, **kwargs}
                response = self._tavily_post("extract", payload, timeout=60)

            if isinstance(response, dict) and response.get("success") is False:
                return response
            response = self._truncate_result_content(response, max_content_chars)
            return {
                "success": True,
                "provider": "Tavily Extract",
                "urls": clean_urls,
                "response": response,
                "results": response.get("results", []) if isinstance(response, dict) else response,
            }
        except Exception as e:
            return {"success": False, "error": f"Tavily extract failed: {e}"}

    def map_tavily(
        self,
        url: str,
        instructions: Optional[str] = None,
        max_depth: int = 1,
        max_breadth: int = 20,
        limit: int = 100,
        select_paths: Optional[Any] = None,
        exclude_paths: Optional[Any] = None,
        select_domains: Optional[Any] = None,
        exclude_domains: Optional[Any] = None,
        allow_external: bool = False,
    ) -> Dict[str, Any]:
        """Discover URLs on a site using Tavily Map."""
        if not url:
            return {"success": False, "error": "A root URL is required"}
        if not self.tavily_key:
            return {"success": False, "error": "Tavily API key missing"}

        kwargs: Dict[str, Any] = {
            "max_depth": max(1, min(int(max_depth), 5)),
            "max_breadth": max(1, min(int(max_breadth), 100)),
            "limit": max(1, min(int(limit), 500)),
            "allow_external": allow_external,
        }
        if instructions:
            kwargs["instructions"] = instructions
        for key, value in {
            "select_paths": select_paths,
            "exclude_paths": exclude_paths,
            "select_domains": select_domains,
            "exclude_domains": exclude_domains,
        }.items():
            normalized = self._normalize_string_list(value)
            if normalized:
                kwargs[key] = normalized

        try:
            client = self._get_tavily_client()
            try:
                response = client.map(url=url, **kwargs)
            except (AttributeError, TypeError):
                response = self._tavily_post("map", {"url": url, **kwargs}, timeout=60)

            if isinstance(response, dict) and response.get("success") is False:
                return response
            return {
                "success": True,
                "provider": "Tavily Map",
                "url": url,
                "response": response,
                "results": response.get("results", []) if isinstance(response, dict) else response,
            }
        except Exception as e:
            return {"success": False, "error": f"Tavily map failed: {e}"}

    def crawl_tavily(
        self,
        url: str,
        instructions: Optional[str] = None,
        chunks_per_source: int = 3,
        max_depth: int = 1,
        max_breadth: int = 20,
        limit: int = 20,
        extract_depth: str = "basic",
        content_format: str = "markdown",
        include_images: bool = False,
        select_paths: Optional[Any] = None,
        exclude_paths: Optional[Any] = None,
        select_domains: Optional[Any] = None,
        exclude_domains: Optional[Any] = None,
        allow_external: bool = False,
        max_content_chars: int = 5000,
    ) -> Dict[str, Any]:
        """Crawl and extract content from a bounded site section using Tavily Crawl."""
        if not url:
            return {"success": False, "error": "A root URL is required"}
        if not self.tavily_key:
            return {"success": False, "error": "Tavily API key missing"}

        kwargs: Dict[str, Any] = {
            "max_depth": max(1, min(int(max_depth), 5)),
            "max_breadth": max(1, min(int(max_breadth), 100)),
            "limit": max(1, min(int(limit), 50)),
            "extract_depth": extract_depth,
            "format": content_format,
            "include_images": include_images,
            "allow_external": allow_external,
        }
        if instructions:
            kwargs["instructions"] = instructions
            kwargs["chunks_per_source"] = max(1, min(int(chunks_per_source), 5))
        for key, value in {
            "select_paths": select_paths,
            "exclude_paths": exclude_paths,
            "select_domains": select_domains,
            "exclude_domains": exclude_domains,
        }.items():
            normalized = self._normalize_string_list(value)
            if normalized:
                kwargs[key] = normalized

        try:
            client = self._get_tavily_client()
            try:
                response = client.crawl(url=url, **kwargs)
            except (AttributeError, TypeError):
                response = self._tavily_post("crawl", {"url": url, **kwargs}, timeout=150)

            if isinstance(response, dict) and response.get("success") is False:
                return response
            response = self._truncate_result_content(response, max_content_chars)
            return {
                "success": True,
                "provider": "Tavily Crawl",
                "url": url,
                "response": response,
                "results": response.get("results", []) if isinstance(response, dict) else response,
            }
        except Exception as e:
            return {"success": False, "error": f"Tavily crawl failed: {e}"}

    def research_tavily(
        self,
        research_input: str,
        model: str = "auto",
        citation_format: str = "numbered",
        wait_for_completion: bool = True,
        timeout_seconds: int = 120,
        poll_interval_seconds: int = 5,
        max_content_chars: int = 20000,
    ) -> Dict[str, Any]:
        """Create a Tavily Research task and optionally poll until it completes."""
        if not research_input:
            return {"success": False, "error": "A research input is required"}
        if not self.tavily_key:
            return {"success": False, "error": "Tavily API key missing"}

        kwargs = {
            "model": model,
            "citation_format": citation_format,
            "stream": False,
        }

        try:
            client = self._get_tavily_client()
            try:
                response = client.research(input=research_input, **kwargs)
            except (AttributeError, TypeError):
                response = self._tavily_post(
                    "research",
                    {"input": research_input, **kwargs},
                    timeout=60,
                )

            if isinstance(response, dict) and response.get("success") is False:
                return response

            if not wait_for_completion or not isinstance(response, dict):
                return {
                    "success": True,
                    "provider": "Tavily Research",
                    "status": response.get("status") if isinstance(response, dict) else None,
                    "request_id": response.get("request_id") if isinstance(response, dict) else None,
                    "response": response,
                }

            request_id = response.get("request_id")
            status = response.get("status")
            if request_id and status not in {"completed", "failed"}:
                deadline = time.time() + max(1, int(timeout_seconds))
                while time.time() < deadline:
                    time.sleep(max(1, int(poll_interval_seconds)))
                    response = self.get_tavily_research_status(
                        request_id,
                        max_content_chars=max_content_chars,
                    )
                    status = response.get("status")
                    if status in {"completed", "failed"}:
                        break

            response = self._truncate_result_content(response, max_content_chars)
            return {
                "success": response.get("status") != "failed",
                "provider": "Tavily Research",
                "status": response.get("status"),
                "request_id": response.get("request_id"),
                "content": response.get("content"),
                "sources": response.get("sources", []),
                "response": response,
            }
        except Exception as e:
            return {"success": False, "error": f"Tavily research failed: {e}"}

    def get_tavily_research_status(
        self,
        request_id: str,
        max_content_chars: int = 20000,
    ) -> Dict[str, Any]:
        """Fetch status/result for an existing Tavily Research task."""
        if not request_id:
            return {"success": False, "error": "A request_id is required"}
        if not self.tavily_key:
            return {"success": False, "error": "Tavily API key missing"}

        try:
            client = self._get_tavily_client()
            try:
                response = client.get_research(request_id)
            except (AttributeError, TypeError):
                response = self._tavily_get(f"research/{request_id}", timeout=30)

            if isinstance(response, dict) and response.get("success") is False:
                return response
            response = self._truncate_result_content(response, max_content_chars)
            return {
                "success": response.get("status") != "failed",
                "provider": "Tavily Research",
                "status": response.get("status"),
                "request_id": response.get("request_id", request_id),
                "content": response.get("content"),
                "sources": response.get("sources", []),
                "response": response,
            }
        except Exception as e:
            return {"success": False, "error": f"Tavily research status failed: {e}"}

    def search_exa(self, query: str, num_results: int = 5, search_type: str = "deep") -> Dict[str, Any]:
        """Perform semantic research search using Exa."""
        if not self.exa_key:
            return {"success": False, "error": "Exa API key missing"}
            
        try:
            # We can use direct requests to remain lightweight and fully custom
            url = "https://api.exa.ai/search"
            headers = {
                "x-api-key": self.exa_key,
                "Content-Type": "application/json"
            }
            payload = {
                "query": query,
                "type": search_type,
                "numResults": min(num_results, 10),
                "contents": {
                    "highlights": True
                }
            }
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                self._increment_usage("exa")
                data = response.json()
                results = []
                for r in data.get("results", []):
                    highlights = r.get("highlights", [])
                    snippet = " ... ".join(highlights) if highlights else r.get("text", "")[:400]
                    results.append({
                        "title": r.get("title", "Exa Result"),
                        "url": r.get("url", ""),
                        "snippet": snippet,
                        "published_date": r.get("publishedDate", ""),
                        "author": r.get("author", "")
                    })
                return {
                    "success": True,
                    "provider": "Exa",
                    "search_type": search_type,
                    "query": query,
                    "results": results,
                    "images": []
                }
            return {"success": False, "error": f"Exa returned status code {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": f"Exa search failed: {e}"}

    def search_images(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Specialized image-only search leveraging Tavily or Brave's visual index."""
        # 1. Primary: Tavily Image Search (gives rich descriptions)
        if self.tavily_key:
            try:
                payload = {
                    "query": query,
                    "max_results": min(max_results, 10),
                    "include_images": True,
                    "include_image_descriptions": True,
                }
                try:
                    client = self._get_tavily_client()
                    response = client.search(**payload)
                except Exception as sdk_error:
                    print(f"[SEARCH ROUTER] Tavily SDK image search unavailable, using REST: {sdk_error}")
                    response = self._tavily_post("search", payload, timeout=60)

                if isinstance(response, dict) and response.get("success") is False:
                    response = {}

                images = self._normalize_image_items(response.get("images", []))
                
                # Also collect web references to go alongside images
                results = []
                for r in response.get("results", []):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("content", "")
                    })
                    
                if images:
                    return {
                        "success": True,
                        "provider": "Tavily Images",
                        "query": query,
                        "results": results,
                        "images": images[:6]
                    }
            except Exception as e:
                print(f"[SEARCH ROUTER] Tavily image search failed: {e}")

        # 2. Fallback: Brave Web Search (parse returned images/infoboxes)
        usage = self._get_usage()
        if self.brave_key and usage.get("brave_count", 0) < MAX_FREE_LIMIT:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "X-Subscription-Token": self.brave_key,
                "Accept": "application/json"
            }
            # We explicitly ask to return web search results
            try:
                response = requests.get(url, headers=headers, params={"q": query, "count": 10}, timeout=10)
                if response.status_code == 200:
                    self._increment_usage("brave")
                    data = response.json()
                    images = []
                    # Extrapolate any image objects or thumbnails in Brave results
                    for item in data.get("web", {}).get("results", []):
                        thumbnail = item.get("thumbnail", {})
                        if thumbnail and thumbnail.get("src"):
                            images.append({
                                "url": thumbnail.get("src"),
                                "description": item.get("title", "")
                            })
                    
                    results = []
                    for item in data.get("web", {}).get("results", []):
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "snippet": item.get("snippet", "")
                        })
                        
                    return {
                        "success": True,
                        "provider": "Brave Images (extrapolated)",
                        "query": query,
                        "results": results[:max_results],
                        "images": images[:6]
                    }
            except Exception as e:
                print(f"[SEARCH ROUTER] Brave image search failed: {e}")

        return {"success": False, "error": "No image search providers succeeded"}
