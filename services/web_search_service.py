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
import threading
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional
from services.content_safety import (
    blocked_query_result,
    is_blocked_url,
    is_explicit_query,
    sanitize_web_payload,
    web_images_enabled,
)

# Constants
USAGE_FILE = "./data/search_usage.json"
MAX_FREE_LIMIT = 1000
DEFAULT_MAX_RESULTS = 10

class WebSearchService:
    """Intelligent router and rate-limiter for web search providers."""

    # ── Brave circuit breaker ────────────────────────────────────────────
    # A failing Brave account (timeout / 401 / 429 / 5xx on BOTH endpoints)
    # used to cost every search up to 2×10 s of dead attempts before the
    # Tavily fallback even started (live 2026-07-18: ~24 s web phases).
    # Class-level because the service is instantiated per search call.
    _BRAVE_COOLDOWN_SECONDS = 600.0
    _brave_down_until = 0.0

    @classmethod
    def _brave_cooling_down(cls) -> bool:
        return time.time() < cls._brave_down_until

    @classmethod
    def _trip_brave_breaker(cls, why: str) -> None:
        cls._brave_down_until = time.time() + cls._BRAVE_COOLDOWN_SECONDS
        print(
            f"[SEARCH ROUTER] Brave circuit OPEN for "
            f"{int(cls._BRAVE_COOLDOWN_SECONDS)}s — {why}"
        )

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

        # Reserve deep-reasoning for genuinely hard synthesis/comparison work.
        # Normal advanced searches should stay on Exa deep.
        reasoning_keywords = [
            "tradeoff", "trade-off", "trade off",
            "disagreement", "disagree", "methodological",
            "competing", "risk", "ranking", "rank",
            "parameter tradeoff", "parameter trade-off",
            "systematic review", "meta-analysis", "evidence landscape",
            "conflicting evidence", "which approach", "pros and cons",
        ]

        if any(kw in q_lower for kw in reasoning_keywords):
            return "deep-reasoning"

        comparison_terms = ["compare", "comparison", "versus", " vs "]
        complexity_terms = [
            "identify", "explain", "evaluate", "synthesize", "recommend",
            "why", "best", "limitations", "assumptions", "implications",
        ]
        if (
            any(kw in q_lower for kw in comparison_terms)
            and word_count >= 12
            and any(kw in q_lower for kw in complexity_terms)
        ):
            return "deep-reasoning"

        return "deep"

    def route_and_search(
        self, 
        query: str, 
        max_results: int = DEFAULT_MAX_RESULTS,
        search_depth: str = "basic"
    ) -> Dict[str, Any]:
        """Classify search intent and route to the optimal provider with fallbacks."""
        if is_explicit_query(query):
            return blocked_query_result(query)

        q_lower = query.lower()
        try:
            max_results = max(1, min(int(max_results), 10))
        except (TypeError, ValueError):
            max_results = DEFAULT_MAX_RESULTS
        
        # 1. Image or Plot intent -> route to Image Search
        if any(kw in q_lower for kw in ["image", "photo", "chart", "map", "plot", "spectrum"]):
            print(f"[SEARCH ROUTER] Routed to Image Search for: {query!r}")
            return sanitize_web_payload(self.search_images(query, max_results=max_results))
            
        # 2. Advanced / deep technical / research intent -> route to Exa.
        # Normal advanced search uses Exa deep; only very hard synthesis uses
        # Exa deep-reasoning inside _determine_exa_type().
        exa_intent = (
            search_depth == "advanced"
            or any(kw in q_lower for kw in [
                "compare", "versus", "vs", "formula", "equations",
                "papers", "documentation", "handbook", "innovations",
                "architecture", "literature", "review", "research",
                "academic",
            ])
        )
        # Exa/Brave return text-only results, so their image tiles come from a
        # separate Tavily call. It used to run SERIALLY after the text search
        # (+2.7 s measured 2026-07-18); prefetch it concurrently instead.
        usage = self._get_usage()
        exa_eligible = bool(
            exa_intent and self.exa_key and usage.get("exa_count", 0) < MAX_FREE_LIMIT
        )
        brave_eligible = bool(
            self.brave_key
            and usage.get("brave_count", 0) < MAX_FREE_LIMIT
            and not self._brave_cooling_down()
        )
        img_thread, img_holder = (None, None)
        if exa_eligible or brave_eligible:
            img_thread, img_holder = self._start_image_prefetch(query)

        if exa_intent:
            if exa_eligible:
                print(f"[SEARCH ROUTER] Routed to Exa (Specialist) for: {query!r}")
                exa_type = self._determine_exa_type(query)
                res = self.search_exa(query, num_results=max_results, search_type=exa_type)
                if res.get("success"):
                    return sanitize_web_payload(
                        self._attach_prefetched_images(res, img_thread, img_holder)
                    )
                print("[SEARCH ROUTER] Exa failed, falling back to Brave/Tavily")
            elif self.exa_key:
                print("[SEARCH ROUTER] Exa monthly free limit (1000) reached. Falling back.")

        # 3. Fresh / Current / RAG fresh info -> route to Brave Search
        if brave_eligible:
            print(f"[SEARCH ROUTER] Routed to Brave (Primary) for: {query!r}")
            res = self.search_brave(query, max_results=max_results)
            if res.get("success"):
                return sanitize_web_payload(
                    self._attach_prefetched_images(res, img_thread, img_holder)
                )
            print("[SEARCH ROUTER] Brave failed, falling back to Tavily")
        elif self.brave_key and self._brave_cooling_down():
            print("[SEARCH ROUTER] Brave circuit breaker open. Falling back to Tavily.")
        elif self.brave_key:
            print("[SEARCH ROUTER] Brave monthly free limit (1000) reached. Falling back.")

        # 4. Fallback: Tavily Search
        if self.tavily_key:
            print(f"[SEARCH ROUTER] Routing to Tavily for: {query!r}")
            # When a prefetch is in flight, reuse those tiles instead of asking
            # the search call for images+descriptions (the slow combination).
            res = self.search_tavily(
                query,
                max_results=max_results,
                search_depth=search_depth,
                include_images=img_thread is None,
            )
            if res.get("success"):
                if img_thread is not None:
                    res = self._attach_prefetched_images(res, img_thread, img_holder)
                return sanitize_web_payload(res)
                
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
                if img_thread is not None:
                    img_thread.join(timeout=4.0)
                    fallback_images = (img_holder or {}).get("images") or []
                else:
                    fallback_images = self._fetch_tavily_images(query) if self.tavily_key else []
                return sanitize_web_payload({
                    "success": True,
                    "provider": "BrowserService (fallback)",
                    "query": query,
                    "results": fallback_results,
                    "raw_text": fallback.get("raw_text", "") if isinstance(fallback, dict) else "",
                    "images": fallback_images
                })
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
    def _normalize_source_url(value: Any) -> str:
        """Return a web URL only when a provider supplied a real source page."""
        url = str(value or "").strip().strip("<>")
        url = url.rstrip(".,;:)]}'\"")
        if not url:
            return ""
        if url.startswith(("http://", "https://")):
            return url
        if url.startswith("www."):
            return f"https://{url}"
        if re.match(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}/\S+$", url):
            return f"https://{url}"
        return ""

    @classmethod
    def _normalize_image_items(
        cls,
        items: Any,
        source_url: str = "",
        source_title: str = "",
    ) -> List[Dict[str, str]]:
        """Normalize provider image payloads into Quasar's web image shape."""
        if not web_images_enabled() or not isinstance(items, list):
            return []

        images: List[Dict[str, str]] = []
        seen_urls = set()
        parent_source_url = cls._normalize_source_url(source_url)
        parent_source_title = str(source_title or "").strip()
        for item in items:
            if isinstance(item, str):
                url = item.strip()
                description = ""
                item_source_url = parent_source_url
                item_source_title = parent_source_title
            elif isinstance(item, dict):
                url = str(item.get("url") or item.get("src") or item.get("image_url") or "").strip()
                description = str(
                    item.get("description")
                    or item.get("title")
                    or item.get("alt")
                    or ""
                ).strip()
                item_source_url = cls._normalize_source_url(
                    item.get("sourceUrl")
                    or item.get("source_url")
                    or item.get("sourcePageUrl")
                    or item.get("source_page_url")
                    or item.get("pageUrl")
                    or item.get("page_url")
                    or item.get("source")
                    or parent_source_url
                )
                item_source_title = str(
                    item.get("sourceTitle")
                    or item.get("source_title")
                    or item.get("sourcePageTitle")
                    or item.get("source_page_title")
                    or item.get("pageTitle")
                    or item.get("page_title")
                    or parent_source_title
                    or ""
                ).strip()
            else:
                continue

            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            image = {"url": url, "description": description}
            if item_source_url:
                image["sourceUrl"] = item_source_url
            if item_source_title:
                image["sourceTitle"] = item_source_title
            images.append(image)

        return images

    @staticmethod
    def _dedupe_image_items(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        images: List[Dict[str, str]] = []
        by_url: Dict[str, int] = {}
        for item in items:
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            key = url.lower().rstrip("/")
            if key in by_url:
                current = images[by_url[key]]
                for field in ("description", "sourceUrl", "sourceTitle"):
                    if not current.get(field) and item.get(field):
                        current[field] = item[field]
                continue
            by_url[key] = len(images)
            images.append(item)
        return images

    @classmethod
    def _collect_tavily_images(cls, response: Any, max_images: int = 6) -> List[Dict[str, str]]:
        """Prefer Tavily images attached to a source result, then generic images."""
        if not isinstance(response, dict):
            return []

        linked_images: List[Dict[str, str]] = []
        for result in response.get("results", []):
            if not isinstance(result, dict):
                continue
            linked_images.extend(
                cls._normalize_image_items(
                    result.get("images", []),
                    source_url=str(result.get("url") or ""),
                    source_title=str(result.get("title") or ""),
                )
            )
        linked_images = cls._dedupe_image_items(linked_images)
        if linked_images:
            return linked_images[:max_images]

        return cls._normalize_image_items(response.get("images", []))[:max_images]

    def _fetch_tavily_images(self, query: str, max_images: int = 6) -> List[Dict[str, str]]:
        """Fetch image tiles for a search result using Tavily's REST API."""
        if not web_images_enabled() or not self.tavily_key or is_explicit_query(query):
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
        return self._collect_tavily_images(response, max_images=max_images)

    def _start_image_prefetch(self, query: str, max_images: int = 6):
        """Start the Tavily image fetch on a thread, concurrent with a text search.

        Returns ``(thread, holder)`` or ``(None, None)`` when images are
        unavailable (no key, images disabled, blocked query).
        """
        if not self.tavily_key or not web_images_enabled() or is_explicit_query(query):
            return None, None
        holder: Dict[str, Any] = {}

        def _fetch():
            try:
                holder["images"] = self._fetch_tavily_images(query, max_images=max_images)
            except Exception as e:
                print(f"[SEARCH ROUTER] Tavily image prefetch failed: {e}")
                holder["images"] = []

        thread = threading.Thread(
            target=_fetch, name="quasar-image-prefetch", daemon=True
        )
        thread.start()
        return thread, holder

    @staticmethod
    def _attach_prefetched_images(
        result: Dict[str, Any],
        img_thread: Optional[threading.Thread],
        img_holder: Optional[Dict[str, Any]],
        wait_seconds: float = 4.0,
    ) -> Dict[str, Any]:
        """Attach prefetched image tiles to a text-only result (bounded wait)."""
        if (
            img_thread is None
            or not isinstance(result, dict)
            or not result.get("success")
            or result.get("images")
        ):
            return result
        img_thread.join(timeout=wait_seconds)
        images = (img_holder or {}).get("images") or []
        if not images:
            return result
        enriched = dict(result)
        enriched["images"] = images[:6]
        enriched["image_provider"] = "Tavily Images"
        return enriched

    # Statuses that mean the account/service is refusing us — worth a cooldown,
    # unlike a 200 with empty results.
    _BRAVE_HARD_STATUSES = {401, 402, 403, 422, 429, 500, 502, 503, 504}

    def search_brave(self, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> Dict[str, Any]:
        """Perform search using Brave LLM Context (primary RAG) or Web Search."""
        if is_explicit_query(query):
            return blocked_query_result(query)
        if not self.brave_key:
            return {"success": False, "error": "Brave API key missing"}
        if self._brave_cooling_down():
            return {"success": False, "error": "Brave circuit breaker open"}
        max_results = max(1, min(int(max_results), 10))

        # Track failure modes across both endpoints; only a hard failure on
        # BOTH trips the breaker. Brave answers in ~2 s when healthy, so 5 s
        # is generous — the old 10 s doubled the worst-case dead time.
        _hard_failures = []

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
            response = requests.get(url, headers=headers, params=params, timeout=5)
            if response.status_code == 200:
                self._increment_usage("brave")
                data = response.json()
                results = self._parse_brave_llm_context(data, max_results=max_results)
                if results:
                    return sanitize_web_payload({
                        "success": True,
                        "provider": "Brave LLM Context",
                        "query": query,
                        "results": results,
                        "images": []
                    })
            elif response.status_code in self._BRAVE_HARD_STATUSES:
                _hard_failures.append(f"llm/context HTTP {response.status_code}")
        except Exception as e:
            _hard_failures.append(f"llm/context {type(e).__name__}")
            print(f"[SEARCH ROUTER] Brave LLM Context API call failed: {e}")

        # Fallback to standard Brave Web Search
        web_url = "https://api.search.brave.com/res/v1/web/search"
        try:
            web_response = requests.get(
                web_url,
                headers=headers,
                params={"q": query, "count": min(max_results, 10), "safesearch": "strict"},
                timeout=5,
            )
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
                return sanitize_web_payload({
                    "success": True,
                    "provider": "Brave Web Search",
                    "query": query,
                    "results": results[:max_results],
                    "images": []
                })
            if web_response.status_code in self._BRAVE_HARD_STATUSES:
                _hard_failures.append(f"web/search HTTP {web_response.status_code}")
        except Exception as e:
            _hard_failures.append(f"web/search {type(e).__name__}")
            print(f"[SEARCH ROUTER] Brave Web Search API call failed: {e}")

        if len(_hard_failures) >= 2:
            self._trip_brave_breaker("; ".join(_hard_failures))

        return {"success": False, "error": "Brave API request failed"}

    def search_tavily(
        self,
        query: str,
        max_results: int = DEFAULT_MAX_RESULTS,
        search_depth: str = "basic",
        include_images: bool = True,
        include_answer: bool = False,
    ) -> Dict[str, Any]:
        """Perform Tavily search with source and optional image metadata.

        ``include_answer`` defaults OFF: Tavily's generated answer roughly
        doubled search latency (measured 7.2 s vs 3.4 s, 2026-07-18) and
        Quasar synthesizes its own summary from the snippets anyway — the raw
        answer was only a no-snippets fallback.
        """
        if is_explicit_query(query):
            return blocked_query_result(query)
        if not self.tavily_key:
            return {"success": False, "error": "Tavily API key missing"}

        max_results = max(1, min(int(max_results), 10))
        include_images = bool(include_images and web_images_enabled())
        payload = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": bool(include_answer),
            "include_images": include_images,
            "include_image_descriptions": include_images,
        }

        try:
            try:
                client = self._get_tavily_client()
                response = client.search(**payload)
            except Exception as sdk_error:
                print(f"[SEARCH ROUTER] Tavily SDK search unavailable, using REST: {sdk_error}")
                response = self._tavily_post("search", payload, timeout=25)

            if isinstance(response, dict) and response.get("success") is False:
                return response

            response_results = response.get("results", []) if isinstance(response, dict) else []
            results = []
            for r in response_results:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", "")
                })
            images = self._collect_tavily_images(response, max_images=6)
            return sanitize_web_payload({
                "success": True,
                "provider": "Tavily",
                "query": query,
                "answer": response.get("answer", "") if isinstance(response, dict) else "",
                "results": results,
                "images": images[:6]
            })
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
        original_url_count = len(clean_urls)
        clean_urls = [url for url in clean_urls if not is_blocked_url(url)]
        blocked_url_count = original_url_count - len(clean_urls)
        if not clean_urls:
            return {
                "success": False,
                "error": (
                    "The requested URL was blocked by the web safety filter."
                    if blocked_url_count
                    else "web_extract_url requires at least one full http(s) URL. Use web_search for keyword queries."
                ),
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
            return sanitize_web_payload({
                "success": True,
                "provider": "Tavily Extract",
                "urls": clean_urls,
                "response": response,
                "results": response.get("results", []) if isinstance(response, dict) else response,
            })
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
        if is_blocked_url(url):
            return {"success": False, "error": "The requested URL was blocked by the web safety filter."}
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
            return sanitize_web_payload({
                "success": True,
                "provider": "Tavily Map",
                "url": url,
                "response": response,
                "results": response.get("results", []) if isinstance(response, dict) else response,
            })
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
        if is_blocked_url(url):
            return {"success": False, "error": "The requested URL was blocked by the web safety filter."}
        if not self.tavily_key:
            return {"success": False, "error": "Tavily API key missing"}

        kwargs: Dict[str, Any] = {
            "max_depth": max(1, min(int(max_depth), 5)),
            "max_breadth": max(1, min(int(max_breadth), 100)),
            "limit": max(1, min(int(limit), 50)),
            "extract_depth": extract_depth,
            "format": content_format,
            "include_images": bool(include_images and web_images_enabled()),
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
            return sanitize_web_payload({
                "success": True,
                "provider": "Tavily Crawl",
                "url": url,
                "response": response,
                "results": response.get("results", []) if isinstance(response, dict) else response,
            })
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
        if is_explicit_query(research_input):
            return blocked_query_result(research_input, provider="Tavily Research")
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
                return sanitize_web_payload({
                    "success": True,
                    "provider": "Tavily Research",
                    "status": response.get("status") if isinstance(response, dict) else None,
                    "request_id": response.get("request_id") if isinstance(response, dict) else None,
                    "response": response,
                })

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
            return sanitize_web_payload({
                "success": response.get("status") != "failed",
                "provider": "Tavily Research",
                "status": response.get("status"),
                "request_id": response.get("request_id"),
                "content": response.get("content"),
                "sources": response.get("sources", []),
                "response": response,
            })
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
            return sanitize_web_payload({
                "success": response.get("status") != "failed",
                "provider": "Tavily Research",
                "status": response.get("status"),
                "request_id": response.get("request_id", request_id),
                "content": response.get("content"),
                "sources": response.get("sources", []),
                "response": response,
            })
        except Exception as e:
            return {"success": False, "error": f"Tavily research status failed: {e}"}

    def search_exa(self, query: str, num_results: int = DEFAULT_MAX_RESULTS, search_type: str = "deep") -> Dict[str, Any]:
        """Perform semantic research search using Exa."""
        if is_explicit_query(query):
            return blocked_query_result(query)
        if not self.exa_key:
            return {"success": False, "error": "Exa API key missing"}
        num_results = max(1, min(int(num_results), 10))
            
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
                "numResults": num_results,
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
                return sanitize_web_payload({
                    "success": True,
                    "provider": "Exa",
                    "search_type": search_type,
                    "query": query,
                    "results": results,
                    "images": []
                })
            return {"success": False, "error": f"Exa returned status code {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": f"Exa search failed: {e}"}

    def search_images(self, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> Dict[str, Any]:
        """Specialized image-only search leveraging Tavily or Brave's visual index."""
        if is_explicit_query(query):
            return blocked_query_result(query)
        if not web_images_enabled():
            return {
                "success": True,
                "provider": "Safety filter",
                "query": query,
                "answer": "General web image tiles are disabled by the safety policy.",
                "results": [],
                "images": [],
                "content_filter": {
                    "enabled": True,
                    "filtered": True,
                    "blocked_query": False,
                    "blocked_results": 0,
                    "blocked_images": 0,
                    "blocked_text_fields": 0,
                    "notice": "General web image tiles are disabled by the safety policy.",
                    "web_images_enabled": False,
                },
            }
        max_results = max(1, min(int(max_results), 10))
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

                images = self._collect_tavily_images(response, max_images=max_results)
                
                # Also collect web references to go alongside images
                results = []
                for r in response.get("results", []):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("content", "")
                    })
                    
                if images:
                    return sanitize_web_payload({
                        "success": True,
                        "provider": "Tavily Images",
                        "query": query,
                        "results": results,
                        "images": images[:6]
                    })
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
                response = requests.get(
                    url,
                    headers=headers,
                    params={"q": query, "count": 10, "safesearch": "strict"},
                    timeout=10,
                )
                if response.status_code == 200:
                    self._increment_usage("brave")
                    data = response.json()
                    images = []
                    # Extrapolate any image objects or thumbnails in Brave results
                    for item in data.get("web", {}).get("results", []):
                        thumbnail = item.get("thumbnail", {})
                        if thumbnail and thumbnail.get("src"):
                            image = {
                                "url": thumbnail.get("src"),
                                "description": item.get("title", "")
                            }
                            source_url = self._normalize_source_url(item.get("url"))
                            if source_url:
                                image["sourceUrl"] = source_url
                            source_title = str(item.get("title") or "").strip()
                            if source_title:
                                image["sourceTitle"] = source_title
                            images.append(image)
                    
                    results = []
                    for item in data.get("web", {}).get("results", []):
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "snippet": item.get("snippet", "")
                        })
                        
                    return sanitize_web_payload({
                        "success": True,
                        "provider": "Brave Images (extrapolated)",
                        "query": query,
                        "results": results[:max_results],
                        "images": images[:6]
                    })
            except Exception as e:
                print(f"[SEARCH ROUTER] Brave image search failed: {e}")

        return {"success": False, "error": "No image search providers succeeded"}
