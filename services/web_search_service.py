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
                res = self.search_exa(query, num_results=max_results)
                if res.get("success"):
                    return res
                print("[SEARCH ROUTER] Exa failed, falling back to Brave/Tavily")
            elif self.exa_key:
                print("[SEARCH ROUTER] Exa monthly free limit (1000) reached. Falling back.")

        # 3. Fresh / Current / RAG fresh info -> route to Brave Search
        usage = self._get_usage()
        if self.brave_key and usage.get("brave_count", 0) < MAX_FREE_LIMIT:
            print(f"[SEARCH ROUTER] Routed to Brave (Primary) for: {query!r}")
            res = self.search_brave(query, max_results=max_results)
            if res.get("success"):
                return res
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
                return {
                    "success": True,
                    "provider": "BrowserService (fallback)",
                    "query": query,
                    "results": fallback,
                    "images": []
                }
            except Exception as e:
                return {"success": False, "error": f"Scraping fallback failed: {e}"}
                
        return {"success": False, "error": "No search providers available or all quotas exceeded"}

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
                results = []
                # Brave LLM Context returns a flat list of excerpts/snippets
                for item in data if isinstance(data, list) else data.get("context", []):
                    results.append({
                        "title": item.get("title", "Brave Snippet"),
                        "url": item.get("url", ""),
                        "snippet": item.get("snippet", "")
                    })
                return {
                    "success": True,
                    "provider": "Brave LLM Context",
                    "query": query,
                    "results": results[:max_results],
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

    def search_tavily(self, query: str, max_results: int = 5, search_depth: str = "basic") -> Dict[str, Any]:
        """Perform search using Tavily client."""
        if not self.tavily_key:
            return {"success": False, "error": "Tavily API key missing"}
            
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=self.tavily_key)
            response = client.search(
                query=query,
                max_results=min(int(max_results), 10),
                search_depth=search_depth,
                include_answer=True
            )
            results = []
            for r in response.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", "")
                })
            return {
                "success": True,
                "provider": "Tavily",
                "query": query,
                "answer": response.get("answer", ""),
                "results": results,
                "images": []
            }
        except Exception as e:
            return {"success": False, "error": f"Tavily search failed: {e}"}

    def search_exa(self, query: str, num_results: int = 5) -> Dict[str, Any]:
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
                "type": "auto",
                "numResults": min(num_results, 10),
                "contents": {
                    "highlights": True
                }
            }
            response = requests.post(url, headers=headers, json=payload, timeout=10)
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
                        "snippet": snippet
                    })
                return {
                    "success": True,
                    "provider": "Exa",
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
                from tavily import TavilyClient
                client = TavilyClient(api_key=self.tavily_key)
                response = client.search(
                    query=query,
                    max_results=min(max_results, 10),
                    include_images=True,
                    include_image_descriptions=True
                )
                images = []
                for img in response.get("images", []):
                    if isinstance(img, dict):
                        images.append({
                            "url": img.get("url", ""),
                            "description": img.get("description", "")
                        })
                    elif isinstance(img, str):
                        images.append({"url": img, "description": ""})
                
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
