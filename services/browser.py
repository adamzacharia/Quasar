"""
BrowserService — Playwright-powered web automation for Quasar AI
Gives the LLM the ability to navigate web pages, search the web, 
and extract content from astronomy portals with no public API.

Exposed as agent tools:
    - web_search(query)           → Google/DuckDuckGo search results
    - navigate_to_url(url)        → Load a URL and return page text
    - read_page()                 → Get text content of current page
    - click_element(selector)     → Click a CSS selector on current page
"""

import asyncio
import re
from typing import Optional

# Playwright is available via the playwright package installed with npx
try:
    from playwright.sync_api import sync_playwright, Page, Browser
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("[WARNING] playwright Python library not found. Install with: pip install playwright")


class BrowserService:
    """Synchronous Playwright browser wrapper for use inside agent tool calls."""

    def __init__(self):
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
        self._available = PLAYWRIGHT_AVAILABLE

    def _ensure_browser(self):
        """Lazily launch the browser on first use."""
        if not self._available:
            raise RuntimeError("Playwright is not installed. Run: conda run -n quasar pip install playwright && playwright install chromium")
        if self._browser is None:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            self._page = self._browser.new_page()
            # Set a realistic user agent
            self._page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })

    def _extract_text(self) -> str:
        """Extract clean text content from the current page."""
        if not self._page:
            return ""
        try:
            # Remove script/style/nav tags first for cleaner output
            text = self._page.evaluate("""
                () => {
                    const body = document.body.cloneNode(true);
                    ['script', 'style', 'nav', 'footer', 'header', 'aside'].forEach(tag => {
                        body.querySelectorAll(tag).forEach(el => el.remove());
                    });
                    return body.innerText;
                }
            """)
            # Collapse whitespace
            text = re.sub(r'\n{3,}', '\n\n', text.strip())
            return text[:8000]  # Cap at 8k chars to stay within token limits
        except Exception as e:
            return f"[Error extracting text: {e}]"

    def web_search(self, query: str, engine: str = "duckduckgo") -> dict:
        """
        Perform a web search and return the top results as text.
        
        Args:
            query: Search query string (e.g., "ALMA observations of M87")
            engine: Search engine to use (default: "duckduckgo")
        
        Returns:
            dict with 'results' list of title/url/snippet
        """
        try:
            self._ensure_browser()
            search_url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}&ia=web"
            self._page.goto(search_url, wait_until="domcontentloaded", timeout=15000)
            self._page.wait_for_timeout(1500)  # Let JS render

            # Extract result links and snippets
            results = self._page.evaluate("""
                () => {
                    const items = Array.from(document.querySelectorAll('[data-result="snippet"]'));
                    return items.slice(0, 5).map(item => {
                        const title = item.querySelector('h2')?.innerText || '';
                        const link = item.querySelector('a')?.href || '';
                        const snippet = item.innerText.replace(title, '').trim().substring(0, 200);
                        return { title, url: link, snippet };
                    });
                }
            """)

            # Fallback to raw text if JS path fails
            if not results:
                raw_text = self._extract_text()
                return {"query": query, "engine": engine, "raw_text": raw_text[:3000], "results": []}

            return {"query": query, "engine": engine, "results": results}
        except Exception as e:
            return {"error": str(e), "query": query}

    def navigate_to_url(self, url: str) -> dict:
        """
        Navigate to a URL and return the page's text content.
        
        Args:
            url: Full URL to navigate to (e.g., "https://almascience.nrao.edu")
        
        Returns:
            dict with 'url', 'title', and 'content'
        """
        try:
            self._ensure_browser()
            self._page.goto(url, wait_until="domcontentloaded", timeout=20000)
            self._page.wait_for_timeout(1000)
            title = self._page.title()
            content = self._extract_text()
            return {"url": url, "title": title, "content": content}
        except Exception as e:
            return {"error": str(e), "url": url}

    def read_page(self) -> dict:
        """
        Read the text content of the currently loaded page.
        
        Returns:
            dict with the current page 'url', 'title', and 'content'
        """
        try:
            self._ensure_browser()
            if not self._page:
                return {"error": "Browser not started. Call navigate_to_url first."}
            url = self._page.url
            title = self._page.title()
            content = self._extract_text()
            return {"url": url, "title": title, "content": content}
        except Exception as e:
            return {"error": str(e)}

    def click_element(self, selector: str, wait_after_ms: int = 1500) -> dict:
        """
        Click an element on the current page by CSS selector.
        
        Args:
            selector: CSS selector for the element to click (e.g., "button.search-btn")
            wait_after_ms: Milliseconds to wait after clicking for the page to update
        
        Returns:
            dict with the page content after the click
        """
        try:
            self._ensure_browser()
            if not self._page:
                return {"error": "Browser not started. Call navigate_to_url first."}
            self._page.click(selector, timeout=5000)
            self._page.wait_for_timeout(wait_after_ms)
            content = self._extract_text()
            return {"clicked": selector, "page_content_after": content}
        except Exception as e:
            return {"error": str(e), "selector": selector}

    def close(self):
        """Clean up playwright resources."""
        try:
            if self._page:
                self._page.close()
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        self._page = None
        self._browser = None
        self._playwright = None
