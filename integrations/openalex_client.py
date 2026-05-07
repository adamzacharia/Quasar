"""OpenAlex scholarly data integration for Quasar.

Complementary data source to NASA ADS — used for:
  • Researcher / author profile lookups (h-index, institution, ORCID, topics)
  • Silent enrichment of ADS paper results (funding data, citation percentiles, OA PDFs)
  • Bibliometric trend aggregation (papers-per-year, country breakdowns)

ADS remains the PRIMARY literature engine for astronomy.  OpenAlex fills gaps
ADS cannot: funding intelligence, institution analytics, author profiles, and
cross-disciplinary discovery across 270 M+ works.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OpenAlexError(Exception):
    """Base exception for OpenAlex service errors."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OpenAlexService:
    """Lightweight REST client for the OpenAlex API (https://api.openalex.org)."""

    BASE_URL = "https://api.openalex.org"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 12.0,
        retry_attempts: int = 3,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENALEX_API_KEY", "")
        self.timeout = timeout
        self.retry_attempts = max(1, retry_attempts)
        self.session = session or requests.Session()

        if not self.api_key:
            logger.warning(
                "OPENALEX_API_KEY not set — requests will be heavily rate-limited."
            )

    # ------------------------------------------------------------------
    # 1.  Researcher / Author Lookup
    # ------------------------------------------------------------------

    def search_authors(
        self,
        query: str,
        *,
        max_results: int = 5,
        filters: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for researchers by name, returning rich profile data.

        Example:
            >>> svc.search_authors("Andrea Isella")
        """
        params: Dict[str, Any] = {
            "search": query,
            "per_page": min(max_results, 100),
        }
        if filters:
            params["filter"] = ",".join(f"{k}:{v}" for k, v in filters.items())

        data = self._get("/authors", params)
        return [self._format_author(a) for a in data.get("results", [])]

    def get_author(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Fetch a single author by OpenAlex ID or ORCID.

        Args:
            identifier: OpenAlex ID (``A5023888391``), full ORCID URL, or
                bare ORCID (``0000-0001-2345-6789``).
        """
        # Normalise bare ORCIDs
        if identifier.replace("-", "").isdigit() and len(identifier) >= 16:
            identifier = f"https://orcid.org/{identifier}"

        try:
            data = self._get(f"/authors/{identifier}", {})
        except OpenAlexError:
            return None

        if not data or "id" not in data:
            return None
        return self._format_author(data)

    def get_author_works(
        self,
        author_id: str,
        *,
        max_results: int = 20,
        sort: str = "cited_by_count:desc",
    ) -> List[Dict[str, Any]]:
        """Fetch top works for a given OpenAlex author ID.

        Args:
            author_id: OpenAlex author ID (e.g. ``A5023888391``).
        """
        # Ensure bare ID
        if author_id.startswith("https://"):
            author_id = author_id.rsplit("/", 1)[-1]

        params: Dict[str, Any] = {
            "filter": f"authorships.author.id:{author_id}",
            "sort": sort,
            "per_page": min(max_results, 100),
        }
        data = self._get("/works", params)
        return [self._format_work(w) for w in data.get("results", [])]

    # ------------------------------------------------------------------
    # 2.  Silent Enrichment Helpers
    # ------------------------------------------------------------------

    def enrich_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """Look up a work by DOI and return enrichment data.

        Returns funding info, citation percentile, OA PDF URL, and topics
        that ADS does not provide.
        """
        # Normalise DOI
        doi = doi.strip()
        if doi.startswith("http"):
            doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")

        try:
            data = self._get(f"/works/doi:{doi}", {})
        except OpenAlexError:
            return None

        if not data or "id" not in data:
            return None

        return self._format_enrichment(data)

    def enrich_batch_dois(self, dois: List[str]) -> Dict[str, Dict[str, Any]]:
        """Batch-enrich up to 50 DOIs in a single request.

        Returns a dict mapping DOI → enrichment data.
        """
        # Clean DOIs
        clean = []
        for d in dois[:50]:
            d = d.strip()
            if d.startswith("http"):
                d = d.replace("https://doi.org/", "").replace("http://doi.org/", "")
            clean.append(d)

        if not clean:
            return {}

        doi_filter = "|".join(clean)
        params: Dict[str, Any] = {
            "filter": f"doi:{doi_filter}",
            "per_page": len(clean),
        }

        try:
            data = self._get("/works", params)
        except OpenAlexError:
            return {}

        result = {}
        for w in data.get("results", []):
            raw_doi = (w.get("doi") or "").replace("https://doi.org/", "")
            if raw_doi:
                result[raw_doi] = self._format_enrichment(w)
        return result

    # ------------------------------------------------------------------
    # 3.  Bibliometric Trends
    # ------------------------------------------------------------------

    def get_topic_trends(
        self,
        search_query: str,
        *,
        year_from: int = 2015,
        year_to: int = 2026,
    ) -> Dict[str, Any]:
        """Get papers-per-year trend for a topic search.

        Returns: ``{"query": ..., "total": ..., "by_year": [{year, count}, ...]}``
        """
        params: Dict[str, Any] = {
            "search": search_query,
            "filter": f"publication_year:{year_from}-{year_to}",
            "group_by": "publication_year",
        }
        data = self._get("/works", params)

        by_year = []
        for g in data.get("group_by", []):
            by_year.append({
                "year": int(g.get("key", 0)),
                "count": g.get("count", 0),
            })
        by_year.sort(key=lambda x: x["year"])

        return {
            "query": search_query,
            "total": data.get("meta", {}).get("count", 0),
            "year_range": f"{year_from}-{year_to}",
            "by_year": by_year,
        }

    def get_funding_landscape(
        self,
        search_query: str,
        *,
        max_results: int = 20,
    ) -> List[Dict[str, Any]]:
        """Find funding/grant information for works matching a search query.

        Returns works with their associated funders and award details.
        """
        params: Dict[str, Any] = {
            "search": search_query,
            "filter": "has_doi:true",
            "sort": "cited_by_count:desc",
            "per_page": min(max_results, 100),
        }
        data = self._get("/works", params)

        funded_works = []
        for w in data.get("results", []):
            awards = w.get("awards") or []
            funders = w.get("funders") or []
            if awards or funders:
                funded_works.append({
                    "title": w.get("display_name", ""),
                    "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                    "year": w.get("publication_year"),
                    "cited_by_count": w.get("cited_by_count", 0),
                    "funders": [
                        {
                            "name": f.get("display_name", ""),
                            "id": f.get("id", ""),
                        }
                        for f in funders
                    ],
                    "awards": [
                        {
                            "funder": a.get("funder_display_name", ""),
                            "award_id": a.get("funder_award_id", ""),
                        }
                        for a in awards
                    ],
                })
        return funded_works

    # ------------------------------------------------------------------
    # Formatters (private)
    # ------------------------------------------------------------------

    def _format_author(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Format a raw OpenAlex author object into a clean profile."""
        oa_id = (raw.get("id") or "").rsplit("/", 1)[-1]
        orcid = raw.get("orcid") or ""
        if orcid.startswith("https://orcid.org/"):
            orcid = orcid.replace("https://orcid.org/", "")

        # Current institution(s)
        institutions = []
        for inst in raw.get("last_known_institutions") or []:
            institutions.append({
                "name": inst.get("display_name", ""),
                "ror": inst.get("ror", ""),
                "country": inst.get("country_code", ""),
                "type": inst.get("type", ""),
            })

        # Affiliation history
        affiliations = []
        for aff in (raw.get("affiliations") or [])[:10]:
            inst = aff.get("institution", {})
            years = aff.get("years", [])
            affiliations.append({
                "institution": inst.get("display_name", ""),
                "country": inst.get("country_code", ""),
                "years": f"{min(years)}-{max(years)}" if years else "",
            })

        # Summary stats
        stats = raw.get("summary_stats") or {}

        # Top topics
        topics = []
        for t in (raw.get("topics") or [])[:8]:
            topics.append({
                "name": t.get("display_name", ""),
                "count": t.get("count", 0),
            })

        # Publication history (last 10 years)
        counts_by_year = []
        for c in (raw.get("counts_by_year") or [])[:10]:
            counts_by_year.append({
                "year": c.get("year"),
                "works": c.get("works_count", 0),
                "citations": c.get("cited_by_count", 0),
            })
        counts_by_year.sort(key=lambda x: x["year"], reverse=True)

        return {
            "name": raw.get("display_name", ""),
            "name_alternatives": raw.get("display_name_alternatives", []),
            "orcid": orcid,
            "works_count": raw.get("works_count", 0),
            "cited_by_count": raw.get("cited_by_count", 0),
            "h_index": stats.get("h_index", 0),
            "i10_index": stats.get("i10_index", 0),
            "mean_citedness_2yr": round(stats.get("2yr_mean_citedness", 0), 2),
            "current_institutions": institutions,
            "affiliation_history": affiliations,
            "top_topics": topics,
            "publication_history": counts_by_year,
        }

    def _format_work(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Format a raw OpenAlex work into a clean paper dict."""
        doi = (raw.get("doi") or "").replace("https://doi.org/", "")
        authors = []
        for auth in (raw.get("authorships") or [])[:5]:
            a = auth.get("author", {})
            insts = [i.get("display_name", "") for i in (auth.get("institutions") or [])[:2]]
            authors.append({
                "name": a.get("display_name", ""),
                "institution": insts[0] if insts else "",
            })

        return {
            "title": raw.get("display_name", ""),
            "doi": doi,
            "year": raw.get("publication_year"),
            "cited_by_count": raw.get("cited_by_count", 0),
            "type": raw.get("type", ""),
            "is_oa": raw.get("is_oa", False),
            "authors": authors,
        }

    def _format_enrichment(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Extract enrichment fields that ADS lacks (funding, percentiles, OA)."""
        # Citation percentile
        percentile = raw.get("citation_normalized_percentile") or {}
        fwci = raw.get("fwci")

        # Open access PDF
        best_oa = raw.get("best_oa_location") or {}
        pdf_url = best_oa.get("pdf_url") or ""

        # Funding — filter to known science/research agencies to avoid noisy
        # OpenAlex NLP extraction (e.g. "European School of Oncology" on astro papers)
        _KNOWN_FUNDER_KEYWORDS = {
            "nsf", "nasa", "esa", "nih", "doe", "darpa", "erc", "ukri",
            "science foundation", "research council", "research foundation",
            "national science", "space agency", "astronomy", "astrophysic",
            "science and technology", "natural sciences", "engineering research",
            "ministry of science", "ministry of education", "european commission",
            "max planck", "helmholtz", "cnrs", "anr", "dfg", "jsps", "nserc",
            "arc ", "australian research", "chinese academy", "cas ",
            "national natural science", "simons foundation", "kavli",
            "moore foundation", "packard", "sloan foundation", "hubble",
            "chandra", "spitzer", "jwst", "alma", "nrao", "noao", "gemini",
            "keck", "subaru", "eso ", "european southern", "stsci",
            "smithsonian", "carnegie", "caltech",
        }
        funders = []
        for f in (raw.get("funders") or []):
            name = f.get("display_name", "")
            name_lower = name.lower()
            # Only include if the funder name matches a known keyword
            if any(kw in name_lower for kw in _KNOWN_FUNDER_KEYWORDS):
                funders.append({
                    "name": name,
                    "id": f.get("id", ""),
                })
        awards = []
        for a in (raw.get("awards") or []):
            awards.append({
                "funder": a.get("funder_display_name", ""),
                "award_id": a.get("funder_award_id", ""),
            })

        # Topics
        topics = [t.get("display_name", "") for t in (raw.get("topics") or [])[:5]]

        return {
            "fwci": round(fwci, 2) if fwci else None,
            "citation_percentile": round(percentile.get("value", 0), 2) if percentile else None,
            "is_top_1_percent": percentile.get("is_in_top_1_percent", False),
            "is_top_10_percent": percentile.get("is_in_top_10_percent", False),
            "oa_pdf_url": pdf_url,
            "funders": funders,
            "awards": awards,
            "topics": topics,
            "openalex_id": (raw.get("id") or "").rsplit("/", 1)[-1],
        }

    # ------------------------------------------------------------------
    # HTTP transport (private)
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a GET request with retries and rate-limit handling."""
        url = f"{self.BASE_URL}{endpoint}"

        if self.api_key:
            params["api_key"] = self.api_key

        delay = 1.0
        for attempt in range(self.retry_attempts):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.Timeout:
                if attempt == self.retry_attempts - 1:
                    raise OpenAlexError("OpenAlex request timed out")
                time.sleep(delay)
                delay *= 2
                continue
            except requests.RequestException as exc:
                if attempt == self.retry_attempts - 1:
                    raise OpenAlexError(f"OpenAlex request error: {exc}") from exc
                time.sleep(delay)
                delay *= 2
                continue

            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", delay))
                logger.warning(
                    "OpenAlex rate limit (attempt %d); retrying in %.1f s",
                    attempt + 1, wait,
                )
                time.sleep(wait)
                delay = min(delay * 2, 30)
                continue

            if resp.status_code >= 400:
                snippet = resp.text.strip()[:500]
                raise OpenAlexError(
                    f"OpenAlex API error {resp.status_code}: {snippet or 'No details'}"
                )

            try:
                return resp.json()
            except ValueError as exc:
                raise OpenAlexError("OpenAlex returned invalid JSON") from exc

        raise OpenAlexError("Exceeded retry attempts for OpenAlex API")
