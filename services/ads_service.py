"""
NASA ADS Integration Service — Paper search via NASA ADS API.

CALLED BY: core/agent.py (tool: search_papers → _search_papers)
CALLS:     NASA ADS REST API (https://api.adsabs.harvard.edu/v1)
           OpenAI API (via ADSQueryBuilder for natural language → ADS query)

KEY CLASSES:
    ADSService      — Main service: search_papers(), search_natural_language(),
                      search_by_target(), search_by_facility(), search_by_frequency()
    ADSQueryBuilder — LLM-backed helper that converts free-text questions into
                      structured ADS query syntax (e.g. object:"M87" AND ALMA)

FALLBACK: If no NASA_ADS_API_KEY is set, returns hardcoded example papers.
"""

from __future__ import annotations

import json
import os
import time
import logging
from typing import Any, Dict, List, Optional, Union

import requests

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

logger = logging.getLogger(__name__)


class ADSServiceError(Exception):
    """Base exception for ADS service errors."""


class ADSQueryBuilderError(ADSServiceError):
    """Raised when natural-language query building fails."""


class ADSService:
    """Service for interacting with the NASA ADS API."""

    _DEFAULT_FIELDS = [
        "title",
        "author",
        "year",
        "bibcode",
        "abstract",
        "citation_count",
        "pub",
        "doi",
        "identifier",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 10.0,
        retry_attempts: int = 3,
        session: Optional[requests.Session] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("NASA_ADS_API_KEY", "")
        self.base_url = base_url or os.getenv(
            "NASA_ADS_BASE_URL", "https://api.adsabs.harvard.edu/v1"
        )
        self.timeout = timeout
        self.retry_attempts = max(1, retry_attempts)
        self.session = session or requests.Session()
        self.user_agent = user_agent or os.getenv(
            "NASA_ADS_USER_AGENT",
            "QuasarADSClient/1.0 (+https://github.com/nraoai/quasar)",
        )

        if not self.api_key:
            logger.warning(
                "NASA ADS API key not found. Paper search will use fallback examples."
            )

    # ------------------------------------------------------------------
    # Public search helpers
    # ------------------------------------------------------------------
    def search_papers(
        self,
        query: str,
        max_results: int = 10,
        sort: str = "date desc",
        fields: Optional[List[str]] = None,
        filters: Optional[List[str]] = None,
        start: int = 0,
    ) -> List[Dict[str, Any]]:
        """Search for papers in NASA ADS."""

        if not query:
            logger.warning("Empty ADS query received; returning no papers.")
            return []

        if not self.api_key:
            return self._get_example_papers(query)

        fields = fields or self._DEFAULT_FIELDS
        rows = max(1, min(max_results, 200))  # ADS hard limit 200 per request

        params: Dict[str, Union[str, int, List[str]]] = {
            "q": query,
            "fl": ",".join(fields),
            "rows": rows,
            "sort": sort,
            "start": max(0, start),
        }

        if filters:
            params["fq"] = filters

        try:
            data = self._perform_get("/search/query", params)
        except ADSServiceError as exc:
            logger.error("ADS search failed: %s", exc)
            return self._get_example_papers(query)

        docs = data.get("response", {}).get("docs", [])
        return self._format_papers(docs)

    def search_natural_language(
        self,
        question: str,
        max_results: int = 10,
        sort: Optional[str] = None,
        filters: Optional[List[str]] = None,
        query_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert a natural-language question into an ADS query and execute it."""

        builder = ADSQueryBuilder(model=query_model)

        try:
            structured = builder.build_query(question, max_results, sort, filters)
        except ADSQueryBuilderError as exc:
            logger.warning(
                "Query builder failed (%s); falling back to heuristic query.", exc
            )
            structured = self._heuristic_query(question, max_results, sort, filters)

        ads_query = structured.get("query", "")
        resolved_sort = structured.get("sort", sort or "date desc")
        resolved_filters = structured.get("filters", filters)
        rows = int(structured.get("rows", max_results))

        papers = self.search_papers(
            ads_query,
            max_results=rows,
            sort=resolved_sort,
            filters=resolved_filters,
        )

        return {
            "query": ads_query,
            "sort": resolved_sort,
            "filters": resolved_filters,
            "rows": rows,
            "papers": papers,
        }
    
    def search_by_target(self, target_name: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search papers related to a specific astronomical target"""
        # Build query for astronomical target
        query = f'object:"{target_name}" OR title:"{target_name}" OR abstract:"{target_name}"'
        query += ' AND (radio OR VLA OR ALMA OR "Very Large Array" OR VLBA OR GBT)'
        
        return self.search_papers(query, max_results)
    
    def search_by_facility(self, facility: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search papers related to a specific radio facility"""
        facility_keywords = {
            'VLA': '("Very Large Array" OR VLA OR JVLA OR EVLA)',
            'ALMA': '(ALMA OR "Atacama Large Millimeter Array")',
            'VLBA': '(VLBA OR "Very Long Baseline Array")',
            'GBT': '(GBT OR "Green Bank Telescope")',
            'GMRT': '(GMRT OR "Giant Metrewave Radio Telescope")',
            'ASKAP': '(ASKAP OR "Australian Square Kilometre Array Pathfinder")',
            'MeerKAT': '(MeerKAT OR "Karoo Array Telescope")'
        }
        
        query = facility_keywords.get(facility.upper(), f'"{facility}"')
        query += ' AND (observation OR data OR survey OR catalog)'
        
        return self.search_papers(query, max_results, sort="date desc")
    
    def search_by_frequency(self, 
                          min_freq_ghz: float, 
                          max_freq_ghz: float,
                          max_results: int = 10) -> List[Dict[str, Any]]:
        """Search papers related to observations in a frequency range"""
        # Convert to common radio astronomy bands
        bands = []
        
        if min_freq_ghz <= 0.5 and max_freq_ghz >= 0.3:
            bands.append('"P-band"')
        if min_freq_ghz <= 1.5 and max_freq_ghz >= 1.0:
            bands.append('"L-band"')
        if min_freq_ghz <= 3.0 and max_freq_ghz >= 2.0:
            bands.append('"S-band"')
        if min_freq_ghz <= 8.0 and max_freq_ghz >= 4.0:
            bands.append('"C-band"')
        if min_freq_ghz <= 12.0 and max_freq_ghz >= 8.0:
            bands.append('"X-band"')
        if min_freq_ghz <= 18.0 and max_freq_ghz >= 12.0:
            bands.append('"Ku-band"')
        if min_freq_ghz <= 26.5 and max_freq_ghz >= 18.0:
            bands.append('"K-band"')
        if min_freq_ghz <= 40.0 and max_freq_ghz >= 26.5:
            bands.append('"Ka-band"')
        
        # Also include specific frequency mentions
        freq_terms = []
        if min_freq_ghz < 10:
            freq_terms.append(f'("{min_freq_ghz:.1f} GHz" OR "{max_freq_ghz:.1f} GHz")')
        else:
            freq_terms.append(f'("{min_freq_ghz:.0f} GHz" OR "{max_freq_ghz:.0f} GHz")')
        
        query_parts = []
        if bands:
            query_parts.append(f"({' OR '.join(bands)})")
        if freq_terms:
            query_parts.append(f"({' OR '.join(freq_terms)})")
        
        query = ' OR '.join(query_parts) if query_parts else f'"{min_freq_ghz:.1f}-{max_freq_ghz:.1f} GHz"'
        query += ' AND (radio OR observation OR survey)'
        
        return self.search_papers(query, max_results)
    
    def get_paper_details(self, bibcode: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific paper"""
        if not self.api_key:
            return None

        params = {
            "q": f"bibcode:{bibcode}",
            "fl": "title,author,year,bibcode,abstract,citation_count,pub,doi,identifier,keyword,aff",
            "rows": 1,
        }

        try:
            data = self._perform_get("/search/query", params)
        except ADSServiceError as exc:
            logger.error("Error getting paper details: %s", exc)
            return None

        docs = data.get("response", {}).get("docs", [])
        if docs:
            return self._format_paper_details(docs[0])
        return None

    # ------------------------------------------------------------------
    # ── nasa-ads-mcp equivalent tools ─────────────────────────────────
    # ------------------------------------------------------------------

    def get_author_papers(self, author: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Find all publications by a specific author (last, first format preferred)."""
        if not self.api_key:
            return []
        query = f'author:"{author}"'
        return self.search_papers(query, max_results=max_results, sort="date desc")

    def get_paper_metrics(self, bibcode: str) -> Dict[str, Any]:
        """Get citation count, read count, and impact metrics for a specific paper."""
        if not self.api_key:
            return {"error": "No API key configured"}
        try:
            data = self._perform_post(
                "/metrics",
                {"bibcodes": [bibcode], "types": ["citations", "reads"]}
            )
            citation_stats = data.get("citation stats", {})
            usage_stats = data.get("usage stats", {})
            return {
                "bibcode": bibcode,
                "total_citations": citation_stats.get("total number of citations", 0),
                "refereed_citations": citation_stats.get("total number of refereed citations", 0),
                "reads_last_90_days": usage_stats.get("recent number of reads", 0),
                "total_reads": usage_stats.get("total number of reads", 0),
                "normalized_citations": citation_stats.get("normalized number of citations", 0),
            }
        except ADSServiceError as exc:
            logger.error("Metrics fetch failed: %s", exc)
            return {"error": str(exc)}

    def get_author_metrics(self, author: str) -> Dict[str, Any]:
        """Get h-index, citation counts, and scholarly impact for an author."""
        if not self.api_key:
            return {"error": "No API key configured"}
        # First get author's bibcodes
        papers = self.get_author_papers(author, max_results=200)
        if not papers:
            return {"error": f"No papers found for author: {author}"}
        bibcodes = [p["bibcode"] for p in papers if p.get("bibcode")][:200]
        try:
            data = self._perform_post(
                "/metrics",
                {"bibcodes": bibcodes, "types": ["indicators", "citations"]}
            )
            indicators = data.get("indicators", {})
            citation_stats = data.get("citation stats", {})
            return {
                "author": author,
                "paper_count": len(bibcodes),
                "h_index": indicators.get("h", 0),
                "m_index": indicators.get("m", 0),
                "i10_index": indicators.get("i10", 0),
                "total_citations": citation_stats.get("total number of citations", 0),
                "refereed_citations": citation_stats.get("total number of refereed citations", 0),
                "normalized_h_index": indicators.get("normalized h", 0),
            }
        except ADSServiceError as exc:
            logger.error("Author metrics failed: %s", exc)
            return {"error": str(exc)}

    def export_bibtex(self, bibcodes: List[str]) -> str:
        """Export properly formatted BibTeX citations for one or more bibcodes."""
        if not self.api_key:
            return "% No API key configured"
        if not bibcodes:
            return "% No bibcodes provided"
        try:
            payload = {"bibcode": bibcodes}
            data = self._perform_post("/export/bibtex", payload)
            return data.get("export", "% No BibTeX returned")
        except ADSServiceError as exc:
            logger.error("BibTeX export failed: %s", exc)
            return f"% Export failed: {exc}"

    # ── Library Management ─────────────────────────────────────────────

    def list_libraries(self) -> List[Dict[str, Any]]:
        """List all personal ADS libraries for the authenticated user."""
        if not self.api_key:
            return []
        try:
            data = self._perform_get("/biblib/libraries", {})
            libraries = data.get("libraries", [])
            return [
                {
                    "id": lib.get("id", ""),
                    "name": lib.get("name", ""),
                    "description": lib.get("description", ""),
                    "num_documents": lib.get("num_documents", 0),
                    "date_created": lib.get("date_created", ""),
                    "public": lib.get("public", False),
                }
                for lib in libraries
            ]
        except ADSServiceError as exc:
            logger.error("List libraries failed: %s", exc)
            return []

    def get_library_papers(self, library_id: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Get papers from a specific personal ADS library."""
        if not self.api_key:
            return []
        try:
            data = self._perform_get(
                f"/biblib/libraries/{library_id}",
                {"rows": max_results, "fl": ",".join(self._DEFAULT_FIELDS)}
            )
            docs = data.get("documents", [])
            return self._format_papers(docs) if docs else []
        except ADSServiceError as exc:
            logger.error("Get library papers failed: %s", exc)
            return []

    def create_library(self, name: str, description: str = "", public: bool = False) -> Dict[str, Any]:
        """Create a new personal ADS library."""
        if not self.api_key:
            return {"error": "No API key configured"}
        try:
            data = self._perform_post(
                "/biblib/libraries",
                {"name": name, "description": description, "public": public, "bibcode": []}
            )
            return {"id": data.get("id", ""), "name": name, "success": True}
        except ADSServiceError as exc:
            logger.error("Create library failed: %s", exc)
            return {"error": str(exc)}

    def add_to_library(self, library_id: str, bibcodes: List[str]) -> Dict[str, Any]:
        """Add papers to an existing personal ADS library by bibcode."""
        if not self.api_key:
            return {"error": "No API key configured"}
        try:
            data = self._perform_post(
                f"/biblib/documents/{library_id}",
                {"bibcode": bibcodes, "action": "add"}
            )
            return {"success": True, "added": len(bibcodes), "response": data}
        except ADSServiceError as exc:
            logger.error("Add to library failed: %s", exc)
            return {"error": str(exc)}


    
    def _format_papers(self, papers: List[Dict]) -> List[Dict[str, Any]]:
        """Format raw ADS response to standardized format"""
        formatted = []
        
        for paper in papers:
            authors = paper.get('author', [])
            if len(authors) > 3:
                author_str = f"{authors[0]} et al."
            else:
                author_str = ", ".join(authors)
            
            formatted.append({
                'title': paper.get('title', ['Unknown'])[0],
                'authors': author_str,
                'year': paper.get('year', 'Unknown'),
                'journal': paper.get('pub', 'Unknown'),
                'bibcode': paper.get('bibcode', ''),
                'abstract': paper.get('abstract', 'No abstract available'),
                'citations': paper.get('citation_count', 0),
                'doi': paper.get('doi', [''])[0] if paper.get('doi') else '',
                'link': f"https://ui.adsabs.harvard.edu/abs/{paper.get('bibcode', '')}"
            })
        
        return formatted
    
    def _format_paper_details(self, paper: Dict) -> Dict[str, Any]:
        """Format detailed paper information"""
        return {
            'title': paper.get('title', ['Unknown'])[0],
            'authors': paper.get('author', []),
            'affiliations': paper.get('aff', []),
            'year': paper.get('year', 'Unknown'),
            'journal': paper.get('pub', 'Unknown'),
            'bibcode': paper.get('bibcode', ''),
            'abstract': paper.get('abstract', 'No abstract available'),
            'citations': paper.get('citation_count', 0),
            'keywords': paper.get('keyword', []),
            'doi': paper.get('doi', [''])[0] if paper.get('doi') else '',
            'identifiers': paper.get('identifier', []),
            'link': f"https://ui.adsabs.harvard.edu/abs/{paper.get('bibcode', '')}"
        }
    
    def _get_example_papers(self, query: str) -> List[Dict[str, Any]]:
        """Return example papers when ADS API is not available"""
        examples = [
            {
                'title': 'The Very Large Array Sky Survey (VLASS): Science Case and Survey Design',
                'authors': 'Lacy, M. et al.',
                'year': '2020',
                'journal': 'PASP',
                'bibcode': '2020PASP..132c5001L',
                'abstract': 'The Very Large Array Sky Survey (VLASS) is a synoptic, all-sky radio sky survey...',
                'citations': 250,
                'doi': '10.1088/1538-3873/ab63eb',
                'link': 'https://ui.adsabs.harvard.edu/abs/2020PASP..132c5001L'
            },
            {
                'title': 'ALMA Observations of Molecular Gas in High-Redshift Galaxies',
                'authors': 'Casey, C. M. et al.',
                'year': '2023',
                'journal': 'ApJ',
                'bibcode': '2023ApJ...950..123C',
                'abstract': 'We present ALMA observations of CO emission in a sample of high-redshift galaxies...',
                'citations': 45,
                'doi': '10.3847/1538-4357/acd123',
                'link': 'https://ui.adsabs.harvard.edu/abs/2023ApJ...950..123C'
            },
            {
                'title': 'Radio Properties of Active Galactic Nuclei: A VLA Survey',
                'authors': 'Smith, J. A. et al.',
                'year': '2024',
                'journal': 'MNRAS',
                'bibcode': '2024MNRAS.520..456S',
                'abstract': 'We present results from a comprehensive VLA survey of radio-loud AGN...',
                'citations': 12,
                'doi': '10.1093/mnras/stad789',
                'link': 'https://ui.adsabs.harvard.edu/abs/2024MNRAS.520..456S'
            }
        ]
        
        # Filter examples based on query keywords
        query_lower = query.lower()
        filtered = []
        
        for paper in examples:
            if any(keyword in paper['title'].lower() or keyword in paper['abstract'].lower() 
                   for keyword in query_lower.split()):
                filtered.append(paper)
        
        return filtered if filtered else examples[:2]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _perform_get(self, endpoint: str, params: Dict[str, Union[str, int, List[str]]]) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{endpoint}"
        headers = self._build_headers()
        delay = 1.0

        for attempt in range(self.retry_attempts):
            try:
                response = self.session.get(
                    url, headers=headers, params=params, timeout=self.timeout
                )
            except requests.Timeout as exc:
                if attempt == self.retry_attempts - 1:
                    raise ADSServiceError("ADS request timed out") from exc
                time.sleep(delay)
                delay *= 2
                continue
            except requests.RequestException as exc:
                if attempt == self.retry_attempts - 1:
                    raise ADSServiceError(f"ADS request error: {exc}") from exc
                time.sleep(delay)
                delay *= 2
                continue

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_time = float(retry_after) if retry_after else delay
                logger.warning(
                    "ADS rate limit hit (attempt %s); retrying in %.1f s",
                    attempt + 1,
                    wait_time,
                )
                time.sleep(wait_time)
                delay = min(delay * 2, 30)
                continue

            if response.status_code >= 400:
                snippet = response.text.strip()[:500]
                raise ADSServiceError(
                    f"ADS API error {response.status_code}: {snippet or 'No details'}"
                )

            try:
                return response.json()
            except ValueError as exc:
                raise ADSServiceError("ADS returned invalid JSON") from exc

        raise ADSServiceError("Exceeded retry attempts for ADS API")

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": self.user_agent}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _perform_post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST helper for ADS metrics/export/library endpoints."""
        url = f"{self.base_url.rstrip('/')}{endpoint}"
        headers = self._build_headers()
        headers["Content-Type"] = "application/json"
        try:
            response = self.session.post(url, headers=headers, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ADSServiceError(f"ADS POST error: {exc}") from exc
        if response.status_code >= 400:
            snippet = response.text.strip()[:500]
            raise ADSServiceError(f"ADS API error {response.status_code}: {snippet or 'No details'}")
        try:
            return response.json()
        except ValueError as exc:
            # Some endpoints (BibTeX) return plain text
            return {"export": response.text}

    def _heuristic_query(
        self,
        question: str,
        default_rows: int,
        sort: Optional[str],
        filters: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Fallback query builder when the LLM builder fails."""
        term = question.strip().replace('"', "")
        if not term:
            query = "*:*"
        else:
            # Use keyword: (ADS controlled vocabulary) + title: + object: for better precision
            # than plain abstract: which catches tangential mentions
            query = (
                f'keyword:"{term}" OR title:"{term}" OR object:"{term}"'
            )
        return {
            "query": query,
            "rows": max(1, default_rows),
            "sort": sort or "score desc",
            "filters": filters or ["property:refereed"],
        }



class ADSQueryBuilder:
    """LLM-backed helper that turns natural language into ADS queries."""

    # Comprehensive ADS syntax reference for the LLM query translator
    _SYSTEM_PROMPT = """\
You are an expert NASA ADS (Astrophysics Data System) query builder.
Convert natural-language astronomy questions into optimal ADS search queries.

Return JSON with exactly these keys:
  query   (string)  — the ADS query string
  rows    (int)     — number of results
  sort    (string)  — sort order
  filters (array of strings, optional) — fq filter strings

═══ FIELD SYNTAX (use the most specific field available) ═══

  keyword:"protoplanetary disks"   — ADS controlled-vocabulary keywords (BEST for topic search)
  title:"disk gaps"                — words in paper title (high precision)
  abstract:"dust continuum"        — words in abstract (medium precision, catches tangential mentions)
  body:"gap opening mechanism"     — full-text search inside the paper (use when abstract is too narrow)
  object:"HL Tau"                  — SIMBAD/NED-linked object (catches ALL name variants automatically)
  author:"Andrews, Sean"           — author name (Last, First)
  ^author:"Andrews, Sean"          — FIRST author only
  orcid:0000-0001-2345-6789        — search by ORCID
  year:2020-2024                   — year range
  bibcode:"2018ApJ...869L..41A"    — specific paper
  doi:"10.3847/..."                — DOI lookup
  inst:"Harvard"                   — institution/affiliation
  aff:"Max Planck"                 — affiliation text search
  facility:"ALMA"                  — facility metadata field (precise)

═══ TELESCOPE/FACILITY FILTERING ═══

  bibgroup:ALMA     — papers curated by NRAO as actually USING ALMA data
  bibgroup:HST      — papers using Hubble data
  bibgroup:CXC      — papers using Chandra data
  bibgroup:ESO      — papers using ESO/VLT data
  bibgroup:Gemini   — papers using Gemini data
  bibgroup:Keck     — papers using Keck data
  bibgroup:NRAO     — papers using any NRAO facility (VLA, VLBA, GBT)
  bibgroup:Spitzer  — papers using Spitzer data
  bibgroup:XMM      — papers using XMM-Newton data
  bibgroup:JCMT     — papers using JCMT data

  PREFER bibgroup: over abstract:"ALMA" — bibgroup is curated by librarians
  and only includes papers that actually used the telescope, not papers that
  casually mention it. Only fall back to abstract:"ALMA" if no bibgroup exists.

═══ PROPERTY FILTERS ═══

  property:refereed      — peer-reviewed only (USE BY DEFAULT)
  property:openaccess    — open access papers
  property:data          — papers with linked datasets
  doctype:article        — journal articles only
  doctype:eprint         — arXiv preprints only
  doctype:inproceedings  — conference proceedings

═══ SECOND-ORDER OPERATORS (for discovery) ═══

  similar(query)    — papers with SIMILAR abstract text to results of query
  trending(query)   — papers that readers of query results are also reading NOW
  useful(query)     — most-cited references FROM the papers matching query (foundational works)
  reviews(query)    — papers that cite the papers citing query results (review articles)
  citations(bibcode:XXX)   — all papers that cite a specific paper
  references(bibcode:XXX)  — all papers cited BY a specific paper

═══ SORT OPTIONS ═══

  "date desc"                — newest first (default for "recent" queries)
  "citation_count desc"      — most cited (use for "best/important/influential" queries)
  "citation_count_norm desc" — normalized citations (corrects for paper age)
  "score desc"               — ADS relevance ranking (good general-purpose)
  "read_count desc"          — most-read papers (popularity)

═══ DECISION RULES ═══

1. For TOPIC searches ("papers on X"), prefer keyword:"X" over abstract:"X".
   keyword: uses ADS controlled vocabulary and is far more precise.

2. For OBJECT searches ("papers about HL Tau"), use object:"HL Tau".
   This leverages SIMBAD cross-matching and catches all name variants.

3. For TELESCOPE searches ("ALMA papers on X"), use bibgroup:ALMA AND keyword:"X".

4. For "recent" queries, sort by "date desc".
   For "best/important/seminal" queries, sort by "citation_count desc".
   For "what's hot/trending" queries, use the trending() operator.

5. ALWAYS include property:refereed in filters unless the user asks for preprints.

6. When the user asks for "seminal/foundational/key" papers, use useful(query).
   When the user asks for "review papers/reviews", use reviews(query).
   When the user asks for "similar papers to X", use similar(query).

═══ EXAMPLES ═══

"recent papers on protoplanetary disks"
→ {"query": "keyword:\\"protoplanetary disks\\"", "sort": "date desc", "rows": 15, "filters": ["property:refereed"]}

"best ALMA papers on disk gaps"
→ {"query": "bibgroup:ALMA AND (keyword:\\"protoplanetary disks\\" AND (title:\\"gap\\" OR title:\\"ring\\"))", "sort": "citation_count desc", "rows": 15, "filters": ["property:refereed"]}

"papers about HL Tau"
→ {"query": "object:\\"HL Tau\\"", "sort": "date desc", "rows": 15, "filters": ["property:refereed"]}

"foundational papers on planet formation"
→ {"query": "useful(keyword:\\"planet formation\\" AND year:2015-2026)", "sort": "score desc", "rows": 20, "filters": ["property:refereed"]}
"""

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or os.getenv("ADS_QUERY_MODEL") or os.getenv(
            "DEFAULT_LLM_MODEL", "gpt-4o-mini"
        )
        self._client: Optional[Any] = None

    def build_query(
        self,
        question: str,
        default_rows: int,
        sort: Optional[str],
        filters: Optional[List[str]],
    ) -> Dict[str, Any]:
        if not question.strip():
            raise ADSQueryBuilderError("Empty natural-language question")

        try:
            response = self._call_model(question, default_rows, sort, filters)
        except ADSQueryBuilderError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ADSQueryBuilderError(str(exc)) from exc

        structured = self._parse_response(response)

        if "query" not in structured or not structured["query"].strip():
            raise ADSQueryBuilderError("Model response missing query field")

        structured.setdefault("rows", default_rows)
        structured.setdefault("sort", sort or "date desc")
        if filters and "filters" not in structured:
            structured["filters"] = filters

        return structured

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _call_model(
        self,
        question: str,
        default_rows: int,
        sort: Optional[str],
        filters: Optional[List[str]],
    ) -> str:
        client = self._get_client()

        user_payload = {
            "question": question,
            "defaults": {
                "rows": default_rows,
                "sort": sort or "date desc",
                "filters": filters or ["property:refereed"],
            },
        }

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": "Respond with JSON only. Natural-language request:\n" + json.dumps(user_payload)},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        return response.choices[0].message.content or ""



    def _parse_response(self, content: str) -> Dict[str, Any]:
        if not content:
            raise ADSQueryBuilderError("Empty response from model")

        content = content.strip()
        # Remove code fences properly (only at boundaries, not arbitrary backticks)
        import re
        fence_match = re.match(r'^```(?:json)?\s*\n?(.*?)\n?```$', content, re.DOTALL)
        if fence_match:
            content = fence_match.group(1).strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ADSQueryBuilderError(f"Could not parse model JSON: {exc}") from exc

        # Normalise filters to list
        filters = data.get("filters")
        if isinstance(filters, str):
            data["filters"] = [filters]

        return data

    def _get_client(self) -> Any:
        if self._client:
            return self._client

        if OpenAI is None:
            raise ADSQueryBuilderError("OpenAI SDK not installed")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ADSQueryBuilderError("OPENAI_API_KEY not configured")

        self._client = OpenAI(api_key=api_key)
        return self._client
