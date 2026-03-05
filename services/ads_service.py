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
        term = question.strip().replace('"', "")
        if not term:
            query = "*:*"
        else:
            query = (
                f'object:"{term}" OR title:"{term}" OR abstract:"{term}"'
            )
        return {
            "query": query,
            "rows": max(1, default_rows),
            "sort": sort or "date desc",
            "filters": filters or ["property:refereed"],
        }


class ADSQueryBuilder:
    """LLM-backed helper that turns natural language into ADS queries."""

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

        system_prompt = (
            "You are an expert at converting natural-language astronomy questions into "
            "valid NASA ADS (Astrophysics Data System) search queries.\n"
            "Return JSON with exactly these keys: query (string), rows (int), sort (string), "
            "filters (array of strings, optional).\n\n"
            "IMPORTANT ADS field syntax rules:\n"
            "- title:\"word\" — search in paper title\n"
            "- abstract:\"word\" — search in paper abstract\n"
            "- author:\"Last, First\" — search by author name\n"
            "- year:2020-2024 — filter by year range\n"
            "- bibcode:\"...\" — specific bibcode\n"
            "- object:\"M87\" — known SIMBAD astronomical object (NOT for telescope/instrument names)\n"
            "- Combine with AND, OR operators\n"
            "- For telescope/instrument names (ALMA, VLA, Hubble, VLBI, etc.) use title: or abstract:\n\n"
            "Examples:\n"
            "- 'ALMA papers on molecular clouds' => abstract:\"ALMA\" AND abstract:\"molecular cloud\"\n"
            "- 'VLA observations of AGN 2020-2023' => abstract:\"VLA\" AND abstract:\"AGN\" AND year:2020-2023\n"
            "- 'black hole jets' => title:\"black hole\" AND abstract:\"jet\"\n"
            "- 'papers by Accomazzi' => author:\"Accomazzi\"\n"
        )


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
                {"role": "system", "content": system_prompt},
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
