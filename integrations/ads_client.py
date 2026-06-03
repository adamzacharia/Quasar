"""NASA ADS Integration Service used by Quasar."""

from __future__ import annotations

import json
import os
import re
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
        "read_count",
        "pub",
        "doi",
        "identifier",
        "keyword",
        "doctype",
        "property",
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
        query = f'"{target_name}" OR title:"{target_name}" OR abstract:"{target_name}" OR keyword:"{target_name}"'
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

    def search_by_observation_identifier(
        self,
        identifier: str,
        max_results: int = 20,
        facility: str = "ALMA",
        filters: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Find papers that explicitly mention an observation/project identifier."""
        clean_identifier = str(identifier or "").strip()
        if not clean_identifier:
            return {
                "query": "",
                "identifier": "",
                "identifier_type": "unknown",
                "papers": [],
            }

        identifier_type = self.classify_observation_identifier(clean_identifier)
        variants = self._observation_identifier_variants(clean_identifier)
        exact_terms = []
        for variant in variants:
            escaped = variant.replace('"', r'\"')
            exact_terms.extend([
                f'body:"{escaped}"',
                f'abstract:"{escaped}"',
                f'title:"{escaped}"',
                f'identifier:"{escaped}"',
            ])

        query = "(" + " OR ".join(exact_terms) + ")"
        facility_clean = str(facility or "").strip().upper()
        if facility_clean == "ALMA" or identifier_type in {"project_code", "mous_uid", "asdm_uid"}:
            query = f"bibgroup:ALMA AND {query}"

        papers = self.search_papers(
            query,
            max_results=max_results,
            sort="score desc",
            filters=filters if filters is not None else ["property:refereed"],
        )

        for paper in papers:
            paper["observation_links"] = [
                {
                    "identifier": clean_identifier,
                    "identifier_type": identifier_type,
                    "relation": "explicit_identifier_search",
                    "confidence": "explicit",
                    "ads_query": query,
                }
            ]

        return {
            "query": query,
            "identifier": clean_identifier,
            "identifier_type": identifier_type,
            "papers": papers,
        }

    @staticmethod
    def classify_observation_identifier(identifier: str) -> str:
        text = str(identifier or "").strip()
        if re.search(r"\b\d{4}\.\d\.\d{5}\.[A-Z]\b", text, re.IGNORECASE):
            return "project_code"
        if text.lower().startswith("uid://"):
            return "mous_uid" if "/x" in text.lower() else "uid"
        if text.lower().startswith("ivo://"):
            return "dataset_id"
        if text.lower().startswith("asdm"):
            return "asdm_uid"
        return "identifier"

    @staticmethod
    def _observation_identifier_variants(identifier: str) -> List[str]:
        text = str(identifier or "").strip()
        variants = [text]
        if text.lower().startswith("uid://"):
            variants.append(text.replace("uid://", "", 1))
        if "/" in text:
            variants.append(text.replace("/", " "))
        seen = set()
        result = []
        for variant in variants:
            clean = variant.strip()
            key = clean.lower()
            if clean and key not in seen:
                seen.add(key)
                result.append(clean)
        return result
    
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

    def get_arxiv_pdf_url(self, identifier: str) -> Optional[str]:
        """
        Get the direct arXiv PDF URL for a given bibcode or arXiv ID.
        If an arXiv ID is provided directly, it returns the URL.
        If a bibcode is provided, it queries ADS to find the associated arXiv ID.
        """
        # If it looks like an arXiv ID (e.g., 1812.04040 or arXiv:1812.04040)
        import re
        arxiv_match = re.search(r'(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)', identifier.lower())
        if arxiv_match:
            arxiv_id = arxiv_match.group(1)
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
        # Otherwise, treat as bibcode and query ADS
        details = self.get_paper_details(identifier)
        if not details:
            return None
            
        identifiers = details.get('identifiers', [])
        for id_str in identifiers:
            arxiv_match = re.search(r'(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)', id_str.lower())
            if arxiv_match:
                arxiv_id = arxiv_match.group(1)
                return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                
        return None
    
    def _format_papers(self, papers: List[Dict]) -> List[Dict[str, Any]]:
        """Format raw ADS response to standardized format"""
        formatted = []
        
        for paper in papers:
            authors = paper.get('author', [])
            if len(authors) > 3:
                author_str = f"{authors[0]} et al."
            else:
                author_str = ", ".join(authors)
            
            properties = paper.get('property', [])
            
            formatted.append({
                'title': paper.get('title', ['Unknown'])[0],
                'authors': author_str,
                'year': paper.get('year', 'Unknown'),
                'journal': paper.get('pub', 'Unknown'),
                'bibcode': paper.get('bibcode', ''),
                'abstract': paper.get('abstract', 'No abstract available'),
                'citations': paper.get('citation_count', 0),
                'reads': paper.get('read_count', 0),
                'doi': paper.get('doi', [''])[0] if paper.get('doi') else '',
                'keywords': paper.get('keyword', [])[:8],
                'doctype': paper.get('doctype', 'unknown'),
                'is_refereed': 'REFEREED' in properties if properties else False,
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
            # Use keyword: (ADS controlled vocabulary) + title: + abstract: + plain text
            # for better precision without causing SolrException
            query = (
                f'keyword:"{term}" OR title:"{term}" OR abstract:"{term}" OR "{term}"'
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
  "HL Tau"                         — plain text search/object name (best for target/object searches)
  author:"Andrews, Sean"           — author name (Last, First)
  ^author:"Andrews, Sean"          — FIRST author only
  orcid:0000-0001-2345-6789        — search by ORCID
  year:2020-2024                   — year range
  bibcode:"2018ApJ...869L..41A"    — specific paper
  doi:"10.3847/..."                — DOI lookup
  inst:"Harvard"                   — institution/affiliation
  aff:"Max Planck"                 — affiliation text search
  facility:"ALMA"                  — facility metadata field (precise)

  CRITICAL: Do NOT use "object:" or "simbad:" field prefixes (e.g. object:"HL Tau" is invalid).
  These prefixes are not supported by the search API and cause 400 Bad Request (SolrException) errors.
  Always search for target/object names as plain text or in title/abstract fields instead.

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
  property:nonarticle    — non-article records
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

═══ BOOLEAN OPERATORS ═══

  AND  — both terms required (default between terms)
  OR   — either term
  NOT  — exclude term
  ()   — grouping

═══ DECISION RULES ═══

1. For TOPIC searches ("papers on X"), prefer keyword:"X" over abstract:"X".
   keyword: uses ADS controlled vocabulary and is far more precise.
   Only add abstract:"X" as a fallback OR if the topic is very niche.

2. For OBJECT searches ("papers about HL Tau"), query the target name as plain text (e.g. "HL Tau").
   Do NOT use "object:" or "simbad:" prefixes under any circumstances, as they cause API errors.

3. For TELESCOPE searches ("ALMA papers on X"), use bibgroup:ALMA AND keyword:"X".
   Do NOT use abstract:"ALMA" — it catches papers that merely mention ALMA.

4. For "recent" queries, sort by "date desc".
   For "best/important/seminal" queries, sort by "citation_count desc".
   For "what's hot/trending" queries, use the trending() operator.
   For general queries with no time preference, sort by "score desc".

5. ALWAYS include property:refereed in filters unless the user specifically
   asks for preprints or arXiv papers.

6. When the user asks for "seminal/foundational/key" papers, use useful(query).
   When the user asks for "review papers/reviews", use reviews(query) or add doctype filter.
   When the user asks for "similar papers to X", use similar(query).

═══ EXAMPLES ═══

"recent papers on protoplanetary disks"
→ {"query": "keyword:\\"protoplanetary disks\\"", "sort": "date desc", "rows": 15, "filters": ["property:refereed"]}

"best ALMA papers on disk gaps"
→ {"query": "bibgroup:ALMA AND (keyword:\\"protoplanetary disks\\" AND (title:\\"gap\\" OR title:\\"ring\\"))", "sort": "citation_count desc", "rows": 15, "filters": ["property:refereed"]}

"papers about HL Tau"
→ {"query": "\\"HL Tau\\"", "sort": "date desc", "rows": 15, "filters": ["property:refereed"]}

"what are people reading about FRBs right now"
→ {"query": "trending(keyword:\\"fast radio bursts\\")", "sort": "score desc", "rows": 15}

"foundational papers on planet formation"
→ {"query": "useful(keyword:\\"planet formation\\" AND year:2015-2026)", "sort": "score desc", "rows": 20, "filters": ["property:refereed"]}

"papers by Sean Andrews on ALMA disk surveys"
→ {"query": "author:\\"Andrews, Sean\\" AND bibgroup:ALMA AND keyword:\\"protoplanetary disks\\"", "sort": "date desc", "rows": 20, "filters": ["property:refereed"]}

"review articles on AGN feedback"
→ {"query": "reviews(keyword:\\"active galactic nuclei\\" AND keyword:\\"feedback\\")", "sort": "citation_count desc", "rows": 15, "filters": ["property:refereed"]}

"papers with JWST data on high-z galaxies since 2023"
→ {"query": "bibgroup:HST AND keyword:\\"high-redshift galaxies\\" AND year:2023-2026", "sort": "date desc", "rows": 15, "filters": ["property:refereed"]}

"full text search for gap opening mechanism in disks"
→ {"query": "body:\\"gap opening\\" AND keyword:\\"protoplanetary disks\\"", "sort": "score desc", "rows": 15, "filters": ["property:refereed"]}
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

        response = client.responses.create(
            model=self.model,
            input="Respond with JSON only. Natural-language request:\n" + json.dumps(user_payload),
            instructions=self._SYSTEM_PROMPT,
            temperature=0.2,
            text={"format": {"type": "json_object"}}
        )

        # Extract text from Responses API
        if hasattr(response, 'output_text'):
            return response.output_text
        elif hasattr(response, 'output'):
            for item in response.output:
                if hasattr(item, 'content') and getattr(item, 'type', None) == "text":
                    return item.content
        return str(response)

    def _parse_response(self, content: str) -> Dict[str, Any]:
        if not content:
            raise ADSQueryBuilderError("Empty response from model")

        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:]
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
