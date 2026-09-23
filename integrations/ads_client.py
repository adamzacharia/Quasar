"""NASA ADS / SciX Integration Service used by Quasar.

Provider abstraction (R8, 2026-07-18): astronomy is transitioning from ADS to
SciX (scixplorer.org) through 2026. Both hosts run the same API software —
live-probed 2026-07-18: ``https://api.scixplorer.org/v1/search/query`` and
``https://api.adsabs.harvard.edu/v1/search/query`` return byte-identical
``401 {"message": "Missing \"Authorization\" in headers."}`` without a token,
same endpoint layout, same Bearer auth. Selection order for the base URL:

1. ``NASA_ADS_BASE_URL`` — explicit override, always wins.
2. ``ADS_API_PROVIDER`` = ``ads`` (default) | ``scix`` — picks the host.

API keys: ``NASA_ADS_API_KEY`` first, then ``SCIX_API_KEY`` as a fallback
(SciX issues its own tokens; either works on both hosts today, but that may
diverge post-transition — hence the split env vars).
"""

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


# Known API hosts. Both serve the identical API surface (verified live
# 2026-07-18 — see module docstring); "scix" exists so deployments can flip
# the default host by env without a code change when the ADS domain sunsets.
_API_PROVIDER_BASE_URLS = {
    "ads": "https://api.adsabs.harvard.edu/v1",
    "scix": "https://api.scixplorer.org/v1",
}


def resolve_ads_base_url() -> str:
    """Resolve the ADS/SciX API base URL from the environment.

    ``NASA_ADS_BASE_URL`` (explicit URL) beats ``ADS_API_PROVIDER``
    (named provider: ``ads`` | ``scix``); unknown provider names fall back
    to classic ADS with a warning rather than failing the whole service.
    """
    explicit = os.getenv("NASA_ADS_BASE_URL", "").strip()
    if explicit:
        return explicit
    provider = os.getenv("ADS_API_PROVIDER", "ads").strip().lower()
    if provider not in _API_PROVIDER_BASE_URLS:
        logger.warning(
            "Unknown ADS_API_PROVIDER=%r — falling back to classic ADS", provider
        )
        provider = "ads"
    return _API_PROVIDER_BASE_URLS[provider]


def resolve_ads_api_key() -> str:
    """Resolve the API token: NASA_ADS_API_KEY first, then SCIX_API_KEY."""
    return os.getenv("NASA_ADS_API_KEY", "") or os.getenv("SCIX_API_KEY", "")


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

    # Request policy shared by EVERY ADS call in this module (search, details,
    # metrics, export, libraries): a 10 s timeout and 2 attempts with a 1 s
    # backoff. services/tool_budgets.py reads the __init__ defaults (which are
    # these constants) so the budget-hierarchy test sees live values;
    # :meth:`worst_case_seconds` is the same arithmetic for callers.
    DEFAULT_TIMEOUT_S = 10.0
    # 2 attempts (was 3): a literature tool chains ADS + OpenAlex + an ALMA
    # TAP lookup under a 150 s guard (services/tool_budgets.py).
    DEFAULT_RETRY_ATTEMPTS = 2
    # A 429 whose Retry-After exceeds this is reported at once instead of
    # slept through (ADS quotas reset daily — a long Retry-After means "not
    # this turn"). Also caps the 429 wait so one call never exceeds
    # ``worst_case_seconds(1)``.
    RATE_LIMIT_WAIT_CAP_S = 10.0
    # ADS search ``rows`` hard maximum per request; also the most bibcodes
    # one author-metrics request will aggregate.
    METRICS_MAX_BIBCODES = 2000
    # ADS export service formats (``POST /export/{format}``).
    EXPORT_FORMATS = ("bibtex", "bibtexabs", "aastex", "endnote", "ris", "icarus", "mnras", "soph", "ads")
    _METRICS_TYPES = ("basic", "citations", "indicators")
    # biblib library ids are URL-safe tokens; anything else is refused before
    # it can reach the path.
    _LIBRARY_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        session: Optional[requests.Session] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or resolve_ads_api_key()
        self.base_url = base_url or resolve_ads_base_url()
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
            # Never fabricate literature results. Without a key we cannot query
            # ADS — surface a typed error so the caller reports "ADS unavailable"
            # rather than presenting invented bibcodes/DOIs as real papers. (C2)
            raise ADSServiceError(
                "NASA ADS API key is not configured — the literature archive "
                "cannot be searched. No papers were returned."
            )

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
            # Propagate the typed error instead of returning fabricated example
            # papers. A caller (agent._search_papers) turns this into a
            # {"success": False, "error": ...} the model can honestly report. (C2)
            logger.error("ADS search failed: %s", exc)
            raise

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
        # Standard 19-char ADS bibcode (R2 reverse direction: paper → data).
        if re.match(r"^[12]\d{3}[A-Za-z][A-Za-z0-9&.]{13}[A-Za-z.]$", text):
            return "bibcode"
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
    
    # ------------------------------------------------------------------
    # Author / metrics / export / library backends (tool-facing)
    # ------------------------------------------------------------------
    # These back get_author_papers, get_paper_metrics, get_author_metrics,
    # export_bibtex, list_ads_libraries, get_ads_library_papers,
    # create_ads_library and add_to_ads_library (core/tool_registrations.py),
    # which until 2026-09-21 called methods this class never defined and so
    # raised AttributeError before any network call. Each is one or two bounded
    # ADS requests through :meth:`_perform_request` (same 10 s x 2 policy as
    # search_papers, tool-budget clamp, host circuit breaker) and returns a
    # structured dict — ``{"success": False, "error": ...}`` on any ADS
    # failure — so the model gets a specific message instead of a traceback.
    # Response shapes probed live against api.adsabs.harvard.edu 2026-09-21:
    # metrics keys carry spaces ("citation stats", "indicators"), an unknown
    # bibcode answers HTTP 200 with an "Error" key, the export service answers
    # {"msg", "export"}, the library list {"count", "libraries"}, and a
    # library id that does not exist answers an HTML 500.

    @classmethod
    def worst_case_seconds(
        cls,
        calls: int = 1,
        *,
        timeout: Optional[float] = None,
        retry_attempts: Optional[int] = None,
    ) -> float:
        """Worst-case sequential wall time of ``calls`` ADS requests.

        Per call: ``attempts x timeout`` plus the backoff sleeps between
        attempts (1 s, 2 s, ...). A 429 path is never longer: each 429 attempt
        returns quickly and sleeps at most ``RATE_LIMIT_WAIT_CAP_S`` (= the
        timeout). Defaults: 2 x 10 s + 1 s = 21 s per call.
        """
        t = cls.DEFAULT_TIMEOUT_S if timeout is None else float(timeout)
        n = cls.DEFAULT_RETRY_ATTEMPTS if retry_attempts is None else max(1, int(retry_attempts))
        backoff = sum(float(2 ** i) for i in range(n - 1))
        return float(max(1, int(calls))) * (n * t + backoff)

    def get_author_papers(
        self,
        author: str,
        max_results: int = 20,
        sort: str = "date desc",
        refereed_only: bool = False,
    ) -> Dict[str, Any]:
        """Papers by one author (``author:"Last, First"``), newest first."""
        name = str(author or "").strip()
        if not name:
            return self._tool_error("An author name is required (use 'Last, First' form).")
        query = f'author:"{self._escape_phrase(name)}"'
        rows = self._clamp_int(max_results, default=20, lo=1, hi=200)
        params: Dict[str, Union[str, int, List[str]]] = {
            "q": query,
            "fl": ",".join(self._DEFAULT_FIELDS),
            "rows": rows,
            "sort": str(sort or "date desc"),
            "start": 0,
        }
        if refereed_only:
            params["fq"] = ["property:refereed"]
        try:
            self._require_api_key()
            data = self._perform_get("/search/query", params)
        except ADSServiceError as exc:
            logger.error("ADS author search failed for %r: %s", name, exc)
            return self._tool_error(str(exc), author=name, query=query)
        response = data.get("response", {}) if isinstance(data, dict) else {}
        papers = self._format_papers(response.get("docs", []) or [])
        num_found = self._clamp_int(response.get("numFound"), default=len(papers), lo=0, hi=10**9)
        return {
            "success": True,
            "author": name,
            "query": query,
            "num_found": num_found,
            "returned": len(papers),
            "truncated": num_found > len(papers),
            "refereed_only": bool(refereed_only),
            "papers": papers,
        }

    def get_paper_metrics(self, bibcode: str) -> Dict[str, Any]:
        """Citation / read / indicator metrics for one paper (``POST /metrics``)."""
        code = str(bibcode or "").strip()
        if not code:
            return self._tool_error("A bibcode is required (e.g. '2018ApJ...869L..41A').")
        try:
            self._require_api_key()
            data = self._perform_post("/metrics", {"bibcodes": [code], "types": list(self._METRICS_TYPES)})
        except ADSServiceError as exc:
            logger.error("ADS metrics failed for %s: %s", code, exc)
            return self._tool_error(str(exc), bibcode=code)
        problem = self._metrics_problem(data, [code])
        if problem:
            return self._tool_error(problem, bibcode=code)
        out = {"success": True, "bibcode": code, "link": self._abs_link(code)}
        out.update(self._summarise_metrics(data))
        return out

    def get_author_metrics(
        self,
        author: str,
        max_papers: int = METRICS_MAX_BIBCODES,
        refereed_only: bool = False,
    ) -> Dict[str, Any]:
        """h-index, i10, citations and reads for an author.

        Two bounded calls: the author's bibcodes (``/search/query``, up to
        ``METRICS_MAX_BIBCODES``) then ``POST /metrics`` over them. ``truncated``
        is True when the author has more papers than were aggregated.
        """
        name = str(author or "").strip()
        if not name:
            return self._tool_error("An author name is required (use 'Last, First' form).")
        query = f'author:"{self._escape_phrase(name)}"'
        rows = self._clamp_int(max_papers, default=self.METRICS_MAX_BIBCODES, lo=1, hi=self.METRICS_MAX_BIBCODES)
        params: Dict[str, Union[str, int, List[str]]] = {
            "q": query,
            "fl": "bibcode",
            "rows": rows,
            "sort": "date desc",
            "start": 0,
        }
        if refereed_only:
            params["fq"] = ["property:refereed"]
        try:
            self._require_api_key()
            search = self._perform_get("/search/query", params)
            response = search.get("response", {}) if isinstance(search, dict) else {}
            bibcodes = [
                str(doc.get("bibcode")).strip()
                for doc in (response.get("docs", []) or [])
                if isinstance(doc, dict) and doc.get("bibcode")
            ]
            num_found = self._clamp_int(response.get("numFound"), default=len(bibcodes), lo=0, hi=10**9)
            if not bibcodes:
                return self._tool_error(
                    f"No ADS papers found for author {name!r}"
                    + (" (refereed only)" if refereed_only else "")
                    + " — check the 'Last, First' spelling.",
                    author=name, query=query, num_found=0,
                )
            data = self._perform_post("/metrics", {"bibcodes": bibcodes, "types": list(self._METRICS_TYPES)})
        except ADSServiceError as exc:
            logger.error("ADS author metrics failed for %r: %s", name, exc)
            return self._tool_error(str(exc), author=name, query=query)
        problem = self._metrics_problem(data, bibcodes)
        if problem:
            return self._tool_error(problem, author=name, query=query)
        out: Dict[str, Any] = {
            "success": True,
            "author": name,
            "query": query,
            "num_found": num_found,
            "papers_considered": len(bibcodes),
            "truncated": num_found > len(bibcodes),
            "refereed_only": bool(refereed_only),
        }
        if out["truncated"]:
            out["note"] = (
                f"Metrics aggregate the {len(bibcodes)} most recent of {num_found} papers; "
                "indicators for the full record may be higher."
            )
        out.update(self._summarise_metrics(data))
        return out

    def export_bibtex(
        self,
        bibcodes: Union[str, List[str]],
        format: str = "bibtex",
        max_authors: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Formatted citations for one or more bibcodes (``POST /export/{format}``)."""
        codes = self._coerce_bibcodes(bibcodes)
        fmt = str(format or "bibtex").strip().lower()
        if not codes:
            return self._tool_error("At least one ADS bibcode is required.")
        if fmt not in self.EXPORT_FORMATS:
            return self._tool_error(
                f"Unsupported export format {fmt!r}; choose one of {', '.join(self.EXPORT_FORMATS)}.",
                bibcodes=codes,
            )
        body: Dict[str, Any] = {"bibcode": codes}
        if max_authors:
            body["maxauthor"] = self._clamp_int(max_authors, default=0, lo=1, hi=500)
        try:
            self._require_api_key()
            data = self._perform_post(f"/export/{fmt}", body)
        except ADSServiceError as exc:
            logger.error("ADS export (%s) failed: %s", fmt, exc)
            return self._tool_error(str(exc), bibcodes=codes, format=fmt)
        export = data.get("export") if isinstance(data, dict) else None
        if not isinstance(export, str) or not export.strip():
            return self._tool_error(
                f"ADS export service returned no {fmt} entries for {', '.join(codes[:5])}.",
                bibcodes=codes, format=fmt,
            )
        missing = [c for c in codes if c not in export]
        out: Dict[str, Any] = {
            "success": True,
            "format": fmt,
            "bibcodes": codes,
            "count": len(codes),
            "entries": len(re.findall(r"^@\w+\{", export, re.M)) if fmt.startswith("bibtex") else None,
            "message": str(data.get("msg", "") or ""),
            "bibtex": export,
        }
        if missing:
            out["missing_bibcodes"] = missing
        return out

    def list_libraries(self) -> Dict[str, Any]:
        """Personal ADS libraries of the account behind the configured token."""
        try:
            self._require_api_key()
            data = self._perform_get("/biblib/libraries", {})
        except ADSServiceError as exc:
            logger.error("ADS library list failed: %s", exc)
            return self._tool_error(str(exc))
        raw = data.get("libraries") if isinstance(data, dict) else None
        libraries = [self._format_library(lib) for lib in (raw or []) if isinstance(lib, dict)]
        return {
            "success": True,
            "count": len(libraries),
            "libraries": libraries,
            "note": "Libraries belong to the ADS account whose API token Quasar is configured with.",
        }

    def get_library_papers(self, library_id: str, max_results: int = 50, start: int = 0) -> Dict[str, Any]:
        """Papers in one library (``GET /biblib/libraries/{id}``).

        biblib returns the bibcodes plus a Solr block for the requested ``fl``;
        if that block is missing the bibcodes are resolved with one ordinary
        search, so the result carries titles either way (2 bounded calls max).
        """
        lid = self._validate_library_id(library_id)
        if lid is None:
            return self._tool_error(
                "A valid ADS library id is required (letters, digits, '-' and '_' only).",
                library_id=str(library_id or ""),
            )
        rows = self._clamp_int(max_results, default=50, lo=1, hi=200)
        offset = self._clamp_int(start, default=0, lo=0, hi=10**7)
        params: Dict[str, Union[str, int, List[str]]] = {
            "start": offset,
            "rows": rows,
            "fl": ",".join(self._DEFAULT_FIELDS),
        }
        try:
            self._require_api_key()
            data = self._perform_get(f"/biblib/libraries/{lid}", params)
        except ADSServiceError as exc:
            logger.error("ADS library %s fetch failed: %s", lid, exc)
            return self._tool_error(self._library_hint(str(exc)), library_id=lid)
        if not isinstance(data, dict):
            return self._tool_error("ADS library service returned an unexpected payload.", library_id=lid)
        documents = [str(b).strip() for b in (data.get("documents") or []) if b]
        solr = data.get("solr") if isinstance(data.get("solr"), dict) else {}
        solr_response = solr.get("response") if isinstance(solr.get("response"), dict) else {}
        docs = solr_response.get("docs") or []
        papers = self._format_papers([d for d in docs if isinstance(d, dict)]) if docs else []
        if documents and not papers:
            wanted = documents[:rows]
            fallback_query = "bibcode:(" + " OR ".join(f'"{b}"' for b in wanted) + ")"
            try:
                papers = self.search_papers(fallback_query, max_results=len(wanted), sort="date desc")
            except ADSServiceError as exc:
                logger.warning("ADS library %s: bibcode resolution failed (%s); returning bibcodes only", lid, exc)
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        total = metadata.get("num_documents")
        if not isinstance(total, int):
            found = solr_response.get("numFound")
            total = found if isinstance(found, int) else len(documents)
        if metadata:
            library = self._format_library({**metadata, "id": metadata.get("id", lid)})
        else:
            library = {"id": lid, "link": self._library_link(lid)}
        return {
            "success": True,
            "library_id": lid,
            "library": library,
            "num_documents": int(total),
            "start": offset,
            "returned": len(papers) if papers else len(documents),
            "bibcodes": documents,
            "papers": papers,
        }

    def create_library(
        self,
        name: str,
        description: str = "",
        public: bool = False,
        bibcodes: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Create a personal library (``POST /biblib/libraries``)."""
        title = str(name or "").strip()
        if not title:
            return self._tool_error("A library name is required.")
        codes = self._coerce_bibcodes(bibcodes)
        body: Dict[str, Any] = {
            "name": title,
            "description": str(description or ""),
            "public": self._as_bool(public),
            "bibcode": codes,
        }
        try:
            self._require_api_key()
            data = self._perform_post("/biblib/libraries", body)
        except ADSServiceError as exc:
            logger.error("ADS create library %r failed: %s", title, exc)
            return self._tool_error(str(exc), name=title)
        if not isinstance(data, dict) or not data.get("id"):
            return self._tool_error("ADS did not return an id for the new library.", name=title)
        library = self._format_library(data)
        return {
            "success": True,
            "library_id": str(data.get("id")),
            "library": library,
            "link": library["link"],
            "bibcodes_added": len(codes),
        }

    def add_to_library(
        self,
        library_id: str,
        bibcodes: Union[str, List[str]],
        action: str = "add",
    ) -> Dict[str, Any]:
        """Add (or remove) bibcodes in a library (``POST /biblib/documents/{id}``)."""
        lid = self._validate_library_id(library_id)
        if lid is None:
            return self._tool_error(
                "A valid ADS library id is required (letters, digits, '-' and '_' only).",
                library_id=str(library_id or ""),
            )
        codes = self._coerce_bibcodes(bibcodes)
        act = str(action or "add").strip().lower()
        if act not in {"add", "remove"}:
            return self._tool_error(f"Unsupported library action {act!r}; use 'add' or 'remove'.", library_id=lid)
        if not codes:
            return self._tool_error("At least one ADS bibcode is required.", library_id=lid)
        try:
            self._require_api_key()
            data = self._perform_post(f"/biblib/documents/{lid}", {"bibcode": codes, "action": act})
        except ADSServiceError as exc:
            logger.error("ADS %s to library %s failed: %s", act, lid, exc)
            return self._tool_error(self._library_hint(str(exc)), library_id=lid, bibcodes=codes)
        counter = "number_added" if act == "add" else "number_removed"
        count = data.get(counter) if isinstance(data, dict) else None
        return {
            "success": True,
            "library_id": lid,
            "action": act,
            "requested": len(codes),
            counter: count if isinstance(count, int) else None,
            "bibcodes": codes,
            "link": self._library_link(lid),
        }

    # ── small helpers for the tool backends ─────────────────────────────
    def _require_api_key(self) -> None:
        if not self.api_key:
            raise ADSServiceError(
                "NASA ADS API key is not configured — set NASA_ADS_API_KEY (or SCIX_API_KEY) "
                "to use the ADS author, metrics, export and library tools."
            )

    @staticmethod
    def _tool_error(message: str, **context: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {"success": False, "error": message}
        out.update({k: v for k, v in context.items() if v is not None})
        return out

    @staticmethod
    def _escape_phrase(text: str) -> str:
        return str(text).replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _clamp_int(value: Any, *, default: int, lo: int, hi: int) -> int:
        try:
            if value is None or str(value).strip() == "":
                number = int(default)
            else:
                number = int(float(value))
        except (TypeError, ValueError):
            number = int(default)
        return max(lo, min(hi, number))

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "public"}
        return bool(value)

    @staticmethod
    def _coerce_bibcodes(bibcodes: Any) -> List[str]:
        if bibcodes is None:
            return []
        if isinstance(bibcodes, str):
            items: List[Any] = re.split(r"[,;\s]+", bibcodes)
        else:
            try:
                items = list(bibcodes)
            except TypeError:
                items = [bibcodes]
        out: List[str] = []
        seen = set()
        for item in items:
            code = str(item or "").strip()
            if code and code not in seen:
                seen.add(code)
                out.append(code)
        return out

    def _validate_library_id(self, library_id: Any) -> Optional[str]:
        lid = str(library_id or "").strip()
        return lid if self._LIBRARY_ID_RE.match(lid) else None

    @staticmethod
    def _library_hint(message: str) -> str:
        # Probed live 2026-09-21: biblib answers an HTML 500 for a library id
        # that does not exist / is not readable with this token.
        if "500" in message:
            return message + " (ADS answers 500 for a library id that does not exist or is not accessible to this token)"
        return message

    @staticmethod
    def _abs_link(bibcode: str) -> str:
        return f"https://ui.adsabs.harvard.edu/abs/{bibcode}"

    @staticmethod
    def _library_link(library_id: str) -> str:
        return f"https://ui.adsabs.harvard.edu/user/libraries/{library_id}"

    def _format_library(self, lib: Dict[str, Any]) -> Dict[str, Any]:
        lid = str(lib.get("id", "") or "")
        num_documents = lib.get("num_documents")
        if num_documents is None and isinstance(lib.get("bibcode"), list):
            num_documents = len(lib["bibcode"])
        return {
            "id": lid,
            "name": lib.get("name", ""),
            "description": lib.get("description", ""),
            "num_documents": num_documents,
            "public": bool(lib.get("public", False)),
            "permission": lib.get("permission"),
            "owner": lib.get("owner"),
            "date_created": lib.get("date_created"),
            "date_last_modified": lib.get("date_last_modified"),
            "link": self._library_link(lid) if lid else None,
        }

    @staticmethod
    def _metrics_problem(data: Any, bibcodes: List[str]) -> Optional[str]:
        """Reason the metrics payload is unusable, or None when it is fine."""
        shown = ", ".join(bibcodes[:5]) + (" ..." if len(bibcodes) > 5 else "")
        if not isinstance(data, dict):
            return "ADS metrics service returned an unexpected payload."
        if "Error" in data:
            info = data.get("Error Info") or data.get("Error")
            return f"ADS has no metrics for {shown}: {info}"
        skipped = data.get("skipped bibcodes") or []
        if bibcodes and all(code in skipped for code in bibcodes):
            return f"ADS has no metrics for {shown} (bibcode not found)."
        return None

    @staticmethod
    def _summarise_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
        basic = data.get("basic stats") or {}
        cites = data.get("citation stats") or {}
        cites_ref = data.get("citation stats refereed") or {}
        ind = data.get("indicators") or {}
        ind_ref = data.get("indicators refereed") or {}
        return {
            "number_of_papers": basic.get("number of papers"),
            "total_citations": cites.get("total number of citations"),
            "refereed_citations": cites.get("total number of refereed citations"),
            "citing_papers": cites.get("number of citing papers"),
            "self_citations": cites.get("number of self-citations"),
            "total_reads": basic.get("total number of reads"),
            "recent_reads": basic.get("recent number of reads"),
            "total_downloads": basic.get("total number of downloads"),
            "h_index": ind.get("h"),
            "g_index": ind.get("g"),
            "i10_index": ind.get("i10"),
            "i100_index": ind.get("i100"),
            "m_index": ind.get("m"),
            "tori": ind.get("tori"),
            "riq": ind.get("riq"),
            "read10": ind.get("read10"),
            "indicators_refereed": ind_ref,
            "basic_stats": basic,
            "citation_stats": cites,
            "citation_stats_refereed": cites_ref,
            "skipped_bibcodes": data.get("skipped bibcodes") or [],
        }

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
    
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _perform_get(
        self,
        endpoint: str,
        params: Dict[str, Union[str, int, List[str]]],
        *,
        timeout: Optional[float] = None,
        retry_attempts: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self._perform_request("GET", endpoint, params=params, timeout=timeout, retry_attempts=retry_attempts)

    def _perform_post(
        self,
        endpoint: str,
        json_body: Dict[str, Any],
        *,
        params: Optional[Dict[str, Union[str, int, List[str]]]] = None,
        timeout: Optional[float] = None,
        retry_attempts: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self._perform_request(
            "POST", endpoint, params=params, json_body=json_body, timeout=timeout, retry_attempts=retry_attempts,
        )

    def _perform_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Union[str, int, List[str]]]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
        retry_attempts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """One bounded ADS API call under the shared retry policy.

        Every ADS request in this module goes through here so the 2026-09-21
        latency rules apply uniformly (services/tool_budgets.py docstring):

        * the timeout (default ``DEFAULT_TIMEOUT_S``) is clamped to the calling
          tool's remaining budget (``bounded_timeout``); a call that cannot get
          1 s is refused before it is sent, and a backoff sleep that would
          outlast the budget is skipped in favour of the typed error;
        * the host circuit breaker (services/host_breaker.py) is checked before
          every attempt and fed every transport failure / gateway status, so
          after ADS refuses, resets or times out once, the next literature call
          in the turn fails in microseconds — ``HostCircuitOpen`` is left to
          propagate: the tool guard turns it into the structured
          INFRASTRUCTURE FAILURE result;
        * the send runs under ``http_budget_hook.suppressed()`` so the
          process-wide requests hook neither clamps nor records it twice;
        * a 429 is retried only when ADS asks for at most
          ``RATE_LIMIT_WAIT_CAP_S``; a longer Retry-After is reported at once.
        """
        from services.host_breaker import HostBreaker, host_of
        from services.http_budget_hook import suppressed
        from services.tool_budgets import BudgetExhausted, bounded_timeout

        verb = str(method or "GET").upper()
        url = f"{self.base_url.rstrip('/')}{endpoint}"
        host = host_of(url)
        headers = self._build_headers()
        if json_body is not None:
            headers["Content-Type"] = "application/json"
        label = f"ADS {verb} {endpoint}"
        delay = 1.0
        base_timeout = float(self.timeout if timeout is None else timeout)
        eff_attempts = self.retry_attempts if retry_attempts is None else max(1, int(retry_attempts))

        for attempt in range(eff_attempts):
            # Dead host / service (from ANY tool this turn): refuse instantly.
            HostBreaker.check(url)
            try:
                eff_timeout = bounded_timeout(base_timeout, minimum=1.0, label=label)
            except BudgetExhausted as exc:
                raise ADSServiceError(f"{label} not sent: {exc}") from exc
            try:
                with suppressed():
                    if verb == "GET":
                        response = self.session.get(
                            url, headers=headers, params=params, timeout=eff_timeout
                        )
                    elif verb == "POST":
                        response = self.session.post(
                            url, headers=headers, params=params, json=json_body, timeout=eff_timeout
                        )
                    else:
                        response = self.session.request(
                            verb, url, headers=headers, params=params, json=json_body, timeout=eff_timeout
                        )
            except requests.Timeout as exc:
                HostBreaker.record_failure(url, exc)
                if attempt == eff_attempts - 1:
                    raise ADSServiceError("ADS request timed out") from exc
                self._sleep_bounded(delay, label)
                delay *= 2
                continue
            except requests.RequestException as exc:
                HostBreaker.record_failure(url, exc)
                if attempt == eff_attempts - 1:
                    raise ADSServiceError(f"ADS request error: {exc}") from exc
                self._sleep_bounded(delay, label)
                delay *= 2
                continue

            status = int(getattr(response, "status_code", 0) or 0)
            if status in (502, 503, 504):
                HostBreaker.record_failure(url, status=status)
                raise ADSServiceError(f"ADS API error {status}: {self._response_snippet(response)}")
            HostBreaker.record_success(url)

            if status == 429:
                retry_after = self._retry_after_seconds(response)
                wait_time = delay if retry_after is None else retry_after
                if wait_time > self.RATE_LIMIT_WAIT_CAP_S:
                    raise ADSServiceError(
                        f"ADS rate limit reached (HTTP 429); ADS asks to retry after {wait_time:.0f} s"
                    )
                if attempt == eff_attempts - 1:
                    raise ADSServiceError("ADS rate limit reached (HTTP 429); retries exhausted")
                logger.warning(
                    "ADS rate limit hit (attempt %s); retrying in %.1f s",
                    attempt + 1,
                    wait_time,
                )
                self._sleep_bounded(wait_time, label)
                delay = min(delay * 2, self.RATE_LIMIT_WAIT_CAP_S)
                continue

            if status >= 400:
                raise ADSServiceError(f"ADS API error {status}: {self._response_snippet(response)}")

            try:
                return response.json()
            except ValueError as exc:
                raise ADSServiceError("ADS returned invalid JSON") from exc

        raise ADSServiceError("Exceeded retry attempts for ADS API")

    def _sleep_bounded(self, seconds: float, label: str) -> None:
        """Backoff that never outlasts the calling tool's remaining budget.

        Outside a tool (scripts, the post-stream citation verifier) it is a
        plain sleep. Inside one, a sleep that would leave less than 1 s for
        the next attempt is replaced by the typed error so the tool returns
        its own message before the guard fires.
        """
        from services.tool_budgets import remaining_seconds

        remaining = remaining_seconds()
        if remaining is not None and seconds + 1.0 > remaining:
            raise ADSServiceError(
                f"{label}: retry abandoned — tool budget nearly exhausted ({remaining:.1f} s left)"
            )
        time.sleep(seconds)

    @staticmethod
    def _retry_after_seconds(response: Any) -> Optional[float]:
        try:
            raw = (getattr(response, "headers", None) or {}).get("Retry-After")
        except Exception:  # pragma: no cover - exotic fake responses
            raw = None
        if raw is None:
            return None
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return None  # HTTP-date form: fall back to the backoff ladder

    @staticmethod
    def _response_snippet(response: Any) -> str:
        text = str(getattr(response, "text", "") or "").strip()
        if text.startswith("<"):
            # biblib answers HTML error pages; keep the words, drop the tags.
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
        return text[:500] or "No details"

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

        # Budget hierarchy (2026-09-21): the SDK default is a 600 s timeout with
        # 2 retries — under a 150 s tool guard the query-builder call alone
        # could outlast the tool. Building an ADS query is a 2-5 s completion.
        try:
            _timeout = float(os.getenv("ADS_QUERY_BUILDER_TIMEOUT_SECONDS", "30") or 30)
        except ValueError:
            _timeout = 30.0
        self._client = OpenAI(api_key=api_key, timeout=_timeout, max_retries=0)
        return self._client
