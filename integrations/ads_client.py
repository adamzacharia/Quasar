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
import threading
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
    # Extra ADS calls a zero-hit literature search may spend on relaxing its
    # query (each is one fast search; the tool guard is 150 s).
    _MAX_RELAXATIONS = 4
    # Wall clock after which no EXTRA request (fuzzy bibcode recovery,
    # relaxation) starts, measured from the start of search_natural_language:
    # the main path plus extras stays inside the 150 s tool guard even at the
    # 21 s per-request worst case (guard CX-14).
    _EXTRA_REQUESTS_WALL_S = 45.0
    _MAX_FUZZY_LOOKUPS = 2

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
        """Convert a natural-language question into an ADS query and execute it.

        Order (2026-10-03 literature-search fix):
        1. Bibcodes, DOIs and arXiv ids in the question are looked up directly
           (a wrong-year bibcode is recovered from bibstem/volume/page). A
           question that is only identifiers returns here.
        2. The LLM query builder (``ADS_QUERY_MODEL``, default gpt-oss-120b on
           TACC) translates the rest; when it fails or is off, the
           deterministic ``heuristic_ads_query`` does.
        3. A zero-hit query is relaxed (year widened, then dropped, journal
           dropped, refereed filter dropped), at most ``_MAX_RELAXATIONS``
           extra ADS calls.
        The result says which path built the query (``query_source``) and what
        was relaxed, so the model can tell an empty literature from a bad query
        instead of re-phrasing the same search.
        """
        question = str(question or "").strip()
        sort = normalize_ads_sort(sort)
        attempts: List[Dict[str, Any]] = []
        deadline = time.monotonic() + self._EXTRA_REQUESTS_WALL_S
        ids = extract_literature_identifiers(question)
        id_papers: List[Dict[str, Any]] = []
        resolved: Dict[str, str] = {}
        matched: Dict[str, str] = {}
        id_errors: List[str] = []
        if ids["any"]:
            id_papers, resolved, matched, id_errors = self.lookup_identifiers(
                ids, attempts=attempts, deadline=deadline)
            if not id_papers and id_errors:
                # A failed lookup is not "ADS has no such paper" (guard CX-04).
                raise ADSServiceError(f"ADS identifier lookup failed: {id_errors[0]}")
            if id_papers and _CITATION_INTENT_RE.search(question):
                # "Which papers cited A (and B) [on topic]?" answers with the
                # citing papers of EVERY resolved identifier, narrowed by any
                # topic words (guard CX-11, CX-23).
                return self._citing_papers_answer(question, ids, id_papers, resolved, matched, id_errors,
                                                  attempts, max_results, deadline)
            if ids["only"] or not ids["rest"]:
                out = {
                    "query": attempts[0]["query"] if attempts else question,
                    "sort": "score desc",
                    "filters": None,
                    "rows": len(id_papers),
                    "papers": id_papers,
                    "query_source": "identifier",
                    "resolved_from": resolved,
                    "matched_ids": matched,
                    "relaxed": [],
                    "attempts": attempts,
                    "builder_error": None,
                    "identifier_errors": id_errors,
                }
                return out
        topic = ids["rest"] if ids["any"] else question
        # Counting questions keep their year exactly: a widened year would
        # answer "how many in 2020" with 2021 papers (guard CX-03).
        keep_year = bool(_COUNT_INTENT_RE.search(topic))

        builder_error: Optional[str] = None
        structured: Optional[Dict[str, Any]] = None
        query_source = "fallback"
        if ids["any"] and time.monotonic() > deadline:
            # The identifier lookups used the time budget: skip the LLM call
            # and use the deterministic query (guard CX-14).
            builder_error = "skipped: tool time budget"
        elif ads_query_builder_enabled():
            try:
                structured = ADSQueryBuilder(model=query_model).build_query(topic, max_results, sort, filters)
                query_source = "llm_builder"
            except ADSQueryBuilderError as exc:
                builder_error = str(exc)[:200]
                logger.warning("Query builder failed (%s); falling back to heuristic query.", exc)
        else:
            builder_error = "query builder disabled (ADS_QUERY_BUILDER=off)"
        if structured is None:
            structured = self._heuristic_query(topic, max_results, _intent_sort(topic, None, sort), filters)

        ads_query = structured.get("query", "")
        # Newest-first only for a request about recent work, whoever chose it:
        # the builder echoes gpt-oss's habitual "date desc" despite prompt rule
        # 11, which buried the BICEP2 detection paper (live 2026-10-03).
        resolved_sort = _intent_sort(topic, structured.get("sort"), sort)
        resolved_filters = structured.get("filters", filters)
        try:
            rows = int(structured.get("rows") or max_results)
        except (TypeError, ValueError):
            rows = int(max_results or 10)
        # Never more rows than the caller asked for (guard CX-12).
        rows = max(1, min(rows, int(max_results or rows)))

        try:
            papers = self.search_papers(ads_query, max_results=rows, sort=resolved_sort, filters=resolved_filters)
        except ADSServiceError as exc:
            # Only a syntax rejection of the builder's query gets the
            # deterministic retry; an outage, 401 or 429 is reported as is
            # (guard CX-13).
            if query_source != "llm_builder" or not _is_query_syntax_error(exc):
                raise
            attempts.append({"query": ads_query, "hits": 0, "error": str(exc)[:160]})
            builder_error = f"ADS rejected the builder query: {str(exc)[:120]}"
            structured = self._heuristic_query(topic, max_results, _intent_sort(topic, None, sort), filters)
            ads_query = structured["query"]
            resolved_sort = _intent_sort(topic, structured.get("sort"), sort)
            resolved_filters = structured.get("filters", filters)
            query_source = "fallback"
            papers = self.search_papers(ads_query, max_results=rows, sort=resolved_sort, filters=resolved_filters)
        attempts.append({"query": ads_query, "hits": len(papers)})
        total: Optional[int] = None
        if keep_year and time.monotonic() < deadline:
            # A counting question needs ADS's numFound, not the page size
            # (guard CX-25).
            try:
                total = self.count_matches(ads_query, filters=resolved_filters)
            except ADSServiceError as exc:
                attempts.append({"query": ads_query, "hits": len(papers), "error": f"count: {str(exc)[:120]}"})

        relaxed: List[str] = []
        if not papers and not keep_year:
            # A counting question is never relaxed: any looser match would be
            # counted as an answer to a different question (guard CX-03).
            rebuild = heuristic_ads_query(topic) if query_source == "llm_builder" else None
            steps = relaxation_steps(ads_query, resolved_filters, keep_year=keep_year, rebuild=rebuild)
            for label, q2, f2 in steps[: self._MAX_RELAXATIONS]:
                if time.monotonic() > deadline:
                    attempts.append({"query": q2, "hits": 0, "error": "skipped: tool time budget"})
                    break
                try:
                    found = self.search_papers(q2, max_results=rows, sort=resolved_sort, filters=f2)
                except ADSServiceError as exc:
                    attempts.append({"query": q2, "hits": 0, "error": str(exc)[:160]})
                    break
                attempts.append({"query": q2, "hits": len(found)})
                relaxed.append(label)
                if found:
                    papers, ads_query, resolved_filters = found, q2, f2
                    break

        if id_papers:
            seen = {p.get("bibcode") for p in id_papers}
            papers = id_papers + [p for p in papers if p.get("bibcode") not in seen]

        return {
            "query": ads_query,
            "sort": resolved_sort,
            "filters": resolved_filters,
            "rows": rows,
            "papers": papers,
            "query_source": query_source,
            "resolved_from": resolved,
            "matched_ids": matched,
            "relaxed": relaxed,
            "relaxed_results": bool(relaxed and papers and not id_papers),
            "total": total,
            "total_requested": keep_year,
            "attempts": attempts,
            "builder_error": builder_error,
            "identifier_errors": id_errors if ids["any"] else [],
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
        """Fallback query builder when the LLM builder fails.

        The old fallback quoted the WHOLE question as one phrase
        (keyword:"<sentence>" OR title:"<sentence>" ...), which matches almost
        nothing, so every fielded request returned 0 papers whenever the
        builder model was unavailable (ArchiveBench AB-D-58..60). This one pulls
        out the structure a question usually carries (a year range, a journal,
        a person's name, 'refereed') into ADS fields and AND-s the remaining
        topic words.
        """
        query = heuristic_ads_query(question)
        return {
            "query": query,
            "rows": max(1, default_rows),
            "sort": sort or "score desc",
            "filters": filters or ["property:refereed"],
        }

    # ------------------------------------------------------------------
    # Identifier lookups (bibcode / DOI / arXiv id)
    # ------------------------------------------------------------------
    def count_matches(self, query: str, filters: Optional[List[str]] = None) -> Optional[int]:
        """ADS numFound for a query (rows=0); None when ADS does not report it."""
        params: Dict[str, Union[str, int, List[str]]] = {"q": query, "fl": "bibcode", "rows": 0}
        if filters:
            params["fq"] = filters
        data = self._perform_get("/search/query", params)
        found = (data or {}).get("response", {}).get("numFound") if isinstance(data, dict) else None
        return int(found) if isinstance(found, int) else None

    def lookup_identifiers(
        self,
        ids: Dict[str, Any],
        *,
        attempts: Optional[List[Dict[str, Any]]] = None,
        deadline: Optional[float] = None,
    ) -> "tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, str], List[str]]":
        """Look up the identifiers ``extract_literature_identifiers`` found.

        Exact ``bibcode:`` / ``doi:`` / ``identifier:"arXiv:..."`` queries with
        no refereed filter (an arXiv-only paper is still the paper asked for).
        A bibcode that does not exist is recovered from its bibstem, volume and
        page, which do not depend on the year: Nature Astronomy published
        2021NatAs...5..655G online in 2020, so a model that writes
        ``2020NatAs...5..655G`` gets the real record and a ``resolved_from``
        note instead of an empty result.

        Returns (papers, resolved_from, matched_ids, errors):
        resolved_from maps a requested identifier that did NOT exist as written
        to the bibcode it was recovered as; matched_ids maps a valid DOI/arXiv
        id to its bibcode (not a correction, guard CX-17); errors lists ADS
        failures so the caller does not report them as "no such paper"
        (guard CX-04).
        """
        attempts = attempts if attempts is not None else []
        papers: List[Dict[str, Any]] = []
        resolved: Dict[str, str] = {}
        matched: Dict[str, str] = {}
        errors: List[str] = []
        seen: set = set()

        def _add(found: List[Dict[str, Any]]) -> None:
            for paper in found:
                bib = paper.get("bibcode") or ""
                if bib and bib not in seen:
                    seen.add(bib)
                    papers.append(paper)

        def _run(query: str, rows: int) -> Optional[List[Dict[str, Any]]]:
            try:
                found = self.search_papers(query, max_results=rows, sort="score desc", filters=None)
            except ADSServiceError as exc:
                attempts.append({"query": query, "hits": 0, "error": str(exc)[:160]})
                errors.append(str(exc)[:160])
                return None
            attempts.append({"query": query, "hits": len(found)})
            return found

        bibcodes = list(ids.get("bibcodes") or [])[:5]
        if bibcodes:
            found = _run(" OR ".join(f'bibcode:"{b}"' for b in bibcodes), len(bibcodes))
            if found is not None:
                _add(found)
                hit = {p.get("bibcode") for p in found}
                missed = [b for b in bibcodes if b not in hit]
                for bib in missed[self._MAX_FUZZY_LOOKUPS:]:
                    # Never silent (guard CX-24): the model can look these up singly.
                    errors.append(f"{bib}: not found as written; wrong-year recovery skipped "
                                  f"(at most {self._MAX_FUZZY_LOOKUPS} per call)")
                for bib in missed[: self._MAX_FUZZY_LOOKUPS]:
                    if deadline is not None and time.monotonic() > deadline:
                        errors.append(f"{bib}: wrong-year recovery skipped (tool time budget)")
                        continue
                    fuzzy = bibcode_fuzzy_query(bib)
                    if not fuzzy:
                        continue
                    best = pick_fuzzy_bibcode_match(bib, _run(fuzzy, 5) or [])
                    if best:
                        resolved[bib] = best.get("bibcode", "")
                        _add([best])
        def _late() -> bool:
            return deadline is not None and time.monotonic() > deadline

        dois = list(ids.get("dois") or [])[:5]
        if dois and _late():
            errors.append("DOI lookup skipped: tool time budget")
        elif dois:
            # doi:("...") with parentheses is a Solr 400 (live 2026-10-03): one clause each.
            found = _run(" OR ".join(f'doi:"{d}"' for d in dois), len(dois))
            if found is not None:
                if len(dois) == 1 and len(found) == 1 and found[0].get("bibcode"):
                    matched[dois[0]] = found[0]["bibcode"]
                _add(found)
        arxiv = list(ids.get("arxiv") or [])[:5]
        if arxiv and _late():
            errors.append("arXiv lookup skipped: tool time budget")
        elif arxiv:
            found = _run(" OR ".join(f'identifier:"arXiv:{a}"' for a in arxiv), len(arxiv))
            if found is not None:
                # Formatted papers do not carry the identifier list (kept out of
                # the model's context); a single id maps to its single record.
                if len(arxiv) == 1 and len(found) == 1 and found[0].get("bibcode"):
                    matched[f"arXiv:{arxiv[0]}"] = found[0]["bibcode"]
                _add(found)
        return papers, resolved, matched, errors

    def _citing_papers_answer(self, question, ids, id_papers, resolved, matched, id_errors, attempts,
                              max_results, deadline) -> Dict[str, Any]:
        bibs = [p.get("bibcode") for p in id_papers if p.get("bibcode")][:5]
        if len(bibs) > 1 and _ALL_OF_RE.search(question):
            # papers citing every one of them: intersect the citation sets
            cq = " AND ".join(f'citations(bibcode:"{b}")' for b in bibs)
            combine = "all"
        else:
            cq = "citations(" + " OR ".join(f'bibcode:"{b}"' for b in bibs) + ")"
            combine = "any"
        if len(bibs) > 1 and combine == "all":
            cq = f"({cq})"
        topic_text = _CONNECTIVE_RE.sub(" ", _CITATION_INTENT_RE.sub(" ", ids.get("rest") or ""))
        topic_q = heuristic_ads_query(topic_text) if topic_text.strip() else "*:*"
        if topic_q != "*:*":
            cq = f"{cq} AND ({topic_q})"
        errors = list(id_errors)
        papers: List[Dict[str, Any]] = []
        total: Optional[int] = None
        if time.monotonic() < deadline:
            try:
                papers = self.search_papers(cq, max_results=max_results, sort="citation_count desc", filters=None)
                attempts.append({"query": cq, "hits": len(papers)})
                # The real number of citing papers, not the page size (guard CX-25).
                total = self.count_matches(cq)
            except ADSServiceError as exc:
                attempts.append({"query": cq, "hits": 0, "error": str(exc)[:160]})
                errors.append(str(exc)[:160])
        else:
            errors.append("citations lookup skipped: tool time budget")
        return {
            "total": total,
            "total_requested": True,
            "combine": combine,
            "query": cq,
            "sort": "citation_count desc",
            "filters": None,
            "rows": len(papers),
            "papers": papers,
            "query_source": "citations",
            "cited_papers": id_papers,
            "resolved_from": resolved,
            "matched_ids": matched,
            "relaxed": [],
            "attempts": attempts,
            "builder_error": None,
            "identifier_errors": errors,
            "next_step": ("These are the most-cited papers citing the requested paper(s). For the ones that "
                          "disputed or re-analysed a paper, call find_citing_papers with its bibcode."),
        }

    # ------------------------------------------------------------------
    # Papers that responded to a paper (2026-10-03)
    # ------------------------------------------------------------------
    def resolve_paper_reference(self, reference: str) -> "tuple[Optional[Dict[str, Any]], Optional[str]]":
        """A bibcode / DOI / arXiv id / title -> (ADS paper record, note).

        Identifiers go through :meth:`lookup_identifiers` (so a wrong-year
        bibcode is recovered); anything else is treated as a title."""
        ref = str(reference or "").strip()
        if not ref:
            return None, "no paper given"
        ids = extract_literature_identifiers(ref)
        if ids["any"]:
            papers, resolved, _matched, errors = self.lookup_identifiers(ids)
            if not papers and errors:
                raise ADSServiceError(f"ADS identifier lookup failed: {errors[0]}")
            if papers:
                note = None
                if resolved:
                    note = "; ".join(f"{k} resolved to {v}" for k, v in resolved.items())
                return papers[0], note
            return None, f"no ADS record for {ref}"
        title = re.sub(r'["()]', " ", ref)
        title = re.sub(r"\s+", " ", title).strip()
        papers = self.search_papers(f'title:"{title}"', max_results=3, sort="score desc")
        if not papers:
            words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", title) if w.lower() not in _HEURISTIC_STOP]
            if words:
                papers = self.search_papers("title:(" + " AND ".join(words[:8]) + ")", max_results=3,
                                            sort="citation_count desc")
        if not papers:
            return None, f"no ADS paper matches the title {ref!r}"
        return papers[0], None

    def find_citing_papers(
        self,
        bibcode: str,
        focus: str = "rebuttals",
        max_results: int = 15,
    ) -> Dict[str, Any]:
        """Papers that cite ``bibcode``, grouped for "who pushed back on X".

        ``focus="rebuttals"`` (default) pulls the citing papers whose title or
        abstract uses rebuttal language ("no evidence", "re-analysis", "upper
        limit", "reply to", ...) and re-ranks them by how much of the cited
        paper's own topic they share: rebuttal words alone also match unrelated
        papers that cite X in passing (live 2026-10-03: a K2-18b paper titled
        "Does Not ..." citing the Venus phosphine paper). ``focus="all"``
        returns the most-cited citing papers; any other text is AND-ed onto
        the citation set as unfielded terms.
        """
        self._require_api_key()
        source, note = self.resolve_paper_reference(bibcode)
        if not source:
            return {"success": False, "error": note or "paper not found", "requested": bibcode}
        bib = source.get("bibcode", "")
        cap = self._clamp_int(max_results, default=15, lo=1, hi=50)
        mode = str(focus or "rebuttals").strip()
        mode_key = mode.lower()
        rebuttal_mode = mode_key in _REBUTTAL_FOCUS
        base = f'citations(bibcode:"{bib}")'
        if mode_key in ("all", "any", "everything"):
            query, rows = base, cap
        elif rebuttal_mode:
            terms = " OR ".join(f'"{t}"' if (" " in t or "-" in t) else t for t in _REBUTTAL_TERMS)
            query, rows = f"{base} AND (title:({terms}) OR abs:({terms}))", 100
        else:
            extra = heuristic_ads_query(mode)
            query = base if extra == "*:*" else f"{base} AND ({extra})"
            rows = cap
        found = self.search_papers(query, max_results=rows, sort="citation_count desc", filters=None)

        def _slim(p: Dict[str, Any], score: Optional[int] = None) -> Dict[str, Any]:
            out = {k: p.get(k) for k in ("bibcode", "title", "authors", "year", "journal", "citations", "link")}
            if score is not None:
                out["relevance"] = score
            return out

        result: Dict[str, Any] = {
            "success": True,
            "bibcode": bib,
            "source_title": source.get("title"),
            "source_year": source.get("year"),
            "focus": mode or "rebuttals",
            "query": query,
        }
        if note:
            result["resolved_from"] = note
        if not rebuttal_mode:
            result["citing"] = [_slim(p) for p in found[:cap]]
            result["count"] = len(result["citing"])
            result["papers"] = found[:cap]
            return result

        topic = _distinctive_terms(source.get("title") or "")
        objects = _object_terms(source.get("title") or "")
        rebuttals: List[tuple] = []
        possible: List[tuple] = []
        replies: List[Dict[str, Any]] = []
        other: List[Dict[str, Any]] = []
        for p in found:
            tl = str(p.get("title") or "").lower()
            al = str(p.get("abstract") or "").lower()
            if _REPLY_TITLE_RE.match(tl):
                replies.append(p)
                continue
            r_title = sum(1 for t in _REBUTTAL_TERMS if t in tl)
            r_abs = sum(1 for t in _REBUTTAL_TERMS if t in al)
            t_title = sum(1 for t in topic if t in tl)
            t_abs = sum(1 for t in topic if t in al)
            # Distinct topic words across title AND abstract: "phosphine" in
            # both a Mars paper's title and abstract is still ONE (CX-08 reopen).
            t_unique = sum(1 for t in topic if t in tl or t in al)
            # Sharing the cited paper's topic IN THE TITLE matters most: "upper
            # limits for phosphine on Mars" cites the Venus paper too.
            # Rebuttal language counts once ("upper limit" and "upper limits"
            # overlap); topic overlap decides the order.
            score = 3 * min(r_title, 1) + min(r_abs, 2) + 3 * min(t_title, 3) + min(t_abs, 2)
            # Lexical overlap cannot tell "re-analysed the Venus data" from "set
            # Mars limits and compared them with Venus" (guard CX-08), so the
            # verdict is tiered. A rebuttal shares two topic words IN THE TITLE
            # or names the cited paper's OBJECT (Venus, K2-18 b, BICEP2,
            # 'Oumuamua) in its title; "K2-18b Does Not Meet the Standards of
            # Evidence for Life" shares one word but is plainly a rebuttal (live
            # 2026-10-03). Overlap only via the abstract is a possible rebuttal
            # the model must check before calling it one.
            # Whole-token match: "K2-18" names "K2-18 b"/"K2-18b" but not
            # "K2-180 b" (follow-up guard CX-17).
            names_object = any(re.search(rf"(?<![a-z0-9]){re.escape(o)}(?![0-9])", tl) for o in objects)
            if not (r_title or r_abs) or score < 4 or (t_unique < min(2, len(topic)) and not names_object):
                other.append(p)
            elif t_title >= min(2, len(topic)) or names_object:
                rebuttals.append((score, p))
            else:
                possible.append((score, p))
        rebuttals.sort(key=lambda sp: (-sp[0], -(sp[1].get("citations") or 0)))
        possible.sort(key=lambda sp: (-sp[0], -(sp[1].get("citations") or 0)))
        result["rebuttals"] = [_slim(p, s) for s, p in rebuttals[:cap]]
        result["possible_rebuttals"] = [_slim(p, s) for s, p in possible[:cap]]
        if possible:
            result["possible_note"] = ("possible_rebuttals share the cited paper's topic only partly in the title; "
                                       "they may discuss a different object. Check each title before calling it "
                                       "a rebuttal.")
        result["replies"] = [_slim(p) for p in replies[:5]]
        result["other_citing"] = [_slim(p) for p in other[:5]]
        result["count"] = len(result["rebuttals"])
        result["scanned"] = len(found)
        if len(found) >= 100:
            # guard CX-15: say the scan was bounded instead of implying completeness.
            result["scan_note"] = ("Ranked among the 100 most-cited citing papers that use rebuttal language; "
                                   "a less-cited response can be missing. Use focus with a topic to narrow.")
        result["papers"] = [p for _, p in rebuttals[:cap]] + replies[:5] + [p for _, p in possible[:cap]]
        if not rebuttals and not replies and not possible:
            # Nothing argued with it in so many words: show who cites it most.
            fallback = self.search_papers(base, max_results=min(cap, 10), sort="citation_count desc", filters=None)
            result["other_citing"] = [_slim(p) for p in fallback]
            result["papers"] = fallback
            result["note"] = ("No citing paper uses rebuttal language together with this paper's topic; "
                              "other_citing lists its most-cited citing papers instead.")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Literature identifiers (2026-10-03 literature-search fix)
# ─────────────────────────────────────────────────────────────────────────────
# A canonical ADS bibcode is EXACTLY 19 characters: YYYY + 5 bibstem + 4 volume
# + 1 qualifier + 4 page + 1 author initial, '.'-padded.
_BIBCODE_IN_TEXT_RE = re.compile(r"(?<![A-Za-z0-9&.])((?:1[6-9]|20)\d{2}[A-Za-z][A-Za-z0-9&.]{13}[A-Za-z.])(?![A-Za-z0-9&.])")
_DOI_IN_TEXT_RE = re.compile(r"(?<![\w/.])(10\.\d{4,9}/[^\s\"'<>,;()\[\]]+)")
# An arXiv id inside a sentence needs its 'arXiv' prefix (a bare 1420.4058 is a
# frequency); old-style archive/NNNNNNN ids carry their own prefix.
_ARXIV_PREFIXED_RE = re.compile(
    r"\barxiv\s*[:/]?\s*(?:abs/)?(\d{4}\.\d{4,5}|[a-z][a-z\-]+(?:\.[A-Za-z]{2})?/\d{7})(?:v\d+)?\b",
    re.IGNORECASE,
)
_ARXIV_OLD_RE = re.compile(
    r"(?<![\w/])((?:astro-ph|gr-qc|hep-ph|hep-th|hep-ex|nucl-th|nucl-ex|physics|math-ph|cond-mat|quant-ph)"
    r"(?:\.[A-Za-z]{2})?/\d{7})(?:v\d+)?\b"
)
_ARXIV_BARE_RE = re.compile(r"^\s*(\d{2}(?:0[1-9]|1[0-2])\.\d{4,5})(?:v\d+)?\s*$")
# Words that only frame a lookup ("look up X and give its title"). A question
# that is identifiers plus these words is a pure lookup: no topic search.
_LOOKUP_FILLER = {
    "look", "up", "lookup", "find", "get", "fetch", "show", "give", "tell", "me", "us", "the", "a", "an", "of",
    "its", "it", "is", "what", "which", "who", "whose", "and", "or", "with", "for", "to", "in", "on", "this",
    "that", "paper", "papers", "article", "articles", "record", "records", "entry", "title", "titles", "author",
    "authors", "first", "bibcode", "bibcodes", "doi", "dois", "arxiv", "id", "ids", "identifier", "identifiers",
    "journal", "reference", "references", "ref", "published", "publication", "details", "abstract", "abstracts",
    "citation", "citations", "cited", "metadata", "info", "information", "about", "please", "can", "you", "ads",
    "nasa", "full", "list", "year", "link", "links", "corresponding", "same", "by", "from", "has", "have",
}


# "Which papers cited X" / "how many papers ..." (guard CX-11, CX-03).
_CITATION_INTENT_RE = re.compile(r"\b(?:cit(?:e|ed|es|ing|ation|ations)|cited\s+by)\b", re.IGNORECASE)
# "papers citing BOTH A and B" means the intersection (guard CX-23 round 3).
_ALL_OF_RE = re.compile(r"\bboth\b|\ball\s+(?:of\s+)?(?:these|them|those|the\s+(?:two|three|papers))\b|\beach\s+of\b",
                        re.IGNORECASE)
_CONNECTIVE_RE = re.compile(r"\b(?:both|either|each|all|these|those|them|of|and|or|which|what|papers?)\b",
                            re.IGNORECASE)
_COUNT_INTENT_RE = re.compile(r"\bhow\s+many\b|\bnumber\s+of\b|\bcount(?:s|ing)?\b|\btotal\b", re.IGNORECASE)


_VALID_SORTS = ("date desc", "date asc", "citation_count desc", "citation_count_norm desc", "score desc",
                "read_count desc", "citation_count asc")
_SORT_ALIASES = {
    "date": "date desc", "newest": "date desc", "recent": "date desc", "latest": "date desc",
    "oldest": "date asc", "citations": "citation_count desc", "citation_count": "citation_count desc",
    "cited": "citation_count desc", "most cited": "citation_count desc", "relevance": "score desc",
    "score": "score desc", "reads": "read_count desc", "read_count": "read_count desc",
}


# Recency wording only: "current-induced" / "a new interpretation" are
# science, but "new papers" / "new results" ask for newly published work
# (follow-up guard CX-08, CX-16).
_RECENT_RE = re.compile(
    r"\b(?:recent(?:ly)?|latest|newest|this\s+year|last\s+(?:year|\d+\s+years|few\s+years)|"
    r"since\s+(?:19|20)\d{2}|upcoming|just\s+published|"
    r"new\s+(?:papers?|results?|studies|publications?|preprints?|work|articles?|observations?))\b",
    re.IGNORECASE,
)
_MOST_CITED_RE = re.compile(
    r"\b(?:most[\s-]+cited|highly[\s-]+cited|top[\s-]+cited|most\s+influential|influential|seminal|landmark|"
    r"best|most\s+important|classic)\b",
    re.IGNORECASE,
)
_OLDEST_RE = re.compile(r"\boldest\b|\bin\s+chronological\s+order\b", re.IGNORECASE)


def _intent_sort(question: str, chosen: Optional[str], caller: Optional[str] = None) -> str:
    """The sort the REQUEST asks for, decided in code (the builder echoed
    gpt-oss's habitual "date desc" despite its prompt rule, live 2026-10-03):
    most-cited wording -> citation_count desc; recent wording -> date desc;
    "oldest" -> date asc; otherwise the builder's / caller's choice, except
    that "date desc" without recent wording becomes relevance (it ranked "the
    paper that first reported X" off the first page)."""
    q = str(question or "")
    if _MOST_CITED_RE.search(q):
        return "citation_count desc"
    if _RECENT_RE.search(q):
        return "date desc"
    if _OLDEST_RE.search(q):
        return "date asc"
    pick = normalize_ads_sort(chosen) or normalize_ads_sort(caller) or "score desc"
    return "score desc" if pick == "date desc" else pick


def normalize_ads_sort(sort: Any) -> Optional[str]:
    """A valid ADS sort or None (let the builder choose). gpt-oss sends
    shorthands such as "date" (live 2026-10-03), which ADS does not accept."""
    text = re.sub(r"\s+", " ", str(sort or "").strip().lower())
    if not text:
        return None
    if text in _VALID_SORTS:
        return text
    return _SORT_ALIASES.get(text)


def _is_query_syntax_error(exc: Exception) -> bool:
    """ADS rejected the query string itself, not the service.

    A 400 is always the query. A 500 counts only when it carries a Solr
    response body (the dotted ``abs:(...)`` failure of 2026-10-03 did); a bare
    "500: Internal Server Error" is the service (guard CX-13 reopen)."""
    text = str(exc)
    if re.search(r"\berror 400\b|SyntaxError|ParseException|org\.apache\.solr", text):
        return True
    return bool(re.search(r"\berror 500\b", text) and re.search(r"responseHeader|error-class|SolrException", text))


def _clean_doi(doi: str) -> str:
    return doi.rstrip(".,;:)]}'\"?!")


def extract_literature_identifiers(text: str) -> Dict[str, Any]:
    """Find ADS bibcodes, DOIs and arXiv ids in a literature request.

    ``only`` is True when nothing but identifiers and lookup filler words is
    left (``"2020NatAs...5..655G"``, ``"look up 2018ApJ...869L..41A"``), so the
    caller can skip the topic search. ``rest`` is the request with the
    identifiers removed (what a topic search should use)."""
    raw = str(text or "")
    rest = raw
    bibcodes: List[str] = []
    dois: List[str] = []
    arxiv: List[str] = []

    for m in _DOI_IN_TEXT_RE.finditer(raw):
        doi = _clean_doi(m.group(1))
        if doi and doi not in dois:
            dois.append(doi)
        rest = rest.replace(m.group(1), " ")
    rest = re.sub(r"\bdoi\s*:\s*", " ", rest, flags=re.IGNORECASE)
    for m in _ARXIV_PREFIXED_RE.finditer(rest):
        if m.group(1) not in arxiv:
            arxiv.append(m.group(1))
    rest = _ARXIV_PREFIXED_RE.sub(" ", rest)
    for m in _ARXIV_OLD_RE.finditer(rest):
        if m.group(1) not in arxiv:
            arxiv.append(m.group(1))
    rest = _ARXIV_OLD_RE.sub(" ", rest)
    bare = _ARXIV_BARE_RE.match(rest)
    if bare and not arxiv:
        arxiv.append(bare.group(1))
        rest = " "
    for m in _BIBCODE_IN_TEXT_RE.finditer(rest):
        bib = m.group(1)
        # A real bibcode is padded with '.' or carries a volume/page; plain
        # 19-letter words never reach here (they need a 4-digit year first).
        if bib not in bibcodes:
            bibcodes.append(bib)
    rest = _BIBCODE_IN_TEXT_RE.sub(" ", rest)

    found = bool(bibcodes or dois or arxiv)
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", rest)]
    content = [w for w in words if w not in _LOOKUP_FILLER and w not in _HEURISTIC_STOP]
    rest = re.sub(r"\s+", " ", rest).strip(" ,;:.?!")
    return {
        "bibcodes": bibcodes,
        "dois": dois,
        "arxiv": arxiv,
        "any": found,
        "only": found and not content,
        "rest": rest if content else "",
    }


def bibcode_fuzzy_query(bibcode: str) -> Optional[str]:
    """ADS query for a bibcode's bibstem + volume + page with NO year.

    ``2020NatAs...5..655G`` -> ``bibstem:"NatAs" AND volume:"5" AND page:"655"``.
    The qualifier letter (L for Letters, A for A&A article numbers) is part of
    the ADS page (``L41``, ``A133``)."""
    bib = str(bibcode or "").strip()
    if len(bib) != 19:
        return None
    bibstem = bib[4:9].strip(".")
    volume = bib[9:13].strip(".")
    qualifier = bib[13]
    page = bib[14:18].strip(".")
    if not bibstem or not page:
        return None
    if qualifier.isalpha():
        page = qualifier + page
    parts = [f'bibstem:"{bibstem}"']
    if volume:
        parts.append(f'volume:"{volume}"')
    parts.append(f'page:"{page}"')
    return " AND ".join(parts)


def pick_fuzzy_bibcode_match(requested: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Choose the record a mistyped bibcode meant: same bibstem/volume/page
    (the query already guarantees that), first-author initial equal, and the
    year within two of the requested one. None when nothing qualifies."""
    req = str(requested or "")
    try:
        req_year = int(req[:4])
    except ValueError:
        return None
    best = None
    best_gap = None
    for paper in candidates or []:
        bib = str(paper.get("bibcode") or "")
        if len(bib) != 19:
            continue
        try:
            year = int(bib[:4])
        except ValueError:
            continue
        gap = abs(year - req_year)
        if gap > 2 or bib[18].upper() != req[18].upper():
            continue
        if best is None or gap < best_gap:
            best, best_gap = paper, gap
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Zero-hit relaxation (2026-10-03)
# ─────────────────────────────────────────────────────────────────────────────
_CLAUSE_VALUE = r'(?:\[[^\]]*\]|"[^"]*"|\([^()]*\)|[^\s()]+)'


def _balanced(query: str) -> bool:
    depth = 0
    in_quote = False
    for ch in query:
        if ch == '"':
            in_quote = not in_quote
        elif not in_quote and ch == "(":
            depth += 1
        elif not in_quote and ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not in_quote


def _tidy_connectives(query: str) -> str:
    q = re.sub(r"\s+", " ", query).strip()
    for _ in range(4):
        q = re.sub(r"\b(AND|OR|NOT)\s+(?=(AND|OR)\b)", "", q)
        q = re.sub(r"\(\s*(AND|OR)\s+", "(", q)
        q = re.sub(r"\s+(AND|OR|NOT)\s*\)", ")", q)
        q = re.sub(r"^\s*(AND|OR)\s+", "", q)
        q = re.sub(r"\s+(AND|OR|NOT)\s*$", "", q)
        q = re.sub(r"\(\s*\)", "", q)
        q = re.sub(r"\s+", " ", q).strip()
    return q


def drop_query_clause(query: str, field: str) -> Optional[str]:
    """Remove every ``field:value`` clause (and its dangling AND/OR) from an
    ADS query. None when the field is absent or the result is empty or
    unbalanced (then that relaxation is skipped)."""
    pattern = re.compile(rf"(?<![\w:])\^?{re.escape(field)}:\s*{_CLAUSE_VALUE}", re.IGNORECASE)
    if not pattern.search(query or ""):
        return None
    out = _tidy_connectives(pattern.sub(" ", query))
    if not out or not _balanced(out) or out == query:
        return None
    return out


def widen_year_clause(query: str) -> Optional["tuple[str, str]"]:
    """``year:2020`` / ``year:2020-2021`` / ``year:[2020 TO 2021]`` widened by
    one year each side. Returns (new_query, label) or None."""
    q = query or ""
    m = re.search(r"\byear:\s*\[\s*(\d{4})\s+TO\s+(\d{4}|\*)\s*\]", q, re.IGNORECASE)
    if m:
        lo = int(m.group(1)) - 1
        hi = m.group(2) if m.group(2) == "*" else str(int(m.group(2)) + 1)
        label = f"widened year {m.group(1)}-{m.group(2)} to {lo}-{hi}"
        return q[:m.start()] + f"year:[{lo} TO {hi}]" + q[m.end():], label
    m = re.search(r'\byear:\s*"?(\d{4})(?:\s*-\s*(\d{4}))?"?(?![\d\-])', q)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        shown = m.group(1) if not m.group(2) else f"{m.group(1)}-{m.group(2)}"
        return q[:m.start()] + f"year:[{lo - 1} TO {hi + 1}]" + q[m.end():], f"widened year {shown} to {lo - 1}-{hi + 1}"
    return None


def relaxation_steps(
    query: str,
    filters: Optional[List[str]],
    *,
    keep_year: bool = False,
    rebuild: Optional[str] = None,
) -> List["tuple[str, str, Optional[List[str]]]"]:
    """Ordered (label, query, filters) relaxations for a zero-hit ADS query:
    widen the year, drop the year and the journal, rebuild the query without
    the LLM (``rebuild``), include non-refereed records. ``keep_year`` (a
    counting question) never touches the year (guard CX-03). Four steps at
    most, so the ``_MAX_RELAXATIONS`` cap never hides one (guard CX-07)."""
    steps: List["tuple[str, str, Optional[List[str]]]"] = []
    current = query
    if not keep_year:
        widened = widen_year_clause(current)
        if widened and widened[0] != current:
            steps.append((widened[1], widened[0], filters))
    loosest = current
    dropped_parts = []
    if not keep_year:
        no_year = drop_query_clause(loosest, "year")
        if no_year:
            loosest, dropped_parts = no_year, ["the year"]
    no_journal = drop_query_clause(loosest, "bibstem")
    if no_journal:
        loosest = no_journal
        dropped_parts.append("the journal")
    if dropped_parts:
        steps.append(("dropped " + " and ".join(dropped_parts), loosest, filters))
    if rebuild and rebuild not in ("*:*", query, loosest):
        steps.append(("rebuilt the query without the LLM", rebuild, filters))
    if filters and any(str(f).strip().lower() == "property:refereed" for f in filters):
        rest = [f for f in filters if str(f).strip().lower() != "property:refereed"]
        steps.append(("included non-refereed records", loosest, rest or None))
    return steps


def validate_ads_query(query: str) -> Optional[str]:
    """Why an ADS query string is unusable, or None when it looks fine."""
    q = str(query or "").strip()
    if not q:
        return "empty query"
    if len(q) > 1500:
        return "query too long"
    if re.search(r"(?<![\w])(object|simbad):", q, re.IGNORECASE):
        return "object:/simbad: fields are not supported by the search API"
    if q.count('"') % 2:
        return "unbalanced quotes"
    if not _balanced(q):
        return "unbalanced parentheses"
    return None


_HEURISTIC_JOURNALS = (
    ("astrophysical journal letters", "ApJL"), ("astrophysical journal supplement", "ApJS"),
    ("astrophysical journal", "ApJ"), ("astronomical journal", "AJ"), ("monthly notices", "MNRAS"),
    ("astronomy and astrophysics", "A&A"), ("astronomy & astrophysics", "A&A"), ("nature astronomy", "NatAs"),
    ("annual review of astronomy", "ARA&A"), ("publications of the astronomical society of the pacific", "PASP"),
)
_HEURISTIC_JOURNAL_ABBR = {"apjl": "ApJL", "apjs": "ApJS", "apj": "ApJ", "mnras": "MNRAS", "a&a": "A&A",
                           "aj": "AJ", "pasp": "PASP", "natas": "NatAs", "ara&a": "ARA&A"}
_HEURISTIC_STOP = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "by", "with", "from", "at", "as", "is", "are",
    "was", "were", "be", "been", "has", "have", "had", "do", "does", "did", "how", "many", "much", "what", "which",
    "who", "whom", "when", "where", "why", "that", "this", "these", "those", "there", "their", "its", "it", "me",
    "give", "list", "find", "show", "tell", "papers", "paper", "articles", "article", "publications", "published",
    "publish", "refereed", "peer", "reviewed", "journal", "between", "since", "after", "before", "inclusive",
    "year", "years", "first", "reported", "report", "any", "all", "some", "about", "into", "please", "can", "you",
    "i", "my", "our", "we", "bibcode", "bibcodes", "doi", "dois", "also", "than", "more", "most", "recent",
    "not", "no", "nor", "near", "to", "via", "using", "use", "used",
}
_NAME_PARTICLES = {"van", "von", "de", "der", "den", "da", "di", "du", "del", "la", "le", "st"}
# A capitalised run is only an AUTHOR when a cue says so ("by Jane Doe",
# "did Jane Doe publish", "Doe et al."). Without a cue, "Venus Greaves" stays
# two topic words: unfielded search matches author names anyway.
_AUTHOR_CUE_BEFORE = {"by", "did", "does", "author", "authors", "authored", "coauthor", "coauthored", "co-authored"}
_AUTHOR_CUE_AFTER = {"et", "publish", "published", "publishes", "wrote", "writes", "authored"}
# Capitalised runs that name a facility or a team, never a person.
_NOT_A_PERSON = {"telescope", "observatory", "space", "array", "survey", "mission", "satellite", "explorer",
                 "interferometer", "collaboration", "team", "project", "consortium", "experiment", "network"}


# Rebuttal language for find_citing_papers (titles and abstracts, lowercase).
# The bare word "not" is excluded: it pulled an unrelated K2-18b paper into
# the Venus phosphine responses (live 2026-10-03).
_REBUTTAL_TERMS = (
    "no evidence", "insufficient evidence", "non-detection", "nondetection", "upper limit", "upper limits",
    "re-analysis", "reanalysis", "reanalyses", "reassessment", "reassessing", "re-examination", "reexamination",
    "revisit", "revisiting", "statistical reliability", "no statistically significant", "complications",
    "comment on", "reply to", "matters arising", "spurious", "artefact", "artifact", "alternative explanation",
    "not confirmed", "unconfirmed", "challenge", "challenges", "does not", "cannot be", "contamination",
    "refute", "refuting", "rebuttal", "inconsistent with", "doubt", "caution", "tension with",
)
_REBUTTAL_FOCUS = {"rebuttals", "rebuttal", "pushback", "criticism", "challenges", "responses", "disputes", ""}
_REPLY_TITLE_RE = re.compile(r"^\s*(reply to|response to|authors?['’]? reply|author['’]s reply)")
_GENERIC_TITLE_WORDS = {
    "detection", "detections", "observations", "observation", "observed", "evidence", "analysis", "study",
    "new", "first", "results", "result", "data", "using", "model", "models", "constraints", "survey",
    "properties", "possible", "gas", "measurement", "measurements", "search", "jwst", "hst", "alma", "from",
    "with", "for", "the", "and", "towards", "toward", "between", "into", "via", "high", "low", "large", "small",
    # instruments / facilities / generic nouns: shared by unrelated papers, never
    # "the same object" (a GJ 3473 b JWST/MIRI paper is not a K2-18 b rebuttal)
    "atmosphere", "atmospheres", "miri", "nirspec", "niriss", "nircam", "nirc", "vla", "vlba", "gbt", "sofia",
    "jcmt", "vlt", "keck", "gemini", "chandra", "xmm", "spitzer", "tess", "kepler", "gaia", "sdss", "lsst",
    "planck", "wmap", "euclid", "eht", "ska", "apex", "iram", "noema", "sma", "great",
}


def _object_terms(title: str) -> List[str]:
    """The cited paper's object / experiment names (lowercase): words with a
    digit (K2-18, BICEP2, 1I/2017), acronyms (DMS), and, when the title is not
    in Title Case, capitalised words after the first (Venus, 'Oumuamua)."""
    words = re.findall(r"[A-Za-z0-9'][A-Za-z0-9\-'/]*", str(title or ""))
    letters = [w for w in words if w[:1].isalpha() or w[:1] == "'"]
    title_case = bool(letters) and sum(1 for w in letters if w.lstrip("'")[:1].isupper()) > 0.6 * len(letters)
    out: List[str] = []
    for idx, w in enumerate(words):
        lw = w.lower().strip("'").rstrip("/")
        if len(lw) < 2 or lw in _HEURISTIC_STOP or lw in _GENERIC_TITLE_WORDS:
            continue
        core = w.strip("'")
        is_obj = (any(ch.isdigit() for ch in core)
                  or (len(core) >= 3 and core.isupper())
                  or (not title_case and idx > 0 and core[:1].isupper()))
        if is_obj and lw not in out:
            out.append(lw)
    return out[:6]


def _distinctive_terms(title: str) -> List[str]:
    """Topic words of a paper title a response would share (lowercase)."""
    out: List[str] = []
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", str(title or "")):
        lw = w.lower().strip("'")
        if len(lw) < 3 or lw in _HEURISTIC_STOP or lw in _GENERIC_TITLE_WORDS:
            continue
        if lw not in out:
            out.append(lw)
    return out[:8]

def _unfielded_term(word: str) -> str:
    """One topic word as an unfielded ADS term; anything beyond letters and
    digits (K2-18, J1234+5678) is quoted so Solr does not read an operator."""
    if re.fullmatch(r"[A-Za-z0-9]+", word):
        return word
    return '"' + word.replace('"', "") + '"'


def heuristic_ads_query(question: str) -> str:
    """Turn a natural-language literature question into a fielded ADS query
    without an LLM: year range -> year:[a TO b], journal name -> bibstem:,
    a cued personal name -> author:"Last, First", everything else -> unfielded
    terms AND-ed together (ADS searches abstract, title, keywords and authors
    for an unfielded term; measured 2026-10-03: ``Greaves phosphine Venus``
    ranks the right paper first where ``abs:(Greaves AND ...)`` finds nothing).
    Falls back to *:* only for an empty question."""
    import re as _re

    text = str(question or "").replace('"', " ").strip()
    # Identifiers are looked up separately; inside a field query their dots
    # make Solr answer HTTP 500.
    text = _BIBCODE_IN_TEXT_RE.sub(" ", text)
    text = _DOI_IN_TEXT_RE.sub(" ", text)
    text = _ARXIV_PREFIXED_RE.sub(" ", text)
    if not text.strip():
        return "*:*"
    parts: List[str] = []
    low = text.lower()
    # years
    m = _re.search(r"\b((?:19|20)\d{2})\s*(?:-|–|to|and|through|until)\s*((?:19|20)\d{2})\b", low)
    if m:
        parts.append(f"year:[{m.group(1)} TO {m.group(2)}]")
        text = text[:m.start()] + " " + text[m.end():]
    else:
        m = _re.search(r"\b(?:since|after|from)\s+((?:19|20)\d{2})\b", low)
        if m:
            parts.append(f"year:[{m.group(1)} TO *]")
            text = text[:m.start()] + " " + text[m.end():]
        else:
            m = _re.search(r"\b(?:in|during)\s+((?:19|20)\d{2})\b", low) or \
                _re.search(r"(?<![\w\-])((?:19|20)\d{2})(?![\w\-])", low)
            if m:
                parts.append(f"year:{m.group(1)}")
                text = text[:m.start()] + " " + text[m.end():]
    # journals
    low = text.lower()
    for phrase, stem in _HEURISTIC_JOURNALS:
        idx = low.find(phrase)
        if idx >= 0:
            parts.append(f'bibstem:"{stem}"')
            text = text[:idx] + " " + text[idx + len(phrase):]
            low = text.lower()
            break
    else:
        for tok in _re.findall(r"[A-Za-z&]+", text):
            stem = _HEURISTIC_JOURNAL_ABBR.get(tok.lower())
            if stem and (tok.isupper() or tok in ("ApJ", "ApJL", "ApJS", "A&A", "NatAs")):
                parts.append(f'bibstem:"{stem}"')
                text = _re.sub(r"\b" + _re.escape(tok) + r"\b", " ", text, count=1)
                break
    # a personal name: 1-4 capitalised tokens (particles allowed) with an
    # author cue right before or after it
    tokens = _re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
    name = None
    for i in range(0, len(tokens)):
        run = []
        j = i
        while j < len(tokens) and (tokens[j][0].isupper() or tokens[j].lower() in _NAME_PARTICLES) and \
                tokens[j].lower() not in _HEURISTIC_STOP:
            run.append(tokens[j])
            j += 1
        caps = [t for t in run if t[0].isupper()]
        if not (1 <= len(run) <= 4 and caps and run[0][0].isupper() and run[-1][0].isupper()
                and not any(t.isupper() and len(t) > 1 for t in run)
                and not any(t.lower() in _NOT_A_PERSON for t in run)):
            continue
        before = tokens[i - 1].lower() if i > 0 else ""
        after = tokens[j].lower() if j < len(tokens) else ""
        if before in _AUTHOR_CUE_BEFORE or after in _AUTHOR_CUE_AFTER:
            name = run
            break
    if name:
        if len(name) == 1:
            parts.append(f'author:"{name[0]}"')
        else:
            first, last = name[0], " ".join(name[1:])
            parts.append(f'author:"{last}, {first}"')
        for t in name:
            text = _re.sub(r"\b" + _re.escape(t) + r"\b", " ", text, count=1)
        text = _re.sub(r"\bet\s+al\b\.?", " ", text, flags=_re.IGNORECASE)
    words = []
    for w in _re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\+\.]*", text):
        w = w.rstrip(".")
        if ".." in w or len(w) <= 1 or w.lower() in _HEURISTIC_STOP or w.upper() in ("AND", "OR", "NOT"):
            continue
        words.append(w)
    words = list(dict.fromkeys(words))[:8]
    if words:
        parts.append(" AND ".join(_unfielded_term(w) for w in words))
    return " AND ".join(parts) if parts else "*:*"


# ─────────────────────────────────────────────────────────────────────────────
# LLM query builder
# ─────────────────────────────────────────────────────────────────────────────
def ads_query_builder_model() -> str:
    """The builder's model. ``ADS_QUERY_MODEL``, else the free TACC
    gpt-oss-120b. It deliberately does NOT follow DEFAULT_LLM_MODEL: a paid
    chat default must never become the background translator, and the old
    code sent that name to the OpenAI SDK, where ``deepseek-v4-pro`` answered
    404 on every call (2026-10-03)."""
    return (os.getenv("ADS_QUERY_MODEL") or "gpt-oss-120b").strip()


def ads_query_builder_enabled() -> bool:
    """``ADS_QUERY_BUILDER=off`` skips the LLM and uses the deterministic path."""
    return os.getenv("ADS_QUERY_BUILDER", "on").strip().lower() not in ("0", "off", "false", "no")


def ads_query_builder_timeout() -> float:
    try:
        return max(1.0, float(os.getenv("ADS_QUERY_BUILDER_TIMEOUT_SECONDS", "15") or 15))
    except ValueError:
        return 15.0


_BUILDER_CACHE: "Dict[tuple, tuple]" = {}
# At most two builder calls in flight: a timed-out call keeps running in its
# daemon thread (the provider SDK has no per-call cancel), so a busy builder
# falls back to the deterministic query instead of stacking calls (guard CX-02).
_BUILDER_SLOTS = threading.BoundedSemaphore(2)
_BUILDER_CACHE_MAX = 256
_BUILDER_CACHE_TTL_S = 3600.0
_BUILDER_CLIENTS: Dict[str, Any] = {}


def _first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """The first balanced {...} in ``text`` parsed as JSON (strings respected)."""
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
                    return obj if isinstance(obj, dict) else None
        start = text.find("{", start + 1)
    return None


def _default_builder_llm_call(instructions: str, prompt: str, model: str, max_tokens: int) -> str:
    """One provider-aware call (TACC / DeepSeek / OpenAI / Anthropic routed by
    model name), the same shape as core/web_planner.py's planner call."""
    from core.llm_client import LLMClient

    client = _BUILDER_CLIENTS.get(model)
    if client is None:
        client = LLMClient(model=model)
        _BUILDER_CLIENTS[model] = client
    # gpt-oss reasoning effort for this short translation call (ignored by
    # other providers). Measured 2026-10-03 on TACC: see the plan folder.
    effort = (os.getenv("ADS_QUERY_BUILDER_REASONING") or "low").strip().lower()
    extra = {"_gpt_oss_reasoning": effort} if effort in ("low", "medium", "high") else {}
    resp = client.responses.create(
        model=model, instructions=instructions, input=prompt,
        temperature=0, max_output_tokens=max_tokens, **extra,
    )
    return str(getattr(resp, "output_text", "") or "")


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

  phosphine Venus                  — UNFIELDED terms: searched in abstract, title, keywords AND author names
  keyword:"protoplanetary disks"   — ADS controlled-vocabulary keywords (good for broad topic search)
  title:"disk gaps"                — words in paper title (high precision)
  abstract:"dust continuum"        — words in abstract (medium precision, catches tangential mentions)
  body:"gap opening mechanism"     — full-text search inside the paper (use when abstract is too narrow)
  "HL Tau"                         — plain text search/object name (best for target/object searches)
  author:"Andrews, Sean"           — author name (Last, First); author:"Greaves" for a surname only
  ^author:"Andrews, Sean"          — FIRST author only
  orcid:0000-0001-2345-6789        — search by ORCID
  year:2020-2024                   — year range
  bibcode:"2018ApJ...869L..41A"    — specific paper
  doi:"10.3847/..."                — DOI lookup
  identifier:"arXiv:1812.04040"    — arXiv id lookup
  bibstem:"NatAs"                  — journal (ApJ, ApJL, MNRAS, A&A, AJ, NatAs, Natur, Sci, PhRvL ...)
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

7. For ONE SPECIFIC PAPER ("the paper that first reported X", a remembered title,
   "Greaves 2020 Nature Astronomy"): use distinctive title words with title:(...)
   or unfielded terms, plus the first author's surname as author:"Surname" when
   given, sort "score desc". NEVER put author names inside abstract:/abs:.

8. A single year the user remembers is often off by one (online vs print
   publication): use year:[Y-1 TO Y+1], never a hard year:Y, when looking for
   one specific paper. Exact years are fine for counting questions
   ("how many papers in 2019").

9. For "who disputed / challenged / pushed back on / responded to X": if the
   bibcode of X is known, use citations(bibcode:"X") AND title:("no evidence"
   OR "re-analysis" OR "upper limit" OR "reply to" OR "comment on" ...) with
   sort "citation_count desc"; otherwise search the topic with those words.

10. Never invent bibcodes, DOIs or arXiv ids, and never add an author:
    clause for a name the request does not contain. Use only what the request
    says; a wrong guess returns nothing.

11. "caller_sort_preference", when present, is the calling model's guess. Use
    "date desc" only when the request asks for recent / latest / new work, and
    "citation_count desc" only for best / most-cited / influential; otherwise
    sort "score desc", which keeps a specific paper on the first page.

═══ EXAMPLES ═══

"recent papers on protoplanetary disks"
→ {"query": "keyword:\\"protoplanetary disks\\"", "sort": "date desc", "rows": 15, "filters": ["property:refereed"]}

"best ALMA papers on disk gaps"
→ {"query": "bibgroup:ALMA AND (keyword:\\"protoplanetary disks\\" AND (title:\\"gap\\" OR title:\\"ring\\"))", "sort": "citation_count desc", "rows": 15, "filters": ["property:refereed"]}

"papers about HL Tau"
→ {"query": "\\"HL Tau\\"", "sort": "score desc", "rows": 15, "filters": ["property:refereed"]}

"what are people reading about FRBs right now"
→ {"query": "trending(keyword:\\"fast radio bursts\\")", "sort": "score desc", "rows": 15}

"foundational papers on planet formation"
→ {"query": "useful(keyword:\\"planet formation\\" AND year:2015-2026)", "sort": "score desc", "rows": 20, "filters": ["property:refereed"]}

"papers by Sean Andrews on ALMA disk surveys"
→ {"query": "author:\\"Andrews, Sean\\" AND bibgroup:ALMA AND keyword:\\"protoplanetary disks\\"", "sort": "score desc", "rows": 20, "filters": ["property:refereed"]}

"review articles on AGN feedback"
→ {"query": "reviews(keyword:\\"active galactic nuclei\\" AND keyword:\\"feedback\\")", "sort": "citation_count desc", "rows": 15, "filters": ["property:refereed"]}

"papers with JWST data on high-z galaxies since 2023"
→ {"query": "bibgroup:HST AND keyword:\\"high-redshift galaxies\\" AND year:2023-2026", "sort": "date desc", "rows": 15, "filters": ["property:refereed"]}

"full text search for gap opening mechanism in disks"
→ {"query": "body:\\"gap opening\\" AND keyword:\\"protoplanetary disks\\"", "sort": "score desc", "rows": 15, "filters": ["property:refereed"]}

"Greaves phosphine Venus Nature Astronomy 2020"
→ {"query": "title:(phosphine Venus) AND author:\\"Greaves\\" AND year:[2019 TO 2021]", "sort": "score desc", "rows": 10, "filters": ["property:refereed"]}

"first paper reporting phosphine in the atmosphere of Venus"
→ {"query": "title:(phosphine Venus)", "sort": "score desc", "rows": 15, "filters": ["property:refereed"]}

"papers that disputed the BICEP2 B-mode detection"
→ {"query": "BICEP2 AND (title:(dust OR foreground OR \\"joint analysis\\") OR abs:(\\"no evidence\\" OR \\"dust contamination\\"))", "sort": "citation_count desc", "rows": 15, "filters": ["property:refereed"]}
"""

    def __init__(self, model: Optional[str] = None, llm_call: Optional[Any] = None) -> None:
        self.model = (model or ads_query_builder_model()).strip()
        self._llm_call = llm_call or _default_builder_llm_call

    def build_query(
        self,
        question: str,
        default_rows: int,
        sort: Optional[str],
        filters: Optional[List[str]],
    ) -> Dict[str, Any]:
        if not str(question or "").strip():
            raise ADSQueryBuilderError("Empty natural-language question")

        key = (self.model, question.strip(), int(default_rows or 0), sort or "", tuple(filters or ()))
        hit = _BUILDER_CACHE.get(key)
        if hit and time.monotonic() - hit[0] < _BUILDER_CACHE_TTL_S:
            logger.info("[ADS BUILDER] cache hit query=%s", hit[1].get("query"))
            return json.loads(json.dumps(hit[1]))

        t0 = time.monotonic()
        try:
            response = self._call_model(question, default_rows, sort, filters)
            structured = self._parse_response(response)
            query = str(structured.get("query") or "").strip()
            problem = validate_ads_query(query)
            if problem:
                raise ADSQueryBuilderError(f"builder produced an unusable query ({problem}): {query[:160]}")
        except ADSQueryBuilderError as exc:
            print(f"[ADS BUILDER] model={self.model} ms={(time.monotonic() - t0) * 1000:.0f} fail({str(exc)[:160]})")
            raise
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[ADS BUILDER] model={self.model} ms={(time.monotonic() - t0) * 1000:.0f} fail({type(exc).__name__}: {str(exc)[:160]})")
            raise ADSQueryBuilderError(str(exc)) from exc

        structured["query"] = query
        try:
            structured["rows"] = max(1, min(int(structured.get("rows") or default_rows), 200))
        except (TypeError, ValueError):
            structured["rows"] = max(1, int(default_rows or 10))
        if default_rows:
            # Never more rows than the caller asked for (guard CX-12).
            structured["rows"] = min(structured["rows"], int(default_rows))
        # The builder's sort wins: gpt-oss sends sort="date desc" on nearly
        # every call, which ranked "the paper that first reported X" off the
        # page and made it re-search for it all turn (live 2026-10-03). The
        # caller's sort is only a preference the builder may follow.
        built_sort = normalize_ads_sort(structured.get("sort"))
        structured["sort"] = built_sort or sort or "score desc"
        built_filters = structured.get("filters")
        if not isinstance(built_filters, list) or not all(isinstance(f, str) for f in built_filters):
            structured["filters"] = filters or ["property:refereed"]
        print(
            f"[ADS BUILDER] model={self.model} ms={(time.monotonic() - t0) * 1000:.0f} ok "
            f"query={query[:200]!r} sort={structured['sort']!r}"
        )
        if len(_BUILDER_CACHE) >= _BUILDER_CACHE_MAX:
            _BUILDER_CACHE.pop(next(iter(_BUILDER_CACHE)), None)
        _BUILDER_CACHE[key] = (time.monotonic(), json.loads(json.dumps(structured)))
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
        user_payload = {
            "question": question,
            "defaults": {
                "rows": default_rows,
                "sort": "score desc",
                "filters": filters or ["property:refereed"],
            },
        }
        if sort:
            # Only a preference: follow it when the request itself is about
            # recent / most-cited work (prompt rule 11).
            user_payload["caller_sort_preference"] = sort
        prompt = "Respond with ONE JSON object only. Natural-language request:\n" + json.dumps(user_payload)
        timeout = ads_query_builder_timeout()
        box: Dict[str, Any] = {}
        # The request-local LLM context (BYOK keys, quota admission, usage
        # recording) is thread-local: carry it into the worker (guard CX-01).
        try:
            from core.llm_client import get_llm_request_context, reinstall_llm_request_context
            request_ctx = get_llm_request_context()
        except Exception:  # pragma: no cover - core not importable
            request_ctx, reinstall_llm_request_context = None, None
        slots = _BUILDER_SLOTS  # release THIS object even if the global is swapped (CX-22)
        if not slots.acquire(blocking=False):
            raise ADSQueryBuilderError("query builder busy (2 calls in flight)")

        def _run() -> None:
            try:
                if reinstall_llm_request_context is not None:
                    with reinstall_llm_request_context(request_ctx):
                        box["text"] = self._llm_call(self._SYSTEM_PROMPT, prompt, self.model, 1200)
                else:
                    box["text"] = self._llm_call(self._SYSTEM_PROMPT, prompt, self.model, 1200)
            except BaseException as exc:  # pragma: no cover - re-raised below
                box["error"] = exc
            finally:
                slots.release()

        worker = threading.Thread(target=_run, name="ads-query-builder", daemon=True)
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            raise ADSQueryBuilderError(f"query builder timed out after {timeout:.0f}s ({self.model})")
        if "error" in box:
            raise ADSQueryBuilderError(f"{type(box['error']).__name__}: {str(box['error'])[:200]}")
        return str(box.get("text") or "")

    def _parse_response(self, content: str) -> Dict[str, Any]:
        if not content or not content.strip():
            raise ADSQueryBuilderError("Empty response from model")

        text = content.strip()
        if "<|" in text:
            try:
                from core.harmony_filter import strip_harmony_markup

                text = strip_harmony_markup(text)
            except Exception:  # pragma: no cover - optional dependency
                pass
        data = _first_json_object(text)
        if data is None:
            raise ADSQueryBuilderError(f"Could not parse model JSON: {text[:120]!r}")

        # Normalise filters to list
        filters = data.get("filters")
        if isinstance(filters, str):
            data["filters"] = [filters]

        return data
