"""Extract and verify ADS bibcodes and DOI citations in generated answers."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Dict, List, MutableMapping, Optional
from urllib.parse import unquote

logger = logging.getLogger(__name__)

BIBCODE_RE = re.compile(r"\b\d{4}[A-Za-z.&]{5}[\w.&]{9}[A-Z]\b")
# Unicode dash variants (U+2010..U+2014, U+2212) are included because LLM output
# typography routinely swaps them for ASCII hyphens — without them a DOI like
# 10.1051/0004‑6361/202243940 truncates to '10.1051/0004' (live P6) and the
# verifier then flags a mangled identifier instead of the real one.
_DASH_VARIANTS = "‐‑‒–—−"
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()+/:A-Za-z0-9" + _DASH_VARIANTS + r"]+")


def extract_citations(text: str) -> Dict[str, List[str]]:
    """Return order-preserving unique ADS bibcodes and DOIs found in text."""

    bibcodes: List[str] = []
    seen_bibcodes = set()
    dois: List[str] = []
    seen_dois = set()

    for variant in _text_variants(text or ""):
        for match in BIBCODE_RE.finditer(variant):
            bibcode = match.group(0)
            if len(bibcode) != 19:
                continue
            if bibcode not in seen_bibcodes:
                seen_bibcodes.add(bibcode)
                bibcodes.append(bibcode)

        for match in DOI_RE.finditer(variant):
            doi = _normalize_doi(match.group(0))
            if doi and doi not in seen_dois:
                seen_dois.add(doi)
                dois.append(doi)

    return {"bibcodes": bibcodes, "dois": dois}


def verify_citations(
    text: str,
    ads_client: Any,
    *,
    max_lookups: int = 25,
    cache: Optional[MutableMapping[str, Dict[str, Any]]] = None,
    deadline_seconds: Optional[float] = 8.0,
) -> Dict[str, Any]:
    """Resolve extracted citations against ADS without raising to callers."""

    citations = extract_citations(text)
    result: Dict[str, Any] = {
        "bibcodes": [],
        "dois": [],
        "unresolved": [],
        "checked": 0,
        "skipped": 0,
        "skipped_items": [],
    }

    unavailable_reason = _ads_unavailable_reason(ads_client)
    if unavailable_reason:
        for bibcode in citations["bibcodes"]:
            result["bibcodes"].append(_entry("bibcode", bibcode, None, unavailable_reason))
        for doi in citations["dois"]:
            result["dois"].append(_entry("doi", doi, None, unavailable_reason))
        return result

    lookup_limit = max(0, int(max_lookups))
    deadline_at = (
        time.monotonic() + deadline_seconds
        if deadline_seconds is not None and deadline_seconds >= 0
        else None
    )

    for kind, identifier in _citation_items(citations):
        cached = _cache_get(cache, kind, identifier)
        if cached is not None:
            item = dict(cached)
        else:
            skip_reason = _skip_reason(result["checked"], lookup_limit, deadline_at)
            if skip_reason:
                item = _entry(kind, identifier, None, skip_reason)
                result["skipped"] += 1
                result["skipped_items"].append(item)
                logger.info("Skipping citation verification for %s %s: %s", kind, identifier, skip_reason)
            else:
                item = (
                    _resolve_bibcode(identifier, ads_client)
                    if kind == "bibcode"
                    else _resolve_doi(identifier, ads_client)
                )
                result["checked"] += 1
                _cache_set(cache, kind, identifier, item)

        bucket = "bibcodes" if kind == "bibcode" else "dois"
        result[bucket].append(item)
        if item.get("resolved") is False:
            result["unresolved"].append(item)

    return result


def format_unverified_citation_warning(verification: Dict[str, Any]) -> str:
    """Build the user-visible warning block for unresolved citations."""

    unresolved = list(verification.get("unresolved") or [])
    if not unresolved:
        return ""

    labels = []
    for item in unresolved[:10]:
        label = "ADS bibcode" if item.get("type") == "bibcode" else "DOI"
        labels.append(f"{label} `{item.get('id', '')}`")
    if len(unresolved) > 10:
        labels.append(f"{len(unresolved) - 10} more")

    joined = ", ".join(labels)
    return (
        "\n\n> **Unverified citations:** the following references could not "
        f"be found in ADS and may be inaccurate: {joined}."
    )


def append_citation_warning(
    text: str,
    ads_client: Any,
    *,
    on_token: Optional[Callable[[str], None]] = None,
    max_lookups: int = 25,
    cache: Optional[MutableMapping[str, Dict[str, Any]]] = None,
    deadline_seconds: Optional[float] = 8.0,
) -> str:
    """Append and optionally stream the citation warning; never raises."""

    if not text or "Unverified citations:" in text:
        return text

    try:
        verification = verify_citations(
            text,
            ads_client,
            max_lookups=max_lookups,
            cache=cache,
            deadline_seconds=deadline_seconds,
        )
        warning = format_unverified_citation_warning(verification)
    except Exception:
        logger.warning("Citation verification failed; continuing without warning", exc_info=True)
        return text

    if not warning:
        return text

    if on_token:
        try:
            on_token(warning)
        except Exception:
            logger.debug("Citation warning token callback failed", exc_info=True)
    return text + warning


def _text_variants(text: str) -> List[str]:
    decoded = unquote(text)
    return [text] if decoded == text else [text, decoded]


def _citation_items(citations: Dict[str, List[str]]):
    for bibcode in citations.get("bibcodes", []):
        yield "bibcode", bibcode
    for doi in citations.get("dois", []):
        yield "doi", doi


def _ads_unavailable_reason(ads_client: Any) -> Optional[str]:
    if ads_client is None:
        return "no_ads_client"
    if hasattr(ads_client, "api_key") and not getattr(ads_client, "api_key", ""):
        return "no_ads_key"
    return None


def _skip_reason(checked: int, max_lookups: int, deadline_at: Optional[float]) -> Optional[str]:
    if checked >= max_lookups:
        return "skipped_max_lookups"
    if deadline_at is not None and time.monotonic() >= deadline_at:
        return "skipped_time_budget"
    return None


def _entry(
    kind: str,
    identifier: str,
    resolved: Optional[bool],
    reason: str,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "type": kind,
        "id": identifier,
        "resolved": resolved,
        "reason": reason,
    }
    if title:
        item["title"] = title
    return item


def _resolve_bibcode(bibcode: str, ads_client: Any) -> Dict[str, Any]:
    if hasattr(ads_client, "_perform_get"):
        return _resolve_ads_query(
            ads_client,
            f'bibcode:"{_escape_ads_phrase(bibcode)}"',
            "bibcode",
            bibcode,
        )

    get_details = getattr(ads_client, "get_paper_details", None)
    if not callable(get_details):
        return _entry("bibcode", bibcode, None, "missing_bibcode_lookup")

    try:
        details = get_details(bibcode)
    except Exception:
        logger.info("ADS bibcode lookup failed for %s", bibcode, exc_info=True)
        return _entry("bibcode", bibcode, None, "ads_error")

    if details:
        return _entry("bibcode", bibcode, True, "found", _paper_title(details))
    return _entry("bibcode", bibcode, False, "not_found")


def _resolve_doi(doi: str, ads_client: Any) -> Dict[str, Any]:
    if hasattr(ads_client, "_perform_get"):
        return _resolve_ads_query(
            ads_client,
            f'doi:"{_escape_ads_phrase(doi)}"',
            "doi",
            doi,
        )

    search = getattr(ads_client, "search_papers", None)
    if not callable(search):
        return _entry("doi", doi, None, "missing_doi_lookup")

    try:
        papers = search(f'doi:"{_escape_ads_phrase(doi)}"', max_results=1, sort="score desc")
    except Exception:
        logger.info("ADS DOI lookup failed for %s", doi, exc_info=True)
        return _entry("doi", doi, None, "ads_error")

    papers = papers or []
    exact = _find_exact_doi_paper(papers, doi)
    if exact:
        return _entry("doi", doi, True, "found", _paper_title(exact))
    if papers and not any(_paper_dois(paper) for paper in papers):
        return _entry("doi", doi, True, "found", _paper_title(papers[0]))
    return _entry("doi", doi, False, "not_found")


def _resolve_ads_query(
    ads_client: Any,
    query: str,
    kind: str,
    identifier: str,
) -> Dict[str, Any]:
    params = {"q": query, "fl": "title,bibcode,doi", "rows": 1}
    # Bound each verification lookup: no retries/backoff and a short timeout, so
    # a slow or rate-limited ADS cannot stall response completion (the verifier
    # runs post-stream and only ever appends an advisory warning). Fall back to
    # the default call if a client doesn't accept the override kwargs.
    try:
        try:
            data = ads_client._perform_get(
                "/search/query", params, timeout=4.0, retry_attempts=1
            )
        except TypeError:
            data = ads_client._perform_get("/search/query", params)
    except Exception:
        logger.info("ADS %s lookup failed for %s", kind, identifier, exc_info=True)
        return _entry(kind, identifier, None, "ads_error")

    docs = data.get("response", {}).get("docs", []) if isinstance(data, dict) else []
    if not docs:
        return _entry(kind, identifier, False, "not_found")

    paper = docs[0]
    if kind == "bibcode" and paper.get("bibcode") != identifier:
        return _entry(kind, identifier, False, "not_found")
    if kind == "doi" and _normalize_doi(identifier) not in _paper_dois(paper):
        return _entry(kind, identifier, False, "not_found")
    return _entry(kind, identifier, True, "found", _paper_title(paper))


def _find_exact_doi_paper(papers: List[Dict[str, Any]], doi: str) -> Optional[Dict[str, Any]]:
    normalized = _normalize_doi(doi)
    for paper in papers:
        if normalized in _paper_dois(paper):
            return paper
    return None


def _paper_dois(paper: Dict[str, Any]) -> List[str]:
    raw = paper.get("doi")
    if raw is None:
        return []
    values = raw if isinstance(raw, list) else [raw]
    return [_normalize_doi(str(value)) for value in values if _normalize_doi(str(value))]


def _paper_title(paper: Dict[str, Any]) -> Optional[str]:
    title = paper.get("title") if isinstance(paper, dict) else None
    if isinstance(title, list):
        return str(title[0]) if title else None
    return str(title) if title else None


def _normalize_doi(value: str) -> str:
    doi = str(value or "").strip()
    # Unicode dash typography → ASCII hyphen, so a real DOI typed with U+2011
    # etc. still resolves against ADS instead of being flagged unverified.
    doi = doi.translate({ord(ch): "-" for ch in _DASH_VARIANTS})
    doi = doi.rstrip(".,;")
    while doi.endswith(")") and doi.count(")") > doi.count("("):
        doi = doi[:-1].rstrip(".,;")
    return doi.lower()


def _escape_ads_phrase(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', r"\"")


def _cache_key(kind: str, identifier: str) -> str:
    return f"citation:{kind}:{identifier.lower()}"


def _cache_get(
    cache: Optional[MutableMapping[str, Dict[str, Any]]],
    kind: str,
    identifier: str,
) -> Optional[Dict[str, Any]]:
    if cache is None:
        return None
    cached = cache.get(_cache_key(kind, identifier))
    return dict(cached) if isinstance(cached, dict) else None


def _cache_set(
    cache: Optional[MutableMapping[str, Dict[str, Any]]],
    kind: str,
    identifier: str,
    item: Dict[str, Any],
) -> None:
    if cache is not None:
        cache[_cache_key(kind, identifier)] = dict(item)
