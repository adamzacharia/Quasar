"""core/router.py — deterministic query-routing helpers extracted verbatim from
QuasarAgent (masterplan S12). Each function takes the agent instance so it can read the
class-level regex/cutoff constants that remain on QuasarAgent. Byte-parity: bodies are the
former method bodies, with attribute accesses rewritten onto the agent parameter."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from core.agent import QuasarAgent


def route_alma_science_archive_query(agent: "QuasarAgent", query: str) -> Optional[Dict[str, Any]]:
    """Map known hard ALMA science prompts to deterministic tool args."""
    raw = str(query or "")
    q = raw.lower()
    cycle_match = re.search(r"\bcycle\s+(\d{1,2})\b", q)
    cycle = int(cycle_match.group(1)) if cycle_match else None

    if cycle is not None and re.search(r"\b(?:sun|solar)\b", q):
        return {"query_type": "cycle_solar_projects", "cycle": cycle}

    if cycle is not None and all(term in q for term in ("12m", "7m")) and re.search(r"total\s+power|\btp\b", q):
        return {"query_type": "cycle_array_combo_projects", "cycle": cycle, "arrays": ["12m", "7m", "TP"]}

    if re.search(r"\bhh\s*212\b", q) and re.search(r"band\s*7|\bb7\b", q):
        return {
            "query_type": "high_resolution_band_data",
            "target": "HH 212",
            "band": 7,
            "max_resolution_arcsec": 0.1,
        }

    if all(term in q for term in ("12co", "13co", "c18o")):
        args: Dict[str, Any] = {
            "query_type": "line_set_projects",
            "band": 6,
            "lines": ["12CO", "13CO", "C18O"],
            "require_same_project": True,
        }
        if re.search(r"protostellar|proto-stellar|disk", q):
            args["topic_filter"] = "protostellar disks"
        return args

    z_match = re.search(r"\bz\s*[=~]?\s*(\d+(?:\.\d+)?)\s*(?:-|to|–)\s*(\d+(?:\.\d+)?)", q)
    if z_match and re.search(r"\bco\b|carbon monoxide|rest frequenc", q):
        return {
            "query_type": "redshifted_line_projects",
            "redshift_min": float(z_match.group(1)),
            "redshift_max": float(z_match.group(2)),
            "rest_species": "CO",
            "science_category": "Galaxy",
            "require_same_project": True,
        }

    if re.search(r"bandwidth\s+switching|spectral\s+setup", q):
        args: Dict[str, Any] = {"query_type": "bandwidth_switching_candidates"}
        if cycle is not None:
            args["cycle"] = cycle
        return args

    return None


def route_cross_archive_source_match_query(agent: "QuasarAgent", query: str) -> Optional[Dict[str, Any]]:
    """Map custom coordinate cross-match prompts to deterministic tool args."""
    raw = str(query or "")
    q = raw.lower()
    archive_aliases = {
        "alma": "ALMA",
        "jwst": "JWST",
        "hst": "HST",
        "mast": "MAST",
        "tess": "TESS",
        "kepler": "KEPLER",
        "k2": "K2",
        "galex": "GALEX",
        "swift": "SWIFT",
    }
    archives: List[str] = []
    for needle, label in archive_aliases.items():
        if re.search(rf"\b{re.escape(needle)}\b", q) and label not in archives:
            archives.append(label)
    if not archives:
        return None

    has_match_intent = bool(re.search(r"\bcross[- ]?match|crossmatch|match\b", q))
    has_coverage_intent = bool(re.search(r"\bboth\b|all requested|every archive|in all\b|which\b|coverage", q))
    has_catalog_signal = bool(re.search(r"\bperseus\b|\bprotostar|\bsource\b|\bRA\s*\d", raw, flags=re.IGNORECASE))
    if not (has_match_intent or (has_coverage_intent and has_catalog_signal)):
        return None

    radius_arcsec = 5.0
    radius_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:arcsec|arcsecond|arcseconds|as|['\"]{2})\b", q)
    if radius_match:
        radius_arcsec = float(radius_match.group(1))

    sources: List[Dict[str, Any]] = []
    source_re = re.compile(
        r"(?P<name>[A-Za-z0-9_.+/-][A-Za-z0-9_.+/-]*(?:\s+[A-Za-z0-9_.+/-]+){0,4})"
        r"\s+(?:at\s+)?RA\s*[=:]?\s*(?P<ra>\d+(?:\.\d+)?)"
        r"\s*(?:,|\s)+Dec\s*[=:]?\s*(?P<dec>[+-]?\d+(?:\.\d+)?)",
        flags=re.IGNORECASE,
    )
    for match in source_re.finditer(raw):
        name = re.sub(
            r"^(?:and|with|sources|catalog|target)\s+",
            "",
            match.group("name").strip(" ,.;:"),
            flags=re.IGNORECASE,
        ).strip()
        name = re.sub(r"\s+at$", "", name, flags=re.IGNORECASE).strip() or f"source_{len(sources) + 1}"
        try:
            ra = float(match.group("ra"))
            dec = float(match.group("dec"))
        except (TypeError, ValueError):
            continue
        sources.append({"source_name": name, "ra": ra, "dec": dec})

    args: Dict[str, Any] = {
        "archives": archives,
        "radius_arcsec": radius_arcsec,
        "require_all_archives": bool(re.search(r"\bboth\b|all requested|every archive|in all\b", q)),
    }
    if sources:
        args["catalog_name"] = "inline"
        args["sources"] = sources
        args["max_sources"] = len(sources)
    elif "perseus" in q and "protostar" in q:
        args["catalog_name"] = "perseus_protostars"
        args["require_all_archives"] = True
    else:
        return None
    return args


def is_live_data_query(agent: "QuasarAgent", query: str) -> bool:
    """True when the query is answered by QUASAR's built-in live-data
    tools (archives, papers, alerts, catalogs, imagery, spectra,
    photometry, cone searches) — web search adds nothing for these."""
    _q = query.lower()
    return bool(
        agent._LIVE_DATA_KEYWORDS_RE.search(_q)
        or agent._LIVE_DATA_VERBS_RE.search(_q)
        or agent._LIVE_DATA_FACILITY_RE.search(_q)
        or agent._LIVE_DATA_CONE_RE.search(_q)
        or agent._PAPER_QUERY_RE.search(_q)
    )


def detect_beyond_cutoff(agent: "QuasarAgent", query: str) -> Optional[str]:
    """
    Check if a query references dates or time periods beyond the LLM's
    training data cutoff.  Returns a web-search query string if detected,
    else None.

    Triggers on:
    - Explicit future years:  "as of March 2025", "in 2026"
    - Freshness keywords:    "latest", "current", "recent", "now",
                              "today", "this year", "this month"

    Does NOT trigger on archive search queries — those hit live databases
    (ALMA, CADC, etc.) directly and don't need web search augmentation.
    """
    _q = query.lower()

    # ── Skip web search for live-data / paper queries ───────────────
    # These are served by dedicated live databases (ALMA archive, ADS,
    # ALeRCE alerts, Data Lab, SparCL, NED, HiPS imagery, ...) — web
    # search adds nothing and just wastes time / clutters the response.
    if agent._is_live_data_query(_q):
        return None

    # 1. Explicit year mentions beyond cutoff
    year_matches = re.findall(r'\b(20[2-9]\d)\b', query)
    for ym in year_matches:
        y = int(ym)
        if y > agent._LLM_CUTOFF_YEAR:
            return query  # whole query is the search string

    # 2. Month+Year combos in cutoff year but after cutoff month
    month_year = re.findall(
        r'(?:january|february|march|april|may|june|july|august|'
        r'september|october|november|december)\s+(20[2-9]\d)',
        _q,
    )
    month_names = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
    }
    for match in re.finditer(
        r'(january|february|march|april|may|june|july|august|'
        r'september|october|november|december)\s+(20[2-9]\d)',
        _q,
    ):
        m_name, m_year = match.group(1), int(match.group(2))
        if m_year > agent._LLM_CUTOFF_YEAR:
            return query
        if m_year == agent._LLM_CUTOFF_YEAR and month_names[m_name] > agent._LLM_CUTOFF_MONTH:
            return query

    # 3. Selective freshness keywords — only high-confidence temporal phrases
    #    that strongly imply the user wants current-year information.
    #    Avoids broad terms like "latest", "current", "recent" which cause
    #    false positives on nearly every query.
    _freshness_pattern = re.search(
        r'\b(?:this\s+year|this\s+month|today|right\s+now|happening\s+now)\b',
        _q,
    )
    if _freshness_pattern:
        return query

    return None
