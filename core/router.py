"""core/router.py — deterministic query-routing helpers extracted verbatim from
QuasarAgent (masterplan S12). Each function takes the agent instance so it can read the
class-level regex/cutoff constants that remain on QuasarAgent. Byte-parity: bodies are the
former method bodies, with attribute accesses rewritten onto the agent parameter."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from core.agent import QuasarAgent


_ARCHIVE_INTENT_RE = re.compile(r"\b(?:alma|archive|observations?|projects?|mous|datasets?)\b")
_BAND_LIST_RE = re.compile(
    r"\bbands?\s+(\d+(?:\s*(?:,|or|and|/|-|\u2013|\u2014|to|through|thru)\s*(?:band\s*)?\d+)*)"
)
_BAND_TOKEN_RE = re.compile(r"(\d+)|(-|\u2013|\u2014|\bto\b|\bthrough\b|\bthru\b)")
_RESOLUTION_RE = re.compile(
    r"(?:under|below|less than|better than|finer than|<)\s*(\d+(?:\.\d+)?)\s*(arcsec(?:ond)?s?|mas|[\"\u2033])"
)
# Negated / archive-name phrasings must not become constraints.
_PUBLIC_RE = re.compile(r"(?<!non-)(?<!non )(?<!not )\bpublic\b")
_SCIENCE_RE = re.compile(r"\bscience\W{1,4}(?:observations?|data|targets?|scans?|only|products?)\b")  # tolerates markdown
# Species commonly requested by name in ALMA line searches (case-sensitive, matched on the raw prompt).
_SPECIES_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:1[23]C(?:1[78])?O|C1[78]O|CO|HCO\+|H13CO\+|DCO\+|HCN|H13CN|DCN|HNC|HN13C|CS|13CS|C34S|SiO|"
    r"N2H\+|N2D\+|CN|SO2|SO|H2O|NH3|NH2D|CH3OH|CH3CN|HC3N|H2CO|OCS|C2H|CCH|CH3CCH|HCOOCH3|CH3OCH3|CH\+|OH|HD)"
    r"(?![A-Za-z0-9])"
)


def _bands_from_prompt(q: str) -> List[int]:
    """Band numbers named in the prompt; inclusive ranges ('6-7', '6 to 7') expanded."""
    bands: List[int] = []
    for match in _BAND_LIST_RE.finditer(q):
        prev: Optional[int] = None
        pending_range = False
        for tok in _BAND_TOKEN_RE.finditer(match[1]):
            if tok.group(1):
                n = int(tok.group(1))
                if pending_range and prev is not None and prev < n <= prev + 9:
                    bands.extend(range(prev + 1, n + 1))
                else:
                    bands.append(n)
                prev, pending_range = n, False
            else:
                pending_range = True
    return list(dict.fromkeys(b for b in bands if 1 <= b <= 10))


def route_alma_science_archive_query(agent: "QuasarAgent", query: str) -> Optional[Dict[str, Any]]:
    """Extract explicit archive constraints without target-specific defaults.

    Every branch is gated on archive intent, values come only from the prompt,
    and the target is deliberately NOT extracted (the tool directive tells the
    model to supply the user's target or coordinates) — a regex guess at a
    multi-token identifier is worse than no guess.
    """
    raw = str(query or "")
    q = raw.lower()
    if not _ARCHIVE_INTENT_RE.search(q):
        return None
    cycle_match = re.search(r"\bcycle\s+(\d{1,2})\b", q)
    cycle = int(cycle_match.group(1)) if cycle_match else None

    if cycle is not None and re.search(r"\b(?:sun|solar)\b", q):
        return {"query_type": "cycle_solar_projects", "cycle": cycle}

    if cycle is not None and all(term in q for term in ("12m", "7m")) and re.search(r"total\s+power|\btp\b", q):
        return {"query_type": "cycle_array_combo_projects", "cycle": cycle, "arrays": ["12m", "7m", "TP"]}

    bands = _bands_from_prompt(q)
    resolution = _RESOLUTION_RE.search(q)
    if "alma" in q and bands and (resolution or re.search(r"high[-\s]?resolution", q)):
        args: Dict[str, Any] = {"query_type": "high_resolution_band_data", "band": bands}
        if resolution:
            args["max_resolution_arcsec"] = float(resolution[1]) / (1000 if resolution[2] == "mas" else 1)
        if _PUBLIC_RE.search(q):
            args["public_only"] = True
        if _SCIENCE_RE.search(q):
            args["science_only"] = True
        return args

    species = list(dict.fromkeys(_SPECIES_RE.findall(raw)))
    if "alma" in q and len(species) > 1 and re.search(r"\blines?\b|molecular|isotop|transitions?", q):
        args = {"query_type": "line_set_projects", "lines": species}
        if bands:
            args["band"] = bands
        return args

    z_match = re.search(r"\bz\s*[=~]?\s*(\d+(?:\.\d+)?)\s*(?:-|to|\u2013)\s*(\d+(?:\.\d+)?)", q)
    if z_match and re.search(r"\bco\b|carbon monoxide|rest frequenc", q):
        return {
            "query_type": "redshifted_line_projects",
            "redshift_min": float(z_match.group(1)),
            "redshift_max": float(z_match.group(2)),
            "rest_species": "CO",
        }

    if re.search(r"bandwidth\s+switching|spectral\s+setup", q):
        args = {"query_type": "bandwidth_switching_candidates"}
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
