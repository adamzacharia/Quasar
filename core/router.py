"""core/router.py — deterministic query-routing helpers extracted verbatim from
QuasarAgent (masterplan S12). Each function takes the agent instance so it can read the
class-level regex/cutoff constants that remain on QuasarAgent. Byte-parity: bodies are the
former method bodies, with attribute accesses rewritten onto the agent parameter."""
from __future__ import annotations

import os
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
    r"N2H\+|N2D\+|CN|SO2|SO|H2O|NH3|NH2D|CH3OH|CH3CN|HC3N|H2CO|OCS|C2H|CCH|CH3CCH|HCOOCH3|CH3OCH3|CH\+|OH|HD|"
    r"\[?C\s?II\]?|\[?N\s?II\]?|\[?O\s?III\]?|\[?O\s?I\]?|\[?C\s?I\]?)"  # fine-structure lines of high-z work
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

    # Precedence (UI benchmark 2026-09-22, D22): the MOST specific signal wins.
    # A redshift interval plus a rest-frame species is a redshifted-line query
    # even when the prompt also says "spectral setup"; a list of >= 2 species
    # is a line-set query; only the literal phrase "bandwidth switching" (or
    # "spectral setup" with no species / redshift) is a BWSW candidate query.
    z_match = re.search(
        r"(?:\bz\s*[=~]?\s*|redshifts?\s+(?:of\s+|between\s+|from\s+)?(?:z\s*[=~]?\s*)?)"
        r"(\d+(?:\.\d+)?)\s*(?:-|to|and|\u2013)\s*(?:z\s*[=~]?\s*)?(\d+(?:\.\d+)?)",
        q,
    )
    species = list(dict.fromkeys(_SPECIES_RE.findall(raw)))
    mentions_co = bool(re.search(r"\bco\b|carbon\s+monoxide", q))
    if z_match and (mentions_co or species or re.search(r"rest\s+frequenc", q)):
        rest = species[0] if species else "CO"
        if mentions_co and "CO" in species:
            rest = "CO"
        return {
            "query_type": "redshifted_line_projects",
            "redshift_min": float(z_match.group(1)),
            "redshift_max": float(z_match.group(2)),
            "rest_species": rest,
        }

    if cycle is not None and re.search(r"\b(?:sun|solar)\b", q):
        return {"query_type": "cycle_solar_projects", "cycle": cycle}

    if cycle is not None and all(term in q for term in ("12m", "7m")) and re.search(r"total\s+power|\btp\b", q):
        return {"query_type": "cycle_array_combo_projects", "cycle": cycle, "arrays": ["12m", "7m", "TP"]}

    bands = _bands_from_prompt(q)
    if "alma" in q and len(species) > 1 and re.search(r"\blines?\b|molecular|isotop|transitions?|same\s+project|observed", q):
        args: Dict[str, Any] = {"query_type": "line_set_projects", "lines": species}
        if bands:
            args["band"] = bands
        return args

    resolution = _RESOLUTION_RE.search(q)
    # "Band 7 ... better than 1 arcsec" is an ALMA request even without the
    # word ALMA (D12: HH212 Band 7 continuum) -- numbered bands with arcsec
    # resolution and archive intent are unambiguous.
    if bands and (resolution or re.search(r"high[-\s]?resolution", q)) and ("alma" in q or resolution):
        args = {"query_type": "high_resolution_band_data", "band": bands}
        if resolution:
            args["max_resolution_arcsec"] = float(resolution[1]) / (1000 if resolution[2] == "mas" else 1)
        if _PUBLIC_RE.search(q):
            args["public_only"] = True
        if _SCIENCE_RE.search(q):
            args["science_only"] = True
        return args

    if re.search(r"bandwidth\s+switching", q) or (re.search(r"spectral\s+setup", q) and not species and not z_match):
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


_DEFAULT_LLM_CUTOFF = (2025, 6)

# Policy / deadline phrases that need the CURRENT published rules. They win
# over the live-data suppression: "Cycle 13 proprietary period for archive
# data" was suppressed by the bare words "data"/"archive" before the policy
# keywords were ever consulted (D6). A bare "Cycle 13" is NOT here: "find ALMA
# Cycle 12 observations of M87" is an archive query.
_POLICY_OVERRIDE_RE = re.compile(
    r"\b(?:proprietary\s+(?:period|time)s?|deadlines?|polic(?:y|ies)|guidelines?|regulations?|"
    r"call[-\s]for[-\s]proposals?|proposers?['’]?s?\s+guide|"
    r"proposal\s+(?:rules?|requirements?|limits?|eligibility)|eligibility\s+rules?|"
    # Phase 2 (WebBench POL-05 / POL-06): access-period and schedule questions are
    # policy questions too; "data" in them used to trip the live-data suppression
    r"exclusive[-\s]access(?:\s+periods?)?|data\s+rights|access\s+periods?|embargo(?:\s+periods?)?|"
    r"large\s+programs?\s+threshold|"
    r"cycle\s*\d{1,2}\s+(?:science\s+)?(?:observ\w*\s+)?(?:start|starts|begin\w*|end|ends|schedule|dates?|timeline)|"
    r"(?:configuration|array)\s+(?:schedule|plans?)|(?:observing|semester)\s+schedule)\b",
    re.IGNORECASE,
)


def policy_web_override(query: str) -> bool:
    """True when the query asks about a policy, deadline or proposal rule that
    changes over time: web search runs even if the query also mentions data."""
    return bool(_POLICY_OVERRIDE_RE.search(str(query or "")))


def llm_knowledge_cutoff() -> tuple:
    """(year, month) of the chat models' training cutoff, from QUASAR_LLM_CUTOFF
    ("YYYY-MM" or "YYYY"); default 2025-06. Read per call so an env change
    needs no restart. A malformed value falls back to the default."""
    raw = (os.getenv("QUASAR_LLM_CUTOFF") or "").strip()
    m = re.fullmatch(r"((?:19|20)\d\d)(?:-(\d{1,2}))?", raw)
    if not m:
        return _DEFAULT_LLM_CUTOFF
    year = int(m.group(1))
    month = int(m.group(2)) if m.group(2) else 12
    if not 1 <= month <= 12:
        return _DEFAULT_LLM_CUTOFF
    return (year, month)


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

    cutoff_year, cutoff_month = llm_knowledge_cutoff()

    # 1. Explicit year mentions beyond cutoff
    year_matches = re.findall(r'\b(20[2-9]\d)\b', query)
    for ym in year_matches:
        y = int(ym)
        if y > cutoff_year:
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
        if m_year > cutoff_year:
            return query
        if m_year == cutoff_year and month_names[m_name] > cutoff_month:
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
