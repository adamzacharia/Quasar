"""Domain packs for web retrieval and the deterministic evidence ordering.

Web search redesign, Phase 2 (PLAN 2.2, 2.3 and the light part of 3.2).

A *domain pack* names the official sites for a family of questions. The
planner (core/web_planner.py) picks one per turn; the pack's queries run with
Tavily ``include_domains`` (plus ONE unrestricted query for recall), and the
evidence block is ordered by authority before relevance so the official page
is what the model cites first.

Everything here is DATA plus a few pure functions: no network, no model call.
The pack list is reviewed by the domain expert (the NRAO astronomer who owns
this repo); edit ``DOMAIN_PACKS`` and the unit tests keep the contract.

Ordering rule (``order_evidence``), deterministic and unit-tested:
  1. official pack domains first (``official_rank`` 0), then primary
     observatory / archive / agency tiers, then institutional or peer
     reviewed, then the general web;
  2. within a rank, a page about the CURRENT document wins: for a question
     about Cycle 13 a page whose title / URL / excerpt names only another
     cycle (Cycle 7) is demoted below pages that name Cycle 13 or no cycle;
  3. then relevance (provider rank), then arrival order.
Full BM25 rerank stays in Phase 3.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

# ── the packs (data) ───────────────────────────────────────────────────────

DOMAIN_PACKS: Dict[str, Dict[str, Any]] = {
    "alma_policy": {
        "label": "ALMA policies, cycles and proposer documentation",
        "include_domains": [
            "almascience.org",
            "almascience.eso.org",
            "almascience.nrao.edu",
            "almascience.nao.ac.jp",
            "almaobservatory.org",
            "alma-telescope.jp",
            "eso.org",
        ],
        "topic": "general",
    },
    "nrao": {
        "label": "NRAO facilities: VLA, VLBA, GBT, ngVLA",
        "include_domains": [
            "science.nrao.edu",
            "public.nrao.edu",
            "nrao.edu",
            "greenbankobservatory.org",
            "ngvla.nrao.edu",
        ],
        "topic": "general",
    },
    "stsci": {
        "label": "STScI: JWST, HST, Roman documentation and calls",
        "include_domains": [
            "stsci.edu",
            "jwst-docs.stsci.edu",
            "hst-docs.stsci.edu",
            "roman-docs.stsci.edu",
            "archive.stsci.edu",
            "science.nasa.gov",
        ],
        "topic": "general",
    },
    "noirlab": {
        "label": "NOIRLab: Rubin / LSST, Gemini, Data Lab, CTIO, KPNO",
        "include_domains": [
            "noirlab.edu",
            "datalab.noirlab.edu",
            "rubinobservatory.org",
            "lsst.org",
            "lsst.io",
            "gemini.edu",
        ],
        "topic": "general",
    },
    "esa": {
        "label": "ESA science missions and cosmos portal",
        "include_domains": [
            "esa.int",
            "cosmos.esa.int",
            "sci.esa.int",
            "esac.esa.int",
        ],
        "topic": "general",
    },
    "transients": {
        "label": "Transient reports: TNS, ATel, GCN, agency news",
        "include_domains": [
            "wis-tns.org",
            "astronomerstelegram.org",
            "gcn.nasa.gov",
            "science.nasa.gov",
            "esa.int",
            "eso.org",
        ],
        "topic": "news",
    },
    "researcher": {
        "label": "Researcher profiles: ORCID, institutional pages",
        # No restriction: university domains cannot be enumerated. The
        # unrestricted query carries recall; .edu / .ac.* pages get an
        # authority boost in the ordering.
        "include_domains": [],
        "boost_suffixes": [".edu", ".ac.uk", ".ac.jp", ".ac.at", ".ac.il", ".ac.za", ".ac.nz", ".ac.in", ".edu.au"],
        "official_domains": ["orcid.org"],
        "topic": "general",
    },
    "general": {
        "label": "No restriction",
        "include_domains": [],
        "topic": "general",
    },
}

DEFAULT_PACK = "general"

FRESHNESS_VALUES = ("day", "week", "month", "year", "any")

# Tavily accepts day|week|month|year; "any" means no time_range.
_BRAVE_FRESHNESS = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}


def known_packs() -> List[str]:
    return list(DOMAIN_PACKS)


def normalize_pack(name: Any) -> str:
    key = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    return key if key in DOMAIN_PACKS else DEFAULT_PACK


def pack_domains(name: Any) -> List[str]:
    """The ``include_domains`` list of a pack ("" / unknown / general -> [])."""
    return list(DOMAIN_PACKS.get(normalize_pack(name), {}).get("include_domains") or [])


def pack_topic(name: Any) -> str:
    return str(DOMAIN_PACKS.get(normalize_pack(name), {}).get("topic") or "general")


def normalize_freshness(value: Any) -> str:
    v = str(value or "").strip().lower()
    aliases = {"today": "day", "daily": "day", "recent": "week", "weekly": "week", "monthly": "month",
               "this_year": "year", "yearly": "year", "none": "any", "": "any", "all": "any"}
    v = aliases.get(v, v)
    return v if v in FRESHNESS_VALUES else "any"


def tavily_time_range(freshness: Any) -> Optional[str]:
    f = normalize_freshness(freshness)
    return None if f == "any" else f


def brave_freshness(freshness: Any) -> Optional[str]:
    return _BRAVE_FRESHNESS.get(normalize_freshness(freshness))


# ── deterministic pack inference (planner off / failed) ────────────────────

_PACK_HINTS: Tuple[Tuple[str, str], ...] = (
    (r"\balma\b|\batacama\b|\bproposer'?s?\s+guide\b", "alma_policy"),
    (r"\bvla\b|\bvlba\b|\bgbt\b|\bngvla\b|\bnrao\b|\bgreen\s*bank\b|\bjansky\b", "nrao"),
    (r"\bjwst\b|\bhst\b|\bhubble\b|\bwebb\b|\bstsci\b|\broman\s+space\s+telescope\b|\bmast\b", "stsci"),
    (r"\brubin\b|\blsst\b|\bnoirlab\b|\bgemini\b|\bdata\s*lab\b|\bdecam\b|\bctio\b|\bkpno\b|\bblanco\b", "noirlab"),
    (r"\besa\b|\beuclid\b|\bgaia\b|\bxmm\b|\bjuice\b|\bplato\b|\bathena\b|\bariel\b", "esa"),
    (r"\bgrb\s*\d|\bsn\s*20\d\d|\bat\s*20\d\d|\bsupernova\b|\bgamma[-\s]ray\s+burst\b|\btransient\b|\bkilonova\b|\bfast\s+radio\s+burst\b|\bfrb\b|\bcomet\b|\binterstellar\s+object\b|\btns\b|\batel\b|\bgcn\b", "transients"),
)


def infer_pack(query: str, *, person: bool = False) -> str:
    """Deterministic pack guess from the query text (used when the planner is
    off or failed). ``person`` = the researcher path is active."""
    if person:
        return "researcher"
    q = str(query or "").lower()
    for pattern, pack in _PACK_HINTS:
        if re.search(pattern, q):
            return pack
    return DEFAULT_PACK


# ── authority ranking ──────────────────────────────────────────────────────

_PRIMARY_TIERS = {"primary"}
_STRONG_TIERS = {"peer_reviewed", "reference", "institutional", "preprint"}


def _host(url_or_domain: str) -> str:
    s = str(url_or_domain or "").strip().lower()
    if "://" in s:
        s = urlsplit(s).hostname or ""
    s = s.split("/", 1)[0].rstrip(".")
    for prefix in ("www.", "m."):
        if s.startswith(prefix) and s.count(".") >= 2:
            s = s[len(prefix):]
    return s


def domain_in(host: str, domains: Iterable[str]) -> bool:
    h = _host(host)
    if not h:
        return False
    for d in domains:
        dd = _host(d)
        if dd and (h == dd or h.endswith("." + dd)):
            return True
    return False


def official_rank(url_or_domain: str, pack: Any, quality: Optional[Dict[str, Any]] = None) -> int:
    """0 = a domain of the chosen pack (the official page for this question),
    1 = primary observatory / archive / agency tier (evidence_quality),
    2 = institutional, peer reviewed, reference or preprint tier,
    3 = everything else."""
    spec = DOMAIN_PACKS.get(normalize_pack(pack), {})
    host = _host(url_or_domain)
    if domain_in(host, spec.get("include_domains") or []) or domain_in(host, spec.get("official_domains") or []):
        return 0
    if any(host.endswith(suffix) for suffix in (spec.get("boost_suffixes") or [])):
        return 1
    tier = str((quality or {}).get("tier") or "").lower()
    if tier in _PRIMARY_TIERS:
        return 1
    if tier in _STRONG_TIERS:
        return 2
    return 3


# ── current-document rule ──────────────────────────────────────────────────

_CYCLE_RE = re.compile(r"\bcycle[\s_-]*(\d{1,2})\b", re.I)
_DR_RE = re.compile(r"\b(?:dr|data\s+release)[\s_-]*(\d{1,2})\b", re.I)


def asked_versions(query: str) -> Dict[str, set]:
    """Version-like ids the question names: {"cycle": {13}, "dr": {2}}."""
    q = str(query or "")
    return {
        "cycle": {int(m) for m in _CYCLE_RE.findall(q)},
        "dr": {int(m) for m in _DR_RE.findall(q)},
    }


def _versions_in(text: str) -> Dict[str, set]:
    t = str(text or "")
    return {
        "cycle": {int(m) for m in _CYCLE_RE.findall(t)},
        "dr": {int(m) for m in _DR_RE.findall(t)},
    }


def currency_penalty(item_text: str, asked: Dict[str, set], *, strong_text: Optional[str] = None) -> int:
    """1 when the page is about ANOTHER version of a document the question
    pins (a Cycle 7 page for a Cycle 13 question), else 0.

    ``strong_text`` (URL + title) decides when it names a version: a
    "documents-and-tools/cycle-12" page whose excerpt also mentions "Cycle 13"
    in a sidebar is still a Cycle 12 page (live 2026-09-24, POL-01). Only when
    URL and title name no version does the excerpt count. A page naming the
    asked version, or no version at all, is not penalised."""
    strong = _versions_in(strong_text) if strong_text else {}
    weak = _versions_in(item_text)
    for kind, wanted in asked.items():
        if not wanted:
            continue
        got = strong.get(kind) or set()
        if not got:
            got = weak.get(kind) or set()
        if got and not (got & wanted):
            return 1
    return 0


def order_evidence(
    items: Sequence[Any],
    *,
    pack: Any = DEFAULT_PACK,
    query: str = "",
    entities: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """Return ``items`` (WebEvidence-like: url, domain, title, excerpt,
    quality, relevance, rank) sorted by authority, then currency, then
    relevance, then arrival. Stable and deterministic."""
    asked = asked_versions(query)
    cycle = (entities or {}).get("cycle") if isinstance(entities, dict) else None
    if cycle:
        try:
            asked["cycle"].add(int(str(cycle).strip()))
        except (TypeError, ValueError):
            pass

    def _key(indexed: Tuple[int, Any]):
        position, ev = indexed
        url = getattr(ev, "url", None) or (ev.get("url") if isinstance(ev, dict) else "") or ""
        quality = getattr(ev, "quality", None) if not isinstance(ev, dict) else ev.get("quality") or ev.get("evidenceQuality")
        title = getattr(ev, "title", None) if not isinstance(ev, dict) else ev.get("title")
        excerpt = getattr(ev, "excerpt", None) if not isinstance(ev, dict) else (ev.get("excerpt") or ev.get("snippet"))
        relevance = getattr(ev, "relevance", None) if not isinstance(ev, dict) else ev.get("relevance")
        try:
            rel = float(relevance or 0.0)
        except (TypeError, ValueError):
            rel = 0.0
        text = " ".join(str(x or "") for x in (title, url, excerpt))
        return (
            official_rank(url, pack, quality if isinstance(quality, dict) else None),
            currency_penalty(text, asked, strong_text=" ".join(str(x or "") for x in (title, url))),
            -rel,
            position,
        )

    return [ev for _, ev in sorted(enumerate(items), key=_key)]


def review_table() -> str:
    """Markdown table of the packs for the domain expert's review."""
    rows = ["| Pack | Purpose | include_domains | Topic |", "|---|---|---|---|"]
    for name, spec in DOMAIN_PACKS.items():
        doms = ", ".join(spec.get("include_domains") or []) or "(none: unrestricted)"
        if spec.get("boost_suffixes"):
            doms += "; boost " + ", ".join(spec["boost_suffixes"])
        if spec.get("official_domains"):
            doms += "; official " + ", ".join(spec["official_domains"])
        rows.append(f"| {name} | {spec.get('label', '')} | {doms} | {spec.get('topic', 'general')} |")
    return "\n".join(rows)
