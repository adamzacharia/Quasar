"""Authority scoring for web evidence surfaced in Quasar.

The score is deliberately source-authority based. It does not look at which
search provider returned the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse


@dataclass(frozen=True)
class AuthorityRule:
    label: str
    tier: str
    score: int
    host_suffixes: tuple[str, ...] = ()
    host_contains: tuple[str, ...] = ()
    path_contains: tuple[str, ...] = ()
    signals: tuple[str, ...] = ()


AUTHORITY_RULES: tuple[AuthorityRule, ...] = (
    AuthorityRule(
        label="Primary archive",
        tier="primary",
        score=96,
        host_contains=(
            "archive.alma",
            "almascience.",
            "mast.stsci.edu",
            "archive.stsci.edu",
            "cadc-ccda.hia-iha.nrc-cnrc.gc.ca",
            "irsa.ipac.caltech.edu",
            "ned.ipac.caltech.edu",
            "heasarc.gsfc.nasa.gov",
        ),
        signals=("official astronomy archive",),
    ),
    AuthorityRule(
        label="Observatory docs",
        tier="primary",
        score=92,
        host_contains=(
            "almaobservatory.org",
            "nrao.edu",
            "eso.org",
            "stsci.edu",
            "esa.int",
            "science.nasa.gov",
        ),
        path_contains=("doc", "guide", "handbook", "manual", "kb", "help"),
        signals=("official observatory or agency documentation",),
    ),
    AuthorityRule(
        label="Peer reviewed",
        tier="peer_reviewed",
        score=90,
        host_contains=(
            "adsabs.harvard.edu",
            "ui.adsabs.harvard.edu",
            "iopscience.iop.org",
            "academic.oup.com",
            "aanda.org",
            "nature.com",
            "science.org",
            "springer.com",
            "link.springer.com",
            "journals.aps.org",
            "doi.org",
        ),
        signals=("journal, DOI, or ADS record",),
    ),
    AuthorityRule(
        label="Science database",
        tier="reference",
        score=84,
        host_contains=(
            "simbad.cds.unistra.fr",
            "vizier.cds.unistra.fr",
            "cdsarc.cds.unistra.fr",
            "splatalogue.online",
            "ned.ipac.caltech.edu",
        ),
        signals=("curated astronomy reference database",),
    ),
    AuthorityRule(
        label="Preprint",
        tier="preprint",
        score=78,
        host_contains=("arxiv.org",),
        signals=("scientific preprint",),
    ),
    AuthorityRule(
        label="Institutional",
        tier="institutional",
        score=72,
        host_suffixes=(".edu", ".gov", ".mil", ".int"),
        signals=("institutional or government domain",),
    ),
    AuthorityRule(
        label="Community reference",
        tier="reference",
        score=58,
        host_contains=("wikipedia.org", "wikidata.org"),
        signals=("community-maintained reference",),
    ),
)

GENERAL_QUALITY = {
    "score": 42,
    "tier": "general",
    "label": "General web",
    "reason": "No strong scientific authority signal detected.",
    "signals": [],
}


def _host_matches(host: str, rule: AuthorityRule) -> bool:
    return any(host == suffix.lstrip(".") or host.endswith(suffix) for suffix in rule.host_suffixes) or any(
        token in host for token in rule.host_contains
    )


def _path_matches(path: str, rule: AuthorityRule) -> bool:
    if not rule.path_contains:
        return True
    return any(token in path for token in rule.path_contains)


def _parse_url(url: str) -> tuple[str, str]:
    parsed = urlparse(str(url or "").strip())
    host = (parsed.hostname or "").lower().removeprefix("www.")
    path = (parsed.path or "").lower()
    return host, path


def assess_web_source_quality(url: str, title: str = "", snippet: str = "") -> Dict[str, Any]:
    """Return a compact source authority assessment for a web result."""
    host, path = _parse_url(url)
    if not host:
        return dict(GENERAL_QUALITY)

    title_l = str(title or "").lower()
    snippet_l = str(snippet or "").lower()
    best: Optional[AuthorityRule] = None

    for rule in AUTHORITY_RULES:
        if _host_matches(host, rule) and _path_matches(path, rule):
            best = rule
            break

    if best is None:
        if any(term in title_l or term in snippet_l for term in ("doi:", "bibcode", "journal", "accepted", "apj", "mnras")):
            return {
                "score": 74,
                "tier": "institutional",
                "label": "Likely scholarly",
                "reason": "Snippet/title contains scholarly publication signals.",
                "signals": ["publication metadata in title or snippet"],
            }
        return dict(GENERAL_QUALITY)

    return {
        "score": best.score,
        "tier": best.tier,
        "label": best.label,
        "reason": "; ".join(best.signals) if best.signals else best.label,
        "signals": list(best.signals),
    }


def annotate_web_source_evidence(source: Dict[str, Any]) -> Dict[str, Any]:
    """Attach evidenceQuality if missing, preserving any higher caller-supplied score."""
    clean = dict(source)
    computed = assess_web_source_quality(
        str(clean.get("url") or ""),
        str(clean.get("title") or ""),
        str(clean.get("snippet") or clean.get("description") or ""),
    )
    existing = clean.get("evidenceQuality") or clean.get("evidence_quality")
    clean["evidenceQuality"] = choose_better_evidence_quality(existing, computed)
    clean.pop("evidence_quality", None)
    return clean


def choose_better_evidence_quality(existing: Any, incoming: Any) -> Dict[str, Any]:
    """Keep the stronger normalized quality record when merging duplicate URLs."""
    existing_q = normalize_evidence_quality(existing)
    incoming_q = normalize_evidence_quality(incoming)
    if not existing_q:
        return incoming_q or dict(GENERAL_QUALITY)
    if not incoming_q:
        return existing_q
    return incoming_q if incoming_q.get("score", 0) > existing_q.get("score", 0) else existing_q


def normalize_evidence_quality(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    score = value.get("score", 0)
    try:
        score_int = max(0, min(100, int(score)))
    except (TypeError, ValueError):
        score_int = 0
    label = str(value.get("label") or "").strip() or "General web"
    tier = str(value.get("tier") or "").strip() or "general"
    reason = str(value.get("reason") or "").strip()
    signals_raw = value.get("signals")
    signals: Iterable[Any] = signals_raw if isinstance(signals_raw, list) else []
    return {
        "score": score_int,
        "tier": tier,
        "label": label,
        "reason": reason,
        "signals": [str(signal) for signal in signals if str(signal).strip()],
    }


def rank_web_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return sources sorted by evidence quality score, preserving input order for ties."""
    annotated = [annotate_web_source_evidence(source) for source in sources if isinstance(source, dict)]
    return [
        source
        for _, source in sorted(
            enumerate(annotated),
            key=lambda item: (item[1].get("evidenceQuality", {}).get("score", 0), -item[0]),
            reverse=True,
        )
    ]
