"""Papers that used ALMA data on a topic, with an archival-use classification (alma_bibliography backend).

Research 2026-10-16 (tmp/almabench-2026-10-15/round4/ads_archival_research.md, all numbers live):

* ADS ``bibgroup:ALMA`` (5,154 records) is the ALMA telescope bibliography (telbib, 5,157). ADS has no archival flag,
  no ALMA project-code field and no PI field.
* ALMA ObsCore links projects to papers: ``SELECT DISTINCT proposal_id, pi_name, proposal_authors, bib_reference FROM
  ivoa.obscore WHERE bib_reference IS NOT NULL`` returns 3,202 projects / 5,103 bibcodes in ~4 s, with the PI and co-Is.
* ALMA's own definition of archival use (Stoehr et al. 2026, arXiv:2601.16687; ALMA publication statistics) is that no
  author of the paper is the PI or a co-I of the project whose data it used. Applied to the whole bibliography this
  author-overlap test gives 39% archival papers in 2024 (official figure: 39%) and agrees with the telbib curators on
  34/34 "PI data" and 7/8 "archival only" papers, but only 4/13 "archival + PI" (large collaborations whose members are
  not on the proposal's co-I list).

So: ADS search (topic + synonyms, newest first) -> literal topical filter -> project linkage from the cached ObsCore
map -> author-overlap class with a stated confidence. Every returned paper comes from the ADS response itself, so no
bibcode is invented, and the result says the list is closed.
"""
from __future__ import annotations

import re
import threading
import time
import unicodedata
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ARCHIVAL_ONLY = "archival only"
ARCHIVAL_PLUS_PI = "archival + PI"
PI_DATA = "PI data"
UNKNOWN = "unknown"
CONFIDENCE = {PI_DATA: "high", ARCHIVAL_ONLY: "medium", ARCHIVAL_PLUS_PI: "low", UNKNOWN: "none"}

BIB_MAP_ADQL = ("SELECT DISTINCT proposal_id, pi_name, proposal_authors, bib_reference FROM ivoa.obscore "
                "WHERE bib_reference IS NOT NULL")
BIB_MAP_TTL_S = 24 * 3600.0
_MAP_LOCK = threading.Lock()
_MAP_CACHE: Dict[str, Any] = {"at": 0.0, "map": None}

METHOD_NOTE = (
    "Archival use is decided the way ALMA's publication statistics define it: a paper used archival data from a project "
    "when none of its authors is that project's PI or a co-I (author lists from ADS, PI/co-Is and paper-project links "
    "from the ALMA archive's ObsCore bib_reference). Limits: 'archival + PI' is often wrong for large collaborations "
    "whose members are not on the proposal; papers the ALMA bibliography has not curated yet (the newest weeks to "
    "months) are missing; name matching is surname + first initial."
)


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii")


def name_key(name: str) -> str:
    """'Surname, First M.' / 'First M. Surname' -> 'surname f' (accents dropped, lowercase)."""
    raw = _ascii(name).strip().lower()
    if not raw:
        return ""
    if "," in raw:
        surname, _, rest = raw.partition(",")
    else:
        parts = raw.split()
        surname, rest = (parts[-1], " ".join(parts[:-1])) if len(parts) > 1 else (raw, "")
    surname = re.sub(r"[^a-z]", "", surname)
    initial = next((c for c in rest.strip() if c.isalpha()), "")
    return f"{surname} {initial}".strip()


def parse_bib_map(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """bibcode -> [{project, pi, team (name keys of PI + co-Is)}] from the ObsCore bib_reference rows."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        project = str(row.get("proposal_id") or "").strip()
        if not project:
            continue
        pi = str(row.get("pi_name") or "").strip()
        coi = [a.strip() for a in str(row.get("proposal_authors") or "").split(";") if a.strip()]
        team = {k for k in (name_key(n) for n in [pi] + coi) if k}
        for bib in str(row.get("bib_reference") or "").split():
            entry = out.setdefault(bib.strip(), [])
            if all(e["project"] != project for e in entry):
                entry.append({"project": project, "pi": pi, "team": team})
    return out


def bib_map(run_query: Callable[[str], Any], *, now: Optional[float] = None) -> Tuple[Dict[str, List[Dict[str, Any]]], str, bool]:
    """The bibcode -> projects map, when it was built, and whether it is complete. Only a complete answer is cached
    (24 h): a row-capped answer can drop a paper's other projects and flip its archival class (guard CX-05)."""
    t = time.time() if now is None else now
    with _MAP_LOCK:
        if _MAP_CACHE["map"] is not None and t - _MAP_CACHE["at"] < BIB_MAP_TTL_S:
            return _MAP_CACHE["map"], time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(_MAP_CACHE["at"])), True
    frame = run_query(BIB_MAP_ADQL)
    truncated = bool(getattr(frame, "attrs", {}).get("truncated"))
    rows = frame.to_dict("records") if hasattr(frame, "to_dict") else list(frame)
    mapping = parse_bib_map(rows)
    if not truncated:
        with _MAP_LOCK:
            _MAP_CACHE.update(at=t, map=mapping)
    return mapping, time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(t)), not truncated


def classify(paper_authors: Sequence[str], projects: Sequence[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Author-overlap class of one paper and the per-project evidence."""
    if not projects:
        return UNKNOWN, []
    authors = {k for k in (name_key(a) for a in paper_authors) if k}
    evidence = []
    for p in projects:
        sv = p["project"].endswith(".SV")  # science-verification data are public by design: always archival use
        overlap = (not sv) and bool(authors & set(p.get("team") or ()))
        evidence.append({"project": p["project"], "pi": p.get("pi"), "author_on_proposal": overlap})
    flags = [e["author_on_proposal"] for e in evidence]
    if not any(flags):
        return ARCHIVAL_ONLY, evidence
    if all(flags):
        return PI_DATA, evidence
    return ARCHIVAL_PLUS_PI, evidence


def topic_patterns(phrases: Sequence[str]) -> List[re.Pattern]:
    pats = []
    for phrase in phrases:
        words = [re.escape(w) for w in str(phrase or "").strip().split() if w]
        if not words:
            continue
        last = words[-1]
        if re.search(r"[^aeiou]y$", last, re.I):  # galaxy -> galaxies
            words[-1] = last[:-1] + r"(?:y|ies)"
        else:
            words[-1] = last + r"(?:s|es)?"
        pats.append(re.compile(r"\b" + r"[\s-]+".join(words) + r"\b", re.I))
    return pats


def topic_evidence(paper: Dict[str, Any], patterns: Sequence[re.Pattern]) -> Optional[str]:
    """Where the topic shows up: title or keywords, or at least two mentions in the abstract; None = off topic."""
    title = paper.get("title")
    title = " ".join(title) if isinstance(title, list) else str(title or "")
    keywords = " ".join(str(k) for k in (paper.get("keyword") or paper.get("keywords") or []))
    abstract = str(paper.get("abstract") or "")
    for pat in patterns:
        if pat.search(title):
            return f"title: '{pat.search(title).group(0)}'"
        if pat.search(keywords):
            return f"keyword: '{pat.search(keywords).group(0)}'"
    for pat in patterns:
        hits = pat.findall(abstract)
        if len(hits) >= 2:
            return f"abstract: '{hits[0]}' x{len(hits)}"
    return None


def ads_query(phrases: Sequence[str]) -> str:
    terms = " OR ".join(f'abs:"{p.strip()}"' for p in phrases if str(p or "").strip())
    return f"bibgroup:ALMA ({terms})"


def summarize(abstract: str, limit: int = 1500) -> str:
    """The abstract in whole sentences up to ``limit`` characters (most abstracts fit), so the answer can say what
    each paper observed and found; the opening sentences alone are usually background (ALMAbench round 5)."""
    text = re.sub(r"\s+", " ", str(abstract or "")).strip()
    if not text:
        return "(no abstract in the ADS record)"
    if len(text) <= limit:
        return text
    out = ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if len(out) + len(sent) + 1 > limit:
            break
        out = f"{out} {sent}".strip()
    return (out or text[:limit]) + " [...]"
