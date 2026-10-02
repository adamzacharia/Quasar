"""Archive-aware pre-flight for model-written ADQL (vo_adql_query, advanced_search).

The generic TAP tools used to send any SELECT as written. On the 2026-10-01
MANNA-evals UI run that meant: NRAO questions sent to the ALMA mirror with
VLA filters, NRAO ADQL with ivoa.obscore / LOWER() (each a slow failure or a
silent zero), and Data Lab TAP ADQL with CONTAINS/CIRCLE or q3c(...) = 1 /
= true (all rejected by Data Lab's parser). This module fixes what has one
correct rewrite and stops what can only fail, BEFORE the network call, with a
hint that names the fix. Rules are keyed on the endpoint host, so an archive
Quasar knows nothing about is passed through untouched.

Sources: MANNA archives/nrao.py and archives/datalab.py (probe-verified notes),
references/references.md section 1 of the 2026-10-01 run (Data Lab TAP accepts
q3c_radial_query(...) = 't' only), services/archive_profiles/nrao.py.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple
from urllib.parse import urlsplit

from services.datalab_sql_policy import (
    _CONTAINS_CIRCLE_RE,
    _Q3C_CALL_RE,
    _contains_compared_to_other,
    _matching_paren,
    _scrub_sql,
)

NRAO_TAP_URL = "https://data-query.nrao.edu/tap"
NRAO_TAP_HOST = "data-query.nrao.edu"
ALMA_HOSTS = ("almascience.eso.org", "almascience.nrao.edu", "almascience.nao.ac.jp", "almascience.org")
DATALAB_HOST = "datalab.noirlab.edu"

# A quoted VLA/VLBA/GBT name used to SELECT rows: the right-hand side of =,
# IN (...) or LIKE. A projected label such as SELECT 'VLA' AS x is not a
# filter and passes (Codex CX-05).
_NON_ALMA_TELESCOPE_RE = re.compile(
    r"(?:=|\bLIKE|\bIN\s*\((?:[^)]*?,)?)\s*'\s*(?:E?VLA|JVLA|VLBA|GBT|VLASS[\w.]*)\s*'", re.IGNORECASE
)
_STRING_FN_RE = re.compile(r"\b(LOWER|UPPER)\s*\(|\bILIKE\b|\|\|", re.IGNORECASE)
_IVOA_OBSCORE_RE = re.compile(r"\bivoa\s*\.\s*obscore\b", re.IGNORECASE)
_OBSCORE_RE = re.compile(r"\b(?:tap_schema\s*\.\s*)?obscore\b", re.IGNORECASE)


class PreflightRejection(ValueError):
    """The query cannot succeed as written; ``hint`` says what to send instead."""

    def __init__(self, message: str, *, hint: str):
        super().__init__(message)
        self.hint = hint


def _host_path(url: str) -> Tuple[str, str]:
    try:
        parts = urlsplit(str(url or "").strip())
    except ValueError:
        return "", ""
    return (parts.hostname or "").lower().rstrip("."), (parts.path or "").rstrip("/").lower()


def preflight(access_url: Optional[str], adql: Optional[str], mode: Optional[str] = "sync") -> Tuple[Optional[str], List[str]]:
    """Return ``(adql_to_send, warnings)`` or raise :class:`PreflightRejection`.
    A query no rule applies to comes back as the SAME object (None stays None)."""
    text = str(adql or "")
    host, path = _host_path(access_url or "")
    if not host or not text.strip():
        return adql, []
    if host == NRAO_TAP_HOST:
        return _nrao(text, mode)
    if host.endswith(".nrao.edu") and host not in ALMA_HOSTS:
        raise PreflightRejection(
            f"{host} is not the NRAO archive's TAP service",
            hint=(f"The NRAO archive (VLA/EVLA/VLBA/GBT) TAP is {NRAO_TAP_URL}, table tap_schema.obscore. "
                  "Re-run vo_adql_query with that access_url; call browse_schema('nrao') for its rules."),
        )
    if host in ALMA_HOSTS:
        _alma(text, host)
        return adql, []
    if host == DATALAB_HOST and path.startswith("/tap"):
        return _datalab_tap(text)
    return adql, []


def _scrub_aligned(text: str) -> str:
    """_scrub_sql with raw offsets preserved: an escaped quote ('') shrinks the
    scrub by one char, so it is swapped for two non-quote placeholders first
    (inside a literal they stay inside it; an empty literal '' simply stops
    being a literal, which hides nothing). Codex CX-14: no comment stripping on
    raw text, which a literal such as 'O''--' would cut wrongly."""
    swapped = text.replace("''", "\x00\x00")
    clean = _scrub_sql(swapped)
    return clean if len(clean) == len(text) else " " * len(text)


def check_alma_adql(adql: Optional[str], host: str = "almascience.eso.org") -> None:
    """ALMA-only guard shared with advanced_search (raises PreflightRejection)."""
    _alma(str(adql or ""), host)


def _alma(text: str, host: str) -> None:
    # Only the WHERE clause can select rows; search the raw text after it (the
    # scrubbed text blanks literals, and the prefix before WHERE holds the
    # same offsets whenever no escaped quote precedes it).
    clean = _scrub_aligned(text)
    where = re.search(r"\bWHERE\b", clean, re.IGNORECASE)
    if not where:
        return
    hit = None
    for m in _NON_ALMA_TELESCOPE_RE.finditer(text, where.end()):
        # the operator must be code: _scrub_sql blanks comments and literal
        # interiors, so a match inside `-- instrument_name = 'VLA'` has a
        # blank there (Codex CX-05 reopen).
        if clean[m.start()] == text[m.start()] and not clean[m.start()].isspace():
            hit = m
            break
    if not hit:
        return
    name = re.search(r"'([^']*)'", hit.group(0)).group(1).strip()
    raise PreflightRejection(
        f"The ALMA archive ({host}) holds ALMA data only; it has no '{name}' rows",
        hint=("almascience.* (including almascience.nrao.edu, ALMA's North American mirror) is not the NRAO "
              f"archive. VLA/EVLA/VLBA/GBT data are in the NRAO archive: vo_adql_query(access_url='{NRAO_TAP_URL}', "
              "adql against tap_schema.obscore with instrument_name = 'EVLA'|'VLA'|'VLBA'|'GBT', mode='auto' or "
              "'async'), or search_by_position / search_by_target with facility='VLA'|'VLBA'|'GBT'."),
    )


def _nrao(text: str, mode: Optional[str]) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    clean = _scrub_sql(text)
    fn = _STRING_FN_RE.search(clean)
    if fn:
        raise PreflightRejection(
            f"NRAO's TAP rejects {fn.group(0).strip().rstrip('(').upper()}: its ADQL has no string functions "
            "(LOWER, UPPER, ILIKE) and no || operator",
            hint=("Match exact case, or OR together LIKE patterns for the case variants, e.g. "
                  "(target_name LIKE '%M87%' OR target_name LIKE '%m87%' OR target_name = '3C274'); "
                  "radio sources often carry radio names, so a CONTAINS cone on (s_ra, s_dec) is more reliable."),
        )
    if _IVOA_OBSCORE_RE.search(clean):
        text = _IVOA_OBSCORE_RE.sub("tap_schema.obscore", text)
        clean = _scrub_sql(text)
        warnings.append("Rewrote ivoa.obscore to tap_schema.obscore: NRAO keeps ObsCore at that non-standard "
                        "location and has no ivoa.obscore table.")
    if _OBSCORE_RE.search(clean) and not re.search(r"\bWHERE\b", clean, re.IGNORECASE):
        raise PreflightRejection(
            "Unfiltered reads of NRAO's tap_schema.obscore fail (sync returns an ERROR after ~60 s; async "
            "COUNT/DISTINCT scans end in ERROR after ~30 min)",
            hint=("Add a selective WHERE: a cone 1=CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', ra, dec, r)) "
                  "for positional asks, or instrument_name = 'EVLA'|'VLA'|'VLBA'|'GBT' / project_code = '...'; "
                  "then run with mode='auto' or 'async' and poll vo_tap_job."),
        )
    if _OBSCORE_RE.search(clean) and str(mode or "sync").lower() == "sync":
        warnings.append("NRAO obscore data reads are load-dependent in sync and often time out; if this fails, "
                        "re-run with mode='auto' (promotes to async) or mode='async'.")
    return text, warnings


def _datalab_tap(text: str) -> Tuple[str, List[str]]:
    """Data Lab's TAP passes ADQL to PostgreSQL untranslated and its parser accepts
    only q3c_radial_query(...) = 't' (references.md s1: = 1, = true, bare and
    CONTAINS/POINT/CIRCLE all fail)."""
    warnings: List[str] = []
    clean = _scrub_sql(text)
    if len(text) != len(clean):  # escaped quotes: offsets unusable, send as written
        return text, warnings
    if re.search(r"\bARRAY\s*\[", clean, re.IGNORECASE):
        raise PreflightRejection(
            "Data Lab's TAP ADQL parser cannot read ARRAY[...] (q3c_poly_query)",
            hint=("Use datalab_sql_query (the query manager runs PostgreSQL and accepts q3c_poly_query and ra/dec "
                  "BETWEEN boxes), or write the box here as ra BETWEEN a AND b AND dec BETWEEN c AND d."),
        )
    out: List[str] = []
    pos = 0
    n = 0
    for m in _CONTAINS_CIRCLE_RE.finditer(text):
        if clean[m.start("kw"): m.end("kw")].upper() != "CONTAINS" or _contains_compared_to_other(text, m):
            continue
        out.append(text[pos: m.start()])
        out.append(f"q3c_radial_query({m.group('ra')}, {m.group('dec')}, {m.group('a')}, {m.group('b')}, {m.group('r')}) = 't'")
        pos = m.end()
        n += 1
    if n:
        text = "".join(out) + text[pos:]
        clean = _scrub_sql(text)
        warnings.append(f"Rewrote {n} ADQL CONTAINS(POINT, CIRCLE) cone(s) to q3c_radial_query(...) = 't': Data Lab "
                        "does not translate ADQL geometry. Same cone, indexed.")
    edits: List[Tuple[int, int, str]] = []
    for m in _Q3C_CALL_RE.finditer(clean):
        close = _matching_paren(clean, m.end() - 1)
        if close is None:
            continue
        after = clean[close + 1:]
        if re.match(r"\s*=\s*'", text[close + 1:]):  # already = 't' (literal blanked in clean)
            continue
        cmp_tail = re.match(r"\s*=\s*(?:1|true)\b", after, re.IGNORECASE)
        head = re.search(r"\b(?:1|true)\s*=\s*$", clean[: m.start()], re.IGNORECASE)
        if cmp_tail:
            edits.append((close + 1, close + 1 + cmp_tail.end(), " = 't'"))
        elif head:
            edits.append((head.start(), m.start(), ""))
            edits.append((close + 1, close + 1, " = 't'"))
        elif not re.match(r"\s*(?:=|<>|!=)", after):
            edits.append((close + 1, close + 1, " = 't'"))
    if edits:
        for start, end, repl in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
            text = text[:start] + repl + text[end:]
        warnings.append("Normalised q3c predicate(s) to = 't': Data Lab's TAP parser rejects a bare q3c call, "
                        "= 1 and = true.")
    return text, warnings


__all__ = ["PreflightRejection", "preflight", "check_alma_adql", "NRAO_TAP_URL"]
