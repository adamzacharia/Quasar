"""Answer-versus-trace verifier: the prose must match what the tools did.

UI benchmark 2026-09-22 evidence (tmp/ui-bench-2026-09-22/):

* L09 re-run -- the answer claims proper-motion, magnitude, colour and
  class_star cuts that the executed SQL never applied.
* L11 -- "the sky-density map is shown below" while no map card was rendered.
* L15 -- a CMD "interpretation" over plots that show field scatter.
* D10 -- a calibrator quasar named as the science target.
* D11 -- ``n_projects`` (30) reported instead of ``unique_projects`` (1).
* D18 -- an ALMA phase timeout reported as "there are no Perseus protostars
  with both ALMA and JWST".

The verifier is DETERMINISTIC (no LLM). It builds a :class:`TraceSummary` of
the turn -- executed SQL/ADQL text, tool arguments, result counts, attached
card titles/ids, tool errors and timed-out / skipped phases -- and checks the
final answer against it:

(a) every numeric cut the answer states (``pmra > -4``, ``class_star > 0.5``,
    ``g - r < 0.9``, ``LIMIT 5000``, ``radius = 2 deg``, ``z between 0.4 and
    0.8`` ...) must occur in the executed SQL/ADQL or the tool arguments;
(b) every "shown / displayed / attached above|below" must have an attached
    card this turn;
(c) every count the answer states ("30 projects", "1 313 sources", "5000
    rows") must appear in some tool result;
(d) if an archive phase timed out or was skipped, the answer may not state a
    null result ("no sources", "none exist") as fact.

On violation nothing is rewritten -- a rewrite would have to invent -- and a
short, factual **Verification** block listing each unsupported claim is
appended; the runner logs ``[VERIFY] unsupported_claims=n``.

Everything here is a leaf (stdlib + json) so it is unit-testable against the
saved ``answer.md`` / ``queries.md`` fixtures of the benchmark folders.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "Claim",
    "TraceSummary",
    "VerificationReport",
    "build_trace_summary",
    "format_verification_block",
    "verify_answer",
]

# ── trace summary ────────────────────────────────────────────────────────

_SQL_KEYS = ("validated_sql", "query_summary", "sql", "adql", "query", "executed_sql", "executed_query")
_COUNT_KEYS = (
    "rowcount", "reported_count", "row_count", "count", "n", "n_rows", "num_rows", "total", "total_count",
    "n_projects", "unique_projects", "n_sources", "n_matches", "matches", "n_results", "num_results",
    "results_count", "n_mous", "n_datasets", "n_execution_blocks", "n_eb", "rows_returned", "returned_rows",
    "hits", "n_hits", "num_projects", "project_count", "source_count", "n_candidates", "n_rows_total",
    "distinct_projects", "n_distinct_projects", "n_rows_returned",
)
_TIMEOUT_MARKERS = ("timed out", "timeout", "budget exhausted", "deadline", "did not answer within",
                    "circuit open", "circuit breaker", "unreachable", "infrastructure failure", "not executed",
                    "skipped", "not attempted", "not queried")


@dataclass
class TraceSummary:
    """What the tools of ONE turn actually did."""

    sql_texts: List[str] = field(default_factory=list)          # executed SQL / ADQL, verbatim
    arg_texts: List[str] = field(default_factory=list)          # tool arguments (JSON), one per call
    counts: set = field(default_factory=set)                    # every integer that appeared as a count
    numbers: set = field(default_factory=set)                   # every number in results (looser check)
    cards: List[Dict[str, Any]] = field(default_factory=list)   # attached visual/data cards {type, title, id}
    errors: List[str] = field(default_factory=list)             # tool error strings
    timed_out_phases: List[str] = field(default_factory=list)   # tool / archive phases that timed out or were skipped
    tools_called: List[str] = field(default_factory=list)
    partial: bool = False
    science_targets: set = field(default_factory=set)       # target_name with science_observation='T'
    calibrator_targets: set = field(default_factory=set)    # target_name of calibrator rows only
    count_pairs: List[Tuple[int, int]] = field(default_factory=list)  # (n_projects in window, unique_projects matching)
    result_texts: List[str] = field(default_factory=list)   # string leaves of tool results (notes, warnings, summaries)
    timed_out_archives: set = field(default_factory=set)    # archive names found in timed_out_phases (CX-20)
    keyed_counts: set = field(default_factory=set)          # (lower-case key, int) of every non-negative integer leaf
    query_arg_texts: List[str] = field(default_factory=list)  # arguments of DATA tools only (not documentation tools, CX-19)

    # Normalised search corpora (computed lazily)
    _sql_norm: Optional[str] = None
    _args_norm: Optional[str] = None
    _all_norm: Optional[str] = None

    def sql_corpus(self) -> str:
        if self._sql_norm is None:
            self._sql_norm = _norm(" \n ".join(self.sql_texts))
        return self._sql_norm

    def args_corpus(self) -> str:
        if self._args_norm is None:
            self._args_norm = _norm(" \n ".join(self.arg_texts))
        return self._args_norm

    def evidence_corpus(self) -> str:
        if self._all_norm is None:
            self._all_norm = self.sql_corpus() + " \n " + self.args_corpus()
        return self._all_norm

    @property
    def has_cards(self) -> bool:
        return bool(self.cards)

    @property
    def has_sql(self) -> bool:
        return bool(self.sql_texts)


_NUM_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:[ ,  ]\d{3})*(?:\.\d+)?(?![\w.])")


def _norm(text: str) -> str:
    """Lower-case, collapse whitespace, unify unicode minus/dashes and
    comparison spellings so answer phrases can be matched against SQL."""
    t = str(text or "")
    t = t.replace("−", "-").replace("–", "-").replace("—", "-").replace("‑", "-")
    t = t.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    t = t.replace("≥", ">=").replace("≤", "<=").replace("≠", "!=")
    t = re.sub(r"\s+", " ", t)
    return t.lower().strip()


def _clean_number(token: str) -> Optional[float]:
    t = token.replace(" ", "").replace(",", "").replace(" ", "").replace(" ", "")
    try:
        return float(t)
    except ValueError:
        return None


def _walk(obj: Any, depth: int = 0):
    """Yield (key_path_last_key, value) for every leaf of a JSON-like object."""
    if depth > 8:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                yield from _walk(v, depth + 1)
            else:
                yield str(k), v
    elif isinstance(obj, list):
        for v in obj[:5000]:
            if isinstance(v, (dict, list)):
                yield from _walk(v, depth + 1)
            else:
                yield "", v


def _parse_maybe_json(text: Any) -> Any:
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s or s[0] not in "{[":
        return None
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return None


def build_trace_summary(
    tool_results: Sequence[Any] = (),
    tool_trace: Sequence[Dict[str, Any]] = (),
    run_results: Sequence[Any] = (),
    extra_sql: Iterable[str] = (),
) -> TraceSummary:
    """Summarise the turn from the runner's evidence:

    ``tool_results`` -- the ``function_call_output`` list (each ``{"output": json}``)
    or raw result dicts; ``tool_trace`` -- ``agent._accumulated_tool_trace`` records
    (``name``, ``arguments``, ``output``, ``ok``, ``sql``, ``request``); ``run_results``
    -- ``agent._accumulated_run_results`` (cards: ``type``, ``title``); ``extra_sql``
    -- any further executed query text (e.g. ALMA TAP provenance)."""
    ts = TraceSummary()
    seen_sql: set = set()

    def add_sql(text: Any) -> None:
        s = str(text or "").strip()
        if len(s) < 8:
            return
        key = _norm(s)
        if key in seen_sql:
            return
        seen_sql.add(key)
        ts.sql_texts.append(s)

    def add_number(v: Any) -> None:
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            ts.numbers.add(float(v))

    def ingest_result(obj: Any, support: bool = True) -> None:
        """``support=False`` (a FAILED / refused call, guard CX-18): record its
        errors and timed-out phases, but nothing it carries (SQL, arguments,
        numbers, text) may substantiate a claim."""
        if isinstance(obj, str):
            parsed = _parse_maybe_json(obj)
            if parsed is None:
                low = obj.lower()
                if any(m in low for m in _TIMEOUT_MARKERS):
                    ts.timed_out_phases.append(obj[:160])
                return
            obj = parsed
        if not isinstance(obj, (dict, list)):
            return
        if isinstance(obj, dict):
            if obj.get("repeated_call") and isinstance(obj.get("previous_result"), dict):
                obj = obj["previous_result"]
            if obj.get("success") is False:
                support = False
            for k in _SQL_KEYS:
                v = obj.get(k)
                if support and isinstance(v, str) and _looks_like_query(v):
                    add_sql(v)
            prov = obj.get("provenance")
            if support and isinstance(prov, dict):
                for k in ("query", "adql", "sql"):
                    if isinstance(prov.get(k), str):
                        add_sql(prov[k])
            if obj.get("partial"):
                ts.partial = True
            err = obj.get("error")
            if isinstance(err, str) and err.strip():
                ts.errors.append(err[:300])
                low = err.lower()
                if any(m in low for m in _TIMEOUT_MARKERS) or obj.get("timeout") or obj.get("infrastructure_failure"):
                    ts.timed_out_phases.append(err[:160])
            elif obj.get("timeout") or obj.get("infrastructure_failure") or obj.get("budget_exhausted"):
                ts.timed_out_phases.append(str(obj.get("note") or obj.get("status") or "timeout")[:160])
            # Archive-phase status columns (cross_archive_match & co.)
            for k in ("archive_errors", "phase_errors", "errors", "warnings", "dead_hosts", "skipped", "skipped_sources"):
                v = obj.get(k)
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        text = f"{kk}: {vv}"
                        ts.errors.append(text[:300])
                        if any(m in text.lower() for m in _TIMEOUT_MARKERS):
                            ts.timed_out_phases.append(text[:160])
                elif isinstance(v, list):
                    for item in v[:50]:
                        text = json.dumps(item, default=str) if isinstance(item, (dict, list)) else str(item)
                        if k in ("archive_errors", "phase_errors", "dead_hosts", "skipped", "skipped_sources") or any(
                            m in text.lower() for m in _TIMEOUT_MARKERS
                        ):
                            ts.errors.append(text[:300])
                            ts.timed_out_phases.append(text[:160])
            for k in ("archive_status", "phase_status", "status_by_archive", "archives"):
                v = obj.get(k)
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        vs = str(vv).lower()
                        if any(m in vs for m in ("timeout", "timed out", "skipped", "unknown", "budget", "circuit", "fail")):
                            ts.timed_out_phases.append(f"{kk}: {vv}"[:160])
            status = str(obj.get("status") or "").lower()
            if status in ("partial", "infrastructure_failure", "timeout", "coverage_gap"):
                ts.timed_out_phases.append(f"status={status}")
        if not support:
            return
        for key, value in _walk(obj):
            kl = key.lower()
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                add_number(value)
                if kl and float(value).is_integer() and value >= 0:
                    ts.keyed_counts.add((kl, int(value)))
                if kl in _COUNT_KEYS or kl.startswith(("n_", "num_", "count", "rowcount")) or kl.endswith(("_count", "count", "_total")):
                    if float(value).is_integer() and value >= 0:
                        ts.counts.add(int(value))
            elif isinstance(value, str):
                if 0 < len(value) <= 2000 and len(ts.result_texts) < 4000:
                    ts.result_texts.append(value)
                if kl in _SQL_KEYS and _looks_like_query(value):
                    add_sql(value)
                elif kl in _COUNT_KEYS:
                    n = _clean_number(value)
                    if n is not None and n.is_integer():
                        ts.counts.add(int(n))

    for item in tool_results or ():
        if isinstance(item, dict) and "output" in item and isinstance(item.get("output"), str):
            ingest_result(item["output"])
        else:
            ingest_result(item)

    for rec in tool_trace or ():
        if not isinstance(rec, dict):
            continue
        name = str(rec.get("name") or "")
        if name:
            ts.tools_called.append(name)
        failed = rec.get("ok") is False
        out = rec.get("output")
        if failed:
            # An unexecuted / rejected call substantiates nothing (CX-18).
            if isinstance(out, str):
                ingest_result(out, support=False)
            continue
        args = rec.get("arguments")
        if isinstance(args, dict) and args:
            blob = json.dumps(args, default=str, sort_keys=True)
            ts.arg_texts.append(blob)
            if name not in DOCUMENTATION_TOOLS:
                ts.query_arg_texts.append(blob)
            for k in _SQL_KEYS:
                if isinstance(args.get(k), str) and _looks_like_query(args[k]):
                    add_sql(args[k])
        if isinstance(rec.get("sql"), str):
            add_sql(rec["sql"])
        req = rec.get("request")
        if isinstance(req, dict) and isinstance(req.get("text"), str) and req.get("kind") in ("adql", "sql", None):
            add_sql(req["text"])
        if isinstance(out, str):
            ingest_result(out)

    for rr in run_results or ():
        if not isinstance(rr, dict):
            continue
        rtype = str(rr.get("type") or "")
        if rtype in ("image", "plotly", "data", "conductor_result", "notebook", "papers", "web_sources"):
            ts.cards.append({
                "type": rtype,
                "title": str(rr.get("title") or rr.get("caption") or rr.get("name") or "")[:160],
                "id": str(rr.get("_result_id") or rr.get("result_id") or rr.get("id") or ""),
                # The producing tool (stamped by the runner as requestTool):
                # a plot_sky_map card IS a map even when its title is the
                # user's caption.
                "tool": str(rr.get("requestTool") or rr.get("tool_name") or rr.get("tool") or ""),
            })
        for k in _SQL_KEYS:
            if isinstance(rr.get(k), str) and _looks_like_query(rr[k]):
                add_sql(rr[k])
        req = rr.get("request")
        if isinstance(req, dict) and isinstance(req.get("text"), str):
            add_sql(req["text"])
        prov = rr.get("provenance")
        if isinstance(prov, dict):
            for k in ("query", "adql", "sql"):
                if isinstance(prov.get(k), str):
                    add_sql(prov[k])
        rc = rr.get("rowcount", rr.get("row_count"))
        if isinstance(rc, int):
            ts.counts.add(rc)
        rows = rr.get("rows") or rr.get("data")
        if isinstance(rows, list):
            ts.counts.add(len(rows))
            for row in rows[:2000]:
                if isinstance(row, dict):
                    for k, v in row.items():
                        add_number(v)
                        if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
                            ts.keyed_counts.add((str(k).lower(), v))
                elif isinstance(row, (list, tuple)):
                    for v in row:
                        add_number(v)

    for s in extra_sql or ():
        add_sql(s)

    # Target roles (calibrator vs science) and window-vs-matching count pairs.
    ts.science_targets, ts.calibrator_targets = _target_roles(tool_results, run_results)

    def collect_pairs(obj: Any) -> None:
        if isinstance(obj, str):
            obj = _parse_maybe_json(obj)
        if isinstance(obj, dict):
            n_all = obj.get("n_projects", obj.get("projects_in_window"))
            n_match = obj.get("unique_projects", obj.get("matching_projects", obj.get("n_matching_projects")))
            if isinstance(n_all, int) and isinstance(n_match, int) and not isinstance(n_all, bool):
                ts.count_pairs.append((n_all, n_match))
            for v in obj.values():
                if isinstance(v, dict):
                    collect_pairs(v)

    for item in tool_results or ():
        collect_pairs(item["output"] if isinstance(item, dict) and isinstance(item.get("output"), str) else item)
    for rr in run_results or ():
        collect_pairs(rr)
    for phase in ts.timed_out_phases:
        ts.timed_out_archives |= _archives_in(phase)
    return ts


# Tools whose arguments are topics / library names, not selections (CX-19).
DOCUMENTATION_TOOLS = frozenset({
    "alma_reference", "alma_reference_table", "code_recipe", "search_documentation", "search_alma_docs",
    "rag_search", "web_search", "search_papers", "alma_bibliography", "get_paper_abstract", "lookup_researcher",
})

_ARCHIVE_NAMES = {
    "alma": ("alma", "asa"), "jwst": ("jwst",), "hst": ("hst", "hubble"), "mast": ("mast",), "simbad": ("simbad",),
    "ned": ("ned",), "gaia": ("gaia",), "datalab": ("data lab", "datalab", "noirlab"), "vizier": ("vizier",),
    "sdss": ("sdss",), "desi": ("desi",), "vla": ("vla", "vlass", "nrao"), "chandra": ("chandra", "cxc"),
    "irsa": ("irsa", "wise", "2mass", "spitzer"), "eso": ("eso",), "cadc": ("cadc",), "ads": ("ads",),
    "splatalogue": ("splatalogue",), "cds": ("cds", "hips", "aladin"),
}


def _archives_in(text: str) -> set:
    low = f" {_norm(text)} "
    found = set()
    for name, words in _ARCHIVE_NAMES.items():
        if any(re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", low) for w in words):
            found.add(name)
    return found


_QUERY_HINT_RE = re.compile(r"\b(select|from|where|with|order by|group by|limit|top)\b", re.I)


def _looks_like_query(text: str) -> bool:
    t = str(text or "")
    return len(t) >= 12 and bool(_QUERY_HINT_RE.search(t))


# ── claims ───────────────────────────────────────────────────────────────

def _display(text: str) -> str:
    """Claim text as shown to the user: plain spaces and ASCII minus (the model
    emits narrow no-break spaces and U+2212 in cuts)."""
    t = str(text or "")
    for ch in (" ", " ", " ", " "):
        t = t.replace(ch, " ")
    t = t.replace("−", "-").replace("‑", "-").replace("–", "-")
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class Claim:
    kind: str          # "cut" | "artifact" | "count" | "null_result" | "calibrator"
    text: str          # the phrase from the answer
    detail: str = ""   # why it is unsupported

    def __post_init__(self) -> None:
        self.text = _display(self.text)

    def as_line(self) -> str:
        return f"{self.text}" + (f" — {self.detail}" if self.detail else "")


@dataclass
class VerificationReport:
    unsupported: List[Claim] = field(default_factory=list)
    checked: Dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.unsupported

    def by_kind(self, kind: str) -> List[Claim]:
        return [c for c in self.unsupported if c.kind == kind]


# Column-like identifiers that appear in Data Lab / ObsCore cuts. Kept to
# words that only make sense as catalogue columns / query knobs so that prose
# such as "3 candidates" or "Band 6" is never mistaken for a cut.
_CUT_COLUMNS = (
    "pmra", "pmdec", "pm", "proper motion", "parallax", "parallax_over_error", "ruwe", "class_star", "spread_model",
    "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "bp_rp", "bp-rp", "g_rp", "g - rp", "g-rp",
    "gmag", "rmag", "imag", "zmag", "ymag", "mag_auto_g", "mag_auto_r", "mag_auto_i", "mag_auto_z",
    "g - r", "g-r", "g−r", "r - i", "r-i", "r−i", "i - z", "i-z", "g - i", "g-i", "w1 - w2", "w1-w2", "r - z", "r-z",
    "flux_g", "flux_r", "flux_z", "dered_mag_g", "dered_mag_r", "dered_mag_z", "type", "extended_class",
    "z", "redshift", "zwarn", "spectype", "desi_target", "s_resolution", "spatial_resolution", "bandwidth",
    "t_exptime", "frequency", "em_min", "em_max", "obs_release_date", "data_rights", "band_list",
    "abs_mag", "absolute magnitude", "m_g", "g_abs", "healpix", "nside", "radius", "limit", "top",
)
_CUT_COLUMNS_RE = "|".join(sorted((re.escape(c) for c in _CUT_COLUMNS), key=len, reverse=True))
_CMP = r"(?:>=|<=|!=|<>|=|>|<|≥|≤|between)"
_NUMBER = r"[-+−]?\d+(?:\.\d+)?"
# "pmra > -4", "class_star > 0.5", "0.3 < bp_rp < 0.9", "z between 0.4 and 0.8",
# "16 < g < 20", "LIMIT 5000", "radius = 2°" -- in prose, bullets or tables.
_CUT_RE = re.compile(
    rf"(?P<lhs>{_NUMBER})\s*(?P<c1><|<=|≤)\s*(?P<col>{_CUT_COLUMNS_RE})\s*(?P<c2><|<=|≤)\s*(?P<rhs>{_NUMBER})"   # 0.3 < bp_rp < 0.9
    rf"|(?P<col2>{_CUT_COLUMNS_RE})\s*(?P<cmp>{_CMP})\s*(?P<val>{_NUMBER})(?:\s*(?:and|–|-|to)\s*(?P<val2>{_NUMBER}))?"  # col > 0.5 / z between a and b
    rf"|(?P<kw>limit|top)\s+(?P<lim>\d{{2,}})",
    re.I,
)
_RADIUS_RE = re.compile(
    r"(?:within|radius(?:\s+of)?|r\s*=)\s*(?:a\s+)?(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>°|deg(?:rees?)?|arcmin(?:utes?)?|′|'|arcsec(?:onds?)?|″|\")",
    re.I,
)

_ARTIFACT_RE = re.compile(
    r"(?:(?:is|are|were|was)\s+)?(?:display|shown|attach|plott|render|generat|present|embedd|included?)\w*\s+"
    r"(?:above|below|here|inline|in\s+the\s+ui|in\s+the\s+card)"
    r"|(?:data\s+cards?|cutouts?|figures?|plots?|images?|diagrams?|maps?|histograms?|thumbnails?|panels?)\s+"
    r"(?:above|below|(?:is|are)\s+(?:shown|displayed|attached|rendered|embedded|included))"
    r"|see\s+the\s+(?:plot|figure|image|cmd|diagram|cutout|map|histogram|data\s+cards?)\s+(?:above|below)",
    re.I,
)

_COUNT_NOUNS = (
    "projects?", "proposals?", "rows?", "sources?", "objects?", "stars?", "galaxies", "galaxy", "observations?",
    "datasets?", "mous", "execution\\s+blocks?", "ebs?", "match(?:es)?", "records?", "entries", "candidates?",
    "detections?", "tables?", "catalogs?", "spectra", "papers?", "publications?", "images?", "cutouts?",
    "epochs?", "points?", "members?", "targets?", "files?", "products?", "lrgs?", "quasars?", "protostars?",
)
_COUNT_RE = re.compile(
    rf"(?<![\w.$€£])(?P<num>\d{{1,3}}(?:[ ,  ]\d{{3}})+|\d+)(?!\s*(?:%|°|deg|arcsec|arcmin|ghz|mhz|mas|mag|kpc|mpc|pc|s\b|sec|min|hours?|h\b|d\b|days?|yr|years?|jy|mjy|k\b|cm|mm|µm|um|nm|å|″|′|:))"
    rf"\s+(?:distinct\s+|unique\s+|public\s+|matching\s+|total\s+|candidate\s+|archive\s+|catalog\s+)?(?P<noun>{'|'.join(_COUNT_NOUNS)})\b",
    re.I,
)
# The number is NOT a result count when it directly follows one of these
# ("Band 6", "Cycle 9", "top 5", "Table 2", "DR3").
_COUNT_CONTEXT_SKIP_RE = re.compile(r"\b(?:band|cycle|top|first|dr|table|figure|fig\.?|step|section|phase|epoch|tier|row|rank|#)\s*[:#]?\s*$", re.I)

_NULL_RESULT_RE = re.compile(
    r"\b(?:there\s+(?:are|is|were|was)\s+no\b|no\s+(?:sources?|objects?|observations?|projects?|matches|data|rows|results|protostars?|galaxies|stars)\s+"
    r"(?:were\s+|was\s+)?(?:found|exist|returned|match|available|satisfy|meet|have|with)\b|none\s+(?:exist|were\s+found|found|matched)\b|"
    r"(?:returned|found|yielded)\s+(?:0|zero|no)\s+(?:\w+\s+){0,2}(?:sources?|objects?|observations?|projects?|matches|results|rows)\b|"
    r"\b(?:0|zero)\s+(?:sources?|observations?|projects?|matches|results)\s+(?:were\s+)?(?:found|returned|exist)\b|"
    r"\bdoes\s+not\s+(?:have|contain)\s+(?:any\s+)?(?:alma|jwst|hst|data|observations?)\b)",
    re.I,
)
_NULL_HEDGE_RE = re.compile(
    r"\b(?:unknown|could\s+not\s+be\s+(?:queried|checked|determined)|not\s+(?:be\s+)?(?:queried|checked|determined|verified)|timed?[\s-]*out|"
    r"time-?out|unavailable|unreachable|outage|budget\s+exhausted|partial|incomplete|cannot\s+confirm|"
    r"may\s+still\s+exist|could\s+not\s+confirm|status\s+unknown|not\s+queried|was\s+skipped|were\s+skipped|"
    r"could\s+not\s+be\s+(?:reached|completed)|did\s+not\s+(?:complete|finish|respond))\b",
    re.I,
)

# Figure nouns in an artifact claim -> words a matching card title/type must
# contain. A claim about a MAP is not satisfied by a histogram card (L11).
_FIGURE_NOUN_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "map": ("map", "density", "footprint", "sky", "healpix"),
    "footprint": ("footprint", "map", "sky", "density"),
    "histogram": ("histogram", "distribution", "n(z)", "hist", "count", "redshift"),
    "cmd": ("cmd", "color-magnitude", "colour-magnitude", "color–magnitude", "colour–magnitude", "magnitude diagram", "hr diagram", "hr"),
    "color-magnitude": ("cmd", "color-magnitude", "colour-magnitude", "color–magnitude", "magnitude"),
    "colour-magnitude": ("cmd", "color-magnitude", "colour-magnitude", "colour–magnitude", "magnitude"),
    "hr diagram": ("hr", "cmd", "color-magnitude", "colour-magnitude", "magnitude"),
    "color-color": ("color-color", "colour-colour", "ccd", "color–color", "colour–colour"),
    "colour-colour": ("color-color", "colour-colour", "ccd", "colour–colour"),
    "cutout": ("cutout", "image", "coadd", "thumbnail", "stamp"),
    "cutouts": ("cutout", "image", "coadd", "thumbnail", "stamp"),
    "image": ("image", "cutout", "coadd", "rgb", "composite", "colour image", "color image", "overlay"),
    "light curve": ("light curve", "lightcurve", "folded", "phase", "period"),
    "light-curve": ("light curve", "lightcurve", "folded", "phase", "period"),
    "lightcurve": ("light curve", "lightcurve", "folded", "phase", "period"),
    "sed": ("sed", "spectral energy", "magnitude vs wavelength", "wavelength"),
    "spectrum": ("spectrum", "spectra"),
    "wedge": ("wedge", "cone", "large-scale", "lss"),
    "cone plot": ("wedge", "cone", "lss"),
    "scatter": ("scatter", "distribution", "sky", "map", "plot"),
    "sky distribution": ("sky", "map", "scatter", "distribution", "position"),
    "table": ("table", "rows", "data"),
    "data card": ("table", "rows", "data"),
}
_FIGURE_NOUN_RE = re.compile(
    r"\b(sky[\s-]*density\s+map|sky\s+distribution|density\s+map|footprint\s+map|footprint|maps?|histograms?|"
    r"colou?r[\s–-]*magnitude\s+diagrams?|colou?r[\s–-]*colou?r\s+diagrams?|cmds?|ccds?|hr\s+diagrams?|"
    r"cutouts?|images?|light[\s-]*curves?|lightcurves?|seds?|spectra|spectrum|wedge\s+plots?|wedges?|cone\s+plots?|"
    r"scatter\s+plots?|scatter|tables?|data\s+cards?)\b",
    re.I,
)


def _noun_key(noun: str) -> str:
    n = _norm(noun).replace("–", "-")
    n = re.sub(r"\s+", " ", n)
    if "density map" in n or n in ("footprint map",):
        return "map"
    if n.endswith("s") and n[:-1] in _FIGURE_NOUN_KEYWORDS:
        n = n[:-1]
    if "magnitude" in n:
        return "cmd"
    if "colour-colour" in n or "color-color" in n or n in ("ccd", "ccds"):
        return "color-color"
    if n in ("light curve", "light-curve", "lightcurve", "light curves", "lightcurves"):
        return "light curve"
    if n.startswith("cone"):
        return "cone plot"
    if n.startswith("wedge"):
        return "wedge"
    if n in ("scatter plot", "scatter plots"):
        return "scatter"
    return n


# Producing tool -> the figure-noun keys its card satisfies (title-independent).
_TOOL_FIGURE_KEYS: Dict[str, Tuple[str, ...]] = {
    "plot_sky_map": ("map", "footprint", "scatter", "sky distribution"),
    "datalab_sky_density_map": ("map", "footprint"),
    "datalab_healpix_density_map": ("map", "footprint"),
    "datalab_density_aggregate": ("map", "footprint"),
    "datalab_density_vetting": ("map", "footprint", "cutout", "image"),
    "datalab_color_magnitude_diagram": ("cmd", "hr diagram"),
    "datalab_selection_diagram": ("cmd", "hr diagram", "color-color", "scatter"),
    "datalab_color_color_diagram": ("color-color",),
    "datalab_image_cutout": ("cutout", "image"),
    "datalab_cutout_grid": ("cutout", "image"),
    "datalab_color_image": ("cutout", "image"),
    "hips_cutout": ("cutout", "image"),
    "hips_rgb_composite": ("cutout", "image"),
    "hips_multiband_panel": ("cutout", "image"),
    "hips_contour_overlay": ("image",),
    "vlass_cutout": ("cutout", "image"),
    "datalab_star_lightcurve": ("light curve",),
    "datalab_period_fold": ("light curve",),
    "plot_space_lightcurve": ("light curve",),
    "ztf_light_curve": ("light curve",),
    "datalab_sed_plot": ("sed",),
    "ned_sed_plot": ("sed",),
    "radio_sed": ("sed",),
    "datalab_lss_wedge": ("wedge", "cone plot", "scatter"),
    "datalab_catalog_scatter": ("scatter", "histogram"),
    "plot_spectrum": ("spectrum",),
    "sparcl_plot_spectrum": ("spectrum",),
    "datalab_target_class_summary": ("histogram", "map", "footprint"),
    "datalab_stream_selection": ("map", "footprint", "scatter", "sky distribution", "cmd"),
    "datalab_satellite_search": ("cmd", "cutout", "image", "map"),
    "archive_overlay": ("image",),
    "overlay_archive_images": ("image",),
}
_POSITIONAL_TITLE_RE = re.compile(r"\b(?:ra|dec|sky|position|footprint|map|on-sky|spatial)\b", re.I)


def _card_matches_noun(noun: str, cards: List[Dict[str, Any]]) -> bool:
    key = _noun_key(noun)
    keywords = _FIGURE_NOUN_KEYWORDS.get(key)
    if not keywords:
        return bool(cards)  # generic noun ("plot", "figure"): any card will do
    for card in cards:
        title = _norm(card.get("title") or "")
        ctype = _norm(card.get("type") or "")
        tool = str(card.get("tool") or "")
        if key in ("table", "data card") and ctype == "data":
            return True
        # "the image attached above" is satisfied by ANY rendered figure card
        # (a density map card is an image too; UI 2026-09-23 L08).
        if key == "image" and ctype in ("image", "plotly"):
            return True
        tool_keys = set(_TOOL_FIGURE_KEYS.get(tool, ()))
        if tool == "datalab_catalog_scatter" and _POSITIONAL_TITLE_RE.search(title):
            tool_keys |= {"map", "footprint", "sky distribution"}
        if key in tool_keys:
            return True
        if title and any(k in title for k in keywords):
            return True
    # Anonymous cards only (no title, no tool): fall back to "some card exists".
    if cards and all(not _norm(c.get("title") or "") and not c.get("tool") for c in cards):
        return True
    return False


# Calibrator scan intents / flags in ObsCore-like rows.
_CALIBRATOR_INTENT_RE = re.compile(r"calibrat|bandpass|phase|flux|pointing|focus|atmosphere|sideband|delay|check", re.I)


def _target_roles(tool_results: Sequence[Any], run_results: Sequence[Any]) -> Tuple[set, set]:
    """(science_targets, calibrator_targets) from any ObsCore-like rows in the
    evidence (``target_name`` with ``science_observation`` / ``scan_intent``)."""
    science: set = set()
    calibrators: set = set()

    def scan_rows(rows: Any) -> None:
        if not isinstance(rows, list):
            return
        for row in rows[:5000]:
            if not isinstance(row, dict):
                continue
            name = str(row.get("target_name") or row.get("target") or "").strip()
            if not name:
                continue
            sci = row.get("science_observation")
            intent = str(row.get("scan_intent") or "")
            is_cal = (sci in ("F", False, "false", "False", 0)) or (
                intent and _CALIBRATOR_INTENT_RE.search(intent) and "target" not in intent.lower()
            )
            is_sci = sci in ("T", True, "true", "True", 1) or ("TARGET" in intent.upper())
            if is_cal and not is_sci:
                calibrators.add(name)
            elif is_sci:
                science.add(name)

    def scan_obj(obj: Any) -> None:
        if isinstance(obj, str):
            obj = _parse_maybe_json(obj)
        if isinstance(obj, dict):
            for k in ("rows", "results", "observations", "data", "records", "items", "preview", "sample_rows", "matches"):
                scan_rows(obj.get(k))
            for v in obj.values():
                if isinstance(v, dict):
                    scan_obj(v)
        elif isinstance(obj, list):
            scan_rows(obj)

    for item in tool_results or ():
        scan_obj(item["output"] if isinstance(item, dict) and isinstance(item.get("output"), str) else item)
    for rr in run_results or ():
        scan_obj(rr)
    return science, calibrators

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_CALIBRATOR_WORDS_RE = re.compile(r"\bcalibrat\w*|\b(?:phase|bandpass|flux|amplitude|polari[sz]ation|check)[\s-]+(?:cal\w*|source)\b|\bnot\s+the\s+(?:science\s+)?target\b", re.I)


_FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def _strip_code_blocks(text: str) -> str:
    """Remove fenced code (``` or ~~~, any length >= 3; a fence closes only on
    the same character with at least the opening length -- CommonMark, guard
    CX-23), indented-by-tab code and inline code spans: cuts inside a code
    sample the answer gives the USER to run are not claims about executed
    queries."""
    out: List[str] = []
    fence: Optional[str] = None
    for line in str(text or "").split("\n"):
        if fence is None:
            m = _FENCE_OPEN_RE.match(line)
            if m:
                fence = m.group(1)
                out.append("")
                continue
            out.append(line)
        else:
            m = re.match(r"^ {0,3}(`{3,}|~{3,})\s*$", line)
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                fence = None
            out.append("")
    t = "\n".join(out)
    t = re.sub(r"(`+)(?!`).+?(?<!`)\1(?!`)", " ", t)  # inline spans of any backtick length
    return t


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _num_variants(token: str) -> List[str]:
    """Spellings of a number to look for in the evidence corpus."""
    n = _clean_number(token)
    out = [token.strip().replace("−", "-")]
    if n is None:
        return out
    if n.is_integer():
        i = int(n)
        out += [str(i), f"{i:,}", f"{i:.1f}", f"{i:.0f}"]
    else:
        out += [repr(n), f"{n:g}", f"{n:.1f}", f"{n:.2f}", f"{n:.3f}"]
        s = f"{n:g}"
        if s.startswith("0."):
            out.append(s[1:])  # ".5"
        if s.startswith("-0."):
            out.append("-" + s[2:])
    return list(dict.fromkeys(v for v in out if v))


def _corpus_has_number(corpus: str, token: str) -> bool:
    for v in _num_variants(token):
        vl = v.lower()
        if re.search(rf"(?<![\w.]){re.escape(vl)}(?![\w])", corpus):
            return True
    return False


def _column_variants(col: str) -> List[str]:
    c = _norm(col)
    out = {c, c.replace(" ", ""), c.replace(" - ", "-"), c.replace("-", " - ")}
    aliases = {
        "proper motion": ["pmra", "pmdec", "pm_total", "sqrt(pmra"],
        "pm": ["pmra", "pmdec", "pm_total", "sqrt(pmra"],
        "redshift": ["z"],
        "absolute magnitude": ["abs_mag", "m_g", "g_abs", "+ 5*log10", "+5*log10", "5 * log10"],
        "abs_mag": ["m_g", "g_abs", "5*log10", "5 * log10"],
        "g - r": ["gmag - rmag", "gmag-rmag", "mag_auto_g - mag_auto_r", "g_r", "gr", "(g - r)", "g-r", "mag_g - mag_r",
                  "dered_mag_g - dered_mag_r", "flux_g", "gmag - big.rmag"],
        "r - i": ["rmag - imag", "rmag-imag", "mag_auto_r - mag_auto_i", "r_i", "ri", "(r - i)", "r-i"],
        "i - z": ["imag - zmag", "imag-zmag", "i_z", "(i - z)", "i-z"],
        "g - i": ["gmag - imag", "gmag-imag", "g-i"],
        "r - z": ["rmag - zmag", "r-z"],
        "w1 - w2": ["w1 - w2", "w1-w2", "flux_w1", "w1mag - w2mag"],
        "bp - rp": ["bp_rp"],
        "bp_rp": ["bp_rp", "bp - rp", "bp-rp"],
        "g": ["phot_g_mean_mag", "gmag", "mag_auto_g", "mag_g", "dered_mag_g"],
        "r": ["rmag", "mag_auto_r", "mag_r", "dered_mag_r"],
        "type": ["type", "extended_class", "class_star"],
        "z": ["z", "redshift"],
        "radius": ["q3c_radial_query", "circle(", "radius", "radius_deg", "radius_arcmin", "radius_arcsec", "cone"],
        "limit": ["limit"],
        "top": ["top", "limit", "max_results", "maxrec"],
    }
    for k, vals in aliases.items():
        if c == k or c.replace("−", "-") == k:
            out.update(vals)
    return sorted(out, key=len, reverse=True)


_CMP_DIR = {">": ">", ">=": ">", "≥": ">", "<": "<", "<=": "<", "≤": "<", "=": "=", "!=": "!", "<>": "!"}
_PRED_NUM = r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"


def _predicate_values(corpus: str, start: int, end: int) -> Tuple[List[str], Optional[str]]:
    """The values (and comparator direction) of the ONE predicate whose column
    occurrence spans ``corpus[start:end]`` (guard CX-15: a value may not be
    borrowed from a neighbouring predicate). Understands ``col OP v``,
    ``col between a and b``, ``v OP col`` (leading bound), JSON arguments
    ``"col_min": v`` / ``"col": [a, b]`` and ``col IN (...)``."""
    tail = corpus[end:end + 80]
    values: List[str] = []
    direction: Optional[str] = None
    m = re.match(rf"\s*between\s+({_PRED_NUM})\s+and\s+({_PRED_NUM})", tail)
    if m:
        return [m.group(1), m.group(2)], "between"
    m = re.match(rf"[\w]*\"?\s*:\s*\[([^\]]*)\]", tail)  # "col": [a, b] / "col_range": [a, b]
    if m:
        return re.findall(_PRED_NUM, m.group(1))[:4], "between"
    m = re.match(rf"([\w]*)\"?\s*:\s*\"?({_PRED_NUM})", tail)  # "col_min": v
    if m:
        suffix = m.group(1)
        direction = ">" if re.search(r"(?:^|_)(?:min|gt|lower|lo)(?:_|$)", suffix) else (
            "<" if re.search(r"(?:^|_)(?:max|lt|upper|hi)(?:_|$)", suffix) else None)
        return [m.group(2)], direction
    m = re.match(rf"\s*\)?\s*(>=|<=|!=|<>|=|>|<)\s*\(?\s*({_PRED_NUM})", tail)
    if m:
        values, direction = [m.group(2)], _CMP_DIR.get(m.group(1))
    m_in = re.match(r"\s*in\s*\(([^)]*)\)", tail)
    if m_in:
        values = re.findall(_PRED_NUM, m_in.group(1))[:20]
        direction = "="
    head = corpus[max(0, start - 40):start]
    m = re.search(rf"({_PRED_NUM})\s*(<=|<|>=|>)\s*\(?\s*$", head)  # 0.3 < col
    if m:
        values.append(m.group(1))
    return values, direction


def _cut_supported(col: str, numbers: List[str], summary: TraceSummary, cmp: Optional[str] = None) -> bool:
    """A stated cut is supported when ONE executed predicate on that column
    carries every stated value (same comparator direction where both sides
    state one), or when a data tool's result text states it verbatim."""
    nums = [n for n in numbers if n]
    corpus = _norm(" \n ".join(summary.sql_texts + summary.query_arg_texts))
    want_dir = _CMP_DIR.get((cmp or "").strip().lower()) if cmp else None
    column_values: List[str] = []  # every value of every predicate ON THIS COLUMN
    for v in _column_variants(col):
        # "_" may follow (JSON keys: pm_total_min_mas_yr, z_range, radius_deg);
        # a letter/digit may not ("pm" never matches "pmra").
        for m in re.finditer(rf"(?<![\w]){re.escape(v)}(?![a-z0-9(])", corpus):
            values, direction = _predicate_values(corpus, m.start(), m.end())
            if not values:
                continue
            column_values.extend(values)
            if not all(_corpus_has_number(" ".join(values), n) for n in nums):
                continue
            if want_dir in (">", "<") and direction in (">", "<") and want_dir != direction and len(nums) == 1:
                continue  # "parallax > 5" is not supported by "parallax < 5"
            return True
    # A two-sided range may be written as two predicates on the SAME column
    # ("z >= 0.4 AND z <= 0.8"): values may combine across that column's own
    # predicates, never across columns.
    if len(nums) > 1 and column_values and all(_corpus_has_number(" ".join(column_values), n) for n in nums):
        return True
    # An EQUALITY on a tool parameter the result echoes ("nside = 256" with
    # result field nside=256; UI 2026-09-23 L08) -- never an inequality.
    if len(nums) == 1 and (cmp or "=").strip() in ("=", "=="):
        val = _clean_number(nums[0])
        if val is not None and val.is_integer() and val >= 0:
            keys = {v.replace(" ", "_") for v in _column_variants(col)}
            if any((k, int(val)) in summary.keyed_counts for k in keys):
                return True
    return _stated_in_result_text(nums, summary)


def _stated_in_result_text(nums: List[str], summary: TraceSummary, span: int = 80) -> bool:
    """Every value appears within one short span of one tool-result string
    (a documented band edge, a tool's own note) -- stated from a tool, not
    invented."""
    if not nums:
        return False
    for leaf in summary.result_texts:
        low = _norm(leaf)
        if len(low) < 2:
            continue
        spans = []
        for n in nums:
            pos = None
            for var in _num_variants(n):
                mm = re.search(rf"(?<![\w.]){re.escape(var.lower())}(?![\w])", low)
                if mm:
                    pos = mm.start()
                    break
            if pos is None:
                break
            spans.append(pos)
        else:
            if max(spans) - min(spans) <= span:
                return True
    return False


def _noun_stem(noun: str) -> str:
    n = re.sub(r"\s+", " ", noun.lower()).strip()
    for suffix, repl in (("ies", ""), ("xies", "x"), ("ches", "ch"), ("es", "e"), ("s", "")):
        if n.endswith(suffix) and len(n) > len(suffix) + 2:
            return n[: -len(suffix)] + repl if suffix != "ies" else n[:-3]
    return n


def _count_in_named_field(n: int, noun: str, summary: TraceSummary) -> bool:
    """A numeric field whose KEY names the claimed noun carries the number
    ("19 projects" <- alma_projects=19; "252 observations" <- alma_observations)."""
    stem = _noun_stem(noun).replace(" ", "_")
    if len(stem) < 3:
        return False
    return any(v == n and stem in k for k, v in summary.keyed_counts)


def _row_cap_executed(n: int, noun: str, summary: TraceSummary) -> bool:
    """"limited to 5 000 rows" describes the executed row CAP: a rows/records/
    entries count equal to an executed LIMIT / TOP / MAXREC (or a data tool's
    limit argument) is supported -- the only way SQL text may support a count."""
    if _noun_stem(noun) not in ("row", "record", "entr", "entrie"):
        return False
    sql = _norm(" \n ".join(summary.sql_texts))
    if re.search(rf"\b(?:limit|top|maxrec)\s*=?\s*{n}\b", sql):
        return True
    args = _norm(" \n ".join(summary.query_arg_texts))
    return bool(re.search(rf'"(?:limit|max_rows|maxrec|max_results|row_limit|top)"\s*:\s*{n}\b', args))


def _count_stated_in_results(num_tok: str, summary: TraceSummary) -> bool:
    nouns = "|".join(_COUNT_NOUNS)
    variants = [re.escape(v.lower()) for v in _num_variants(num_tok)]
    pat = re.compile(rf"(?<![\w.])(?:{'|'.join(variants)})(?![\w.])\s+(?:[a-z_-]+\s+){{0,2}}(?:{nouns})\b", re.I)
    return any(pat.search(_norm(leaf)) for leaf in summary.result_texts)


def verify_answer(
    answer: str,
    summary: TraceSummary,
    *,
    check_cuts: bool = True,
    check_artifacts: bool = True,
    check_counts: bool = True,
    check_null_results: bool = True,
) -> VerificationReport:
    """Run the deterministic checks and return the unsupported claims."""
    report = VerificationReport()
    text = str(answer or "")
    if not text.strip():
        return report
    prose = _strip_code_blocks(text)
    # Never re-verify our own appended blocks / guard notices.
    prose = re.sub(r"(?ms)^>\s*(?:🔍|⚠️|🔗|✅).*?$", " ", prose)
    prose = re.sub(r"(?ms)^\*\*Verification\*\*.*?(?=\n\n|\Z)", " ", prose)

    # Cuts are policed only when a DATA query ran (SQL / ADQL or a data tool's
    # arguments): a documentation turn's "211-275 GHz" is not a selection (CX-19).
    if check_cuts and (summary.has_sql or summary.query_arg_texts):
        seen = set()
        for m in _CUT_RE.finditer(prose):
            cmp = None
            if m.group("lhs"):
                col, nums = m.group("col"), [m.group("lhs"), m.group("rhs")]
            elif m.group("col2"):
                col, nums, cmp = m.group("col2"), [m.group("val"), m.group("val2") or ""], m.group("cmp")
            else:
                col, nums = m.group("kw"), [m.group("lim")]
            phrase = _norm(m.group(0))
            if phrase in seen:
                continue
            seen.add(phrase)
            report.checked["cut"] = report.checked.get("cut", 0) + 1
            if not _cut_supported(col, [n for n in nums if n], summary, cmp=cmp):
                report.unsupported.append(Claim("cut", m.group(0).strip(), "this cut does not appear in any executed query or tool argument"))
        for m in _RADIUS_RE.finditer(prose):
            val, unit = m.group("val"), m.group("unit").lower()
            phrase = _norm(m.group(0))
            if phrase in seen:
                continue
            seen.add(phrase)
            report.checked["cut"] = report.checked.get("cut", 0) + 1
            corpus = _norm(" \n ".join(summary.sql_texts + summary.query_arg_texts))
            n = _clean_number(val) or 0.0
            candidates = [val]
            if unit.startswith(("arcmin", "′", "'")):
                candidates += [f"{n / 60:g}", f"{n / 60:.4f}", f"{n / 60:.6f}", f"{n / 60:.7f}", f"{n:g}"]
            elif unit.startswith(("arcsec", "″", '"')):
                candidates += [f"{n / 3600:g}", f"{n / 3600:.6f}", f"{n / 3600:.8f}", f"{n:g}"]
            else:
                candidates += [f"{n:g}", f"{n:.1f}"]
            if not any(_corpus_has_number(corpus, c) for c in candidates):
                report.unsupported.append(Claim("cut", m.group(0).strip(), "no executed query or tool argument uses this search radius"))

    if check_artifacts:
        seen_a = set()
        for sentence in _sentences(prose):
            m = _ARTIFACT_RE.search(sentence)
            if not m:
                continue
            report.checked["artifact"] = report.checked.get("artifact", 0) + 1
            phrase = m.group(0).strip()
            if not summary.has_cards:
                if _norm(phrase) in seen_a:
                    continue
                seen_a.add(_norm(phrase))
                report.unsupported.append(Claim("artifact", phrase, "no plot, image or data card was attached this turn"))
                continue
            # A card exists -- but is it the one the sentence talks about? A
            # "sky-density map ... shown below" is not satisfied by a histogram
            # card (L11). Only specific figure nouns are checked this way.
            # The claim's SUBJECT = figure nouns up to the end of the claim
            # phrase ("the CMD and sky-density map are shown above"); every one
            # must have a matching card (CX-22). Nouns after it ("... cutouts
            # below show peaks from the density map") are not claimed.
            nouns = [n.group(0) for n in _FIGURE_NOUN_RE.finditer(sentence) if n.start() < m.end()]
            specific = [n for n in nouns if _noun_key(n) in _FIGURE_NOUN_KEYWORDS]
            missing = [n for n in specific if not _card_matches_noun(n, summary.cards)]
            if missing:
                specific = missing
                key = _norm(sentence)
                if key in seen_a:
                    continue
                seen_a.add(key)
                report.unsupported.append(Claim(
                    "artifact", sentence[:160],
                    f"no attached card matches '{specific[0]}' (cards this turn: "
                    + ", ".join(c.get("title") or c.get("type") or "?" for c in summary.cards[:4]) + ")",
                ))

    if check_counts and (summary.counts or summary.numbers or summary.has_sql):
        seen_c = set()
        for m in _COUNT_RE.finditer(prose):
            start = max(0, m.start() - 40)
            context = prose[start:m.end() + 20]
            if _COUNT_CONTEXT_SKIP_RE.search(prose[start:m.start()]):
                continue
            num_tok = m.group("num")
            n = _clean_number(num_tok)
            if n is None:
                continue
            noun = m.group("noun").lower()
            key = (int(n), noun)
            if key in seen_c:
                continue
            seen_c.add(key)
            report.checked["count"] = report.checked.get("count", 0) + 1
            # A count is supported only by a COUNT-valued field (rowcount,
            # n_*, *_count, rows returned ...), by a field whose key names the
            # claimed noun (alma_projects=19 for "19 projects"), or by a tool's
            # own text stating that number next to a noun -- never by an arbitrary numeric leaf
            # (a dec of 30 is not "30 projects") or by SQL/argument text
            # (CX-16). 0 and 1 are checked too (CX-17).
            supported = (int(n) in summary.counts or _count_in_named_field(int(n), noun, summary)
                         or _count_stated_in_results(num_tok, summary) or _row_cap_executed(int(n), noun, summary))
            if not supported:
                report.unsupported.append(Claim("count", m.group(0).strip(), "this number does not appear in any tool result"))

    if check_null_results and summary.timed_out_phases:
        for sentence in _sentences(prose):
            s_norm = sentence.replace("‑", "-").replace("‐", "-")
            if _NULL_RESULT_RE.search(s_norm) and not _NULL_HEDGE_RE.search(s_norm):
                # A null result about an archive whose phase did NOT time out
                # ("No MAST sources were found" while only ALMA timed out) is a
                # real result (CX-20). Unattributable timeouts stay conservative.
                named = _archives_in(s_norm)
                if named and summary.timed_out_archives and not (named & summary.timed_out_archives):
                    continue
                report.checked["null_result"] = report.checked.get("null_result", 0) + 1
                report.unsupported.append(Claim(
                    "null_result", sentence[:200],
                    "an archive phase timed out or was skipped this turn, so the result is unknown, not null",
                ))

    # (e) a calibrator named as the observed target (D10: J0238+1636, a phase
    # calibrator, presented as the "Solar active region").
    if summary.calibrator_targets:
        for name in sorted(summary.calibrator_targets, key=len, reverse=True):
            if len(name) < 4 or name in summary.science_targets:
                continue
            # A sentence-final period is not part of the name (J0238+1636.).
            name_re = re.compile(rf"(?<![\w+.-]){re.escape(name)}(?![\w+-]|\.\w)")
            mentions = [snt for snt in _sentences(prose) if name_re.search(snt)]
            # Correctly described as a calibrator (CX-21): not a violation.
            if mentions and all(_CALIBRATOR_WORDS_RE.search(snt) for snt in mentions):
                continue
            if mentions:
                report.checked["calibrator"] = report.checked.get("calibrator", 0) + 1
                report.unsupported.append(Claim(
                    "calibrator", name,
                    "this target is a calibrator in the returned rows (science_observation='F' / calibration scan intent), not the science target",
                ))

    # (f) the unfiltered project count reported where the tool gave a
    # matching count (D11: n_projects=30 in the window, unique_projects=1
    # matched all three arrays).
    if summary.count_pairs:
        for total, matching in summary.count_pairs:
            if total == matching:
                continue
            if re.search(rf"(?<![\d.])(?:≈|~|about\s+|roughly\s+)?{total}\s+(?:distinct\s+|unique\s+)?projects?\b", prose, re.I) and not re.search(
                rf"(?<![\d.]){matching}\s+(?:distinct\s+|unique\s+|matching\s+)?projects?\b", prose, re.I
            ):
                report.checked["count_pair"] = report.checked.get("count_pair", 0) + 1
                report.unsupported.append(Claim(
                    "count", f"{total} projects",
                    f"{total} is the number of projects in the searched window; the tool's count of projects MATCHING the criteria is {matching} (unique_projects)",
                ))
    return report


def format_verification_block(report: VerificationReport, *, max_items: int = 8) -> str:
    """The visible block appended to an answer with unsupported claims."""
    if report.ok:
        return ""
    lines = ["", "", "> 🔍 **Verification** — the following statements are not supported by what the tools actually ran this turn:"]
    kinds = {
        "cut": "selection cut not in the executed query",
        "artifact": "figure/card claimed but none was attached",
        "count": "number not found in any tool result",
        "null_result": "null result stated while a phase timed out or was skipped",
    }
    for claim in report.unsupported[:max_items]:
        text = claim.text.replace("\n", " ").strip()
        if len(text) > 140:
            text = text[:137] + "…"
        lines.append(f"> - {kinds.get(claim.kind, claim.kind)}: “{text}”")
    if len(report.unsupported) > max_items:
        lines.append(f"> - … and {len(report.unsupported) - max_items} more")
    lines.append(">")  # ends the list so the closing sentence is its own paragraph
    lines.append("> Treat these as unverified; the Show-query panels above show exactly what was executed.")
    return "\n".join(lines)
