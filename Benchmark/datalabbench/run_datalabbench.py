"""
DataLabBench v1 — runner, scorer and reporter
=============================================
Benchmarks Quasar's NOIRLab Astro Data Lab subsystem against the 15-question
DataLabBench v1 set (see dlb_dataset_v1.py) with step-level partial credit.

Usage:
    python Benchmark/datalabbench/run_datalabbench.py                  # full run, local backend
    python Benchmark/datalabbench/run_datalabbench.py --questions DLB-02 DLB-03
    python Benchmark/datalabbench/run_datalabbench.py --api-url https://... --auth-token <jwt>
    python Benchmark/datalabbench/run_datalabbench.py --skip-judge     # deterministic checks only
    python Benchmark/datalabbench/run_datalabbench.py --dry-run
    python Benchmark/datalabbench/run_datalabbench.py --self-test     # offline scorer integrity check
    python Benchmark/datalabbench/run_datalabbench.py --emit-rubric   # regenerate RUBRIC.md from the dataset

Scoring model (summary — full docs in RUBRIC.md):
  * every question has 100 points of checkpoints (auto + judge)
  * auto checkpoints:   points * (#checks passed / #checks)      <- partial credit
  * judge checkpoints:  points * judge_credit in [0, 1]          <- partial credit
  * penalties (global guardrails + per-question) subtract afterwards; floor 0
  * overall = tier-weighted mean of question percentages (see TIER_WEIGHTS)
"""

import os
import sys
import json
import time
import re
import argparse
import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# Windows consoles default to cp1252 which cannot print the Unicode minus
# signs that appear verbatim in the benchmark prompts — force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from dlb_dataset_v1 import (  # noqa: E402
    BENCH_NAME, BENCH_VERSION, TIER_WEIGHTS, TIER_LABELS,
    QUESTIONS, GLOBAL_PENALTIES, validate_dataset,
)

# ---------------------------------------------------------------------------
# Evidence bundle
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    output: str = ""
    ok: bool = True
    sql: str = ""      # structured SQL field recorded by the agent trace


@dataclass
class Evidence:
    """Everything the scorer can look at for one benchmark question.

    ``sql_texts`` holds EXECUTED SQL only (arguments/outputs of successful
    tool calls). SQL that merely appears in the final answer text lives in
    ``response_sql_texts`` and never earns auto credit — otherwise a model
    could paste the reference query in prose without running it.
    """
    response_text: str = ""
    calls: List[ToolCall] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    sql_texts: List[str] = field(default_factory=list)            # executed
    response_sql_texts: List[str] = field(default_factory=list)   # prose only
    errors: List[str] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    trace_source: str = "none"     # tool_trace | fallback | none
    raw_event_counts: Dict[str, int] = field(default_factory=dict)

    # -- derived text blobs (cached) --
    _args_blob: Optional[str] = None
    _ok_args_blob: Optional[str] = None
    _sql_blob: Optional[str] = None

    def args_blob(self) -> str:
        """Arguments of ALL calls (incl. failed) — for intent checks."""
        if self._args_blob is None:
            self._args_blob = self._serialize_calls(self.calls)
        return self._args_blob

    def ok_args_blob(self) -> str:
        """Arguments of successful calls only — for execution-credit checks.

        Narrative fields (``reason``, plot ``title``, ...) are scrubbed:
        mentioning a cut in ``datalab_sql_query.reason`` must never count as
        having executed it."""
        if self._ok_args_blob is None:
            self._ok_args_blob = self._serialize_calls(
                [c for c in self.calls if c.ok], scrub_narrative=True)
        return self._ok_args_blob

    @staticmethod
    def _serialize_calls(calls: List["ToolCall"], scrub_narrative: bool = False) -> str:
        parts = []
        for c in calls:
            arguments = _scrub_narrative_args(c.arguments) if scrub_narrative else c.arguments
            try:
                parts.append(json.dumps({c.name: arguments}, sort_keys=True, default=str))
            except Exception:
                parts.append(str(arguments))
            # Structured cuts (value_cuts / morphology / color_cut) are also
            # rendered in SQL-like form so rubric patterns written against the
            # reference SQL ("pm > 100", "g - r BETWEEN ...") match tool args.
            parts.extend(_flatten_structured_cuts(arguments))
        return "\n".join(parts)

    def sql_blob(self) -> str:
        if self._sql_blob is None:
            self._sql_blob = "\n---\n".join(self.sql_texts)
        return self._sql_blob

    def trace_blob(self) -> str:
        """Executed evidence only: successful-call args + executed SQL."""
        return "\n".join([self.sql_blob(), self.ok_args_blob()])

    def any_blob(self) -> str:
        return "\n".join([self.sql_blob(), self.args_blob(), self.response_text])


# Argument keys that carry narrative text rather than executed parameters
# (e.g. datalab_sql_query.reason — a required free-text justification — or a
# plot title). Execution-credit blobs (trace_regex) must not match rubric
# patterns against them; intent-tier blobs (args_regex/any_regex) keep them.
_NARRATIVE_ARG_KEYS = frozenset({
    "reason", "title", "subtitle", "caption", "label", "labels",
    "description", "notes", "comment", "comments", "explanation",
    "summary", "rationale", "justification", "message",
})


def _scrub_narrative_args(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _scrub_narrative_args(x) for k, x in v.items()
                if k not in _NARRATIVE_ARG_KEYS}
    if isinstance(v, (list, tuple)):
        return [_scrub_narrative_args(x) for x in v]
    return v


def _flatten_structured_cuts(args: Any) -> List[str]:
    """Render structured cut dicts as SQL-like strings for pattern matching."""
    out: List[str] = []

    def walk(v):
        if isinstance(v, dict):
            if "column" in v:
                between = v.get("between")
                if isinstance(between, (list, tuple)) and len(between) == 2:
                    out.append(f"{v['column']} BETWEEN {between[0]} AND {between[1]}")
                elif "op" in v or "value" in v:
                    out.append(f"{v['column']} {v.get('op', '=')} {v.get('value', '')}")
            bands = v.get("bands")
            if isinstance(bands, (list, tuple)) and len(bands) == 2:
                out.append(f"{bands[0]} - {bands[1]} BETWEEN {v.get('min', '')} AND {v.get('max', '')}")
            for vv in v.values():
                walk(vv)
        elif isinstance(v, (list, tuple)):
            for vv in v:
                walk(vv)

    walk(args)
    return out


_SQL_FENCE_RE = re.compile(r"```(?:sql|postgres(?:ql)?)\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_QUERY_SUMMARY_RE = re.compile(r'"(?:query_summary|validated_sql|query)"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _clean_sql(sql: str) -> str:
    sql = (sql or "").strip()
    try:
        # tool outputs are JSON-escaped ("\\n" etc.) — unescape best-effort
        if "\\n" in sql or '\\"' in sql:
            sql = json.loads(f'"{sql}"')
    except Exception:
        pass
    return sql


# Only Data Lab tools EXECUTE their sql/query/adql arguments (and echo
# query_summary/validated_sql in their outputs) against TAP. A free-text
# `query` argument on any other successful tool (web_search, ADS, ...) is a
# search string, not run SQL, and must earn no sql_regex/position credit.
_SQL_CAPABLE_TOOL_RE = re.compile(r"^datalab_", re.IGNORECASE)


def extract_executed_sql(calls: List[ToolCall]) -> List[str]:
    """SQL that actually ran: args/outputs/structured field of SUCCESSFUL
    calls to SQL-capable (Data Lab) tools."""
    seen, out = set(), []

    def add(sql: Optional[str]):
        sql = _clean_sql(sql or "")
        key = " ".join(sql.split())[:400]
        if key and key not in seen:
            seen.add(key)
            out.append(sql)

    for c in calls:
        if not c.ok or not _SQL_CAPABLE_TOOL_RE.match(c.name or ""):
            continue
        if c.sql:
            add(c.sql)
        for k in ("sql", "query", "adql"):
            v = c.arguments.get(k)
            if isinstance(v, str):
                add(v)
        if c.output:
            for m in _QUERY_SUMMARY_RE.finditer(c.output):
                add(m.group(1))
    return out


def extract_response_sql(response_text: str) -> List[str]:
    """SQL the model merely wrote in its answer (report/judge context only)."""
    return [_clean_sql(m.group(1)) for m in _SQL_FENCE_RE.finditer(response_text or "")
            if m.group(1).strip()]


# ---------------------------------------------------------------------------
# Numeric helpers (q3c argument extraction, safe arithmetic like 10.0/60.0)
# ---------------------------------------------------------------------------

_NUM_EXPR_RE = re.compile(r"^[\d.\s/*+eE()-]+$")


def safe_number(expr: str) -> Optional[float]:
    expr = (expr or "").strip()
    if not expr or len(expr) > 40 or not _NUM_EXPR_RE.match(expr):
        return None
    try:
        val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - charset-restricted
        return float(val)
    except Exception:
        return None


_Q3C_RADIAL_RE = re.compile(
    r"q3c_radial_query\s*\(\s*[\w.\"]+\s*,\s*[\w.\"]+\s*,"
    r"\s*([^,()]+|\([^)]*\))\s*,\s*([^,()]+|\([^)]*\))\s*,\s*([^,()]+|\([^)]*\))\s*\)",
    re.IGNORECASE,
)


def q3c_cones_from_sql(sql_blob: str) -> List[Dict[str, float]]:
    cones = []
    for m in _Q3C_RADIAL_RE.finditer(sql_blob or ""):
        ra = safe_number(m.group(1))
        dec = safe_number(m.group(2))
        radius = safe_number(m.group(3))
        if ra is not None and dec is not None:
            cones.append({"ra": ra, "dec": dec, "radius": radius})
    return cones


def _as_float(v) -> Optional[float]:
    try:
        if isinstance(v, bool):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def candidate_positions(ev: Evidence, include_failed: bool = False) -> List[Dict[str, float]]:
    """(ra, dec[, radius]) candidates from SUCCESSFUL tool calls + executed SQL."""
    pos = []
    for c in ev.calls:
        if not c.ok and not include_failed:
            continue
        a = c.arguments or {}
        ra, dec = _as_float(a.get("ra")), _as_float(a.get("dec"))
        if ra is not None and dec is not None:
            pos.append({"ra": ra, "dec": dec, "radius": _as_float(a.get("radius_deg"))})
        ra2, dec2 = _as_float(a.get("sia_ra")), _as_float(a.get("sia_dec"))
        if ra2 is not None and dec2 is not None:
            pos.append({"ra": ra2, "dec": dec2, "radius": _as_float(a.get("sia_fov_deg"))})
        for peak in a.get("peaks") or []:
            if isinstance(peak, dict):
                pra, pdec = _as_float(peak.get("ra")), _as_float(peak.get("dec"))
                if pra is not None and pdec is not None:
                    pos.append({"ra": pra, "dec": pdec, "radius": None})
        bbox = [_as_float(a.get(k)) for k in ("ra_min", "ra_max", "dec_min", "dec_max")]
        if all(v is not None for v in bbox):
            pos.append({"ra": (bbox[0] + bbox[1]) / 2, "dec": (bbox[2] + bbox[3]) / 2, "radius": None})
    for cone in q3c_cones_from_sql(ev.sql_blob()):
        pos.append({"ra": cone["ra"], "dec": cone["dec"], "radius": cone.get("radius")})
    return pos


# radius-bearing args with a conversion factor to degrees
_RADIUS_ARGS = {
    "radius_deg": 1.0, "fov_deg": 1.0, "tile_radius_deg": 1.0,
    "sia_fov_deg": 1.0, "width_deg": 1.0, "height_deg": 1.0,
    "radius_arcmin": 1.0 / 60.0, "radius_arcsec": 1.0 / 3600.0,
}


def candidate_radii(ev: Evidence) -> List[float]:
    radii = []
    for c in ev.calls:
        if not c.ok:
            continue
        for k, factor in _RADIUS_ARGS.items():
            v = _as_float((c.arguments or {}).get(k))
            if v is not None:
                radii.append(v * factor)
    for cone in q3c_cones_from_sql(ev.sql_blob()):
        if cone.get("radius") is not None:
            radii.append(cone["radius"])
    return radii


# ---------------------------------------------------------------------------
# Check evaluation engine
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    passed: bool
    note: str


def _match_tools(ev: Evidence, tools: List[str]) -> List[ToolCall]:
    if not tools:
        return list(ev.calls)
    wanted = set(tools)
    return [c for c in ev.calls if c.name in wanted]


def _arg_matches(check: Dict[str, Any], value: Any) -> bool:
    if "equals" in check:
        want = check["equals"]
        if isinstance(want, str) and isinstance(value, str):
            return want.strip().lower() == value.strip().lower()
        return value == want
    if "contains" in check:
        return isinstance(value, str) and str(check["contains"]).lower() in value.lower()
    if "one_of" in check:
        opts = check["one_of"]
        if isinstance(value, str):
            return value.strip().lower() in {str(o).strip().lower() for o in opts}
        return value in opts
    fv = _as_float(value)
    if "approx" in check:
        tol = float(check.get("tol", 0.0))
        return fv is not None and abs(fv - float(check["approx"])) <= tol
    ok = True
    if "min" in check:
        ok = ok and fv is not None and fv >= float(check["min"])
    if "max" in check:
        ok = ok and fv is not None and fv <= float(check["max"])
    if not any(k in check for k in ("equals", "contains", "one_of", "approx", "min", "max")):
        return value is not None  # arg merely present
    return ok


_ROWSCAN_FROM_RE = re.compile(r"\bFROM\s+([\w]+\.[\w]+)", re.IGNORECASE)


def _sql_is_rowlevel(sql: str) -> bool:
    s = sql.upper()
    if "GROUP BY" in s or re.search(r"\bCOUNT\s*\(", s):
        return False
    if "TAP_SCHEMA." in s:
        return False  # metadata browsing, not a catalog scan
    return bool(_ROWSCAN_FROM_RE.search(sql))


def _sql_unbounded_rowscan(sql: str) -> bool:
    """Row-level catalog pull with no q3c bound and no key equality.

    Per guardrail #1 a LIMIT alone does NOT make it safe — with selective
    WHERE cuts Postgres may scan a large fraction of the catalog before
    filling the LIMIT. LIMIT discipline is scored as a positive criterion
    in the rubrics instead.
    """
    if not _sql_is_rowlevel(sql):
        return False
    s = sql.upper()
    if "Q3C_" in s:
        return False
    if re.search(r"\b(ID|OBJID|SPECOBJID|FIELDID|TARGETID)\s*=", s):
        return False
    return True


def _sql_between_rowscan(sql: str) -> bool:
    if not _sql_is_rowlevel(sql):
        return False
    s = sql.upper()
    if "Q3C_" in s:
        return False
    return bool(re.search(r"\b(RA|DEC)\s+BETWEEN\b", s))


def _sql_flat_q3c_join(sql: str) -> bool:
    """The PDF anti-example: a q3c_join in a statement that does not
    MATERIALIZE the reduced small side. Evaluated PER STATEMENT so a good
    CTE elsewhere cannot mask a bad flat join."""
    s = sql.upper()
    return "Q3C_JOIN" in s and "MATERIALIZED" not in s


def eval_check(check: Dict[str, Any], ev: Evidence) -> CheckResult:
    kind = check.get("kind")

    if kind == "any":
        subs = [eval_check(c, ev) for c in check.get("of", [])]
        hit = next((s for s in subs if s.passed), None)
        return CheckResult(hit is not None, hit.note if hit else "no alternative matched")

    if kind == "all":
        subs = [eval_check(c, ev) for c in check.get("of", [])]
        bad = next((s for s in subs if not s.passed), None)
        return CheckResult(bad is None, "all conditions met" if bad is None else f"failed: {bad.note}")

    if kind == "not":
        sub = eval_check(check["of"], ev)
        return CheckResult(not sub.passed, f"negated({sub.note})")

    if kind == "tool_called":
        hits = _match_tools(ev, check.get("tools", []))
        names = sorted({c.name for c in hits})
        return CheckResult(bool(hits), f"called: {', '.join(names)}" if hits else
                           f"none of {check.get('tools')} called")

    if kind == "tool_ok":
        hits = [c for c in _match_tools(ev, check.get("tools", [])) if c.ok]
        names = sorted({c.name for c in hits})
        return CheckResult(bool(hits), f"succeeded: {', '.join(names)}" if hits else
                           f"no successful call among {check.get('tools')}")

    if kind == "tool_arg":
        # Execution checkpoints must not be satisfiable by FAILED calls; a
        # decision checkpoint (tool may legitimately fail, e.g. an SIA
        # coverage gap) opts in with include_failed: true.
        include_failed = bool(check.get("include_failed"))
        for c in _match_tools(ev, check.get("tools", [])):
            if not c.ok and not include_failed:
                continue
            if check["arg"] in (c.arguments or {}) and _arg_matches(check, c.arguments[check["arg"]]):
                return CheckResult(True, f"{c.name}.{check['arg']}={c.arguments[check['arg']]!r}")
        return CheckResult(False, f"no {'call' if include_failed else 'successful call'} "
                                  f"with matching arg {check['arg']!r}")

    if kind == "position_near":
        tol = float(check.get("tol_deg", 0.1))
        want_ra, want_dec = float(check["ra"]), float(check["dec"])
        import math
        for p in candidate_positions(ev, include_failed=bool(check.get("include_failed"))):
            dra = abs(p["ra"] - want_ra) * math.cos(math.radians(want_dec))
            ddec = abs(p["dec"] - want_dec)
            if dra <= tol and ddec <= tol:
                return CheckResult(True, f"position ({p['ra']:.4f}, {p['dec']:.4f}) within {tol} deg")
        return CheckResult(False, f"no tool/SQL position within {tol} deg of ({want_ra}, {want_dec})")

    if kind == "radius_near":
        tol = float(check.get("tol", 0.01))
        want = float(check["value"])
        for r in candidate_radii(ev):
            if abs(r - want) <= tol:
                return CheckResult(True, f"radius {r:.6g} ~ {want:.6g}")
        return CheckResult(False, f"no radius within {tol} of {want}")

    if kind in ("sql_regex", "args_regex", "text_regex", "any_regex", "trace_regex"):
        blob = {"sql_regex": ev.sql_blob(), "args_regex": ev.args_blob(),
                "text_regex": ev.response_text, "any_regex": ev.any_blob(),
                "trace_regex": ev.trace_blob()}[kind]
        m = re.search(check["pattern"], blob or "", re.IGNORECASE)
        return CheckResult(bool(m), f"matched {m.group(0)[:60]!r}" if m else
                           f"pattern {check['pattern']!r} not found in {kind[:-6]}")

    if kind == "image_emitted":
        n = len(ev.images)
        return CheckResult(n > 0, f"{n} image(s) emitted" if n else "no image produced")

    if kind == "image_count":
        n = len(ev.images)
        need = int(check.get("min", 1))
        return CheckResult(n >= need, f"{n} image(s) emitted (need >= {need})")

    if kind == "sql_unbounded_rowscan":
        for sql in ev.sql_texts:
            if _sql_unbounded_rowscan(sql):
                return CheckResult(True, f"unbounded row scan: {' '.join(sql.split())[:100]!r}")
        return CheckResult(False, "no unbounded row-level scan found")

    if kind == "sql_between_rowscan":
        for sql in ev.sql_texts:
            if _sql_between_rowscan(sql):
                return CheckResult(True, f"BETWEEN-box row scan: {' '.join(sql.split())[:100]!r}")
        return CheckResult(False, "no BETWEEN-box row scan found")

    if kind == "sql_flat_q3c_join":
        for sql in ev.sql_texts:
            if _sql_flat_q3c_join(sql):
                return CheckResult(True, f"flat q3c_join: {' '.join(sql.split())[:100]!r}")
        return CheckResult(False, "no flat q3c_join found")

    return CheckResult(False, f"unknown check kind {kind!r}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class CheckpointScore:
    id: str
    type: str
    desc: str
    points: int
    credit: Optional[float]        # 0..1; None = not judged (e.g. --skip-judge)
    earned: Optional[float]
    detail: List[str] = field(default_factory=list)
    judge_reason: str = ""


@dataclass
class PenaltyHit:
    id: str
    desc: str
    points: int
    note: str


@dataclass
class QuestionResult:
    id: str
    tier: int
    title: str
    prompt: str
    response_time_s: float = 0.0
    error: Optional[str] = None
    judge_error: Optional[str] = None
    trace_source: str = "none"
    checkpoints: List[CheckpointScore] = field(default_factory=list)
    penalties: List[PenaltyHit] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    n_tool_calls: int = 0
    n_images: int = 0

    # ---- derived ----
    @property
    def auto_max(self) -> float:
        return sum(c.points for c in self.checkpoints if c.type == "auto")

    @property
    def auto_earned(self) -> float:
        return sum(c.earned or 0.0 for c in self.checkpoints if c.type == "auto")

    @property
    def judged(self) -> bool:
        return all(c.credit is not None for c in self.checkpoints if c.type == "judge") \
            and self.judge_error is None

    @property
    def penalty_points(self) -> float:
        return sum(p.points for p in self.penalties)

    @property
    def total_earned(self) -> Optional[float]:
        if not self.judged:
            return None
        raw = sum(c.earned or 0.0 for c in self.checkpoints)
        return max(0.0, raw - self.penalty_points)

    @property
    def percentage(self) -> Optional[float]:
        e = self.total_earned
        return None if e is None else e  # max is always 100

    @property
    def auto_percentage(self) -> float:
        if self.auto_max == 0:
            return 0.0
        return max(0.0, (self.auto_earned - self.penalty_points)) / self.auto_max * 100.0

    @property
    def grade(self) -> str:
        p = self.percentage
        if p is None:
            return "?"
        if p >= 97: return "A+"
        if p >= 90: return "A"
        if p >= 75: return "B"
        if p >= 60: return "C"
        if p >= 40: return "D"
        return "F"


def score_auto_checkpoints(question: Dict[str, Any], ev: Evidence) -> List[CheckpointScore]:
    out = []
    for cp in question["checkpoints"]:
        if cp["type"] != "auto":
            out.append(CheckpointScore(cp["id"], "judge", cp["desc"], cp["points"],
                                       credit=None, earned=None))
            continue
        results = [eval_check(chk, ev) for chk in cp["checks"]]
        passed = sum(1 for r in results if r.passed)
        credit = passed / len(results) if results else 0.0
        out.append(CheckpointScore(
            cp["id"], "auto", cp["desc"], cp["points"],
            credit=credit, earned=round(cp["points"] * credit, 2),
            detail=[("PASS " if r.passed else "MISS ") + r.note for r in results],
        ))
    return out


def apply_penalties(question: Dict[str, Any], ev: Evidence) -> List[PenaltyHit]:
    hits = []
    disabled = set(question.get("disable_penalties", []))
    for pen in list(GLOBAL_PENALTIES) + list(question.get("penalties", [])):
        if pen["id"] in disabled:
            continue
        gate = pen.get("only_if_text")
        if gate and not re.search(gate, ev.response_text or "", re.IGNORECASE):
            continue
        res = eval_check(pen["detect"], ev)
        if res.passed:
            hits.append(PenaltyHit(pen["id"], pen["desc"], pen["points"], res.note))
    return hits


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are the scoring judge for DataLabBench, a benchmark that evaluates an
AI astronomy assistant (Quasar) on NOIRLab Astro Data Lab catalog-science tasks
(TAP/ADQL queries, q3c spatial functions, SIA image cutouts, plots).

You receive: the benchmark question, the ground-truth solution sketch, the
assistant's final answer, and a trace of the tool calls it actually made
(with arguments and truncated outputs). Score ONLY the listed judge
checkpoints. For each, return a fractional credit between 0.0 and 1.0
following the checkpoint's guidance. Use the tool trace as the source of
truth: claims in the answer that are not backed by a tool call or output
must NOT earn credit (fabrication earns 0 for the affected checkpoint).
Partial credit is expected — score each element of a checkpoint's guidance
proportionally.

Return STRICT JSON:
{"checkpoints": {"<id>": {"credit": <0.0-1.0>, "reason": "<one short sentence>"}, ...}}
"""


def extract_json_object(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Judge returned empty output")
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object in judge output: {cleaned[:200]!r}")
    parsed = json.loads(cleaned[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Judge output JSON is not an object")
    return parsed


def summarize_trace_for_judge(ev: Evidence, max_calls: int = 40) -> str:
    lines = []
    for c in ev.calls[:max_calls]:
        args = json.dumps(c.arguments, default=str)
        if len(args) > 400:
            args = args[:400] + "…"
        out = " ".join((c.output or "").split())
        if len(out) > 350:
            out = out[:350] + "…"
        lines.append(f"- {c.name}({args}) -> {'OK' if c.ok else 'ERROR'} :: {out}")
    if len(ev.calls) > max_calls:
        lines.append(f"... {len(ev.calls) - max_calls} more calls omitted")
    lines.append(f"[images emitted: {len(ev.images)}]")
    if ev.errors:
        lines.append(f"[stream errors: {ev.errors[:3]}]")
    return "\n".join(lines) if lines else "(no tool calls captured)"


class LLMJudge:
    def __init__(self, model: str):
        self.model = model

    def judge(self, question: Dict[str, Any], ev: Evidence,
              checkpoints: List[Dict[str, Any]], max_attempts: int = 3) -> Dict[str, Dict]:
        from core.llm_client import LLMClient
        client = LLMClient(model=self.model)

        cp_lines = []
        for cp in checkpoints:
            cp_lines.append(f"- {cp['id']} ({cp['points']} pts): {cp['desc']}\n  Guidance: {cp['guidance']}")
        response_text = ev.response_text or "[NO TEXT RESPONSE]"
        if len(response_text) > 12000:
            response_text = response_text[:12000] + "\n…[truncated]"

        user_prompt = f"""## Benchmark question ({question['id']}, tier {question['tier']})
{question['prompt']}

## Ground-truth solution sketch (from the benchmark source)
Expected actions: {question.get('reference_actions', '(n/a)')}
Reference SQL: {question.get('reference_sql') or '(no SQL reference — metadata/SIA task)'}

## Judge checkpoints to score
{chr(10).join(cp_lines)}

## Assistant's tool-call trace (source of truth)
{summarize_trace_for_judge(ev)}

## Assistant's final answer
{response_text}

Score each checkpoint id with fractional credit per its guidance. Return the JSON now."""

        last_err = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = client.responses.create(
                    model=self.model,
                    instructions=JUDGE_SYSTEM_PROMPT,
                    input=user_prompt,
                    text={"format": {"type": "json_object"}},
                    temperature=0.1,
                )
                raw = extract_json_object(getattr(resp, "output_text", "") or "")
                cps = raw.get("checkpoints", raw)
                normalized = {}
                for cp in checkpoints:
                    entry = cps.get(cp["id"])
                    if entry is None:
                        raise ValueError(f"judge omitted checkpoint {cp['id']}")
                    credit = float(entry.get("credit", 0.0))
                    normalized[cp["id"]] = {
                        "credit": max(0.0, min(1.0, credit)),
                        "reason": str(entry.get("reason", "")).strip(),
                    }
                return normalized
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < max_attempts:
                    wait = 2 ** attempt
                    print(f"\n    [WARN] judge attempt {attempt}/{max_attempts} failed: {e}; retry in {wait}s",
                          flush=True)
                    time.sleep(wait)
        raise RuntimeError(f"Judge failed after {max_attempts} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Quasar API client (SSE)
# ---------------------------------------------------------------------------

def login_for_token(api_url: str, timeout: int = 30) -> Optional[str]:
    """Try the local test login if no token was supplied."""
    import httpx
    user = os.getenv("QUASAR_BENCH_USER", "1@1")
    pw = os.getenv("QUASAR_BENCH_PASS", "1")
    try:
        r = httpx.post(f"{api_url.rstrip('/')}/api/auth/login",
                       json={"username": user, "password": pw}, timeout=timeout)
        if r.status_code == 200:
            token = r.json().get("token")
            if token:
                print(f"  [auth] logged in as {user}")
                return token
        print(f"  [auth] login as {user} failed ({r.status_code}); continuing anonymously")
    except Exception as e:  # noqa: BLE001
        print(f"  [auth] login skipped ({e}); continuing anonymously")
    return None


def query_quasar(question: str, api_url: str, model: str, timeout: int,
                 auth_token: Optional[str]) -> tuple[Evidence, List[dict], float]:
    """Send one benchmark prompt; return (evidence, raw_events, elapsed_s)."""
    import httpx

    url = f"{api_url.rstrip('/')}/api/chat"
    payload = {"message": question, "model": model}
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    ev = Evidence()
    raw_events: List[dict] = []
    t0 = time.time()

    with httpx.Client(timeout=httpx.Timeout(timeout, connect=30)) as client:
        with client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    ev.response_text += data_str
                    continue
                etype = event.get("type", "?")
                ev.raw_event_counts[etype] = ev.raw_event_counts.get(etype, 0) + 1
                raw_events.append(event)

                if etype == "token":
                    ev.response_text += event.get("content", "")
                elif etype == "tool_trace":
                    for call in event.get("calls", []):
                        ev.calls.append(ToolCall(
                            name=str(call.get("name", "")),
                            arguments=call.get("arguments") or {},
                            output=str(call.get("output", ""))[:4000],
                            ok=bool(call.get("ok", True)),
                            sql=str(call.get("sql", "")),
                        ))
                    ev.trace_source = "tool_trace"
                elif etype == "image":
                    url_val = event.get("url") or event.get("image_url") or ""
                    if url_val:
                        ev.images.append(url_val)
                elif etype == "usage":
                    ev.usage = {
                        "input": int(event.get("inputTokens", 0)),
                        "output": int(event.get("outputTokens", 0)),
                        "total": int(event.get("totalTokens", 0)),
                    }
                elif etype == "error":
                    ev.errors.append(str(event.get("content", ""))[:500])

    elapsed = time.time() - t0

    # Fallback trace when the backend doesn't emit tool_trace yet.
    if ev.trace_source != "tool_trace":
        seen = set()
        for event in raw_events:
            if event.get("type") == "tool_call" and event.get("name"):
                key = (event["name"], json.dumps(event.get("input") or {}, sort_keys=True, default=str))
                if key in seen:
                    continue
                seen.add(key)
                ev.calls.append(ToolCall(name=event["name"],
                                         arguments=event.get("input") or {},
                                         output=str(event.get("output", ""))[:4000]))
            elif event.get("type") == "run_progress" and event.get("tool"):
                if (event["tool"], "{}") not in seen:
                    seen.add((event["tool"], "{}"))
                    ev.calls.append(ToolCall(name=event["tool"]))
        ev.trace_source = "fallback" if ev.calls else "none"

    # Images may also come back inline in markdown / data URIs.
    for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", ev.response_text or ""):
        ev.images.append(m.group(1))
    for c in ev.calls:
        if "image_url" in (c.output or "") and not ev.images:
            m = re.search(r'"image_url"\s*:\s*"([^"]+)"', c.output)
            if m:
                ev.images.append(m.group(1))

    ev.sql_texts = extract_executed_sql(ev.calls)
    ev.response_sql_texts = extract_response_sql(ev.response_text)
    return ev, raw_events, elapsed


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_question(q: Dict[str, Any], args, auth_token: Optional[str],
                 judge: Optional[LLMJudge], out_dir: Path) -> QuestionResult:
    result = QuestionResult(id=q["id"], tier=q["tier"], title=q["title"], prompt=q["prompt"])
    qdir = out_dir / q["id"]
    qdir.mkdir(parents=True, exist_ok=True)

    ev = Evidence()
    raw_events: List[dict] = []
    try:
        print("    -> querying Quasar…", end=" ", flush=True)
        ev, raw_events, elapsed = query_quasar(q["prompt"], args.api_url, args.model,
                                               args.timeout, auth_token)
        result.response_time_s = round(elapsed, 1)
        print(f"[OK] ({elapsed:.0f}s, {len(ev.response_text)} chars, "
              f"{len(ev.calls)} tool calls [{ev.trace_source}], {len(ev.images)} images)")
    except Exception as e:  # noqa: BLE001
        result.error = str(e)
        print(f"[ERR] {e}")

    result.trace_source = ev.trace_source
    result.usage = ev.usage
    result.n_tool_calls = len(ev.calls)
    result.n_images = len(ev.images)

    # ── deterministic scoring ──
    result.checkpoints = score_auto_checkpoints(q, ev)
    result.penalties = apply_penalties(q, ev)

    # ── judge scoring ──
    judge_cps = [cp for cp in q["checkpoints"] if cp["type"] == "judge"]
    if judge and judge_cps:
        try:
            print("    -> judging…", end=" ", flush=True)
            verdicts = judge.judge(q, ev, judge_cps)
            for cs in result.checkpoints:
                if cs.type == "judge":
                    v = verdicts[cs.id]
                    cs.credit = v["credit"]
                    cs.earned = round(cs.points * v["credit"], 2)
                    cs.judge_reason = v["reason"]
            print("[OK]")
        except Exception as e:  # noqa: BLE001
            result.judge_error = str(e)
            print(f"[ERR] {e}")

    # ── persist per-question artifacts ──
    (qdir / "response.md").write_text(ev.response_text or "(empty)", encoding="utf-8")
    with (qdir / "events.jsonl").open("w", encoding="utf-8") as f:
        for event in raw_events:
            f.write(json.dumps(event, default=str) + "\n")
    (qdir / "trace.json").write_text(json.dumps({
        "trace_source": ev.trace_source,
        "calls": [asdict(c) for c in ev.calls],
        "sql_texts": ev.sql_texts,
        "images": ev.images,
        "errors": ev.errors,
    }, indent=2, default=str), encoding="utf-8")
    (qdir / "score.json").write_text(json.dumps({
        "checkpoints": [asdict(c) for c in result.checkpoints],
        "penalties": [asdict(p) for p in result.penalties],
        "auto_percentage": result.auto_percentage,
        "percentage": result.percentage,
    }, indent=2), encoding="utf-8")

    return result


def overall_rollup(results: List[QuestionResult]) -> Dict[str, Any]:
    judged = [r for r in results if r.judged and r.percentage is not None]
    # The headline score is only valid when EVERY selected question was
    # judged — a partial judge outage would otherwise silently bias it.
    if judged and len(judged) == len(results):
        w_full = sum(TIER_WEIGHTS[r.tier] for r in judged)
        full = sum(r.percentage * TIER_WEIGHTS[r.tier] for r in judged) / w_full
    else:
        full = None
    w_auto = sum(TIER_WEIGHTS[r.tier] for r in results)
    auto = (sum(r.auto_percentage * TIER_WEIGHTS[r.tier] for r in results) / w_auto) if w_auto else 0.0
    per_tier = {}
    for tier in sorted({r.tier for r in results}):
        tier_rs = [r for r in results if r.tier == tier]
        tier_judged = [r for r in tier_rs if r.judged and r.percentage is not None]
        per_tier[tier] = {
            "label": TIER_LABELS.get(tier, ""),
            "full_pct": (sum(r.percentage for r in tier_judged) / len(tier_judged)) if tier_judged else None,
            "auto_pct": sum(r.auto_percentage for r in tier_rs) / len(tier_rs),
            "n": len(tier_rs),
        }
    return {"full_pct": full, "auto_pct": auto, "per_tier": per_tier,
            "judged": len(judged), "total": len(results)}


# ---------------------------------------------------------------------------
# Charts + report
# ---------------------------------------------------------------------------

def generate_charts(results: List[QuestionResult], out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [charts] matplotlib not available — skipping charts")
        return

    plt.rcParams.update({
        "figure.facecolor": "#0f172a", "axes.facecolor": "#1e293b",
        "text.color": "#e2e8f0", "axes.labelcolor": "#e2e8f0",
        "xtick.color": "#94a3b8", "ytick.color": "#94a3b8",
        "axes.edgecolor": "#334155", "font.size": 11,
    })
    tier_colors = {1: "#4ade80", 2: "#a3e635", 3: "#facc15", 4: "#fb923c",
                   5: "#f87171", 6: "#e879f9", 7: "#818cf8"}

    # 1. Score by question
    fig, ax = plt.subplots(figsize=(13, 6))
    ids = [r.id for r in results]
    pct = [r.percentage if r.percentage is not None else r.auto_percentage for r in results]
    hatch = ["" if r.judged else "//" for r in results]
    bars = ax.bar(range(len(ids)), pct, color=[tier_colors[r.tier] for r in results],
                  edgecolor="#475569")
    for b, h in zip(bars, hatch):
        b.set_hatch(h)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)")
    ax.axhline(100, color="#22c55e", ls="--", alpha=0.5)
    ax.set_title(f"{BENCH_NAME} v{BENCH_VERSION} — score by question "
                 f"(hatched = auto-only)", fontweight="bold", pad=14)
    for b, p in zip(bars, pct):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{p:.0f}",
                ha="center", fontsize=8, color="#cbd5e1")
    plt.tight_layout()
    fig.savefig(out_dir / "scores_by_question.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Score by tier
    fig, ax = plt.subplots(figsize=(10, 5))
    tiers = sorted({r.tier for r in results})
    vals = []
    for t in tiers:
        rs = [r for r in results if r.tier == t]
        vals.append(np.mean([r.percentage if r.percentage is not None else r.auto_percentage
                             for r in rs]))
    bars = ax.bar([f"T{t}" for t in tiers], vals, color=[tier_colors[t] for t in tiers],
                  edgecolor="#475569", width=0.6)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Average score (%)")
    ax.set_title("Score by tier (T1 easy → T7 open-ended)", fontweight="bold", pad=14)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / "scores_by_tier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Checkpoint credit heat map
    max_cp = max(len(r.checkpoints) for r in results)
    grid = np.full((len(results), max_cp), np.nan)
    for i, r in enumerate(results):
        for j, c in enumerate(r.checkpoints):
            grid[i, j] = c.credit if c.credit is not None else np.nan
    fig, ax = plt.subplots(figsize=(1.2 * max_cp + 3, 0.5 * len(results) + 2))
    im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(max_cp))
    ax.set_xticklabels([f"C{j+1}" for j in range(max_cp)])
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([r.id for r in results], fontsize=9)
    for i, r in enumerate(results):
        for j, c in enumerate(r.checkpoints):
            if c.credit is not None:
                ax.text(j, i, f"{c.credit:.1f}", ha="center", va="center",
                        fontsize=7, color="#111827")
    fig.colorbar(im, label="checkpoint credit")
    ax.set_title("Checkpoint credit matrix (gray = unjudged)", fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(out_dir / "checkpoint_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [charts] 3 charts saved to {out_dir}/")


def generate_report(results: List[QuestionResult], out_dir: Path, args,
                    rollup: Dict[str, Any]):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full = rollup["full_pct"]
    lines = [
        f"# {BENCH_NAME} v{BENCH_VERSION} — Report",
        "",
        f"**Date:** {now}  ",
        f"**API:** `{args.api_url}`  ",
        f"**Model under test:** `{args.model}`  ",
        f"**Judge model:** `{args.judge_model if not args.skip_judge else '(skipped)'}`  ",
        f"**Questions:** {len(results)} (judged: {rollup['judged']}/{rollup['total']})",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Overall score (tier-weighted)** | **{f'{full:.1f} / 100' if full is not None else 'N/A (judge skipped/failed)'}** |",
        f"| Deterministic (auto-only) score | {rollup['auto_pct']:.1f} / 100 |",
        f"| API errors | {sum(1 for r in results if r.error)} |",
        f"| Judge failures | {sum(1 for r in results if r.judge_error)} |",
        f"| Total tokens (Quasar side) | {sum(r.usage.get('total', 0) for r in results):,} |",
        "",
        "![scores](scores_by_question.png)",
        "![tiers](scores_by_tier.png)",
        "![matrix](checkpoint_matrix.png)",
        "",
        "## Per-tier",
        "",
        "| Tier | Focus | Questions | Full score | Auto score |",
        "|---|---|---|---|---|",
    ]
    for t, info in rollup["per_tier"].items():
        fp = f"{info['full_pct']:.1f}%" if info["full_pct"] is not None else "—"
        lines.append(f"| T{t} | {info['label']} | {info['n']} | {fp} | {info['auto_pct']:.1f}% |")

    # Improvement targets: weighted lost points, worst first.
    # Unjudged judge checkpoints (credit None, e.g. --skip-judge) are NOT
    # losses — they are simply unmeasured this run.
    losses = []
    for r in results:
        for c in r.checkpoints:
            if c.credit is None:
                continue
            lost = c.points * (1 - c.credit) * TIER_WEIGHTS[r.tier]
            if lost > 0.5:
                losses.append((lost, r.id, c))
        for p in r.penalties:
            losses.append((p.points * TIER_WEIGHTS[r.tier], r.id,
                           CheckpointScore(p.id, "penalty", p.desc, -p.points, 0.0, 0.0,
                                           detail=[p.note])))
    losses.sort(key=lambda x: -x[0])
    lines += ["", "## Top improvement targets (weighted lost points)", "",
              "| Weighted loss | Question | Checkpoint | What was missed |", "|---|---|---|---|"]
    for lost, qid, c in losses[:20]:
        what = c.judge_reason or (c.detail[0] if c.detail else c.desc)
        lines.append(f"| {lost:.1f} | {qid} | {c.id} ({c.type}) | {what[:140]} |")

    lines += ["", "---", "", "## Question details", ""]
    for r in results:
        pct = f"{r.percentage:.1f}%" if r.percentage is not None else f"auto {r.auto_percentage:.1f}%"
        lines += [
            f"### {r.id} (T{r.tier}): {r.title} — {pct} [{r.grade}]",
            "",
            f"*{r.prompt}*",
            "",
            f"Response time {r.response_time_s:.0f}s · {r.n_tool_calls} tool calls "
            f"({r.trace_source}) · {r.n_images} images"
            + (f" · **API ERROR:** `{r.error}`" if r.error else "")
            + (f" · **JUDGE ERROR:** `{r.judge_error}`" if r.judge_error else ""),
            "",
            "| Checkpoint | Type | Points | Credit | Earned | Notes |",
            "|---|---|---|---|---|---|",
        ]
        for c in r.checkpoints:
            credit = f"{c.credit:.2f}" if c.credit is not None else "—"
            earned = f"{c.earned:.1f}" if c.earned is not None else "—"
            note = c.judge_reason or ("; ".join(c.detail)[:200] if c.detail else "")
            note = note.replace("|", "\\|")
            lines.append(f"| {c.id}: {c.desc[:70]} | {c.type} | {c.points} | {credit} | {earned} | {note} |")
        if r.penalties:
            lines.append("")
            for p in r.penalties:
                lines.append(f"- **PENALTY {p.id} (−{p.points})**: {p.desc} — {p.note}")
        lines.append("")

    path = out_dir / "DataLabBench_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [report] {path}")


def emit_rubric_md(path: Path):
    """Regenerate the human-readable rubric from the dataset (single source of truth)."""
    lines = [
        f"# {BENCH_NAME} v{BENCH_VERSION} — Scoring Rubric",
        "",
        "> GENERATED from `dlb_dataset_v1.py` by `run_datalabbench.py --emit-rubric`.",
        "> Do not edit by hand — edit the dataset and regenerate.",
        "",
        "## How scoring works",
        "",
        "Every question is worth **100 points**, split across weighted **checkpoints**:",
        "",
        "- **auto** checkpoints are scored deterministically against the captured evidence",
        "  (tool-call trace incl. arguments, extracted SQL, response text, emitted images).",
        "  A checkpoint contains N checks; earned credit = `points × passed/N` → **partial",
        "  credit for partially completed steps**.",
        "- **judge** checkpoints are scored by an LLM judge that returns fractional credit",
        "  in [0, 1] per its guidance, grounded in the tool trace (unsupported claims = 0).",
        "- **Penalties** (guardrail violations) subtract points after summing; floor at 0.",
        "",
        "**Overall bench score** = tier-weighted mean of question percentages:",
        "",
        "| Tier | Weight | Focus |",
        "|---|---|---|",
    ]
    for t, w in TIER_WEIGHTS.items():
        lines.append(f"| T{t} | {w} | {TIER_LABELS[t]} |")
    lines += [
        "",
        "Grades: A+ ≥97, A ≥90, B ≥75, C ≥60, D ≥40, F <40.",
        "",
        "## Global guardrail penalties (every question)",
        "",
        "| ID | Points | Trigger |",
        "|---|---|---|",
    ]
    for p in GLOBAL_PENALTIES:
        lines.append(f"| {p['id']} | −{p['points']} | {p['desc']} |")
    lines += ["", "---", ""]

    for q in QUESTIONS:
        lines += [
            f"## {q['id']} (Tier {q['tier']}, weight {TIER_WEIGHTS[q['tier']]}): {q['title']}",
            "",
            f"**Prompt:** {q['prompt']}",
            "",
            f"**Exercises:** {q['exercises']}",
            "",
            f"**Expected actions:** {q['reference_actions']}",
            "",
        ]
        if q.get("reference_sql"):
            lines += ["**Reference query:**", "", "```sql", q["reference_sql"], "```", ""]
        lines += ["| Checkpoint | Type | Points | Criterion |", "|---|---|---|---|"]
        for cp in q["checkpoints"]:
            crit = cp["desc"]
            if cp["type"] == "judge":
                crit += f" — *{cp['guidance']}*"
            else:
                crit += f" ({len(cp['checks'])} checks, each worth {cp['points']}/{len(cp['checks'])} pts)"
            crit = crit.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {cp['id']} | {cp['type']} | {cp['points']} | {crit} |")
        extra = []
        for pen in q.get("penalties", []):
            extra.append(f"- **{pen['id']} (−{pen['points']})**: {pen['desc']}")
        if q.get("disable_penalties"):
            extra.append(f"- Global penalties disabled here: {', '.join(q['disable_penalties'])} "
                         f"(the reference solution itself uses that pattern legitimately)")
        if extra:
            lines += [""] + extra
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[rubric] wrote {path}")


# ---------------------------------------------------------------------------
# Self-test (offline — no API, no judge LLM)
# ---------------------------------------------------------------------------

def _mk_ev(calls=None, text="", images=None, sqls=None) -> Evidence:
    ev = Evidence(response_text=text, calls=calls or [], images=images or [])
    ev.trace_source = "tool_trace" if ev.calls else "none"
    ev.sql_texts = (sqls or []) + extract_executed_sql(ev.calls)
    ev.response_sql_texts = extract_response_sql(text)
    return ev


def run_self_test() -> int:
    failures = []

    def check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    print("\nDataLabBench self-test (offline)\n" + "=" * 40)

    # 0. dataset validity
    problems = validate_dataset()
    check("dataset validates", not problems, "; ".join(problems[:3]))

    # 1. numeric expression eval (10.0/60.0 radius in SQL)
    check("safe_number('10.0/60.0')", abs((safe_number("10.0/60.0") or 0) - 0.16667) < 1e-3)
    check("safe_number rejects code", safe_number("__import__('os')") is None)

    # 2. q3c cone extraction from SQL
    cones = q3c_cones_from_sql(
        "SELECT COUNT(*) FROM gaia_dr3.gaia_source "
        "WHERE q3c_radial_query(ra, dec, 229.022, -0.112, 10.0/60.0)")
    check("q3c cone parsed", len(cones) == 1 and abs(cones[0]["ra"] - 229.022) < 1e-6
          and abs(cones[0]["radius"] - 0.16667) < 1e-3)

    # 3. full-credit DLB-02 via structured tool
    q02 = next(q for q in QUESTIONS if q["id"] == "DLB-02")
    ev = _mk_ev(calls=[ToolCall(
        name="datalab_cone_count",
        arguments={"catalog": "gaia_dr3", "table": "gaia_source",
                   "ra": 229.022, "dec": -0.112, "radius_deg": 0.16667},
        output=json.dumps({"success": True, "reported_count": 2415,
                           "query_summary": "SELECT COUNT(*) FROM gaia_dr3.gaia_source "
                                            "WHERE q3c_radial_query(ra, dec, 229.022, -0.112, 0.16667)"}),
    )], text="There are 2,415 Gaia DR3 sources within 10 arcminutes of Palomar 5.")
    cps = score_auto_checkpoints(q02, ev)
    auto_earned = sum(c.earned or 0 for c in cps if c.type == "auto")
    check("DLB-02 auto full credit (70)", abs(auto_earned - 70.0) < 1e-6, f"earned={auto_earned}")
    pens = apply_penalties(q02, ev)
    check("DLB-02 no penalties", not pens, str([p.id for p in pens]))

    # 4. partial credit: wrong radius, no server-side count
    ev_partial = _mk_ev(calls=[ToolCall(
        name="datalab_select_catalog_rows",
        arguments={"catalog": "gaia_dr3", "table": "gaia_source",
                   "ra": 229.022, "dec": -0.112, "radius_deg": 0.5, "limit": 500},
        output='{"success": true, "rowcount": 500}',
    )], text="I selected 500 rows near Palomar 5.")
    cps_p = score_auto_checkpoints(q02, ev_partial)
    earned_p = sum(c.earned or 0 for c in cps_p if c.type == "auto")
    # C1 (20) catalog ok, C2 (20) position ok, C3 (15) radius wrong -> 0, C4 (15) no COUNT -> 0
    check("DLB-02 partial credit (40)", abs(earned_p - 40.0) < 1e-6, f"earned={earned_p}")

    # 5. GP-00 fires when numbers are claimed with zero successful datalab calls
    ev_fab = _mk_ev(text="There are 2415 sources within 10 arcmin of Palomar 5.")
    pens_fab = apply_penalties(q02, ev_fab)
    check("GP-00 fabrication penalty fires", any(p.id == "GP-00" for p in pens_fab))

    # 6. GP-02 flat q3c_join penalty + MATERIALIZED exemption
    q09 = next(q for q in QUESTIONS if q["id"] == "DLB-09")
    ev_flat = _mk_ev(
        calls=[ToolCall(name="datalab_sql_query",
                        arguments={"sql": "SELECT n.ra FROM nsc_dr2.object n, gaia_dr3.gaia_source g "
                                          "WHERE q3c_radial_query(n.ra, n.dec, 229.022, -0.112, 0.5) "
                                          "AND q3c_join(n.ra, n.dec, g.ra, g.dec, 1.0/3600.0)",
                                   "expert_ack": True, "reason": "test"},
                        output='{"success": true, "rowcount": 10}')],
        text="Crossmatched 10 stars.")
    pens_flat = apply_penalties(q09, ev_flat)
    check("GP-02 flat-join penalty fires", any(p.id == "GP-02" for p in pens_flat))

    ev_mat = _mk_ev(
        calls=[ToolCall(name="datalab_q3c_crossmatch",
                        arguments={"small_catalog": "gaia_dr3", "big_catalog": "nsc_dr2",
                                   "ra": 229.022, "dec": -0.112, "radius_deg": 0.5},
                        output=json.dumps({"success": True, "query_summary":
                                           "WITH g AS MATERIALIZED (SELECT ra, dec FROM gaia_dr3.gaia_source "
                                           "WHERE q3c_radial_query(ra, dec, 229.022, -0.112, 0.5)) "
                                           "SELECT n.ra FROM g, nsc_dr2.object AS n "
                                           "WHERE q3c_join(g.ra, g.dec, n.ra, n.dec, 1.0/3600.0)"}))],
        text="Crossmatched via the planner-safe builder; 1,234 matches.")
    pens_mat = apply_penalties(q09, ev_mat)
    check("GP-02 exempt with MATERIALIZED", not any(p.id == "GP-02" for p in pens_mat),
          str([p.id for p in pens_mat]))
    cps09 = score_auto_checkpoints(q09, ev_mat)
    c1 = next(c for c in cps09 if c.id == "C1")
    c2 = next(c for c in cps09 if c.id == "C2")
    check("DLB-09 C1 crossmatch route", c1.credit == 1.0, c1.detail[0] if c1.detail else "")
    check("DLB-09 C2 join orientation", c2.credit == 1.0, c2.detail[0] if c2.detail else "")

    # 7. GP-01 unbounded row scan (a LIMIT alone does NOT exempt — guardrail #1)
    check("GP-01 detects unbounded scan",
          _sql_unbounded_rowscan("SELECT ra, dec FROM nsc_dr2.object WHERE gmag < 20"))
    check("GP-01 fires despite LIMIT", _sql_unbounded_rowscan(
        "SELECT ra FROM nsc_dr2.object WHERE gmag < 20 LIMIT 1000000"))
    check("GP-01 spares q3c cone", not _sql_unbounded_rowscan(
        "SELECT ra FROM nsc_dr2.object WHERE q3c_radial_query(ra, dec, 1, 2, 0.5)"))
    check("GP-01 spares aggregates", not _sql_unbounded_rowscan(
        "SELECT ring256, COUNT(*) FROM nsc_dr2.object GROUP BY ring256"))
    check("GP-01 spares tap_schema", not _sql_unbounded_rowscan(
        "SELECT table_name, description FROM tap_schema.tables WHERE schema_name='vhs_dr5'"))
    check("GP-03 detects BETWEEN row box", _sql_between_rowscan(
        "SELECT ra FROM nsc_dr2.object WHERE ra BETWEEN 70 AND 90 AND dec BETWEEN -70 AND -50"))
    check("GP-03 spares q3c", not _sql_between_rowscan(
        "SELECT ra FROM nsc_dr2.object WHERE q3c_radial_query(ra, dec, 1, 2, 0.5) "
        "AND gmag BETWEEN 9 AND 25"))

    # 7b. GP-02 is per-statement: a good MATERIALIZED crossmatch elsewhere must
    # not mask a separate flat q3c_join statement.
    check("flat join detected per-statement", _sql_flat_q3c_join(
        "SELECT n.ra FROM nsc_dr2.object n, gaia_dr3.gaia_source g "
        "WHERE q3c_join(n.ra, n.dec, g.ra, g.dec, 1.0/3600.0)"))
    check("materialized join spared", not _sql_flat_q3c_join(
        "WITH g AS MATERIALIZED (SELECT ra, dec FROM gaia_dr3.gaia_source) "
        "SELECT n.ra FROM g, nsc_dr2.object n WHERE q3c_join(g.ra, g.dec, n.ra, n.dec, 1.0/3600.0)"))
    ev_mixed = _mk_ev(calls=[
        ToolCall(name="datalab_q3c_crossmatch", arguments={},
                 output=json.dumps({"success": True, "query_summary":
                                    "WITH g AS MATERIALIZED (SELECT ra FROM gaia_dr3.gaia_source) "
                                    "SELECT n.ra FROM g, nsc_dr2.object n "
                                    "WHERE q3c_join(g.ra, g.dec, n.ra, n.dec, 1.0/3600.0)"})),
        ToolCall(name="datalab_sql_query",
                 arguments={"sql": "SELECT n.ra FROM nsc_dr2.object n, gaia_dr3.gaia_source g "
                                   "WHERE q3c_join(n.ra, n.dec, g.ra, g.dec, 1.0/3600.0)",
                            "expert_ack": True, "reason": "test"},
                 output='{"success": true}'),
    ], text="done")
    pens_mixed = apply_penalties(q09, ev_mixed)
    check("GP-02 fires on mixed statements", any(p.id == "GP-02" for p in pens_mixed),
          str([p.id for p in pens_mixed]))

    # 7c. Prose SQL earns nothing: reference SQL pasted in the answer with only
    # a schema-listing call must score 0 auto points on DLB-02 (no GP-00 —
    # a Data Lab tool DID succeed, the model just never ran the query).
    ev_prose = _mk_ev(
        calls=[ToolCall(name="datalab_list_catalogs", arguments={},
                        output='{"success": true, "count": 10}')],
        text="Run this:\n```sql\nSELECT COUNT(*) FROM gaia_dr3.gaia_source "
             "WHERE q3c_radial_query(ra, dec, 229.022, -0.112, 10.0/60.0)\n```\n"
             "That returns about 2415 sources.")
    cps_prose = score_auto_checkpoints(q02, ev_prose)
    earned_prose = sum(c.earned or 0 for c in cps_prose if c.type == "auto")
    check("prose-only SQL earns 0 auto", earned_prose == 0.0, f"earned={earned_prose}")
    check("prose SQL captured for report only", len(ev_prose.response_sql_texts) == 1
          and not ev_prose.sql_texts)

    # 7d. Failed calls contribute no positions/radii.
    ev_failed = _mk_ev(calls=[ToolCall(
        name="datalab_cone_count",
        arguments={"catalog": "gaia_dr3", "ra": 229.022, "dec": -0.112, "radius_deg": 0.1667},
        output='{"success": false, "error": "boom"}', ok=False)])
    check("failed call yields no position", not eval_check(
        {"kind": "position_near", "ra": 229.022, "dec": -0.112, "tol_deg": 0.05}, ev_failed).passed)
    check("failed call yields no tool_arg", not eval_check(
        {"kind": "tool_arg", "tools": [], "arg": "catalog", "equals": "gaia_dr3"}, ev_failed).passed)
    check("include_failed opts back in", eval_check(
        {"kind": "tool_arg", "tools": [], "arg": "catalog", "equals": "gaia_dr3",
         "include_failed": True}, ev_failed).passed)
    check("include_failed position opts in", eval_check(
        {"kind": "position_near", "ra": 229.022, "dec": -0.112, "tol_deg": 0.05,
         "include_failed": True}, ev_failed).passed)
    check("radius alias arcmin converts", any(
        abs(r - 0.1667) < 1e-3 for r in candidate_radii(_mk_ev(calls=[ToolCall(
            name="x", arguments={"radius_arcmin": 10.0}, output='{"success": true}')]))))

    # 7e. Structured cuts flatten into SQL-like strings so reference-style
    # rubric patterns match tool arguments too.
    ev_cuts = _mk_ev(calls=[ToolCall(
        name="datalab_density_aggregate",
        arguments={"catalog": "delve_dr3", "table": "coadd_objects",
                   "value_cuts": [{"column": "pm", "op": ">", "value": 100},
                                  {"column": "zwarn", "op": "=", "value": 0}],
                   "morphology": {"column": "ext_coadd", "between": [0, 1]},
                   "color_cut": {"bands": ["g", "r"], "min": -0.5, "max": 0.5}},
        output='{"success": true}')])
    for label, pat in [("value cut", r"pm\s*>\s*\d+"), ("zwarn cut", r"zwarn\s*=\s*0"),
                       ("between cut", r"ext_coadd BETWEEN 0 AND 1"),
                       ("color cut", r"g\s*-\s*r"), ("color bound", r"-\s*0\.5")]:
        check(f"structured {label} matches trace_regex",
              eval_check({"kind": "trace_regex", "pattern": pat}, ev_cuts).passed)

    # 7f. SQL smuggled through a non-SQL tool's `query` argument is NOT
    # executed SQL: no sql_regex credit, no position/radius from it.
    ev_smuggle = _mk_ev(
        calls=[ToolCall(name="web_search",
                        arguments={"query": "SELECT COUNT(*) FROM gaia_dr3.gaia_source "
                                            "WHERE q3c_radial_query(ra, dec, 229.022, -0.112, 0.16667)"},
                        output='{"success": true, "results": 5}')],
        text="There are about 2415 sources.")
    check("non-SQL tool query yields no executed SQL", not ev_smuggle.sql_texts)
    check("smuggled SQL fails sql_regex", not eval_check(
        {"kind": "sql_regex", "pattern": r"gaia_dr3\.gaia_source"}, ev_smuggle).passed)

    # 7g. Narrative fields (reason/title) of successful calls earn no
    # execution credit — but still count for intent-tier args_regex.
    ev_reason = _mk_ev(calls=[ToolCall(
        name="datalab_sql_query",
        arguments={"sql": "SELECT ra, dec FROM nsc_dr2.object "
                          "WHERE q3c_radial_query(ra, dec, 229.022, -0.112, 0.1)",
                   "expert_ack": True,
                   "reason": "will try a class_star morphology cut next"},
        output='{"success": true, "rowcount": 42}')])
    check("narrative reason earns no trace credit", not eval_check(
        {"kind": "trace_regex", "pattern": r"class_star"}, ev_reason).passed)
    check("narrative reason still visible to args_regex", eval_check(
        {"kind": "args_regex", "pattern": r"class_star"}, ev_reason).passed)
    check("executed SQL from same call keeps trace credit", eval_check(
        {"kind": "trace_regex", "pattern": r"q3c_radial_query"}, ev_reason).passed)

    # 8. image_count + position from peaks
    ev_img = _mk_ev(calls=[ToolCall(name="datalab_cutout_grid",
                                    arguments={"peaks": [{"ra": 185.43, "dec": -31.99}],
                                               "fov_deg": 0.05})],
                    images=["a.png", "b.png"])
    check("image_count min 2", eval_check({"kind": "image_count", "min": 2}, ev_img).passed)
    check("position from peaks", eval_check(
        {"kind": "position_near", "ra": 185.41, "dec": -31.98, "tol_deg": 0.2}, ev_img).passed)

    # 9. judge plumbing with a fake judge (partial credit propagates)
    class FakeJudge:
        def judge(self, question, ev, checkpoints):
            return {cp["id"]: {"credit": 0.5, "reason": "half"} for cp in checkpoints}

    qr = QuestionResult(id="DLB-02", tier=1, title="t", prompt="p")
    qr.checkpoints = score_auto_checkpoints(q02, ev)
    verdicts = FakeJudge().judge(q02, ev, [c for c in q02["checkpoints"] if c["type"] == "judge"])
    for cs in qr.checkpoints:
        if cs.type == "judge":
            cs.credit = verdicts[cs.id]["credit"]
            cs.earned = round(cs.points * cs.credit, 2)
    # auto 70 + judge 0.5*(15+15)=15 -> 85
    check("judge partial credit rollup (85)", abs((qr.total_earned or 0) - 85.0) < 1e-6,
          f"total={qr.total_earned}")
    check("grade B for 85", qr.grade == "B", qr.grade)

    # 10. rollup math
    r1 = QuestionResult(id="A", tier=1, title="", prompt="")
    r1.checkpoints = [CheckpointScore("C1", "auto", "", 100, 1.0, 100.0)]
    r2 = QuestionResult(id="B", tier=7, title="", prompt="")
    r2.checkpoints = [CheckpointScore("C1", "auto", "", 100, 0.5, 50.0)]
    ro = overall_rollup([r1, r2])
    expect = (100 * 1.0 + 50 * 2.4) / (1.0 + 2.4)
    check("tier-weighted rollup", abs(ro["full_pct"] - expect) < 1e-6, f"{ro['full_pct']}")

    print("=" * 40)
    if failures:
        print(f"SELF-TEST FAILED: {len(failures)} failure(s): {failures}")
        return 1
    print("SELF-TEST PASSED: scorer, penalties, partial credit and rollup all OK.")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=f"{BENCH_NAME} v{BENCH_VERSION} runner")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--model", default="gpt-4.1", help="Quasar model under test")
    parser.add_argument("--judge-model", default=os.getenv("DLB_JUDGE_MODEL", "gpt-4o"))
    parser.add_argument("--timeout", type=int, default=600,
                        help="Seconds per question (these are long multi-tool runs)")
    parser.add_argument("--questions", nargs="*", default=None,
                        help="Question ids to run, e.g. DLB-02 DLB-09")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--auth-token", default=None,
                        help="Bearer token (default: QUASAR_API_TOKEN env, else test login)")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Deterministic checks only (no LLM judge)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--emit-rubric", action="store_true",
                        help="Regenerate RUBRIC.md from the dataset and exit")
    args = parser.parse_args()

    problems = validate_dataset()
    if problems:
        print("DATASET INVALID:")
        for p in problems:
            print(" -", p)
        sys.exit(2)

    if args.self_test:
        sys.exit(run_self_test())

    if args.emit_rubric:
        emit_rubric_md(HERE / "RUBRIC.md")
        return

    selected = QUESTIONS
    if args.questions:
        wanted = {w.upper() for w in args.questions}
        selected = [q for q in QUESTIONS if q["id"].upper() in wanted]
        if not selected:
            print(f"No questions matched: {args.questions}")
            sys.exit(1)

    if args.dry_run:
        print(f"\n{BENCH_NAME} v{BENCH_VERSION} — dry run ({len(selected)} questions)\n")
        for q in selected:
            n_auto = sum(1 for c in q["checkpoints"] if c["type"] == "auto")
            n_judge = len(q["checkpoints"]) - n_auto
            print(f"  [{q['id']}] T{q['tier']} (w={TIER_WEIGHTS[q['tier']]}) {q['title']}")
            print(f"      {n_auto} auto + {n_judge} judge checkpoints; "
                  f"prompt: {q['prompt'][:90]}…\n")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else \
        HERE / "results" / f"{timestamp}_{args.model.replace('/', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    auth_token = args.auth_token or os.getenv("QUASAR_API_TOKEN") or login_for_token(args.api_url)
    judge = None if args.skip_judge else LLMJudge(args.judge_model)

    print(f"\n{'=' * 64}")
    print(f"  {BENCH_NAME} v{BENCH_VERSION}")
    print(f"  API: {args.api_url} | model: {args.model} | judge: "
          f"{'(skipped)' if args.skip_judge else args.judge_model}")
    print(f"  questions: {len(selected)} | output: {out_dir}")
    print(f"{'=' * 64}\n")

    results: List[QuestionResult] = []
    for i, q in enumerate(selected, 1):
        print(f"  [{i}/{len(selected)}] {q['id']} (T{q['tier']}): {q['title']}")
        results.append(run_question(q, args, auth_token, judge, out_dir))

    rollup = overall_rollup(results)

    (out_dir / "results.json").write_text(json.dumps({
        "bench": BENCH_NAME, "version": BENCH_VERSION,
        "date": timestamp, "api_url": args.api_url, "model": args.model,
        "judge_model": None if args.skip_judge else args.judge_model,
        "rollup": rollup,
        "questions": [{
            **{k: getattr(r, k) for k in ("id", "tier", "title", "response_time_s",
                                          "error", "judge_error", "trace_source",
                                          "n_tool_calls", "n_images", "usage")},
            "percentage": r.percentage,
            "auto_percentage": r.auto_percentage,
            "grade": r.grade,
            "checkpoints": [asdict(c) for c in r.checkpoints],
            "penalties": [asdict(p) for p in r.penalties],
        } for r in results],
    }, indent=2, default=str), encoding="utf-8")

    generate_charts(results, out_dir)
    generate_report(results, out_dir, args, rollup)

    print(f"\n{'=' * 64}")
    full = rollup["full_pct"]
    print(f"  OVERALL: {f'{full:.1f} / 100' if full is not None else 'N/A (unjudged)'}"
          f"   (auto-only: {rollup['auto_pct']:.1f} / 100)")
    for r in results:
        pct = f"{r.percentage:5.1f}%" if r.percentage is not None else f"a{r.auto_percentage:5.1f}%"
        pens = f"  penalties: {', '.join(p.id for p in r.penalties)}" if r.penalties else ""
        print(f"    {r.id}  T{r.tier}  {pct}  [{r.grade}]{pens}")
    print(f"{'=' * 64}")
    print(f"  Full report: {out_dir / 'DataLabBench_report.md'}")


if __name__ == "__main__":
    main()
