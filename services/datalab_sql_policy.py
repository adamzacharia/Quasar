"""Governor for Data Lab SQL emitted by builders or expert raw mode."""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from services import datalab_registry as registry


DEFAULT_ROW_CAP = 500
MAX_ROW_CAP = 5000
# INTO is included so `SELECT ra INTO TEMP ...` (a table-creating side effect) is rejected.
DDL_DML_RE = re.compile(
    r"\b(ALTER|ANALYZE|COPY|CREATE|DELETE|DROP|GRANT|INSERT|INTO|MERGE|REINDEX|REVOKE|TRUNCATE|UPDATE|VACUUM)\b",
    re.IGNORECASE,
)
FORBIDDEN_STORAGE_RE = re.compile(r"\b(MYDB|VOSPACE|STORE|PUT|SAVEAS)\b", re.IGNORECASE)
FROM_JOIN_RE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\b", re.IGNORECASE)
LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)


class DatalabPolicyError(ValueError):
    """Policy violation with an actionable fix hint."""

    def __init__(self, message: str, *, fix_hint: str):
        super().__init__(message)
        self.fix_hint = fix_hint

    def to_dict(self) -> Dict[str, str]:
        return {"error": str(self), "fix_hint": self.fix_hint}


@dataclass
class ValidatedQuery:
    sql: str
    source: str
    meta: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def validate(sql: str, *, source: str = "builder", meta: Optional[Mapping[str, Any]] = None) -> ValidatedQuery:
    """Validate and, when safe, row-cap a Data Lab SQL query."""

    query = _strip_one_semicolon(str(sql or "").strip())
    if not query:
        _raise("Data Lab SQL is empty", "Use a SELECT query built by a Data Lab builder.")
    clean = _scrub_sql(query)
    first = _first_keyword(clean)
    if first not in {"SELECT", "WITH"}:
        _raise("Only SELECT or WITH ... SELECT queries are allowed", "Use a read-only SELECT query.")
    if first == "WITH" and not re.search(r"\bSELECT\b", clean, re.IGNORECASE):
        _raise("WITH queries must contain a SELECT", "End the CTE with a SELECT statement.")
    if _has_multiple_statements(clean):
        _raise("Multiple SQL statements are not allowed", "Submit one SELECT query at a time.")
    if DDL_DML_RE.search(clean):
        _raise("DDL/DML statements are not allowed", "Use read-only SELECT queries only.")
    if FORBIDDEN_STORAGE_RE.search(clean):
        _raise("MyDB, VOSpace, and storage tokens are not allowed in P0", "Use catalog SELECT queries without persistent storage side effects.")

    metadata = dict(meta or {})
    warnings = list(metadata.get("warnings") or [])
    # ADQL habits (TOP n, CONTAINS/POINT/CIRCLE, q3c(...) = 1) become the
    # PostgreSQL the query manager runs, BEFORE column grounding (which would
    # otherwise read `TOP` as an unknown column).
    query, clean = _translate_adql_dialect(query, clean, warnings)
    tables = _tables(clean)
    if not tables:
        _raise("No registered Data Lab table found", "Query a registered catalog.table from the Data Lab registry.")

    _check_column_references(clean, first, tables, warnings)
    query, clean, box_rewritten = _rewrite_radec_box(query, clean, tables, warnings)
    if source != "builder":
        query, clean = _add_nan_guards(query, clean, warnings)
        _warn_nan_unsafe_cuts(query, clean, warnings)
        query, clean = _inject_default_quality_cuts(query, clean, tables, warnings)

    has_radial = bool(re.search(r"\bq3c_radial_query\s*\(", clean, re.IGNORECASE))
    has_poly = bool(re.search(r"\bq3c_poly_query\s*\(", clean, re.IGNORECASE))
    has_q3c_join = bool(re.search(r"\bq3c_join\s*\(", clean, re.IGNORECASE))
    has_box = _has_ra_dec_between(clean)
    box_allowed = _all_boxes_allowed(tables)
    if has_box and not box_allowed and not box_rewritten:
        _raise(
            "This RA/Dec BETWEEN box could not be turned into an index-backed q3c polygon "
            "(supported: one single-table SELECT, both bounds numeric or simple a +/- b "
            f"expressions, no OR/NOT, 0 <= ra <= 360, sides <= {MAX_BOX_SIDE_DEG:g} deg, not "
            "touching a pole)",
            "Write the box as q3c_poly_query(ra, dec, ARRAY[ra1, dec1, ra2, dec2, ra3, dec3, ra4, dec4]) "
            "plus the exact BETWEEN predicates, or use q3c_radial_query for a cone.",
        )
    if has_box and box_allowed:
        warnings.append("Accepted registry-approved RA/Dec BETWEEN box query.")

    if has_q3c_join:
        if not (source == "builder" and metadata.get("builder") == "q3c_crossmatch" and metadata.get("q3c_join_authorized")):
            _raise(
                "q3c_join is only allowed from the structured crossmatch builder",
                "Use datalab_q3c_crossmatch instead of raw q3c_join SQL.",
            )
        # Verify planner-safe small-then-big arg order using the columns the builder
        # recorded in meta, so crossmatches on non-ra/dec tables (e.g. DESI
        # mean_fiber_ra/dec) validate too instead of being falsely rejected.
        sg = re.escape(str(metadata.get("join_small_alias", "g")))
        bg = re.escape(str(metadata.get("join_big_alias", "big")))
        sra = re.escape(str(metadata.get("join_small_ra", "ra")))
        sdec = re.escape(str(metadata.get("join_small_dec", "dec")))
        bra = re.escape(str(metadata.get("join_big_ra", "ra")))
        bdec = re.escape(str(metadata.get("join_big_dec", "dec")))
        join_pat = rf"q3c_join\s*\(\s*{sg}\.{sra}\s*,\s*{sg}\.{sdec}\s*,\s*{bg}\.{bra}\s*,\s*{bg}\.{bdec}\s*,"
        if not re.search(join_pat, clean, re.IGNORECASE):
            _raise(
                "q3c_join must use the planner-safe small-then-big argument order from build_q3c_crossmatch",
                "Build crossmatches with build_q3c_crossmatch so the small side is materialized first.",
            )

    is_aggregate = _is_aggregate(clean)
    aggregate_safe = is_aggregate and any(_qualified(catalog, table) in registry.aggregate_safe_tables() for catalog, table in tables)
    exact_id_bound = bool(source == "builder" and metadata.get("exact_id_bound"))
    # Equality on a registry-listed indexed column (e.g. smash fieldid=169, id='169.429960')
    # is the PDF-canonical bound for field-partitioned tables — accept it from any source.
    indexed_eq = _has_indexed_equality(query, clean, tables)
    if indexed_eq:
        warnings.append(f"Accepted indexed equality bound on {indexed_eq}.")
    spatial_bound = has_radial or has_poly or has_q3c_join or (has_box and box_allowed) or exact_id_bound or bool(indexed_eq)
    if not spatial_bound and not aggregate_safe:
        band = _index_ordered_band(clean, tables)
        if band is not None:
            spatial_bound = True
            warnings.append(
                f"Accepted a {band} BETWEEN band without a cone: it is ordered by the indexed {band} "
                f"column with LIMIT <= {DEFAULT_ROW_CAP}, so the server reads only the first rows "
                "of the band from the index. These are the first rows in that order, not a sample."
            )
    if not spatial_bound and not aggregate_safe:
        if _single_axis_band(clean, tables):
            _raise(
                "An RA-only or Dec-only BETWEEN band spans the whole other axis, so a row-level "
                "read of it is an unbounded scan",
                f"Add ORDER BY <that column> with LIMIT <= {DEFAULT_ROW_CAP} (an index-ordered read), "
                "add the other axis as a BETWEEN bound (the box becomes a q3c polygon automatically), "
                "or ask for SELECT COUNT(*) instead.",
            )
        _raise(
            "Row-level Data Lab queries require a q3c spatial bound, an indexed-column equality "
            "(e.g. smash fieldid = N or id = '...'), or a registry-approved aggregate",
            "Add q3c_radial_query/q3c_poly_query, filter on an indexed id/fieldid column, "
            "use a box_ok table box, or use an aggregate builder.",
        )

    row_level = not is_aggregate
    # Output cap: always cap row-level queries; ALSO cap expert/raw aggregates.
    # A builder aggregate is trusted to size its own LIMIT (a fine HEALPix map can
    # legitimately exceed MAX_ROW_CAP pixels), but a raw expert aggregate such as
    # `SELECT source_id, COUNT(*) FROM gaia_dr3.gaia_source GROUP BY source_id`
    # returns one row per source and must not run uncapped.
    needs_cap = row_level or source != "builder"
    final_sql = query
    if needs_cap:
        limit = _limit_value(clean)
        if limit is None:
            # Self-describing injection (RE-B1): the comment travels with the
            # provenance SQL so the cap can never read as a science choice,
            # and platform_row_cap marks it machine-readably.
            final_sql = f"{query}\nLIMIT {DEFAULT_ROW_CAP} /* platform row cap, not a science cut */"
            warnings.append(
                f"Row cap LIMIT {DEFAULT_ROW_CAP} added by the query governor — platform cost "
                "control, not a science cut. If it truncates the result, tell the user and offer "
                "the uncapped path (async_submit background job or an aggregate builder)."
            )
            metadata.setdefault("row_limit", DEFAULT_ROW_CAP)
            metadata.setdefault("platform_row_cap", DEFAULT_ROW_CAP)
        elif limit > MAX_ROW_CAP:
            _raise(
                f"LIMIT {limit} exceeds the Data Lab row cap {MAX_ROW_CAP}",
                f"Reduce LIMIT to <= {MAX_ROW_CAP} (the platform row cap), or get the full "
                "selection through a builder aggregate or an async_submit background job.",
            )
        else:
            # Record the effective cap so execution layers can detect results
            # that exactly filled it (LIMIT-truncated, storage-order slices).
            metadata.setdefault("row_limit", limit)

    return ValidatedQuery(sql=final_sql, source=source, meta=metadata, warnings=warnings)


# Rows come back in storage order, which is spatially clustered — a result that
# exactly filled its LIMIT is a corner-of-the-field slice, not a sample (live
# P7: the "field 169" map excluded Hydra II; live P9: the "Pal 5" map excluded
# Pal 5 itself).
TRUNCATION_GUIDANCE = (
    "rows are a storage-order, spatially clustered slice of the full selection — NOT a "
    "complete or random sample. Never build a sky-distribution or density map from this "
    "result; use datalab_density_aggregate or datalab_density_vetting (server-side GROUP "
    "BY over EVERY row) instead."
)


def limit_truncation_warning(rowcount: Any, row_limit: Any) -> Optional[str]:
    """Warning text when a result exactly filled its row cap, else None."""
    try:
        limit = int(row_limit)
        count = int(rowcount)
    except (TypeError, ValueError):
        return None
    if limit > 0 and count >= limit:
        return f"Result hit its row cap (LIMIT {limit}): " + TRUNCATION_GUIDANCE
    return None


def _strip_one_semicolon(sql: str) -> str:
    return sql[:-1].rstrip() if sql.endswith(";") else sql


# ── ADQL -> PostgreSQL dialect translation (MANNA evals 2026-10-01, T07/T14/T17) ──
#
# Every governed Data Lab tool sends PostgreSQL to the query manager, but users
# and models write ADQL. Three habits used to fail outright: `SELECT TOP n`
# (the governor then appended LIMIT 500 and Postgres rejected the pair),
# CONTAINS(POINT, CIRCLE) (Data Lab does not translate ADQL geometry) and
# `q3c_radial_query(...) = 1` (boolean = integer). Each is rewritten to the
# form that runs, and the rewrite is disclosed in the warnings. Rewrites need
# offset-stable scrubbing: _scrub_sql shrinks only on escaped quotes (''), so
# raw/clean offsets agree whenever the lengths agree.

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_NUM_EXPR = rf"{_NUM}(?:\s*[-+]\s*{_NUM})*"
_TOP_RE = re.compile(r"^(\s*SELECT\s+(?:(?:DISTINCT|ALL)\s+)?)TOP\s+(\d+)\s+", re.IGNORECASE)
_IDENT = r"[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)?"
_CONTAINS_CIRCLE_RE = re.compile(
    rf"(?P<pre>\b1\s*=\s*)?(?P<kw>\bCONTAINS)\s*\(\s*POINT\s*\(\s*(?:'[^']*'\s*,\s*)?(?P<ra>{_IDENT})\s*,\s*"
    rf"(?P<dec>{_IDENT})\s*\)\s*,\s*CIRCLE\s*\(\s*(?:'[^']*'\s*,\s*)?(?P<a>{_NUM})\s*,\s*(?P<b>{_NUM})\s*,\s*"
    rf"(?P<r>{_NUM})\s*\)\s*\)(?P<post>\s*=\s*1\b)?",
    re.IGNORECASE,
)
_Q3C_CALL_RE = re.compile(r"\bq3c_(?:radial|poly|ellipse)_query\s*\(", re.IGNORECASE)
MAX_BOX_SIDE_DEG = 20.0
MAX_BAND_WIDTH_DEG = 10.0


def _translate_adql_dialect(query: str, clean: str, warnings: List[str]) -> tuple[str, str]:
    query, clean = _translate_top(query, clean, warnings)
    if len(query) == len(clean):
        query, clean = _translate_contains_circle(query, clean, warnings)
    if len(query) == len(clean):
        query, clean = _strip_q3c_equals_one(query, clean, warnings)
    return query, clean


def _translate_top(query: str, clean: str, warnings: List[str]) -> tuple[str, str]:
    """`SELECT [DISTINCT] TOP n ...` -> `SELECT [DISTINCT] ... LIMIT n` (top-level only).

    The text before TOP holds no string literal, so its raw and scrubbed
    offsets always agree. Compound queries are left alone (a trailing LIMIT
    would bind to the whole UNION, not to the first branch)."""
    match = _TOP_RE.match(clean)
    if not match or re.search(r"\b(UNION|INTERSECT|EXCEPT)\b", clean, re.IGNORECASE):
        return query, clean
    n = int(match.group(2))
    body = query[: match.end(1)] + query[match.end():]
    existing = _limit_value(_scrub_sql(body))
    if existing is None:
        body = f"{body.rstrip()}\nLIMIT {n}"
        warnings.append(f"Translated ADQL 'SELECT TOP {n}' to PostgreSQL 'LIMIT {n}' (Data Lab's query manager runs PostgreSQL).")
    else:
        warnings.append(f"Dropped ADQL 'TOP {n}': the query already has LIMIT {existing}, which PostgreSQL uses.")
    return body, _scrub_sql(body)


def _translate_contains_circle(query: str, clean: str, warnings: List[str]) -> tuple[str, str]:
    """ADQL point-in-circle -> q3c_radial_query(ra, dec, ra0, dec0, r), the indexed cone
    Data Lab runs (live: CONTAINS fails with 'function point(...) does not exist')."""
    out: List[str] = []
    pos = 0
    rewritten = 0
    for m in _CONTAINS_CIRCLE_RE.finditer(query):
        # The keyword must be real code, not text inside a literal or comment.
        if clean[m.start("kw"): m.end("kw")].upper() != "CONTAINS" or _contains_compared_to_other(query, m):
            continue
        out.append(query[pos: m.start()])
        out.append(f"q3c_radial_query({m.group('ra')}, {m.group('dec')}, {m.group('a')}, {m.group('b')}, {m.group('r')})")
        pos = m.end()
        rewritten += 1
    if not rewritten:
        return query, clean
    new_query = "".join(out) + query[pos:]
    warnings.append(
        f"Rewrote {rewritten} ADQL CONTAINS(POINT, CIRCLE) cone(s) to q3c_radial_query(ra, dec, ra0, dec0, radius_deg): "
        "Data Lab does not translate ADQL geometry. Same cone, indexed."
    )
    return new_query, _scrub_sql(new_query)


def _strip_q3c_equals_one(query: str, clean: str, warnings: List[str]) -> tuple[str, str]:
    """`q3c_*_query(...) = 1` / `1 = q3c_*_query(...)` -> the bare boolean predicate.
    PostgreSQL rejects boolean = integer; `= 't'`, `= true` and bare forms are left as written."""
    edits: List[tuple[int, int, str]] = []
    for m in _Q3C_CALL_RE.finditer(clean):
        close = _matching_paren(clean, m.end() - 1)
        if close is None:
            continue
        tail = re.match(r"\s*=\s*1\b", clean[close + 1:])
        if tail:
            edits.append((close + 1, close + 1 + tail.end(), ""))
            continue
        head = re.search(r"\b1\s*=\s*$", clean[: m.start()])
        if head:
            edits.append((head.start(), m.start(), ""))
    if not edits:
        return query, clean
    new_query = query
    for start, end, repl in sorted(edits, reverse=True):
        new_query = new_query[:start] + repl + new_query[end:]
    warnings.append("Removed '= 1' after q3c predicate(s): PostgreSQL rejects boolean = integer; q3c functions return a boolean.")
    return new_query, _scrub_sql(new_query)


def _matching_paren(clean: str, open_idx: int) -> Optional[int]:
    depth = 0
    for i in range(open_idx, len(clean)):
        if clean[i] == "(":
            depth += 1
        elif clean[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def _eval_num_expr(expr: str) -> Optional[float]:
    """Value of `a`, `a + b`, `a - b - c` (numeric literals only), else None."""
    text = re.sub(r"\s+", "", expr)
    head = re.match(_NUM, text)
    if not head:
        return None
    value = float(head.group(0))
    rest = text[head.end():]
    for m in re.finditer(rf"([-+])({_NUM})", rest):
        value = value + float(m.group(2)) if m.group(1) == "+" else value - float(m.group(2))
    consumed = "".join(m.group(0) for m in re.finditer(rf"([-+])({_NUM})", rest))
    return value if consumed == rest else None


def _between_bounds(segment: str, column: str) -> List[tuple[float, float, tuple[int, int]]]:
    """Every `[alias.]column BETWEEN <expr> AND <expr>` in segment, evaluated."""
    found = []
    pat = rf"(?<![\w.])(?:[A-Za-z_]\w*\.)?{re.escape(column)}\s+BETWEEN\s+({_NUM_EXPR})\s+AND\s+({_NUM_EXPR})(?![\w.])"
    for m in re.finditer(pat, segment, re.IGNORECASE):
        lo, hi = _eval_num_expr(m.group(1)), _eval_num_expr(m.group(2))
        if lo is None or hi is None:
            return []
        found.append((lo, hi, m.span()))
    return found


def _table_radec(catalog: str, table: str) -> tuple[str, str]:
    try:
        info = registry.describe_table(catalog, table)
        return str(info.get("ra_column") or "ra"), str(info.get("dec_column") or "dec")
    except Exception:  # noqa: BLE001 - unregistered tables already raised in _tables
        return "ra", "dec"


def _simple_single_table(clean: str, tables: List[tuple[str, str]]) -> bool:
    # NOT forms that cannot negate a BETWEEN box (IS NOT NULL, NOT IN, NOT
    # LIKE) are fine; a bare NOT or any OR could put the box inside a negated
    # or alternative branch, where an ANDed polygon would change the result.
    negations = re.sub(r"\bIS\s+NOT\s+NULL\b|\bNOT\s+(?:IN|LIKE|ILIKE)\b", " ", clean, flags=re.IGNORECASE)
    return (
        len(tables) == 1
        and len(re.findall(r"\bSELECT\b", clean, re.IGNORECASE)) == 1
        and not re.search(r"\b(JOIN|UNION|INTERSECT|EXCEPT|OR|NOT)\b", negations, re.IGNORECASE)
    )


def _contains_compared_to_other(text: str, m: "re.Match[str]") -> bool:
    """CONTAINS(...) compared with anything but 1 (e.g. = 0, 'outside the
    circle') has no q3c_radial_query equivalent here: leave it alone."""
    if m.group("post") is None and re.match(r"\s*(?:=|<>|!=|<|>)", text[m.end():]):
        return True
    if m.group("pre") is None and re.search(r"(?:=|<>|!=|<|>)\s*$", text[: m.start()]):
        return True
    return False


def _gc_bulge_deg(dec_deg: float, ra_span_deg: float) -> float:
    """How far (deg) a great-circle polygon edge between two corners at the same
    Dec departs from that constant-Dec line, at its midpoint."""
    import math

    d = math.radians(dec_deg)
    half = math.radians(ra_span_deg) / 2.0
    mid = math.degrees(math.atan(math.tan(d) / math.cos(half)))
    return abs(mid - dec_deg)


def _fmt_deg(x: float) -> str:
    text = f"{x:.10f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def _rewrite_radec_box(
    query: str, clean: str, tables: List[tuple[str, str]], warnings: List[str]
) -> tuple[str, str, bool]:
    """An RA/Dec BETWEEN box on a q3c table -> `q3c_poly_query(...) AND (<original WHERE>)`.

    The original BETWEEN predicates stay, so the result is EXACTLY the box the
    query asked for; the polygon only lets q3c use its index. q3c polygon
    edges are great circles, which bow poleward of a constant-Dec edge, so the
    polygon is padded by twice that bow (plus a hair in RA) to be a strict
    superset of the box. Conservative: one single-table SELECT, no OR/NOT,
    numeric bounds, no RA wrap, sides <= MAX_BOX_SIDE_DEG, away from the poles.
    Live check 2026-10-02: the M87 0.1 deg box counts 1,846 either way.
    """
    if len(query) != len(clean) or not tables or _all_boxes_allowed(tables):
        return query, clean, False
    if not _simple_single_table(clean, tables):
        return query, clean, False
    catalog, table = tables[0]
    ra_col, dec_col = _table_radec(catalog, table)
    where_m = _WHERE_SEGMENT_RE.search(clean)
    if not where_m:
        return query, clean, False
    where = clean[where_m.start(1): where_m.end(1)]
    ra_b, dec_b = _between_bounds(where, ra_col), _between_bounds(where, dec_col)
    if len(ra_b) != 1 or len(dec_b) != 1:
        return query, clean, False
    # No BETWEEN on these columns outside the WHERE (e.g. in a CASE projection).
    for col in (ra_col, dec_col):
        if len(re.findall(rf"(?<![\w.])(?:[A-Za-z_]\w*\.)?{re.escape(col)}\s+BETWEEN\b", clean, re.IGNORECASE)) != 1:
            return query, clean, False
    ra_lo, ra_hi = sorted(ra_b[0][:2])
    dec_lo, dec_hi = sorted(dec_b[0][:2])
    if not (0.0 <= ra_lo < ra_hi <= 360.0 and -90.0 < dec_lo < dec_hi < 90.0):
        return query, clean, False
    if ra_hi - ra_lo > MAX_BOX_SIDE_DEG or dec_hi - dec_lo > MAX_BOX_SIDE_DEG:
        return query, clean, False
    import math

    span = ra_hi - ra_lo
    pad_dec = 2.0 * max(_gc_bulge_deg(dec_lo, span), _gc_bulge_deg(dec_hi, span)) + 1e-6
    p_dec_lo, p_dec_hi = dec_lo - pad_dec, dec_hi + pad_dec
    if p_dec_lo <= -89.9 or p_dec_hi >= 89.9:
        return query, clean, False
    pad_ra = 1e-6 / max(math.cos(math.radians(max(abs(p_dec_lo), abs(p_dec_hi)))), 1e-3)
    p_ra_lo, p_ra_hi = max(0.0, ra_lo - pad_ra), min(360.0, ra_hi + pad_ra)
    corners = [(p_ra_lo, p_dec_lo), (p_ra_hi, p_dec_lo), (p_ra_hi, p_dec_hi), (p_ra_lo, p_dec_hi)]
    array = ", ".join(f"{_fmt_deg(r)}, {_fmt_deg(d)}" for r, d in corners)
    poly = f"q3c_poly_query({ra_col}, {dec_col}, ARRAY[{array}])"
    c0, c1 = where_m.span(1)
    cond = query[c0:c1].strip()
    tail = query[c1:]
    new_query = f"{query[:c0]} {poly} AND ({cond}){(' ' + tail.lstrip()) if tail.strip() else ''}".rstrip()
    warnings.append(
        f"Rewrote the RA/Dec BETWEEN box (ra {_fmt_deg(ra_lo)}..{_fmt_deg(ra_hi)}, dec {_fmt_deg(dec_lo)}..{_fmt_deg(dec_hi)}) "
        "as an index-backed q3c_poly_query AND the original BETWEEN predicates: the result is exactly that box "
        "(the polygon is only the index bound). Note an RA/Dec box is not a cone: RA width in degrees is not "
        "sky width unless scaled by cos(dec)."
    )
    return new_query, _scrub_sql(new_query), True


def _single_axis_band(clean: str, tables: List[tuple[str, str]]) -> bool:
    if len(tables) != 1:
        return False
    ra_col, dec_col = _table_radec(*tables[0])
    where = _where_segment(clean)
    return bool(_between_bounds(where, ra_col) or _between_bounds(where, dec_col))


def _index_ordered_band(clean: str, tables: List[tuple[str, str]]) -> Optional[str]:
    """Column name when a row-level read is an index-ordered slice of one RA (or
    Dec) band: `col BETWEEN a AND b ... ORDER BY col [ASC|DESC] LIMIT k`, k <=
    DEFAULT_ROW_CAP, band <= MAX_BAND_WIDTH_DEG. ra/dec carry b-tree indexes on
    the Data Lab q3c tables (live tap_schema indexed=1 for smash_dr2.object and
    nsc_dr2.object, 2026-10-02), so the server reads only the first k rows of
    the band. Covers the MANNA T07 prompt verbatim."""
    if not _simple_single_table(clean, tables):
        return None
    if re.search(r"\bGROUP\s+BY\b", clean, re.IGNORECASE):
        return None
    limit = _limit_value(clean)
    if limit is None or limit > DEFAULT_ROW_CAP:
        return None
    where = _where_segment(clean)
    for col in _table_radec(*tables[0]):
        bounds = _between_bounds(where, col)
        if len(bounds) != 1:
            continue
        lo, hi = sorted(bounds[0][:2])
        if hi - lo > MAX_BAND_WIDTH_DEG:
            continue
        order = re.search(r"\bORDER\s+BY\s+(?:[A-Za-z_]\w*\.)?([A-Za-z_]\w*)\b", clean, re.IGNORECASE)
        if order and order.group(1).lower() == col.lower():
            return col
    return None


def _scrub_sql(sql: str) -> str:
    chars: List[str] = []
    i = 0
    in_single = False
    in_double = False
    while i < len(sql):
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < len(sql) else ""
        if in_single:
            chars.append(" ")
            if ch == "'" and nxt == "'":
                i += 2
                continue
            if ch == "'":
                in_single = False
            i += 1
            continue
        if in_double:
            chars.append(" ")
            if ch == '"':
                in_double = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            chars.append(" ")
            i += 1
            continue
        if ch == '"':
            in_double = True
            chars.append(" ")
            i += 1
            continue
        if ch == "-" and nxt == "-":
            while i < len(sql) and sql[i] != "\n":
                chars.append(" ")
                i += 1
            continue
        if ch == "/" and nxt == "*":
            chars.extend("  ")
            i += 2
            while i + 1 < len(sql) and not (sql[i] == "*" and sql[i + 1] == "/"):
                chars.append(" ")
                i += 1
            if i + 1 < len(sql):
                chars.extend("  ")
                i += 2
            continue
        chars.append(ch)
        i += 1
    return "".join(chars)


def _first_keyword(clean: str) -> str:
    match = re.search(r"\b([A-Za-z]+)\b", clean)
    return match.group(1).upper() if match else ""


# Bare lower-bound / not-equal cuts on nullable float columns silently admit
# NaN rows (Data Lab stores missing values as NaN; Postgres orders NaN above
# every real number). Builders add the guard themselves; expert/raw SQL gets a
# warning so the model re-runs with `AND col < 'Infinity'` guards. A rewrite is
# deliberately NOT attempted: _scrub_sql is not offset-preserving, so a regex
# rewrite could corrupt queries containing escaped quotes.
NAN_UNSAFE_CUT_RE = re.compile(r"\b([a-z_][\w]*)\s*(?:>=|>|!=|<>)\s*[-+]?[\d.]", re.IGNORECASE)
# Columns that are structurally never NaN (positions, ids, counters, bins).
_NAN_EXEMPT_COLUMNS = {
    "ra", "dec", "ra_bin", "dec_bin", "glon", "glat", "mjd", "fieldid",
    "exptime", "nepochs", "ndet", "nobs", "z_bin", "healpix", "nest4096",
    "ring256", "targetid", "source_id", "objid", "source_count",
}


# A (possibly alias-qualified) column compared to a numeric literal with a
# NaN-unsafe operator — the auto-rewrite target.
NAN_GUARDABLE_RE = re.compile(
    r"\b((?:[a-z_][\w]*\.)?([a-z_][\w]*))\s*(>=|>|!=|<>)\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)


def _add_nan_guards(query: str, clean: str, warnings: List[str]) -> tuple[str, str]:
    """Auto-guard NaN-unsafe numeric cuts in expert/raw SQL.

    Advisory warnings alone demonstrably do not work: on the live P6 re-test
    gpt-oss wrote bare cuts, ignored the warning, and shipped a 5,000-row
    selection that was 97% NaN (4,839/5,000 rows with no astrometry). Rewriting
    is only safe when the regex cannot touch quoted or commented text, so the
    rewrite fires ONLY for SQL with no string literals or comments — anything
    else falls through to _warn_nan_unsafe_cuts.
    """
    if any(tok in query for tok in ("'", '"', "--", "/*")):
        return query, clean

    def _adjacent_arithmetic(start: int, end: int) -> bool:
        """True when the matched comparison is an operand of surrounding
        arithmetic — e.g. the 'b > 0.5' inside 'a - b > 0.5' or the 'a > 0.5'
        inside 'a > 0.5 - b'. Wrapping such a match in a boolean paren produced
        invalid SQL ('numeric - boolean'), 400-ing every expert color/difference
        cut (dl-nan-guard-corrupts-arithmetic-cuts). These fall through to the
        advisory _warn_nan_unsafe_cuts path instead."""
        i = start - 1
        while i >= 0 and query[i].isspace():
            i -= 1
        if i >= 0 and query[i] in "+-*/":
            return True
        j = end
        while j < len(query) and query[j].isspace():
            j += 1
        return j < len(query) and query[j] in "+-*/"

    guarded: List[str] = []

    def _sub(match: "re.Match[str]") -> str:
        full, col, op, num = match.group(1), match.group(2), match.group(3), match.group(4)
        if col.lower() in _NAN_EXEMPT_COLUMNS:
            return match.group(0)
        if _adjacent_arithmetic(match.start(), match.end()):
            return match.group(0)
        # An existing upper bound on the same column already excludes NaN.
        if re.search(rf"\b{re.escape(col)}\s*(?:<=?|BETWEEN)\s", clean, re.IGNORECASE):
            return match.group(0)
        guarded.append(f"{full} {op} {num}")
        op_norm = "!=" if op == "<>" else op
        return f"({full} {op_norm} {num} AND {full} < 'Infinity'::float8)"

    rewritten = NAN_GUARDABLE_RE.sub(_sub, query)
    if not guarded:
        return query, clean
    warnings.append(
        "Auto-added NaN finiteness guards (col < 'Infinity') to: " + "; ".join(guarded) + ". "
        "Data Lab stores missing floats as NaN, which Postgres orders ABOVE every real number, "
        "so bare >/>=/!= cuts would have admitted every missing-value row. This is a data-validity "
        "guardrail added by the query governor, not a science cut."
    )
    return rewritten, _scrub_sql(rewritten)


_CLAUSE_TAIL_RE = re.compile(r"\b(GROUP\s+BY|ORDER\s+BY|LIMIT|OFFSET)\b", re.IGNORECASE)


def _inject_default_quality_cuts(
    query: str, clean: str, tables: List[tuple[str, str]], warnings: List[str]
) -> tuple[str, str]:
    """Apply registry survey-quality defaults to expert/raw SQL.

    The structured tools merge these defaults themselves, but expert SQL relied
    on a prompt rule the model follows only partially (live P11: the DESI
    z-histogram applied zwarn=0 but omitted survey='main' AND main_primary, so
    every low-z bin was inflated 11-27% by sv-survey duplicates). Injection is
    restricted to single-table, single-SELECT queries whose scrub is offset-
    preserving (no comments, no escaped quotes): the WHERE clause is located on
    the literal-blanked text and spliced into the raw text at the same offsets,
    so plain string literals like spectype = 'GALAXY' are safe. A column the
    SQL mentions anywhere is treated as caller intent and never overridden.
    """
    if "--" in query or "/*" in query:
        return query, clean
    if len(query) != len(clean):  # escaped quotes shrink the scrub — offsets unusable
        return query, clean
    if len(tables) != 1 or re.search(r"\b(JOIN|UNION)\b", clean, re.IGNORECASE):
        return query, clean
    if len(re.findall(r"\bSELECT\b", clean, re.IGNORECASE)) != 1:
        return query, clean
    catalog, table = tables[0]
    try:
        defaults = registry.default_quality_cuts(catalog, table)
    except Exception:  # noqa: BLE001 - unregistered tables simply have no defaults
        return query, clean
    missing = [
        d for d in defaults
        if not re.search(rf"\b{re.escape(str(d['column']))}\b", clean, re.IGNORECASE)
    ]
    if not missing:
        return query, clean
    preds = []
    for d in missing:
        val = d["value"]
        lit = f"'{val}'" if isinstance(val, str) else str(val)
        preds.append(f"{d['column']} {d['op']} {lit}")
    pred_sql = " AND ".join(preds)
    # Locate clauses on CLEAN (literals blanked, so 'WHERE'-in-a-string cannot
    # match) and splice QUERY at the same offsets (lengths verified equal).
    where_m = _WHERE_SEGMENT_RE.search(clean)
    if where_m:
        # Parenthesize the original condition: a top-level OR would otherwise
        # let its right leg escape the injected cuts.
        c0, c1 = where_m.span(1)
        cond = query[c0:c1].strip()
        new_query = f"{query[:c0]} {pred_sql} AND ({cond}) {query[c1:]}".rstrip()
    else:
        tail_m = _CLAUSE_TAIL_RE.search(clean)
        if tail_m:
            i = tail_m.start()
            new_query = f"{query[:i]}WHERE {pred_sql} {query[i:]}"
        else:
            new_query = f"{query} WHERE {pred_sql}"
    warnings.append(
        f"Auto-applied {catalog}.{table} registry quality cuts to expert SQL: {pred_sql}. "
        "These are platform survey-quality defaults (not user-requested science cuts) that make "
        "results comparable to the survey's canonical selection — disclose them in the answer; "
        "include your own cut on those columns to override."
    )
    return new_query, _scrub_sql(new_query)


def _warn_nan_unsafe_cuts(raw: str, clean: str, warnings: List[str]) -> None:
    unguarded: List[str] = []
    for match in NAN_UNSAFE_CUT_RE.finditer(clean):
        col = match.group(1).lower()
        if col in _NAN_EXEMPT_COLUMNS or col in unguarded:
            continue
        # Any upper bound on the same column already excludes NaN. The
        # 'Infinity' guard is checked against the RAW sql because _scrub_sql
        # blanks string literals.
        has_upper = re.search(rf"\b{re.escape(col)}\s*(?:<=?|BETWEEN)\s", clean, re.IGNORECASE)
        has_inf_guard = re.search(rf"\b{re.escape(col)}\s*<\s*'Infinity'", raw, re.IGNORECASE)
        if not has_upper and not has_inf_guard:
            unguarded.append(col)
    if unguarded:
        warnings.append(
            "NaN-unsafe cuts on: " + ", ".join(unguarded) + ". Data Lab stores missing floats as NaN, "
            "which Postgres orders ABOVE every real number — bare >/>=/!= cuts admit every missing-value "
            "row. If these are nullable float columns, re-run with a finiteness guard, "
            "e.g. `AND parallax_over_error < 'Infinity'`."
        )


def _has_multiple_statements(clean: str) -> bool:
    return ";" in clean.strip().rstrip(";")


def _tables(clean: str) -> List[tuple[str, str]]:
    found: List[tuple[str, str]] = []
    for match in FROM_JOIN_RE.finditer(clean):
        catalog, table = match.group(1).lower(), match.group(2).lower()
        registry.describe_table(catalog, table)
        item = (catalog, table)
        if item not in found:
            found.append(item)
    return found


def _qualified(catalog: str, table: str) -> str:
    return f"{catalog}.{table}"


def _has_ra_dec_between(clean: str) -> bool:
    return bool(
        re.search(r"\bra\b\s+BETWEEN\b", clean, re.IGNORECASE)
        and re.search(r"\bdec\b\s+BETWEEN\b", clean, re.IGNORECASE)
    )


def _all_boxes_allowed(tables: List[tuple[str, str]]) -> bool:
    return bool(tables) and all(registry.region_strategy(catalog, table) == "box_ok" for catalog, table in tables)


_WHERE_SEGMENT_RE = re.compile(
    r"\bWHERE\b(.*?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|\bOFFSET\b|$)",
    re.IGNORECASE | re.DOTALL,
)


def _where_segment(text: str) -> str:
    match = _WHERE_SEGMENT_RE.search(text)
    return match.group(1) if match else ""


def _has_indexed_equality(raw_sql: str, clean: str, tables: List[tuple[str, str]]) -> Optional[str]:
    """Return the column name when the query is genuinely bounded by an equality
    on a registry indexed-bound column (e.g. smash fieldid = 169, id = '169.429960').

    Deliberately conservative — this loosening must not become a bypass:
    - single-table queries only (multi-table joins need q3c bounds);
    - no OR anywhere in the scrubbed logic (`col = 1 OR 1=1` is unbounded);
    - the equality must appear inside the WHERE clause (a projection like
      `SELECT fieldid = 169 AS is_field` is not a bound — Codex guard finding);
    - a NUMERIC equality must appear in the literal/comment-scrubbed text, so
      column names inside string literals or comments never count;
    - a QUOTED equality must show `col =` in the scrubbed text (real code, since
      scrubbing blanks literal interiors) AND `col = '<literal>'` in the raw
      text, so the RHS is a literal — never another column reference.
    """
    if len(tables) != 1:
        return None
    if re.search(r"\bOR\b", clean, re.IGNORECASE):
        return None
    clean_where = _where_segment(clean)
    raw_where = _where_segment(raw_sql)
    if not clean_where.strip():
        return None
    catalog, table = tables[0]
    for col in registry.indexed_bound_columns(catalog, table):
        col_re = re.escape(col)
        numeric_in_clean = re.search(rf"\b{col_re}\s*=\s*\d[\w.\-+]*", clean_where, re.IGNORECASE)
        eq_in_clean = re.search(rf"\b{col_re}\s*=", clean_where, re.IGNORECASE)
        quoted_in_raw = re.search(rf"\b{col_re}\s*=\s*'[^']+'", raw_where, re.IGNORECASE)
        if numeric_in_clean or (eq_in_clean and quoted_in_raw):
            return col
    return None


# ── column grounding against the curated registry + cached live tap_schema ──
#
# The blog-parity goal: SQL that references a hallucinated column name fails
# fast with real suggestions instead of a cryptic server error. Ground truth is
# registry.known_columns (curated fast-path ∪ cached live tap_schema.columns).

_SQL_KEYWORDS = frozenset({
    "select", "from", "where", "and", "or", "not", "as", "on", "join", "inner",
    "left", "right", "full", "outer", "cross", "with", "group", "by", "order",
    "limit", "offset", "having", "distinct", "between", "in", "is", "null",
    "like", "ilike", "case", "when", "then", "else", "end", "asc", "desc",
    "nulls", "first", "last", "true", "false", "union", "all", "exists", "any",
    "cast", "using", "escape", "array",
    # window/expression keywords that appear as bare tokens
    "over", "partition", "rows", "range", "unbounded", "preceding", "following",
    "current", "row", "epoch", "for",
    # type names (CAST targets and ::casts)
    "numeric", "double", "precision", "integer", "bigint", "smallint", "real",
    "float", "text", "varchar", "char", "boolean", "date", "timestamp", "interval",
})

_QUALIFIED_REF_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)\b")
# Identifiers not adjacent to a dot (qualified refs are handled separately) and
# not preceded by a word char (so the 'e3' inside the literal 1e3 never matches).
_BARE_IDENT_RE = re.compile(r"(?<![\w.$])([A-Za-z_]\w*)(?![\w.])")
_FROM_ALIAS_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+([A-Za-z_]\w*)\.([A-Za-z_]\w*)(?:\s+(?:AS\s+)?([A-Za-z_]\w*))?",
    re.IGNORECASE,
)
_AS_ALIAS_RE = re.compile(r"\bAS\s+([A-Za-z_]\w*)", re.IGNORECASE)


def _known_column_maps(tables: List[tuple[str, str]]) -> Optional[Dict[str, Dict[str, Any]]]:
    known: Dict[str, Dict[str, Any]] = {}
    for catalog, table in tables:
        info = registry.known_columns(catalog, table)
        if info is None or not info.get("columns"):
            return None
        known[_qualified(catalog, table)] = info
    return known


def _column_suggestions(name: str, candidates: set[str]) -> List[str]:
    matches = difflib.get_close_matches(name.lower(), sorted(candidates), n=3, cutoff=0.55)
    if not matches:
        # Fall back to prefix/substring hits so e.g. 'w1' still suggests w1mpro.
        matches = [c for c in sorted(candidates) if name.lower()[:3] and name.lower()[:3] in c][:3]
    return matches


def _check_column_references(clean: str, first_keyword: str, tables: List[tuple[str, str]], warnings: List[str]) -> None:
    """Reject (or, when unverifiable, warn about) column names that exist in
    neither the curated registry nor the cached live TAP schema.

    Deliberately conservative to avoid false positives:
    - identifiers followed by ``(`` are function calls — skipped;
    - output aliases (``AS x``), table aliases, and catalog/table tokens are allowed;
    - bare (unqualified) identifiers are only checked for single-table non-WITH
      queries; qualified ``alias.col`` refs are checked whenever the alias maps
      to a registered table;
    - a miss is a hard error only when the live schema for every referenced
      table is cached (``authoritative``); curated lists are subsets, so
      curated-only misses just warn.
    """

    if os.getenv("DATALAB_COLUMN_CHECK", "1").strip().lower() in {"0", "false", "off"}:
        return
    known = _known_column_maps(tables)
    if known is None:
        return
    authoritative = all(info.get("authoritative") for info in known.values())
    all_columns: set[str] = set()
    for info in known.values():
        all_columns.update(info["columns"])

    text = re.sub(r"::\s*[A-Za-z_]\w*", " ", clean)  # strip ::numeric-style casts

    alias_map: Dict[str, str] = {}
    for match in _FROM_ALIAS_RE.finditer(text):
        catalog, table = match.group(1).lower(), match.group(2).lower()
        qualified = _qualified(catalog, table)
        if qualified not in known:
            continue
        alias_map.setdefault(table, qualified)
        alias = (match.group(3) or "").lower()
        if alias and alias not in _SQL_KEYWORDS:
            alias_map.setdefault(alias, qualified)

    allowed: set[str] = set(_SQL_KEYWORDS)
    allowed.update(all_columns)
    allowed.update(alias_map.keys())
    for catalog, table in tables:
        allowed.add(catalog)
        allowed.add(table)
    for match in _AS_ALIAS_RE.finditer(text):
        allowed.add(match.group(1).lower())

    output_aliases = {match.group(1).lower() for match in _AS_ALIAS_RE.finditer(text)}
    unknown: Dict[str, set[str]] = {}

    for match in _QUALIFIED_REF_RE.finditer(text):
        prefix, column = match.group(1).lower(), match.group(2).lower()
        qualified = alias_map.get(prefix)
        if qualified is None:
            continue  # CTE alias or the catalog.table token itself
        rest = text[match.end():].lstrip()
        if rest[:1] in ("(", "["):
            continue  # function call / array subscript
        # Check STRICTLY against the alias's own table (a column that exists
        # only on some OTHER joined table is still wrong here — CX-02), plus
        # output aliases and keywords.
        if (
            column not in known[qualified]["columns"]
            and column not in output_aliases
            and column not in _SQL_KEYWORDS
        ):
            unknown.setdefault(column, set()).update(known[qualified]["columns"])

    if len(tables) == 1 and first_keyword == "SELECT":
        for match in _BARE_IDENT_RE.finditer(text):
            token = match.group(1).lower()
            if token in allowed:
                continue
            rest = text[match.end():].lstrip()
            if rest[:1] in ("(", "["):
                continue  # function call / ARRAY[...] constructor
            unknown.setdefault(token, set()).update(all_columns)

    if not unknown:
        return
    parts = []
    for name in sorted(unknown):
        suggestions = _column_suggestions(name, unknown[name])
        parts.append(f"'{name}'" + (f" (did you mean: {', '.join(suggestions)}?)" if suggestions else ""))
    tables_text = ", ".join(sorted(known))
    message = (
        f"Unknown column(s) {'; '.join(parts)} for {tables_text} — "
        "checked against the live Data Lab tap_schema" if authoritative else
        f"Column(s) {'; '.join(parts)} not found in the curated registry for {tables_text} "
        "(live schema not cached yet, so this may be a real but unregistered column)"
    )
    if authoritative:
        _raise(
            message,
            "Use datalab_describe_table(catalog, table) to see the real column names, "
            "then rewrite the query with those columns.",
        )
    warnings.append(message + ".")


def _is_aggregate(clean: str) -> bool:
    return bool(
        re.search(r"\bGROUP\s+BY\b", clean, re.IGNORECASE)
        or re.search(r"\bCOUNT\s*\(", clean, re.IGNORECASE)
        or re.search(r"\bAVG\s*\(", clean, re.IGNORECASE)
        or re.search(r"\bSUM\s*\(", clean, re.IGNORECASE)
    )


def _limit_value(clean: str) -> Optional[int]:
    matches = list(LIMIT_RE.finditer(clean))
    if not matches:
        return None
    return int(matches[-1].group(1))


def _raise(message: str, hint: str) -> None:
    raise DatalabPolicyError(message, fix_hint=hint)


__all__ = ["DatalabPolicyError", "ValidatedQuery", "validate"]
