"""Governor for Data Lab SQL emitted by builders or expert raw mode."""

from __future__ import annotations

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
    tables = _tables(clean)
    if not tables:
        _raise("No registered Data Lab table found", "Query a registered catalog.table from the Data Lab registry.")

    has_radial = bool(re.search(r"\bq3c_radial_query\s*\(", clean, re.IGNORECASE))
    has_poly = bool(re.search(r"\bq3c_poly_query\s*\(", clean, re.IGNORECASE))
    has_q3c_join = bool(re.search(r"\bq3c_join\s*\(", clean, re.IGNORECASE))
    has_box = _has_ra_dec_between(clean)
    box_allowed = _all_boxes_allowed(tables)
    if has_box and not box_allowed:
        _raise(
            "Bare RA/Dec BETWEEN boxes are not allowed for q3c registry tables",
            "Use q3c_radial_query/q3c_poly_query or a table marked region_strategy='box_ok'.",
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
    spatial_bound = has_radial or has_poly or has_q3c_join or (has_box and box_allowed) or exact_id_bound
    if not spatial_bound and not aggregate_safe:
        _raise(
            "Row-level Data Lab queries require a q3c spatial bound or a registry-approved aggregate",
            "Add q3c_radial_query/q3c_poly_query, use a box_ok table box, or use an aggregate builder.",
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
            final_sql = f"{query}\nLIMIT {DEFAULT_ROW_CAP}"
            warnings.append(f"Injected LIMIT {DEFAULT_ROW_CAP} output cap.")
        elif limit > MAX_ROW_CAP:
            _raise(
                f"LIMIT {limit} exceeds the Data Lab row cap {MAX_ROW_CAP}",
                f"Use LIMIT <= {MAX_ROW_CAP} or a builder aggregate.",
            )

    return ValidatedQuery(sql=final_sql, source=source, meta=metadata, warnings=warnings)


def _strip_one_semicolon(sql: str) -> str:
    return sql[:-1].rstrip() if sql.endswith(";") else sql


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
