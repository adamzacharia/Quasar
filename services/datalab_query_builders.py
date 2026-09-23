"""Pure SQL builders for NOIRLab Astro Data Lab P0 catalog tools."""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from services import datalab_registry as registry

# A safe SQL column identifier. The registry's column lists are NOT exhaustive (catalogs have
# hundreds of columns), so we accept any safe identifier and let Data Lab validate it server-side
# rather than hard-failing a whole task on an unlisted-but-real column.
_SAFE_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


DEFAULT_ROW_LIMIT = 500
MAX_ROW_LIMIT = 5000
# Appended to every LIMIT the PLATFORM chose (no limit requested, or the
# request was clamped to the ceiling) so the provenance SQL can never read as
# a science choice (RE-B1, NOIRLab beta eval). Deliberately contains no digits
# and not the word "AND" so SQL-shape checks and NaN-guard regexes never see it.
PLATFORM_ROW_CAP_COMMENT = "/* platform row cap, not a science cut */"
# Injected sentinel-magnitude guards carry this marker for the same reason —
# provenance readers must see data validity, not a saturation/noise-floor cut.
SENTINEL_GUARD_COMMENT = "/* sentinel-removal guard, not a science cut */"
# Single source of truth for sentinel-magnitude bounds (RE-B2): real photometry
# lives inside this range; 99/99.99/-99/... padding marks MISSING measurements.
SENTINEL_MAG_RANGE = registry.SENTINEL_MAG_RANGE
# A model-supplied morphology threshold this many times larger or smaller than
# the table's registered star/galaxy convention is almost certainly a threshold
# conflation (e.g. class_star's 0.5 applied to spread_model — NOIRLab beta
# eval). Such cuts get a LOUD warning; they are never silently clamped.
MORPHOLOGY_DEVIATION_FACTOR = 20.0
# Cap cone radius so an all-sky cone (radius_deg=180/360) can't seq-scan a catalog.
MAX_CONE_RADIUS_DEG = float(os.getenv("DATALAB_MAX_CONE_RADIUS_DEG", "30"))
# Whitelisted comparison operators for structured selection cuts.
_CUT_OPS = {"<", ">", "<=", ">=", "=", "!="}
# Operator spellings LLMs frequently emit that mean the same thing: Python/JS
# equality "==" and SQL-standard not-equal "<>".
_CUT_OP_ALIASES = {"==": "=", "<>": "!="}
# Data Lab stores missing floats as NaN (not SQL NULL), and Postgres orders NaN
# ABOVE every real number — so a bare `col > x` / `col >= x` / `col != x` cut
# silently admits every NaN row (live P6: identical predicates gave 1,960 rows
# on Data Lab vs 8 on the ESA archive; adding the guard gave exactly 8).
# `<`/`<=` cuts already exclude NaN. The ::float8 cast keeps the guard valid on
# integer columns too (an unadorned 'Infinity' literal would fail to coerce).
_NAN_UNSAFE_OPS = {">", ">=", "!="}
_NAN_GUARD = "< 'Infinity'::float8"


def _normalize_cut_op(op: Any) -> str:
    """Canonicalize a value-cut operator, mapping common aliases ('==' -> '=')."""
    text = str(op or "").strip()
    text = _CUT_OP_ALIASES.get(text, text)
    if text not in _CUT_OPS:
        raise ValueError(f"unsupported value-cut operator {op!r}")
    return text


def _cut_rhs(op: str, value: Any) -> str:
    """Render the right-hand side of a value cut as a safe SQL literal.

    Numeric values (including numeric strings like '169') become numeric
    literals. Non-numeric strings are allowed ONLY for '='/'!=' and become
    single-quoted, quote-escaped string literals (e.g. ``class = 'GALAXY'``),
    which the SQL policy layer recognizes. Inequality operators still require a
    number.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        pass
    else:
        if math.isnan(number):
            raise ValueError("value-cut value cannot be NaN")
        if math.isinf(number):
            # Models write explicit finiteness guards ({"op": "<", "value":
            # "Infinity"}) after the NaN-convention prompt rule; format() would
            # render bare `inf`, which Data Lab reads as a column name (live P9:
            # "Unknown column(s) 'inf'"). Emit the quoted float8 literal.
            return f"'{'-' if number < 0 else ''}Infinity'::float8"
        return _num(number)
    if op not in ("=", "!="):
        raise ValueError(f"value-cut operator {op!r} requires a numeric value, got {value!r}")
    text = str(value).strip()
    # Tolerate a value the model already wrapped in quotes.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1]
    if not text or len(text) > 128 or any(ord(ch) < 32 for ch in text):
        raise ValueError(f"invalid string value for value cut: {value!r}")
    return "'" + text.replace("'", "''") + "'"


def _cut_predicate(col: str, op: str, value: Any) -> str:
    """Render one value-cut predicate, NaN-guarded for NaN-unsafe operators.

    String-valued cuts never need the guard — only numeric comparisons
    (including quoted ±Infinity literals: `col > '-Infinity'` admits NaN)
    can leak NaN rows.
    """
    rhs = _cut_rhs(op, value)
    numeric_rhs = not rhs.startswith("'") or rhs.endswith("Infinity'::float8")
    if op in _NAN_UNSAFE_OPS and numeric_rhs:
        return f"({col} {op} {rhs} AND {col} {_NAN_GUARD})"
    return f"{col} {op} {rhs}"


def build_cone_count(catalog: str, table: str, *, ra: float, dec: float, radius_deg: float) -> Tuple[str, Dict[str, Any]]:
    info = _table_info(catalog, table)
    ra_col, dec_col = info["ra_column"], info["dec_column"]
    _validate_sky(ra, dec, radius_deg)
    sql = (
        f"SELECT COUNT(*) AS row_count\n"
        f"FROM {info['qualified_name']}\n"
        f"WHERE q3c_radial_query({ra_col}, {dec_col}, {_num(ra)}, {_num(dec)}, {_num(radius_deg)})"
    )
    return sql, _meta("cone_count", info, spatial_bound=True, aggregate=True)


def build_cone_select(
    catalog: str,
    table: str,
    *,
    ra: float,
    dec: float,
    radius_deg: float,
    columns: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    predicates: Optional[Sequence[str]] = None,
    default_limit: int = DEFAULT_ROW_LIMIT,
) -> Tuple[str, Dict[str, Any]]:
    info = _table_info(catalog, table)
    ra_col, dec_col = info["ra_column"], info["dec_column"]
    _validate_sky(ra, dec, radius_deg)
    # Row-level pulls always need SOME cap (cost governance); when the caller
    # did not choose one, the platform default is applied and FLAGGED so the
    # provenance can never read as a science choice (RE-B1). `default_limit`
    # lets one-shot diagram tools declare their plotting budget as that
    # platform default instead of baking it into the tool schema.
    row_limit, cap_reason = _resolve_limit(limit, default=default_limit)
    select_cols = _select_columns(info, columns)
    # Extra cuts must come from build_catalog_predicates (registry-validated),
    # so the LIMIT budget is spent on rows that survive the selection.
    where_parts = [f"q3c_radial_query({ra_col}, {dec_col}, {_num(ra)}, {_num(dec)}, {_num(radius_deg)})"]
    for pred in (predicates or []):
        text = str(pred).strip()
        if text:
            where_parts.append(f"({text})")
    sql = (
        f"SELECT {select_cols}\n"
        f"FROM {info['qualified_name']}\n"
        f"WHERE " + "\n  AND ".join(where_parts) + "\n"
        + _limit_clause(row_limit, cap_reason)
    )
    meta = _meta("cone_select", info, spatial_bound=True, row_limit=row_limit)
    return sql, _flag_platform_cap(meta, row_limit, cap_reason)


def build_key_select(
    catalog: str,
    table: str,
    *,
    key_column: str,
    key_value: Any,
    columns: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    predicates: Optional[Sequence[str]] = None,
    default_limit: int = DEFAULT_ROW_LIMIT,
) -> Tuple[str, Dict[str, Any]]:
    """Select rows by an indexed KEY (``fieldid = 169`` on SMASH tables) with
    NO cone: the UI benchmark's L07 used ``q3c_radial_query(ra, dec, 0, 0, 5)``
    as a dummy cone to reach a field. The key column must be a registered,
    indexed bound column of the table."""
    info = _table_info(catalog, table)
    indexed = set(indexed_bound_columns_for(info))
    if key_column not in (info.get("columns") or []):
        raise ValueError(f"unknown column {key_column!r} for {info['qualified_name']}")
    if indexed and key_column not in indexed:
        raise ValueError(
            f"{key_column!r} is not an indexed bound column of {info['qualified_name']} "
            f"(indexed: {', '.join(sorted(indexed))}); a key select on it would scan the table."
        )
    row_limit, cap_reason = _resolve_limit(limit, default=default_limit)
    select_cols = _select_columns(info, columns)
    if isinstance(key_value, (int, float)) and not isinstance(key_value, bool):
        literal = _num(float(key_value)) if isinstance(key_value, float) else str(int(key_value))
    else:
        text = str(key_value).replace("'", "''")
        literal = f"'{text}'"
    where_parts = [f"{key_column} = {literal}"]
    for pred in (predicates or []):
        text = str(pred).strip()
        if text:
            where_parts.append(f"({text})")
    sql = (
        f"SELECT {select_cols}\n"
        f"FROM {info['qualified_name']}\n"
        f"WHERE " + "\n  AND ".join(where_parts) + "\n"
        + _limit_clause(row_limit, cap_reason)
    )
    meta = _meta("key_select", info, spatial_bound=False, row_limit=row_limit, key_column=key_column)
    meta["indexed_key"] = True
    return sql, _flag_platform_cap(meta, row_limit, cap_reason)


def indexed_bound_columns_for(info: Dict[str, Any]) -> List[str]:
    """Indexed key columns for a described table (registry-backed)."""
    from services import datalab_registry as reg

    try:
        return list(reg.indexed_bound_columns(info["catalog"], info["table"]))
    except Exception:
        return []


def build_rectangular_region_select(
    catalog: str,
    table: str,
    *,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    columns: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    info = _table_info(catalog, table)
    _validate_rect(ra_min, ra_max, dec_min, dec_max)
    row_limit, cap_reason = _resolve_limit(limit)
    select_cols = _select_columns(info, columns)
    ra_col, dec_col = info["ra_column"], info["dec_column"]
    warnings: List[str] = []
    # ra_min > ra_max: the box wraps through RA=0/360 — split it into the two
    # non-wrapping sub-boxes [ra_min,360)∪[0,ra_max] OR'd together
    # (ra-wrap-rect-footprint-unsupported).
    wraps = float(ra_min) > float(ra_max)
    ra_ranges = [(ra_min, 360.0), (0.0, ra_max)] if wraps else [(ra_min, ra_max)]
    if wraps:
        warnings.append(
            f"RA bounds wrap through 0/360: interpreted as [{_num(ra_min)}°, 360°) ∪ "
            f"[0°, {_num(ra_max)}°] ({_num((360.0 - float(ra_min)) + float(ra_max))}° span), "
            "queried as two OR'd sub-boxes. Swap the bounds if you meant the "
            "complementary band."
        )
    if info["region_strategy"] == "box_ok":
        boxes = [
            f"{ra_col} BETWEEN {_num(lo)} AND {_num(hi)} "
            f"AND {dec_col} BETWEEN {_num(dec_min)} AND {_num(dec_max)}"
            for lo, hi in ra_ranges
        ]
        where = boxes[0] if len(boxes) == 1 else "(" + ") OR (".join(boxes) + ")"
        warnings.append("Used registry-approved RA/Dec BETWEEN box for this table.")
        spatial = "box"
    else:
        polys = [
            (
                f"q3c_poly_query({ra_col}, {dec_col}, "
                f"ARRAY[{_num(lo)}, {_num(hi)}, {_num(hi)}, {_num(lo)}], "
                f"ARRAY[{_num(dec_min)}, {_num(dec_min)}, {_num(dec_max)}, {_num(dec_max)}])"
            )
            for lo, hi in ra_ranges
        ]
        where = polys[0] if len(polys) == 1 else "(" + " OR ".join(polys) + ")"
        spatial = "q3c_poly_query"
    sql = (
        f"SELECT {select_cols}\nFROM {info['qualified_name']}\nWHERE {where}\n"
        + _limit_clause(row_limit, cap_reason)
    )
    meta = _meta("rectangular_region_select", info, spatial_bound=True, row_limit=row_limit)
    meta.update({"spatial_strategy": spatial, "warnings": warnings})
    return sql, _flag_platform_cap(meta, row_limit, cap_reason)


def build_density_aggregate(
    catalog: str,
    table: str,
    *,
    mode: str = "grid",
    step_deg: float = 0.1,
    healpix_column: Optional[str] = None,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    radius_deg: Optional[float] = None,
    all_sky: bool = False,
    predicates: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    field_bound: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    info = _table_info(catalog, table)
    row_limit, cap_reason = _resolve_limit(limit, maximum=MAX_ROW_LIMIT, default=MAX_ROW_LIMIT)
    ra_col, dec_col = info["ra_column"], info["dec_column"]
    # Require an explicit region: an unbounded GROUP BY over a billion-row catalog
    # is a full-catalog scan. Give a cone, opt in explicitly with all_sky=True, or
    # (field_bound) rely on an indexed-equality predicate such as SMASH
    # `fieldid = 169` — the canonical bound for field-partitioned tables, and the
    # only way to map a WHOLE survey field without guessing its center (live P7:
    # a guessed 0.6° cone cut Hydra II out of the map).
    has_cone = ra is not None and dec is not None and radius_deg is not None
    warnings: List[str] = []
    clauses: List[str] = []
    if has_cone:
        _validate_sky(float(ra), float(dec), float(radius_deg))
        clauses.append(f"q3c_radial_query({ra_col}, {dec_col}, {_num(float(ra))}, {_num(float(dec))}, {_num(float(radius_deg))})")
    elif field_bound and (predicates or []):
        warnings.append("Aggregate bounded by an indexed-equality predicate (e.g. fieldid = N) instead of a cone.")
    elif all_sky is True:
        warnings.append("Unbounded all-sky aggregate: scans the whole catalog server-side and may be slow/expensive.")
    else:
        raise ValueError(
            "density aggregate requires a cone (ra, dec, radius_deg), an indexed-equality "
            "value_cut (e.g. fieldid = 169 on SMASH tables), or an explicit all_sky=True "
            "(an unbounded aggregate scans the whole catalog)"
        )
    # Selection cuts (color/magnitude/morphology) must come from build_catalog_predicates
    # so columns are registry-validated; they are appended to the WHERE.
    for pred in (predicates or []):
        text = str(pred).strip()
        if text:
            clauses.append(f"({text})")
    where = ("WHERE " + "\n  AND ".join(clauses) + "\n") if clauses else ""
    mode_key = str(mode or "grid").strip().lower()
    if mode_key == "healpix":
        hpix = _healpix_column(info, healpix_column)
        sql = (
            f"SELECT {hpix} AS healpix, COUNT(*) AS source_count\n"
            f"FROM {info['qualified_name']}\n"
            f"{where}"
            f"GROUP BY {hpix}\n"
            f"ORDER BY source_count DESC\n"
            + _limit_clause(row_limit, cap_reason)
        )
    elif mode_key == "grid":
        step = _positive(step_deg, "step_deg")
        sql = (
            f"SELECT ROUND({ra_col} / {_num(step)}) * {_num(step)} AS ra_bin,\n"
            f"       ROUND({dec_col} / {_num(step)}) * {_num(step)} AS dec_bin,\n"
            f"       COUNT(*) AS source_count\n"
            f"FROM {info['qualified_name']}\n"
            f"{where}"
            f"GROUP BY ra_bin, dec_bin\n"
            f"ORDER BY source_count DESC\n"
            + _limit_clause(row_limit, cap_reason)
        )
    else:
        raise ValueError("density aggregate mode must be grid or healpix")
    meta = _meta("density_aggregate", info, aggregate=True, spatial_bound=has_cone, row_limit=row_limit)
    meta["warnings"] = warnings
    _flag_platform_cap(meta, row_limit, cap_reason)
    if mode_key == "healpix":
        # Record the column's authoritative pixelization: decoding RING pixels
        # with the renderer's NESTED default scatters cells across the sky
        # (live DS-P8: a 2° Galactic-center cone rendered as two blobs 100° apart).
        hp_entry = next((h for h in (info.get("healpix_columns") or []) if h.get("name") == hpix), None)
        if hp_entry:
            meta["healpix"] = {
                "column": hpix,
                "nside": hp_entry.get("nside"),
                "scheme": hp_entry.get("scheme"),
            }
    return sql, meta


def build_catalog_predicates(
    catalog: str,
    table: str,
    *,
    color_cut: Optional[Mapping[str, Any]] = None,
    value_cuts: Optional[Sequence[Mapping[str, Any]]] = None,
    morphology: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Build safe, registry-validated WHERE predicate fragments for catalog selections.

    color_cut:  {"bands": ["mag_auto_g", "mag_auto_r"], "min": -0.5, "max": 0.5}
    value_cuts: [{"column": "mag_auto_g", "op": ">", "value": 19.5}, ...]  (op in _CUT_OPS)
    morphology: {"column": "ext_coadd", "between": [0, 1]} |
                {"column": "class_star", "op": ">", "value": 0.5}   (0.5 is class_star-ONLY) |
                {"column": "spread_model_r", "between": [-0.003, 0.003]}  (DES stellar cut) |
                {"column": "ext_coadd", "in": [0, 1]}
    Every column is checked against the registry; operators are whitelisted; values are
    numeric-formatted — so the output is safe to append to a builder WHERE clause.
    Callers surfacing warnings should also run morphology_deviation_warning() on the
    morphology cut (orders-of-magnitude threshold conflations, RE-B3).
    """
    info = _table_info(catalog, table)
    preds: List[str] = []
    if color_cut:
        bands = list(color_cut.get("bands") or [])
        if len(bands) != 2:
            raise ValueError("color_cut.bands must be exactly two registered columns")
        b0, b1 = _column(info, bands[0]), _column(info, bands[1])
        expr = f"({b0} - {b1})"
        if color_cut.get("min") is not None:
            pred = f"{expr} >= {_num(float(color_cut['min']))}"
            if color_cut.get("max") is None:
                # A min-only color cut has no `<=` leg to exclude NaN colors.
                pred = f"({pred} AND {expr} {_NAN_GUARD})"
            preds.append(pred)
        if color_cut.get("max") is not None:
            preds.append(f"{expr} <= {_num(float(color_cut['max']))}")
    for vc in (value_cuts or []):
        col = _column(info, vc["column"])
        op = _normalize_cut_op(vc.get("op", ""))
        preds.append(_cut_predicate(col, op, vc["value"]))
    if morphology:
        col = _column(info, morphology["column"])
        if morphology.get("in"):
            vals = ", ".join(_num(float(v)) for v in morphology["in"])
            preds.append(f"{col} IN ({vals})")
        if morphology.get("between"):
            lo, hi = morphology["between"]
            preds.append(f"{col} BETWEEN {_num(float(lo))} AND {_num(float(hi))}")
        if morphology.get("op"):
            op = _normalize_cut_op(morphology["op"])
            preds.append(_cut_predicate(col, op, morphology["value"]))
    return preds


def build_q3c_crossmatch(
    *,
    small_catalog: str = "gaia_dr3",
    small_table: str = "gaia_source",
    big_catalog: str = "nsc_dr2",
    big_table: str = "object",
    ra: float,
    dec: float,
    radius_deg: float,
    match_radius_arcsec: float = 1.0,
    small_columns: Optional[Sequence[str]] = None,
    big_columns: Optional[Sequence[str]] = None,
    small_limit: Optional[int] = None,
    limit: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    small = _table_info(small_catalog, small_table)
    big = _table_info(big_catalog, big_table)
    _validate_sky(ra, dec, radius_deg)
    match_radius_deg = _positive(match_radius_arcsec, "match_radius_arcsec") / 3600.0
    # The pre-join CTE cap is a platform budget exactly like the outer LIMIT:
    # None means the caller made no choice, so the default (10000) must be
    # flagged — SQL comment + platform_row_caps metadata — never read as a
    # science cut (RE-B1 / guard CX-01). An explicit in-range value is the
    # caller's own provenance and stays unflagged.
    small_row_limit, small_cap_reason = _resolve_limit(small_limit, maximum=50000, default=10000)
    row_limit, cap_reason = _resolve_limit(limit)
    small_select = _select_columns(small, small_columns, required=[small["ra_column"], small["dec_column"]])
    big_select = _prefixed_columns("big", _column_list(big, big_columns, required=[big["ra_column"], big["dec_column"]]))
    sra, sdec = small["ra_column"], small["dec_column"]
    bra, bdec = big["ra_column"], big["dec_column"]
    sql = (
        "WITH g AS MATERIALIZED (\n"
        f"    SELECT {small_select}\n"
        f"    FROM {small['qualified_name']}\n"
        f"    WHERE q3c_radial_query({sra}, {sdec}, {_num(ra)}, {_num(dec)}, {_num(radius_deg)})\n"
        # The CTE cap must be nearest-first too: a storage-order LIMIT here
        # selects a spatially clustered corner of the cone BEFORE the join, and
        # no outer ORDER BY can undo it (live P9 post-fix run: the map still
        # showed the far-west edge because only the outer SELECT was ordered).
        f"    ORDER BY q3c_dist({sra}, {sdec}, {_num(ra)}, {_num(dec)})\n"
        f"    LIMIT {small_row_limit}"
        + (f" {PLATFORM_ROW_CAP_COMMENT}" if small_cap_reason else "")
        + "\n"
        ")\n"
        "SELECT g.*,\n"
        f"       {big_select}\n"
        "FROM g\n"
        f"JOIN {big['qualified_name']} AS big\n"
        f"  ON q3c_join(g.{sra}, g.{sdec}, big.{bra}, big.{bdec}, {_num(match_radius_deg)})\n"
        # Nearest-to-center first: when the row cap bites, the kept slice contains
        # the cone CENTER (the cluster/stream the user asked about) instead of a
        # storage-order corner chunk (live P9: the map excluded Pal 5 itself).
        f"ORDER BY q3c_dist(g.{sra}, g.{sdec}, {_num(ra)}, {_num(dec)})\n"
        + _limit_clause(row_limit, cap_reason)
    )
    meta = _meta("q3c_crossmatch", small, spatial_bound=True, row_limit=row_limit)
    _flag_platform_cap(meta, row_limit, cap_reason)
    meta.update(
        {
            "big_catalog": big["catalog"],
            "big_table": big["table"],
            "small_limit": small_row_limit,
            "match_radius_deg": match_radius_deg,
            "q3c_join_authorized": True,
            # Record the join columns/aliases so the governor can verify planner-safe
            # arg order for ANY registered table (not just ra/dec ones, e.g. DESI mean_fiber_*).
            "join_small_alias": "g",
            "join_big_alias": "big",
            "join_small_ra": sra,
            "join_small_dec": sdec,
            "join_big_ra": bra,
            "join_big_dec": bdec,
        }
    )
    # Named platform budget for the inner CTE cap (guard CX-01): rides
    # meta['platform_row_caps'] so the outer int meta['platform_row_cap'] —
    # consumed via int() by every provenance stamper — is never clobbered.
    _flag_named_platform_cap(
        meta,
        name="small_side",
        row_limit=small_row_limit,
        platform_reason=small_cap_reason,
        what="Crossmatch small-side (pre-join CTE) row cap",
    )
    return sql, meta


def build_bitmask_select(
    catalog: str,
    table: str,
    *,
    bitmask_column: str,
    bit_name: str,
    ra: float,
    dec: float,
    radius_deg: float,
    columns: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    info = _table_info(catalog, table)
    _validate_sky(ra, dec, radius_deg)
    bit = _bit_value(info, bitmask_column, bit_name)
    row_limit, cap_reason = _resolve_limit(limit)
    select_cols = _select_columns(info, columns)
    ra_col, dec_col = info["ra_column"], info["dec_column"]
    mask = 1 << bit
    sql = (
        f"SELECT {select_cols}\n"
        f"FROM {info['qualified_name']}\n"
        f"WHERE q3c_radial_query({ra_col}, {dec_col}, {_num(ra)}, {_num(dec)}, {_num(radius_deg)})\n"
        f"  AND (({bitmask_column} & {mask}) != 0)\n"
        + _limit_clause(row_limit, cap_reason)
    )
    meta = _meta("bitmask_select", info, spatial_bound=True, row_limit=row_limit)
    return sql, _flag_platform_cap(meta, row_limit, cap_reason)


def build_zhistogram(
    catalog: str,
    table: str,
    *,
    z_column: str = "z",
    bin_width: float = 0.01,
    bin: Optional[float] = None,
    limit: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    info = _table_info(catalog, table)
    z_col = _column(info, z_column)
    width_value = bin_width if bin is None else bin
    width = _positive(width_value, "bin_width")
    row_limit, cap_reason = _resolve_limit(limit, maximum=MAX_ROW_LIMIT, default=MAX_ROW_LIMIT)
    sql = (
        f"SELECT ROUND(({z_col} / {_num(width)})::numeric, 0) * {_num(width)} AS z_bin,\n"
        f"       COUNT(*) AS source_count\n"
        f"FROM {info['qualified_name']}\n"
        f"WHERE {z_col} IS NOT NULL AND {z_col} {_NAN_GUARD}\n"
        f"GROUP BY z_bin\n"
        f"ORDER BY z_bin\n"
        + _limit_clause(row_limit, cap_reason)
    )
    meta = _meta("zhistogram", info, aggregate=True, row_limit=row_limit)
    return sql, _flag_platform_cap(meta, row_limit, cap_reason)


def build_footprint_aggregate(catalog: str, table: str, *, limit: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
    info = _table_info(catalog, table)
    row_limit, cap_reason = _resolve_limit(limit, maximum=MAX_ROW_LIMIT, default=MAX_ROW_LIMIT)
    ra_col, dec_col = info["ra_column"], info["dec_column"]
    sql = (
        f"SELECT ROUND({ra_col}) AS ra_deg, ROUND({dec_col}) AS dec_deg, COUNT(*) AS source_count\n"
        f"FROM {info['qualified_name']}\n"
        f"GROUP BY ra_deg, dec_deg\n"
        f"ORDER BY source_count DESC\n"
        + _limit_clause(row_limit, cap_reason)
    )
    meta = _meta("footprint_aggregate", info, aggregate=True, row_limit=row_limit)
    return sql, _flag_platform_cap(meta, row_limit, cap_reason)


def build_sed_select(
    *,
    ra: float,
    dec: float,
    radius_deg: float,
    limit: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    # Forced unWISE W1/W2 only — W3/W4 are NOT in ls_dr9.tractor dered_mag_* (guardrail).
    columns = [
        "ra", "dec", "type", "dered_mag_g", "dered_mag_r", "dered_mag_z",
        "dered_mag_w1", "dered_mag_w2",
    ]
    sql, meta = build_cone_select("ls_dr9", "tractor", ra=ra, dec=dec, radius_deg=radius_deg, columns=columns, limit=limit)
    meta["builder"] = "sed_select"
    return sql, meta


# Multi-epoch tables (per-epoch photometry rows) usable for variability work.
_EPOCH_TABLE_COLUMNS = {"id", "mjd", "cmag", "cerr"}


def _epoch_table_info(catalog: str) -> Dict[str, Any]:
    """Registry info for a catalog's multi-epoch `source` table, or a clear error."""
    info = _table_info(catalog, "source")
    missing = _EPOCH_TABLE_COLUMNS - set(info.get("columns") or [])
    if missing:
        raise ValueError(
            f"{info['qualified_name']} is not a multi-epoch photometry table "
            f"(missing columns: {sorted(missing)}). Use a SMASH source table."
        )
    return info


def _band_predicate(band: Optional[str]) -> str:
    """Optional single-band filter clause for SMASH-style `filter` columns."""
    if band is None or not str(band).strip():
        return ""
    text = str(band).strip().lower()
    if not re.fullmatch(r"[a-z0-9]{1,8}", text):
        raise ValueError(f"Invalid band {band!r}; use a single filter name like 'g'.")
    return f"\n  AND filter = '{text}'"


def build_variability_rank(
    catalog: str = "smash_dr1",
    *,
    ra: float,
    dec: float,
    radius_deg: float,
    band: Optional[str] = None,
    min_epochs: int = 10,
    limit: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Per-object variability ranking over a multi-epoch cone (the blog's
    'select high-variability stars' step): scatter, amplitude, and a
    scatter-to-error significance ratio, ranked most-variable first."""
    info = _epoch_table_info(catalog)
    _validate_sky(float(ra), float(dec), float(radius_deg))
    epochs = int(min_epochs) if min_epochs else 10
    if epochs < 2:
        raise ValueError("min_epochs must be >= 2 (variability needs repeat epochs)")
    row_limit, cap_reason = _resolve_limit(limit, default=100)
    band_clause = _band_predicate(band)
    group_by_filter = not bool(band_clause)
    filter_column = "       filter,\n" if group_by_filter else ""
    # Wrap-safe per-object mean RA: plain AVG(ra) across the RA=0/360 seam
    # averages epochs at 359.999° and 0.001° to ~180°, so the reported position
    # and dist_arcsec were nonsense and nearest-candidate selection rejected
    # the true target (variability-avg-ra-wrap). Circular mean of the unit
    # vectors; CASE re-ranges atan2's (-180,180] output to [0,360).
    mean_ra = "degrees(atan2(AVG(sin(radians(ra))), AVG(cos(radians(ra)))))"
    mean_ra = f"CASE WHEN {mean_ra} < 0 THEN {mean_ra} + 360.0 ELSE {mean_ra} END"
    sql = (
        f"SELECT id,\n"
        f"{filter_column}"
        f"       {mean_ra} AS ra, AVG(dec) AS dec,\n"
        # Distance from the cone center: when the user gave an exact target
        # position, the right star is the NEAREST candidate, not the most
        # variable one (live P14: the model folded a variable 2.2' away).
        f"       q3c_dist({mean_ra}, AVG(dec), {_num(float(ra))}, {_num(float(dec))}) * 3600.0 AS dist_arcsec,\n"
        f"       COUNT(*) AS nepochs,\n"
        f"       AVG(cmag) AS mean_mag,\n"
        f"       STDDEV(cmag) AS mag_rms,\n"
        f"       MAX(cmag) - MIN(cmag) AS amplitude,\n"
        f"       AVG(cerr) AS mean_err,\n"
        f"       STDDEV(cmag) / NULLIF(AVG(cerr), 0) AS var_snr\n"
        f"FROM {info['qualified_name']}\n"
        f"WHERE q3c_radial_query(ra, dec, {_num(float(ra))}, {_num(float(dec))}, {_num(float(radius_deg))})\n"
        # Sentinel padding (99/-99) marks MISSING epochs — the unified
        # SENTINEL_MAG_RANGE guard, annotated so it never reads as science.
        f"  AND cmag > {_num(SENTINEL_MAG_RANGE[0])} AND cmag < {_num(SENTINEL_MAG_RANGE[1])} "
        f"{SENTINEL_GUARD_COMMENT}{band_clause}\n"
        f"GROUP BY id{', filter' if group_by_filter else ''}\n"
        f"HAVING COUNT(*) >= {epochs}\n"
        f"ORDER BY var_snr DESC NULLS LAST\n"
        + _limit_clause(row_limit, cap_reason)
    )
    meta = _meta(
        "variability_rank", info,
        spatial_bound=True, aggregate=True, row_limit=row_limit,
        min_epochs=epochs, grouped_by_filter=group_by_filter,
        **({"band": str(band).strip().lower()} if band_clause else {}),
    )
    return sql, _flag_platform_cap(meta, row_limit, cap_reason)


def build_variable_star_select(
    *,
    catalog: str = "smash_dr1",
    source_id: Optional[str] = None,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    limit: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    info = _epoch_table_info(catalog)
    row_limit, cap_reason = _resolve_limit(limit)
    cols = _select_columns(info, ["id", "ra", "dec", "mjd", "filter", "cmag", "cerr"])
    exact_id_bound = False
    if source_id:
        safe_id = str(source_id).replace("'", "''")
        where = f"id = '{safe_id}'"
        spatial = False
        exact_id_bound = True
    else:
        if ra is None or dec is None:
            raise ValueError("Provide source_id or both ra and dec")
        _validate_sky(float(ra), float(dec), 1.0 / 3600.0)
        where = f"q3c_radial_query(ra, dec, {_num(float(ra))}, {_num(float(dec))}, {_num(1.0 / 3600.0)})"
        spatial = True
    sql = (
        f"SELECT {cols}\n"
        f"FROM {info['qualified_name']}\n"
        f"WHERE {where}\n"
        # Unified sentinel guard (was an unexplained `cmag < 99`, which even
        # let -99 sentinels through) — data validity, not science.
        f"  AND cmag > {_num(SENTINEL_MAG_RANGE[0])} AND cmag < {_num(SENTINEL_MAG_RANGE[1])} "
        f"{SENTINEL_GUARD_COMMENT}\n"
        f"ORDER BY mjd\n"
        + _limit_clause(row_limit, cap_reason)
    )
    meta = _meta("variable_star_select", info, spatial_bound=spatial, row_limit=row_limit)
    if exact_id_bound:
        meta["exact_id_bound"] = True
    return sql, _flag_platform_cap(meta, row_limit, cap_reason)


def _table_info(catalog: str, table: str) -> Dict[str, Any]:
    return registry.describe_table(catalog, table)


def _meta(builder: str, info: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    meta = {
        "source": "builder",
        "builder": builder,
        "catalog": info["catalog"],
        "table": info["table"],
        "qualified_name": info["qualified_name"],
        "aggregate": bool(extra.pop("aggregate", False)),
        "spatial_bound": bool(extra.pop("spatial_bound", False)),
    }
    meta.update(extra)
    return meta


def _column_list(info: Dict[str, Any], columns: Optional[Sequence[str]], *, required: Optional[Sequence[str]] = None) -> List[str]:
    requested = list(columns or list(info.get("columns") or []))
    for col in required or []:
        if col not in requested:
            requested.insert(0, col)
    result = []
    for col in requested:
        clean = _column(info, col)  # permissive: keeps any safe identifier, not only registered ones
        if clean not in result:
            result.append(clean)
    if not result:
        raise ValueError("No valid columns selected")
    return result


def _select_columns(info: Dict[str, Any], columns: Optional[Sequence[str]], *, required: Optional[Sequence[str]] = None) -> str:
    return ", ".join(_column_list(info, columns, required=required))


def _prefixed_columns(prefix: str, columns: Iterable[str]) -> str:
    return ",\n       ".join(f"{prefix}.{col} AS {prefix}_{col}" for col in columns)


def _column(info: Dict[str, Any], column: str) -> str:
    text = str(column or "").strip().lower()
    allowed = set(info.get("columns") or [])
    if text in allowed:
        return text
    # Not in the seed registry — accept any safe identifier and let Data Lab validate it,
    # instead of hard-failing the task (the registry can't list every column of a catalog).
    if _SAFE_IDENT_RE.match(text):
        return text
    raise ValueError(f"Invalid column identifier {column!r} for {info['qualified_name']}")


def _healpix_column(info: Dict[str, Any], column: Optional[str]) -> str:
    healpix = [item["name"] for item in info.get("healpix_columns") or []]
    selected = str(column or (healpix[0] if healpix else "")).strip().lower()
    if selected not in healpix:
        raise ValueError(f"No registered HEALPix column {column!r} for {info['qualified_name']}")
    return selected


def _bit_value(info: Dict[str, Any], column: str, bit_name: str) -> int:
    col = _column(info, column)
    bitmasks = registry.DATALAB_CATALOGS[info["catalog"]].get("bitmasks", {})
    bits = bitmasks.get(col) or {}
    key = str(bit_name or "").strip().upper()
    if key not in bits:
        raise ValueError(f"Unknown bit {bit_name!r} for {info['qualified_name']}.{col}")
    return int(bits[key])


def _validate_sky(ra: float, dec: float, radius_deg: float) -> None:
    _finite(ra, "ra")
    _finite(dec, "dec")
    _positive(radius_deg, "radius_deg")
    if float(radius_deg) > MAX_CONE_RADIUS_DEG:
        raise ValueError(
            f"radius_deg {radius_deg} exceeds the max cone radius {MAX_CONE_RADIUS_DEG} deg; "
            "tile the region into smaller cones (q3c_radial_query/q3c_poly_query) instead"
        )
    if not 0.0 <= float(ra) < 360.0:
        raise ValueError("ra must be in [0, 360) degrees")
    if not -90.0 <= float(dec) <= 90.0:
        raise ValueError("dec must be in [-90, 90] degrees")


def _validate_rect(ra_min: float, ra_max: float, dec_min: float, dec_max: float) -> None:
    for name, value in (("ra_min", ra_min), ("ra_max", ra_max), ("dec_min", dec_min), ("dec_max", dec_max)):
        _finite(value, name)
    # ra_min > ra_max is a legal box wrapping through RA=0/360 (e.g. 358°→2°,
    # ra-wrap-rect-footprint-unsupported); only a degenerate zero-width box is
    # rejected.
    if not 0.0 <= float(ra_min) < 360.0 or not 0.0 <= float(ra_max) <= 360.0 or float(ra_min) == float(ra_max):
        raise ValueError(
            "RA bounds must satisfy 0 <= ra_min, ra_max <= 360 with ra_min != ra_max "
            "(ra_min > ra_max means the box wraps through RA=0/360)"
        )
    if not -90.0 <= float(dec_min) < float(dec_max) <= 90.0:
        raise ValueError("Dec bounds must satisfy -90 <= dec_min < dec_max <= 90")


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive(value: float, name: str) -> float:
    number = _finite(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _resolve_limit(
    value: Optional[int],
    *,
    maximum: int = MAX_ROW_LIMIT,
    default: int = DEFAULT_ROW_LIMIT,
) -> Tuple[int, Optional[str]]:
    """(row_limit, platform_reason) for a caller-supplied limit.

    platform_reason is None when the caller's explicit, in-range limit was used
    verbatim (their choice, their provenance). It names WHY the platform picked
    the number when the caller passed None (auto default) or a value above the
    ceiling (clamped) — those caps must be flagged, never read as science
    (RE-B1). 0/negative stay explicit errors (do NOT silently fall back to the
    default — that would hide an intended LIMIT 0).
    """
    if value is None:
        return int(default), "no row limit was requested; automatic platform cap applied"
    number = int(value)
    if number <= 0:
        raise ValueError("limit must be positive")
    if number > int(maximum):
        return int(maximum), f"requested limit {number} exceeds the platform ceiling {int(maximum)}; clamped"
    return number, None


def _limit_clause(row_limit: int, platform_reason: Optional[str]) -> str:
    """The SQL LIMIT text — self-describing when the platform chose the cap."""
    if platform_reason:
        return f"LIMIT {int(row_limit)} {PLATFORM_ROW_CAP_COMMENT}"
    return f"LIMIT {int(row_limit)}"


def _flag_platform_cap(meta: Dict[str, Any], row_limit: int, platform_reason: Optional[str]) -> Dict[str, Any]:
    """Stamp platform-applied caps into meta (platform_row_cap + warning).

    The warning rides meta['warnings'] into the governor's validated warnings,
    so every result built from this SQL discloses that the cap is platform cost
    governance, not a science choice (RE-B1).
    """
    if platform_reason:
        meta["platform_row_cap"] = int(row_limit)
        meta.setdefault("warnings", []).append(
            f"Row cap LIMIT {int(row_limit)} applied by the platform ({platform_reason}) — "
            "cost governance, not a science cut and not user-requested. If it truncates the "
            "result, disclose the cap and offer the uncapped path (async_submit background "
            "job or a server-side aggregate builder)."
        )
    return meta


def _flag_named_platform_cap(
    meta: Dict[str, Any],
    *,
    name: str,
    row_limit: int,
    platform_reason: Optional[str],
    what: str,
) -> Dict[str, Any]:
    """Stamp a NAMED platform-chosen budget (crossmatch small-side CTE cap,
    orchestration-derived candidate budgets, ...) into meta.

    Named budgets accumulate under ``meta['platform_row_caps']``
    (e.g. ``{"small_side": 10000}``) so the scalar ``meta['platform_row_cap']``
    — consumed as an int by every provenance stamper — keeps its outer-LIMIT
    meaning. Same disclosure contract as :func:`_flag_platform_cap`: platform
    cost governance, never a science cut (RE-B1 / guard CX-01, CX-09).
    """
    if platform_reason:
        meta.setdefault("platform_row_caps", {})[str(name)] = int(row_limit)
        meta.setdefault("warnings", []).append(
            f"{what} LIMIT {int(row_limit)} applied by the platform ({platform_reason}) — "
            "cost governance, not a science cut and not user-requested. Disclose it if it "
            "could truncate the result."
        )
    return meta


def _sentinel_mag_predicates(catalog: str, table: str, columns: Sequence[str]) -> List[str]:
    """Registry-validated sentinel-removal predicates for magnitude columns.

    One guard pair per column, with the LAST predicate carrying the
    self-describing SQL comment so provenance readers see data validity, not a
    saturation/noise-floor science cut (RE-B2).
    """
    lo, hi = SENTINEL_MAG_RANGE
    cuts: List[Dict[str, Any]] = []
    for col in columns:
        cuts.append({"column": col, "op": ">", "value": lo})
        cuts.append({"column": col, "op": "<", "value": hi})
    preds = build_catalog_predicates(catalog, table, value_cuts=cuts)
    if preds:
        preds[-1] = f"{preds[-1]} {SENTINEL_GUARD_COMMENT}"
    return preds


_MORPH_BAND_SUFFIX_RE = re.compile(r"_(?:u|g|r|i|z|y|vr|j|h|k|ks|w1|w2|w3|w4)$")


def _morph_family(column: str) -> str:
    """Column family for morphology comparison: per-band suffixes stripped so
    spread_model_g compares against the registered spread_model_r convention."""
    return _MORPH_BAND_SUFFIX_RE.sub("", str(column or "").strip().lower())


def _morph_scale(cut: Mapping[str, Any]) -> Optional[float]:
    """Characteristic |threshold| of a morphology cut, or None when it has no
    meaningful continuous scale (discrete 'in' codes, zero thresholds)."""
    try:
        if cut.get("between"):
            lo, hi = cut["between"]
            scale = max(abs(float(lo)), abs(float(hi)))
        elif cut.get("op") and cut.get("value") is not None:
            scale = abs(float(cut["value"]))
        else:
            return None
    except (TypeError, ValueError):
        return None
    return scale if scale > 0 else None


def morphology_deviation_warning(
    catalog: str,
    table: str,
    morphology: Optional[Mapping[str, Any]],
) -> Optional[str]:
    """LOUD warning when a caller-supplied morphology threshold is orders of
    magnitude off the table's registered star/galaxy convention (RE-B3).

    The classic failure: transplanting class_star's 0.5 onto spread_model
    (DES convention |spread_model_r| < 0.003 — a 167x deviation). The cut is
    executed as given (the user may genuinely want it); it is never clamped.
    Returns None when the table has no registered convention, the columns are
    different families, or the values are within MORPHOLOGY_DEVIATION_FACTOR.
    """
    if not morphology:
        return None
    try:
        reference = registry.point_source_cut(catalog, table)
    except Exception:  # noqa: BLE001 - unknown table => nothing to compare against
        return None
    if not reference:
        return None
    col = str(morphology.get("column") or "").strip().lower()
    ref_col = str(reference.get("column") or "").strip().lower()
    if not col or not ref_col or _morph_family(col) != _morph_family(ref_col):
        return None
    supplied = _morph_scale(morphology)
    expected = _morph_scale(reference)
    if supplied is None or expected is None:
        return None
    ratio = supplied / expected
    if 1.0 / MORPHOLOGY_DEVIATION_FACTOR < ratio < MORPHOLOGY_DEVIATION_FACTOR:
        return None
    fold = ratio if ratio >= 1 else 1.0 / ratio
    if reference.get("between"):
        lo, hi = reference["between"]
        ref_desc = f"{ref_col} BETWEEN {_num(float(lo))} AND {_num(float(hi))}"
    else:
        ref_desc = f"{ref_col} {reference.get('op')} {_num(float(reference['value']))}"
    return (
        f"MORPHOLOGY THRESHOLD CHECK: the supplied cut {col} at {supplied:g} deviates ~{fold:.0f}x "
        f"from the registered {catalog}.{table} star/galaxy convention ({ref_desc}). Thresholds near "
        "0.5 are a class_star-style classifier convention and are almost certainly WRONG for "
        "spread_model-style columns (DES uses |spread_model_r| < 0.003). The cut was executed as "
        "given (NOT clamped) — verify the threshold before trusting the star/galaxy split."
    )


def _num(value: float) -> str:
    return format(float(value), ".15g")


__all__ = [
    "MAX_ROW_LIMIT",
    "MORPHOLOGY_DEVIATION_FACTOR",
    "PLATFORM_ROW_CAP_COMMENT",
    "SENTINEL_GUARD_COMMENT",
    "SENTINEL_MAG_RANGE",
    "morphology_deviation_warning",
    "build_bitmask_select",
    "build_catalog_predicates",
    "build_cone_count",
    "build_cone_select",
    "build_density_aggregate",
    "build_footprint_aggregate",
    "build_q3c_crossmatch",
    "build_rectangular_region_select",
    "build_sed_select",
    "build_variability_rank",
    "build_variable_star_select",
    "build_zhistogram",
]
