"""Pure SQL builders for NOIRLab Astro Data Lab P0 catalog tools."""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from services import datalab_registry as registry

# A safe SQL column identifier. The registry's column lists are NOT exhaustive (catalogs have
# hundreds of columns), so we accept any safe identifier and let Data Lab validate it server-side
# rather than hard-failing a whole task on an unlisted-but-real column.
_SAFE_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


DEFAULT_ROW_LIMIT = 500
MAX_ROW_LIMIT = 5000
# Cap cone radius so an all-sky cone (radius_deg=180/360) can't seq-scan a catalog.
MAX_CONE_RADIUS_DEG = float(os.getenv("DATALAB_MAX_CONE_RADIUS_DEG", "30"))
# Whitelisted comparison operators for structured selection cuts.
_CUT_OPS = {"<", ">", "<=", ">=", "=", "!="}
# Operator spellings LLMs frequently emit that mean the same thing: Python/JS
# equality "==" and SQL-standard not-equal "<>".
_CUT_OP_ALIASES = {"==": "=", "<>": "!="}


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
        return _num(float(value))
    except (TypeError, ValueError):
        pass
    if op not in ("=", "!="):
        raise ValueError(f"value-cut operator {op!r} requires a numeric value, got {value!r}")
    text = str(value).strip()
    # Tolerate a value the model already wrapped in quotes.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1]
    if not text or len(text) > 128 or any(ord(ch) < 32 for ch in text):
        raise ValueError(f"invalid string value for value cut: {value!r}")
    return "'" + text.replace("'", "''") + "'"


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
    limit: int = DEFAULT_ROW_LIMIT,
    predicates: Optional[Sequence[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    info = _table_info(catalog, table)
    ra_col, dec_col = info["ra_column"], info["dec_column"]
    _validate_sky(ra, dec, radius_deg)
    row_limit = _limit(limit)
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
        f"LIMIT {row_limit}"
    )
    return sql, _meta("cone_select", info, spatial_bound=True, row_limit=row_limit)


def build_rectangular_region_select(
    catalog: str,
    table: str,
    *,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    columns: Optional[Sequence[str]] = None,
    limit: int = DEFAULT_ROW_LIMIT,
) -> Tuple[str, Dict[str, Any]]:
    info = _table_info(catalog, table)
    _validate_rect(ra_min, ra_max, dec_min, dec_max)
    row_limit = _limit(limit)
    select_cols = _select_columns(info, columns)
    ra_col, dec_col = info["ra_column"], info["dec_column"]
    warnings: List[str] = []
    if info["region_strategy"] == "box_ok":
        where = (
            f"{ra_col} BETWEEN {_num(ra_min)} AND {_num(ra_max)} "
            f"AND {dec_col} BETWEEN {_num(dec_min)} AND {_num(dec_max)}"
        )
        warnings.append("Used registry-approved RA/Dec BETWEEN box for this table.")
        spatial = "box"
    else:
        where = (
            f"q3c_poly_query({ra_col}, {dec_col}, "
            f"ARRAY[{_num(ra_min)}, {_num(ra_max)}, {_num(ra_max)}, {_num(ra_min)}], "
            f"ARRAY[{_num(dec_min)}, {_num(dec_min)}, {_num(dec_max)}, {_num(dec_max)}])"
        )
        spatial = "q3c_poly_query"
    sql = f"SELECT {select_cols}\nFROM {info['qualified_name']}\nWHERE {where}\nLIMIT {row_limit}"
    meta = _meta("rectangular_region_select", info, spatial_bound=True, row_limit=row_limit)
    meta.update({"spatial_strategy": spatial, "warnings": warnings})
    return sql, meta


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
    limit: int = MAX_ROW_LIMIT,
) -> Tuple[str, Dict[str, Any]]:
    info = _table_info(catalog, table)
    row_limit = _limit(limit, maximum=MAX_ROW_LIMIT)
    ra_col, dec_col = info["ra_column"], info["dec_column"]
    # Require an explicit region: an unbounded GROUP BY over a billion-row catalog
    # is a full-catalog scan. Give a cone, or opt in explicitly with all_sky=True.
    has_cone = ra is not None and dec is not None and radius_deg is not None
    warnings: List[str] = []
    clauses: List[str] = []
    if has_cone:
        _validate_sky(float(ra), float(dec), float(radius_deg))
        clauses.append(f"q3c_radial_query({ra_col}, {dec_col}, {_num(float(ra))}, {_num(float(dec))}, {_num(float(radius_deg))})")
    elif all_sky:
        warnings.append("Unbounded all-sky aggregate: scans the whole catalog server-side and may be slow/expensive.")
    else:
        raise ValueError(
            "density aggregate requires a cone (ra, dec, radius_deg) or an explicit all_sky=True "
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
            f"LIMIT {row_limit}"
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
            f"LIMIT {row_limit}"
        )
    else:
        raise ValueError("density aggregate mode must be grid or healpix")
    meta = _meta("density_aggregate", info, aggregate=True, spatial_bound=has_cone, row_limit=row_limit)
    meta["warnings"] = warnings
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
                {"column": "class_star", "op": ">", "value": 0.5} |
                {"column": "ext_coadd", "in": [0, 1]}
    Every column is checked against the registry; operators are whitelisted; values are
    numeric-formatted — so the output is safe to append to a builder WHERE clause.
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
            preds.append(f"{expr} >= {_num(float(color_cut['min']))}")
        if color_cut.get("max") is not None:
            preds.append(f"{expr} <= {_num(float(color_cut['max']))}")
    for vc in (value_cuts or []):
        col = _column(info, vc["column"])
        op = _normalize_cut_op(vc.get("op", ""))
        preds.append(f"{col} {op} {_cut_rhs(op, vc['value'])}")
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
            preds.append(f"{col} {op} {_cut_rhs(op, morphology['value'])}")
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
    small_limit: int = 10000,
    limit: int = DEFAULT_ROW_LIMIT,
) -> Tuple[str, Dict[str, Any]]:
    small = _table_info(small_catalog, small_table)
    big = _table_info(big_catalog, big_table)
    _validate_sky(ra, dec, radius_deg)
    match_radius_deg = _positive(match_radius_arcsec, "match_radius_arcsec") / 3600.0
    small_row_limit = _limit(small_limit, maximum=50000)
    row_limit = _limit(limit)
    small_select = _select_columns(small, small_columns, required=[small["ra_column"], small["dec_column"]])
    big_select = _prefixed_columns("big", _column_list(big, big_columns, required=[big["ra_column"], big["dec_column"]]))
    sra, sdec = small["ra_column"], small["dec_column"]
    bra, bdec = big["ra_column"], big["dec_column"]
    sql = (
        "WITH g AS MATERIALIZED (\n"
        f"    SELECT {small_select}\n"
        f"    FROM {small['qualified_name']}\n"
        f"    WHERE q3c_radial_query({sra}, {sdec}, {_num(ra)}, {_num(dec)}, {_num(radius_deg)})\n"
        f"    LIMIT {small_row_limit}\n"
        ")\n"
        "SELECT g.*,\n"
        f"       {big_select}\n"
        "FROM g\n"
        f"JOIN {big['qualified_name']} AS big\n"
        f"  ON q3c_join(g.{sra}, g.{sdec}, big.{bra}, big.{bdec}, {_num(match_radius_deg)})\n"
        f"LIMIT {row_limit}"
    )
    meta = _meta("q3c_crossmatch", small, spatial_bound=True, row_limit=row_limit)
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
    limit: int = DEFAULT_ROW_LIMIT,
) -> Tuple[str, Dict[str, Any]]:
    info = _table_info(catalog, table)
    _validate_sky(ra, dec, radius_deg)
    bit = _bit_value(info, bitmask_column, bit_name)
    row_limit = _limit(limit)
    select_cols = _select_columns(info, columns)
    ra_col, dec_col = info["ra_column"], info["dec_column"]
    mask = 1 << bit
    sql = (
        f"SELECT {select_cols}\n"
        f"FROM {info['qualified_name']}\n"
        f"WHERE q3c_radial_query({ra_col}, {dec_col}, {_num(ra)}, {_num(dec)}, {_num(radius_deg)})\n"
        f"  AND (({bitmask_column} & {mask}) != 0)\n"
        f"LIMIT {row_limit}"
    )
    return sql, _meta("bitmask_select", info, spatial_bound=True, row_limit=row_limit)


def build_zhistogram(
    catalog: str,
    table: str,
    *,
    z_column: str = "z",
    bin_width: float = 0.01,
    bin: Optional[float] = None,
    limit: int = MAX_ROW_LIMIT,
) -> Tuple[str, Dict[str, Any]]:
    info = _table_info(catalog, table)
    z_col = _column(info, z_column)
    width_value = bin_width if bin is None else bin
    width = _positive(width_value, "bin_width")
    row_limit = _limit(limit, maximum=MAX_ROW_LIMIT)
    sql = (
        f"SELECT ROUND(({z_col} / {_num(width)})::numeric, 0) * {_num(width)} AS z_bin,\n"
        f"       COUNT(*) AS source_count\n"
        f"FROM {info['qualified_name']}\n"
        f"WHERE {z_col} IS NOT NULL\n"
        f"GROUP BY z_bin\n"
        f"ORDER BY z_bin\n"
        f"LIMIT {row_limit}"
    )
    return sql, _meta("zhistogram", info, aggregate=True, row_limit=row_limit)


def build_footprint_aggregate(catalog: str, table: str, *, limit: int = MAX_ROW_LIMIT) -> Tuple[str, Dict[str, Any]]:
    info = _table_info(catalog, table)
    row_limit = _limit(limit, maximum=MAX_ROW_LIMIT)
    ra_col, dec_col = info["ra_column"], info["dec_column"]
    sql = (
        f"SELECT ROUND({ra_col}) AS ra_deg, ROUND({dec_col}) AS dec_deg, COUNT(*) AS source_count\n"
        f"FROM {info['qualified_name']}\n"
        f"GROUP BY ra_deg, dec_deg\n"
        f"ORDER BY source_count DESC\n"
        f"LIMIT {row_limit}"
    )
    return sql, _meta("footprint_aggregate", info, aggregate=True, row_limit=row_limit)


def build_sed_select(
    *,
    ra: float,
    dec: float,
    radius_deg: float,
    limit: int = DEFAULT_ROW_LIMIT,
) -> Tuple[str, Dict[str, Any]]:
    # Forced unWISE W1/W2 only — W3/W4 are NOT in ls_dr9.tractor dered_mag_* (guardrail).
    columns = [
        "ra", "dec", "type", "dered_mag_g", "dered_mag_r", "dered_mag_z",
        "dered_mag_w1", "dered_mag_w2",
    ]
    sql, meta = build_cone_select("ls_dr9", "tractor", ra=ra, dec=dec, radius_deg=radius_deg, columns=columns, limit=limit)
    meta["builder"] = "sed_select"
    return sql, meta


def build_variable_star_select(
    *,
    source_id: Optional[str] = None,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    limit: int = DEFAULT_ROW_LIMIT,
) -> Tuple[str, Dict[str, Any]]:
    info = _table_info("smash_dr1", "source")
    row_limit = _limit(limit)
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
        f"FROM smash_dr1.source\n"
        f"WHERE {where}\n"
        f"  AND cmag < 99\n"
        f"ORDER BY mjd\n"
        f"LIMIT {row_limit}"
    )
    meta = _meta("variable_star_select", info, spatial_bound=spatial, row_limit=row_limit)
    if exact_id_bound:
        meta["exact_id_bound"] = True
    return sql, meta


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
    if not 0.0 <= float(ra_min) < 360.0 or not 0.0 <= float(ra_max) <= 360.0 or float(ra_min) >= float(ra_max):
        raise ValueError("RA bounds must satisfy 0 <= ra_min < ra_max <= 360")
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


def _limit(value: int, *, maximum: int = MAX_ROW_LIMIT) -> int:
    # Treat None as the default, but 0 / negative are explicit errors (do NOT
    # silently fall back to the default — that would hide an intended LIMIT 0).
    number = DEFAULT_ROW_LIMIT if value is None else int(value)
    if number <= 0:
        raise ValueError("limit must be positive")
    return min(number, maximum)


def _num(value: float) -> str:
    return format(float(value), ".15g")


__all__ = [
    "MAX_ROW_LIMIT",
    "build_bitmask_select",
    "build_catalog_predicates",
    "build_cone_count",
    "build_cone_select",
    "build_density_aggregate",
    "build_footprint_aggregate",
    "build_q3c_crossmatch",
    "build_rectangular_region_select",
    "build_sed_select",
    "build_variable_star_select",
    "build_zhistogram",
]
