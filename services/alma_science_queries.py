"""Deterministic ALMA archive query helpers for science questions.

The agent owns network access to TAP/ALMiner.  This module keeps the
science-specific filtering, grouping, and line-coverage logic isolated so
LLM routing can call one tool and receive auditable tables/counts.
"""

from __future__ import annotations

import math
import re
from datetime import date
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


LINE_REST_FREQ_GHZ: Dict[str, float] = {
    "12CO(1-0)": 115.2712018,
    "12CO(2-1)": 230.5380000,
    "12CO(3-2)": 345.7959899,
    "13CO(1-0)": 110.201354,
    "13CO(2-1)": 220.398684,
    "13CO(3-2)": 330.587965,
    "C18O(1-0)": 109.782173,
    "C18O(2-1)": 219.560354,
    "C18O(3-2)": 329.330552,
    "CO(1-0)": 115.2712018,
    "CO(2-1)": 230.5380000,
    "CO(3-2)": 345.7959899,
    "CO(4-3)": 461.0407682,
    "CO(5-4)": 576.2679305,
    "CO(6-5)": 691.4730763,
    "CO(7-6)": 806.6518060,
    "CO(8-7)": 921.7997000,
    # Dense-gas tracers (CDMS rest frequencies), low-J ladders.
    "HCN(1-0)": 88.6316022,
    "HCN(2-1)": 177.2611115,
    "HCN(3-2)": 265.8864343,
    "HCN(4-3)": 354.5054779,
    "HCO+(1-0)": 89.1885247,
    "HCO+(2-1)": 178.3750563,
    "HCO+(3-2)": 267.5576259,
    "HCO+(4-3)": 356.7342230,
    "HNC(1-0)": 90.6635680,
    "HNC(2-1)": 181.3247580,
    "HNC(3-2)": 271.9811420,
    "HNC(4-3)": 362.6303030,
    "CS(2-1)": 97.9809533,
    "CS(3-2)": 146.9690287,
    "CS(5-4)": 244.9355565,
    "CS(7-6)": 342.8828503,
    # Atomic fine-structure lines of high-z work (laboratory rest
    # frequencies, CDMS/JPL). Redshifted into the ALMA bands for z >~ 1-9.
    "[CII]158um": 1900.536900,
    "[NII]205um": 1461.131406,
    "[NII]122um": 2459.380100,
    "[OIII]88um": 3393.006244,
    "[OIII]52um": 5785.879590,
    "[OI]63um": 4744.777490,
    "[OI]145um": 2060.068860,
    "[CI](1-0)": 492.160651,
    "[CI](2-1)": 809.341970,
}

# Bracketed / bare species tokens -> the transitions they mean (the router's
# _SPECIES_RE recognises "[C II]", "CII", "[OIII]" ...; guard CX-29: they used
# to fall back silently to the CO ladder while the answer said [C II]).
FINE_STRUCTURE_SPECIES: Dict[str, List[str]] = {
    "CII": ["[CII]158um"],
    "NII": ["[NII]205um", "[NII]122um"],
    "OIII": ["[OIII]88um", "[OIII]52um"],
    "OI": ["[OI]63um", "[OI]145um"],
    "CI": ["[CI](1-0)", "[CI](2-1)"],
}


class UnsupportedSpecies(ValueError):
    """A named rest species with no rest-frequency entry: never substitute CO."""


def _fine_structure_key(token: str) -> Optional[str]:
    t = str(token or "").replace(" ", "").upper()
    if t.startswith("[") and t.endswith("]"):
        t = t[1:-1]
    return t if t in FINE_STRUCTURE_SPECIES else None

CO_LADDER_LINES = [
    "CO(1-0)",
    "CO(2-1)",
    "CO(3-2)",
    "CO(4-3)",
    "CO(5-4)",
    "CO(6-5)",
    "CO(7-6)",
    "CO(8-7)",
]

UNIT_TO_GHZ = {
    "ghz": 1.0,
    "mhz": 1e-3,
    "khz": 1e-6,
    "hz": 1e-9,
}

FREQUENCY_RANGE_RE = re.compile(
    r"(?P<lo>\d+(?:\.\d+)?)\s*(?P<lo_unit>GHz|MHz|kHz|Hz)?"
    r"\s*(?:\.\.|-|to)\s*"
    r"(?P<hi>\d+(?:\.\d+)?)\s*(?P<hi_unit>GHz|MHz|kHz|Hz)?",
    re.IGNORECASE,
)

# In ALMA frequency_support each SPW range is immediately followed by its
# channel resolution, e.g. ``[86.24..88.11GHz, 976.56kHz, ...]``. This captures
# that resolution token when it directly follows a parsed range.
RESOLUTION_AFTER_RANGE_RE = re.compile(
    r"\s*,\s*(?P<res>\d+(?:\.\d+)?)\s*(?P<unit>GHz|MHz|kHz|Hz)",
    re.IGNORECASE,
)
_RES_UNIT_TO_KHZ = {"ghz": 1e6, "mhz": 1e3, "khz": 1.0, "hz": 1e-3}


def as_text(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    return "" if text.lower() == "nan" else text


def escape_adql(value: str) -> str:
    return as_text(value).replace("'", "''")


def normalize_target_alias(target: str) -> str:
    """Normalize whitespace; let the resolver interpret catalog identifiers."""
    clean = as_text(target)
    return re.sub(r"\s+", " ", clean)


# alma-cycle-prefix-wrong-pre-cycle8: the cycle→proposal-year relation is NOT
# linear below Cycle 8, so an explicit lookup is required. Irregularities:
# Cycle 0 was the 2011 Early Science call (project codes 2011.0.*, suffix .0.);
# Cycle 2 spanned longer than a year, so there was NO 2014 proposal call
# (Cycle 3 jumps to 2015); the 2020 call was cancelled because of COVID, so
# Cycle 8 became the 2021 call (no 2020 proposal year). year = cycle + 2013
# holds only from Cycle 8 onward.
_CYCLE_TO_PROPOSAL_YEAR: Dict[int, int] = {
    0: 2011,
    1: 2012,
    2: 2013,
    3: 2015,
    4: 2016,
    5: 2017,
    6: 2018,
    7: 2019,
    8: 2021,
    9: 2022,
    10: 2023,
    11: 2024,
    12: 2025,
    13: 2026,
}


def cycle_to_project_prefix(cycle: int) -> str:
    """Return the public project-code prefix for an ALMA cycle.

    ALMA project codes use the proposal-call year as their first component
    (Cycle 10 → 2023.1.*, Cycle 0 → 2011.0.*). The mapping is tabulated for
    Cycles 0-11 and extrapolated as ``year = cycle + 2013`` only for later
    cycles (valid from Cycle 8 onward; see _CYCLE_TO_PROPOSAL_YEAR).
    """
    cycle_int = int(cycle)
    if cycle_int < 0 or cycle_int > 30:
        raise ValueError("ALMA cycle must be between 0 and 30")
    year = _CYCLE_TO_PROPOSAL_YEAR.get(cycle_int, cycle_int + 2013)
    suffix = "0" if cycle_int == 0 else "1"
    return f"{year}.{suffix}."


def cycle_project_prefixes(
    cycle: int,
    *,
    include_supplemental: bool = True,
    include_ddt: bool = True,
) -> List[str]:
    """Every proposal_id prefix that belongs to one ALMA cycle.

    A cycle is not only its main call (``<year>.1.``): Cycle 7 also had the
    2019.2 supplemental call, and every cycle carries DDT projects with a
    letter period (``2019.A.``). Counting only ``.1.`` undercounts a cycle
    (skill: SKILL.md "UID and project-code grammar"; cycle-capabilities.md).
    Cycle 0 keeps its ``.0.`` special case.
    """
    main = cycle_to_project_prefix(cycle)
    year = main.split(".")[0]
    prefixes = [main]
    if int(cycle) != 0:
        if include_supplemental:
            prefixes.append(f"{year}.2.")
        if include_ddt:
            prefixes.append(f"{year}.A.")
    return prefixes


def cycle_periods_disclosure(
    cycle: int,
    *,
    include_supplemental: bool = True,
    include_ddt: bool = True,
) -> str:
    """Human-readable statement of which project-code periods a cycle query included."""
    prefixes = cycle_project_prefixes(
        cycle, include_supplemental=include_supplemental, include_ddt=include_ddt
    )
    year = prefixes[0].split(".")[0]
    parts = [f"main call {prefixes[0]}*"]
    if int(cycle) != 0:
        parts.append(f"supplemental call {year}.2.*" if include_supplemental
                     else "supplemental-call projects EXCLUDED")
        parts.append(f"DDT letter period {year}.A.*" if include_ddt
                     else "DDT projects EXCLUDED")
    return f"Cycle {int(cycle)} project codes included: " + "; ".join(parts) + "."


def project_prefix_where(
    cycle: int,
    *,
    include_supplemental: bool = True,
    include_ddt: bool = True,
) -> str:
    """ADQL predicate selecting every project code of one cycle (see
    :func:`cycle_project_prefixes`)."""
    prefixes = cycle_project_prefixes(
        cycle, include_supplemental=include_supplemental, include_ddt=include_ddt
    )
    clauses = [f"proposal_id LIKE '{prefix}%'" for prefix in prefixes]
    if len(clauses) == 1:
        return clauses[0]
    return "(" + " OR ".join(clauses) + ")"


# The standard ObsCore projection every template fetches. Row-grain columns
# (asdm_uid, group_ous_uid, schedblock_name) make MOUS/EB aggregation
# possible; s_region/is_mosaic expose the footprint; data_rights and
# obs_release_date carry access; science_observation/scan_intent/qa2_passed
# carry intent and QA; access_url is the DataLink URL of the MOUS.
OBSCORE_BASE_COLUMNS: Tuple[str, ...] = (
    "target_name", "proposal_id", "member_ous_uid", "group_ous_uid", "asdm_uid",
    "obs_publisher_did", "schedblock_name",
    "frequency", "bandwidth", "frequency_support", "band_list",
    "antenna_arrays", "dataproduct_type", "calib_level",
    "scientific_category", "science_keyword", "obs_title", "pi_name",
    "s_ra", "s_dec", "s_region", "is_mosaic",
    "t_exptime", "s_resolution", "spatial_resolution",
    "obs_release_date", "data_rights", "science_observation", "scan_intent",
    "qa2_passed", "access_url", "sensitivity_10kms", "cont_sensitivity_bandwidth",
    "t_min", "t_max",
)


def _format_columns(columns: Sequence[str]) -> str:
    lines: List[str] = []
    line: List[str] = []
    for column in columns:
        line.append(column)
        if len(line) == 4:
            lines.append(", ".join(line))
            line = []
    if line:
        lines.append(", ".join(line))
    return ",\n       ".join(lines)


def select_obscore_query(where_clause: str, *, top: int = 5000, order_by: str = "proposal_id") -> str:
    top = max(1, min(int(top or 5000), 20000))
    return f"""
SELECT TOP {top}
       {_format_columns(OBSCORE_BASE_COLUMNS)}
FROM ivoa.obscore
WHERE {where_clause}
ORDER BY {order_by}
"""


def truncation_info(df: Optional[pd.DataFrame], max_results: int) -> Dict[str, Any]:
    """Did a TOP-limited fetch hit its cap? (skill: capabilities/limits are
    operational settings — never report a capped fetch as the complete answer)."""
    rows = int(len(df)) if df is not None else 0
    cap = int(max_results or 0)
    truncated = bool((cap and rows >= cap) or (df is not None and df.attrs.get("truncated")))
    warning = ""
    if truncated:
        warning = (
            f"TAP fetch hit the TOP {cap} row cap (ordered by proposal_id, so the lowest "
            "project codes come first); counts and project lists below are INCOMPLETE. "
            "Report derived counts as 'at least' (lower bounds), never as reliable totals. "
            "Use server-side aggregation or exhaust pagination for exact counts."
        )
    return {"rows_fetched": rows, "row_cap": cap, "truncated": truncated,
            "count_is_lower_bound": truncated, "warning": warning}


def distinct_project_count_query(where_clause: str) -> str:
    """Count projects at the server without a coverage-row TOP limit."""
    return ("SELECT COUNT(DISTINCT proposal_id) AS n_projects FROM ivoa.obscore "
            f"WHERE ({where_clause}) AND proposal_id IS NOT NULL")


def public_band_inventory_query(as_of_date: Optional[str] = None) -> str:
    """Distinct publicly accessible band tokens, optionally by release date.

    A historical cut describes current records released by that date; it is
    not a reconstruction of archive holdings or receiver capabilities then.
    """
    where = "data_rights = 'Public' AND band_list IS NOT NULL"
    if as_of_date:
        canonical = date.fromisoformat(as_of_date).isoformat()
        where += f" AND obs_release_date < '{date.fromordinal(date.fromisoformat(canonical).toordinal() + 1).isoformat()}'"
    return f"SELECT DISTINCT band_list FROM ivoa.obscore WHERE {where} ORDER BY band_list"


def fetch_cycle_array_metadata(
    fetch: Callable[[str], pd.DataFrame], cycle: int, *, page_size: int = 5000,
    max_pages: int = 20,
) -> pd.DataFrame:
    """Exhaust DISTINCT array metadata with bounded, non-OFFSET keyset pages.

    Fetch is the caller's existing budgeted TAP adapter. Four NULL partitions
    avoid relying on service-specific NULL ordering or unsupported COALESCE.
    Repeated EBs/SPWs collapse at the server. On any incomplete page stream,
    preserve metadata already obtained and explicitly mark the count a bound.
    Array membership remains the documented antenna/SB heuristic.
    """
    size = max(1, min(int(page_size), 20000))
    columns = ("proposal_id", "antenna_arrays", "schedblock_name")
    frames, queries = [], []
    complete, error = True, ""
    for antenna_null, sb_null in ((False, False), (False, True), (True, False), (True, True)):
        parts = [project_prefix_where(cycle), "proposal_id IS NOT NULL"]
        keys = ["proposal_id"]
        for column, is_null in (("antenna_arrays", antenna_null), ("schedblock_name", sb_null)):
            parts.append(f"{column} IS {'NULL' if is_null else 'NOT NULL'}")
            if not is_null:
                keys.append(column)
        cursor = None
        while True:
            if len(queries) >= max_pages:
                complete, error = False, "Pagination budget exhausted"
                break
            where = " AND ".join(parts)
            if cursor is not None:
                greater = []
                for index, key in enumerate(keys):
                    equal = [f"{keys[j]} = '{escape_adql(cursor[j])}'" for j in range(index)]
                    greater.append("(" + " AND ".join(equal + [f"{key} > '{escape_adql(cursor[index])}'"]) + ")")
                where += " AND (" + " OR ".join(greater) + ")"
            query = (f"SELECT DISTINCT TOP {size} {', '.join(columns)} FROM ivoa.obscore "
                     f"WHERE {where} ORDER BY {', '.join(keys)}")
            queries.append(query)
            try:
                page = fetch(query)
                if not set(columns).issubset(page.columns):
                    raise ValueError("TAP response lacks array-count metadata columns")
                if page.empty:
                    if page.attrs.get("truncated"):
                        raise ValueError("TAP returned an empty truncated page")
                    break
                new_cursor = tuple(as_text(page.iloc[-1][key]) for key in keys)
                if cursor is not None and new_cursor <= cursor:
                    raise ValueError("TAP pagination made no forward progress")
                frames.append(page)
                cursor = new_cursor
                if len(page) < size and not page.attrs.get("truncated"):
                    break
            except Exception as exc:
                complete, error = False, str(exc)
                break
        if not complete:
            break
    result = pd.concat(frames, ignore_index=True).drop_duplicates(list(columns)) if frames else pd.DataFrame(columns=columns)
    result.attrs.update(truncated=not complete, count_is_lower_bound=not complete,
                        pagination_complete=complete, pagination_queries=queries,
                        pagination_error=error, metadata_grain="distinct project/antenna/SB combinations")
    return result


SPEED_OF_LIGHT_M_GHZ = 0.299792458  # metres * GHz


def wavelength_overlap_where(nu_lo_ghz: float, nu_hi_ghz: float) -> str:
    """ADQL: rows whose [em_min, em_max] wavelength span overlaps [nu_lo, nu_hi] GHz.

    em_min/em_max are WAVELENGTHS IN METRES (skill guardrail 2), so the
    frequency window [nu_lo, nu_hi] is the wavelength window
    [c/nu_hi, c/nu_lo]; overlap is em_min <= c/nu_lo AND em_max >= c/nu_hi.
    This is the coarse prefilter; exact SPW coverage still needs
    frequency_support (see :func:`parse_frequency_support_windows`).
    """
    lo = float(min(nu_lo_ghz, nu_hi_ghz))
    hi = float(max(nu_lo_ghz, nu_hi_ghz))
    if lo <= 0 or hi <= 0:
        raise ValueError("frequency bounds must be positive GHz")
    lam_max = SPEED_OF_LIGHT_M_GHZ / lo
    lam_min = SPEED_OF_LIGHT_M_GHZ / hi
    return f"(em_min <= {lam_max:.12g} AND em_max >= {lam_min:.12g})"


def alma_cone_where(ra: float, dec: float, radius_deg: float, *, footprint: bool = True) -> str:
    """Cone predicate over the s_region FOOTPRINT unioned with the representative
    point (s_ra, s_dec).

    s_ra/s_dec is a representative position only: a mosaic whose footprint
    overlaps the cone without its centre falling inside is missed by the
    point test (skill: archive-query.md ADQL patterns). INTERSECTS(CIRCLE,
    s_region) catches those; the point test is kept in the OR so rows with a
    NULL or single-pointing s_region (the Total Power known issue) are not
    lost. ``footprint=False`` is the point-only fallback for a service that
    rejects INTERSECTS.
    """
    circle = f"CIRCLE('ICRS', {float(ra):.8f}, {float(dec):.8f}, {float(radius_deg):.8f})"
    point = f"CONTAINS(POINT('ICRS', s_ra, s_dec), {circle}) = 1"
    if not footprint:
        return point
    return f"(INTERSECTS({circle}, s_region) = 1 OR {point})"


def alma_cone_filters_where(
    ra: float,
    dec: float,
    radius_deg: float,
    *,
    public: bool = True,
    footprint: bool = True,
    band: Any = None,
    max_resolution_arcsec: Optional[float] = None,
    science_only: bool = False,
) -> str:
    """Cone predicate PLUS the user's structured filters, all in ADQL.

    UI benchmark 2026-09-22 (D16): "Band 6 observations of M83" ran a TOP 100
    cone with no band filter and kept whatever Band 6 rows fell in the first
    100 -- the band, resolution and science filters belong in the WHERE so the
    row cap applies AFTER them.
    """
    where = alma_cone_where(ra, dec, radius_deg, footprint=footprint)
    if public:
        where += " AND data_rights = 'Public'"
    bands = requested_bands(band) if band is not None else []
    if band is not None and not bands:
        raise ValueError("band must contain ALMA band numbers from 1 to 10")
    if bands:
        where += " AND (" + " OR ".join(band_token_where(b) for b in bands) + ")"
    if max_resolution_arcsec is not None:
        threshold = float(max_resolution_arcsec)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("max_resolution_arcsec must be finite and positive")
        where += f" AND spatial_resolution <= {threshold:g}"
    if science_only:
        where += " AND science_observation = 'T'"
    return where


def alma_cone_adql(
    ra: float,
    dec: float,
    radius_deg: float,
    *,
    public: bool = True,
    top: Optional[int] = None,
    footprint: bool = True,
    columns: Sequence[str] = OBSCORE_BASE_COLUMNS,
    band: Any = None,
    max_resolution_arcsec: Optional[float] = None,
    science_only: bool = False,
) -> str:
    """The cone query the ALMA client executes (and what provenance reports)."""
    where = alma_cone_filters_where(
        ra, dec, radius_deg, public=public, footprint=footprint, band=band,
        max_resolution_arcsec=max_resolution_arcsec, science_only=science_only,
    )
    top_clause = f"TOP {max(1, min(int(top), 20000))} " if top else ""
    return f"SELECT {top_clause}{', '.join(columns)} FROM ivoa.obscore WHERE {where}"


def alma_cone_count_adql(
    ra: float,
    dec: float,
    radius_deg: float,
    *,
    public: bool = True,
    footprint: bool = True,
    band: Any = None,
    max_resolution_arcsec: Optional[float] = None,
    science_only: bool = False,
) -> str:
    """Companion ``SELECT COUNT(*)`` for the same cone + filters, so a
    TOP-capped result can say "N rows total, showing M"."""
    where = alma_cone_filters_where(
        ra, dec, radius_deg, public=public, footprint=footprint, band=band,
        max_resolution_arcsec=max_resolution_arcsec, science_only=science_only,
    )
    return f"SELECT COUNT(*) AS total_rows, COUNT(DISTINCT member_ous_uid) AS total_mous FROM ivoa.obscore WHERE {where}"


def science_cone_query(ra, dec, radius_arcsec=60.0, *, band=None,
                       max_resolution_arcsec=None, public_only=False,
                       science_only=False, top=5000):
    """Apply explicit science constraints before the archive's row cap."""
    if not all(math.isfinite(float(v)) for v in (ra, dec, radius_arcsec)):
        raise ValueError("Coordinates and radius must be finite")
    if not (0 <= float(ra) < 360 and -90 <= float(dec) <= 90 and 0 < float(radius_arcsec) <= 648000):
        raise ValueError("Invalid ICRS coordinates or cone radius")
    if not 1 <= int(top) <= 20000:
        raise ValueError("top must be between 1 and 20000")
    where = alma_cone_where(float(ra), float(dec), float(radius_arcsec) / 3600)
    bands = requested_bands(band)
    if band is not None and not bands:
        raise ValueError("band must contain ALMA band numbers from 1 to 10")
    if bands:
        where += " AND (" + " OR ".join(band_token_where(b) for b in bands) + ")"
    if max_resolution_arcsec is not None:
        threshold = float(max_resolution_arcsec)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("max_resolution_arcsec must be finite and positive")
        where += f" AND spatial_resolution < {threshold}"
    if public_only:
        where += " AND data_rights = 'Public'"
    if science_only:
        where += " AND science_observation = 'T'"
    columns = ("member_ous_uid", "proposal_id", "target_name", "s_ra", "s_dec",
               "band_list", "spatial_resolution", "data_rights", "science_observation", "qa2_passed")
    # No ORDER BY: it made the NRAO proxy time out (502 after 62 s, live 2026-09-16)
    # while the unsorted query returned in 25 s; grouping sorts by MOUS in Python.
    return f"SELECT TOP {int(top)} {', '.join(columns)} FROM ivoa.obscore WHERE {where}"


def summarize_mous(df: pd.DataFrame) -> pd.DataFrame:
    """One observation per MOUS; minimum beam size and union of observed bands."""
    if df.empty:
        return df.copy()
    required = {"member_ous_uid", "band_list", "spatial_resolution"}
    if not required.issubset(df.columns):
        raise ValueError("Archive result lacks the identifiers or metadata needed for MOUS grouping")
    valid = df["member_ous_uid"].notna() & df["member_ous_uid"].astype(str).str.strip().ne("")
    missing = int((~valid).sum())
    rows = []
    for uid, group in df[valid].groupby("member_ous_uid", sort=True):
        row = group.iloc[0].to_dict()
        row["member_ous_uid"] = uid
        band_set = {b for v in group.band_list for b in band_tokens(v) if str(b).isdigit()}
        row["band_list"] = " ".join(sorted(band_set, key=int))
        row["bands"] = sorted(int(b) for b in band_set)
        row["spatial_resolution"] = pd.to_numeric(group.spatial_resolution, errors="coerce").min()
        row["rows"] = len(group)
        if "qa2_passed" in group:
            row["qa2_passed"] = "; ".join(dict.fromkeys(as_text(v) for v in group.qa2_passed))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs.update(df.attrs)
    if missing:
        out.attrs.update(partial=True, ungroupable_rows=missing)
    return out


_SUNYAEV_RE = re.compile(r"sunyaev|zel'?dovich", re.IGNORECASE)
_SOLAR_SYSTEM_RE = re.compile(r"solar\s+system|\bcomet|trans-neptunian|\bTNO\b|asteroid|\bplanet", re.IGNORECASE)


def solar_where() -> str:
    """Predicate for observations OF THE SUN.

    ``LIKE '%sun%'`` matches the 'Sunyaev-Zel'dovich effect' science keyword
    and ``'%solar%'`` matches ALMA's 'Solar system' category and its
    comet/TNO keywords, so the official scientific_category 'Sun' is the
    primary predicate and the text fallbacks are the word 'sun' anchored on
    word edges (ADQL has no word boundaries; space-anchored alternatives
    approximate them) plus the ALMA keyword 'Solar flares'-style phrases
    ('solar ' followed by a non-system word is NOT attempted). Sunyaev and
    Solar-System rows are excluded in ADQL and again by
    :func:`exclude_sunyaev`.
    """
    alternatives: List[str] = [
        "LOWER(scientific_category) = 'sun'",
        "LOWER(scientific_category) LIKE 'sun %'",
        "LOWER(scientific_category) LIKE '% sun'",
        "LOWER(scientific_category) LIKE '% sun %'",
    ]
    for column in ("target_name", "science_keyword", "obs_title"):
        alternatives.append(f"LOWER({column}) = 'sun'")
        alternatives.append(f"LOWER({column}) LIKE 'sun %'")
        alternatives.append(f"LOWER({column}) LIKE '% sun'")
        alternatives.append(f"LOWER({column}) LIKE '% sun %'")
        alternatives.append(f"LOWER({column}) LIKE '%the sun%'")
    return (
        "(" + " OR ".join(alternatives) + ")"
        " AND LOWER(science_keyword) NOT LIKE '%sunyaev%'"
        " AND LOWER(obs_title) NOT LIKE '%sunyaev%'"
        " AND LOWER(scientific_category) NOT LIKE '%solar system%'"
    )


def exclude_sunyaev(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows a text match on 'sun' let through that are NOT the Sun:
    Sunyaev-Zel'dovich cluster projects and Solar-System (comet / TNO /
    planet) projects."""
    if df is None or df.empty:
        return df
    text_cols = [c for c in ("science_keyword", "obs_title", "scientific_category", "target_name") if c in df.columns]
    if not text_cols:
        return df
    blob = df[text_cols].astype(str).agg(" ".join, axis=1)
    keep = ~blob.str.contains(_SUNYAEV_RE, regex=True, na=False)
    if "scientific_category" in df.columns or "science_keyword" in df.columns:
        cat_cols = [c for c in ("scientific_category", "science_keyword") if c in df.columns]
        cats = df[cat_cols].astype(str).agg(" ".join, axis=1)
        keep &= ~cats.str.contains(_SOLAR_SYSTEM_RE, regex=True, na=False)
    return df[keep].copy()


def first_nonempty(values: Iterable[Any]) -> str:
    for value in values:
        text = as_text(value)
        if text:
            return text
    return ""


def col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


_BAND_SPLIT_RE = re.compile(r"[\s,;/|]+")


def band_tokens(value: Any) -> List[str]:
    """Tolerant tokenizer for the ObsCore ``band_list`` column.

    Live values are numeric and SPACE delimited (``'6'``, band-to-band
    ``'5 10'``); historical/display forms say ``'BAND 6'`` or use commas.
    Splitting on ',' alone yields the single token ``'5 10'`` and silently
    drops every multi-band row (skill guardrail 3).
    """
    text = as_text(value)
    if not text:
        return []
    text = re.sub(r"(?i)\bband\b", " ", text)
    text = re.sub(r"(?i)\bB(?=\d)", "", text)
    return [t for t in _BAND_SPLIT_RE.split(text.strip()) if t]


def normalize_band_token(band: Any) -> str:
    """'Band 6' / 'B6' / 6 / '6' -> '6'."""
    tokens = band_tokens(band)
    return tokens[0] if tokens else ""


def requested_bands(band: Any) -> List[str]:
    """Normalize a user/LLM band argument (int, '6', '6,7', '6 and 7', [6, 7])."""
    if band is None:
        return []
    if isinstance(band, (list, tuple, set)):
        out: List[str] = []
        for item in band:
            out.extend(requested_bands(item))
        return out
    text = re.sub(r"(?i)\band\b", " ", as_text(band))
    return [t for t in band_tokens(text) if t.isdigit() and 1 <= int(t) <= 10]


def row_matches_band(value: Any, bands: Sequence[Any]) -> bool:
    wanted = {normalize_band_token(b) for b in bands if normalize_band_token(b)}
    if not wanted:
        return True
    return bool(wanted & set(band_tokens(value)))


def is_multi_band(value: Any) -> bool:
    return len(band_tokens(value)) > 1


def filter_band(df: pd.DataFrame, band: Any) -> pd.DataFrame:
    """Keep rows whose band_list contains ANY requested band (token match)."""
    if df is None or df.empty or band is None:
        return df
    wanted = requested_bands(band)
    if not wanted:
        return df
    band_col = col(df, ["band_list", "Band", "band"])
    if not band_col:
        return df
    mask = df[band_col].apply(lambda v: row_matches_band(v, wanted))
    return df[mask].copy()


def _resolution_arcsec(row: pd.Series) -> Optional[float]:
    for name in ("s_resolution", "spatial_resolution", "resolution"):
        if name not in row:
            continue
        try:
            value = float(row.get(name))
            if value == value:
                return value
        except (TypeError, ValueError):
            continue
    return None


def filter_resolution(df: pd.DataFrame, max_arcsec: Optional[float]) -> pd.DataFrame:
    if df is None or df.empty or max_arcsec is None:
        return df
    out = df.copy()
    out["_resolution_arcsec"] = out.apply(_resolution_arcsec, axis=1)
    return out[out["_resolution_arcsec"].notna() & (out["_resolution_arcsec"] < float(max_arcsec))].copy()


_ANTENNA_PREFIX_RE = re.compile(r"(?:^|[\s:,;])(DV|DA|CM|PM)\d{2}\b", re.IGNORECASE)
_ANTENNA_PREFIX_TO_ARRAY = {"DV": "12m", "DA": "12m", "CM": "7m", "PM": "TP"}


def infer_arrays(text: Any) -> List[str]:
    """Infer which ALMA arrays a row used.

    The live ObsCore ``antenna_arrays`` column is a blank-separated list of
    ``Pad:Antenna`` pairs such as ``'A004:DV07 A025:CM03 J505:PM03'`` — it
    never contains the words '12m', 'ACA' or 'TP'. Antenna-name prefixes are
    the heuristic the skill recommends (``DV``/``DA`` 12-m, ``CM`` 7-m, ``PM``
    Total Power; listobs-and-intents.md "Which array? Infer it"). Prose
    tokens ('12m Array', 'Total Power', '_TM1'/'_7M'/'_TP' SB-name suffixes)
    are still recognised as a fallback for display-shaped values. This is a
    heuristic, not an identity contract: heterogeneous EBs can mix 7-m and
    12-m antennas and the result then lists both.
    """
    raw = as_text(text)
    arrays: List[str] = []
    for match in _ANTENNA_PREFIX_RE.finditer(raw):
        label = _ANTENNA_PREFIX_TO_ARRAY[match.group(1).upper()]
        if label not in arrays:
            arrays.append(label)
    if arrays:
        return arrays
    clean = raw.lower()
    if re.search(r"\b12\s*m\b|12m|array.*twelve|_tm[12]\b|tm[12]", clean):
        arrays.append("12m")
    if re.search(r"\b7\s*m\b|7m|aca|morita", clean):
        arrays.append("7m")
    if re.search(r"\btp\b|_tp\b|total\s*power|single\s*dish", clean):
        arrays.append("TP")
    return arrays


def infer_arrays_from_schedblock(name: Any) -> List[str]:
    """``_TM1``/``_TM2`` -> 12m, ``_7M`` -> 7m, ``_TP`` -> TP (SB-name heuristic)."""
    clean = as_text(name).upper()
    arrays: List[str] = []
    if re.search(r"_TM[12]\b", clean):
        arrays.append("12m")
    if re.search(r"_7M\b", clean):
        arrays.append("7m")
    if re.search(r"_TP\b", clean):
        arrays.append("TP")
    return arrays


def detect_arrays(row: pd.Series) -> List[str]:
    """Union of antenna-prefix and SB-name heuristics for one ObsCore row."""
    arrays: List[str] = []
    for key in ("antenna_arrays", "array", "arrays"):
        if key in row.index:
            for label in infer_arrays(row.get(key)):
                if label not in arrays:
                    arrays.append(label)
    if "schedblock_name" in row.index:
        for label in infer_arrays_from_schedblock(row.get("schedblock_name")):
            if label not in arrays:
                arrays.append(label)
    return arrays


def _nunique(group: pd.DataFrame, column: str) -> Optional[int]:
    if column not in group.columns:
        return None
    values = group[column].dropna().astype(str).str.strip()
    values = values[(values != "") & (values.str.lower() != "nan")]
    return int(values.nunique())


def aggregate_counts(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Row-grain honesty for one ObsCore frame (skill guardrail 1).

    ObsCore rows repeat per execution block, per field and per spectral
    coverage, so ``len(df)`` is NOT a number of observations. Report rows,
    distinct MOUS (``member_ous_uid`` = datasets), distinct EBs (``asdm_uid``)
    and distinct projects separately; ``None`` means the column was absent.
    """
    if df is None or not hasattr(df, "columns"):
        return {"rows": 0, "n_mous": None, "n_eb": None, "n_projects": None}
    counts: Dict[str, Any] = {
        "rows": int(len(df)),
        "n_mous": _nunique(df, "member_ous_uid"),
        "n_eb": _nunique(df, "asdm_uid"),
        "n_projects": None,
    }
    project_col = col(df, ["proposal_id", "project_code"])
    if project_col:
        counts["n_projects"] = _nunique(df, project_col)
    if "data_rights" in df.columns:
        rights = df["data_rights"].astype(str).str.strip().str.lower()
        counts["n_public_rows"] = int((rights == "public").sum())
        counts["n_proprietary_rows"] = int((rights == "proprietary").sum())
    if df.attrs.get("count_is_lower_bound") or df.attrs.get("truncated"):
        counts["count_is_lower_bound"] = True
    return counts


def counts_note(counts: Dict[str, Any]) -> str:
    """One sentence stating rows vs datasets vs executions (never 'N observations')."""
    rows = int(counts.get("rows") or 0)
    parts = [f"{rows} archive row{'s' if rows != 1 else ''}"]
    if counts.get("n_mous") is not None:
        parts.append(f"{counts['n_mous']} dataset{'s' if counts['n_mous'] != 1 else ''} (distinct member_ous_uid)")
    if counts.get("n_eb") is not None:
        parts.append(f"{counts['n_eb']} execution block{'s' if counts['n_eb'] != 1 else ''} (distinct asdm_uid)")
    if counts.get("n_projects") is not None:
        parts.append(f"{counts['n_projects']} project{'s' if counts['n_projects'] != 1 else ''}")
    note = ", ".join(parts)
    if counts.get("count_is_lower_bound"):
        note = "At least " + note + "; counts are lower bounds because the archive fetch is incomplete"
    if counts.get("n_proprietary_rows"):
        note += f"; {counts['n_proprietary_rows']} row(s) are proprietary (data_rights)"
    return note + ". Rows repeat per execution/field/spectral coverage; do not call rows observations."


def projects_with_array_combo(df: pd.DataFrame, required_arrays: Sequence[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    project_col = col(df, ["proposal_id", "project_code"])
    array_col = col(df, ["antenna_arrays", "array", "arrays"])
    if not project_col or not array_col:
        return pd.DataFrame()

    required = {item.upper().replace("TOTAL POWER", "TP").replace("7M", "7m").replace("12M", "12m") for item in required_arrays}
    normalized_required = {"TP" if item == "TP" else item.lower() for item in required}
    rows: List[Dict[str, Any]] = []
    for project, group in df.groupby(project_col, dropna=True):
        arrays = set()
        for _, row in group.iterrows():
            arrays.update(detect_arrays(row))
        normalized_arrays = {"TP" if item == "TP" else item.lower() for item in arrays}
        if normalized_required <= normalized_arrays:
            rows.append({
                "proposal_id": as_text(project),
                "arrays_found": ", ".join(sorted(arrays)),
                "rows": int(len(group)),
                "n_mous": _nunique(group, "member_ous_uid"),
                "n_eb": _nunique(group, "asdm_uid"),
                "array_inference": "heuristic: antenna prefixes DV/DA=12m, CM=7m, PM=TP (+ SB-name suffix); the archive delivers per-MOUS and has not combined arrays",
                "target_name": first_nonempty(group["target_name"].tolist()) if "target_name" in group.columns else "",
                "pi_name": first_nonempty(group["pi_name"].tolist()) if "pi_name" in group.columns else "",
                "obs_title": first_nonempty(group["obs_title"].tolist()) if "obs_title" in group.columns else "",
            })
    return pd.DataFrame(rows)


def observation_interval_ghz(row: pd.Series) -> Optional[tuple[float, float]]:
    try:
        freq = float(row.get("frequency"))
        bandwidth_hz = float(row.get("bandwidth") or 0)
        bandwidth_ghz = bandwidth_hz / 1e9 if bandwidth_hz > 1e5 else bandwidth_hz
        if freq != freq or bandwidth_ghz != bandwidth_ghz:
            return None
        return (freq - 0.5 * bandwidth_ghz, freq + 0.5 * bandwidth_ghz)
    except (TypeError, ValueError):
        return None


def _value_to_ghz(value: float, unit: Optional[str]) -> float:
    if unit:
        return float(value) * UNIT_TO_GHZ.get(unit.lower(), 1.0)
    # TAP frequency columns are GHz, but DataLink/frequency-support strings can
    # sometimes carry raw Hz-like numbers without a unit.
    if abs(float(value)) > 1e5:
        return float(value) / 1e9
    return float(value)


def parse_frequency_support_windows(value: Any) -> List[Dict[str, Optional[float]]]:
    """Parse ALMA frequency_support into per-SPW windows with channel resolution.

    The archive represents spectral windows as strings such as
    ``[86.24..88.11GHz, 976.56kHz, ...] U [88.10..89.98GHz, 488.28kHz, ...]``.
    Each returned dict carries ``low_ghz``/``high_ghz`` and, when present, the
    SPW channel ``resolution_khz`` (the token immediately after the range).
    Different separators (``..``/``-``/``to``) and units (GHz/MHz/kHz/Hz) are
    accepted so the parser is robust across frequency_support formats.
    """
    text = as_text(value)
    if not text:
        return []

    windows: List[Dict[str, Optional[float]]] = []
    for match in FREQUENCY_RANGE_RE.finditer(text):
        lo_unit = match.group("lo_unit")
        hi_unit = match.group("hi_unit") or lo_unit
        lo = _value_to_ghz(float(match.group("lo")), lo_unit or hi_unit)
        hi = _value_to_ghz(float(match.group("hi")), hi_unit or lo_unit)
        if lo != lo or hi != hi:
            continue
        low, high = sorted((lo, hi))
        if high <= low:
            continue
        resolution_khz: Optional[float] = None
        tail = RESOLUTION_AFTER_RANGE_RE.match(text, match.end())
        if tail:
            resolution_khz = round(
                float(tail.group("res")) * _RES_UNIT_TO_KHZ[tail.group("unit").lower()],
                6,
            )
        windows.append(
            {
                "low_ghz": round(low, 9),
                "high_ghz": round(high, 9),
                "resolution_khz": resolution_khz,
            }
        )

    deduped: List[Dict[str, Optional[float]]] = []
    seen: set = set()
    for window in sorted(windows, key=lambda w: (w["low_ghz"], w["high_ghz"])):
        key = (window["low_ghz"], window["high_ghz"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(window)
    return deduped


def parse_frequency_support_intervals(value: Any) -> List[Tuple[float, float]]:
    """Parse ALMA frequency_support ranges into GHz intervals (compat shim).

    The archive commonly represents spectral windows as strings containing
    ranges such as ``218.0..220.0GHz``.  This parser intentionally accepts only
    explicit ranges, then callers fall back to ``frequency +/- bandwidth/2``.
    """
    return [
        (window["low_ghz"], window["high_ghz"])
        for window in parse_frequency_support_windows(value)
    ]


def observation_intervals_ghz(row: pd.Series) -> List[Tuple[float, float]]:
    intervals = parse_frequency_support_intervals(row.get("frequency_support"))
    if intervals:
        return intervals
    fallback = observation_interval_ghz(row)
    return [fallback] if fallback else []


def observation_windows_ghz(row: pd.Series) -> List[Dict[str, Optional[float]]]:
    """Per-SPW windows (with resolution) for one ObsCore row, with fallback."""
    windows = parse_frequency_support_windows(row.get("frequency_support"))
    if windows:
        return windows
    fallback = observation_interval_ghz(row)
    if fallback:
        return [
            {"low_ghz": fallback[0], "high_ghz": fallback[1], "resolution_khz": None}
        ]
    return []


def line_names_for_input(lines: Sequence[str]) -> List[str]:
    output: List[str] = []
    for line in lines:
        raw = as_text(line)
        if not raw:
            continue
        compact = raw.replace(" ", "").upper()
        aliases = {
            "12CO": "12CO(2-1)",
            "13CO": "13CO(2-1)",
            "C18O": "C18O(2-1)",
            "CO": "CO(2-1)",
        }
        # line-name-case-sensitive-silent-drop: fall back to the UPPERCASED
        # compact token, not the original-case string, so lower/mixed-case
        # inputs with an explicit transition (e.g. 'co(1-0)') still resolve
        # against the all-uppercase LINE_REST_FREQ_GHZ keys.
        fs = _fine_structure_key(compact)
        if fs is not None and compact.upper() not in LINE_REST_FREQ_GHZ:
            name = FINE_STRUCTURE_SPECIES[fs][0]
            if name not in output:
                output.append(name)
            continue
        name = aliases.get(compact, compact)
        if name not in LINE_REST_FREQ_GHZ and f"{name}(2-1)" in LINE_REST_FREQ_GHZ:
            name = f"{name}(2-1)"
        if name in LINE_REST_FREQ_GHZ and name not in output:
            output.append(name)
    return output


def line_names_for_species(species: Sequence[str] | str) -> List[str]:
    """Transitions for the requested rest species. No species -> the CO
    ladder (the documented default). A NAMED species with no rest-frequency
    entry raises :class:`UnsupportedSpecies` -- it is never replaced by CO
    while the answer keeps the requested label (guard CX-29)."""
    raw_items = [species] if isinstance(species, str) else list(species or [])
    raw_items = [item for item in raw_items if as_text(item).strip()]
    if not raw_items:
        return CO_LADDER_LINES.copy()
    output: List[str] = []
    unknown: List[str] = []
    for item in raw_items:
        compact = as_text(item).replace(" ", "").upper()
        if compact in {"CO", "12CO"}:
            for name in CO_LADDER_LINES:
                if name not in output:
                    output.append(name)
            continue
        fs = _fine_structure_key(compact)
        if fs is not None:
            for name in FINE_STRUCTURE_SPECIES[fs]:
                if name not in output:
                    output.append(name)
            continue
        # A bare molecule name means its whole ladder (as for CO): a redshifted
        # search must consider every transition that can land in a band.
        ladder = [k for k in LINE_REST_FREQ_GHZ if k.upper().startswith(compact + "(")] if "(" not in compact else []
        found = ladder or line_names_for_input([compact])
        if not found:
            unknown.append(as_text(item))
        for name in found:
            if name not in output:
                output.append(name)
    if unknown and not output:
        supported = sorted({k.split("(")[0] for k in LINE_REST_FREQ_GHZ if not k.startswith("[")} | {f"[{k}]" for k in FINE_STRUCTURE_SPECIES})
        raise UnsupportedSpecies(
            f"rest species {', '.join(unknown)} has no rest-frequency entry in this tool "
            f"(supported: {', '.join(supported)}); it was NOT replaced by CO -- give the rest frequency explicitly"
        )
    return output


def annotate_line_coverage(df: pd.DataFrame, lines: Sequence[str], z: float = 0.0) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    line_names = line_names_for_input(lines)
    out = df.copy()

    def covered(row: pd.Series) -> str:
        intervals = observation_intervals_ghz(row)
        if not intervals:
            return ""
        hits = []
        for name in line_names:
            observed_freq = LINE_REST_FREQ_GHZ[name] / (1 + float(z or 0.0))
            if any(lo <= observed_freq <= hi for lo, hi in intervals):
                hits.append(name)
        return ", ".join(hits)

    out["covered_lines"] = out.apply(covered, axis=1)
    return out[out["covered_lines"].astype(bool)].copy()


def projects_covering_all_lines(df: pd.DataFrame, lines: Sequence[str], z: float = 0.0) -> pd.DataFrame:
    annotated = annotate_line_coverage(df, lines, z=z)
    if annotated.empty:
        return pd.DataFrame()
    project_col = col(annotated, ["proposal_id", "project_code"])
    if not project_col or "member_ous_uid" not in annotated.columns:
        return pd.DataFrame()
    required = set(line_names_for_input(lines))
    rows: List[Dict[str, Any]] = []
    # A proposal can contain unrelated targets and tunings. Preserve the
    # MOUS and target context described in references/archive-query.md, row
    # granularity: do not combine one target's CO with another's isotopologues.
    annotated = annotated[annotated["member_ous_uid"].map(as_text).ne("")]
    group_columns = [project_col, "member_ous_uid"]
    if "target_name" in annotated:
        group_columns.append("target_name")
    for identity, group in annotated.groupby(group_columns, dropna=False):
        project, mous = identity[:2]
        found = set()
        for value in group["covered_lines"].tolist():
            found.update(part.strip() for part in as_text(value).split(",") if part.strip())
        if required <= found:
            rows.append({
                "proposal_id": as_text(project),
                "member_ous_uid": as_text(mous),
                "coverage_basis": "all requested lines within the same MOUS and target SPW set",
                "covered_lines": ", ".join(sorted(found)),
                "rows": int(len(group)),
                "n_mous": _nunique(group, "member_ous_uid"),
                "n_eb": _nunique(group, "asdm_uid"),
                "target_name": first_nonempty(group["target_name"].tolist()) if "target_name" in group.columns else "",
                "band_list": first_nonempty(group["band_list"].tolist()) if "band_list" in group.columns else "",
                "pi_name": first_nonempty(group["pi_name"].tolist()) if "pi_name" in group.columns else "",
                "obs_title": first_nonempty(group["obs_title"].tolist()) if "obs_title" in group.columns else "",
                "science_keyword": first_nonempty(group["science_keyword"].tolist()) if "science_keyword" in group.columns else "",
            })
    return pd.DataFrame(rows)


_EXTRAGALACTIC_RE = re.compile(
    r"galax(?:y|ies)|extragalactic|cosmolog|active galactic|\bAGN\b|\bSMG\b|quasar",
    re.IGNORECASE,
)
_CALIBRATION_RE = re.compile(r"calibrat|bandpass|\bphase\b|\bflux\b|\bcheck\b", re.IGNORECASE)


def extragalactic_science_where() -> str:
    """Require science intent before the row cap; categories are proposal metadata."""
    return (
        "science_observation = 'T' AND ("
        "LOWER(scientific_category) LIKE '%galax%' OR "
        "LOWER(scientific_category) LIKE '%cosmolog%' OR "
        "LOWER(scientific_category) LIKE '%active%' OR "
        "LOWER(science_keyword) LIKE '%galax%' OR "
        "LOWER(science_keyword) LIKE '%agn%' OR "
        "LOWER(science_keyword) LIKE '%quasar%') "
        "AND (scientific_category IS NULL OR LOWER(scientific_category) NOT LIKE '%solar system%') "
        "AND (science_keyword IS NULL OR LOWER(science_keyword) NOT LIKE '%calibrat%') "
        "AND (scientific_category IS NULL OR LOWER(scientific_category) NOT LIKE '%calibrat%')"
    )


def is_extragalactic_science_row(row: pd.Series) -> bool:
    """Reject calibration/solar-system rows even under extragalactic proposals.

    science_observation=T is documented in references/archive-query.md.
    Missing science/classification metadata cannot establish target membership.
    A proposal category alone never establishes a target redshift.
    """
    if as_text(row.get("science_observation")).lower() not in {"t", "true", "1"}:
        return False
    metadata = " ".join(as_text(row.get(k)) for k in ("scientific_category", "science_keyword"))
    if _SOLAR_SYSTEM_RE.search(metadata) or _CALIBRATION_RE.search(metadata):
        return False
    intent = as_text(row.get("scan_intent"))
    if _CALIBRATION_RE.search(intent) and "target" not in intent.lower():
        return False
    return bool(_EXTRAGALACTIC_RE.search(metadata))


def redshifted_line_windows(rest_species: Sequence[str] | str, z_min: float, z_max: float) -> List[Dict[str, Any]]:
    """Explicit per-transition nu_obs = nu_rest / (1 + z) intervals."""
    z_lo, z_hi = sorted((float(z_min), float(z_max)))
    if not all(math.isfinite(z) and z > -1 for z in (z_lo, z_hi)):
        raise ValueError("Redshift bounds must be finite and greater than -1")
    return [{"transition": name, "rest_frequency_ghz": LINE_REST_FREQ_GHZ[name],
             "observed_min_ghz": LINE_REST_FREQ_GHZ[name] / (1 + z_hi),
             "observed_max_ghz": LINE_REST_FREQ_GHZ[name] / (1 + z_lo)}
            for name in line_names_for_species(rest_species)]


def redshifted_line_projects(
    df: pd.DataFrame,
    *,
    rest_species: Sequence[str] | str = "CO",
    z_min: float = 1.0,
    z_max: float = 2.0,
) -> pd.DataFrame:
    """Return coverage-compatible science targets, never inferred source redshifts."""
    if df is None or df.empty:
        return pd.DataFrame()
    project_col = col(df, ["proposal_id", "project_code"])
    if not project_col:
        return pd.DataFrame()

    z_lo = float(min(z_min, z_max))
    z_hi = float(max(z_min, z_max))
    windows = redshifted_line_windows(rest_species, z_lo, z_hi)
    hit_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        if not is_extragalactic_science_row(row):
            continue
        # Representative frequency +/- total bandwidth can bridge SPW gaps.
        # Missing actual SPW metadata is insufficient for this science claim.
        intervals = parse_frequency_support_intervals(row.get("frequency_support"))
        if not intervals:
            continue
        for window in windows:
            line_name, rest_freq = window["transition"], window["rest_frequency_ghz"]
            obs_min, obs_max = window["observed_min_ghz"], window["observed_max_ghz"]
            for lo, hi in intervals:
                inter_lo = max(lo, obs_min)
                inter_hi = min(hi, obs_max)
                if inter_lo > inter_hi:
                    continue
                inferred_z_min = max(z_lo, rest_freq / inter_hi - 1.0)
                inferred_z_max = min(z_hi, rest_freq / inter_lo - 1.0)
                hit_rows.append({
                    "proposal_id": as_text(row.get(project_col)),
                    "member_ous_uid": as_text(row.get("member_ous_uid")),
                    "asdm_uid": as_text(row.get("asdm_uid")),
                    "target_name": as_text(row.get("target_name")),
                    "transition": line_name,
                    "rest_frequency_ghz": round(rest_freq, 6),
                    "observed_frequency_range_ghz": f"{inter_lo:.3f}-{inter_hi:.3f}",
                    "coverage_compatible_redshift_range": f"{inferred_z_min:.3f}-{inferred_z_max:.3f}",
                    "band_list": as_text(row.get("band_list")),
                    "pi_name": as_text(row.get("pi_name")),
                    "obs_title": as_text(row.get("obs_title")),
                    "science_keyword": as_text(row.get("science_keyword")),
                })

    hits = pd.DataFrame(hit_rows)
    if hits.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for (project, target, mous), group in hits.groupby(["proposal_id", "target_name", "member_ous_uid"], dropna=False):
        rows.append({
            "proposal_id": as_text(project),
            "target_name": as_text(target),
            "member_ous_uid": as_text(mous),
            "transitions": ", ".join(sorted(set(group["transition"].tolist()))),
            "observed_frequency_ranges_ghz": "; ".join(sorted(set(group["observed_frequency_range_ghz"].tolist()))[:8]),
            "coverage_compatible_redshift_ranges": "; ".join(sorted(set(group["coverage_compatible_redshift_range"].tolist()))[:8]),
            "target_redshift": None,
            "target_redshift_status": "unknown: no target redshift measurement supplied by ObsCore",
            "redshift_interpretation": "coverage-compatible only; not evidence that the target lies in the requested redshift range",
            "line_frequency_windows": windows,
            "rows": int(len(group)),
            "n_mous": _nunique(group[group["member_ous_uid"] != ""], "member_ous_uid"),
            "n_eb": _nunique(group[group["asdm_uid"] != ""], "asdm_uid"),
            "band_list": first_nonempty(group["band_list"].tolist()),
            "pi_name": first_nonempty(group["pi_name"].tolist()),
            "obs_title": first_nonempty(group["obs_title"].tolist()),
            "science_keyword": first_nonempty(group["science_keyword"].tolist()),
        })
    return pd.DataFrame(rows).sort_values(["rows", "proposal_id"], ascending=[False, True])


def _band_union(group: pd.DataFrame) -> str:
    if "band_list" not in group.columns:
        return ""
    tokens: List[str] = []
    for value in group["band_list"].tolist():
        for token in band_tokens(value):
            if token not in tokens:
                tokens.append(token)
    return " ".join(sorted(tokens, key=lambda t: (not t.isdigit(), int(t) if t.isdigit() else 0, t)))


def _release_span(group: pd.DataFrame) -> Tuple[str, str]:
    if "obs_release_date" not in group.columns:
        return "", ""
    values = sorted(v for v in (as_text(x) for x in group["obs_release_date"].tolist()) if v)
    if not values:
        return "", ""
    return values[0], values[-1]


def summarize_projects(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    project_col = col(df, ["proposal_id", "project_code"])
    if not project_col:
        return df
    rows: List[Dict[str, Any]] = []
    for project, group in df.groupby(project_col, dropna=True):
        arrays: List[str] = []
        for _, row in group.iterrows():
            for label in detect_arrays(row):
                if label not in arrays:
                    arrays.append(label)
        release_min, release_max = _release_span(group)
        entry: Dict[str, Any] = {
            "proposal_id": as_text(project),
            "target_name": first_nonempty(group["target_name"].tolist()) if "target_name" in group.columns else "",
            "rows": int(len(group)),
            "n_mous": _nunique(group, "member_ous_uid"),
            "n_eb": _nunique(group, "asdm_uid"),
            "band_list": _band_union(group),
            "arrays": ", ".join(arrays),
            "resolution_arcsec": min([v for v in (_resolution_arcsec(row) for _, row in group.iterrows()) if v is not None], default=""),
            "pi_name": first_nonempty(group["pi_name"].tolist()) if "pi_name" in group.columns else "",
            "obs_title": first_nonempty(group["obs_title"].tolist()) if "obs_title" in group.columns else "",
            "obs_release_date_min": release_min,
            "obs_release_date_max": release_max,
        }
        if "data_rights" in group.columns:
            rights = group["data_rights"].astype(str).str.strip().str.lower()
            entry["n_public_rows"] = int((rights == "public").sum())
            entry["n_proprietary_rows"] = int((rights == "proprietary").sum())
        rows.append(entry)
    out = pd.DataFrame(rows)
    sort_cols = ["rows", "proposal_id"]
    ascending = [False, True]
    if out["n_mous"].notna().any():
        sort_cols = ["n_mous", "rows", "proposal_id"]
        ascending = [False, False, True]
    return out.sort_values(sort_cols, ascending=ascending, na_position="last")


# ─────────────────────────────────────────────────────────────────────────────
# R2 — sensitivity-driven discovery + archive↔literature ObsCore joins.
# Column availability live-verified against the ALMA TAP 2026-07-18 (73 obscore
# columns incl. sensitivity_10kms [mJy/beam], cont_sensitivity_bandwidth
# [mJy/beam], bib_reference, pub_title, publication_year, first_author).
# bib_reference is a SPACE-SEPARATED bibcode list per row; pub_title and
# first_author are concatenated blobs without a per-publication delimiter, so
# bibcodes are the only mechanically joinable key.
# ─────────────────────────────────────────────────────────────────────────────

# Standard 19-character ADS bibcode: YYYYJJJJJVVVVMPPPPA.
_BIBCODE_RE = re.compile(r"^[12]\d{3}[A-Za-z][A-Za-z0-9&.]{13}[A-Za-z.]$")

_PROJECT_CODE_RE = re.compile(r"^\d{4}\.[0-9A-Za-z]\.\d{5}\.[A-Z]$")


def looks_like_bibcode(value: Any) -> bool:
    return bool(_BIBCODE_RE.match(as_text(value)))


def split_bibcodes(value: Any) -> List[str]:
    """Split an obscore ``bib_reference`` blob into individual bibcodes."""
    tokens = as_text(value).split()
    seen: set = set()
    out: List[str] = []
    for token in tokens:
        if looks_like_bibcode(token) and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def band_token_where(band: Any) -> str:
    """Exact-token match on the space-delimited band_list column.

    A bare substring LIKE would make band=1 also match Band 10 (the
    line_set_projects CAP-06 lesson) — match the token in every position.
    """
    token = escape_adql(str(band).strip())
    return (
        "("
        f"band_list = '{token}' OR band_list LIKE '{token} %' "
        f"OR band_list LIKE '% {token}' OR band_list LIKE '% {token} %'"
        ")"
    )


def sensitivity_where(
    sensitivity_mjy: float,
    *,
    continuum: bool = False,
    band: Any = None,
    science_category: str = "",
) -> Tuple[str, str]:
    """WHERE clause for sensitivity-driven archival discovery.

    Returns ``(where_clause, sensitivity_column)``. A row qualifies when its
    achieved sensitivity is at or below the requested rms (deeper == smaller
    mJy/beam). ``sensitivity_10kms`` is the line sensitivity per 10 km/s
    channel; ``cont_sensitivity_bandwidth`` is the aggregated-continuum
    sensitivity over the full bandwidth.
    """
    threshold = float(sensitivity_mjy)
    if not (threshold > 0):
        raise ValueError("sensitivity_mjy must be a positive rms in mJy/beam")
    column = "cont_sensitivity_bandwidth" if continuum else "sensitivity_10kms"
    parts = [f"{column} > 0", f"{column} <= {threshold:g}"]
    if band is not None and str(band).strip():
        parts.append(band_token_where(band))
    category = as_text(science_category)
    if category:
        parts.append(
            f"LOWER(scientific_category) LIKE '%{escape_adql(category).lower()}%'"
        )
    return " AND ".join(parts), column


def publication_join_where(identifier: str) -> Tuple[str, str]:
    """WHERE clause joining ObsCore rows to publications for one identifier.

    Recognizes ALMA project codes (``2019.1.00123.S``), MOUS UIDs
    (``uid://...``), and ADS bibcodes (reverse direction: which archived data
    did this paper use). Returns ``(where_clause, identifier_kind)``.
    """
    text = as_text(identifier)
    if not text:
        raise ValueError("identifier is required")
    safe = escape_adql(text)
    if text.lower().startswith("uid://"):
        return f"member_ous_uid = '{safe}'", "mous_uid"
    if _PROJECT_CODE_RE.match(text):
        return f"proposal_id = '{safe}'", "project_code"
    if looks_like_bibcode(text):
        # bib_reference is a space-separated list — LIKE with the exact
        # bibcode; bibcodes never contain spaces so no token ambiguity.
        return f"bib_reference LIKE '%{safe}%'", "bibcode"
    raise ValueError(
        f"Unrecognized identifier '{text}': expected an ALMA project code "
        "(2019.1.00123.S), a MOUS UID (uid://...), or an ADS bibcode."
    )


_PUBLICATION_COLUMNS = (
    "bib_reference", "pub_title", "publication_year", "first_author",
)
_SENSITIVITY_COLUMNS = ("sensitivity_10kms", "cont_sensitivity_bandwidth")


def select_obscore_query_extended(
    where_clause: str,
    *,
    extra_columns: Sequence[str] = (),
    top: int = 5000,
    order_by: str = "proposal_id",
) -> str:
    """The standard obscore SELECT plus extra columns (R2 templates)."""
    top = max(1, min(int(top or 5000), 20000))
    extras = [c for c in extra_columns if c not in OBSCORE_BASE_COLUMNS]
    base_columns = _format_columns(OBSCORE_BASE_COLUMNS)
    if extras:
        base_columns += ",\n       " + ", ".join(extras)
    return f"""
SELECT TOP {top}
       {base_columns}
FROM ivoa.obscore
WHERE {where_clause}
ORDER BY {order_by}
"""


def summarize_sensitivity(df: pd.DataFrame, sensitivity_column: str) -> pd.DataFrame:
    """Per-project summary with the best (smallest) achieved sensitivity."""
    summary = summarize_projects(df)
    if summary is None or summary.empty or df is None or df.empty:
        return summary
    if sensitivity_column not in df.columns or "proposal_id" not in summary.columns:
        return summary
    values = pd.to_numeric(df[sensitivity_column], errors="coerce")
    project_col = col(df, ["proposal_id", "project_code"])
    best = (
        df.assign(_sens=values)[values > 0]
        .groupby(project_col)["_sens"]
        .min()
        .rename(f"best_{sensitivity_column}_mjy_beam")
    )
    summary = summary.merge(best, left_on="proposal_id", right_index=True, how="left")
    sens_col = f"best_{sensitivity_column}_mjy_beam"
    return summary.sort_values([sens_col, "proposal_id"], na_position="last")


def summarize_publication_links(df: pd.DataFrame) -> pd.DataFrame:
    """Per-project publication join: deduped bibcodes + observation counts."""
    if df is None or df.empty:
        return pd.DataFrame()
    project_col = col(df, ["proposal_id", "project_code"])
    if not project_col:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for project, group in df.groupby(project_col, dropna=True):
        bibcodes: List[str] = []
        for value in group.get("bib_reference", pd.Series(dtype=str)).tolist():
            for code in split_bibcodes(value):
                if code not in bibcodes:
                    bibcodes.append(code)
        mous_uids = (
            group["member_ous_uid"].dropna().astype(str).unique().tolist()
            if "member_ous_uid" in group.columns else []
        )
        rows.append({
            "proposal_id": as_text(project),
            "n_publications": len(bibcodes),
            "bibcodes": " ".join(bibcodes),
            "rows": int(len(group)),
            "n_mous": _nunique(group, "member_ous_uid"),
            "member_ous_uids": " ".join(mous_uids[:20]),
            "target_name": first_nonempty(group["target_name"].tolist()) if "target_name" in group.columns else "",
            "band_list": first_nonempty(group["band_list"].tolist()) if "band_list" in group.columns else "",
            "pi_name": first_nonempty(group["pi_name"].tolist()) if "pi_name" in group.columns else "",
            "obs_title": first_nonempty(group["obs_title"].tolist()) if "obs_title" in group.columns else "",
        })
    return pd.DataFrame(rows).sort_values(
        ["n_publications", "rows", "proposal_id"], ascending=[False, False, True]
    )


def collect_publications(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Flat, deduped publication list for the observation↔paper graph."""
    if df is None or df.empty or "bib_reference" not in df.columns:
        return []
    seen: set = set()
    publications: List[Dict[str, str]] = []
    for value in df["bib_reference"].tolist():
        for code in split_bibcodes(value):
            if code not in seen:
                seen.add(code)
                publications.append({
                    "bibcode": code,
                    "ads_url": f"https://ui.adsabs.harvard.edu/abs/{code}",
                })
    return publications


def bandwidth_switching_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Rank projects with spectral-setup diversity that may indicate bandwidth switching.

    This is a diagnostic, not a proof.  It intentionally returns reasons and a
    confidence label instead of a binary answer.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    project_col = col(df, ["proposal_id", "project_code"])
    if not project_col:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for project, group in df.groupby(project_col, dropna=True):
        bandwidth_values = pd.to_numeric(group["bandwidth"], errors="coerce").dropna() if "bandwidth" in group.columns else pd.Series(dtype=float)
        distinct_bandwidths = bandwidth_values.round(3).nunique() if not bandwidth_values.empty else 0
        distinct_freqs = pd.to_numeric(group["frequency"], errors="coerce").dropna().round(3).nunique() if "frequency" in group.columns else 0
        distinct_support = group["frequency_support"].dropna().astype(str).nunique() if "frequency_support" in group.columns else 0
        # Guardrail 1: never count SPWs by summing frequency_support over
        # repeated rows — the same MOUS setup appears once per EB/field. A
        # parseable frequency_support string is counted once per
        # (member_ous_uid, string); unparseable rows fall back to the
        # per-row frequency +/- bandwidth/2 window.
        seen_setups: set = set()
        interval_counts: List[int] = []
        interval_widths: List[float] = []
        for _, row in group.iterrows():
            parsed = parse_frequency_support_intervals(row.get("frequency_support"))
            if parsed:
                key = (as_text(row.get("member_ous_uid")), as_text(row.get("frequency_support")))
                if key in seen_setups:
                    continue
                seen_setups.add(key)
                intervals = parsed
            else:
                intervals = observation_intervals_ghz(row)
            interval_counts.append(len(intervals))
            interval_widths.extend(hi - lo for lo, hi in intervals if hi > lo)
        total_spw_intervals = sum(interval_counts)
        target_text = " ".join(group.get("target_name", pd.Series(dtype=str)).dropna().astype(str).tolist()).lower()

        score = 0
        reasons: List[str] = []
        if total_spw_intervals >= 8:
            score += 2
            reasons.append(f"{total_spw_intervals} parsed spectral-window intervals")
        if distinct_bandwidths >= 3:
            score += 2
            reasons.append(f"{distinct_bandwidths} distinct bandwidth values")
        elif distinct_bandwidths == 2:
            score += 1
            reasons.append("2 distinct bandwidth values")
        if distinct_freqs >= 8:
            score += 2
            reasons.append(f"{distinct_freqs} distinct spectral-window frequencies")
        if distinct_support >= 3:
            score += 1
            reasons.append(f"{distinct_support} distinct frequency-support strings")
        if interval_widths and min(interval_widths) < 0.5 and max(interval_widths) > 1.5:
            score += 2
            reasons.append("mixed narrow and wide spectral-window widths")
        if distinct_support >= 2 and distinct_freqs >= 4 and len(group) >= distinct_freqs:
            score += 1
            reasons.append("repeated or multi-tuning spectral setups")
        if re.search(r"\b(cal|bandpass|phase|flux|check|j\d{3,4}[+-]\d{3,4})\b", target_text):
            score += 1
            reasons.append("calibration-like target names present")
        if len(group) >= 12:
            score += 1
            reasons.append(f"{len(group)} observation/subband rows")

        confidence = "unlikely"
        if score >= 6:
            confidence = "likely"
        elif score >= 3:
            confidence = "possible"
        if score == 0:
            continue
        rows.append({
            "proposal_id": as_text(project),
            "confidence": confidence,
            "score": score,
            "reasons": "; ".join(reasons),
            "diagnostic_warning": "Diagnostic only: Bandwidth Switching likelihood is inferred from public spectral setup metadata, not proven calibration intent.",
            "target_name": first_nonempty(group["target_name"].tolist()) if "target_name" in group.columns else "",
            "rows": int(len(group)),
            "n_mous": _nunique(group, "member_ous_uid"),
            "n_eb": _nunique(group, "asdm_uid"),
            "pi_name": first_nonempty(group["pi_name"].tolist()) if "pi_name" in group.columns else "",
            "obs_title": first_nonempty(group["obs_title"].tolist()) if "obs_title" in group.columns else "",
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["score", "proposal_id"], ascending=[False, True])
