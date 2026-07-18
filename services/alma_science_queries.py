"""Deterministic ALMA archive query helpers for science questions.

The agent owns network access to TAP/ALMiner.  This module keeps the
science-specific filtering, grouping, and line-coverage logic isolated so
LLM routing can call one tool and receive auditable tables/counts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
}

CO_LADDER_LINES = [
    "CO(1-0)",
    "CO(2-1)",
    "CO(3-2)",
    "CO(4-3)",
    "CO(5-4)",
    "CO(6-5)",
    "CO(7-6)",
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
    """Normalize compact common names that archive resolvers often space out."""
    clean = as_text(target)
    if re.fullmatch(r"hh\s*212", clean, flags=re.IGNORECASE):
        return "HH 212"
    return clean


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


def project_prefix_where(cycle: int) -> str:
    return f"proposal_id LIKE '{cycle_to_project_prefix(cycle)}%'"


def select_obscore_query(where_clause: str, *, top: int = 5000, order_by: str = "proposal_id") -> str:
    top = max(1, min(int(top or 5000), 20000))
    return f"""
SELECT TOP {top}
       target_name, proposal_id, member_ous_uid, obs_publisher_did,
       frequency, bandwidth, frequency_support, band_list,
       antenna_arrays, dataproduct_type, calib_level,
       scientific_category, science_keyword, obs_title, pi_name,
       s_ra, s_dec, t_exptime, s_resolution, spatial_resolution,
       obs_release_date
FROM ivoa.obscore
WHERE {where_clause}
ORDER BY {order_by}
"""


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


def filter_band(df: pd.DataFrame, band: Optional[int | str]) -> pd.DataFrame:
    if df is None or df.empty or not band:
        return df
    band_col = col(df, ["band_list", "Band", "band"])
    if not band_col:
        return df
    needle = str(band).strip().replace("Band", "").replace("band", "").strip()
    mask = df[band_col].astype(str).str.contains(rf"(^|[^0-9]){re.escape(needle)}([^0-9]|$)", regex=True, na=False)
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


def infer_arrays(text: Any) -> List[str]:
    clean = as_text(text).lower()
    arrays: List[str] = []
    if re.search(r"\b12\s*m\b|12m|array.*twelve|tm[12]", clean):
        arrays.append("12m")
    if re.search(r"\b7\s*m\b|7m|aca|morita", clean):
        arrays.append("7m")
    if re.search(r"\btp\b|total\s*power|single\s*dish", clean):
        arrays.append("TP")
    return arrays


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
        for value in group[array_col].tolist():
            arrays.update(infer_arrays(value))
        normalized_arrays = {"TP" if item == "TP" else item.lower() for item in arrays}
        if normalized_required <= normalized_arrays:
            rows.append({
                "proposal_id": as_text(project),
                "arrays_found": ", ".join(sorted(arrays)),
                "observations": int(len(group)),
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
        name = aliases.get(compact, compact)
        if name not in LINE_REST_FREQ_GHZ and f"{name}(2-1)" in LINE_REST_FREQ_GHZ:
            name = f"{name}(2-1)"
        if name in LINE_REST_FREQ_GHZ and name not in output:
            output.append(name)
    return output


def line_names_for_species(species: Sequence[str] | str) -> List[str]:
    raw_items = [species] if isinstance(species, str) else list(species or [])
    output: List[str] = []
    for item in raw_items:
        compact = as_text(item).replace(" ", "").upper()
        if compact in {"CO", "12CO"}:
            for name in CO_LADDER_LINES:
                if name not in output:
                    output.append(name)
            continue
        for name in line_names_for_input([compact]):
            if name not in output:
                output.append(name)
    return output or CO_LADDER_LINES.copy()


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
    if not project_col:
        return pd.DataFrame()
    required = set(line_names_for_input(lines))
    rows: List[Dict[str, Any]] = []
    for project, group in annotated.groupby(project_col, dropna=True):
        found = set()
        for value in group["covered_lines"].tolist():
            found.update(part.strip() for part in as_text(value).split(",") if part.strip())
        if required <= found:
            rows.append({
                "proposal_id": as_text(project),
                "covered_lines": ", ".join(sorted(found)),
                "observations": int(len(group)),
                "target_name": first_nonempty(group["target_name"].tolist()) if "target_name" in group.columns else "",
                "band_list": first_nonempty(group["band_list"].tolist()) if "band_list" in group.columns else "",
                "pi_name": first_nonempty(group["pi_name"].tolist()) if "pi_name" in group.columns else "",
                "obs_title": first_nonempty(group["obs_title"].tolist()) if "obs_title" in group.columns else "",
                "science_keyword": first_nonempty(group["science_keyword"].tolist()) if "science_keyword" in group.columns else "",
            })
    return pd.DataFrame(rows)


def redshifted_line_projects(
    df: pd.DataFrame,
    *,
    rest_species: Sequence[str] | str = "CO",
    z_min: float = 1.0,
    z_max: float = 2.0,
) -> pd.DataFrame:
    """Summarize projects whose spectral setup overlaps redshifted rest lines."""
    if df is None or df.empty:
        return pd.DataFrame()
    project_col = col(df, ["proposal_id", "project_code"])
    if not project_col:
        return pd.DataFrame()

    z_lo = float(min(z_min, z_max))
    z_hi = float(max(z_min, z_max))
    line_names = line_names_for_species(rest_species)
    hit_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        intervals = observation_intervals_ghz(row)
        if not intervals:
            continue
        for line_name in line_names:
            rest_freq = LINE_REST_FREQ_GHZ[line_name]
            obs_min = rest_freq / (1.0 + z_hi)
            obs_max = rest_freq / (1.0 + z_lo)
            for lo, hi in intervals:
                inter_lo = max(lo, obs_min)
                inter_hi = min(hi, obs_max)
                if inter_lo > inter_hi:
                    continue
                inferred_z_min = max(z_lo, rest_freq / inter_hi - 1.0)
                inferred_z_max = min(z_hi, rest_freq / inter_lo - 1.0)
                hit_rows.append({
                    "proposal_id": as_text(row.get(project_col)),
                    "target_name": as_text(row.get("target_name")),
                    "transition": line_name,
                    "rest_frequency_ghz": round(rest_freq, 6),
                    "observed_frequency_range_ghz": f"{inter_lo:.3f}-{inter_hi:.3f}",
                    "inferred_redshift_range": f"{inferred_z_min:.3f}-{inferred_z_max:.3f}",
                    "band_list": as_text(row.get("band_list")),
                    "pi_name": as_text(row.get("pi_name")),
                    "obs_title": as_text(row.get("obs_title")),
                    "science_keyword": as_text(row.get("science_keyword")),
                })

    hits = pd.DataFrame(hit_rows)
    if hits.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for project, group in hits.groupby("proposal_id", dropna=True):
        rows.append({
            "proposal_id": as_text(project),
            "target_name": first_nonempty(group["target_name"].tolist()),
            "transitions": ", ".join(sorted(set(group["transition"].tolist()))),
            "observed_frequency_ranges_ghz": "; ".join(sorted(set(group["observed_frequency_range_ghz"].tolist()))[:8]),
            "inferred_redshift_ranges": "; ".join(sorted(set(group["inferred_redshift_range"].tolist()))[:8]),
            "observations": int(len(group)),
            "band_list": first_nonempty(group["band_list"].tolist()),
            "pi_name": first_nonempty(group["pi_name"].tolist()),
            "obs_title": first_nonempty(group["obs_title"].tolist()),
            "science_keyword": first_nonempty(group["science_keyword"].tolist()),
        })
    return pd.DataFrame(rows).sort_values(["observations", "proposal_id"], ascending=[False, True])


def summarize_projects(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    project_col = col(df, ["proposal_id", "project_code"])
    if not project_col:
        return df
    rows: List[Dict[str, Any]] = []
    for project, group in df.groupby(project_col, dropna=True):
        rows.append({
            "proposal_id": as_text(project),
            "target_name": first_nonempty(group["target_name"].tolist()) if "target_name" in group.columns else "",
            "observations": int(len(group)),
            "band_list": first_nonempty(group["band_list"].tolist()) if "band_list" in group.columns else "",
            "arrays": first_nonempty(group["antenna_arrays"].tolist()) if "antenna_arrays" in group.columns else "",
            "resolution_arcsec": min([v for v in (_resolution_arcsec(row) for _, row in group.iterrows()) if v is not None], default=""),
            "pi_name": first_nonempty(group["pi_name"].tolist()) if "pi_name" in group.columns else "",
            "obs_title": first_nonempty(group["obs_title"].tolist()) if "obs_title" in group.columns else "",
            "obs_release_date": first_nonempty(group["obs_release_date"].tolist()) if "obs_release_date" in group.columns else "",
        })
    return pd.DataFrame(rows).sort_values(["observations", "proposal_id"], ascending=[False, True])


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
    base_columns = (
        "target_name, proposal_id, member_ous_uid, obs_publisher_did,\n"
        "       frequency, bandwidth, frequency_support, band_list,\n"
        "       antenna_arrays, dataproduct_type, calib_level,\n"
        "       scientific_category, science_keyword, obs_title, pi_name,\n"
        "       s_ra, s_dec, t_exptime, s_resolution, spatial_resolution,\n"
        "       obs_release_date"
    )
    if extra_columns:
        base_columns += ",\n       " + ", ".join(extra_columns)
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
            "observations": int(len(group)),
            "member_ous_uids": " ".join(mous_uids[:20]),
            "target_name": first_nonempty(group["target_name"].tolist()) if "target_name" in group.columns else "",
            "band_list": first_nonempty(group["band_list"].tolist()) if "band_list" in group.columns else "",
            "pi_name": first_nonempty(group["pi_name"].tolist()) if "pi_name" in group.columns else "",
            "obs_title": first_nonempty(group["obs_title"].tolist()) if "obs_title" in group.columns else "",
        })
    return pd.DataFrame(rows).sort_values(
        ["n_publications", "observations", "proposal_id"], ascending=[False, False, True]
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
        interval_counts = [len(observation_intervals_ghz(row)) for _, row in group.iterrows()]
        total_spw_intervals = sum(interval_counts)
        interval_widths = [
            hi - lo
            for _, row in group.iterrows()
            for lo, hi in observation_intervals_ghz(row)
            if hi > lo
        ]
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
            "observations": int(len(group)),
            "pi_name": first_nonempty(group["pi_name"].tolist()) if "pi_name" in group.columns else "",
            "obs_title": first_nonempty(group["obs_title"].tolist()) if "obs_title" in group.columns else "",
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["score", "proposal_id"], ascending=[False, True])
