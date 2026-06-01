"""Cross-archive source matching helpers for coordinate source catalogs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# Small built-in seed catalog of well-known Perseus embedded protostars.
# Coordinates are ICRS degrees and serve as query centers; users can later
# replace this with an uploaded/catalog-sourced list.
PERSEUS_PROTOSTARS: List[Dict[str, Any]] = [
    {"source_name": "L1448 IRS3B", "ra": 51.4126, "dec": 30.7343},
    {"source_name": "L1448-mm", "ra": 51.4087, "dec": 30.7348},
    {"source_name": "NGC 1333 IRAS 2A", "ra": 52.2657, "dec": 31.2420},
    {"source_name": "NGC 1333 IRAS 4A", "ra": 52.2943, "dec": 31.2239},
    {"source_name": "NGC 1333 IRAS 4B", "ra": 52.3001, "dec": 31.2189},
    {"source_name": "SVS 13", "ra": 52.2650, "dec": 31.2677},
    {"source_name": "B1-c", "ra": 53.3164, "dec": 31.1636},
    {"source_name": "B1-b", "ra": 53.3150, "dec": 31.1330},
    {"source_name": "HH 211", "ra": 55.9803, "dec": 32.0030},
    {"source_name": "IC 348 MMS", "ra": 55.9670, "dec": 32.0296},
    {"source_name": "B5-IRS1", "ra": 56.9153, "dec": 32.8739},
    {"source_name": "L1455 IRS1", "ra": 51.9996, "dec": 30.1487},
]


def as_text(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    return "" if text.lower() == "nan" else text


def _first_value(row: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and as_text(row.get(key)):
            return row.get(key)
    return None


def normalize_source_catalog(
    catalog_name: str = "perseus_protostars",
    sources: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    max_sources: Optional[int] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Return a normalized coordinate source catalog.

    ``sources`` lets the agent cross-match any caller-provided catalog.  Each
    item may use common aliases such as name/source_name/target and ra/s_ra.
    The built-in Perseus catalog remains available for backwards compatibility.
    """
    limit = max(1, int(max_sources or 500))

    if sources:
        normalized: List[Dict[str, Any]] = []
        for idx, raw in enumerate(sources):
            if not isinstance(raw, dict):
                continue
            name = as_text(_first_value(raw, ["source_name", "name", "target_name", "target", "id"]))
            ra_value = _first_value(raw, ["ra", "s_ra", "ra_deg", "RA", "RA_deg"])
            dec_value = _first_value(raw, ["dec", "s_dec", "dec_deg", "DEC", "Dec", "DEC_deg"])
            try:
                ra = float(ra_value)
                dec = float(dec_value)
            except (TypeError, ValueError):
                continue
            if not (0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0):
                continue
            normalized.append({
                "source_name": name or f"source_{idx + 1}",
                "ra": ra,
                "dec": dec,
            })
            if len(normalized) >= limit:
                break
        if not normalized:
            raise ValueError("No valid sources were provided. Each source needs name/source_name, ra, and dec in degrees.")
        return "inline_sources", normalized

    catalog_key = as_text(catalog_name or "perseus_protostars").lower()
    if catalog_key in {"perseus_protostars", "perseus"}:
        return "perseus_protostars", PERSEUS_PROTOSTARS[:limit]

    raise ValueError(
        f"Unsupported catalog_name: {catalog_name}. Provide `sources` with RA/Dec for arbitrary catalogs."
    )


def alma_bulk_cone_adql(sources: Iterable[Dict[str, Any]], radius_arcsec: float, top: int = 5000) -> str:
    radius_deg = max(float(radius_arcsec), 0.1) / 3600.0
    conditions = []
    for source in sources:
        conditions.append(
            "CONTAINS(POINT('ICRS', s_ra, s_dec), "
            f"CIRCLE('ICRS', {float(source['ra']):.8f}, {float(source['dec']):.8f}, {radius_deg:.8f})) = 1"
        )
    where = " OR ".join(conditions) if conditions else "1 = 0"
    return f"""
SELECT TOP {max(1, min(int(top or 5000), 20000))}
       target_name, proposal_id, member_ous_uid, obs_publisher_did,
       s_ra, s_dec, frequency, bandwidth, band_list, dataproduct_type,
       scientific_category, science_keyword, obs_title, pi_name,
       t_exptime, s_resolution, obs_release_date
FROM ivoa.obscore
WHERE {where}
ORDER BY proposal_id
"""


def _angular_sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    import math

    ra1r, dec1r = math.radians(ra1), math.radians(dec1)
    ra2r, dec2r = math.radians(ra2), math.radians(dec2)
    dra = (ra2r - ra1r) * math.cos(0.5 * (dec1r + dec2r))
    ddec = dec2r - dec1r
    return math.degrees(math.sqrt(dra * dra + ddec * ddec)) * 3600.0


def attach_nearest_source(
    observations: pd.DataFrame,
    sources: List[Dict[str, Any]],
    *,
    radius_arcsec: float,
    ra_col: str = "s_ra",
    dec_col: str = "s_dec",
) -> pd.DataFrame:
    if observations is None or observations.empty or ra_col not in observations.columns or dec_col not in observations.columns:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for _, row in observations.iterrows():
        try:
            ra = float(row[ra_col])
            dec = float(row[dec_col])
        except (TypeError, ValueError):
            continue
        matches = [
            (source, _angular_sep_arcsec(float(source["ra"]), float(source["dec"]), ra, dec))
            for source in sources
        ]
        source, sep = min(matches, key=lambda item: item[1])
        if sep <= radius_arcsec:
            enriched = row.to_dict()
            enriched["source_name"] = source["source_name"]
            enriched["source_ra"] = source["ra"]
            enriched["source_dec"] = source["dec"]
            enriched["match_sep_arcsec"] = round(sep, 3)
            rows.append(enriched)
    return pd.DataFrame(rows)


def summarize_alma_jwst_matches(
    sources: List[Dict[str, Any]],
    alma_matches: pd.DataFrame,
    mast_by_source: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    return summarize_cross_archive_matches(sources, alma_matches, mast_by_source, ["ALMA", "JWST"])


def summarize_cross_archive_matches(
    sources: List[Dict[str, Any]],
    alma_matches: pd.DataFrame,
    mast_by_source: Dict[str, pd.DataFrame],
    archives: Sequence[str],
    require_all_archives: bool = False,
) -> pd.DataFrame:
    """Summarize per-source matches across the requested archive families."""
    requested = {as_text(a).upper() for a in (archives or ["ALMA", "JWST"])}
    mast_requested = bool(requested & {"MAST", "JWST", "HST", "TESS", "KEPLER", "K2", "GALEX", "SWIFT"})
    rows: List[Dict[str, Any]] = []
    for source in sources:
        name = source["source_name"]
        alma = alma_matches[alma_matches["source_name"] == name] if alma_matches is not None and not alma_matches.empty else pd.DataFrame()
        mast = mast_by_source.get(name, pd.DataFrame())
        mast_filtered = mast
        mission_filters = sorted(requested & {"JWST", "HST", "TESS", "KEPLER", "K2", "GALEX", "SWIFT"})
        if mission_filters and not mast.empty:
            collection_col = "telescope" if "telescope" in mast.columns else "obs_collection"
            if collection_col in mast.columns:
                pattern = "|".join(mission_filters)
                mast_filtered = mast[mast[collection_col].astype(str).str.upper().str.contains(pattern, na=False)].copy()

        if require_all_archives:
            if "ALMA" in requested and alma.empty:
                continue
            if mast_requested and mast_filtered.empty:
                continue

        mast_id_col = "project_code" if "project_code" in mast_filtered.columns else "proposal_id"
        collection_col = "telescope" if "telescope" in mast_filtered.columns else "obs_collection"
        row = {
            "source_name": name,
            "s_ra": float(source["ra"]),
            "s_dec": float(source["dec"]),
            "alma_observations": int(len(alma)),
            "alma_projects": int(alma["proposal_id"].nunique()) if "proposal_id" in alma.columns else 0,
            "alma_project_ids": ", ".join(sorted({as_text(v) for v in alma.get("proposal_id", pd.Series(dtype=str)).dropna().tolist() if as_text(v)})[:8]),
            "mast_observations": int(len(mast_filtered)),
            "mast_programs": int(mast_filtered[mast_id_col].nunique()) if mast_id_col in mast_filtered.columns else 0,
            "mast_collections": ", ".join(sorted({as_text(v) for v in mast_filtered.get(collection_col, pd.Series(dtype=str)).dropna().tolist() if as_text(v)})[:6]) if collection_col in mast_filtered.columns else "",
            "mast_instruments": ", ".join(sorted({as_text(v) for v in mast_filtered.get("instrument_name", pd.Series(dtype=str)).dropna().tolist() if as_text(v)})[:6]),
        }
        if "JWST" in requested:
            row["jwst_observations"] = row["mast_observations"]
            row["jwst_programs"] = row["mast_programs"]
            row["jwst_instruments"] = row["mast_instruments"]
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["alma_projects", "mast_observations", "source_name"], ascending=[False, False, True]) if rows else pd.DataFrame()
