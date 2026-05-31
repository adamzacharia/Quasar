"""Cross-archive source matching helpers for ALMA + MAST/JWST workflows."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

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
    rows: List[Dict[str, Any]] = []
    for source in sources:
        name = source["source_name"]
        alma = alma_matches[alma_matches["source_name"] == name] if alma_matches is not None and not alma_matches.empty else pd.DataFrame()
        mast = mast_by_source.get(name, pd.DataFrame())
        jwst = mast
        if not mast.empty:
            collection_col = "telescope" if "telescope" in mast.columns else "obs_collection"
            if collection_col in mast.columns:
                jwst = mast[mast[collection_col].astype(str).str.upper().str.contains("JWST", na=False)].copy()

        if alma.empty or jwst.empty:
            continue

        rows.append({
            "source_name": name,
            "s_ra": float(source["ra"]),
            "s_dec": float(source["dec"]),
            "alma_observations": int(len(alma)),
            "alma_projects": int(alma["proposal_id"].nunique()) if "proposal_id" in alma.columns else 0,
            "alma_project_ids": ", ".join(sorted({as_text(v) for v in alma.get("proposal_id", pd.Series(dtype=str)).dropna().tolist() if as_text(v)})[:8]),
            "jwst_observations": int(len(jwst)),
            "jwst_programs": int(jwst["project_code"].nunique()) if "project_code" in jwst.columns else (
                int(jwst["proposal_id"].nunique()) if "proposal_id" in jwst.columns else 0
            ),
            "jwst_instruments": ", ".join(sorted({as_text(v) for v in jwst.get("instrument_name", pd.Series(dtype=str)).dropna().tolist() if as_text(v)})[:6]),
        })

    return pd.DataFrame(rows).sort_values(["alma_projects", "jwst_observations", "source_name"], ascending=[False, False, True]) if rows else pd.DataFrame()
