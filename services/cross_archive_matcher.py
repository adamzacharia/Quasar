"""Cross-archive source matching helpers for coordinate source catalogs."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
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


VANDAM_PATH = Path(__file__).resolve().parent / "reference_data" / "perseus_vandam_protostars.json"


def perseus_protostars() -> Tuple[List[Dict[str, Any]], str]:
    """The Perseus protostar list and its provenance: the VANDAM survey sample (Tobin et al. 2016, ApJ 818, 73,
    97 protostars) when the reference file is present, else the 12-source seed list. ALMAbench 2026-10-15 A25: the
    seed list gave '4 of 12' where the expected answer is about 40 of the roughly 100 VANDAM protostars."""
    try:
        data = json.loads(VANDAM_PATH.read_text(encoding="utf-8"))
        rows = [{"source_name": r["source_name"] + (f" ({r['other_names']})" if r.get("other_names") else ""),
                 "ra": float(r["ra"]), "dec": float(r["dec"])} for r in data["sources"]]
        if rows:
            return rows, data.get("citation", "VANDAM")
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return list(PERSEUS_PROTOSTARS), "built-in 12-source seed list (VANDAM reference file missing)"


def as_text(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    return "" if text.lower() == "nan" else text


def _first_value(row: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and as_text(row.get(key)):
            return row.get(key)
    return None


_VIZIER_NAME_COLUMNS = ("Name", "name", "Source", "SName", "ID", "Id", "Seq", "SimbadName", "Field", "FieldA", "Object")


def _vizier_get_catalogs(cat_id: str, limit: int):
    from astroquery.vizier import Vizier

    # _RAJ2000/_DEJ2000 are VizieR's computed J2000 positions in decimal degrees, whatever the table's own columns.
    return Vizier(columns=["*", "_RAJ2000", "_DEJ2000"], row_limit=limit).get_catalogs(cat_id)


def load_vizier_catalog(cat_id: str, *, limit: int = 500, fetch=None) -> Tuple[str, List[Dict[str, Any]]]:
    """Any published source list from VizieR as a cross-match catalogue: ``J/ApJ/890/130`` (first table) or an exact
    table id ``J/ApJ/890/130/table2``. Live 2026-10-16: the VANDAM Orion catalogue (5 tables) loads in 2.9 s. The
    label carries the table id, its VizieR description and the other tables, so the answer can cite the catalogue."""
    cat_id = cat_id.strip().strip("/")
    if not re.match(r"^[A-Za-z0-9+./_-]+$", cat_id):
        raise ValueError(f"Not a VizieR catalogue id: {cat_id!r} (expected e.g. J/ApJ/890/130 or J/ApJ/890/130/table2)")
    tables = (fetch or _vizier_get_catalogs)(cat_id, limit)
    keys = list(tables.keys())
    if not keys:
        raise ValueError(f"VizieR returned no table for {cat_id!r}")
    key = cat_id if cat_id in keys else keys[0]
    table = tables[key]
    names = [c for c in _VIZIER_NAME_COLUMNS if c in table.colnames]
    rows: List[Dict[str, Any]] = []
    for idx, rec in enumerate(table):
        try:
            ra, dec = float(rec["_RAJ2000"]), float(rec["_DEJ2000"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(ra) and math.isfinite(dec)):
            continue
        name = next((as_text(rec[c]).strip() for c in names if as_text(rec[c]).strip()), "") or f"{key.split('/')[-1]}_{idx + 1}"
        rows.append({"source_name": name, "ra": ra, "dec": dec})
        if len(rows) >= limit:
            break
    if not rows:
        raise ValueError(f"VizieR table {key} has no usable J2000 positions")
    desc = as_text(getattr(table, "meta", {}).get("description", "")).strip()
    others = [k for k in keys if k != key]
    label = f"vizier:{key}" + (f" ({desc})" if desc else "") + (f"; other tables: {', '.join(others)}" if others else "")
    return label, rows


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

    raw_name = as_text(catalog_name or "perseus_protostars").strip()
    if raw_name.lower().startswith("vizier:"):
        return load_vizier_catalog(raw_name.split(":", 1)[1].strip(), limit=limit)

    catalog_key = raw_name.lower()
    if catalog_key in {"perseus_protostars", "perseus", "vandam", "perseus_vandam"}:
        rows, _citation = perseus_protostars()
        return "perseus_protostars", rows[:limit]

    raise ValueError(
        f"Unsupported catalog_name: {catalog_name}. Provide `sources` with RA/Dec for arbitrary catalogs."
    )


def alma_bulk_cone_adql(sources: Iterable[Dict[str, Any]], radius_arcsec: float, top: int = 5000,
                        *, footprint: bool = True) -> str:
    from services.alma_science_queries import alma_cone_where

    radius_deg = max(float(radius_arcsec), 0.1) / 3600.0
    conditions = []
    for source in sources:
        # Footprint-aware cone (INTERSECTS on s_region OR the representative
        # point) so a mosaic covering the source is not missed (A-78).
        conditions.append(alma_cone_where(float(source["ra"]), float(source["dec"]), radius_deg,
                                          footprint=footprint))
    where = " OR ".join(conditions) if conditions else "1 = 0"
    return f"""
SELECT TOP {max(1, min(int(top or 5000), 20000))}
       target_name, proposal_id, member_ous_uid, asdm_uid, obs_publisher_did,
       s_ra, s_dec, s_region, frequency, bandwidth, band_list, dataproduct_type,
       scientific_category, science_keyword, obs_title, pi_name,
       t_exptime, s_resolution, obs_release_date, data_rights
FROM ivoa.obscore
WHERE {where}
ORDER BY proposal_id
"""


def catalog_box(sources: Sequence[Dict[str, Any]], margin_deg: float = 0.25) -> Optional[Tuple[float, float, float, float]]:
    """(ra_min, ra_max, dec_min, dec_max) around a compact catalogue (one region such as a cloud or a deep field:
    RA span < 20 deg, no RA wrap, area < 50 deg^2), else None. The margin (0.25 deg, RA scaled by 1/cos dec) keeps
    pointings centred outside the catalogue whose field still covers an edge source: ALMA fields up to 0.5 deg across
    (guard CX-01; the 0.05-deg margin dropped a 0.2-deg mosaic centred 0.08 deg outside)."""
    if not sources:
        return None
    ras = [float(s["ra"]) for s in sources]
    decs = [float(s["dec"]) for s in sources]
    import math

    de_lo, de_hi = min(decs) - margin_deg, max(decs) + margin_deg
    if max(abs(de_lo), abs(de_hi)) >= 89.0:
        # Near a pole an RA box cannot hold the margin (it needs > 14 deg of RA at |dec| 89); the caller falls back
        # to per-source cones (guard CX-01 reopen: a cosine floor silently cut the box short at dec -89).
        return None
    ra_margin = margin_deg / math.cos(math.radians(max(abs(de_lo), abs(de_hi))))
    ra_lo, ra_hi = min(ras) - ra_margin, max(ras) + ra_margin

    area = (ra_hi - ra_lo) * math.cos(math.radians(0.5 * (de_lo + de_hi))) * (de_hi - de_lo)
    if ra_hi - ra_lo > 20.0 or ra_lo < 0.0 or ra_hi > 360.0 or area > 50.0:
        return None
    return ra_lo, ra_hi, de_lo, de_hi


def alma_box_adql(box: Tuple[float, float, float, float]) -> str:
    """ONE query for every ALMA science pointing in a box: distinct (MOUS, source, pointing, field of view). For the
    97-source VANDAM Perseus list it answers in ~9 s where per-source cones ran out of the tool budget (2026-10-16)."""
    ra_lo, ra_hi, de_lo, de_hi = box
    return (
        "SELECT DISTINCT member_ous_uid, proposal_id, target_name, s_ra, s_dec, s_fov, band_list, obs_release_date, "
        "data_rights FROM ivoa.obscore WHERE "
        f"s_ra BETWEEN {ra_lo:.5f} AND {ra_hi:.5f} AND s_dec BETWEEN {de_lo:.5f} AND {de_hi:.5f} "
        "AND science_observation = 'T'"
    )


def attach_covering_sources(
    observations: pd.DataFrame,
    sources: List[Dict[str, Any]],
    *,
    radius_arcsec: float,
) -> pd.DataFrame:
    """Rows whose field covers a source: separation from the pointing centre <= max(radius, s_fov/2). Unlike
    :func:`attach_nearest_source` a mosaic covering several sources counts for each of them, and a pointing whose
    centre is offset from the source still counts when its field contains it (the matching rule 'keep a source
    when the archive covers its position')."""
    if observations is None or observations.empty or not {"s_ra", "s_dec"} <= set(observations.columns):
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for _, row in observations.iterrows():
        try:
            ra, dec = float(row["s_ra"]), float(row["s_dec"])
        except (TypeError, ValueError):
            continue
        try:
            half_fov = float(row.get("s_fov")) * 3600.0 / 2.0
        except (TypeError, ValueError):
            half_fov = 0.0
        if half_fov != half_fov:  # NaN
            half_fov = 0.0
        reach = max(float(radius_arcsec), half_fov)
        for source in sources:
            sep = _angular_sep_arcsec(float(source["ra"]), float(source["dec"]), ra, dec)
            if sep <= reach:
                enriched = row.to_dict()
                enriched.update(source_name=source["source_name"], source_ra=source["ra"], source_dec=source["dec"],
                                match_sep_arcsec=round(sep, 3))
                rows.append(enriched)
    return pd.DataFrame(rows)


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
