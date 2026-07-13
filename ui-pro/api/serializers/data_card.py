"""Data-card serialization: per-archive column maps, demographics, QA2, links.

Extracted verbatim from ``api/main.py``. ``_build_data_card_event`` turns an
agent ``run_result`` (with a pandas DataFrame) into the ``type: "data"`` SSE
event payload the frontend renders as a table card — including the hardcoded
per-archive display-column maps, sky-preview thumbnails, demographics, and the
per-archive browse link.

Behaviour-preservation note: ``import pandas as pd`` is intentionally kept
*function-local* inside ``_build_data_card_event`` (exactly as it was in
``main.py``). ``_compute_demographics`` references ``pd`` but never imports it
and has no module-level ``pd`` in scope, so its ``obs_release_date`` /
resolution-bin branches raise ``NameError`` which their surrounding
``try/except Exception`` swallows — meaning ``observationYears`` /
``resolutionBins`` never populate today. Do NOT hoist pandas to module scope:
that would silently change data-card output. (Tracked as a latent bug for a
separate, benchmark-gated fix.)
"""

import json
import re
from typing import Any, Dict, List, Optional

from api import bootstrap  # noqa: F401  (sys.path + shims for the utils import below)

from utils.archive_links import build_archive_link, infer_archive_kind

from api.serializers.cadc import _fetch_cadc_preview_urls


def _normalize_qa2_value(value: Any) -> str:
    """Normalize archive-table QA2 values for the existing UI badge.

    ALMA's matrix exposes science QA2 as PASS or SEMIPASS. The ObsCore
    qa2_passed boolean uses T/F, where F corresponds to SEMIPASS in that
    matrix rather than a third visible "Fail" state.
    """
    if isinstance(value, bool):
        return "Pass" if value else "SemiPass"
    if value is None:
        return "Unknown"

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>", "-", "--", "unknown"}:
        return "Unknown"

    compact = re.sub(r"[\s_-]+", "", text).lower()
    if compact in {"t", "true", "y", "yes", "1", "pass", "passed", "qa2pass", "qa2passed"}:
        return "Pass"
    if compact in {"f", "false", "n", "no", "0", "fail", "failed", "qa2fail", "qa2failed"}:
        return "SemiPass"
    if compact in {"semipass", "semipassed", "qa2semipass", "qa2semipassed"}:
        return "SemiPass"
    if "semipass" in compact:
        return "SemiPass"
    if "fail" in compact:
        return "SemiPass"
    if "pass" in compact:
        return "Pass"
    return text


def _qa2_status_from_table(df: Any) -> Optional[Any]:
    """Return QA2 status from table columns; do not derive it from report PDFs."""
    for column in ("qa2_status", "QA2", "qa2_passed", "qa2Passed", "qa2"):
        if hasattr(df, "columns") and column in df.columns:
            statuses = df[column].map(_normalize_qa2_value)
            if statuses.astype(str).str.lower().ne("unknown").any():
                return statuses
    return None


# Per-observation footprint overlays (STC-S s_region) for the sky map. A.footprintsFromSTCS
# parses these on the client. Bounded so the card payload stays small and Aladin stays
# responsive even for large obscore result sets. (T7.2)
_STCS_SHAPE_TOKENS = ("POLYGON", "CIRCLE", "BOX", "POSITION", "UNION", "CONVEX", "ELLIPSE")
_MAX_SKY_FOOTPRINTS = 512
_MAX_STCS_LEN = 20000


def _looks_like_stcs(text: str) -> bool:
    """True when a string plausibly is an STC-S region (starts with a shape token).

    STC-S puts the shape first (``POLYGON ICRS ...``, ``CIRCLE J2000 ...``,
    ``Union(...)``), so a prefix check both accepts real regions and rejects
    stray text / NaN without a full parser."""
    return text.strip().upper().startswith(_STCS_SHAPE_TOKENS)


# ── Demographics & FITS estimation helper ────────────────────
def _compute_demographics(df) -> tuple:
    """Compute distribution data, FITS estimate, and sky coordinates from a search DataFrame.
    Returns (demographics_dict, fits_estimate_int).
    demographics_dict may include a 'skyCoords' key with [{ra, dec}, ...] entries.
    """
    import math
    demographics: Dict[str, Any] = {}

    if "band_list" in df.columns:
        band_counts = df["band_list"].astype(str).value_counts().head(8)
        demographics["bands"] = {str(k): int(v) for k, v in band_counts.items()}

    if "project_code" in df.columns:
        proj_counts = df["project_code"].value_counts().head(6)
        demographics["projects"] = {str(k): int(v) for k, v in proj_counts.items()}
    elif "proposal_id" in df.columns:
        proj_counts = df["proposal_id"].value_counts().head(6)
        demographics["projects"] = {str(k): int(v) for k, v in proj_counts.items()}

    # Multi-telescope support (CADC results have obs_collection)
    if "obs_collection" in df.columns:
        tel_counts = df["obs_collection"].value_counts().head(8)
        demographics["telescopes"] = {str(k): int(v) for k, v in tel_counts.items()}

    if "instrument_name" in df.columns:
        inst_counts = df["instrument_name"].value_counts().head(6)
        demographics["instruments"] = {str(k): int(v) for k, v in inst_counts.items()}

    # ── Sky coordinates for sky-map widget ──────────────────────
    # Each entry carries the row's POSITIONAL index `i` within the same
    # df.head(10000) slice the table rows[] are built from, plus a display
    # label from the best identifier column — this is what lets a clicked
    # sky marker open the matching table row (click-to-inspect).
    ra_col = next((c for c in ["s_ra", "ra"] if c in df.columns), None)
    dec_col = next((c for c in ["s_dec", "dec"] if c in df.columns), None)
    label_col = next(
        (c for c in ["target_name", "oid", "source_id", "sparcl_id", "obs_id", "id", "name"]
         if c in df.columns),
        None,
    )
    if ra_col and dec_col:
        coords = []
        for i, (_, row) in enumerate(df.head(10000).iterrows()):
            try:
                ra_v = float(row[ra_col])
                dec_v = float(row[dec_col])
                # Filter out NaN and zero-coordinate rows (invalid positions)
                if math.isnan(ra_v) or math.isnan(dec_v):
                    continue
                if abs(ra_v) < 0.001 and abs(dec_v) < 0.001:
                    continue
                coord = {"ra": round(ra_v, 4), "dec": round(dec_v, 4), "i": i}
                if label_col is not None:
                    label_v = row[label_col]
                    if label_v is not None and str(label_v).strip() and str(label_v).lower() != "nan":
                        coord["label"] = str(label_v)[:80]
                coords.append(coord)
            except (ValueError, TypeError):
                pass
        if coords:
            demographics["skyCoords"] = coords

    # ── Per-observation footprints (STC-S s_region) for the sky map ──
    # ObsCore rows (ALMA TAP `SELECT *`, VO) carry an s_region polygon per row;
    # the interactive sky map draws them via A.footprintsFromSTCS. `i` matches
    # the skyCoords / rows[] positional index so a footprint ties back to its
    # row. Bounded (count + per-region length) to keep the payload small. (T7.2)
    if "s_region" in df.columns:
        footprints: List[Dict[str, Any]] = []
        footprints_truncated = False
        for i, (_, row) in enumerate(df.head(10000).iterrows()):
            if len(footprints) >= _MAX_SKY_FOOTPRINTS:
                footprints_truncated = True
                break
            raw = row["s_region"]
            if raw is None:
                continue
            text = str(raw).strip()
            if not text or text.lower() == "nan" or len(text) > _MAX_STCS_LEN:
                continue
            if not _looks_like_stcs(text):
                continue
            footprint: Dict[str, Any] = {"stcs": text, "i": i}
            if label_col is not None:
                label_v = row[label_col]
                if label_v is not None and str(label_v).strip() and str(label_v).lower() != "nan":
                    footprint["label"] = str(label_v)[:80]
            footprints.append(footprint)
        if footprints:
            demographics["skyFootprints"] = footprints
            if footprints_truncated:
                demographics["skyFootprintsTruncated"] = True

    # ── Observation year timeline ────────────────────────────────
    if "obs_release_date" in df.columns:
        try:
            years = pd.to_datetime(df["obs_release_date"], errors="coerce").dt.year.dropna()
            if not years.empty:
                year_counts = years.astype(int).value_counts().sort_index()
                demographics["observationYears"] = {str(k): int(v) for k, v in year_counts.items()}
        except Exception:
            pass

    # ── Science categories ───────────────────────────────────────
    if "scientific_category" in df.columns:
        try:
            cat_counts = df["scientific_category"].dropna().value_counts().head(6)
            if not cat_counts.empty and len(cat_counts) > 1:
                demographics["scienceCategories"] = {str(k): int(v) for k, v in cat_counts.items()}
        except Exception:
            pass

    # ── Angular resolution distribution ──────────────────────────
    res_col = next((c for c in ["spatial_resolution", "s_resolution"] if c in df.columns), None)
    if res_col:
        try:
            resolutions = pd.to_numeric(df[res_col], errors="coerce").dropna()
            if not resolutions.empty and len(resolutions) > 2:
                bins = [0, 0.1, 0.5, 1.0, 5.0, float("inf")]
                labels = ['<0.1"', '0.1-0.5"', '0.5-1"', '1-5"', '>5"']
                binned = pd.cut(resolutions, bins=bins, labels=labels)
                res_counts = binned.value_counts()
                res_dict = {str(k): int(v) for k, v in res_counts.items() if v > 0}
                if len(res_dict) > 1:
                    demographics["resolutionBins"] = res_dict
        except Exception:
            pass

    # FITS file estimation
    fits_estimate = 0
    if "member_ous_uid" in df.columns:
        unique_mous = df["member_ous_uid"].dropna().nunique()
        fits_estimate = unique_mous * 5  # ~5 FITS products per MOUS (conservative)
    elif "obs_publisher_did" in df.columns:
        fits_estimate = int(df["obs_publisher_did"].dropna().nunique())
    elif len(df) > 0:
        fits_estimate = len(df) * 3  # Generic estimate for non-ALMA archives

    return demographics, fits_estimate


def _default_archive_source(df, source_hint: str = "", filter_label: str = "") -> str:
    """Return the label shown above a data card when the tool did not provide one."""
    if filter_label:
        return filter_label
    if source_hint:
        return source_hint

    archive_kind = infer_archive_kind(df, source_hint=source_hint, filter_label=filter_label)
    if archive_kind == "cadc":
        if hasattr(df, "columns") and "obs_collection" in df.columns:
            collections = []
            for value in df["obs_collection"].dropna().astype(str):
                text = value.strip()
                if text and text not in collections:
                    collections.append(text)
                if len(collections) >= 3:
                    break
            if collections:
                return " / ".join(collections)
        return "CADC"
    if archive_kind == "alma":
        return "ALMA Archive"
    if archive_kind == "mast":
        return "MAST"
    if archive_kind == "eso":
        return "ESO"
    if archive_kind == "irsa":
        return "IRSA"
    return "Archive"


def _build_data_card_event(_run_result: dict) -> Optional[tuple]:
    """Build a data card SSE event from a run_result dict.

    Returns (sse_event_str, rich_dt_dict) or None if the result can't be serialized.
    This is extracted so it can be called both during streaming (eager)
    and after streaming (fallback), avoiding the 2-4s delay.
    """
    import math
    import pandas as pd

    result_type = _run_result.get("type", "")
    if result_type != "data":
        return None

    df = _run_result.get("data")
    if df is None or not hasattr(df, "to_dict"):
        return None

    try:
        table_kind = str(_run_result.get("table_kind", "") or "")
        _source = _run_result.get("source", "")
        _filter_label = _run_result.get("filter_label", "")
        archive_kind = infer_archive_kind(df, source_hint=_source, filter_label=_filter_label)
        if archive_kind == "alma" and table_kind != "alma_project_picker":
            qa2_status = _qa2_status_from_table(df)
            if qa2_status is not None:
                # Copy before adding UI-only normalization so we do not mutate agent-owned results.
                df = df.copy()
                df["qa2_status"] = qa2_status
        product_display_cols = [
            ("filename", "File"),
            ("product_kind", "Product"),
            ("size_mb", "Size (MB)"),
            ("proposal_id", "Proposal ID"),
            ("target_name", "Target"),
            ("scan_intent", "Scan Intent"),
            ("member_ous_uid", "MOUS ID"),
            ("qa2_status", "QA2"),
            ("qa2_passed", "QA2"),
            ("triage_status", "Triage"),
            ("readiness_score", "Readiness"),
            ("warnings", "Warnings"),
        ]
        project_picker_cols = [
            ("proposal_id", "Proposal ID"),
            ("target_name", "Target"),
            ("band_list", "Band"),
            ("observations", "Observations"),
            ("member_ous_count", "MOUS Count"),
            ("dataproduct_type", "Type"),
            ("pi_name", "PI"),
            ("obs_title", "Project Title"),
            ("obs_release_date", "Release Date"),
        ]
        alma_display_cols = [
            ("obs_publisher_did", "Project"),
            ("target_name", "Target"),
            ("scan_intent", "Scan Intent"),
            ("qa2_status", "QA2"),
            ("qa2_passed", "QA2"),
            ("obs_collection", "Telescope"),
            ("instrument_name", "Instrument"),
            ("band_list", "Band"),
            ("frequency", "Freq (GHz)"),
            ("frequency_support", "Freq Support"),
            ("cont_sensitivity_bandwidth", "Cont. Sens. (mJy/beam)"),
            ("dataproduct_type", "Type"),
            ("calib_level", "Cal Level"),
            ("spatial_resolution", "Ang. Res. (arcsec)"),
            ("s_resolution", "Ang. Res. (arcsec)"),
            ("velocity_resolution", "Vel. Res. (km/s)"),
            ("spatial_scale_max", "Max Recov. Scale (arcsec)"),
            ("t_exptime", "Int. Time (s)"),
            ("antenna_arrays", "Array"),
            ("is_mosaic", "Mosaic"),
            ("s_fov", "FOV (arcsec)"),
            ("scientific_category", "Science Category"),
            ("science_keyword", "Science Keyword"),
            ("pol_states", "Polarization"),
            ("pwv", "PWV (mm)"),
            ("pi_name", "PI"),
            ("proposal_authors", "Authors"),
            ("obs_release_date", "Release Date"),
            ("obs_title", "Project Title"),
            ("schedblock_name", "SB Name"),
            ("proposal_id", "Proposal ID"),
            ("member_ous_uid", "MOUS ID"),
            ("group_ous_uid", "Group OUS ID"),
            ("asdm_uid", "ASDM UID"),
        ]
        mmu_display_cols = [
            ("source_id", "Source ID"), ("object_id", "Object ID"), ("targetid", "Target ID"),
            ("ticid", "TIC ID"), ("name", "Name"), ("obsid", "ObsID"),
            ("ra", "RA (deg)"), ("dec", "Dec (deg)"),
            ("parallax", "Parallax (mas)"), ("parallax_error", "Parallax err"),
            ("pmra", "PM RA (mas/yr)"), ("pmdec", "PM Dec (mas/yr)"),
            ("phot_g_mean_mag", "G (mag)"), ("phot_bp_mean_mag", "BP (mag)"), ("phot_rp_mean_mag", "RP (mag)"),
            ("radial_velocity", "Radial Vel."), ("ruwe", "RUWE"),
            ("redshift", "Redshift"), ("z", "Redshift"), ("Z", "Redshift"),
            ("zerr", "z err"), ("ZERR", "z err"), ("Z_ERR", "z err"),
            ("ZWARN", "Z warn"), ("ZWARNING", "Z warn"), ("EBV", "E(B-V)"),
            ("FLUX_G", "Flux g"), ("FLUX_R", "Flux r"), ("FLUX_Z", "Flux z"),
            ("VDISP", "V disp (km/s)"),
            ("SPECTROFLUX_G", "Spec flux g"), ("SPECTROFLUX_R", "Spec flux r"), ("SPECTROFLUX_I", "Spec flux i"),
            ("flux_aper_b", "Flux (b)"), ("flux_significance_b", "Flux signif."),
            ("hard_hm", "HR hm"), ("hard_hs", "HR hs"), ("hard_ms", "HR ms"),
            ("var_index_b", "Var index"), ("var_prob_b", "Var prob"),
            ("mag", "Mag"), ("flux", "Flux"), ("tessmag", "TESS (mag)"),
            ("teff", "Teff"), ("logg", "log g"), ("radius", "Radius"),
            ("class", "Class"), ("classification", "Class"), ("spectype", "Spec Type"), ("subtype", "Subtype"), ("subclass", "Subclass"),
            ("plate", "Plate"), ("mjd", "MJD"), ("fiberid", "Fiber ID"), ("exposure", "Exposure"),
            ("survey", "Survey"), ("catalog", "Catalog"),
        ]
        external_display_cols = [
            ("oid", "OID"), ("ndet", "Detections"),
            ("meanra", "RA (deg)"), ("meandec", "Dec (deg)"),
            ("firstmjd", "First MJD"), ("lastmjd", "Last MJD"),
            ("classalerce", "Class"), ("classification", "Class"), ("class", "Class"),
            ("probability", "Probability"), ("prob", "Probability"), ("classifier", "Classifier"),
            ("sparcl_id", "SparCL ID"), ("ra", "RA (deg)"), ("dec", "Dec (deg)"),
            ("distance_arcsec", "Distance (arcsec)"), ("redshift", "Redshift"),
            ("spectype", "Spec Type"), ("data_release", "Data Release"),
        ]
        if table_kind == "alma_products":
            alma_display_cols = product_display_cols
        elif table_kind == "alma_project_picker":
            alma_display_cols = project_picker_cols
        elif table_kind == "mmu_hats":
            known_mmu_cols = {raw for raw, _ in mmu_display_cols}
            alma_display_cols = list(mmu_display_cols) + [(col, str(col)) for col in df.columns if col not in known_mmu_cols]
        elif table_kind == "external_catalog":
            known_external_cols = {raw for raw, _ in external_display_cols}
            alma_display_cols = list(external_display_cols) + [(col, str(col)) for col in df.columns if col not in known_external_cols]
        seen_display = set()
        sel_cols, display_cols = [], []
        for raw, nice in alma_display_cols:
            if raw in df.columns and nice not in seen_display:
                sel_cols.append(raw)
                display_cols.append(nice)
                seen_display.add(nice)

        if not sel_cols:
            sel_cols = list(df.columns[:8])
            display_cols = sel_cols

        _MAX_TABLE_ROWS = 10000
        sub = df[sel_cols].head(_MAX_TABLE_ROWS).copy()
        sub.columns = display_cols

        per_row_links = []
        if table_kind == "alma_products" and "access_url" in df.columns:
            per_row_links = df["access_url"].head(_MAX_TABLE_ROWS).fillna("").tolist()
        elif archive_kind == "cadc" and "obs_id" in df.columns:
            # Build browsable CADC archive links (not raw DataLink URLs)
            _collections = df["obs_collection"].head(_MAX_TABLE_ROWS).fillna("").tolist() if "obs_collection" in df.columns else [""] * min(len(df), _MAX_TABLE_ROWS)
            _obs_ids = df["obs_id"].head(_MAX_TABLE_ROWS).fillna("").tolist()
            import urllib.parse as _urlparse
            per_row_links = [
                f"https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/search/?Observation.observationID={_urlparse.quote(str(oid), safe='')}"
                if oid else ""
                for oid, col in zip(_obs_ids, _collections)
            ]
        elif "member_ous_uid" in df.columns:
            # ALMA archive links via member OUS UID
            per_row_links = [
                f"https://almascience.nrao.edu/aq/?member_ous_id={v}"
                if pd.notna(v) and str(v).strip() else ""
                for v in df["member_ous_uid"].head(_MAX_TABLE_ROWS)
            ]
        elif "access_url" in df.columns:
            # Fallback: use access_url directly (non-DataLink sources)
            per_row_links = df["access_url"].head(_MAX_TABLE_ROWS).fillna("").tolist()

        def _fmt(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return ""
            if isinstance(v, float):
                return f"{v:.3f}".rstrip("0").rstrip(".")
            return str(v)[:60]

        for col in sub.columns:
            sub[col] = sub[col].map(_fmt)

        rows = sub.to_dict("records")
        for i, link in enumerate(per_row_links):
            if link and i < len(rows):
                rows[i]["_link"] = link

        # ── Inject sky preview thumbnail URLs ──────────
        ra_col = next((c for c in ["s_ra", "ra"] if c in df.columns), None)
        dec_col = next((c for c in ["s_dec", "dec"] if c in df.columns), None)
        has_preview = False

        is_cadc = archive_kind == "cadc"
        cadc_collections = set()
        if is_cadc and "obs_collection" in df.columns:
            cadc_collections = set(df["obs_collection"].dropna().astype(str).unique())

        cadc_preview_map = {}
        if is_cadc and "obs_publisher_did" in df.columns:
            pub_ids = df["obs_publisher_did"].head(_MAX_TABLE_ROWS).dropna().astype(str).tolist()
            try:
                cadc_preview_map = _fetch_cadc_preview_urls(pub_ids)
            except Exception:
                pass

        if cadc_preview_map:
            for i, (_, orig_row) in enumerate(df.head(_MAX_TABLE_ROWS).iterrows()):
                if i >= len(rows):
                    break
                pub_id = str(orig_row.get("obs_publisher_did", "")).strip()
                if pub_id in cadc_preview_map:
                    rows[i]["_preview"] = cadc_preview_map[pub_id]
                    has_preview = True
                elif ra_col and dec_col:
                    try:
                        ra_v = float(orig_row[ra_col])
                        dec_v = float(orig_row[dec_col])
                        if not (math.isnan(ra_v) or math.isnan(dec_v)):
                            rows[i]["_preview"] = (
                                f"https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
                                f"?hips=CDS%2FP%2FDSS2%2Fcolor&width=120&height=120"
                                f"&fov=0.033&projection=TAN&coordsys=icrs"
                                f"&ra={ra_v:.6f}&dec={dec_v:.6f}&format=jpg"
                            )
                            has_preview = True
                    except (ValueError, TypeError):
                        pass
        elif ra_col and dec_col:
            for i, (_, orig_row) in enumerate(df.head(_MAX_TABLE_ROWS).iterrows()):
                if i >= len(rows):
                    break
                try:
                    ra_v = float(orig_row[ra_col])
                    dec_v = float(orig_row[dec_col])
                    if not (math.isnan(ra_v) or math.isnan(dec_v)):
                        rows[i]["_preview"] = (
                            f"https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
                            f"?hips=CDS%2FP%2FDSS2%2Fcolor&width=120&height=120"
                            f"&fov=0.033&projection=TAN&coordsys=icrs"
                            f"&ra={ra_v:.6f}&dec={dec_v:.6f}&format=jpg"
                        )
                        has_preview = True
                except (ValueError, TypeError):
                    pass

        # ── Demographics & FITS estimation ─────────────
        demographics, fits_estimate = _compute_demographics(df)
        if table_kind == "alma_products":
            demographics = {}
            if "filename" in df.columns:
                fits_estimate = int(
                    df["filename"].astype(str).str.contains(r"\.fits?(\.gz)?$", case=False, regex=True, na=False).sum()
                )
            else:
                fits_estimate = len(df)
        elif table_kind == "alma_project_picker":
            fits_estimate = 0
        elif table_kind == "mmu_hats":
            fits_estimate = 0
        elif table_kind == "external_catalog":
            fits_estimate = 0

        # ── Detect archive source dynamically ─────────
        _detected_source = _default_archive_source(
            df,
            source_hint=_source,
            filter_label=_filter_label,
        )

        metrics = [{"label": "Results", "value": len(df), "color": "blue"}]
        if "band_list" in df.columns:
            metrics.append({
                "label": "Bands",
                "value": int(df["band_list"].astype(str).nunique()),
                "color": "purple",
            })
        if fits_estimate > 0:
            metrics.append({
                "label": "FITS" if table_kind == "alma_products" else "Est. FITS",
                "value": f"{fits_estimate}" if table_kind == "alma_products" else f"~{fits_estimate}",
                "color": "amber",
            })
        if "obs_collection" in df.columns:
            metrics.append({
                "label": "Telescopes",
                "value": int(df["obs_collection"].nunique()),
                "color": "emerald",
            })

        # ── Build archive link (per-archive) ──────────
        archive_link = build_archive_link(
            df,
            source_hint=_source,
            filter_label=_filter_label,
        )

        warnings = list(_run_result.get("warnings") or [])
        table_payload = {
            "type": "data",
            "metrics": metrics,
            "columns": list(sub.columns),
            "rows": rows,
            "sourceName": _detected_source,
            "warnings": warnings,
            "partial": bool(_run_result.get("partial") or warnings),
            "archiveLink": archive_link,
            "hasRowLinks": any(bool(r.get("_link")) for r in rows),
            "hasPreview": has_preview,
            "demographics": demographics if demographics else None,
            "fitsEstimate": fits_estimate if fits_estimate > 0 else None,
            "tableKind": table_kind or None,
        }
        table_event_str = f"data: {json.dumps(table_payload)}\n\n"
        rich_dt = table_payload.copy()
        rich_dt.pop("type", None)
        return (table_event_str, rich_dt)
    except Exception as e:
        print(f"[WARN] Could not build data card: {e}")
        return None
