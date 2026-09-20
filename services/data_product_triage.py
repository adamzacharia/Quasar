"""Helpers for ALMA data-product discovery and FITS triage.

This module intentionally stays free of network calls.  The agent owns
archive/DataLink/FITS service calls; these helpers classify requests,
summarize candidate projects, rank products, and turn FITS headers into
usable triage signals.

Product taxonomy (INT-3; skill products-and-qa.md "Product FITS naming token
families", identifiers-and-packaging.md "What DataLink offers per MOUS"):
there is no clean machine-readable taxonomy of file kinds, so classification
rests on filename conventions plus DataLink ``semantics`` and is deliberately
conservative — the raw filename is always kept.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from services.alma_science_queries import band_tokens, requested_bands, row_matches_band


PROJECT_CODE_RE = re.compile(r"\b\d{4}\.[0-9A-Za-z]\.\d{5}\.[A-Za-z]\b")
MOUS_UID_RE = re.compile(
    r"\buid://[A-Za-z0-9]+/[A-Za-z0-9]+/[A-Za-z0-9]+\b|"
    r"\buid___[A-Za-z0-9]+_[A-Za-z0-9]+_[A-Za-z0-9]+\b"
)
DATASET_ID_RE = re.compile(r"\bivo://[^\s,;]+", re.IGNORECASE)
BAND_RE = re.compile(r"\bband\s*([1-9]|10)\b|\bb([1-9]|10)\b", re.IGNORECASE)

# Calibrator-intent tokens in product filenames (``<source>_<intent>``).
_INTENT_TOKENS = {
    "_sci": "science", "_ph": "phase calibrator", "_bp": "bandpass calibrator",
    "_chk": "check source", "_flux": "flux calibrator", "_pol": "polarization calibrator",
    "_amp": "amplitude calibrator", "pol_leak": "polarization calibrator",
}
_TAR_PART_RE = re.compile(r"_(\d{3})_of_(\d{3})\.tar", re.IGNORECASE)


def as_text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def extract_band(text: str) -> str:
    match = BAND_RE.search(as_text(text))
    if not match:
        return ""
    return (match.group(1) or match.group(2) or "").strip()


def classify_alma_product_request(text: str) -> Dict[str, str]:
    """Classify a product-triage request into exact ID vs target lookup."""
    raw = as_text(text)
    project = PROJECT_CODE_RE.search(raw)
    if project:
        return {
            "kind": "project_code",
            "identifier": project.group(0).upper(),
            "target": "",
            "band": extract_band(raw),
        }

    mous = MOUS_UID_RE.search(raw)
    if mous:
        return {
            "kind": "mous_uid",
            "identifier": mous.group(0),
            "target": "",
            "band": extract_band(raw),
        }

    dataset = DATASET_ID_RE.search(raw)
    if dataset:
        return {
            "kind": "dataset_id",
            "identifier": dataset.group(0),
            "target": "",
            "band": extract_band(raw),
        }

    target = raw
    target = re.sub(r"\b(fetch|find|get|list|show|triage|inspect|analyze|analyse)\b", " ", target, flags=re.I)
    target = re.sub(r"\b(alma|data|products?|fits|files?|archive|archives?|for|from|of|the|please)\b", " ", target, flags=re.I)
    target = BAND_RE.sub(" ", target)
    target = re.sub(r"\s+", " ", target).strip(" ,.;:")
    return {
        "kind": "target",
        "identifier": "",
        "target": target or raw,
        "band": extract_band(raw),
    }


def filter_observations_by_band(df: pd.DataFrame, band: str) -> pd.DataFrame:
    """Filter an ALMA observation dataframe to rows matching a band value.

    Token-aware (band_list is space delimited; band-to-band rows read '5 10').
    """
    if df is None or df.empty or not band:
        return df

    band_col = next((col for col in ("band_list", "Band", "band") if col in df.columns), None)
    if not band_col:
        return df

    wanted = requested_bands(band)
    if not wanted:
        return df
    mask = df[band_col].apply(lambda v: row_matches_band(v, wanted))
    return df[mask].copy()


def _first_nonempty(series: Iterable[Any]) -> str:
    for value in series:
        text = as_text(value)
        if text and text.lower() != "nan":
            return text
    return ""


def _nunique(group: pd.DataFrame, column: str) -> int:
    if column not in group.columns:
        return 0
    values = group[column].dropna().astype(str).str.strip()
    values = values[(values != "") & (values.str.lower() != "nan")]
    return int(values.nunique())


def summarize_project_options(df: pd.DataFrame, max_projects: Optional[int] = None) -> pd.DataFrame:
    """Build a project-code picker table from ALMA observation rows.

    Counts are reported at the right grain: ``rows`` (ObsCore coverage
    records), ``member_ous_count`` (datasets) and ``eb_count`` (distinct
    asdm_uid) — never rows relabelled as observations.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    project_col = next((col for col in ("proposal_id", "project_code", "obs_publisher_did") if col in df.columns), None)
    if not project_col:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for project, group in df.groupby(project_col, dropna=True):
        project_text = as_text(project)
        if not project_text:
            continue
        bands: List[str] = []
        for column in ("band_list", "Band"):
            if column in group.columns:
                for value in group[column].dropna().tolist():
                    for token in band_tokens(value):
                        if token not in bands:
                            bands.append(token)
                break
        bands.sort(key=lambda t: (not t.isdigit(), int(t) if t.isdigit() else 0, t))

        target = _first_nonempty(group["target_name"].tolist()) if "target_name" in group.columns else ""
        title = _first_nonempty(group["obs_title"].tolist()) if "obs_title" in group.columns else ""
        pi = _first_nonempty(group["pi_name"].tolist()) if "pi_name" in group.columns else ""
        release_values = sorted(v for v in (as_text(x) for x in group["obs_release_date"].tolist()) if v and v.lower() != "nan") if "obs_release_date" in group.columns else []
        product_type = _first_nonempty(group["dataproduct_type"].tolist()) if "dataproduct_type" in group.columns else ""
        mous_count = _nunique(group, "member_ous_uid")
        eb_count = _nunique(group, "asdm_uid")
        data_rights = ""
        if "data_rights" in group.columns:
            rights = sorted({as_text(v) for v in group["data_rights"].dropna().tolist() if as_text(v)})
            data_rights = ", ".join(rights)

        rows.append({
            "proposal_id": project_text,
            "target_name": target,
            "band_list": " ".join(bands),
            "rows": int(len(group)),
            "member_ous_count": mous_count,
            "eb_count": eb_count,
            "dataproduct_type": product_type,
            "data_rights": data_rights,
            "pi_name": pi,
            "obs_title": title,
            "obs_release_date": release_values[0] if release_values else "",
            "obs_release_date_max": release_values[-1] if release_values else "",
        })

    rows.sort(key=lambda row: (-int(row.get("member_ous_count") or 0), -int(row.get("rows") or 0), row.get("proposal_id", "")))
    if max_projects:
        return pd.DataFrame(rows[:max_projects])
    return pd.DataFrame(rows)


def unique_values(df: pd.DataFrame, columns: Iterable[str], limit: int = 20) -> List[str]:
    values: List[str] = []
    if df is None or df.empty:
        return values
    for col in columns:
        if col not in df.columns:
            continue
        for value in df[col].dropna().astype(str).tolist():
            clean = as_text(value)
            if clean and clean not in values:
                values.append(clean)
            if len(values) >= limit:
                return values
    return values


def is_fits_product(file_info: Dict[str, Any]) -> bool:
    filename = as_text(file_info.get("filename")).lower()
    content_type = as_text(file_info.get("content_type")).lower()
    return (
        filename.endswith((".fits", ".fit", ".fits.gz", ".fit.gz"))
        or "fits" in content_type
    )


def is_compressed(file_info: Dict[str, Any]) -> bool:
    return as_text(file_info.get("filename")).lower().endswith((".gz", ".tgz", ".tar.gz"))


def product_intent(file_info: Dict[str, Any]) -> str:
    """Science target vs calibrator intent from the ``<source>_<intent>`` token."""
    filename = as_text(file_info.get("filename")).lower()
    for token, label in _INTENT_TOKENS.items():
        if token in filename:
            return label
    return "science" if is_fits_product(file_info) else ""


def product_role(file_info: Dict[str, Any]) -> str:
    """Coarse delivery role: science FITS, pb/mask, README, QA report, weblog,
    scripts, calibration tar, auxiliary tar, product tar, raw ASDM, ..."""
    filename = as_text(file_info.get("filename")).lower()
    semantics = as_text(file_info.get("semantics")).lower()
    if not filename:
        return "unknown"
    if filename.endswith(".asdm.sdm.tar") or "asdm.sdm" in filename or semantics in {"#progenitor", "#package"} and "asdm" in filename:
        return "raw ASDM (per-EB, restore only)"
    if "readme" in filename:
        return "README"
    if "qa2_report" in filename or "qa0_report" in filename or ("qa" in filename and filename.endswith(".pdf")):
        return "QA report"
    if "weblog" in filename:
        return "weblog"
    if "aquareport" in filename or "pipeline_manifest" in filename or filename.endswith(".xml"):
        return "pipeline artifact"
    if "scriptforpi" in filename or "casa_pipescript" in filename or "casa_piperestorescript" in filename or "scriptforcalibration" in filename or filename.endswith(".py"):
        return "script"
    if "auxproducts" in filename or "caltables" in filename or "flagversions" in filename or "calibration" in filename:
        return "calibration"
    if "_auxiliary.tar" in filename or filename.endswith("auxiliary.tar"):
        return "auxiliary tar"
    if _TAR_PART_RE.search(filename):
        return "product tar (split)"
    if filename.endswith((".tar", ".tgz", ".tar.gz")):
        return "tar bundle"
    if is_fits_product(file_info):
        if ".pb." in filename or filename.endswith((".pb.fits", ".pb.fits.gz")):
            return "primary beam"
        if ".mask." in filename or filename.endswith((".mask.fits", ".mask.fits.gz")):
            return "clean mask"
        if ".psf." in filename or ".residual." in filename or ".model." in filename:
            return "imaging diagnostic"
        return "science FITS"
    return "other"


def tar_parts(file_info: Dict[str, Any]) -> Dict[str, Optional[int]]:
    match = _TAR_PART_RE.search(as_text(file_info.get("filename")))
    if not match:
        return {"tar_part": None, "tar_parts_total": None}
    return {"tar_part": int(match.group(1)), "tar_parts_total": int(match.group(2))}


def classify_product(file_info: Dict[str, Any]) -> str:
    filename = as_text(file_info.get("filename")).lower()
    description = as_text(file_info.get("description")).lower()
    text = f"{filename} {description}"
    if not is_fits_product(file_info):
        role = product_role(file_info)
        if role.startswith("raw ASDM"):
            return "raw ASDM tar"
        if role in {"auxiliary tar", "product tar (split)", "tar bundle", "calibration"}:
            return "archive bundle"
        if role in {"README", "QA report", "weblog", "script", "pipeline artifact"}:
            return role
        if ".ms" in filename or "measurement" in text:
            return "measurement set"
        return "non-FITS product"
    if ".pb." in filename or filename.endswith((".pb.fits", ".pb.fits.gz")):
        return "primary beam FITS"
    if ".mask." in filename or filename.endswith((".mask.fits", ".mask.fits.gz")):
        return "clean mask FITS"
    if "pbcor" in text:
        if "cube" in text:
            return "primary-beam-corrected cube"
        if ".cont." in filename or "continuum" in text or ".mfs." in filename:
            return "primary-beam-corrected continuum"
        return "primary-beam-corrected FITS"
    if "mom0" in text or "moment0" in text or "moment_0" in text:
        return "moment-0 map"
    if "mom1" in text or "moment1" in text or "moment_1" in text:
        return "moment-1 map"
    if "cube" in text or "spectral" in text:
        return "spectral cube"
    if "cont" in text or "continuum" in text:
        return "continuum image"
    if "image" in text:
        return "FITS image"
    return "FITS product"


def category_counts(files: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Count delivery roles so triage can state what the MOUS actually holds."""
    counts: Dict[str, int] = {}
    for file_info in files:
        role = product_role(file_info)
        counts[role] = counts.get(role, 0) + 1
    return counts


def _size_mb(file_info: Dict[str, Any]) -> Optional[float]:
    value = file_info.get("size_mb")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def product_rank(file_info: Dict[str, Any]) -> tuple:
    filename = as_text(file_info.get("filename")).lower()
    size_mb = _size_mb(file_info)
    fits_rank = 0 if is_fits_product(file_info) else 5
    intent = product_intent(file_info)
    intent_rank = 0 if intent in {"science", ""} else 1  # calibrator images after science
    pbcor_rank = 0 if "pbcor" in filename else 1
    diag_rank = 1 if any(tok in filename for tok in (".psf.", ".residual.", ".model.", ".pb.", ".mask.")) else 0
    moment_rank = 0 if any(token in filename for token in ("mom0", "moment0", "moment_0")) else 1
    cube_rank = 1 if "cube" in filename else 0
    large_rank = 1 if (size_mb is not None and size_mb > 500) else 0  # unknown size is neutral
    return (fits_rank, intent_rank, pbcor_rank, diag_rank, moment_rank, cube_rank, large_rank,
            size_mb if size_mb is not None else 0.0, filename)


def header_warnings(metadata: Dict[str, Any], product_kind: str = "") -> List[str]:
    if not metadata or not metadata.get("success", False):
        return ["FITS header not readable from remote URL"]

    warnings: List[str] = []
    headers = metadata.get("all_headers") or {}
    if not metadata.get("image_size"):
        warnings.append("missing image dimensions")
    if not (headers.get("CTYPE1") and headers.get("CTYPE2")):
        warnings.append("missing celestial WCS")
    if metadata.get("beam_major_arcsec") is None or metadata.get("beam_minor_arcsec") is None:
        warnings.append("missing synthesized beam")
    if as_text(metadata.get("bunit")).lower() in {"", "unknown"}:
        warnings.append("missing brightness unit")
    if "cube" in product_kind.lower() and metadata.get("rest_freq_ghz") is None and not headers.get("CTYPE3"):
        warnings.append("spectral cube has unclear spectral axis")
    return warnings


def readiness_score(metadata: Dict[str, Any], warnings: List[str], file_info: Dict[str, Any]) -> int:
    """Header-completeness score (0-100). It is NOT a science-readiness
    verdict and does not penalize file size or unknown size."""
    score = 100
    if not metadata or not metadata.get("success", False):
        score -= 45
    score -= min(40, len(warnings) * 12)
    return max(0, min(100, score))


def build_product_row(
    file_info: Dict[str, Any],
    *,
    member_ous_uid: str = "",
    proposal_id: str = "",
    target_name: str = "",
    scan_intent: str = "",
    qa2_passed: Any = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    kind = classify_product(file_info)
    warnings = header_warnings(metadata or {}, kind) if metadata is not None else []
    score = readiness_score(metadata or {}, warnings, file_info) if metadata is not None else ""
    size_mb = _size_mb(file_info)
    parts = tar_parts(file_info)
    row = {
        "filename": as_text(file_info.get("filename")),
        "product_kind": kind,
        "role": product_role(file_info),
        "intent": product_intent(file_info),
        "size_mb": size_mb if size_mb is not None else "",
        "size_known": size_mb is not None,
        "semantics": as_text(file_info.get("semantics")),
        "proposal_id": proposal_id,
        "target_name": target_name,
        "scan_intent": scan_intent,
        "qa2_passed": qa2_passed,
        "member_ous_uid": member_ous_uid,
        "triage_status": "header checked" if metadata is not None else "listed only",
        "readiness_score": score,
        "warnings": "; ".join(warnings),
        "access_url": as_text(file_info.get("access_url")),
    }
    if parts["tar_part"] is not None:
        row["tar_part"] = parts["tar_part"]
        row["tar_parts_total"] = parts["tar_parts_total"]
    return row
