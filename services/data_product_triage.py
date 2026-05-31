"""Helpers for ALMA data-product discovery and FITS triage.

This module intentionally stays free of network calls.  The agent owns
archive/DataLink/FITS service calls; these helpers classify requests,
summarize candidate projects, rank products, and turn FITS headers into
usable triage signals.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


PROJECT_CODE_RE = re.compile(r"\b\d{4}\.\d\.\d{5}\.[A-Za-z]\b")
MOUS_UID_RE = re.compile(
    r"\buid://[A-Za-z0-9]+/[A-Za-z0-9]+/[A-Za-z0-9]+\b|"
    r"\buid___[A-Za-z0-9]+_[A-Za-z0-9]+_[A-Za-z0-9]+\b"
)
DATASET_ID_RE = re.compile(r"\bivo://[^\s,;]+", re.IGNORECASE)
BAND_RE = re.compile(r"\bband\s*([3-9]|10)\b|\bb([3-9]|10)\b", re.IGNORECASE)


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
    """Filter an ALMA observation dataframe to rows matching a band value."""
    if df is None or df.empty or not band:
        return df

    band_col = next((col for col in ("band_list", "Band", "band") if col in df.columns), None)
    if not band_col:
        return df

    needle = as_text(band)
    mask = df[band_col].astype(str).str.contains(rf"(^|[^0-9]){re.escape(needle)}([^0-9]|$)", regex=True, na=False)
    return df[mask].copy()


def _first_nonempty(series: Iterable[Any]) -> str:
    for value in series:
        text = as_text(value)
        if text and text.lower() != "nan":
            return text
    return ""


def summarize_project_options(df: pd.DataFrame, max_projects: Optional[int] = None) -> pd.DataFrame:
    """Build a project-code picker table from ALMA observation rows."""
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
        bands = []
        if "band_list" in group.columns:
            bands = sorted({as_text(v) for v in group["band_list"].dropna().tolist() if as_text(v)})
        elif "Band" in group.columns:
            bands = sorted({as_text(v) for v in group["Band"].dropna().tolist() if as_text(v)})

        target = _first_nonempty(group["target_name"].tolist()) if "target_name" in group.columns else ""
        title = _first_nonempty(group["obs_title"].tolist()) if "obs_title" in group.columns else ""
        pi = _first_nonempty(group["pi_name"].tolist()) if "pi_name" in group.columns else ""
        release = _first_nonempty(group["obs_release_date"].tolist()) if "obs_release_date" in group.columns else ""
        product_type = _first_nonempty(group["dataproduct_type"].tolist()) if "dataproduct_type" in group.columns else ""
        mous_count = int(group["member_ous_uid"].dropna().nunique()) if "member_ous_uid" in group.columns else 0

        rows.append({
            "proposal_id": project_text,
            "target_name": target,
            "band_list": ", ".join(bands),
            "observations": int(len(group)),
            "member_ous_count": mous_count,
            "dataproduct_type": product_type,
            "pi_name": pi,
            "obs_title": title,
            "obs_release_date": release,
        })

    rows.sort(key=lambda row: (-int(row.get("member_ous_count") or 0), -int(row.get("observations") or 0), row.get("proposal_id", "")))
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


def classify_product(file_info: Dict[str, Any]) -> str:
    filename = as_text(file_info.get("filename")).lower()
    description = as_text(file_info.get("description")).lower()
    text = f"{filename} {description}"
    if not is_fits_product(file_info):
        if filename.endswith((".tar", ".tgz", ".tar.gz")):
            return "archive bundle"
        if ".ms" in filename or "measurement" in text:
            return "measurement set"
        return "non-FITS product"
    if "pbcor" in text:
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


def product_rank(file_info: Dict[str, Any]) -> tuple:
    filename = as_text(file_info.get("filename")).lower()
    size_mb = float(file_info.get("size_mb") or 0)
    fits_rank = 0 if is_fits_product(file_info) else 5
    pbcor_rank = 0 if "pbcor" in filename else 1
    moment_rank = 0 if any(token in filename for token in ("mom0", "moment0", "moment_0")) else 1
    cube_rank = 1 if "cube" in filename else 0
    large_rank = 1 if size_mb > 500 else 0
    return (fits_rank, pbcor_rank, moment_rank, cube_rank, large_rank, size_mb, filename)


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
    score = 100
    if not metadata or not metadata.get("success", False):
        score -= 45
    score -= min(40, len(warnings) * 12)
    try:
        if float(file_info.get("size_mb") or 0) > 500:
            score -= 10
    except (TypeError, ValueError):
        pass
    return max(0, min(100, score))


def build_product_row(
    file_info: Dict[str, Any],
    *,
    member_ous_uid: str = "",
    proposal_id: str = "",
    target_name: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    kind = classify_product(file_info)
    warnings = header_warnings(metadata or {}, kind) if metadata is not None else []
    score = readiness_score(metadata or {}, warnings, file_info) if metadata is not None else ""
    return {
        "filename": as_text(file_info.get("filename")),
        "product_kind": kind,
        "size_mb": file_info.get("size_mb") or 0,
        "proposal_id": proposal_id,
        "target_name": target_name,
        "member_ous_uid": member_ous_uid,
        "triage_status": "header checked" if metadata is not None else "listed only",
        "readiness_score": score,
        "warnings": "; ".join(warnings),
        "access_url": as_text(file_info.get("access_url")),
    }
