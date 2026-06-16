"""ALMA QA2 report lookup helpers.

The ALMA archive exposes small per-MOUS QA2 report PDFs with a predictable
filename derived from the Member OUS UID. These helpers keep that URL mapping,
PDF extraction, and capped batch lookup logic out of the API view layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, wait
import re
from typing import Callable, Dict, Iterable, List, Optional
from urllib.parse import quote

import requests


QA2_UNKNOWN = "Unknown"
QA2_REPORT_BASE_URL = "https://almascience.nrao.edu/dataPortal"


@dataclass(frozen=True)
class QA2StatusResult:
    normalized_uid: str
    status: str = QA2_UNKNOWN
    report_url: str = ""
    error: str = ""


@dataclass(frozen=True)
class QA2BatchResult:
    statuses: Dict[str, str] = field(default_factory=dict)
    requested: int = 0
    looked_up: int = 0
    capped: bool = False
    timed_out: bool = False
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def incomplete_count(self) -> int:
        return sum(1 for status in self.statuses.values() if status == QA2_UNKNOWN)


def normalize_mous_uid(mous_uid: object) -> str:
    """Return the archive filename slug for a Member OUS UID.

    Examples:
        uid://A001/X12a3/X407 -> A001_X12a3_X407
        uid___A001_X12a3_X407 -> A001_X12a3_X407
    """
    text = str(mous_uid or "").strip()
    if not text or text.lower() in {"nan", "none", "-"}:
        return ""

    filename_match = re.search(r"member\.uid___([A-Za-z0-9_]+)\.qa2_report\.pdf", text)
    if filename_match:
        return filename_match.group(1)

    text = text.strip().strip("\"'")
    text = text.replace("\\", "/")
    text = re.sub(r"^https?://[^/]+/dataPortal/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.qa2_report\.pdf$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^member\.uid___", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^uid___", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^uid://", "", text, flags=re.IGNORECASE)
    text = text.strip("/")

    parts = [part for part in re.split(r"[/_]+", text) if part]
    if len(parts) < 3:
        return ""
    return "_".join(parts)


def qa2_report_url(mous_uid: object, base_url: str = QA2_REPORT_BASE_URL) -> str:
    normalized = normalize_mous_uid(mous_uid)
    if not normalized:
        return ""
    filename = f"member.uid___{normalized}.qa2_report.pdf"
    return f"{base_url.rstrip('/')}/{quote(filename, safe='._')}"


def parse_qa2_status(text: str) -> str:
    """Extract a normalized QA2 status from report text."""
    source = text or ""
    match = re.search(
        r"QA2\s+Status\s*:?\s*(?:[^\w]{0,20})"
        r"(Semi\s*-?\s*Pass|SemiPass|Pass|Fail)\b",
        source,
        flags=re.IGNORECASE,
    )
    if not match:
        return QA2_UNKNOWN

    compact = re.sub(r"[\s_-]+", "", match.group(1)).lower()
    if compact == "semipass":
        return "SemiPass"
    if compact == "pass":
        return "Pass"
    if compact == "fail":
        return "Fail"
    return match.group(1).strip() or QA2_UNKNOWN


def extract_qa2_status_from_pdf_bytes(raw_pdf: bytes, max_pages: int = 2) -> str:
    if not raw_pdf:
        return QA2_UNKNOWN

    try:
        import fitz  # PyMuPDF
    except Exception:
        return QA2_UNKNOWN

    try:
        doc = fitz.open(stream=raw_pdf, filetype="pdf")
        try:
            page_count = min(max_pages, len(doc))
            text = "\n".join(doc[i].get_text("text") for i in range(page_count))
        finally:
            doc.close()
    except Exception:
        return QA2_UNKNOWN

    return parse_qa2_status(text)


def fetch_qa2_status(
    mous_uid: object,
    timeout_s: float = 6.0,
    http_get: Callable[..., requests.Response] = requests.get,
) -> QA2StatusResult:
    normalized = normalize_mous_uid(mous_uid)
    if not normalized:
        return QA2StatusResult(normalized_uid="", error="Invalid MOUS UID")

    url = qa2_report_url(normalized)
    try:
        response = http_get(
            url,
            timeout=timeout_s,
            headers={"Accept": "application/pdf"},
        )
        response.raise_for_status()
    except Exception as exc:
        return QA2StatusResult(
            normalized_uid=normalized,
            report_url=url,
            error=str(exc) or "QA2 report request failed",
        )

    content_type = str(response.headers.get("Content-Type", "")).lower()
    if "pdf" not in content_type and not response.content.startswith(b"%PDF"):
        return QA2StatusResult(
            normalized_uid=normalized,
            report_url=url,
            error=f"Unexpected QA2 content type: {content_type or 'unknown'}",
        )

    status = extract_qa2_status_from_pdf_bytes(response.content)
    return QA2StatusResult(normalized_uid=normalized, status=status, report_url=url)


def fetch_qa2_statuses(
    mous_uids: Iterable[object],
    max_lookup: int = 40,
    per_request_timeout_s: float = 6.0,
    overall_timeout_s: float = 10.0,
    max_workers: int = 8,
) -> QA2BatchResult:
    """Fetch QA2 status for unique MOUS UIDs with cap and overall timeout."""
    unique: List[str] = []
    seen = set()
    for value in mous_uids:
        normalized = normalize_mous_uid(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)

    statuses = {normalized: QA2_UNKNOWN for normalized in unique}
    if not unique:
        return QA2BatchResult(statuses=statuses, requested=0, looked_up=0)

    max_lookup = max(0, int(max_lookup or 0))
    selected = unique[:max_lookup]
    capped = len(unique) > len(selected)
    if not selected:
        return QA2BatchResult(
            statuses=statuses,
            requested=len(unique),
            looked_up=0,
            capped=capped,
        )

    worker_count = max(1, min(int(max_workers or 1), len(selected)))
    executor = ThreadPoolExecutor(max_workers=worker_count)
    future_to_uid = {
        executor.submit(fetch_qa2_status, uid, per_request_timeout_s): uid
        for uid in selected
    }
    done, pending = wait(future_to_uid.keys(), timeout=max(0.1, float(overall_timeout_s)))

    errors: Dict[str, str] = {}
    timed_out = bool(pending)
    for future in pending:
        uid = future_to_uid[future]
        errors[uid] = "QA2 lookup timed out"
        future.cancel()

    for future in done:
        uid = future_to_uid[future]
        try:
            result = future.result()
        except Exception as exc:
            errors[uid] = str(exc) or "QA2 lookup failed"
            continue
        statuses[uid] = result.status or QA2_UNKNOWN
        if result.error:
            errors[uid] = result.error

    executor.shutdown(wait=False, cancel_futures=True)
    return QA2BatchResult(
        statuses=statuses,
        requested=len(unique),
        looked_up=len(done),
        capped=capped,
        timed_out=timed_out,
        errors=errors,
    )
