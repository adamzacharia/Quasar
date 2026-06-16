"""ALMA QA2 report lookup helpers.

The ALMA archive exposes small per-MOUS QA2 report PDFs with a predictable
filename derived from the Member OUS UID. These helpers keep that URL mapping,
PDF extraction, and capped batch lookup logic out of the API view layer.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
import os
import re
from pathlib import Path
import sqlite3
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter


QA2_UNKNOWN = "Unknown"
QA2_REPORT_BASE_URL = "https://almascience.nrao.edu/dataPortal"
QA2_CACHEABLE_STATUSES = {"Pass", "SemiPass", "Fail"}
QA2_CACHE_FILENAME = "alma_qa2_statuses.sqlite3"

_CACHE_LOCK = threading.Lock()
_SESSION_LOCK = threading.Lock()
_SHARED_SESSION: Optional[requests.Session] = None
_SHARED_SESSION_POOL_SIZE = 0


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
    cache_hits: int = 0
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


def _env_int(name: str, default: int, min_value: int = 0) -> int:
    try:
        return max(min_value, int(os.getenv(name, str(default)) or default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, min_value: float = 0.0) -> float:
    try:
        return max(min_value, float(os.getenv(name, str(default)) or default))
    except (TypeError, ValueError):
        return default


def _cache_path(cache_path: Optional[object] = None) -> Path:
    if cache_path:
        return Path(str(cache_path))
    configured = os.getenv("QUASAR_QA2_CACHE_PATH")
    if configured:
        return Path(configured)
    return Path(os.getenv("CACHE_DIR", "./cache")) / QA2_CACHE_FILENAME


def _get_shared_session(pool_size: int = 40) -> requests.Session:
    global _SHARED_SESSION, _SHARED_SESSION_POOL_SIZE

    pool_size = max(1, int(pool_size or 1))
    with _SESSION_LOCK:
        if _SHARED_SESSION is None or pool_size > _SHARED_SESSION_POOL_SIZE:
            if _SHARED_SESSION is not None:
                try:
                    _SHARED_SESSION.close()
                except Exception:
                    pass
            session = requests.Session()
            adapter = HTTPAdapter(
                pool_connections=pool_size,
                pool_maxsize=pool_size,
                max_retries=0,
                pool_block=False,
            )
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            _SHARED_SESSION = session
            _SHARED_SESSION_POOL_SIZE = pool_size
        return _SHARED_SESSION


def _shared_get(url: str, **kwargs) -> requests.Response:
    pool_size = _env_int("QUASAR_QA2_MAX_WORKERS", 40, min_value=1)
    return _get_shared_session(pool_size=pool_size).get(url, **kwargs)


def _ensure_cache_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS qa2_status_cache (
            normalized_uid TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            report_url TEXT NOT NULL,
            fetched_at REAL NOT NULL
        )
        """
    )


def _read_cached_statuses(
    normalized_uids: Iterable[str],
    ttl_days: float,
    cache_path: Optional[object] = None,
) -> Dict[str, str]:
    uids = [uid for uid in normalized_uids if uid]
    if not uids or ttl_days <= 0:
        return {}

    path = _cache_path(cache_path)
    if not path.exists():
        return {}

    cutoff = time.time() - (ttl_days * 86400.0)
    placeholders = ",".join("?" for _ in uids)
    params = [*uids, cutoff, *sorted(QA2_CACHEABLE_STATUSES)]
    try:
        with _CACHE_LOCK:
            with sqlite3.connect(path) as conn:
                rows = conn.execute(
                    f"""
                    SELECT normalized_uid, status
                    FROM qa2_status_cache
                    WHERE normalized_uid IN ({placeholders})
                      AND fetched_at >= ?
                      AND status IN ({",".join("?" for _ in QA2_CACHEABLE_STATUSES)})
                    """,
                    params,
                ).fetchall()
    except sqlite3.Error:
        return {}

    return {str(uid): str(status) for uid, status in rows}


def _write_cached_statuses(
    results: Iterable[QA2StatusResult],
    cache_path: Optional[object] = None,
) -> None:
    rows = [
        (
            result.normalized_uid,
            result.status,
            result.report_url,
            time.time(),
        )
        for result in results
        if result.normalized_uid
        and result.status in QA2_CACHEABLE_STATUSES
        and not result.error
    ]
    if not rows:
        return

    path = _cache_path(cache_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _CACHE_LOCK:
            with sqlite3.connect(path) as conn:
                _ensure_cache_table(conn)
                conn.executemany(
                    """
                    INSERT INTO qa2_status_cache (normalized_uid, status, report_url, fetched_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(normalized_uid) DO UPDATE SET
                        status = excluded.status,
                        report_url = excluded.report_url,
                        fetched_at = excluded.fetched_at
                    """,
                    rows,
                )
                conn.commit()
    except sqlite3.Error:
        return


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
            for index in range(page_count):
                status = parse_qa2_status(doc[index].get_text("text"))
                if status != QA2_UNKNOWN:
                    return status
        finally:
            doc.close()
    except Exception:
        return QA2_UNKNOWN

    return QA2_UNKNOWN


def fetch_qa2_status(
    mous_uid: object,
    timeout_s: float = 6.0,
    http_get: Optional[Callable[..., requests.Response]] = None,
) -> QA2StatusResult:
    normalized = normalize_mous_uid(mous_uid)
    if not normalized:
        return QA2StatusResult(normalized_uid="", error="Invalid MOUS UID")

    url = qa2_report_url(normalized)
    getter = http_get or _shared_get
    try:
        response = getter(
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
    max_lookup: Optional[int] = None,
    per_request_timeout_s: float = 6.0,
    overall_timeout_s: float = 10.0,
    max_workers: Optional[int] = None,
    cache_ttl_days: Optional[float] = None,
    cache_path: Optional[object] = None,
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

    if max_lookup is None:
        max_lookup = _env_int("QUASAR_QA2_MAX_LOOKUP", 40, min_value=0)
    if max_workers is None:
        max_workers = _env_int("QUASAR_QA2_MAX_WORKERS", 40, min_value=1)
    if cache_ttl_days is None:
        cache_ttl_days = _env_float("QUASAR_QA2_CACHE_TTL_DAYS", 30.0, min_value=0.0)

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

    cached = _read_cached_statuses(selected, cache_ttl_days, cache_path=cache_path)
    statuses.update(cached)
    to_fetch = [uid for uid in selected if uid not in cached]
    cache_hits = len(cached)
    if not to_fetch:
        return QA2BatchResult(
            statuses=statuses,
            requested=len(unique),
            looked_up=len(selected),
            cache_hits=cache_hits,
            capped=capped,
        )

    worker_count = max(1, min(int(max_workers or 1), len(to_fetch)))
    _get_shared_session(pool_size=worker_count)
    executor = ThreadPoolExecutor(max_workers=worker_count)
    future_to_uid = {
        executor.submit(fetch_qa2_status, uid, per_request_timeout_s): uid
        for uid in to_fetch
    }
    done, pending = wait(future_to_uid.keys(), timeout=max(0.1, float(overall_timeout_s)))

    errors: Dict[str, str] = {}
    timed_out = bool(pending)
    cacheable_results: List[QA2StatusResult] = []
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
        elif result.status in QA2_CACHEABLE_STATUSES:
            cacheable_results.append(result)

    executor.shutdown(wait=False, cancel_futures=True)
    _write_cached_statuses(cacheable_results, cache_path=cache_path)
    return QA2BatchResult(
        statuses=statuses,
        requested=len(unique),
        looked_up=cache_hits + len(done),
        cache_hits=cache_hits,
        capped=capped,
        timed_out=timed_out,
        errors=errors,
    )
