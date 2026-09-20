# integrations/datalink.py
"""
ALMA DataLink Client — File-Level Archive Access

Queries the ALMA DataLink service to enumerate the access alternatives of a
Member OUS UID.  This is the layer that bridges observation-level metadata
(from ALminer/TAP) to the actual downloadable files a researcher needs.

CALLED BY: core/agent.py (tools: list_alma_files, triage_alma_data_products,
           inspect_fits_header), capabilities/alma.py (download preflight)
CALLS:     astroquery.alma (DataLink), ALMA archive HTTPS (VOTable fallback)

Typed inventory (INT-3, skill: identifiers-and-packaging.md "What DataLink
offers per MOUS"): DataLink 1.1 rows are NOT all downloadable files. Exactly
one of ``access_url`` / ``service_def`` / ``error_message`` identifies a row's
access alternative, and an ``access_url`` can itself be a nested DataLink
service. ``list_files`` therefore classifies every row and reports:

  files      direct downloadable files (semantics, content_type, size kept;
             size is ``None`` when the service omits content_length)
  services   service-descriptor rows (SODA etc.), never listed as files
  nested     rows whose access_url is another DataLink endpoint
  errors     per-row error_message rows and service-level faults

and a ``state``:

  ok                    rows returned
  empty_or_unauthorized valid VOTable with zero rows — normal anonymous answer
                        for a proprietary MOUS; NOT "the MOUS does not exist"
  not_found             the service returned a #error / NotFoundFault
  unavailable           every transport failed (outage / timeout)
  partial_parse         only the regex fallback could read the response
"""

from __future__ import annotations

import fnmatch
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# Try astroquery.alma — the primary way to access ALMA DataLink
try:
    from astroquery.alma import Alma
    ASTROQUERY_AVAILABLE = True
except ImportError:
    ASTROQUERY_AVAILABLE = False
    print("[DataLink] WARNING: astroquery not installed. DataLink will use fallback HTTP.")

# Try pyvo for direct DataLink VOTable parsing (fallback)
try:
    import pyvo
    PYVO_AVAILABLE = True
except ImportError:
    PYVO_AVAILABLE = False


ALMA_MIRROR = os.getenv("QUASAR_ALMA_MIRROR", "https://almascience.nrao.edu").rstrip("/")

STATE_OK = "ok"
STATE_EMPTY = "empty_or_unauthorized"
STATE_NOT_FOUND = "not_found"
STATE_UNAVAILABLE = "unavailable"
STATE_PARTIAL = "partial_parse"

_UID_CANON_RE = re.compile(r"uid://([A-Za-z0-9]+)/([A-Za-z0-9]+)/([A-Za-z0-9]+)", re.IGNORECASE)
_UID_SANITIZED_RE = re.compile(r"uid___([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)", re.IGNORECASE)
_UID_BARE_RE = re.compile(r"^([A-Za-z0-9]+)/([A-Za-z0-9]+)/([A-Za-z0-9]+)$")
_VOTABLE_ERROR_RE = re.compile(
    r'<INFO[^>]*name="QUERY_STATUS"[^>]*value="ERROR"[^>]*>(.*?)</INFO>|'
    r'<INFO[^>]*value="ERROR"[^>]*name="QUERY_STATUS"[^>]*>(.*?)</INFO>|'
    r'<INFO[^>]*name="QUERY_STATUS"[^>]*value="ERROR"[^>]*/>',
    re.IGNORECASE | re.DOTALL,
)
_NOT_FOUND_RE = re.compile(r"NotFound|not\s+found|UsageFault|does not exist", re.IGNORECASE)


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "--", "<na>", "masked"} else text


def _length(value: Any) -> Optional[int]:
    """Per-link byte count, or None when the service omitted it. astropy fills
    an empty ``long`` cell with 0 rather than a mask, and no real DataLink
    deliverable is 0 bytes, so 0 is treated as unknown too."""
    text = _text(value)
    if not text:
        return None
    try:
        n = int(float(text))
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def row_to_entry(row: Any, colnames: List[str]) -> Dict[str, Any]:
    """Normalize one DataLink row (astropy Row or dict) into a typed entry dict.

    Shared by the astroquery and HTTP paths so ``semantics``/``description``/
    ALMA-local columns survive on both (A-65).
    """
    def get(name: str) -> Any:
        if name not in colnames:
            return None
        try:
            return row[name]
        except Exception:
            return None

    access_url = _text(get("access_url"))
    service_def = _text(get("service_def"))
    error_message = _text(get("error_message"))
    content_type = _text(get("content_type"))
    semantics = _text(get("semantics"))
    description = _text(get("description"))
    content_length = _length(get("content_length"))
    filename = access_url.split("/")[-1].split("?")[0] if access_url else ""

    if error_message:
        kind = "error"
    elif service_def and not access_url:
        kind = "service"
    elif access_url and ("/datalink/" in access_url.lower() or "content=datalink" in content_type.lower()):
        kind = "nested"
    elif access_url:
        kind = "file"
    else:
        kind = "unknown"

    size_mb = round(content_length / (1024 * 1024), 2) if content_length is not None else None
    entry: Dict[str, Any] = {
        "kind": kind,
        "filename": filename,
        "size_mb": size_mb,
        "content_length": content_length,
        "size_known": content_length is not None,
        "access_url": access_url,
        "content_type": content_type,
        "description": description,
        "semantics": semantics,
    }
    if service_def:
        entry["service_def"] = service_def
    if error_message:
        entry["error_message"] = error_message
    known = {"access_url", "service_def", "error_message", "content_type", "semantics",
             "description", "content_length", "ID", "id"}
    extra = {}
    for name in colnames:
        if name in known:
            continue
        value = _text(get(name))
        if value:
            extra[name] = value
    if extra:
        entry["extra"] = extra
    return entry


def partition_entries(entries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {"files": [], "services": [], "nested": [], "errors": [], "unknown": []}
    for entry in entries:
        kind = entry.get("kind")
        if kind == "file":
            out["files"].append(entry)
        elif kind == "service":
            out["services"].append(entry)
        elif kind == "nested":
            out["nested"].append(entry)
        elif kind == "error":
            out["errors"].append(entry)
        else:
            out["unknown"].append(entry)
    return out


def size_summary(files: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Byte preflight over typed file rows (skill: plan bytes from DataLink
    content_length, allow missing lengths)."""
    known = [f for f in files if f.get("content_length") is not None]
    total = sum(int(f["content_length"]) for f in known)
    return {
        "n_files": len(files),
        "n_size_known": len(known),
        "n_size_unknown": len(files) - len(known),
        "total_known_bytes": total,
        "total_known_gb": round(total / 1e9, 3),
    }


class DataLinkClient:
    """
    Client for ALMA DataLink protocol — enumerates the access alternatives
    within a Member OUS dataset.

    Usage:
        client = DataLinkClient()
        inventory = client.list_files("uid://A001/X1590/X30a8", pattern="*pbcor.fits")
        inventory["files"]  # typed file rows: filename, size_mb (None = unknown), access_url, semantics...
        inventory["state"]  # ok | empty_or_unauthorized | not_found | unavailable | partial_parse
    """

    # ALMA DataLink endpoints — mirror-first (QUASAR_ALMA_MIRROR, default NRAO),
    # the other regional copy only as a transport fallback.
    DATALINK_ENDPOINTS = [
        f"{ALMA_MIRROR}/datalink/sync",
        "https://almascience.eso.org/datalink/sync",
        "https://almascience.nrao.edu/datalink/sync",
    ]

    def __init__(self):
        """Initialize the DataLink client."""
        self.alma = None
        if ASTROQUERY_AVAILABLE:
            try:
                self.alma = Alma()
                self.alma.archive_url = ALMA_MIRROR
                print("[DataLink] Initialized with astroquery.alma")
            except Exception as e:
                print(f"[DataLink] astroquery.alma init failed: {e}")

    # ── Public API ──────────────────────────────────────────────────────────

    def list_files(
        self,
        mous_uid: str,
        pattern: Optional[str] = None,
        expand_tarfiles: bool = True,
    ) -> Dict[str, Any]:
        """
        Enumerate the DataLink rows of a MOUS UID, typed.

        Returns a dict with keys:
          - success: bool (False only for transport failure or an explicit
            not-found fault; an EMPTY table is success=True)
          - state: ok | empty_or_unauthorized | not_found | unavailable | partial_parse
          - mous_uid, total_files, files (kind == 'file', pattern-filtered),
            services, nested, errors (strings), partial_parse (bool),
            size_summary, category_counts
          - error: str (only when success is False)
        """
        mous_uid = self._normalize_uid(mous_uid)
        print(f"[DataLink] Listing files for {mous_uid} (pattern={pattern})")

        transport_errors: List[str] = []
        entries: Optional[List[Dict[str, Any]]] = None
        partial_parse = False
        fault: Optional[str] = None

        # Strategy 1: astroquery.alma.get_data_info() — most reliable
        if self.alma is not None:
            try:
                entries = self._list_via_astroquery(mous_uid, expand_tarfiles)
            except Exception as e:
                message = str(e)
                if _NOT_FOUND_RE.search(message):
                    fault = message
                else:
                    transport_errors.append(f"astroquery: {message}")
                print(f"[DataLink] astroquery approach failed: {e}")

        # Strategy 2: Direct DataLink VOTable HTTP request
        if entries is None and fault is None:
            try:
                entries, partial_parse, fault, http_errors = self._list_via_http(mous_uid)
                transport_errors.extend(http_errors)
            except Exception as e:
                transport_errors.append(f"http: {e}")
                print(f"[DataLink] HTTP approach failed: {e}")

        if fault is not None:
            return {
                "success": False,
                "state": STATE_NOT_FOUND,
                "mous_uid": mous_uid,
                "total_files": 0,
                "files": [], "services": [], "nested": [],
                "errors": [fault],
                "partial_parse": False,
                "error": f"DataLink reported an error for {mous_uid}: {fault}",
            }
        if entries is None:
            detail = "; ".join(transport_errors) or "no DataLink endpoint answered"
            return {
                "success": False,
                "state": STATE_UNAVAILABLE,
                "mous_uid": mous_uid,
                "total_files": 0,
                "files": [], "services": [], "nested": [],
                "errors": transport_errors,
                "partial_parse": False,
                "error": (
                    f"Could not retrieve the DataLink table for {mous_uid} (archive unavailable or "
                    f"timed out: {detail}). This is a transport failure, not evidence that the MOUS "
                    "is invalid or empty."
                ),
            }

        groups = partition_entries(entries)
        files = self._apply_pattern_filter(groups["files"], pattern)
        row_errors = [
            (e.get("error_message") or "DataLink error row") + (f" ({e['filename']})" if e.get("filename") else "")
            for e in groups["errors"]
        ]
        if not entries:
            state = STATE_EMPTY
        elif partial_parse:
            state = STATE_PARTIAL
        else:
            state = STATE_OK
        result: Dict[str, Any] = {
            "success": True,
            "state": state,
            "mous_uid": mous_uid,
            "total_files": len(files),
            "total_rows": len(entries),
            "files": files,
            "services": groups["services"],
            "nested": groups["nested"],
            "errors": row_errors + transport_errors,
            "partial_parse": partial_parse,
            "size_summary": size_summary(files),
            "category_counts": {
                "files": len(groups["files"]), "services": len(groups["services"]),
                "nested_datalink": len(groups["nested"]), "error_rows": len(groups["errors"]),
                "unknown": len(groups["unknown"]),
            },
        }
        if state == STATE_EMPTY:
            result["message"] = (
                "The DataLink table for this MOUS is valid but EMPTY: no links are visible under the "
                "current (anonymous) authorization. This is the normal answer for a proprietary MOUS; "
                "it does NOT mean the UID is invalid (an invalid UID returns an explicit error)."
            )
        if partial_parse:
            result["message"] = (
                "DataLink VOTable could only be read by a regex fallback: sizes/content types are "
                "unknown and non-URL rows (services, errors) may be missing."
            )
        return result

    def list_files_batch(
        self,
        mous_uids: List[str],
        pattern: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List files for multiple MOUS UIDs at once.

        Returns:
            Dict with:
              - success: bool (any MOUS answered)
              - results: Dict[mous_uid -> list of file dicts]
              - states: Dict[mous_uid -> state]
              - total_files: int (across all MOUSs)
        """
        all_results = {}
        states = {}
        total = 0
        errors = []

        for uid in mous_uids:
            result = self.list_files(uid, pattern=pattern)
            states[uid] = result.get("state")
            if result["success"]:
                all_results[uid] = result["files"]
                total += result["total_files"]
            else:
                errors.append(f"{uid}: {result.get('error', 'Unknown error')}")

        return {
            "success": len(all_results) > 0,
            "results": all_results,
            "states": states,
            "total_files": total,
            "mous_count": len(all_results),
            "errors": errors if errors else None,
        }

    def get_file_access_url(
        self, mous_uid: str, filename: str
    ) -> Optional[str]:
        """
        Get the direct access URL for a specific file within a MOUS dataset.

        Returns the URL string or None if not found.
        """
        result = self.list_files(mous_uid)
        if result["success"]:
            for f in result["files"]:
                if f["filename"] == filename or filename in f["filename"]:
                    return f.get("access_url")
        return None

    def download_file(
        self,
        access_url: str,
        output_dir: str = "./downloads",
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Download a single file from the ALMA archive.

        Args:
            access_url: Direct URL to the file
            output_dir: Directory to save the file
            filename: Optional filename override

        Returns:
            Dict with success, path, size_mb
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if filename is None:
            # Extract filename from URL
            filename = access_url.split("/")[-1].split("?")[0]
            if not filename:
                filename = "alma_download.fits"

        output_path = os.path.join(output_dir, filename)

        try:
            print(f"[DataLink] Downloading {filename}...")
            resp = requests.get(access_url, stream=True, timeout=300)
            resp.raise_for_status()

            total_size = 0
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    total_size += len(chunk)

            size_mb = total_size / (1024 * 1024)
            print(f"[DataLink] Downloaded {filename} ({size_mb:.1f} MB)")

            return {
                "success": True,
                "path": output_path,
                "filename": filename,
                "size_mb": round(size_mb, 2),
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Download failed: {e}",
                "url": access_url,
            }

    # ── Internal Methods ────────────────────────────────────────────────────

    def _list_via_astroquery(
        self, mous_uid: str, expand_tarfiles: bool = True
    ) -> List[Dict[str, Any]]:
        """Use astroquery.alma.get_data_info() to list rows (typed)."""
        print(f"[DataLink] Querying via astroquery for {mous_uid}")

        # get_data_info returns a Table with columns:
        #   access_url, content_length, content_type, semantics, description
        #   (+ service_def / error_message on DataLink 1.1 services)
        info_table = self.alma.get_data_info(mous_uid, expand_tarfiles=expand_tarfiles)

        if info_table is None or len(info_table) == 0:
            return []

        col_names = list(info_table.colnames) if hasattr(info_table, 'colnames') else []
        entries = [row_to_entry(row, col_names) for row in info_table]
        entries = [e for e in entries if e["kind"] != "unknown" or e.get("description")]
        print(f"[DataLink] Found {len(entries)} DataLink rows via astroquery")
        return entries

    def _list_via_http(self, mous_uid: str) -> Tuple[Optional[List[Dict[str, Any]]], bool, Optional[str], List[str]]:
        """
        Fallback: query the DataLink endpoint directly via HTTP and parse
        the VOTable response.

        Returns (entries | None, partial_parse, fault | None, transport_errors).
        """
        print(f"[DataLink] Querying via direct HTTP for {mous_uid}")
        transport_errors: List[str] = []
        seen = set()
        for endpoint in self.DATALINK_ENDPOINTS:
            if endpoint in seen:
                continue
            seen.add(endpoint)
            try:
                url = f"{endpoint}?ID={mous_uid}"
                resp = requests.get(url, timeout=60)
                if resp.status_code == 404:
                    return None, False, f"HTTP 404 from {endpoint} (NotFound)", transport_errors
                resp.raise_for_status()
                entries, partial, fault = self._parse_votable_response(resp.text)
                return entries, partial, fault, transport_errors
            except Exception as e:
                transport_errors.append(f"{endpoint}: {e}")
                print(f"[DataLink] {endpoint} failed: {e}")
                continue

        return None, False, None, transport_errors

    def _parse_votable_response(self, xml_text: str) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
        """Parse a DataLink VOTable XML response into typed entries.

        Returns (entries, partial_parse, fault). A service-level
        QUERY_STATUS=ERROR / NotFoundFault becomes ``fault``; a valid table
        with zero rows returns ([], False, None) — the empty state.
        """
        fault = self._votable_fault(xml_text)
        if fault:
            return [], False, fault

        entries: List[Dict[str, Any]] = []
        parsed_ok = False
        try:
            import io
            from astropy.io.votable import parse as parse_votable

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                votable = parse_votable(io.BytesIO(xml_text.encode("utf-8")))
            first = votable.get_first_table() if votable.resources else None
            if first is None:
                # A resource with INFO but no TABLE: valid, empty.
                parsed_ok = True
            else:
                table = first.to_table()
                vot_cols = list(table.colnames) if hasattr(table, 'colnames') else []
                entries = [row_to_entry(row, vot_cols) for row in table]
                parsed_ok = True
        except Exception as e:
            print(f"[DataLink] VOTable parsing failed: {e}")

        if parsed_ok:
            print(f"[DataLink] Parsed {len(entries)} DataLink rows from VOTable")
            return entries, False, None

        # Very basic regex fallback: every http(s) URL inside a <TD> cell.
        # Content type is NOT fabricated and sizes stay unknown (A-66).
        urls = re.findall(r"<TD[^>]*>\s*(https?://[^<\s]+)\s*</TD>", xml_text, flags=re.IGNORECASE)
        for url in urls:
            entries.append({
                "kind": "nested" if "/datalink/" in url.lower() else "file",
                "filename": url.split("/")[-1].split("?")[0],
                "size_mb": None,
                "content_length": None,
                "size_known": False,
                "access_url": url,
                "content_type": "",
                "description": "",
                "semantics": "",
            })
        print(f"[DataLink] Regex fallback extracted {len(entries)} URL rows")
        return entries, True, None

    @staticmethod
    def _votable_fault(xml_text: str) -> Optional[str]:
        match = _VOTABLE_ERROR_RE.search(xml_text or "")
        if match:
            message = (match.group(1) or match.group(2) or "").strip()
            return re.sub(r"\s+", " ", message) or "QUERY_STATUS=ERROR"
        if _NOT_FOUND_RE.search(xml_text or "") and "<TABLEDATA" not in (xml_text or ""):
            snippet = re.sub(r"<[^>]+>", " ", xml_text or "")
            snippet = re.sub(r"\s+", " ", snippet).strip()
            return snippet[:300] or "NotFoundFault"
        return None

    def _apply_pattern_filter(
        self, files: List[Dict[str, Any]], pattern: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Filter file list by glob pattern on filename."""
        if not pattern:
            return files

        filtered = [
            f for f in files
            if fnmatch.fnmatch(f["filename"], pattern)
            or fnmatch.fnmatch(f["filename"].lower(), pattern.lower())
        ]
        print(f"[DataLink] Pattern '{pattern}' filtered {len(files)} -> {len(filtered)} files")
        return filtered

    @staticmethod
    def _normalize_uid(uid: str) -> str:
        """
        Normalize a MOUS UID to the canonical ``uid://X/X/X`` form.

        Handles: 'uid://A001/X1590/X30a8', 'uid___A001_X1590_X30a8',
        'member.uid___A001_X1590_X30a8', 'A001/X1590/X30a8', trailing
        slashes, and project-prefixed filenames containing a UID (A-92).
        """
        text = str(uid or "").strip().strip("\"'").rstrip("/")
        match = _UID_CANON_RE.search(text)
        if match:
            return f"uid://{match.group(1)}/{match.group(2)}/{match.group(3)}"
        match = _UID_SANITIZED_RE.search(text)
        if match:
            return f"uid://{match.group(1)}/{match.group(2)}/{match.group(3)}"
        match = _UID_BARE_RE.match(text)
        if match:
            return f"uid://{match.group(1)}/{match.group(2)}/{match.group(3)}"
        return text


__all__ = [
    "DataLinkClient", "row_to_entry", "partition_entries", "size_summary",
    "STATE_OK", "STATE_EMPTY", "STATE_NOT_FOUND", "STATE_UNAVAILABLE", "STATE_PARTIAL",
]
