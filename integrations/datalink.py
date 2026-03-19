# integrations/datalink.py
"""
ALMA DataLink Client — File-Level Archive Access

Queries the ALMA DataLink service to enumerate individual data products
(FITS files, measurement sets, etc.) within a Member OUS UID.  This is
the layer that bridges observation-level metadata (from ALminer/TAP) to
the actual downloadable files a researcher needs.

CALLED BY: core/agent.py (tools: list_alma_files, inspect_fits_header)
CALLS:     astroquery.alma (DataLink), ALMA archive HTTPS (direct header reads)

Key capabilities:
  1. list_files(mous_uid, pattern) — enumerate files in a MOUS dataset
  2. list_files_batch(mous_uids, pattern) — bulk enumeration across multiple MOUSs
  3. get_file_access_url(mous_uid, filename) — direct download URL for one file
  4. download_file(access_url, output_dir) — download a single file
"""

from __future__ import annotations

import fnmatch
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

warnings.filterwarnings("ignore")

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


class DataLinkClient:
    """
    Client for ALMA DataLink protocol — enumerates individual files
    within a Member OUS dataset.

    Usage:
        client = DataLinkClient()
        files = client.list_files("uid://A001/X1590/X30a8", pattern="*pbcor.fits")
        # Returns list of dicts with filename, size, access_url, content_type
    """

    # ALMA DataLink endpoints (try ESO first, then NRAO)
    DATALINK_ENDPOINTS = [
        "https://almascience.eso.org/datalink/sync",
        "https://almascience.nrao.edu/datalink/sync",
    ]

    def __init__(self):
        """Initialize the DataLink client."""
        self.alma = None
        if ASTROQUERY_AVAILABLE:
            try:
                self.alma = Alma()
                # Use NRAO mirror for North American sources
                self.alma.archive_url = "https://almascience.nrao.edu"
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
        List all files available for a given MOUS UID.

        Args:
            mous_uid: ALMA Member OUS UID (e.g. "uid://A001/X1590/X30a8")
            pattern: Optional glob pattern to filter filenames (e.g. "*pbcor.fits")
            expand_tarfiles: If True, list individual files inside tar archives

        Returns:
            Dict with keys:
              - success: bool
              - mous_uid: str
              - total_files: int
              - files: List[dict] — each with filename, size_mb, access_url, content_type
              - error: str (only on failure)
        """
        mous_uid = self._normalize_uid(mous_uid)
        print(f"[DataLink] Listing files for {mous_uid} (pattern={pattern})")

        # Strategy 1: astroquery.alma.get_data_info() — most reliable
        if self.alma is not None:
            try:
                result = self._list_via_astroquery(mous_uid, expand_tarfiles)
                if result:
                    files = self._apply_pattern_filter(result, pattern)
                    return {
                        "success": True,
                        "mous_uid": mous_uid,
                        "total_files": len(files),
                        "files": files,
                    }
            except Exception as e:
                print(f"[DataLink] astroquery approach failed: {e}")

        # Strategy 2: Direct DataLink VOTable HTTP request
        try:
            result = self._list_via_http(mous_uid)
            if result:
                files = self._apply_pattern_filter(result, pattern)
                return {
                    "success": True,
                    "mous_uid": mous_uid,
                    "total_files": len(files),
                    "files": files,
                }
        except Exception as e:
            print(f"[DataLink] HTTP approach failed: {e}")

        return {
            "success": False,
            "mous_uid": mous_uid,
            "total_files": 0,
            "files": [],
            "error": "Could not retrieve file list. The MOUS UID may be invalid or the archive may be unavailable.",
        }

    def list_files_batch(
        self,
        mous_uids: List[str],
        pattern: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List files for multiple MOUS UIDs at once.

        Returns:
            Dict with:
              - success: bool
              - results: Dict[mous_uid -> list of file dicts]
              - total_files: int (across all MOUSs)
        """
        all_results = {}
        total = 0
        errors = []

        for uid in mous_uids:
            result = self.list_files(uid, pattern=pattern)
            if result["success"]:
                all_results[uid] = result["files"]
                total += result["total_files"]
            else:
                errors.append(f"{uid}: {result.get('error', 'Unknown error')}")

        return {
            "success": len(all_results) > 0,
            "results": all_results,
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
        """Use astroquery.alma.get_data_info() to list files."""
        print(f"[DataLink] Querying via astroquery for {mous_uid}")

        # get_data_info returns a Table with columns:
        #   access_url, content_length, content_type, semantics, description
        info_table = self.alma.get_data_info(mous_uid, expand_tarfiles=expand_tarfiles)

        if info_table is None or len(info_table) == 0:
            return []

        files = []
        for row in info_table:
            access_url = str(row.get("access_url", ""))
            content_length = row.get("content_length", 0)
            content_type = str(row.get("content_type", ""))
            description = str(row.get("description", ""))
            semantics = str(row.get("semantics", ""))

            # Extract filename from access_url
            filename = access_url.split("/")[-1].split("?")[0] if access_url else ""

            # Clean up: skip auxiliary/metadata-only entries
            if not filename or filename.startswith("."):
                continue

            size_mb = 0
            try:
                size_mb = round(float(content_length) / (1024 * 1024), 2)
            except (ValueError, TypeError):
                pass

            files.append({
                "filename": filename,
                "size_mb": size_mb,
                "access_url": access_url,
                "content_type": content_type,
                "description": description,
                "semantics": semantics,
            })

        print(f"[DataLink] Found {len(files)} files via astroquery")
        return files

    def _list_via_http(self, mous_uid: str) -> List[Dict[str, Any]]:
        """
        Fallback: query the DataLink endpoint directly via HTTP and parse
        the VOTable response.
        """
        print(f"[DataLink] Querying via direct HTTP for {mous_uid}")

        for endpoint in self.DATALINK_ENDPOINTS:
            try:
                url = f"{endpoint}?ID={mous_uid}"
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()

                # Parse VOTable XML response
                return self._parse_votable_response(resp.text)
            except Exception as e:
                print(f"[DataLink] {endpoint} failed: {e}")
                continue

        return []

    def _parse_votable_response(self, xml_text: str) -> List[Dict[str, Any]]:
        """Parse a DataLink VOTable XML response into a list of file dicts."""
        files = []

        if PYVO_AVAILABLE:
            try:
                import io
                from astropy.io.votable import parse as parse_votable

                votable = parse_votable(io.BytesIO(xml_text.encode("utf-8")))
                table = votable.get_first_table().to_table()

                for row in table:
                    access_url = str(row.get("access_url", ""))
                    filename = access_url.split("/")[-1].split("?")[0] if access_url else ""
                    content_length = row.get("content_length", 0)
                    content_type = str(row.get("content_type", ""))

                    if not filename or filename.startswith("."):
                        continue

                    size_mb = 0
                    try:
                        size_mb = round(float(content_length) / (1024 * 1024), 2)
                    except (ValueError, TypeError):
                        pass

                    files.append({
                        "filename": filename,
                        "size_mb": size_mb,
                        "access_url": access_url,
                        "content_type": content_type,
                    })
            except Exception as e:
                print(f"[DataLink] VOTable parsing failed: {e}")

        if not files:
            # Very basic regex fallback for access_url extraction
            urls = re.findall(r'https?://[^\s<>"]+\.fits[^\s<>"]*', xml_text)
            for url in urls:
                filename = url.split("/")[-1].split("?")[0]
                files.append({
                    "filename": filename,
                    "size_mb": 0,
                    "access_url": url,
                    "content_type": "application/fits",
                })

        print(f"[DataLink] Parsed {len(files)} files from VOTable")
        return files

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
        print(f"[DataLink] Pattern '{pattern}' filtered {len(files)} → {len(filtered)} files")
        return filtered

    @staticmethod
    def _normalize_uid(uid: str) -> str:
        """
        Normalize a MOUS UID to the standard format.
        Handles: 'uid://A001/X1590/X30a8', 'uid___A001_X1590_X30a8',
                 'A001/X1590/X30a8', etc.
        """
        uid = uid.strip()

        # If it's already in uid:// format, return as-is
        if uid.startswith("uid://"):
            return uid

        # Convert underscore format to uid:// format
        if uid.startswith("uid___"):
            parts = uid.replace("uid___", "").split("_")
            if len(parts) == 3:
                return f"uid://{parts[0]}/{parts[1]}/{parts[2]}"

        # If it's just the three parts separated by /
        if "/" in uid and not uid.startswith("uid"):
            return f"uid://{uid}"

        return uid