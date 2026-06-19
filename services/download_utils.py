# services/download_utils.py
"""
Download utility with streaming progress reporting.

Used by agent tools that download large files (FITS, archive data)
to emit progress events via the SSE pipeline.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Optional

import requests


def download_with_progress(
    url: str,
    dest_path: str,
    *,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    chunk_size: int = 64 * 1024,  # 64 KB
    timeout: int = 300,
) -> str:
    """
    Stream-download *url* to *dest_path*, emitting progress events.

    Args:
        url:               Remote file URL.
        dest_path:         Local destination path.
        progress_callback: Called with a dict:
            {
                "filename":         str,
                "downloaded_bytes": int,
                "total_bytes":      int | None,  (None if Content-Length missing)
                "speed_kbps":       float,
                "percent":          float | None,
                "eta_seconds":      float | None,
            }
        chunk_size:        Bytes per read chunk.
        timeout:           Request timeout in seconds.

    Returns:
        Absolute path of the downloaded file.
    """
    filename = os.path.basename(dest_path) or url.rsplit("/", 1)[-1] or "download"

    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()

    total = int(resp.headers.get("Content-Length", 0)) or None
    downloaded = 0
    start_time = time.time()
    last_report_time = 0.0

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                # Report at most every 0.5 seconds to avoid flooding SSE
                if progress_callback and (now - last_report_time >= 0.5):
                    elapsed = max(now - start_time, 0.001)
                    speed = (downloaded / 1024) / elapsed  # KB/s
                    eta_seconds = (
                        max(0.0, (total - downloaded) / max(1.0, speed * 1024))
                        if total
                        else None
                    )

                    progress_callback({
                        "filename": filename,
                        "downloaded_bytes": downloaded,
                        "total_bytes": total,
                        "speed_kbps": round(speed, 1),
                        "percent": round((downloaded / total) * 100, 1) if total else None,
                        "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
                    })
                    last_report_time = now

    # Final 100% event
    if progress_callback:
        elapsed = max(time.time() - start_time, 0.001)
        progress_callback({
            "filename": filename,
            "downloaded_bytes": downloaded,
            "total_bytes": total or downloaded,
            "speed_kbps": round((downloaded / 1024) / elapsed, 1),
            "percent": 100.0,
            "eta_seconds": 0.0,
        })

    return os.path.abspath(dest_path)
