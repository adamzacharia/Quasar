"""CDS X-Match sync API client: crossmatch a user object list (or a stored
Data Lab result) against VizieR catalogs / SIMBAD.

Live-verified 2026-07-12 against https://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync
(the service's EXPLAIN response documents the contract used here):
  request=xmatch, cat1=FILE upload (colRA1/colDec1 name the coordinate columns),
  cat2=simbad|vizier:<id>, distMaxArcsec<=180, selection=best|all,
  RESPONSEFORMAT=csv.
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd
import requests

XMATCH_SYNC_URL = os.getenv(
    "CDS_XMATCH_URL", "https://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync"
)
MAX_UPLOAD_ROWS = int(os.getenv("CDS_XMATCH_MAX_UPLOAD_ROWS", "10000"))
MAX_RADIUS_ARCSEC = 180.0  # hard service limit (distMaxArcsec max=180.0)

# Friendly aliases -> CDS cat2 identifiers. VizieR table ids are stable
# catalog handles; 'simbad' is the special NAME the service accepts directly.
CATALOG_ALIASES: Dict[str, str] = {
    "simbad": "simbad",
    "2mass": "vizier:II/246/out",
    "twomass": "vizier:II/246/out",
    "allwise": "vizier:II/328/allwise",
    "unwise": "vizier:II/363/unwise",
    "gaia_dr3": "vizier:I/355/gaiadr3",
    "gaia": "vizier:I/355/gaiadr3",
    "sdss_dr12": "vizier:V/147/sdss12",
    "sdss": "vizier:V/147/sdss12",
    "panstarrs_dr1": "vizier:II/349/ps1",
    "ps1": "vizier:II/349/ps1",
    "tycho2": "vizier:I/259/tyc2",
    "nvss": "vizier:VIII/65/nvss",
    "first": "vizier:VIII/92/first14",
    "galex": "vizier:II/335/galex_ais",
}


class CdsXmatchError(RuntimeError):
    """Transport/contract failures against the CDS X-Match service."""


def resolve_catalog(catalog: str) -> str:
    """Map an alias or explicit id to a cat2 value the service accepts."""
    text = str(catalog or "").strip()
    if not text:
        raise ValueError("xmatch catalog is required (e.g. '2mass', 'gaia_dr3', 'vizier:II/246/out').")
    lowered = text.lower()
    if lowered in CATALOG_ALIASES:
        return CATALOG_ALIASES[lowered]
    if lowered == "simbad" or lowered.startswith("vizier:"):
        return text if lowered.startswith("vizier:") else "simbad"
    # Bare VizieR table id like 'II/246/out'
    if "/" in text:
        return f"vizier:{text}"
    raise ValueError(
        f"Unknown xmatch catalog {catalog!r}. Use an alias ({', '.join(sorted(CATALOG_ALIASES))}) "
        "or an explicit VizieR id like 'vizier:II/246/out'."
    )


def xmatch_dataframe(
    frame: pd.DataFrame,
    *,
    catalog: str,
    ra_column: str = "ra",
    dec_column: str = "dec",
    radius_arcsec: float = 5.0,
    selection: str = "best",
    timeout: float = 120.0,
    url: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload ``frame`` and crossmatch it against ``catalog``.

    Returns {"dataframe": matches, "matched_rows", "uploaded_rows", "cat2",
    "radius_arcsec", "selection"}. Every uploaded column comes back with the
    match columns appended (angDist in arcsec first, per the service contract).
    """

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("xmatch needs a non-empty table of objects.")
    if ra_column not in frame.columns or dec_column not in frame.columns:
        raise ValueError(
            f"xmatch table must carry the coordinate columns {ra_column!r}/{dec_column!r}; "
            f"got: {list(map(str, frame.columns))[:20]}"
        )
    radius = float(radius_arcsec)
    if not (0.0 < radius <= MAX_RADIUS_ARCSEC):
        raise ValueError(f"radius_arcsec must be in (0, {MAX_RADIUS_ARCSEC:g}] (service limit).")
    mode = str(selection or "best").strip().lower()
    if mode not in {"best", "all"}:
        raise ValueError("selection must be 'best' or 'all'.")
    cat2 = resolve_catalog(catalog)

    upload = frame
    truncated = False
    if len(upload) > MAX_UPLOAD_ROWS:
        upload = upload.head(MAX_UPLOAD_ROWS)
        truncated = True
    csv_payload = upload.to_csv(index=False)

    data = {
        "request": "xmatch",
        "distMaxArcsec": f"{radius:g}",
        "selection": mode,
        "responseFormat": "csv",
        "cat2": cat2,
        "colRA1": ra_column,
        "colDec1": dec_column,
    }
    files = {"cat1": ("upload.csv", csv_payload.encode("utf-8"), "text/csv")}
    try:
        resp = requests.post(url or XMATCH_SYNC_URL, data=data, files=files, timeout=float(timeout))
    except requests.RequestException as exc:
        raise CdsXmatchError(f"CDS X-Match request failed: {exc}") from exc
    body = resp.text or ""
    if resp.status_code != 200:
        # Errors come back as a VOTable with an explanatory INFO block.
        raise CdsXmatchError(
            f"CDS X-Match returned HTTP {resp.status_code}: {_error_snippet(body)}"
        )
    # Uploaded string columns must come back as strings: default dtype inference
    # would turn an id like "00123" into integer 123, silently losing the
    # leading zeroes. The response echoes uploaded columns by name. (guard CX-11)
    string_columns = {
        str(col): str
        for col in upload.columns
        if upload[col].dtype == object
    }
    try:
        matches = pd.read_csv(io.StringIO(body), dtype=string_columns or None)
    except Exception as exc:  # noqa: BLE001 - surface a clear transport error
        raise CdsXmatchError(
            f"Could not parse CDS X-Match CSV response: {exc}; body starts: {body[:200]!r}"
        ) from exc

    return {
        "dataframe": matches,
        "matched_rows": int(len(matches)),
        "uploaded_rows": int(len(upload)),
        "upload_truncated": truncated,
        "cat2": cat2,
        "radius_arcsec": radius,
        "selection": mode,
        "endpoint": url or XMATCH_SYNC_URL,
    }


def objects_to_dataframe(objects: List[Mapping[str, Any]]) -> pd.DataFrame:
    """Normalize a user-supplied object list to an upload table (adds a row id)."""
    if not objects:
        raise ValueError("objects list is empty.")
    frame = pd.DataFrame([dict(obj) for obj in objects])
    if "id" not in frame.columns:
        frame.insert(0, "id", [f"obj{i + 1}" for i in range(len(frame))])
    return frame


def _error_snippet(body: str) -> str:
    """Pull the human-readable message out of the service's VOTable error."""
    import re

    match = re.search(r"<INFO[^>]*>(.*?)</INFO>", body, re.DOTALL)
    text = (match.group(1) if match else body).strip()
    return " ".join(text.split())[:400]


__all__ = [
    "CATALOG_ALIASES",
    "CdsXmatchError",
    "MAX_RADIUS_ARCSEC",
    "objects_to_dataframe",
    "resolve_catalog",
    "xmatch_dataframe",
]
