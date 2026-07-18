"""Full-result export (Feature 2: "cite the CSV, not the plot").

The data card the UI renders is a PREVIEW — capped at 10k rows, display columns
only, floats clipped for legibility. Downloading that preview and calling it the
dataset is the silent-partial failure this router exists to prevent. Everything
here streams the FULL frame out of the result store: every row, every column,
unformatted. If the frame can no longer be resolved we fail loud (404/410) so
the caller re-runs the query — we never substitute a partial for a whole.

Result lifetime is DATALAB_RESULT_TTL_SECONDS (1h) on a process-local memory
tier backed by a DiskCache, so an id can outlive a worker but not a redeploy.

Ownership: every card-served id is stamped with its requesting user at
serialization time (data_card.py stamps reused capability-minted ids and only
mints owned ones). Any id still lacking owner_id is therefore legacy or
agent-internal — this route refuses those outright rather than serve them to
whichever authenticated user guesses the id. (f2-CX-01)
"""

import csv
import io
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterator

import pandas as pd

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.deps import get_current_user

router = APIRouter()

# Rows per flush. Small enough that a client sees bytes promptly (and Render's
# proxy sees keepalive traffic), large enough that per-batch overhead is noise.
_EXPORT_BATCH_ROWS = 2000

_EXPIRED_DETAIL = (
    "This result is no longer available — results are kept for a limited time "
    "and this one has expired. Re-run the query to export it."
)


def _export_max_rows() -> int:
    """Safety ceiling on an export. 0/absent = unlimited (stream it all)."""
    try:
        return int(os.getenv("QUASAR_RESULT_EXPORT_MAX_ROWS", "0") or 0)
    except ValueError:
        return 0


def _resolve_or_raise(result_id: str, current_user: dict):
    """Resolve an id to (frame, meta) or raise the correct loud failure."""
    from services.datalab_result_store import default_result_store

    # My-table (dlt_) keys never carry owner_id metadata, and legacy shared
    # tables use guessable name slugs — serving them here would let any
    # authenticated user export another user's saved table by name. My-tables
    # already have their own owner-scoped route (/api/datalab/mytables/{name});
    # this route serves dlr_ ids only. (UIAPI-05 / dl-export-owner-gap)
    if str(result_id or "").startswith("dlt_"):
        raise HTTPException(status_code=404, detail=f"Unknown result id: {result_id}")

    frame, meta, status = default_result_store().lookup(result_id)
    if status == "unknown":
        raise HTTPException(status_code=404, detail=f"Unknown result id: {result_id}")
    if status in {"expired", "gone"}:
        # 410 for both: "expired" is definitive, "gone" means it aged out of
        # both tiers or belongs to another worker's memory. We cannot tell those
        # apart without a durable manifest, and the user's next step is the same
        # either way — re-run the query.
        raise HTTPException(status_code=410, detail=_EXPIRED_DETAIL)

    owner = str(meta.get("owner_id") or "").strip()
    if not owner or owner != str(current_user.get("sub") or ""):
        # Ownerless metadata is refused too, not just a mismatched owner: with
        # serialization-time stamping in place, an id with no owner_id is
        # legacy/agent-internal and must not be exportable cross-user. Same
        # detail as unknown — don't confirm the id exists. (f2-CX-01)
        raise HTTPException(status_code=404, detail=f"Unknown result id: {result_id}")
    return frame, meta


def _cell(value: Any) -> Any:
    """Render None/NaN/NaT/pd.NA as empty; pass every other value through untouched."""
    if value is None:
        return ""
    try:
        # pd.isna covers NaN and NaT plus the nullable-dtype pd.NA scalar,
        # which the old self-inequality check missed (pd.NA != pd.NA raises,
        # and csv.writer then rendered the literal "<NA>"). Scalars yield a
        # bool; array-likes yield an array and keep pass-through. (f2-CX-07)
        if pd.isna(value) is True:
            return ""
    except Exception:  # noqa: BLE001 - exotic values: keep as-is
        pass
    return value


def _iter_csv(frame, max_rows: int) -> Iterator[str]:
    """Yield the frame as CSV text in row batches.

    Uses the csv module directly rather than df.to_csv so nothing re-formats or
    clips a value on the way out: the export must be byte-for-byte the data the
    query returned, not the display rendering of it.
    """
    columns = [str(c) for c in frame.columns]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    # Flush the header on its own: a zero-row result must still export its
    # column schema, not a 0-byte file (UIAPI-13 / dl-export-empty-frame-no-header).
    yield buffer.getvalue()
    buffer.seek(0)
    buffer.truncate(0)

    total = len(frame) if max_rows <= 0 else min(len(frame), max_rows)
    for start in range(0, total, _EXPORT_BATCH_ROWS):
        stop = min(start + _EXPORT_BATCH_ROWS, total)
        for row in frame.iloc[start:stop].itertuples(index=False, name=None):
            writer.writerow([_cell(v) for v in row])
        yield buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)


def _safe_filename(meta: Dict[str, Any], result_id: str) -> str:
    raw = str(meta.get("source") or "").strip() or result_id
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw)
    return f"{cleaned[:60] or result_id}.csv"


@router.get("/api/results/{result_id}/export.csv")
async def export_result_csv(
    result_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Stream the COMPLETE result as CSV — all rows, all columns, unformatted."""
    frame, meta = _resolve_or_raise(result_id, current_user)

    max_rows = _export_max_rows()
    total = int(len(frame))
    exported = total if max_rows <= 0 else min(total, max_rows)

    headers = {
        "Content-Disposition": f'attachment; filename="{_safe_filename(meta, result_id)}"',
        "X-Quasar-Rowcount": str(exported),
        "X-Quasar-Total-Rows": str(total),
        "Cache-Control": "no-store",
    }
    if exported < total:
        # A ceiling-clipped export is still partial data — say so in-band rather
        # than let it pass as complete.
        headers["X-Quasar-Truncated"] = "1"

    return StreamingResponse(
        _iter_csv(frame, max_rows),
        media_type="text/csv; charset=utf-8",
        headers=headers,
    )


@router.get("/api/results/{result_id}/meta")
async def get_result_meta(
    result_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Report what an export would contain: rowcount, columns, provenance, expiry."""
    from services.datalab_result_store import default_result_store

    frame, meta = _resolve_or_raise(result_id, current_user)
    max_rows = _export_max_rows()
    total = int(len(frame))
    # Expiry disclosure (f2-CX-10): the store's lookup() carries the payload's
    # creation time under the reserved "_created_at" key; combined with the
    # store TTL that yields the expiresAt the plan's meta contract asks for.
    ttl = int(default_result_store().ttl_seconds or 0)
    created = float(meta.get("_created_at") or 0.0)

    def _iso(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    return {
        "resultId": result_id,
        "totalRows": total,
        "columns": [str(c) for c in frame.columns],
        "exportMaxRows": max_rows or None,
        "exportRows": total if max_rows <= 0 else min(total, max_rows),
        "source": meta.get("source") or None,
        "toolName": meta.get("tool_name") or None,
        "request": meta.get("request") or None,
        "createdAt": _iso(created) if created else None,
        "ttlSeconds": ttl if ttl > 0 else None,
        "expiresAt": _iso(created + ttl) if (created and ttl > 0) else None,
    }
