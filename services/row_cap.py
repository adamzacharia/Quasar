"""One way to say "this result hit its row cap" across the archive search tools.

MANNA evals 2026-10-01 (MQ17, MQ22): search_eso_archive returned its 500-row
cap as ``total_results: 500`` and gpt-oss answered "500 observations" (the
archive holds 4,386 by pointing). A capped row count is never a count. When a
result fills its cap, the tool result carries:

    truncated: true, returned: N, row_cap: CAP,
    total: <exact COUNT(*) or "unknown">, count_is_lower_bound: bool,
    row_cap_note: what to tell the user.

Results below the cap are left byte-identical (no keys added).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

_TOP_RE = re.compile(r"^\s*SELECT\s+(?:(?:DISTINCT|ALL)\s+)?TOP\s+(\d+)\b", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)\s*(?:/\*.*?\*/\s*)?$", re.IGNORECASE | re.DOTALL)

COUNT_RULE = (
    "A row count that equals the tool's row cap is a lower bound, never the number of observations or "
    "sources: answer count questions with a COUNT(*) query, or say 'at least N'."
)


def adql_row_limit(adql: Any) -> Optional[int]:
    """The top-level TOP n (or trailing LIMIT n) a query asked for, else None."""
    try:
        from services.datalab_sql_policy import _scrub_sql

        clean = _scrub_sql(str(adql or ""))
    except Exception:  # noqa: BLE001
        clean = str(adql or "")
    m = _TOP_RE.match(clean) or _LIMIT_RE.search(clean.strip())
    return int(m.group(1)) if m else None


def row_cap_fields(returned: Any, cap: Any, total: Any = None, *, unit: str = "rows") -> Dict[str, Any]:
    """Keys to merge into a tool result that filled its cap; {} when it did not."""
    try:
        n = int(returned)
        c = int(cap)
    except (TypeError, ValueError):
        return {}
    if c <= 0 or n < c:
        return {}
    exact = isinstance(total, int) and not isinstance(total, bool) and total >= n
    if exact and total == n:
        return {}  # exactly the cap exists: nothing was cut
    if exact:
        note = (f"Row cap reached: {n} {unit} returned out of {total} in the archive (exact COUNT(*)). "
                f"Report {total} as the total and say only the first {n} are listed.")
    else:
        note = (f"Row cap reached: {n} {unit} returned and the true total is unknown (at least {n}). "
                f"Do not report {n} as the number of observations or sources: run a COUNT(*) query, "
                f"or say 'at least {n}'.")
    return {
        "truncated": True,
        "returned": n,
        "row_cap": c,
        "total": total if exact else "unknown",
        "count_is_lower_bound": not exact,
        "row_cap_note": note,
    }


def with_row_cap(out: Dict[str, Any], returned: Any, cap: Any, total: Any = None, *, unit: str = "rows") -> Dict[str, Any]:
    """``out`` plus the row-cap keys (and the note folded into ``note``) when capped."""
    fields = row_cap_fields(returned, cap, total, unit=unit)
    if not fields or not isinstance(out, dict):
        return out
    merged = dict(out)
    merged.update(fields)
    if merged.get("note"):
        merged["note"] = f"{fields['row_cap_note']} {merged['note']}"
    return merged


__all__ = ["COUNT_RULE", "adql_row_limit", "row_cap_fields", "with_row_cap"]
