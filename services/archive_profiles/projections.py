"""Deterministic, size-capped projections of ArchiveProfiles for browse_schema.

Contract (duel task-81b3b73-2145, DX-07/DX-17):
* archive projection <= 6 KiB UTF-8 JSON; table slice <= 20 KiB;
* truncation only at projection boundaries — drop whole trailing list
  entries, never cut strings — and always report counts + ``truncated``;
* the archive projection shows table SUMMARIES (keys) only, <=3
  prompt-ranked pitfalls, exactly min(3, authored) compact golden
  invocations; the table slice shows the full curated TableProfile plus
  scoped pitfalls/examples/surfaces.

Data Lab dynamic tier: curated tables merge the authored profile with
``datalab_registry.describe_table``; expansion/live-TAP-schema tables are
served from the live registry entry, marked ``schema_source="live_tap_schema"``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from services.archive_profiles.schema import (
    ArchiveProfile,
    Pitfall,
    ProfileRef,
    SqlRequestEvidence,
)

ARCHIVE_PROJECTION_MAX_BYTES = 6 * 1024
TABLE_PROJECTION_MAX_BYTES = 20 * 1024

_SCOPE_NOTE_HINT = "Canonical/common fields and parameters; not exhaustive."


def _json_bytes(payload: Dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))


def _surface_summary(profile: ArchiveProfile) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for surface in profile.query_surfaces:
        row: Dict[str, Any] = {
            "id": surface.id,
            "tool": surface.tool,
            "request_kind": surface.request_kind,
            "purpose": surface.purpose,
        }
        params = [
            {k: v for k, v in (("name", p.name), ("unit", p.unit),
                                ("allowed_values", list(p.allowed_values) if p.allowed_values else None),
                                ("description", p.description)) if v is not None}
            for p in surface.parameters
        ]
        if params:
            row["parameters"] = params
        rows.append(row)
    return rows


def _ranked_pitfalls(pitfalls: tuple, limit: int) -> List[Dict[str, Any]]:
    ranked = sorted(
        (p for p in pitfalls if p.prompt_rank is not None), key=lambda p: p.prompt_rank
    )
    unranked = [p for p in pitfalls if p.prompt_rank is None]
    rows = []
    for p in (ranked + unranked)[:limit]:
        row = {"id": p.id, "summary": p.summary}
        rows.append(row)
    return rows


def _compact_goldens(profile: ArchiveProfile, limit: int = 3) -> List[Dict[str, Any]]:
    rows = []
    for g in profile.golden_examples[:limit]:
        rows.append(
            {
                "intent": g.intent,
                "tool": g.invocation.tool,
                "arguments": g.invocation.arguments,
                **({"note": g.note} if g.note else {}),
            }
        )
    return rows


def _pitfall_applies_to_table(pitfall: Pitfall, archive: str, table_key: str) -> bool:
    for ref in pitfall.applies_to:
        if ref.kind == "archive" and ref.ref == archive:
            return True
        if ref.kind == "table" and ref.ref == table_key:
            return True
        if ref.kind == "column" and ref.ref.startswith(f"{table_key}:"):
            return True
    return False


def _shrink_to_cap(payload: Dict[str, Any], cap: int, droppable_keys: List[str]) -> Dict[str, Any]:
    """Drop whole trailing entries from the named list fields (in order) until
    the payload fits the byte cap. Never cuts strings. If the payload STILL
    exceeds the cap after every droppable entry is gone, the mandatory fields
    themselves are oversized — that is an authoring error and must fail
    loudly (duel contract; guard CX-02), not ship an oversize payload."""
    payload["truncated"] = False
    if _json_bytes(payload) <= cap:
        return payload
    for key in droppable_keys:
        while isinstance(payload.get(key), list) and payload[key] and _json_bytes(payload) > cap:
            payload[key].pop()
            payload["truncated"] = True
        if _json_bytes(payload) <= cap:
            return payload
    raise ValueError(
        f"projection for {payload.get('archive')}:{payload.get('table', '<archive>')} is "
        f"{_json_bytes(payload)} bytes even after dropping {droppable_keys} — exceeds the "
        f"{cap}-byte cap; shorten the authored descriptions/pitfalls"
    )


def archive_projection(profile: ArchiveProfile) -> Dict[str, Any]:
    """The compact no-table browse_schema payload."""
    table_keys = sorted((profile.tables or {}).keys())
    payload: Dict[str, Any] = {
        "kind": "archive",
        "archive": profile.archive,
        "description": profile.description,
        "scope_note": profile.scope_note,
        "query_surfaces": _surface_summary(profile),
        "endpoints": [
            {"id": e.id, "protocol": e.protocol, "url": str(e.url)} for e in profile.endpoints
        ],
        "pitfalls": _ranked_pitfalls(profile.pitfalls, limit=3),
        "golden_examples": _compact_goldens(profile, limit=3),
        "table_count": len(table_keys),
        "returned_table_count": len(table_keys),
    }
    if profile.tables is not None:
        payload["table_keys"] = table_keys
        payload["hint"] = (
            f"Call browse_schema('{profile.archive}', '<table_key>') for full columns/units."
        )
    else:
        payload["hint"] = (
            "This archive has no static tables — query through the surfaces above."
        )
    shrunk = _shrink_to_cap(
        payload, ARCHIVE_PROJECTION_MAX_BYTES,
        droppable_keys=["golden_examples", "table_keys", "pitfalls", "query_surfaces"],
    )
    if shrunk["truncated"] and "table_keys" in shrunk:
        shrunk["returned_table_count"] = len(shrunk["table_keys"])
    return shrunk


def table_projection(profile: ArchiveProfile, table_key: str) -> Optional[Dict[str, Any]]:
    """The focused one-table payload, or None if the key is not curated
    (the caller decides how to fall back — e.g. Data Lab's live tier)."""
    tables = profile.tables or {}
    table = tables.get(table_key)
    if table is None:
        return None
    dumped = table.model_dump(mode="json", exclude_none=True)
    payload: Dict[str, Any] = {
        "kind": "table",
        "archive": profile.archive,
        "table": table_key,
        "schema_source": "curated_profile",
        "scope_note": profile.scope_note,
        **dumped,
        "pitfalls": [
            {"id": p.id, "summary": p.summary, **({"detail": p.detail} if p.detail else {})}
            for p in profile.pitfalls
            if _pitfall_applies_to_table(p, profile.archive, table_key)
        ],
        "golden_examples": [
            {
                "intent": g.intent,
                "tool": g.invocation.tool,
                "arguments": g.invocation.arguments,
                **({"note": g.note} if g.note else {}),
            }
            for g in profile.golden_examples
            if _golden_mentions_table(g, table_key)
        ][:3],
        "citations": [
            c.model_dump(mode="json", exclude_none=True)
            for c in profile.citations
            if c.id in table.citation_ids
        ],
    }
    return _shrink_to_cap(
        payload, TABLE_PROJECTION_MAX_BYTES,
        droppable_keys=["golden_examples", "pitfalls"],
    )


def live_table_projection(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Cap a LIVE-tier Data Lab table payload (columns read from the live TAP
    schema, potentially hundreds of entries) to the same slice byte budget as
    curated tables, dropping trailing columns with an honest count + flag
    (guard CX-03)."""
    cols = payload.get("columns")
    payload["column_count"] = len(cols) if isinstance(cols, list) else 0
    # Placeholder with the maximum digit width; the true value written after
    # shrinking can only be <= this many digits, so bytes only go down.
    payload["returned_column_count"] = payload["column_count"]
    shrunk = _shrink_to_cap(payload, TABLE_PROJECTION_MAX_BYTES, droppable_keys=["columns"])
    if shrunk["truncated"]:
        # The note itself costs bytes — append it, then re-shrink so the
        # final payload still honors the cap.
        shrunk["hint"] = (
            str(shrunk.get("hint") or "")
            + " Column list truncated to fit the size cap — datalab_describe_table "
              "returns the full list."
        ).strip()
        shrunk = _shrink_to_cap(shrunk, TABLE_PROJECTION_MAX_BYTES, droppable_keys=["columns"])
        shrunk["truncated"] = True
    shrunk["returned_column_count"] = (
        len(shrunk["columns"]) if isinstance(shrunk.get("columns"), list) else 0
    )
    return shrunk


def _golden_mentions_table(golden, table_key: str) -> bool:
    args = golden.invocation.arguments
    catalog_table = None
    if isinstance(args, dict) and args.get("catalog") and args.get("table"):
        catalog_table = f"{args['catalog']}.{args['table']}"
    if catalog_table == table_key:
        return True
    if isinstance(golden.request, SqlRequestEvidence):
        sql = args.get(golden.request.argument, "")
        return isinstance(sql, str) and table_key in sql
    return any(table_key in str(v) for v in args.values() if isinstance(v, str))


__all__ = [
    "ARCHIVE_PROJECTION_MAX_BYTES",
    "TABLE_PROJECTION_MAX_BYTES",
    "archive_projection",
    "live_table_projection",
    "table_projection",
]
