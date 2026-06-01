"""Quasar JWST MCP server.

This server exposes JWST-focused MAST archive tools from Quasar as MCP tools.
Run from the repository root with:

    python JWST_MCP/server.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from mcp.server.fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrations.mast_client import MASTClient  # noqa: E402

mcp = FastMCP("quasar-jwst")
_mast_client: Optional[MASTClient] = None
_last_jwst_observations: Optional[pd.DataFrame] = None


def _client() -> MASTClient:
    global _mast_client
    if _mast_client is None:
        _mast_client = MASTClient()
    return _mast_client


def _table_payload(df: pd.DataFrame, max_rows: int = 100) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"success": True, "row_count": 0, "columns": [], "rows": []}
    safe = df.head(max_rows).where(pd.notnull(df), None)
    return {
        "success": True,
        "row_count": int(len(df)),
        "returned_rows": int(len(safe)),
        "columns": [str(c) for c in safe.columns],
        "rows": safe.to_dict("records"),
    }


@mcp.tool()
def jwst_search_by_target(
    target_name: str,
    instrument: Optional[str] = None,
    radius: str = "30s",
    max_rows: int = 100,
) -> Dict[str, Any]:
    """Search MAST for JWST observations of a named target.

    Use for prompts such as "find JWST observations of NGC 1333" or
    "show JWST NIRCam data for M87".
    """
    global _last_jwst_observations
    df = _client().search_by_target(
        target=target_name,
        mission="JWST",
        instrument=instrument,
        radius=radius,
        max_results=max_rows,
    )
    _last_jwst_observations = df
    payload = _table_payload(df, max_rows=max_rows)
    payload["query"] = {"target_name": target_name, "instrument": instrument, "radius": radius}
    return payload


@mcp.tool()
def jwst_search_by_position(
    ra: float,
    dec: float,
    radius_arcmin: float = 1.0,
    instrument: Optional[str] = None,
    max_rows: int = 100,
) -> Dict[str, Any]:
    """Search JWST observations around ICRS RA/Dec in degrees."""
    global _last_jwst_observations
    df = _client().search_by_position(
        ra=ra,
        dec=dec,
        radius_arcmin=radius_arcmin,
        mission="JWST",
        instrument=instrument,
        max_results=max_rows,
    )
    _last_jwst_observations = df
    payload = _table_payload(df, max_rows=max_rows)
    payload["query"] = {"ra": ra, "dec": dec, "radius_arcmin": radius_arcmin, "instrument": instrument}
    return payload


@mcp.tool()
def jwst_search_by_criteria(
    instrument: Optional[str] = None,
    proposal_id: Optional[str] = None,
    filters: Optional[str] = None,
    target_name: Optional[str] = None,
    dataproduct_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_rows: int = 100,
) -> Dict[str, Any]:
    """Advanced JWST MAST search by instrument, program, filter, target, or date."""
    global _last_jwst_observations
    date_range: Optional[Tuple[str, str]] = (start_date, end_date) if start_date and end_date else None
    df = _client().search_by_criteria(
        mission="JWST",
        instrument=instrument,
        proposal_id=proposal_id,
        filters=filters,
        target_name=target_name,
        dataproduct_type=dataproduct_type,
        date_range=date_range,
        max_results=max_rows,
    )
    _last_jwst_observations = df
    payload = _table_payload(df, max_rows=max_rows)
    payload["query"] = {
        "instrument": instrument,
        "proposal_id": proposal_id,
        "filters": filters,
        "target_name": target_name,
        "dataproduct_type": dataproduct_type,
        "start_date": start_date,
        "end_date": end_date,
    }
    return payload


@mcp.tool()
def jwst_get_products(
    product_type: Optional[str] = "SCIENCE",
    extension: Optional[str] = "fits",
    max_rows: int = 100,
) -> Dict[str, Any]:
    """List file-level JWST products for the last JWST search.

    Call `jwst_search_by_target`, `jwst_search_by_position`, or
    `jwst_search_by_criteria` first.
    """
    if _last_jwst_observations is None or _last_jwst_observations.empty:
        return {"success": False, "error": "No JWST observations cached. Run a JWST search first."}
    df = _client().get_product_list(
        _last_jwst_observations,
        productType=product_type,
        extension=extension,
    )
    payload = _table_payload(df, max_rows=max_rows)
    payload["query"] = {"product_type": product_type, "extension": extension}
    return payload


@mcp.tool()
def jwst_observation_summary(rows: list[dict[str, Any]]) -> Dict[str, Any]:
    """Summarize JWST observation rows returned by another MCP call."""
    df = pd.DataFrame(rows or [])
    if df.empty:
        return {"success": True, "row_count": 0, "summary": {}}
    summary: Dict[str, Any] = {"row_count": int(len(df))}
    for col in ["instrument_name", "filters", "target_name", "proposal_id", "dataproduct_type"]:
        if col in df.columns:
            summary[col] = df[col].astype(str).value_counts().head(10).to_dict()
    return {"success": True, "summary": summary}


if __name__ == "__main__":
    mcp.run()
