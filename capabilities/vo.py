"""
capabilities/vo.py — the Virtual Observatory registry-discovery family as
transport-pure capabilities (P1 family migration #5).

Logic relocated VERBATIM from ``core/agent.py`` (the ``_vo_find_services`` /
``_vo_list_tables`` / ``_vo_describe_table`` / ``_vo_adql_query`` /
``_vo_cone_search`` methods). Structural changes only:

  * the lazy service getter is injected as a bound callable
    (``get_vo_registry_service`` → the agent's ``_get_vo_registry_service``)
    and called INSIDE the legacy ``try`` at the same position, so a failing
    ``VoRegistryService`` constructor produces the byte-identical
    caught-and-typed error dict (not a NoneType attribute error);
  * the two shared agent helpers stay agent-side and are injected bound:
    ``external_catalog_table_result`` (sets the UI table card — a transport
    concern owned by the agent) and ``live_imagery_coordinates`` (target →
    ra/dec resolution shared with the imaging tools);
  * every legacy method began with ``self.last_run_result = None``; the
    provider clears it at ctx-build time (per call, immediately before
    ``run()``) — equivalent ordering, same as the Data Lab provider;
  * every path returns a :class:`ToolResult` whose ``native`` payload is the
    exact legacy output dict (byte-parity), including the pass-through of the
    service's own failure dicts (``if not result.get("success"): return result``).

None of these five methods carried ``@log_tool``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from capabilities.base import BaseCapability, ToolResult

logger = logging.getLogger(__name__)


def _native(out: Dict[str, Any]) -> ToolResult:
    """Wrap a legacy output dict as a byte-parity ToolResult."""
    ok = bool(isinstance(out, dict) and out.get("success"))
    err = out.get("error") if isinstance(out, dict) else None
    return ToolResult(
        success=ok,
        error=(str(err) if (err is not None and not ok) else None),
        native=out,
    )


class _In(BaseModel):
    model_config = ConfigDict(extra="ignore")


class VoFindServicesInput(_In):
    keywords: Optional[str]
    service_type: Optional[str] = None
    waveband: Optional[str] = None
    max_rows: Optional[int] = 30


class VoFindServices(BaseCapability):
    name = "vo_find_services"
    description = (
        "Discover Virtual Observatory services (TAP/SIA/SSA/cone) by keyword and waveband — "
        "finds archives Quasar has no built-in client for. Follow with vo_list_tables / "
        "vo_adql_query on the access_url."
    )
    category = "archive"
    InputModel = VoFindServicesInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        get_service = ctx.service("get_vo_registry_service")
        table_result = ctx.service("external_catalog_table_result")
        keywords, service_type = inp.keywords, inp.service_type
        waveband, max_rows = inp.waveband, inp.max_rows
        try:
            result = get_service().registry_search(
                keywords, service_type=service_type, waveband=waveband, max_rows=max_rows
            )
            if not result.get("success"):
                return _native(result)
            columns = ["short_name", "title", "service_type", "waveband", "access_url"]
            rows = result.get("rows") or []
            present_columns = [col for col in columns if any(col in row for row in rows)] or columns
            return _native(table_result(
                rows,
                columns=present_columns,
                source="IVOA Registry",
                filter_label=f"VO services for {keywords!r}"
                             + (f" [{service_type}]" if service_type else ""),
                tool_name="vo_find_services",
                warnings=result.get("warnings", []),
                provenance=result.get("provenance", {}),
            ))
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class VoListTablesInput(_In):
    access_url: Optional[str]
    keyword: Optional[str] = None
    max_tables: Optional[int] = 50


class VoListTables(BaseCapability):
    name = "vo_list_tables"
    description = "List (and keyword-filter) the tables of any TAP service found via vo_find_services."
    category = "archive"
    InputModel = VoListTablesInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        get_service = ctx.service("get_vo_registry_service")
        table_result = ctx.service("external_catalog_table_result")
        access_url, keyword, max_tables = inp.access_url, inp.keyword, inp.max_tables
        try:
            result = get_service().list_tables(
                access_url, keyword=keyword, max_tables=max_tables
            )
            if not result.get("success"):
                return _native(result)
            columns = ["table_name", "n_columns", "description"]
            rows = result.get("rows") or []
            present_columns = [col for col in columns if any(col in row for row in rows)] or columns
            return _native(table_result(
                rows,
                columns=present_columns,
                source=access_url,
                filter_label=f"TAP tables" + (f" matching {keyword!r}" if keyword else ""),
                tool_name="vo_list_tables",
                warnings=result.get("warnings", []),
                provenance=result.get("provenance", {}),
            ))
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class VoDescribeTableInput(_In):
    access_url: Optional[str]
    table_name: Optional[str]


class VoDescribeTable(BaseCapability):
    name = "vo_describe_table"
    description = "Column schema (names, datatypes, units, UCDs) of a table on any TAP service — call before writing ADQL."
    category = "archive"
    InputModel = VoDescribeTableInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        get_service = ctx.service("get_vo_registry_service")
        table_result = ctx.service("external_catalog_table_result")
        access_url, table_name = inp.access_url, inp.table_name
        try:
            result = get_service().describe_table(access_url, table_name)
            if not result.get("success"):
                return _native(result)
            columns = ["name", "datatype", "unit", "ucd", "description"]
            rows = result.get("rows") or []
            present_columns = [col for col in columns if any(col in row for row in rows)] or columns
            return _native(table_result(
                rows,
                columns=present_columns,
                source=f"Schema: {table_name}",
                filter_label=f"columns of {table_name}",
                tool_name="vo_describe_table",
                warnings=result.get("warnings", []),
                provenance=result.get("provenance", {}),
            ))
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class VoAdqlQueryInput(_In):
    access_url: Optional[str]
    adql: Optional[str]
    max_rows: Optional[int] = 200


class VoAdqlQuery(BaseCapability):
    name = "vo_adql_query"
    description = (
        "Run a guarded SELECT-only ADQL query against any TAP service URL. On ADQL errors "
        "the server's message is returned - read it and fix the query."
    )
    category = "archive"
    InputModel = VoAdqlQueryInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        get_service = ctx.service("get_vo_registry_service")
        table_result = ctx.service("external_catalog_table_result")
        access_url, adql, max_rows = inp.access_url, inp.adql, inp.max_rows
        try:
            result = get_service().run_adql(access_url, adql, max_rows=max_rows)
            if not result.get("success"):
                return _native(result)
            rows = result.get("rows") or []
            columns = list(result.get("columns") or [])
            warnings = list(result.get("warnings") or [])
            if len(columns) > 12:
                columns = columns[:12]
                warnings.append("Displaying the first 12 of the result's columns.")
            return _native(table_result(
                rows,
                columns=columns or ["result"],
                source=f"TAP: {access_url}",
                filter_label=(adql or "")[:120],
                tool_name="vo_adql_query",
                warnings=warnings,
                provenance=result.get("provenance", {}),
            ))
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class VoConeSearchInput(_In):
    access_url: Optional[str]
    target_name: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    radius_deg: Optional[float] = 0.1
    max_rows: Optional[int] = 100


class VoConeSearch(BaseCapability):
    name = "vo_cone_search"
    description = "Cone search any VO simple-cone-search service by position."
    category = "archive"
    InputModel = VoConeSearchInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        get_service = ctx.service("get_vo_registry_service")
        table_result = ctx.service("external_catalog_table_result")
        resolve = ctx.service("live_imagery_coordinates")
        access_url, target_name, ra, dec = inp.access_url, inp.target_name, inp.ra, inp.dec
        radius_deg, max_rows = inp.radius_deg, inp.max_rows
        try:
            ra_f, dec_f, label = resolve(target_name=target_name, ra=ra, dec=dec)
            result = get_service().cone_search(
                access_url, ra_f, dec_f, radius_deg=radius_deg, max_rows=max_rows
            )
            if not result.get("success"):
                return _native(result)
            rows = result.get("rows") or []
            columns = list(result.get("columns") or [])
            warnings = list(result.get("warnings") or [])
            if len(columns) > 12:
                columns = columns[:12]
                warnings.append("Displaying the first 12 of the result's columns.")
            radius_label = float(result.get("provenance", {}).get("radius_deg", radius_deg))
            return _native(table_result(
                rows,
                columns=columns or ["result"],
                source=f"SCS: {access_url}",
                filter_label=f"cone at {label}, r={radius_label:g} deg",
                tool_name="vo_cone_search",
                warnings=warnings,
                provenance=result.get("provenance", {}),
            ))
        except Exception as e:
            return _native({"success": False, "error": str(e)})


CAPABILITIES: List[BaseCapability] = [
    VoFindServices(),
    VoListTables(),
    VoDescribeTable(),
    VoAdqlQuery(),
    VoConeSearch(),
]

__all__ = [
    "CAPABILITIES",
    "VoFindServices", "VoListTables", "VoDescribeTable",
    "VoAdqlQuery", "VoConeSearch",
]
