"""
capabilities/datalab.py — the Data Lab family as transport-pure capabilities
(P1 reference migration).

Logic relocated VERBATIM from ``core/agent.py`` (the ``_datalab_*`` methods +
``_execute_datalab_sql`` + ``_datalab_error`` + ``_datalab_fit_rows`` +
``_datalab_query_summary``). The only changes are structural:

  * the Data Lab CLIENT and RESULT STORE arrive via ``CallContext`` (injected),
    not ``self._get_datalab_client()`` / ``self._get_datalab_result_store()``;
  * ``self.last_run_result = None`` (an SSE/data-card transport concern) moves to
    the native adapter's context provider — a capability never touches it;
  * every path returns a :class:`ToolResult`. Its ``native`` payload is the exact
    legacy output dict, so the native tool call is byte-for-byte identical and
    the DataLabBench baseline cannot drift.

The pure helpers (``services.datalab_query_builders`` / ``datalab_sql_policy`` /
``datalab_registry``) are imported directly — they are stateless logic, not
per-request state, so importing them keeps the capability transport-pure.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from capabilities.base import BaseCapability, CallContext, Provenance, ToolResult
from services import (
    datalab_orchestration,
    datalab_query_builders,
    datalab_registry,
    datalab_sql_policy,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (relocated verbatim from agent.py)
# ─────────────────────────────────────────────────────────────────────────────
def _fit_rows(frame, max_rows: int, *, char_budget: int = 6000):
    """Return (rows, truncated) trimmed so the JSON stays under char_budget."""
    total = int(len(frame))
    if total == 0:
        return [], False
    n = max(1, min(int(max_rows or 1), total))
    rows = frame.head(n).to_dict(orient="records")
    truncated = total > len(rows)
    blob = json.dumps(rows, default=str)
    if len(blob) > char_budget:
        avg = max(1, len(blob) // max(1, len(rows)))
        rows = rows[: max(1, char_budget // avg)]
        truncated = True
        while len(rows) > 1 and len(json.dumps(rows, default=str)) > char_budget:
            rows = rows[: max(1, len(rows) - 5)]
    if rows and len(json.dumps(rows, default=str)) > char_budget:
        ncols = max(1, len(rows[0]))
        per_cell = max(40, char_budget // (len(rows) * ncols))
        rows = [
            {k: (v[:per_cell] + "…" if isinstance(v, str) and len(v) > per_cell else v) for k, v in r.items()}
            for r in rows
        ]
        truncated = True
    return rows, truncated


def _query_summary(sql: str) -> str:
    compact = " ".join(str(sql or "").split())
    return compact[:700] + ("..." if len(compact) > 700 else "")


def datalab_error(error: Exception) -> ToolResult:
    """Map an exception to a typed-error ToolResult (never fabricates data)."""
    payload: Dict[str, Any] = {"success": False, "error": str(error)}
    fix_hint = None
    if isinstance(error, datalab_sql_policy.DatalabPolicyError):
        fix_hint = error.fix_hint
    else:
        text = str(error).lower()
        if "column" in text and "does not exist" in text:
            fix_hint = (
                "A column name in the query is wrong. Use the HINT in the error if present, or call "
                "datalab_describe_table(catalog, table) to list the valid columns, then re-run the "
                "corrected query. Do not give up after this error."
            )
        elif "relation" in text and "does not exist" in text:
            fix_hint = (
                "The table name is wrong. Call datalab_list_catalogs and datalab_describe_table to "
                "find the correct schema-qualified table, then re-run the query."
            )
        elif "not found for expression" in str(error) or "unsupported expression" in text:
            fix_hint = (
                "Fix the expression using the available columns listed in the error, then call the "
                "plot tool again. Derived quantities can be computed inline, e.g. "
                "'phot_g_mean_mag + 5*log10(parallax/100)'."
            )
    if fix_hint is not None:
        payload["fix_hint"] = fix_hint
    return ToolResult(success=False, error=str(error), native=payload,
                      meta=({"fix_hint": fix_hint} if fix_hint else {}))


def execute_datalab_sql(
    sql: str,
    meta: Dict[str, Any],
    *,
    tool_name: str,
    source: str = "builder",
    ctx: CallContext,
) -> ToolResult:
    """Validate → execute → store → summarize. Byte-parity with the legacy
    ``QuasarAgent._execute_datalab_sql``."""
    try:
        # Fetch client/store INSIDE the try so a missing/None client or store is
        # mapped through datalab_error (matching the legacy in-try acquisition),
        # not raised as a bare KeyError. (P1 review C)
        client = ctx.service("datalab_client")
        store = ctx.result_store
        validated = datalab_sql_policy.validate(sql, source=source, meta=meta)
        result = client.query(sql=validated.sql, fmt="pandas")
        store_meta = {
            **validated.meta,
            "tool_name": tool_name,
            "validated_sql": validated.sql,
            "warnings": validated.warnings,
            "provenance": {
                **result.provenance,
                "query": validated.sql,
                "tool_name": tool_name,
                "policy_source": source,
                **(
                    {"healpix": validated.meta["healpix"]}
                    if isinstance(validated.meta, dict) and validated.meta.get("healpix")
                    else {}
                ),
            },
        }
        result_id = store.put(result.dataframe, store_meta)
        preview_df = result.dataframe
        preview_reordered = False
        if len(preview_df) > 10:
            _nan_counts = preview_df.isna().sum(axis=1)
            if int(_nan_counts.head(10).sum()) > 0:
                preview_df = preview_df.loc[_nan_counts.sort_values(kind="stable").index]
                preview_reordered = True
        preview_rows, preview_more = _fit_rows(preview_df, 10, char_budget=4000)
        summary: Dict[str, Any] = {
            "success": True,
            "tool_name": tool_name,
            "result_id": result_id,
            "rowcount": int(len(result.dataframe)),
            "columns": result.columns[:30],
            "warnings": validated.warnings,
            "catalog": validated.meta.get("catalog") or result.provenance.get("catalog"),
            "table": validated.meta.get("table") or result.provenance.get("table"),
            "query_summary": _query_summary(validated.sql),
            "preview": preview_rows,
            "preview_truncated": preview_more,
            "note": (
                "Preview shows the most complete rows (some rows contain NaNs; the full "
                "result keeps its original order); fetch up to 5000 rows with "
                "datalab_get_result(result_id)."
                if preview_reordered
                else "Preview shows the first rows; fetch up to 5000 rows with datalab_get_result(result_id)."
            ),
        }
        if "row_count" in result.dataframe.columns and not result.dataframe.empty:
            summary["reported_count"] = int(result.dataframe.iloc[0]["row_count"])
        return ToolResult(
            success=True,
            result_id=result_id,
            warnings=list(validated.warnings or []),
            provenance=Provenance(
                service="datalab",
                query=validated.sql,
                tool_name=tool_name,
                rowcount=int(len(result.dataframe)),
                endpoint=(result.provenance or {}).get("endpoint"),
            ),
            native=summary,
        )
    except Exception as e:
        return datalab_error(e)


# ─────────────────────────────────────────────────────────────────────────────
# Input models (mirror the legacy method signatures; extra args ignored)
# ─────────────────────────────────────────────────────────────────────────────
class _In(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ListCatalogsInput(_In):
    pass


class DescribeTableInput(_In):
    catalog: str
    table: str


class ConeCountInput(_In):
    catalog: str
    table: str
    ra: float
    dec: float
    radius_deg: float


class SelectCatalogRowsInput(_In):
    catalog: str
    table: str
    ra: float
    dec: float
    radius_deg: float
    columns: Optional[List[str]] = None
    limit: int = 500
    value_cuts: Optional[List[Dict[str, Any]]] = None
    color_cut: Optional[Dict[str, Any]] = None
    morphology: Optional[Dict[str, Any]] = None


class Q3cCrossmatchInput(_In):
    ra: float
    dec: float
    radius_deg: float
    small_catalog: str = "gaia_dr3"
    small_table: str = "gaia_source"
    big_catalog: str = "nsc_dr2"
    big_table: str = "object"
    match_radius_arcsec: float = 1.0
    small_columns: Optional[List[str]] = None
    big_columns: Optional[List[str]] = None
    small_limit: int = 10000
    limit: int = 500


class SqlQueryInput(_In):
    sql: str
    # `Any` (not bool) so pydantic does NOT coerce "true"/1/"yes"/"on" to True.
    # The legacy gate is a strict identity check (`expert_ack is not True`); a
    # bool field would silently open the restricted raw-SQL gate for stringified
    # truthy values that gpt-oss/deepseek emit. Parity preserved. (P1 review A)
    expert_ack: Any = False
    reason: str = ""


class GetResultInput(_In):
    result_id: str
    max_rows: int = 200


class DensityAggregateInput(_In):
    # Every optional field is declared Optional[...] with the LEGACY default so an
    # explicit JSON `null` from the model does not raise a pydantic ValidationError
    # — it is forwarded to the query builder unchanged, exactly as the legacy
    # method did (`str(mode or "grid")` etc. is the builder's job, not ours).
    # `catalog`/`table` stay required non-nullable, matching the legacy signature.
    # Do NOT add `or <default>` coercions here: that would change what the builder
    # receives and silently break this parity migration. (docs/v2 P1)
    catalog: str
    table: str
    mode: Optional[str] = "grid"
    step_deg: Optional[float] = 0.1
    healpix_column: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    radius_deg: Optional[float] = None
    # `Any` (not bool) so pydantic does NOT coerce "false"/"0"/"no"/"off" to False
    # (and does not reject "maybe" outright). The builder's gate is a bare Python
    # truthiness check (`elif all_sky:`), so the legacy method — which forwarded
    # the raw JSON value — treated ANY non-empty string as all-sky. A `bool` field
    # would silently flip that branch. Same hazard, same fix as SqlQueryInput
    # .expert_ack above. Parity preserved. (P1 review)
    all_sky: Any = False
    color_cut: Optional[Dict[str, Any]] = None
    value_cuts: Optional[List[Dict[str, Any]]] = None
    morphology: Optional[Dict[str, Any]] = None
    limit: Optional[int] = 5000


# ─────────────────────────────────────────────────────────────────────────────
# Capabilities
# ─────────────────────────────────────────────────────────────────────────────
class ListCatalogs(BaseCapability):
    name = "datalab_list_catalogs"
    description = "List the available NOIRLab Astro Data Lab catalogs."
    category = "datalab"
    InputModel = ListCatalogsInput
    annotations = {"read_only": True, "cost": "cheap"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            catalogs = datalab_registry.list_catalogs()
            native = {"success": True, "catalogs": catalogs, "count": len(catalogs)}
            return ToolResult(success=True, data=catalogs, native=native)
        except Exception as e:
            return ToolResult(success=False, error=str(e), native={"success": False, "error": str(e)})


class DescribeTable(BaseCapability):
    name = "datalab_describe_table"
    description = "Describe a Data Lab catalog table (columns, types)."
    category = "datalab"
    InputModel = DescribeTableInput
    annotations = {"read_only": True, "cost": "cheap"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            table = datalab_registry.describe_table(inp.catalog, inp.table)
            return ToolResult(success=True, data=table, native={"success": True, "table": table})
        except Exception as e:
            return ToolResult(success=False, error=str(e), native={"success": False, "error": str(e)})


class ConeCount(BaseCapability):
    name = "datalab_cone_count"
    description = "Count catalog sources in a cone, server-side."
    category = "datalab"
    InputModel = ConeCountInput
    annotations = {"read_only": True, "cost": "moderate"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            sql, meta = datalab_query_builders.build_cone_count(
                inp.catalog, inp.table, ra=inp.ra, dec=inp.dec, radius_deg=inp.radius_deg
            )
        except Exception as e:
            return datalab_error(e)
        return execute_datalab_sql(sql, meta, tool_name=self.name, ctx=ctx)


class SelectCatalogRows(BaseCapability):
    name = "datalab_select_catalog_rows"
    description = "Select capped rows from a Data Lab catalog cone with governed SQL."
    category = "datalab"
    InputModel = SelectCatalogRowsInput
    annotations = {"read_only": True, "cost": "moderate"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            predicates = None
            if inp.value_cuts or inp.color_cut or inp.morphology:
                predicates = datalab_query_builders.build_catalog_predicates(
                    inp.catalog, inp.table,
                    color_cut=inp.color_cut, value_cuts=inp.value_cuts, morphology=inp.morphology,
                )
            sql, meta = datalab_query_builders.build_cone_select(
                inp.catalog, inp.table, ra=inp.ra, dec=inp.dec, radius_deg=inp.radius_deg,
                columns=inp.columns, limit=inp.limit, predicates=predicates,
            )
        except Exception as e:
            return datalab_error(e)
        return execute_datalab_sql(sql, meta, tool_name=self.name, ctx=ctx)


class Q3cCrossmatch(BaseCapability):
    name = "datalab_q3c_crossmatch"
    description = "Positional Q3C crossmatch of a small catalog against a big indexed one."
    category = "datalab"
    InputModel = Q3cCrossmatchInput
    annotations = {"read_only": True, "cost": "expensive"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            sql, meta = datalab_query_builders.build_q3c_crossmatch(
                small_catalog=inp.small_catalog, small_table=inp.small_table,
                big_catalog=inp.big_catalog, big_table=inp.big_table,
                ra=inp.ra, dec=inp.dec, radius_deg=inp.radius_deg,
                match_radius_arcsec=inp.match_radius_arcsec,
                small_columns=inp.small_columns, big_columns=inp.big_columns,
                small_limit=inp.small_limit, limit=inp.limit,
            )
        except Exception as e:
            return datalab_error(e)
        return execute_datalab_sql(sql, meta, tool_name=self.name, ctx=ctx)


class SqlQuery(BaseCapability):
    name = "datalab_sql_query"
    description = "Restricted expert/debug raw ADQL/SQL execution (requires expert_ack + reason)."
    category = "datalab"
    InputModel = SqlQueryInput
    annotations = {"read_only": True, "cost": "expensive"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            if inp.expert_ack is not True or not str(inp.reason or "").strip():
                native = {
                    "success": False,
                    "error": "datalab_sql_query is restricted expert/debug mode and requires expert_ack=true plus a reason.",
                }
                return ToolResult(success=False, error=native["error"], native=native)
            meta = {"source": "expert", "builder": "raw_sql", "expert_reason": str(inp.reason).strip()}
        except Exception as e:
            return datalab_error(e)
        return execute_datalab_sql(inp.sql, meta, tool_name=self.name, source="expert", ctx=ctx)


class GetResult(BaseCapability):
    name = "datalab_get_result"
    description = "Fetch the rows of a stored Data Lab result_id (capped at max_rows <= 5000)."
    category = "datalab"
    InputModel = GetResultInput
    annotations = {"read_only": True, "cost": "cheap"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            res = ctx.result_store.get(inp.result_id)
            cap = max(1, min(int(inp.max_rows or 200), 5000))
            rows, truncated = _fit_rows(res.dataframe, cap)
            native = {
                "success": True,
                "result_id": inp.result_id,
                "rowcount": int(len(res.dataframe)),
                "returned_rows": len(rows),
                "columns": res.columns,
                "rows": rows,
                "provenance": res.provenance,
                "truncated": truncated,
            }
            return ToolResult(success=True, result_id=inp.result_id, data=rows,
                              columns=res.columns, native=native)
        except Exception as e:
            return datalab_error(e)


class DensityAggregate(BaseCapability):
    # Description copied VERBATIM from the legacy inline registration so the
    # LLM-facing surface is identical whether the tool is reached through the
    # agent registry (which keeps its own `parameters=` schema) or straight
    # through `adapters.native.build_tool`. (docs/v2 P1)
    name = "datalab_density_aggregate"
    description = (
        "Aggregate Data Lab source density by RA/Dec grid or registered HEALPix column over a cone "
        "region; returns a stable result_id. Requires a cone (ra/dec/radius_deg) unless all_sky=true "
        "is set explicitly. Wide cones that exceed the 60s sync window are automatically tiled into "
        "sub-cones and merged — do NOT hand-tile the region yourself; call once with the full cone."
    )
    category = "datalab"
    InputModel = DensityAggregateInput
    annotations = {"read_only": True, "cost": "expensive"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            predicates = datalab_query_builders.build_catalog_predicates(
                inp.catalog, inp.table, color_cut=inp.color_cut, value_cuts=inp.value_cuts,
                morphology=inp.morphology,
            )
            sql, meta = datalab_query_builders.build_density_aggregate(
                inp.catalog, inp.table, mode=inp.mode, step_deg=inp.step_deg,
                healpix_column=inp.healpix_column, ra=inp.ra, dec=inp.dec,
                radius_deg=inp.radius_deg, all_sky=inp.all_sky, predicates=predicates,
                limit=inp.limit,
            )
            has_cone = inp.ra is not None and inp.dec is not None and inp.radius_deg is not None
            # Once one aggregate on this table has sync-timed-out this turn,
            # go straight to tiling for further wide cones — the doomed 60s
            # sync attempt per call burned ~3 minutes of live DS-P8's clock.
            # The set is PER-TURN state owned by the agent (reset in
            # stream_response_api) and handed in through CallContext, so this
            # capability mutates it without ever touching `self` on the agent.
            timeout_tables = ctx.service("datalab_agg_timeout_tables")
            table_key = f"{inp.catalog}.{inp.table}".lower()
            skip_sync = has_cone and float(inp.radius_deg) >= 2.0 and table_key in timeout_tables
            sync_result: Optional[ToolResult] = None
            if skip_sync:
                out: Dict[str, Any] = {
                    "success": False,
                    "error": "sync skipped: earlier aggregate on this table timed out",
                }
            else:
                sync_result = execute_datalab_sql(
                    sql, meta, tool_name="datalab_density_aggregate", ctx=ctx
                )
                out = sync_result.to_native()
            # Wide-cone sync timeout (anonymous tokens cannot use the async-job
            # path — live P8 both models): auto-tile the cone into sub-cones
            # sized for the 60s window and merge, instead of failing the tool.
            if (
                not out.get("success")
                and ("timed out" in str(out.get("error", "")).lower() or skip_sync)
                and has_cone
                and float(inp.radius_deg) >= 2.0
            ):
                logger.info(
                    "[DATALAB] density aggregate timed out at radius %s° — "
                    "auto-tiling (%s.%s, mode=%s, sync_skipped=%s)",
                    inp.radius_deg, inp.catalog, inp.table, inp.mode, skip_sync,
                )
                # Repeat attempts on a table that already proved slow get a
                # smaller tiling budget, so the model keeps enough turn clock
                # for more probes and the final render (live DS-P8 attempt 3:
                # four 210s probes of the Galactic centre ate the whole 900s).
                # NB: computed BEFORE the add() — the FIRST timeout on a table
                # gets the full budget (None); only a REPEAT gets 120s.
                _budget_override = 120.0 if table_key in timeout_tables else None
                timeout_tables.add(table_key)
                tiled = datalab_orchestration.tiled_density_aggregate(
                    inp.catalog, inp.table, mode=inp.mode, step_deg=inp.step_deg,
                    healpix_column=inp.healpix_column,
                    ra=float(inp.ra), dec=float(inp.dec), radius_deg=float(inp.radius_deg),
                    predicates=predicates, limit=inp.limit,
                    max_seconds=_budget_override,
                    client=ctx.service("datalab_client"),
                    result_store=ctx.result_store,
                )
                return ToolResult(
                    success=bool(isinstance(tiled, dict) and tiled.get("success")),
                    native=tiled,
                )
            # Non-tiling path: hand back the ToolResult execute_datalab_sql built,
            # untouched, so result_id / provenance / warnings survive for the
            # canonical (non-native) consumers.
            if sync_result is not None:
                return sync_result
            return ToolResult(success=False, error=str(out.get("error")), native=out)
        except Exception as e:
            return datalab_error(e)


# ─────────────────────────────────────────────────────────────────────────────
# Plot-from-result_id capabilities (image-producing)
# ─────────────────────────────────────────────────────────────────────────────
def _run_analysis_plot(
    fn_name: str,
    params: Dict[str, Any],
    ctx: CallContext,
    *,
    extra_services: Optional[Dict[str, str]] = None,
) -> ToolResult:
    """Shared runner for ``services.datalab_analysis`` plot functions.

    The plot logic lives in datalab_analysis (pure; the result store — and, for
    the SED, the SVO client — are injected). This returns the raw plot result
    (with image_base64/path/plotly_spec) as ToolResult data. The native adapter's
    image wrapper (``QuasarAgent._datalab_image_tool_fn``) does the SSE/UI
    transport: last_run_result + stripping the heavy base64 from the LLM-facing
    dict. The capability itself stays transport-pure. (docs/v2 P1)"""
    from services import datalab_analysis

    fn = getattr(datalab_analysis, fn_name)
    call_kw = dict(params)
    call_kw["result_store"] = ctx.result_store
    for kw, svc_key in (extra_services or {}).items():
        call_kw[kw] = ctx.service(svc_key)
    try:
        result = fn(**call_kw)
    except Exception as e:
        return datalab_error(e)
    if not isinstance(result, dict):
        native = {"success": False, "error": "plot function returned no result"}
        return ToolResult(success=False, error=native["error"], native=native)
    return ToolResult(
        success=bool(result.get("success")),
        error=(None if result.get("success") else result.get("error")),
        native=result,
    )


class CatalogScatterInput(_In):
    result_id: str
    x_expr: str
    y_expr: str
    color_by: Optional[str] = None
    invert_y: bool = False
    invert_x: bool = False
    title: str = "Data Lab catalog scatter"
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    overlay_locus: Optional[str] = None


class CatalogScatter(BaseCapability):
    name = "datalab_catalog_scatter"
    description = "Render a stored Data Lab result_id as a CMD/CCD/HR scatter plot."
    category = "datalab"
    InputModel = CatalogScatterInput
    annotations = {"read_only": True, "cost": "moderate", "produces": "image"}

    def run(self, inp, ctx) -> ToolResult:
        return _run_analysis_plot("catalog_scatter", inp.model_dump(), ctx)


class SkyDensityMapInput(_In):
    result_id: str
    mode: str = "hist2d"
    ra_col: Optional[str] = None
    dec_col: Optional[str] = None
    count_col: str = "source_count"
    bins: int = 80
    healpix_col: str = "healpix"
    nside: Optional[int] = None
    order: str = "nested"
    matched_filter: bool = False
    sigma_small: float = 1.0
    sigma_large: float = 3.0
    peak_threshold: float = 3.0
    max_peaks: int = 10
    log_scale: bool = True
    title: str = "Data Lab sky density map"


class SkyDensityMap(BaseCapability):
    name = "datalab_sky_density_map"
    description = "Render a stored Data Lab result_id as a RA/Dec or HEALPix density map."
    category = "datalab"
    InputModel = SkyDensityMapInput
    annotations = {"read_only": True, "cost": "moderate", "produces": "image"}

    def run(self, inp, ctx) -> ToolResult:
        return _run_analysis_plot("sky_density_map", inp.model_dump(), ctx)


class PeriodFoldInput(_In):
    result_id: str
    time_col: str = "mjd"
    mag_col: str = "cmag"
    error_col: Optional[str] = "cerr"
    band: Optional[str] = None
    band_col: str = "filter"
    min_frequency: float = 1.0
    max_frequency: float = 10.0
    title: str = "Data Lab period-folded light curve"


class PeriodFold(BaseCapability):
    name = "datalab_period_fold"
    description = "Lomb-Scargle period search + phase-fold of a stored Data Lab result_id."
    category = "datalab"
    InputModel = PeriodFoldInput
    annotations = {"read_only": True, "cost": "moderate", "produces": "image"}

    def run(self, inp, ctx) -> ToolResult:
        return _run_analysis_plot("period_fold", inp.model_dump(), ctx)


class SedPlotInput(_In):
    result_id: str
    row_index: int = 0
    filter_columns: Optional[Dict[str, str]] = None
    title: str = "Data Lab SED"


class SedPlot(BaseCapability):
    name = "datalab_sed_plot"
    description = "Render an SED from a stored Data Lab result_id using SVO FPS wavelengths."
    category = "datalab"
    InputModel = SedPlotInput
    annotations = {"read_only": True, "cost": "moderate", "produces": "image"}

    def run(self, inp, ctx) -> ToolResult:
        return _run_analysis_plot(
            "sed_plot", inp.model_dump(), ctx, extra_services={"svo_client": "svo_fps_client"}
        )


class LssWedgeInput(_In):
    result_id: str
    ra_col: Optional[str] = None
    dec_col: Optional[str] = None
    z_col: str = "z"
    class_col: Optional[str] = None
    pie_slice: bool = False
    title: str = "Data Lab large-scale structure wedge"


class LssWedge(BaseCapability):
    name = "datalab_lss_wedge"
    description = "Render a stored spectroscopic Data Lab result_id as a comoving LSS wedge."
    category = "datalab"
    InputModel = LssWedgeInput
    annotations = {"read_only": True, "cost": "moderate", "produces": "image"}

    def run(self, inp, ctx) -> ToolResult:
        return _run_analysis_plot("lss_wedge", inp.model_dump(), ctx)


# ─────────────────────────────────────────────────────────────────────────────
# SIA image capabilities (coordinate-resolved cutouts / color images)
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_coords(target_name, ra, dec, ctx: CallContext):
    """Resolve (ra, dec, label) via the injected coordinate resolver — the same
    _datalab_coordinates the inline tools use (byte-identical). (docs/v2 P1)"""
    return ctx.service("resolve_coordinates")(target_name=target_name, ra=ra, dec=dec)


class ImageCutoutInput(_In):
    fov_deg: float
    ra: Optional[float] = None
    dec: Optional[float] = None
    target_name: Optional[str] = None
    band: str = "g"
    catalog: str = "ls_dr9"
    endpoint: Optional[str] = None
    title: Optional[str] = None


class ImageCutout(BaseCapability):
    name = "datalab_image_cutout"
    description = "Render a single-band NOIRLab Astro Data Lab SIA cutout at RA/Dec or a resolvable target name."
    category = "datalab"
    InputModel = ImageCutoutInput
    annotations = {"read_only": True, "cost": "moderate", "produces": "image"}

    def run(self, inp, ctx) -> ToolResult:
        service = ctx.service("datalab_image_service")
        try:
            ra_f, dec_f, label = _resolve_coords(inp.target_name, inp.ra, inp.dec, ctx)
            caption = inp.title or f"Data Lab {inp.band}-band cutout: {label}"
            result = service.cutout(
                ra_f, dec_f, float(inp.fov_deg),
                band=inp.band, catalog=inp.catalog, endpoint=inp.endpoint, title=caption,
            )
            # Auto-substitute a working band when the requested one has no tiles.
            no_image = not (result.get("image_base64") or result.get("path"))
            suggested = list(result.get("suggested_bands") or [])
            if no_image and suggested:
                sub_band = suggested[0]
                sub_caption = inp.title or f"Data Lab {sub_band}-band cutout: {label} (requested {inp.band}, not available here)"
                retry = service.cutout(
                    ra_f, dec_f, float(inp.fov_deg),
                    band=sub_band, catalog=inp.catalog, endpoint=inp.endpoint, title=sub_caption,
                )
                if retry.get("image_base64") or retry.get("path"):
                    retry["band_substituted"] = {"requested": str(inp.band), "used": sub_band}
                    retry["note"] = (
                        f"No usable {inp.band}-band tiles at this position; rendered the {sub_band}-band "
                        f"cutout instead. State the substitution to the user."
                    )
                    retry["_caption"] = sub_caption
                    return ToolResult(success=bool(retry.get("success")), native=retry)
            result["_caption"] = caption
            return ToolResult(
                success=bool(result.get("success")),
                error=(None if result.get("success") else result.get("error")),
                native=result,
            )
        except Exception as e:
            return datalab_error(e)


class ColorImageInput(_In):
    fov_deg: float
    ra: Optional[float] = None
    dec: Optional[float] = None
    target_name: Optional[str] = None
    catalog: str = "ls_dr9"
    endpoint: Optional[str] = None
    bands: Optional[List[str]] = None
    q: float = 8.0
    stretch: float = 0.5
    title: Optional[str] = None


class ColorImage(BaseCapability):
    name = "datalab_color_image"
    description = "Render a NOIRLab Astro Data Lab SIA color image at RA/Dec or a resolvable target name."
    category = "datalab"
    InputModel = ColorImageInput
    annotations = {"read_only": True, "cost": "moderate", "produces": "image"}

    def run(self, inp, ctx) -> ToolResult:
        service = ctx.service("datalab_image_service")
        try:
            ra_f, dec_f, label = _resolve_coords(inp.target_name, inp.ra, inp.dec, ctx)
            caption = inp.title or f"Data Lab color image: {label}"
            result = service.color_image(
                ra_f, dec_f, float(inp.fov_deg),
                catalog=inp.catalog, endpoint=inp.endpoint, bands=inp.bands,
                q=float(inp.q), stretch=float(inp.stretch), title=caption,
            )
            result["_caption"] = caption
            return ToolResult(
                success=bool(result.get("success")),
                error=(None if result.get("success") else result.get("error")),
                native=result,
            )
        except Exception as e:
            return datalab_error(e)


class CutoutGridInput(_In):
    peaks: List[Dict[str, Any]]
    fov_deg: float
    band: str = "g"
    catalog: str = "ls_dr9"
    endpoint: Optional[str] = None
    title: str = "Data Lab cutout grid"


class CutoutGrid(BaseCapability):
    name = "datalab_cutout_grid"
    description = "Render a grid of SIA cutouts at a list of peak positions."
    category = "datalab"
    InputModel = CutoutGridInput
    annotations = {"read_only": True, "cost": "moderate", "produces": "image"}

    def run(self, inp, ctx) -> ToolResult:
        service = ctx.service("datalab_image_service")
        try:
            result = service.cutout_grid(
                inp.peaks, float(inp.fov_deg),
                band=inp.band, catalog=inp.catalog, endpoint=inp.endpoint, title=inp.title,
            )
            result["_caption"] = inp.title
            return ToolResult(
                success=bool(result.get("success")),
                error=(None if result.get("success") else result.get("error")),
                native=result,
            )
        except Exception as e:
            return datalab_error(e)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration capabilities (one-shot color diagrams; sky-area guard; jobs)
# ─────────────────────────────────────────────────────────────────────────────
class ConfirmSkyAreaInput(_In):
    ra_min: float
    ra_max: float
    dec_min: float
    dec_max: float
    tile_radius_deg: float = 2.0


class ConfirmSkyArea(BaseCapability):
    name = "datalab_confirm_sky_area"
    description = "Estimate/confirm the sky area (tile count) before a wide fan-out scan."
    category = "datalab"
    InputModel = ConfirmSkyAreaInput
    annotations = {"read_only": True, "cost": "cheap"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            fp = {"ra_min": inp.ra_min, "ra_max": inp.ra_max, "dec_min": inp.dec_min, "dec_max": inp.dec_max}
            native = {"success": True, **datalab_orchestration.confirm_sky_area(fp, float(inp.tile_radius_deg))}
            return ToolResult(success=True, native=native)
        except Exception as e:
            return datalab_error(e)


class ColorColorDiagramInput(_In):
    catalog: str
    table: Optional[str] = None
    radius_deg: Optional[float] = 0.5
    ra: Optional[float] = None
    dec: Optional[float] = None
    target_name: Optional[str] = None
    x_bands: Optional[List[str]] = None
    y_bands: Optional[List[str]] = None
    split_col: Optional[str] = None
    split_threshold: Optional[float] = 0.005
    limit: Optional[int] = 3000
    title: Optional[str] = None
    point_sources: bool = False
    morphology: Optional[Dict[str, Any]] = None
    value_cuts: Optional[List[Dict[str, Any]]] = None


class ColorColorDiagram(BaseCapability):
    name = "datalab_color_color_diagram"
    description = "One-shot color-color diagram for a Data Lab catalog cone (with morphology split)."
    category = "datalab"
    InputModel = ColorColorDiagramInput
    annotations = {"read_only": True, "cost": "moderate", "produces": "image"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            table = inp.table or datalab_registry.default_table(inp.catalog)
            radius_deg = float(inp.radius_deg) if inp.radius_deg is not None else 0.5
            split_threshold = float(inp.split_threshold) if inp.split_threshold is not None else 0.005
            limit = int(inp.limit) if inp.limit is not None else 3000
            ra_f, dec_f, label = _resolve_coords(inp.target_name, inp.ra, inp.dec, ctx)
            out = datalab_orchestration.color_color_diagram(
                inp.catalog, table, ra_f, dec_f, radius_deg,
                x_bands=tuple(inp.x_bands) if inp.x_bands else ("g", "r"),
                y_bands=tuple(inp.y_bands) if inp.y_bands else ("r", "i"),
                split_col=inp.split_col, split_threshold=split_threshold, limit=limit,
                title=inp.title or f"{inp.catalog} color-color: {label}",
                point_sources=bool(inp.point_sources), morphology=inp.morphology,
                value_cuts=inp.value_cuts,
            )
            if isinstance(out, dict):
                out["_caption"] = inp.title or f"Color-color diagram: {label}"
            return ToolResult(success=bool(isinstance(out, dict) and out.get("success")), native=out)
        except Exception as e:
            return datalab_error(e)


class ColorMagnitudeDiagramInput(_In):
    catalog: str
    table: Optional[str] = None
    radius_deg: Optional[float] = 0.4
    ra: Optional[float] = None
    dec: Optional[float] = None
    target_name: Optional[str] = None
    blue_band: str = "g"
    red_band: str = "r"
    mag_band: Optional[str] = None
    limit: Optional[int] = 5000
    title: Optional[str] = None
    point_sources: bool = False
    morphology: Optional[Dict[str, Any]] = None
    value_cuts: Optional[List[Dict[str, Any]]] = None


class ColorMagnitudeDiagram(BaseCapability):
    name = "datalab_color_magnitude_diagram"
    description = "One-shot color-magnitude (CMD/HR) diagram for a Data Lab catalog cone."
    category = "datalab"
    InputModel = ColorMagnitudeDiagramInput
    annotations = {"read_only": True, "cost": "moderate", "produces": "image"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            table = inp.table or datalab_registry.default_table(inp.catalog)
            radius_deg = float(inp.radius_deg) if inp.radius_deg is not None else 0.4
            limit = int(inp.limit) if inp.limit is not None else 5000
            blue_band = inp.blue_band or "g"
            red_band = inp.red_band or "r"
            ra_f, dec_f, label = _resolve_coords(inp.target_name, inp.ra, inp.dec, ctx)
            out = datalab_orchestration.color_magnitude_diagram(
                inp.catalog, table, ra_f, dec_f, radius_deg,
                blue_band=blue_band, red_band=red_band, mag_band=inp.mag_band, limit=limit,
                title=inp.title or f"{inp.catalog} CMD: {label}",
                point_sources=bool(inp.point_sources), morphology=inp.morphology,
                value_cuts=inp.value_cuts,
            )
            if isinstance(out, dict):
                out["_caption"] = inp.title or f"Color-magnitude diagram: {label}"
            return ToolResult(success=bool(isinstance(out, dict) and out.get("success")), native=out)
        except Exception as e:
            return datalab_error(e)


class JobIdInput(_In):
    job_id: str


class JobStatus(BaseCapability):
    name = "datalab_job_status"
    description = "Poll the status of a Data Lab background job (e.g. a tiled search)."
    category = "datalab"
    InputModel = JobIdInput
    annotations = {"read_only": True, "cost": "cheap"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            out = {"success": True, **ctx.service("datalab_job_service").status(inp.job_id)}
            # Job-aware turn ending (live DS-P15: the model polled a slow tiled
            # scan 18x until it silently hit HARD_MAX_ITERATIONS with no closing
            # message). After a few polls of a still-running job, tell the model
            # to stop polling and end the turn gracefully. The counter is
            # PER-TURN state owned by the agent and injected via CallContext;
            # it is fetched only on the non-terminal path, exactly as the legacy
            # method did. (docs/v2 P1)
            if str(out.get("status", "")).lower() in {"queued", "running"}:
                counts = ctx.service("datalab_job_poll_counts")
                counts[str(inp.job_id)] = counts.get(str(inp.job_id), 0) + 1
                if counts[str(inp.job_id)] >= 3:
                    out["stop_polling"] = True
                    out["instruction"] = (
                        f"This job is still {out.get('status')} server-side after "
                        f"{counts[str(inp.job_id)]} polls. STOP polling now. End your answer: "
                        "summarize any results you already have, state that job "
                        f"{inp.job_id} is still running, and tell the user to ask you to "
                        "check it again in a few minutes (datalab_job_status / "
                        "datalab_job_results). Do NOT call datalab_job_status again this turn."
                    )
            return ToolResult(success=True, native=out)
        except Exception as e:
            return datalab_error(e)


class JobResults(BaseCapability):
    name = "datalab_job_results"
    description = "Fetch the results of a background Data Lab job."
    category = "datalab"
    InputModel = JobIdInput
    annotations = {"read_only": True, "cost": "cheap"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            native = {"success": True, **ctx.service("datalab_job_service").results(inp.job_id)}
            return ToolResult(success=True, native=native)
        except Exception as e:
            return datalab_error(e)


class JobCancel(BaseCapability):
    name = "datalab_job_cancel"
    description = "Cancel a running background Data Lab job."
    category = "datalab"
    InputModel = JobIdInput
    annotations = {"read_only": False, "cost": "cheap"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            native = {"success": True, **ctx.service("datalab_job_service").cancel(inp.job_id)}
            return ToolResult(success=True, native=native)
        except Exception as e:
            return datalab_error(e)


# The migrated Data Lab sub-family. SQL/catalog core (incl. the auto-tiling
# density aggregate) + result_id plot tools + SIA imaging + orchestration/jobs.
# Still inline in core/agent.py: density_vetting, tiled_search, export_notebook.
SQL_CAPABILITIES: List[BaseCapability] = [
    ListCatalogs(),
    DescribeTable(),
    ConeCount(),
    SelectCatalogRows(),
    Q3cCrossmatch(),
    SqlQuery(),
    GetResult(),
    DensityAggregate(),
]

# Image-producing capabilities — registered via the agent's IMAGE wrapper
# (_datalab_image_tool_fn) so the artifact is emitted to the UI. Covers both
# result_id plots and SIA cutouts/color images.
PLOT_CAPABILITIES: List[BaseCapability] = [
    CatalogScatter(),
    SkyDensityMap(),
    PeriodFold(),
    SedPlot(),
    LssWedge(),
    ImageCutout(),
    ColorImage(),
    CutoutGrid(),
    ColorColorDiagram(),
    ColorMagnitudeDiagram(),
]

# Non-image orchestration/job capabilities — registered via the plain
# _datalab_tool_fn wrapper (no image transport).
MISC_CAPABILITIES: List[BaseCapability] = [
    ConfirmSkyArea(),
    JobStatus(),
    JobResults(),
    JobCancel(),
]

CAPABILITIES: List[BaseCapability] = SQL_CAPABILITIES + PLOT_CAPABILITIES + MISC_CAPABILITIES

__all__ = [
    "CAPABILITIES", "SQL_CAPABILITIES", "PLOT_CAPABILITIES", "MISC_CAPABILITIES",
    "execute_datalab_sql",
    "datalab_error",
    "ListCatalogs", "DescribeTable", "ConeCount", "SelectCatalogRows",
    "Q3cCrossmatch", "SqlQuery", "GetResult", "DensityAggregate",
    "CatalogScatter", "SkyDensityMap", "PeriodFold", "SedPlot", "LssWedge",
    "ImageCutout", "ColorImage", "CutoutGrid",
    "ColorColorDiagram", "ColorMagnitudeDiagram",
    "ConfirmSkyArea", "JobStatus", "JobResults", "JobCancel",
]
