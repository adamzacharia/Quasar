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
from typing import Any, Dict, List, Literal, Optional

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


# Public alias: non-datalab agent tools (MMU/external-catalog table summaries)
# reuse the same row-fitting logic. One implementation — the agent's private
# `_datalab_fit_rows` copy was retired in its favor. (docs/v2 P1)
fit_rows = _fit_rows


def datalab_error(
    error: Exception,
    *,
    sql: Optional[str] = None,
    tool_name: Optional[str] = None,
) -> ToolResult:
    """Map an exception to a typed-error ToolResult (never fabricates data).

    When the failure happened AFTER a query was attempted, pass ``sql`` so the
    provenance surface still shows the exact request that failed (CX-09) —
    a failed call is precisely when the user most wants to see the query.
    """
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
    provenance = None
    if sql:
        provenance = Provenance(service="datalab", query=sql, tool_name=tool_name)
    return ToolResult(success=False, error=str(error), native=payload,
                      provenance=provenance,
                      meta=({"fix_hint": fix_hint} if fix_hint else {}))


def execute_datalab_sql(
    sql: str,
    meta: Dict[str, Any],
    *,
    tool_name: str,
    source: str = "builder",
    ctx: CallContext,
    async_submit: Any = False,
) -> ToolResult:
    """Validate → execute → store → summarize. Byte-parity with the legacy
    ``QuasarAgent._execute_datalab_sql``.

    With ``async_submit`` truthy the validated query is submitted as a job and
    the job_id is returned UP FRONT (not only on sync timeout): a real
    server-side Data Lab job when a login token is configured, else a local
    threaded run (the anonymous token gets HTTP 401 from the server job API).
    Poll with datalab_job_status / fetch with datalab_job_results."""
    # The EXECUTED text is validated.sql (the policy may add LIMITs/rewrites);
    # keep it for the error path so failures report the query that actually ran,
    # not the pre-validation input (CX-09-residual).
    executed_sql = sql
    try:
        # Fetch client/store INSIDE the try so a missing/None client or store is
        # mapped through datalab_error (matching the legacy in-try acquisition),
        # not raised as a bare KeyError. (P1 review C)
        client = ctx.service("datalab_client")
        store = ctx.result_store
        # Fail BEFORE issuing the remote query if the result store is missing.
        # The agent's context provider wraps service getters in a `_lazy` guard
        # (a failing constructor for an unrelated service yields None instead of
        # aborting the whole call); without this check a None store would let
        # `client.query()` fire and only then crash at `store.put()` — an
        # unnecessary remote query past the legacy fail-before-query boundary,
        # which built the store eagerly at context construction. (guard verify)
        if store is None:
            raise RuntimeError(
                "Data Lab result store is unavailable; cannot persist the query "
                "result_id. Not issuing the remote query."
            )
        # Nightly-cache driver for the live TAP schema the governor grounds
        # column names on: non-blocking (daemon thread), throttled, never raises.
        datalab_registry.ensure_tap_schema_fresh(client)
        validated = datalab_sql_policy.validate(sql, source=source, meta=meta)
        executed_sql = validated.sql  # CX-09-residual
        if async_submit:
            # Must return BEFORE client.query: with async_=True the client
            # returns a plain jobid string, not a DatalabResult.
            return _submit_async_query(validated, tool_name=tool_name, source=source, ctx=ctx, client=client)
        result = client.query(sql=validated.sql, fmt="pandas")
        # A result that exactly filled its LIMIT is a spatially-biased,
        # storage-order slice (live P7/P9) — warn the model and stamp the
        # provenance so downstream plot tools can refuse to present it as a
        # sky distribution.
        row_limit = validated.meta.get("row_limit") if isinstance(validated.meta, dict) else None
        trunc_warning = datalab_sql_policy.limit_truncation_warning(len(result.dataframe), row_limit)
        if trunc_warning:
            validated.warnings.append(trunc_warning)
        store_meta = {
            **validated.meta,
            "tool_name": tool_name,
            "validated_sql": validated.sql,
            "warnings": validated.warnings,
            # Scope the stored result to its requesting user so the export
            # route's owner guard is not vacuously skipped
            # (dl-export-owner-gap / UIAPI-05).
            **({"owner_id": str(ctx.user_id)} if ctx.user_id else {}),
            # A LIMIT-capped SELECT stored here is itself a slice of the
            # matching rows — ride the F2 upstream channel so any card built
            # from this result can never read as complete. The remote total is
            # unknown (service-side cap), so no upstream_total. (scan-L5)
            **({"upstream_truncated": True} if trunc_warning else {}),
            "provenance": {
                **result.provenance,
                "query": validated.sql,
                "tool_name": tool_name,
                "policy_source": source,
                **(
                    {"row_limit": int(row_limit), "limit_truncated": True}
                    if trunc_warning
                    else {}
                ),
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
        if trunc_warning:
            summary["limit_truncated"] = True
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
        # A sync-timeout fallback can leave a REAL server job running (or
        # errored) — register it so datalab_job_status/results resolve for this
        # user instead of "Unknown Data Lab job" (live P14: the error message
        # pointed at a jobid the job tools could not see).
        fallback_jobid = getattr(e, "jobid", None)
        if fallback_jobid:
            try:
                ctx.service("datalab_job_service").register_external(
                    str(fallback_jobid),
                    kind="server_query",
                    params={"tool_name": tool_name, "sql": executed_sql, "policy_source": source},
                    status="running",
                    **_job_owner_kwargs(ctx),
                )
            except Exception:  # noqa: BLE001 - registration is best-effort
                pass
        # executed_sql is validated.sql once validation succeeded — the text
        # that actually ran (CX-09-residual); before that point it is the input.
        return datalab_error(e, sql=executed_sql, tool_name=tool_name)


def _job_owner_kwargs(ctx: CallContext) -> Dict[str, str]:
    user_id = str(ctx.user_id or "").strip()
    return {"owner_id": user_id} if user_id else {}


def _store_user_kwargs(ctx: CallContext) -> Dict[str, str]:
    user_id = str(ctx.user_id or "").strip()
    return {"user_id": user_id} if user_id else {}


def _submit_async_query(validated, *, tool_name: str, source: str, ctx: CallContext, client) -> ToolResult:
    """First-class async submit: return a job_id up front for a governed query.

    Real server-side Data Lab job when the client carries a login token;
    otherwise a local threaded run (live-verified: /status and /results reject
    the anonymous token with HTTP 401, so submitting anon server jobs would
    strand them unpollable)."""
    from integrations.datalab_client import ANON_TOKEN

    job_service = ctx.service("datalab_job_service")
    params = {
        "tool_name": tool_name,
        "sql": validated.sql,
        "policy_source": source,
        "catalog": validated.meta.get("catalog"),
        "table": validated.meta.get("table"),
        # Builder healpix {column, nside, scheme} must survive into the job
        # record so JobResults can re-attach it when persisting server-job rows
        # — without it sky_density_map decodes RING pixels as NESTED with a
        # guessed nside (dl-async-healpix-scheme-lost).
        **(
            {"healpix": validated.meta["healpix"]}
            if isinstance(validated.meta, dict) and validated.meta.get("healpix")
            else {}
        ),
    }
    poll_note = (
        "Poll ONCE with datalab_job_status(job_id) if the user is waiting, then end the "
        "turn; fetch rows later with datalab_job_results(job_id)."
    )
    if getattr(client, "token", None) and client.token != ANON_TOKEN:
        job_id = client.submit(sql=validated.sql)
        job_service.register_external(
            job_id,
            kind="server_query",
            params=params,
            **_job_owner_kwargs(ctx),
        )
        native = {
            "success": True,
            "job_id": str(job_id),
            "job_kind": "server",
            "status": "submitted",
            "tool_name": tool_name,
            "warnings": validated.warnings,
            "note": "Query submitted as a server-side Data Lab job (survives restarts). " + poll_note,
        }
        return ToolResult(
            success=True,
            warnings=list(validated.warnings or []),
            # The submitted SQL is this call's exact request (CX-09) — surface
            # it even though the rows arrive later via the job tools.
            provenance=Provenance(service="datalab", query=validated.sql, tool_name=tool_name),
            native=native,
        )

    # Anonymous token → local threaded runner. Capture plain locals (the client
    # and store are long-lived singletons), never ctx — the closure runs on the
    # job service worker after this request's CallContext is gone.
    sql_text = validated.sql
    store = ctx.result_store
    owner_id = str(ctx.user_id) if ctx.user_id else None
    healpix_meta = (
        validated.meta.get("healpix") if isinstance(validated.meta, dict) else None
    )
    row_limit = validated.meta.get("row_limit") if isinstance(validated.meta, dict) else None
    store_meta = {
        **validated.meta,
        "tool_name": tool_name,
        "validated_sql": sql_text,
        "warnings": validated.warnings,
        **({"owner_id": owner_id} if owner_id else {}),  # dl-export-owner-gap
    }

    def _job(cancel_check):
        result = client.query(sql=sql_text, fmt="pandas", async_fallback=False)
        # Mirror the sync path's provenance stamps: healpix {column,nside,scheme}
        # (dl-async-healpix-scheme-lost) and the row-cap truncation flag so a
        # capped async result can never be plotted as complete.
        trunc_warning = datalab_sql_policy.limit_truncation_warning(
            len(result.dataframe), row_limit
        )
        job_meta = dict(store_meta)
        if trunc_warning:
            job_meta["warnings"] = list(job_meta.get("warnings") or []) + [trunc_warning]
        result_id = store.put(result.dataframe, {
            **job_meta,
            "provenance": {
                **result.provenance,
                "query": sql_text,
                "tool_name": tool_name,
                "policy_source": source,
                **({"healpix": healpix_meta} if healpix_meta else {}),
                **(
                    {"row_limit": int(row_limit), "limit_truncated": True}
                    if trunc_warning
                    else {}
                ),
            },
        })
        return {
            "result_id": result_id,
            "rowcount": int(len(result.dataframe)),
            "columns": [str(col) for col in result.dataframe.columns][:30],
            **({"warnings": [trunc_warning]} if trunc_warning else {}),
            "note": "Fetch rows with datalab_get_result(result_id).",
        }

    job_id = job_service.start(
        "async_query",
        _job,
        params=params,
        **_job_owner_kwargs(ctx),
    )
    native = {
        "success": True,
        "job_id": job_id,
        "job_kind": "local",
        "status": "queued",
        "tool_name": tool_name,
        "warnings": validated.warnings,
        "note": (
            "No DATALAB_TOKEN configured, so the query runs as a LOCAL background job "
            "(server-side jobs need a Data Lab login). " + poll_note
        ),
    }
    return ToolResult(
        success=True,
        warnings=list(validated.warnings or []),
        provenance=Provenance(service="datalab", query=validated.sql, tool_name=tool_name),
        native=native,
    )


# Server job states → the local job-status vocabulary the model already knows.
_SERVER_STATE_MAP = {
    "QUEUED": "queued",
    "SUBMITTED": "queued",
    "EXECUTING": "running",
    "RUNNING": "running",
    "COMPLETED": "succeeded",
    "ERROR": "failed",
    "ABORTED": "canceled",
    "ABORT": "canceled",
}


def _require_server_job_client(ctx: CallContext):
    """Client for a server-side job id, with a typed error for the anon token."""
    from integrations.datalab_client import ANON_TOKEN

    client = ctx.service("datalab_client")
    if getattr(client, "token", None) == ANON_TOKEN:
        raise RuntimeError(
            "This is a server-side Data Lab job id, and the /status //results job API "
            "rejects the anonymous token (HTTP 401). Set DATALAB_TOKEN to a real Data "
            "Lab login token, or rerun the query without async_submit."
        )
    return client


# ─────────────────────────────────────────────────────────────────────────────
# Input models (mirror the legacy method signatures; extra args ignored)
# ─────────────────────────────────────────────────────────────────────────────
class _In(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ListCatalogsInput(_In):
    # 'registered' = curated governed subset (fast, offline); 'all' also reads
    # the complete schema list from the live tap_schema. Literal so a typo
    # surfaces as a validation error instead of silently meaning 'registered'.
    scope: Literal["registered", "all"] = "registered"


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
    # Optional like the diagram tools: defaults to the catalog's primary table
    # (live P6 burned a round on a pydantic "Field required" for `table`).
    table: Optional[str] = None
    ra: float
    dec: float
    radius_deg: float
    columns: Optional[List[str]] = None
    limit: int = 500
    value_cuts: Optional[List[Dict[str, Any]]] = None
    color_cut: Optional[Dict[str, Any]] = None
    morphology: Optional[Dict[str, Any]] = None
    # `Any` (not bool) — downstream is a bare truthiness check, and stringified
    # "true"/"false" from gpt-oss/deepseek must keep legacy truthiness (same
    # hazard/fix as SqlQueryInput.expert_ack below).
    async_submit: Any = False


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
    async_submit: Any = False  # `Any`, not bool — see expert_ack above


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
    async_submit: Any = False  # `Any`, not bool — see all_sky above
    confirm: Any = False  # `Any`, not bool — see all_sky above (HITL wide-area gate)


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
            native = {
                "success": True,
                "scope": "registered",
                "catalogs": catalogs,
                "count": len(catalogs),
                "note": (
                    "This is the CURATED, registry-governed subset of Data Lab (best structured-"
                    "builder and SQL-governor support), not the complete service inventory — "
                    "call with scope='all' for the full live schema list."
                ),
            }
            if str(inp.scope or "registered").strip().lower() != "all":
                return ToolResult(success=True, data=catalogs, native=native)
            # scope='all': the complete live tap_schema.schemas inventory, with
            # the curated subset marked registry_governed. Network failure keeps
            # the curated list usable — with an explicit note, never silently.
            try:
                schemas = datalab_registry.list_all_schemas(ctx.service("datalab_client"))
                native.update({
                    "scope": "all",
                    "schemas": schemas,
                    "schema_count": len(schemas),
                    "note": (
                        "Complete live Data Lab schema list (tap_schema.schemas). Entries with "
                        "registry_governed=true are the curated subset in 'catalogs' with full "
                        "structured-builder support; other schemas become queryable after "
                        "datalab_describe_table caches their live columns."
                    ),
                })
            except Exception as live_error:  # noqa: BLE001 - degrade to the curated list
                native["note"] = (
                    "The full live schema list was unavailable "
                    f"({live_error}); showing only the curated registry-governed subset."
                )
            return ToolResult(success=True, data=native, native=native)
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
            table = inp.table or datalab_registry.default_table(inp.catalog)
            # Registry default quality cuts (DESI zwarn/survey/main_primary, DES
            # flags, SDSS zwarning) apply unless the caller cuts the same column
            # (live P11: unfiltered LRG counts were 11-27% inflated per z-bin).
            merged_cuts, quality_note = datalab_registry.merge_default_quality_cuts(
                inp.catalog, table, inp.value_cuts
            )
            predicates = None
            if merged_cuts or inp.color_cut or inp.morphology:
                predicates = datalab_query_builders.build_catalog_predicates(
                    inp.catalog, table,
                    color_cut=inp.color_cut, value_cuts=merged_cuts, morphology=inp.morphology,
                )
            sql, meta = datalab_query_builders.build_cone_select(
                inp.catalog, table, ra=inp.ra, dec=inp.dec, radius_deg=inp.radius_deg,
                columns=inp.columns, limit=inp.limit, predicates=predicates,
            )
            if quality_note:
                meta.setdefault("warnings", []).append(quality_note)
        except Exception as e:
            return datalab_error(e)
        return execute_datalab_sql(sql, meta, tool_name=self.name, ctx=ctx, async_submit=inp.async_submit)


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
        return execute_datalab_sql(inp.sql, meta, tool_name=self.name, source="expert", ctx=ctx,
                                   async_submit=inp.async_submit)


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
        "region; returns a stable result_id. Requires a cone (ra/dec/radius_deg), OR an "
        "indexed-equality value_cut (e.g. SMASH fieldid = 169 — maps the WHOLE survey field, no "
        "cone guessing), OR an explicit all_sky=true. Wide cones that exceed the 60s sync window "
        "are automatically tiled into sub-cones and merged — do NOT hand-tile the region yourself; "
        "call once with the full cone."
    )
    category = "datalab"
    InputModel = DensityAggregateInput
    annotations = {"read_only": True, "cost": "expensive"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            # HITL wide-area gate (live P15: a 30° "quick look" ≈ 2,800 deg² of
            # server-side aggregation died in async ERROR with no confirmation
            # step; same pattern as datalab_tiled_search's confirm).
            if (
                inp.ra is not None and inp.dec is not None and inp.radius_deg is not None
                and not inp.confirm
            ):
                decision = datalab_orchestration.confirm_cone_area(
                    float(inp.ra), float(inp.dec), float(inp.radius_deg)
                )
                if decision.get("needs_confirmation"):
                    return ToolResult(success=False, error=decision.get("message"), native={
                        "success": False,
                        "needs_confirmation": True,
                        **decision,
                        "hint": "Re-call datalab_density_aggregate with confirm=true only if the "
                                "user explicitly asked for a region this large.",
                    })
            merged_cuts, quality_note = datalab_registry.merge_default_quality_cuts(
                inp.catalog, inp.table, inp.value_cuts
            )
            predicates = datalab_query_builders.build_catalog_predicates(
                inp.catalog, inp.table, color_cut=inp.color_cut, value_cuts=merged_cuts,
                morphology=inp.morphology,
            )
            # An equality cut on a registry-indexed column (SMASH fieldid = 169)
            # bounds the aggregate on its own — the canonical way to map a whole
            # survey field without guessing a cone center (live P7).
            try:
                idx_cols = set(datalab_registry.indexed_bound_columns(inp.catalog, inp.table))
            except Exception:
                idx_cols = set()
            field_bound = any(
                str(vc.get("op", "")).strip() in ("=", "==")
                and str(vc.get("column", "")).strip().lower() in idx_cols
                for vc in (inp.value_cuts or [])
            )
            sql, meta = datalab_query_builders.build_density_aggregate(
                inp.catalog, inp.table, mode=inp.mode, step_deg=inp.step_deg,
                healpix_column=inp.healpix_column, ra=inp.ra, dec=inp.dec,
                radius_deg=inp.radius_deg, all_sky=inp.all_sky, predicates=predicates,
                limit=inp.limit, field_bound=field_bound,
            )
            if quality_note:
                meta.setdefault("warnings", []).append(quality_note)
            if inp.async_submit:
                # First-class job path: hand back a job_id up front; skips the
                # sync attempt AND the tiling fallback entirely.
                return execute_datalab_sql(
                    sql, meta, tool_name="datalab_density_aggregate", ctx=ctx, async_submit=True
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
                    owner_id=(str(ctx.user_id) if ctx.user_id else None),  # dl-export-owner-gap
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
    description = (
        "Lomb-Scargle period search + phase-fold of a stored Data Lab result_id. "
        "Default search window is periods 0.1-1.0 d (min/max_frequency 1-10 per day) — "
        "widen it for longer periods. Output includes period_significant and warnings: "
        "when it says no_significant_period, report exactly that. Mixed-band light "
        "curves auto-fold only the best-sampled band."
    )
    category = "datalab"
    InputModel = PeriodFoldInput
    annotations = {"read_only": True, "cost": "moderate", "produces": "image"}

    def run(self, inp, ctx) -> ToolResult:
        return _run_analysis_plot("period_fold", inp.model_dump(), ctx)


class SedPlotInput(_In):
    result_id: str
    # None (not 0) so the service can tell "no selector" from "row 0": with no
    # selector at all, multi-row results render the sample overlay (two live
    # P10 runs proved the model won't pass sample_n on its own). Models also
    # send explicit "row_index": null alongside sample_n — must stay accepted.
    row_index: Optional[int] = None
    row_indices: Optional[List[int]] = None
    sample_n: Optional[int] = None
    filter_columns: Optional[Dict[str, str]] = None
    title: str = "Data Lab SED"


class SedPlot(BaseCapability):
    name = "datalab_sed_plot"
    description = (
        "Render SED(s) from a stored Data Lab result_id using SVO FPS wavelengths. "
        "This tool IS multi-object: with no row selector it overlays up to 300 "
        "objects as faint lines with the per-band median highlighted (sample_n=N "
        "or row_indices control the sample). For SEDs of a sample, call it ONCE "
        'like {"result_id": "dlr_..."} or {"result_id": "dlr_...", "sample_n": 300} '
        "— never loop per row. Pass row_index=N only for ONE specific object."
    )
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


class SiaSearchInput(_In):
    ra: Optional[float] = None
    dec: Optional[float] = None
    target_name: Optional[str] = None
    fov_deg: float = 0.1
    band: Optional[str] = None
    catalog: Optional[str] = None
    endpoint: Optional[str] = None
    limit: int = 100


# Preferred inventory columns, shown in preview order when present. The full
# SIA row set (incl. access_url) is kept in the stored result.
_SIA_DISPLAY_COLUMNS = [
    "obs_bandpass", "exptime", "proctype", "prodtype",
    "instrument", "telescope", "mjd_obs", "date_obs", "survey",
]


class SiaSearch(BaseCapability):
    """Surface the Data Lab SIA image INVENTORY (Data Lab parity 2026-07):
    the search that previously ran only as an internal step of the cutout /
    color-image tools, now returned as a browsable table so a user can see
    what imaging exists at a position (bands, depths, epochs) and pick a
    specific row before pulling a cutout."""

    name = "datalab_sia_search"
    description = (
        "List the Data Lab SIA image inventory covering a position: bands, exposure "
        "times, proc/prod types, and access URLs, stored under a result_id. Use this to "
        "see what imaging exists (and how deep) BEFORE datalab_image_cutout / "
        "datalab_color_image; fetch rows with datalab_get_result."
    )
    category = "datalab"
    InputModel = SiaSearchInput
    annotations = {"read_only": True, "cost": "moderate"}

    def run(self, inp, ctx) -> ToolResult:
        service = ctx.service("datalab_image_service")
        try:
            import pandas as pd

            store = ctx.result_store
            if store is None:
                raise RuntimeError(
                    "Data Lab result store is unavailable; cannot persist the SIA "
                    "inventory result_id. Not issuing the remote query."
                )
            ra_f, dec_f, label = _resolve_coords(inp.target_name, inp.ra, inp.dec, ctx)
            result = service.search(
                ra_f, dec_f, float(inp.fov_deg),
                catalog=inp.catalog, endpoint=inp.endpoint,
            )
            rows = list(result.get("rows") or [])
            warnings: List[str] = []
            if inp.band:
                want = str(inp.band).strip().lower()
                rows = [
                    row for row in rows
                    if str(row.get("obs_bandpass") or "").strip().lower().startswith(want)
                ]
            limit = max(1, min(int(inp.limit or 100), 1000))
            upstream_total = len(rows)
            if len(rows) > limit:
                warnings.append(f"SIA returned {len(rows)} rows; keeping the first {limit}.")
                rows = rows[:limit]
            frame = pd.DataFrame(rows)
            # The stored frame is itself a truncated slice of the remote result
            # — record that STRUCTURALLY, not just as a warning string, so no
            # card downstream can offer it as a complete dataset (f2-CX-21).
            _upstream_flags = (
                {"upstream_truncated": True, "upstream_total": upstream_total}
                if upstream_total > len(rows) else {}
            )
            result_id = store.put(frame, {
                "tool_name": self.name,
                **({"owner_id": str(ctx.user_id)} if ctx.user_id else {}),  # dl-export-owner-gap (f2-CX-01)
                "provenance": result.get("provenance") or {},
                "warnings": warnings,
                **_upstream_flags,
            })
            display_cols = [c for c in _SIA_DISPLAY_COLUMNS if c in frame.columns]
            preview_rows, preview_more = _fit_rows(
                frame[display_cols] if display_cols else frame, 10, char_budget=3000
            )
            summary: Dict[str, Any] = {
                "success": True,
                "tool_name": self.name,
                "result_id": result_id,
                "rowcount": int(len(frame)),
                "coverage_gap": bool(result.get("coverage_gap")),
                "used_endpoint": result.get("used_endpoint"),
                "position": {"ra": ra_f, "dec": dec_f, "label": label, "fov_deg": float(inp.fov_deg)},
                "columns": [str(c) for c in frame.columns][:30],
                "preview": preview_rows,
                "preview_truncated": preview_more,
                "warnings": warnings,
                **_upstream_flags,  # (f2-CX-21)
                "note": (
                    "Inventory preview only (access URLs are in the stored rows); fetch up to "
                    "5000 rows with datalab_get_result(result_id), then render a chosen band/"
                    "position with datalab_image_cutout."
                ),
            }
            return ToolResult(
                success=True,
                result_id=result_id,
                warnings=warnings,
                provenance=Provenance(
                    service="datalab",
                    endpoint=str(result.get("used_endpoint") or ""),
                    tool_name=self.name,
                    rowcount=int(len(frame)),
                ),
                native=summary,
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
    split_threshold: Optional[float] = 0.003  # DES DR1 spread_model convention
    limit: Optional[int] = 3000
    title: Optional[str] = None
    point_sources: bool = False
    morphology: Optional[Dict[str, Any]] = None
    value_cuts: Optional[List[Dict[str, Any]]] = None
    x_expr: Optional[str] = None
    y_expr: Optional[str] = None


class ColorColorDiagram(BaseCapability):
    name = "datalab_color_color_diagram"
    description = (
        "One-shot color-color diagram for a Data Lab catalog cone (with morphology split). "
        "x_expr/y_expr switch both axes to derived expressions of real columns "
        "(single panel; e.g. x_expr='bp_rp')."
    )
    category = "datalab"
    InputModel = ColorColorDiagramInput
    annotations = {"read_only": True, "cost": "moderate", "produces": "image"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            table = inp.table or datalab_registry.default_table(inp.catalog)
            radius_deg = float(inp.radius_deg) if inp.radius_deg is not None else 0.5
            split_threshold = float(inp.split_threshold) if inp.split_threshold is not None else 0.003
            limit = int(inp.limit) if inp.limit is not None else 3000
            ra_f, dec_f, label = _resolve_coords(inp.target_name, inp.ra, inp.dec, ctx)
            out = datalab_orchestration.color_color_diagram(
                inp.catalog, table, ra_f, dec_f, radius_deg,
                x_bands=tuple(inp.x_bands) if inp.x_bands else ("g", "r"),
                y_bands=tuple(inp.y_bands) if inp.y_bands else ("r", "i"),
                split_col=inp.split_col, split_threshold=split_threshold, limit=limit,
                title=inp.title or f"{inp.catalog} color-color: {label}",
                point_sources=bool(inp.point_sources), morphology=inp.morphology,
                value_cuts=inp.value_cuts, x_expr=inp.x_expr, y_expr=inp.y_expr,
                owner_id=(str(ctx.user_id) if ctx.user_id else None),  # dl-export-owner-gap
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
    x_expr: Optional[str] = None
    y_expr: Optional[str] = None
    # WD-locus overlay + side-of-line counts (B3). Previously this field did
    # not exist, so extra="ignore" silently SWALLOWED the argument the agent
    # prompt rule steers models to pass — no locus drawn, no n_wd_candidates
    # to quote (found live 2026-07-18).
    overlay_locus: Optional[str] = None


class ColorMagnitudeDiagram(BaseCapability):
    name = "datalab_color_magnitude_diagram"
    description = (
        "One-shot color-magnitude (CMD/HR) diagram for a Data Lab catalog cone. "
        "For derived axes (e.g. a Gaia HR diagram) pass BOTH x_expr and y_expr — "
        "x_expr='bp_rp', y_expr='phot_g_mean_mag + 5*log10(parallax) - 10' — "
        "instead of band names; the y axis is rendered inverted."
    )
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
                value_cuts=inp.value_cuts, x_expr=inp.x_expr, y_expr=inp.y_expr,
                overlay_locus=inp.overlay_locus,
                owner_id=(str(ctx.user_id) if ctx.user_id else None),  # dl-export-owner-gap
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
    description = (
        "Poll the status of a Data Lab background job — local 'dlj_' ids (tiled "
        "search, local async queries) and real server-side job ids from async_submit."
    )
    category = "datalab"
    InputModel = JobIdInput
    annotations = {"read_only": True, "cost": "cheap"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            job_id = str(inp.job_id).strip()
            if job_id.startswith("dlj_"):
                out = {
                    "success": True,
                    **ctx.service("datalab_job_service").status(
                        job_id, **_job_owner_kwargs(ctx)
                    ),
                }
            else:
                # Server-side job (jobid minted by the Data Lab query manager;
                # also reachable from a sync-timeout fallback error message).
                if ctx.user_id:
                    try:
                        ctx.service("datalab_job_service").status(
                            job_id, **_job_owner_kwargs(ctx)
                        )
                    except KeyError:
                        # A fallback job the local service never saw (pre-fix
                        # runs, other workers). Server jobids are unguessable
                        # tokens; fall through to the live server status, and
                        # re-register the id for this owner so the Jobs panel
                        # and the update_external sync below work after a
                        # restart (dl-server-job-registry-restart-orphan).
                        try:
                            ctx.service("datalab_job_service").ensure_external(
                                job_id, **_job_owner_kwargs(ctx)
                            )
                        except Exception:  # noqa: BLE001 - re-registration is best-effort
                            pass
                client = _require_server_job_client(ctx)
                state = str(client.status(job_id)).strip().upper()
                out = {
                    "success": True,
                    "job_id": job_id,
                    "kind": "server_query",
                    "status": _SERVER_STATE_MAP.get(state, state.lower() or "unknown"),
                    "server_status": state,
                }
                if out["status"] == "succeeded":
                    out["note"] = "Fetch the rows with datalab_job_results(job_id)."
                elif out["status"] == "failed":
                    # Attach the server's actual error text so the model can
                    # adapt instead of guessing (live P15: bare "ERROR").
                    try:
                        out["error"] = str(client.error(job_id))[:400]
                    except Exception:  # noqa: BLE001 - reason fetch best-effort
                        pass
                try:
                    ctx.service("datalab_job_service").update_external(
                        job_id, status=out["status"], **_job_owner_kwargs(ctx)
                    )
                except Exception:
                    pass  # panel sync is best-effort; the live status is the answer
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
    description = (
        "Fetch the results of a background Data Lab job (local 'dlj_' or server-side "
        "id). Server-job rows are stored and returned as a result_id + preview."
    )
    category = "datalab"
    InputModel = JobIdInput
    annotations = {"read_only": True, "cost": "cheap"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            job_id = str(inp.job_id).strip()
            if job_id.startswith("dlj_"):
                record = ctx.service("datalab_job_service").results(
                    job_id, **_job_owner_kwargs(ctx)
                )
                # A failed job's results are not a success — the model must see
                # the failure, not a success wrapper around it. (guard CX-08)
                ok = str(record.get("status", "")).lower() != "failed"
                native = {"success": ok, **record}
                return ToolResult(
                    success=ok,
                    error=(None if ok else str(record.get("error") or "Data Lab job failed")),
                    native=native,
                )
            # Read the local record for its params (catalog/table/query/healpix)
            # and as the ownership pre-check. A KeyError here must NOT fail the
            # fetch: after a backend restart the in-memory registry is empty
            # while the job still exists server-side — the exact fall-through
            # JobStatus already has (CAP-02); re-register the id for this owner
            # instead (dl-server-job-registry-restart-orphan).
            record_params: Dict[str, Any] = {}
            try:
                record = ctx.service("datalab_job_service").status(
                    job_id, **_job_owner_kwargs(ctx)
                )
                record_params = dict(record.get("params") or {})
            except KeyError:
                if ctx.user_id:
                    try:
                        ctx.service("datalab_job_service").ensure_external(
                            job_id, **_job_owner_kwargs(ctx)
                        )
                    except Exception:  # noqa: BLE001 - re-registration is best-effort
                        pass
            except Exception:  # noqa: BLE001 - params enrichment is best-effort
                record_params = {}
            client = _require_server_job_client(ctx)
            store = ctx.result_store
            if store is None:
                raise RuntimeError(
                    "Data Lab result store is unavailable; cannot persist the job rows."
                )
            result = client.results(job_id)
            # Re-attach what the submit-time record knew: catalog/table/query
            # (client.results() pre-fills them with None) and the builder's
            # healpix {column, nside, scheme} so sky_density_map decodes async
            # aggregates with the right scheme (dl-async-healpix-scheme-lost).
            record_prov = {
                key: record_params[key]
                for key in ("catalog", "table", "healpix")
                if record_params.get(key) is not None
            }
            if record_params.get("sql"):
                record_prov["query"] = record_params["sql"]
            result_id = store.put(result.dataframe, {
                "tool_name": self.name,
                **({"owner_id": str(ctx.user_id)} if ctx.user_id else {}),  # dl-export-owner-gap
                "provenance": {**result.provenance, **record_prov, "jobid": job_id},
            })
            rows, truncated = _fit_rows(result.dataframe, 10, char_budget=4000)
            native = {
                "success": True,
                "job_id": job_id,
                "kind": "server_query",
                "status": "succeeded",
                "result_id": result_id,
                "rowcount": int(len(result.dataframe)),
                "columns": result.columns[:30],
                "preview": rows,
                "preview_truncated": truncated,
                "note": "Fetch up to 5000 rows with datalab_get_result(result_id).",
            }
            try:
                ctx.service("datalab_job_service").update_external(
                    job_id,
                    status="succeeded",
                    result={"result_id": result_id, "rowcount": native["rowcount"]},
                    **_job_owner_kwargs(ctx),
                )
            except Exception:
                pass
            return ToolResult(success=True, result_id=result_id, native=native)
        except Exception as e:
            return datalab_error(e)


class JobCancel(BaseCapability):
    name = "datalab_job_cancel"
    description = "Cancel a running background Data Lab job (local 'dlj_' or server-side id)."
    category = "datalab"
    InputModel = JobIdInput
    annotations = {"read_only": False, "cost": "cheap"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            job_id = str(inp.job_id).strip()
            if job_id.startswith("dlj_"):
                native = {
                    "success": True,
                    **ctx.service("datalab_job_service").cancel(
                        job_id, **_job_owner_kwargs(ctx)
                    ),
                }
                return ToolResult(success=True, native=native)
            if ctx.user_id:
                try:
                    ctx.service("datalab_job_service").status(
                        job_id, **_job_owner_kwargs(ctx)
                    )
                except KeyError:
                    # Same fall-through as JobStatus (CAP-02): a restart wipes
                    # the in-memory registry while the server job lives on;
                    # server jobids are unguessable tokens. Re-register for
                    # this owner so the panel reflects the cancel.
                    try:
                        ctx.service("datalab_job_service").ensure_external(
                            job_id, **_job_owner_kwargs(ctx)
                        )
                    except Exception:  # noqa: BLE001 - re-registration is best-effort
                        pass
            client = _require_server_job_client(ctx)
            client.abort(job_id)
            native = {"success": True, "job_id": job_id, "kind": "server_query", "status": "canceled"}
            try:
                ctx.service("datalab_job_service").update_external(
                    job_id, status="canceled", **_job_owner_kwargs(ctx)
                )
            except Exception:
                pass
            return ToolResult(success=True, native=native)
        except Exception as e:
            return datalab_error(e)


# ─────────────────────────────────────────────────────────────────────────────
# CDS X-Match: user list (or stored result) vs VizieR catalogs / SIMBAD
# ─────────────────────────────────────────────────────────────────────────────
class XmatchUserListInput(_In):
    catalog: str
    objects: Optional[List[Dict[str, Any]]] = None
    result_id: Optional[str] = None
    ra_column: str = "ra"
    dec_column: str = "dec"
    radius_arcsec: float = 5.0
    selection: str = "best"


class XmatchUserList(BaseCapability):
    name = "xmatch_user_list"
    description = (
        "Crossmatch a user object list (ra/dec dicts) OR a stored Data Lab result_id "
        "against a VizieR catalog or SIMBAD via the CDS X-Match service; returns a "
        "result_id with every uploaded column plus the match columns (angDist arcsec)."
    )
    category = "datalab"
    InputModel = XmatchUserListInput
    annotations = {"read_only": True, "cost": "moderate"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            from services import cds_xmatch

            store = ctx.result_store
            if store is None:
                raise RuntimeError("Data Lab result store is unavailable; cannot persist the match table.")
            if bool(inp.objects) == bool(str(inp.result_id or "").strip()):
                raise ValueError("Provide exactly one of objects (list of {ra,dec,...}) or result_id.")
            source_provenance: Dict[str, Any] = {}
            if inp.objects:
                frame = cds_xmatch.objects_to_dataframe(inp.objects)
            else:
                source_result = store.get(str(inp.result_id).strip())
                frame = source_result.dataframe
                source_provenance = dict(source_result.provenance or {})
            out = cds_xmatch.xmatch_dataframe(
                frame,
                catalog=inp.catalog,
                ra_column=inp.ra_column,
                dec_column=inp.dec_column,
                radius_arcsec=inp.radius_arcsec,
                selection=inp.selection,
            )
            matches = out.pop("dataframe")
            # A LIMIT-truncated source is a spatially biased, storage-order
            # slice — the derived match table inherits that bias, so the
            # truncation stamp must survive into the NEW result's provenance
            # or downstream plots lose the TRUNCATED SAMPLE caption
            # (dl-xmatch-truncation-provenance-drop).
            source_truncated = bool(source_provenance.get("limit_truncated"))
            trunc_carryover = (
                {
                    "limit_truncated": True,
                    **(
                        {"row_limit": source_provenance["row_limit"]}
                        if source_provenance.get("row_limit") is not None
                        else {}
                    ),
                }
                if source_truncated
                else {}
            )
            result_id = store.put(matches, {
                "tool_name": self.name,
                **({"owner_id": str(ctx.user_id)} if ctx.user_id else {}),  # dl-export-owner-gap
                "provenance": {
                    "service": "cds_xmatch",
                    "endpoint": out["endpoint"],
                    "cat2": out["cat2"],
                    "radius_arcsec": out["radius_arcsec"],
                    "selection": out["selection"],
                    "uploaded_rows": out["uploaded_rows"],
                    **({"source_result_id": str(inp.result_id).strip()} if inp.result_id else {}),
                    **({"source_provenance": source_provenance} if source_provenance else {}),
                    **trunc_carryover,
                },
            })
            rows, truncated = _fit_rows(matches, 10, char_budget=4000)
            native = {
                "success": True,
                "result_id": result_id,
                "matched_rows": out["matched_rows"],
                "uploaded_rows": out["uploaded_rows"],
                "upload_truncated": out["upload_truncated"],
                **(
                    {
                        "warnings": [
                            "The SOURCE result was LIMIT-truncated (a storage-order, "
                            "spatially biased slice) — this match table inherits that "
                            "bias; do not present it as the full selection."
                        ]
                    }
                    if source_truncated
                    else {}
                ),
                "catalog": out["cat2"],
                "radius_arcsec": out["radius_arcsec"],
                "selection": out["selection"],
                "columns": [str(col) for col in matches.columns][:30],
                "preview": rows,
                "preview_truncated": truncated,
                "note": (
                    "angDist is the match separation in arcsec. Fetch up to 5000 rows with "
                    "datalab_get_result(result_id); save keepers with datalab_save_result."
                ),
                "citation": "Crossmatch by the CDS X-Match service (CDS, Strasbourg).",
            }
            return ToolResult(success=True, result_id=result_id, native=native)
        except Exception as e:
            return datalab_error(e)


# ─────────────────────────────────────────────────────────────────────────────
# My-tables (MyDB-lite): durable named tables on the result store
# ─────────────────────────────────────────────────────────────────────────────
class SaveResultInput(_In):
    result_id: str
    name: str
    description: Optional[str] = None


class MyTableNameInput(_In):
    name: str


class ListMyTablesInput(_In):
    pass


class SaveResult(BaseCapability):
    name = "datalab_save_result"
    description = (
        "Save a Data Lab result_id as a durable named table (MyDB-lite): unlike "
        "result_ids (1-hour TTL), saved tables survive restarts. Reload with "
        "datalab_load_my_table(name)."
    )
    category = "datalab"
    InputModel = SaveResultInput
    annotations = {"read_only": False, "cost": "cheap"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            store = ctx.result_store
            if store is None:
                raise RuntimeError("Data Lab result store is unavailable; cannot save the table.")
            entry = store.save_result(
                inp.result_id,
                inp.name,
                description=inp.description,
                **_store_user_kwargs(ctx),
            )
            native = {
                "success": True,
                "my_table": entry,
                "note": f"Saved as '{entry['name']}'. Reload anytime with datalab_load_my_table.",
            }
            return ToolResult(success=True, native=native)
        except Exception as e:
            return datalab_error(e)


class ListMyTables(BaseCapability):
    name = "datalab_list_my_tables"
    description = "List the user's saved Data Lab tables (names, row counts, provenance)."
    category = "datalab"
    InputModel = ListMyTablesInput
    annotations = {"read_only": True, "cost": "cheap"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            store = ctx.result_store
            if store is None:
                raise RuntimeError("Data Lab result store is unavailable.")
            tables = store.list_my_tables(**_store_user_kwargs(ctx))
            native = {"success": True, "my_tables": tables, "count": len(tables)}
            return ToolResult(success=True, native=native)
        except Exception as e:
            return datalab_error(e)


class LoadMyTable(BaseCapability):
    name = "datalab_load_my_table"
    description = (
        "Load a saved Data Lab table by name into a fresh result_id usable by every "
        "result_id-consuming tool (plots, crossmatch, get_result)."
    )
    category = "datalab"
    InputModel = MyTableNameInput
    annotations = {"read_only": True, "cost": "cheap"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            store = ctx.result_store
            if store is None:
                raise RuntimeError("Data Lab result store is unavailable.")
            result = store.load_my_table(inp.name, **_store_user_kwargs(ctx))
            result_id = store.put(result.dataframe, {
                "tool_name": self.name,
                **({"owner_id": str(ctx.user_id)} if ctx.user_id else {}),  # dl-export-owner-gap
                "provenance": dict(result.provenance),
            })
            rows, truncated = _fit_rows(result.dataframe, 10, char_budget=4000)
            native = {
                "success": True,
                "name": str(inp.name).strip().lower(),
                "result_id": result_id,
                "rowcount": int(len(result.dataframe)),
                "columns": result.columns[:30],
                "provenance": result.provenance,
                "preview": rows,
                "preview_truncated": truncated,
                "note": "Fetch up to 5000 rows with datalab_get_result(result_id).",
            }
            return ToolResult(success=True, result_id=result_id, native=native)
        except Exception as e:
            return datalab_error(e)


class DensityVettingInput(_In):
    # `radius_deg` is REQUIRED-but-nullable: the legacy signature took it as a
    # required positional (missing key → error) while the body null-coerced an
    # explicit JSON null to the documented 0.5 default. `Optional[float]` with
    # NO default mirrors that exactly. The other optionals carry the legacy
    # defaults and are null-coerced in run(), line-for-line with the inline
    # method. (docs/v2 P1)
    catalog: str
    table: str
    radius_deg: Optional[float]
    ra: Optional[float] = None
    dec: Optional[float] = None
    target_name: Optional[str] = None
    step_deg: Optional[float] = 0.05
    color_cut: Optional[Dict[str, Any]] = None
    value_cuts: Optional[List[Dict[str, Any]]] = None
    morphology: Optional[Dict[str, Any]] = None
    top_n: Optional[int] = 5
    fov_deg: Optional[float] = 0.05
    band: Optional[str] = "g"
    confirm: Any = False  # `Any`, not bool — HITL wide-area gate (see all_sky notes)


class DensityVetting(BaseCapability):
    name = "datalab_density_vetting"
    description = (
        "P12: find the densest catalog cells within a cone (with optional "
        "color/magnitude/morphology cuts) and pull a SIA cutout grid of the "
        "top-N densest locations to eyeball."
    )
    category = "datalab"
    InputModel = DensityVettingInput
    annotations = {"read_only": True, "cost": "expensive", "produces": "image"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            # Null-coerce optional numerics: models sometimes send explicit
            # "step_deg": null / "fov_deg": null / "top_n": null, which would
            # crash on float(None)/int(None). Fall back to the documented defaults
            # (mirrors the color-magnitude/color-color handlers).
            radius_deg = 0.5 if inp.radius_deg is None else float(inp.radius_deg)
            step_deg = 0.05 if inp.step_deg is None else float(inp.step_deg)
            top_n = 5 if inp.top_n is None else int(inp.top_n)
            fov_deg = 0.05 if inp.fov_deg is None else float(inp.fov_deg)
            band = inp.band or "g"
            ra_f, dec_f, label = _resolve_coords(inp.target_name, inp.ra, inp.dec, ctx)
            # HITL wide-area gate: density_vetting accepted radius 30° (~2,800
            # deg²) with no confirmation and burned three async-ERROR scans
            # (live P15).
            if not inp.confirm:
                decision = datalab_orchestration.confirm_cone_area(ra_f, dec_f, radius_deg)
                if decision.get("needs_confirmation"):
                    return ToolResult(success=False, error=decision.get("message"), native={
                        "success": False,
                        "needs_confirmation": True,
                        **decision,
                        "hint": "Re-call datalab_density_vetting with confirm=true only if the "
                                "user explicitly asked for a region this large; otherwise shrink "
                                "the radius (≤2.5°) or ask the user.",
                    })
            out = datalab_orchestration.density_then_cutouts(
                inp.catalog, inp.table, ra_f, dec_f, radius_deg, step_deg=step_deg,
                color_cut=inp.color_cut, value_cuts=inp.value_cuts, morphology=inp.morphology,
                top_n=top_n, fov_deg=fov_deg, band=band,
                client=ctx.service("datalab_client"),
                result_store=ctx.result_store,
                image_service=ctx.service("datalab_image_service"),
                owner_id=(str(ctx.user_id) if ctx.user_id else None),  # dl-export-owner-gap
            )
            out["target"] = label
            # The cutout grid must reach the UI as an image card (2026-07-04 live
            # test: the base64 grid stayed buried in the tool output, the chat
            # showed nothing, and the answer still told the user to "eyeball the
            # cutouts above"). The image here is NESTED in out["cutout_grid"],
            # not the top-level result, so the capability only MARKS the grid
            # with the computed private _caption — under exactly the legacy
            # attach condition — and the agent's image wrapper
            # (_datalab_image_tool_fn(..., nested_key="cutout_grid")) pops the
            # mark, sets the UI card, and strips the base64. Transport stays at
            # the adapter boundary. (docs/v2 P1)
            grid = out.get("cutout_grid")
            if isinstance(grid, dict) and (grid.get("image_base64") or grid.get("path")):
                grid = dict(grid)
                grid.setdefault("success", True)
                grid["_caption"] = f"Density-peak cutout grid: {label} (top {int(top_n)})"
                out["cutout_grid"] = grid
            return ToolResult(success=bool(out.get("success")), native=out)
        except Exception as e:
            return datalab_error(e)


class TiledSearchInput(_In):
    catalog: str
    table: str
    ra_min: float
    ra_max: float
    dec_min: float
    dec_max: float
    # Optionals carry the legacy defaults but stay nullable: an explicit JSON
    # null flows into the same float()/int() conversions the inline method ran
    # inside its try, so a null maps to the identical typed error (via
    # datalab_error), not a differently-worded pydantic one. (docs/v2 P1)
    tile_radius_deg: Optional[float] = 2.0
    step_deg: Optional[float] = 0.05
    color_cut: Optional[Dict[str, Any]] = None
    value_cuts: Optional[List[Dict[str, Any]]] = None
    morphology: Optional[Dict[str, Any]] = None
    peak_threshold: Optional[float] = 3.0
    max_tiles: Optional[int] = 64
    candidate_budget: Optional[int] = 50
    # `Any` (not bool): the legacy HITL gate is a bare truthiness check
    # (`decision["needs_confirmation"] and not confirm`), so a stringified
    # "false"/"0" from the model counted as CONFIRMED and ran the scan. A bool
    # field would coerce those to False and bounce the scan back for
    # confirmation. Same hazard, same fix as SqlQueryInput.expert_ack and
    # DensityAggregateInput.all_sky above. Parity preserved. (docs/v2 P1)
    confirm: Any = False


class TiledSearch(BaseCapability):
    name = "datalab_tiled_search"
    description = (
        "P15: tiled region-bounded overdensity search over a footprint. Runs a "
        "server-side density aggregate per q3c cone tile, finds matched-filter "
        "peaks, and ranks candidates. Executes as a background job; for a large "
        "area it returns needs_confirmation first — re-call with confirm=true "
        "after confirming the sky area with the user."
    )
    category = "datalab"
    InputModel = TiledSearchInput
    annotations = {"read_only": True, "cost": "expensive"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            fp = {"ra_min": inp.ra_min, "ra_max": inp.ra_max, "dec_min": inp.dec_min, "dec_max": inp.dec_max}
            decision = datalab_orchestration.confirm_sky_area(
                fp, float(inp.tile_radius_deg), max_tiles=int(inp.max_tiles)
            )
            # HITL gate: don't start a wide scan until the user confirms the area.
            if decision["needs_confirmation"] and not inp.confirm:
                native = {"success": False, "needs_confirmation": True, **decision,
                          "hint": "Re-call datalab_tiled_search with confirm=true to run the scan over this area."}
                return ToolResult(success=False, native=native)

            # Plain locals so the closure is self-contained, exactly like the
            # legacy inline closure (it runs later on the job service's worker
            # thread, long after this request's CallContext is gone). The scan
            # resolves its own default client/result store at execution time,
            # verbatim with the legacy job body.
            catalog, table = inp.catalog, inp.table
            tile_radius_deg, step_deg = inp.tile_radius_deg, inp.step_deg
            color_cut, value_cuts, morphology = inp.color_cut, inp.value_cuts, inp.morphology
            peak_threshold, max_tiles, candidate_budget = inp.peak_threshold, inp.max_tiles, inp.candidate_budget
            scan_owner = str(ctx.user_id) if ctx.user_id else None  # dl-export-owner-gap

            def _job(cancel_check):
                return datalab_orchestration.tiled_sky_scan(
                    catalog, table, fp, tile_radius_deg=float(tile_radius_deg), step_deg=float(step_deg),
                    color_cut=color_cut, value_cuts=value_cuts, morphology=morphology,
                    peak_threshold=float(peak_threshold), max_tiles=int(max_tiles),
                    candidate_budget=int(candidate_budget), confirm=True, cancel_check=cancel_check,
                    owner_id=scan_owner,
                )

            job_id = ctx.service("datalab_job_service").start(
                "tiled_sky_scan",
                _job,
                params={"catalog": catalog, "table": table, **fp},
                **_job_owner_kwargs(ctx),
            )
            native = {"success": True, "job_id": job_id, "status": "queued", **decision,
                      "note": "Tiled scan started; poll with datalab_job_status / datalab_job_results."}
            return ToolResult(success=True, native=native)
        except Exception as e:
            return datalab_error(e)


class ExportNotebookInput(_In):
    # All-optional (the legacy schema has required=[]). `title`/`sia_fov_deg`
    # carry the legacy defaults but stay nullable: an explicit JSON null is
    # forwarded verbatim — the inline method applied no null-coercion here.
    title: Optional[str] = "NOIRLab Data Lab analysis"
    sql: Optional[str] = None
    catalog: Optional[str] = None
    table: Optional[str] = None
    sia_ra: Optional[float] = None
    sia_dec: Optional[float] = None
    sia_fov_deg: Optional[float] = 0.1
    sia_endpoint: Optional[str] = None
    svo_filters: Optional[List[str]] = None


class ExportNotebook(BaseCapability):
    name = "datalab_export_notebook"
    description = (
        "Export a reproducible Jupyter notebook for a Data Lab analysis: the "
        "governed TAP SQL (qc.query(sql=...)), an optional SIA cutout recipe, an "
        "optional SVO filter-wavelength lookup, and a data-citation cell."
    )
    category = "datalab"
    InputModel = ExportNotebookInput
    annotations = {"read_only": True, "cost": "cheap"}

    def run(self, inp, ctx) -> ToolResult:
        # NB: the injected notebook generator (the agent's bound
        # _generate_notebook) sets last_run_result to the notebook card — the
        # deliverable — so this tool registers with clear_card=False and the
        # generator arrives as a service instead of being reimplemented here.
        # The card write happens inside the agent-owned callable, never in this
        # capability. (docs/v2 P1)
        try:
            from services.notebook_gen import datalab_notebook_steps
            citation = None
            if inp.catalog:
                try:
                    citation = datalab_registry.citation(inp.catalog)
                except Exception:
                    citation = None
            sia = None
            if inp.sia_ra is not None and inp.sia_dec is not None:
                sia = {"ra": inp.sia_ra, "dec": inp.sia_dec, "fov_deg": inp.sia_fov_deg, "endpoint": inp.sia_endpoint}
            steps = datalab_notebook_steps(
                sql=inp.sql, catalog=inp.catalog, table=inp.table, sia=sia,
                svo_filters=inp.svo_filters, citation=citation,
            )
            out = ctx.service("generate_notebook")(inp.title, steps)
            return ToolResult(
                success=bool(isinstance(out, dict) and out.get("success")),
                error=(out.get("error") if isinstance(out, dict) and not out.get("success") else None),
                native=out,
            )
        except Exception as e:
            return datalab_error(e)


class VariableCandidatesInput(_In):
    catalog: str = "smash_dr1"
    ra: Optional[float] = None
    dec: Optional[float] = None
    target_name: Optional[str] = None
    radius_deg: float = 0.2
    band: Optional[str] = None
    min_epochs: int = 10
    limit: int = 100


class VariableCandidates(BaseCapability):
    """The blog's 'select high-variability stars' step (Data Lab parity 2026-07):
    a per-object variability aggregate over a multi-epoch cone, ranked by the
    scatter-to-error significance ratio. Chains into datalab_star_lightcurve /
    datalab_period_fold."""

    name = "datalab_variable_candidates"
    description = (
        "Rank variable-star candidates in a multi-epoch catalog cone (SMASH source "
        "tables): per-object epoch count, mean magnitude, magnitude scatter, amplitude, "
        "and scatter/error significance, most-variable first. Each row carries "
        "dist_arcsec from the cone center and the output includes nearest_candidate — "
        "when the user gave an exact target position, analyze the NEAREST candidate, "
        "not the top-ranked one. Feed a candidate's id into datalab_star_lightcurve, "
        "then datalab_period_fold."
    )
    category = "datalab"
    InputModel = VariableCandidatesInput
    annotations = {"read_only": True, "cost": "moderate"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            ra, dec = inp.ra, inp.dec
            if (ra is None or dec is None) and inp.target_name:
                ra, dec, _label = _resolve_coords(inp.target_name, ra, dec, ctx)
            if ra is None or dec is None:
                raise ValueError("Provide ra/dec or a resolvable target_name.")
            sql, meta = datalab_query_builders.build_variability_rank(
                inp.catalog, ra=float(ra), dec=float(dec), radius_deg=inp.radius_deg,
                band=inp.band, min_epochs=inp.min_epochs, limit=inp.limit,
            )
        except Exception as e:
            return datalab_error(e)
        out = execute_datalab_sql(sql, meta, tool_name=self.name, ctx=ctx)
        # Rows are ranked by variability significance, and the preview shows only
        # the head — the star AT the user's position can be invisible to the
        # model (live P14: it folded the top-ranked variable 2.2' away instead of
        # the target, whose modest var_snr didn't even make the row cap). Surface
        # the nearest candidate explicitly; when nothing ranked lies within a few
        # arcsec, probe a 5" cone at the exact center so the target star always
        # appears.
        native = getattr(out, "native", None)
        if isinstance(native, dict) and native.get("success") and native.get("result_id"):
            try:
                frame = ctx.result_store.get(native["result_id"]).dataframe
                nearest_row = None
                if frame is not None and len(frame) and "dist_arcsec" in frame.columns:
                    nearest_row = frame.loc[[frame["dist_arcsec"].idxmin()]]
                    if float(nearest_row["dist_arcsec"].iloc[0]) > 5.0:
                        nearest_row = None
                if nearest_row is None and inp.radius_deg > 5.0 / 3600.0:
                    probe_sql, probe_meta = datalab_query_builders.build_variability_rank(
                        inp.catalog, ra=float(ra), dec=float(dec), radius_deg=5.0 / 3600.0,
                        band=inp.band, min_epochs=min(inp.min_epochs, 10), limit=1,
                    )
                    validated = datalab_sql_policy.validate(probe_sql, source="builder", meta=probe_meta)
                    probe = ctx.service("datalab_client").query(sql=validated.sql).dataframe
                    if probe is not None and len(probe):
                        nearest_row = probe.head(1)
                if nearest_row is not None:
                    nearest, _ = _fit_rows(nearest_row, 1)
                    if nearest:
                        native["nearest_candidate"] = nearest[0]
                        native["note"] = (
                            str(native.get("note") or "")
                            + " Rows are ranked most-variable first; nearest_candidate is the "
                            "variable closest to the query position — use ITS id when the "
                            "user asked about a specific star at these coordinates."
                        ).strip()
            except Exception:
                pass
        return out


class StarLightcurveInput(_In):
    catalog: str = "smash_dr1"
    source_id: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    limit: int = 500


class StarLightcurve(BaseCapability):
    """Multi-epoch light-curve retrieval for ONE star (wires the previously
    unregistered build_variable_star_select builder; Data Lab parity 2026-07)."""

    name = "datalab_star_lightcurve"
    description = (
        "Fetch the multi-epoch light curve (mjd, filter, cmag, cerr) of one star from a "
        "SMASH source table by source id or exact position; returns a result_id ready for "
        "datalab_period_fold. Use datalab_variable_candidates first to find good targets."
    )
    category = "datalab"
    InputModel = StarLightcurveInput
    annotations = {"read_only": True, "cost": "moderate"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            sql, meta = datalab_query_builders.build_variable_star_select(
                catalog=inp.catalog, source_id=inp.source_id,
                ra=inp.ra, dec=inp.dec, limit=inp.limit,
            )
        except Exception as e:
            return datalab_error(e)
        return execute_datalab_sql(sql, meta, tool_name=self.name, ctx=ctx)


# The migrated Data Lab family — complete. SQL/catalog core (incl. the
# auto-tiling density aggregate) + result_id plot tools + SIA imaging +
# orchestration/jobs + density vetting + tiled search + notebook export.
# No Data Lab tool remains inline in core/agent.py.
SQL_CAPABILITIES: List[BaseCapability] = [
    ListCatalogs(),
    DescribeTable(),
    ConeCount(),
    SelectCatalogRows(),
    Q3cCrossmatch(),
    SqlQuery(),
    GetResult(),
    DensityAggregate(),
    VariableCandidates(),
    StarLightcurve(),
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
    # Nested image: the grid card lives in out["cutout_grid"], so the agent
    # registers this one with nested_key="cutout_grid".
    DensityVetting(),
]

# Non-image orchestration/job capabilities — registered via the plain
# _datalab_tool_fn wrapper (no image transport).
MISC_CAPABILITIES: List[BaseCapability] = [
    ConfirmSkyArea(),
    SiaSearch(),
    JobStatus(),
    JobResults(),
    JobCancel(),
    TiledSearch(),
    SaveResult(),
    ListMyTables(),
    LoadMyTable(),
    XmatchUserList(),
    # Registered with clear_card=False: its notebook card is the deliverable.
    ExportNotebook(),
]

CAPABILITIES: List[BaseCapability] = SQL_CAPABILITIES + PLOT_CAPABILITIES + MISC_CAPABILITIES

__all__ = [
    "CAPABILITIES", "SQL_CAPABILITIES", "PLOT_CAPABILITIES", "MISC_CAPABILITIES",
    "execute_datalab_sql",
    "datalab_error",
    "fit_rows",
    "ListCatalogs", "DescribeTable", "ConeCount", "SelectCatalogRows",
    "Q3cCrossmatch", "SqlQuery", "GetResult", "DensityAggregate",
    "VariableCandidates", "StarLightcurve",
    "CatalogScatter", "SkyDensityMap", "PeriodFold", "SedPlot", "LssWedge",
    "ImageCutout", "ColorImage", "CutoutGrid",
    "ColorColorDiagram", "ColorMagnitudeDiagram",
    "ConfirmSkyArea", "SiaSearch", "JobStatus", "JobResults", "JobCancel",
    "SaveResult", "ListMyTables", "LoadMyTable", "XmatchUserList",
    "DensityVetting", "TiledSearch", "ExportNotebook",
]
