"""
capabilities/schema.py — the uniform ``browse_schema`` capability (Feature 3).

One tool, every archive: ``browse_schema(archive, table?)`` returns the
curated ArchiveProfile projection for ADS/SIA/VO/Splatalogue/ALMA, and for
Data Lab wraps the existing registry (``describe_table`` / the live
TAP-schema tier) merged with the authored profile. Grounding stays
tool-mediated: the system prompt only carries a one-line-per-archive pointer
to this tool (gated by ``QUASAR_SCHEMA_GROUNDING``).

Transport-pure: no thread-locals, no env reads, no SSE — a stateless read
over code-defined profiles, so the CallContext needs no services.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict

from capabilities.base import BaseCapability, Provenance, ToolResult


class BrowseSchemaInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    archive: str
    table: Optional[str] = None


class BrowseSchema(BaseCapability):
    name = "browse_schema"
    description = (
        "Browse the curated schema profile of an archive BEFORE writing a query: "
        "canonical tables/columns with units, top pitfalls, and golden example tool calls. "
        "Archives: datalab, alma, ads_openalex, sia_hips, vo, splatalogue (aliases accepted). "
        "Pass table (e.g. 'des_dr1.main' or 'ivoa.obscore') for full column-level detail."
    )
    category = "archive"
    InputModel = BrowseSchemaInput
    annotations = {"read_only": True, "cost": "cheap"}
    # Hand-tuned schema: the pydantic-derived one renders Optional[str] as
    # anyOf[string,null], which no other registered tool uses and which some
    # provider tool-schema paths handle badly. Plain types only.
    json_schema = {
        "type": "object",
        "properties": {
            "archive": {
                "type": "string",
                "description": (
                    "Archive slug or alias: datalab, alma, ads_openalex (ads/openalex/papers), "
                    "sia_hips (hips), vo (vizier), splatalogue."
                ),
            },
            "table": {
                "type": "string",
                "description": (
                    "Optional table key for full column-level detail, e.g. 'des_dr1.main', "
                    "'gaia_dr3.gaia_source', 'ivoa.obscore'."
                ),
            },
        },
        "required": ["archive"],
    }

    def run(self, inp: BrowseSchemaInput, ctx) -> ToolResult:
        from services import archive_profiles
        from services.archive_profiles import projections

        try:
            profile = archive_profiles.get_profile(inp.archive)
        except KeyError as exc:
            return ToolResult.fail(str(exc))

        provenance = Provenance(
            service="archive_profiles",
            tool_name=self.name,
            query=f"browse_schema({inp.archive!r}"
                  + (f", {inp.table!r})" if inp.table else ")"),
            retrieved_at=getattr(ctx, "now", None),
        )

        if not inp.table:
            payload = projections.archive_projection(profile)
            if profile.archive == "datalab":
                payload["live_table_keys"] = self._datalab_live_keys()
                payload["live_hint"] = (
                    "live_table_keys resolve through the live TAP-schema CACHE "
                    "(schema_source='live_tap_schema'). If browsing one fails because the "
                    "cache is not yet populated, run any governed Data Lab query first "
                    "(the SQL path refreshes the cache), then retry."
                )
            return ToolResult(success=True, data=payload, provenance=provenance)

        if profile.tables is None:
            surfaces = ", ".join(s.tool for s in profile.query_surfaces)
            return ToolResult.fail(
                f"Archive '{profile.archive}' has no static tables — query through its "
                f"surfaces instead: {surfaces}. Call browse_schema('{profile.archive}') "
                "for parameters and pitfalls."
            )

        table_key = inp.table.strip()
        if profile.archive == "datalab":
            return self._datalab_table(profile, table_key, provenance)

        payload = projections.table_projection(profile, table_key)
        if payload is None:
            known = ", ".join(sorted(profile.tables.keys()))
            return ToolResult.fail(
                f"Unknown table {table_key!r} for archive '{profile.archive}'. "
                f"Curated tables: {known}"
            )
        return ToolResult(success=True, data=payload, provenance=provenance)

    # ── Data Lab: curated + live TAP-schema tiers ─────────────────────────
    @staticmethod
    def _datalab_live_keys() -> list:
        from services import datalab_registry

        return datalab_registry.expansion_qualified_tables()

    def _datalab_table(self, profile, table_key: str, provenance) -> ToolResult:
        from services import datalab_registry
        from services.archive_profiles import projections

        # Accept 'catalog.table' or a bare catalog (resolved to its default table).
        if "." in table_key:
            catalog, table = table_key.split(".", 1)
        else:
            catalog = table_key
            try:
                table = datalab_registry.default_table(catalog)
            except ValueError as exc:
                return ToolResult.fail(f"{exc} Curated tables: "
                                       + ", ".join(sorted(profile.tables.keys())))
        qualified = f"{catalog}.{table}"

        payload = projections.table_projection(profile, qualified)
        if payload is not None:
            return ToolResult(success=True, data=payload, provenance=provenance)

        # Live/expansion tier: columns come from the cached live TAP schema,
        # never from authored metadata (deliberately outside the curated gate).
        try:
            described = datalab_registry.describe_table(catalog, table)
        except ValueError as exc:
            known = ", ".join(sorted(profile.tables.keys()))
            live = ", ".join(datalab_registry.expansion_qualified_tables())
            # NOTE: datalab_describe_table reads the SAME live cache, so it is
            # not a fallback for an uncached expansion table — the cache is
            # populated by the Data Lab SQL path (guard CX-08).
            return ToolResult.fail(
                f"{exc} Curated tables: {known}. Live-schema tables: {live} — these need "
                "the TAP-schema cache; if one fails to resolve, run any governed Data Lab "
                "query first (the SQL path refreshes the cache), then retry. "
                "datalab_list_catalogs lists everything without needing the cache."
            )
        payload = {
            "kind": "table",
            "archive": "datalab",
            "table": described.get("qualified_name", qualified),
            "schema_source": "live_tap_schema",
            "scope_note": profile.scope_note,
            **{k: described[k] for k in (
                "columns", "ra_column", "dec_column", "region_strategy",
                "healpix_columns", "morphology", "bitmasks", "aggregate_safe",
                "footprint", "citation",
            ) if k in described},
            "hint": (
                "Column list read from the live Data Lab TAP schema (not curated) — "
                "units/dtypes may be absent; verify with datalab_describe_table if unsure."
            ),
        }
        # Live tables can carry hundreds of columns — same slice byte budget
        # as curated tables, trailing columns dropped honestly (guard CX-03).
        payload = projections.live_table_projection(payload)
        return ToolResult(success=True, data=payload, provenance=provenance)


CAPABILITIES = [BrowseSchema()]
