"""Generic Virtual Observatory profile (non-tabular: tables=None).

VO tables are DYNAMIC — discovered per service — so this profile grounds the
discovery chain and ADQL discipline instead of a static table list:
vo_find_services -> vo_list_tables -> vo_describe_table -> vo_adql_query.
"""

from __future__ import annotations

from services.archive_profiles.schema import ArchiveProfile

PROFILE = ArchiveProfile.model_validate(
    {
        "archive": "vo",
        "aliases": ("virtual observatory", "tap", "vizier", "ivoa"),
        "description": (
            "Generic Virtual Observatory access for archives with no built-in client: registry "
            "discovery of TAP/SIA/SSA/cone services, live table/column introspection, and "
            "guarded SELECT-only ADQL."
        ),
        "endpoints": [
            {
                "id": "vizier_tap",
                "description": "VizieR TAP (the most common vo_* target; thousands of catalogs).",
                "url": "https://tapvizier.cds.unistra.fr/TAPVizieR/tap",
                "protocol": "tap",
            },
        ],
        "query_surfaces": [
            {
                "id": "find_services",
                "purpose": "Registry search for services by keyword/waveband — the entry point when no built-in tool covers an archive.",
                "tool": "vo_find_services",
                "request_kind": "structured_args",
                "parameters": [
                    {"name": "service_type", "json_type": "string",
                     "allowed_values": ["tap", "sia", "ssa", "scs"],
                     "description": "Optional service-type filter."},
                ],
            },
            {
                "id": "list_tables",
                "purpose": "List a TAP service's tables (keyword-filtered).",
                "tool": "vo_list_tables",
                "request_kind": "structured_args",
                "parameters": [
                    {"name": "keyword", "json_type": "string",
                     "description": "Substring filter — STRONGLY recommended on big services like VizieR."},
                ],
            },
            {
                "id": "describe_table",
                "purpose": "Column names/datatypes/units/UCDs of one table — ALWAYS call before writing ADQL.",
                "tool": "vo_describe_table",
                "request_kind": "structured_args",
            },
            {
                "id": "adql",
                "purpose": "Guarded SELECT-only ADQL against any TAP service.",
                "tool": "vo_adql_query",
                "request_kind": "adql",
                "query_argument": "adql",
                "parameters": [
                    {"name": "max_rows", "json_type": "integer", "unit": "1",
                     "description": "Row cap (service-side caps also apply)."},
                ],
            },
            {
                "id": "cone_search",
                "purpose": "Simple cone search against an SCS service.",
                "tool": "vo_cone_search",
                "request_kind": "structured_args",
                "parameters": [
                    {"name": "radius_deg", "json_type": "number", "unit": "deg",
                     "description": "Cone radius in degrees, capped at 5."},
                ],
            },
        ],
        "tables": None,
        "pitfalls": [
            {
                "id": "inspect_before_adql",
                "summary": "follow vo_find_services -> vo_list_tables -> vo_describe_table -> vo_adql_query; ALWAYS inspect the live schema before writing ADQL — VO tables are dynamic",
                "applies_to": [{"kind": "archive", "ref": "vo"}],
                "prompt_rank": 1,
            },
            {
                "id": "quote_odd_names",
                "summary": "double-quote table names containing '/' or '+' (VizieR names like \"VIII/100/gleamegc\"); queries are SELECT-only",
                "applies_to": [{"kind": "surface", "ref": "adql"}],
                "prompt_rank": 2,
            },
            {
                "id": "vizier_keyword",
                "summary": "always pass a keyword to vo_list_tables on big services like VizieR — unfiltered listings truncate uselessly",
                "applies_to": [{"kind": "surface", "ref": "list_tables"}],
            },
            {
                "id": "read_server_errors",
                "summary": "on ADQL errors the server's message is returned — read it and fix the query rather than retrying blindly",
                "applies_to": [{"kind": "surface", "ref": "adql"}],
            },
        ],
        "golden_examples": [
            {
                "id": "discover_gleam",
                "intent": "Find a TAP service holding the GLEAM radio survey.",
                "invocation": {
                    "tool": "vo_find_services",
                    "arguments": {"keywords": "GLEAM radio survey", "service_type": "tap"},
                },
            },
            {
                "id": "describe_before_query",
                "intent": "Inspect the GLEAM extragalactic catalog's columns before querying.",
                "invocation": {
                    "tool": "vo_describe_table",
                    "arguments": {
                        "access_url": "https://tapvizier.cds.unistra.fr/TAPVizieR/tap",
                        "table_name": "VIII/100/gleamegc",
                    },
                },
            },
            {
                "id": "bright_gleam_sources",
                "intent": "Bright GLEAM sources via ADQL (after describing the table).",
                "invocation": {
                    "tool": "vo_adql_query",
                    "arguments": {
                        "access_url": "https://tapvizier.cds.unistra.fr/TAPVizieR/tap",
                        "adql": 'SELECT RAJ2000, DEJ2000, Fintwide FROM "VIII/100/gleamegc" WHERE Fintwide > 1.0',
                        "max_rows": 100,
                    },
                },
                "request": {"kind": "adql", "argument": "adql"},
                "note": "Slash-containing VizieR table name double-quoted.",
            },
        ],
        "citations": [
            {"id": "cite_ivoa", "text": "IVOA Virtual Observatory protocols (TAP/SIA/SSA/SCS)",
             "url": "https://www.ivoa.net/"},
        ],
    }
)
