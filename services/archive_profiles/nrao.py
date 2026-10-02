"""NRAO Science Data Archive profile (VLA / EVLA / VLBA / GBT; TAP at data-query.nrao.edu).

Why it exists: on the 2026-10-01 MANNA-evals UI run, prompts that said "NRAO
obscore" were sent to almascience.nrao.edu (the ALMA North American mirror,
which has no VLA/VLBA/GBT data) with ivoa.obscore, LOWER() and bare sync
reads, the exact pitfalls below. The NRAO endpoint was only known inside the
VLA helpers in integrations/tap.py.

Sources: MANNA archives/nrao.py (NSF-Simons CosmicAI, @3204ba8; its probes
ran live 2026-07-16 and 2026-09-10) and integrations/tap.py
(NRAO_TAP_HOST, FACILITY_INSTRUMENTS). data-query.nrao.edu timed out from
both this machine and the TACC VM on 2026-10-01/02, so none of these claims
could be re-probed here; scripts/audit_archive_profiles.py reports them as
UNREACHABLE (never as a pass) until the service answers again.
"""

from __future__ import annotations

from services.archive_profiles.schema import ArchiveProfile

_TAP = "https://data-query.nrao.edu/tap"
_CRAB = "CIRCLE('ICRS', 83.6331, 22.0145, 0.1)"
_3C273 = "CIRCLE('ICRS', 187.2779, 2.0524, 0.5)"

PROFILE = ArchiveProfile.model_validate(
    {
        "archive": "nrao",
        "aliases": ("nrao_archive", "vla", "evla", "jvla", "vlba", "gbt"),
        "description": (
            "NRAO Science Data Archive: VLA (historical and Karl G. Jansky VLA, instrument 'EVLA'), "
            "VLBA, GBT and GMVA observations, plus mirrored ALMA scan rows. ObsCore over TAP at "
            f"{_TAP}, table tap_schema.obscore (NOT ivoa.obscore). almascience.nrao.edu is the ALMA "
            "mirror, not this archive."
        ),
        "endpoints": [
            {
                "id": "nrao_tap",
                "description": (
                    "NRAO archive TAP (VOSI /availability and /tables work; /capabilities is a 404). "
                    "Use vo_adql_query with mode='auto' or 'async', then vo_tap_job."
                ),
                "url": _TAP,
                "protocol": "tap",
            },
        ],
        "query_surfaces": [
            {
                "id": "position_search",
                "purpose": "Default: VLA/VLBA/GBT observations at a sky position (outage-aware NRAO client).",
                "tool": "search_by_position",
                "request_kind": "structured_args",
                "parameters": [
                    {"name": "facility", "json_type": "string",
                     "allowed_values": ["VLA", "VLBA", "GBT"],
                     "description": "Pick the NRAO telescope; the default (ALMA) searches the ALMA archive instead."},
                ],
            },
            {
                "id": "adql",
                "purpose": "Custom ADQL on tap_schema.obscore (vo_adql_query with this access_url).",
                "tool": "vo_adql_query",
                "request_kind": "adql",
                "query_argument": "adql",
                "endpoint_ids": ["nrao_tap"],
                "parameters": [
                    {"name": "mode", "json_type": "string", "allowed_values": ["sync", "auto", "async"],
                     "description": "'auto' (sync, promoted to async on timeout) or 'async' for every obscore data read."},
                ],
            },
            {
                "id": "async_job",
                "purpose": "Poll an async NRAO job until COMPLETED, then read its rows.",
                "tool": "vo_tap_job",
                "request_kind": "structured_args",
                "parameters": [
                    {"name": "action", "json_type": "string", "allowed_values": ["status", "results", "abort"],
                     "description": "status until phase is COMPLETED, then results."},
                ],
            },
        ],
        "tables": {
            "tap_schema.obscore": {
                "purpose": "ObsCore-style metadata for every NRAO telescope (non-standard location).",
                "grain": "one row per SCAN, not per observation or execution block",
                "ra_column": "s_ra",
                "dec_column": "s_dec",
                "columns": [
                    {"name": "obs_id", "dtype": "string", "role": "identifier",
                     "description": "Observation identifier."},
                    {"name": "obs_publisher_did", "dtype": "string", "role": "identifier",
                     "description": "Publisher dataset id; GROUP BY it (or project_code) for per-observation counts."},
                    {"name": "project_code", "dtype": "string", "role": "category",
                     "description": "Proposal code, e.g. '13B-088' or 'VLASS3.2' (select VLASS with project_code LIKE 'VLASS%')."},
                    {"name": "instrument_name", "dtype": "string", "role": "category",
                     "description": "Case-sensitive: 'EVLA', 'VLA', 'VLBA', 'GBT', 'GMVA', 'ALMA' (mirrored rows)."},
                    {"name": "facility_name", "dtype": "string", "role": "category",
                     "description": "'NRAO' for every telescope except the mirrored ALMA rows ('ALMA')."},
                    {"name": "target_name", "dtype": "string", "role": "identifier",
                     "description": "Proposer string or radio designation (M87 -> '3C274'); VLASS uses packed J2000 names."},
                    {"name": "s_ra", "dtype": "float", "role": "ra", "unit": "deg",
                     "description": "Pointing RA (ICRS)."},
                    {"name": "s_dec", "dtype": "float", "role": "dec", "unit": "deg",
                     "description": "Pointing Dec (ICRS)."},
                    {"name": "t_min", "dtype": "float", "role": "time", "unit": "d",
                     "description": "Start time, MJD (ObsCore convention). ORDER BY t_min DESC for 'most recent'."},
                    {"name": "freq_min", "dtype": "float", "role": "measurement", "unit": "Hz",
                     "description": "NRAO extension: lowest frequency in Hz (about 1% off em_max)."},
                    {"name": "freq_max", "dtype": "float", "role": "measurement", "unit": "Hz",
                     "description": "NRAO extension: highest frequency in Hz."},
                    {"name": "configuration", "dtype": "string", "role": "category",
                     "description": "VLA array configuration (A/B/C/D and hybrids)."},
                    {"name": "access_url", "dtype": "string", "role": "provenance",
                     "description": "Execution-block link for the web Archive Access Tool (access_format is not a MIME type)."},
                ],
                "hints": {"endpoint_ids": ["nrao_tap"]},
            },
        },
        "pitfalls": [
            {
                "id": "obscore_at_tap_schema",
                "summary": "VLA/VLBA/GBT TAP is https://data-query.nrao.edu/tap; ObsCore there is tap_schema.obscore (ivoa.obscore does not exist)",
                "detail": (
                    "NRAO keeps ObsCore at the non-standard tap_schema.obscore; querying ivoa.obscore "
                    "fails with a bare 'table not found'. almascience.nrao.edu/tap is the ALMA North "
                    "American mirror: it serves ALMA only and must never stand in for the NRAO archive."
                ),
                "applies_to": [{"kind": "archive", "ref": "nrao"}],
                "prompt_rank": 1,
                "error_triggers": ["ivoa.obscore"],
                "audit": {"expect": "nonempty", "endpoint_id": "nrao_tap",
                          "adql": "SELECT table_name FROM tap_schema.tables WHERE table_name = 'tap_schema.obscore'"},
            },
            {
                "id": "no_string_functions",
                "summary": "LOWER(), UPPER(), ILIKE and || are rejected: match exact case, or OR together LIKE case variants",
                "detail": (
                    "NRAO's ADQL has no string functions ('Function [LOWER] is not found in TapSchema'). "
                    "For a case-insensitive name match write the variants out, e.g. (target_name LIKE '%M87%' "
                    "OR target_name LIKE '%m87%'), and remember radio designations (M87 is often '3C274'); a "
                    "cone on s_ra/s_dec is more reliable than any name match."
                ),
                "applies_to": [{"kind": "table", "ref": "tap_schema.obscore"}],
                "prompt_rank": 2,
                "error_triggers": ["LOWER(", "UPPER(", "ILIKE", "||"],
                "audit": {"expect": "error", "endpoint_id": "nrao_tap",
                          "adql": ("SELECT TOP 1 table_name FROM tap_schema.tables "
                                   "WHERE LOWER(table_name) = 'tap_schema.obscore'"),
                          "error_contains": "LOWER"},
            },
            {
                "id": "async_and_selective",
                "summary": "obscore data reads: mode='auto' or 'async' (sync times out) plus a selective WHERE (cone, instrument_name, project_code)",
                "detail": (
                    "Unfiltered reads fail even as SELECT TOP 1 * in sync (HTTP 200 with QUERY_STATUS=ERROR "
                    "after ~60 s), and unfiltered COUNT(*)/DISTINCT scans fail even async (~30 min). Any "
                    "selective WHERE completes async in 2 to 12 min; a CONTAINS cone on (s_ra, s_dec) is "
                    "fastest for positional asks. Do not add fake geometry to a non-positional ask. "
                    "Metadata queries on tap_schema.tables/columns are fast in sync."
                ),
                "applies_to": [{"kind": "table", "ref": "tap_schema.obscore"}],
                "prompt_rank": 3,
                "audit": {"expect": "manual",
                          "reason": ("Load-dependent timing (MANNA probes 2026-07-16 and 2026-09-10); the failing "
                                     "case takes ~30 min, so no single probe settles it.")},
            },
            {
                "id": "almascience_nrao_is_alma",
                "summary": "almascience.nrao.edu is the ALMA North American mirror: ALMA data only, never the NRAO VLA/VLBA/GBT archive",
                "applies_to": [{"kind": "archive", "ref": "nrao"}],
                "audit": {"expect": "manual",
                          "reason": "Host-role claim (ALMA regional mirror), not a TAP probe."},
            },
            {
                "id": "instrument_values",
                "summary": "instrument_name is case-sensitive: 'EVLA' (Jansky VLA), 'VLA' (historical), 'VLBA', 'GBT', plus 'GMVA' and mirrored 'ALMA'",
                "applies_to": [{"kind": "column", "ref": "tap_schema.obscore:instrument_name"}],
                "audit": {"expect": "columns_present", "endpoint_id": "nrao_tap",
                          "table": "tap_schema.obscore", "columns": ["instrument_name", "facility_name"]},
            },
            {
                "id": "rows_are_scans",
                "summary": "rows are scans: count observations with GROUP BY project_code or obs_publisher_did, and name the unit you report",
                "applies_to": [{"kind": "table", "ref": "tap_schema.obscore"}],
                "audit": {"expect": "manual", "reason": "Row-granularity claim about the data model."},
            },
            {
                "id": "radio_designations",
                "summary": "sources use radio names (M87 = 3C274, Cygnus A = 3C405, Cen A = NGC5128) and VLASS packed J2000 names: search by position",
                "applies_to": [{"kind": "column", "ref": "tap_schema.obscore:target_name"}],
                "audit": {"expect": "manual", "reason": "Naming convention across many rows."},
            },
            {
                "id": "no_dataproduct_subtype",
                "summary": "the standard ObsCore column dataproduct_subtype is absent from tap_schema.obscore",
                "applies_to": [{"kind": "table", "ref": "tap_schema.obscore"}],
                "audit": {"expect": "columns_absent", "endpoint_id": "nrao_tap",
                          "table": "tap_schema.obscore", "columns": ["dataproduct_subtype"]},
            },
            {
                "id": "outage_is_not_no_data",
                "summary": "if data-query.nrao.edu times out the archive is down: say so, never report 'no observations'",
                "applies_to": [{"kind": "archive", "ref": "nrao"}],
                "audit": {"expect": "manual",
                          "reason": "Behavioural rule; NRAO timed out from two networks on 2026-10-01/02."},
            },
        ],
        "golden_examples": [
            {
                "id": "vla_at_position",
                "intent": "VLA observations of the Crab Nebula (structured, outage-aware).",
                "invocation": {
                    "tool": "search_by_position",
                    "arguments": {"ra": 83.6331, "dec": 22.0145, "radius": 0.1, "facility": "VLA"},
                },
            },
            {
                "id": "evla_cone_async",
                "intent": "25 EVLA rows near the Crab Nebula.",
                "invocation": {
                    "tool": "vo_adql_query",
                    "arguments": {
                        "access_url": _TAP,
                        "adql": ("SELECT obs_id, project_code, target_name, instrument_name, t_min, s_ra, s_dec "
                                 "FROM tap_schema.obscore WHERE instrument_name = 'EVLA' AND "
                                 f"1=CONTAINS(POINT('ICRS', s_ra, s_dec), {_CRAB})"),
                        "max_rows": 25,
                        "mode": "auto",
                    },
                },
                "request": {"kind": "adql", "argument": "adql"},
            },
            {
                "id": "case_insensitive_name",
                "intent": "Observations whose target name matches 'm87' case-insensitively (no LOWER()).",
                "invocation": {
                    "tool": "vo_adql_query",
                    "arguments": {
                        "access_url": _TAP,
                        "adql": ("SELECT obs_id, target_name, instrument_name, t_min FROM tap_schema.obscore "
                                 "WHERE target_name LIKE '%M87%' OR target_name LIKE '%m87%' "
                                 "OR target_name = '3C274'"),
                        "max_rows": 10,
                        "mode": "async",
                    },
                },
                "request": {"kind": "adql", "argument": "adql"},
            },
            {
                "id": "instruments_in_a_cone",
                "intent": "Which instrument_name values appear (grouped over a 0.5 deg cone around 3C 273, not a full scan).",
                "invocation": {
                    "tool": "vo_adql_query",
                    "arguments": {
                        "access_url": _TAP,
                        "adql": ("SELECT instrument_name, COUNT(*) AS n_scans FROM tap_schema.obscore "
                                 f"WHERE 1=CONTAINS(POINT('ICRS', s_ra, s_dec), {_3C273}) GROUP BY instrument_name"),
                        "max_rows": 50,
                        "mode": "async",
                    },
                },
                "request": {"kind": "adql", "argument": "adql"},
            },
        ],
        "citations": [
            {"id": "cite_nrao_scripted", "text": "NRAO: scripted access to the NRAO archive (TAP)",
             "url": "https://science.nrao.edu/facilities/vla/archive/scripted-access-to-the-nrao-archive"},
        ],
    }
)
