"""ESO Science Archive profile (TAP at archive.eso.org/tap_obs).

The dedicated search_eso_archive tool covers the common "what did VLT
instrument X observe here" question; raw ADQL goes through vo_adql_query.
Audited claims verified live on 2026-09-24 (scripts/audit_archive_profiles.py).
"""

from __future__ import annotations

from services.archive_profiles.schema import ArchiveProfile

_TAP = "https://archive.eso.org/tap_obs"
_ORION = "CIRCLE('ICRS',83.82,-5.39,0.05)"

PROFILE = ArchiveProfile.model_validate(
    {
        "archive": "eso",
        "aliases": ("eso_archive", "eso_science_archive", "vlt"),
        "description": (
            "ESO Science Archive: VLT/VLTI, La Silla and VISTA/VST data. Processed Phase 3 "
            "products in ivoa.ObsCore; raw frames in dbo.raw."
        ),
        "endpoints": [
            {
                "id": "eso_tap",
                "description": "ESO TAP (tap_obs): ivoa.ObsCore, dbo.raw, dbo.ssa.",
                "url": _TAP,
                "protocol": "tap",
            },
        ],
        "query_surfaces": [
            {
                "id": "instrument_search",
                "purpose": "Default: observations of a target by a named ESO instrument (MUSE, KMOS, X-Shooter, ...).",
                "tool": "search_eso_archive",
                "request_kind": "structured_args",
                "parameters": [
                    {"name": "radius_arcmin", "json_type": "number", "unit": "arcmin",
                     "description": "Search radius in arcminutes (default 1)."},
                ],
            },
            {
                "id": "adql",
                "purpose": "Custom ADQL on ivoa.ObsCore or dbo.raw (vo_adql_query with this access_url).",
                "tool": "vo_adql_query",
                "request_kind": "adql",
                "query_argument": "adql",
                "endpoint_ids": ["eso_tap"],
                "parameters": [
                    {"name": "max_rows", "json_type": "integer", "unit": "1",
                     "description": "Row cap for the sync result."},
                ],
            },
        ],
        "tables": {
            "ivoa.ObsCore": {
                "purpose": "Processed Phase 3 science products (reduced images, cubes, spectra, catalogs).",
                "grain": "one row per data product (dp_id); several products can share one obs_id",
                "ra_column": "s_ra",
                "dec_column": "s_dec",
                "columns": [
                    {"name": "dp_id", "dtype": "string", "role": "identifier",
                     "description": "Unique ESO product identifier."},
                    {"name": "obs_id", "dtype": "string", "role": "identifier",
                     "description": "Observation identifier; not unique per row."},
                    {"name": "obs_collection", "dtype": "string", "role": "category",
                     "description": "Phase 3 collection (survey or instrument pipeline stream)."},
                    {"name": "instrument_name", "dtype": "string", "role": "category",
                     "description": "Instrument, e.g. MUSE, XSHOOTER."},
                    {"name": "dataproduct_type", "dtype": "string", "role": "category",
                     "description": "image, cube, spectrum, measurements, ..."},
                    {"name": "s_ra", "dtype": "float", "role": "ra", "unit": "deg",
                     "description": "Product centre RA (ICRS)."},
                    {"name": "s_dec", "dtype": "float", "role": "dec", "unit": "deg",
                     "description": "Product centre Dec (ICRS)."},
                    {"name": "em_min", "dtype": "float", "role": "measurement", "unit": "m",
                     "description": "Minimum wavelength in METRES."},
                    {"name": "em_max", "dtype": "float", "role": "measurement", "unit": "m",
                     "description": "Maximum wavelength in METRES."},
                    {"name": "t_exptime", "dtype": "float", "role": "measurement", "unit": "s",
                     "description": "Total exposure time."},
                    {"name": "abmaglim", "dtype": "float", "role": "quality", "unit": "mag",
                     "description": "Limiting AB magnitude (imaging products)."},
                    {"name": "p3orig", "dtype": "string", "role": "provenance",
                     "description": "Phase 3 origin: IDP (ESO-internal pipeline) or EDP (external, community)."},
                    {"name": "access_url", "dtype": "string", "role": "provenance",
                     "description": "Product download URL."},
                ],
                "hints": {"endpoint_ids": ["eso_tap"]},
            },
            "dbo.raw": {
                "purpose": "Raw frames as they came off the telescope (not in ObsCore).",
                "grain": "one row per raw file (dp_id)",
                "ra_column": "ra",
                "dec_column": "dec",
                "columns": [
                    {"name": "dp_id", "dtype": "string", "role": "identifier",
                     "description": "Raw file identifier."},
                    {"name": "instrument", "dtype": "string", "role": "category",
                     "description": "Instrument name (column is instrument, not instrument_name)."},
                    {"name": "dp_cat", "dtype": "string", "role": "category",
                     "description": "SCIENCE, CALIB, ACQUISITION, ..."},
                    {"name": "object", "dtype": "string", "role": "category",
                     "description": "Target name as entered by the observer."},
                    {"name": "ra", "dtype": "float", "role": "ra", "unit": "deg",
                     "description": "Pointing RA."},
                    {"name": "dec", "dtype": "float", "role": "dec", "unit": "deg",
                     "description": "Pointing Dec."},
                    {"name": "exposure", "dtype": "float", "role": "measurement", "unit": "s",
                     "description": "Exposure time."},
                    {"name": "prog_id", "dtype": "string", "role": "identifier",
                     "description": "ESO programme ID."},
                ],
                "hints": {"endpoint_ids": ["eso_tap"]},
            },
        },
        "pitfalls": [
            {
                "id": "obscore_is_phase3_only",
                "summary": "ivoa.ObsCore holds only processed Phase 3 products (calib_level 2-4); raw frames live in dbo.raw (instrument, dp_cat, object, exposure)",
                "applies_to": [{"kind": "archive", "ref": "eso"}],
                "prompt_rank": 1,
                # Verified 2026-09-24: DISTINCT calib_level over ObsCore = {2, 3, 4}.
                "audit": {"expect": "empty", "endpoint_id": "eso_tap",
                          "adql": ("SELECT TOP 1 dp_id FROM ivoa.ObsCore WHERE calib_level < 2 "
                                   "OR calib_level > 4 OR calib_level IS NULL")},
            },
            {
                "id": "rows_are_products",
                "summary": "ObsCore rows are products (unique dp_id), and one obs_id can repeat: count DISTINCT obs_id for observations",
                "applies_to": [{"kind": "table", "ref": "ivoa.ObsCore"}],
                "prompt_rank": 2,
                "audit": {"expect": "nonempty", "endpoint_id": "eso_tap",
                          "adql": ("SELECT obs_id, COUNT(*) AS n FROM ivoa.ObsCore WHERE "
                                   f"1=CONTAINS(POINT('ICRS',s_ra,s_dec),{_ORION}) "
                                   "GROUP BY obs_id HAVING COUNT(*) > 1")},
            },
            {
                "id": "wavelengths_in_metres",
                "summary": "em_min/em_max are wavelengths in METRES (500 nm = 5e-7)",
                "applies_to": [{"kind": "column", "ref": "ivoa.ObsCore:em_min"},
                               {"kind": "column", "ref": "ivoa.ObsCore:em_max"}],
                "prompt_rank": 3,
                "audit": {"expect": "nonempty", "endpoint_id": "eso_tap",
                          "adql": ("SELECT TOP 1 dp_id FROM ivoa.ObsCore WHERE instrument_name = 'MUSE' "
                                   "AND em_min > 4e-7 AND em_max < 1e-6")},
            },
            {
                "id": "mixed_case_table_name",
                "summary": "the table is published as ivoa.ObsCore; lower-case ivoa.obscore also works on this service",
                "applies_to": [{"kind": "table", "ref": "ivoa.ObsCore"}],
                "audit": {"expect": "ok", "endpoint_id": "eso_tap",
                          "adql": "SELECT TOP 1 obs_id FROM ivoa.obscore"},
            },
            {
                "id": "footprint_intersects",
                "summary": "for coverage use 1=INTERSECTS(s_region, CIRCLE(...)); CONTAINS(POINT(s_ra,s_dec), ...) only tests the product centre",
                "applies_to": [{"kind": "table", "ref": "ivoa.ObsCore"}],
                "audit": {"expect": "nonempty", "endpoint_id": "eso_tap",
                          "adql": f"SELECT TOP 2 obs_id FROM ivoa.ObsCore WHERE 1=INTERSECTS(s_region,{_ORION})"},
            },
            {
                "id": "raw_column_names_differ",
                "summary": "dbo.raw uses instrument/object/exposure, not the ObsCore names instrument_name/target_name/t_exptime",
                "applies_to": [{"kind": "table", "ref": "dbo.raw"}],
                "error_triggers": ["dbo.raw"],
                "audit": {"expect": "columns_absent", "endpoint_id": "eso_tap", "table": "dbo.raw",
                          "columns": ["instrument_name", "t_exptime"]},
            },
            {
                "id": "raw_geometry_needs_filters",
                "summary": "an unfiltered CONTAINS/POINT cone on dbo.raw fails ('Latitude values must be between -90 and 90'): add instrument/dp_cat filters or use an ra/dec BETWEEN box",
                "detail": (
                    "Some dbo.raw rows carry out-of-range coordinates, and the server's geography "
                    "conversion aborts the whole query when the cone reaches them. Narrowing by "
                    "instrument and dp_cat first (as the raw golden example does) avoids it, as "
                    "does a plain ra/dec BETWEEN box."
                ),
                "applies_to": [{"kind": "table", "ref": "dbo.raw"}],
                "error_triggers": ["Latitude values must be"],
                "audit": {"expect": "manual",
                          "reason": ("The failing probe needs more than 60 s before the server errors, too "
                                     "slow for a routine audit. Observed 2026-09-24: an unfiltered "
                                     "CONTAINS cone at (83.8, -5.4, 0.1 deg) on dbo.raw returned "
                                     "'24201: Latitude values must be between -90 and 90'; the same "
                                     "cone with instrument/dp_cat filters succeeded.")},
            },
        ],
        "golden_examples": [
            {
                "id": "muse_on_target",
                "intent": "MUSE observations of NGC 1068.",
                "invocation": {
                    "tool": "search_eso_archive",
                    "arguments": {"target_name": "NGC 1068", "instrument": "MUSE"},
                },
            },
            {
                "id": "phase3_cubes_in_orion",
                "intent": "Reduced MUSE cubes overlapping the Orion Nebula core.",
                "invocation": {
                    "tool": "vo_adql_query",
                    "arguments": {
                        "access_url": _TAP,
                        "adql": ("SELECT dp_id, obs_id, instrument_name, t_exptime, access_url "
                                 "FROM ivoa.ObsCore WHERE instrument_name = 'MUSE' AND "
                                 f"dataproduct_type = 'cube' AND 1=INTERSECTS(s_region,{_ORION})"),
                        "max_rows": 50,
                    },
                },
                "request": {"kind": "adql", "argument": "adql"},
            },
            {
                "id": "raw_science_frames",
                "intent": "Raw X-Shooter science frames near SN 1987A.",
                "invocation": {
                    "tool": "vo_adql_query",
                    "arguments": {
                        "access_url": _TAP,
                        "adql": ("SELECT dp_id, object, exposure, prog_id FROM dbo.raw "
                                 "WHERE instrument = 'XSHOOTER' AND dp_cat = 'SCIENCE' AND "
                                 "1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',83.8667,-69.2697,0.05))"),
                        "max_rows": 100,
                    },
                },
                "request": {"kind": "adql", "argument": "adql"},
                "note": "Raw frames are in dbo.raw with their own column names.",
            },
        ],
        "citations": [
            {"id": "cite_eso_programmatic", "text": "ESO Archive programmatic access (TAP)",
             "url": "https://archive.eso.org/programmatic/"},
        ],
    }
)
