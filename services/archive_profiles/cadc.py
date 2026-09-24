"""Canadian Astronomy Data Centre profile (TAP at /argus, SIA 2.0 at /sia).

search_cadc_archive is the default target search; vo_adql_query covers custom
ADQL and vo_image_search the SIA service. Audited claims verified live on
2026-09-24 (scripts/audit_archive_profiles.py).
"""

from __future__ import annotations

from services.archive_profiles.schema import ArchiveProfile

_HOST = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca"
_TAP = f"{_HOST}/argus"
_ORION = "CIRCLE('ICRS',83.82,-5.39,0.05)"

PROFILE = ArchiveProfile.model_validate(
    {
        "archive": "cadc",
        "aliases": ("cadc_archive", "canadian_astronomy_data_centre"),
        "description": (
            "Canadian Astronomy Data Centre: multi-mission archive (JWST, HST, CFHT, Gemini, JCMT, "
            "TESS, VLASS, ...). CAOM2 tables plus an ivoa.ObsCore view over TAP; SIA 2.0 images."
        ),
        "endpoints": [
            {
                "id": "cadc_tap",
                "description": "CADC TAP at /argus (the old /tap path returns 404).",
                "url": _TAP,
                "protocol": "tap",
            },
            {
                "id": "cadc_sia2",
                "description": "CADC SIA 2.0 (vo_image_search archive='cadc'); access_url is a DataLink.",
                "url": f"{_HOST}/sia",
                "protocol": "sia",
            },
        ],
        "query_surfaces": [
            {
                "id": "target_search",
                "purpose": "Default: CADC observations of a named target across missions.",
                "tool": "search_cadc_archive",
                "request_kind": "structured_args",
                "parameters": [
                    {"name": "max_results", "json_type": "integer", "unit": "1",
                     "description": "Row cap (default 500)."},
                ],
            },
            {
                "id": "adql",
                "purpose": "Custom ADQL on ivoa.ObsCore or caom2.* (vo_adql_query with this access_url).",
                "tool": "vo_adql_query",
                "request_kind": "adql",
                "query_argument": "adql",
                "endpoint_ids": ["cadc_tap"],
                "parameters": [
                    {"name": "mode", "json_type": "string", "allowed_values": ["sync", "auto", "async"],
                     "description": "Use 'auto' for anything that is not tightly constrained."},
                ],
            },
            {
                "id": "images",
                "purpose": "Images and cubes covering a position (SIA 2.0).",
                "tool": "vo_image_search",
                "request_kind": "structured_args",
                "endpoint_ids": ["cadc_sia2"],
                "parameters": [
                    {"name": "radius_deg", "json_type": "number", "unit": "deg",
                     "description": "Search radius, capped at 2."},
                ],
            },
        ],
        "tables": {
            "ivoa.ObsCore": {
                "purpose": "IVOA ObsCore view over every CADC collection.",
                "grain": "one row per data product; obs_id can repeat across rows",
                "ra_column": "s_ra",
                "dec_column": "s_dec",
                "columns": [
                    {"name": "obs_collection", "dtype": "string", "role": "category",
                     "description": "Mission/collection: JWST, HST, CFHT, GEMINI, JCMT, TESS, VLASS, ..."},
                    {"name": "obs_id", "dtype": "string", "role": "identifier",
                     "description": "Observation identifier (not unique per row)."},
                    {"name": "instrument_name", "dtype": "string", "role": "category",
                     "description": "Instrument (e.g. NIRCam, WFC3, MegaPrime)."},
                    {"name": "dataproduct_type", "dtype": "string", "role": "category",
                     "description": "image, cube, spectrum, ..."},
                    {"name": "calib_level", "dtype": "integer", "role": "quality", "unit_state": "not_applicable",
                     "description": "ObsCore calibration level 0 (raw) .. 3 (science-ready) / 4 (analysis)."},
                    {"name": "s_ra", "dtype": "float", "role": "ra", "unit": "deg",
                     "description": "Product centre RA."},
                    {"name": "s_dec", "dtype": "float", "role": "dec", "unit": "deg",
                     "description": "Product centre Dec."},
                    {"name": "em_min", "dtype": "float", "role": "measurement", "unit": "m",
                     "description": "Minimum wavelength in METRES."},
                    {"name": "em_max", "dtype": "float", "role": "measurement", "unit": "m",
                     "description": "Maximum wavelength in METRES."},
                    {"name": "t_exptime", "dtype": "float", "role": "measurement", "unit": "s",
                     "description": "Exposure time."},
                    {"name": "access_url", "dtype": "string", "role": "provenance",
                     "description": "DataLink URL (a VOTable listing files), not the file itself."},
                ],
                "hints": {"endpoint_ids": ["cadc_tap"]},
            },
            "caom2.Observation": {
                "purpose": "CAOM2 observation table (native CADC model).",
                "grain": "one row per observation",
                "columns": [
                    {"name": "collection", "dtype": "string", "role": "category",
                     "description": "Mission/collection (ObsCore calls this obs_collection)."},
                    {"name": "observationID", "dtype": "string", "role": "identifier",
                     "description": "Observation identifier within the collection."},
                    {"name": "instrument_name", "dtype": "string", "role": "category",
                     "description": "Instrument name."},
                ],
                "hints": {"endpoint_ids": ["cadc_tap"]},
            },
        },
        "pitfalls": [
            {
                "id": "tap_lives_at_argus",
                "summary": "CADC TAP is at /argus (the old /tap path is a 404); ObsCore is ivoa.ObsCore, CAOM2 tables are caom2.*",
                "applies_to": [{"kind": "archive", "ref": "cadc"}],
                "prompt_rank": 1,
                "audit": {"expect": "ok", "endpoint_id": "cadc_tap",
                          "adql": "SELECT TOP 1 collection FROM caom2.Observation"},
            },
            {
                "id": "collection_column_per_table",
                "summary": "mission filter is obs_collection on ivoa.ObsCore but collection on caom2.Observation (obs_collection errors there)",
                "applies_to": [{"kind": "table", "ref": "caom2.Observation"}],
                "prompt_rank": 2,
                "error_triggers": ["caom2."],
                "audit": {"expect": "columns_absent", "endpoint_id": "cadc_tap",
                          "table": "caom2.Observation", "columns": ["obs_collection"]},
            },
            {
                "id": "constrain_every_query",
                "summary": "unconstrained DISTINCT/aggregates over ivoa.ObsCore time out: always constrain position or collection, and use mode='auto'",
                "applies_to": [{"kind": "table", "ref": "ivoa.ObsCore"}],
                "prompt_rank": 3,
                "audit": {"expect": "manual",
                          "reason": ("A timeout is not a single-probe check; a full-table DISTINCT "
                                     "timed out after 60 s on 2026-09-24.")},
            },
            {
                "id": "access_url_is_datalink",
                "summary": "ObsCore/SIA access_url is a DataLink VOTable; follow its #this row to reach the FITS file",
                "applies_to": [{"kind": "column", "ref": "ivoa.ObsCore:access_url"}],
                "audit": {"expect": "nonempty", "endpoint_id": "cadc_tap",
                          "adql": ("SELECT TOP 1 obs_id FROM ivoa.ObsCore WHERE obs_collection = 'JWST' "
                                   "AND access_format LIKE '%datalink%'")},
            },
            {
                "id": "wavelengths_in_metres",
                "summary": "em_min/em_max are wavelengths in METRES",
                "applies_to": [{"kind": "column", "ref": "ivoa.ObsCore:em_min"}],
                "audit": {"expect": "columns_present", "endpoint_id": "cadc_tap",
                          "table": "ivoa.ObsCore", "columns": ["em_min", "em_max", "t_exptime", "calib_level"]},
            },
            {
                "id": "footprint_intersects",
                "summary": "for coverage use 1=INTERSECTS(s_region, CIRCLE(...)); it finds products whose footprint overlaps, not only centred ones",
                "applies_to": [{"kind": "table", "ref": "ivoa.ObsCore"}],
                "audit": {"expect": "nonempty", "endpoint_id": "cadc_tap",
                          "adql": f"SELECT TOP 2 obs_id FROM ivoa.ObsCore WHERE 1=INTERSECTS(s_region,{_ORION})"},
            },
        ],
        "golden_examples": [
            {
                "id": "target_across_missions",
                "intent": "CADC observations of the Crab Nebula.",
                "invocation": {
                    "tool": "search_cadc_archive",
                    "arguments": {"target_name": "Crab Nebula"},
                },
            },
            {
                "id": "jwst_science_products",
                "intent": "Science-ready JWST products overlapping the Orion Nebula core.",
                "invocation": {
                    "tool": "vo_adql_query",
                    "arguments": {
                        "access_url": _TAP,
                        "adql": ("SELECT obs_id, instrument_name, dataproduct_type, em_min, em_max, access_url "
                                 "FROM ivoa.ObsCore WHERE obs_collection = 'JWST' AND calib_level >= 3 AND "
                                 f"1=INTERSECTS(s_region,{_ORION})"),
                        "max_rows": 100,
                        "mode": "auto",
                    },
                },
                "request": {"kind": "adql", "argument": "adql"},
            },
            {
                "id": "images_at_position",
                "intent": "JWST images and cubes covering M87's core.",
                "invocation": {
                    "tool": "vo_image_search",
                    "arguments": {"archive": "cadc", "target_name": "M87", "radius_deg": 0.02,
                                  "collection": "JWST"},
                },
            },
        ],
        "citations": [
            {"id": "cite_cadc", "text": "Canadian Astronomy Data Centre (CADC) TAP and SIA services",
             "url": "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/"},
        ],
    }
)
