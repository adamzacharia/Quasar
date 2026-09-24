"""ESA Gaia Archive profile (TAP at ESAC), queried through the generic vo_*
tools plus the dedicated gaia_distance tool.

Every pitfall with an ``audit`` was verified live on 2026-09-24 and is
re-checked by scripts/audit_archive_profiles.py. The ESA-vs-Data-Lab naming
trap (gaiadr3 vs gaia_dr3) is the one models hit most: Data Lab mirrors Gaia
under its own schema names, and browse_schema('datalab') documents that copy.
"""

from __future__ import annotations

from services.archive_profiles.schema import ArchiveProfile

_TAP = "https://gea.esac.esa.int/tap-server/tap"

PROFILE = ArchiveProfile.model_validate(
    {
        "archive": "gaia",
        "aliases": ("esa_gaia", "gaia_archive", "gea"),
        "description": (
            "ESA Gaia Archive (ESAC): the authoritative Gaia DR3/DR2 TAP service. Astrometry, "
            "photometry and radial velocities in gaiadr3.gaia_source; astrophysical parameters, "
            "variability and orbits in sibling gaiadr3.* tables joined on source_id."
        ),
        "endpoints": [
            {
                "id": "gaia_tap",
                "description": "ESA Gaia TAP (sync and async; ADQL geometry supported).",
                "url": _TAP,
                "protocol": "tap",
            },
        ],
        "query_surfaces": [
            {
                "id": "adql",
                "purpose": "ADQL against the ESA Gaia TAP service (vo_adql_query with this access_url; mode='auto' for heavy queries).",
                "tool": "vo_adql_query",
                "request_kind": "adql",
                "query_argument": "adql",
                "endpoint_ids": ["gaia_tap"],
                "parameters": [
                    {"name": "max_rows", "json_type": "integer", "unit": "1",
                     "description": "Row cap for the sync result."},
                    {"name": "mode", "json_type": "string",
                     "allowed_values": ["sync", "auto", "async"],
                     "description": "Use 'auto' or 'async' for large scans or joins."},
                ],
            },
            {
                "id": "distances",
                "purpose": "Bailer-Jones (2021) distances for DR3 sources near a position; prefer this over inverting parallax.",
                "tool": "gaia_distance",
                "request_kind": "structured_args",
                "parameters": [
                    {"name": "radius_arcsec", "json_type": "number", "unit": "arcsec",
                     "description": "Cone radius in arcseconds, capped at 300."},
                ],
            },
        ],
        "tables": {
            "gaiadr3.gaia_source": {
                "purpose": "Gaia DR3 main source catalogue: one row per source.",
                "grain": "one row per Gaia DR3 source (source_id)",
                "ra_column": "ra",
                "dec_column": "dec",
                "columns": [
                    {"name": "source_id", "dtype": "integer", "role": "identifier",
                     "unit_state": "not_applicable",
                     "description": "Unique source identifier; the join key to every other gaiadr3.* table."},
                    {"name": "ra", "dtype": "float", "role": "ra", "unit": "deg",
                     "description": "Right ascension (ICRS) at epoch ref_epoch."},
                    {"name": "dec", "dtype": "float", "role": "dec", "unit": "deg",
                     "description": "Declination (ICRS) at epoch ref_epoch."},
                    {"name": "ref_epoch", "dtype": "float", "role": "time", "unit": "yr",
                     "description": "Reference epoch of the positions: 2016.0 (Julian year) for DR3."},
                    {"name": "parallax", "dtype": "float", "role": "measurement", "unit": "mas",
                     "description": "Parallax in milliarcseconds; can be negative for faint stars."},
                    {"name": "parallax_error", "dtype": "float", "role": "uncertainty", "unit": "mas",
                     "description": "Parallax standard error."},
                    {"name": "pmra", "dtype": "float", "role": "measurement", "unit": "mas/yr",
                     "description": "Proper motion in RA times cos(dec)."},
                    {"name": "pmdec", "dtype": "float", "role": "measurement", "unit": "mas/yr",
                     "description": "Proper motion in declination."},
                    {"name": "ruwe", "dtype": "float", "role": "quality", "unit": "1",
                     "description": "Renormalised unit weight error; below about 1.4 indicates a well-behaved single-star astrometric solution."},
                    {"name": "phot_g_mean_mag", "dtype": "float", "role": "measurement", "unit": "mag",
                     "description": "G-band mean magnitude (Vega)."},
                    {"name": "bp_rp", "dtype": "float", "role": "measurement", "unit": "mag",
                     "description": "BP minus RP colour."},
                    {"name": "radial_velocity", "dtype": "float", "role": "measurement", "unit": "km/s",
                     "description": "Radial velocity; null for most sources (only bright stars have one)."},
                ],
                "hints": {"endpoint_ids": ["gaia_tap"]},
            },
        },
        "pitfalls": [
            {
                "id": "esa_vs_datalab_schema_names",
                "summary": "ESA schemas have no underscore: gaiadr3.gaia_source (gaia_dr3.gaia_source is Data Lab's copy and fails here as 'unresolved identifiers')",
                "detail": (
                    "The same Gaia release is published under different schema names by different "
                    "archives. On the ESA TAP service use gaiadr3.* / gaiadr2.*; the underscored "
                    "gaia_dr3.* names belong to NOIRLab Data Lab (see browse_schema('datalab'))."
                ),
                "applies_to": [{"kind": "archive", "ref": "gaia"}],
                "prompt_rank": 1,
                "error_triggers": ["gaia_dr3.", "gaia_edr3.", "gaia_dr2."],
                "audit": {"expect": "error", "endpoint_id": "gaia_tap",
                          "adql": "SELECT TOP 1 source_id FROM gaia_dr3.gaia_source",
                          "error_contains": "unresolved"},
            },
            {
                "id": "units_and_epoch",
                "summary": "parallax is mas and pm mas/yr; positions are at epoch J2016.0 (ref_epoch), so propagate proper motion before matching other-epoch catalogs",
                "applies_to": [{"kind": "table", "ref": "gaiadr3.gaia_source"}],
                "prompt_rank": 2,
                "audit": {"expect": "nonempty", "endpoint_id": "gaia_tap",
                          "adql": "SELECT TOP 1 source_id FROM gaiadr3.gaia_source WHERE ref_epoch = 2016.0"},
            },
            {
                "id": "distance_not_inverse_parallax",
                "summary": "do not invert noisy or negative parallaxes into distances: use gaia_distance (Bailer-Jones 2021, external.gaiaedr3_distance)",
                "applies_to": [{"kind": "column", "ref": "gaiadr3.gaia_source:parallax"}],
                "prompt_rank": 3,
                # Checks the archive-side half of the advice: the Bailer-Jones
                # distances gaia_distance relies on are published and populated.
                # "Do not invert parallax" itself is methodology, not archive state.
                "audit": {"expect": "nonempty", "endpoint_id": "gaia_tap",
                          "adql": "SELECT TOP 1 source_id FROM external.gaiaedr3_distance WHERE r_med_geo > 0"},
            },
            {
                "id": "ruwe_in_dr3_main_table",
                "summary": "ruwe is a column of gaiadr3.gaia_source (cut ruwe < 1.4 for clean astrometry)",
                "applies_to": [{"kind": "column", "ref": "gaiadr3.gaia_source:ruwe"}],
                "audit": {"expect": "columns_present", "endpoint_id": "gaia_tap",
                          "table": "gaiadr3.gaia_source", "columns": ["ruwe", "parallax", "pmra", "pmdec"]},
            },
            {
                "id": "dr2_has_no_ruwe_column",
                "summary": "gaiadr2.gaia_source has no ruwe column (DR2 published RUWE in a separate table); prefer DR3 unless DR2 is asked for",
                "applies_to": [{"kind": "archive", "ref": "gaia"}],
                "audit": {"expect": "columns_absent", "endpoint_id": "gaia_tap",
                          "table": "gaiadr2.gaia_source", "columns": ["ruwe"]},
            },
            {
                "id": "split_tables_join_on_source_id",
                "summary": "astrophysical parameters, variability and orbits are separate gaiadr3.* tables: JOIN them to gaia_source ON source_id",
                "applies_to": [{"kind": "archive", "ref": "gaia"}],
                "audit": {"expect": "nonempty", "endpoint_id": "gaia_tap",
                          "adql": "SELECT TOP 1 source_id FROM gaiadr3.astrophysical_parameters"},
            },
            {
                "id": "adql_geometry_supported",
                "summary": "standard ADQL cones work here: 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',ra0,dec0,r_deg))",
                "applies_to": [{"kind": "surface", "ref": "adql"}],
                "audit": {"expect": "nonempty", "endpoint_id": "gaia_tap",
                          "adql": ("SELECT TOP 3 source_id FROM gaiadr3.gaia_source WHERE "
                                   "1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',187.70593,12.39112,0.02))")},
            },
        ],
        "golden_examples": [
            {
                "id": "cone_bright_stars",
                "intent": "Gaia DR3 stars within 1 arcmin of M67's centre brighter than G = 15, with clean astrometry.",
                "invocation": {
                    "tool": "vo_adql_query",
                    "arguments": {
                        "access_url": _TAP,
                        "adql": ("SELECT source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag, ruwe "
                                 "FROM gaiadr3.gaia_source WHERE 1=CONTAINS(POINT('ICRS',ra,dec),"
                                 "CIRCLE('ICRS',132.846,11.814,0.0167)) AND phot_g_mean_mag < 15 AND ruwe < 1.4 "
                                 "ORDER BY phot_g_mean_mag"),
                        "max_rows": 200,
                    },
                },
                "request": {"kind": "adql", "argument": "adql"},
            },
            {
                "id": "join_astrophysical_parameters",
                "intent": "Effective temperatures from the DR3 astrophysical-parameters table for bright stars near a position.",
                "invocation": {
                    "tool": "vo_adql_query",
                    "arguments": {
                        "access_url": _TAP,
                        "adql": ("SELECT g.source_id, g.phot_g_mean_mag, a.teff_gspphot "
                                 "FROM gaiadr3.gaia_source AS g JOIN gaiadr3.astrophysical_parameters AS a "
                                 "ON g.source_id = a.source_id WHERE 1=CONTAINS(POINT('ICRS',g.ra,g.dec),"
                                 "CIRCLE('ICRS',56.75,24.12,0.1)) AND g.phot_g_mean_mag < 12"),
                        "max_rows": 50,
                        "mode": "auto",
                    },
                },
                "request": {"kind": "adql", "argument": "adql"},
                "note": "Split tables join on source_id; mode='auto' guards against a slow join.",
            },
            {
                "id": "distance_to_named_star",
                "intent": "Distance to Barnard's Star.",
                "invocation": {
                    "tool": "gaia_distance",
                    "arguments": {"target_name": "Barnard's Star"},
                },
            },
        ],
        "citations": [
            {"id": "cite_gaia_dr3", "text": "Gaia Collaboration, Vallenari et al. 2023, A&A 674, A1 (Gaia DR3)",
             "doi": "10.1051/0004-6361/202243940"},
            {"id": "cite_bj2021", "text": "Bailer-Jones et al. 2021, AJ 161, 147 (Gaia EDR3 distances)",
             "doi": "10.3847/1538-3881/abd806"},
        ],
    }
)
