"""ALMA Science Archive profile (ivoa.obscore via TAP + capability tools).

Canonical/common grounding only. Units are the LIVE ObsCore units (dated
column snapshot: tests/fixtures/alma_obscore_columns_2025-12.txt, cross-checked
against the vendored ALMA data skill, third_party/alma-data-skill, reviewed
2026-07-18): frequency GHz but **bandwidth Hz**; em_min/em_max wavelengths in
METERS; velocity_resolution m/s; t_min/t_max MJD; obs_release_date ISO;
access_estsize kbyte; s_resolution/spatial_resolution/spatial_scale_max arcsec.

Row grain: ivoa.obscore rows are COVERAGE RECORDS finer than a dataset
(repeated per execution block, per field/source, per spectral coverage).
Aggregate by member_ous_uid for datasets, COUNT(DISTINCT asdm_uid) for EBs.
"""

from __future__ import annotations

from services.archive_profiles.schema import ArchiveProfile

PROFILE = ArchiveProfile.model_validate(
    {
        "archive": "alma",
        "aliases": ("alma_archive", "almascience"),
        # Regional mirrors serve the same /tap, /sia2, /datalink paths.
        "mirror_hosts": ("almascience.org", "almascience.eso.org", "almascience.nao.ac.jp"),
        "description": (
            "ALMA Science Archive: interferometric observations searched by target/position/"
            "frequency, deterministic science-query templates, DataLink file inventories, "
            "and TAP/ADQL over ivoa.obscore."
        ),
        "endpoints": [
            {
                "id": "alma_tap",
                "description": "ALMA TAP service (ivoa.obscore), NRAO mirror. Stay on one mirror per request.",
                "url": "https://almascience.nrao.edu/tap",
                "protocol": "tap",
            },
            {
                "id": "alma_datalink",
                "description": "ALMA DataLink (per-MOUS file enumeration): <mirror>/datalink/sync?ID=<uid>.",
                "url": "https://almascience.nrao.edu/datalink/sync",
                "protocol": "rest",
            },
            {
                "id": "alma_sia2",
                "description": "ALMA SIA 2.0 image/cube search (vo_image_search archive='alma'); access_url is a MOUS DataLink.",
                "url": "https://almascience.nrao.edu/sia2",
                "protocol": "sia",
            },
        ],
        "query_surfaces": [
            {
                "id": "target_search",
                "purpose": "Default archive search by target name (multi-target via commas). Resolver + footprint cone.",
                "tool": "search_by_target",
                "request_kind": "structured_args",
                "parameters": [
                    {"name": "band", "json_type": "string",
                     "description": "ALMA band number(s) AS A STRING ('6' or '6,7'). Pass ONLY if the user explicitly asks. Matches band_list tokens, so band-to-band rows ('5 10') are kept."},
                    {"name": "max_resolution", "json_type": "number", "unit": "arcsec",
                     "description": "Maximum angular resolution. Pass ONLY if the user specifies one."},
                    {"name": "min_freq_ghz", "json_type": "number", "unit": "GHz",
                     "description": "Minimum frequency. Pass ONLY if the user specifies."},
                    {"name": "max_freq_ghz", "json_type": "number", "unit": "GHz",
                     "description": "Maximum frequency. Pass ONLY if the user specifies."},
                    {"name": "public_only", "json_type": "boolean",
                     "description": "Adds data_rights = 'Public'. Default false returns proprietary rows too (they are labelled)."},
                ],
                "endpoint_ids": ["alma_tap"],
            },
            {
                "id": "science_templates",
                "purpose": "Deterministic templates for hard archive-science questions (cycle counts, array combos, line sets, sensitivity-driven discovery, publication joins). Results report rows, MOUS and EB counts separately.",
                "tool": "query_alma_science_archive",
                "request_kind": "structured_args",
                "parameters": [
                    {"name": "query_type", "json_type": "string",
                     "allowed_values": ["cycle_solar_projects", "cycle_array_combo_projects",
                                         "high_resolution_band_data", "line_set_projects",
                                         "redshifted_line_projects", "bandwidth_switching_candidates",
                                         "sensitivity_search", "data_publications"],
                     "description": "Template to run — use these instead of hand-writing ADQL."},
                    {"name": "max_resolution_arcsec", "json_type": "number", "unit": "arcsec",
                     "description": "Resolution ceiling for high_resolution_band_data (no cut when omitted)."},
                    {"name": "sensitivity_mjy", "json_type": "number", "unit": "mJy/beam",
                     "description": "sensitivity_search: required rms — returns data with achieved sensitivity <= this."},
                    {"name": "identifier", "json_type": "string",
                     "description": "data_publications: project code, MOUS uid://..., or ADS bibcode (reverse paper->data)."},
                ],
                "endpoint_ids": ["alma_tap"],
            },
            {
                "id": "frequency_search",
                "purpose": "Search the archive by frequency range (em_min/em_max wavelength overlap; representative 'frequency' is NOT used).",
                "tool": "search_by_frequency",
                "request_kind": "structured_args",
                "parameters": [
                    {"name": "min_freq_ghz", "json_type": "number", "unit": "GHz",
                     "description": "Lower frequency bound (GHz, required)."},
                    {"name": "max_freq_ghz", "json_type": "number", "unit": "GHz",
                     "description": "Upper frequency bound (GHz, required)."},
                ],
                "endpoint_ids": ["alma_tap"],
            },
            {
                "id": "line_coverage",
                "purpose": "One named transition + target: resolve with Splatalogue and verify exact spectral-window coverage from frequency_support.",
                "tool": "find_alma_line_coverage",
                "request_kind": "structured_args",
            },
            {
                "id": "file_inventory",
                "purpose": "Enumerate a MOUS's DataLink rows (typed: files, service descriptors, nested DataLink, errors). Empty means 'no links visible under this authorization', not 'invalid UID'.",
                "tool": "list_alma_files",
                "request_kind": "structured_args",
                "parameters": [
                    {"name": "mous_uid", "json_type": "string",
                     "description": "Member OUS UID (uid://A001/Xnnn/Xnnn or uid___A001_Xnnn_Xnnn)."},
                ],
                "endpoint_ids": ["alma_datalink"],
            },
            {
                "id": "guidance",
                "purpose": "On-demand reference sections of the vendored ALMA data skill (units, footprints, DataLink packaging, QA2/restore, spectral frames, mosaics/TP, cycles, pipeline history).",
                "tool": "browse_alma_guidance",
                "request_kind": "structured_args",
            },
            {
                "id": "raw_adql",
                "purpose": "Escape hatch: raw SELECT-only ADQL against the ALMA TAP when no template fits.",
                "tool": "vo_adql_query",
                "request_kind": "adql",
                "query_argument": "adql",
                "endpoint_ids": ["alma_tap"],
            },
        ],
        "tables": {
            "ivoa.obscore": {
                "purpose": (
                    "Coverage records of ALMA observations: positions/footprints, bands, spectral "
                    "setup, resolution, sensitivity, project/MOUS/EB identifiers, access rights."
                ),
                "grain": (
                    "coverage record — repeated per execution block (asdm_uid), per field/source, "
                    "per spectral coverage; aggregate by member_ous_uid for datasets (MOUS), "
                    "COUNT(DISTINCT asdm_uid) for executions; rows are NOT observations"
                ),
                "ra_column": "s_ra",
                "dec_column": "s_dec",
                "columns": [
                    {"name": "target_name", "dtype": "string", "role": "identifier",
                     "description": "PI-entered target name — free text, inconsistent ('IC342'/'ic 342'/'IC342'). "
                                    "NOT for source selection: resolve the name and cone-search the footprint instead; "
                                    "use only for labelling, or as an explicitly-flagged string match (moving/solar targets)."},
                    {"name": "proposal_id", "dtype": "string", "role": "identifier",
                     "description": "ALMA project code <year>.<period>.<number>.<type>, e.g. 2019.1.00123.S; period 1 = main call, 2 = supplemental, a letter (A) = DDT."},
                    {"name": "member_ous_uid", "dtype": "string", "role": "identifier",
                     "description": "Member OUS UID (uid://...) — THE grouping key: the unit of calibration, QA2 and delivery. Aggregate rows by it."},
                    {"name": "group_ous_uid", "dtype": "string", "role": "identifier",
                     "description": "Parent Group OUS UID — sibling MOUSs (12-m/7-m/TP) meant for combination share it."},
                    {"name": "asdm_uid", "dtype": "string", "role": "identifier",
                     "description": "Execution-block (raw ASDM) UID; COUNT(DISTINCT asdm_uid) = number of executions. Identify EBs by this column, not by the A002 prefix."},
                    {"name": "schedblock_name", "dtype": "string", "role": "identifier",
                     "description": "Scheduling-block name; _TM1/_TM2, _7M, _TP suffixes are array HEURISTICS."},
                    {"name": "obs_publisher_did", "dtype": "string", "role": "identifier",
                     "description": "Unique dataset identifier."},
                    {"name": "s_ra", "dtype": "float", "unit": "deg", "role": "ra",
                     "description": "Representative ICRS right ascension — inadequate for mosaics; use s_region for spatial tests."},
                    {"name": "s_dec", "dtype": "float", "unit": "deg", "role": "dec",
                     "description": "Representative ICRS declination."},
                    {"name": "s_region", "dtype": "string", "role": "measurement",
                     "description": "STC-S footprint — the correct column for spatial intersection: INTERSECTS(CIRCLE('ICRS',ra,dec,r), s_region) = 1. Known issue: Total Power rows may show one pointing."},
                    {"name": "is_mosaic", "dtype": "string", "role": "category",
                     "description": "'T'/'F' — mosaic footprints extend far beyond s_ra/s_dec."},
                    {"name": "s_resolution", "dtype": "float", "unit": "arcsec", "role": "measurement",
                     "description": "Estimated synthesized beam."},
                    {"name": "spatial_resolution", "dtype": "float", "unit": "arcsec", "role": "measurement",
                     "description": "Estimated synthesized beam (same estimate as s_resolution)."},
                    {"name": "spatial_scale_max", "dtype": "float", "unit": "arcsec", "role": "measurement",
                     "description": "Maximum recoverable scale; flux on larger scales is progressively under-recovered (not a step cutoff)."},
                    {"name": "t_exptime", "dtype": "float", "unit": "s", "role": "measurement",
                     "description": "Total integration time."},
                    {"name": "t_min", "dtype": "float", "unit": "d", "role": "time",
                     "description": "Observation start, MJD days (NOT ISO — obs_release_date is the ISO one)."},
                    {"name": "t_max", "dtype": "float", "unit": "d", "role": "time",
                     "description": "Observation end, MJD days."},
                    {"name": "em_min", "dtype": "float", "unit": "m", "role": "measurement",
                     "description": "Minimum WAVELENGTH in meters. Frequency window [nu1, nu2] GHz overlaps when em_min <= 0.299792458/nu1 AND em_max >= 0.299792458/nu2."},
                    {"name": "em_max", "dtype": "float", "unit": "m", "role": "measurement",
                     "description": "Maximum WAVELENGTH in meters."},
                    {"name": "band_list", "dtype": "string", "role": "category",
                     "description": "SPACE-delimited band tokens as VARCHAR: '6', band-to-band '5 10'. Match tokens (band_list = '6' OR LIKE '6 %' OR LIKE '% 6' OR LIKE '% 6 %'), never a bare substring."},
                    {"name": "frequency", "dtype": "float", "unit": "GHz", "role": "measurement",
                     "description": "One REPRESENTATIVE frequency; not a coverage bound."},
                    {"name": "bandwidth", "dtype": "float", "unit": "Hz", "role": "measurement",
                     "description": "Aggregate bandwidth in Hz (NOT GHz) summed over non-contiguous SPWs — frequency +/- bandwidth/2 is NOT the covered range; parse frequency_support."},
                    {"name": "frequency_support", "dtype": "string", "role": "measurement",
                     "description": "U-joined per-SPW windows '[216.90..218.88GHz,31250.00kHz,...] U [...]'; the only column that gives exact spectral coverage."},
                    {"name": "velocity_resolution", "dtype": "float", "unit": "m/s", "role": "measurement",
                     "description": "Aggregate velocity resolution estimate in m/s (NOT km/s)."},
                    {"name": "antenna_arrays", "dtype": "string", "role": "category",
                     "description": "Blank-separated Pad:Antenna pairs ('A004:DV07 A025:CM03'); DV/DA = 12-m, CM = 7-m, PM = Total Power (heuristic). No 'array' column exists."},
                    {"name": "cont_sensitivity_bandwidth", "dtype": "float", "unit": "mJy/beam", "role": "measurement",
                     "description": "Estimated continuum rms over the aggregated bandwidth (indication only; smaller = deeper)."},
                    {"name": "sensitivity_10kms", "dtype": "float", "unit": "mJy/beam", "role": "measurement",
                     "description": "Estimated line rms per 10 km/s channel (smaller = deeper)."},
                    {"name": "data_rights", "dtype": "string", "role": "quality",
                     "description": "'Public' or 'Proprietary' — filter data_rights = 'Public' for anonymous access; never recompute rights from dates."},
                    {"name": "obs_release_date", "dtype": "string", "role": "time",
                     "description": "ISO timestamp of public release; a year-3000 value is a proprietary placeholder, not a promise. Can differ per MOUS."},
                    {"name": "access_url", "dtype": "string", "role": "provenance",
                     "description": "The MOUS's DataLink URL (a VOTable listing files) — NOT a direct file download. Enumerate with list_alma_files."},
                    {"name": "access_format", "dtype": "string", "role": "provenance",
                     "description": "Content format of access_url (DataLink VOTable)."},
                    {"name": "access_estsize", "dtype": "integer", "unit": "kbyte", "role": "measurement",
                     "description": "Coarse dataset size estimate in kbyte, nullable; plan downloads from DataLink content_length instead."},
                    {"name": "science_observation", "dtype": "string", "role": "quality",
                     "description": "'T' science rows, 'F' calibration-intent rows — add science_observation = 'T' for science-target counts."},
                    {"name": "scan_intent", "dtype": "string", "role": "category",
                     "description": "Multi-valued intents (TARGET, BANDPASS, PHASE, FLUX, ...); science data via scan_intent LIKE '%TARGET%'."},
                    {"name": "qa2_passed", "dtype": "string", "role": "quality",
                     "description": "Boolean archive flag 'T'/'F'; does NOT encode PASS/SEMIPASS/FAIL (the QA2 report PDF is authoritative). Filtering on 'T' must be disclosed."},
                    {"name": "obs_collection", "dtype": "string", "role": "provenance",
                     "description": "Collection label; reprocessed ARI-L products are not the original QA2 delivery — introspect live values."},
                    {"name": "bib_reference", "dtype": "string", "role": "provenance",
                     "description": "SPACE-SEPARATED list of ADS bibcodes of publications that used this observation (join key to the literature)."},
                    {"name": "pub_title", "dtype": "string", "role": "provenance",
                     "description": "Publication titles concatenated WITHOUT a delimiter — display only; join on bib_reference bibcodes instead."},
                    {"name": "pol_states", "dtype": "string", "role": "category",
                     "description": "Polarization states, e.g. 'XX YY'."},
                ],
                "citation_ids": ["cite_alma", "cite_alma_skill"],
            },
        },
        "pitfalls": [
            {
                "id": "row_grain",
                "summary": "obscore rows repeat per EB/field/SPW: group by member_ous_uid for datasets, COUNT(DISTINCT asdm_uid) for executions; rows are not observations",
                "detail": (
                    "ivoa.obscore rows are coverage records finer than a dataset: repeated across execution "
                    "blocks (asdm_uid), sources/fields and spectral coverage. MOUS-level work aggregates by "
                    "member_ous_uid; count executions as distinct asdm_uid; never sum frequency_support "
                    "over grouped rows; per-MOUS values such as obs_release_date can differ across rows."
                ),
                "applies_to": [{"kind": "table", "ref": "ivoa.obscore"}],
                "prompt_rank": 1,
                # Live audits verified 2026-09-24 (scripts/audit_archive_profiles.py).
                "audit": {"expect": "nonempty", "endpoint_id": "alma_tap",
                          "adql": ("SELECT member_ous_uid, COUNT(*) AS n FROM ivoa.obscore WHERE "
                                   "1=INTERSECTS(CIRCLE('ICRS',187.70593,12.39112,0.01), s_region) "
                                   "GROUP BY member_ous_uid HAVING COUNT(*) > 1")},
            },
            {
                "id": "units_and_footprints",
                "summary": "units: bandwidth Hz, frequency GHz, em_min/em_max METERS, velocity_resolution m/s, t_min MJD; cone: INTERSECTS(CIRCLE, s_region) OR point",
                "detail": (
                    "Frequency coverage: prefilter with em_min <= 0.299792458/nu_lo AND em_max >= 0.299792458/nu_hi, "
                    "then decide exact coverage from frequency_support (frequency +/- bandwidth/2 spans the "
                    "inter-sideband gap and is wrong). Spatial: s_ra/s_dec is a representative point; a true "
                    "cone is INTERSECTS(CIRCLE('ICRS',ra,dec,r), s_region) = 1, unioned with the point test "
                    "so NULL/single-pointing (Total Power) footprints are not lost."
                ),
                "applies_to": [{"kind": "table", "ref": "ivoa.obscore"},
                                {"kind": "surface", "ref": "raw_adql"}],
                "prompt_rank": 2,
                # The unit claims themselves, as TAP_SCHEMA publishes them (t_min is MJD, unit 'd').
                "audit": {"expect": "column_units", "endpoint_id": "alma_tap", "table": "ivoa.obscore",
                          "units": {"bandwidth": "Hz", "frequency": "GHz", "em_min": "m", "em_max": "m",
                                    "velocity_resolution": "m/s", "t_min": "d"}},
            },
            {
                "id": "qa2_and_datalink",
                "summary": "qa2_passed T/F cannot give PASS/SEMIPASS/FAIL (read the QA2 report); access_url = MOUS DataLink URL, not a file; no calibrated MS delivered",
                "detail": (
                    "The standard delivery holds caltables + scripts + QA + selected FITS products; a calibrated "
                    "MeasurementSet must be restored with scriptForPI under the package's CASA version or "
                    "obtained from an ARC/SRDP service. An empty anonymous DataLink table for a valid "
                    "proprietary MOUS differs from a NotFound error for an invalid UID."
                ),
                "applies_to": [{"kind": "table", "ref": "ivoa.obscore"},
                                {"kind": "surface", "ref": "file_inventory"}],
                "prompt_rank": 3,
                "audit": {"expect": "nonempty", "endpoint_id": "alma_tap",
                          "adql": ("SELECT TOP 1 obs_id FROM ivoa.obscore WHERE "
                                   "1=INTERSECTS(CIRCLE('ICRS',187.70593,12.39112,0.01), s_region) AND access_url LIKE '%datalink%'")},
            },
            {
                "id": "minimal_parameters",
                "summary": "pass ONLY the filters the user asked for — a plain request is search_by_target(target_name=...) with NO band/resolution/frequency defaults",
                "applies_to": [{"kind": "surface", "ref": "target_search"}],
            },
            {
                "id": "name_resolver_rule",
                "summary": "resolve source names via SIMBAD/NED to a footprint cone (search_by_target does this); NEVER string-match PI-entered target_name — if unavoidable, flag it",
                "applies_to": [{"kind": "surface", "ref": "target_search"},
                                {"kind": "surface", "ref": "raw_adql"},
                                {"kind": "table", "ref": "ivoa.obscore"}],
            },
            {
                "id": "band_list_tokens",
                "summary": "band_list is space-delimited ('5 10' for band-to-band): match tokens, never bare substrings; confirm the science band from frequency_support",
                "applies_to": [{"kind": "column", "ref": "ivoa.obscore:band_list"}],
                "audit": {"expect": "nonempty", "endpoint_id": "alma_tap",
                          "adql": "SELECT TOP 1 band_list FROM ivoa.obscore WHERE band_list LIKE '% %'"},
            },
            {
                "id": "cycle_project_codes",
                "summary": "a cycle's codes include the supplemental call (.2.) and DDT letter periods (.A.); no 2014/2020 call years — templates disclose the periods counted",
                "applies_to": [{"kind": "surface", "ref": "science_templates"}],
            },
            {
                "id": "templates_over_adql",
                "summary": "for cycle counts, solar projects, array combinations, or required line sets use query_alma_science_archive templates — not hand-written ADQL",
                "applies_to": [{"kind": "surface", "ref": "science_templates"}],
            },
            {
                "id": "named_transition",
                "summary": "one named transition + target -> find_alma_line_coverage (resolves it and checks exact spectral windows); check_co_lines is for whole-ladder requests",
                "applies_to": [{"kind": "surface", "ref": "line_coverage"}],
            },
            {
                "id": "top_truncation",
                "summary": "TAP fetches are capped (TOP 5000 by default, ordered by proposal_id); results carry rows_fetched/truncated — never report a capped fetch as complete",
                "applies_to": [{"kind": "surface", "ref": "science_templates"},
                                {"kind": "surface", "ref": "raw_adql"}],
            },
        ],
        "golden_examples": [
            {
                "id": "plain_target",
                "intent": "Find ALMA data of M87 (no constraints stated).",
                "invocation": {"tool": "search_by_target", "arguments": {"target_name": "M87"}},
                "note": "No invented band/resolution filters — omitting a filter means 'no filter'.",
            },
            {
                "id": "cycle_solar",
                "intent": "How many Cycle 10 projects observed the Sun?",
                "invocation": {
                    "tool": "query_alma_science_archive",
                    "arguments": {"query_type": "cycle_solar_projects", "cycle": 10},
                },
                "note": "Template uses scientific_category = 'Sun' and excludes Sunyaev-Zel'dovich text matches.",
            },
            {
                "id": "band6_frequency_window",
                "intent": "Observations covering 230-240 GHz.",
                "invocation": {
                    "tool": "search_by_frequency",
                    "arguments": {"min_freq_ghz": 230, "max_freq_ghz": 240},
                },
                "note": "Executed as an em_min/em_max wavelength overlap, not frequency BETWEEN.",
            },
            {
                "id": "adql_footprint_cone_public",
                "intent": "Public Band 6 science rows whose footprint overlaps a 0.1 deg cone at (10.68, 41.27), grouped per MOUS.",
                "invocation": {
                    "tool": "vo_adql_query",
                    "arguments": {
                        "access_url": "https://almascience.nrao.edu/tap",
                        "adql": (
                            "SELECT member_ous_uid, proposal_id, band_list, "
                            "COUNT(DISTINCT asdm_uid) AS n_eb, MIN(obs_release_date) AS released "
                            "FROM ivoa.obscore "
                            "WHERE INTERSECTS(CIRCLE('ICRS', 10.68, 41.27, 0.1), s_region) = 1 "
                            "AND (band_list = '6' OR band_list LIKE '6 %' OR band_list LIKE '% 6' OR band_list LIKE '% 6 %') "
                            "AND data_rights = 'Public' AND science_observation = 'T' "
                            "GROUP BY member_ous_uid, proposal_id, band_list"
                        ),
                        "max_rows": 100,
                    },
                },
                "request": {"kind": "adql", "argument": "adql"},
                "note": "Footprint cone via s_region; band_list token match; EBs counted as DISTINCT asdm_uid.",
            },
            {
                "id": "mous_file_inventory",
                "intent": "Which files does MOUS uid://A001/X1590/X30a8 deliver?",
                "invocation": {"tool": "list_alma_files", "arguments": {"mous_uid": "uid://A001/X1590/X30a8"}},
                "note": "Rows are typed (file / service / nested DataLink / error); an empty table means not visible under this authorization.",
            },
        ],
        "unit_conventions": [
            {
                "id": "obscore_units",
                "statement": (
                    "obscore em_min/em_max are wavelengths in meters; frequency in GHz but bandwidth in Hz; "
                    "velocity_resolution m/s; s_resolution/spatial_resolution/spatial_scale_max arcsec; "
                    "t_exptime seconds; t_min/t_max MJD days; obs_release_date ISO string; access_estsize kbyte."
                ),
                "applies_to": [{"kind": "table", "ref": "ivoa.obscore"}],
            },
        ],
        "citations": [
            {
                "id": "cite_alma",
                "text": "ALMA Science Archive",
                "url": "https://almascience.nrao.edu/",
            },
            {
                "id": "cite_alma_skill",
                "text": "Working with ALMA data skill (vendored third_party/alma-data-skill, reviewed 2026-07-18)",
            },
        ],
    }
)
