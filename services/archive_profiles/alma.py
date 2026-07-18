"""ALMA Science Archive profile (ivoa.obscore via TAP + capability tools).

Canonical/common grounding only — the obscore column subset below mirrors the
ALMA_TAP_SCHEMA block the agent already sees (core/prompts), with units made
mechanical. Frequencies GHz, wavelengths METERS, resolutions arcsec, times MJD.
"""

from __future__ import annotations

from services.archive_profiles.schema import ArchiveProfile

PROFILE = ArchiveProfile.model_validate(
    {
        "archive": "alma",
        "aliases": ("alma_archive", "almascience"),
        "description": (
            "ALMA Science Archive: interferometric observations searched by target/position/"
            "frequency, deterministic science-query templates, and TAP/ADQL over ivoa.obscore."
        ),
        "endpoints": [
            {
                "id": "alma_tap",
                "description": "ALMA TAP service (ivoa.obscore).",
                "url": "https://almascience.nrao.edu/tap",
                "protocol": "tap",
            },
        ],
        "query_surfaces": [
            {
                "id": "target_search",
                "purpose": "Default archive search by target name (multi-target via commas).",
                "tool": "search_by_target",
                "request_kind": "structured_args",
                "parameters": [
                    {"name": "band", "json_type": "string",
                     "description": "ALMA band number(s) AS A STRING ('6' or '6,7'). Pass ONLY if the user explicitly asks."},
                    {"name": "max_resolution", "json_type": "number", "unit": "arcsec",
                     "description": "Maximum angular resolution. Pass ONLY if the user specifies one."},
                    {"name": "min_freq_ghz", "json_type": "number", "unit": "GHz",
                     "description": "Minimum frequency. Pass ONLY if the user specifies."},
                    {"name": "max_freq_ghz", "json_type": "number", "unit": "GHz",
                     "description": "Maximum frequency. Pass ONLY if the user specifies."},
                ],
                "endpoint_ids": ["alma_tap"],
            },
            {
                "id": "science_templates",
                "purpose": "Deterministic templates for hard archive-science questions (cycle counts, array combos, line sets, sensitivity-driven discovery, publication joins).",
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
                     "description": "Resolution ceiling for high_resolution_band_data."},
                    {"name": "sensitivity_mjy", "json_type": "number", "unit": "mJy/beam",
                     "description": "sensitivity_search: required rms — returns data with achieved sensitivity <= this."},
                    {"name": "identifier", "json_type": "string",
                     "description": "data_publications: project code, MOUS uid://..., or ADS bibcode (reverse paper->data)."},
                ],
                "endpoint_ids": ["alma_tap"],
            },
            {
                "id": "frequency_search",
                "purpose": "Search the archive by frequency range.",
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
                "purpose": "One named transition + target: resolve with Splatalogue and verify exact spectral-window coverage.",
                "tool": "find_alma_line_coverage",
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
                "purpose": "One observation record per member OUS product: positions, bands, resolution, sensitivity, project ids.",
                "grain": "one row per observation (member OUS science product)",
                "ra_column": "s_ra",
                "dec_column": "s_dec",
                "columns": [
                    {"name": "target_name", "dtype": "string", "role": "identifier",
                     "description": "PI-entered target name — free text, inconsistent ('NGC1068'/'ngc 1068'/'N1068'). "
                                    "NOT for source selection: resolve the name and cone-search s_ra/s_dec instead; "
                                    "use only for labelling, or as an explicitly-flagged string match."},
                    {"name": "s_ra", "dtype": "float", "unit": "deg", "role": "ra",
                     "description": "ICRS right ascension (J2000)."},
                    {"name": "s_dec", "dtype": "float", "unit": "deg", "role": "dec",
                     "description": "ICRS declination (J2000)."},
                    {"name": "s_resolution", "dtype": "float", "unit": "arcsec", "role": "measurement",
                     "description": "Angular resolution."},
                    {"name": "t_exptime", "dtype": "float", "unit": "s", "role": "measurement",
                     "description": "Total integration time."},
                    {"name": "t_min", "dtype": "float", "unit": "d", "role": "time",
                     "description": "Observation start (MJD)."},
                    {"name": "t_max", "dtype": "float", "unit": "d", "role": "time",
                     "description": "Observation end (MJD)."},
                    {"name": "em_min", "dtype": "float", "unit": "m", "role": "measurement",
                     "description": "Minimum WAVELENGTH in meters (not frequency)."},
                    {"name": "em_max", "dtype": "float", "unit": "m", "role": "measurement",
                     "description": "Maximum WAVELENGTH in meters."},
                    {"name": "band_list", "dtype": "string", "role": "category",
                     "description": "ALMA band number(s) as a VARCHAR, e.g. '6' — compare as a string."},
                    {"name": "frequency", "dtype": "float", "unit": "GHz", "role": "measurement",
                     "description": "Central frequency."},
                    {"name": "bandwidth", "dtype": "float", "unit": "GHz", "role": "measurement",
                     "description": "Total bandwidth."},
                    {"name": "proposal_id", "dtype": "string", "role": "identifier",
                     "description": "ALMA project code, e.g. 2019.1.00123.S."},
                    {"name": "obs_publisher_did", "dtype": "string", "role": "identifier",
                     "description": "Unique dataset identifier."},
                    {"name": "member_ous_uid", "dtype": "string", "role": "identifier",
                     "description": "Member OUS UID (uid://...) — the data-delivery unit."},
                    {"name": "access_url", "dtype": "string", "role": "provenance",
                     "description": "Data download URL."},
                    {"name": "cont_sensitivity_bandwidth", "dtype": "float", "unit": "mJy/beam", "role": "measurement",
                     "description": "Continuum sensitivity over the full bandwidth (achieved rms; smaller = deeper)."},
                    {"name": "sensitivity_10kms", "dtype": "float", "unit": "mJy/beam", "role": "measurement",
                     "description": "Line sensitivity per 10 km/s channel (achieved rms; smaller = deeper)."},
                    {"name": "bib_reference", "dtype": "string", "role": "provenance",
                     "description": "SPACE-SEPARATED list of ADS bibcodes of publications that used this observation (join key to the literature)."},
                    {"name": "pub_title", "dtype": "string", "role": "provenance",
                     "description": "Publication titles concatenated WITHOUT a delimiter — display only; join on bib_reference bibcodes instead."},
                    {"name": "velocity_resolution", "dtype": "float", "unit": "km/s", "role": "measurement",
                     "description": "Velocity resolution."},
                    {"name": "pol_states", "dtype": "string", "role": "category",
                     "description": "Polarization states, e.g. 'XX YY'."},
                    {"name": "science_observation", "dtype": "string", "role": "quality",
                     "description": "Science-observation flag."},
                    {"name": "scan_intent", "dtype": "string", "role": "category",
                     "description": "Scan intent; science data via scan_intent LIKE '%TARGET%'."},
                    {"name": "qa2_passed", "dtype": "string", "role": "quality",
                     "description": "QA2 flag: 'T' = PASS, 'F' = SEMIPASS."},
                ],
                "citation_ids": ["cite_alma"],
            },
        },
        "pitfalls": [
            {
                "id": "minimal_parameters",
                "summary": "pass ONLY the filters the user asked for — a plain request is search_by_target(target_name=...) with NO band/resolution/frequency defaults",
                "applies_to": [{"kind": "surface", "ref": "target_search"}],
                "prompt_rank": 1,
            },
            {
                "id": "name_resolver_rule",
                "summary": "resolve source names via SIMBAD/NED to a cone on s_ra/s_dec (search_by_target does this); NEVER string-match PI-entered target_name — if unavoidable, flag it",
                "applies_to": [{"kind": "surface", "ref": "target_search"},
                                {"kind": "surface", "ref": "raw_adql"},
                                {"kind": "table", "ref": "ivoa.obscore"}],
            },
            {
                "id": "obscore_units",
                "summary": "obscore units: band_list is a VARCHAR ('6'); em_min/em_max are wavelengths in METERS; frequency GHz; t_exptime s; s_resolution arcsec; t_min/t_max MJD",
                "applies_to": [{"kind": "table", "ref": "ivoa.obscore"}],
                "prompt_rank": 2,
            },
            {
                "id": "templates_over_adql",
                "summary": "for cycle counts, solar projects, array combinations, or required line sets use query_alma_science_archive templates — not hand-written ADQL",
                "applies_to": [{"kind": "surface", "ref": "science_templates"}],
            },
            {
                "id": "intent_and_qa2",
                "summary": "science-target rows via scan_intent LIKE '%TARGET%'; QA2-passed data via qa2_passed = 'T'",
                "applies_to": [{"kind": "table", "ref": "ivoa.obscore"}],
            },
            {
                "id": "named_transition",
                "summary": "one named transition + target -> find_alma_line_coverage (resolves it and checks exact spectral windows); check_co_lines is for whole-ladder requests",
                "applies_to": [{"kind": "surface", "ref": "line_coverage"}],
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
            },
            {
                "id": "band6_frequency_window",
                "intent": "Observations covering 230-240 GHz.",
                "invocation": {
                    "tool": "search_by_frequency",
                    "arguments": {"min_freq_ghz": 230, "max_freq_ghz": 240},
                },
            },
            {
                "id": "adql_high_res_band6",
                "intent": "High-resolution Band 6 science observations via raw ADQL (no template fits).",
                "invocation": {
                    "tool": "vo_adql_query",
                    "arguments": {
                        "access_url": "https://almascience.nrao.edu/tap",
                        "adql": (
                            "SELECT target_name, proposal_id, s_resolution, frequency "
                            "FROM ivoa.obscore "
                            "WHERE band_list = '6' AND s_resolution < 0.1 "
                            "AND scan_intent LIKE '%TARGET%'"
                        ),
                        "max_rows": 100,
                    },
                },
                "request": {"kind": "adql", "argument": "adql"},
                "note": "band_list compared as a STRING; resolution in arcsec.",
            },
        ],
        "unit_conventions": [
            {
                "id": "obscore_units",
                "statement": (
                    "obscore em_min/em_max are wavelengths in meters; frequency/bandwidth in GHz; "
                    "s_resolution arcsec; t_exptime seconds; t_min/t_max MJD."
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
        ],
    }
)
