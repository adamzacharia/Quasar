"""Splatalogue spectral-line profile (non-tabular: tables=None).

Parameter-service archive: the grounding that matters is unit discipline
(frequency bounds in GHz; energy bounds interpreted per energy_type) and
rest-vs-redshifted frequency routing.
"""

from __future__ import annotations

from services.archive_profiles.schema import ArchiveProfile

PROFILE = ArchiveProfile.model_validate(
    {
        "archive": "splatalogue",
        "aliases": ("lines", "spectral lines", "line list"),
        "description": (
            "Splatalogue molecular line database: identify lines near a frequency, search "
            "transitions of a molecule, and run filtered frequency-range line surveys."
        ),
        "endpoints": [
            {
                "id": "splatalogue",
                "description": "Splatalogue line-list service.",
                "url": "https://splatalogue.online/",
                "protocol": "line_list",
            },
        ],
        "query_surfaces": [
            {
                "id": "identify_line",
                "purpose": "What line is near this frequency? (line identification around a rest frequency).",
                "tool": "identify_spectral_line",
                "request_kind": "parameter_service",
                "parameters": [
                    {"name": "frequency_ghz", "json_type": "number", "unit": "GHz",
                     "description": "REST frequency to search around, e.g. 230.538."},
                    {"name": "tolerance_ghz", "json_type": "number", "unit": "GHz",
                     "description": "Search half-window (default 0.01 = 10 MHz)."},
                ],
                "endpoint_ids": ["splatalogue"],
            },
            {
                "id": "molecule_lines",
                "purpose": "Transitions of one molecule, optionally filtered by frequency/energy/intensity.",
                "tool": "search_lines_by_molecule",
                "request_kind": "parameter_service",
                "parameters": [
                    {"name": "freq_min_ghz", "json_type": "number", "unit": "GHz",
                     "description": "Lower rest-frequency bound."},
                    {"name": "freq_max_ghz", "json_type": "number", "unit": "GHz",
                     "description": "Upper rest-frequency bound."},
                    {"name": "energy_type", "json_type": "string",
                     "allowed_values": ["el_cm1", "eu_cm1", "el_k", "eu_k"],
                     "description": "Sets the UNITS of energy_min/energy_max: *_k in kelvin, *_cm1 in 1/cm."},
                    {"name": "intensity_type", "json_type": "string",
                     "allowed_values": ["CDMS/JPL (log)", "Sij-mu2", "Aij (log)"],
                     "description": "Sets the scale of intensity_lower_limit."},
                ],
                "endpoint_ids": ["splatalogue"],
            },
            {
                "id": "range_survey",
                "purpose": "Filtered line survey over a frequency window (line-confusion checks, catalog comparisons).",
                "tool": "search_spectral_lines",
                "request_kind": "parameter_service",
                "parameters": [
                    {"name": "freq_min_ghz", "json_type": "number", "unit": "GHz",
                     "description": "Lower rest-frequency bound (required)."},
                    {"name": "freq_max_ghz", "json_type": "number", "unit": "GHz",
                     "description": "Upper rest-frequency bound (required)."},
                    {"name": "line_lists", "json_type": "array",
                     "description": "Source catalogs to include (CDMS, JPL, SLAIM, LovasNIST, ...)."},
                ],
                "endpoint_ids": ["splatalogue"],
            },
        ],
        "tables": None,
        "pitfalls": [
            {
                "id": "ghz_rest_frame",
                "summary": "all frequency bounds are GHz in the REST frame; for a named transition toward a target use find_alma_line_coverage — never redshift frequencies by hand",
                "applies_to": [{"kind": "archive", "ref": "splatalogue"}],
                "prompt_rank": 1,
            },
            {
                "id": "energy_type_units",
                "summary": "energy_min/energy_max are interpreted per energy_type (el_k/eu_k in K, el_cm1/eu_cm1 in 1/cm) — a mismatched pair silently filters the wrong lines",
                "applies_to": [{"kind": "surface", "ref": "molecule_lines"}],
                "prompt_rank": 2,
            },
            {
                "id": "intensity_scale",
                "summary": "intensity_lower_limit is scaled by intensity_type (CDMS/JPL log, Sij-mu2, Aij log) — state which scale you filtered on",
                "applies_to": [{"kind": "surface", "ref": "molecule_lines"}],
            },
        ],
        "golden_examples": [
            {
                "id": "co_21_window",
                "intent": "CO transitions in the 1.3 mm window.",
                "invocation": {
                    "tool": "search_lines_by_molecule",
                    "arguments": {"molecule_name": "CO", "freq_min_ghz": 220, "freq_max_ghz": 235},
                },
            },
            {
                "id": "identify_230538",
                "intent": "Which line sits at 230.538 GHz?",
                "invocation": {
                    "tool": "identify_spectral_line",
                    "arguments": {"frequency_ghz": 230.538},
                },
            },
            {
                "id": "band7_survey",
                "intent": "Astronomically observed lines in a Band 7 spectral window.",
                "invocation": {
                    "tool": "search_spectral_lines",
                    "arguments": {"freq_min_ghz": 342.0, "freq_max_ghz": 346.0,
                                   "only_astronomically_observed": True},
                },
            },
        ],
        "unit_conventions": [
            {
                "id": "freq_ghz",
                "statement": "Frequencies are GHz everywhere; energies follow energy_type (K or 1/cm).",
            },
        ],
        "citations": [
            {"id": "cite_splatalogue", "text": "Splatalogue (NRAO spectral line catalog)",
             "url": "https://splatalogue.online/"},
        ],
    }
)
