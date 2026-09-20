"""
ALMABench v0 — held-out ALMA archive questions + machine-readable rubrics
=========================================================================

Twelve questions on the DataLabBench checkpoint DSL (see
Benchmark/datalabbench/dlb_dataset_v1.py for the check-kind semantics), each
expressing one or more guardrails of the reviewed "Working with ALMA data"
skill (third_party/alma-data-skill):

  1. ObsCore rows are coverage records — report rows, MOUS and EBs separately
  2. ObsCore units (bandwidth Hz, em_min/em_max metres, t_min MJD, m/s)
  3. band_list tolerance (band-to-band '5 10')
  4. the standard delivery has no calibrated MS (scriptForPI restore)
  6. DataLink states and byte preflight
  8. qa2_passed cannot encode PASS/SEMIPASS/FAIL
 10. branch to special handling (mosaics, TP, moving targets)
 plus footprint cones (INTERSECTS on s_region), wavelength-overlap prefilters,
 cycle project-code periods, the Technical Handbook sensitivity equation and
 the TOP-truncation disclosure.

HELD-OUT DESIGN: these prompts were written from the skill's guardrails, not
from the unit tests or the fix list that implemented them, and they are not
used anywhere in the codebase (no router rule, no golden example, no test
fixture quotes them). Real identifiers are public: NGC 253, Sgr A*, M87,
Orion KL; MOUS uid://A001/X133d/X1d1 is the skill's own worked example. If
an identifier goes stale, the rubric still scores process (which tool, which
ADQL shape, what was disclosed), not one archive number.

Auto checkpoints score the executed trace (tool arguments, executed ADQL via
the uniform request provenance, response text); judge checkpoints grade the
answer against the trace (fabrication is a zero). Run with
Benchmark/almabench/run_almabench.py.
"""

BENCH_NAME = "ALMABench"
BENCH_VERSION = "0.1"

TIER_WEIGHTS = {1: 1.0, 2: 1.2, 3: 1.4, 4: 1.6, 5: 1.8}
TIER_LABELS = {
    1: "Single archive lookup with honest counts",
    2: "Frequency / footprint / QA2 semantics",
    3: "Deterministic templates and calculators",
    4: "Products, restore and special cases",
    5: "Open-ended archive science (multi-step)",
}

# Tools whose success signals a real ALMA archive interaction (GP-00).
ALMA_QUERY_TOOLS = [
    "search_by_target", "search_by_position", "search_by_frequency", "advanced_search",
    "search_alma_with_keywords", "search_alma_co_in_redshift_range", "query_alma_science_archive",
    "find_alma_line_coverage", "check_line_coverage", "check_co_lines", "list_alma_files",
    "triage_alma_data_products", "get_observation_details", "get_alma_qa2_status",
    "browse_alma_guidance", "browse_schema", "calculate_alma_sensitivity", "calculate_beam",
    "calculate_doppler_shift", "vo_adql_query", "generate_casa_imaging_script",
    "generate_casa_calibration_script", "inspect_fits_header",
]

# Tools that EXECUTE ADQL against the ALMA TAP: their `request.kind == 'adql'`
# provenance (and query/adql args) count as executed SQL for sql_regex /
# trace_regex / position_near checks. The runner installs this pattern into the
# shared scorer (run_datalabbench.SQL_CAPABLE_TOOL_RE).
ALMA_SQL_CAPABLE_TOOL_PATTERN = (
    r"^(search_by_target|search_by_position|search_by_frequency|advanced_search|"
    r"search_alma_co_in_redshift_range|query_alma_science_archive|vo_adql_query|"
    r"match_cross_archive_sources|get_observation_details)$"
)

_ROWS_AS_OBSERVATIONS = r"\b\d[\d,]*\s+(?:ALMA\s+)?observations\b"
_GRAIN_WORDS = r"\bMOUS\b|member_ous|\bdataset|execution block|\basdm|\bEBs?\b|coverage record|\brows?\b"

GLOBAL_PENALTIES = [
    {
        "id": "GP-00",
        "desc": "Fabricated result: the answer presents concrete archive results but NO ALMA "
                "tool call succeeded during the run.",
        "points": 20,
        "detect": {"kind": "not", "of": {"kind": "tool_ok", "tools": ALMA_QUERY_TOOLS}},
        "only_if_text": (
            r"(?i)(?:there\s+are|found|returned|retrieved|yields?|contains?|"
            r"total(?:s|ing)?\s*(?:of|:|=)?|count(?:s)?\s*(?:of|:|=|is))\s*[\d,]+"
            r"|\b[\d,]{1,12}\s+(?:observations?|datasets?|projects?|MOUSs?|rows?|files?|execution blocks?)\b"
            r"|\|\s*-?[\d.,eE+]+\s*\|"
        ),
    },
    {
        "id": "GP-A1",
        "desc": "Rows reported as observations: the answer states 'N observations' and never "
                "mentions the MOUS / dataset / execution-block / row grain (skill guardrail 1).",
        "points": 10,
        "detect": {"kind": "all", "of": [
            {"kind": "text_regex", "pattern": _ROWS_AS_OBSERVATIONS},
            {"kind": "not", "of": {"kind": "text_regex", "pattern": _GRAIN_WORDS}},
        ]},
    },
    {
        "id": "GP-A2",
        "desc": "Executed ADQL treats the aggregate bandwidth as one contiguous window "
                "(frequency +/- bandwidth/2) — physically wrong for ALMA tunings (guardrail 2/3).",
        "points": 10,
        "detect": {"kind": "trace_regex", "pattern": r"0\.5\s*\*\s*bandwidth|bandwidth\s*/\s*2(?!\d)"},
    },
    {
        "id": "GP-A3",
        "desc": "Executed cone tests only the representative point (CONTAINS(POINT(s_ra,s_dec)) "
                "with no INTERSECTS on s_region) — mosaics overlapping the cone are missed.",
        "points": 5,
        "detect": {"kind": "all", "of": [
            {"kind": "trace_regex", "pattern": r"CONTAINS\s*\(\s*POINT\s*\(\s*'ICRS'\s*,\s*s_ra"},
            {"kind": "not", "of": {"kind": "trace_regex", "pattern": r"INTERSECTS\s*\("}},
        ]},
    },
]

_JUDGE_FABRICATION = (
    " Zero credit for any number, identifier or file name that does not appear in any tool "
    "output in the trace (fabricated content)."
)

QUESTIONS = [

    # ══════════════════════════ TIER 1 ══════════════════════════
    {
        "id": "ALMA-01",
        "tier": 1,
        "title": "Band 6 inventory of NGC 253 at the right grain",
        "prompt": (
            "How much ALMA Band 6 data exists for NGC 253? Report the number of archive rows, "
            "the number of distinct datasets (MOUS) and the number of execution blocks separately, "
            "and say how many of the rows are public."
        ),
        "exercises": "search_by_target with a band token filter; rows vs member_ous_uid vs asdm_uid; data_rights.",
        "reference_actions": "search_by_target(target_name='NGC 253', band='6'); read counts / count_summary; report rows, n_mous, n_eb, public split.",
        "reference_sql": None,
        "checkpoints": [
            {"id": "C1", "type": "auto", "points": 20,
             "desc": "Searched the ALMA archive by target with a Band 6 filter",
             "checks": [
                 {"kind": "tool_ok", "tools": ["search_by_target", "search_by_position", "advanced_search", "query_alma_science_archive"]},
                 {"kind": "any", "of": [
                     {"kind": "tool_arg", "tools": ["search_by_target"], "arg": "band", "contains": "6"},
                     {"kind": "sql_regex", "pattern": r"band_list"},
                 ]},
             ]},
            {"id": "C2", "type": "auto", "points": 20,
             "desc": "Answer distinguishes rows from datasets (MOUS) and execution blocks",
             "checks": [
                 {"kind": "text_regex", "pattern": r"\bMOUS\b|member_ous|\bdatasets?\b"},
                 {"kind": "text_regex", "pattern": r"execution block|\bEBs?\b|asdm"},
                 {"kind": "text_regex", "pattern": r"\brows?\b"},
             ]},
            {"id": "C3", "type": "auto", "points": 15,
             "desc": "Public / proprietary split stated (data_rights)",
             "checks": [{"kind": "text_regex", "pattern": r"public|proprietary|data_rights"}]},
            {"id": "C4", "type": "judge", "points": 30,
             "desc": "Counts are consistent with the tool output and not conflated",
             "guidance": "Full credit if the three numbers (rows, MOUS, EBs) match the tool's counts/count_summary "
                         "and rows are never called observations. Half credit if only rows and MOUS are right."
                         + _JUDGE_FABRICATION},
            {"id": "C5", "type": "judge", "points": 15,
             "desc": "Explains why rows exceed datasets (repeat per EB/field/SPW)",
             "guidance": "Credit for stating that ObsCore rows repeat per execution block, field and spectral window. "
                         "No credit for claiming rows are observations."},
        ],
    },

    {
        "id": "ALMA-02",
        "tier": 1,
        "title": "Cycle 7 solar projects with period disclosure",
        "prompt": (
            "Which ALMA Cycle 7 projects observed the Sun? State which project-code periods you counted "
            "(main call, supplemental call, DDT) and make sure Sunyaev-Zel'dovich projects are not included."
        ),
        "exercises": "cycle_solar_projects template; 2019.1/2019.2/2019.A disclosure; SZ exclusion.",
        "reference_actions": "query_alma_science_archive(query_type='cycle_solar_projects', cycle=7); relay warnings/query_summary.",
        "reference_sql": None,
        "checkpoints": [
            {"id": "C1", "type": "auto", "points": 25,
             "desc": "Used the deterministic solar template for Cycle 7",
             "checks": [
                 {"kind": "tool_arg", "tools": ["query_alma_science_archive"], "arg": "query_type", "equals": "cycle_solar_projects"},
                 {"kind": "tool_arg", "tools": ["query_alma_science_archive"], "arg": "cycle", "approx": 7, "tol": 0},
                 {"kind": "tool_ok", "tools": ["query_alma_science_archive"]},
             ]},
            {"id": "C2", "type": "auto", "points": 20,
             "desc": "Discloses the project-code periods counted (2019.1 / 2019.2 / 2019.A)",
             "checks": [
                 {"kind": "text_regex", "pattern": r"2019\.1|main call"},
                 {"kind": "text_regex", "pattern": r"2019\.2|supplemental"},
                 {"kind": "text_regex", "pattern": r"2019\.A|DDT|director"},
             ]},
            {"id": "C3", "type": "auto", "points": 15,
             "desc": "Mentions the Sunyaev-Zel'dovich exclusion / the category predicate",
             "checks": [{"kind": "text_regex", "pattern": r"Sunyaev|Zel'?dovich|\bSZ\b|scientific_category"}]},
            {"id": "C4", "type": "judge", "points": 25,
             "desc": "Project list/count matches the tool output and is not fabricated",
             "guidance": "Full credit if every listed project code appears in the tool output and the count equals "
                         "unique_projects (or explains a truncation warning)." + _JUDGE_FABRICATION},
            {"id": "C5", "type": "judge", "points": 15,
             "desc": "Truncation / completeness honestly handled",
             "guidance": "Credit if the answer relays a rows_fetched/truncated warning when present, or states the "
                         "fetch was complete when truncated=false. Zero if a truncated fetch is presented as complete."},
        ],
    },

    # ══════════════════════════ TIER 2 ══════════════════════════
    {
        "id": "ALMA-03",
        "tier": 2,
        "title": "CO(2-1) coverage at z = 0.04-0.05 from spectral windows",
        "prompt": (
            "Find ALMA observations whose spectral windows actually cover CO(2-1) for galaxies at "
            "redshift 0.04 to 0.05. Explain how coverage was decided and give the observed frequency range."
        ),
        "exercises": "Wavelength-overlap prefilter + frequency_support verification; observed frequency 219.6-221.7 GHz.",
        "reference_actions": "search_alma_co_in_redshift_range(z_min=0.04, z_max=0.05); relay coverage_method and covering_spw_ghz.",
        "reference_sql": "WHERE em_min <= 0.299792458/219.56 AND em_max >= 0.299792458/221.67",
        "checkpoints": [
            {"id": "C1", "type": "auto", "points": 20,
             "desc": "Used the CO-redshift tool (or exact line-coverage tool) for the right z range",
             "checks": [
                 {"kind": "any", "of": [
                     {"kind": "tool_ok", "tools": ["search_alma_co_in_redshift_range"]},
                     {"kind": "tool_ok", "tools": ["find_alma_line_coverage", "query_alma_science_archive"]},
                 ]},
                 {"kind": "any", "of": [
                     {"kind": "tool_arg", "tools": ["search_alma_co_in_redshift_range"], "arg": "z_min", "approx": 0.04, "tol": 0.005},
                     {"kind": "tool_arg", "tools": ["query_alma_science_archive"], "arg": "redshift_min", "approx": 0.04, "tol": 0.005},
                     {"kind": "tool_arg", "tools": ["find_alma_line_coverage"], "arg": "redshift", "min": 0.035, "max": 0.055},
                 ]},
             ]},
            {"id": "C2", "type": "auto", "points": 20,
             "desc": "Executed ADQL prefilters on em_min/em_max (wavelength overlap), not frequency +/- bandwidth/2",
             "checks": [
                 {"kind": "trace_regex", "pattern": r"em_min\s*<=|em_max\s*>="},
                 {"kind": "not", "of": {"kind": "trace_regex", "pattern": r"0\.5\s*\*\s*bandwidth"}},
             ]},
            {"id": "C3", "type": "auto", "points": 15,
             "desc": "Answer states the observed frequency window (~219.6-221.7 GHz)",
             "checks": [{"kind": "text_regex", "pattern": r"21[9]\.\d|22[01]\.\d"}]},
            {"id": "C4", "type": "auto", "points": 15,
             "desc": "Explains coverage was decided from spectral windows (frequency_support)",
             "checks": [{"kind": "text_regex", "pattern": r"frequency_support|spectral window|SPW"}]},
            {"id": "C5", "type": "judge", "points": 30,
             "desc": "Reported observations/coverage agree with the tool output",
             "guidance": "Full credit if the listed targets/projects and the covered-transition annotations come from "
                         "the tool results and approximate rows (coverage_method 'APPROXIMATE') are flagged as such."
                         + _JUDGE_FABRICATION},
        ],
    },

    {
        "id": "ALMA-04",
        "tier": 2,
        "title": "Footprint cone with a frequency overlap near Orion KL",
        "prompt": (
            "List ALMA datasets whose footprint overlaps a 0.1 degree circle around RA 83.81, Dec -5.37 "
            "and whose spectral coverage overlaps 345-350 GHz. Use the observation footprint, not just the "
            "catalog position, and state how many distinct MOUSs match."
        ),
        "exercises": "INTERSECTS(CIRCLE, s_region) footprint cone; em_min/em_max overlap; MOUS aggregation.",
        "reference_actions": "advanced_search with INTERSECTS + em_min/em_max, or search_by_position then frequency filter; report distinct member_ous_uid.",
        "reference_sql": (
            "SELECT member_ous_uid, proposal_id, band_list, COUNT(DISTINCT asdm_uid) FROM ivoa.obscore "
            "WHERE INTERSECTS(CIRCLE('ICRS', 83.81, -5.37, 0.1), s_region) = 1 "
            "AND em_min <= 0.299792458/345 AND em_max >= 0.299792458/350 GROUP BY member_ous_uid, proposal_id, band_list"
        ),
        "checkpoints": [
            {"id": "C1", "type": "auto", "points": 20,
             "desc": "Cone centered on (83.81, -5.37) with ~0.1 deg radius",
             "checks": [
                 {"kind": "position_near", "ra": 83.81, "dec": -5.37, "tol_deg": 0.05},
                 {"kind": "radius_near", "value": 0.1, "tol": 0.02},
             ]},
            {"id": "C2", "type": "auto", "points": 25,
             "desc": "Executed ADQL uses the s_region footprint (INTERSECTS), not only the s_ra/s_dec point",
             "checks": [{"kind": "trace_regex", "pattern": r"INTERSECTS\s*\(.*s_region"}]},
            {"id": "C3", "type": "auto", "points": 15,
             "desc": "Frequency overlap expressed through em_min/em_max or verified from frequency_support",
             "checks": [{"kind": "any", "of": [
                 {"kind": "trace_regex", "pattern": r"em_min|em_max"},
                 {"kind": "tool_ok", "tools": ["search_by_frequency"]},
                 {"kind": "text_regex", "pattern": r"frequency_support"},
             ]}]},
            {"id": "C4", "type": "auto", "points": 15,
             "desc": "Distinct MOUS count stated",
             "checks": [{"kind": "text_regex", "pattern": r"\bMOUS|member_ous|distinct dataset"}]},
            {"id": "C5", "type": "judge", "points": 25,
             "desc": "Datasets listed match the tool output; no invented identifiers",
             "guidance": "Full credit if every MOUS/project listed appears in a tool output and the MOUS count is "
                         "the number of distinct member_ous_uid, not the row count." + _JUDGE_FABRICATION},
        ],
    },

    {
        "id": "ALMA-05",
        "tier": 2,
        "title": "Three-state QA2 for one MOUS",
        "prompt": (
            "Did ALMA dataset uid://A001/X133d/X1d1 pass QA2? Tell me whether it is PASS, SEMIPASS or FAIL "
            "and where that information comes from."
        ),
        "exercises": "qa2_passed cannot encode the three-state QA2; get_alma_qa2_status reads the report.",
        "reference_actions": "get_alma_qa2_status(mous_uid='uid://A001/X133d/X1d1'); cite the QA2 report; never infer SEMIPASS from the flag.",
        "reference_sql": None,
        "checkpoints": [
            {"id": "C1", "type": "auto", "points": 30,
             "desc": "Resolved QA2 from the report tool (not the flag alone)",
             "checks": [
                 {"kind": "tool_called", "tools": ["get_alma_qa2_status"]},
                 {"kind": "tool_arg", "tools": ["get_alma_qa2_status"], "arg": "mous_uid", "contains": "X133d/X1d1"},
             ]},
            {"id": "C2", "type": "auto", "points": 20,
             "desc": "Answer names the QA2 report as the source and the flag's limitation",
             "checks": [
                 {"kind": "text_regex", "pattern": r"QA2 report|qa2_report|report"},
                 {"kind": "text_regex", "pattern": r"qa2_passed|flag|T/F|boolean"},
             ]},
            {"id": "C3", "type": "judge", "points": 35,
             "desc": "Disposition stated equals the tool's qa2_status (or Unknown handled honestly)",
             "guidance": "Full credit if the stated PASS/SEMIPASS/FAIL equals the get_alma_qa2_status result; if the "
                         "tool returned Unknown, full credit only for saying the disposition could not be read "
                         "(and NOT inferring SEMIPASS or PASS from qa2_passed)." + _JUDGE_FABRICATION},
            {"id": "C4", "type": "judge", "points": 15,
             "desc": "Explains what SEMIPASS means for science use if relevant",
             "guidance": "Credit for explaining that SEMIPASS means delivered with documented shortfalls (read the QA2 "
                         "report/README), or that the flag alone cannot distinguish PASS from SEMIPASS."},
        ],
    },

    # ══════════════════════════ TIER 3 ══════════════════════════
    {
        "id": "ALMA-06",
        "tier": 3,
        "title": "Cycle 8 projects combining 12-m, 7-m and Total Power",
        "prompt": (
            "Which ALMA Cycle 8 projects used the 12-m Array together with the 7-m and Total Power arrays? "
            "Explain how the arrays were identified and what caveat applies to combining them."
        ),
        "exercises": "cycle_array_combo_projects; antenna-prefix heuristic (DV/DA, CM, PM); archive delivers per MOUS.",
        "reference_actions": "query_alma_science_archive(query_type='cycle_array_combo_projects', cycle=8, arrays=['12m','7m','TP']).",
        "reference_sql": None,
        "checkpoints": [
            {"id": "C1", "type": "auto", "points": 25,
             "desc": "Used the array-combination template for Cycle 8",
             "checks": [
                 {"kind": "tool_arg", "tools": ["query_alma_science_archive"], "arg": "query_type", "equals": "cycle_array_combo_projects"},
                 {"kind": "tool_arg", "tools": ["query_alma_science_archive"], "arg": "cycle", "approx": 8, "tol": 0},
                 {"kind": "tool_ok", "tools": ["query_alma_science_archive"]},
             ]},
            {"id": "C2", "type": "auto", "points": 20,
             "desc": "Explains the antenna-name heuristic (DV/DA, CM, PM) or SB-name suffixes",
             "checks": [{"kind": "text_regex", "pattern": r"\bDV\b|\bDA\b|\bCM\b|\bPM\b|antenna|_TM1|_7M|_TP|heuristic"}]},
            {"id": "C3", "type": "auto", "points": 15,
             "desc": "States that the archive delivers per MOUS and has not combined the arrays",
             "checks": [{"kind": "text_regex", "pattern": r"not (?:been )?combined|per[- ]MOUS|separate(?:ly)?|sibling"}]},
            {"id": "C4", "type": "judge", "points": 25,
             "desc": "Project list/count matches the tool output",
             "guidance": "Full credit if every project code listed appears in the tool results with arrays_found "
                         "covering 12m, 7m and TP, and the count matches." + _JUDGE_FABRICATION},
            {"id": "C5", "type": "judge", "points": 15,
             "desc": "Cycle project-code periods and truncation disclosed",
             "guidance": "Credit for relaying which periods (2021.1 / 2021.2 / 2021.A) were counted and any "
                         "truncation warning; zero if a truncated fetch is called complete."},
        ],
    },

    {
        "id": "ALMA-07",
        "tier": 3,
        "title": "Band 6 continuum sensitivity per the Technical Handbook",
        "prompt": (
            "Estimate the ALMA 12-m Array Band 6 continuum point-source sensitivity for 1 hour on source "
            "with 43 antennas, 7.5 GHz bandwidth and dual polarization. Show the equation and the efficiency "
            "factors you used, and say what the official calculator would do differently."
        ),
        "exercises": "Technical Handbook eq. 9.8: eta_q 0.96, eta_c 0.88, N(N-1), Table 9.3 aperture efficiency.",
        "reference_actions": "calculate_alma_sensitivity(band=6, t_integration_s=3600, n_antennas=43, bandwidth_ghz=7.5, n_polarizations=2).",
        "reference_sql": None,
        "checkpoints": [
            {"id": "C1", "type": "auto", "points": 25,
             "desc": "Ran the sensitivity calculator with the requested inputs",
             "checks": [
                 {"kind": "tool_ok", "tools": ["calculate_alma_sensitivity"]},
                 {"kind": "tool_arg", "tools": ["calculate_alma_sensitivity"], "arg": "band", "approx": 6, "tol": 0},
                 {"kind": "tool_arg", "tools": ["calculate_alma_sensitivity"], "arg": "t_integration_s", "approx": 3600, "tol": 60},
             ]},
            {"id": "C2", "type": "auto", "points": 20,
             "desc": "States the quantization (0.96) and correlator (0.88) efficiencies",
             "checks": [
                 {"kind": "text_regex", "pattern": r"0\.96|quantization"},
                 {"kind": "text_regex", "pattern": r"0\.88|correlator efficiency"},
             ]},
            {"id": "C3", "type": "auto", "points": 15,
             "desc": "Equation shown with N(N-1) under the square root",
             "checks": [{"kind": "text_regex", "pattern": r"N\s*\(\s*N\s*-\s*1\s*\)|N\(N−1\)|N\*\(N-1\)"}]},
            {"id": "C4", "type": "judge", "points": 25,
             "desc": "Numerical result matches the tool output (about 16 uJy/beam for these inputs)",
             "guidance": "Full credit if the quoted rms equals the tool's continuum_sensitivity value (roughly "
                         "16 microJy/beam: 126.5 uJy at 60 s scaled by sqrt(60/3600)) with correct units; zero for a "
                         "number not in the tool output." + _JUDGE_FABRICATION},
            {"id": "C5", "type": "judge", "points": 15,
             "desc": "Says what the official ALMA Sensitivity Calculator adds",
             "guidance": "Credit for noting the ASC derives Tsys from receiver temperature, PWV octile and source "
                         "elevation (and that the estimate excludes overheads/imaging losses)."},
        ],
    },

    {
        "id": "ALMA-08",
        "tier": 3,
        "title": "Band 10 data of Sgr A* including band-to-band rows",
        "prompt": (
            "Find ALMA Band 10 observations of Sgr A*. Make sure observations taken in band-to-band "
            "(B2B) mode, where band_list holds two bands, are included, and say how you handled them."
        ),
        "exercises": "band_list tolerance ('5 10'); token match; multi-band disclosure.",
        "reference_actions": "search_by_target(target_name='Sgr A*', band='10'); relay the multi-band warning.",
        "reference_sql": None,
        "checkpoints": [
            {"id": "C1", "type": "auto", "points": 25,
             "desc": "Searched Sgr A* with a Band 10 filter",
             "checks": [
                 {"kind": "tool_ok", "tools": ["search_by_target", "search_by_position", "advanced_search"]},
                 {"kind": "any", "of": [
                     {"kind": "tool_arg", "tools": ["search_by_target"], "arg": "band", "contains": "10"},
                     {"kind": "sql_regex", "pattern": r"band_list"},
                 ]},
             ]},
            {"id": "C2", "type": "auto", "points": 20,
             "desc": "Band-to-band / multi-band rows addressed explicitly",
             "checks": [{"kind": "text_regex", "pattern": r"band[- ]to[- ]band|\bB2B\b|multi[- ]band|two bands|'?5 10'?"}]},
            {"id": "C3", "type": "auto", "points": 15,
             "desc": "If raw ADQL was used, band_list matched as tokens (not a bare substring)",
             "checks": [{"kind": "any", "of": [
                 {"kind": "not", "of": {"kind": "tool_called", "tools": ["advanced_search", "vo_adql_query"]}},
                 {"kind": "trace_regex", "pattern": r"band_list\s*=\s*'10'|LIKE\s*'% 10'|LIKE\s*'10 %'|LIKE\s*'% 10 %'"},
             ]}]},
            {"id": "C4", "type": "judge", "points": 25,
             "desc": "Counts reported at the right grain and consistent with the tool output",
             "guidance": "Full credit if rows / MOUS / EB counts (or the note) come from the tool and rows are not "
                         "called observations." + _JUDGE_FABRICATION},
            {"id": "C5", "type": "judge", "points": 15,
             "desc": "Notes that the science band should be confirmed from frequency_support for B2B rows",
             "guidance": "Credit for saying the higher band is usually the science band but frequency_support confirms it."},
        ],
    },

    # ══════════════════════════ TIER 4 ══════════════════════════
    {
        "id": "ALMA-09",
        "tier": 4,
        "title": "Typed DataLink inventory of a MOUS",
        "prompt": (
            "What files does the ALMA archive deliver for MOUS uid://A001/X133d/X1d1? Group them by kind "
            "(science FITS, auxiliary/scripts, README, raw data), say which sizes are unknown, and tell me "
            "whether a calibrated measurement set is among them."
        ),
        "exercises": "list_alma_files typed rows; content_length unknown; no calibrated MS in the delivery.",
        "reference_actions": "list_alma_files(mous_uid='uid://A001/X133d/X1d1'); relay category_counts, size_summary, state.",
        "reference_sql": None,
        "checkpoints": [
            {"id": "C1", "type": "auto", "points": 25,
             "desc": "Enumerated the MOUS with the DataLink tool",
             "checks": [
                 {"kind": "tool_ok", "tools": ["list_alma_files", "triage_alma_data_products"]},
                 {"kind": "any", "of": [
                     {"kind": "tool_arg", "tools": ["list_alma_files"], "arg": "mous_uid", "contains": "X133d"},
                     {"kind": "tool_arg", "tools": ["triage_alma_data_products"], "arg": "identifier_or_target", "contains": "X133d"},
                 ]},
             ]},
            {"id": "C2", "type": "auto", "points": 20,
             "desc": "Files grouped by kind (FITS / tar / README / raw ASDM)",
             "checks": [
                 {"kind": "text_regex", "pattern": r"\.fits|FITS"},
                 {"kind": "text_regex", "pattern": r"auxiliary|\.tar|README|asdm|raw"},
             ]},
            {"id": "C3", "type": "auto", "points": 20,
             "desc": "States that no calibrated MS is delivered (restore via scriptForPI)",
             "checks": [{"kind": "text_regex", "pattern": r"(?:no|not) (?:a )?calibrated (?:measurement set|MS|visibilit)|scriptForPI|restore"}]},
            {"id": "C4", "type": "judge", "points": 20,
             "desc": "Unknown sizes and DataLink state handled honestly",
             "guidance": "Full credit if the answer reports how many sizes were unknown (or that all were known) and, "
                         "for an empty table, explains 'no links visible under anonymous access' rather than 'invalid UID'."
                         + _JUDGE_FABRICATION},
            {"id": "C5", "type": "judge", "points": 15,
             "desc": "File names come from the tool output",
             "guidance": "Zero credit if any listed filename does not appear in the tool output; full credit for a faithful summary."},
        ],
    },

    {
        "id": "ALMA-10",
        "tier": 4,
        "title": "Imaging archival ALMA data: restore before tclean",
        "prompt": (
            "I downloaded the ALMA archive package for project 2019.1.00001.S and want to make a continuum "
            "image in CASA. Give me the steps and a script, and be explicit about what I must do before imaging."
        ),
        "exercises": "Guardrail 4: restore the MS with scriptForPI under the package CASA version; imaging script afterwards.",
        "reference_actions": "browse_alma_guidance(topic='products-qa-restore') and/or generate_casa_imaging_script; state scriptForPI + CASA version matching.",
        "reference_sql": None,
        "checkpoints": [
            {"id": "C1", "type": "auto", "points": 20,
             "desc": "Consulted the restore guidance or the imaging generator",
             "checks": [{"kind": "tool_ok", "tools": ["browse_alma_guidance", "generate_casa_imaging_script", "generate_casa_calibration_script"]}]},
            {"id": "C2", "type": "auto", "points": 25,
             "desc": "Names scriptForPI and CASA-version matching as the prerequisite",
             "checks": [
                 {"kind": "text_regex", "pattern": r"scriptForPI"},
                 {"kind": "text_regex", "pattern": r"CASA (?:version|release)|pipeline version|QA2 report|README|weblog"},
             ]},
            {"id": "C3", "type": "auto", "points": 15,
             "desc": "Mentions raw ASDMs / the auxiliary package or the ARC/SRDP calibrated-MS alternative",
             "checks": [{"kind": "text_regex", "pattern": r"ASDM|auxiliary|SRDP|ARC|calibrated (?:MS|measurement set|visibilities)"}]},
            {"id": "C4", "type": "judge", "points": 25,
             "desc": "Does not present a generic manual calibration as the ALMA path",
             "guidance": "Full credit if the answer states ALMA data are pipeline-calibrated and the package has no "
                         "calibrated MS, so restore comes first; zero if it offers a manual bandpass/gain recipe "
                         "(e.g. setjy Butler-JPL-Horizons on a quasar) as the standard ALMA procedure."},
            {"id": "C5", "type": "judge", "points": 15,
             "desc": "Imaging script is sensible for a restored MS",
             "guidance": "Credit for tclean on the restored calibrated/ MS with datacolumn/spw considered, cont.dat or "
                         "line exclusion for continuum, and no re-flagging of the pipeline-calibrated data."},
        ],
    },

    # ══════════════════════════ TIER 5 ══════════════════════════
    {
        "id": "ALMA-11",
        "tier": 5,
        "title": "Cycle 10 Band 6 projects covering the CO isotopologue set",
        "prompt": (
            "Which ALMA Cycle 10 Band 6 projects cover 12CO, 13CO and C18O J=2-1 within the same project? "
            "Report projects, distinct MOUSs and archive rows separately and say whether the archive fetch was complete."
        ),
        "exercises": "line_set_projects template; band token predicate; per-project rows/n_mous/n_eb; TOP truncation.",
        "reference_actions": "query_alma_science_archive(query_type='line_set_projects', band=6, lines=['12CO','13CO','C18O'], cycle=10) then filter to 2023.* if needed; relay truncated.",
        "reference_sql": None,
        "checkpoints": [
            {"id": "C1", "type": "auto", "points": 25,
             "desc": "Used the line-set template (or exact line coverage) for Band 6",
             "checks": [
                 {"kind": "any", "of": [
                     {"kind": "tool_arg", "tools": ["query_alma_science_archive"], "arg": "query_type", "equals": "line_set_projects"},
                     {"kind": "tool_ok", "tools": ["find_alma_line_coverage"]},
                 ]},
                 {"kind": "tool_ok", "tools": ["query_alma_science_archive", "find_alma_line_coverage", "advanced_search"]},
             ]},
            {"id": "C2", "type": "auto", "points": 20,
             "desc": "Reports projects, MOUSs and rows as separate numbers",
             "checks": [
                 {"kind": "text_regex", "pattern": r"project"},
                 {"kind": "text_regex", "pattern": r"\bMOUS|member_ous|dataset"},
                 {"kind": "text_regex", "pattern": r"\brows?\b"},
             ]},
            {"id": "C3", "type": "auto", "points": 15,
             "desc": "Completeness of the fetch stated (truncated or complete)",
             "checks": [{"kind": "text_regex", "pattern": r"truncat|complete|row cap|TOP \d+|incomplete"}]},
            {"id": "C4", "type": "judge", "points": 25,
             "desc": "Projects listed are Cycle 10 (2023.*) and come from the tool output",
             "guidance": "Full credit if every project code listed appears in the tool results and is a 2023.* code "
                         "(or the answer explains the template is not cycle-scoped and filtered accordingly)."
                         + _JUDGE_FABRICATION},
            {"id": "C5", "type": "judge", "points": 15,
             "desc": "Line coverage decided from spectral windows, not from frequency +/- bandwidth/2",
             "guidance": "Credit for stating coverage was checked against frequency_support / spectral windows at the "
                         "rest frequencies 230.538, 220.399 and 219.560 GHz."},
        ],
    },

    {
        "id": "ALMA-12",
        "tier": 5,
        "title": "Public vs proprietary M87 datasets with release dates",
        "prompt": (
            "Which ALMA datasets (MOUS) exist for M87, which of them are public today and which are still "
            "proprietary, and when do the proprietary ones become public? Distinguish rows from datasets."
        ),
        "exercises": "data_rights + obs_release_date per MOUS; grain; 3000-01-01 placeholder awareness.",
        "reference_actions": "search_by_target(target_name='M87'); group by member_ous_uid; report data_rights and obs_release_date min/max.",
        "reference_sql": None,
        "checkpoints": [
            {"id": "C1", "type": "auto", "points": 20,
             "desc": "Searched M87 in the ALMA archive",
             "checks": [
                 {"kind": "tool_ok", "tools": ["search_by_target", "search_by_position", "advanced_search", "query_alma_science_archive"]},
                 {"kind": "any", "of": [
                     {"kind": "tool_arg", "tools": ["search_by_target"], "arg": "target_name", "contains": "M87"},
                     {"kind": "tool_arg", "tools": ["search_by_target"], "arg": "target_name", "contains": "M 87"},
                     {"kind": "position_near", "ra": 187.706, "dec": 12.391, "tol_deg": 0.1},
                 ]},
             ]},
            {"id": "C2", "type": "auto", "points": 20,
             "desc": "Uses data_rights (not recomputed from dates) and release dates",
             "checks": [
                 {"kind": "text_regex", "pattern": r"data_rights|proprietary|public"},
                 {"kind": "text_regex", "pattern": r"release|obs_release_date"},
             ]},
            {"id": "C3", "type": "auto", "points": 15,
             "desc": "Distinguishes rows from MOUS datasets",
             "checks": [
                 {"kind": "text_regex", "pattern": r"\bMOUS|member_ous|dataset"},
                 {"kind": "text_regex", "pattern": r"\brows?\b"},
             ]},
            {"id": "C4", "type": "judge", "points": 30,
             "desc": "Public/proprietary split and dates match the tool output",
             "guidance": "Full credit if the MOUS counts and release dates come from the tool output, proprietary "
                         "placeholders (year 3000) are explained as placeholders, and the split matches data_rights."
                         + _JUDGE_FABRICATION},
            {"id": "C5", "type": "judge", "points": 15,
             "desc": "Does not overstate: rows are not observations; truncation relayed",
             "guidance": "Credit for correct grain language and for relaying any row-cap/truncation warning."},
        ],
    },
]

# Faithfulness/correctness axis overrides (same convention as DataLabBench v1.2).
AXIS_OVERRIDES = {
    "ALMA-02": {"C5": "faithfulness"},
    "ALMA-06": {"C5": "faithfulness"},
    "ALMA-10": {"C4": "faithfulness"},
    "ALMA-11": {"C5": "faithfulness"},
    "ALMA-12": {"C5": "faithfulness"},
}


def validate_dataset():
    """Sanity-check the dataset: ids unique, points sum to 100, kinds known."""
    errors = []
    seen = set()
    known_kinds = {
        "tool_called", "tool_ok", "tool_arg", "position_near", "radius_near",
        "sql_regex", "args_regex", "text_regex", "any_regex", "trace_regex",
        "image_emitted", "image_count", "any", "all", "not",
        "sql_unbounded_rowscan", "sql_between_rowscan", "sql_flat_q3c_join",
    }

    def walk(check, path):
        kind = check.get("kind")
        if kind not in known_kinds:
            errors.append(f"{path}: unknown check kind {kind!r}")
        for sub in check.get("of", []) if isinstance(check.get("of"), list) else []:
            walk(sub, path + f".{kind}")
        if isinstance(check.get("of"), dict):
            walk(check["of"], path + f".{kind}")

    for q in QUESTIONS:
        if q["id"] in seen:
            errors.append(f"duplicate id {q['id']}")
        seen.add(q["id"])
        total = sum(c["points"] for c in q["checkpoints"])
        if total != 100:
            errors.append(f"{q['id']}: checkpoint points sum to {total}, expected 100")
        if q["tier"] not in TIER_WEIGHTS:
            errors.append(f"{q['id']}: unknown tier {q['tier']}")
        for cp in q["checkpoints"]:
            if cp["type"] == "auto":
                if not cp.get("checks"):
                    errors.append(f"{q['id']}.{cp['id']}: auto checkpoint without checks")
                for i, chk in enumerate(cp.get("checks", [])):
                    walk(chk, f"{q['id']}.{cp['id']}[{i}]")
            elif cp["type"] == "judge":
                if not cp.get("guidance"):
                    errors.append(f"{q['id']}.{cp['id']}: judge checkpoint without guidance")
            else:
                errors.append(f"{q['id']}.{cp['id']}: unknown type {cp['type']}")
    for pen in GLOBAL_PENALTIES:
        walk(pen["detect"], f"penalty {pen['id']}")
    return errors
