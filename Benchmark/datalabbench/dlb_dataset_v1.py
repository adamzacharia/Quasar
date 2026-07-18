"""
DataLabBench v1 — question set + machine-readable scoring rubrics
=================================================================

15 questions distilled from "Suggested science-user prompts for an LLM + Astro
Data Lab MCP server" (the source PDF), ordered by increasing complexity across
7 tiers. Every question carries a 100-point rubric of weighted CHECKPOINTS.

Checkpoint types
----------------
- ``auto``  : scored deterministically against the captured evidence
              (tool-call trace with arguments, extracted SQL, final response
              text, emitted images). A checkpoint holds a list of CHECKS; the
              earned credit is ``points * passed_checks / total_checks`` —
              this is where step-level partial credit comes from.
- ``judge`` : scored by an LLM judge which returns a fractional credit in
              [0, 1] for the checkpoint, guided by ``guidance``.

Check DSL (kind → semantics)
----------------------------
- tool_called   {tools:[...]}                       any listed tool was called
- tool_ok       {tools:[...]}                       any listed tool was called AND succeeded
- tool_arg      {tools:[...], arg, equals|contains|one_of|approx+tol|min|max}
- position_near {ra, dec, tol_deg}                  any tool call (ra/dec, sia_ra/sia_dec,
                                                    peaks[], tiled bbox center) or a
                                                    q3c_radial_query(...) in captured SQL
                                                    is within tol of the target
- radius_near   {value, tol}                        radius_deg / fov_deg / tile_radius_deg arg,
                                                    or the q3c_radial_query radius, near value
- sql_regex     {pattern}                           case-insensitive regex over EXECUTED SQL
                                                    (successful DATA LAB tool calls only —
                                                    SQL pasted into the answer text or into a
                                                    non-SQL tool's `query` arg never counts)
- args_regex    {pattern}                           regex over JSON-serialized tool arguments
                                                    of ALL calls (intent, incl. failed calls)
- trace_regex   {pattern}                           regex over executed evidence only:
                                                    successful-call args + executed SQL, with
                                                    narrative fields (reason/title/...) scrubbed.
                                                    Use for "the cut was actually APPLIED".
- text_regex    {pattern}                           regex over the final response text
- any_regex     {pattern}                           regex over SQL + args + text combined
                                                    (gameable by prose — avoid for credit)
- image_emitted {}                                  at least one image/plot was produced
- image_count   {min}                               at least ``min`` images were produced
- any / all     {of:[subchecks]}                    boolean composition

Penalties
---------
Global penalties (GLOBAL_PENALTIES) apply to every question unless the
question lists the penalty id in ``disable_penalties``. Question-specific
``penalties`` add to them. A penalty subtracts points AFTER checkpoints are
summed; scores floor at 0.

Provenance
----------
``reference_sql`` / ``reference_actions`` are copied from the source PDF and
are the ground-truth solution sketch. Coordinates, thresholds and LIMITs are
illustrative per the PDF; rubric tolerances account for that.
"""

BENCH_NAME = "DataLabBench"
# v1.1 (2026-07-17): retroactively acknowledges the rubric hardening that
# landed between 2026-07-02 and 2026-07-13 under an unchanged "1.0" tag
# (any_regex→trace_regex conversions, lenient alternatives removed, GP-02
# rewritten to sql_flat_q3c_join) — scores from runs before/after that window
# are NOT comparable. v1.1 additionally: claim-shaped GP-00 gate (honest
# outage reports no longer dock 20), same-statement scoping for the DLB-10
# C4 / DLB-12+13 C7 row-cap checks, per-statement GP-02 splitting, penalty
# scaling in auto-only scores, and fallback-trace ok derivation (harness).
#
# v1.2 (2026-07-18, R6/R3): ADDITIVE reporting axes only — per-question
# scores are computed exactly as in v1.1 (same checkpoints, same points,
# same penalties). New: (a) every checkpoint maps to a ReplicationBench-style
# axis — "faithfulness" (did the agent follow the task/approach as asked) or
# "correctness" (is the produced content/number right) — via
# axis_for_checkpoint(); (b) a fabrication axis aggregating GP-00 hits and
# zero-credit on fabrication-guarded judge checkpoints; (c) the harness
# records the backend's mechanical citation recall/precision metrics
# (citation_metrics SSE event, R3) as informational per-question fields.
# v1.1 and v1.2 total scores remain directly comparable; the axes are new
# columns, not a rescoring.
BENCH_VERSION = "1.2"

# ---------------------------------------------------------------------------
# v1.2 axis mapping (R6)
# ---------------------------------------------------------------------------
# Default rule: auto checkpoints verify PROCESS (right tools, right query
# shape, right region) → faithfulness; judge checkpoints grade CONTENT
# (correct science, correct numbers, correct plots) → correctness.
# AXIS_OVERRIDES lists the judge checkpoints that actually grade
# task/approach compliance (answer form, region choice, strategy design,
# execution-model awareness) rather than content.
CHECKPOINT_AXES = ("faithfulness", "correctness")
DEFAULT_AXIS_BY_TYPE = {"auto": "faithfulness", "judge": "correctness"}
AXIS_OVERRIDES = {
    "DLB-02": {"C5": "faithfulness"},   # answer-form compliance (a plain count)
    "DLB-04": {"C5": "faithfulness"},   # explicit FOV reasoning (approach)
    "DLB-05": {"C7": "faithfulness"},   # quality cuts applied (methodology)
    "DLB-07": {"C6": "faithfulness"},   # method explained (binning/kernel)
    "DLB-08": {"C4": "faithfulness"},   # bounded-region task compliance
    "DLB-09": {"C7": "faithfulness"},   # q3c execution-model awareness
    "DLB-11": {"C7": "faithfulness"},   # bitmask semantics explained
    "DLB-12": {"C7": "faithfulness"},   # vetting guidance / honest coverage
    "DLB-13": {"C3": "faithfulness"},   # region-choice task compliance
    "DLB-14": {"C8": "faithfulness"},   # coherent end-to-end narrative
    "DLB-15": {"C1": "faithfulness"},   # strategy designed before executing
}

# Judge checkpoints whose guidance makes fabrication a zero — derived
# mechanically from the guidance text so the list can never drift from the
# rubric wording.
_FABRICATION_GUARD_RE = None  # compiled lazily below (re imported at module use)


def axis_for_checkpoint(question_id: str, checkpoint: dict) -> str:
    """Return the v1.2 reporting axis for one checkpoint (additive, R6)."""
    override = AXIS_OVERRIDES.get(question_id, {}).get(checkpoint.get("id"))
    if override:
        return override
    return DEFAULT_AXIS_BY_TYPE.get(checkpoint.get("type", "auto"), "faithfulness")


def fabrication_guard_ids(question: dict) -> list:
    """Checkpoint ids whose guidance zeroes fabricated content."""
    import re as _re
    global _FABRICATION_GUARD_RE
    if _FABRICATION_GUARD_RE is None:
        _FABRICATION_GUARD_RE = _re.compile(
            r"fabricat|invented|not appear in any tool output|"
            r"consistent with the tool output",
            _re.IGNORECASE,
        )
    return [
        cp["id"] for cp in question.get("checkpoints", [])
        if cp.get("type") == "judge"
        and _FABRICATION_GUARD_RE.search(cp.get("guidance") or "")
    ]

# Tier → weight used in the overall roll-up (higher tiers count more).
TIER_WEIGHTS = {1: 1.0, 2: 1.2, 3: 1.4, 4: 1.6, 5: 1.8, 6: 2.0, 7: 2.4}

TIER_LABELS = {
    1: "Discovery & single trivial query",
    2: "Single-survey selection / one image",
    3: "Quality cuts, classification, kinematics",
    4: "Spatial structure (matched filter, density maps)",
    5: "Cross-survey combination (server-side TAP joins)",
    6: "Full TAP + SIA combination workflows",
    7: "Open-ended research (LLM designs the whole strategy)",
}

# Tools whose success signals a real Data Lab interaction (used by GP-00).
DATALAB_QUERY_TOOLS = [
    "datalab_list_catalogs", "datalab_describe_table", "datalab_cone_count",
    "datalab_select_catalog_rows", "datalab_density_aggregate",
    "datalab_q3c_crossmatch", "datalab_sql_query", "datalab_get_result",
    "datalab_image_cutout", "datalab_color_image", "datalab_cutout_grid",
    "svo_filter_wavelength", "datalab_catalog_scatter",
    "datalab_sky_density_map", "datalab_period_fold", "datalab_sed_plot",
    "datalab_lss_wedge", "datalab_density_vetting",
    "datalab_color_color_diagram", "datalab_color_magnitude_diagram",
    "datalab_tiled_search", "datalab_confirm_sky_area",
    "datalab_job_status", "datalab_job_results",
]

# ---------------------------------------------------------------------------
# Global penalties (PDF "Query-size guardrails" + anti-fabrication)
# ---------------------------------------------------------------------------

GLOBAL_PENALTIES = [
    {
        "id": "GP-00",
        "desc": "Fabricated result: the answer presents concrete results but NO "
                "Data Lab tool call succeeded during the run.",
        "points": 20,
        # Triggers when NO datalab tool succeeded. The scorer negates tool_ok.
        "detect": {"kind": "not", "of": {"kind": "tool_ok", "tools": DATALAB_QUERY_TOOLS}},
        # v1.1: gate on CLAIM-shaped numbers, not any digit — an honest
        # "Data Lab is unreachable (HTTP 503), could not run the cone at
        # RA 229.02" echoes digits without presenting results, and docking it
        # -20 punished exactly the honest degradation the bench rewards.
        "only_if_text": (
            r"(?i)(?:there\s+are|found|returned|retrieved|yields?|contains?|"
            r"total(?:s|ing)?\s*(?:of|:|=)?|count(?:s)?\s*(?:of|:|=|is))\s*[\d,]+"
            r"|\b[\d,]{1,12}\s+(?:sources?|objects?|rows?|stars?|galaxies|"
            r"quasars?|candidates?|matches|detections?|entries|spectra)\b"
            r"|\|\s*-?[\d.,eE+]+\s*\|"  # a numeric markdown-table cell
        ),
    },
    {
        "id": "GP-01",
        "desc": "Unbounded row-level scan: executed SQL selects rows from a "
                "catalog with no q3c spatial bound and no key-equality filter "
                "(guardrail #1 — a LIMIT alone does not make the scan cheap; "
                "row-cap discipline is scored separately as positive criteria).",
        "points": 10,
        "detect": {"kind": "sql_unbounded_rowscan"},
    },
    {
        "id": "GP-02",
        "desc": "q3c_join anti-pattern: a statement with a flat q3c_join and no "
                "MATERIALIZED CTE reduction of the small side (guardrail #2 — "
                "the exact anti-example the PDF says never to run). Evaluated "
                "per executed statement.",
        "points": 15,
        "detect": {"kind": "sql_flat_q3c_join"},
    },
    {
        "id": "GP-03",
        "desc": "ra/dec BETWEEN box used as the ONLY spatial bound of a row-level "
                "(non-aggregate) query — the q3c functional index cannot serve it "
                "(guardrail #1).",
        "points": 8,
        "detect": {"kind": "sql_between_rowscan"},
    },
]

# ---------------------------------------------------------------------------
# The 15 questions
# ---------------------------------------------------------------------------

QUESTIONS = [

    # ══════════════════════════ TIER 1 ══════════════════════════
    {
        "id": "DLB-01",
        "tier": 1,
        "title": "Catalog discovery (LMC + near-infrared)",
        "prompt": (
            "Which Data Lab catalogs cover the Large Magellanic Cloud, and "
            "which of those include near-infrared photometry? List the "
            "relevant table names."
        ),
        "exercises": "Dataset/schema metadata (no row query). Candidates: nsc_dr2, smash_dr2, vhs_dr5, gaia_dr3.",
        "reference_actions": "Call schema/metadata listing (qc.schema / dataset catalog), reason over coverage + bandpasses, return a curated table list.",
        "reference_sql": None,
        "checkpoints": [
            {
                "id": "C1", "type": "auto", "points": 20,
                "desc": "Used schema/metadata tooling instead of answering purely from memory",
                "checks": [
                    {"kind": "tool_called", "tools": ["datalab_list_catalogs", "datalab_describe_table"]},
                    {"kind": "tool_ok", "tools": ["datalab_list_catalogs", "datalab_describe_table"]},
                ],
            },
            {
                "id": "C2", "type": "auto", "points": 20,
                "desc": "Names the candidate LMC catalogs (VHS + at least one of SMASH/NSC/Gaia)",
                "checks": [
                    {"kind": "text_regex", "pattern": r"vhs[\s_-]?dr?5|vista hemisphere"},
                    {"kind": "text_regex", "pattern": r"smash|nsc[\s_-]?dr?2|gaia[\s_-]?dr?3"},
                ],
            },
            {
                "id": "C3", "type": "judge", "points": 25,
                "desc": "Near-infrared identification is correct",
                "guidance": "Full credit only if VHS (J/H/Ks bands) is identified as the near-infrared catalog. "
                            "Deduct if NIR capability is wrongly attributed to optical-only surveys "
                            "(SMASH ugriz, NSC, DES grizY is only barely NIR at Y — attributing NIR to Gaia G/BP/RP is wrong). "
                            "Half credit if VHS named but bands not stated.",
            },
            {
                "id": "C4", "type": "judge", "points": 20,
                "desc": "Lists concrete TABLE names, not just catalog/schema names",
                "guidance": "Full credit for real table names (e.g. vhs_dr5.vhs_cat_v3, smash_dr2.object, "
                            "nsc_dr2.object, gaia_dr3.gaia_source). Half credit if only schemas listed. "
                            "Zero if tables are invented (verify against the tool outputs in the trace).",
            },
            {
                "id": "C5", "type": "judge", "points": 15,
                "desc": "Coverage reasoning (why these catalogs cover the LMC)",
                "guidance": "Credit for correctly reasoning about footprints: SMASH targets the Magellanic "
                            "system, VHS covers the southern hemisphere incl. LMC, Gaia is all-sky, NSC is "
                            "wide-southern. Partial credit for weaker but not wrong reasoning.",
            },
        ],
    },

    {
        "id": "DLB-02",
        "tier": 1,
        "title": "Cone-search count (Gaia DR3 around Palomar 5)",
        "prompt": (
            "How many Gaia DR3 sources lie within 10 arcminutes of Palomar 5 "
            "(RA = 229.022, Dec = −0.112)?"
        ),
        "exercises": "gaia_dr3.gaia_source via TAP; q3c_radial_query.",
        "reference_actions": "One COUNT(*) with q3c_radial_query; return the number.",
        "reference_sql": (
            "SELECT COUNT(*) FROM gaia_dr3.gaia_source "
            "WHERE q3c_radial_query(ra, dec, 229.022, -0.112, 10.0/60.0)"
        ),
        "checkpoints": [
            {
                "id": "C1", "type": "auto", "points": 20,
                "desc": "Queried gaia_dr3.gaia_source",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "tool_arg", "tools": [], "arg": "catalog", "equals": "gaia_dr3"},
                        {"kind": "sql_regex", "pattern": r"gaia_dr3\.gaia_source"},
                    ]},
                ],
            },
            {
                "id": "C2", "type": "auto", "points": 20,
                "desc": "Cone centered on Palomar 5 (229.022, -0.112)",
                "checks": [{"kind": "position_near", "ra": 229.022, "dec": -0.112, "tol_deg": 0.05}],
            },
            {
                "id": "C3", "type": "auto", "points": 15,
                "desc": "Radius = 10 arcmin converted correctly to 0.1667 deg",
                "checks": [{"kind": "radius_near", "value": 0.166667, "tol": 0.005}],
            },
            {
                "id": "C4", "type": "auto", "points": 15,
                "desc": "Count computed server-side (COUNT(*) / cone-count tool), not by pulling rows",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "tool_ok", "tools": ["datalab_cone_count"]},
                        {"kind": "sql_regex", "pattern": r"COUNT\s*\("},
                    ]},
                ],
            },
            {
                "id": "C5", "type": "judge", "points": 15,
                "desc": "A specific integer count is reported as THE answer",
                "guidance": "Full credit if a single concrete source count is stated plainly. "
                            "Half if buried/hedged. Zero if no number or only a code sketch.",
            },
            {
                "id": "C6", "type": "judge", "points": 15,
                "desc": "Reported number is faithful to the tool output",
                "guidance": "Compare the stated count against reported_count / rowcount in the tool trace. "
                            "Full credit iff they match exactly. Zero if the number does not appear in any "
                            "tool output (fabrication).",
            },
        ],
    },

    # ══════════════════════════ TIER 2 ══════════════════════════
    {
        "id": "DLB-03",
        "tier": 2,
        "title": "Color–magnitude diagram (Draco dwarf, NSC DR2)",
        "prompt": (
            "Get g and r magnitudes for point sources within 0.4° of the Draco "
            "dwarf (RA = 260.06, Dec = +57.92) from NSC DR2 and plot a g vs "
            "(g−r) CMD."
        ),
        "exercises": "nsc_dr2.object TAP; cone + star/morphology cut; scatter CMD.",
        "reference_actions": "Cone + class_star/quality cuts -> dataframe -> scatter CMD.",
        "reference_sql": (
            "SELECT gmag, rmag, gmag - rmag AS gr FROM nsc_dr2.object "
            "WHERE q3c_radial_query(ra, dec, 260.06, 57.92, 0.4) "
            "AND class_star > 0.5 AND gmag < 24 AND rmag < 24"
        ),
        "checkpoints": [
            {
                "id": "C1", "type": "auto", "points": 15,
                "desc": "Queried nsc_dr2.object",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "tool_arg", "tools": [], "arg": "catalog", "equals": "nsc_dr2"},
                        {"kind": "sql_regex", "pattern": r"nsc_dr2\.object"},
                    ]},
                ],
            },
            {
                "id": "C2", "type": "auto", "points": 15,
                "desc": "Cone at Draco (260.06, +57.92) with radius 0.4 deg",
                "checks": [
                    {"kind": "position_near", "ra": 260.06, "dec": 57.92, "tol_deg": 0.05},
                    {"kind": "radius_near", "value": 0.4, "tol": 0.05},
                ],
            },
            {
                "id": "C3", "type": "auto", "points": 15,
                "desc": "Point-source (morphology) cut actually applied in the executed query",
                "checks": [{"kind": "trace_regex", "pattern": r"class_star"}],
            },
            {
                "id": "C4", "type": "auto", "points": 20,
                "desc": "A CMD plot was actually produced",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "tool_ok", "tools": ["datalab_color_magnitude_diagram"]},
                        {"kind": "all", "of": [
                            {"kind": "tool_ok", "tools": ["datalab_select_catalog_rows"]},
                            {"kind": "tool_ok", "tools": ["datalab_catalog_scatter"]},
                        ]},
                    ]},
                    {"kind": "image_emitted"},
                ],
            },
            {
                "id": "C5", "type": "judge", "points": 20,
                "desc": "CMD is correctly constructed (g vs g-r, magnitude axis inverted, g/r from NSC)",
                "guidance": "Full credit: y-axis g magnitude (inverted, bright up), x-axis g-r color. "
                            "Half credit for axes swapped or inversion missing/unstated. "
                            "Check tool args (mag_band/blue_band/red_band or x_expr/y_expr/invert_y).",
            },
            {
                "id": "C6", "type": "judge", "points": 15,
                "desc": "Sensible depth/quality handling and interpretation",
                "guidance": "Credit for magnitude limits (~g,r < 24) or equivalent quality bounds, and for "
                            "any correct interpretation (Draco's old population / MSTO / RGB visible). "
                            "Partial credit if only one of the two.",
            },
        ],
    },

    {
        "id": "DLB-04",
        "tier": 2,
        "title": "Single color cutout (center of M31)",
        "prompt": "Show me a color image of the center of M31 from the DECam Legacy Surveys.",
        "exercises": "SIA service (coadd_all), 3-band cutout; FOV choice is a decision.",
        "reference_actions": "Resolve M31 -> choose modest FOV (~0.1-0.2 deg for 'center'; D25~3 deg) -> "
                             "dec-corrected size -> deepest Stack/image per band -> Lupton RGB. "
                             "NB: M31 at dec +41 may have little/no DECam coadd coverage — "
                             "0 rows is a coverage gap, not a code bug.",
        "reference_sql": None,
        "checkpoints": [
            {
                "id": "C1", "type": "auto", "points": 20,
                "desc": "Used the SIA color-image path (not just text)",
                "checks": [
                    {"kind": "tool_called", "tools": ["datalab_color_image", "datalab_image_cutout"]},
                ],
            },
            {
                "id": "C2", "type": "auto", "points": 15,
                "desc": "Pointed at M31 (10.6847, +41.2687) or resolved it by name "
                        "(decision check: a coverage-gap failure must not zero it)",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "position_near", "ra": 10.6847, "dec": 41.2687, "tol_deg": 0.3,
                         "include_failed": True},
                        {"kind": "args_regex", "pattern": r"m\s?31|andromeda"},
                    ]},
                ],
            },
            {
                "id": "C3", "type": "auto", "points": 15,
                "desc": "FOV chosen for the CENTER (modest, ~0.05-0.5 deg; not the whole 3-deg disk; "
                        "decision check: counts even if the SIA search finds no coverage)",
                "checks": [
                    {"kind": "tool_arg", "tools": ["datalab_color_image", "datalab_image_cutout"],
                     "arg": "fov_deg", "min": 0.02, "max": 0.5, "include_failed": True},
                ],
            },
            {
                "id": "C4", "type": "judge", "points": 30,
                "desc": "Correct outcome handling: a color image OR an honest coverage-gap explanation",
                "guidance": "Full credit if a color image of M31's center is rendered, OR if the search "
                            "returned no DECam coverage and the answer explains it as a coverage gap "
                            "(M31 at dec +41 is at the northern edge of DECam surveys) and offers an "
                            "alternative. ZERO credit if it claims an image that no tool produced, or "
                            "blames its own code for what is a coverage gap.",
            },
            {
                "id": "C5", "type": "judge", "points": 10,
                "desc": "FOV reasoning is explicit (center vs. D25 ~ 3 deg)",
                "guidance": "Full credit if the answer justifies the FOV from M31's apparent size / the "
                            "'center' intent. Half credit for a sensible FOV with no reasoning.",
            },
            {
                "id": "C6", "type": "judge", "points": 10,
                "desc": "Color composition handled correctly",
                "guidance": "Credit for correct RGB band assignment (red=i/z, green=r, blue=g) or "
                            "for the tool's auto-selection being described; Lupton stretch parameters "
                            "are a plus. No credit if bands are scrambled.",
            },
        ],
    },

    # ══════════════════════════ TIER 3 ══════════════════════════
    {
        "id": "DLB-05",
        "tier": 3,
        "title": "Star/galaxy separation + color-color (DES DR1)",
        "prompt": (
            "Using DES DR1 around RA = 30, Dec = −50, separate stars from "
            "galaxies and show me g−r vs r−i color–color diagrams for each "
            "population."
        ),
        "exercises": "des_dr1.main TAP; spread_model_r morphology; two-panel CCD.",
        "reference_actions": "Query with morphology flag + colors -> two-panel color-color diagram.",
        "reference_sql": (
            "SELECT mag_auto_g - mag_auto_r AS gr, mag_auto_r - mag_auto_i AS ri, "
            "CASE WHEN spread_model_r > 0.003 THEN 'galaxy' ELSE 'star' END AS morph "
            "FROM des_dr1.main WHERE q3c_radial_query(ra, dec, 30.0, -50.0, 0.5) "
            "AND mag_auto_i BETWEEN 16 AND 23 AND flags_g = 0 AND flags_r = 0 AND flags_i = 0 "
            "AND fluxerr_auto_g > 0 AND fluxerr_auto_r > 0 AND fluxerr_auto_i > 0"
        ),
        "checkpoints": [
            {
                "id": "C1", "type": "auto", "points": 15,
                "desc": "Queried des_dr1.main",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "tool_arg", "tools": [], "arg": "catalog", "equals": "des_dr1"},
                        {"kind": "sql_regex", "pattern": r"des_dr1\.main"},
                    ]},
                ],
            },
            {
                "id": "C2", "type": "auto", "points": 10,
                "desc": "Cone near (30, -50)",
                "checks": [{"kind": "position_near", "ra": 30.0, "dec": -50.0, "tol_deg": 0.3}],
            },
            {
                "id": "C3", "type": "auto", "points": 20,
                "desc": "Morphological star/galaxy split via spread_model (in the executed query/tool args)",
                "checks": [
                    {"kind": "trace_regex", "pattern": r"spread_model"},
                    {"kind": "trace_regex", "pattern": r"0\.00[2-9]"},
                ],
            },
            {
                "id": "C4", "type": "auto", "points": 15,
                "desc": "Both colors formed: g-r AND r-i",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "tool_ok", "tools": ["datalab_color_color_diagram"]},
                        {"kind": "all", "of": [
                            {"kind": "sql_regex", "pattern": r"g\s*-\s*(mag_auto_)?r|mag_auto_g\s*-\s*mag_auto_r"},
                            {"kind": "sql_regex", "pattern": r"r\s*-\s*(mag_auto_)?i|mag_auto_r\s*-\s*mag_auto_i"},
                        ]},
                    ]},
                ],
            },
            {
                "id": "C5", "type": "auto", "points": 10,
                "desc": "A color-color plot was rendered",
                "checks": [{"kind": "image_emitted"}],
            },
            {
                "id": "C6", "type": "judge", "points": 15,
                "desc": "Stars and galaxies shown as SEPARATE populations (two panels or clearly split)",
                "guidance": "Full credit for a two-panel CCD (stars | galaxies) or an explicit split. "
                            "Half if both are in one panel but distinguishable. Zero if populations mixed.",
            },
            {
                "id": "C7", "type": "judge", "points": 15,
                "desc": "Photometric quality cuts applied or discussed",
                "guidance": "Credit for flags_g/r/i = 0, fluxerr_auto_* > 0, and a magnitude window "
                            "(~16 < i < 23) — or the structured tool's documented equivalents. "
                            "Partial credit proportional to how many of the three appear.",
            },
        ],
    },

    {
        "id": "DLB-06",
        "tier": 3,
        "title": "Proper-motion + parallax selection (Gaia white dwarfs)",
        "prompt": (
            "Find high-proper-motion white-dwarf candidates in Gaia DR3 in a "
            "5°-radius patch of the southern sky (say around RA = 60, Dec = −50): "
            "significant parallax, large total proper motion, and absolute "
            "magnitudes on the WD sequence. Give me the HR diagram."
        ),
        "exercises": "gaia_dr3.gaia_source; pm, bp_rp, parallax, astrometric quality; HR diagram.",
        "reference_actions": "Cone + real Gaia astrometric-quality cuts -> absolute mag -> HR diagram with WD locus.",
        "reference_sql": (
            "SELECT bp_rp, phot_g_mean_mag + 5*LOG10(parallax) - 10 AS abs_g, parallax, pm "
            "FROM gaia_dr3.gaia_source WHERE q3c_radial_query(ra, dec, 60.0, -50.0, 5.0) "
            "AND parallax_over_error > 4 AND parallax > 0.25 AND ruwe < 1.4 "
            "AND ipd_frac_multi_peak <= 2 AND astrometric_sigma5d_max < 1.5 "
            "AND pm > 100 AND phot_g_mean_mag + 5*LOG10(parallax) - 10 > 10"
        ),
        "checkpoints": [
            {
                "id": "C1", "type": "auto", "points": 15,
                "desc": "Gaia DR3 cone at (60, -50) with ~5 deg radius",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "tool_arg", "tools": [], "arg": "catalog", "equals": "gaia_dr3"},
                        {"kind": "sql_regex", "pattern": r"gaia_dr3\.gaia_source"},
                    ]},
                    {"kind": "position_near", "ra": 60.0, "dec": -50.0, "tol_deg": 1.0},
                    {"kind": "radius_near", "value": 5.0, "tol": 1.0},
                ],
            },
            {
                "id": "C2", "type": "auto", "points": 20,
                "desc": "Real astrometric-quality + kinematic cuts applied in the executed query",
                "checks": [
                    {"kind": "trace_regex", "pattern": r"parallax_over_error"},
                    {"kind": "trace_regex", "pattern": r"ruwe"},
                    {"kind": "trace_regex", "pattern": r"\bpm\b.{0,12}(>|&gt;).{0,6}\d|pm\s*>\s*\d+"},
                ],
            },
            {
                "id": "C3", "type": "auto", "points": 15,
                "desc": "Absolute magnitude derived from parallax in the executed query/plot expression",
                "checks": [
                    {"kind": "trace_regex", "pattern": r"5\s*\*?\s*log10\s*\(\s*parallax|abs_g"},
                ],
            },
            {
                "id": "C4", "type": "auto", "points": 10,
                "desc": "An HR diagram was plotted",
                "checks": [
                    {"kind": "image_emitted"},
                    {"kind": "any", "of": [
                        {"kind": "tool_ok", "tools": ["datalab_catalog_scatter", "datalab_color_magnitude_diagram"]},
                        {"kind": "text_regex", "pattern": r"HR diagram|Hertzsprung"},
                    ]},
                ],
            },
            {
                "id": "C5", "type": "judge", "points": 20,
                "desc": "White-dwarf selection is physically sound",
                "guidance": "Full credit: positive significant parallax, LARGE total PM (~>100 mas/yr), "
                            "and a faint absolute-magnitude cut (abs G > ~10) placing candidates on the "
                            "WD sequence — all three present. Deduct ~1/3 for each missing/wrong element.",
            },
            {
                "id": "C6", "type": "judge", "points": 20,
                "desc": "HR diagram correct + WD locus highlighted/interpreted",
                "guidance": "Full credit: bp_rp on x, absolute G on y (inverted), WD sequence below the "
                            "main sequence identified or highlighted. Half for a correct plot without "
                            "WD identification.",
            },
        ],
    },

    # ══════════════════════════ TIER 4 ══════════════════════════
    {
        "id": "DLB-07",
        "tier": 4,
        "title": "Overdensity hunt / matched filter (SMASH field 169)",
        "prompt": (
            "Help me look for a stellar overdensity — a possible dwarf companion "
            "— in SMASH DR1 field 169. Select blue main-sequence stars and find "
            "where they clump on the sky."
        ),
        "exercises": "smash_dr1.object; color box; 2D density / Mexican-hat convolution. Field 169 is the Hydra II field.",
        "reference_actions": "Blue stars over the field -> 2D histogram + difference-of-Gaussians filter -> peak coords + significance.",
        "reference_sql": (
            "SELECT ra, dec, gmag, gmag - rmag AS gr FROM smash_dr1.object "
            "WHERE fieldid = 169 AND depthflag > 1 AND ABS(sharp) < 0.5 "
            "AND gmag - rmag BETWEEN -0.5 AND 0.5 AND gmag BETWEEN 9 AND 25"
        ),
        "checkpoints": [
            {
                "id": "C1", "type": "auto", "points": 15,
                "desc": "SMASH DR1 object table, field 169 (or the Hydra II cone)",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "tool_arg", "tools": [], "arg": "catalog", "equals": "smash_dr1"},
                        {"kind": "sql_regex", "pattern": r"smash_dr1\.object"},
                    ]},
                    {"kind": "any", "of": [
                        {"kind": "trace_regex", "pattern": r"fieldid.{0,12}169"},
                        {"kind": "position_near", "ra": 185.43, "dec": -31.99, "tol_deg": 0.5},
                    ]},
                ],
            },
            {
                "id": "C2", "type": "auto", "points": 15,
                "desc": "Stellar morphology + blue main-sequence color box in the executed query",
                "checks": [
                    {"kind": "trace_regex", "pattern": r"sharp|class_star"},
                    {"kind": "trace_regex", "pattern": r"-\s*0\.5"},
                ],
            },
            {
                "id": "C3", "type": "auto", "points": 15,
                "desc": "Sky-density analysis actually run (server-side aggregate or density tool)",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "tool_ok", "tools": ["datalab_density_aggregate", "datalab_density_vetting"]},
                        {"kind": "sql_regex", "pattern": r"GROUP\s+BY"},
                    ]},
                ],
            },
            {
                "id": "C4", "type": "auto", "points": 10,
                "desc": "Matched-filter / peak detection actually executed",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "tool_arg", "tools": ["datalab_sky_density_map"], "arg": "matched_filter", "equals": True},
                        {"kind": "trace_regex", "pattern": r"matched_filter|peak_threshold|sigma_(small|large)"},
                    ]},
                ],
            },
            {
                "id": "C5", "type": "judge", "points": 25,
                "desc": "Recovers the Hydra II overdensity with concrete peak coordinates",
                "guidance": "Full credit if a peak is reported within ~0.15 deg of (185.43, -31.99) "
                            "(Hydra II) with some significance measure. Half credit for a peak list "
                            "that contains it less prominently, or coordinates without significance. "
                            "Zero if no concrete peak position is reported.",
            },
            {
                "id": "C6", "type": "judge", "points": 20,
                "desc": "Method quality: binning/kernel/significance explained; density map shown",
                "guidance": "Credit for a rendered density map, a sensible bin size (arcmin-scale), a "
                            "smoothing/matched-filter justification, and a significance estimate. "
                            "Score proportionally.",
            },
        ],
    },

    {
        "id": "DLB-08",
        "tier": 4,
        "title": "HEALPix stellar-density map (NSC DR2, Milky Way structure)",
        "prompt": (
            "Make a stellar density map of a ~20° × 20° region from NSC DR2 to "
            "reveal Milky Way structure — bin by HEALPix and show log counts."
        ),
        "exercises": "nsc_dr2 TAP aggregate; precomputed HEALPix column (ring256/nest4096) + COUNT.",
        "reference_actions": "GROUP BY the HEALPix index server-side (one row per pixel), bounded footprint -> healpy map, log10 counts.",
        "reference_sql": (
            "SELECT ring256, AVG(ra) AS ra0, AVG(dec) AS dec0, COUNT(ring256) AS nb "
            "FROM nsc_dr2.object WHERE ra BETWEEN 70 AND 90 AND dec BETWEEN -70 AND -50 "
            "AND class_star > 0.5 GROUP BY ring256"
        ),
        # The reference itself uses a BETWEEN box on an AGGREGATE — that is the
        # notebook idiom and must not be penalized here.
        "disable_penalties": ["GP-03"],
        "checkpoints": [
            {
                "id": "C1", "type": "auto", "points": 15,
                "desc": "NSC DR2 aggregated server-side",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "tool_arg", "tools": [], "arg": "catalog", "equals": "nsc_dr2"},
                        {"kind": "sql_regex", "pattern": r"nsc_dr2\.object"},
                    ]},
                    {"kind": "any", "of": [
                        {"kind": "tool_ok", "tools": ["datalab_density_aggregate"]},
                        {"kind": "sql_regex", "pattern": r"GROUP\s+BY"},
                    ]},
                ],
            },
            {
                "id": "C2", "type": "auto", "points": 20,
                "desc": "HEALPix binning via a precomputed column, in the executed aggregate",
                "checks": [
                    {"kind": "trace_regex", "pattern": r"ring256|nest4096|healpix"},
                ],
            },
            {
                "id": "C3", "type": "auto", "points": 15,
                "desc": "Aggregate returns one row per pixel (COUNT + GROUP BY / healpix mode), not raw rows",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "tool_arg", "tools": ["datalab_density_aggregate"], "arg": "mode", "equals": "healpix"},
                        {"kind": "all", "of": [
                            {"kind": "sql_regex", "pattern": r"COUNT\s*\("},
                            {"kind": "sql_regex", "pattern": r"GROUP\s+BY"},
                        ]},
                    ]},
                ],
            },
            {
                "id": "C4", "type": "judge", "points": 15,
                "desc": "Region is a bounded ~20x20 deg footprint (not all-sky, not a tiny cone)",
                "guidance": "Full credit for a defined ~400 deg^2 region (e.g. 70<ra<90, -70<dec<-50, or a "
                            "~10 deg-radius cone). Half for a materially smaller/larger but still bounded "
                            "region. Zero for an unbounded all-sky scan.",
            },
            {
                "id": "C5", "type": "auto", "points": 10,
                "desc": "A density map was rendered",
                "checks": [
                    {"kind": "image_emitted"},
                    {"kind": "tool_ok", "tools": ["datalab_sky_density_map"]},
                ],
            },
            {
                "id": "C6", "type": "judge", "points": 15,
                "desc": "Log-scaled counts + true HEALPix pixel geometry",
                "guidance": "Full credit: log10(counts) color scale AND a real HEALPix map (healpy-style, "
                            "pixel shapes) rather than a scatter of pixel centers. Half for one of the two.",
            },
            {
                "id": "C7", "type": "judge", "points": 10,
                "desc": "Milky Way structure interpretation",
                "guidance": "Credit for pointing at visible structure (disk gradient, LMC/SMC if in view, "
                            "globular clusters, dust lanes). Brief but correct = full.",
            },
        ],
    },

    # ══════════════════════════ TIER 5 ══════════════════════════
    {
        "id": "DLB-09",
        "tier": 5,
        "title": "Stream tracing via crossmatch (Palomar 5 tidal tails)",
        "prompt": (
            "Combine NSC DR2 photometry with Gaia DR3 proper motions around "
            "Palomar 5 to trace its tidal tails — select stream stars by proper "
            "motion and CMD, then plot their on-sky distribution."
        ),
        "exercises": "nsc_dr2.object x gaia_dr3.gaia_source via q3c_join; PM cut + CMD mask. "
                     "Must follow q3c's execution model (materialize small side; indexed side second).",
        "reference_actions": "Reduce Gaia (cone + PM cut) in a MATERIALIZED CTE -> q3c_join against NSC "
                             "(indexed side) -> CMD mask -> on-sky scatter.",
        "reference_sql": (
            "WITH g AS MATERIALIZED (SELECT ra, dec, pmra, pmdec, pm FROM gaia_dr3.gaia_source "
            "WHERE q3c_radial_query(ra, dec, 229.022, -0.112, 0.5) AND pm < 5) "
            "SELECT n.ra, n.dec, n.gmag, n.gmag - n.rmag AS gr, g.pmra, g.pmdec "
            "FROM g, nsc_dr2.object AS n WHERE q3c_join(g.ra, g.dec, n.ra, n.dec, 1.0/3600.0)"
        ),
        "checkpoints": [
            {
                "id": "C1", "type": "auto", "points": 20,
                "desc": "Planner-safe crossmatch executed (builder tool OR MATERIALIZED-CTE SQL)",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "tool_ok", "tools": ["datalab_q3c_crossmatch"]},
                        {"kind": "all", "of": [
                            {"kind": "sql_regex", "pattern": r"MATERIALIZED"},
                            {"kind": "sql_regex", "pattern": r"q3c_join\s*\("},
                        ]},
                    ]},
                ],
            },
            {
                "id": "C2", "type": "auto", "points": 15,
                "desc": "Small side is Gaia, big indexed side is NSC (correct join orientation)",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "all", "of": [
                            {"kind": "tool_arg", "tools": ["datalab_q3c_crossmatch"], "arg": "small_catalog", "equals": "gaia_dr3"},
                            {"kind": "tool_arg", "tools": ["datalab_q3c_crossmatch"], "arg": "big_catalog", "equals": "nsc_dr2"},
                        ]},
                        {"kind": "sql_regex", "pattern": r"WITH\s+\w+\s+AS\s+MATERIALIZED[\s\S]{0,200}gaia_dr3"},
                    ]},
                ],
            },
            {
                "id": "C3", "type": "auto", "points": 10,
                "desc": "Cone centered on Palomar 5",
                "checks": [{"kind": "position_near", "ra": 229.022, "dec": -0.112, "tol_deg": 0.1}],
            },
            {
                "id": "C4", "type": "auto", "points": 10,
                "desc": "Proper-motion selection applied in the executed crossmatch (Pal 5 is LOW-PM)",
                "checks": [{"kind": "trace_regex", "pattern": r"\bpm\b\s*(<|&lt;)\s*\d|pmra|pmdec"}],
            },
            {
                "id": "C5", "type": "auto", "points": 10,
                "desc": "CMD/photometric selection on the NSC side (executed query/plot expressions)",
                "checks": [{"kind": "trace_regex", "pattern": r"gmag|g\s*-\s*r"}],
            },
            {
                "id": "C6", "type": "judge", "points": 20,
                "desc": "On-sky distribution of stream candidates plotted; tidal tails discussed",
                "guidance": "Full credit: RA/Dec scatter of the selected stars with the leading/trailing "
                            "tail orientation discussed. Half: plot without interpretation, or "
                            "interpretation without plot.",
            },
            {
                "id": "C7", "type": "judge", "points": 15,
                "desc": "Execution-model awareness (why materialize the small side; why NSC is the indexed side)",
                "guidance": "Full credit if the answer (or its tool choice) reflects the q3c execution "
                            "model: reduce small side first, big catalog on the indexed side, ~1 arcsec "
                            "match radius. Half credit for a correct result with no rationale.",
            },
        ],
    },

    {
        "id": "DLB-10",
        "tier": 5,
        "title": "Multi-survey SED from photometry (Coma red galaxies)",
        "prompt": (
            "Build optical-to-mid-infrared SEDs for a small sample (a few "
            "hundred) of red galaxies within 1° of the Coma cluster (RA = 194.95, "
            "Dec = +27.98) by combining Legacy Surveys DR9 grz photometry with "
            "the survey's forced WISE (W1/W2) photometry. Give me magnitude vs "
            "wavelength."
        ),
        "exercises": "ls_dr9.tractor carries dereddened grz AND forced unWISE W1/W2 in ONE table "
                     "(no cross-catalog join!); wavelengths from the SVO Filter Profile Service.",
        "reference_actions": "Cone + extended + red cut + LIMIT -> SVO effective wavelengths (a property of "
                             "the FILTER, not memorized constants) -> per-object mag vs lambda.",
        "reference_sql": (
            "SELECT ra, dec, dered_mag_g, dered_mag_r, dered_mag_z, dered_mag_w1, dered_mag_w2 "
            "FROM ls_dr9.tractor WHERE q3c_radial_query(ra, dec, 194.95, 27.98, 1.0) "
            "AND type != 'PSF' AND snr_g > 5 AND snr_r > 5 AND snr_z > 5 "
            "AND dered_mag_g - dered_mag_r > 1.0 LIMIT 500"
        ),
        "penalties": [
            {
                "id": "QP-10a",
                "desc": "Joined LS DR9 with a separate WISE catalog even though tractor already "
                        "carries forced W1/W2 (explicit PDF guardrail).",
                "points": 10,
                "detect": {"kind": "sql_regex", "pattern": r"allwise|catwise|unwise\.\w+|join[\s\S]{0,80}wise"},
            },
        ],
        "checkpoints": [
            {
                "id": "C1", "type": "auto", "points": 15,
                "desc": "Single-table ls_dr9.tractor source (grz + forced W1/W2 together)",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "tool_arg", "tools": [], "arg": "catalog", "equals": "ls_dr9"},
                        {"kind": "sql_regex", "pattern": r"ls_dr9\.tractor"},
                    ]},
                    {"kind": "trace_regex", "pattern": r"dered_mag_w1|\bw1\b"},
                ],
            },
            {
                "id": "C2", "type": "auto", "points": 10,
                "desc": "Cone at Coma (194.95, +27.98) ~1 deg",
                "checks": [
                    {"kind": "position_near", "ra": 194.95, "dec": 27.98, "tol_deg": 0.3},
                    {"kind": "radius_near", "value": 1.0, "tol": 0.3},
                ],
            },
            {
                "id": "C3", "type": "auto", "points": 15,
                "desc": "Red, extended-source selection with S/N floor in the executed query",
                "checks": [
                    {"kind": "trace_regex", "pattern": r"type\s*(!=|<>)\s*'?PSF"},
                    {"kind": "trace_regex", "pattern": r"snr_"},
                    {"kind": "trace_regex", "pattern": r"(dered_mag_)?g\s*-\s*(dered_mag_)?r\s*(>|&gt;)"},
                ],
            },
            {
                "id": "C4", "type": "auto", "points": 10,
                "desc": "Sample capped to 'a few hundred' (LIMIT / limit arg <= 1000)",
                # v1.1: scoped to the ROW-PULL tools / same-statement SQL — an
                # unrelated tool's limit arg (datalab_list_catalogs limit=50)
                # used to satisfy this while the actual pull was uncapped.
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "tool_arg",
                         "tools": ["datalab_select_catalog_rows", "datalab_sql_query"],
                         "arg": "limit", "max": 1000},
                        {"kind": "tool_arg", "tools": ["datalab_sed_plot"],
                         "arg": "sample_n", "max": 1000},
                        {"kind": "sql_regex",
                         "pattern": r"(?is)FROM\s+\S+[^;]*LIMIT\s+\d{1,4}\b"},
                    ]},
                ],
            },
            {
                "id": "C5", "type": "auto", "points": 20,
                "desc": "Wavelengths resolved from the SVO Filter Profile Service, not hardcoded "
                        "(datalab_sed_plot is SVO-backed internally, so either tool qualifies)",
                "checks": [
                    {"kind": "tool_ok", "tools": ["svo_filter_wavelength", "datalab_sed_plot"]},
                ],
            },
            {
                "id": "C6", "type": "auto", "points": 10,
                "desc": "An SED plot was rendered",
                "checks": [{"kind": "image_emitted"}],
            },
            {
                "id": "C7", "type": "judge", "points": 20,
                "desc": "SED is correctly constructed",
                "guidance": "Full credit: magnitude vs wavelength (microns), log-x, inverted y (brighter "
                            "up), all five bands g/r/z/W1/W2 present, per-object curves. Deduct "
                            "proportionally for missing bands, linear-x over 0.4-4.6 um, or upright mag axis.",
            },
        ],
    },

    {
        "id": "DLB-11",
        "tier": 5,
        "title": "Targeting-bitmask catalog selection (DESI DR1 LRGs)",
        "prompt": (
            "From the DESI DR1 redshift catalog, select luminous red galaxies "
            "(LRG target class) between z = 0.4 and 0.8, and show their redshift "
            "distribution and sky footprint."
        ),
        "exercises": "desi_dr1.zpix + desi_target bitmask + zwarn via TAP (no spectra).",
        "reference_actions": "Bitmask AND + quality cuts; server-side GROUP BY z-bin for the histogram; "
                             "GROUP BY rounded mean_fiber_ra/dec for the footprint.",
        "reference_sql": (
            "SELECT ROUND(z::numeric, 2) AS z_bin, COUNT(*) AS n FROM desi_dr1.zpix "
            "WHERE survey = 'main' AND main_primary AND (desi_target & 1) != 0 "
            "AND zwarn = 0 AND spectype = 'GALAXY' AND z BETWEEN 0.4 AND 0.8 "
            "GROUP BY z_bin ORDER BY z_bin"
        ),
        "disable_penalties": ["GP-03"],  # z BETWEEN 0.4 AND 0.8 is not a sky box
        "checkpoints": [
            {
                "id": "C1", "type": "auto", "points": 15,
                "desc": "Queried desi_dr1.zpix",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "tool_arg", "tools": [], "arg": "catalog", "equals": "desi_dr1"},
                        {"kind": "sql_regex", "pattern": r"desi_dr1\.zpix"},
                    ]},
                ],
            },
            {
                "id": "C2", "type": "auto", "points": 20,
                "desc": "LRG bitmask + spectroscopic quality cuts in the executed query",
                "checks": [
                    {"kind": "trace_regex", "pattern": r"desi_target\s*&\s*1|desi_mask"},
                    {"kind": "trace_regex", "pattern": r"zwarn\s*=\s*0"},
                    {"kind": "trace_regex", "pattern": r"spectype\s*=\s*'?GALAXY"},
                    {"kind": "trace_regex", "pattern": r"survey\s*=\s*'?main|main_primary"},
                ],
            },
            {
                "id": "C3", "type": "auto", "points": 10,
                "desc": "Redshift window 0.4 <= z <= 0.8 in the executed query",
                "checks": [
                    {"kind": "trace_regex", "pattern": r"0\.4"},
                    {"kind": "trace_regex", "pattern": r"0\.8"},
                ],
            },
            {
                "id": "C4", "type": "auto", "points": 15,
                "desc": "Server-side aggregation (GROUP BY z-bin / sky cell), not a multi-million-row pull",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "sql_regex", "pattern": r"GROUP\s+BY"},
                        {"kind": "tool_ok", "tools": ["datalab_density_aggregate"]},
                    ]},
                ],
            },
            {
                "id": "C5", "type": "auto", "points": 10,
                "desc": "BOTH deliverables rendered (z-distribution AND sky footprint)",
                "checks": [{"kind": "image_count", "min": 2}],
            },
            {
                "id": "C6", "type": "judge", "points": 20,
                "desc": "Deliverables are scientifically correct",
                "guidance": "z-histogram: n(z) over 0.4-0.8 with sensible bins (~0.01-0.05). Footprint: "
                            "RA/Dec (mean_fiber_ra/dec) density or scatter showing the DESI main-survey "
                            "footprint. Full credit needs both correct; half for one.",
            },
            {
                "id": "C7", "type": "judge", "points": 10,
                "desc": "Bitmask semantics explained (LRG = bit 0 of desi_target; main survey primacy)",
                "guidance": "Full credit for explaining the bitwise AND selection and the survey='main' / "
                            "main_primary quality context. Half for using it correctly without explanation.",
            },
        ],
    },

    # ══════════════════════════ TIER 6 ══════════════════════════
    {
        "id": "DLB-12",
        "tier": 6,
        "title": "Density search → image vetting (Hydra II, chained)",
        "prompt": (
            "Search NSC DR2 for the densest stellar clump within 1° of the Hydra "
            "II dwarf region, then pull DECam image cutouts of the top few "
            "candidate locations so I can eyeball them."
        ),
        "exercises": "nsc_dr2.object TAP overdensity + SIA cutouts at peaks (chained TAP -> SIA).",
        "reference_actions": "Density aggregate (0.05-deg bins, star+blue cuts) -> top-N cells -> per cell "
                             "deepest g-band Stack cutout -> multi-panel grid (label no-coverage panels).",
        "reference_sql": (
            "SELECT ROUND(ra/0.05)*0.05 AS ra_bin, ROUND(dec/0.05)*0.05 AS dec_bin, COUNT(*) AS n "
            "FROM nsc_dr2.object WHERE q3c_radial_query(ra, dec, 185.41, -31.98, 1.0) "
            "AND class_star > 0.5 AND gmag - rmag BETWEEN -0.5 AND 0.5 "
            "GROUP BY ra_bin, dec_bin ORDER BY n DESC LIMIT 5"
        ),
        "checkpoints": [
            {
                "id": "C1", "type": "auto", "points": 20,
                "desc": "Chained density->cutout workflow executed (one-shot vetting tool OR aggregate+grid)",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "tool_ok", "tools": ["datalab_density_vetting"]},
                        {"kind": "all", "of": [
                            {"kind": "tool_ok", "tools": ["datalab_density_aggregate"]},
                            {"kind": "tool_ok", "tools": ["datalab_cutout_grid", "datalab_image_cutout"]},
                        ]},
                    ]},
                ],
            },
            {
                "id": "C2", "type": "auto", "points": 10,
                "desc": "Cone within 1 deg of Hydra II (185.41, -31.98)",
                "checks": [
                    {"kind": "position_near", "ra": 185.41, "dec": -31.98, "tol_deg": 0.2},
                    {"kind": "radius_near", "value": 1.0, "tol": 0.3},
                ],
            },
            {
                "id": "C3", "type": "auto", "points": 10,
                "desc": "Stellar + blue color cuts in the executed density query",
                "checks": [
                    {"kind": "trace_regex", "pattern": r"class_star|sharp"},
                    {"kind": "trace_regex", "pattern": r"-\s*0\.5"},
                ],
            },
            {
                "id": "C4", "type": "auto", "points": 10,
                "desc": "Top-N kept small (a 'few' candidates, N <= 10)",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "tool_arg", "tools": ["datalab_density_vetting"], "arg": "top_n", "max": 10},
                        {"kind": "sql_regex", "pattern": r"LIMIT\s+([1-9]|10)\b"},
                    ]},
                ],
            },
            {
                "id": "C5", "type": "auto", "points": 15,
                "desc": "Cutout imagery rendered",
                "checks": [
                    {"kind": "image_emitted"},
                    {"kind": "tool_ok", "tools": ["datalab_density_vetting", "datalab_cutout_grid", "datalab_image_cutout"]},
                ],
            },
            {
                "id": "C6", "type": "judge", "points": 20,
                "desc": "Candidates ranked with coordinates + counts; densest clump identified",
                "guidance": "Full credit: an ordered list of peak (ra, dec) with source counts, densest "
                            "first (the densest clump should be Hydra II itself, ~185.43, -31.99). "
                            "Half: peaks without ordering or counts.",
            },
            {
                "id": "C7", "type": "judge", "points": 15,
                "desc": "Vetting guidance + honest handling of missing coverage",
                "guidance": "Credit for telling the user what to look for in the cutouts (compact stellar "
                            "concentration vs. chance clump / cluster contamination) and for labeling "
                            "panels without SIA coverage instead of failing or faking them.",
            },
        ],
    },

    {
        "id": "DLB-13",
        "tier": 6,
        "title": "Large-scale-structure wedge (SDSS Great Wall)",
        "prompt": (
            "Select galaxies from SDSS/BOSS in a thin redshift slice and make a "
            "cone/wedge plot to show the cosmic web — pick a region like the "
            "SDSS Great Wall."
        ),
        "exercises": "sdss_dr17.specobj via TAP; comoving distance; wedge/3D scatter.",
        "reference_actions": "ra/dec/z + quality query -> astropy comoving distance -> spherical->Cartesian "
                             "-> wedge (thin Dec slice) with equal axis scaling, or 2D pie slice.",
        "reference_sql": (
            "SELECT ra, dec, z FROM sdss_dr17.specobj WHERE class = 'GALAXY' AND zwarning = 0 "
            "AND ra BETWEEN 190 AND 240 AND dec BETWEEN 0 AND 5 AND z BETWEEN 0.0 AND 0.10"
        ),
        # Explicit guardrail exception: the PDF's own reference query is a
        # row-level RA/Dec BETWEEN box over sdss_dr17.specobj — a compact
        # spectroscopic table where the notebook idiom is a plain box and a
        # seq scan is affordable. GP-01/GP-03 are therefore waived HERE ONLY;
        # bounding is still scored positively via C7.
        "disable_penalties": ["GP-01", "GP-03"],
        "checkpoints": [
            {
                "id": "C1", "type": "auto", "points": 15,
                "desc": "Queried sdss_dr17.specobj",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "tool_arg", "tools": [], "arg": "catalog", "equals": "sdss_dr17"},
                        {"kind": "sql_regex", "pattern": r"sdss_dr17\.specobj"},
                    ]},
                ],
            },
            {
                "id": "C2", "type": "auto", "points": 15,
                "desc": "Galaxy + quality + thin-z selection in the executed query",
                "checks": [
                    {"kind": "trace_regex", "pattern": r"class\s*=\s*'?GALAXY"},
                    {"kind": "trace_regex", "pattern": r"zwarning\s*=\s*0"},
                    {"kind": "trace_regex", "pattern": r"z\s+BETWEEN\s+0(\.0)?\s+AND\s+0\.1|z\s*(<|&lt;)=?\s*0\.1"},
                ],
            },
            {
                "id": "C3", "type": "judge", "points": 15,
                "desc": "Region choice targets the SDSS Great Wall geometry",
                "guidance": "Full credit for RA ~150-250, a THIN Dec slice (a few deg), z <~ 0.1 — or an "
                            "equivalent well-justified slice. The Dec slice being thin is what makes the "
                            "wedge readable; deduct half if the slice is fat (>~10 deg).",
            },
            {
                "id": "C4", "type": "auto", "points": 15,
                "desc": "Wedge plot actually rendered",
                "checks": [
                    {"kind": "tool_ok", "tools": ["datalab_lss_wedge"]},
                    {"kind": "image_emitted"},
                ],
            },
            {
                "id": "C5", "type": "judge", "points": 20,
                "desc": "Comoving-space construction is correct",
                "guidance": "Full credit: redshifts converted to comoving distance with a named cosmology "
                            "(e.g. Planck18), spherical->Cartesian (or polar r=d, theta=RA pie slice), "
                            "and axis scaling that preserves Mpc-per-unit (no stretched thin slab). "
                            "Deduct ~1/3 for each missing element.",
            },
            {
                "id": "C6", "type": "judge", "points": 10,
                "desc": "Cosmic-web interpretation",
                "guidance": "Credit for identifying walls/filaments/voids and naming the Great Wall "
                            "feature if visible.",
            },
            {
                "id": "C7", "type": "auto", "points": 10,
                "desc": "Pull is bounded (region + z cuts and/or LIMIT / tool row cap)",
                # v1.1: bound must live in the SAME statement as the row-level
                # FROM (a BETWEEN in an earlier COUNT probe used to launder an
                # unbounded full-table pull), and the tool_arg is scoped to the
                # row-pull tools.
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "sql_regex",
                         "pattern": r"(?is)FROM\s+\S+[^;]*(LIMIT\s+\d+|BETWEEN)"},
                        {"kind": "tool_arg",
                         "tools": ["datalab_select_catalog_rows", "datalab_sql_query",
                                   "datalab_lss_wedge"],
                         "arg": "limit", "max": 100000},
                    ]},
                ],
            },
        ],
    },

    {
        "id": "DLB-14",
        "tier": 6,
        "title": "Variable-star characterization (RR Lyrae in Hydra II)",
        "prompt": (
            "I have a candidate variable star at RA = 185.4311, Dec = −31.9953 "
            "(an RR Lyrae in the Hydra II field). Find its multi-epoch SMASH "
            "photometry, phase-fold the light curve to get the period, and pull "
            "an image cutout of the field."
        ),
        "exercises": "smash_dr1.source multi-epoch TAP time series + Lomb-Scargle + SIA cutout. "
                     "This star is SMASH object 169.429960.",
        "reference_actions": "1-arcsec cone (or id key) -> g-band valid epochs (cmag<99) ordered by mjd -> "
                             "Lomb-Scargle over P=0.1-1 d -> phase-fold -> field cutout via coadd_all.",
        "reference_sql": (
            "SELECT mjd, cmag, cerr, filter FROM smash_dr1.source "
            "WHERE q3c_radial_query(ra, dec, 185.4311, -31.9953, 1.0/3600.0) "
            "AND filter = 'g' AND cmag < 99 ORDER BY mjd"
        ),
        "checkpoints": [
            {
                "id": "C1", "type": "auto", "points": 15,
                "desc": "Multi-epoch smash_dr1.source queried (not the coadd object table)",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "all", "of": [
                            {"kind": "tool_arg", "tools": [], "arg": "catalog", "equals": "smash_dr1"},
                            {"kind": "tool_arg", "tools": [], "arg": "table", "equals": "source"},
                        ]},
                        {"kind": "sql_regex", "pattern": r"smash_dr1\.source"},
                    ]},
                ],
            },
            {
                "id": "C2", "type": "auto", "points": 10,
                "desc": "Star isolated by a ~1-arcsec cone at (185.4311, -31.9953) or by object id",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "all", "of": [
                            {"kind": "position_near", "ra": 185.4311, "dec": -31.9953, "tol_deg": 0.01},
                            {"kind": "radius_near", "value": 0.000278, "tol": 0.001},
                        ]},
                        {"kind": "trace_regex", "pattern": r"169\.429960"},
                    ]},
                ],
            },
            {
                "id": "C3", "type": "auto", "points": 10,
                "desc": "Valid-epoch filtering (cmag < 99) and a single band, in the executed query",
                "checks": [
                    {"kind": "trace_regex", "pattern": r"cmag\s*(<|&lt;)\s*99"},
                    {"kind": "trace_regex", "pattern": r"filter\s*=\s*'?g|\"filter\""},
                ],
            },
            {
                "id": "C4", "type": "auto", "points": 20,
                "desc": "Lomb-Scargle period search + phase fold executed",
                "checks": [
                    {"kind": "tool_ok", "tools": ["datalab_period_fold"]},
                    {"kind": "trace_regex", "pattern": r"lomb|scargle|phase.?fold|period_fold"},
                ],
            },
            {
                "id": "C5", "type": "auto", "points": 10,
                "desc": "Folded light curve rendered",
                "checks": [{"kind": "image_emitted"}],
            },
            {
                "id": "C6", "type": "judge", "points": 15,
                "desc": "Period is reported and plausible for an RR Lyrae",
                "guidance": "Full credit: a concrete period in days within 0.1-1.0 d (RRab typically "
                            "~0.4-0.9 d, RRc ~0.2-0.45 d), consistent with the tool output. Zero for a "
                            "period outside the physical range or none reported.",
            },
            {
                "id": "C7", "type": "auto", "points": 10,
                "desc": "Field image cutout also pulled",
                "checks": [
                    {"kind": "tool_called", "tools": ["datalab_image_cutout", "datalab_color_image", "datalab_cutout_grid"]},
                ],
            },
            {
                "id": "C8", "type": "judge", "points": 10,
                "desc": "Coherent end-to-end narrative (epochs -> period -> fold -> field)",
                "guidance": "Credit for reporting number of epochs used, the folded curve's sawtooth shape "
                            "if visible, and tying the cutout back to the star's field.",
            },
        ],
    },

    # ══════════════════════════ TIER 7 ══════════════════════════
    {
        "id": "DLB-15",
        "tier": 7,
        "title": "Discover new Milky Way satellites (open-ended strategy)",
        "prompt": (
            "I want to discover new Milky Way satellite dwarf-galaxy candidates. "
            "Devise and carry out a search strategy using Data Lab's deep imaging "
            "catalogs, and give me a ranked list of candidate positions with "
            "supporting CMDs and image cutouts."
        ),
        "exercises": "Full agency: choose survey (DELVE/NSC/DES), tiled TAP density + matched filter, "
                     "CMD verification, SIA cutouts, ranking. Highest scan risk -> confirm area first.",
        "reference_actions": "Schema discovery -> pick catalog -> tiled region-bounded density GROUP BY "
                             "(hpix_1024, old/metal-poor point-source cuts) -> matched filter -> rank "
                             "-> per-candidate CMD + cutout -> ranked report. Confirm sky area before "
                             "fanning out (guardrail #5).",
        "reference_sql": (
            "SELECT AVG(ra) AS ra0, AVG(dec) AS dec0, hpix_1024, COUNT(*) AS n "
            "FROM delve_dr3.coadd_objects WHERE q3c_radial_query(ra, dec, :tile_ra, :tile_dec, 2.0) "
            "AND (mag_auto_g - mag_auto_r) < 0.75 AND mag_auto_g > 19.5 AND magerr_auto_g < 0.2 "
            "AND ext_coadd BETWEEN 0 AND 1 GROUP BY hpix_1024"
        ),
        "checkpoints": [
            {
                "id": "C1", "type": "judge", "points": 15,
                "desc": "A real search strategy is designed before executing",
                "guidance": "Full credit: names the survey choice (DELVE/NSC/DES) with a reason (depth, "
                            "southern coverage), describes the matched-filter density approach and the "
                            "old/metal-poor CMD selection, and states how candidates will be vetted and "
                            "ranked. Deduct proportionally for missing elements.",
            },
            {
                "id": "C2", "type": "auto", "points": 15,
                "desc": "Guardrail: sky area confirmed/estimated BEFORE the wide fan-out",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "tool_called", "tools": ["datalab_confirm_sky_area"]},
                        {"kind": "text_regex", "pattern": r"confirm|how (wide|large|big)|which region|sky area"},
                    ]},
                ],
            },
            {
                "id": "C3", "type": "auto", "points": 20,
                "desc": "Tiled, region-bounded, server-side density search with point-source + color cuts",
                "checks": [
                    {"kind": "any", "of": [
                        {"kind": "tool_ok", "tools": ["datalab_tiled_search"]},
                        {"kind": "tool_ok", "tools": ["datalab_density_aggregate"]},
                    ]},
                    {"kind": "trace_regex", "pattern": r"ext_coadd|class_star|sharp"},
                    {"kind": "trace_regex", "pattern": r"color_cut|mag_auto_g\s*-\s*mag_auto_r|g\s*-\s*r"},
                ],
            },
            {
                "id": "C4", "type": "auto", "points": 10,
                "desc": "Background-job orchestration for the long scan (status/results polling)",
                "checks": [
                    {"kind": "tool_called", "tools": ["datalab_job_status", "datalab_job_results"]},
                ],
            },
            {
                "id": "C5", "type": "judge", "points": 15,
                "desc": "Ranked candidate list with positions + significance",
                "guidance": "Full credit: an ordered table of candidate (ra, dec) with a density/matched-"
                            "filter significance per candidate. Known satellites recovered (e.g. Hydra II "
                            "if in the footprint) count as validation, not failure. Half: unranked or "
                            "significance-free list.",
            },
            {
                "id": "C6", "type": "auto", "points": 10,
                "desc": "Per-candidate verification evidence gathered (CMD and/or cutouts)",
                "checks": [
                    {"kind": "tool_ok", "tools": ["datalab_color_magnitude_diagram", "datalab_cutout_grid",
                                                   "datalab_density_vetting", "datalab_image_cutout"]},
                    {"kind": "image_emitted"},
                ],
            },
            {
                "id": "C7", "type": "judge", "points": 15,
                "desc": "CMD-based vetting interpreted correctly",
                "guidance": "Full credit if candidate CMDs are checked for an old, metal-poor population "
                            "(tight MSTO/RGB, possible BHB) and obvious false positives (clusters of "
                            "galaxies, chip artifacts) are screened out or flagged.",
            },
        ],
    },
]


def get_question(qid):
    for q in QUESTIONS:
        if q["id"] == qid:
            return q
    raise KeyError(qid)


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
        for pen in q.get("penalties", []):
            walk(pen["detect"], f"{q['id']}.{pen['id']}")
    for pen in GLOBAL_PENALTIES:
        walk(pen["detect"], f"GLOBAL.{pen['id']}")
    if len(QUESTIONS) != 15:
        errors.append(f"expected 15 questions, found {len(QUESTIONS)}")

    # v1.2 axis mapping (R6): overrides must reference real question and
    # checkpoint ids with valid axes, and every checkpoint must resolve.
    by_id = {q["id"]: q for q in QUESTIONS}
    for qid, overrides in AXIS_OVERRIDES.items():
        q = by_id.get(qid)
        if q is None:
            errors.append(f"AXIS_OVERRIDES: unknown question {qid}")
            continue
        cp_ids = {cp["id"] for cp in q["checkpoints"]}
        for cid, axis in overrides.items():
            if cid not in cp_ids:
                errors.append(f"AXIS_OVERRIDES: {qid}.{cid} does not exist")
            if axis not in CHECKPOINT_AXES:
                errors.append(f"AXIS_OVERRIDES: {qid}.{cid} has invalid axis {axis!r}")
    for q in QUESTIONS:
        for cp in q["checkpoints"]:
            if axis_for_checkpoint(q["id"], cp) not in CHECKPOINT_AXES:
                errors.append(f"{q['id']}.{cp['id']}: unresolvable axis")
    return errors


if __name__ == "__main__":
    problems = validate_dataset()
    if problems:
        print("DATASET INVALID:")
        for p in problems:
            print(" -", p)
        raise SystemExit(1)
    print(f"{BENCH_NAME} v{BENCH_VERSION}: {len(QUESTIONS)} questions OK; "
          f"all checkpoint rubrics sum to 100.")
