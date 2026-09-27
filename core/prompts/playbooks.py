"""Per-turn workflow playbooks for the v2 system prompt.

A playbook is guidance that used to live in the always-on system prompt but
only matters for one workflow (white-dwarf selection, stream membership,
density maps, survey color imagery, open-ended regions, researcher profiles,
ALMA product triage). The runner selects playbooks once at round 0 from the
bare user query and the resolved intents, and again after each tool round
from the tools that were actually called, so a missed lexical trigger still
gets the guidance before the model interprets the results.

Pure module: no agent state. Turn-local state lives in ``TurnState`` owned
by the runner for one turn. Nothing here touches tool outputs or the UI
event stream; the rendered text is appended to the current-turn user input.

Budget: ``INITIAL_BUDGET`` tokens at round 0 and ``FOLLOWUP_BUDGET`` for
tool-observed activation, ``TOTAL_BUDGET`` per turn. Selection is
deterministic (priority, then id), deduplicated by id, and never truncates a
playbook mid-sentence: a playbook that does not fit is omitted and logged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

# Sentinel headers: the shim prunes earlier turns' context by these exact strings
# (core/llm_client.prune_turn_context_history). The bracketed tag keeps a user who
# quotes the words "Turn context" from ever matching.
TURN_CONTEXT_HEADER = "Turn context [qv2]"
TURN_CONTEXT_UPDATE_HEADER = "Turn context update [qv2] (applies to this turn only):"

INITIAL_BUDGET = 1100
FOLLOWUP_BUDGET = 500
TOTAL_BUDGET = 1600

# A query "names a region" when it carries coordinates, a catalog designation,
# a survey field id, or an "around / near / within N deg of <target>" phrase.
_COORD_RE = re.compile(
    r"\b(?:ra|dec|l|b)\s*[=:~]?\s*[-+]?\d"
    r"|\d+(?:\.\d+)?\s*(?:deg|°|arcmin|arcsec|′|″)"
    r"|\b(?:ngc|ic|ugc|m|pal(?:omar)?|abell|hydra|draco|sculptor|fornax|carina|sextans|leo)\s?\w{0,3}\d{0,4}\b"
    r"|\bsmash\s+(?:dr\d\s+)?field\b|\bfield\s*\d+\b"
    r"|\b(?:around|near|toward|towards|centered\s+on|centred\s+on|in\s+the\s+direction\s+of|within\s+[\d.]+\s*\S*\s+of)\s+(?:the\s+)?[A-Z][\w-]*",
    re.I,
)


def _rx(*patterns: str) -> tuple:
    return tuple(re.compile(p, re.I | re.S) for p in patterns)


@dataclass(frozen=True)
class Playbook:
    id: str
    priority: int                      # lower runs first
    text: str
    intent_keys: tuple = ()            # resolved-intent keys from the runner
    query_patterns: tuple = ()         # compiled regexes on the bare user query
    oneshot_tools: tuple = ()          # detect_oneshot_intent()["tool"] values
    observed_tools: tuple = ()         # tools actually called this turn
    result_flags: tuple = ()           # flags found in tool results this turn ...
    flag_tools: tuple = ()             # ... but only when one of THESE tools produced them
    requires_no_region: bool = False   # only when the query names no region

    @property
    def token_cost(self) -> int:
        from core.prompts.system_core import count_tokens

        return count_tokens(self.text)


@dataclass
class TurnState:
    """Turn-local record of what was sent; owned by the runner for one turn."""
    sent_ids: Set[str] = field(default_factory=set)
    spent_tokens: int = 0
    omitted: List[str] = field(default_factory=list)
    log: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SURVEY_IMAGERY = """[PLAYBOOK survey_imagery] Survey-named color imagery (DECam, Legacy Surveys, DES, coadd)
- Call datalab_color_image with ra, dec, and fov only. It selects an available three-band triplet and renders the Lupton RGB; do not hand-pick bands or pre-judge coverage.
- Legacy Surveys DR9 imaging bands are g, r, z. There is no i band in DR9; never conclude "no color image" because i is missing.
- Pick the field of view from the target's apparent size and the "center" intent (M31's D25 is about 3 degrees, so "center of M31" means roughly 0.1 to 0.2 degrees), not a fixed constant.
- The coadd_all cutout service has genuinely broken or partial tiles at some bright nearby galaxies (the exact center of M31 has only usable z-band tiles). When datalab_color_image returns coverage_gap, the tool has already tried the same survey's official color HiPS as completion; a remaining gap means a color image from that survey cannot be made at this position.
- NEVER silently substitute another survey's imagery for a "color image" request: no hips_multiband_panel, no DSS2, 2MASS, or WISE panels presented as the answer. Report the gap honestly (which bands were usable, why the composite failed) and offer an explicit choice: (a) a same-survey single-band cutout in a usable band (datalab_image_cutout, or hips_cutout with an ls_g, ls_r, ls_i, or ls_z single-band HiPS; those aliases serve the Legacy Surveys DR10 HiPS, which does carry i, and the no-i rule above is about DR9 SIA tiles), or (b) an optical color view from a different survey (for example hips_cutout DSS2 color) explicitly labeled with that survey's name, never presented as the requested survey. If they choose (b), state the source survey and the image service from the result's source_service field.
- Say an image is shown only when a tool rendered one this turn."""

_WD_SELECTION = """[PLAYBOOK wd_selection] White-dwarf candidates and absolute-magnitude HR diagrams
- The routed one-shot for this workflow is datalab_selection_diagram: quality preset gaia_astrometric (parallax_over_error > 5, ruwe < 1.4, parallax > 0), a total-proper-motion floor when asked, absolute magnitude from parallax, and overlay_locus='wd' so the result reports n_wd_candidates and a wd_locus_note. Prefer it.
- Fallback only if that tool is unavailable: datalab_color_magnitude_diagram with x_expr='bp_rp', y_expr='phot_g_mean_mag + 5*log10(parallax) - 10', point_sources=true, overlay_locus='wd'.
- Quote the tool-computed candidate count. Never eyeball the diagram, never call the lower main sequence a "WD cooling track", and never answer WD counts from memory.
- Include the astrometric quality cuts (with their finiteness guards) in the selection before quoting counts, and state them in the answer."""

_STREAM_MEMBERSHIP = """[PLAYBOOK stream_membership] Stream, tidal-tail, and cluster membership science
- A raw positional crossmatch is only step one. Apply the science cuts server-side (value_cuts for the proper-motion window, color_cut for the population or CMD locus) and make the final sky and CMD plots from the selected member sample. Never present the raw crossmatch as the result. State the exact cuts in the answer.
- For the on-sky plot use the proper-motion and CMD selected rows over the full cone (for example the Gaia datalab_select_catalog_rows result or datalab_stream_selection), not a row-capped crossmatch result: the crossmatch LIMIT slices the sample to a spatial corner and the map misses the cluster or stream.
- Claim the map shows the cluster or tails only if the cluster center actually lies within the plotted RA and Dec range."""

_DENSITY_MAPS = """[PLAYBOOK density_maps] Density, footprint, and overdensity maps
- Never build a sky-density or overdensity map from a row-limited pull (datalab_select_catalog_rows, datalab_sql_query rows, crossmatch rows): capped results are storage-order, spatially clustered slices and the map shows one corner of the field.
- Use datalab_density_aggregate (server-side GROUP BY counts every row) or datalab_density_vetting (finds and ranks peaks). For regions wider than a few degrees use the coarse HEALPix column (for example ring256). Wide cones auto-tile; call once with the full cone.
- For a whole named survey field (for example SMASH field 169) bound the aggregate with the indexed value cut fieldid = N and no cone; never guess a cone center for a named field.
- For overdensity hunts use density_vetting or pass matched_filter=true to datalab_sky_density_map, and report the detected peak RA and Dec in the answer.
- If a result carries a "hit its row cap" warning, do not plot its sky distribution; rerun with an aggregate and relay the truncation.
- If datalab_density_vetting times out, fall back to datalab_density_aggregate for the peaks and one datalab_cutout_grid call for all peak cutouts. Never one datalab_image_cutout per peak, and never retry the timed-out vetting call."""

_OPEN_REGIONS = """[PLAYBOOK open_regions] Open-ended or named analysis regions
- When the user leaves the region open, choose a validated field: a known satellite, or the tool's preset (south_gradient for wide stellar-density maps, smc, sgr_stream, delve_south). The LMC is too dense for a full map in one turn. Never default to the Galactic Centre. Density scans keep their default point-source and blue or old-population colour cuts.
- "The SDSS Great Wall" uses the adopted analysis window RA 150 to 220 degrees, Dec 0 to 5 degrees, z <= 0.1. State it as the adopted window, not as a definition of the structure."""

_BLAZAR_MMDC = """[PLAYBOOK blazar_mmdc] Blazar SEDs, emission models and variability (MMDC)
- Multi-epoch or time-resolved SEDs, and any blazar SED: mmdc_sed with start_date/end_date. Other targets: ned_sed_plot. If mmdc_sed returns not_mmdc_source, call its fallback_tool and say MMDC has no entry; never describe MMDC data for it.
- Window rule: state the rule the tool applied (contained keeps only bins inside the window; overlap also keeps bins extending past it) with the window in MJD and dates, the straddling-row count and the undated archival count. Never call out-of-window data in-window. For a short window, contained can drop long Fermi bins; say so and offer window_mode='overlap' rather than silently switching.
- Upper limits are drawn and counted separately; never quote one as a detection.
- Fits: mmdc_model mode='fit' with the SED result_id so the fit uses exactly the rows shown, then mmdc_model_job until done. Report parameters with errors from the tool, the rows dropped and binned, any at_bound flags and the surrogate validity caveat. Redshift only from the tool (MMDC or NED) with its source; never guess. Hadronic neutrino likelihoods need explicit n_icecube and dt, or x1, x2 and y.
- Variability: variability_analysis on a light-curve result_id (mmdc_lightcurve, ztf_light_curve, plot_space_lightcurve, datalab_star_lightcurve). Quote Fvar with its error, flare blocks, lags with errors, and index-flux trends only when the tool's p < 0.05.
- Blazar catalogs: 5BZCAT is VizieR VII/274/bzcat5 (class, redshift) and 4LAC is J/ApJ/892/105/4lac (4FGL name, flux, photon index); query them with catalog_query. Fermi-LAT light curves independent of MMDC: fermi_lcr_lightcurve.
- End with the MMDC acknowledgment and the citations the tool returns."""

_RESEARCHER_PROFILE = """[PLAYBOOK researcher_profile] Researcher profile format
Present the lookup_researcher result in this exact order. Fields the tool did not return stay absent; never invent them.
1. Header: "## Profile: [Full Name]" with email and personal webpage when web results supply them.
2. Identity: ORCID, alternative name forms, current institution(s).
3. Academic Metrics as a Markdown table with rows Publications, Citations, h-index, i10-index, 2yr Mean Citedness.
4. Research Focus: bulleted list of top topics.
5. Affiliation History: chronological list with year ranges.
6. Recent Research Activity as a Markdown table with Year, Works, Citations columns (last 5 to 10 years).
7. Summary: a brief narrative paragraph.
If web results are available, put the email, personal webpage, and recent news or awards at the top. Do not name the underlying data provider."""

PLAYBOOKS: List[Playbook] = [
    Playbook(
        id="survey_imagery", priority=10, text=_SURVEY_IMAGERY,
        query_patterns=_rx(r"\b(?:colou?r|rgb|three[- ]colou?r)\b.{0,60}\b(?:image|composite|picture|cutout)\b.{0,80}\b(?:decam|legacy\s+surveys?|\bdes\b|coadd|ls\s*dr\d+)|\b(?:decam|legacy\s+surveys?|\bdes\b|coadd|ls\s*dr\d+)\b.{0,80}\b(?:colou?r|rgb)\b.{0,40}\b(?:image|composite|picture)\b"),
        observed_tools=("datalab_color_image",),
        result_flags=("coverage_gap",),
        flag_tools=("datalab_color_image", "datalab_image_cutout", "hips_cutout", "datalab_cutout_grid"),
    ),
    Playbook(
        id="wd_selection", priority=20, text=_WD_SELECTION,
        query_patterns=_rx(r"\bwhite[- ]dwarfs?\b|\bWDs?\b.{0,40}\b(?:candidate|sequence|cooling|locus)|\b(?:hr|hertzsprung)\b.{0,20}\bdiagram\b.{0,80}\b(?:absolute|parallax)"),
        oneshot_tools=("datalab_selection_diagram",),
        observed_tools=("datalab_selection_diagram",),
    ),
    Playbook(
        id="stream_membership", priority=30, text=_STREAM_MEMBERSHIP,
        query_patterns=_rx(r"\btidal\s+tails?\b|\bstellar\s+streams?\b|\bstream\s+(?:stars|members|selection)\b|\bmember(?:ship)?\s+(?:stars|selection)\b|\bpal(?:omar)?\s*5\b.{0,80}\b(?:tail|stream|proper[- ]motion)"),
        oneshot_tools=("datalab_stream_selection",),
        observed_tools=("datalab_stream_selection",),
    ),
    Playbook(
        id="density_maps", priority=40, text=_DENSITY_MAPS,
        query_patterns=_rx(r"\b(?:over)?densit(?:y|ies)\b|\bclumps?\b|\bfootprint\b.{0,30}\b(?:map|plot)|\bhealpix\b|\bwhere\s+(?:do|does)\s+(?:they|it|the\s+\w+)\s+(?:clump|cluster|concentrate)"),
        oneshot_tools=("datalab_healpix_density_map", "datalab_satellite_search"),
        observed_tools=("datalab_density_aggregate", "datalab_sky_density_map", "datalab_density_vetting", "datalab_tiled_search"),
        result_flags=("row_cap",),
        flag_tools=("datalab_select_catalog_rows", "datalab_sql_query", "datalab_q3c_crossmatch"),
    ),
    Playbook(
        id="open_regions", priority=50, text=_OPEN_REGIONS,
        query_patterns=_rx(r"\bgreat\s+wall\b|\bpick\s+a\s+(?:region|field|patch)\b|\banywhere\s+(?:in|on)\s+the\s+sky\b|\bsome\s+(?:region|field|patch)\b|\ba\s+(?:region|field|patch)\s+like\b"),
        oneshot_tools=("datalab_healpix_density_map", "datalab_satellite_search", "datalab_stream_selection"),
        requires_no_region=True,
    ),
    Playbook(
        id="blazar_mmdc", priority=35, text=_BLAZAR_MMDC,
        query_patterns=_rx(r"\bblazars?\b|\bbl\s*lac|\bfsrqs?\b|\bmulti[- ]?epoch\b.{0,40}\bs(?:ed|pectral\s+energy)|\btime[- ]resolved\s+s(?:ed|pectral\s+energy)|\b(?:ssc|eic|hadronic|lepto[- ]?hadronic)\b.{0,40}\b(?:model|fit)|\bfractional\s+variability\b|\bfvar\b|\bbayesian\s+blocks?\b|\bmmdc\b"),
        oneshot_tools=("mmdc_sed", "mmdc_lightcurve", "mmdc_model", "variability_analysis"),
        observed_tools=("mmdc_sed", "mmdc_lightcurve", "mmdc_model", "mmdc_model_job", "variability_analysis"),
    ),
    Playbook(
        id="researcher_profile", priority=60, text=_RESEARCHER_PROFILE,
        intent_keys=("researcher",),
        observed_tools=("lookup_researcher",),
    ),
    # No alma_products playbook: the runner's product directives already carry
    # the picker and row-number rules (core/runner.py product_directive), and
    # the remaining guidance (band filter, no bulk downloads) lives in the
    # triage_alma_data_products description amendment. Duel DX-14: one
    # authoritative instruction per workflow.
]

_BY_ID = {p.id: p for p in PLAYBOOKS}


def get_playbook(pid: str) -> Optional[Playbook]:
    return _BY_ID.get(pid)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _names_region(query: str) -> bool:
    return bool(_COORD_RE.search(query or ""))


# Presets the one-shot router falls back to when the user left the region open
# (core/oneshot_routing.py). A target-derived preset (hydra2, a named field) is
# NOT open-ended.
OPEN_REGION_PRESETS = frozenset({"south_gradient", "delve_south", "smc", "sgr_stream"})
_REGION_ARG_KEYS = ("ra", "dec", "target", "target_name", "cluster_name", "smash_field", "field", "fieldid", "preset", "region")


def _region_left_open(oneshot_tool: Optional[str], oneshot_args: Optional[Dict[str, Any]], region_named: bool) -> bool:
    """True when the workflow needs a region and the user did not supply one."""
    if oneshot_args:
        preset = oneshot_args.get("preset")
        if preset in OPEN_REGION_PRESETS:
            return True
        if any(oneshot_args.get(k) not in (None, "", []) for k in _REGION_ARG_KEYS):
            return False
        return not region_named
    if oneshot_tool:
        return not region_named
    return not region_named


def _fits(pb: Playbook, state: TurnState, budget: int, spent_in_call: int) -> bool:
    return spent_in_call + pb.token_cost <= budget and state.spent_tokens + pb.token_cost <= TOTAL_BUDGET


def _take(candidates: Iterable[Playbook], state: TurnState, budget: int, stage: str) -> List[Playbook]:
    chosen: List[Playbook] = []
    spent = 0
    for pb in sorted(candidates, key=lambda p: (p.priority, p.id)):
        if pb.id in state.sent_ids:
            continue
        if not _fits(pb, state, budget, spent):
            state.omitted.append(pb.id)
            state.log.append(f"{stage}: omitted {pb.id} ({pb.token_cost} tok, budget {budget})")
            continue
        chosen.append(pb)
        spent += pb.token_cost
        state.sent_ids.add(pb.id)
        state.spent_tokens += pb.token_cost
        state.log.append(f"{stage}: selected {pb.id} ({pb.token_cost} tok)")
    return chosen


def select_initial_playbooks(
    query: str,
    resolved_intent: Optional[Dict[str, Any]] = None,
    available_tools: Optional[Set[str]] = None,
    budget: int = INITIAL_BUDGET,
    state: Optional[TurnState] = None,
) -> List[Playbook]:
    """Round-0 selection from the bare user query and the runner's resolved
    intents: ``{"oneshot_tool": str|None, "researcher": bool,
    "product_triage": bool}``. ``available_tools`` (when given) drops a
    playbook whose observed tools are all unavailable in this arm."""
    state = state if state is not None else TurnState()
    intent = resolved_intent or {}
    oneshot_tool = intent.get("oneshot_tool")
    q = query or ""
    region_named = _names_region(q)
    candidates: List[Playbook] = []
    for pb in PLAYBOOKS:
        hit = False
        if oneshot_tool and oneshot_tool in pb.oneshot_tools:
            hit = True
        if not hit and any(intent.get(k) for k in pb.intent_keys):
            hit = True
        if not hit and any(rx.search(q) for rx in pb.query_patterns):
            hit = True
        if not hit:
            continue
        if pb.requires_no_region and not any(rx.search(q) for rx in pb.query_patterns):
            # Region defaults only help when the user left the region open. The
            # one-shot router already resolved the region into arguments: an
            # open-ended request lands on one of the default presets, a named
            # target/field on ra/dec, cluster_name, smash_field or a target
            # preset. Fall back to the lexical check when no router args exist.
            if not _region_left_open(oneshot_tool, intent.get("oneshot_args"), region_named):
                continue
        if available_tools is not None and pb.observed_tools and not (set(pb.observed_tools) & available_tools):
            continue
        candidates.append(pb)
    return _take(candidates, state, budget, "round0")


def select_tool_followups(
    tool_names: Sequence[str],
    result_flags: Iterable[str],
    state: TurnState,
    budget: int = FOLLOWUP_BUDGET,
) -> List[Playbook]:
    """Activation from the tools actually called in the round just finished
    (and flags seen in their results). Only playbooks not yet sent."""
    names = set(tool_names or ())
    flags = set(result_flags or ())
    # A result flag activates a playbook only when one of the tools that make
    # the flag meaningful was called (a satellite-search cutout grid reporting
    # coverage_gap is not a color-image request; a CMD result noting a row cap
    # is not a density map). Observed 2026-09-25 A/B run 1: both fired spuriously.
    candidates = [
        pb for pb in PLAYBOOKS
        if pb.id not in state.sent_ids
        and (
            (names & set(pb.observed_tools))
            or ((flags & set(pb.result_flags)) and (names & set(pb.flag_tools)))
        )
    ]
    return _take(candidates, state, budget, "followup")


# Flags are read by VALUE: '"coverage_gap": false' or a numeric '"row_cap": 500'
# (the configured limit, not a hit) must not activate a playbook. Matches JSON
# (true) and Python-repr (True) payloads.
_COVERAGE_GAP_TRUE_RE = re.compile(r"\bcoverage_gap[\"']?\s*[:=]\s*true\b", re.I)
_ROW_CAP_TRUE_RE = re.compile(r"\brow_cap(?:_hit)?[\"']?\s*[:=]\s*true\b", re.I)


def result_flags_from_outputs(tool_results: Iterable[Dict[str, Any]]) -> Set[str]:
    """Cheap flag extraction from function_call_output payloads: only the two
    signals the playbooks key on, and only when they are set."""
    flags: Set[str] = set()
    for item in tool_results or ():
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue
        out = str(item.get("output", ""))
        if _COVERAGE_GAP_TRUE_RE.search(out):
            flags.add("coverage_gap")
        if "hit its row cap" in out or _ROW_CAP_TRUE_RE.search(out):
            flags.add("row_cap")
    return flags


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_turn_context(turn_date: str, route_directive: str = "", playbooks: Sequence[Playbook] = ()) -> str:
    """The per-turn block appended to the user input. ``turn_date`` is an ISO
    date with timezone label, captured once at turn start."""
    lines = ["", "", TURN_CONTEXT_HEADER, f"- Current date: {turn_date}"]
    if route_directive:
        lines.append(f"- Routing: {route_directive.strip()}")
    if playbooks:
        lines.append("- Workflow playbooks for this turn (they apply to this turn only):")
        for pb in playbooks:
            lines.append("")
            lines.append(pb.text)
    return "\n".join(lines)


def render_followup_item(playbooks: Sequence[Playbook]) -> Optional[Dict[str, str]]:
    """A user-role input item carrying newly activated playbooks, appended
    after a round's tool outputs. None when nothing was activated."""
    if not playbooks:
        return None
    body = TURN_CONTEXT_UPDATE_HEADER + "\n\n" + "\n\n".join(pb.text for pb in playbooks)
    return {"role": "user", "content": body}
