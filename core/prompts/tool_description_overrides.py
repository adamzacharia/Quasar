"""v2 tool-description amendments.

Rules that the legacy system prompt stated about one specific tool move next
to that tool's schema, where the model reads them at call time. The
amendments are APPENDED to the registered description when the v2 prompt is
active; the registry itself is never mutated, so the legacy arm and the
rollback path see the original text. Applied in
QuasarAgent._build_tools_for_responses_api.

Keep each amendment to the behaviour the legacy prompt encoded; do not
shorten or rewrite the base descriptions here (that is a separate follow-up
with its own A/B).
"""
from __future__ import annotations

from typing import Dict

OVERRIDES: Dict[str, str] = {
    "search_by_target": (
        "If this returns empty for a valid target name, call resolve_target for RA/Dec and then "
        "search_by_position. Several targets with the same constraints go in one call as a comma-separated "
        "string (target_name='M87, Sz65'); several bands as band='6,7' in ONE call, never one call per band. "
        "Targets with different constraints take separate calls per target and constraint pair, or the parsed "
        "form target_name='M87 in band 6, Sz65 in band 7'. Each call produces its own data card."
    ),
    "search_by_position": (
        "Use after resolve_target when a valid name returned nothing from search_by_target."
    ),
    "datalab_sql_query": (
        "Every query needs a bound: a q3c cone (q3c_radial_query), an indexed equality (SMASH fieldid = 169, "
        "id = '169.429960', DESI targetid = N), a registry-approved BETWEEN box, or a GROUP BY aggregate on an "
        "aggregate-safe table; all-sky row-level pulls are rejected. Data Lab stores missing floats as NaN and "
        "Postgres orders NaN above every number, so a one-sided lower bound or a not-equal cut on a nullable "
        "float column (parallax, pm, pmra, pmdec, parallax_over_error, magnitudes, colors, snr_*, chi2) must "
        "carry the finiteness guard, for example WHERE parallax_over_error > 5 AND parallax_over_error < "
        "'Infinity' AND pm > 150 AND pm < 'Infinity'; two-sided ranges and upper bounds are already safe. "
        "Never guess column names: call datalab_describe_table first (NSC DR2 uses gmag/rmag). If a query "
        "returns a jobid, poll datalab_job_status a few times only; when the result says stop_polling, end the "
        "turn and tell the user the job is still running."
    ),
    "datalab_describe_table": (
        "Call this before hand-writing SQL whenever the column names are not certain, and after a "
        "'column ... does not exist' error to correct the name."
    ),
    "datalab_job_status": (
        "Poll a few times only. When the result says stop_polling, stop and tell the user the job is still running."
    ),
    "datalab_select_catalog_rows": (
        "Selects by an indexed key too (key_column='fieldid', key_value=169); never use a dummy cone for that. "
        "Registry survey-quality cuts are applied automatically (DESI zpix: zwarn=0 and zcat_primary; DES: "
        "flags_g/r/i=0; SDSS specobj: zwarning=0) unless you pass your own cut on those columns; state the "
        "applied cuts when reporting counts. When the user implies an object class on a spectroscopic catalog "
        "(galaxies or LRGs: spectype='GALAXY' on DESI zpix, class='GALAXY' on SDSS specobj; quasars: 'QSO'), "
        "add that class cut yourself. Do not build a sky-density map from these capped rows."
    ),
    "datalab_color_magnitude_diagram": (
        "For Gaia or any absolute-magnitude HR diagram pass x_expr and y_expr, for example x_expr='bp_rp', "
        "y_expr='phot_g_mean_mag + 5*log10(parallax) - 10'; this renders an interactive plot, so prefer it over "
        "select_rows then catalog_scatter (static). For white-dwarf questions prefer datalab_selection_diagram; "
        "if you use this tool instead, pass overlay_locus='wd' and quote its n_wd_candidates rather than "
        "eyeballing the diagram."
    ),
    "datalab_selection_diagram": (
        "This is the routed tool for white-dwarf candidate searches: use the gaia_astrometric preset, a "
        "proper-motion floor when asked, and overlay_locus='wd'; quote the returned n_wd_candidates and "
        "wd_locus_note, never a visual estimate."
    ),
    "datalab_sed_plot": (
        "Multi-object: for 'SEDs of a sample / a few hundred objects' call it ONCE with sample_n (for example "
        "{\"result_id\": \"dlr_...\", \"sample_n\": 300}); it overlays up to 300 SEDs with the per-band median "
        "highlighted. Use row_index only for ONE object; never loop per row and never call the tool "
        "single-object. Wavelengths come from svo_filter_wavelength, never from memory."
    ),
    "svo_filter_wavelength": (
        "Always the source of filter wavelengths for SED plots; do not quote wavelengths from memory."
    ),
    "datalab_color_image": (
        "Pass ra, dec, and fov only; pick the fov from the target's apparent size and the 'center' intent "
        "(the center of M31 is about 0.1 to 0.2 deg), not a constant. LS DR9 imaging bands are g, r, z (no i); "
        "never conclude 'no color image' because i is missing. On coverage_gap, report which bands were usable "
        "and offer the user a same-survey single-band cutout or an explicitly labeled different-survey color "
        "view; NEVER silently substitute another survey's imagery, and repeat the result's source_service."
    ),
    "datalab_image_cutout": (
        "Also the same-survey single-band fallback when a color composite has a coverage gap."
    ),
    "hips_cutout": (
        "The ls_g, ls_r, ls_i, ls_z aliases serve the Legacy Surveys DR10 HiPS (which carries i); DR9 SIA "
        "tiles have no i band. When you use this for a request that named a different survey, label the "
        "result with the survey actually shown."
    ),
    "datalab_cutout_grid": (
        "For the peaks of a density search this is always ONE call with all peaks."
    ),
    "datalab_density_aggregate": (
        "Counts every row server-side, so it is the correct basis for density and footprint maps; for a whole "
        "named field bound with the indexed value cut (fieldid = N) and no cone. Use the coarse HEALPix column "
        "(for example ring256) for regions wider than a few degrees. It is also the fallback when "
        "datalab_density_vetting times out (then one datalab_cutout_grid call for the peaks; never retry the "
        "timed-out vetting call)."
    ),
    "datalab_sky_density_map": (
        "Never feed it a row-capped result (select_rows, sql rows, crossmatch rows); use an aggregate "
        "result_id. For overdensity hunts pass matched_filter=true and report the detected peak RA/Dec in the "
        "answer."
    ),
    "datalab_density_vetting": (
        "If it times out, fall back to datalab_density_aggregate plus one datalab_cutout_grid call; do not retry it."
    ),
    "datalab_q3c_crossmatch": (
        "Always use this for a two-catalog positional crossmatch; never hand-write q3c_join SQL. For membership "
        "science the crossmatch is step one: apply the proper-motion and CMD cuts server-side and plot the "
        "selected sample over the full cone, not these row-capped rows."
    ),
    "datalab_list_catalogs": (
        "Never list every catalog as covering a target; curate by footprint (the LMC is not covered by SDSS, "
        "DESI, LS DR9, or DES)."
    ),
    "survey_covers_position": (
        "Use before broad 'is there data' or 'which surveys observed X' claims. For exact archive IDs or "
        "product downloads, or if MOCServer fails, continue with the requested archive tool and report the "
        "preflight issue."
    ),
    "survey_coverage": (
        "Preflight for broad availability questions; if MOCServer fails, continue with the requested archive "
        "tool and report the preflight issue."
    ),
    "galactic_extinction": (
        "Use whenever photometry, colors, or distance moduli need dereddening (E(B-V), A_lambda)."
    ),
    "search_papers": (
        "After this tool runs successfully, do not repeat the whole paper list: the interface renders them as "
        "cards. When the user asks which paper(s) or for bibcodes / DOIs, name each one with its bibcode as "
        "returned. Report an empty or failed search briefly. If the user gave a proposal ID, project code, MOUS, "
        "ASDM UID, or dataset identifier, use search_papers_by_observation_id instead. Never use web_search "
        "for paper requests. When resolved_from is present, report the resolved bibcode, not the one asked for."
    ),
    "find_citing_papers": (
        "Results render as paper cards. Name the original paper and each responding paper with its bibcode as "
        "returned, grouped as the tool groups them (rebuttals, replies). Only cite bibcodes the tool returned."
    ),
    "search_papers_by_observation_id": (
        "Call it only when the user explicitly asks for papers connected to a specific identifier. The backend "
        "already auto-links the top ALMA project codes of search_by_target and search_by_position results to "
        "ADS; do not call this merely to reproduce that linking. Results render "
        "as paper cards; do not repeat the whole list in text, but name the papers with their bibcodes when the "
        "user asks for them."
    ),
    "lookup_researcher": (
        "Present the result with the researcher_profile format (header, identity, metrics table, focus, "
        "affiliation history, activity table, summary). Do not name the underlying data provider or link to its "
        "pages in the answer."
    ),
    "get_research_trends": (
        "Do not name the underlying data provider in the answer."
    ),
    "radio_sed": (
        "Always repeat the returned flags and state that v1 uses TGSS/GLEAM/SUMSS/NVSS/FIRST catalog fluxes "
        "without resolution matching, flux-scale corrections, or image-plane photometry."
    ),
    "period_search": (
        "Always report the FAP with any period; when the result says no significant period, say so with the FAP "
        "rather than claiming a period."
    ),
    "search_space_lightcurves": (
        "First step for TESS/Kepler variability: list availability, then plot_space_lightcurve or period_search."
    ),
    "monitor_add_target": (
        "For 'keep an eye on', 'alert me', 'monitor' requests: add the target, then call monitor_check_now and "
        "report only NEW alerts. Manage the watchlist with monitor_list_targets and monitor_remove_target."
    ),
    "monitor_check_now": (
        "Report only the alerts that are new since the previous check."
    ),
    "find_alma_line_coverage": (
        "Use this once for one named transition and target (for example 'Check CO(2-1) line coverage for M87'). "
        "Do not use the broad check_co_lines for a named transition, and do not report other CO ladder "
        "transitions as matches."
    ),
    "check_co_lines": (
        "Only for explicit requests to inspect the whole CO/13CO/C18O ladder in prior search results; a single "
        "named transition goes to find_alma_line_coverage."
    ),
    "search_mmu_hats_catalog": (
        "Examples: 'What Gaia sources are near M87?' -> catalog_key='gaia', target_name='M87'. 'Find ALMA data "
        "for M87' -> search_by_target, not this tool. For combined requests (ALMA data for M87 and Gaia sources "
        "in the field) call the archive tool FIRST, then enrich the field with this tool. For catalog-to-catalog "
        "matching use crossmatch_mmu_hats_catalogs within a bounded cone; if it fails, run two bounded cone "
        "searches and say so."
    ),
    "vo_list_tables": (
        "Pass a keyword on big services such as VizieR. Quote table names containing '/' or '+' in double "
        "quotes in the ADQL that follows."
    ),
    "vo_adql_query": (
        "Inspect the schema (vo_describe_table) before writing ADQL. Quote table names containing '/' or '+' "
        "in double quotes. ESA Gaia, ESO, CADC and NRAO have curated notes: browse_schema('gaia'|'eso'|'cadc'|'nrao'). "
        "For images or cubes at a position use vo_image_search instead."
    ),
    "moving_object_check": (
        "Use when a transient could be a known asteroid or comet."
    ),
    "solar_system_ephemeris": (
        "Use for planet, asteroid, or comet positions, distances, magnitude, and visibility over a date range."
    ),
    "gaia_distance": (
        "Use for stellar distances from parallax; do not compute 1/parallax by hand."
    ),
    "ned_distance": (
        "Use for galaxy distances (redshift-independent indicators)."
    ),
    "velocity_frame_distance": (
        "Use for flow-corrected Hubble distances of nearby galaxies."
    ),
    "query_alma_science_archive": (
        "Use only when no alma_* one-shot tool (alma_project_census, alma_source_summary, "
        "alma_public_band_status, alma_bibliography) expresses the question; never answer these counts from "
        "memory and do not hand-write ADQL unless this tool cannot express the query."
    ),
    "triage_alma_data_products": (
        "Pass a stated band preference so the project picker is filtered first. Never auto-download large "
        "products; remote header inspection and product listing are safe."
    ),
    "web_search": (
        "Keyword query -> this tool. A full URL to read, summarize, or quote -> web_extract_url. A site root plus "
        "'find pages' -> web_map_site. A site section plus 'crawl' or 'docs' -> web_crawl_site. A comprehensive "
        "report or comparison -> web_research."
    ),
    "web_research": (
        "Never for archive data or papers; those have dedicated tools."
    ),
    "filter_results": (
        "Use after a search when the user asks for constraints such as 'resolution < 0.05' on archive results; "
        "not needed when a dedicated tool already applied the cut server-side."
    ),
}


def effective_description(name: str, base: str) -> str:
    """The description the model sees in the v2 arm."""
    extra = OVERRIDES.get(name)
    if not extra:
        return base
    base = (base or "").rstrip()
    sep = "" if not base else (" " if base.endswith((".", "!", "?", ")")) else ". ")
    return f"{base}{sep}{extra}"
