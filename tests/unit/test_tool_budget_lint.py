"""Tool-schema token lint (token plan rank 3) and tool-pack coverage (rank 1).

Builds the real agent once (local env, ~16 s) and checks the serialized tool
list: every tool belongs to the core set or a pack, pack tables name only
registered tools, and no tool grows past its token budget. Budgets are a
ratchet: a new tool must fit DEFAULT_CAP; an existing oversize tool may not
grow more than 5% past the size recorded here (2026-10-02). To shrink one,
lower its entry.
"""
import os
import sys

import pytest

DEFAULT_CAP = 350
SLACK = 1.05
BUDGETS = {
    "ads_search": 477,
    "advanced_search": 438,
    "alma_project_census": 612,  # constrained nullable unions keep anyOf (guard CX-06)
    "browse_alma_guidance": 672,
    "calculate_alma_sensitivity": 630,
    "catalog_crossmatch": 755,
    "catalog_query": 557,
    "datalab_color_color_diagram": 758,
    "datalab_color_image": 463,
    "datalab_color_magnitude_diagram": 743,
    "datalab_density_aggregate": 699,
    "datalab_density_vetting": 439,
    "datalab_healpix_density_map": 398,  # +null in type lists (guard CX-06)
    "datalab_list_catalogs": 531,
    "datalab_lss_wedge": 379,
    "datalab_q3c_crossmatch": 401,
    "datalab_satellite_search": 673,  # +null in type lists (guard CX-06)
    "datalab_sed_sample": 466,
    "datalab_select_catalog_rows": 617,
    "datalab_selection_diagram": 621,  # +null in type lists (guard CX-06)
    "datalab_sia_search": 850,
    "datalab_sky_density_map": 437,
    "datalab_sql_query": 516,
    "datalab_stream_selection": 419,  # +null in type lists (guard CX-06)
    "detect_sources": 474,
    "exoplanet_archive": 626,
    "fermi_lcr_lightcurve": 418,
    "fit_gaussian_source": 464,
    "gaia_archive_query": 542,
    "generate_casa_calibration_script": 398,
    "generate_casa_imaging_script": 495,
    "heasarc_observations": 529,
    "hips_aperture_photometry": 449,
    "hips_cutout": 437,
    "hips_rgb_composite": 409,
    "image_statistics": 372,
    "match_cross_archive_sources": 415,
    "measure_region": 466,
    "mmdc_lightcurve": 379,
    "mmdc_model": 830,
    "mmdc_sed": 604,
    "moc_operations": 394,
    "query_alma_science_archive": 1030,
    "radial_profile": 439,
    "search_alma_co_in_redshift_range": 383,
    "search_by_target": 780,
    "search_lines_by_molecule": 493,
    "search_mast_by_criteria": 482,
    "search_mmu_hats_catalog": 442,
    "search_papers": 440,
    "search_spectral_lines": 566,
    "simbad_query": 462,
    "sparcl_search_spectra": 367,
    "sparcl_stack_spectra": 558,
    "survey_coverage": 388,
    "triage_alma_data_products": 394,
    "variability_analysis": 701,
    "vo_image_search": 480,
    "web_crawl_site": 466,
    "web_map_site": 352,
    "web_search": 383,
}


@pytest.fixture(scope="module")
def serialized_tools():
    env = {"QUASAR_ENV": "development", "ENVIRONMENT": "development", "QUASAR_FORCE_LOCAL_DB": "1",
           "TURSO_DATABASE_URL": "", "TURSO_AUTH_TOKEN": ""}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ui-pro"))
        from core.agent import QuasarAgent

        try:
            agent = QuasarAgent()
        except Exception as exc:
            # A construction regression must fail the lint, not skip it (guard
            # CX-14). Opt-out only for environments without agent deps.
            if os.getenv("QUASAR_SKIP_AGENT_LINT") == "1":
                pytest.skip(f"agent could not be built here: {exc}")
            raise
        return agent._build_tools_for_responses_api() or []
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_every_tool_is_in_core_or_a_pack(serialized_tools):
    from core import tool_packs as tp

    names = {t["name"] for t in serialized_tools if t.get("type") == "function"}
    covered = set(tp.CORE) | {n for spec in tp.PACKS.values() for n in spec["tools"]}
    assert sorted(names - covered) == [], "add new tools to core/tool_packs.py (CORE or a pack)"
    assert sorted(covered - names - {tp.FIND_TOOLS}) == [], "pack tables name tools that are not registered"


def test_tool_schema_token_budgets(serialized_tools):
    from core.call_tokens import count

    over = {}
    for t in serialized_tools:
        n = count(t)
        cap = BUDGETS.get(t["name"], DEFAULT_CAP)
        if n > cap * SLACK:
            over[t["name"]] = (n, cap)
    assert over == {}, f"tool schemas over budget (tokens, budget): {over}"

