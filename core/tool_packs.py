# core/tool_packs.py
"""
Per-turn tool packs: send the model only the tools a turn needs.

CALLED BY: core/runner.py (tool list for each turn, find_tools execution,
           tools the model calls outside the offered set)
CALLS:     nothing heavy (regexes over the question; pure functions)

Every LLM call used to carry all ~200 tool schemas (about 40k billed tokens
on TACC, token plan tmp/token-budget-2026-10-01/PLAN.md). A turn now gets:

* CORE: a small fixed set that answers most simple questions;
* packs chosen from signals already in the question (archive names,
  catalogue words, literature words, one-shot intents, tools named in the
  route directives), plus packs of the tools this conversation called in
  its last turns, so a follow-up ("now plot it") keeps its tools;
* ``find_tools``: a meta-tool the model calls when it needs a capability
  that is not loaded; the matching tools are added for the next round.

Order is fixed (core first, then packs in PACK_ORDER, each pack in its own
order) so the system prompt + core prefix is byte-identical across turns and
users, which keeps provider prefix caches warm (plan rank 5).

Guards against a missing tool: find_tools; a call to a registered tool that
was not offered still executes and its pack is added (runner); a round-0
answer that says it lacks a tool is re-sampled once with the full set
(``says_lacks_tool``, runner). ``QUASAR_TOOL_PACKS=0`` restores the old
behaviour (every tool on every call). Benchmark allowlist arms bypass packs.
"""

from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

ENV = "QUASAR_TOOL_PACKS"
FIND_TOOLS = "find_tools"


def enabled() -> bool:
    return os.getenv(ENV, "1").strip().lower() not in ("0", "false", "no", "off")


# ── Pack membership ─────────────────────────────────────────────────────
# Names that are not registered (optional services, renamed tools) are
# simply skipped when the list is built, so this table can be a superset.

CORE: Sequence[str] = (
    FIND_TOOLS,
    "web_search",
    "search_by_target",
    "resolve_target",
    "simbad_query",
    "search_mast",
    "catalog_find",
    "catalog_query",
    "search_papers",
    "web_extract_url",
    "alma_reference",
    "hips_cutout",
    "convert_coordinates",
    "calculate_redshift",
    "calculate_doppler_shift",
)

PACKS: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


def _pack(name: str, summary: str, tools: Sequence[str], pattern: Optional[str]) -> None:
    PACKS[name] = {
        "summary": summary,
        "tools": tuple(tools),
        "re": re.compile(pattern, re.I | re.S) if pattern else None,
    }


_pack(
    "alma",
    "ALMA Science Archive: hard archive queries, project census, per-source MOUS summary, observation details, "
    "cone search, QA2, files/products, archive links, public-band status, result filters and plots",
    (
        "query_alma_science_archive", "alma_project_census", "alma_source_summary", "alma_public_band_status",
        "alma_archive_link", "get_observation_details", "get_alma_qa2_status", "list_alma_files",
        "triage_alma_data_products", "search_by_position", "filter_results", "plot_alma_results",
        "browse_alma_guidance", "code_recipe", "advanced_search", "search_alma_with_keywords",
        "search_alma_co_in_redshift_range", "search_by_frequency", "find_alma_line_coverage", "check_co_lines",
        "check_line_coverage", "calculate_alma_sensitivity", "calculate_beam", "analyze_uv_coverage",
        "alma_bibliography", "search_papers_by_observation_id", "download_alma_data", "download_data",
        "plot_sky_map", "vlass_cutout",
    ),
    r"\balma\b|\basa\b|\bmous\b|uid://|\bband\s*\d|\bbands?\s+\d|\b20\d\d\.\d\.\d{5}\b|\bqa2\b|\bcycle\s*\d|"
    r"sub-?mm|submillimet|millimet|interferomet|\bbaselines?\b|\buv[- ]coverage|\bspws?\b|spectral window|"
    r"\bsensitivity\b|synthesi[sz]ed beam|\bbeam\s+size|\bvla\b|\bvlass\b|\bnrao\b|\bobservations?\b.{0,40}\barchive|"
    r"\barchive\b.{0,40}\bobservations?\b|\bproject\s+code|\bpi\s+name|proprietary|\bmeasurement\s*sets?\b",
)
_pack(
    "casa_scripts",
    "CASA calibration/imaging scripts and Jupyter notebooks",
    ("generate_casa_imaging_script", "generate_casa_calibration_script", "generate_jupyter_notebook"),
    r"\bcasa\b|tclean|\bcalibrat|imaging\s+script|\bnotebook\b|jupyter|\.ipynb|\bscript\b",
)
_pack(
    "multi_archive",
    "Other archives: MAST (JWST/HST/TESS) detailed search and products, ESO, CADC, IRSA, HEASARC "
    "(Chandra/XMM/Swift), cross-archive source matching, FITS header inspection, archive image overlays",
    (
        "search_mast_by_criteria", "get_mast_products", "download_mast_data", "search_eso_archive",
        "search_cadc_archive", "search_irsa", "heasarc_observations", "cross_archive_match",
        "match_cross_archive_sources", "match_perseus_protostars_alma_jwst", "cross_match_source",
        "inspect_fits_header", "archive_overlay", "overlay_archive_images", "survey_coverage",
    ),
    r"\bmast\b|\bjwst\b|\bhst\b|hubble|\bwebb\b|\bnircam\b|\bmiri\b|\bniriss\b|\bnirspec\b|\btess\b|\bkepler\b|"
    r"\bk2\b|\beso\b|\bvlt\b|\bmuse\b|\bsphere\b|\bcadc\b|\birsa\b|spitzer|herschel|\bchandra\b|\bxmm\b|"
    r"\bswift\b|nustar|heasarc|x-?ray|cross[- ]?match|cross[- ]archive|both\s+alma\s+and|\boverlay\b|"
    r"\bfits\b|\bjcmt\b|\bgemini\b|\bsubaru\b|\bproducts?\b|\bdownload",
)
_pack(
    "catalogs",
    "Catalogues: VizieR/Gaia/SIMBAD/NED queries and table crossmatch, Gaia distances, extinction, NED distances, "
    "exoplanets, pulsars, multi-database source lookups",
    (
        "catalog_crossmatch", "gaia_archive_query", "gaia_distance", "galactic_extinction", "ned_distance",
        "exoplanet_archive", "pulsar_lookup", "search_pulsars", "browse_schema", "search_catalog",
        "cross_match_source", "search_irsa", "vo_find_services",
    ),
    r"catalog|catalogue|vizier|simbad|\bgaia\b|parallax|proper\s+motion|\bdistances?\b|extinction|reddening|"
    r"e\(b-v\)|\bned\b|exoplanet|\bplanets?\b|pulsar|\bsdss\b|2mass|allwise|\bwise\b|\bw[1-4]\b|pan-?starrs|"
    r"\bschema\b|\bcolumns?\b|\btables?\b|stars?\s+(?:within|near|around)|cross[- ]?match|\bmagnitudes?\b|"
    r"\b(?:kepler|k2|toi|wasp|hat-p|gj|kelt|corot|tres|trappist|qatar|xo)-?\s?\d+\s?[b-h]?\b|orbital\s+period|"
    r"radial[- ]velocity|\btransit|\bhost\s+star|\bsources?\b.{0,40}\bwithin\b|within\s+\d+(?:\.\d+)?\s*(?:arc)?(?:min|sec|deg)|"
    # Capability questions answer from browse_schema, not memory (MANNA evals T02/T03).
    r"\bquirks?\b|\bpitfalls?\b|which\s+(?:astronomical\s+)?archives|archives?\s+(?:do\s+)?you\s+(?:have|know|support|access)",
)
_pack(
    "vo_coverage",
    "VO services and coverage: which surveys cover a position, survey footprints, MOC algebra, generic "
    "TAP/ADQL, SIA image and cone services, Multimodal Universe HATS catalogues",
    (
        "survey_coverage", "survey_covers_position", "survey_footprint", "moc_operations",
        "vo_find_services", "vo_list_tables", "vo_describe_table", "vo_adql_query", "vo_tap_job",
        "vo_cone_search", "vo_image_search", "list_mmu_hats_catalogs", "search_mmu_hats_catalog",
        "crossmatch_mmu_hats_catalogs",
    ),
    r"\bcover(?:s|ed|ing|age)?\b|footprint|\bmoc\b|\badql\b|\btap\b|\bvo\b|virtual\s+observatory|cone\s+search|"
    r"\bhats\b|multimodal\s+universe|\bmmu\b|lsdb|\bimaged\b|\bobserved\s+(?:by|with)\b|\bsia\b|\bssa\b|"
    r"which\s+surveys|\bany\s+survey|services?\b|"
    # MANNA evals 2026-10-01: NRAO ADQL (vo_adql_query at data-query.nrao.edu)
    # and "direct access / FITS URL" asks (vo_image_search) need this pack.
    r"\bnrao\b|\bobscore\b|\bvlba\b|\bevla\b|\bgbt\b|access\s+url|fits\s+(?:url|image|file)",
)
_pack(
    "spectral_lines",
    "Spectral lines: Splatalogue searches by molecule or frequency, line identification, line coverage",
    ("search_spectral_lines", "search_lines_by_molecule", "identify_spectral_line", "find_alma_line_coverage"),
    r"spectral\s+lines?|\btransitions?\b|molecul|splatalogue|rest\s+frequenc|\bghz\b|\bmhz\b|\bco\s*\(|\b13co\b|"
    r"\bc18o\b|\bhcn\b|\bhco\+|\bch3oh\b|methanol|maser|line\s+list|\blines?\b.{0,30}\bfrequenc|"
    r"\bfrequenc\w*\b.{0,30}\blines?\b|\bspecies\b|\bdeuter|\bisotopolog",
)
_pack(
    "literature",
    "Literature: fielded NASA ADS search with counts, abstracts, metrics, BibTeX, author papers and metrics, "
    "researcher profiles (OpenAlex), trends, consensus, paper method extraction, ADS libraries",
    (
        "ads_search", "find_citing_papers", "get_paper_abstract", "get_paper_metrics", "export_bibtex",
        "get_author_papers",
        "get_author_metrics", "lookup_researcher", "get_research_trends", "evaluate_consensus",
        "extract_paper_details", "reproduce_paper_methods", "search_papers_by_observation_id", "alma_bibliography",
        "list_ads_libraries", "get_ads_library_papers", "create_ads_library", "add_to_ads_library",
    ),
    r"\bpapers?\b|publication|\barticles?\b|literature|\bads\b|arxiv|bibcode|bibtex|\bcit(?:e|ed|ation)|"
    r"h-?index|\bauthors?\b|researcher|professor|\borcid\b|consensus|\btrends?\b|\blibrar(?:y|ies)\b|"
    r"published|\bjournal\b|\bstudies\b|\bstudy\b|who\s+is\b|"
    r"disput|challeng|rebut|refut|push(?:ed)?\s+back|responded|re-?analys",
)
_pack(
    "imaging",
    "Images: survey cutouts and multi-band panels, VLASS, finding charts, RGB composites, contour overlays, "
    "FITS rendering, source detection, aperture photometry, Gaussian fits, region stats, radial profiles, "
    "image statistics and differences",
    (
        "hips_multiband_panel", "vlass_cutout", "get_sky_image", "generate_finding_chart", "hips_rgb_composite",
        "hips_contour_overlay", "hips_aperture_photometry", "detect_sources", "fit_gaussian_source",
        "measure_region", "radial_profile", "image_statistics", "image_difference", "render_fits_image",
        "overlay_fits_images", "inspect_fits_header", "vlass_epoch_comparison",
    ),
    r"\bimages?\b|\bimaging\b|picture|cutout|show\s+me|look\s+like|finding\s+chart|\brgb\b|colou?r\s+composite|"
    r"contour|aperture|detect\s+sources|source\s+detection|gaussian|radial\s+profile|"
    r"surface\s+brightness|\bnoise\b|\brms\b|\bfits\b|\bhips\b|aladin|\bpanels?\b|multi-?wavelength|"
    r"\brender|visuali[sz]|\bfield\s+of\b|\bdss\b|\boverlay\b",
)
_pack(
    "cubes",
    "Spectral cubes: moment maps, spectrum extraction, line fitting, position-velocity slices, spectrum plots",
    ("compute_moment_map", "extract_spectrum", "fit_spectral_line", "pv_slice", "plot_spectrum"),
    r"\bcubes?\b|moment[- ]?(?:map|[012]\b)|extract\w*\s+(?:a\s+|the\s+)?spectr|spectrum\s+at|"
    r"pv[- ]?(?:slice|diagram)|position[- ]velocity|fit\w*\s+(?:a\s+|the\s+)?(?:spectral\s+)?line|line\s+profile|"
    r"plot\w*\s+(?:a\s+|the\s+)?spectrum",
)
_pack(
    "time_domain",
    "Time domain and solar system: TESS/Kepler light curves, period search, ZTF alerts, TNS transients, "
    "sky-monitor watchlist, gravitational waves (GWOSC/GWTC), GCN circulars, Fermi-LAT light curves, "
    "variability analysis, asteroids/comets, JPL Horizons ephemerides",
    (
        "search_space_lightcurves", "plot_space_lightcurve", "period_search", "search_ztf_alerts", "ztf_object",
        "tns_object", "monitor_add_target", "monitor_check_now", "monitor_list_targets", "monitor_remove_target",
        "get_latest_gw_events", "search_gwtc_catalog", "summarize_gcn_circular", "fermi_lcr_lightcurve",
        "variability_analysis", "moving_object_check", "solar_system_ephemeris", "ztf_light_curve", "ztf_stamps", "gaia_archive_query",
    ),
    r"light\s*curves?|lightcurve|\bperiods?\b|periodogram|variab|transient|supernova|\bsn\s?20\d\d|\bat\s?20\d\d|"
    r"\btns\b|\bztf\b|\balerts?\b|gamma[- ]ray\s+burst|\bgrb\b|\bgcn\b|gravitational|\bgw\d|\bgw\b|ligo|kagra|"
    r"asteroid|comet|minor\s+planet|ephemeri|horizons|solar\s+system|\bmonitor|watchlist|\bflares?\b|outburst|"
    r"\bfermi\b|\btess\b|\bkepler\b|\beclips|\bepochs?\b|\bmerger|rr\s*lyr|cepheid|pulsation",
)
_pack(
    "blazar_sed",
    "Blazars and SEDs: MMDC multi-epoch SEDs, light curves and emission-model fits, NED literature SEDs, "
    "radio continuum SEDs, Fermi-LAT light curves, variability analysis",
    (
        "mmdc_sed", "mmdc_lightcurve", "mmdc_model", "mmdc_model_job", "ned_sed_plot", "radio_sed",
        "fermi_lcr_lightcurve", "variability_analysis",
    ),
    r"blazar|bl\s*lac|\bfsrq|\bmmdc\b|\bseds?\b|spectral\s+energy\s+distribution|\bssc\b|\beic\b|inverse\s+compton|"
    r"hadronic|neutrino|\b4fgl\b|radio\s+(?:sed|spectrum|continuum)|broad-?band|\bmrk\s*\d|markarian|\b3c\s*\d|"
    r"\bpks\b|\bnvss\b|\bfirst\b\s+survey|\bgleam\b|\btgss\b|\bsumss\b",
)
_pack(
    "datalab",
    "NOIRLab Astro Data Lab catalogue access: list/describe catalogues (DELVE, DES, NSC, SMASH, Legacy Surveys, "
    "Gaia, DESI, SDSS), governed cone counts and row selections, SQL, q3c crossmatch, stored results and saved "
    "tables, background jobs, notebooks",
    (
        "datalab_list_catalogs", "datalab_describe_table", "datalab_cone_count", "datalab_select_catalog_rows",
        "datalab_sql_query", "datalab_q3c_crossmatch", "xmatch_user_list", "datalab_get_result",
        "datalab_save_result", "datalab_list_my_tables", "datalab_load_my_table", "datalab_export_notebook",
        "datalab_job_status", "datalab_job_results", "datalab_job_cancel", "svo_filter_wavelength",
    ),
    r"data\s?lab|noirlab|\bdelve\b|\bdes\b|\bdes\s*dr|decam|\bnsc\b|\bsmash\b|legacy\s+survey|\bls\s*dr\d|\bdr9\b|"
    r"\bdr10\b|\bdesi\b|\bsdss\b|\bboss\b|\bq3c\b|healpix|density\s+map|over-?densit|\bsatellites?\b|"
    r"dwarf\s+(?:galax|companion|satellite)|stellar\s+stream|tidal|colou?r[- ]magnitude|colou?r[- ]colou?r|"
    r"\bcmd\b|\bhr\s+diagram|hertzsprung|white\s+dwarf|\bmydb\b|my\s+tables?|result_id|\blss\b|cosmic\s+web|"
    r"\bwedge\b|n\(z\)|redshift\s+distribution|\bsql\b|\bgalaxy\s+sample|\blrgs?\b|"
    r"\belgs?\b|\bstars?\b.{0,40}\b(?:within|cone|degree|arcmin)|\bcone\b|\bglobular|open\s+cluster|"
    r"\bsources?\b.{0,40}\bwithin\b|within\s+\d+(?:\.\d+)?\s*arcmin",
)
_pack(
    "datalab_science",
    "Data Lab one-call science: CMD / colour-colour / HR selection diagrams, scatter plots, sky and HEALPix "
    "density maps, overdensity vetting and tiled searches, dwarf-satellite and tidal-stream searches, variable "
    "stars, light curves and period folds, galaxy SED samples, DESI n(z) and large-scale-structure wedges",
    (
        "datalab_color_magnitude_diagram", "datalab_color_color_diagram", "datalab_selection_diagram",
        "datalab_catalog_scatter", "datalab_sky_density_map", "datalab_healpix_density_map",
        "datalab_density_aggregate", "datalab_density_vetting", "datalab_confirm_sky_area", "datalab_tiled_search",
        "datalab_satellite_search", "datalab_stream_selection", "datalab_variable_candidates",
        "datalab_star_lightcurve", "datalab_period_fold", "datalab_sed_sample", "datalab_sed_plot",
        "datalab_target_class_summary", "datalab_lss_wedge", "datalab_cutout_grid",
    ),
    r"healpix|density\s+map|over-?densit|\bsatellites?\b|dwarf\s+(?:galax|companion|satellite)|stellar\s+stream|"
    r"tidal|colou?r[- ]magnitude|colou?r[- ]colou?r|\bcmd\b|\bhr\s+diagram|hertzsprung|white\s+dwarf|\blss\b|"
    r"cosmic\s+web|\bwedge\b|n\(z\)|redshift\s+distribution|\bgalaxy\s+sample|\blrgs?\b|\belgs?\b",
)
_pack(
    "datalab_images",
    "Data Lab images: SIA image inventory (bandpass, exposure, stack type), single-band cutouts, cutout grids, "
    "colour images",
    ("datalab_sia_search", "datalab_image_cutout", "datalab_cutout_grid", "datalab_color_image"),
    r"\bsia\b|image\s+inventory|\bexposures?\b|exptime|\bstacks?\b|cutouts?|\bimages?\b|colou?r\s+image|"
    r"\bimaging\b|\bpicture|\bdeepest\b",
)
_pack(
    "sparcl_spectra",
    "SPARCL optical spectra (DESI/SDSS/BOSS): find, survey-scale search, retrieve, plot, stack",
    ("sparcl_find_spectra", "sparcl_search_spectra", "sparcl_get_spectrum", "sparcl_plot_spectrum",
     "sparcl_stack_spectra"),
    r"sparcl|\bspectr(?:um|a|oscop\w*)\b|\bdesi\b|\bsdss\b|\bboss\b|\bstack\w*\b|co-?add",
)
_pack(
    "web_advanced",
    "Web beyond search: extract a specific URL, crawl or map a site, multi-source web research, browser",
    (
        "web_extract_url", "web_crawl_site", "web_map_site", "web_research", "web_research_status",
        "navigate_to_url", "read_page", "click_element",
    ),
    r"https?://|\bwww\.|\burls?\b|\bwebsite\b|\bweb\s*page\b|\bcrawl|\bsite\b|\bthis\s+page|\bbrowse|"
    r"\bextract\w*\s+(?:from|the)\b|\bin[- ]depth\s+research|\bdeep\s+research",
)
_pack(
    "calculators",
    "Calculators: velocity frames and distances, beam size, ALMA sensitivity, coordinate and redshift tools",
    ("velocity_frame_distance", "calculate_beam", "calculate_alma_sensitivity", "galactic_extinction"),
    r"\bconvert|\bcalculat|\bcompute|velocity\s+frame|\blsr\b|\bgsr\b|\bcmb\s+frame|hubble\s+flow|"
    r"luminosity\s+distance|lookback|angular\s+(?:size|diameter)|\bkpc\b|\bmpc\b",
)
_pack(
    "image_generation",
    "AI image generation (illustrations, artist's impressions) on the user's own Gemini/OpenAI key",
    ("generate_image",),
    r"\b(?:generate|create|draw|make|render|paint|design|sketch|illustrate)\w*\b.{0,40}"
    r"\b(?:images?|pictures?|illustrations?|drawings?|artwork|art|logo|poster|icon|cartoon|sketch|"
    r"artist'?s\s+(?:impression|concept|rendering))\b|"
    r"\bartist'?s\s+(?:impression|concept)\b|\bnano[- ]?banana\b|\bgpt-image\b|\bimage\s+generation\b",
)

PACK_ORDER: Sequence[str] = tuple(PACKS.keys())

# With the Data Lab pack selected, these words also need the science pack.
_DATALAB_WEAK_RE = re.compile(
    r"\bplot|diagram|\bmaps?\b|scatter|light\s*curve|\bperiods?\b|variab|\bseds?\b|spectral\s+energy|selection|"
    r"proper\s+motion|parallax|absolute\s+mag|\bclusters?\b|\bdwarfs?\b|\bstreams?\b|candidates?|\bdensity|"
    r"bitmask|target\s+class|\bquasars?\b|\bqsos?\b|redshift|\bcutouts?\b|\bgrid\b|\bpeaks?\b",
    re.I,
)

# One-line pack descriptions for the find_tools schema (sent on every call).
_SHORT = {
    'alma': 'ALMA archive queries, census, MOUS summaries, QA2, files, guidance, code, calculators, downloads',
    'casa_scripts': 'CASA calibration/imaging scripts, Jupyter notebooks',
    'multi_archive': 'MAST/JWST/HST/TESS products, ESO, CADC, IRSA, HEASARC, cross-archive matching, overlays',
    'catalogs': 'VizieR/Gaia/SIMBAD/NED catalogues and crossmatch, distances, extinction, exoplanets, pulsars',
    'vo_coverage': 'survey coverage/footprints/MOC, generic TAP/ADQL/SIA/cone services, MMU HATS',
    'spectral_lines': 'Splatalogue line search and identification',
    'literature': 'ADS search/metrics/BibTeX, authors, researcher profiles, trends, paper methods, libraries',
    'imaging': 'cutouts, panels, RGB, contours, FITS rendering, photometry, source detection, image stats',
    'cubes': 'spectral cubes: moment maps, spectra, line fits, PV slices',
    'time_domain': 'light curves, periods, ZTF/TNS transients, monitor, GW events, GCN, Fermi, asteroids, ephemerides',
    'blazar_sed': 'MMDC blazar SEDs/light curves/model fits, NED and radio SEDs',
    'datalab': 'Data Lab catalogue listing, counts, row selection, SQL, crossmatch, saved tables, jobs',
    'datalab_science': 'Data Lab CMD/colour diagrams, density maps, satellites, streams, variables, SEDs, n(z), LSS',
    'datalab_images': 'Data Lab SIA image inventory, cutouts, colour images',
    'sparcl_spectra': 'SPARCL DESI/SDSS spectra search, plot, stack',
    'web_advanced': 'extract/crawl/map web pages, deep web research, browser',
    'calculators': 'velocity frames, beam, ALMA sensitivity, extinction',
}
for _p, _t in _SHORT.items():
    if _p in PACKS:
        PACKS[_p]["short"] = _t

# One-shot intents and route directives name the tool they need; these are
# looked up by name, so a forced tool is always offered.
_TOOL_NAME_RE = re.compile(r"`?\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b`?")


def packs_for_tool(name: str) -> List[str]:
    return [p for p, spec in PACKS.items() if name in spec["tools"]]


def packs_from_text(text: str) -> List[str]:
    t = str(text or "")
    if not t.strip():
        return []
    return [p for p, spec in PACKS.items() if spec["re"] is not None and spec["re"].search(t)]


def tool_names_in_text(text: str, registered: Iterable[str]) -> List[str]:
    reg = set(registered)
    seen: List[str] = []
    for m in _TOOL_NAME_RE.finditer(str(text or "")):
        n = m.group(1)
        if n in reg and n not in seen:
            seen.append(n)
    return seen


# ── Per-conversation memory of the tools a chat used ─────────────────────

_STICKY_TURNS = 2
_MAX_CONVS = 1000
_conv_lock = threading.Lock()
_conv_tools: "OrderedDict[str, List[Set[str]]]" = OrderedDict()


def remember_called(conversation_id: Optional[str], called: Iterable[str]) -> None:
    """Record the tools one turn of this conversation actually called."""
    if not conversation_id:
        return
    names = {str(n) for n in called if n and n != FIND_TOOLS}
    with _conv_lock:
        turns = _conv_tools.pop(str(conversation_id), [])
        turns = (turns + [names])[-_STICKY_TURNS:]
        _conv_tools[str(conversation_id)] = turns
        while len(_conv_tools) > _MAX_CONVS:
            _conv_tools.popitem(last=False)


def recently_called(conversation_id: Optional[str]) -> Set[str]:
    if not conversation_id:
        return set()
    with _conv_lock:
        turns = _conv_tools.get(str(conversation_id)) or []
        out: Set[str] = set()
        for s in turns:
            out |= s
        return out


def prior_user_message(history: Any, current_query: str) -> str:
    """The user's previous message in this chat ('' when there is none)."""
    cur = str(current_query or "").strip()
    for m in reversed(list(history or [])):
        if isinstance(m, dict) and m.get("role") == "user":
            c = str(m.get("content", "") or "").strip()
            if c and c != cur:
                return c[:2000]
    return ""


# ── Selection ────────────────────────────────────────────────────────────

def select(
    query: str,
    *,
    registered: Iterable[str],
    directive_text: str = "",
    forced_tools: Iterable[str] = (),
    conversation_id: Optional[str] = None,
    prior_user_text: str = "",
    web_mode: str = "auto",
    has_attachments: bool = False,
    extra_packs: Iterable[str] = (),
) -> Dict[str, Any]:
    """Which tools to offer this turn: {"names": [...ordered...], "packs": [...], "reasons": {...}}."""
    reg = list(registered)
    reg_set = set(reg)
    reasons: Dict[str, str] = {}
    packs: List[str] = []

    def add_pack(p: str, why: str) -> None:
        if p in PACKS and p not in packs:
            packs.append(p)
            reasons[p] = why

    for p in packs_from_text(query):
        add_pack(p, "question")
    for p in extra_packs:
        add_pack(p, "router")
    # A short follow-up ("and in band 7?", "plot that") inherits the previous
    # question's packs.
    if prior_user_text and len(str(query or "")) < 160:
        for p in packs_from_text(prior_user_text):
            add_pack(p, "previous question")
    extra_tools: List[str] = []
    for n in list(forced_tools) + tool_names_in_text(directive_text, reg_set):
        if n in reg_set and n not in extra_tools:
            extra_tools.append(n)
            for p in packs_for_tool(n):
                add_pack(p, f"route needs {n}")
    for n in sorted(recently_called(conversation_id)):
        if n in reg_set:
            if n not in extra_tools:
                extra_tools.append(n)
            for p in packs_for_tool(n):
                add_pack(p, f"chat used {n}")
    if "datalab" in packs and "datalab_science" not in packs and _DATALAB_WEAK_RE.search(str(query or "")):
        add_pack("datalab_science", "data lab + analysis words")
    if web_mode == "always":
        add_pack("web_advanced", "web mode always")
    if has_attachments:
        add_pack("imaging", "attachment")
        add_pack("cubes", "attachment")

    names = order_names(reg, packs, extra_tools)
    return {"names": names, "packs": packs, "reasons": reasons, "extra_tools": extra_tools}


def order_names(registered: Sequence[str], packs: Sequence[str], extra: Iterable[str] = ()) -> List[str]:
    """Core first, then packs in PACK_ORDER, then any extra registered tools
    (in registry order). Stable for a given set, so prefixes cache."""
    reg_set = set(registered)
    out: List[str] = []
    seen: Set[str] = set()

    def push(n: str) -> None:
        if n in reg_set and n not in seen:
            seen.add(n)
            out.append(n)

    for n in CORE:
        push(n)
    for p in PACK_ORDER:
        if p in packs:
            for n in PACKS[p]["tools"]:
                push(n)
    extra_set = set(extra)
    for n in registered:
        if n in extra_set:
            push(n)
    return out


# ── find_tools meta-tool ─────────────────────────────────────────────────

def find_tools_schema() -> Dict[str, Any]:
    lines = [f"- {p}: {spec.get('short') or spec['summary']}" for p, spec in PACKS.items()]
    return {
        "type": "function",
        "name": FIND_TOOLS,
        "description": (
            "Only part of Quasar's toolset is loaded for this turn. When none of the loaded tools can do what "
            "the request needs, call this FIRST with what you need (and the packs you think fit); the matching "
            "tools become callable in your next step. Never tell the user a capability is unavailable before "
            "calling this. Packs:\n" + "\n".join(lines)
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "need": {"type": "string", "description": "What you need to do, in a few words."},
                "packs": {"type": "array", "items": {"type": "string", "enum": list(PACKS.keys())},
                          "description": "Packs to load, if you know them."},
            },
            "required": ["need"],
        },
    }


_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "to", "for", "and", "or", "in", "on", "with", "from", "by", "at", "is", "i",
         "need", "want", "tool", "tools", "data", "get", "find", "use", "do", "this", "that", "my", "me"}


def resolve_find_tools(args: Dict[str, Any], registry_tools: Sequence[Any], offered: Iterable[str]) -> Dict[str, Any]:
    """Tools to add for a find_tools call: named packs + packs whose triggers
    match ``need`` + the best keyword matches over tool names/descriptions.
    Returns {"add": [names not yet offered], "packs": [...], "text": tool output}."""
    if not isinstance(args, dict):  # valid JSON but not an object (guard CX-04)
        args = {"need": str(args or "")}
    need = str(args.get("need", "") or "")
    _packs_arg = args.get("packs") or []
    if isinstance(_packs_arg, str):
        _packs_arg = [_packs_arg]
    asked = [str(p) for p in (_packs_arg if isinstance(_packs_arg, list) else []) if str(p) in PACKS]
    packs = list(dict.fromkeys(asked + packs_from_text(need)))
    offered_set = set(offered)
    by_name = {getattr(t, "name", ""): t for t in registry_tools}
    names: List[str] = []
    for p in packs:
        names.extend(n for n in PACKS[p]["tools"] if n in by_name)
    words = {w for w in _WORD_RE.findall(need.lower()) if w not in _STOP and len(w) > 2}
    if words:
        scored = []
        for n, t in by_name.items():
            hay = (n.replace("_", " ") + " " + str(getattr(t, "description", ""))[:400]).lower()
            hits = sum(1 for w in words if w in hay)
            name_hits = sum(1 for w in words if w in n)
            if hits:
                scored.append((name_hits * 2 + hits, n))
        scored.sort(key=lambda x: (-x[0], x[1]))
        names.extend(n for _, n in scored[:8])
    add = [n for n in dict.fromkeys(names) if n not in offered_set and n != FIND_TOOLS]
    if add:
        lines = []
        for n in add:
            desc = str(getattr(by_name.get(n), "description", "") or "").strip().split("\n")[0]
            first = re.split(r"(?<=[.!?])\s", desc, maxsplit=1)[0][:160]
            lines.append(f"- {n}: {first}")
        text = ("Loaded " + str(len(add)) + " more tool(s); call them directly now:\n" + "\n".join(lines))
    elif packs and any(n in by_name and n in offered_set for p in packs for n in PACKS[p]["tools"]):
        text = ("The tools for that (" + ", ".join(packs) + ") are already loaded; call them directly. "
                "If none of them fits, say what is missing.")
    elif packs:
        # e.g. web tools while the user switched web search off (guard CX-03)
        text = ("The tools for that (" + ", ".join(packs) + ") are not available in this session. Answer with "
                "the loaded tools, and tell the user plainly what could not be done.")
    else:
        text = ("No additional tools match that need. Pack names: " + ", ".join(PACKS.keys()) +
                ". Call find_tools again with a pack name, or answer with the loaded tools.")
    return {"add": add, "packs": packs, "text": text}


# ── "I don't have a tool for that" detector (round-0 retry with all tools) ──

_LACKS_TOOL_RE = re.compile(
    r"\b(?:i\s+(?:do\s+not|don'?t|cannot|can'?t)\s+(?:currently\s+)?(?:have|access)\s+(?:access\s+to\s+)?"
    r"(?:a|any|the)?\s*(?:\w+\s+){0,3}(?:tools?|functions?|capabilit\w+)|"
    r"(?:no|none\s+of\s+(?:the|my))\s+(?:available\s+)?(?:tools?|functions?)\s+(?:available\s+)?(?:for|to|that|can|lets?)|"
    r"(?:not|isn'?t)\s+(?:available|supported)\s+(?:in|with|among)\s+(?:my|the|this)\s+(?:current\s+)?(?:tool\s*set|toolset|tools)|"
    r"(?:lack|without)\s+(?:a|the|any)\s+(?:\w+\s+){0,2}tool)",
    re.I,
)


def says_lacks_tool(text: str) -> bool:
    return bool(_LACKS_TOOL_RE.search(str(text or "")))


__all__ = [
    "ENV", "FIND_TOOLS", "CORE", "PACKS", "PACK_ORDER", "enabled", "select", "order_names", "packs_for_tool",
    "packs_from_text", "tool_names_in_text", "remember_called", "recently_called", "find_tools_schema",
    "resolve_find_tools", "says_lacks_tool",
]
