"""core/tool_registrations.py — the QuasarAgent tool-registration body extracted verbatim
from _register_tools (masterplan S13). Byte-parity: the former method body, with every
attribute access rewritten from method form to the `agent` parameter."""
from __future__ import annotations

from typing import TYPE_CHECKING

from core.tools import Tool
from services.mmu_hats import MMU_HATS_CATALOGS

if TYPE_CHECKING:
    from core.agent import QuasarAgent


def register_tools(agent: "QuasarAgent") -> None:
    """Register available tools with the agent using OpenAI Schemas"""

    # Search tools
    agent.tool_registry.register(Tool(
        name="search_by_position",
        description="Search NRAO archives by sky position (cone search)",
        function=agent._alma_tool_fn("search_by_position", log_name="_search_by_position"),
        parameters={
            "type": "object",
            "properties": {
                "ra": {"type": "number", "description": "Right ascension in degrees"},
                "dec": {"type": "number", "description": "Declination in degrees"},
                "radius": {"type": "number", "description": "Search radius in degrees (default 0.5)"},
                "facility": {"type": "string", "enum": ["VLA", "VLBA", "ALMA", "GBT"], "description": "Observatory facility. Default to ALMA."},
                "band": {"type": "string", "description": "ALMA band number(s) to filter (3-10). Pass a single band like '6' or multiple like '6,7'."},
                "scan_intent": {"type": "string", "description": "ALMA scan intent to filter, such as TARGET, BANDPASS, PHASE, FLUX, or WVR. ONLY pass if user explicitly asks for a scan intent."},
                "max_results": {"type": "integer", "description": "Maximum results to return"}
            },
            "required": ["ra", "dec"]
        }
    ))

    agent.tool_registry.register(Tool(
        name="search_by_target",
        description=(
            "Search the ALMA archive (default) by target name. Supports multiple targets "
            "separated by 'and' or comma (e.g. 'M87 and Sz65' or 'M87, NGC 1068').\n"
            "Pass facility='VLA', 'VLBA', or 'GBT' to search the NRAO archive instead — "
            "ONLY when the user explicitly asks for those telescopes.\n"
            "CRITICAL: ONLY pass optional filter parameters (band, resolution, frequency, scan_intent) "
            "if the user EXPLICITLY mentions them. Do NOT invent default values. "
            "If the user just says 'Find ALMA data of M87', pass ONLY target_name='M87' "
            "with NO other parameters — this returns ALL observations across all bands.\n"
            "Only pass band=6 if the user says 'Band 6'. Only pass max_resolution if "
            "the user specifies a resolution constraint. Omitting a filter means 'no filter'."
        ),
        function=agent._alma_tool_fn("search_by_target", log_name="_search_by_target"),
        parameters={
            "type": "object",
            "properties": {
                "target_name":    {"type": "string",  "description": "Astronomical target name (e.g. 'TW Hya', 'HL Tau'). For multiple targets use comma or 'and': 'M87, NGC 1068'."},
                "facility":       {"type": "string",  "enum": ["ALMA", "VLA", "VLBA", "GBT"], "description": "Observatory. Default ALMA."},
                "band":           {"type": "string", "description": "ALMA band number(s) to filter (3-10). For a single band pass '6'. For multiple bands pass comma-separated like '6,7'. ONLY pass if user explicitly asks for a specific band."},
                "max_resolution": {"type": "number",  "description": "Maximum angular resolution in arcsec. ONLY pass if user specifies."},
                "min_resolution": {"type": "number",  "description": "Minimum angular resolution in arcsec. ONLY pass if user specifies."},
                "min_freq_ghz":   {"type": "number",  "description": "Minimum frequency in GHz. ONLY pass if user specifies."},
                "max_freq_ghz":   {"type": "number",  "description": "Maximum frequency in GHz. ONLY pass if user specifies."},
                "min_exp_s":      {"type": "number",  "description": "Minimum integration time in seconds. ONLY pass if user specifies."},
                "scan_intent":    {"type": "string",  "description": "ALMA scan intent to filter, such as TARGET, BANDPASS, PHASE, FLUX, or WVR. ONLY pass if user explicitly asks for a scan intent."},
                "public_only":    {"type": "boolean", "description": "Only return publicly available data."},
            },
            "required": ["target_name"]
        }
    ))

    agent.tool_registry.register(Tool(
        name="search_by_frequency",
        description="Search archives by frequency range. Defaults to ALMA; pass facility='VLA'/'VLBA'/'GBT' for the NRAO archive.",
        function=agent._alma_tool_fn("search_by_frequency", log_name="_search_by_frequency"),
        parameters={
            "type": "object",
            "properties": {
                "min_freq_ghz": {"type": "number", "description": "Minimum frequency in GHz"},
                "max_freq_ghz": {"type": "number", "description": "Maximum frequency in GHz"},
                "facility": {"type": "string", "enum": ["ALMA", "VLA", "VLBA", "GBT"], "description": "Observatory facility. Default ALMA."},
                "max_results": {"type": "integer", "description": "Max results"}
            },
            "required": ["min_freq_ghz", "max_freq_ghz"]
        }
    ))

    # ── CADC Multi-wavelength Archive Search ──────────────────
    agent.tool_registry.register(Tool(
        name="search_cadc_archive",
        description=(
            "Search the Canadian Astronomy Data Centre (CADC) for multi-wavelength observations "
            "from JWST, HST, JCMT, CFHT, Gemini, and other telescopes. Uses the IVOA ObsCore TAP "
            "service. Complements ALMA searches with optical/infrared/submm data."
        ),
        function=agent._alma_tool_fn("search_cadc_archive", log_name="_search_cadc"),
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "Target name to resolve via SIMBAD (e.g. 'M31', 'TW Hya')"},
                "ra":          {"type": "number", "description": "RA in decimal degrees (ICRS). Used if target_name is not given."},
                "dec":         {"type": "number", "description": "Dec in decimal degrees (ICRS). Used if target_name is not given."},
                "radius":      {"type": "number", "description": "Search radius in degrees. Default 0.05 (~3 arcmin).", "default": 0.05},
                "collection":  {"type": "string", "description": "Filter by telescope collection (e.g. 'JWST', 'HST', 'JCMT', 'CFHT', 'Gemini'). Leave empty for all."},
                "max_results": {"type": "integer", "description": "Maximum results to return. Default 500.", "default": 500},
            },
            "required": []
        }
    ))

    agent.tool_registry.register(Tool(
        name="get_observation_details",
        description="Get detailed information about a specific observation",
        function=agent._alma_tool_fn("get_observation_details"),
        parameters={
            "type": "object",
            "properties": {
                "obs_id": {"type": "string", "description": "Observation ID or execution block ID"}
            },
            "required": ["obs_id"]
        }
    ))

    # Basic download (legacy)
    agent.tool_registry.register(Tool(
        name="download_data",
        description="Legacy download tool (Use download_alma_data instead)",
        function=agent._alma_tool_fn("download_data"),
        parameters={
            "type": "object",
            "properties": {
                 "obs_id": {"type": "string", "description": "Observation ID"}
            },
            "required": ["obs_id"]
        }
    ))

    agent.tool_registry.register(Tool(
        name="analyze_uv_coverage",
        description="Analyze UV coverage for an observation",
        function=agent._alma_tool_fn("analyze_uv_coverage"),
        parameters={
            "type": "object",
            "properties": {
                "ms_path": {"type": "string", "description": "Path to measurement set"}
            },
            "required": ["ms_path"]
        }
    ))

    # NEW: ALminer Tools
    agent.tool_registry.register(Tool(
        name="search_alma_with_keywords",
        description="Search ALMA archives using specific keywords (pi_name, project_code, etc.)",
        function=agent._alma_tool_fn("search_alma_with_keywords"),
        parameters={
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "object", 
                    "description": "Dictionary of keywords e.g. {'pi_name': 'Smith', 'project_code': '2017.1...'}",
                    "additionalProperties": True 
                }
            },
            "required": ["keywords"]
        }
    ))

    agent.tool_registry.register(Tool(
        name="advanced_search",
        description=(
            "Execute a custom ADQL/TAP query directly on the ALMA Science Archive (ivoa.obscore table).\n"
            "ALMA ONLY — NOT for NOIRLab Data Lab catalogs (gaia_dr3/des_dr1/desi_dr1/nsc_dr2/smash/...): "
            "use datalab_sql_query for those.\n"
            "IMPORTANT: The obscore table has NO 'redshift' column. Use frequency/bandwidth containment instead.\n"
            "To find observations covering a specific frequency nu_ghz:\n"
            "  WHERE (frequency - 0.5*bandwidth/1e9) < {nu_ghz} AND (frequency + 0.5*bandwidth/1e9) > {nu_ghz}\n"
            "Key columns: target_name, s_ra, s_dec, frequency (GHz), bandwidth (Hz), scientific_category,\n"
            "  science_keyword, proposal_id, member_ous_uid, t_exptime, s_resolution, band_list.\n"
            "Add OFFSET 0 ROWS FETCH NEXT 500 ROWS ONLY to limit results."
        ),
        function=agent._alma_tool_fn("advanced_search"),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "ADQL TAP query string for ivoa.obscore"}
            },
            "required": ["query"]
        }
    ))

    agent.tool_registry.register(Tool(
        name="search_alma_co_in_redshift_range",
        description=(
            "Search the ALMA archive for observations that cover CO emission lines "
            "for galaxies at a given redshift range. Handles the CO rest-frequency → "
            "observed-frequency conversion and TAP frequency-containment query automatically. "
            "Use this for any query like 'galaxies at z=1-2 with CO coverage' or "
            "'ALMA CO detections at high redshift'.\n"
            "CO transitions checked: J=1-0 (115.3 GHz), J=2-1 (230.5), J=3-2 (345.8), "
            "J=4-3 (461.0), J=5-4 (576.3), J=6-5 (691.5), J=7-6 (806.7).\n"
            "Returns: target_name, proposal_id, CO_transition, obs_frequency_ghz, bandwidth_ghz."
        ),
        function=agent._alma_tool_fn("search_alma_co_in_redshift_range"),
        parameters={
            "type": "object",
            "properties": {
                "z_min": {"type": "number", "description": "Minimum redshift (e.g. 1.0)"},
                "z_max": {"type": "number", "description": "Maximum redshift (e.g. 2.0)"},
                "science_category": {
                    "type": "string",
                    "description": "Optional ALMA science category filter (e.g. 'Galaxy evolution', 'Cosmology', 'Active galaxies'). Leave empty for all.",
                    "default": ""
                },
                "max_results": {"type": "integer", "description": "Maximum rows to return (default 500)", "default": 500}
            },
            "required": ["z_min", "z_max"]
        }
    ))

    agent.tool_registry.register(Tool(
        name="query_alma_science_archive",
        description=(
            "Run deterministic ALMA Science Archive query templates for hard archive-science questions. "
            "Use this instead of raw ADQL for: Cycle N project counts, Sun/solar projects, projects using "
            "12m+7m+total-power arrays, high-resolution Band N continuum candidates for a target, projects "
            "covering a required molecular line set such as 12CO/13CO/C18O in the same project, and "
            "bandwidth-switching calibration diagnostics."
        ),
        function=agent._alma_tool_fn("query_alma_science_archive"),
        parameters={
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": [
                        "cycle_solar_projects",
                        "cycle_array_combo_projects",
                        "high_resolution_band_data",
                        "line_set_projects",
                        "redshifted_line_projects",
                        "bandwidth_switching_candidates",
                    ],
                    "description": "Specific ALMA science/archive query template to run."
                },
                "cycle": {"type": "integer", "description": "ALMA cycle number, e.g. 10."},
                "target": {"type": "string", "description": "Target/source name, e.g. HH212."},
                "band": {"type": "integer", "description": "ALMA band number, e.g. 6 or 7."},
                "max_resolution_arcsec": {"type": "number", "description": "Maximum angular resolution in arcsec for high-resolution data."},
                "arrays": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Required array types, e.g. ['12m','7m','TP']."
                },
                "lines": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Required lines, e.g. ['12CO','13CO','C18O']."
                },
                "topic_filter": {"type": "string", "description": "Optional science keyword/category filter such as protostellar disks."},
                "redshift_min": {"type": "number", "description": "Minimum redshift for redshifted rest-line searches."},
                "redshift_max": {"type": "number", "description": "Maximum redshift for redshifted rest-line searches."},
                "rest_species": {"type": "string", "description": "Rest species for line searches, e.g. CO, 12CO, 13CO, C18O."},
                "science_category": {"type": "string", "description": "Optional ALMA science category filter, e.g. Galaxy evolution."},
                "require_same_project": {"type": "boolean", "description": "Require requested line matches in the same proposal_id. Default true."},
                "include_adql": {"type": "boolean", "description": "Include executed ADQL in provenance. Default true."},
                "max_results": {"type": "integer", "description": "Maximum TAP rows to fetch before grouping. Default 5000."},
            },
            "required": ["query_type"]
        },
        category="archive"
    ))

    agent.tool_registry.register(Tool(
        name="match_cross_archive_sources",
        description=(
            "Cross-match a source catalog against ALMA and MAST/JWST observations. "
            "Use this for source-list location questions such as Perseus protostars observed with ALMA and JWST, "
            "or pass explicit source coordinates for any catalog."
        ),
        function=agent._alma_tool_fn("match_cross_archive_sources"),
        parameters={
            "type": "object",
            "properties": {
                "catalog_name": {"type": "string", "description": "Catalog key. Built-in: 'perseus_protostars'. For arbitrary catalogs, pass sources."},
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_name": {"type": "string"},
                            "ra": {"type": "number", "description": "ICRS right ascension in degrees."},
                            "dec": {"type": "number", "description": "ICRS declination in degrees."},
                        },
                        "required": ["ra", "dec"],
                    },
                    "description": "Optional inline coordinate catalog. Each item needs source_name/name plus ra and dec in degrees."
                },
                "archives": {"type": "array", "items": {"type": "string"}, "description": "Archives to match, e.g. ['ALMA','JWST']."},
                "radius_arcsec": {"type": "number", "description": "Match radius in arcsec. Default 5."},
                "max_sources": {"type": "integer", "description": "Max catalog sources to test. Default 12."},
                "max_alma_rows": {"type": "integer", "description": "Max ALMA TAP rows to fetch. Default 5000."},
                "max_mast_results_per_source": {"type": "integer", "description": "Max MAST/JWST rows per source. Default 80."},
                "require_all_archives": {"type": "boolean", "description": "If true, only return sources matched in every requested archive. Default false for diagnostic cross-match tables."},
            },
            "required": []
        },
        category="archive"
    ))

    agent.tool_registry.register(Tool(
        name="match_perseus_protostars_alma_jwst",
        description=(
            "Cross-match a built-in Perseus protostar source list against ALMA and MAST/JWST observations. "
            "Use this for questions like 'Show locations of protostars in Perseus observed with ALMA and JWST'. "
            "Returns sources with both ALMA and JWST matches, counts, project/program IDs, and sky coordinates."
        ),
        function=agent._alma_tool_fn("match_perseus_protostars_alma_jwst"),
        parameters={
            "type": "object",
            "properties": {
                "radius_arcsec": {"type": "number", "description": "Match radius in arcsec. Default 5."},
                "max_sources": {"type": "integer", "description": "Max built-in Perseus sources to test. Default 12."},
                "max_alma_rows": {"type": "integer", "description": "Max ALMA TAP rows to fetch. Default 5000."},
                "max_mast_results_per_source": {"type": "integer", "description": "Max MAST/JWST rows per source. Default 80."},
            },
            "required": []
        },
        category="archive"
    ))

    # plot_alma_results is registered once below under "Publication Plotting Tools"
    
    agent.tool_registry.register(Tool(
        name="download_alma_data",
        description="Download ALMA data (FITS) for current results",
        function=agent._alma_tool_fn("download_alma_data"),
        parameters={
            "type": "object",
            "properties": {
                "dry_run": {"type": "boolean", "description": "If true, only simulates download. Default False.", "default": False}
            },
            "required": []
        }
    ))


    # NEW: Advanced ALminer Tools
    agent.tool_registry.register(Tool(
        name="find_alma_line_coverage",
        description=(
            "Resolve one named spectral transition with Splatalogue and return only "
            "ALMA projects whose exact spectral windows cover its observed frequency."
        ),
        function=agent._alma_tool_fn("find_alma_line_coverage"),
        parameters={
            "type": "object",
            "properties": {
                "target_name": {
                    "type": "string",
                    "description": "Astronomical target name, e.g. M87",
                },
                "species": {
                    "type": "string",
                    "description": "Molecular species/formula, e.g. CO",
                },
                "transition": {
                    "type": "string",
                    "description": "Exact transition text, e.g. 2-1",
                },
                "redshift": {
                    "type": "number",
                    "description": "Optional explicit redshift; overrides SIMBAD/NED",
                },
                "tolerance_mhz": {
                    "type": "number",
                    "description": "Optional additional frequency tolerance in MHz",
                },
                "velocity_width_kms": {
                    "type": "number",
                    "description": "Optional full velocity width in km/s",
                },
            },
            "required": ["target_name", "species", "transition"],
        },
        category="archive",
    ))

    agent.tool_registry.register(Tool(
        name="check_line_coverage",
        description="Check if specific lines are covered in the LAST search results.",
        function=agent._alma_tool_fn("check_line_coverage"),
        parameters={
            "type": "object",
            "properties": {
                "line_freq_ghz": {"type": "number", "description": "Frequency in GHz"},
                "z": {"type": "number", "description": "Redshift (default 0.0)"},
                "line_name": {"type": "string", "description": "Name of the line"}
            },
            "required": ["line_freq_ghz"]
        }
    ))

    agent.tool_registry.register(Tool(
        name="check_co_lines",
        description="Check for CO, 13CO, and C18O lines in the LAST search results.",
        function=agent._alma_tool_fn("check_co_lines"),
        parameters={
            "type": "object",
            "properties": {
                "z": {"type": "number", "description": "Redshift (default 0.0)"}
            },
            "required": []
        }
    ))

    agent.tool_registry.register(Tool(
        name="search_catalog",
        description="Search for a catalog of objects (Name, RA, Dec)",
        function=agent._alma_tool_fn("search_catalog"),
        parameters={
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Name": {"type": "string"},
                            "RAJ2000": {"type": "number"},
                            "DEJ2000": {"type": "number"}
                        },
                         "required": ["Name"]
                    },
                    "description": "List of objects with Name, RA, Dec"
                }
            },
            "required": ["objects"]
        }
    ))

    # NEW: Fix 3 - Target name resolution using SIMBAD
    agent.tool_registry.register(Tool(
        name="resolve_target",
        description="Resolve a target name to RA/Dec coordinates using SIMBAD. Use this if search_by_target returns empty for a valid target name.",
        function=agent._calc_tool_fn("resolve_target"),
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "The astronomical target name to resolve (e.g., RXJ1347-1145, M31)"}
            },
            "required": ["target_name"]
        }
    ))

    # NEW: Fix 2 - Deterministic filtering tool
    agent.tool_registry.register(Tool(
        name="filter_results",
        description="Apply numeric filters to the LAST ALMA/archive search results table (e.g. 'resolution < 0.05 arcsec', 'sensitivity > 10 mJy'). This does NOT see Data Lab catalog results — for those, put the cut in the query itself (datalab_select_catalog_rows value_cuts, datalab_sql_query WHERE) or use the one-shot diagram tools' point_sources/morphology options.",
        function=agent._alma_tool_fn("filter_results"),
        parameters={
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "Column name to filter (e.g., 'resolution', 'sensitivity', 'Band')"},
                "operator": {"type": "string", "enum": ["<", ">", "<=", ">=", "==", "!="], "description": "Comparison operator"},
                "value": {"type": "number", "description": "Numeric value to compare against"}
            },
            "required": ["column", "operator", "value"]
        }
    ))

    # ── Browser Control Tools ──────────────────────────────────
    agent.tool_registry.register(Tool(
        name="web_search",
        description=(
            "Search the web for real-time information: astronomy news, telescope schedules, "
            "observatory announcements, instrument specs, call-for-proposals, or operational status updates. "
            "NEVER use this for finding papers or publications — use search_papers (NASA ADS) instead. "
            "Uses Brave, Tavily, and Exa through Quasar's web search router for grounded, source-cited results. "
            "Use the user's query as-is — do NOT add years or dates unless the user explicitly mentioned them. "
            "For keyword queries, use this tool directly and answer from its returned sources. "
            "Do NOT call navigate_to_url after web_search unless the user explicitly asks to open a specific result URL "
            "or the search result is insufficient and you need one full page from a known http(s) URL. "
            "Examples: 'ALMA proprietary period policy', 'JWST cycle 4 call for proposals', "
            "'VLA sensitivity at 1.4 GHz'."
        ),
        function=agent._tavily_web_search,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Web search query string"},
                "max_results": {"type": "integer", "description": "Number of results to return (default 10, max 10)"},
                "search_depth": {"type": "string", "enum": ["basic", "advanced"], "description": "'basic' for quick real-time search, 'advanced' for Exa deep search; very hard comparisons/research route to Exa deep-reasoning (default: basic)"},
            },
            "required": ["query"]
        }
    ))

    agent.tool_registry.register(Tool(
        name="web_extract_url",
        description=(
            "Extract clean markdown/text from one or more specific URLs using Tavily Extract. "
            "Use when the user gives URLs and asks to read, summarize, quote, or pull page content. "
            "Only pass full http(s) URLs. Never pass search terms or keyword queries here; use web_search for those. "
            "Do not call this immediately after web_search unless the final answer truly needs full-page text "
            "from a specific result URL. "
            "For JavaScript-heavy pages, set extract_depth='advanced'."
        ),
        function=agent._tavily_extract_url,
        parameters={
            "type": "object",
            "properties": {
                "urls": {
                    "type": "string",
                    "description": "Single full http(s) URL, comma-separated URLs, or list of URLs. Max 20. Not a keyword query.",
                },
                "query": {"type": "string", "description": "Optional focus query to return only relevant chunks."},
                "chunks_per_source": {"type": "integer", "description": "Relevant chunks per URL when query is provided. 1-5, default 3."},
                "extract_depth": {"type": "string", "enum": ["basic", "advanced"], "description": "Use advanced for JS-heavy pages."},
                "include_images": {"type": "boolean", "description": "Whether to include image URLs from the pages."},
                "content_format": {"type": "string", "enum": ["markdown", "text"], "description": "Extracted content format."},
            },
            "required": ["urls"],
        }
    ))

    agent.tool_registry.register(Tool(
        name="web_map_site",
        description=(
            "Discover URLs on a website using Tavily Map without extracting page content. "
            "Use before crawling a large site, or when the user asks for site structure or to find the right page."
        ),
        function=agent._tavily_map_site,
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Root URL to map."},
                "instructions": {"type": "string", "description": "Optional natural-language filter, e.g. 'find API authentication docs'."},
                "max_depth": {"type": "integer", "description": "Crawl depth for URL discovery. 1-5, default 1."},
                "max_breadth": {"type": "integer", "description": "Links explored per page, default 20."},
                "limit": {"type": "integer", "description": "Maximum URLs to return. Default 100, hard-capped at 500."},
                "select_paths": {"type": "array", "items": {"type": "string"}, "description": "Regex path patterns to include."},
                "exclude_paths": {"type": "array", "items": {"type": "string"}, "description": "Regex path patterns to exclude."},
                "select_domains": {"type": "array", "items": {"type": "string"}, "description": "Regex domain patterns to include."},
                "exclude_domains": {"type": "array", "items": {"type": "string"}, "description": "Regex domain patterns to exclude."},
                "allow_external": {"type": "boolean", "description": "Whether to include external links. Default false."},
            },
            "required": ["url"],
        }
    ))

    agent.tool_registry.register(Tool(
        name="web_crawl_site",
        description=(
            "Crawl a bounded website section with Tavily Crawl and extract content from discovered pages. "
            "Use for documentation sections or site areas where multiple pages are needed. Keep limit small unless the user asks for broad coverage."
        ),
        function=agent._tavily_crawl_site,
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Root URL to crawl."},
                "instructions": {"type": "string", "description": "Semantic focus for relevant pages/chunks."},
                "chunks_per_source": {"type": "integer", "description": "Chunks per page when instructions are provided. 1-5, default 3."},
                "max_depth": {"type": "integer", "description": "Crawl depth. 1-5, default 1."},
                "max_breadth": {"type": "integer", "description": "Links explored per page, default 20."},
                "limit": {"type": "integer", "description": "Maximum pages to extract. Default 20, hard-capped at 50."},
                "extract_depth": {"type": "string", "enum": ["basic", "advanced"], "description": "Use advanced for JS-heavy pages."},
                "content_format": {"type": "string", "enum": ["markdown", "text"], "description": "Extracted content format."},
                "include_images": {"type": "boolean", "description": "Whether to include image URLs from crawled pages."},
                "select_paths": {"type": "array", "items": {"type": "string"}, "description": "Regex path patterns to include."},
                "exclude_paths": {"type": "array", "items": {"type": "string"}, "description": "Regex path patterns to exclude."},
                "select_domains": {"type": "array", "items": {"type": "string"}, "description": "Regex domain patterns to include."},
                "exclude_domains": {"type": "array", "items": {"type": "string"}, "description": "Regex domain patterns to exclude."},
                "allow_external": {"type": "boolean", "description": "Whether to include external links. Default false."},
            },
            "required": ["url"],
        }
    ))

    agent.tool_registry.register(Tool(
        name="web_research",
        description=(
            "Run Tavily Research for comprehensive multi-source web research with citations. "
            "Use for deep web reports, market/landscape comparisons, or current-topic investigations. "
            "Do not use for astronomy paper searches; use search_papers for publications."
        ),
        function=agent._tavily_research,
        parameters={
            "type": "object",
            "properties": {
                "research_input": {"type": "string", "description": "Research task or question."},
                "model": {"type": "string", "enum": ["mini", "pro", "auto"], "description": "mini for narrow tasks, pro for broad comparisons, auto by default."},
                "citation_format": {"type": "string", "enum": ["numbered", "mla", "apa", "chicago"], "description": "Citation style."},
                "wait_for_completion": {"type": "boolean", "description": "Poll for completion before returning. Default true."},
                "timeout_seconds": {"type": "integer", "description": "Maximum polling time. Default 120 seconds."},
            },
            "required": ["research_input"],
        }
    ))

    agent.tool_registry.register(Tool(
        name="web_research_status",
        description="Get the status or completed content for a Tavily Research request_id.",
        function=agent._tavily_research_status,
        parameters={
            "type": "object",
            "properties": {
                "request_id": {"type": "string", "description": "Tavily Research request ID."}
            },
            "required": ["request_id"],
        }
    ))

    agent.tool_registry.register(Tool(
        name="navigate_to_url",
        description=(
            "Navigate the browser to a specific full http(s) URL and return its page content. "
            "Use for interactive browser-only tasks on ALMA archive, NASA ADS, ESO portal, VizieR, arXiv pages, etc. "
            "Do NOT use this for keyword searches and do NOT call it after web_search unless the user explicitly "
            "asked to open/navigate to a specific URL."
        ),
        function=lambda **kw: agent.browser_service.navigate_to_url(**kw),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to navigate to (must start with http:// or https://)"}
            },
            "required": ["url"]
        }
    ))

    agent.tool_registry.register(Tool(
        name="read_page",
        description="Read the text content of the currently open browser page. Call after navigate_to_url to get the full page content.",
        function=lambda **kw: agent.browser_service.read_page(**kw),
        parameters={
            "type": "object",
            "properties": {},
            "required": []
        }
    ))

    agent.tool_registry.register(Tool(
        name="click_element",
        description="Click a button, link, or other element on the current browser page by CSS selector.",
        function=lambda **kw: agent.browser_service.click_element(**kw),
        parameters={
            "type": "object",
            "properties": {
                "selector": {"type": "string", "description": "CSS selector for the element to click (e.g., 'button.search', '#submit', 'a.download-link')"},
                "wait_after_ms": {"type": "integer", "description": "Milliseconds to wait after clicking (default: 1500)"}
            },
            "required": ["selector"]
        }
    ))

    # ── Publication Plotting Tools ─────────────────────────────
    agent.tool_registry.register(Tool(
        name="plot_alma_results",
        description=(
            "Generate a plot from the last ALMA search results. Two modes:\n"
            "1. Quick overview: pass plot_type='sky', 'frequency', or 'overview' for pre-built plots.\n"
            "2. Publication-quality scatter: pass x_column and y_column for a custom ApJ/MNRAS-style "
            "scatter plot (300 DPI, colorblind-safe).\n"
            "If plot_type is given, x_column/y_column are ignored. Use after any search."
        ),
        function=agent._alma_tool_fn("plot_alma_results"),
        parameters={
            "type": "object",
            "properties": {
                "plot_type": {"type": "string", "enum": ["sky", "frequency", "overview"], "description": "Quick overview plot type. If provided, x_column/y_column are ignored."},
                "x_column": {"type": "string", "description": "Column for x-axis (e.g. 'frequency', 'spatial_resolution', 'band')"},
                "y_column": {"type": "string", "description": "Column for y-axis"},
                "color_by": {"type": "string", "description": "Column to color-code points by (e.g. 'band', 'facility')"},
                "title": {"type": "string", "description": "Plot title"},
                "dark_mode": {"type": "boolean", "description": "Use dark background for presentations/posters"},
            },
            "required": []
        }
    ))

    agent.tool_registry.register(Tool(
        name="plot_sky_map",
        description=(
            "Generate a publication-quality RA/Dec sky distribution map from the last "
            "ALMA search results, or from a Data Lab result_id (scatter of its rows)."
        ),
        function=agent._plot_sky_map_tool,
        parameters={
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional Data Lab dlr_/dlt_ result id to plot instead of the last archive search"},
                "ra_col": {"type": "string", "description": "Column name for Right Ascension"},
                "dec_col": {"type": "string", "description": "Column name for Declination"},
                "color_by": {"type": "string", "description": "Column for color coding"},
                "title": {"type": "string", "description": "Plot title"},
            },
            "required": []
        }
    ))

    agent.tool_registry.register(Tool(
        name="plot_spectrum",
        description="Generate a publication-quality 1D spectral line profile plot with optional error bars and molecular line ID labels.",
        function=lambda **kw: agent.plotting_service.plot_spectrum(**kw),
        parameters={
            "type": "object",
            "properties": {
                "frequencies": {"type": "array", "items": {"type": "number"}, "description": "Frequency values (GHz)"},
                "fluxes": {"type": "array", "items": {"type": "number"}, "description": "Flux density values (Jy)"},
                "title": {"type": "string", "description": "Plot title"},
                "errors": {"type": "array", "items": {"type": "number"}, "description": "Optional error bars (same length as fluxes)"},
            },
            "required": ["frequencies", "fluxes"]
        }
    ))

    # ── Splatalogue Line ID Tools ───────────────────────────────
    agent.tool_registry.register(Tool(
        name="identify_spectral_line",
        description="Identify molecular spectral lines near a given rest frequency using the Splatalogue database. Essential for ALMA/VLA spectral line identification.",
        function=lambda **kw: agent.splatalogue_tool.identify_spectral_line(**kw),
        parameters={
            "type": "object",
            "properties": {
                "frequency_ghz": {"type": "number", "description": "Rest frequency to search around (GHz), e.g. 230.538"},
                "tolerance_ghz": {"type": "number", "description": "Search window ± around the frequency in GHz (default: 0.01 = 10 MHz)"},
                "top_n": {"type": "integer", "description": "Max candidate lines to return (default: 5)"},
            },
            "required": ["frequency_ghz"]
        }
    ))

    agent.tool_registry.register(Tool(
        name="search_lines_by_molecule",
        description="Search Splatalogue for spectral-line transitions of a molecule, with optional frequency, energy, intensity, transition, and catalog filters.",
        function=lambda **kw: agent.splatalogue_tool.search_lines_by_molecule(**kw),
        parameters={
            "type": "object",
            "properties": {
                "molecule_name": {"type": "string", "description": "Molecule name (e.g., 'CO', 'HCN', 'CH3OH', 'SiO')"},
                "freq_min_ghz": {"type": "number", "description": "Minimum frequency filter (GHz)"},
                "freq_max_ghz": {"type": "number", "description": "Maximum frequency filter (GHz)"},
                "top_n": {"type": "integer", "minimum": 1, "description": "Maximum number of normalized transitions to return"},
                "transition": {"type": "string", "description": "Optional quantum-number or transition filter, e.g. '2-1'"},
                "energy_min": {"type": "number", "description": "Optional lower energy bound"},
                "energy_max": {"type": "number", "description": "Optional upper energy bound"},
                "energy_type": {
                    "type": "string",
                    "enum": ["el_cm1", "eu_cm1", "el_k", "eu_k"],
                    "description": "Energy field used for energy_min/energy_max",
                },
                "intensity_lower_limit": {"type": "number", "description": "Optional lower line-intensity threshold"},
                "intensity_type": {
                    "type": "string",
                    "enum": ["CDMS/JPL (log)", "Sij-mu2", "Aij (log)"],
                    "description": "Intensity field used by intensity_lower_limit",
                },
                "line_lists": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["LovasNIST", "SLAIM", "JPL", "CDMS", "ToyaMA", "OSU", "TopModel", "Recombination", "RFI"],
                    },
                    "description": "Splatalogue source catalogs to include",
                },
                "only_astronomically_observed": {"type": "boolean", "description": "Only return transitions observed in space"},
                "only_nrao_recommended": {"type": "boolean", "description": "Only return NRAO-recommended frequencies"},
            },
            "required": ["molecule_name"]
        }
    ))

    agent.tool_registry.register(Tool(
        name="search_spectral_lines",
        description="Run an advanced Splatalogue frequency-range query. Use this for filtered line surveys, line-confusion checks, and catalog comparisons.",
        function=lambda **kw: agent.splatalogue_tool.search_spectral_lines(**kw),
        parameters={
            "type": "object",
            "properties": {
                "freq_min_ghz": {"type": "number", "description": "Minimum rest frequency in GHz"},
                "freq_max_ghz": {"type": "number", "description": "Maximum rest frequency in GHz"},
                "molecule_name": {"type": "string", "description": "Optional molecule/species name or formula"},
                "transition": {"type": "string", "description": "Optional quantum-number or transition filter"},
                "energy_min": {"type": "number", "description": "Optional lower energy bound"},
                "energy_max": {"type": "number", "description": "Optional upper energy bound"},
                "energy_type": {
                    "type": "string",
                    "enum": ["el_cm1", "eu_cm1", "el_k", "eu_k"],
                    "description": "Energy field used for energy_min/energy_max",
                },
                "intensity_lower_limit": {"type": "number", "description": "Optional lower line-intensity threshold"},
                "intensity_type": {
                    "type": "string",
                    "enum": ["CDMS/JPL (log)", "Sij-mu2", "Aij (log)"],
                    "description": "Intensity field used by intensity_lower_limit",
                },
                "version": {
                    "type": "string",
                    "enum": ["v1.0", "v2.0", "v3.0", "vall"],
                    "description": "Splatalogue data version",
                },
                "exclude": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["atmospheric", "potential", "probable", "known", "none"]},
                    "description": "Species classifications to exclude",
                },
                "line_lists": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["LovasNIST", "SLAIM", "JPL", "CDMS", "ToyaMA", "OSU", "TopModel", "Recombination", "RFI"],
                    },
                    "description": "Splatalogue source catalogs to include",
                },
                "only_astronomically_observed": {"type": "boolean", "description": "Only return transitions observed in space"},
                "only_nrao_recommended": {"type": "boolean", "description": "Only return NRAO-recommended frequencies"},
                "top_n": {"type": "integer", "minimum": 1, "description": "Maximum number of normalized transitions to return"},
            },
            "required": ["freq_min_ghz", "freq_max_ghz"]
        }
    ))

    # ── Multi-archive Cross-matcher Tools ─────────────────────
    agent.tool_registry.register(Tool(
        name="cross_match_source",
        description="Query multiple astronomical archives (Simbad, NED, MAST, VizieR, Fermi 4FGL) in parallel for a source. Returns a unified multi-wavelength summary.",
        function=lambda **kw: agent.multi_archive.cross_match_source(**kw),
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "Source name recognized by Simbad (e.g., 'M87', 'HL Tau', 'NGC 1275')"},
                "archives": {"type": "array", "items": {"type": "string"}, "description": "Specific archives to query, e.g. ['simbad', 'ned', 'mast']. Defaults to all."},
            },
            "required": ["target_name"]
        }
    ))
    # ── MAST Archive Tools (JWST, HST, TESS, Kepler) ──────────
    agent.tool_registry.register(Tool(
        name="search_mast",
        description=(
            "Search the MAST archive for observations from JWST, HST, TESS, Kepler, "
            "and other space telescopes. Use this for any JWST or HST data queries. "
            "Returns observation metadata including target, instrument, filters, "
            "exposure time, and data product type."
        ),
        function=agent._archives_tool_fn("search_mast"),
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "Astronomical target name (e.g., 'M87', 'Carina Nebula', 'TRAPPIST-1')"},
                "mission": {"type": "string", "description": "Filter by mission: 'JWST', 'HST', 'TESS', 'Kepler'. Leave empty for all missions."},
                "instrument": {"type": "string", "description": "Filter by instrument (e.g., 'NIRCAM', 'NIRSPEC', 'MIRI', 'ACS', 'WFC3'). Leave empty for all."},
                "radius": {"type": "string", "description": "Search radius (e.g., '30s' for 30 arcsec, '1m' for 1 arcmin). Default '30s'."},
                "ra": {"type": "number", "description": "RA in degrees (use instead of target_name for positional search)"},
                "dec": {"type": "number", "description": "Dec in degrees (use instead of target_name for positional search)"},
            },
            "required": []
        }
    ))

    agent.tool_registry.register(Tool(
        name="search_mast_by_criteria",
        description=(
            "Advanced MAST search with rich filtering: program ID, date range, "
            "filter name, data product type, etc. Use this when users ask for "
            "specific JWST/HST programs, particular filters (F200W, F444W), "
            "or time-constrained searches."
        ),
        function=agent._archives_tool_fn("search_mast_by_criteria"),
        parameters={
            "type": "object",
            "properties": {
                "mission": {"type": "string", "description": "Mission name (JWST, HST, TESS, Kepler)"},
                "instrument": {"type": "string", "description": "Instrument name (NIRCAM, NIRSPEC, MIRI, ACS, WFC3)"},
                "proposal_id": {"type": "string", "description": "Specific proposal/program ID (e.g., '1345' for JADES)"},
                "filters": {"type": "string", "description": "Filter name (e.g., 'F200W', 'F444W', 'F115W')"},
                "target_name": {"type": "string", "description": "Target name for the search"},
                "dataproduct_type": {"type": "string", "description": "'image', 'spectrum', 'cube', 'timeseries'"},
                "start_date": {"type": "string", "description": "Start date for time filter (ISO format, e.g., '2022-07-01')"},
                "end_date": {"type": "string", "description": "End date for time filter (ISO format, e.g., '2023-07-01')"},
            },
            "required": []
        }
    ))

    agent.tool_registry.register(Tool(
        name="get_mast_products",
        description=(
            "Get file-level product list for the LAST MAST search results. "
            "Shows individual data files available for download (filenames, sizes, URLs). "
            "Call this AFTER a search_mast or search_mast_by_criteria call."
        ),
        function=agent._archives_tool_fn("get_mast_products"),
        parameters={
            "type": "object",
            "properties": {
                "product_type": {"type": "string", "description": "Filter by type: 'SCIENCE', 'CALIBRATION', 'PREVIEW'. Default all."},
                "extension": {"type": "string", "description": "Filter by file extension: 'fits', 'jpg', etc."},
            },
            "required": []
        }
    ))

    # ── ESO Science Archive Tools (VLT instruments) ───────────
    agent.tool_registry.register(Tool(
        name="search_eso_archive",
        description=(
            "Search the ESO Science Archive for VLT instrument observations. "
            "Supports instruments: MUSE, KMOS, X-Shooter, FORS2, HAWK-I, UVES, "
            "SPHERE, GRAVITY, ESPRESSO, and more. Uses TAP/ADQL queries against "
            "the ESO ObsCore table."
        ),
        function=agent._archives_tool_fn("search_eso_archive"),
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "Astronomical target name (e.g., 'NGC 1068', 'Eta Carinae')"},
                "instrument": {"type": "string", "description": "ESO instrument (e.g., 'MUSE', 'KMOS', 'XSHOOTER', 'FORS2', 'HAWK-I', 'UVES', 'SPHERE')"},
                "ra": {"type": "number", "description": "RA in degrees (use instead of target_name)"},
                "dec": {"type": "number", "description": "Dec in degrees (use instead of target_name)"},
                "radius_arcmin": {"type": "number", "description": "Search radius in arcminutes. Default 1.0."},
            },
            "required": []
        }
    ))

    # ── IRSA Infrared Archive Tools (WISE, 2MASS, Spitzer) ────
    agent.tool_registry.register(Tool(
        name="search_irsa",
        description=(
            "Search the IRSA (Infrared Science Archive) catalog services for "
            "infrared source catalogs. Catalogs include AllWISE, 2MASS Point "
            "Source, 2MASS Extended Source, and Spitzer SEIP."
        ),
        function=agent._archives_tool_fn("search_irsa"),
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "Astronomical target name (e.g., 'M31', 'NGC 253')"},
                "catalog": {"type": "string", "description": "IRSA catalog: 'allwise' (default), '2mass', '2mass_xsc', 'seip'. Or a specific IRSA catalog ID."},
                "radius_arcsec": {"type": "number", "description": "Search radius in arcseconds. Default 30."},
                "ra": {"type": "number", "description": "RA in degrees (use instead of target_name)"},
                "dec": {"type": "number", "description": "Dec in degrees (use instead of target_name)"},
            },
            "required": []
        }
    ))

    # ── Sky Survey Image Tools ────────────────────────────────

    # Data Lab P0 catalog-TAP tools
    agent.tool_registry.register(Tool(
        name="datalab_list_catalogs",
        description="List supported NOIRLab Astro Data Lab P0 catalogs and registered tables.",
        function=agent._datalab_tool_fn("datalab_list_catalogs"),
        parameters={"type": "object", "properties": {}, "required": []},
        category="datalab",
    ))

    # ── Cross-archive schema grounding (Feature 3) ────────────
    # One uniform profile tool for every archive; for Data Lab it wraps the
    # registry below, for ADS/SIA/VO/ALMA/Splatalogue it serves the curated
    # ArchiveProfile. Stateless read — the CallContext needs no services.
    from adapters.native import build_tool as _build_schema_tool
    from capabilities.base import CallContext as _SchemaCallContext
    from capabilities.schema import BrowseSchema as _BrowseSchema
    agent.tool_registry.register(
        _build_schema_tool(_BrowseSchema(), _SchemaCallContext)
    )

    agent.tool_registry.register(Tool(
        name="datalab_describe_table",
        description="Describe a registered Data Lab catalog table, columns, region strategy, morphology hints, and citation.",
        function=agent._datalab_tool_fn("datalab_describe_table"),
        parameters={
            "type": "object",
            "properties": {
                "catalog": {"type": "string", "description": "Registered catalog, e.g. gaia_dr3 or nsc_dr2."},
                "table": {"type": "string", "description": "Registered table within the catalog, e.g. gaia_source or object."},
            },
            "required": ["catalog", "table"],
        },
        category="datalab",
    ))

    agent.tool_registry.register(Tool(
        name="datalab_cone_count",
        description="Count rows in a Data Lab catalog cone using a governed q3c_radial_query builder.",
        function=agent._datalab_tool_fn("datalab_cone_count"),
        parameters={
            "type": "object",
            "properties": {
                "catalog": {"type": "string"},
                "table": {"type": "string"},
                "ra": {"type": "number", "description": "ICRS right ascension in degrees."},
                "dec": {"type": "number", "description": "ICRS declination in degrees."},
                "radius_deg": {"type": "number", "description": "Cone radius in degrees."},
            },
            "required": ["catalog", "table", "ra", "dec", "radius_deg"],
        },
        category="datalab",
    ))

    agent.tool_registry.register(Tool(
        name="datalab_select_catalog_rows",
        description="Select capped rows from a Data Lab catalog cone using governed structured SQL; returns result_id, not the full table. Apply selection cuts server-side via value_cuts/color_cut/morphology so the row budget is spent on rows you want.",
        function=agent._datalab_tool_fn("datalab_select_catalog_rows"),
        parameters={
            "type": "object",
            "properties": {
                "catalog": {"type": "string"},
                "table": {"type": "string"},
                "ra": {"type": "number"},
                "dec": {"type": "number"},
                "radius_deg": {"type": "number"},
                "columns": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer", "default": 500},
                "value_cuts": {"type": "array", "items": {"type": "object"}, "description": "e.g. [{'column':'parallax_over_error','op':'>','value':5}]"},
                "color_cut": {"type": "object", "description": "{'bands':['gmag','rmag'],'min':-0.5,'max':0.5}"},
                "morphology": {"type": "object", "description": "{'column':'class_star','op':'>','value':0.5}"},
                "async_submit": {"type": "boolean", "default": False, "description": "Submit as a background job and return job_id immediately (server-side with DATALAB_TOKEN, else local); poll datalab_job_status."},
            },
            "required": ["catalog", "table", "ra", "dec", "radius_deg"],
        },
        category="datalab",
    ))

    agent.tool_registry.register(Tool(
        name="datalab_density_aggregate",
        description="Aggregate Data Lab source density by RA/Dec grid or registered HEALPix column over a cone region; returns a stable result_id. Requires a cone (ra/dec/radius_deg) unless all_sky=true is set explicitly. Wide cones that exceed the 60s sync window are automatically tiled into sub-cones and merged — do NOT hand-tile the region yourself; call once with the full cone.",
        function=agent._datalab_tool_fn("datalab_density_aggregate"),
        parameters={
            "type": "object",
            "properties": {
                "catalog": {"type": "string"},
                "table": {"type": "string"},
                "mode": {"type": "string", "enum": ["grid", "healpix"], "default": "grid"},
                "step_deg": {"type": "number", "default": 0.1},
                "healpix_column": {"type": "string"},
                "ra": {"type": "number", "description": "Cone center RA (deg); with dec+radius_deg bounds the aggregate."},
                "dec": {"type": "number", "description": "Cone center Dec (deg)."},
                "radius_deg": {"type": "number", "description": "Cone radius (deg) bounding the aggregate."},
                "all_sky": {"type": "boolean", "default": False, "description": "Explicitly run an unbounded whole-catalog aggregate (slow/expensive)."},
                "color_cut": {"type": "object", "description": "e.g. {'bands':['gmag','rmag'],'min':-0.5,'max':0.5}"},
                "value_cuts": {"type": "array", "items": {"type": "object"}, "description": "e.g. [{'column':'gmag','op':'>','value':19.5}]"},
                "morphology": {"type": "object", "description": "e.g. {'column':'class_star','op':'>','value':0.5}"},
                "limit": {"type": "integer", "default": 5000},
                "async_submit": {"type": "boolean", "default": False, "description": "Submit as a background job and return job_id immediately (skips the sync attempt and auto-tiling); poll datalab_job_status."},
            },
            "required": ["catalog", "table"],
        },
        category="datalab",
    ))

    agent.tool_registry.register(Tool(
        name="datalab_variable_candidates",
        description=(
            "Rank variable-star candidates in a multi-epoch catalog cone (SMASH source "
            "tables): per-object epoch count, mean magnitude, magnitude scatter, amplitude, "
            "and scatter/error significance, most-variable first. Feed a candidate's id into "
            "datalab_star_lightcurve, then datalab_period_fold."
        ),
        function=agent._datalab_tool_fn("datalab_variable_candidates"),
        parameters={
            "type": "object",
            "properties": {
                "catalog": {"type": "string", "default": "smash_dr1", "description": "Multi-epoch catalog: smash_dr1 or smash_dr2."},
                "ra": {"type": "number"},
                "dec": {"type": "number"},
                "target_name": {"type": "string", "description": "Alternative to ra/dec."},
                "radius_deg": {"type": "number", "default": 0.2},
                "band": {"type": "string", "description": "Optional single filter (e.g. 'g') to restrict epochs to one band."},
                "min_epochs": {"type": "integer", "default": 10, "description": "Minimum epochs per object (>=2)."},
                "limit": {"type": "integer", "default": 100},
            },
            "required": [],
        },
        category="datalab",
    ))

    agent.tool_registry.register(Tool(
        name="datalab_star_lightcurve",
        description=(
            "Fetch the multi-epoch light curve (mjd, filter, cmag, cerr) of one star from a "
            "SMASH source table by source id or exact position; returns a result_id ready for "
            "datalab_period_fold. Use datalab_variable_candidates first to find good targets."
        ),
        function=agent._datalab_tool_fn("datalab_star_lightcurve"),
        parameters={
            "type": "object",
            "properties": {
                "catalog": {"type": "string", "default": "smash_dr1", "description": "Multi-epoch catalog: smash_dr1 or smash_dr2."},
                "source_id": {"type": "string", "description": "SMASH source id (from datalab_variable_candidates)."},
                "ra": {"type": "number", "description": "Exact position alternative to source_id."},
                "dec": {"type": "number"},
                "limit": {"type": "integer", "default": 500},
            },
            "required": [],
        },
        category="datalab",
    ))

    agent.tool_registry.register(Tool(
        name="datalab_q3c_crossmatch",
        description="Planner-safe Data Lab q3c crossmatch: materializes the small Gaia-like side first and joins the large indexed catalog second.",
        function=agent._datalab_tool_fn("datalab_q3c_crossmatch"),
        parameters={
            "type": "object",
            "properties": {
                "small_catalog": {"type": "string", "default": "gaia_dr3"},
                "small_table": {"type": "string", "default": "gaia_source"},
                "big_catalog": {"type": "string", "default": "nsc_dr2"},
                "big_table": {"type": "string", "default": "object"},
                "ra": {"type": "number"},
                "dec": {"type": "number"},
                "radius_deg": {"type": "number"},
                "match_radius_arcsec": {"type": "number", "default": 1.0},
                "small_columns": {"type": "array", "items": {"type": "string"}},
                "big_columns": {"type": "array", "items": {"type": "string"}},
                "small_limit": {"type": "integer", "default": 10000},
                "limit": {"type": "integer", "default": 500},
            },
            "required": ["ra", "dec", "radius_deg"],
        },
        category="datalab",
    ))

    agent.tool_registry.register(Tool(
        name="datalab_sql_query",
        description=(
            "EXPERT/DEBUG ONLY: run governed raw native SQL against Data Lab. "
            "Requires expert_ack=true and a reason; q3c_join remains blocked outside the structured crossmatch builder. "
            "Returns result_id only, not the full table."
        ),
        function=agent._datalab_tool_fn("datalab_sql_query"),
        parameters={
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "Read-only native SQL SELECT/WITH query."},
                "expert_ack": {"type": "boolean", "description": "Must be true to acknowledge expert/debug raw SQL mode."},
                "reason": {"type": "string", "description": "Brief justification for not using structured builders."},
                "async_submit": {"type": "boolean", "default": False, "description": "Submit as a background job and return job_id immediately (server-side with DATALAB_TOKEN, else local); poll datalab_job_status."},
            },
            "required": ["sql", "expert_ack", "reason"],
        },
        category="datalab",
    ))

    agent.tool_registry.register(Tool(
        name="datalab_get_result",
        description="Fetch the rows of a stored Data Lab result_id (from a prior datalab_* tool), capped at max_rows (<=5000).",
        function=agent._datalab_tool_fn("datalab_get_result"),
        parameters={
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "result_id returned by a datalab_* tool."},
                "max_rows": {"type": "integer", "default": 200, "description": "Maximum rows to return (capped at 5000)."},
            },
            "required": ["result_id"],
        },
        category="datalab",
    ))

    # Data Lab P1 SIA image, SVO, and catalog-analysis tools
    agent.tool_registry.register(Tool(
        name="datalab_sia_search",
        description=(
            "List the Data Lab SIA image inventory covering a position: bands, exposure "
            "times, proc/prod types, and access URLs, stored under a result_id. Use this to "
            "see what imaging exists (and how deep) BEFORE datalab_image_cutout / "
            "datalab_color_image; fetch rows with datalab_get_result."
        ),
        function=agent._datalab_tool_fn("datalab_sia_search"),
        parameters={
            "type": "object",
            "properties": {
                "ra": {"type": "number", "description": "ICRS right ascension in degrees."},
                "dec": {"type": "number", "description": "ICRS declination in degrees."},
                "target_name": {"type": "string", "description": "Optional target name to resolve if ra/dec are not supplied."},
                "fov_deg": {"type": "number", "default": 0.1, "description": "Search field of view in degrees."},
                "band": {"type": "string", "description": "Optional band prefix filter (e.g. g, r, i)."},
                "catalog": {"type": "string", "description": "Registered Data Lab catalog used to choose the SIA endpoint."},
                "endpoint": {"type": "string", "description": "Optional explicit SIA endpoint override."},
                "limit": {"type": "integer", "default": 100, "description": "Max inventory rows kept, capped at 1000."},
            },
            "required": [],
        },
        category="datalab",
    ))

    agent.tool_registry.register(Tool(
        name="datalab_image_cutout",
        description="Render a single-band NOIRLab Astro Data Lab SIA cutout at RA/Dec or a resolvable target name.",
        function=agent._datalab_image_tool_fn("datalab_image_cutout"),
        parameters={
            "type": "object",
            "properties": {
                "ra": {"type": "number", "description": "ICRS right ascension in degrees."},
                "dec": {"type": "number", "description": "ICRS declination in degrees."},
                "target_name": {"type": "string", "description": "Optional target name to resolve if ra/dec are not supplied."},
                "fov_deg": {"type": "number", "default": 0.05, "description": "Cutout field of view in degrees."},
                "band": {"type": "string", "default": "g", "description": "Band prefix, e.g. g, r, or i."},
                "catalog": {"type": "string", "default": "ls_dr9", "description": "Registered Data Lab catalog used to choose SIA endpoint."},
                "endpoint": {"type": "string", "description": "Optional explicit SIA endpoint override."},
                "title": {"type": "string"},
            },
            "required": ["fov_deg"],
        },
        category="datalab",
    ))

    agent.tool_registry.register(Tool(
        name="datalab_color_image",
        description="Render a Data Lab color image from deepest SIA stack images, reprojected to a common WCS before Lupton RGB composition. Auto-selects RGB bands (red=i or z, green=r, blue=g) unless 'bands' is given.",
        function=agent._datalab_image_tool_fn("datalab_color_image"),
        parameters={
            "type": "object",
            "properties": {
                "ra": {"type": "number"},
                "dec": {"type": "number"},
                "target_name": {"type": "string"},
                "fov_deg": {"type": "number", "default": 0.05},
                "catalog": {"type": "string", "default": "ls_dr9"},
                "endpoint": {"type": "string"},
                "bands": {"type": "array", "items": {"type": "string"}, "description": "Optional (red, green, blue) band override, e.g. ['z','r','g']. Defaults to auto-selection."},
                "q": {"type": "number", "default": 8.0},
                "stretch": {"type": "number", "default": 0.5},
                "title": {"type": "string"},
            },
            "required": ["fov_deg"],
        },
        category="datalab",
    ))

    agent.tool_registry.register(Tool(
        name="datalab_cutout_grid",
        description="Render a multi-panel Data Lab SIA cutout grid for peak coordinates; panels without coverage are labeled instead of failing the grid.",
        function=agent._datalab_image_tool_fn("datalab_cutout_grid"),
        parameters={
            "type": "object",
            "properties": {
                "peaks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ra": {"type": "number"},
                            "dec": {"type": "number"},
                            "label": {"type": "string"},
                        },
                        "required": ["ra", "dec"],
                    },
                },
                "fov_deg": {"type": "number", "default": 0.05},
                "band": {"type": "string", "default": "g"},
                "catalog": {"type": "string", "default": "ls_dr9"},
                "endpoint": {"type": "string"},
                "title": {"type": "string"},
            },
            "required": ["peaks", "fov_deg"],
        },
        category="datalab",
    ))

    agent.tool_registry.register(Tool(
        name="svo_filter_wavelength",
        description="Look up effective and pivot wavelengths for SVO FPS filter IDs, including LS DR9 shorthand g/r/z/w1/w2.",
        function=agent._svo_filter_wavelength,
        parameters={
            "type": "object",
            "properties": {
                "filter_id": {"type": "string", "description": "Single filter ID or shorthand."},
                "filter_ids": {"type": "array", "items": {"type": "string"}, "description": "Multiple filter IDs or shorthands."},
            },
            "required": [],
        },
        category="datalab",
    ))

    agent.tool_registry.register(Tool(
        name="datalab_catalog_scatter",
        description=(
            "Render a Data Lab result_id as a CMD/CCD/HR-style scatter plot using safe column expressions. "
            "x_expr/y_expr combine columns of the stored result with + - * / ** %, parentheses, numeric literals, "
            "and functions log10/log/sqrt/abs/exp/power (e.g. absolute magnitude: "
            "'phot_g_mean_mag + 5*log10(parallax/100)' with parallax in mas). Only columns present in the stored "
            "result can be referenced — derived columns like M_G do NOT pre-exist; compute them inline here or "
            "alias them in the SQL SELECT first."
        ),
        function=agent._datalab_image_tool_fn("datalab_catalog_scatter"),
        parameters={
            "type": "object",
            "properties": {
                "result_id": {"type": "string"},
                "x_expr": {
                    "type": "string",
                    "description": "Expression over result columns for the x axis, e.g. 'bp_rp' or 'g - r'.",
                },
                "y_expr": {
                    "type": "string",
                    "description": "Expression over result columns for the y axis, e.g. 'phot_g_mean_mag + 5*log10(parallax/100)'.",
                },
                "color_by": {"type": "string"},
                "invert_y": {"type": "boolean", "default": False},
                "invert_x": {"type": "boolean", "default": False},
                "title": {"type": "string"},
                "x_label": {"type": "string"},
                "y_label": {"type": "string"},
                "overlay_locus": {"type": "string"},
            },
            "required": ["result_id", "x_expr", "y_expr"],
        },
        category="datalab",
    ))

    agent.tool_registry.register(Tool(
        name="datalab_sky_density_map",
        description=(
            "Render a Data Lab result_id as a RA/Dec density map or sparse HEALPix density map, "
            "with optional matched-filter peak detection. The colorbar is LOG-scaled by default "
            "(log_scale=true) — leave it on for 'log counts'/'log source count' requests; set "
            "log_scale=false only when the user explicitly wants a linear count scale."
        ),
        function=agent._datalab_image_tool_fn("datalab_sky_density_map"),
        parameters={
            "type": "object",
            "properties": {
                "result_id": {"type": "string"},
                "mode": {"type": "string", "enum": ["hist2d", "healpix"], "default": "hist2d"},
                "ra_col": {"type": "string"},
                "dec_col": {"type": "string"},
                "count_col": {"type": "string", "default": "source_count"},
                "bins": {"type": "integer", "default": 80},
                "healpix_col": {"type": "string", "default": "healpix"},
                "nside": {"type": "integer"},
                "order": {"type": "string", "default": "nested"},
                "matched_filter": {"type": "boolean", "default": False},
                "sigma_small": {"type": "number", "default": 1.0},
                "sigma_large": {"type": "number", "default": 3.0},
                "peak_threshold": {"type": "number", "default": 3.0},
                "max_peaks": {"type": "integer", "default": 10},
                "log_scale": {"type": "boolean", "default": True, "description": "Log-scale the count colorbar (matplotlib LogNorm). True honors 'log counts' requests; set false for a linear scale."},
                "title": {"type": "string"},
            },
            "required": ["result_id"],
        },
        category="datalab",
    ))

    agent.tool_registry.register(Tool(
        name="datalab_period_fold",
        description=(
            "Run Lomb-Scargle period search on a stored Data Lab light curve and render the folded light curve. "
            "Multi-band light curves (NSC/DES/SMASH interleave g/r/i/z epochs in one table) should be folded ONE band "
            "at a time — pass band (e.g. 'g') to restrict to a single filter; mixing bands smears the phased curve."
        ),
        function=agent._datalab_image_tool_fn("datalab_period_fold"),
        parameters={
            "type": "object",
            "properties": {
                "result_id": {"type": "string"},
                "time_col": {"type": "string", "default": "mjd"},
                "mag_col": {"type": "string", "default": "cmag"},
                "error_col": {"type": "string", "default": "cerr"},
                "band": {"type": "string", "description": "Single filter to fold (e.g. 'g', 'r', 'i', 'z'). Restricts to rows where band_col equals this value. Leave unset to fold all rows."},
                "band_col": {"type": "string", "default": "filter", "description": "Column holding the filter/band label (NSC/SMASH use 'filter')."},
                "min_frequency": {"type": "number", "default": 1.0},
                "max_frequency": {"type": "number", "default": 10.0},
                "title": {"type": "string"},
            },
            "required": ["result_id"],
        },
        category="datalab",
    ))

    agent.tool_registry.register(Tool(
        name="datalab_sed_plot",
        description="Render an LS DR9-style SED from a stored Data Lab result_id using SVO FPS wavelengths.",
        function=agent._datalab_image_tool_fn("datalab_sed_plot"),
        parameters={
            "type": "object",
            "properties": {
                "result_id": {"type": "string"},
                "row_index": {"type": "integer", "default": 0},
                "filter_columns": {"type": "object", "additionalProperties": {"type": "string"}},
                "title": {"type": "string"},
            },
            "required": ["result_id"],
        },
        category="datalab",
    ))

    agent.tool_registry.register(Tool(
        name="datalab_lss_wedge",
        description="Render a stored spectroscopic Data Lab result_id as a comoving large-scale-structure wedge or 3D scatter plot.",
        function=agent._datalab_image_tool_fn("datalab_lss_wedge"),
        parameters={
            "type": "object",
            "properties": {
                "result_id": {"type": "string"},
                "ra_col": {"type": "string"},
                "dec_col": {"type": "string"},
                "z_col": {"type": "string", "default": "z"},
                "class_col": {"type": "string"},
                "pie_slice": {"type": "boolean", "default": False},
                "title": {"type": "string"},
            },
            "required": ["result_id"],
        },
        category="datalab",
    ))
    agent.tool_registry.register(Tool(
        name="datalab_density_vetting",
        description="P12: find the densest catalog cells within a cone (with optional color/magnitude/morphology cuts) and pull a SIA cutout grid of the top-N densest locations to eyeball.",
        function=agent._datalab_image_tool_fn("datalab_density_vetting", nested_key="cutout_grid"),
        parameters={
            "type": "object",
            "properties": {
                "catalog": {"type": "string"},
                "table": {"type": "string"},
                "radius_deg": {"type": "number"},
                "ra": {"type": "number"},
                "dec": {"type": "number"},
                "target_name": {"type": "string"},
                "step_deg": {"type": "number", "default": 0.05},
                "color_cut": {"type": "object", "description": "e.g. {'bands':['gmag','rmag'],'min':-0.5,'max':0.5}"},
                "value_cuts": {"type": "array", "items": {"type": "object"}, "description": "e.g. [{'column':'gmag','op':'<','value':25}]"},
                "morphology": {"type": "object", "description": "e.g. {'column':'class_star','op':'>','value':0.5} or {'column':'ext_coadd','between':[0,1]}"},
                "top_n": {"type": "integer", "default": 5},
                "fov_deg": {"type": "number", "default": 0.05},
                "band": {"type": "string", "default": "g"},
            },
            "required": ["catalog", "table", "radius_deg"],
        },
        category="datalab",
    ))
    agent.tool_registry.register(Tool(
        name="datalab_color_color_diagram",
        description="ONE-SHOT color-color diagram (e.g. g-r vs r-i) for a catalog cone. Queries + plots in a single call; auto-splits into stars vs galaxies (2 panels) using the catalog's morphology column (e.g. DES spread_model_r) unless split_col is given. Sentinel magnitudes (99.99) are excluded automatically. Use this for 'show me a color-color diagram'/'separate stars from galaxies' requests — do NOT chain separate query+plot tools.",
        function=agent._datalab_image_tool_fn("datalab_color_color_diagram"),
        parameters={
            "type": "object",
            "properties": {
                "catalog": {"type": "string"},
                "table": {"type": "string", "description": "Optional; defaults to the catalog's primary table."},
                "radius_deg": {"type": "number", "default": 0.5},
                "ra": {"type": "number"},
                "dec": {"type": "number"},
                "target_name": {"type": "string"},
                "x_bands": {"type": "array", "items": {"type": "string"}, "description": "Two bands for the x color, e.g. ['g','r']."},
                "y_bands": {"type": "array", "items": {"type": "string"}, "description": "Two bands for the y color, e.g. ['r','i']."},
                "split_col": {"type": "string", "description": "Morphology column to split stars/galaxies (auto from registry if omitted, e.g. spread_model_r)."},
                "split_threshold": {"type": "number", "default": 0.005},
                "limit": {"type": "integer", "default": 3000},
                "point_sources": {"type": "boolean", "default": False, "description": "True = keep only point sources via the catalog's registered star cut (single panel, no star/galaxy split)."},
                "morphology": {"type": "object", "description": "Explicit morphology cut applied in the SQL WHERE, e.g. {'column':'class_star','op':'>','value':0.5} or {'column':'ext_coadd','between':[0,1]}; overrides point_sources (single panel, no star/galaxy split)."},
                "value_cuts": {"type": "array", "items": {"type": "object"}, "description": "Extra server-side cuts, e.g. [{'column':'flags_g','op':'=','value':0}]."},
            },
            "required": ["catalog"],
        },
        category="datalab",
    ))
    agent.tool_registry.register(Tool(
        name="datalab_color_magnitude_diagram",
        description="ONE-SHOT color-magnitude diagram (CMD): mag_band vs (blue-red) color for a catalog cone. Queries + plots in a single call (magnitude axis inverted). Sentinel magnitudes (99.99) are excluded automatically; set point_sources=true when the user asks for stars/point sources. Use this for 'plot a CMD'/'g vs g-r' requests instead of chaining query+plot tools.",
        function=agent._datalab_image_tool_fn("datalab_color_magnitude_diagram"),
        parameters={
            "type": "object",
            "properties": {
                "catalog": {"type": "string"},
                "table": {"type": "string", "description": "Optional; defaults to the catalog's primary table."},
                "radius_deg": {"type": "number", "default": 0.4},
                "ra": {"type": "number"},
                "dec": {"type": "number"},
                "target_name": {"type": "string"},
                "blue_band": {"type": "string", "default": "g"},
                "red_band": {"type": "string", "default": "r"},
                "mag_band": {"type": "string", "description": "Magnitude (y) band; defaults to blue_band."},
                "limit": {"type": "integer", "default": 5000},
                "point_sources": {"type": "boolean", "default": False, "description": "True = keep only point sources via the catalog's registered star cut (e.g. NSC class_star>0.5)."},
                "morphology": {"type": "object", "description": "Explicit morphology cut applied in the SQL WHERE, e.g. {'column':'class_star','op':'>','value':0.5} or {'column':'ext_coadd','between':[0,1]}; overrides point_sources."},
                "value_cuts": {"type": "array", "items": {"type": "object"}, "description": "Extra server-side cuts, e.g. [{'column':'parallax_over_error','op':'>','value':5}]."},
            },
            "required": ["catalog"],
        },
        category="datalab",
    ))
    agent.tool_registry.register(Tool(
        name="datalab_tiled_search",
        description="P15: tiled region-bounded overdensity search over a footprint. Runs a server-side density aggregate per q3c cone tile, finds matched-filter peaks, and ranks candidates. Executes as a background job; for a large area it returns needs_confirmation first — re-call with confirm=true after confirming the sky area with the user.",
        function=agent._datalab_tool_fn("datalab_tiled_search"),
        parameters={
            "type": "object",
            "properties": {
                "catalog": {"type": "string"},
                "table": {"type": "string"},
                "ra_min": {"type": "number"},
                "ra_max": {"type": "number"},
                "dec_min": {"type": "number"},
                "dec_max": {"type": "number"},
                "tile_radius_deg": {"type": "number", "default": 2.0},
                "step_deg": {"type": "number", "default": 0.05},
                "color_cut": {"type": "object"},
                "value_cuts": {"type": "array", "items": {"type": "object"}},
                "morphology": {"type": "object"},
                "peak_threshold": {"type": "number", "default": 3.0},
                "max_tiles": {"type": "integer", "default": 64},
                "candidate_budget": {"type": "integer", "default": 50},
                "confirm": {"type": "boolean", "default": False},
            },
            "required": ["catalog", "table", "ra_min", "ra_max", "dec_min", "dec_max"],
        },
        category="datalab",
    ))
    agent.tool_registry.register(Tool(
        name="datalab_confirm_sky_area",
        description="Estimate the sky area and tile count for a tiled search before fanning out (guardrail: confirm wide scans with the user first).",
        function=agent._datalab_tool_fn("datalab_confirm_sky_area"),
        parameters={
            "type": "object",
            "properties": {
                "ra_min": {"type": "number"}, "ra_max": {"type": "number"},
                "dec_min": {"type": "number"}, "dec_max": {"type": "number"},
                "tile_radius_deg": {"type": "number", "default": 2.0},
            },
            "required": ["ra_min", "ra_max", "dec_min", "dec_max"],
        },
        category="datalab",
    ))
    agent.tool_registry.register(Tool(
        name="datalab_job_status",
        description="Poll the status of a Data Lab background job — local 'dlj_' ids (tiled search, local async queries) and real server-side job ids from async_submit.",
        function=agent._datalab_tool_fn("datalab_job_status"),
        parameters={"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]},
        category="datalab",
    ))
    agent.tool_registry.register(Tool(
        name="datalab_job_results",
        description="Fetch the result of a Data Lab background job (local 'dlj_' or server-side id). Server-job rows are stored and returned as a result_id + preview.",
        function=agent._datalab_tool_fn("datalab_job_results"),
        parameters={"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]},
        category="datalab",
    ))
    agent.tool_registry.register(Tool(
        name="datalab_job_cancel",
        description="Cancel a running Data Lab background job (local 'dlj_' or server-side id).",
        function=agent._datalab_tool_fn("datalab_job_cancel"),
        parameters={"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]},
        category="datalab",
    ))
    agent.tool_registry.register(Tool(
        name="datalab_save_result",
        description="Save a Data Lab result_id as a durable named table (MyDB-lite): unlike result_ids (1-hour TTL), saved tables survive restarts. Reload with datalab_load_my_table(name).",
        function=agent._datalab_tool_fn("datalab_save_result"),
        parameters={
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "result_id returned by a datalab_* tool."},
                "name": {"type": "string", "description": "Table name: 1-64 chars, letter first, then letters/digits/_/- (e.g. 'lmc_rr_lyrae_candidates')."},
                "description": {"type": "string", "description": "Optional one-line note shown in the My tables panel."},
            },
            "required": ["result_id", "name"],
        },
        category="datalab",
    ))
    agent.tool_registry.register(Tool(
        name="datalab_list_my_tables",
        description="List the user's saved Data Lab tables (names, row counts, provenance).",
        function=agent._datalab_tool_fn("datalab_list_my_tables"),
        parameters={"type": "object", "properties": {}, "required": []},
        category="datalab",
    ))
    agent.tool_registry.register(Tool(
        name="datalab_load_my_table",
        description="Load a saved Data Lab table by name into a fresh result_id usable by every result_id-consuming tool (plots, crossmatch, get_result).",
        function=agent._datalab_tool_fn("datalab_load_my_table"),
        parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        category="datalab",
    ))
    agent.tool_registry.register(Tool(
        name="xmatch_user_list",
        description="Crossmatch a user object list (ra/dec dicts) OR a stored Data Lab result_id against a VizieR catalog or SIMBAD via the CDS X-Match service; returns a result_id with every uploaded column plus the match columns (angDist arcsec).",
        function=agent._datalab_tool_fn("xmatch_user_list"),
        parameters={
            "type": "object",
            "properties": {
                "catalog": {"type": "string", "description": "Alias (simbad, 2mass, allwise, unwise, gaia_dr3, sdss_dr12, ps1, tycho2, nvss, first, galex) or explicit 'vizier:<table id>' e.g. 'vizier:II/246/out'."},
                "objects": {"type": "array", "items": {"type": "object"}, "description": "User list: [{'ra':10.68,'dec':41.27,'name':'M31'}, ...]. Alternative to result_id."},
                "result_id": {"type": "string", "description": "Stored Data Lab result to upload instead of an objects list."},
                "ra_column": {"type": "string", "default": "ra"},
                "dec_column": {"type": "string", "default": "dec"},
                "radius_arcsec": {"type": "number", "default": 5.0, "description": "Match radius in arcsec (service max 180)."},
                "selection": {"type": "string", "enum": ["best", "all"], "default": "best"},
            },
            "required": ["catalog"],
        },
        category="datalab",
    ))
    agent.tool_registry.register(Tool(
        name="datalab_export_notebook",
        description="Export a reproducible Jupyter notebook for a Data Lab analysis: the governed TAP SQL (qc.query(sql=...)), an optional SIA cutout recipe, an optional SVO filter-wavelength lookup, and a data-citation cell.",
        function=agent._datalab_tool_fn("datalab_export_notebook", clear_card=False),
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "default": "NOIRLab Data Lab analysis"},
                "sql": {"type": "string", "description": "The governed native SQL to reproduce (e.g. a datalab tool's query_summary)."},
                "catalog": {"type": "string"},
                "table": {"type": "string"},
                "sia_ra": {"type": "number"},
                "sia_dec": {"type": "number"},
                "sia_fov_deg": {"type": "number", "default": 0.1},
                "sia_endpoint": {"type": "string"},
                "svo_filters": {"type": "array", "items": {"type": "string"}},
            },
            "required": [],
        },
        category="datalab",
    ))
    # -- Multimodal Universe HATS catalogs (LSDB / Hugging Face) --
    agent.tool_registry.register(Tool(
        name="list_mmu_hats_catalogs",
        description="List available Multimodal Universe HATS catalogs (Gaia, DESI, SDSS, TESS, Chandra) that Quasar can cone-search from Hugging Face via LSDB. These provide catalog/source properties, not archive observations.",
        function=agent._list_mmu_hats_catalogs,
        parameters={"type": "object", "properties": {}, "required": []},
        category="mmu_hats",
    ))
    agent.tool_registry.register(Tool(
        name="search_mmu_hats_catalog",
        description=(
            "Cone-search a Multimodal Universe HATS catalog using LSDB/Hugging Face. "
            "Use this for catalog/source properties such as Gaia astrometry, DESI/SDSS redshifts, "
            "TESS source metadata, Chandra spectra metadata, or cross-survey source enrichment "
            "around a sky position. Do NOT use this for finding archive observations, project/proposal IDs, "
            "or FITS/data products -- use the archive tools for those."
        ),
        function=agent._search_mmu_hats_catalog,
        parameters={
            "type": "object",
            "properties": {
                "catalog_key": {"type": "string", "enum": sorted(MMU_HATS_CATALOGS)},
                "ra": {"type": "number", "description": "ICRS right ascension in degrees."},
                "dec": {"type": "number", "description": "ICRS declination in degrees."},
                "radius_arcsec": {"type": "number", "default": 120, "description": "Cone radius in arcseconds; capped by MMU_HATS_MAX_RADIUS_ARCSEC."},
                "columns": {"type": "array", "items": {"type": "string"}},
                "max_rows": {"type": "integer", "default": 500, "description": "Maximum rows returned to the UI; capped by MMU_HATS_MAX_ROWS."},
                "target_name": {"type": "string", "description": "Resolve this name to RA/Dec instead of passing ra/dec."},
            },
            "required": [],
        },
        category="mmu_hats",
    ))
    agent.tool_registry.register(Tool(
        name="crossmatch_mmu_hats_catalogs",
        description=(
            "Crossmatch two Multimodal Universe HATS catalogs (e.g. gaia x desi_edr_sv3) "
            "within a bounded cone region using LSDB margin caches. Requires ra/dec/radius -- "
            "all-sky crossmatches are not allowed. If this fails, run two bounded "
            "search_mmu_hats_catalog cone searches instead."
        ),
        function=agent._crossmatch_mmu_hats_catalogs,
        parameters={
            "type": "object",
            "properties": {
                "left_catalog_key": {"type": "string", "enum": sorted(MMU_HATS_CATALOGS)},
                "right_catalog_key": {"type": "string", "enum": sorted(MMU_HATS_CATALOGS)},
                "ra": {"type": "number"},
                "dec": {"type": "number"},
                "radius_arcsec": {"type": "number", "default": 120},
                "match_radius_arcsec": {"type": "number", "default": 1.0},
                "columns_left": {"type": "array", "items": {"type": "string"}},
                "columns_right": {"type": "array", "items": {"type": "string"}},
                "max_rows": {"type": "integer", "default": 500},
            },
            "required": ["left_catalog_key", "right_catalog_key", "ra", "dec"],
        },
        category="mmu_hats",
    ))

    # -- Live imagery and external spectra (hips2fits, ZTF, NED, SparCL) --
    agent.tool_registry.register(Tool(
        name="hips_cutout",
        description=(
            "Fetch a live CDS hips2fits PNG cutout for 'show me X', 'what does X look like', "
            "or multiwavelength postage-stamp requests. Supports aliases optical/dss/dss2, sdss, "
            "2mass/nir, wise/mir, galex/uv, rosat/xray, fermi/gamma, vlass/radio, or raw HiPS IDs. "
            "Use this for broad survey imagery; hips2fits exposes roughly 1000 HiPS surveys."
        ),
        function=agent._hips_cutout,
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "Resolve this target name to RA/Dec if ra/dec are not supplied."},
                "ra": {"type": "number", "description": "ICRS right ascension in degrees."},
                "dec": {"type": "number", "description": "ICRS declination in degrees."},
                "survey": {"type": "string", "default": "optical", "description": "Survey alias or raw HiPS ID. Aliases: optical/dss/dss2, dss2_red, sdss, 2mass/nir, wise/mir, galex/uv, rosat/xray, fermi/gamma, vlass/radio."},
                "fov_deg": {"type": "number", "default": 0.25},
                "width": {"type": "integer", "default": 512},
            },
            "required": [],
        },
        category="archive",
    ))
    agent.tool_registry.register(Tool(
        name="hips_multiband_panel",
        description=(
            "Render a multi-panel CDS hips2fits survey view for appearance or multiwavelength "
            "postage-stamp requests. Defaults to optical, 2MASS, and WISE; failed panels are labeled."
        ),
        function=agent._hips_multiband_panel,
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string"},
                "ra": {"type": "number"},
                "dec": {"type": "number"},
                "surveys": {"type": "array", "items": {"type": "string"}, "default": ["optical", "2mass", "wise"]},
                "fov_deg": {"type": "number", "default": 0.25},
            },
            "required": [],
        },
        category="archive",
    ))
    agent.tool_registry.register(Tool(
        name="vlass_cutout",
        description="Fetch a VLASS 3 GHz radio-continuum cutout via hips2fits. Use for radio appearance; VLASS covers Dec > -40 deg only.",
        function=agent._vlass_cutout,
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string"},
                "ra": {"type": "number"},
                "dec": {"type": "number"},
                "fov_deg": {"type": "number", "default": 0.1},
            },
            "required": [],
        },
        category="archive",
    ))
    agent.tool_registry.register(Tool(
        name="search_ztf_alerts",
        description="Search ALeRCE/ZTF alert objects near a target or sky position for transients and variability; returns a UI table of object IDs and detection metadata.",
        function=agent._search_ztf_alerts,
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string"},
                "ra": {"type": "number"},
                "dec": {"type": "number"},
                "radius_arcsec": {"type": "number", "default": 120},
                "max_rows": {"type": "integer", "default": 25},
            },
            "required": [],
        },
        category="archive",
    ))
    agent.tool_registry.register(Tool(
        name="ztf_light_curve",
        description="Plot an ALeRCE/ZTF light curve for an alert object oid, including detections and non-detection limits.",
        function=agent._ztf_light_curve,
        parameters={"type": "object", "properties": {"oid": {"type": "string"}}, "required": ["oid"]},
        category="analysis",
    ))
    agent.tool_registry.register(Tool(
        name="ztf_stamps",
        description="Render ALeRCE/ZTF science, template, and difference stamp PNG panels for an alert object oid and optional candid.",
        function=agent._ztf_stamps,
        parameters={
            "type": "object",
            "properties": {"oid": {"type": "string"}, "candid": {"type": "string"}},
            "required": ["oid"],
        },
        category="analysis",
    ))
    agent.tool_registry.register(Tool(
        name="search_space_lightcurves",
        description="List available TESS/Kepler/K2 light curves for a target; returns a table whose index can be used with plot_space_lightcurve or period_search.",
        function=agent._search_space_lightcurves,
        parameters={
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "mission": {"type": "string", "enum": ["TESS", "Kepler", "K2"]},
                "max_rows": {"type": "integer", "default": 20},
            },
            "required": ["target"],
        },
        category="archive",
    ))
    agent.tool_registry.register(Tool(
        name="plot_space_lightcurve",
        description="Download and plot a TESS/Kepler/K2 light curve for a target.",
        function=agent._plot_space_lightcurve,
        parameters={
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "mission": {"type": "string", "enum": ["TESS", "Kepler", "K2"]},
                "index": {"type": "integer", "default": 0},
            },
            "required": ["target"],
        },
        category="analysis",
    ))
    agent.tool_registry.register(Tool(
        name="period_search",
        description="Lomb-Scargle period search plus phase-folded plot on a TESS/Kepler light curve or a ZTF object by ALeRCE oid; returns best period, FAP, and top alternatives.",
        function=agent._period_search,
        parameters={
            "type": "object",
            "properties": {
                "source": {"type": "string", "enum": ["tess", "kepler", "ztf"]},
                "identifier": {"type": "string"},
                "mission": {"type": "string", "enum": ["TESS", "Kepler", "K2"]},
                "min_period_d": {"type": "number", "default": 0.05},
                "max_period_d": {"type": "number", "default": 30.0},
                "fid": {"type": "integer"},
                "index": {"type": "integer", "default": 0},
            },
            "required": ["source", "identifier"],
        },
        category="analysis",
    ))
    agent.tool_registry.register(Tool(
        name="ned_sed_plot",
        description="Plot a literature SED from NED photometry for a named target.",
        function=agent._ned_sed_plot,
        parameters={"type": "object", "properties": {"target_name": {"type": "string"}}, "required": ["target_name"]},
        category="analysis",
    ))
    agent.tool_registry.register(Tool(
        name="sparcl_find_spectra",
        description="Search NOIRLab SparCL for actual DESI/SDSS optical spectra near a target or position, not just redshift catalog values.",
        function=agent._sparcl_find_spectra,
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string"},
                "ra": {"type": "number"},
                "dec": {"type": "number"},
                "radius_arcsec": {"type": "number", "default": 60},
                "data_release": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer", "default": 20},
            },
            "required": [],
        },
        category="archive",
    ))
    agent.tool_registry.register(Tool(
        name="sparcl_search_spectra",
        description=(
            "Survey-scale SPARCL spectrum search by physical constraints: spectype "
            "(GALAXY/STAR/QSO), redshift range, and data release (DESI-DR1, SDSS-DR17) "
            "across 31M spectra — optionally combined with a cone around a target/position. "
            "Returns sparcl_ids ready for sparcl_get_spectrum / sparcl_plot_spectrum. "
            "Use sparcl_find_spectra instead for a pure position-only cone lookup."
        ),
        function=agent._spectra_tool_fn("sparcl_search_spectra"),
        parameters={
            "type": "object",
            "properties": {
                "spectype": {"type": "string", "enum": ["GALAXY", "STAR", "QSO"], "description": "Spectral classification to select."},
                "redshift_min": {"type": "number", "description": "Lower redshift bound (pairs with redshift_max)."},
                "redshift_max": {"type": "number", "description": "Upper redshift bound (pairs with redshift_min)."},
                "data_release": {"type": "array", "items": {"type": "string"}, "description": "SPARCL data releases; defaults to DESI-DR1 + SDSS-DR17."},
                "target_name": {"type": "string", "description": "Optional target to add a cone constraint around."},
                "ra": {"type": "number"},
                "dec": {"type": "number"},
                "radius_arcsec": {"type": "number", "description": "Cone radius when a target/position is given. Default 60."},
                "limit": {"type": "integer", "default": 100, "description": "Max spectra to return, capped at 500."},
            },
            "required": [],
        },
        category="archive",
    ))
    agent.tool_registry.register(Tool(
        name="sparcl_get_spectrum",
        description=(
            "Bulk-retrieve SPARCL spectrum arrays (wavelength, flux, ivar, model) for up to "
            "50 sparcl_ids as a stored data product. Returns per-spectrum scalars (n_points, "
            "wavelength range, redshift, spectype, median S/N) plus a result_id whose rows are "
            "the long-format arrays — fetch with datalab_get_result or chain into analysis. "
            "Use sparcl_plot_spectrum to just LOOK at one spectrum."
        ),
        function=agent._spectra_tool_fn("sparcl_get_spectrum"),
        parameters={
            "type": "object",
            "properties": {
                "sparcl_ids": {"type": "array", "items": {"type": "string"}, "description": "1-50 SPARCL UUIDs from a prior search."},
                "include": {"type": "array", "items": {"type": "string"}, "description": "Optional array/scalar fields; defaults to wavelength, flux, ivar, model, redshift, spectype."},
            },
            "required": ["sparcl_ids"],
        },
        category="analysis",
    ))
    agent.tool_registry.register(Tool(
        name="sparcl_stack_spectra",
        description=(
            "Stack (co-add) SPARCL spectra in bins so weak features emerge: either bin by "
            "redshift with constraints (spectype + redshift_min/max), or chain from a stored "
            "result_id whose rows carry sparcl_id plus any numeric bin_column (e.g. a g-r color "
            "computed in a Data Lab query). Spectra are rest-frame shifted, median-normalized, "
            "resampled to a common grid, and ivar-weighted averaged per bin. Returns the stacked-"
            "spectra plot and a result_id with the stacked arrays. Keep n_bins*n_per_bin modest "
            "(<=600); large stacks take minutes."
        ),
        function=agent._spectra_image_tool_fn("sparcl_stack_spectra"),
        parameters={
            "type": "object",
            "properties": {
                "spectype": {"type": "string", "enum": ["GALAXY", "STAR", "QSO"], "description": "Constraint mode: spectral class to stack."},
                "redshift_min": {"type": "number", "description": "Constraint mode: lower edge of the redshift bin axis."},
                "redshift_max": {"type": "number", "description": "Constraint mode: upper edge of the redshift bin axis."},
                "data_release": {"type": "array", "items": {"type": "string"}, "description": "SPARCL data releases; defaults to DESI-DR1 + SDSS-DR17."},
                "result_id": {"type": "string", "description": "Chained mode: stored result whose rows have sparcl_id + bin_column."},
                "bin_column": {"type": "string", "description": "Chained mode: numeric column of the stored result to bin on."},
                "bin_edges": {"type": "array", "items": {"type": "number"}, "description": "Explicit bin edges (else n_bins equal-width bins)."},
                "n_bins": {"type": "integer", "default": 4, "description": "Number of bins, capped at 12."},
                "n_per_bin": {"type": "integer", "default": 25, "description": "Spectra per bin, capped at 200 (total <= 600)."},
                "rest_frame": {"type": "boolean", "default": True, "description": "Shift to rest frame using each spectrum's redshift."},
                "weighting": {"type": "string", "enum": ["ivar", "uniform", "median"], "default": "ivar"},
                "normalize": {"type": "boolean", "default": True, "description": "Median-normalize each spectrum before stacking."},
                "title": {"type": "string"},
            },
            "required": [],
        },
        category="analysis",
    ))
    agent.tool_registry.register(Tool(
        name="sparcl_plot_spectrum",
        description="Plot an actual SparCL optical spectrum by sparcl_id with optional model overlay and redshifted line markers.",
        function=agent._sparcl_plot_spectrum,
        parameters={
            "type": "object",
            "properties": {
                "sparcl_id": {"type": "string"},
                "mark_lines": {"type": "boolean", "default": True},
                "smooth": {"type": "integer", "default": 0},
            },
            "required": ["sparcl_id"],
        },
        category="analysis",
    ))
    agent.tool_registry.register(Tool(
        name="get_sky_image",
        description=(
            "Fetch a sky survey cutout image for a target. Returns a FITS file "
            "and PNG preview from surveys like DSS2 (optical), 2MASS (near-IR), "
            "SDSS (optical), WISE (mid-IR), NVSS/FIRST (radio). "
            "Use this when users ask for 'an image of', 'show me', 'DSS image', "
            "or 'what does X look like'."
        ),
        function=agent._viz_tool_fn("get_sky_image"),
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "Astronomical target name (e.g., 'M87', 'Carina Nebula')"},
                "survey": {"type": "string", "description": "Survey name: 'dss2' (default optical), '2mass', 'sdss', 'wise', 'nvss', 'first'. Or specific like 'DSS2 Red', '2MASS-J'."},
                "radius_arcmin": {"type": "number", "description": "Image radius in arcminutes. Default 5."},
                "ra": {"type": "number", "description": "RA in degrees (alternative to target_name)"},
                "dec": {"type": "number", "description": "Dec in degrees (alternative to target_name)"},
            },
            "required": []
        }
    ))

    # ── Data Download Tools ───────────────────────────────────
    agent.tool_registry.register(Tool(
        name="download_mast_data",
        description=(
            "Download FITS files from the MAST archive (JWST/HST data). "
            "Call this AFTER search_mast to download actual science data files. "
            "Downloads to ~/quasar_data/mast/ by default. Has a safety limit of 10 files."
        ),
        function=agent._archives_tool_fn("download_mast_data"),
        parameters={
            "type": "object",
            "properties": {
                "product_type": {"type": "string", "description": "Filter: 'SCIENCE' (default), 'CALIBRATION', 'PREVIEW'"},
                "extension": {"type": "string", "description": "File extension filter: 'fits' (default), 'jpg', etc."},
                "max_files": {"type": "integer", "description": "Max files to download (default 10, safety limit)"},
            },
            "required": []
        }
    ))

    # ── CASA Script Generator Tools ────────────────────────────
    agent.tool_registry.register(Tool(
        name="generate_casa_imaging_script",
        description="Generate a complete CASA tclean imaging script for ALMA/VLA data. Returns ready-to-run Python code for radio interferometry imaging.",
        function=lambda **kw: agent.casa_generator.generate_casa_imaging_script(**kw),
        parameters={
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Target source field name (as in the MS)"},
                "vis": {"type": "string", "description": "Path to calibrated Measurement Set (.ms)"},
                "band": {"type": "string", "description": "ALMA band number (e.g. '6', '3', '7')"},
                "cell": {"type": "string", "description": "Cell size, e.g. '0.02arcsec'"},
                "imsize": {"type": "integer", "description": "Square image size in pixels"},
                "weighting": {"type": "string", "enum": ["briggs", "natural", "uniform"], "description": "Visibility weighting scheme"},
                "robust": {"type": "number", "description": "Briggs robust parameter (-2 to +2)"},
                "threshold": {"type": "string", "description": "Clean stopping threshold (e.g. '0.1mJy')"},
                "specmode": {"type": "string", "enum": ["mfs", "cube"], "description": "'mfs' for continuum, 'cube' for spectral line"},
            },
            "required": ["target", "vis"]
        }
    ))

    agent.tool_registry.register(Tool(
        name="generate_casa_calibration_script",
        description="Generate a CASA manual calibration script for ALMA/VLA data reduction. Returns ready-to-run Python code for bandpass, gain, and flux calibration.",
        function=lambda **kw: agent.casa_generator.generate_casa_calibration_script(**kw),
        parameters={
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Science target field name"},
                "vis": {"type": "string", "description": "Input Measurement Set path"},
                "flux_cal": {"type": "string", "description": "Flux calibrator field name"},
                "phase_cal": {"type": "string", "description": "Phase calibrator field name"},
                "refant": {"type": "string", "description": "Reference antenna name (e.g. 'DA41')"},
            },
            "required": ["target", "vis", "flux_cal", "phase_cal", "refant"]
        }
    ))

    # ── GCN / GW Alert Tools ───────────────────────────────────
    agent.tool_registry.register(Tool(
        name="get_latest_gw_events",
        description="Get the latest gravitational wave events from the GWOSC (Gravitational Wave Open Science Center) catalog. Use for multi-messenger astronomy queries.",
        function=lambda **kw: agent.gcn_monitor.get_latest_gcn_alerts(**kw),
        parameters={
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Number of recent events to return (default: 10)"},
                "event_type": {"type": "string", "description": "Filter by type: 'BBH', 'BNS', 'NSBH', or None for all"},
            },
            "required": []
        }
    ))

    agent.tool_registry.register(Tool(
        name="search_gwtc_catalog",
        description="Search the GWTC gravitational wave transient catalog with mass, distance, and type filters.",
        function=lambda **kw: agent.gcn_monitor.search_gwtc_catalog(**kw),
        parameters={
            "type": "object",
            "properties": {
                "mass_min_solar": {"type": "number", "description": "Minimum total mass in solar masses"},
                "mass_max_solar": {"type": "number", "description": "Maximum total mass in solar masses"},
                "distance_max_mpc": {"type": "number", "description": "Maximum luminosity distance (Mpc)"},
                "event_type": {"type": "string", "description": "Event type: 'BBH', 'BNS', 'NSBH'"},
            },
            "required": []
        }
    ))

    agent.tool_registry.register(Tool(
        name="summarize_gcn_circular",
        description="Fetch and parse a NASA GCN (Gamma-ray Coordinates Network) circular by number. Extracts key parameters: event name, trigger time, coordinates, and summary.",
        function=lambda **kw: agent.gcn_monitor.summarize_gcn_circular(**kw),
        parameters={
            "type": "object",
            "properties": {
                "circular_number": {"type": "integer", "description": "GCN circular number (e.g., 33000)"},
            },
            "required": ["circular_number"]
        }
    ))

    # ── NASA ADS Literature Tools ──────────────────────────────────────
    _ads = agent.ads_client  # may be None if no key

    agent.tool_registry.register(Tool(
        name="search_papers",
        description=(
            "Search the NASA ADS database for astronomical papers. "
            "Returns titles, authors, abstracts, citation counts, DOIs, and a direct link to each paper on NASA ADS. "
            "IMPORTANT: Pass the user's request as natural language — an internal AI query builder will "
            "automatically translate it into optimal ADS syntax using keyword searches, bibgroup filters, "
            "SIMBAD object linking, second-order discovery operators (trending, similar, useful), and more.\n"
            "Examples of what to pass as query:\n"
            "- 'recent papers on protoplanetary disks'\n"
            "- 'best ALMA papers on disk gaps'\n"
            "- 'papers about HL Tau'\n"
            "- 'what are people reading about FRBs right now'\n"
            "- 'foundational papers on planet formation'\n"
            "- 'review articles on AGN feedback'\n"
            "- 'papers by Sean Andrews on disk surveys'\n"
            "Do NOT try to construct ADS field syntax yourself — just pass the natural language query."
        ),
        function=agent._papers_tool_fn("search_papers", log_name="_search_papers"),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query describing what papers the user wants (e.g. 'recent ALMA papers on protoplanetary disk gaps')"},
                "max_results": {"type": "integer", "description": "Number of results to return (default 15, max 50)"},
                "sort": {"type": "string", "description": "Sort order: 'date desc' (newest first), 'citation_count desc' (most cited), 'score desc' (relevance). Default: 'date desc'"},
            },
            "required": ["query"]
        },
        category="literature"
    ))

    agent.tool_registry.register(Tool(
        name="search_papers_by_observation_id",
        description=(
            "Find NASA ADS papers explicitly connected to a specific archive identifier. "
            "Use this instead of generic search_papers when the user provides an ALMA project/proposal code "
            "(e.g. 2019.1.00123.S), MOUS/member_ous_uid (uid://...), ASDM UID, or archive dataset ID. "
            "The lookup uses exact identifier searches and returns provenance metadata for the graph."
        ),
        function=agent._papers_tool_fn(
            "search_papers_by_observation_id",
            log_name="_search_papers_by_observation_identifier",
        ),
        parameters={
            "type": "object",
            "properties": {
                "identifier": {"type": "string", "description": "Project/proposal code, MOUS UID, ASDM UID, or archive dataset identifier."},
                "facility": {"type": "string", "description": "Facility/bibgroup hint, default ALMA."},
                "max_results": {"type": "integer", "description": "Number of results to return (default 20, max 50)"},
            },
            "required": ["identifier"]
        },
        category="literature"
    ))

    agent.tool_registry.register(Tool(
        name="get_author_papers",
        description="Find all papers published by a specific author. Use 'Last, First' format for best results.",
        function=lambda author, max_results=20, **kw: (
            agent.ads_client.get_author_papers(author, max_results=max_results)
            if agent.ads_client else {"error": "ADS client not configured"}
        ),
        parameters={
            "type": "object",
            "properties": {
                "author": {"type": "string", "description": "Author name, e.g. 'Accomazzi, Alberto'"},
                "max_results": {"type": "integer", "description": "Max papers to return (default 20)"},
            },
            "required": ["author"]
        },
        category="literature"
    ))

    agent.tool_registry.register(Tool(
        name="get_paper_metrics",
        description="Get citation count, read statistics, and impact metrics for a specific paper by its ADS bibcode.",
        function=lambda bibcode, **kw: (
            agent.ads_client.get_paper_metrics(bibcode)
            if agent.ads_client else {"error": "ADS client not configured"}
        ),
        parameters={
            "type": "object",
            "properties": {
                "bibcode": {"type": "string", "description": "ADS bibcode, e.g. '2020PASP..132c5001L'"},
            },
            "required": ["bibcode"]
        },
        category="literature"
    ))

    agent.tool_registry.register(Tool(
        name="get_author_metrics",
        description="Calculate scholarly impact metrics for an author: h-index, i10-index, total citations, refereed citations.",
        function=lambda author, **kw: (
            agent.ads_client.get_author_metrics(author)
            if agent.ads_client else {"error": "ADS client not configured"}
        ),
        parameters={
            "type": "object",
            "properties": {
                "author": {"type": "string", "description": "Author name, e.g. 'Accomazzi, Alberto'"},
            },
            "required": ["author"]
        },
        category="literature"
    ))

    agent.tool_registry.register(Tool(
        name="export_bibtex",
        description="Export properly formatted BibTeX citations for one or more papers given their ADS bibcodes. Use when the user asks to cite papers or needs BibTeX.",
        function=lambda bibcodes, **kw: (
            agent.ads_client.export_bibtex(bibcodes if isinstance(bibcodes, list) else [bibcodes])
            if agent.ads_client else "% ADS client not configured"
        ),
        parameters={
            "type": "object",
            "properties": {
                "bibcodes": {
                    "oneOf": [
                        {"type": "string", "description": "Single bibcode"},
                        {"type": "array", "items": {"type": "string"}, "description": "List of bibcodes"}
                    ],
                    "description": "ADS bibcode(s) to export"
                },
            },
            "required": ["bibcodes"]
        },
        category="literature"
    ))

    agent.tool_registry.register(Tool(
        name="get_paper_abstract",
        description="Get the full abstract and detailed metadata (keywords, affiliations) for a specific paper by bibcode.",
        function=lambda bibcode, **kw: (
            agent.ads_client.get_paper_details(bibcode)
            if agent.ads_client else {"error": "ADS client not configured"}
        ),
        parameters={
            "type": "object",
            "properties": {
                "bibcode": {"type": "string", "description": "ADS bibcode of the paper"},
            },
            "required": ["bibcode"]
        },
        category="literature"
    ))

    agent.tool_registry.register(Tool(
        name="list_ads_libraries",
        description="List all personal ADS paper libraries for the authenticated ADS user.",
        function=lambda **kw: (
            agent.ads_client.list_libraries()
            if agent.ads_client else {"error": "ADS client not configured"}
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": []
        },
        category="literature"
    ))

    agent.tool_registry.register(Tool(
        name="extract_paper_details",
        description="Download a scientific paper by its arXiv ID or ADS bibcode and extract specific details (e.g. beam size, flux density, telescope configuration) using an LLM QA pass over the full text. Use when the user asks specific questions about the contents of a published paper.",
        function=agent._papers_tool_fn("extract_paper_details"),
        parameters={
            "type": "object",
            "properties": {
                "identifier": {"type": "string", "description": "arXiv ID (e.g. '1812.04040') or ADS bibcode (e.g. '2018ApJ...869L..41A')"},
                "query": {"type": "string", "description": "The specific question to ask about the paper's contents (e.g. 'What was the exact angular resolution achieved for AS 209?')"}
            },
            "required": ["identifier", "query"]
        },
        category="literature"
    ))

    agent.tool_registry.register(Tool(
        name="evaluate_consensus",
        description=(
            "Evaluate the scientific consensus on a research question by searching NASA ADS for the most-cited "
            "papers on the topic, reading all their abstracts, and producing a structured analysis. "
            "The output includes: overall consensus level (Strong Agreement / Divided / etc.), "
            "which specific papers agree vs disagree, WHY they disagree (different methods, data, assumptions), "
            "key evidence from each side with proper citations, how the consensus has evolved over time, "
            "and what open questions remain. "
            "Use this when the user asks questions like: 'Do scientists agree on X?', 'What does the field think about X?', "
            "'Is there consensus on X?', 'What's the current understanding of X?', or any question where "
            "a literature-wide summary would be more useful than individual paper results."
        ),
        function=agent._papers_tool_fn("evaluate_consensus", log_name="_evaluate_consensus"),
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The scientific question to evaluate consensus on (e.g. 'Is planet migration necessary for hot Jupiter formation?')"},
                "max_papers": {"type": "integer", "description": "Number of top-cited papers to analyze (default 20, max 50)"},
            },
            "required": ["question"]
        },
        category="literature"
    ))

    agent.tool_registry.register(Tool(
        name="reproduce_paper_methods",
        description="Extract the methodology from a published paper (by arXiv ID or ADS bibcode) and construct a Python/CASA data reduction script that replicates its steps. Use when a user asks 'how did they reduce the data for this paper' or 'reproduce this paper'.",
        function=agent._papers_tool_fn("reproduce_paper_methods"),
        parameters={
            "type": "object",
            "properties": {
                "identifier": {"type": "string", "description": "arXiv ID (e.g. '1812.04040') or ADS bibcode (e.g. '2018ApJ...869L..41A')"}
            },
            "required": ["identifier"]
        },
        category="literature"
    ))

    agent.tool_registry.register(Tool(
        name="get_ads_library_papers",
        description="Get papers stored in a specific personal ADS library by library ID.",
        function=lambda library_id, max_results=50, **kw: (
            agent.ads_client.get_library_papers(library_id, max_results=max_results)
            if agent.ads_client else {"error": "ADS client not configured"}
        ),
        parameters={
            "type": "object",
            "properties": {
                "library_id": {"type": "string", "description": "ADS library ID"},
                "max_results": {"type": "integer", "description": "Max papers to return"},
            },
            "required": ["library_id"]
        },
        category="literature"
    ))

    agent.tool_registry.register(Tool(
        name="create_ads_library",
        description="Create a new personal ADS paper library to organize papers by topic or project.",
        function=lambda name, description="", public=False, **kw: (
            agent.ads_client.create_library(name, description=description, public=public)
            if agent.ads_client else {"error": "ADS client not configured"}
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Library name"},
                "description": {"type": "string", "description": "Library description"},
                "public": {"type": "boolean", "description": "Whether the library is public"},
            },
            "required": ["name"]
        },
        category="literature"
    ))

    agent.tool_registry.register(Tool(
        name="add_to_ads_library",
        description="Add papers to an existing personal ADS library by their bibcodes.",
        function=lambda library_id, bibcodes, **kw: (
            agent.ads_client.add_to_library(library_id, bibcodes if isinstance(bibcodes, list) else [bibcodes])
            if agent.ads_client else {"error": "ADS client not configured"}
        ),
        parameters={
            "type": "object",
            "properties": {
                "library_id": {"type": "string", "description": "ADS library ID to add to"},
                "bibcodes": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}}
                    ],
                    "description": "Bibcode(s) to add"
                },
            },
            "required": ["library_id", "bibcodes"]
        },
        category="literature"
    ))

    # ── OpenAlex Researcher & Bibliometric Tools ──────────────────────
    _oalex = agent.openalex_client

    agent.tool_registry.register(Tool(
        name="lookup_researcher",
        description=(
            "Look up a researcher/scientist by name or ORCID to get their full academic profile: "
            "current institution, h-index, i10-index, total publications, total citations, "
            "ORCID, Scopus ID, research topics, affiliation history, and publication trend "
            "over the last 10 years.  Powered by OpenAlex (90M+ disambiguated authors).\n"
            "Use this when the user asks about a person, wants to know who someone is, "
            "or wants contact/institutional information about a researcher.\n"
            "Examples: 'Who is Andrea Isella?', 'Tell me about Crystal Brogan', "
            "'Look up ORCID 0000-0001-2345-6789'"
        ),
        function=agent._papers_tool_fn("lookup_researcher", log_name="_lookup_researcher"),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Researcher name (e.g. 'Andrea Isella') or ORCID "
                        "(e.g. '0000-0001-2345-6789')"
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max author matches to return (default 3)",
                },
            },
            "required": ["query"]
        },
        category="literature"
    ))

    agent.tool_registry.register(Tool(
        name="get_research_trends",
        description=(
            "Get a bibliometric trend showing papers-per-year for a given topic or search "
            "query.  Returns total paper count and yearly breakdown.\n"
            "Use when the user asks 'How much research is being done on X?', "
            "'Is interest in X growing?', 'Publication trends for FRBs'.\n"
            "Also returns the funding landscape — which funders (NSF, NASA, ESA, etc.) "
            "have funded research on the topic."
        ),
        function=agent._papers_tool_fn("get_research_trends", log_name="_get_research_trends"),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic or search query (e.g. 'fast radio bursts', 'ALMA protoplanetary disks')",
                },
                "year_from": {
                    "type": "integer",
                    "description": "Start year for the trend (default 2015)",
                },
                "year_to": {
                    "type": "integer",
                    "description": "End year for the trend (default 2026)",
                },
            },
            "required": ["query"]
        },
        category="literature"
    ))

    agent.tool_registry.register(Tool(
        name="generate_jupyter_notebook",
        description="Generate a runnable Jupyter Notebook (.ipynb) for data analysis workflows. Use this when the user asks for code to analyze data, make maps, or perform reductions.",
        function=lambda title, steps, **kw: agent._generate_notebook(title, steps),
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Title of the notebook"},
                "steps": {
                    "type": "array",
                    "description": "List of notebook cells (markdown or code)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["markdown", "code"]},
                            "content": {"type": "string", "description": "The exact cell content. Use complete astropy/spectral-cube/numpy code for code cells."}
                        },
                        "required": ["type", "content"]
                    }
                }
            },
            "required": ["title", "steps"]
        },
        category="analysis"
    ))

    # ── FITS Image Rendering Tools ─────────────────────────────
    agent.tool_registry.register(Tool(
        name="render_fits_image",
        description=(
            "Download a FITS file from an archive URL and render it as a "
            "publication-quality image displayed inline in the chat. Use this "
            "when the user asks to SEE or VISUALIZE data. The URL should come "
            "from a prior archive search (access_url or datalink URL)."
        ),
        function=agent._viz_tool_fn("render_fits_image"),
        parameters={
            "type": "object",
            "properties": {
                "url":      {"type": "string", "description": "Direct URL to the FITS file (from access_url, DataLink, or CADC cutout service)."},
                "title":    {"type": "string", "description": "Title for the rendered image (e.g., 'JWST NIRCam F200W — Hubble Ultra Deep Field')."},
                "colormap": {"type": "string", "description": "Matplotlib colormap. Default 'inferno'. Options: 'viridis', 'plasma', 'magma', 'gray', 'hot'."},
                "stretch":  {"type": "string", "enum": ["sqrt", "log", "linear", "asinh"], "description": "Pixel stretch. Default 'sqrt'."},
            },
            "required": ["url"]
        },
        category="analysis"
    ))

    agent.tool_registry.register(Tool(
        name="overlay_fits_images",
        description=(
            "Download two FITS files and create an overlay composite: one rendered "
            "as a colorscale background, the other as contours on top. Uses WCS "
            "reprojection to align them. Perfect for showing ALMA contours on "
            "JWST/HST colorscale images."
        ),
        function=agent._viz_tool_fn("overlay_fits_images"),
        parameters={
            "type": "object",
            "properties": {
                "base_url":       {"type": "string", "description": "URL to the base/background FITS file (rendered as colorscale)."},
                "contour_url":    {"type": "string", "description": "URL to the FITS file rendered as contours on top."},
                "base_label":     {"type": "string", "description": "Label for the base image (e.g., 'JWST NIRCam'). Default 'JWST'."},
                "contour_label":  {"type": "string", "description": "Label for the contour image (e.g., 'ALMA Band 6'). Default 'ALMA'."},
                "base_cmap":      {"type": "string", "description": "Colormap for base image. Default 'inferno'."},
                "contour_levels": {"type": "integer", "description": "Number of contour levels. Default 8."},
            },
            "required": ["base_url", "contour_url"]
        },
        category="analysis"
    ))

    agent.tool_registry.register(Tool(
        name="overlay_archive_images",
        description=(
            "End-to-end archive image overlay workflow. Queries MAST/JWST for a "
            "background FITS image and ALMA/DataLink for contour FITS near a named "
            "region, WCS-aligns them, and renders a PNG. Use for requests like "
            "'Overlay ALMA contours on JWST image for HUDF'."
        ),
        function=agent._viz_tool_fn("overlay_archive_images"),
        parameters={
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "Named region or source, e.g. HUDF, M87, HH 212. Can also contain decimal RA/Dec."},
                "ra_deg": {"type": "number", "description": "Optional ICRS right ascension in degrees. Use with dec_deg for arbitrary regions."},
                "dec_deg": {"type": "number", "description": "Optional ICRS declination in degrees. Use with ra_deg for arbitrary regions."},
                "base_archive": {"type": "string", "description": "Base image archive, default MAST."},
                "base_collection": {"type": "string", "description": "Base collection/mission, default JWST."},
                "contour_archive": {"type": "string", "description": "Contour archive, default ALMA."},
                "radius_arcmin": {"type": "number", "description": "Search radius around the region center. Default 1."},
                "max_product_mb": {"type": "number", "description": "Maximum FITS product size to select. Default 150."},
            },
            "required": ["region"]
        },
        category="analysis"
    ))

    agent.tool_registry.register(Tool(
        name="compute_moment_map",
        description=(
            "Download a FITS spectral cube and compute a moment map. "
            "Moment 0 = integrated intensity (total emission). "
            "Moment 1 = velocity field (mean velocity). "
            "Moment 2 = velocity dispersion (turbulence). "
            "Use this for ALMA cubes when the user asks about emission maps, "
            "velocity fields, or line intensity maps."
        ),
        function=agent._viz_tool_fn("compute_moment_map"),
        parameters={
            "type": "object",
            "properties": {
                "url":          {"type": "string",  "description": "Direct URL to the FITS spectral cube."},
                "order":        {"type": "integer", "description": "Moment order: 0 (intensity), 1 (velocity), 2 (dispersion). Default 0."},
                "title":        {"type": "string",  "description": "Title for the rendered image."},
                "colormap":     {"type": "string",  "description": "Colormap for moment 0. Moment 1 uses RdBu_r, moment 2 uses magma. Default 'inferno'."},
                "freq_min_ghz": {"type": "number",  "description": "Optional: only use channels above this frequency (GHz) for the moment."},
                "freq_max_ghz": {"type": "number",  "description": "Optional: only use channels below this frequency (GHz) for the moment."},
            },
            "required": ["url"]
        },
        category="analysis"
    ))

    agent.tool_registry.register(Tool(
        name="extract_spectrum",
        description=(
            "Download a FITS spectral cube and extract a 1D spectrum at a given "
            "sky position (RA/Dec) or pixel coordinate. If no position is given, "
            "extracts at the peak emission pixel. Pass radius_arcsec for an "
            "aperture-integrated spectrum (Jy/beam cubes convert to Jy) — "
            "single-pixel spectra underestimate resolved sources. The spectrum "
            "is plotted as flux vs frequency/velocity and displayed inline."
        ),
        function=agent._viz_tool_fn("extract_spectrum"),
        parameters={
            "type": "object",
            "properties": {
                "url":     {"type": "string", "description": "Direct URL to the FITS spectral cube."},
                "ra_deg":  {"type": "number", "description": "RA in decimal degrees (ICRS). Optional."},
                "dec_deg": {"type": "number", "description": "Dec in decimal degrees (ICRS). Optional."},
                "x_pixel": {"type": "integer", "description": "X pixel coordinate. Optional. Use if RA/Dec not available."},
                "y_pixel": {"type": "integer", "description": "Y pixel coordinate. Optional."},
                "title":   {"type": "string",  "description": "Title for the spectrum plot."},
                "radius_arcsec": {"type": "number", "description": "Optional aperture radius in arcsec for an integrated spectrum instead of a single pixel."},
            },
            "required": ["url"]
        },
        category="analysis"
    ))

    # ── Spectral Line Profile Fitter (R3) ──────────────────────
    agent.tool_registry.register(Tool(
        name="fit_spectral_line",
        description=(
            "Download a FITS spectral cube, extract a 1D spectrum at a given "
            "position, and fit a Gaussian profile to the strongest line. "
            "Returns peak flux, FWHM (in frequency and velocity), center "
            "frequency, and integrated flux. The fit is overlaid on the "
            "spectrum plot."
        ),
        function=agent._viz_tool_fn("fit_spectral_line"),
        parameters={
            "type": "object",
            "properties": {
                "url":     {"type": "string", "description": "Direct URL to the FITS spectral cube."},
                "ra_deg":  {"type": "number", "description": "RA in decimal degrees (ICRS). Optional."},
                "dec_deg": {"type": "number", "description": "Dec in decimal degrees (ICRS). Optional."},
                "x_pixel": {"type": "integer", "description": "X pixel coordinate. Optional."},
                "y_pixel": {"type": "integer", "description": "Y pixel coordinate. Optional."},
                "title":   {"type": "string",  "description": "Title for the spectrum plot."},
            },
            "required": ["url"]
        },
        category="analysis"
    ))

    # ── Quantitative image analysis + hips2fits FITS-mode products (2026-07) ──
    _IMG_INPUT_PROPS = {
        "url":         {"type": "string", "description": "Direct FITS URL (archive access_url, hips2fits FITS URL, SODA cutout). Preferred when available."},
        "survey":      {"type": "string", "description": "HiPS survey alias or raw ID for a cutout when no url is given (optical/dss2, sdss, 2mass, wise, galex, xray, vlass/radio, ...)."},
        "target_name": {"type": "string", "description": "Target to resolve to RA/Dec when no url/ra/dec is given."},
        "ra":          {"type": "number", "description": "ICRS right ascension in degrees (with survey mode)."},
        "dec":         {"type": "number", "description": "ICRS declination in degrees (with survey mode)."},
        "fov_deg":     {"type": "number", "description": "Cutout field of view in degrees (survey mode). Default 0.25."},
        "width":       {"type": "integer", "description": "Cutout width in pixels (survey mode, max 2048). Default 512."},
        "title":       {"type": "string", "description": "Title for the rendered figure."},
    }
    agent.tool_registry.register(Tool(
        name="image_statistics",
        description=(
            "Measure sigma-clipped statistics, robust MAD noise, a 5-sigma "
            "point-source limit, coverage/blankness fractions, and a pixel-value "
            "histogram for a FITS image (url) or any survey cutout (survey + "
            "position). Use to answer 'is my source detectable in this survey' "
            "or 'is this cutout blank' before deeper analysis."
        ),
        function=agent._viz_tool_fn("image_statistics"),
        parameters={"type": "object", "properties": dict(_IMG_INPUT_PROPS), "required": []},
        category="analysis",
    ))
    agent.tool_registry.register(Tool(
        name="detect_sources",
        description=(
            "Detect sources in a FITS image or survey cutout (photutils "
            "DAOStarFinder, segmentation fallback) and measure aperture "
            "photometry with local background annuli. Returns an annotated "
            "detection image plus a source list with RA/Dec, peak, aperture "
            "flux, and SNR (integrated Jy for Jy/beam radio maps). Use for "
            "'how many sources are in this field and how bright are they'."
        ),
        function=agent._viz_tool_fn("detect_sources"),
        parameters={
            "type": "object",
            "properties": {
                **_IMG_INPUT_PROPS,
                "threshold_sigma": {"type": "number", "description": "Detection threshold in background sigma. Default 5."},
                "fwhm_arcsec": {"type": "number", "description": "Expected source FWHM in arcsec (defaults to the beam, else 3 px)."},
                "max_sources": {"type": "integer", "description": "Max sources returned, brightest first. Default 100, cap 500."},
            },
            "required": [],
        },
        category="analysis",
    ))
    agent.tool_registry.register(Tool(
        name="measure_region",
        description=(
            "Measure statistics inside a sky region on a FITS image or survey "
            "cutout: sum, mean/median, MAD RMS, area in arcsec^2, and integrated "
            "flux in Jy for Jy/beam maps with a beam. Accepts a DS9 region "
            "string (circle/ellipse/box/polygon — paste straight from DS9/CARTA) "
            "or ra/dec + radius_arcsec. Renders the region on the image."
        ),
        function=agent._viz_tool_fn("measure_region"),
        parameters={
            "type": "object",
            "properties": {
                **_IMG_INPUT_PROPS,
                "region": {"type": "string", "description": "DS9 region string in sky coords, e.g. 'circle(150.1d, 2.2d, 30\")'."},
                "radius_arcsec": {"type": "number", "description": "Circular aperture radius in arcsec (with ra/dec) when no region string is given."},
            },
            "required": [],
        },
        category="analysis",
    ))
    agent.tool_registry.register(Tool(
        name="fit_gaussian_source",
        description=(
            "Fit a 2D Gaussian to a source in a FITS image (CASA imfit "
            "workflow): peak, integrated flux (Jy for Jy/beam maps), fitted "
            "FWHM sizes/PA, and the beam-deconvolved size or a 'consistent "
            "with point source' verdict when the header has a restoring beam. "
            "Renders a data/model/residual panel. Defaults to the peak pixel."
        ),
        function=agent._viz_tool_fn("fit_gaussian_source"),
        parameters={
            "type": "object",
            "properties": {
                **_IMG_INPUT_PROPS,
                "x_pixel": {"type": "number", "description": "X pixel position of the source (alternative to ra/dec)."},
                "y_pixel": {"type": "number", "description": "Y pixel position of the source."},
                "box_arcsec": {"type": "number", "description": "Fit box full width in arcsec (defaults to ~6 beam majors)."},
            },
            "required": [],
        },
        category="analysis",
    ))
    agent.tool_registry.register(Tool(
        name="radial_profile",
        description=(
            "Azimuthally averaged radial profile + curve of growth at a "
            "position in a FITS image or survey cutout: FWHM, half-light "
            "radius, total flux with a convergence check, beam HWHM marked "
            "for radio maps. The standard extended-vs-point-source and "
            "asymptotic-flux diagnostic."
        ),
        function=agent._viz_tool_fn("radial_profile"),
        parameters={
            "type": "object",
            "properties": {
                **_IMG_INPUT_PROPS,
                "x_pixel": {"type": "number", "description": "X pixel position (alternative to ra/dec)."},
                "y_pixel": {"type": "number", "description": "Y pixel position."},
                "max_radius_arcsec": {"type": "number", "description": "Outer profile radius in arcsec (default: quarter of the image)."},
            },
            "required": [],
        },
        category="analysis",
    ))
    agent.tool_registry.register(Tool(
        name="hips_contour_overlay",
        description=(
            "Overlay one survey as CONTOURS on another survey's image at any "
            "position using calibrated hips2fits FITS cutouts — e.g. VLASS "
            "radio contours on DSS2 optical, WISE on SDSS. Works for any HiPS "
            "survey alias or raw ID; no archive FITS products needed. The "
            "classic multiwavelength counterpart/proposal figure."
        ),
        function=agent._viz_tool_fn("hips_contour_overlay"),
        parameters={
            "type": "object",
            "properties": {
                "base_survey":    {"type": "string", "description": "Survey rendered as the color image. Default 'optical' (DSS2)."},
                "contour_survey": {"type": "string", "description": "Survey rendered as contours. Default 'vlass'."},
                "target_name":    {"type": "string"},
                "ra":             {"type": "number"},
                "dec":            {"type": "number"},
                "fov_deg":        {"type": "number", "default": 0.25},
                "width":          {"type": "integer", "default": 512},
                "contour_levels": {"type": "integer", "default": 8},
                "base_cmap":      {"type": "string", "default": "inferno"},
            },
            "required": [],
        },
        category="analysis",
    ))
    agent.tool_registry.register(Tool(
        name="image_difference",
        description=(
            "WCS-align two FITS images (reproject), background/gain match, "
            "subtract, and render A, aligned B, and the A−B residual with "
            "residual statistics. Accepts two FITS urls OR two survey aliases "
            "+ one position. Use for epoch-to-epoch transient checks (e.g. "
            "DSS1 vs DSS2, two VLASS epochs) and morphology comparisons."
        ),
        function=agent._viz_tool_fn("image_difference"),
        parameters={
            "type": "object",
            "properties": {
                "url_a":       {"type": "string", "description": "FITS URL for image A (the reference)."},
                "url_b":       {"type": "string", "description": "FITS URL for image B (reprojected onto A)."},
                "survey_a":    {"type": "string", "description": "Survey alias for A when using cutout mode."},
                "survey_b":    {"type": "string", "description": "Survey alias for B when using cutout mode."},
                "target_name": {"type": "string"},
                "ra":          {"type": "number"},
                "dec":         {"type": "number"},
                "fov_deg":     {"type": "number", "default": 0.25},
                "width":       {"type": "integer", "default": 512},
                "label_a":     {"type": "string"},
                "label_b":     {"type": "string"},
                "scale_match": {"type": "boolean", "default": True, "description": "Background/gain match B to A before subtracting."},
                "title":       {"type": "string"},
            },
            "required": [],
        },
        category="analysis",
    ))
    agent.tool_registry.register(Tool(
        name="hips_rgb_composite",
        description=(
            "Build a Lupton three-color RGB composite from any three "
            "SINGLE-BAND HiPS surveys (R, G, B order) at a position. Use "
            "single-band aliases — 2mass_j/2mass_h/2mass_k, sdss_g/r/i/z, "
            "wise_w1..w4, galex_nuv/fuv, dss2_red/dss2_blue — or raw IDs like "
            "'CDS/P/SDSS9/i'. Avoid the color aliases (wise/2mass/sdss/optical): "
            "they are multi-plane and make poor channels. Channels arrive "
            "pixel-aligned from hips2fits; blank layers are rejected. Default "
            "is 2MASS K/H/J."
        ),
        function=agent._viz_tool_fn("hips_rgb_composite"),
        parameters={
            "type": "object",
            "properties": {
                "surveys":     {"type": "array", "items": {"type": "string"}, "description": "Exactly three single-band surveys in R, G, B order."},
                "target_name": {"type": "string"},
                "ra":          {"type": "number"},
                "dec":         {"type": "number"},
                "fov_deg":     {"type": "number", "default": 0.25},
                "width":       {"type": "integer", "default": 512},
                "stretch":     {"type": "number", "default": 5.0, "description": "Lupton stretch parameter."},
                "q":           {"type": "number", "default": 8.0, "description": "Lupton Q (softening) parameter."},
                "title":       {"type": "string"},
            },
            "required": [],
        },
        category="analysis",
    ))
    agent.tool_registry.register(Tool(
        name="vlass_epoch_comparison",
        description=(
            "Compare VLASS 3 GHz epochs (2017→now) at a position for radio "
            "variability/transient triage: per-epoch Quicklook cutouts from "
            "CADC rendered as a shared-stretch panel with blinkable frames, "
            "plus per-epoch peak flux, RMS, and a variability verdict that "
            "folds in the ~15% Quicklook systematic. Dec > -40 only. "
            "Complements vlass_cutout (median stack, no time axis)."
        ),
        function=agent._viz_tool_fn("vlass_epoch_comparison"),
        parameters={
            "type": "object",
            "properties": {
                "target_name":   {"type": "string"},
                "ra":            {"type": "number"},
                "dec":           {"type": "number"},
                "radius_arcsec": {"type": "number", "default": 60, "description": "Cutout radius per epoch (10–600 arcsec)."},
                "max_epochs":    {"type": "integer", "default": 6},
                "title":         {"type": "string"},
            },
            "required": [],
        },
        category="analysis",
    ))
    agent.tool_registry.register(Tool(
        name="moc_operations",
        description=(
            "MOC coverage algebra: intersection/union/difference of survey "
            "footprints (MOCServer dataset IDs from survey_coverage), with "
            "the resulting sky area in deg^2 and an interactive sky view of "
            "the derived footprint. Optionally pass ra_list/dec_list to flag "
            "which targets fall inside it. Answers 'where do these surveys "
            "overlap?' and 'which of my candidates have joint coverage?'."
        ),
        function=agent._viz_tool_fn("moc_operations"),
        parameters={
            "type": "object",
            "properties": {
                "survey_ids": {"type": "array", "items": {"type": "string"}, "description": "2+ MOCServer dataset IDs (e.g. 'CDS/P/DES-DR2/g'); difference = first minus the rest."},
                "operation":  {"type": "string", "enum": ["intersection", "union", "difference"], "default": "intersection"},
                "target_name": {"type": "string", "description": "Optional view center; defaults to a point inside the derived MOC."},
                "ra":         {"type": "number"},
                "dec":        {"type": "number"},
                "fov_deg":    {"type": "number", "default": 20.0},
                "survey":     {"type": "string", "default": "optical", "description": "Base imagery for the sky view."},
                "order":      {"type": "integer", "default": 8, "description": "HEALPix MOC order (3-10)."},
                "ra_list":    {"type": "array", "items": {"type": "number"}, "description": "Optional target RAs to test against the derived MOC."},
                "dec_list":   {"type": "array", "items": {"type": "number"}, "description": "Optional target Decs (same length as ra_list)."},
            },
            "required": ["survey_ids"],
        },
        category="archive",
    ))
    agent.tool_registry.register(Tool(
        name="pv_slice",
        description=(
            "Extract a position-velocity (PV) diagram from a FITS spectral "
            "cube along an ARBITRARY sky path (pvextractor): two endpoints "
            "in ICRS degrees (e.g. along a disk major axis from "
            "fit_gaussian_source) plus an optional averaging width in arcsec. "
            "The canonical rotation/outflow diagnostic for ALMA/VLA cubes."
        ),
        function=agent._viz_tool_fn("pv_slice"),
        parameters={
            "type": "object",
            "properties": {
                "url":          {"type": "string", "description": "Direct URL to the FITS spectral cube."},
                "ra_start":     {"type": "number", "description": "Path start RA (ICRS degrees)."},
                "dec_start":    {"type": "number", "description": "Path start Dec."},
                "ra_end":       {"type": "number", "description": "Path end RA."},
                "dec_end":      {"type": "number", "description": "Path end Dec."},
                "width_arcsec": {"type": "number", "description": "Optional averaging width perpendicular to the path."},
                "title":        {"type": "string"},
            },
            "required": ["url", "ra_start", "dec_start", "ra_end", "dec_end"],
        },
        category="analysis",
    ))

    # ── Astronomy Calculators (U9, U10, R6, R4) ───────────────
    agent.tool_registry.register(Tool(
        name="calculate_redshift",
        description=(
            "Compute cosmological quantities for a given redshift z using "
            "Planck18 cosmology. Returns luminosity distance, angular diameter "
            "distance, comoving distance, lookback time, age of the universe "
            "at that epoch, and the physical scale (kpc per arcsecond). "
            "Use for any question about distances, ages, or scales at a "
            "given redshift."
        ),
        function=agent._calc_tool_fn("calculate_redshift"),
        parameters={
            "type": "object",
            "properties": {
                "z": {"type": "number", "description": "Cosmological redshift (must be >= 0)."},
            },
            "required": ["z"]
        },
        category="analysis"
    ))

    agent.tool_registry.register(Tool(
        name="convert_coordinates",
        description=(
            "Convert sky coordinates between ICRS (RA/Dec), Galactic (l/b), "
            "Ecliptic (lon/lat), FK5 (J2000), and FK4 (B1950) frames. "
            "Returns the position in ALL frames at once. "
            "Use for coordinate transformations, epoch precession, or when "
            "the user gives Galactic coordinates and needs RA/Dec."
        ),
        function=agent._calc_tool_fn("convert_coordinates"),
        parameters={
            "type": "object",
            "properties": {
                "ra":           {"type": "number", "description": "RA or longitude in degrees (for ICRS/Ecliptic/FK5 input)."},
                "dec":          {"type": "number", "description": "Dec or latitude in degrees."},
                "l":            {"type": "number", "description": "Galactic longitude in degrees (for Galactic input)."},
                "b":            {"type": "number", "description": "Galactic latitude in degrees (for Galactic input)."},
                "input_frame":  {"type": "string", "description": "Source frame: 'icrs', 'galactic', 'ecliptic', 'fk5', 'fk4'. Default 'icrs'."},
                "output_frame": {"type": "string", "description": "Target frame (all frames are always returned). Default 'galactic'."},
            },
            "required": []
        },
        category="analysis"
    ))

    agent.tool_registry.register(Tool(
        name="calculate_beam",
        description=(
            "Calculate the synthesized beam size for a radio interferometer "
            "given the maximum baseline and observing frequency. For ALMA, "
            "you can specify an array configuration name (C-1 through C-10) "
            "instead of a raw baseline length. Returns beam size in arcsec "
            "and milliarcsec."
        ),
        function=agent._calc_tool_fn("calculate_beam"),
        parameters={
            "type": "object",
            "properties": {
                "frequency_ghz":  {"type": "number", "description": "Observing frequency in GHz."},
                "max_baseline_m": {"type": "number", "description": "Maximum baseline in meters. Optional if array_config is given."},
                "array_config":   {"type": "string", "description": "ALMA config name: C-1 through C-10. Overrides max_baseline_m."},
            },
            "required": ["frequency_ghz"]
        },
        category="analysis"
    ))

    agent.tool_registry.register(Tool(
        name="calculate_alma_sensitivity",
        description=(
            "Estimate ALMA continuum and spectral line sensitivity using "
            "the radiometer equation. Returns noise level in mJy/beam and "
            "uJy/beam for given band, bandwidth, and integration time. "
            "Includes Tsys scaling for weather (PWV). Use when the user "
            "asks about ALMA sensitivity, noise levels, or integration "
            "time estimates."
        ),
        function=agent._calc_tool_fn("calculate_alma_sensitivity"),
        parameters={
            "type": "object",
            "properties": {
                "band":             {"type": "integer", "description": "ALMA band number (3-10)."},
                "bandwidth_ghz":    {"type": "number",  "description": "Total continuum bandwidth in GHz. Default 7.5."},
                "t_integration_s":  {"type": "number",  "description": "On-source integration time in seconds. Default 60."},
                "n_antennas":       {"type": "integer", "description": "Number of antennas. Default 50."},
                "n_polarizations":  {"type": "integer", "description": "Number of polarizations (1 or 2). Default 2."},
                "channel_width_khz":{"type": "number",  "description": "Spectral channel width in kHz (for line sensitivity). Optional."},
                "pwv_mm":           {"type": "number",  "description": "Precipitable water vapor in mm. Default 1.0."},
            },
            "required": ["band"]
        },
        category="analysis"
    ))

    # ── Finding Chart Generator (O6) ──────────────────────────
    agent.tool_registry.register(Tool(
        name="generate_finding_chart",
        description=(
            "Generate a publication-quality finding chart for a target. "
            "Creates a DSS2 or 2MASS image with WCS axes, a target "
            "crosshair marker, N/E compass arrows, and an angular scale "
            "bar. Use when the user needs a finding chart for observations "
            "or proposals."
        ),
        function=agent._viz_tool_fn("generate_finding_chart"),
        parameters={
            "type": "object",
            "properties": {
                "target":      {"type": "string", "description": "Target name (e.g., 'M87', 'NGC 1068')."},
                "ra":          {"type": "number", "description": "RA in degrees (alternative to target)."},
                "dec":         {"type": "number", "description": "Dec in degrees (alternative to target)."},
                "survey":      {"type": "string", "description": "Sky survey: 'DSS2 Red', '2MASS-J', 'WISE 3.4', etc. Default 'DSS2 Red'."},
                "fov_arcmin":  {"type": "number", "description": "Field of view in arcminutes. Default 5."},
                "title":       {"type": "string", "description": "Custom chart title."},
            },
            "required": []
        },
        category="analysis"
    ))

    # ── DataLink + FITS Remote Header Tools (Phase 0) ──────────
    agent.tool_registry.register(Tool(
        name="list_alma_files",
        description=(
            "List all deliverable files (images, cubes, continuum maps) for a "
            "given ALMA MOUS UID via the DataLink protocol. Returns filenames, "
            "sizes, and direct access URLs.  Use after a search to discover "
            "which data products (e.g. *pbcor.fits) are available for download "
            "or remote FITS header inspection."
        ),
        function=agent._list_alma_files,
        parameters={
            "type": "object",
            "properties": {
                "mous_uid": {
                    "type": "string",
                    "description": "MOUS UID, e.g. 'uid://A001/X1590/X30ae'"
                },
                "filename_pattern": {
                    "type": "string",
                    "description": "Optional glob filter, e.g. '*.pbcor.fits' to only show primary-beam-corrected images"
                },
            },
            "required": ["mous_uid"]
        },
        category="archive"
    ))

    agent.tool_registry.register(Tool(
        name="triage_alma_data_products",
        description=(
            "Discover and triage ALMA deliverable data products. Use this when the user asks "
            "to fetch, list, inspect, or triage ALMA FITS/data products. Exact project/proposal "
            "codes, MOUS UIDs, ASDM UIDs, and dataset IDs are routed directly. Generic target "
            "names are treated as ambiguous: the tool returns a project-code picker table and "
            "asks the user to choose before product triage. If a project picker was just shown, "
            "row-number replies such as '#4' or 'use number 4' are resolved to that project. "
            "It lists DataLink products and reads "
            "remote FITS headers only; it does not download large science files."
        ),
        function=agent._triage_alma_data_products,
        parameters={
            "type": "object",
            "properties": {
                "identifier_or_target": {
                    "type": "string",
                    "description": "ALMA project code, MOUS UID, dataset ID, or target name from the user request."
                },
                "band": {
                    "type": "string",
                    "description": "Optional ALMA band preference, e.g. '6' or 'Band 7'."
                },
                "max_projects": {
                    "type": "integer",
                    "description": "Optional max project-code options to show for ambiguous target searches. By default all matched projects are shown."
                },
                "max_mous": {
                    "type": "integer",
                    "description": "Max MOUS datasets to inspect for exact IDs. Default 5."
                },
                "max_products": {
                    "type": "integer",
                    "description": "Max product rows to list. Default 40."
                },
                "max_header_checks": {
                    "type": "integer",
                    "description": "Max FITS products to inspect with remote header reads. Default 6."
                },
            },
            "required": ["identifier_or_target"]
        },
        category="archive"
    ))

    agent.tool_registry.register(Tool(
        name="inspect_fits_header",
        description=(
            "Read key metadata from a remote FITS file header WITHOUT downloading "
            "the full file. Returns beam size (BMAJ/BMIN in arcsec), RMS noise/sensitivity, "
            "rest frequency, target name, pixel scale, and image dimensions. "
            "Use after list_alma_files to inspect the data quality of each product."
        ),
        function=agent._inspect_fits_header,
        parameters={
            "type": "object",
            "properties": {
                "access_url": {
                    "type": "string",
                    "description": "Direct HTTPS URL to the FITS file (from list_alma_files access_url)"
                },
            },
            "required": ["access_url"]
        },
        category="archive"
    ))

    agent.tool_registry.register(Tool(
        name="survey_coverage",
        description="List surveys/datasets whose sky coverage (MOC) includes a position or region; use first for data-existence/coverage questions before broad archive searches, with optional dataproduct_type, regime, or keyword filters.",
        function=agent._survey_coverage,
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "Astronomical target name to resolve, e.g. NGC 253 or 3C 273."},
                "ra": {"type": "number", "description": "Right ascension in decimal degrees (ICRS). Used with dec if target_name is omitted."},
                "dec": {"type": "number", "description": "Declination in decimal degrees (ICRS). Used with ra if target_name is omitted."},
                "radius_deg": {"type": "number", "description": "Cone radius in degrees. Default 0 for a point query.", "default": 0.0},
                "dataproduct_type": {"type": "string", "enum": ["image", "catalog", "cube"], "description": "Optional product type filter."},
                "keyword": {"type": "string", "description": "Optional case-insensitive substring filter on dataset ID and title."},
                "regime": {"type": "string", "enum": ["radio", "mm/sub-mm", "infrared", "optical", "UV", "X-ray", "gamma"], "description": "Optional wavelength-regime filter derived from em_min/em_max."},
                "max_rows": {"type": "integer", "description": "Maximum rows to return, capped at 200.", "default": 50},
            },
            "required": []
        },
        category="archive"
    ))

    agent.tool_registry.register(Tool(
        name="survey_footprint",
        description=(
            "Draw survey footprints (MOC coverage maps) on an interactive sky view: takes "
            "MOCServer dataset ids (the `id` values from survey_coverage rows, e.g. "
            "'CDS/P/SDSS9/color') plus a position anchor, and renders a HiPS card whose "
            "Interactive view overlays each survey's footprint in a distinct color with a "
            "legend. Use after survey_coverage when the user wants to SEE where surveys "
            "overlap, not just a table."
        ),
        function=agent._survey_footprint,
        parameters={
            "type": "object",
            "properties": {
                "survey_ids": {"type": "array", "items": {"type": "string"}, "description": "1-4 MOCServer dataset ids from survey_coverage rows."},
                "target_name": {"type": "string", "description": "Position anchor to center the view on."},
                "ra": {"type": "number"},
                "dec": {"type": "number"},
                "fov_deg": {"type": "number", "default": 20.0, "description": "Field of view of the static preview in degrees."},
                "survey": {"type": "string", "default": "optical", "description": "Base imagery survey alias (optical/2mass/wise/...)."},
                "order": {"type": "integer", "default": 8, "description": "MOC HEALPix order (3-10); higher = finer edges, larger payload."},
            },
            "required": ["survey_ids"],
        },
        category="archive",
    ))

    agent.tool_registry.register(Tool(
        name="survey_covers_position",
        description="Check whether a named survey (e.g. VLASS, SDSS, GLEAM) covers a given position; returns covered true/false and matching dataset IDs.",
        function=agent._survey_covers_position,
        parameters={
            "type": "object",
            "properties": {
                "survey_keyword": {"type": "string", "description": "Survey name or keyword to match in MOCServer dataset IDs/titles."},
                "target_name": {"type": "string", "description": "Astronomical target name to resolve."},
                "ra": {"type": "number", "description": "Right ascension in decimal degrees (ICRS). Used with dec if target_name is omitted."},
                "dec": {"type": "number", "description": "Declination in decimal degrees (ICRS). Used with ra if target_name is omitted."},
            },
            "required": ["survey_keyword"]
        },
        category="archive"
    ))

    agent.tool_registry.register(Tool(
        name="galactic_extinction",
        description=(
            "Galactic dust reddening E(B-V) (SFD98 + Schlafly-Finkbeiner 2011) "
            "and per-band extinction A_lambda at a sky position -- use before any "
            "photometric correction, color, or distance-modulus work."
        ),
        function=agent._galactic_extinction,
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "Astronomical target name to resolve, e.g. M87 or 3C 273."},
                "ra": {"type": "number", "description": "Right ascension in decimal degrees (ICRS). Used with dec if target_name is omitted."},
                "dec": {"type": "number", "description": "Declination in decimal degrees (ICRS). Used with ra if target_name is omitted."},
                "bands": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional extinction bands. Exact keys include V, sdss_r, ps1_g, J, Ks, W1. Omit for all supported bands."
                },
            },
            "required": []
        },
        category="archive"
    ))

    agent.tool_registry.register(Tool(
        name="gaia_distance",
        description="Bailer-Jones (2021) geometric/photogeometric distances for Gaia DR3 sources near a position -- the correct way to turn parallax into distance for stars.",
        function=agent._gaia_distance,
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "Astronomical target name to resolve, e.g. Barnard's Star."},
                "ra": {"type": "number", "description": "Right ascension in decimal degrees (ICRS). Used with dec if target_name is omitted."},
                "dec": {"type": "number", "description": "Declination in decimal degrees (ICRS). Used with ra if target_name is omitted."},
                "radius_arcsec": {"type": "number", "description": "Cone radius in arcseconds, capped at 300. Default 10; named-target default calls expand to 30 for high-proper-motion tolerance.", "default": 10},
                "max_rows": {"type": "integer", "description": "Maximum rows to return, capped at 50.", "default": 10},
            },
            "required": []
        },
        category="archive"
    ))

    agent.tool_registry.register(Tool(
        name="ned_distance",
        description="NED redshift-independent distance measurements (Cepheids, TRGB, SNIa, ...) for a named galaxy, with median summary.",
        function=agent._ned_distance,
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "Named galaxy, e.g. M83 or NGC 253."},
            },
            "required": ["target_name"]
        },
        category="archive"
    ))

    agent.tool_registry.register(Tool(
        name="velocity_frame_distance",
        description="Convert a heliocentric velocity or redshift to GSR, Local Group, and CMB frames and give Hubble-flow distances (Planck18 H0) -- use for nearby-galaxy distances and flow corrections.",
        function=agent._velocity_frame_distance,
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "Astronomical target name to resolve."},
                "ra": {"type": "number", "description": "Right ascension in decimal degrees (ICRS). Used with dec if target_name is omitted."},
                "dec": {"type": "number", "description": "Declination in decimal degrees (ICRS). Used with ra if target_name is omitted."},
                "v_helio_kms": {"type": "number", "description": "Heliocentric velocity in km/s. Provide exactly one of v_helio_kms or z."},
                "z": {"type": "number", "description": "Redshift converted relativistically to heliocentric velocity. Provide exactly one of v_helio_kms or z."},
            },
            "required": []
        },
        category="analysis"
    ))

    agent.tool_registry.register(Tool(
        name="search_pulsars",
        description="Search the ATNF pulsar catalogue around a sky position; returns period, DM, 1400 MHz flux, distance, binarity, associations.",
        function=agent._search_pulsars,
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "Astronomical target name to resolve, e.g. Crab Nebula."},
                "ra": {"type": "number", "description": "Right ascension in decimal degrees (ICRS). Used with dec if target_name is omitted."},
                "dec": {"type": "number", "description": "Declination in decimal degrees (ICRS). Used with ra if target_name is omitted."},
                "radius_deg": {"type": "number", "description": "Cone radius in degrees, capped at 30. Default 1.", "default": 1.0},
                "max_rows": {"type": "integer", "description": "Maximum rows to return, capped at 200. Default 25.", "default": 25},
            },
            "required": []
        },
        category="archive"
    ))

    agent.tool_registry.register(Tool(
        name="pulsar_lookup",
        description="Look up a pulsar by J/B name in the ATNF catalogue and return its full timing/derived parameters.",
        function=agent._pulsar_lookup,
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Pulsar J-name or B-name, with or without the PSR prefix."},
            },
            "required": ["name"]
        },
        category="archive"
    ))
    agent.tool_registry.register(Tool(
        name="solar_system_ephemeris",
        description="JPL Horizons ephemeris for a planet, asteroid, or comet over a date range: RA/Dec, observer/heliocentric distances, magnitude, and elongation.",
        function=agent._solar_system_ephemeris,
        parameters={
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Solar-system body name or designation, e.g. 'Ceres', 'Mars', '2020 SO'."},
                "start": {"type": "string", "description": "Start date/time, ISO, e.g. '2026-07-03'."},
                "stop": {"type": "string", "description": "Stop date/time, ISO, e.g. '2026-07-08'."},
                "step": {"type": "string", "description": "Step size, e.g. '1d', '6h', '30m'.", "default": "1d"},
            },
            "required": ["target", "start", "stop"]
        },
        category="archive"
    ))
    agent.tool_registry.register(Tool(
        name="moving_object_check",
        description="List known asteroids/comets inside a field at a given epoch (IMCCE SkyBoT) — use to check whether a transient or odd detection is a known moving object.",
        function=agent._moving_object_check,
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "Target name to resolve for the field centre."},
                "ra": {"type": "number", "description": "RA in decimal degrees (ICRS); used with dec if target_name is omitted."},
                "dec": {"type": "number", "description": "Dec in decimal degrees (ICRS); used with ra if target_name is omitted."},
                "radius_deg": {"type": "number", "description": "Cone radius in degrees (<=10).", "default": 0.2},
                "epoch": {"type": "string", "description": "Epoch (ISO UTC or JD); defaults to now."},
            },
            "required": []
        },
        category="archive"
    ))
    agent.tool_registry.register(Tool(
        name="radio_sed",
        description=(
            "Compile a compact-source radio continuum SED from TGSS, GLEAM, SUMSS, NVSS, and FIRST catalog fluxes; "
            "fit the spectral index alpha (S~nu^alpha), plot it, and return flags for resolution/epoch caveats."
        ),
        function=agent._radio_sed,
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "Astronomical target name to resolve, e.g. 3C 273 or M87."},
                "ra": {"type": "number", "description": "Right ascension in decimal degrees (ICRS). Used with dec if target_name is omitted."},
                "dec": {"type": "number", "description": "Declination in decimal degrees (ICRS). Used with ra if target_name is omitted."},
                "radius_arcsec": {"type": "number", "description": "Cone-search radius in arcsec, capped at 120. Default 30.", "default": 30.0},
            },
            "required": []
        },
        category="analysis"
    ))
    agent.tool_registry.register(Tool(
        name="monitor_add_target",
        description="Add a sky position to the standing ZTF-alert watchlist; Quasar remembers it across sessions and reports only NEW alerts on each check.",
        function=agent._monitor_add_target,
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Label for the watchlist entry, e.g. 'SN 2026abc field'."},
                "target_name": {"type": "string", "description": "Astronomical name to resolve for the position (defaults to `name`)."},
                "ra": {"type": "number", "description": "RA in decimal degrees (ICRS); used with dec if no name resolves."},
                "dec": {"type": "number", "description": "Dec in decimal degrees (ICRS)."},
                "radius_arcsec": {"type": "number", "description": "Match radius in arcsec, capped at 600.", "default": 120},
                "note": {"type": "string", "description": "Optional free-text note."},
            },
            "required": ["name"]
        },
        category="analysis"
    ))
    agent.tool_registry.register(Tool(
        name="monitor_list_targets",
        description="List the standing sky-monitor watchlist with hit counts and last-checked times.",
        function=agent._monitor_list_targets,
        parameters={"type": "object", "properties": {}, "required": []},
        category="analysis"
    ))
    agent.tool_registry.register(Tool(
        name="monitor_remove_target",
        description="Remove a sky-monitor watchlist entry (and its recorded alerts) by id.",
        function=agent._monitor_remove_target,
        parameters={
            "type": "object",
            "properties": {"target_id": {"type": "integer", "description": "Watchlist entry id from monitor_list_targets."}},
            "required": ["target_id"]
        },
        category="analysis"
    ))
    agent.tool_registry.register(Tool(
        name="monitor_check_now",
        description="Check the watchlist (or one target) against ALeRCE/ZTF NOW and report only alerts that are new since the previous check.",
        function=agent._monitor_check_now,
        parameters={
            "type": "object",
            "properties": {"target_id": {"type": "integer", "description": "Optional: check only this watchlist entry."}},
            "required": []
        },
        category="analysis"
    ))
    agent.tool_registry.register(Tool(
        name="vo_find_services",
        description="Discover Virtual Observatory services (TAP/SIA/SSA/cone) by keyword and waveband — finds archives Quasar has no built-in client for. Follow with vo_list_tables / vo_adql_query on the access_url.",
        function=agent._vo_tool_fn("vo_find_services"),
        parameters={
            "type": "object",
            "properties": {
                "keywords": {"type": "string", "description": "Search keywords, e.g. 'HI 21cm survey' or 'GLEAM'."},
                "service_type": {"type": "string", "enum": ["tap", "sia", "ssa", "scs"], "description": "Optional service type filter."},
                "waveband": {"type": "string", "description": "Optional waveband filter, e.g. 'radio', 'x-ray'."},
                "max_rows": {"type": "integer", "description": "Max services to return, capped at 100.", "default": 30},
            },
            "required": ["keywords"]
        },
        category="archive"
    ))
    agent.tool_registry.register(Tool(
        name="vo_list_tables",
        description="List (and keyword-filter) the tables of any TAP service found via vo_find_services.",
        function=agent._vo_tool_fn("vo_list_tables"),
        parameters={
            "type": "object",
            "properties": {
                "access_url": {"type": "string", "description": "TAP service base URL."},
                "keyword": {"type": "string", "description": "Substring filter on table name/description — strongly recommended for big services like VizieR."},
                "max_tables": {"type": "integer", "description": "Max tables to list, capped at 200.", "default": 50},
            },
            "required": ["access_url"]
        },
        category="archive"
    ))
    agent.tool_registry.register(Tool(
        name="vo_describe_table",
        description="Column schema (names, datatypes, units, UCDs) of a table on any TAP service — call before writing ADQL.",
        function=agent._vo_tool_fn("vo_describe_table"),
        parameters={
            "type": "object",
            "properties": {
                "access_url": {"type": "string", "description": "TAP service base URL."},
                "table_name": {"type": "string", "description": "Exact table name from vo_list_tables."},
            },
            "required": ["access_url", "table_name"]
        },
        category="archive"
    ))
    agent.tool_registry.register(Tool(
        name="vo_adql_query",
        description="Run a guarded SELECT-only ADQL query against any TAP service URL. On ADQL errors the server's message is returned - read it and fix the query.",
        function=agent._vo_tool_fn("vo_adql_query"),
        parameters={
            "type": "object",
            "properties": {
                "access_url": {"type": "string", "description": "TAP service base URL."},
                "adql": {"type": "string", "description": "SELECT-only ADQL. Quote table names containing '/' or '+' in double quotes."},
                "max_rows": {"type": "integer", "description": "Row cap (service cap also applies).", "default": 200},
            },
            "required": ["access_url", "adql"]
        },
        category="archive"
    ))
    agent.tool_registry.register(Tool(
        name="vo_cone_search",
        description="Cone search any VO simple-cone-search service by position.",
        function=agent._vo_tool_fn("vo_cone_search"),
        parameters={
            "type": "object",
            "properties": {
                "access_url": {"type": "string", "description": "SCS service base URL."},
                "target_name": {"type": "string", "description": "Target name to resolve."},
                "ra": {"type": "number", "description": "RA in decimal degrees (ICRS)."},
                "dec": {"type": "number", "description": "Dec in decimal degrees (ICRS)."},
                "radius_deg": {"type": "number", "description": "Cone radius in degrees, capped at 5.", "default": 0.1},
                "max_rows": {"type": "integer", "description": "Row cap.", "default": 100},
            },
            "required": ["access_url"]
        },
        category="archive"
    ))
