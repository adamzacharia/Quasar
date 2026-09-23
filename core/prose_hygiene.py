"""User-facing prose hygiene: keep internal identifiers out of the answer text.

UI benchmark 2026-09-22 (D06, D07, D09, D12, D18 and most DataLab answers):
the model narrates its work with internal tool names (``datalab_density_vetting``,
``query_alma_science_archive``, ``search_by_target``, ``match_cross_archive_sources``)
and pastes raw JSON argument payloads into the prose. Those belong in the
Show-query panels, which already carry the exact request.

:func:`humanize_prose` rewrites a final answer:

* every registered tool name (and the ``name(...)`` / ``name(args)`` call
  forms) becomes plain language -- "the Data Lab density-vetting step",
  "the ALMA science-archive query" -- outside fenced/inline code;
* raw single-line JSON argument dumps (``{"catalog": "gaia_dr3", ...}``) are
  replaced with a pointer to the query panel;
* orphan image placeholders left after URL stripping (``![CMD 1]``) are dropped;
* ``<br>`` tags become line breaks (or a separator inside table rows).

Deterministic, stdlib-only, idempotent. Text inside code fences and inline
code is left untouched: code the user asked for legitimately names tools and
APIs.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

__all__ = [
    "PREFIX_LABELS",
    "SPECIAL_LABELS",
    "find_internal_identifiers",
    "humanize_prose",
    "humanize_tool_name",
    "strip_br_tags",
    "strip_image_placeholders",
]

# Prefix -> service label used when humanising a snake_case tool name.
PREFIX_LABELS: Dict[str, str] = {
    "datalab_": "Data Lab",
    "sparcl_": "SPARCL",
    "hips_": "HiPS",
    "vlass_": "VLASS",
    "ztf_": "ZTF",
    "ned_": "NED",
    "gaia_": "Gaia",
    "vo_": "VO",
    "web_": "web",
    "monitor_": "watchlist",
    "svo_": "SVO",
    "alma_": "ALMA",
    "mast_": "MAST",
    "ads_": "ADS",
}

# Tool names whose word-by-word expansion would read badly.
SPECIAL_LABELS: Dict[str, str] = {
    "query_alma_science_archive": "the ALMA science-archive query",
    "search_by_target": "the ALMA target search",
    "search_by_position": "the ALMA position search",
    "search_by_frequency": "the ALMA frequency search",
    "advanced_search": "the ALMA advanced search",
    "search_alma_with_keywords": "the ALMA keyword search",
    "search_alma_co_in_redshift_range": "the redshifted-CO ALMA search",
    "get_observation_details": "the ALMA observation-details lookup",
    "triage_alma_data_products": "the ALMA data-product triage",
    "list_alma_files": "the ALMA file listing",
    "download_alma_data": "the ALMA download",
    "get_alma_qa2_status": "the ALMA QA2 status lookup",
    "match_cross_archive_sources": "the cross-archive source match",
    "match_perseus_protostars_alma_jwst": "the Perseus protostar ALMA/JWST match",
    "overlay_archive_images": "the archive image overlay",
    "find_alma_line_coverage": "the ALMA line-coverage check",
    "search_mast": "the MAST search",
    "search_mast_by_criteria": "the MAST criteria search",
    "get_mast_products": "the MAST product listing",
    "download_mast_data": "the MAST download",
    "search_papers": "the literature search",
    "search_papers_by_observation_id": "the observation-linked literature search",
    "resolve_target": "the target-name resolution",
    "browse_schema": "the schema lookup",
    "browse_alma_guidance": "the ALMA guidance lookup",
    "datalab_sql_query": "the Data Lab SQL query",
    "datalab_select_catalog_rows": "the Data Lab row selection",
    "datalab_cone_count": "the Data Lab cone count",
    "datalab_q3c_crossmatch": "the Data Lab cross-match",
    "datalab_density_aggregate": "the Data Lab density aggregate",
    "datalab_density_vetting": "the Data Lab density vetting",
    "datalab_color_magnitude_diagram": "the colour-magnitude diagram",
    "datalab_color_color_diagram": "the colour-colour diagram",
    "datalab_sky_density_map": "the sky density map",
    "datalab_catalog_scatter": "the catalogue scatter plot",
    "datalab_image_cutout": "the Data Lab image cutout",
    "datalab_color_image": "the Data Lab colour image",
    "datalab_cutout_grid": "the Data Lab cutout grid",
    "datalab_star_lightcurve": "the Data Lab light curve",
    "datalab_period_fold": "the period fold",
    "datalab_sed_plot": "the SED plot",
    "datalab_lss_wedge": "the large-scale-structure wedge plot",
    "datalab_describe_table": "the Data Lab table description",
    "datalab_list_catalogs": "the Data Lab catalogue listing",
    "datalab_job_status": "the Data Lab job status check",
    "datalab_job_results": "the Data Lab job results",
    "datalab_confirm_sky_area": "the sky-area confirmation",
    "datalab_tiled_search": "the Data Lab tiled search",
    "datalab_variable_candidates": "the Data Lab variability search",
    "datalab_sia_search": "the Data Lab image search",
    "plot_sky_map": "the sky map",
    "plot_alma_results": "the ALMA results plot",
    "plot_spectrum": "the spectrum plot",
    "hips_cutout": "the HiPS cutout",
    "hips_multiband_panel": "the HiPS multi-band panel",
    "hips_contour_overlay": "the HiPS contour overlay",
    "hips_rgb_composite": "the HiPS RGB composite",
    "web_search": "the web search",
    "xmatch_user_list": "the CDS cross-match",
    "cross_match_source": "the multi-archive source cross-match",
    "alma_project_census": "the ALMA project census",
    "alma_source_summary": "the ALMA source summary",
    "alma_public_band_status": "the ALMA public-band status",
    "alma_archive_link": "the ALMA archive link builder",
    "alma_bibliography": "the ALMA bibliography search",
    "cross_archive_match": "the cross-archive match",
    "archive_overlay": "the archive image overlay",
    "code_recipe": "the code recipe",
    "alma_reference": "the ALMA reference lookup",
    "alma_reference_table": "the ALMA reference table",
    "datalab_healpix_density_map": "the HEALPix density map",
    "datalab_stream_selection": "the stream-star selection",
    "datalab_selection_diagram": "the selection diagram",
    "datalab_target_class_summary": "the target-class summary",
    "datalab_satellite_search": "the satellite search",
}

_ACRONYMS = {"sql", "adql", "tap", "sia", "cmd", "sed", "rgb", "hips", "qa2", "mous", "ads", "mast", "ned", "vo",
             "url", "id", "fits", "pv", "moc", "ztf", "gw", "gcn", "jwst", "hst", "alma", "co", "lss", "hr", "svo",
             "irsa", "eso", "cadc", "vlass", "uv", "casa", "sparcl", "q3c", "healpix"}


def humanize_tool_name(name: str) -> str:
    """``datalab_density_vetting`` -> ``the Data Lab density vetting``."""
    n = str(name or "").strip().strip("`")
    if not n:
        return ""
    if n in SPECIAL_LABELS:
        return SPECIAL_LABELS[n]
    label = ""
    rest = n
    for prefix, service in PREFIX_LABELS.items():
        if n.startswith(prefix):
            label, rest = service, n[len(prefix):]
            break
    words = []
    for w in rest.split("_"):
        if not w:
            continue
        words.append(w.upper() if w.lower() in _ACRONYMS else w.replace("-", " "))
    body = " ".join(words)
    return f"the {label} {body} step".replace("  ", " ") if label else f"the {body} step"


_CODE_SPLIT_RE = re.compile(r"(```.*?```|~~~.*?~~~|`[^`\n]*`)", re.S)


_FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_INLINE_CODE_RE = re.compile(r"(?<!`)(`+)(?!`)(?:.+?)(?<!`)\1(?!`)")


def _split_inline(text: str) -> List[Tuple[bool, str]]:
    out: List[Tuple[bool, str]] = []
    pos = 0
    for m in _INLINE_CODE_RE.finditer(text):
        if m.start() > pos:
            out.append((False, text[pos:m.start()]))
        out.append((True, m.group(0)))
        pos = m.end()
    if pos < len(text):
        out.append((False, text[pos:]))
    return out


def _split_code(text: str) -> List[Tuple[bool, str]]:
    """[(is_code, segment)] preserving order and every character. Fenced
    blocks follow CommonMark: ``` or ~~~ of any length >= 3, closed only by
    the same character with at least the opening length (a ```` block may
    contain ```); an unclosed fence runs to the end. Inline spans may use any
    backtick run length."""
    out: List[Tuple[bool, str]] = []
    prose: List[str] = []
    code: List[str] = []
    fence: Optional[str] = None
    for line in str(text or "").splitlines(keepends=True):
        if fence is None:
            m = _FENCE_OPEN_RE.match(line)
            if m:
                if prose:
                    out.extend(_split_inline("".join(prose)))
                    prose = []
                fence = m.group(1)
                code = [line]
                continue
            prose.append(line)
        else:
            code.append(line)
            m = re.match(r"^ {0,3}(`{3,}|~{3,})\s*$", line)
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                out.append((True, "".join(code)))
                code, fence = [], None
    if fence is not None and code:
        out.append((True, "".join(code)))
    if prose:
        out.extend(_split_inline("".join(prose)))
    return [(c, seg) for c, seg in out if seg]


def _identifier_regex(tool_names: Iterable[str]) -> Optional[re.Pattern]:
    names = sorted({str(n).strip() for n in tool_names if n and str(n).strip()}, key=len, reverse=True)
    if not names:
        return None
    alt = "|".join(re.escape(n) for n in names)
    # name, optional trailing "(...)" call with a short argument list
    return re.compile(rf"(?<![\w.])(?:functions\.)?(?P<name>{alt})(?P<call>\((?:[^()\n]|\([^()\n]*\))*\))?(?![\w])")


# Inline JSON argument dumps: a single-line {...} object with quoted keys
# (tool arguments the model echoed into the prose).
_JSON_DUMP_RE = re.compile(r"\{\s*\"[A-Za-z_][\w]*\"\s*:[^{}\n]{0,600}\}")
_JSON_ARG_KEYS = ("catalog", "table", "ra", "dec", "radius", "sql", "query_type", "target", "target_name", "band",
                  "result_id", "tool", "arguments", "columns", "x_expr", "y_expr", "cuts", "value_cuts", "healpix_column",
                  "mode", "step_deg", "radius_deg", "radius_arcmin", "radius_arcsec", "max_results", "archives")
_PANEL_POINTER = "(the exact query is in the Show-query panel)"

_IMAGE_PLACEHOLDER_RE = re.compile(r"!\[([^\]\n]*)\](?!\()")
_BR_RE = re.compile(r"<br\s*/?>", re.I)


def strip_image_placeholders(text: str) -> str:
    """Drop ``![alt]`` tokens left behind once an image URL was stripped
    (L15: ``![CMD 1]`` in table cells). ``![alt](url)`` is handled upstream."""
    return _IMAGE_PLACEHOLDER_RE.sub("", text or "")


def strip_br_tags(text: str) -> str:
    """``<br>`` -> a hard line break outside tables, a separator inside a
    GFM table row (a newline would break the row; D02)."""
    if not text or "<br" not in text.lower():
        return text or ""
    out = []
    for line in text.split("\n"):
        if line.lstrip().startswith("|"):
            out.append(_BR_RE.sub(" · ", line))
        else:
            out.append(_BR_RE.sub("  \n", line))
    return "\n".join(out)


def find_internal_identifiers(text: str, tool_names: Iterable[str]) -> List[str]:
    """Tool identifiers present in the PROSE (code excluded). For the tests."""
    rx = _identifier_regex(tool_names)
    if rx is None:
        return []
    found: List[str] = []
    for is_code, seg in _split_code(text or ""):
        if is_code:
            continue
        found.extend(m.group("name") for m in rx.finditer(seg))
    return found


def humanize_prose(text: str, tool_names: Iterable[str], *, drop_json_dumps: bool = True) -> str:
    """Rewrite ``text`` for the user (see module docstring). Code untouched."""
    if not text:
        return text or ""
    rx = _identifier_regex(tool_names)
    names = {str(n).strip() for n in tool_names if n}
    pieces: List[str] = []
    code_segments: List[str] = []
    for is_code, seg in _split_code(text):
        if is_code:
            # `datalab_sed_plot` / `search_by_target(...)` as an inline code span
            # is a tool NAME, not code the user asked for (live 2026-09-23 L10):
            # humanise it. Fenced blocks and real code spans stay untouched.
            m = re.fullmatch(r"`\s*(?:functions\.)?([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^()\n]*\))?\s*`", seg)
            if m and m.group(1) in names:
                pieces.append(humanize_tool_name(m.group(1)))
                continue
            # Protected: a placeholder through the global clean-ups below, so
            # `<br>` or `Client()` inside code is never rewritten (guard CX-25).
            code_segments.append(seg)
            pieces.append(f"\x00CODE{len(code_segments) - 1}\x00")
            continue
        s = seg
        if rx is not None:
            def _sub(m: "re.Match[str]") -> str:
                label = humanize_tool_name(m.group("name"))
                # "the X step(args)" -> "the X step"; a following "tool"/"call"
                # word would read doubled ("the X step tool"), so swallow it.
                return label

            s = rx.sub(_sub, s)
            s = re.sub(r"\b(the [^.\n]{2,60}?(?:step|query|search|lookup|diagram|map|plot|cutout|match|check|fold|listing|"
                       r"vetting|aggregate|count|selection|description|image|grid|panel|composite|overlay|triage|download|"
                       r"census|summary|status|builder|recipe|table|curve|wedge|resolution))\s+(?:tool|function|call)\b", r"\1", s)
            # Article doubling: "the the X" / "The the X" / "a the X"
            s = re.sub(r"\b(?:the|a|an)\s+(the\s+)", r"\1", s, flags=re.I)
            s = re.sub(r"\bThe the\b", "The", s)
        if drop_json_dumps:
            def _json_sub(m: "re.Match[str]") -> str:
                blob = m.group(0)
                if any(f'"{k}"' in blob for k in _JSON_ARG_KEYS):
                    return _PANEL_POINTER
                return blob

            s = _JSON_DUMP_RE.sub(_json_sub, s)
        s = strip_image_placeholders(s)
        pieces.append(s)
    out = "".join(pieces)
    out = strip_br_tags(out)
    # Collapse doubled pointers / leftover empty call parentheses.
    out = re.sub(rf"(?:{re.escape(_PANEL_POINTER)}\s*){{2,}}", _PANEL_POINTER + " ", out)
    out = re.sub(r"\(\s*\)", "", out)
    return re.sub(r"\x00CODE(\d+)\x00", lambda m: code_segments[int(m.group(1))], out)
