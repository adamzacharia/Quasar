"""Route benchmark-shaped questions to the one-shot tools (capabilities/alma_tools.py).

UI benchmark 2026-09-22: questions that one deterministic tool can answer lost
points in open-ended multi-round loops (D08, D13-D15, D17, D20) or in the
TOP-capped science templates (D09-D12, D18, D19, D21, D22). This module is
pure and regex-only:

* :func:`detect_oneshot_intent` -- URL/link, public-band-status, code-recipe and
  ALMA-bibliography intents that previously had NO routing at all, plus the
  Data Lab one-shot tools (capabilities/datalab_tools.py: satellite search,
  WD selection diagram, stream selection, DESI class summary, HEALPix map);
* :func:`census_directive` / :func:`cross_archive_directive` /
  :func:`overlay_directive` -- the MANDATORY instruction text for the three
  intents the runner already detected, now naming the one-shot tools with
  concrete arguments built from the router's extraction.

The directive is advice injected into round 0 (the tool call is forced); the
model keeps the final say on arguments.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from typing import Any, Dict, List, Optional

__all__ = [
    "census_arguments",
    "census_directive",
    "cross_archive_directive",
    "detect_oneshot_intent",
    "overlay_directive",
]

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], start=1)}

_ALMA_ARCHIVE_RE = re.compile(r"\b(?:alma|asa)\b.{0,40}\b(?:archive|data)\b|\balma\s+science\s+archive\b|\balma\b", re.I)
_URL_RE = re.compile(r"\b(?:urls?|links?|hyperlinks?|deep[- ]links?|web\s+address(?:es)?|permalinks?)\b", re.I)
_PUBLIC_BANDS_RE = re.compile(r"\bbands?\b.{0,80}\bpublic\s+data\b|\bpublic\s+data\b.{0,80}\bbands?\b", re.I | re.S)
_CODE_RE = re.compile(r"\b(?:python|astroquery|pyvo|alminer|tap\b|adql|code|script|programmatic\w*|jupyter|notebook)\b", re.I)
_HOWTO_RE = re.compile(r"\b(?:how\s+(?:do|can|could|would|should)\s+i|how\s+to|is\s+there\s+a\s+way|show\s+me\s+(?:\w+\s+){0,2}ways?|ways?\s+to|example|snippet)\b", re.I)
_PAPERS_RE = re.compile(r"\b(?:papers?|publications?|articles?|literature|bibliograph\w*)\b", re.I)
_USED_ALMA_DATA_RE = re.compile(
    r"\b(?:used|using|use|based\s+on|from)\b.{0,30}\b(?:data|observations?)\b.{0,30}\balma\b|"
    r"\balma\s+science\s+archive\b|\balma\s+(?:archive|data)\b", re.I | re.S)


def _as_of_date(query: str) -> Optional[str]:
    q = query.lower()
    m = re.search(r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d{2})\b", q)
    if m:
        try:
            return _dt.date(int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2))).isoformat()
        except ValueError:
            return None
    m = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", q)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def _target_for_link(query: str) -> Optional[str]:
    m = re.search(r"\b(?:query\s+of|of|for)\s+(?:the\s+source\s+)?([A-Za-z]{1,6}\s?\d{1,5}[A-Za-z0-9+\-]*)\b", query)
    return m.group(1).strip() if m else None


def _bands(query: str) -> List[int]:
    try:
        from core.router import _bands_from_prompt

        return _bands_from_prompt(query.lower())
    except Exception:  # pragma: no cover
        return []


def _topic(query: str) -> Optional[str]:
    m = re.search(r"\b(?:studying|interested\s+in|on\s+the\s+topic\s+of|about)\s+([a-z][a-z0-9 \-]{3,60}?)(?:[.,;]|\s+(?:that|which|using|in\s+the|please)\b|$)", query, re.I)
    if not m:
        return None
    topic = re.sub(r"^(?:studying|researching|working\s+on|the\s+topic\s+of)\s+", "", m.group(1).strip(), flags=re.I)
    return topic or None


def _n_items(query: str) -> Optional[int]:
    m = re.search(r"\b(\d{1,2})\s+(?:most\s+recent\s+|latest\s+|newest\s+)?(?:publications|papers|articles)\b", query, re.I)
    return int(m.group(1)) if m else None


def _code_libraries(query: str) -> List[str]:
    q = query.lower()
    libs: List[str] = []
    if "alminer" in q:
        libs.append("alminer")
    if re.search(r"\bpyvo\b", q) or (re.search(r"\btap\b|\badql\b", q) and "astroquery" not in q):
        libs.append("pyvo")
    if "astroquery" in q or (not libs and re.search(r"\bpython\b|\bcode\b|\bscript\b", q)):
        libs.append("astroquery")
    return libs or ["astroquery"]


def _code_task(query: str) -> str:
    q = query.lower()
    if re.search(r"datalink|download|files?\b|products?", q):
        return "list and download DataLink files"
    if re.search(r"\badql\b|\btap\b|custom query", q):
        return "ADQL via TAP"
    if re.search(r"\bcone\b|coordinates?|position|radius", q):
        return "cone search"
    return "query by object name"


def detect_oneshot_intent(query: str) -> Optional[Dict[str, Any]]:
    """The one-shot tool a question maps to, with suggested arguments and the
    MANDATORY directive text; None when no one-shot tool applies."""
    q = str(query or "")
    if not q.strip():
        return None
    alma = bool(_ALMA_ARCHIVE_RE.search(q))

    # D20: publications that used ALMA archive data -> ADS bibgroup:ALMA
    if alma and _PAPERS_RE.search(q) and _USED_ALMA_DATA_RE.search(q):
        args: Dict[str, Any] = {"n": _n_items(q) or 10}
        topic = _topic(q)
        if topic:
            args["topic"] = topic
        return {
            "tool": "alma_bibliography", "args": args,
            "directive": (
                "\n\nMANDATORY INSTRUCTION: The user wants publications that USED ALMA archive data. You MUST call "
                f"`alma_bibliography` now (suggested arguments: {json.dumps(args)}; set `topic` to the research subject). "
                "It searches the ADS ALMA bibliographic group (papers the ALMA telbib tagged as using ALMA data). Present the "
                "returned table (title, first author, year, journal, bibcode, ALMA project codes) and summarise from the returned "
                "abstract summaries only. Do NOT call search_papers or web_search for this."
            ),
        }

    # D17: a URL / link into the ALMA archive
    if alma and _URL_RE.search(q) and re.search(r"\b(?:archive|asa|query|search)\b", q, re.I):
        args = {}
        target = _target_for_link(q)
        if target:
            args["target"] = target
        bands = _bands(q)
        if bands:
            args["band"] = bands
        return {
            "tool": "alma_archive_link", "args": args,
            "directive": (
                "\n\nMANDATORY INSTRUCTION: The user asked for a URL into the ALMA Science Archive. You MUST call "
                f"`alma_archive_link` now (suggested arguments: {json.dumps(args)}). Give the returned `asa_url` and "
                "`tap_sync_url` VERBATIM with their verified/unverified labels. Never compose an archive URL yourself."
            ),
        }

    # D08: how many bands have PUBLIC data (as of a date) -> live archive state
    if alma and _PUBLIC_BANDS_RE.search(q):
        args = {}
        as_of = _as_of_date(q)
        if as_of:
            args["as_of"] = as_of
        return {
            "tool": "alma_public_band_status", "args": args,
            "directive": (
                "\n\nMANDATORY INSTRUCTION: The question asks about the CURRENT state of the ALMA archive. You MUST call "
                f"`alma_public_band_status` now (suggested arguments: {json.dumps(args)}). Report the number of receiver bands "
                "from its `reference` table and the number of bands WITH PUBLIC DATA from the live query (with the as-of date "
                "and query date). Never answer archive holdings from memory."
            ),
        }

    # D13-D15: code for querying the ALMA archive
    if alma and _CODE_RE.search(q) and (_HOWTO_RE.search(q) or re.search(r"\bthrough\s+python\b|\busing\s+(?:tap|alminer|astroquery|pyvo)\b", q, re.I)):
        libs = _code_libraries(q)
        task = _code_task(q)
        calls = [{"task": task, "library": lib} for lib in libs]
        return {
            "tool": "code_recipe", "args": calls[0], "calls": calls,
            "directive": (
                "\n\nMANDATORY INSTRUCTION: The user wants code to query the ALMA Science Archive. You MUST call `code_recipe` "
                f"now -- once per library: {json.dumps(calls)}. Present each returned snippet VERBATIM in a ```python block "
                "and explain it briefly; use ONLY the function names in the returned code (they are tested). Cite the "
                "returned documentation URL. Do NOT invent API calls such as Alma.get_data_links or Alma.query_sql."
            ),
        }
    if not alma:
        return _datalab_intent(q)
    return None


# ── Data Lab one-shot intents (DataLabBench L06-L09, L11, L15) ───────────

_NUM = r"([+\-\u2212]?\d+(?:\.\d+)?)"
_RA_RE = re.compile(r"\bRA\s*[=:~]?\s*" + _NUM + r"\s*(?:°|deg\w*)?", re.I)
_DEC_RE = re.compile(r"\bDec(?:l(?:ination)?)?\s*[=:~]?\s*" + _NUM + r"\s*(?:°|deg\w*)?", re.I)
_RADIUS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:°|deg(?:rees?)?)[\s-]*(?:radius|cone|circle)|radius\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(?:°|deg)", re.I)
_SMASH_FIELD_RE = re.compile(r"\bsmash\b.{0,40}?\bfield\s*(?:id\s*)?#?\s*(\d{1,4})\b", re.I | re.S)
_WD_RE = re.compile(r"\bwhite[\s-]*dwarfs?\b|\bWDs?\b", re.I)
_HR_RE = re.compile(r"\b(?:HR|hertzsprung|colou?r[\s-]*magnitude|CMD)\b|\babsolute\s+magnitudes?\b", re.I)
_OVERDENSITY_RE = re.compile(r"\b(?:over[\s-]?densit\w*|dwarf\s+(?:companion|galax\w*|satellite)s?|satellites?|clump\w*|ultra[\s-]*faint)\b", re.I)
_DISCOVER_RE = re.compile(r"\b(?:discover\w*|search\w*|find|look\s+for|hunt\w*|candidates?|new)\b", re.I)
_DEEP_IMAGING_RE = re.compile(r"\b(?:data\s?lab|deep\s+imaging|delve|des\b|decam|nsc|noirlab\s+source\s+catalog|smash|legacy\s+survey)", re.I)
_DENSITY_MAP_RE = re.compile(r"\b(?:stellar\s+)?density\s+map\b|\bstar[\s-]*counts?\s+map\b|\bhealpix\b", re.I)
_STREAM_RE = re.compile(r"\btidal\s+tails?\b|\bstellar\s+stream\b|\bstream\s+(?:stars|members)\b|\btrace\s+(?:its|the)\s+(?:tails?|stream)\b", re.I)
_CLUSTER_NAME_RE = re.compile(r"\b(Palomar\s*\d+|Pal\s*\d+|NGC\s*\d{3,4}|M\s*\d{1,3}|GD[\s-]*1|Omega\s+Cen\w*)\b", re.I)
_DESI_CLASS_RE = re.compile(r"\b(LRGs?|luminous\s+red\s+galax\w*|ELGs?|emission[\s-]*line\s+galax\w*|QSOs?|quasars?|BGS|bright\s+galaxy\s+survey)\b", re.I)
_Z_RANGE_RE = re.compile(r"\bz\s*(?:=|between|from)?\s*" + _NUM + r"\s*(?:and|to|-|–|<\s*z\s*<)\s*" + _NUM, re.I)
_NAMED_REGION_RE = re.compile(r"\b(LMC|large\s+magellanic|SMC|small\s+magellanic|magellanic\s+bridge|sagittarius|sgr\b|anti[\s-]*cent(?:er|re)|north\s+galactic\s+pole|NGP)\b", re.I)


def _num(tok: str) -> float:
    return float(str(tok).replace("\u2212", "-"))


def _coords(q: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    ra, dec = _RA_RE.search(q), _DEC_RE.search(q)
    if ra and dec:
        out["ra"], out["dec"] = _num(ra.group(1)), _num(dec.group(1))
    rad = _RADIUS_RE.search(q)
    if rad:
        out["radius_deg"] = float(rad.group(1) or rad.group(2))
    return out


def _named_region_preset(q: str) -> Optional[str]:
    m = _NAMED_REGION_RE.search(q)
    if not m:
        return None
    t = m.group(1).lower()
    if "bridge" in t:
        return "magellanic_bridge"
    if t.startswith("lmc") or "large" in t:
        return "lmc"
    if t.startswith("smc") or "small" in t:
        return "smc"
    if t.startswith("sag") or t.startswith("sgr"):
        return "sgr_stream"
    if "anti" in t:
        return "anticenter"
    return "ngp"


def _desi_class(tok: str) -> str:
    t = tok.lower()
    if t.startswith("lrg") or "luminous" in t:
        return "LRG"
    if t.startswith("elg") or "emission" in t:
        return "ELG"
    if t.startswith("qso") or t.startswith("quasar"):
        return "QSO"
    return "BGS_ANY"


def _dl(tool: str, args: Dict[str, Any], body: str) -> Dict[str, Any]:
    return {
        "tool": tool, "args": args,
        "directive": (f"\n\nMANDATORY INSTRUCTION: You MUST call `{tool}` now (suggested arguments: {json.dumps(args)}). "
                      + body + " The figures it produces are already displayed ABOVE your answer -- describe them, do not "
                      "re-create them with other plotting tools, and quote counts only from the tool result."),
    }


def _datalab_intent(q: str) -> Optional[Dict[str, Any]]:
    # L07: an overdensity search inside a named SMASH field
    m = _SMASH_FIELD_RE.search(q)
    if m and (_OVERDENSITY_RE.search(q) or re.search(r"\bblue\b.{0,30}\bstars?\b", q, re.I)):
        return _dl("datalab_satellite_search", {"smash_field": int(m.group(1))},
                   "It scans exactly that SMASH field (fieldid), selects blue main-sequence point sources, builds the density "
                   "grid, ranks peaks by significance over the local background, screens known satellites, and attaches "
                   "cutouts and CMDs. Report the ranked peaks with their significance and whether a known object was "
                   "re-detected; state the selection cuts from `cuts_applied`.")
    # L15: an open-ended satellite discovery
    if (re.search(r"\b(?:satellite|dwarf)[\s-]*(?:dwarf[\s-]*)?galax\w*|\bmilky\s+way\s+satellites?\b|\bultra[\s-]*faint\b", q, re.I)
            and _DISCOVER_RE.search(q) and _DEEP_IMAGING_RE.search(q)):
        args: Dict[str, Any] = {"survey": "delve", "preset": "delve_south"}
        c = _coords(q)
        if {"ra", "dec"} <= set(c):
            args = {"survey": "delve", "region": {"ra": c["ra"], "dec": c["dec"], "radius": c.get("radius_deg", 3.0)}}
        return _dl("datalab_satellite_search", args,
                   "Before the results, STATE THE STRATEGY in 3-5 steps (point-source + old/metal-poor colour-magnitude "
                   "selection -> server-side density grid -> peak significance over the local background -> known-object "
                   "screening -> CMD + cutout vetting) and the search region with the reason it was chosen. Then give the "
                   "ranked candidate table (position, significance, verdict) from the tool result.")
    # L06: white-dwarf candidates with an HR diagram (Gaia)
    if _WD_RE.search(q) and (_HR_RE.search(q) or re.search(r"\bproper[\s-]*motion\b", q, re.I)) and re.search(r"\bgaia\b", q, re.I):
        args = {"catalog": "gaia_dr3", "x_expr": "bp_rp", "y_expr": "phot_g_mean_mag", "abs_mag_from_parallax": True,
                "quality_preset": "gaia_high_pm", "overlay_locus": "white_dwarf"}
        args.update(_coords(q))
        if re.search(r"\b(?:high|large|significant)[\s-]*(?:total\s+)?proper[\s-]*motion\b", q, re.I):
            args["pm_total_min_mas_yr"] = 50.0
        return _dl("datalab_selection_diagram", args,
                   "It applies the parallax-significance/RUWE cuts, the total-proper-motion cut and M_G = G + 5 log10(parallax/100) "
                   "server-side, tiles wide cones, and draws the HR diagram with the white-dwarf locus. Report how many "
                   "sources passed and how many fall on the WD sequence, with the exact cuts from the result.")
    # L09: tidal tails / stream around a named cluster
    if _STREAM_RE.search(q) and re.search(r"\bproper[\s-]*motions?\b|\bgaia\b", q, re.I):
        args = {}
        cm = _CLUSTER_NAME_RE.search(q)
        if cm:
            args["cluster_name"] = re.sub(r"\s+", " ", cm.group(1)).strip()
        c = _coords(q)
        if {"ra", "dec"} <= set(c):
            args.update({"ra": c["ra"], "dec": c["dec"]})
        return _dl("datalab_stream_selection", args,
                   "It crossmatches NSC DR2 with Gaia DR3 proper motions, derives the proper-motion window and CMD mask from "
                   "the cluster core (unless given), and plots the selected stream stars on the sky. Report the windows used "
                   "and the member count, and say whether tails are visible.")
    # L11: DESI target-class redshift distribution + footprint
    dm = _DESI_CLASS_RE.search(q)
    if dm and re.search(r"\bdesi\b", q, re.I) and re.search(r"\bredshift\s+distribution\b|\bfootprint\b|\bn\(z\)", q, re.I):
        args = {"catalog": "desi_dr1", "target_class": _desi_class(dm.group(1))}
        zr = _Z_RANGE_RE.search(q)
        if zr:
            args["z_range"] = [_num(zr.group(1)), _num(zr.group(2))]
        return _dl("datalab_target_class_summary", args,
                   "It selects the class by its targeting bitmask (not SPECTYPE) with good redshifts in the range and returns "
                   "the redshift histogram and the sky-footprint map in one figure. Report the total count and the bitmask "
                   "predicate used.")
    # L08: a large-area stellar density map
    if _DENSITY_MAP_RE.search(q) and re.search(r"\b(?:stellar|stars?|milky\s+way|nsc|healpix)\b", q, re.I) and not re.search(r"\bgalax(?:y|ies)\s+density\b", q, re.I):
        args = {"catalog": "nsc_dr2"}
        preset = _named_region_preset(q)
        c = _coords(q)
        if {"ra", "dec"} <= set(c):
            args["region"] = {"ra": c["ra"], "dec": c["dec"], "radius": c.get("radius_deg", 10.0)}
            region_note = "Use the region the user gave."
        elif preset:
            args["preset"] = preset
            region_note = f"Use the `{preset}` preset the user's wording names."
        else:
            args["preset"] = "lmc"
            region_note = ("The user did not name a region: use the `lmc` preset (a 10 deg cone on the Large Magellanic Cloud -- "
                           "the canonical Milky Way/Magellanic structure example) and SAY that you chose it and why, offering "
                           "other presets (anticenter, sgr_stream, ngp). Do NOT ask a clarifying question first.")
        return _dl("datalab_healpix_density_map", args,
                   region_note + " It bins point sources by HEALPix server-side (tiled) and shows log counts.")
    return None


# ── directives for the intents the runner already detects ────────────────

def census_arguments(route: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """(tool, arguments) for a query_alma_science_archive route."""
    if not route:
        return None
    qt = route.get("query_type")
    if qt == "cycle_solar_projects":
        return {"tool": "alma_project_census", "args": {"cycle": route.get("cycle"), "constraint": {"type": "solar"}}}
    if qt == "cycle_array_combo_projects":
        return {"tool": "alma_project_census", "args": {"cycle": route.get("cycle"), "constraint": {"type": "arrays", "arrays": route.get("arrays") or ["12m", "7m", "TP"]}}}
    if qt == "line_set_projects":
        c: Dict[str, Any] = {"type": "lines", "lines": route.get("lines") or []}
        if route.get("band"):
            c["band"] = route["band"]
        return {"tool": "alma_project_census", "args": {"constraint": c}}
    if qt == "redshifted_line_projects":
        return {"tool": "alma_project_census", "args": {"constraint": {"type": "redshift", "zmin": route.get("redshift_min"), "zmax": route.get("redshift_max"), "species": route.get("rest_species") or "CO"}}}
    if qt == "bandwidth_switching_candidates":
        args: Dict[str, Any] = {"constraint": {"type": "bandwidth_switching"}}
        if route.get("cycle") is not None:
            args["cycle"] = route["cycle"]
        return {"tool": "alma_project_census", "args": args}
    if qt == "high_resolution_band_data":
        args = {"bands": route.get("band") or []}
        if route.get("max_resolution_arcsec") is not None:
            args["max_resolution_arcsec"] = route["max_resolution_arcsec"]
        if route.get("public_only"):
            args["public_only"] = True
        return {"tool": "alma_source_summary", "args": args}
    return None


def census_directive(route: Optional[Dict[str, Any]]) -> str:
    plan = census_arguments(route)
    if plan is None:
        return (
            "\n\nMANDATORY INSTRUCTION: The user is asking a live ALMA Science Archive count/filter question. Call the one-shot "
            "tool that fits: `alma_project_census` (constraint.type solar | arrays | lines | redshift | bandwidth_switching; "
            "counts computed at the archive server) or `alma_source_summary` (a target's data with band/resolution filters). "
            "Quote the tool's `headline` count and its completeness statement. Do NOT answer from memory."
        )
    extra = ""
    if plan["tool"] == "alma_source_summary":
        extra = " Also pass the user's target name as `target` (or ra/dec)."
    return (
        "\n\nMANDATORY INSTRUCTION: The user is asking a live ALMA Science Archive question that one tool answers at the "
        f"archive server. You MUST call `{plan['tool']}` now with these arguments: {json.dumps(plan['args'])}.{extra} "
        "Report the tool's `headline` count (for arrays: the number of projects using ALL arrays, never the cycle-window "
        "count), its completeness statement ('complete' or 'partial: ... not scanned'), and the per-project table from the "
        "data card. Science scans only: never present a calibrator (e.g. a quasar phase calibrator) as the observed target. "
        "For redshift questions the z range is coverage-compatible, NOT a measured redshift. Do NOT answer from memory."
    )


def cross_archive_directive(route: Optional[Dict[str, Any]]) -> str:
    route_text = json.dumps(route) if route else '{"catalog_name":"perseus_protostars","archives":["ALMA","JWST"],"radius_arcsec":5,"require_all_archives":true}'
    return (
        "\n\nMANDATORY INSTRUCTION: The user is asking for cross-archive source locations. You MUST call `cross_archive_match` "
        f"now with these arguments: {route_text}. It queries ALMA with one point cone per source and MAST per source, and "
        "returns a per-archive status column. Report its `headline`; if `alma_status_unknown_for` > 0, say plainly that the "
        "ALMA status of those sources is UNKNOWN (the phase timed out) -- never that they lack ALMA data."
    )


def overlay_directive(query: str) -> str:
    q = str(query or "")
    field = "Hubble Ultra Deep Field" if re.search(r"hubble\s+ultra\s+deep\s+field|\bhudf\b|ultra\s+deep\s+field", q, re.I) else None
    base = "HST" if re.search(r"\b(?:hst|hubble\s+space\s+telescope)\b", q, re.I) and not re.search(r"\bjwst\b", q, re.I) else "JWST"
    args: Dict[str, Any] = {"base": {"mission": base}, "contour": {"mission": "ALMA"}}
    if field:
        args["field"] = field
    return (
        "\n\nMANDATORY INSTRUCTION: The user wants ALMA contours over a JWST/HST colourscale image. You MUST call "
        f"`archive_overlay` now (arguments: {json.dumps(args)}; set `field` to the named region or pass ra/dec). If it returns "
        "status='coverage_gap', do NOT search further: tell the user which side is missing and offer the alternatives it lists."
    )
