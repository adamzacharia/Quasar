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
                      + body + " The figures it produces are already shown as cards (call them 'the card', not above/below) -- describe them, do not "
                      "re-create them with other plotting tools, and quote counts only from the tool result."),
    }


_SATELLITE_FIELD_PRESETS = (
    (re.compile(r"\bhydra\s*(?:ii|2)\b", re.I), "hydra2"),
    (re.compile(r"\bfornax\s+(?:dwarf|dsph|field)\b", re.I), "fornax_field"),
)


def _satellite_field_preset(q: str) -> Optional[str]:
    """Satellite-search preset for a named known-satellite field, or None."""
    for rx, preset in _SATELLITE_FIELD_PRESETS:
        if rx.search(q):
            return preset
    return None


def _datalab_intent(q: str) -> Optional[Dict[str, Any]]:
    # L01: which catalogs cover a region AND which have some photometry: one
    # UNFILTERED listing (UI 2026-09-24 run 2: a band filter on the first call
    # dropped every covering catalog without that band)
    cov = re.search(r"\bwhich\b.{0,40}\bcatalog\w*\b.{0,40}\bcover\w*\s+(?:the\s+)?(.+?)(?:,|\?|\band\b)", q, re.I | re.S)
    if cov and re.search(r"\bwhich\s+of\s+(?:those|these|them)\b|\balso\b|\band\s+which\b", q, re.I):
        target = cov.group(1).strip()
        target = {"large magellanic cloud": "LMC", "small magellanic cloud": "SMC"}.get(target.lower(), target)
        return _dl("datalab_list_catalogs", {"target": target},
                   "Call it ONCE with the target and NO band filter: its `coverage_summary` lists every covering catalog "
                   "(`covered`, with `covered_footprints`) and the coverage-unverified ones, plus `by_wavelength_regime`. "
                   "Answer BOTH parts: first all catalogs that cover the region with their table names and why they cover "
                   "it, then the subset with the requested photometry (from the catalogs' bands).")
    # L07: an overdensity search inside a named SMASH field
    m = _SMASH_FIELD_RE.search(q)
    if m and (_OVERDENSITY_RE.search(q) or re.search(r"\bblue\b.{0,30}\bstars?\b", q, re.I)):
        return _dl("datalab_satellite_search", {"smash_field": int(m.group(1))},
                   "It scans exactly that SMASH field (fieldid), selects blue main-sequence point sources, builds the density "
                   "grid, ranks peaks by significance over the local background, screens known satellites, and attaches "
                   "cutouts and CMDs. Report the ranked peaks with their significance, the stars in the aperture vs the expected "
                   "background, and whether a known object was re-detected; state the selection cuts from `cuts_applied` and the "
                   "method from `method` (cell size in arcmin, aperture radius, background annulus, significance formula). For "
                   "each CMD card, quote its `population_test` verdict.")
    # L12: densest clumps near a known satellite field, then cutouts to vet.
    # Raw-count peaks ranked Hydra II fifth behind field clumps (09-23); the
    # satellite search ranks by significance over the local background.
    sat_preset = _satellite_field_preset(q)
    if (sat_preset and re.search(r"\b(?:dens\w*|over[\s-]?densit\w*|clumps?|peaks?|concentrations?)\b", q, re.I)
            and re.search(r"\bcut[\s-]*outs?\b|\bimages?\b|\beyeball\b|\bvet\w*\b", q, re.I)):
        survey = "delve" if re.search(r"\bdelve\b", q, re.I) else ("des" if re.search(r"\bdes\b(?:\s*dr\d)?", q, re.I) else "nsc")
        sat_args: Dict[str, Any] = {"survey": survey, "preset": sat_preset}
        rm = (re.search(r"\bwithin\s+(?:a\s+)?(\d+(?:\.\d+)?)\s*(?:°|deg(?:rees?)?\b)", q, re.I)
              or re.search(r"(\d+(?:\.\d+)?)\s*(?:°|deg(?:rees?)?\b)[\s-]*(?:radius|cone|circle)\b", q, re.I)
              or re.search(r"\bradius\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:°|deg(?:rees?)?\b)", q, re.I))
        if rm:
            sat_args["radius_deg"] = float(rm.group(1))  # the user's radius wins over the preset's
        return _dl("datalab_satellite_search", sat_args,
                   "It ranks density peaks by SIGNIFICANCE over a local annulus background (raw counts favour field clumps), "
                   "matches known objects (the named dwarf itself is a re-detection, not a new clump), screens large-galaxy / "
                   "bright-star artefacts, and attaches the cutout grid of the top peaks plus CMDs. Report the ranked peaks "
                   "(position, significance, stars in the aperture vs the expected background from `aperture_count` / "
                   "`background_in_aperture`, verdict), say which peak is the named dwarf and quote its CMD `population_test` "
                   "verdict, and tell the user what to look for in the cutouts (a faint, compact, resolved stellar concentration "
                   "vs. a galaxy or a bright-star halo). Only repeat verdicts the tool returned.")
    # L13: an LSS wedge / cosmic-web slice (SDSS Great Wall)
    if (re.search(r"\bwedge\b|\bpie[\s-]*(?:slice|diagram|plot)\b|\bcone\s*/\s*wedge\b|\bcosmic\s+web\b|\bgreat\s+wall\b", q, re.I)
            and re.search(r"\bsdss\b|\bboss\b|\bgalax(?:y|ies)\b|\bredshift\b", q, re.I)
            # the wedge tool selects SDSS DR17: another named survey is not hijacked
            and not re.search(r"\b(?:desi|2df\w*|6df\w*|gama|lamost|vipers|deep2|zcosmos|hectomap|2mrs)\b", q, re.I)):
        args = {}
        zr = _Z_RANGE_RE.search(q)
        if zr:
            args.update({"z_min": _num(zr.group(1)), "z_max": _num(zr.group(2))})
        return _dl("datalab_lss_wedge", args,
                   "Call it WITHOUT a result_id: it selects SDSS DR17 galaxies (class = 'GALAXY', zwarning = 0) in a THIN slice "
                   "-- by default the equatorial stripe RA 150-220, Dec -1.25..+1.25, z <= 0.1, where Gott et al. (2005) "
                   "identified the SDSS Great Wall -- thins it uniformly to the row cap, converts z to comoving distance "
                   "(Planck18) and draws the pie-slice wedge. Report the slice, the galaxy count, the cosmology, and describe "
                   "the walls, filaments and voids that are visible (name the Great Wall only if the plot shows it). Quote "
                   "the distance and redshift of the densest structure from `radial_density` (most_overdense_shell), never "
                   "an estimate read off the plot.")
    # L14: one variable star -> SMASH epochs, period fold, a DECam field cutout
    if (re.search(r"\bmulti[\s-]*epoch\b|\blight[\s-]*curves?\b", q, re.I)
            and re.search(r"\bphase[\s-]*fold\w*|\bperiod\b", q, re.I)
            and re.search(r"\bsmash\b", q, re.I) and re.search(r"\bcut[\s-]*outs?\b|\bimages?\b", q, re.I)):
        c = _coords(q)
        star = {"ra": c["ra"], "dec": c["dec"]} if {"ra", "dec"} <= set(c) else {}
        lc_args = {"catalog": "smash_dr1", **star}
        cut_args = {**star, "fov_deg": 0.05, "band": "g", "catalog": "nsc_dr2"}
        return {
            "tool": "datalab_star_lightcurve", "args": lc_args,
            "directive": (
                f"\n\nMANDATORY INSTRUCTION: Run these three Data Lab tools in order. (1) `datalab_star_lightcurve` "
                f"{json.dumps(lc_args)} for the multi-epoch SMASH photometry. (2) `datalab_period_fold` on its result_id "
                "with band='g', min_frequency=1, max_frequency=10 (periods 0.1-1 d). (3) `datalab_image_cutout` "
                f"{json.dumps(cut_args)}: a DECam g-band cutout from Data Lab SIA centred on the star, with the star "
                "marked by a reticle (not DSS or another survey). Report the number of epochs used, the period with its "
                "false-alarm probability, the light-curve shape from the fold's `shape` (e.g. sawtooth: fast rise, slow "
                "decline), and say the cutout is centred on the star (marked). The figures are already shown as cards "
                "(call them 'the card'); quote numbers only from the tool results."
            ),
        }
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
                   "ranked candidate table (position, significance, verdict) from the tool result. A verdict 'likely artefact' "
                   "(large galaxy / bright-star mask) is a screened false positive: say so. Call the tool ONCE; describe only "
                   "the cards it returned -- `vetting_summary` says which ranks have a cutout / CMD. For each CMD card, quote "
                   "its `population_test`: the verdict (coherent old population / field-like / inconclusive), the old-population "
                   "excess and significance, and which features (turnoff, RGB, BHB) are in excess. Report the `background_job` "
                   "(job id, status history, and which source the density result came from), or say plainly that none ran.")
    # L10: SEDs of a galaxy sample from Legacy Surveys grz + forced WISE W1/W2
    if (re.search(r"\bSEDs?\b|\bspectral\s+energy\s+distributions?\b", q, re.I) and re.search(r"\bgalax", q, re.I)
            and re.search(r"\blegacy\s+surveys?\b|\bls\s*dr\s*9\b|\bdecals\b|\bW1\b|\bunwise\b|\bwise\b", q, re.I)):
        args = {k: v for k, v in _coords(q).items() if k in ("ra", "dec", "radius_deg")}
        wm = re.search(r"\bwithin\s+(?:a\s+)?(\d+(?:\.\d+)?)\s*(?:°|deg(?:rees?)?\b)", q, re.I)
        if wm and "radius_deg" not in args:
            args["radius_deg"] = float(wm.group(1))
        if not {"ra", "dec"} <= set(args):
            tm = re.search(r"\b(?:around|near|of|in)\s+(?:the\s+)?([A-Z][\w-]*(?:\s+[A-Z0-9][\w-]*)*\s+cluster)\b", q)
            if tm:
                args["target_name"] = tm.group(1)
        if not re.search(r"\bred\b", q, re.I):
            args.update({"gr_min": None, "rz_min": None})
        nm = re.search(r"\b(\d{2,4})\s+(?:red\s+)?galax", q, re.I)
        if nm:
            args["limit"] = max(10, min(1000, int(nm.group(1))))
        return _dl("datalab_sed_sample", args,
                   "It selects the sample from ONE table, ls_dr9.tractor (dereddened g/r/z and forced unWISE W1/W2 together; "
                   "no WISE catalogue join): cone, extended sources (type != 'PSF'), S/N floors, the red colour cut and a "
                   "LIMIT of a few hundred, all in the SQL, then draws magnitude vs wavelength with SVO effective wavelengths "
                   "(log wavelength, brighter up, per-object curves and the median). Report the sample size and every cut "
                   "from `cuts_applied`.")
    # L06: white-dwarf candidates with an HR diagram (Gaia)
    if _WD_RE.search(q) and (_HR_RE.search(q) or re.search(r"\bproper[\s-]*motion\b", q, re.I)) and re.search(r"\bgaia\b", q, re.I):
        args = {"catalog": "gaia_dr3", "x_expr": "bp_rp", "y_expr": "phot_g_mean_mag", "abs_mag_from_parallax": True,
                "quality_preset": "gaia_high_pm", "overlay_locus": "white_dwarf"}
        args.update(_coords(q))
        if re.search(r"\b(?:high|large|significant)[\s-]*(?:total\s+)?proper[\s-]*motion\b", q, re.I):
            # ~>100 mas/yr is the usual high-PM white-dwarf floor (DLB-06 C5).
            args["pm_total_min_mas_yr"] = 100.0
        # White dwarfs sit at absolute G > ~10: an explicit cut in the SQL.
        args["abs_mag_min"] = 10.0
        return _dl("datalab_selection_diagram", args,
                   "It applies the parallax-significance/RUWE cuts, the total-proper-motion cut, M_G = G + 5 log10(parallax) - 10 "
                   "and the M_G > 10 white-dwarf floor server-side, tiles wide cones, and draws the HR diagram with the "
                   "white-dwarf locus. Report how many sources passed and how many fall on the WD sequence, with the exact "
                   "cuts from the result.")
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
                   "the cluster core (unless given), and plots the selected stream stars on the sky plus a smoothed density "
                   "map. Report the windows used and the member count; give the `tail_test` result WITH its numbers (best "
                   "position angle, stars in that strip vs the perpendicular strip, significance) and say plainly whether "
                   "tails were recovered; explain the query design from `execution_model` (small side materialized, NSC "
                   "on the indexed side, match radius).")
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
    # L03: a g vs (g-r) CMD of a cone from a named survey -> the CMD tool
    # (UI 2026-09-24: the model picked the expression diagram, which dropped
    # point_sources and has no depth / features blocks)
    if (re.search(r"\bCMD\b|\bcolou?r[\s-]*magnitude\s+diagram\b", q, re.I)
            and re.search(r"\bg\s*vs\.?\s*\(?\s*g\s*[−\-]\s*r|\bg\s+and\s+r\b|\bg\s*[−\-]\s*r\b", q, re.I)
            and not re.search(r"\bproper[\s-]*motions?\b|\btidal\b|\bstream\b|\bsatellites?\b|\bwhite[\s-]*dwarfs?\b|\bgaia\b|\bcandidates?\b", q, re.I)):
        c = _coords(q)
        if {"ra", "dec"} <= set(c):
            cat = ("des_dr1" if re.search(r"\bdes\b", q, re.I) else "delve_dr3" if re.search(r"\bdelve\b", q, re.I)
                   else "smash_dr2" if re.search(r"\bsmash\b", q, re.I) else "nsc_dr2")
            args: Dict[str, Any] = {"catalog": cat, "ra": c["ra"], "dec": c["dec"], "blue_band": "g", "red_band": "r",
                                    "point_sources": bool(re.search(r"\bpoint[\s-]*sources?\b|\bstars?\b", q, re.I))}
            wm = re.search(r"\bwithin\s+(?:a\s+)?(\d+(?:\.\d+)?)\s*(?:°|deg(?:rees?)?\b)", q, re.I)
            if "radius_deg" in c or wm:
                args["radius_deg"] = c.get("radius_deg") or float(wm.group(1))
            return _dl("datalab_color_magnitude_diagram", args,
                       "It selects the cone with the survey's registered point-source cut and a documented faint depth bound "
                       "in the SQL, draws g vs g-r with the magnitude axis inverted, and returns `depth` (bound applied, "
                       "turnover) and `features` (old-population turnoff / RGB / BHB excess over the surrounding field). "
                       "Report the cuts from `cuts_applied`, state the depth limit, and say which population features are "
                       "present according to `features`.")
    # L04: a colour image of a named object from DECam / Legacy Surveys / DES
    if (re.search(r"\bcolou?r\s+(?:image|composite|picture)\b", q, re.I)
            and re.search(r"\bdecam\b|\blegacy\s+surveys?\b|\bdes\b|\bdark\s+energy\s+survey\b", q, re.I)):
        tm = re.search(r"\b(M\s?\d{1,3}|NGC\s?\d{1,4}|IC\s?\d{1,4}|Andromeda|Triangulum)\b", q, re.I)
        args = {"catalog": "des_dr1" if re.search(r"\bdes\b|\bdark\s+energy\s+survey\b", q, re.I) else "ls_dr9"}
        if tm:
            args["target_name"] = re.sub(r"\s+", "", tm.group(1)) if re.match(r"(?:M|NGC|IC)\s", tm.group(1), re.I) else tm.group(1)
        c = _coords(q)
        if {"ra", "dec"} <= set(c):
            args.update({"ra": c["ra"], "dec": c["dec"]})
        if re.search(r"\bcent(?:er|re)\b|\bcore\b|\bnucleus\b|\bbulge\b", q, re.I):
            args["fov_deg"] = 0.1
        return _dl("datalab_color_image", args,
                   "Whatever the outcome, explain it from the result: if `coverage_gap`, give `coverage_reason` and offer the "
                   "`alternatives` (never switch surveys unasked); justify the field of view with `fov_rationale`; say which "
                   "bands map to red, green and blue and the stretch from `rgb_mapping`.")
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
            args["preset"] = "south_gradient"
            region_note = ("The user did not name a region: use the `south_gradient` preset (a 10 deg cone at RA 75, Dec -30 "
                           "spanning Galactic latitude ~ -26 to -46 deg: a density gradient toward the plane plus the globular "
                           "cluster M79 -- it completes within one turn) and SAY that you chose it and why, offering other presets "
                           "(ngp, sgr_stream, smc; the LMC is very dense and may not finish in one turn). Do NOT ask a "
                           "clarifying question first.")
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
