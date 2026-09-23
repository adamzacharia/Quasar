"""Documentation-RAG routing override for how-to / reference questions.

UI benchmark 2026-09-22: the documentation search ran on 9 of 22 domain
turns; the invented specifics (a non-existent ``pipeline_version`` ObsCore
column, ``Alma.get_data_links``, ASA URL parameters ``target=``/``band=``,
"Eq. 9.8 in the Handbook") cluster exactly where it did NOT run (D06, D07,
D08, D13, D14, D15, D17). The runner's gate skipped those because they also
matched its archive-fetch patterns ("find ... observations").

:func:`documentation_rag_override` is a pure, regex-only decision applied
AFTER the runner's existing gates: it forces the documentation search ON when
the message is a how-to / reference question about ALMA, CASA, the archive,
astroquery, the pipeline, QA2, DataLink, TAP or ALMiner. It never turns RAG
off, and RAG never blocks tools -- a live archive query still runs when the
question needs one (D08 "as of March 25, 2026").
"""
from __future__ import annotations

import re
from typing import Dict

__all__ = ["documentation_rag_override", "rag_override_reasons"]

_TOPIC_RE = re.compile(
    r"\b(?:alma|casa|archive|asa\b|astroquery|pipeline|qa2|weblog|datalink|data\s*link|tap\b|adql|alminer|"
    r"obscore|ivoa|proprietary|correlator|configurations?|baselines?|band\s*\d|bands|receivers?|"
    r"technical\s+handbook|handbook|proposer'?s?\s+guide|observing\s+tool|calibrat\w+|"
    r"measurement\s+set|\bms\b|fits|weblog|manifest|readme|data\s+products?|delivered|download\w*|"
    r"bandwidth\s+switching|spectral\s+setup|sensitivity|resolution|cycle\s+\d+)\b",
    re.I,
)

_HOWTO_RE = re.compile(
    r"\b(?:how\s+(?:do|can|could|should|would|to|does|is|are|many|much)|is\s+there\s+(?:a|any)\s+way|"
    r"which|what\s+(?:is|are|types?|kinds?|sort|does|do|can|would|information|steps?)|where\s+(?:do|can|is|are|would)|"
    r"explain|describe|tell\s+me\s+about|what\s+can\s+you\s+tell|show\s+me\s+(?:two|three|\d+|several|different)\s+ways?|"
    r"walk\s+me\s+through|guide|tutorial|example|snippet|code|python|script|policy|policies|documentation|"
    r"generate\s+the\s+url|url\s+for|link\s+for|likely\s+needed|what\s+are\s+the\s+different)\b",
    re.I,
)

# Requests whose ONLY intent is fetching/plotting data from a survey catalogue
# (Data Lab benchmark shapes) -- no documentation lookup.
_CATALOG_DATA_RE = re.compile(
    r"\b(?:gaia|des\s+dr\d|desi|nsc|smash|delve|legacy\s+surveys?|ls_dr\d|sdss|boss|vhs|pan-?starrs|unwise|2mass|"
    r"data\s?lab|datalab|healpix|cmd\b|color-?magnitude|colour-?magnitude|light\s?curves?|proper\s+motions?|parallax|"
    r"dwarf\s+galax\w+|satellite|tidal|overdensit\w+|redshift\s+slice|great\s+wall|seds?\b|cutouts?|image\s+of)\b",
    re.I,
)


# Terms that only make sense for ALMA / its tooling (the RAG corpus).
_ALMA_SPECIFIC_RE = re.compile(
    r"\b(?:alma|casa|asa|qa2|alminer|weblog|proposer'?s?\s+guide|observing\s+tool|technical\s+handbook|"
    r"bandwidth\s+switching|band\s*(?:[1-9]|10)\b|measurement\s+set|mous|member\s+ous|atacama)\b",
    re.I,
)
# Another facility / archive named: generic topics (FITS, resolution,
# download, archive, data products ...) then belong to IT, not to the ALMA
# manuals (guard CX-28).
_OTHER_FACILITY_RE = re.compile(
    r"\b(?:mast|hst|hubble|jwst|webb|chandra|xmm|spitzer|herschel|tess|kepler|k2|galex|swift|fermi|euclid|"
    r"rubin|lsst|ztf|pan-?starrs|sdss|boss|desi|gaia|2mass|wise|unwise|irsa|ned|simbad|vizier|eso|vlt|muse|"
    r"keck|gemini|subaru|noirlab|data\s?lab|decam|des\b|legacy\s+surveys?|vla|vlass|meerkat|askap|lofar|"
    r"sma\b|noema|iram|jcmt|apex|gbt|arecibo|heasarc|cadc|mast\s+portal|astroquery\.mast|astroquery\.eso)\b",
    re.I,
)


def rag_override_reasons(user_query: str) -> Dict[str, bool]:
    q = str(user_query or "")
    return {
        "topic": bool(_TOPIC_RE.search(q)),
        "howto": bool(_HOWTO_RE.search(q)),
        "catalog_data": bool(_CATALOG_DATA_RE.search(q)),
        "alma_specific": bool(_ALMA_SPECIFIC_RE.search(q)),
        "other_facility": bool(_OTHER_FACILITY_RE.search(q)),
    }


# Archive DATA requests phrased as questions ("How many projects in Cycle 10
# observed the sun?", "List the protostellar disks ... for which ..."): the
# answer comes from a live query, not from the Handbook.
_ARCHIVE_DATA_RE = re.compile(
    r"\b(?:how\s+many\s+(?:alma\s+)?(?:projects?|observations?|datasets?|mous|sources?|galaxies|targets?|execution\s+blocks?)|"
    r"list\s+(?:the|all)\b|show\s+me\s+(?:all|the\s+locations?|locations?|images?|two\s+images)|find\s+(?:alma\s+)?observations|"
    r"observed\s+(?:the\s+sun|with\s+alma)|make\s+a\s+table\s+of|give\s+me\s+a\s+summary\s+of\s+the\s+band)\b",
    re.I,
)


# An explicit how-to opening beats the archive-data shapes: "How do I use
# Astroquery to find ALMA observations of M83?" (D14) is a documentation
# question even though it says "find ... observations".
_STRONG_HOWTO_RE = re.compile(
    r"^\s*(?:\w+[,:!-]?\s+){0,3}?(?:how\s+(?:do|can|could|would|should)\s+(?:i|we|one|you)|how\s+to|is\s+there\s+(?:a|any)\s+way|"
    r"show\s+me\s+(?:two|three|\d+|several|different)\s+ways?|what\s+(?:types?|kinds?)\s+of|explain|describe|"
    r"what\s+can\s+you\s+tell|where\s+(?:do|can)\s+i|which\s+(?:tool|method|python|package|module))\b",
    re.I,
)


def documentation_rag_override(user_query: str) -> bool:
    """True when the message is a how-to / reference question about ALMA and
    its tooling (see module docstring); False for pure data requests."""
    q = str(user_query or "")
    r = rag_override_reasons(q)
    if r["catalog_data"] and not re.search(r"\balma\b", q, re.I):
        return False
    if _ARCHIVE_DATA_RE.search(q) and not _STRONG_HOWTO_RE.search(q):
        return False
    if r["other_facility"] and not r["alma_specific"]:
        return False  # "How do I download HST FITS files from MAST?" is not an ALMA-manual question
    return r["topic"] and r["howto"]
