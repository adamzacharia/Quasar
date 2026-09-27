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

__all__ = [
    "documentation_rag_override",
    "rag_override_reasons",
    "facility_documentation_question",
    "live_archive_fact_request",
    "knowledge_question",
    "rag_directive",
]

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


# Facilities whose own documentation is now in the corpus (RAG refresh
# 2026-09-25: JDox Cycle 6 pages, ESO data access policy + Phase 3 standard,
# VLA OSS + deadlines, Data Lab manual + dataset pages, DESI DR1, Legacy
# Surveys DR11, IVOA recommendations, ZTF supplement, Gaia DR4 / Euclid DR1
# release pages). MAST/HST/Chandra are NOT covered, so they stay excluded.
_COVERED_FACILITY_RE = re.compile(
    r"\b(?:jwst|webb|eso|vlt|vla|vlass|data\s?lab|datalab|noirlab|desi|"
    r"legacy\s+surveys?|ls[\s_-]?dr1[01]|ivoa|obscore|datalink|adql|table\s+access\s+protocol|"
    r"simple\s+image\s+access|ztf|euclid|gaia\s+dr4)\b",
    re.I,
)
# Knowledge about the facility itself. STRONG cues name a documented topic
# (deadline, policy, known issues, manual...) and survive a polite request
# verb ("Show me the JWST Cycle 6 proposal deadline", "Get the ESO data access
# policy"). WEAK cues ("when is...", "is X available") only count when no
# request verb is present (guard CX-18/CX-19).
_FACILITY_KNOWLEDGE_STRONG_RE = re.compile(
    r"\b(?:deadlines?|call\s+for\s+proposals|cfp|proposal\s+(?:cycle|round|process)|semester|20\d\d[ab]|"
    r"exclusive\s+access|proprietary|data\s+rights|(?:data\s+)?(?:access\s+)?polic(?:y|ies)|"
    r"(?:data\s+)?release\s+(?:date|notes|schedule|timeline)|known\s+issues?|documentation|manual|"
    r"status\s+summary)\b",
    re.I,
)
_FACILITY_KNOWLEDGE_WEAK_RE = re.compile(
    r"\b(?:when\s+(?:is|will|does|do)\s+(?:the\s+)?(?:[\w.-]+\s+){0,5}?"
    r"(?:due|released?|release|call|cycle|dr\d+|available|public|opens?|starts?|begins?|announced?)|"
    r"(?:is|are)\s+(?:the\s+)?(?:[\w-]+\s+){0,3}?(?:dr\d+|data\s+releases?(?:\s+\d+)?|releases?|catalog(?:ue)?s?|surveys?|datasets?)\s+"
    r"(?:[\w-]+\s+){0,3}?(?:available|hosted|served|released|public)|"
    r"which\s+(?:[\w-]+\s+){0,3}?(?:data\s*sets?|datasets|releases|surveys)\s+(?:does|do|are|is)|"
    r"(?:does|do)\s+[\w\s-]{0,30}?\b(?:host|serve|offer)|"
    r"what\s+(?:is|are)\s+(?:the\s+)?(?:difference|new|changes)|"
    r"how\s+(?:do|does|can|should)\s+(?:i|we|one|you)\s+(?:cite|acknowledge|propose|apply|submit|"
    r"write\s+an?\s+adql|use\s+(?:tap|adql|datalink)))\b",
    re.I,
)
# Always a data request: never documentation, whatever else the message says.
_HARD_DATA_RE = re.compile(
    r"\b(?:plot|map|draw|histogram|redshift\s+distribution|cone\s+search|cross-?match|cutouts?|image\s+of|"
    r"light\s?curves?|cmd|seds?|select\s+(?:all|the|stars|galaxies|sources)|"
    r"how\s+many\s+(?:sources|stars|galaxies|objects|rows)|are\s+there\s+(?:any\s+)?(?:public|archival)|"
    r"which\s+(?:[\w-]+\s+)?programs?|observations?\s+(?:of|taken)|fluxe?s?|magnitudes?|"
    r"spectr(?:a|um)\s+of)\b|\b(?:ra|dec)\s*=",
    re.I,
)
# Request verbs: data requests unless a STRONG documentation topic is named.
_SOFT_DATA_RE = re.compile(
    r"\b(?:show\s+me|get\s+(?:g|r|i|z|the|me)|list\s+the|find\s+(?:all|the|high|sources|stars|galaxies|candidates)|"
    r"retrieve|fetch|download\s+(?:the|all|images?|spectra)|make\s+a)\b",
    re.I,
)
# The verb's object must be the documentation noun itself ("Get the ESO data
# access policy"); "data access policy" / "data rights" are document nouns.
_SOFT_VERB_DOC_OBJECT_RE = re.compile(
    r"\b(?:show\s+me|get|list|find|retrieve|fetch|download|make)\s+"
    # Only an allow-list may sit between the verb and the documentation noun
    # (articles, facility / programme names, cycle or semester, numbers,
    # "proposal", "access"), so any data noun ("data", "exposures", "rows",
    # "reduced products", ...) breaks the match (guard CX-24, rounds 2-4).
    r"(?:(?:the|a|an|current|latest|official|new|full|updated|jwst|webb|eso|vlt|vla|vlass|data\s?lab|datalab|"
    r"noirlab|desi|legacy|surveys?|ls|dr\d+|ivoa|ztf|euclid|gaia|stsci|nrao|cycle|semester|\d+[ab]?|"
    r"proposal|submission|general|science|key|access|exclusive|for)\s+){0,6}?"
    r"(?:(?:data\s+)?(?:access\s+)?polic(?:y|ies)|data\s+rights|deadlines?|call\s+for\s+proposals|cfp|"
    r"release\s+(?:date|notes|schedule|timeline)|known\s+issues?|documentation|manual|status\s+summary)\b",
    re.I,
)


def facility_documentation_question(user_query: str) -> bool:
    """True for a policy / deadline / release / documentation question about a
    non-ALMA facility whose docs are in the corpus (see _COVERED_FACILITY_RE),
    e.g. "When is the JWST Cycle 6 proposal deadline?" or "Is LS DR11 available
    in Data Lab?". Never true for a data request (plot/select/cone search...)."""
    q = str(user_query or "")
    if not _COVERED_FACILITY_RE.search(q) or _HARD_DATA_RE.search(q):
        return False
    if _SOFT_DATA_RE.search(q):
        # A request verb survives only when its object is a documentation
        # noun ("Show me the JWST Cycle 6 proposal deadline"), not data that
        # merely carries a policy word ("Fetch ESO proprietary data") (CX-24).
        return bool(_SOFT_VERB_DOC_OBJECT_RE.search(q))
    return bool(_FACILITY_KNOWLEDGE_STRONG_RE.search(q) or _FACILITY_KNOWLEDGE_WEAK_RE.search(q))


# Live archive facts: values the archive holds today (a project's PI, its
# member OUS count, a planet count "according to" an archive, whether public
# data exist). The manuals cannot supply them. ArchiveBench dev 2026-09-26
# (AB-D-41, 48, 49, 51): with documentation context injected, gpt-oss answered
# "the ALMA documentation excerpts you provided do not contain ..." without
# calling a single tool, and scored 0 on all four.
_PROJECT_CODE_RE = re.compile(r"\b\d{4}\.\d\.\d{5}\.[a-z]\b", re.I)
_LIVE_FACT_RE = re.compile(
    r"\bwho\s+(?:is|was|are|were)\s+(?:the\s+)?(?:pis?|principal\s+investigators?)\b"
    r"|\bhow\s+many\b[^?]{0,200}?\baccording\s+to\b"
    r"|\b(?:is|are)\s+there\s+(?:any\s+)?(?:public|archival)\b[^?]{0,80}?\b(?:data|observations?|projects?|spectra|images?)\b"
    r"|\b(?:does|do)\s+(?:the\s+)?(?:alma|archive|mast|heasarc|irsa|eso)\b[^?]{0,40}?\b(?:hold|have)\s+(?:any\s+)?public\b"
    r"|\bwhich\s+public\s+(?:[\w()-]+\s+){0,2}?(?:projects?|observations?|datasets?|programs?|programmes?)\b",
    re.I,
)


def live_archive_fact_request(user_query: str) -> bool:
    """True when the question asks for a live archive fact (see _LIVE_FACT_RE):
    a project code, "who is the PI", "how many ... according to <archive>",
    "is there public ... data", "which public ... projects". An explicit how-to
    opening ("How do I find who the PI of 2019.1.00001.S is?") still counts as
    documentation (:data:`_STRONG_HOWTO_RE`)."""
    q = str(user_query or "")
    if _STRONG_HOWTO_RE.search(q):
        return False
    if _LIVE_FACT_RE.search(q):
        return True
    # A project code alone is a data request unless the user is asking for help
    # processing data (ALMABench: "I downloaded the ALMA archive package for
    # project 2019.1.00001.S ... give me the steps and a script").
    if not _PROJECT_CODE_RE.search(q):
        return False
    # A mixed request ("give me the PI and title of ..., and explain how to
    # calibrate it") still needs the archive for its facts (guard CX-26 round 3).
    return not _processing_request(q) or bool(_PROJECT_FACT_RE.search(q))


_PROJECT_FACT_RE = re.compile(
    r"\b(?:pis?|principal\s+investigators?|title|called|named|mous|member\s+(?:ous|observing\s+units?)|"
    r"how\s+many|(?:is|are)\s+(?:it|they|the\s+data)\s+(?:public|released)|release\s+date|proprietary\s+period|"
    r"when\s+was|observed\s+(?:on|in|when))\b",
    re.I,
)


# Processing help = a request framing AND a processing task. Either alone is a
# fact question: "Which pipeline version processed 2016.1.00484.L?" (CX-03),
# "Which CASA command was used to calibrate ..." (CX-26 verify), "Give me the
# title and PI of ..." all stay live archive facts.
_PROCESSING_REQUEST_RE = re.compile(
    r"\b(?:give|show|write|send|tell)\s+me\b|\bhow\s+(?:do|can|should|would)\s+(?:i|we|one|you)\b|"
    r"\b(?:i|we)\s+(?:want|need|would\s+like|am\s+trying|are\s+trying)\s+to\b|\bhelp\s+me\b|\bi\s+downloaded\b",
    re.I,
)
_PROCESSING_TASK_RE = re.compile(
    r"\b(?:steps?|scripts?|commands?|tclean|clean|re-?imag\w*|imag(?:e|ing)|cubes?|continuum|self-?cal\w*|"
    r"calibrat(?:e|ion)|reduc(?:e|tion)|restor(?:e|ing)|scriptforpi|measurement\s+sets?|casa)\b",
    re.I,
)


def _processing_request(q: str) -> bool:
    return bool(_PROCESSING_REQUEST_RE.search(q) and _PROCESSING_TASK_RE.search(q))


# How-to / policy / procedure wording: the documentation IS the answer.
_PROCEDURE_POLICY_RE = re.compile(
    r"\b(?:how\s+(?:do|can|should|would)\s+(?:i|we|one|you)|how\s+to|procedures?|polic(?:y|ies)|"
    r"proprietary\s+period|deadlines?|guidelines?|requirements?|recommend\w*|best\s+practices?|"
    r"what\s+is\s+the\s+(?:difference|process|rule)|explain|describe)\b",
    re.I,
)


def knowledge_question(user_query: str) -> bool:
    """True for a genuine how-to / policy / procedure / reference question,
    where the documentation context is the answer. Never true for a live
    archive fact request."""
    q = str(user_query or "")
    if live_archive_fact_request(q):
        return False
    return documentation_rag_override(q) or bool(_PROCEDURE_POLICY_RE.search(q))


# Round-0 note appended when documentation context was injected and no other
# route directive fired (core/runner.py).
KNOWLEDGE_DIRECTIVE = (
    "\n\n[SYSTEM NOTE: This is a how-to / policy / procedure question. Answer it from the DOCUMENTATION "
    "CONTEXT above, with its citations. Do NOT call `search_papers`: the user is not asking for papers. "
    "If part of the question needs a live archive fact (a PI, project code, count, coverage, public data, "
    "release date or catalogue value), call the archive tool for that part instead of saying the "
    "documentation does not contain it.]"
)
BACKGROUND_DIRECTIVE = (
    "\n\n[SYSTEM NOTE: The DOCUMENTATION CONTEXT above is background only, not the source of the answer. "
    "It does not hold live archive facts. For any PI, project code, count, coverage, public-data status, "
    "release date or catalogue value, call the archive tools and answer from their results. Never answer "
    "that the documentation excerpts do not contain the fact. Do NOT call `search_papers` unless the user "
    "asks for papers.]"
)


def rag_directive(user_query: str) -> str:
    """The note for a turn that carries documentation context and no other
    route directive: KNOWLEDGE for how-to / policy questions, BACKGROUND (docs
    are context, tools give the facts) otherwise."""
    return KNOWLEDGE_DIRECTIVE if knowledge_question(user_query) else BACKGROUND_DIRECTIVE


def documentation_rag_override(user_query: str) -> bool:
    """True when the message is a how-to / reference question about ALMA and
    its tooling (see module docstring), or a policy / deadline / release /
    documentation question about another facility the corpus covers
    (:func:`facility_documentation_question`); False for pure data requests."""
    q = str(user_query or "")
    if live_archive_fact_request(q):
        return False  # the archive answers it, not the manuals (AB-D-41/48/51)
    if facility_documentation_question(q):
        return True
    r = rag_override_reasons(q)
    if r["catalog_data"] and not re.search(r"\balma\b", q, re.I):
        return False
    if _ARCHIVE_DATA_RE.search(q) and not _STRONG_HOWTO_RE.search(q):
        return False
    if r["other_facility"] and not r["alma_specific"]:
        return False  # "How do I download HST FITS files from MAST?" is not an ALMA-manual question
    return r["topic"] and r["howto"]
