"""Small per-turn guards for the runner's tool loop (2026-10-03).

Pure functions and one tiny tracker, kept out of core/runner.py so they can be
unit-tested without building an agent. They fix three failures seen on one
gpt-oss-120b literature turn (tmp/literature-search-fix-2026-10-03/PLAN.md):

* ``repair_tool_name``: gpt-oss leaked harmony tokens into a function name
  (``search_papers<|channel|>commentary``), the registry lookup failed with
  "Unknown tool" and the round was wasted.
* ``LiteratureSearchGuard``: six rewordings of one failing ADS search are six
  "new" calls to the canonical repeat detector, so they spent the whole
  8-round budget. After a streak of empty searches a near-duplicate one is
  answered with a hint instead of being run.
* ``looks_like_leaked_reasoning`` / ``strip_leaked_reasoning``: in the forced
  final round gpt-oss sometimes writes its analysis ("The user asks: ... We
  need to ...") as the answer text.
* ``budget_reason_label``: the "Tool budget reached" step now says which limit
  tripped.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Set

# ─────────────────────────────────────────────────────────────────────────────
# Tool names
# ─────────────────────────────────────────────────────────────────────────────
_HARMONY_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")


def sanitize_tool_name(name: Any) -> str:
    """Strip gpt-oss harmony residue from a function name.

    ``search_papers<|channel|>commentary`` -> ``search_papers``;
    ``functions.search_papers`` -> ``search_papers``. Anything after the first
    control token is channel chatter, never part of the name."""
    text = str(name or "")
    if "<|" in text:
        text = text.split("<|", 1)[0]
    text = _HARMONY_TOKEN_RE.sub("", text).strip()
    if text.startswith("functions."):
        text = text[len("functions."):]
    # a trailing constrain hint ("search_papers json") is not part of a name
    text = text.split()[0] if text.split() else ""
    return text


def repair_tool_name(name: Any, known: Iterable[str]) -> str:
    """The registered tool a mangled name meant, or the name unchanged.

    Only repairs when the sanitized name is a known tool, so a genuinely
    unknown name still takes the runner's "Unknown tool" path."""
    raw = str(name or "")
    known_set: Set[str] = set(known or ())
    if raw in known_set:
        return raw
    clean = sanitize_tool_name(raw)
    return clean if clean and clean in known_set else raw


# ─────────────────────────────────────────────────────────────────────────────
# Budget reason
# ─────────────────────────────────────────────────────────────────────────────
def budget_reason_label(reason: Optional[str]) -> str:
    """Plain words for why a turn's tool budget ended."""
    text = str(reason or "").strip()
    if not text:
        return ""
    m = re.search(r"max tool rounds reached \((\d+)\)", text)
    if m:
        return f"all {m.group(1)} tool rounds used"
    m = re.search(r"tool time budget reached \((\d+)s\)", text)
    if m:
        return f"time limit reached after {m.group(1)} s"
    if "repeated tool calls" in text:
        return "the same calls kept repeating"
    if "token budget" in text:
        return "token budget reached"
    return text


def budget_status_text(reason: Optional[str]) -> str:
    """The Research-timeline step. Keeps the "Tool budget reached" prefix the
    UI classifies on (ui-pro/src/lib/research-timeline.js WRAP_RE)."""
    label = budget_reason_label(reason)
    middle = f" ({label})" if label else ""
    return f"Tool budget reached{middle}, composing the final answer from collected results"


# ─────────────────────────────────────────────────────────────────────────────
# Literature search streak guard
# ─────────────────────────────────────────────────────────────────────────────
LITERATURE_SEARCH_TOOLS = frozenset({"search_papers"})
_STOP = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "by", "with", "from", "at", "as", "is", "are",
    "was", "were", "be", "paper", "papers", "article", "articles", "publication", "publications", "find",
    "search", "about", "that", "this", "which", "what", "first", "reported", "reporting", "report", "ads",
    "nasa", "refereed", "journal", "bibcode", "bibcodes",
}


_IDENTIFIER_RE = re.compile(
    r"(?:1[6-9]|20)\d{2}[A-Za-z][A-Za-z0-9&.]{13}[A-Za-z.]"  # bibcode
    r"|\b10\.\d{4,9}/\S+"                                      # DOI
    r"|\barxiv\s*:?\s*\d{4}\.\d{4,5}|\b[a-z\-]+(?:\.[A-Z]{2})?/\d{7}\b",  # arXiv
    re.IGNORECASE,
)


def _terms(query: Any) -> Set[str]:
    words = re.findall(r"[a-z0-9][a-z0-9\-\.]*", str(query or "").lower())
    return {w.strip(".") for w in words if w.strip(".") and w.strip(".") not in _STOP}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _hits(result_str: Any) -> Optional[int]:
    try:
        obj = json.loads(result_str) if isinstance(result_str, str) else result_str
    except (TypeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("success") is False or obj.get("error"):
        # A failed search is not an empty literature (guard CX-05).
        return None
    if isinstance(obj.get("count"), int):
        return obj["count"]
    if isinstance(obj.get("papers"), list):
        return len(obj["papers"])
    return None


class LiteratureSearchGuard:
    """Per-turn record of literature searches and their hit counts.

    ``check`` returns a hint (and the call is NOT run) when the last
    ``streak`` literature searches all came back empty and this one is a
    near-duplicate (term Jaccard >= ``similarity``) of an earlier empty one.
    A genuinely different search is always allowed."""

    def __init__(self, streak: int = 3, similarity: float = 0.5,
                 tools: Iterable[str] = LITERATURE_SEARCH_TOOLS) -> None:
        self.streak = max(1, int(streak))
        self.similarity = float(similarity)
        self.tools = frozenset(tools)
        self._history: List[Dict[str, Any]] = []

    def check(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        if tool_name not in self.tools or not isinstance(args, dict):
            return None
        recent = self._history[-self.streak:]
        if len(recent) < self.streak or any(h["hits"] != 0 for h in recent):
            return None
        if _IDENTIFIER_RE.search(str(args.get("query") or "")):
            # A bibcode / DOI / arXiv id is the exact lookup the hint asks
            # for; never block it (guard CX-06).
            return None
        terms = _terms(args.get("query"))
        if not any(h["hits"] == 0 and _jaccard(terms, h["terms"]) >= self.similarity for h in self._history):
            return None
        tried = "; ".join(repr(h["query"])[:90] for h in recent)
        return (
            f"Not run: the last {len(recent)} literature searches returned no papers ({tried}) and this one is "
            "a rewording of them. Stop rephrasing. Either call search_papers once with something substantially "
            "different (a bibcode, DOI or arXiv id from the user, or two distinctive title words plus the first "
            "author's surname with no year), or answer now with what you have and say the search found nothing."
        )

    def record(self, tool_name: str, args: Dict[str, Any], result_str: Any) -> None:
        if tool_name not in self.tools or not isinstance(args, dict):
            return
        hits = _hits(result_str)
        if hits is None:
            return
        self._history.append({"query": str(args.get("query") or ""), "terms": _terms(args.get("query")),
                              "hits": hits})


# ─────────────────────────────────────────────────────────────────────────────
# "Who pushed back on X" follow-up
# ─────────────────────────────────────────────────────────────────────────────
_PUSHBACK_RE = re.compile(
    r"\b(?:push(?:ed|es|ing)?\s+back|disput\w*|challeng\w*|rebut\w*|refut\w*|respon(?:d|ded|ses?)\s+to|"
    r"criticis\w*|criticiz\w*|critiqu\w*|alternative\s+explanations?|contested|cast\s+doubt|"
    r"(?:re-?analy[sz]\w*|failed\s+to\s+(?:confirm|reproduce|replicate)))",
    re.IGNORECASE,
)


def asks_for_responses(user_query: Any) -> bool:
    """The user wants the papers that responded to / disputed a paper."""
    return bool(_PUSHBACK_RE.search(str(user_query or "")))


def citing_followup_hint(user_query: Any, tool_name: str, result_str: Any,
                         called_tools: Iterable[str]) -> Optional[str]:
    """``result_str`` with a ``next_step`` pointing at find_citing_papers, or None.

    Live 2026-10-03: gpt-oss answered every "who pushed back on X" question
    by re-wording search_papers and never called find_citing_papers, which
    returns the rebuttals in one call. Fires once per turn (only while
    find_citing_papers has not been called), only for a search that found
    papers, and only when the user's question asks for responses."""
    if tool_name != "search_papers" or "find_citing_papers" in set(called_tools or ()):
        return None
    if not asks_for_responses(user_query):
        return None
    try:
        obj = json.loads(result_str) if isinstance(result_str, str) else None
    except (TypeError, ValueError):
        return None
    if not isinstance(obj, dict) or not obj.get("success") or obj.get("next_step"):
        return None
    papers = [p for p in (obj.get("papers") or []) if isinstance(p, dict) and p.get("bibcode")]
    if not papers:
        return None

    def _year(p: Dict[str, Any]) -> int:
        try:
            return int(str(p.get("year") or p["bibcode"][:4])[:4])
        except (TypeError, ValueError):
            return 9999

    # The original claim usually predates its responses, so the earliest
    # results are the candidates; the model picks (guard CX-20).
    candidates = sorted(papers, key=_year)[:3]
    listed = "; ".join(f"{p['bibcode']} ({str(p.get('title') or '')[:60]})" for p in candidates)
    obj["next_step"] = (
        "NEXT CALL: find_citing_papers. The user asked which papers disputed, challenged or responded to a "
        f"paper. Identify the ORIGINAL paper (earliest candidates here: {listed}) and call find_citing_papers "
        "with its bibcode (focus='rebuttals', or the user's angle such as 'alternative explanations'). It "
        "returns the responding papers in one call; searching search_papers for each response one by one "
        "misses most of them."
    )
    return json.dumps(obj, ensure_ascii=False)


# A title that answers another paper is not the original claim.
_RESPONSE_TITLE_RE = re.compile(
    r"\b(?:no evidence|re-?analys\w*|reassess\w*|revisit\w*|upper limits?|non-?detection|reply|response|"
    r"comment|matters arising|cannot|does not|not confirmed|insufficient|complications|addendum|erratum|"
    r"corrigendum|retraction|challeng\w*|refut\w*|rebuttal|disput\w*|critique|critical\s+assessment|"
    r"questioning|against|tension|caution|doubt\w*)\b",
    re.IGNORECASE,
)


def pick_original_candidate(papers: Any) -> Optional[Dict[str, Any]]:
    """The likeliest ORIGINAL paper among search results: the most cited
    paper whose title does not read as a response (a claim that drew
    responses usually out-cites them, but a heavily cited rebuttal must not
    be picked: follow-up guard CX-03), earliest year on a tie. Falls back to
    all papers when every title reads as a response."""
    rows = [p for p in (papers or []) if isinstance(p, dict) and p.get("bibcode")]
    pool = [p for p in rows if not _RESPONSE_TITLE_RE.search(str(p.get("title") or ""))] or rows
    best = None
    for p in pool:
        try:
            cites = int(p.get("citations") or 0)
        except (TypeError, ValueError):
            cites = 0
        try:
            year = int(str(p.get("year") or p["bibcode"][:4])[:4])
        except (TypeError, ValueError):
            year = 9999
        key = (cites, -year)
        if best is None or key > best[0]:
            best = (key, p)
    return best[1] if best else None


def responses_focus(user_query: Any) -> str:
    """find_citing_papers focus for the user's angle."""
    if re.search(r"\balternative\s+explanations?\b", str(user_query or ""), re.IGNORECASE):
        return "alternative explanations"
    return "rebuttals"


def queued_citing_note(result_str: Any, seed: Dict[str, Any]) -> str:
    """``result_str`` whose next_step says Quasar is running
    find_citing_papers for ``seed`` (the next tool output in this round)."""
    obj = json.loads(result_str) if isinstance(result_str, str) else dict(result_str or {})
    obj["next_step"] = (
        f"Quasar is also running find_citing_papers for {seed.get('bibcode')} "
        f"({str(seed.get('title') or '')[:80]}), the likeliest original paper; its result comes after this "
        "round's other tool outputs. Answer from it (bibcodes exactly as given; possible_rebuttals only after checking their "
        "titles). If that is not the original paper, call find_citing_papers with the right bibcode instead of "
        "re-searching."
    )
    return json.dumps(obj, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Leaked reasoning in a final answer
# ─────────────────────────────────────────────────────────────────────────────
# An optional channel-style label in front of the monologue ("Analysis:").
_LEAK_LABEL = r"(?:(?:analysis|reasoning|thinking|thoughts?|commentary|internal notes?)\s*[:\-]\s*)?"
# Verbs about producing THIS reply, not about doing science: "We need to answer
# / retrieve / compose" is a plan, "We need to use independent dust maps" can be
# the answer itself (guard CX-09).
_META_VERBS = (r"(?:answer|respond|reply|provide|produce|compose|craft|output|retrieve|call|summari[sz]e|"
               r"give the|write the|search|look up|format)")
_LEAK_OPENER_RE = re.compile(
    r"^\s*(?:\*\*)?" + _LEAK_LABEL + r"(?:\*\*)?(?:"
    r"the user(?:'s)? (?:asks|asked|is asking|wants|wanted|requests|requested|request|needs|question)"
    r"|user (?:asks|wants|requests)"
    r"|we (?:need|have|must|should) (?:to )?" + _META_VERBS +
    r"|(?:so|now|ok(?:ay)?|alright),?(?: so)? (?:the user|we need|we should|we must|let(?:'|’)s)\b"
    r"|let(?:'|’)s (?:produce|craft|answer|compose|write|summari[sz]e|respond)"
    r"|i (?:need|should|must|will need) to " + _META_VERBS +
    r")",
    re.IGNORECASE,
)
_LABEL_ONLY_RE = re.compile(r"^\s*(?:\*\*)?" + _LEAK_LABEL.replace(")?", ")") + r"(?:\*\*)?\s*$", re.IGNORECASE)
_LABELLED_RE = re.compile(r"^\s*(?:\*\*)?(?:analysis|reasoning|thinking|thoughts?|commentary)\s*[:\-]", re.IGNORECASE)
_SELF_TALK_RE = re.compile(
    r"\(i think\)|\bmaybe\?|\bactually,|\blet(?:'|’)s (?:check|see|recall|think)\b|\bwe need to\b|\bthe user\b"
    r"|\bwe have (?:to|results)\b|\bwe should\b|\bi recall\b|\bnot sure\b|\bhmm\b|\bprobably\b.{0,40}\?",
    re.IGNORECASE,
)


def looks_like_leaked_reasoning(text: Any) -> bool:
    """True when the START of an answer is the model talking to itself.

    A third-person opener ("The user asks", "We need to answer") is enough;
    otherwise it takes three or more self-talk markers inside the first 600
    characters together with a mention of "the user". A normal answer that
    says "we need to" once, deep in a paragraph, does not trip it."""
    head = str(text or "").lstrip()[:600]
    if not head:
        return False
    if _LEAK_OPENER_RE.match(head):
        return True
    markers = _SELF_TALK_RE.findall(head)
    if _LABELLED_RE.match(head) and markers:
        return True  # "Analysis: ... we need to ..." (guard CX-09)
    return len(markers) >= 3 and bool(re.search(r"\bthe user\b", head, re.IGNORECASE))


def _paragraph_is_reasoning(par: str) -> bool:
    p = par.strip()
    if not p or _LABEL_ONLY_RE.match(p):
        return True
    if _LEAK_OPENER_RE.match(p):
        return True
    return len(_SELF_TALK_RE.findall(p[:600])) >= 2


def strip_leaked_reasoning(text: Any) -> str:
    """Drop the leading paragraphs that are reasoning; keep the rest.

    Returns "" when every paragraph is reasoning (the caller then composes an
    answer another way)."""
    paragraphs = re.split(r"\n\s*\n", str(text or ""))
    i = 0
    while i < len(paragraphs) and _paragraph_is_reasoning(paragraphs[i]):
        i += 1
    return "\n\n".join(paragraphs[i:]).strip()


LEAK_RESAMPLE_NOTE = (
    "[SYSTEM CONTINUATION] Your last message described your reasoning instead of answering. Write ONLY the "
    "answer for the user now, using the tool results above: no restating of the question, no 'we need to', no "
    "plan. Do not call tools."
)


def model_leaks_reasoning(model: Any) -> bool:
    """Models whose forced-final-round text gets the leak check (gpt-oss
    family; its analysis channel can arrive as plain content)."""
    return "gpt-oss" in str(model or "").lower()
