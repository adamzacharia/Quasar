"""Mechanical citation recall/precision metrics for synthesized answers (R3).

The two-metric verifiability framework (Liu et al., arXiv:2304.09848):

* **citation recall** — fraction of claim sentences that are supported;
* **citation precision** — fraction of attached citations that support
  their sentence.

This module computes deliberately *mechanical* approximations with zero
extra LLM calls (the LLM-judged variant is the R13 verification gate):

* a **claim sentence** is a prose sentence carrying a checkable payload —
  a citation (ADS bibcode / DOI) or a numeric value;
* a claim sentence counts as **supported** when it carries a citation that
  resolved in ADS, or when the majority of its numeric tokens appear in the
  turn's tool-result evidence;
* **precision** is the fraction of extracted citations that resolve in ADS
  (an existence check — it cannot detect a real-but-irrelevant citation,
  which is why ``method`` is stamped on every result).

Never raises: callers run inside synthesis finalization.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from services.citation_verifier import BIBCODE_RE, DOI_RE, extract_citations

logger = logging.getLogger(__name__)

METRICS_METHOD = "mechanical_v1"

# Sentence boundary: terminal punctuation followed by whitespace + capital/digit
# (avoids splitting on decimal points and most abbreviations).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])")
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
# A bare 4-digit year is almost always bibliographic context, not a
# measured value — do not demand tool-result evidence for it.
_YEAR_RE = re.compile(r"^(1[89]|20)\d{2}$")


def split_sentences(text: str) -> List[str]:
    """Split markdown prose into sentences, dropping non-prose lines."""
    if not text:
        return []
    cleaned = _CODE_BLOCK_RE.sub(" ", text)
    prose_lines: List[str] = []
    for line in cleaned.splitlines():
        if _HEADING_RE.match(line) or _TABLE_ROW_RE.match(line):
            continue
        stripped = line.strip()
        if not stripped:
            prose_lines.append("")  # keep paragraph boundaries
            continue
        # Strip list markers but keep the sentence text.
        stripped = re.sub(r"^[-*+]\s+|^\d+\.\s+", "", stripped)
        prose_lines.append(stripped)
    paragraph_text = "\n".join(prose_lines)
    sentences: List[str] = []
    for block in paragraph_text.split("\n"):
        if not block.strip():
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(block):
            sentence = sentence.strip()
            if len(sentence) >= 15:
                sentences.append(sentence)
    return sentences


def _sentence_numbers(sentence: str) -> List[str]:
    """Numeric tokens outside citation strings; bare years excluded."""
    scrubbed = BIBCODE_RE.sub(" ", sentence)
    scrubbed = DOI_RE.sub(" ", scrubbed)
    numbers = []
    for token in _NUMBER_RE.findall(scrubbed):
        if len(token) < 2 or _YEAR_RE.match(token):
            continue
        numbers.append(token)
    return numbers


def is_claim_sentence(sentence: str) -> bool:
    """A sentence with a checkable payload: a citation or a numeric value."""
    if sentence.endswith("?"):
        return False
    citations = extract_citations(sentence)
    if citations["bibcodes"] or citations["dois"]:
        return True
    return bool(_sentence_numbers(sentence))


def _resolved_citation_ids(verification: Optional[Dict[str, Any]]) -> Dict[str, set]:
    resolved: set = set()
    seen: set = set()
    for bucket in ("bibcodes", "dois"):
        for item in (verification or {}).get(bucket, []) or []:
            identifier = str(item.get("id", "") or "")
            if not identifier:
                continue
            seen.add(identifier)
            if item.get("resolved") is True:
                resolved.add(identifier)
    return {"resolved": resolved, "seen": seen}


def _normalize_evidence(evidence_texts: Optional[List[str]]) -> str:
    if not evidence_texts:
        return ""
    return " ".join(str(t) for t in evidence_texts if t)


def compute_citation_metrics(
    text: str,
    verification: Optional[Dict[str, Any]] = None,
    evidence_texts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute mechanical citation recall/precision for one answer.

    ``verification`` is the ``verify_citations`` result (may be None when ADS
    is unavailable — resolution then counts nothing as resolved and precision
    is reported over zero resolved lookups). ``evidence_texts`` are stringified
    tool results/traces for the numeric-support check.
    """
    try:
        sentences = split_sentences(text or "")
        claim_sentences = [s for s in sentences if is_claim_sentence(s)]
        ids = _resolved_citation_ids(verification)
        evidence_blob = _normalize_evidence(evidence_texts)

        supported = 0
        for sentence in claim_sentences:
            citations = extract_citations(sentence)
            sentence_ids = citations["bibcodes"] + citations["dois"]
            if sentence_ids and any(cid in ids["resolved"] for cid in sentence_ids):
                supported += 1
                continue
            numbers = _sentence_numbers(sentence)
            if numbers and evidence_blob:
                matched = sum(1 for n in numbers if n in evidence_blob)
                if matched * 2 >= len(numbers):
                    supported += 1

        total_citations = len(ids["seen"])
        resolved_citations = len(ids["resolved"])
        recall = (supported / len(claim_sentences)) if claim_sentences else None
        precision = (resolved_citations / total_citations) if total_citations else None

        return {
            "method": METRICS_METHOD,
            "sentences": len(sentences),
            "claim_sentences": len(claim_sentences),
            "supported_claim_sentences": supported,
            "citation_recall": round(recall, 4) if recall is not None else None,
            "citations_total": total_citations,
            "citations_resolved": resolved_citations,
            "citation_precision": round(precision, 4) if precision is not None else None,
        }
    except Exception:
        logger.warning("Citation metrics computation failed", exc_info=True)
        return {
            "method": METRICS_METHOD,
            "error": "metrics_failed",
        }


def record_citation_metrics_on_request_context(metrics: Dict[str, Any]) -> None:
    """Attach metrics to the request-scoped LLM context (never raises).

    The SSE layer owns that context object and surfaces the metrics in the
    turn's run_meta (eval mode reads them from there). Channels without an
    accounting context (Telegram/WhatsApp) simply skip this.
    """
    try:
        from core.llm_client import get_llm_request_context

        ctx = get_llm_request_context()
        if ctx is not None:
            ctx.citation_metrics = dict(metrics)
    except Exception:
        logger.debug("Could not record citation metrics on request context", exc_info=True)
