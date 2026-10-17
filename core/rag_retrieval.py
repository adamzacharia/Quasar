"""Documentation retrieval for how-to answers: sub-question queries, junk filter, per-source cap.

A corpus audit (2026-10-16, tmp/almabench-2026-10-15/round4/rag_coverage.md) found that the ALMA documentation corpus
held about three quarters of the points expert answers needed, but the single whole-question search delivered few
of them: the runner injected only the top 3 chunks, one document often filled every slot, near-empty header-only
chunks (253 of 8,328) and base64 image chunks from notebooks (306) took scarce slots, and a multi-part question
retrieved worse than a short query per part. This module fixes the retrieval itself, generically:

* one search for the whole question plus one per sub-question (split on sentence and question boundaries), run
  concurrently;
* junk chunks dropped (body under ``MIN_BODY_CHARS`` after the document header, base64 runs removed first);
* results interleaved across the queries (each sub-question contributes its best chunk before any gets a second),
  de-duplicated, at most ``PER_SOURCE_CAP`` chunks per source document, ``MAX_CHUNKS`` in total.

Sub-questions alone did not lift recall (10 of 54 audited points reached the model either way): the missing passages
sit under words the question does not use. So the sub-queries also include up to ``MAX_SUBQUERIES`` documentation
queries written by Quasar's auxiliary fast model (``QUASAR_DOC_QUERY_MODEL`` / ``QUASAR_FAST_MODEL``, default
gpt-oss-120b, the model the web planner and the paper-intent check already use) for the aspects a complete answer
needs (definitions, procedure, biases and checks, typical values, where the information lives). Bounded by
``QUASAR_DOC_QUERY_TIMEOUT`` (6 s); on timeout or error the sentence split is used alone.

``QUASAR_RAG_MULTI_QUERY=0`` restores the single-query behaviour (top 3), for A/B measurement;
``QUASAR_RAG_QUERY_EXPANSION=0`` keeps multi-query retrieval without the model-written queries.
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

MAX_CHUNKS = 6
PER_SOURCE_CAP = 2
MAX_SUBQUERIES = 4
MIN_BODY_CHARS = 120
_HEADER_RE = re.compile(r"^\[Document:[^\]]*\]\s*")
_BASE64_RE = re.compile(r"[A-Za-z0-9+/=]{200,}")
_SPLIT_RE = re.compile(r"(?<=[?.;])\s+")


def multi_query_enabled() -> bool:
    return os.getenv("QUASAR_RAG_MULTI_QUERY", "1").strip().lower() not in {"0", "false", "no", "off"}


def sub_queries(question: str, max_parts: int = MAX_SUBQUERIES) -> List[str]:
    """The sentences / sub-questions of a multi-part question (at least 4 words each), in order, deduplicated."""
    parts: List[str] = []
    for piece in _SPLIT_RE.split(str(question or "").strip()):
        piece = piece.strip(" \n\t-*")
        if len(piece.split()) >= 4 and piece not in parts:
            parts.append(piece)
    return parts[:max_parts] if len(parts) > 1 else []


_EXPANSION_PROMPT = (
    "You help retrieve passages from ALMA and radio-astronomy documentation (ALMA Technical Handbook, Proposer's "
    "Guide, Science Archive manual, pipeline and CASA documentation) for a user's question. Write up to {n} short "
    "search queries (5 to 12 words each) for the passages a complete expert answer needs: definitions of the key "
    "terms, the procedure, what can bias the result and how to check it, typical values or limits, version or cycle "
    "dependence, and where the information is found (which file, product, web page or tool). Prefer aspects the "
    "question implies but does not name. Output one query per line, no numbering, nothing else.\n\nQuestion: {q}"
)


def expansion_enabled() -> bool:
    return os.getenv("QUASAR_RAG_QUERY_EXPANSION", "1").strip().lower() not in {"0", "false", "no", "off"}


def _default_expand_call(prompt: str) -> str:
    from core.llm_client import LLMClient

    model = (os.getenv("QUASAR_DOC_QUERY_MODEL") or os.getenv("QUASAR_FAST_MODEL") or "gpt-oss-120b").strip()
    resp = LLMClient(model=model).responses.create(model=model, input=prompt, temperature=0, max_output_tokens=1200)
    return str(getattr(resp, "output_text", "") or "")


def parse_expansion(text: str, max_parts: int = MAX_SUBQUERIES) -> List[str]:
    out: List[str] = []
    for line in str(text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
        if 3 <= len(line.split()) <= 20 and len(line) <= 200 and line not in out:
            out.append(line)
    return out[:max_parts]


# At most this many expansion calls in flight process-wide: a timed-out call keeps running on its daemon thread until
# the provider answers, so repeated timeouts must not pile up (guard CX-04). When the slots are busy, no expansion.
MAX_EXPANSIONS_IN_FLIGHT = 2
_EXPANSION_SLOTS = threading.BoundedSemaphore(MAX_EXPANSIONS_IN_FLIGHT)


def _spawn(fn: Callable[[], Any], name: str) -> Tuple[threading.Thread, Dict[str, Any]]:
    """Run ``fn`` on a DAEMON thread that carries the caller's LLM request context (BYOK routing, quota and usage
    accounting live in a thread-local; guard CX-02). Returns the thread and a box with 'value' or 'error'."""
    from core.llm_client import get_llm_request_context, reinstall_llm_request_context

    ctx = get_llm_request_context()
    box: Dict[str, Any] = {}

    def body() -> None:
        try:
            with reinstall_llm_request_context(ctx):
                box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - reported through the box
            box["error"] = exc

    t = threading.Thread(target=body, name=name, daemon=True)
    t.start()
    return t, box


def expand_queries(question: str, call=None, timeout: Optional[float] = None) -> List[str]:
    """Model-written documentation queries for a question; [] on timeout, error, busy slots or when disabled."""
    if not expansion_enabled():
        return []
    budget = float(timeout if timeout is not None else os.getenv("QUASAR_DOC_QUERY_TIMEOUT", "6") or 6)
    if not _EXPANSION_SLOTS.acquire(blocking=False):
        return []
    prompt = _EXPANSION_PROMPT.format(n=MAX_SUBQUERIES, q=str(question)[:1500])

    def job() -> str:
        try:
            return (call or _default_expand_call)(prompt)
        finally:
            _EXPANSION_SLOTS.release()   # released when the call really ends, not when we stop waiting
    t, box = _spawn(job, "rag-expand")
    t.join(budget)
    if t.is_alive() or "error" in box:
        return []
    return parse_expansion(box.get("value") or "")


def clean_chunk_text(text: str) -> Tuple[str, bool]:
    """(text without base64 runs, is_junk). Junk = under MIN_BODY_CHARS of body after the document header."""
    raw = str(text or "")
    stripped = _BASE64_RE.sub(" [binary data removed] ", raw) if _BASE64_RE.search(raw) else raw
    body = _HEADER_RE.sub("", stripped).replace("[binary data removed]", "").strip()
    return stripped, len(body) < MIN_BODY_CHARS


def _source_of(doc: Any) -> str:
    meta = getattr(doc, "metadata", {}) or {}
    return str(meta.get("source_file") or meta.get("source") or meta.get("doc_title") or "unknown")


def _key_of(doc: Any) -> Tuple[str, Any, str]:
    meta = getattr(doc, "metadata", {}) or {}
    return (_source_of(doc), meta.get("page"), str(getattr(doc, "page_content", ""))[:200])


def retrieve(rag_service: Any, question: str, *, min_year: Optional[int] = None, min_score: float = 0.35,
             max_chunks: int = MAX_CHUNKS, per_source_cap: int = PER_SOURCE_CAP,
             expand_call=None) -> Tuple[List[Any], Dict[str, Any]]:
    """Chunks for a how-to answer and the diagnostics of the whole-question search.

    Falls back to the single whole-question search (top 3, the old behaviour) when disabled. A failed sub-query is
    skipped; a failed whole-question search raises as before (the caller handles it)."""
    if not multi_query_enabled():
        docs, diag = rag_service.search_with_diagnostics(question, min_year=min_year, min_score=min_score)
        return list(docs)[:3], dict(diag or {})

    def run(q: str):
        return rag_service.search_with_diagnostics(q, min_year=min_year, min_score=min_score)

    started = time.monotonic()
    budget = float(os.getenv("QUASAR_RAG_RETRIEVAL_BUDGET", "12") or 12)
    main_thread, main_box = _spawn(lambda: run(question), "rag-main")   # starts while the expansion call runs
    expanded = expand_queries(question, call=expand_call)
    queries = [question]
    for q in expanded + sub_queries(question):
        if q not in queries and len(queries) <= 2 * MAX_SUBQUERIES:
            queries.append(q)
    optional = [_spawn(lambda q=q: run(q), "rag-sub") for q in queries[1:]]
    main_thread.join()                      # the whole-question search is the old path: wait for it as before
    if "error" in main_box:
        raise main_box["error"]
    main_docs, diag = main_box["value"]
    ranked: List[List[Any]] = [list(main_docs)]
    late = 0
    for t, box in optional:                 # optional searches share one budget (guard CX-03); late ones are dropped
        t.join(max(0.0, budget - (time.monotonic() - started)))
        if t.is_alive() or "error" in box:
            late += t.is_alive()
            ranked.append([])
        else:
            ranked.append(list(box["value"][0]))

    chosen: List[Any] = []
    seen: set = set()
    per_source: Dict[str, int] = {}
    dropped_junk = 0
    depth = max((len(r) for r in ranked), default=0)
    for i in range(depth):
        for results in ranked:
            if i >= len(results) or len(chosen) >= max_chunks:
                continue
            doc = results[i]
            key = _key_of(doc)
            if key in seen:
                continue
            seen.add(key)
            text, junk = clean_chunk_text(getattr(doc, "page_content", ""))
            if junk:
                dropped_junk += 1
                continue
            src = _source_of(doc)
            if per_source.get(src, 0) >= per_source_cap:
                continue
            per_source[src] = per_source.get(src, 0) + 1
            if text != doc.page_content:
                try:
                    doc.page_content = text
                except Exception:  # noqa: BLE001 - immutable doc: keep the original text
                    pass
            chosen.append(doc)
    diag = dict(diag or {})
    diag["multi_query"] = {"queries": queries, "expanded": expanded, "chunks": len(chosen), "dropped_junk": dropped_junk,
                           "late_subqueries_dropped": late,
                           "sources": sorted(per_source)}
    return chosen, diag
