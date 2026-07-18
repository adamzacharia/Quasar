# services/rag_service.py
"""
RAG Service — Retrieval Augmented Generation for ALMA documentation.

CALLED BY: core/agent.py (context retrieval before every LLM call)
           scripts/ingest_alma_docs.py, scripts/ingest_docs_folder.py
CALLS:     Qdrant Cloud (vector store), OpenAI Embeddings

PURPOSE:
    Ingests PDFs (ALMA Manual, user docs) into chunked vector embeddings
    with rich metadata (year, category, cycle, etc.), then performs
    semantic similarity search with optional metadata filtering to
    provide relevant, up-to-date context to the LLM.

COLLECTIONS (Qdrant):
    alma_general      — Shared ALMA documentation (with metadata)
    user_{id}_personal — Per-user uploaded documents
    proposal_rubrics  — TAC rubric guidelines

METADATA SCHEMA (per chunk in alma_general):
    doc_year       (int)  — Publication/version year (e.g. 2025)
    doc_month      (int)  — Publication month (1-12) when available
    doc_day        (int)  — Publication day (1-31) when available
    doc_category   (str)  — Document category (e.g. "technical_handbook")
    doc_title      (str)  — Human-readable title
    alma_cycle     (str)  — ALMA cycle if detected (e.g. "Cycle 12")
    source_file    (str)  — Original filename
    page           (int)  — Page number within document
    total_pages    (int)  — Total pages in document
    chunk_index    (int)  — Sequential chunk position
    ingested_at    (str)  — ISO timestamp of ingestion
    file_size_kb   (int)  — File size in kilobytes
    is_personal    (bool) — Whether this is a personal document
"""

import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from services.vector_db import (
    upsert_vectors,
    search_vectors,
    scroll_all,
    delete_by_filter,
    delete_collection,
    create_payload_index,
    collection_count,
    ensure_collection,
)
from qdrant_client.models import PayloadSchemaType

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Collection names
# ──────────────────────────────────────────────────────────────────
GENERAL_COLLECTION = "alma_general"
RUBRICS_COLLECTION = "proposal_rubrics"

def _personal_collection(user_id: str) -> str:
    """Return the Qdrant collection name for a user's personal docs."""
    return f"user_{user_id}_personal"


# ──────────────────────────────────────────────────────────────────
# Document category mapping (filename pattern → category)
# ──────────────────────────────────────────────────────────────────
CATEGORY_MAP = {
    "technical-handbook":     "technical_handbook",
    "technical_handbook":     "technical_handbook",
    "proposers-guide":        "proposers_guide",
    "proposers_guide":        "proposers_guide",
    "pipeline":               "pipeline",
    "user-policies":          "user_policies",
    "user_policies":          "user_policies",
    "ot-usermanual":          "observing_tool",
    "ot-refmanual":           "observing_tool",
    "ot-quickstart":          "observing_tool",
    "large-program":          "large_programs",
    "large_program":          "large_programs",
    "archive-primer":         "archive",
    "archive_primer":         "archive",
    "downloading":            "archive",
    "scheduling-blocks":      "scheduling",
    "scheduling_blocks":      "scheduling",
    "snoopi":                 "snoopi",
    "phase2":                 "phase2",
    "review-process":         "review_process",
    "review_process":         "review_process",
    "principles-review":      "review_process",
    "tap_columns":            "archive",
    "tap-columns":            "archive",
    "reference-manual":       "reference",
    "reference_manual":       "reference",
    "known-issues":           "pipeline",
    "known_issues":           "pipeline",
    "rlm":                    "internal",
    "mem0":                   "internal",
}

# Smalltalk / off-domain patterns — fast reject
_OFFDOMAIN_PATTERNS = re.compile(
    r'^\s*(?:'
    r'(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|howdy|yo)\b'
    r'|(?:tell\s+me\s+a\s+joke|make\s+me\s+laugh|something\s+funny)'
    r'|(?:thanks?|thank\s+you|thx|cheers|great|ok|okay|cool|nice|awesome|perfect|got\s+it)'
    r'|(?:bye|goodbye|see\s+you|later|ciao)'
    r'|(?:who\s+are\s+you|what\s+(?:are|can)\s+you|your\s+name)'
    r'|(?:what\s+(?:is|are)\s+\d+\s*[\+\-\*/x×÷]\s*\d+)'  # basic arithmetic
    r'|(?:write\s+(?:a\s+)?(?:poem|song|story|essay|haiku))'
    r'|(?:translate\s+.+\s+(?:to|into)\s+\w+)'
    r')\b',
    re.IGNORECASE,
)

# Curated domain keywords — fast accept
_DOMAIN_KEYWORDS = {
    "alma", "band", "frequency", "calibration", "correlator", "antenna",
    "baseline", "spectral", "continuum", "imaging", "pipeline", "casa",
    "interferometry", "receiver", "sensitivity", "proposal", "proprietary",
    "archive", "cycle", "mosaic", "polarization", "flux", "beam",
    "spectral window", "spw", "bandwidth", "scheduling", "phase",
    "technical handbook", "vla", "vlba", "gbt", "radio", "submillimeter",
    "millimeter", "ghz", "mhz", "jy", "arcsec", "fits", "measurement set",
    "uvfits", "tclean", "galaxy", "quasar", "pulsar", "nebula", "star",
    "planet", "redshift", "luminosity", "magnitude", "photometry",
    "spectroscopy", "emission", "absorption", "telescope", "observatory",
    "observation", "survey", "catalog", "astrometry", "cosmology",
    "dark matter", "dark energy", "supernova", "black hole", "exoplanet",
    "protoplanetary", "molecular cloud", "interstellar", "circumstellar",
    "agn", "smbh", "ism", "igm", "cmb", "h2", "co ", "hcn", "sio",
    "jwst", "hst", "hubble", "chandra", "xmm", "spitzer", "herschel",
    "noema", "iram", "jcmt", "sofia", "ska", "lofar", "meerkat",
    "atacama", "eso", "nasa", "esa",
}

# Broad/ambiguous keywords that also appear in everyday non-astronomy text
# ("rock band", "submit a proposal", "phase margin", "radio buttons"). These count
# as domain-relevant ONLY when the query is not a smalltalk/off-domain stem.
_AMBIGUOUS_KEYWORDS = {
    "band", "proposal", "phase", "radio", "star", "survey", "catalog", "beam",
    "flux", "planet", "cycle", "mosaic", "archive", "scheduling", "pipeline",
    "imaging", "sensitivity", "continuum", "observation", "bandwidth",
}

def is_domain_relevant(query: str) -> bool:
    """Fast, zero-LLM-call domain relevance check.

    Tiered to satisfy both failure modes: a real science question that opens with a
    conversational stem ("what can you tell me about ALMA Band 6") must pass, while a
    non-astronomy prompt that merely contains a broad word ("my rock band", "phase
    margin") must not.
    """
    q_lower = query.lower()
    strong = _DOMAIN_KEYWORDS - _AMBIGUOUS_KEYWORDS
    # 1. Strong, unambiguous astronomy keyword → accept even after a conversational stem.
    if any(kw in q_lower for kw in strong):
        return True
    # 2. Off-domain / smalltalk stem → reject. Runs BEFORE the broad-keyword check so
    #    ambiguous words (band/phase/radio/...) can't rescue a clearly off-domain prompt.
    if _OFFDOMAIN_PATTERNS.match(query):
        return False
    # 3. Broad/ambiguous keyword with no off-domain stem → accept.
    if any(kw in q_lower for kw in _AMBIGUOUS_KEYWORDS):
        return True
    # 4. Ambiguous — default to False (conservative).
    return False

# ALMA cycle → approximate year mapping
CYCLE_YEAR_MAP = {
    "1": 2013, "2": 2014, "3": 2015, "4": 2016, "5": 2017,
    "6": 2018, "7": 2019, "8": 2020, "9": 2021, "10": 2022,
    "11": 2023, "12": 2024, "13": 2025, "14": 2026,
}

_SEMANTIC_TIE_TOLERANCE = 0.005


def _coerce_doc_year(value: Any) -> Optional[int]:
    """Return a usable doc_year int, or None for missing/invalid values."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        match = re.search(r'\b(20[1-3]\d)\b', value)
        if match:
            return int(match.group(1))
    return None


def _cycle_label_for_year(year: int, docs: List[Document]) -> str:
    """Format a year as an ALMA cycle label when possible."""
    for doc in docs:
        metadata = doc.metadata or {}
        if _coerce_doc_year(metadata.get("doc_year")) == year:
            alma_cycle = str(metadata.get("alma_cycle") or "").strip()
            if alma_cycle:
                return f"{alma_cycle} ({year})"

    for cycle, mapped_year in CYCLE_YEAR_MAP.items():
        if mapped_year == year:
            return f"Cycle {cycle} ({year})"
    return str(year)


def detect_year_conflicts(
    docs: List[Document],
    span_threshold: int = 2,
) -> Optional[Dict[str, Any]]:
    """Detect stale/current ALMA documentation mixed in returned chunks."""
    years = sorted({
        year
        for doc in docs
        for year in [_coerce_doc_year((doc.metadata or {}).get("doc_year"))]
        if year is not None
    })
    if not years:
        return None

    min_year = years[0]
    max_year = years[-1]
    span = max_year - min_year
    if span < span_threshold:
        return None

    min_label = _cycle_label_for_year(min_year, docs)
    max_label = _cycle_label_for_year(max_year, docs)
    return {
        "min_year": min_year,
        "max_year": max_year,
        "span": span,
        "years": years,
        "message": (
            f"Retrieved ALMA docs span {min_label} to {max_label}; "
            "newer specs may supersede older ones."
        ),
    }


def _build_filter_conditions(
    *,
    year: Optional[int] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    category: Optional[str] = None,
    source_file: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
    """Build exact-match and range condition dicts for vector search."""
    filter_conditions: Dict[str, str] = {}
    range_conditions: Dict[str, Dict[str, int]] = {}

    if category:
        filter_conditions["doc_category"] = category
    if source_file:
        filter_conditions["source_file"] = source_file

    if year is not None:
        range_conditions["doc_year"] = {"gte": year, "lte": year}
    elif min_year is not None or max_year is not None:
        year_range: Dict[str, int] = {}
        if min_year is not None:
            year_range["gte"] = min_year
        if max_year is not None:
            year_range["lte"] = max_year
        range_conditions["doc_year"] = year_range

    return filter_conditions, range_conditions


def _rank_scores(
    scores: List[float],
    *,
    reverse: bool = True,
    tolerance: float = 0.0,
) -> Dict[int, int]:
    """Return RRF-compatible ranks, preserving ties within tolerance.

    Ties are measured against the first score in the current tie group (the
    group "leader"), not the immediately preceding score. Comparing to the
    previous score makes ties transitive — a run of small adjacent steps
    (e.g. 0.900, 0.896, 0.892) could collapse into one rank even though the
    endpoints differ by more than ``tolerance``. Leader-relative grouping
    prevents that chaining.
    """
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=reverse)
    ranks: Dict[int, int] = {}
    leader_score: Optional[float] = None
    current_rank = 0

    for position, idx in enumerate(ranked):
        score = float(scores[idx])
        if leader_score is None or abs(score - leader_score) > tolerance:
            current_rank = position
            leader_score = score
        ranks[idx] = current_rank

    return ranks


def _build_recency_rank(
    candidates: List[Document],
    current_year: Optional[int] = None,
) -> Dict[int, int]:
    """Rank candidates by doc_year recency, with missing years last."""
    years_by_index = {
        idx: year
        for idx, doc in enumerate(candidates)
        for year in [_coerce_doc_year((doc.metadata or {}).get("doc_year"))]
        if year is not None
    }
    if not years_by_index:
        return {idx: 0 for idx in range(len(candidates))}

    if current_year is None:
        current_year = max(years_by_index.values())

    unique_years = sorted(
        set(years_by_index.values()),
        key=lambda year: (max(current_year - year, 0), -year),
    )
    ranks_by_year = {year: rank for rank, year in enumerate(unique_years)}
    missing_rank = len(unique_years)
    return {
        idx: ranks_by_year.get(years_by_index.get(idx), missing_rank)
        for idx in range(len(candidates))
    }


def _build_citation_rank(candidates: List[Document]) -> Dict[int, int]:
    """Rank candidates by doc_citations (OpenAlex cited_by_count), missing last.

    R4 (Pathfinder-style citation weighting): mirrors _build_recency_rank —
    rank 0 is the most-cited distinct count; chunks without a citation count
    rank after every chunk that has one.
    """
    counts_by_index: Dict[int, int] = {}
    for idx, doc in enumerate(candidates):
        raw = (doc.metadata or {}).get("doc_citations")
        try:
            if raw is not None and str(raw).strip() != "":
                counts_by_index[idx] = max(0, int(raw))
        except (TypeError, ValueError):
            continue
    if not counts_by_index:
        return {idx: 0 for idx in range(len(candidates))}

    unique_counts = sorted(set(counts_by_index.values()), reverse=True)
    ranks_by_count = {count: rank for rank, count in enumerate(unique_counts)}
    missing_rank = len(unique_counts)
    return {
        idx: ranks_by_count.get(counts_by_index.get(idx), missing_rank)
        for idx in range(len(candidates))
    }


# Query-conditional weighting (Pathfinder's approach): boost the recency or
# citation tiebreaker only when the query itself signals that intent. The
# boost stays inside the bounded tiebreak budget, so relevance always wins.
_RECENCY_INTENT_RE = re.compile(
    r"\b(recent|latest|newest|new(est)?\s+results?|current|up[- ]to[- ]date|"
    r"this\s+year|last\s+year|202[4-9]|203\d)\b",
    re.IGNORECASE,
)
_AUTHORITY_INTENT_RE = re.compile(
    r"\b(seminal|foundational|landmark|classic|influential|important|"
    r"highly[- ]cited|most[- ]cited|best[- ]known|canonical)\b",
    re.IGNORECASE,
)
_INTENT_BOOST = 3.0


def _ranking_intent_multipliers(query: str) -> Tuple[float, float]:
    """Return (recency_multiplier, citation_multiplier) for a query."""
    text = query or ""
    recency_mult = _INTENT_BOOST if _RECENCY_INTENT_RE.search(text) else 1.0
    citation_mult = _INTENT_BOOST if _AUTHORITY_INTENT_RE.search(text) else 1.0
    return recency_mult, citation_mult


# ──────────────────────────────────────────────────────────────────
# Metadata extraction helpers
# ──────────────────────────────────────────────────────────────────

def _extract_year_from_filename(filename: str) -> Optional[int]:
    """Try to extract a year from the filename.

    Matches patterns like: _2025.pdf, -2025., 2025_, (2025)
    """
    # Match 4-digit year in filename
    matches = re.findall(r'(?:_|-|\b)(20[1-3]\d)(?:_|-|\.|\b)', filename)
    if matches:
        return int(matches[-1])  # Take last match (more likely to be version year)
    return None


def _extract_date_from_pdf_metadata(file_path: str) -> Dict[str, Optional[int]]:
    """Extract year, month, day from PDF internal metadata (CreationDate, ModDate).

    Returns:
        Dict with keys 'year', 'month', 'day' (any may be None).
    """
    result: Dict[str, Optional[int]] = {"year": None, "month": None, "day": None}
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        info = doc.metadata
        doc.close()
        if not info:
            return result

        # Try modDate first (more likely to reflect the version), then creationDate
        for field in ["modDate", "creationDate"]:
            val = info.get(field, "")
            if val:
                # PDF date format: D:YYYYMMDDHHmmSS+TZ or just raw string
                date_match = re.search(r'D:(\d{4})(\d{2})(\d{2})', str(val))
                if date_match:
                    result["year"] = int(date_match.group(1))
                    result["month"] = int(date_match.group(2))
                    result["day"] = int(date_match.group(3))
                    return result
                # Fallback: just extract year
                year_match = re.search(r'(20[1-3]\d)', str(val))
                if year_match:
                    result["year"] = int(year_match.group(1))
                    return result
    except Exception:
        pass
    return result


def _extract_title_from_pdf(file_path: str, filename: str) -> str:
    """Extract a human-readable title from PDF metadata or filename."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        info = doc.metadata
        doc.close()
        if info:
            title = info.get("title", "")
            if title and len(title) > 3 and not title.startswith("Microsoft"):
                return str(title).strip()
    except Exception:
        pass

    # Fallback: clean the filename into a title
    name = os.path.splitext(filename)[0]
    name = re.sub(r'[-_]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name.title()


def _extract_year_from_content(pages_text: List[str], max_pages: int = 5) -> Optional[int]:
    """Scan the first N pages for year references."""
    for page_text in pages_text[:max_pages]:
        # Look for "Cycle N" references first
        cycle_match = re.search(r'Cycle\s+(\d{1,2})', page_text, re.IGNORECASE)
        if cycle_match:
            cycle_num = cycle_match.group(1)
            if cycle_num in CYCLE_YEAR_MAP:
                return CYCLE_YEAR_MAP[cycle_num]

        # Look for standalone years in context like "2025 edition", "Version 2025"
        year_match = re.search(
            r'(?:version|edition|release|dated?|updated?|copyright|©)\s*:?\s*(20[1-3]\d)',
            page_text,
            re.IGNORECASE,
        )
        if year_match:
            return int(year_match.group(1))

    return None


_DOI_RE = re.compile(r'\b(10\.\d{4,9}/[^\s"<>\)\]]+)', re.IGNORECASE)


def _extract_doi_from_content(pages_text: List[str], max_pages: int = 3) -> Optional[str]:
    """Scan the first pages for a DOI (R4: enables citation-count enrichment)."""
    for page_text in pages_text[:max_pages]:
        match = _DOI_RE.search(page_text or "")
        if match:
            # Strip common trailing punctuation picked up by the greedy tail.
            return match.group(1).rstrip(".,;")
    return None


def _detect_alma_cycle(pages_text: List[str], max_pages: int = 10) -> Optional[str]:
    """Detect the ALMA Cycle referenced in the document."""
    for page_text in pages_text[:max_pages]:
        match = re.search(r'Cycle\s+(\d{1,2})', page_text, re.IGNORECASE)
        if match:
            return f"Cycle {match.group(1)}"
    return None


def _classify_category(filename: str) -> str:
    """Map a filename to a document category."""
    fn_lower = filename.lower()
    for pattern, category in CATEGORY_MAP.items():
        if pattern in fn_lower:
            return category
    return "general"


def extract_document_metadata(file_path: str, pages_text: Optional[List[str]] = None, original_filename: Optional[str] = None) -> Dict[str, Any]:
    """Extract rich metadata from a document file.

    Args:
        file_path:  Full path to the document.
        pages_text: Optional list of page texts (to avoid re-reading).
        original_filename: Optional original filename if file_path is temporary.

    Returns:
        Dict with: doc_year, doc_category, doc_title, alma_cycle,
                   file_size_kb, total_pages, ingested_at.
    """
    filename = original_filename or os.path.basename(file_path)

    # Date detection (priority: filename → PDF metadata → content scan)
    year = _extract_year_from_filename(filename)
    month = None
    day = None

    # Try PDF metadata for full date (year + month + day)
    pdf_date = _extract_date_from_pdf_metadata(file_path)
    if year is None:
        year = pdf_date["year"]
    month = pdf_date["month"]
    day = pdf_date["day"]

    if year is None and pages_text:
        year = _extract_year_from_content(pages_text)
    if year is None:
        # Last resort: file modification time
        try:
            mtime = os.path.getmtime(file_path)
            dt = datetime.fromtimestamp(mtime)
            year = dt.year
            if month is None:
                month = dt.month
            if day is None:
                day = dt.day
        except Exception:
            year = 2024  # Safe default

    # Category
    category = _classify_category(filename)

    # Title
    title = _extract_title_from_pdf(file_path, filename)

    # ALMA Cycle
    alma_cycle = None
    if pages_text:
        alma_cycle = _detect_alma_cycle(pages_text)

    # File size
    try:
        file_size_kb = int(os.path.getsize(file_path) / 1024)
    except Exception:
        file_size_kb = 0

    # DOI (R4): enables best-effort OpenAlex citation enrichment at ingest.
    doc_doi = _extract_doi_from_content(pages_text) if pages_text else None

    return {
        "doc_year": year,
        "doc_month": month,
        "doc_day": day,
        "doc_category": category,
        "doc_title": title,
        "alma_cycle": alma_cycle or "",
        "doc_doi": doc_doi or "",
        "file_size_kb": file_size_kb,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


# ──────────────────────────────────────────────────────────────────
# RAG Service
# ──────────────────────────────────────────────────────────────────

class RAGService:
    """
    Service for Retrieval Augmented Generation with Personal Collections

    Supports:
    - General RAG (shared ALMA documentation with rich metadata)
    - Personal RAG (user-specific documents)
    - Metadata-filtered search (year, category, source)
    - Progress callbacks for UI
    - Combined search across both collections
    """

    def __init__(self, persist_directory: str = None, user_id: str = None):
        # persist_directory kept for backward-compat but now ignored
        # (all storage is in Qdrant Cloud)
        self.user_id = user_id
        self.embeddings = OpenAIEmbeddings()

        # Collection names
        self.general_collection = GENERAL_COLLECTION
        self.personal_collection = _personal_collection(user_id) if user_id else None
        self.rubrics_collection = RUBRICS_COLLECTION

        # Ensure collections exist
        ensure_collection(self.general_collection)
        if self.personal_collection:
            ensure_collection(self.personal_collection)
        ensure_collection(self.rubrics_collection)

    # ------------------------------------------------------------------
    # Document loading
    # ------------------------------------------------------------------

    def _ocr_pdf(self, file_path: str) -> List[Document]:
        """OCR a scanned PDF using GPT-4o vision transcription."""
        import fitz
        import base64
        from openai import OpenAI

        print(f"[RAG] Attempting OCR vision transcription for {file_path}")
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            print("[RAG] No OPENAI_API_KEY found, cannot perform OCR.")
            return []

        client = OpenAI(api_key=openai_key)
        documents = []

        try:
            doc = fitz.open(file_path)
            for page_idx, page in enumerate(doc):
                # Render page to an image
                # 150 DPI is a good balance between speed/cost and quality
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                base64_image = base64.b64encode(img_bytes).decode("utf-8")

                # Call GPT-4o to transcribe the page
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a precise document transcription engine. "
                                "Transcribe all text from the provided document page image exactly as it appears. "
                                "Maintain the logical structure. Do not summarize. If there is no text, reply with nothing."
                            )
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=2000,
                    temperature=0.0,
                )

                transcribed_text = response.choices[0].message.content or ""
                if transcribed_text.strip():
                    documents.append(Document(
                        page_content=transcribed_text,
                        metadata={
                            "page": page_idx,
                            "source": file_path
                        }
                    ))
            doc.close()
            print(f"[RAG] Successfully OCR'd {len(documents)} pages from {file_path}")
            return documents
        except Exception as e:
            print(f"[RAG] OCR fallback failed for {file_path}: {e}")
            return []

    def _load_document(self, file_path: str) -> List[Document]:
        """Load document based on file type.

        Uses PyMuPDF4LLM for PDFs — significantly better at extracting text
        from scientific documents with multi-column layouts, equations, and tables.
        Falls back to GPT-4o OCR if no readable text can be extracted.
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            try:
                loader = PyMuPDF4LLMLoader(file_path)
                docs = loader.load()
                # Check if it has any readable text
                total_len = sum(len(d.page_content.strip()) for d in docs)
                if total_len < 50:
                    print(f"[RAG] PDF '{file_path}' has very little text ({total_len} chars). Triggering OCR vision fallback...")
                    ocr_docs = self._ocr_pdf(file_path)
                    if ocr_docs:
                        return ocr_docs
                return docs
            except Exception as e:
                print(f"[RAG] PyMuPDF4LLM failed to load '{file_path}': {e}. Triggering OCR vision fallback...")
                ocr_docs = self._ocr_pdf(file_path)
                if ocr_docs:
                    return ocr_docs
                raise
        elif ext in ['.txt', '.md']:
            loader = TextLoader(file_path, encoding='utf-8')
            return loader.load()
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # ------------------------------------------------------------------
    # Ingestion (with rich metadata)
    # ------------------------------------------------------------------

    def ingest_document(
        self,
        file_path: str,
        personal: bool = False,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        original_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a single document into the vector store with rich metadata.

        Args:
            file_path: Path to the document
            personal: If True, add to personal collection
            progress_callback: Function(status_msg, percent) for progress updates
            extra_metadata: Optional additional metadata to merge into each chunk
            original_filename: Optional original filename if file_path is temporary

        Returns:
            Dict with success status, chunk count, extracted metadata, etc.
        """
        try:
            filename = original_filename or os.path.basename(file_path)

            # Step 1: Load document
            if progress_callback:
                progress_callback(f"Loading {filename}...", 10)
            documents = self._load_document(file_path)

            # Step 2: Extract rich metadata
            if progress_callback:
                progress_callback(f"Extracting metadata from {filename}...", 20)

            pages_text = [doc.page_content for doc in documents]
            doc_meta = extract_document_metadata(file_path, pages_text, original_filename=original_filename)
            total_pages = len(documents)
            doc_meta["total_pages"] = total_pages

            # R4: best-effort OpenAlex citation-count enrichment (needs a DOI).
            # Non-fatal by design — ingest must never fail on a metadata lookup.
            if doc_meta.get("doc_doi") and os.getenv("QUASAR_RAG_CITATION_ENRICH", "1") != "0":
                try:
                    from integrations.openalex_client import OpenAlexClient
                    enrichment = OpenAlexClient().enrich_by_doi(doc_meta["doc_doi"])
                    if enrichment and enrichment.get("cited_by_count") is not None:
                        doc_meta["doc_citations"] = int(enrichment["cited_by_count"])
                except Exception as _cite_err:
                    logger.debug("OpenAlex citation enrichment skipped: %s", _cite_err)

            # Step 3: Split into chunks
            if progress_callback:
                progress_callback(f"Splitting {total_pages} pages...", 30)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)

            # Guard: skip documents with no extractable text (e.g. scanned PDFs)
            if not chunks:
                if progress_callback:
                    progress_callback(f"⚠ No text could be extracted from {filename}", 100)
                return {
                    "success": False,
                    "error": "No text could be extracted (possibly a scanned/image-only PDF)",
                    "filename": filename,
                }

            # Step 4: Create embeddings and upsert (in batches to avoid API limits)
            if progress_callback:
                progress_callback(f"Creating embeddings for {len(chunks)} chunks...", 50)

            # Choose target collection
            if personal and self.personal_collection:
                target = self.personal_collection
            else:
                target = self.general_collection

            # Embed in batches of EMBED_BATCH_SIZE to avoid OpenAI rate limits
            import time
            EMBED_BATCH_SIZE = 100
            texts = [chunk.page_content for chunk in chunks]
            vectors = []
            for batch_start in range(0, len(texts), EMBED_BATCH_SIZE):
                batch_end = min(batch_start + EMBED_BATCH_SIZE, len(texts))
                batch_texts = texts[batch_start:batch_end]
                if progress_callback:
                    pct = 50 + int(30 * batch_end / len(texts))
                    progress_callback(
                        f"Embedding batch {batch_start//EMBED_BATCH_SIZE + 1} "
                        f"({batch_end}/{len(texts)} chunks)...", pct
                    )
                # Retry up to 3 times for transient API errors
                for attempt in range(3):
                    try:
                        batch_vectors = self.embeddings.embed_documents(batch_texts)
                        vectors.extend(batch_vectors)
                        break
                    except Exception as embed_err:
                        if attempt < 2:
                            wait = 2 ** (attempt + 1)
                            print(f"[RAG] Embedding batch failed (attempt {attempt+1}), "
                                  f"retrying in {wait}s: {embed_err}")
                            time.sleep(wait)
                        else:
                            raise

            # Build IDs and payloads with rich metadata
            ids = [str(uuid.uuid4()) for _ in chunks]
            payloads = []
            for idx, chunk in enumerate(chunks):
                pay = {
                    "text": chunk.page_content,
                    "source_file": filename,
                    "is_personal": personal,
                    "chunk_index": idx,
                    # Rich metadata fields
                    "doc_year": doc_meta["doc_year"],
                    "doc_month": doc_meta.get("doc_month"),
                    "doc_day": doc_meta.get("doc_day"),
                    "doc_category": doc_meta["doc_category"],
                    "doc_title": doc_meta["doc_title"],
                    "alma_cycle": doc_meta["alma_cycle"],
                    "total_pages": doc_meta["total_pages"],
                    "file_size_kb": doc_meta["file_size_kb"],
                    "ingested_at": doc_meta["ingested_at"],
                }
                if self.user_id and personal:
                    pay["user_id"] = self.user_id
                # Merge extra metadata if provided
                if extra_metadata:
                    for k, v in extra_metadata.items():
                        if k not in pay:
                            pay[k] = v
                # Carry over page number and other loader metadata
                for k, v in chunk.metadata.items():
                    if k not in pay:
                        pay[k] = str(v) if not isinstance(v, (str, int, float, bool)) else v
                payloads.append(pay)

            # Step 5: Upsert to Qdrant
            if progress_callback:
                progress_callback(f"Storing {len(chunks)} chunks in Qdrant...", 80)

            upsert_vectors(target, ids, vectors, payloads)

            if progress_callback:
                progress_callback(f"✓ Ingested {filename}", 100)

            return {
                "success": True,
                "filename": filename,
                "pages": total_pages,
                "chunks": len(chunks),
                "personal": personal,
                "metadata": doc_meta,
            }

        except Exception as e:
            if progress_callback:
                progress_callback(f"✗ Failed: {str(e)}", 0)
            return {
                "success": False,
                "error": str(e),
                "filename": os.path.basename(file_path)
            }

    def ingest_directory(self, directory: str, extensions: List[str] = ['.pdf', '.txt']) -> Dict[str, bool]:
        """Ingest all supported documents from a directory (general collection)"""
        results = {}

        for filename in os.listdir(directory):
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions:
                file_path = os.path.join(directory, filename)
                result = self.ingest_document(file_path, personal=False)
                results[filename] = result.get("success", False)

        success = sum(1 for v in results.values() if v)
        print(f"\n=== Ingestion Complete: {success}/{len(results)} documents ===")
        return results

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def wipe_general_collection(self) -> bool:
        """Delete and recreate the general collection (for clean re-ingestion)."""
        deleted = delete_collection(self.general_collection)
        ensure_collection(self.general_collection)
        return deleted

    def create_metadata_indexes(self):
        """Create payload indexes on frequently filtered metadata fields."""
        try:
            create_payload_index(
                self.general_collection, "doc_year", PayloadSchemaType.INTEGER
            )
        except Exception as e:
            print(f"[RAG] Index on doc_year may already exist: {e}")

        try:
            create_payload_index(
                self.general_collection, "doc_month", PayloadSchemaType.INTEGER
            )
        except Exception as e:
            print(f"[RAG] Index on doc_month may already exist: {e}")

        try:
            create_payload_index(
                self.general_collection, "doc_category", PayloadSchemaType.KEYWORD
            )
        except Exception as e:
            print(f"[RAG] Index on doc_category may already exist: {e}")

        try:
            create_payload_index(
                self.general_collection, "source_file", PayloadSchemaType.KEYWORD
            )
        except Exception as e:
            print(f"[RAG] Index on source_file may already exist: {e}")

    # ------------------------------------------------------------------
    # Search (with metadata filtering)
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
        include_personal: bool = True,
        year: Optional[int] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        category: Optional[str] = None,
        source_file: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[Document]:
        """
        Hybrid search: semantic (Qdrant) + keyword (BM25) reranking.

        1. Over-fetches 4× from Qdrant using dense vector similarity.
        2. Applies BM25 keyword scoring on the returned chunks.
        3. Combines both rankings via Reciprocal Rank Fusion (RRF).
        4. Returns the top-k results.

        This catches exact acronym/keyword matches that pure embedding
        similarity misses (e.g., "SB execution fraction" vs "scheduling
        block execution fraction") with zero re-ingestion.

        Args:
            query: Search query
            k: Number of final results to return
            include_personal: Whether to include personal docs
            year: Exact year filter (e.g. 2025)
            min_year: Minimum year (inclusive, e.g. 2024)
            max_year: Maximum year (inclusive, e.g. 2025)
            category: Document category filter (e.g. "technical_handbook")
            source_file: Filter by specific source filename

        Returns:
            Combined list of relevant documents with metadata
        """
        docs, _diagnostics = self.search_with_diagnostics(
            query,
            k=k,
            include_personal=include_personal,
            year=year,
            min_year=min_year,
            max_year=max_year,
            category=category,
            source_file=source_file,
            min_score=min_score,
        )
        return docs

    def search_with_diagnostics(
        self,
        query: str,
        k: int = 5,
        include_personal: bool = True,
        year: Optional[int] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        category: Optional[str] = None,
        source_file: Optional[str] = None,
        min_score: float = 0.0,
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """Hybrid search with structured diagnostics for freshness warnings."""
        # Over-fetch factor: retrieve more candidates for BM25 reranking
        fetch_k = k * 4

        results = []
        query_vector = self.embeddings.embed_query(query)

        filter_conditions, range_conditions = _build_filter_conditions(
            year=year,
            min_year=min_year,
            max_year=max_year,
            category=category,
            source_file=source_file,
        )

        # Search general collection (over-fetch for reranking)
        try:
            general_hits = search_vectors(
                self.general_collection,
                query_vector,
                limit=fetch_k,
                filter_conditions=filter_conditions or None,
                range_conditions=range_conditions or None,
            )
            for hit in general_hits:
                meta = {kk: vv for kk, vv in hit["payload"].items() if kk != "text"}
                meta["_semantic_score"] = round(hit["score"], 4)
                doc = Document(
                    page_content=hit["payload"].get("text", ""),
                    metadata=meta,
                )
                results.append(doc)
        except Exception as e:
            print(f"General search failed: {e}")

        # Search personal collection (no metadata filters — personal docs are unstructured)
        if include_personal and self.personal_collection:
            try:
                personal_hits = search_vectors(
                    self.personal_collection,
                    query_vector,
                    limit=fetch_k,
                )
                for hit in personal_hits:
                    meta = {kk: vv for kk, vv in hit["payload"].items() if kk != "text"}
                    meta["_semantic_score"] = round(hit["score"], 4)
                    doc = Document(
                        page_content=hit["payload"].get("text", ""),
                        metadata=meta,
                    )
                    results.append(doc)
            except Exception as e:
                print(f"Personal search failed: {e}")

        # Apply hybrid BM25 reranking if we have enough candidates
        if len(results) > k:
            results = self._hybrid_rerank(query, results, k)
        else:
            # Not enough candidates to rerank — just set _score = _semantic_score
            for doc in results:
                doc.metadata["_score"] = doc.metadata.get("_semantic_score", 0.0)

        # Filter out vectors below the semantic relevance threshold
        if min_score > 0.0:
            results = [d for d in results if d.metadata.get("_semantic_score", 0.0) >= min_score]

        year_conflict = detect_year_conflicts(results)
        if year_conflict:
            for doc in results:
                doc.metadata["_year_conflict"] = year_conflict
        else:
            for doc in results:
                doc.metadata.pop("_year_conflict", None)

        return results, {"year_conflict": year_conflict}

    @staticmethod
    def _hybrid_rerank(
        query: str,
        candidates: List[Document],
        k: int,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        rrf_k: int = 60,
        *,
        recency_weight: float = 0.1,
        citation_weight: float = 0.1,
        current_year: Optional[int] = None,
    ) -> List[Document]:
        """Rerank candidates using Reciprocal Rank Fusion (semantic + BM25 + recency + citations).

        Combines semantic similarity rank (from Qdrant) with BM25 keyword
        relevance rank. Uses RRF: score = Σ (weight / (rrf_k + rank)).

        Args:
            query:           The user's search query.
            candidates:      Over-fetched Document list with _semantic_score.
            k:               Number of results to return after reranking.
            semantic_weight: Weight for the semantic (embedding) signal.
            bm25_weight:     Weight for the BM25 (keyword) signal.
            rrf_k:           RRF smoothing constant (standard is 60).
            recency_weight:  Small supplemental weight for doc_year recency.
            citation_weight: Small supplemental weight for doc_citations
                             (OpenAlex cited_by_count; R4 Pathfinder-style).
            current_year:    Optional current ALMA doc year; defaults to the
                             maximum candidate doc_year.

        Both supplemental weights are query-conditional: recency-intent
        queries ("latest", "recent") boost the recency side and
        authority-intent queries ("seminal", "highly cited") the citation
        side, but the combined bonus stays strictly inside the smallest gap
        between distinct relevance scores — relevance always wins.

        Returns:
            Top-k documents sorted by combined RRF score.
        """
        if not candidates:
            return []

        semantic_scores = [
            float(doc.metadata.get("_semantic_score", 0) or 0)
            for doc in candidates
        ]
        semantic_rank = _rank_scores(
            semantic_scores,
            reverse=True,
            tolerance=_SEMANTIC_TIE_TOLERANCE,
        )

        bm25_available = False
        bm25_scores = [0.0 for _ in candidates]
        bm25_rank = {idx: 0 for idx in range(len(candidates))}

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            # Optional dependency: keep search working with semantic + recency.
            pass
        else:
            bm25_available = True
            # Tokenize for BM25 (simple whitespace + lowercase)
            tokenized_corpus = [
                doc.page_content.lower().split() for doc in candidates
            ]
            tokenized_query = query.lower().split()

            # Build BM25 index over the candidate set
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = [float(score) for score in bm25.get_scores(tokenized_query)]
            bm25_rank = _rank_scores(bm25_scores, reverse=True, tolerance=1e-12)

        recency_rank = _build_recency_rank(candidates, current_year=current_year)
        citation_rank = _build_citation_rank(candidates)

        # Relevance = semantic + BM25 RRF. Recency/citations are deliberately
        # NOT free additive RRF terms: as one, a newer-but-less-relevant chunk
        # could outscore a stronger semantic match (their range can exceed the
        # gap between adjacent relevance ranks). Instead, bound the combined
        # bonus to be strictly smaller than the smallest gap between *distinct*
        # relevance scores, so these signals can only reorder candidates whose
        # relevance is (near-)identical — never override a real relevance
        # difference.
        relevance = [
            semantic_weight / (rrf_k + semantic_rank[i])
            + bm25_weight / (rrf_k + bm25_rank[i])
            for i in range(len(candidates))
        ]
        distinct_relevance = sorted(set(relevance))
        min_gap = min(
            (hi - lo for lo, hi in zip(distinct_relevance, distinct_relevance[1:])),
            default=0.0,
        )
        # The weights are fractions of the minimum relevance gap the tiebreaker
        # may use (default 0.1 each). Query intent can boost one side, and the
        # combined budget is capped below the gap so the invariant holds at any
        # weight. When all relevance scores are equal (min_gap == 0) there are
        # no groups to cross, so a tiny absolute span is enough to order ties.
        recency_mult, citation_mult = _ranking_intent_multipliers(query)
        rec_w = max(0.0, recency_weight) * recency_mult
        cit_w = max(0.0, citation_weight) * citation_mult
        # A flat rank list carries no signal (all chunks share one rank —
        # e.g. no chunk has a citation count): drop that side entirely so it
        # neither eats the tiebreak budget nor adds a constant offset.
        if len(set(recency_rank.values())) <= 1:
            rec_w = 0.0
        if len(set(citation_rank.values())) <= 1:
            cit_w = 0.0
        total_w = rec_w + cit_w
        # The bonus budget is the weight-sum fraction of the gap (capped below
        # it), normalized over the active weights — with recency alone at the
        # 0.1 default this reduces exactly to the previous recency-only span.
        budget = (min(0.9, total_w) * min_gap) if min_gap > 0 else 1e-6
        max_recency_rank = max(recency_rank.values(), default=0) or 1
        max_citation_rank = max(citation_rank.values(), default=0) or 1

        rrf_scores = []
        for i in range(len(candidates)):
            # newest / most-cited (rank 0) → full factor; oldest/least → ~0.
            recency_factor = 1 - recency_rank[i] / max_recency_rank
            citation_factor = 1 - citation_rank[i] / max_citation_rank
            if total_w > 0:
                bonus = budget * (rec_w * recency_factor + cit_w * citation_factor) / total_w
            else:
                bonus = 0.0
            rrf_scores.append((i, relevance[i] + bonus))

        # Sort by combined score (relevance dominates; the supplemental
        # signals only break relevance ties).
        rrf_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k with combined score
        reranked = []
        for idx, score in rrf_scores[:k]:
            doc = candidates[idx]
            doc.metadata["_score"] = round(score, 6)
            if bm25_available:
                doc.metadata["_bm25_score"] = round(float(bm25_scores[idx]), 4)
            else:
                doc.metadata.pop("_bm25_score", None)
            doc.metadata["_reranked"] = True
            reranked.append(doc)

        return reranked

    # ------------------------------------------------------------------
    # Rubrics
    # ------------------------------------------------------------------

    def ingest_rubric(self, file_path: str, progress_callback: Optional[Callable[[str, int], None]] = None) -> Dict[str, Any]:
        """Ingest a proposal rubric document into the rubrics collection"""
        try:
            filename = os.path.basename(file_path)
            if progress_callback: progress_callback(f"Loading {filename}...", 10)
            documents = self._load_document(file_path)
            if progress_callback: progress_callback(f"Splitting {len(documents)} pages...", 30)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_documents(documents)

            if progress_callback: progress_callback(f"Creating embeddings...", 50)

            texts = [chunk.page_content for chunk in chunks]
            vectors = self.embeddings.embed_documents(texts)

            ids = [str(uuid.uuid4()) for _ in chunks]
            payloads = [
                {"text": chunk.page_content, "source_file": filename, "type": "rubric"}
                for chunk in chunks
            ]

            upsert_vectors(self.rubrics_collection, ids, vectors, payloads)

            if progress_callback: progress_callback(f"✓ Ingested rubric {filename}", 100)
            return {"success": True, "filename": filename, "chunks": len(chunks)}
        except Exception as e:
            if progress_callback: progress_callback(f"✗ Failed: {str(e)}", 0)
            return {"success": False, "error": str(e)}

    def search_rubrics(self, query: str, k: int = 5) -> List[Document]:
        """Search the proposal rubrics collection"""
        try:
            query_vector = self.embeddings.embed_query(query)
            hits = search_vectors(self.rubrics_collection, query_vector, limit=k)
            return [
                Document(
                    page_content=h["payload"].get("text", ""),
                    metadata={kk: vv for kk, vv in h["payload"].items() if kk != "text"},
                )
                for h in hits
            ]
        except Exception as e:
            print(f"Rubrics search failed: {e}")
        return []

    # ------------------------------------------------------------------
    # Stats & management
    # ------------------------------------------------------------------

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about vector stores"""
        stats = {
            "general": {"status": "empty", "count": 0},
            "personal": {"status": "empty", "count": 0}
        }

        try:
            cnt = collection_count(self.general_collection)
            stats["general"] = {"status": "ready" if cnt > 0 else "empty", "count": cnt}
        except Exception as e:
            print(f"General collection stats failed: {e}")
            stats["general"]["status"] = "error"

        if self.personal_collection:
            try:
                cnt = collection_count(self.personal_collection)
                stats["personal"] = {"status": "ready" if cnt > 0 else "empty", "count": cnt}
            except Exception as e:
                print(f"Personal collection stats failed: {e}")
                stats["personal"]["status"] = "error"

        return stats

    def get_personal_documents(self) -> List[str]:
        """Get list of documents in personal collection"""
        if not self.personal_collection:
            return []

        try:
            points = scroll_all(self.personal_collection, limit=1000)
            files = set()
            for p in points:
                sf = p["payload"].get("source_file")
                if sf:
                    files.add(sf)
            return sorted(list(files))
        except Exception as e:
            print(f"Get personal documents failed: {e}")
        return []

    def delete_personal_document(self, filename: str) -> bool:
        """Delete a document from personal collection by filename"""
        if not self.personal_collection:
            return False

        try:
            return delete_by_filter(self.personal_collection, {"source_file": filename})
        except Exception as e:
            print(f"Delete failed: {e}")

        return False


# Backwards compatibility alias
def legacy_init(persist_directory=None):
    return RAGService(persist_directory=persist_directory, user_id=None)
