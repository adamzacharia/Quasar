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

import os
import re
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable, Union
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

def is_domain_relevant(query: str) -> bool:
    """Fast, zero-LLM-call domain relevance check."""
    # 1. Fast reject: known off-domain patterns
    if _OFFDOMAIN_PATTERNS.match(query):
        return False
    # 2. Fast accept: any astronomy/ALMA keyword present
    q_lower = query.lower()
    if any(kw in q_lower for kw in _DOMAIN_KEYWORDS):
        return True
    # 3. Ambiguous — default to False (conservative)
    return False

# ALMA cycle → approximate year mapping
CYCLE_YEAR_MAP = {
    "1": 2013, "2": 2014, "3": 2015, "4": 2016, "5": 2017,
    "6": 2018, "7": 2019, "8": 2020, "9": 2021, "10": 2022,
    "11": 2023, "12": 2024, "13": 2025, "14": 2026,
}


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

    return {
        "doc_year": year,
        "doc_month": month,
        "doc_day": day,
        "doc_category": category,
        "doc_title": title,
        "alma_cycle": alma_cycle or "",
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
        # Over-fetch factor: retrieve more candidates for BM25 reranking
        fetch_k = k * 4

        results = []
        query_vector = self.embeddings.embed_query(query)

        # Build filter conditions
        filter_conditions = {}
        range_conditions = {}

        if category:
            filter_conditions["doc_category"] = category
        if source_file:
            filter_conditions["source_file"] = source_file

        if year:
            # Exact year match via range (gte=year, lte=year)
            range_conditions["doc_year"] = {"gte": year, "lte": year}
        else:
            if min_year or max_year:
                year_range = {}
                if min_year:
                    year_range["gte"] = min_year
                if max_year:
                    year_range["lte"] = max_year
                range_conditions["doc_year"] = year_range

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

        return results

    @staticmethod
    def _hybrid_rerank(
        query: str,
        candidates: List[Document],
        k: int,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        rrf_k: int = 60,
    ) -> List[Document]:
        """Rerank candidates using Reciprocal Rank Fusion (semantic + BM25).

        Combines semantic similarity rank (from Qdrant) with BM25 keyword
        relevance rank. Uses RRF: score = Σ (weight / (rrf_k + rank)).

        Args:
            query:           The user's search query.
            candidates:      Over-fetched Document list with _semantic_score.
            k:               Number of results to return after reranking.
            semantic_weight: Weight for the semantic (embedding) signal.
            bm25_weight:     Weight for the BM25 (keyword) signal.
            rrf_k:           RRF smoothing constant (standard is 60).

        Returns:
            Top-k documents sorted by combined RRF score.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            # Fallback: if rank_bm25 not installed, return top-k by semantic only
            candidates.sort(
                key=lambda d: d.metadata.get("_semantic_score", 0), reverse=True
            )
            for doc in candidates[:k]:
                doc.metadata["_score"] = doc.metadata.get("_semantic_score", 0.0)
            return candidates[:k]

        if not candidates:
            return []

        # Tokenize for BM25 (simple whitespace + lowercase)
        tokenized_corpus = [
            doc.page_content.lower().split() for doc in candidates
        ]
        tokenized_query = query.lower().split()

        # Build BM25 index over the candidate set
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(tokenized_query)

        # Build semantic rank (already sorted by Qdrant score)
        semantic_ranked = sorted(
            range(len(candidates)),
            key=lambda i: candidates[i].metadata.get("_semantic_score", 0),
            reverse=True,
        )
        semantic_rank = {idx: rank for rank, idx in enumerate(semantic_ranked)}

        # Build BM25 rank
        bm25_ranked = sorted(
            range(len(candidates)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )
        bm25_rank = {idx: rank for rank, idx in enumerate(bm25_ranked)}

        # Reciprocal Rank Fusion
        rrf_scores = []
        for i in range(len(candidates)):
            sem_rrf = semantic_weight / (rrf_k + semantic_rank[i])
            bm25_rrf = bm25_weight / (rrf_k + bm25_rank[i])
            combined = sem_rrf + bm25_rrf
            rrf_scores.append((i, combined))

        # Sort by combined RRF score
        rrf_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k with combined score
        reranked = []
        for idx, score in rrf_scores[:k]:
            doc = candidates[idx]
            doc.metadata["_score"] = round(score, 6)
            doc.metadata["_bm25_score"] = round(float(bm25_scores[idx]), 4)
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
