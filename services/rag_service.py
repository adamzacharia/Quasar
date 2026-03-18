# services/rag_service.py
"""
RAG Service — Retrieval Augmented Generation for ALMA documentation.

CALLED BY: core/agent.py (context retrieval before every LLM call)
           scripts/ingest_alma_docs.py, scripts/ingest_manual.py
CALLS:     Qdrant Cloud (vector store), OpenAI Embeddings

PURPOSE:
    Ingests PDFs (ALMA Manual, user docs) into chunked vector embeddings,
    then performs semantic similarity search to provide relevant context
    to the LLM before generating responses.

COLLECTIONS (Qdrant):
    alma_general      — Shared ALMA documentation
    user_{id}_personal — Per-user uploaded documents
    proposal_rubrics  — TAC rubric guidelines
"""

import os
import uuid
from typing import List, Dict, Any, Optional, Callable
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from services.vector_db import (
    upsert_vectors,
    search_vectors,
    scroll_all,
    delete_by_filter,
    collection_count,
    ensure_collection,
)


# Qdrant collection names
GENERAL_COLLECTION = "alma_general"
RUBRICS_COLLECTION = "proposal_rubrics"

def _personal_collection(user_id: str) -> str:
    """Return the Qdrant collection name for a user's personal docs."""
    # Sanitise user_id (UUIDs contain hyphens which are fine for Qdrant)
    return f"user_{user_id}_personal"


class RAGService:
    """
    Service for Retrieval Augmented Generation with Personal Collections

    Supports:
    - General RAG (shared ALMA documentation)
    - Personal RAG (user-specific documents)
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
    # Document loading (unchanged)
    # ------------------------------------------------------------------

    def _load_document(self, file_path: str) -> List[Document]:
        """Load document based on file type"""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext in ['.txt', '.md']:
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return loader.load()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_document(
        self,
        file_path: str,
        personal: bool = False,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Ingest a single document into the vector store

        Args:
            file_path: Path to the document
            personal: If True, add to personal collection
            progress_callback: Function(status_msg, percent) for progress updates

        Returns:
            Dict with success status, chunk count, etc.
        """
        try:
            filename = os.path.basename(file_path)

            # Step 1: Load document
            if progress_callback:
                progress_callback(f"Loading {filename}...", 10)
            documents = self._load_document(file_path)

            # Step 2: Split into chunks
            if progress_callback:
                progress_callback(f"Splitting {len(documents)} pages...", 30)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)

            # Step 3: Create embeddings and upsert
            if progress_callback:
                progress_callback(f"Creating embeddings for {len(chunks)} chunks...", 50)

            # Choose target collection
            if personal and self.personal_collection:
                target = self.personal_collection
            else:
                target = self.general_collection

            # Embed all chunk texts
            texts = [chunk.page_content for chunk in chunks]
            vectors = self.embeddings.embed_documents(texts)

            # Build IDs and payloads
            ids = [str(uuid.uuid4()) for _ in chunks]
            payloads = []
            for chunk in chunks:
                pay = {
                    "text": chunk.page_content,
                    "source_file": filename,
                    "is_personal": personal,
                }
                if self.user_id and personal:
                    pay["user_id"] = self.user_id
                # Carry over any existing metadata
                for k, v in chunk.metadata.items():
                    if k not in pay:
                        pay[k] = str(v) if not isinstance(v, (str, int, float, bool)) else v
                payloads.append(pay)

            # Upsert to Qdrant
            if progress_callback:
                progress_callback(f"Storing {len(chunks)} chunks in Qdrant...", 80)

            upsert_vectors(target, ids, vectors, payloads)

            if progress_callback:
                progress_callback(f"✓ Ingested {filename}", 100)

            return {
                "success": True,
                "filename": filename,
                "pages": len(documents),
                "chunks": len(chunks),
                "personal": personal
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
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 5, include_personal: bool = True) -> List[Document]:
        """
        Search for relevant documents in both general and personal collections

        Args:
            query: Search query
            k: Number of results per collection
            include_personal: Whether to include personal docs

        Returns:
            Combined list of relevant documents
        """
        results = []
        query_vector = self.embeddings.embed_query(query)

        # Search general collection
        try:
            general_hits = search_vectors(self.general_collection, query_vector, limit=k)
            for hit in general_hits:
                doc = Document(
                    page_content=hit["payload"].get("text", ""),
                    metadata={k: v for k, v in hit["payload"].items() if k != "text"},
                )
                results.append(doc)
        except Exception as e:
            print(f"General search failed: {e}")

        # Search personal collection
        if include_personal and self.personal_collection:
            try:
                personal_hits = search_vectors(self.personal_collection, query_vector, limit=k)
                for hit in personal_hits:
                    doc = Document(
                        page_content=hit["payload"].get("text", ""),
                        metadata={k: v for k, v in hit["payload"].items() if k != "text"},
                    )
                    results.append(doc)
            except Exception as e:
                print(f"Personal search failed: {e}")

        return results

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
                    metadata={k: v for k, v in h["payload"].items() if k != "text"},
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
        except:
            stats["general"]["status"] = "error"

        if self.personal_collection:
            try:
                cnt = collection_count(self.personal_collection)
                stats["personal"] = {"status": "ready" if cnt > 0 else "empty", "count": cnt}
            except:
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
        except:
            pass
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
