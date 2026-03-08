# services/rag_service.py
"""
RAG Service — Retrieval Augmented Generation for ALMA documentation.

CALLED BY: core/agent.py (context retrieval before every LLM call)
           scripts/ingest_alma_docs.py, scripts/ingest_manual.py
CALLS:     ChromaDB (vector store at chroma_db/), OpenAI Embeddings

PURPOSE:
    Ingests PDFs (ALMA Manual, user docs) into chunked vector embeddings,
    then performs semantic similarity search to provide relevant context
    to the LLM before generating responses.

COLLECTIONS:
    General  — Shared ALMA documentation (chroma_db/)
    Personal — Per-user uploaded documents (chroma_db/user_{id}/)
"""

import os
from typing import List, Dict, Any, Optional, Callable
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

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
        if persist_directory is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            persist_directory = os.path.join(root_dir, "chroma_db")
            
        self.base_directory = persist_directory
        self.user_id = user_id
        self.embeddings = OpenAIEmbeddings()
        
        # General collection (shared ALMA docs)
        self.general_dir = persist_directory
        self.general_store = None
        
        # Personal collection (user docs)
        self.personal_dir = os.path.join(persist_directory, f"user_{user_id}") if user_id else None
        self.personal_store = None
        
        # Rubrics collection (TAC guidelines)
        self.rubrics_dir = os.path.join(persist_directory, "proposal_rubrics")
        self.rubrics_store = None
        
        # Initialize stores
        self._init_stores()
    
    def _init_stores(self):
        """Initialize vector stores if they exist"""
        # General store
        if os.path.exists(self.general_dir):
            try:
                self.general_store = Chroma(
                    persist_directory=self.general_dir,
                    embedding_function=self.embeddings
                )
            except:
                pass
        
        # Personal store
        if self.personal_dir and os.path.exists(self.personal_dir):
            try:
                self.personal_store = Chroma(
                    persist_directory=self.personal_dir,
                    embedding_function=self.embeddings
                )
            except:
                pass

        # Rubrics store
        if os.path.exists(self.rubrics_dir):
            try:
                self.rubrics_store = Chroma(
                    persist_directory=self.rubrics_dir,
                    embedding_function=self.embeddings
                )
            except:
                pass

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
            
            # Add metadata
            for chunk in chunks:
                chunk.metadata['source_file'] = filename
                chunk.metadata['is_personal'] = personal
                if self.user_id and personal:
                    chunk.metadata['user_id'] = self.user_id
            
            # Step 3: Create embeddings
            if progress_callback:
                progress_callback(f"Creating embeddings for {len(chunks)} chunks...", 50)
            
            # Determine target store and directory
            if personal:
                target_dir = self.personal_dir
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)
                
                if self.personal_store is None:
                    self.personal_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=self.embeddings,
                        persist_directory=target_dir
                    )
                else:
                    self.personal_store.add_documents(chunks)
            else:
                target_dir = self.general_dir
                if self.general_store is None:
                    self.general_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=self.embeddings,
                        persist_directory=target_dir
                    )
                else:
                    self.general_store.add_documents(chunks)
            
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
        
        # Search general collection
        if self.general_store:
            try:
                general_results = self.general_store.similarity_search(query, k=k)
                results.extend(general_results)
            except Exception as e:
                print(f"General search failed: {e}")
        
        # Search personal collection
        if include_personal and self.personal_store:
            try:
                personal_results = self.personal_store.similarity_search(query, k=k)
                results.extend(personal_results)
            except Exception as e:
                print(f"Personal search failed: {e}")
        
        return results

    def ingest_rubric(self, file_path: str, progress_callback: Optional[Callable[[str, int], None]] = None) -> Dict[str, Any]:
        """Ingest a proposal rubric document into the rubrics collection"""
        try:
            filename = os.path.basename(file_path)
            if progress_callback: progress_callback(f"Loading {filename}...", 10)
            documents = self._load_document(file_path)
            if progress_callback: progress_callback(f"Splitting {len(documents)} pages...", 30)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_documents(documents)
            
            for chunk in chunks:
                chunk.metadata['source_file'] = filename
                chunk.metadata['type'] = 'rubric'
            
            if progress_callback: progress_callback(f"Creating embeddings...", 50)
            
            if not os.path.exists(self.rubrics_dir):
                os.makedirs(self.rubrics_dir, exist_ok=True)
            
            if self.rubrics_store is None:
                self.rubrics_store = Chroma.from_documents(chunks, self.embeddings, persist_directory=self.rubrics_dir)
            else:
                self.rubrics_store.add_documents(chunks)
                
            if progress_callback: progress_callback(f"✓ Ingested rubric {filename}", 100)
            return {"success": True, "filename": filename, "chunks": len(chunks)}
        except Exception as e:
            if progress_callback: progress_callback(f"✗ Failed: {str(e)}", 0)
            return {"success": False, "error": str(e)}

    def search_rubrics(self, query: str, k: int = 5) -> List[Document]:
        """Search the proposal rubrics collection"""
        if self.rubrics_store:
            try:
                return self.rubrics_store.similarity_search(query, k=k)
            except Exception as e:
                print(f"Rubrics search failed: {e}")
        return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about both vector stores"""
        stats = {
            "general": {"status": "empty", "count": 0},
            "personal": {"status": "empty", "count": 0}
        }
        
        if self.general_store:
            try:
                stats["general"] = {
                    "status": "ready",
                    "count": self.general_store._collection.count()
                }
            except:
                stats["general"]["status"] = "error"
        
        if self.personal_store:
            try:
                stats["personal"] = {
                    "status": "ready", 
                    "count": self.personal_store._collection.count()
                }
            except:
                stats["personal"]["status"] = "error"
        
        return stats
    
    def get_personal_documents(self) -> List[str]:
        """Get list of documents in personal collection"""
        if not self.personal_store:
            return []
        
        try:
            # Get unique source files from metadata
            all_docs = self.personal_store.get()
            if all_docs and 'metadatas' in all_docs:
                files = set()
                for meta in all_docs['metadatas']:
                    if meta and 'source_file' in meta:
                        files.add(meta['source_file'])
                return sorted(list(files))
        except:
            pass
        return []
    
    def delete_personal_document(self, filename: str) -> bool:
        """Delete a document from personal collection"""
        if not self.personal_store:
            return False
        
        try:
            # Get all docs with this source file
            all_docs = self.personal_store.get()
            if all_docs and 'ids' in all_docs and 'metadatas' in all_docs:
                ids_to_delete = []
                for i, meta in enumerate(all_docs['metadatas']):
                    if meta and meta.get('source_file') == filename:
                        ids_to_delete.append(all_docs['ids'][i])
                
                if ids_to_delete:
                    self.personal_store.delete(ids=ids_to_delete)
                    return True
        except Exception as e:
            print(f"Delete failed: {e}")
        
        return False

# Backwards compatibility alias
def legacy_init(persist_directory=None):
    return RAGService(persist_directory=persist_directory, user_id=None)
