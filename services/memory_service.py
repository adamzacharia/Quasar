# services/memory_service.py
"""
Memory Service — Long-term user memory using ChromaDB vector store.

CALLED BY: core/agent.py (_update_memory, context retrieval)
CALLS:     ChromaDB (langchain_community.vectorstores.Chroma)

Stores implicit user facts extracted from conversations (e.g.,
"User is interested in protoplanetary disks") and retrieves them
via semantic search to personalize future responses.
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class MemoryService:
    """
    Long-Term Memory Service (Mem0-inspired)
    Stores and retrieves user facts/preferences across sessions.
    """
    
    def __init__(self, persist_directory: str = None):
        if persist_directory is None:
            # Default to project_root/chroma_db
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            persist_directory = os.path.join(root_dir, "chroma_db")
            
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        
        # Initialize vector store (separate collection for memory)
        try:
            self.vector_store = Chroma(
                collection_name="user_memory",
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            print(f"MemoryService initialized at {persist_directory}")
        except Exception as e:
            print(f"MemoryService init failed: {e}")

    def add_memory(self, content: str, user_id: str = "user", metadata: Dict = None) -> bool:
        """Add a new memory fact"""
        if self.vector_store is None:
            return False
            
        try:
            meta = metadata or {}
            meta.update({
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "type": "fact"
            })
            
            doc = Document(page_content=content, metadata=meta)
            self.vector_store.add_documents([doc])
            return True
        except Exception as e:
            print(f"Failed to add memory: {e}")
            return False

    def search_memories(self, query: str, user_id: str = "user", k: int = 3) -> List[str]:
        """Search for relevant memories"""
        if self.vector_store is None:
            return []
            
        try:
            # Filter by user_id if possible (Chroma supports where filter)
            results = self.vector_store.similarity_search(
                query, 
                k=k,
                filter={"user_id": user_id}
            )
            return [doc.page_content for doc in results]
        except Exception as e:
            print(f"Memory search failed: {e}")
            return []

    def get_all_memories(self, user_id: str = "user", limit: int = 100) -> List[str]:
        """Get all memories for a user (most recent first)"""
        if self.vector_store is None:
            return []
            
        try:
            # Chroma doesn't have a simple "get all", so we search with empty query or generic term
            # Actually, we can use get()
            results = self.vector_store.get(where={"user_id": user_id}, limit=limit)
            return results['documents'] if results else []
        except Exception as e:
            print(f"Get all memories failed: {e}")
            return []
