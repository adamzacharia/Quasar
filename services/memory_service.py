# services/memory_service.py
"""
Memory Service — Long-term user memory using Qdrant vector store.

CALLED BY: core/agent.py (_update_memory, context retrieval)
CALLS:     Qdrant Cloud (via services/vector_db.py), OpenAI Embeddings

Stores implicit user facts extracted from conversations (e.g.,
"User is interested in protoplanetary disks") and retrieves them
via semantic search to personalize future responses.
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_openai import OpenAIEmbeddings

from services.vector_db import (
    upsert_vectors,
    search_vectors,
    scroll_all,
    ensure_collection,
)

# Qdrant collection for long-term user memory
MEMORY_COLLECTION = "user_memory"


class MemoryService:
    """
    Long-Term Memory Service (Mem0-inspired)
    Stores and retrieves user facts/preferences across sessions.
    Uses Qdrant Cloud for persistent vector storage.
    """

    def __init__(self, persist_directory: str = None):
        # persist_directory kept for backward-compat but now ignored
        self.embeddings = OpenAIEmbeddings()

        # Ensure the memory collection exists
        try:
            ensure_collection(MEMORY_COLLECTION)
            print(f"MemoryService initialized (Qdrant collection: {MEMORY_COLLECTION})")
        except Exception as e:
            print(f"MemoryService init failed: {e}")

    def add_memory(self, content: str, user_id: str = "user", metadata: Dict = None) -> bool:
        """Add a new memory fact"""
        try:
            # Embed the content
            vector = self.embeddings.embed_query(content)

            # Build payload
            pay = metadata.copy() if metadata else {}
            pay.update({
                "text": content,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "type": "fact",
            })

            mem_id = str(uuid.uuid4())
            upsert_vectors(MEMORY_COLLECTION, [mem_id], [vector], [pay])
            return True
        except Exception as e:
            print(f"Failed to add memory: {e}")
            return False

    def search_memories(self, query: str, user_id: str = "user", k: int = 3) -> List[str]:
        """Search for relevant memories"""
        try:
            query_vector = self.embeddings.embed_query(query)
            hits = search_vectors(
                MEMORY_COLLECTION,
                query_vector,
                limit=k,
                filter_conditions={"user_id": user_id},
            )
            return [hit["payload"].get("text", "") for hit in hits]
        except Exception as e:
            print(f"Memory search failed: {e}")
            return []

    def get_all_memories(self, user_id: str = "user", limit: int = 100) -> List[str]:
        """Get all memories for a user (most recent first)"""
        try:
            points = scroll_all(
                MEMORY_COLLECTION,
                filter_conditions={"user_id": user_id},
                limit=limit,
            )
            return [p["payload"].get("text", "") for p in points]
        except Exception as e:
            print(f"Get all memories failed: {e}")
            return []
