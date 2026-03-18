# services/vector_db.py
"""
Centralized Vector Database Service — Qdrant Cloud + in-memory fallback.

CALLED BY: services/rag_service.py, services/memory_service.py
CALLS:     qdrant_client (Qdrant Cloud) OR qdrant_client (in-memory local)

Provides a singleton QdrantClient and helpers for upserting/searching
vectors. When QDRANT_URL is set, queries go to Qdrant Cloud (persistent).
Otherwise, uses an in-memory Qdrant instance (for development).
"""

import os
import uuid
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    ScrollRequest,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

_USE_CLOUD = bool(QDRANT_URL and QDRANT_API_KEY)

# Embedding dimension for OpenAI text-embedding-ada-002 / text-embedding-3-small
EMBEDDING_DIM = 1536

# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------
_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Return the singleton QdrantClient (cloud or in-memory)."""
    global _client
    if _client is None:
        if _USE_CLOUD:
            _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            print(f"[VectorDB] Connected to Qdrant Cloud: {QDRANT_URL}")
        else:
            _client = QdrantClient(":memory:")
            print("[VectorDB] Using in-memory Qdrant (dev mode)")
    return _client


def ensure_collection(name: str, dim: int = EMBEDDING_DIM):
    """Create a collection if it doesn't already exist."""
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"[VectorDB] Created collection: {name}")


def upsert_vectors(
    collection: str,
    ids: List[str],
    vectors: List[List[float]],
    payloads: List[Dict[str, Any]],
):
    """Upsert points into a Qdrant collection."""
    client = get_qdrant_client()
    ensure_collection(collection, dim=len(vectors[0]) if vectors else EMBEDDING_DIM)

    points = [
        PointStruct(id=uid, vector=vec, payload=pay)
        for uid, vec, pay in zip(ids, vectors, payloads)
    ]
    client.upsert(collection_name=collection, points=points)


def search_vectors(
    collection: str,
    query_vector: List[float],
    limit: int = 5,
    filter_conditions: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Search for similar vectors. Returns list of {id, score, payload}."""
    client = get_qdrant_client()

    # Check collection exists
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        return []

    # Build optional filter
    q_filter = None
    if filter_conditions:
        must = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filter_conditions.items()
        ]
        q_filter = Filter(must=must)

    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=limit,
        query_filter=q_filter,
        with_payload=True,
    )

    return [
        {
            "id": str(hit.id),
            "score": hit.score,
            "payload": hit.payload or {},
        }
        for hit in results.points
    ]


def scroll_all(
    collection: str,
    filter_conditions: Optional[Dict[str, str]] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Scroll (paginate) through all points matching a filter."""
    client = get_qdrant_client()

    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        return []

    q_filter = None
    if filter_conditions:
        must = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filter_conditions.items()
        ]
        q_filter = Filter(must=must)

    points, _next_offset = client.scroll(
        collection_name=collection,
        scroll_filter=q_filter,
        limit=limit,
        with_payload=True,
    )
    return [
        {"id": str(p.id), "payload": p.payload or {}}
        for p in points
    ]


def delete_by_filter(collection: str, filter_conditions: Dict[str, str]) -> bool:
    """Delete points matching a metadata filter."""
    client = get_qdrant_client()

    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        return False

    must = [
        FieldCondition(key=k, match=MatchValue(value=v))
        for k, v in filter_conditions.items()
    ]
    client.delete(
        collection_name=collection,
        points_selector=Filter(must=must),
    )
    return True


def collection_count(collection: str) -> int:
    """Get the number of points in a collection."""
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        return 0
    info = client.get_collection(collection)
    return info.points_count or 0


def is_using_cloud() -> bool:
    """Return True when connected to Qdrant Cloud."""
    return _USE_CLOUD
