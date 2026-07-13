# services/vector_db.py
"""
Centralized Vector Database Service — Qdrant Cloud + in-memory fallback.

CALLED BY: services/rag_service.py, services/memory_service.py
CALLS:     qdrant_client (Qdrant Cloud) OR qdrant_client (in-memory local)

Provides a singleton QdrantClient and helpers for upserting/searching
vectors. When QDRANT_URL is set, queries go to Qdrant Cloud (persistent).
Otherwise, uses an in-memory Qdrant instance (for development).

Supports:
  - Exact-match filtering (keyword fields)
  - Range filtering (integer fields like doc_year)
  - Payload indexing for fast filtered search
"""

import os
import uuid
from typing import List, Dict, Any, Optional, Union

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    ScrollRequest,
    PayloadSchemaType,
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


def delete_collection(name: str) -> bool:
    """Delete an entire collection. Returns True if deleted, False if not found."""
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        print(f"[VectorDB] Collection '{name}' not found — nothing to delete")
        return False
    client.delete_collection(collection_name=name)
    print(f"[VectorDB] Deleted collection: {name}")
    return True


def create_payload_index(
    collection: str,
    field_name: str,
    schema_type: PayloadSchemaType,
):
    """Create a payload index on a field for faster filtered search.

    Args:
        collection:  Name of the Qdrant collection.
        field_name:  Payload key to index (e.g. 'doc_year').
        schema_type: One of PayloadSchemaType.INTEGER, .KEYWORD, .FLOAT, etc.
    """
    client = get_qdrant_client()
    client.create_payload_index(
        collection_name=collection,
        field_name=field_name,
        field_schema=schema_type,
    )
    print(f"[VectorDB] Created payload index: {collection}.{field_name} ({schema_type})")


def upsert_vectors(
    collection: str,
    ids: List[str],
    vectors: List[List[float]],
    payloads: List[Dict[str, Any]],
    batch_size: int = 100,
):
    """Upsert points into a Qdrant collection, batched to avoid timeouts."""
    client = get_qdrant_client()
    ensure_collection(collection, dim=len(vectors[0]) if vectors else EMBEDDING_DIM)

    points = [
        PointStruct(id=uid, vector=vec, payload=pay)
        for uid, vec, pay in zip(ids, vectors, payloads)
    ]

    # Batch upsert to avoid Qdrant Cloud write timeouts on large uploads
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection, points=batch)



def _build_filter(
    filter_conditions: Optional[Dict[str, str]] = None,
    range_conditions: Optional[Dict[str, Dict[str, Union[int, float]]]] = None,
) -> Optional[Filter]:
    """Build a Qdrant Filter from exact-match and/or range conditions.

    Args:
        filter_conditions: Dict of {field: value} for exact MatchValue.
        range_conditions:  Dict of {field: {gte: N, lte: N, gt: N, lt: N}}.
                           E.g. {"doc_year": {"gte": 2024}}

    Returns:
        A Qdrant Filter or None.
    """
    must = []

    if filter_conditions:
        for k, v in filter_conditions.items():
            must.append(FieldCondition(key=k, match=MatchValue(value=v)))

    if range_conditions:
        for k, bounds in range_conditions.items():
            must.append(FieldCondition(key=k, range=Range(**bounds)))

    return Filter(must=must) if must else None


def search_vectors(
    collection: str,
    query_vector: List[float],
    limit: int = 5,
    filter_conditions: Optional[Dict[str, str]] = None,
    range_conditions: Optional[Dict[str, Dict[str, Union[int, float]]]] = None,
) -> List[Dict[str, Any]]:
    """Search for similar vectors with optional exact-match and range filters.

    Args:
        collection:       Qdrant collection name.
        query_vector:     Embedding vector for the query.
        limit:            Max results.
        filter_conditions: Exact match filters {field: value}.
        range_conditions:  Range filters {field: {gte: N, lte: N, ...}}.

    Returns:
        List of {id, score, payload} dicts.
    """
    client = get_qdrant_client()

    # Check collection exists
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        return []

    q_filter = _build_filter(filter_conditions, range_conditions)

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
    """Scroll (paginate) through points matching a filter, following
    ``next_offset`` until ``limit`` points are collected or the scroll is
    exhausted. (C15: the single-page version dropped ``next_offset``, so a
    short server page silently truncated the result below ``limit``.)"""
    client = get_qdrant_client()

    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        return []

    q_filter = _build_filter(filter_conditions)

    out: List[Dict[str, Any]] = []
    offset = None
    while len(out) < limit:
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=q_filter,
            limit=limit - len(out),
            with_payload=True,
            offset=offset,
        )
        out.extend(
            {"id": str(p.id), "payload": p.payload or {}}
            for p in points
        )
        if next_offset is None or not points:
            break
        offset = next_offset
    return out[:limit]


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
