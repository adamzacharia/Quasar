# services/vector_db.py
"""
Centralized Vector Database Service — Qdrant Cloud + in-memory fallback.

CALLED BY: services/rag_service.py, services/personal_docs.py
CALLS:     qdrant_client (Qdrant Cloud) OR qdrant_client (in-memory local)

Provides a singleton QdrantClient and helpers for upserting/searching
vectors. Backend selection, in priority order:
  1. QDRANT_PATH set          -> embedded local Qdrant persisted on disk at that
                                 path (one process at a time holds the lock).
  2. QDRANT_URL + QDRANT_API_KEY -> Qdrant Cloud (persistent).
  3. neither                  -> in-memory Qdrant (dev only, lost at exit).

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
# Persistent embedded store. Wins over Cloud so a dead/suspended cluster can be
# bypassed without touching the Cloud credentials (2026-09-25 outage).
QDRANT_PATH = (os.environ.get("QDRANT_PATH") or "").strip() or None
if QDRANT_PATH and not os.path.isabs(QDRANT_PATH):
    # A relative path is anchored at the repo root, never the process CWD:
    # the backend chdirs into ui-pro/ while scripts run from the root, and
    # CWD-relative resolution would silently split the store (CX-20).
    QDRANT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), QDRANT_PATH)
if QDRANT_PATH:
    QDRANT_PATH = os.path.normpath(QDRANT_PATH)

_USE_LOCAL_PATH = bool(QDRANT_PATH)
_USE_CLOUD = bool(QDRANT_URL and QDRANT_API_KEY) and not _USE_LOCAL_PATH

# Embedding dimension for OpenAI text-embedding-ada-002 / text-embedding-3-small
EMBEDDING_DIM = 1536

# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------
_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Return the singleton QdrantClient (local path, cloud or in-memory)."""
    global _client
    if _client is None:
        if _USE_LOCAL_PATH:
            os.makedirs(QDRANT_PATH, exist_ok=True)
            try:
                _client = QdrantClient(path=QDRANT_PATH)
            except RuntimeError as exc:
                # Embedded Qdrant takes an exclusive directory lock: only one
                # process (backend OR an ingest script) can open the store.
                raise RuntimeError(
                    f"Local Qdrant store {QDRANT_PATH} is locked by another process "
                    "(the backend or another ingest). Stop it first, or run a Qdrant "
                    f"server for concurrent access. ({exc})"
                ) from exc
            print(f"[VectorDB] Using local persistent Qdrant at {QDRANT_PATH}")
        elif _USE_CLOUD:
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


def delete_by_filter(
    collection: str,
    filter_conditions: Dict[str, str],
    exclude_conditions: Optional[Dict[str, str]] = None,
) -> bool:
    """Delete points matching a metadata filter. ``exclude_conditions`` keeps
    points whose field equals the given value (Qdrant ``must_not``); points
    lacking the field are still deleted."""
    client = get_qdrant_client()

    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        return False

    must = [
        FieldCondition(key=k, match=MatchValue(value=v))
        for k, v in filter_conditions.items()
    ]
    must_not = [
        FieldCondition(key=k, match=MatchValue(value=v))
        for k, v in (exclude_conditions or {}).items()
    ]
    client.delete(
        collection_name=collection,
        points_selector=Filter(must=must, must_not=must_not or None),
    )
    return True


def delete_older_runs(collection: str, source_file: str, run_id: str, run_ts: Union[int, float]) -> bool:
    """Delete a source's chunks from runs OLDER than (run_id, run_ts), plus
    legacy chunks that carry no ``ingest_ts``. Newer runs are never touched,
    so two concurrent replacements of the same source converge on the newest
    one instead of deleting each other (guard CX-25)."""
    from qdrant_client.models import IsEmptyCondition, PayloadField

    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        return False
    source = FieldCondition(key="source_file", match=MatchValue(value=source_file))
    client.delete(
        collection_name=collection,
        points_selector=Filter(
            must=[source],
            must_not=[FieldCondition(key="ingest_run", match=MatchValue(value=run_id))],
            should=[
                FieldCondition(key="ingest_ts", range=Range(lt=run_ts)),
                IsEmptyCondition(is_empty=PayloadField(key="ingest_ts")),
            ],
        ),
    )
    # Deliberately NO self-deletion when a newer run is visible: that run may
    # still be a partial upload that later rolls back, and deleting ourselves
    # would leave the source empty (guard round 4, CX-02/CX-23). If an older
    # run finishes after a newer one, both stay until the next replacement:
    # a duplicate, never a loss. Concurrent ingests of one source are not a
    # supported workflow (the embedded store is single-process anyway).
    return True


def delete_personal_doc_points(collection: str, doc_id: Optional[str], source_file: str,
                               legacy_by_filename: bool) -> int:
    """Delete one personal document's points and return how many REMAIN.

    New uploads carry ``doc_id`` on every chunk, so deleting one copy of a
    twice-uploaded file no longer wipes the other copy (audit S20). Legacy
    chunks (ingested before doc_id existed) are removed by filename only when
    ``legacy_by_filename`` (the caller checked no other listed row shares the
    filename), and only those WITHOUT a doc_id. The count afterwards is the
    caller's proof the delete happened; a non-zero result must keep the row.
    """
    from qdrant_client.models import IsEmptyCondition, PayloadField

    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        return 0
    filters = []
    if doc_id:
        filters.append(Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]))
    if legacy_by_filename and source_file:
        filters.append(Filter(must=[
            FieldCondition(key="source_file", match=MatchValue(value=source_file)),
            IsEmptyCondition(is_empty=PayloadField(key="doc_id")),
        ]))
    remaining = 0
    for f in filters:
        client.delete(collection_name=collection, points_selector=f)
    for f in filters:
        remaining += client.count(collection_name=collection, count_filter=f, exact=True).count
    return remaining


def count_doc_points(collection: str, doc_id: str) -> int:
    """Number of points stamped with ``doc_id`` (0 => a legacy, pre-doc_id upload)."""
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing or not doc_id:
        return 0
    f = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
    return client.count(collection_name=collection, count_filter=f, exact=True).count


def ensure_personal_indexes(collection: str) -> None:
    """KEYWORD payload indexes for the personal-collection delete/count filters
    (Qdrant Cloud strict mode rejects filtering on unindexed fields). Never raises."""
    for field in ("doc_id", "source_file"):
        try:
            create_payload_index(collection, field, PayloadSchemaType.KEYWORD)
        except Exception as e:
            print(f"[VectorDB] payload index {collection}.{field} not created (non-fatal): {e}")


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


def backend_label() -> str:
    """Human-readable backend description for logs and inventory reports."""
    if _USE_LOCAL_PATH:
        return f"local:{QDRANT_PATH}"
    if _USE_CLOUD:
        return f"cloud:{QDRANT_URL}"
    return "memory"


def get_collection_info(collection: str) -> Dict[str, Any]:
    """Point count for a collection ({} when it does not exist)."""
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        return {}
    info = client.get_collection(collection)
    return {"points_count": info.points_count or 0, "vectors_count": info.points_count or 0}
