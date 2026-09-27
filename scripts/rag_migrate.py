"""
Copy RAG collections between Qdrant stores without re-embedding.

Typical use: build and verify the corpus in a local embedded store
(QDRANT_PATH), then publish it to Qdrant Cloud (QDRANT_URL + QDRANT_API_KEY
from .env). Point ids, vectors and payloads are copied as-is (upserts are
idempotent, so a retried batch cannot duplicate), payload indexes are created
and verified, and point counts are checked at the end.

Safety: the script never deletes anything. A target collection that already
holds points is refused, so a failed copy can never leave you with neither the
old nor the new corpus (guard CX-08). To replace a cloud collection, delete it
deliberately first (Qdrant console, or scripts/rag_inventory.py), then run this.

Usage:
    python scripts/rag_migrate.py --from-path cache/qdrant_v3 --to-cloud
    python scripts/rag_migrate.py --from-path cache/qdrant_v3 --to-cloud --collections alma_general

User collections (user_*) are skipped unless named explicitly.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

INDEXES: Dict[str, str] = {
    "doc_year": "integer", "doc_month": "integer", "page": "integer",
    "doc_category": "keyword", "source_file": "keyword", "alma_cycle": "keyword",
    "doc_status": "keyword", "facility": "keyword", "content_kind": "keyword",
    "ingest_run": "keyword",
}


class MigrationError(RuntimeError):
    pass


def copy_collection(src, dst, name: str, batch: int = 128, require_indexes: bool = True,
                    log=print, resume: bool = False) -> Dict[str, int]:
    """Copy one collection src -> dst. Returns {"source": n, "target": n}.

    ``resume=True`` continues an interrupted copy into a partially filled
    target: every source point is upserted again (same ids, idempotent) and
    the final count check rejects a target that holds anything extra.

    Raises MigrationError when the target already holds points (unless
    resume), when point
    counts differ afterwards, or (require_indexes) when a payload index is
    missing on the target."""
    from qdrant_client.models import PayloadSchemaType, PointStruct

    info = src.get_collection(name)
    existing = {c.name for c in dst.get_collections().collections}
    if name in existing:
        if dst.count(name, exact=True).count and not resume:
            raise MigrationError(f"target collection {name!r} is not empty; refusing to overwrite "
                                 "(use --resume to finish an interrupted copy)")
    else:
        dst.create_collection(collection_name=name, vectors_config=info.config.params.vectors)

    n_src = src.count(name, exact=True).count
    offset, copied, t0 = None, 0, time.time()
    while True:
        pts, offset = src.scroll(name, limit=batch, offset=offset, with_payload=True, with_vectors=True)
        if pts:
            points = [PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in pts]
            for attempt in range(4):
                try:
                    dst.upsert(collection_name=name, points=points, wait=True)
                    break
                except Exception as e:  # same ids on retry: idempotent
                    if attempt == 3:
                        raise
                    log(f"  retry {attempt + 1} after {e}")
                    time.sleep(2 ** attempt)
            copied += len(points)
            if copied % 1024 < batch:
                log(f"  {name}: {copied}/{n_src}  ({time.time() - t0:.0f}s)")
        if offset is None or not pts:
            break

    if require_indexes:
        for field, kind in INDEXES.items():
            schema = PayloadSchemaType.INTEGER if kind == "integer" else PayloadSchemaType.KEYWORD
            dst.create_payload_index(collection_name=name, field_name=field, field_schema=schema, wait=True)
        have = set((dst.get_collection(name).payload_schema or {}).keys())
        missing = sorted(set(INDEXES) - have)
        if missing:
            raise MigrationError(f"{name}: payload indexes missing on target: {missing}")

    n_dst = dst.count(name, exact=True).count
    if n_dst != n_src:
        raise MigrationError(f"{name}: count mismatch, source {n_src}, target {n_dst}")
    return {"source": n_src, "target": n_dst}


def main():
    ap = argparse.ArgumentParser(description="Copy Qdrant collections (no re-embedding)")
    ap.add_argument("--from-path", required=True, help="Source embedded store directory")
    ap.add_argument("--to-cloud", action="store_true", help="Target = QDRANT_URL/QDRANT_API_KEY from .env")
    ap.add_argument("--collections", nargs="*", default=None)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--resume", action="store_true",
                    help="Finish an interrupted copy into a partially filled target (idempotent upserts)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if not args.to_cloud:
        ap.error("only --to-cloud is supported")

    from dotenv import dotenv_values
    from qdrant_client import QdrantClient

    env = dotenv_values(os.path.join(ROOT, ".env"))
    url, key = env.get("QDRANT_URL"), env.get("QDRANT_API_KEY")
    if not url or not key:
        sys.exit("QDRANT_URL / QDRANT_API_KEY missing in .env")

    src_path = args.from_path if os.path.isabs(args.from_path) else os.path.join(ROOT, args.from_path)
    src = QdrantClient(path=src_path)
    dst = QdrantClient(url=url, api_key=key, timeout=120)
    names: List[str] = args.collections or [
        c.name for c in src.get_collections().collections if not c.name.startswith("user_")
    ]
    print(f"source {src_path}\ntarget {url}\ncollections {names}")
    failed = False
    for name in names:
        try:
            r = copy_collection(src, dst, name, batch=args.batch, resume=args.resume)
            print(f"OK {name}: source {r['source']}, target {r['target']}, indexes verified")
        except MigrationError as e:
            failed = True
            print(f"FAILED {e}")
    src.close()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
