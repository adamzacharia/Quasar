"""Personalization (personal RAG knowledge-base) upload/list/delete endpoints."""

import asyncio
import datetime as _dt
import tempfile
import uuid
from pathlib import Path as _Path
from typing import List as PyList

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.deps import _executor, _get_pers_db, get_current_user

router = APIRouter()

# What RAGService._load_document can actually read (services/rag_service.py).
# Anything else is refused up front with a clear message instead of a 0/N
# "success" (audit 2026-10-01: CSV/JSON/DOCX were advertised and then failed).
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@router.post("/api/personalization/upload")
async def personalization_upload(
    files: PyList[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Upload and index documents into the user's personal RAG collection."""
    user_id = current_user["sub"]
    results = []

    for f in files:
        raw = await f.read()
        ext = (_Path(f.filename or "file").suffix or ".txt").lower()
        if ext not in SUPPORTED_EXTENSIONS:
            results.append({"filename": f.filename, "success": False,
                            "error": f"Unsupported file type {ext}: upload PDF, TXT or MD."})
            continue

        # Write to a temp file so RAGService can read it
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        # The document id exists BEFORE ingestion and is stamped on every
        # chunk, so deleting one copy of a twice-uploaded file removes only
        # that copy's vectors (audit S20).
        doc_id = str(uuid.uuid4())
        try:
            loop = asyncio.get_event_loop()

            def _ingest():
                from services.rag_service import RAGService
                svc = RAGService(user_id=user_id)
                return svc.ingest_document(
                    tmp_path, personal=True, original_filename=f.filename,
                    extra_metadata={"doc_id": doc_id},
                )

            result = await loop.run_in_executor(_executor, _ingest)

            if result.get("success"):
                try:
                    conn = _get_pers_db()
                    try:
                        conn.execute(
                            "INSERT INTO documents VALUES (?,?,?,?,?,?)",
                            (doc_id, user_id, f.filename, len(raw),
                             result.get("chunks", 0), _dt.datetime.utcnow().isoformat())
                        )
                        conn.commit()
                    finally:
                        conn.close()
                except Exception:
                    # No listed row means the user could never delete these
                    # chunks: remove them before reporting the failure.
                    def _rollback():
                        from services.vector_db import delete_personal_doc_points
                        delete_personal_doc_points(f"user_{user_id}_personal", doc_id, f.filename, False)
                    try:
                        await loop.run_in_executor(_executor, _rollback)
                    except Exception as _rb_err:
                        print(f"[WARN] rollback of unlisted chunks for {doc_id} failed: {_rb_err}")
                    raise
                results.append({"filename": f.filename, "success": True, "chunks": result.get("chunks", 0)})
            else:
                results.append({"filename": f.filename, "success": False, "error": result.get("error", "Unknown error")})
        except Exception as e:
            results.append({"filename": f.filename, "success": False, "error": str(e)})
        finally:
            try: _Path(tmp_path).unlink()
            except: pass

    successes = sum(1 for r in results if r["success"])
    return {
        "message": f"{successes}/{len(results)} documents indexed successfully.",
        "all_succeeded": successes == len(results),
        "results": results,
    }


@router.get("/api/personalization/documents")
async def personalization_list(current_user: dict = Depends(get_current_user)):
    """List all documents in the user's personal knowledge base."""
    user_id = current_user["sub"]
    conn = _get_pers_db()
    rows = conn.execute(
        "SELECT id, filename, size_bytes, uploaded_at, chunk_count FROM documents WHERE user_id=? ORDER BY uploaded_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "filename": r[1], "size_bytes": r[2], "uploaded_at": r[3], "chunk_count": r[4]}
        for r in rows
    ]


@router.delete("/api/personalization/document/{doc_id}")
async def personalization_delete(doc_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a document from the user's personal knowledge base.

    The row is removed only after the vectors are verifiably gone; a failed or
    partial vector delete returns 502 and keeps the row listed, so the UI never
    shows a document as deleted while its text is still searchable.
    """
    user_id = current_user["sub"]
    conn = _get_pers_db()
    row = conn.execute("SELECT filename FROM documents WHERE id=? AND user_id=?", (doc_id, user_id)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")
    filename = row[0]
    # Legacy chunks (no doc_id) can only be matched by filename. They belong to
    # this row unless ANOTHER listed row of the same name is itself legacy
    # (has no doc_id-stamped points); a newer stamped copy does not own them.
    other_ids = [r[0] for r in conn.execute(
        "SELECT id FROM documents WHERE user_id=? AND filename=? AND id<>?",
        (user_id, filename, doc_id),
    ).fetchall()]
    conn.close()

    try:
        loop = asyncio.get_event_loop()

        def _delete():
            from services.rag_service import RAGService
            from services.vector_db import count_doc_points
            svc = RAGService(user_id=user_id)
            coll = f"user_{user_id}_personal"
            other_legacy = any(count_doc_points(coll, oid) == 0 for oid in other_ids)
            return svc.delete_personal_document(filename, doc_id=doc_id, legacy_by_filename=not other_legacy)

        remaining = await loop.run_in_executor(_executor, _delete)
    except Exception as e:
        print(f"[WARN] Vector delete failed for {doc_id}: {e}")
        raise HTTPException(status_code=502, detail="Could not delete the document's search index; it is still listed. Try again.")
    if remaining:
        print(f"[WARN] Vector delete incomplete for {doc_id}: {remaining} points remain")
        raise HTTPException(status_code=502, detail="Deletion did not complete; the document is still listed. Try again.")

    conn = _get_pers_db()
    conn.execute("DELETE FROM documents WHERE id=? AND user_id=?", (doc_id, user_id))
    conn.commit()
    conn.close()
    return {"success": True}
