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

        # Write to a temp file so RAGService can read it
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        try:
            loop = asyncio.get_event_loop()

            def _ingest():
                from services.rag_service import RAGService
                svc = RAGService(user_id=user_id)
                return svc.ingest_document(tmp_path, personal=True, original_filename=f.filename)

            result = await loop.run_in_executor(_executor, _ingest)

            if result.get("success"):
                doc_id = str(uuid.uuid4())
                conn = _get_pers_db()
                conn.execute(
                    "INSERT INTO documents VALUES (?,?,?,?,?,?)",
                    (doc_id, user_id, f.filename, len(raw),
                     result.get("chunks", 0), _dt.datetime.utcnow().isoformat())
                )
                conn.commit()
                conn.close()
                results.append({"filename": f.filename, "success": True, "chunks": result.get("chunks", 0)})
            else:
                results.append({"filename": f.filename, "success": False, "error": result.get("error", "Unknown error")})
        except Exception as e:
            results.append({"filename": f.filename, "success": False, "error": str(e)})
        finally:
            try: _Path(tmp_path).unlink()
            except: pass

    successes = sum(1 for r in results if r["success"])
    return {"message": f"{successes}/{len(results)} documents indexed successfully.", "results": results}


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
    """Delete a document from the user's personal knowledge base."""
    user_id = current_user["sub"]
    conn = _get_pers_db()
    row = conn.execute("SELECT filename FROM documents WHERE id=? AND user_id=?", (doc_id, user_id)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    filename = row[0]

    # Remove from vector store
    try:
        loop = asyncio.get_event_loop()
        def _delete():
            from services.rag_service import RAGService
            svc = RAGService(user_id=user_id)
            svc.delete_personal_document(filename)
        await loop.run_in_executor(_executor, _delete)
    except Exception as e:
        print(f"[WARN] Vector delete failed: {e}")

    conn.execute("DELETE FROM documents WHERE id=? AND user_id=?", (doc_id, user_id))
    conn.commit()
    conn.close()
    return {"success": True}
