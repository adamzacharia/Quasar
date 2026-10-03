"""Saved (bookmarked) papers: per-user, persisted across reloads, redeploys and devices."""

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_current_user, saved_papers_service
from api.models import SavedPaperUpsert
from services.saved_papers_service import SavedPaperError

router = APIRouter()


@router.get("/api/saved-papers")
async def list_saved_papers(current_user: dict = Depends(get_current_user)):
    """List the user's bookmarked papers in the order they were saved."""
    return {"papers": saved_papers_service.list_papers(current_user["sub"])}


@router.put("/api/saved-papers")
async def save_paper(req: SavedPaperUpsert, current_user: dict = Depends(get_current_user)):
    """Bookmark a paper (idempotent: re-saving the same key updates it in place)."""
    try:
        saved_papers_service.save_paper(current_user["sub"], req.key, req.paper)
    except SavedPaperError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "ok"}


# The key rides in the query string, not the path: DOI-based keys contain "/",
# which a path segment would split even when percent-encoded.
@router.delete("/api/saved-papers")
async def remove_saved_paper(
    key: str = Query(..., min_length=1),
    current_user: dict = Depends(get_current_user),
):
    """Remove a bookmark. Removing one that is not saved still succeeds."""
    try:
        saved_papers_service.remove_paper(current_user["sub"], key)
    except SavedPaperError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "ok"}
