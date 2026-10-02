"""Settings > Memory: view, edit, delete, pause, clear, export and undo the
user's long-term memory (services/user_memory_service.py). Every route is
scoped to the authenticated user; there is no cross-user read or write."""

import asyncio

from fastapi import APIRouter, Body, Depends, HTTPException

from api.deps import get_current_user

router = APIRouter()


def _svc():
    from services.user_memory_service import get_user_memory_service
    return get_user_memory_service()


def _uid(current_user: dict) -> str:
    uid = (current_user or {}).get("sub")
    if not uid:
        raise HTTPException(status_code=401, detail="Sign in to use Memory")
    return uid


@router.get("/api/memory")
async def memory_get(current_user: dict = Depends(get_current_user)):
    """Active profile, the slot schema the UI renders, pause state and recent changes."""
    from services.user_memory_service import SLOTS
    uid = _uid(current_user)
    svc = _svc()
    items, paused, events = await asyncio.gather(
        asyncio.to_thread(svc.active_items, uid),
        asyncio.to_thread(svc.is_paused, uid),
        asyncio.to_thread(svc.recent_events, uid, 20),
    )
    schema = [
        {"slot": k, "label": v["label"], "kind": v["kind"], "group": v.get("group", "Other"),
         "choices": [{"value": c, "label": l} for c, l in v.get("choices", {}).items()] if v["kind"] == "enum" else None,
         "max_items": v.get("max_items")}
        for k, v in SLOTS.items()
    ]
    return {"paused": paused, "items": items, "schema": schema, "recent": events}


@router.put("/api/memory/slot/{slot}")
async def memory_set(slot: str, body: dict = Body(...), current_user: dict = Depends(get_current_user)):
    """Set a slot by hand (source=manual: confirmed, never overwritten by inference)."""
    from services.user_memory_service import MemoryValidationError
    uid = _uid(current_user)
    try:
        status, event = await asyncio.to_thread(_svc().set_slot, uid, slot, body.get("value"), source="manual")
    except MemoryValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": status, "event": event}


@router.delete("/api/memory/slot/{slot}")
async def memory_delete_slot(slot: str, current_user: dict = Depends(get_current_user)):
    """Forget a slot completely: every saved version of it is deleted."""
    uid = _uid(current_user)
    removed = await asyncio.to_thread(_svc().delete_slot, uid, slot)
    return {"success": True, "removed": removed}


@router.post("/api/memory/pause")
async def memory_pause(body: dict = Body(...), current_user: dict = Depends(get_current_user)):
    """Pause (keep, but neither read nor write) or resume memory."""
    uid = _uid(current_user)
    paused = bool(body.get("paused"))
    await asyncio.to_thread(_svc().set_paused, uid, paused)
    return {"paused": paused}


@router.delete("/api/memory")
async def memory_clear(current_user: dict = Depends(get_current_user)):
    """Delete ALL memory for this user: values, history and change log."""
    uid = _uid(current_user)
    removed = await asyncio.to_thread(_svc().clear_all, uid)
    return {"success": True, "removed": removed}


@router.get("/api/memory/export")
async def memory_export(current_user: dict = Depends(get_current_user)):
    """Everything stored about this user in Memory, as JSON."""
    uid = _uid(current_user)
    return await asyncio.to_thread(_svc().export, uid)


@router.post("/api/memory/events/{event_id}/undo")
async def memory_undo(event_id: str, current_user: dict = Depends(get_current_user)):
    """Undo one memory change (the 'Memory updated' chip's Undo)."""
    uid = _uid(current_user)
    ok = await asyncio.to_thread(_svc().undo_event, uid, event_id)
    if not ok:
        raise HTTPException(status_code=409, detail="This change can no longer be undone (it was already undone or replaced).")
    return {"success": True}
