"""Conversation history CRUD endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from api.deps import conversation_service, get_current_user
from api.models import ConversationCreate, ConversationTitleUpdate

router = APIRouter()


@router.get("/api/conversations")
async def list_conversations(current_user: dict = Depends(get_current_user)):
    """List a user's conversations, most recent first."""
    user_id = current_user["sub"]
    convos = conversation_service.get_user_conversations(user_id, limit=50)
    return {"conversations": convos}


@router.post("/api/conversations")
async def create_conversation_endpoint(req: ConversationCreate, current_user: dict = Depends(get_current_user)):
    """Create a new empty conversation."""
    user_id = current_user["sub"]
    conv_id = conversation_service.create_conversation(user_id, req.title, req.model)
    return {"id": conv_id, "title": req.title, "model": req.model}


@router.get("/api/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str, current_user: dict = Depends(get_current_user)):
    """Fetch all messages for a conversation."""
    user_id = current_user["sub"]
    messages = conversation_service.get_conversation_messages_for_user(conversation_id, user_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"messages": messages}


@router.put("/api/conversations/{conversation_id}/title")
async def update_conversation_title_endpoint(
    conversation_id: str,
    req: ConversationTitleUpdate,
    current_user: dict = Depends(get_current_user),
):
    """Update a conversation's title."""
    user_id = current_user["sub"]
    updated = conversation_service.update_conversation_title_for_user(
        conversation_id,
        user_id,
        req.title,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "ok"}


@router.delete("/api/conversations/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a conversation and all its messages."""
    try:
        user_id = current_user["sub"]
        deleted = conversation_service.delete_conversation_for_user(conversation_id, user_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
        print(f"[INFO] Deleted conversation {conversation_id} for user {current_user.get('sub', 'unknown')}")
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Failed to delete conversation {conversation_id}: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
