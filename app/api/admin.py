"""Admin API for conversation inspection, memory moderation, bot settings, and retention cleanup."""

import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.orm import Session
from app.core.security import verify_admin_key
from app.database.connection import get_db
from app.database.models import (
    BotSettings,
    Conversation,
    Memory,
    Message,
    User,
)
from app.database.repositories import get_or_create_bot_settings
from app.memory.retention import cleanup_expired_data

router = APIRouter(tags=["Admin"], dependencies=[Depends(verify_admin_key)])


# --- Pydantic Schemas for Admin Operations ---

class MemoryEditRequest(BaseModel):
    category: Optional[str] = None
    content: Optional[str] = None
    status: Optional[str] = None


class BotSettingsUpdateRequest(BaseModel):
    is_global_paused: Optional[bool] = None
    cooldown_seconds: Optional[int] = None
    response_frequency: Optional[float] = None
    relevance_threshold: Optional[float] = None


# --- Retention Cleanup (Stage 5) ---

@router.post("/cleanup")
@router.post("/admin/cleanup")
def trigger_cleanup(db: Session = Depends(get_db)) -> dict:
    """Manually trigger message and summary retention cleanup."""
    result = cleanup_expired_data(db)
    return {"status": "success", **result}


# --- Conversations Management (Stage 6) ---

@router.get("/conversations")
@router.get("/admin/conversations")
def list_conversations(db: Session = Depends(get_db), limit: int = 50) -> List[dict]:
    """List recent conversations."""
    conversations = (
        db.execute(select(Conversation).order_by(desc(Conversation.updated_at)).limit(limit))
        .scalars()
        .all()
    )
    return [
        {
            "id": str(c.id),
            "external_id": c.external_id,
            "user_id": str(c.user_id) if c.user_id else None,
            "conversation_type": c.conversation_type,
            "trigger_mode": c.trigger_mode,
            "paused": c.paused,
            "human_handoff": c.human_handoff,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        }
        for c in conversations
    ]


@router.get("/conversations/{conversation_id}/messages")
@router.get("/admin/conversations/{conversation_id}/messages")
def get_conversation_messages(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
    limit: int = 100,
) -> List[dict]:
    """Retrieve messages for a specific conversation."""
    conv = db.execute(select(Conversation).where(Conversation.id == conversation_id)).scalars().first()
    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    messages = (
        db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
            .limit(limit)
        )
        .scalars()
        .all()
    )
    return [
        {
            "id": str(m.id),
            "external_id": m.external_id,
            "sender_id": m.sender_id,
            "direction": m.direction,
            "text": m.text,
            "processing_status": m.processing_status,
            "expires_at": m.expires_at.isoformat() if m.expires_at else None,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in messages
    ]


@router.post("/conversations/{conversation_id}/handoff/resume")
@router.post("/admin/conversations/{conversation_id}/handoff/resume")
def resume_conversation_handoff(conversation_id: uuid.UUID, db: Session = Depends(get_db)) -> dict:
    """Resume automated bot replies after human handoff is resolved."""
    conv = db.execute(select(Conversation).where(Conversation.id == conversation_id)).scalars().first()
    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    conv.human_handoff = False
    db.commit()
    return {"status": "success", "conversation_id": str(conv.id), "human_handoff": False}


# --- Memory Moderation (Stage 6) ---

@router.get("/memories/pending")
@router.get("/admin/memories/pending")
def list_pending_memories(db: Session = Depends(get_db), limit: int = 50) -> List[dict]:
    """List pending memories awaiting owner approval."""
    pending = (
        db.execute(
            select(Memory)
            .where(Memory.status == "pending")
            .order_by(desc(Memory.created_at))
            .limit(limit)
        )
        .scalars()
        .all()
    )
    return [
        {
            "id": str(m.id),
            "user_id": str(m.user_id),
            "category": m.category,
            "content": m.content,
            "status": m.status,
            "source_message_id": str(m.source_message_id) if m.source_message_id else None,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in pending
    ]


@router.get("/memories/{user_id}")
@router.get("/admin/memories/{user_id}")
def get_user_memories(user_id: uuid.UUID, db: Session = Depends(get_db)) -> List[dict]:
    """Retrieve all memories stored for a specific user."""
    memories = (
        db.execute(
            select(Memory).where(Memory.user_id == user_id).order_by(desc(Memory.created_at))
        )
        .scalars()
        .all()
    )
    return [
        {
            "id": str(m.id),
            "user_id": str(m.user_id),
            "category": m.category,
            "content": m.content,
            "status": m.status,
            "source_message_id": str(m.source_message_id) if m.source_message_id else None,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in memories
    ]


@router.post("/memories/{memory_id}/approve")
@router.post("/admin/memories/{memory_id}/approve")
def approve_memory(memory_id: uuid.UUID, db: Session = Depends(get_db)) -> dict:
    """Approve a pending memory so it can be injected into conversation prompts."""
    memory = db.execute(select(Memory).where(Memory.id == memory_id)).scalars().first()
    if not memory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")

    memory.status = "approved"
    db.commit()
    db.refresh(memory)
    return {"status": "success", "id": str(memory.id), "memory_status": memory.status}


@router.post("/memories/{memory_id}/reject")
@router.post("/admin/memories/{memory_id}/reject")
def reject_memory(memory_id: uuid.UUID, db: Session = Depends(get_db)) -> dict:
    """Reject a memory candidate."""
    memory = db.execute(select(Memory).where(Memory.id == memory_id)).scalars().first()
    if not memory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")

    memory.status = "rejected"
    db.commit()
    db.refresh(memory)
    return {"status": "success", "id": str(memory.id), "memory_status": memory.status}


@router.patch("/memories/{memory_id}")
@router.patch("/admin/memories/{memory_id}")
def edit_memory(
    memory_id: uuid.UUID,
    body: MemoryEditRequest,
    db: Session = Depends(get_db),
) -> dict:
    """Edit memory content, category, or status."""
    memory = db.execute(select(Memory).where(Memory.id == memory_id)).scalars().first()
    if not memory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")

    if body.category is not None:
        memory.category = body.category.strip().lower()
    if body.content is not None:
        memory.content = body.content.strip()
    if body.status is not None:
        memory.status = body.status.strip().lower()

    db.commit()
    db.refresh(memory)
    return {
        "status": "success",
        "id": str(memory.id),
        "category": memory.category,
        "content": memory.content,
        "memory_status": memory.status,
    }


@router.delete("/memories/{memory_id}")
@router.delete("/admin/memories/{memory_id}")
def delete_memory(memory_id: uuid.UUID, db: Session = Depends(get_db)) -> dict:
    """Permanently delete a memory record."""
    memory = db.execute(select(Memory).where(Memory.id == memory_id)).scalars().first()
    if not memory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")

    db.delete(memory)
    db.commit()
    return {"status": "success", "deleted_id": str(memory_id)}


# --- Bot Settings Controls (Stage 6) ---

@router.get("/settings")
@router.get("/admin/settings")
def get_bot_settings(db: Session = Depends(get_db)) -> dict:
    """Retrieve global bot operational settings."""
    settings = get_or_create_bot_settings(db)
    return {
        "id": str(settings.id),
        "is_global_paused": settings.is_global_paused,
        "cooldown_seconds": settings.cooldown_seconds,
        "response_frequency": settings.response_frequency,
        "relevance_threshold": settings.relevance_threshold,
    }


@router.patch("/settings")
@router.patch("/admin/settings")
def update_bot_settings(
    body: BotSettingsUpdateRequest,
    db: Session = Depends(get_db),
) -> dict:
    """Update global bot settings."""
    settings = get_or_create_bot_settings(db)

    if body.is_global_paused is not None:
        settings.is_global_paused = body.is_global_paused
    if body.cooldown_seconds is not None:
        settings.cooldown_seconds = body.cooldown_seconds
    if body.response_frequency is not None:
        settings.response_frequency = body.response_frequency
    if body.relevance_threshold is not None:
        settings.relevance_threshold = body.relevance_threshold

    db.commit()
    db.refresh(settings)
    return {
        "status": "success",
        "is_global_paused": settings.is_global_paused,
        "cooldown_seconds": settings.cooldown_seconds,
        "response_frequency": settings.response_frequency,
        "relevance_threshold": settings.relevance_threshold,
    }


@router.post("/settings/pause")
@router.post("/admin/settings/pause")
def pause_bot_globally(db: Session = Depends(get_db)) -> dict:
    """Globally pause all automated replies across all conversations."""
    settings = get_or_create_bot_settings(db)
    settings.is_global_paused = True
    db.commit()
    return {"status": "success", "is_global_paused": True}


@router.post("/settings/resume")
@router.post("/admin/settings/resume")
def resume_bot_globally(db: Session = Depends(get_db)) -> dict:
    """Globally resume automated replies across all conversations."""
    settings = get_or_create_bot_settings(db)
    settings.is_global_paused = False
    db.commit()
    return {"status": "success", "is_global_paused": False}
