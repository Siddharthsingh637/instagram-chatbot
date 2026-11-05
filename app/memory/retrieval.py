"""Memory retrieval queries for conversation context and admin moderation."""

import uuid
from typing import List, Optional
from sqlalchemy import desc, select
from sqlalchemy.orm import Session
from app.database.models import Memory


def get_approved_memories_for_user(db: Session, user_id: uuid.UUID) -> List[Memory]:
    """Retrieve all approved memories for a user, ordered chronologically."""
    query = (
        select(Memory)
        .where(Memory.user_id == user_id, Memory.status == "approved")
        .order_by(Memory.created_at)
    )
    return list(db.execute(query).scalars().all())


def get_all_memories_for_user(db: Session, user_id: uuid.UUID) -> List[Memory]:
    """Retrieve all memories (pending, approved, rejected) for a user."""
    query = select(Memory).where(Memory.user_id == user_id).order_by(desc(Memory.created_at))
    return list(db.execute(query).scalars().all())


def get_pending_memories(db: Session, limit: int = 50) -> List[Memory]:
    """Retrieve pending memories awaiting admin approval."""
    query = (
        select(Memory)
        .where(Memory.status == "pending")
        .order_by(desc(Memory.created_at))
        .limit(limit)
    )
    return list(db.execute(query).scalars().all())
