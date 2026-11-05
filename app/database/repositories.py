"""Database repository functions for queries and transactions."""

import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.database.models import (
    BotSettings,
    Conversation,
    InstagramAccount,
    Message,
    ProcessedEvent,
    User,
    UserPreference,
)


def get_or_create_bot_settings(db: Session) -> BotSettings:
    """Retrieve the global bot settings singleton, or create it with defaults."""
    settings = db.execute(select(BotSettings)).scalars().first()
    if not settings:
        settings = BotSettings(
            id=uuid.uuid4(),
            is_global_paused=False,
            cooldown_seconds=10,
            response_frequency=1.0,
            relevance_threshold=0.6,
        )
        db.add(settings)
        db.commit()
        db.refresh(settings)
    return settings


def is_event_processed(db: Session, external_id: str) -> bool:
    """Check if an event with this external ID has already been recorded."""
    query = select(ProcessedEvent).where(ProcessedEvent.external_id == external_id)
    return db.execute(query).scalars().first() is not None


def record_processed_event(db: Session, external_id: str, event_type: str = "instagram_message") -> ProcessedEvent:
    """Record an event ID to prevent duplicate processing."""
    event = ProcessedEvent(
        id=uuid.uuid4(),
        external_id=external_id,
        event_type=event_type,
        processed_at=datetime.now(timezone.utc),
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def get_or_create_user(db: Session, external_id: str, username: Optional[str] = None) -> User:
    """Retrieve an existing user by external ID or create a new one."""
    user = db.execute(select(User).where(User.external_id == external_id)).scalars().first()
    if not user:
        user = User(
            id=uuid.uuid4(),
            external_id=external_id,
            username=username,
        )
        db.add(user)
        # Create default preference for user
        pref = UserPreference(
            id=uuid.uuid4(),
            user_id=user.id,
            preferred_persona="casual",
            tone="friendly",
        )
        db.add(pref)
        db.commit()
        db.refresh(user)
    elif username and user.username != username:
        user.username = username
        db.commit()
        db.refresh(user)
    return user


def get_or_create_conversation(
    db: Session,
    external_id: str,
    user_id: Optional[uuid.UUID] = None,
    conversation_type: str = "dm",
    instagram_account_id: Optional[uuid.UUID] = None,
) -> Conversation:
    """Retrieve an existing conversation or create a new one."""
    conv = db.execute(select(Conversation).where(Conversation.external_id == external_id)).scalars().first()
    if not conv:
        conv = Conversation(
            id=uuid.uuid4(),
            external_id=external_id,
            user_id=user_id,
            conversation_type=conversation_type,
            instagram_account_id=instagram_account_id,
            trigger_mode="always",
            paused=False,
            human_handoff=False,
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
    return conv
