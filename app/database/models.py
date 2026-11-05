"""SQLAlchemy 2.0 database models for Instagram Bot."""

import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database.connection import Base


def utcnow() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class TimestampMixin:
    """Mixin providing created_at and updated_at timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )


class InstagramAccount(Base, TimestampMixin):
    """Instagram accounts/pages connected to this bot instance."""

    __tablename__ = "instagram_accounts"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    external_id: Mapped[str] = mapped_column(
        String(128), unique=True, index=True, nullable=False
    )
    username: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    conversations: Mapped[List["Conversation"]] = relationship(
        "Conversation", back_populates="instagram_account"
    )


class User(Base, TimestampMixin):
    """Users interacting with the chatbot across platforms."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    external_id: Mapped[str] = mapped_column(
        String(128), unique=True, index=True, nullable=False
    )
    username: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    conversations: Mapped[List["Conversation"]] = relationship(
        "Conversation", back_populates="user"
    )
    memories: Mapped[List["Memory"]] = relationship(
        "Memory", back_populates="user", cascade="all, delete-orphan"
    )
    preference: Mapped[Optional["UserPreference"]] = relationship(
        "UserPreference", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )


class Conversation(Base, TimestampMixin):
    """Conversation threads (DMs or groups)."""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    external_id: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    instagram_account_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("instagram_accounts.id", ondelete="SET NULL"),
        nullable=True,
    )
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    conversation_type: Mapped[str] = mapped_column(
        String(32), default="dm", nullable=False
    )  # dm | group
    trigger_mode: Mapped[str] = mapped_column(
        String(64), default="always", nullable=False
    )  # always | mention_only | mention_plus_occasional
    paused: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    human_handoff: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    instagram_account: Mapped[Optional["InstagramAccount"]] = relationship(
        "InstagramAccount", back_populates="conversations"
    )
    user: Mapped[Optional["User"]] = relationship(
        "User", back_populates="conversations"
    )
    messages: Mapped[List["Message"]] = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )
    summaries: Mapped[List["ConversationSummary"]] = relationship(
        "ConversationSummary", back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base, TimestampMixin):
    """Individual messages sent or received in conversations."""

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    external_id: Mapped[Optional[str]] = mapped_column(
        String(255), unique=True, index=True, nullable=True
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sender_id: Mapped[str] = mapped_column(
        String(128), nullable=False, index=True
    )
    direction: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # in | out
    text: Mapped[str] = mapped_column(Text, nullable=False)
    processing_status: Mapped[str] = mapped_column(
        String(32), default="stored", nullable=False
    )  # stored | summarized | extracted
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, nullable=False
    )

    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )


class Memory(Base, TimestampMixin):
    """Extracted knowledge and facts about a user."""

    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    category: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default="pending", nullable=False, index=True
    )  # pending | approved | rejected
    source_message_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
    )

    user: Mapped["User"] = relationship("User", back_populates="memories")


class ConversationSummary(Base, TimestampMixin):
    """Rolling summaries of conversations for efficient LLM context."""

    __tablename__ = "conversation_summaries"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    last_message_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), index=True, nullable=True
    )

    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="summaries"
    )


class UserPreference(Base, TimestampMixin):
    """Personalized user preferences for chatbot interaction."""

    __tablename__ = "user_preferences"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )
    preferred_persona: Mapped[str] = mapped_column(
        String(64), default="casual", nullable=False
    )
    tone: Mapped[str] = mapped_column(
        String(64), default="friendly", nullable=False
    )

    user: Mapped["User"] = relationship("User", back_populates="preference")


class BotSettings(Base, TimestampMixin):
    """Global bot settings and operational thresholds."""

    __tablename__ = "bot_settings"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    is_global_paused: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    cooldown_seconds: Mapped[int] = mapped_column(
        Integer, default=10, nullable=False
    )
    response_frequency: Mapped[float] = mapped_column(
        Float, default=1.0, nullable=False
    )
    relevance_threshold: Mapped[float] = mapped_column(
        Float, default=0.6, nullable=False
    )


class ProcessedEvent(Base, TimestampMixin):
    """Event deduplication ledger to guarantee idempotency."""

    __tablename__ = "processed_events"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    external_id: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    event_type: Mapped[str] = mapped_column(
        String(64), default="instagram_message", nullable=False
    )
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class OutboundMessage(Base, TimestampMixin):
    """Record of every outgoing message attempt and status."""

    __tablename__ = "outbound_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    recipient_id: Mapped[str] = mapped_column(
        String(128), nullable=False, index=True
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default="pending", nullable=False, index=True
    )  # pending | sent | failed
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    provider_message_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
