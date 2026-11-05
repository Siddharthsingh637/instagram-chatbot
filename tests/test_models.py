"""Unit tests verifying models creation, UUID primary keys, and relationships."""

import uuid
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from app.database.models import (
    BotSettings,
    Conversation,
    ConversationSummary,
    InstagramAccount,
    Memory,
    Message,
    OutboundMessage,
    ProcessedEvent,
    User,
    UserPreference,
)


def test_models_lifecycle_and_relationships(db: Session):
    """Test full lifecycle of all 10 models, foreign keys, and relationships."""
    # 1. InstagramAccount
    account = InstagramAccount(
        external_id="ig_page_12345",
        username="my_bot_page",
        name="My Bot Page",
    )
    db.add(account)
    db.commit()
    db.refresh(account)
    assert isinstance(account.id, uuid.UUID)
    assert account.created_at is not None

    # 2. User & 7. UserPreference
    user = User(
        external_id="ig_user_98765",
        username="test_customer",
        full_name="Test Customer",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    assert isinstance(user.id, uuid.UUID)

    pref = UserPreference(
        user_id=user.id,
        preferred_persona="humorous",
        tone="witty",
    )
    db.add(pref)
    db.commit()
    db.refresh(pref)
    assert pref.user_id == user.id

    # 3. Conversation
    conv = Conversation(
        external_id="thread_123",
        instagram_account_id=account.id,
        user_id=user.id,
        conversation_type="dm",
        trigger_mode="always",
        paused=False,
        human_handoff=False,
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    assert conv.user_id == user.id
    assert conv.instagram_account_id == account.id

    # 4. Message
    now = datetime.now(timezone.utc)
    msg = Message(
        external_id="mid_99999",
        conversation_id=conv.id,
        sender_id=user.external_id,
        direction="in",
        text="Hello bot!",
        processing_status="stored",
        expires_at=now + timedelta(hours=6),
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    assert msg.conversation_id == conv.id
    assert msg.processing_status == "stored"

    # 5. Memory
    mem = Memory(
        user_id=user.id,
        category="preferences",
        content="User likes dark roast coffee",
        status="pending",
        source_message_id=msg.id,
    )
    db.add(mem)
    db.commit()
    db.refresh(mem)
    assert mem.user_id == user.id
    assert mem.source_message_id == msg.id
    assert mem.status == "pending"

    # 6. ConversationSummary
    summary = ConversationSummary(
        conversation_id=conv.id,
        summary="User introduced themselves and likes coffee",
        last_message_id=msg.id,
        expires_at=now + timedelta(hours=6),
    )
    db.add(summary)
    db.commit()
    db.refresh(summary)
    assert summary.conversation_id == conv.id

    # 8. BotSettings
    bot_settings = BotSettings(
        is_global_paused=False,
        cooldown_seconds=15,
        response_frequency=1.0,
        relevance_threshold=0.7,
    )
    db.add(bot_settings)
    db.commit()
    db.refresh(bot_settings)
    assert bot_settings.cooldown_seconds == 15
    assert bot_settings.relevance_threshold == 0.7

    # 9. ProcessedEvent
    proc_event = ProcessedEvent(
        external_id="evt_unique_12345",
        event_type="instagram_message",
    )
    db.add(proc_event)
    db.commit()
    db.refresh(proc_event)
    assert proc_event.external_id == "evt_unique_12345"

    # 10. OutboundMessage
    outbound = OutboundMessage(
        conversation_id=conv.id,
        recipient_id=user.external_id,
        text="Hi there, how can I help you?",
        status="sent",
        provider_message_id="mid_sent_555",
    )
    db.add(outbound)
    db.commit()
    db.refresh(outbound)
    assert outbound.status == "sent"
    assert outbound.conversation_id == conv.id
