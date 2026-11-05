"""Tests for Stage 5: Memory with owner control, sensitive filter, batch extraction, and retention cleanup."""

import uuid
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.core.config import get_settings
from app.database.models import Conversation, ConversationSummary, Memory, Message, User
from app.memory.manager import (
    CandidateMemory,
    MemoryExtractionResponse,
    MemoryManager,
    is_sensitive_data,
)
from app.memory.retention import cleanup_expired_data
from app.memory.summarizer import ConversationSummarizer
from tests.test_ai_providers import FakeProvider


def test_sensitive_data_filter():
    """Verify that sensitive categories and PII patterns are strictly rejected."""
    # 1. Passwords / Secrets
    assert is_sensitive_data("My password is supersecret123", "account")[0] is True
    assert is_sensitive_data("Here is the api_key for the server", "general")[0] is True

    # 2. Government IDs
    assert is_sensitive_data("My ssn is 123-45-6789", "profile")[0] is True
    assert is_sensitive_data("Passport number is P8473821", "identity")[0] is True

    # 3. Financial data
    assert is_sensitive_data("My credit card number is 4111111111111111", "payment")[0] is True
    assert is_sensitive_data("My salary is 150000 per year", "finance")[0] is True
    assert is_sensitive_data("Bank account routing number is 123456", "finance")[0] is True

    # 4. Medical / Health
    assert is_sensitive_data("I was diagnosed with bipolar disorder", "health")[0] is True
    assert is_sensitive_data("Taking prescription medication daily", "personal")[0] is True

    # 5. Precise physical address
    assert is_sensitive_data("I live at 123 Main Street, Apt 4B", "location")[0] is True

    # 6. Disallowed category name
    assert is_sensitive_data("Something innocuous", "medical")[0] is True
    assert is_sensitive_data("Something innocuous", "financial")[0] is True

    # 7. Safe personal disclosures - MUST PASS
    assert is_sensitive_data("I love drinking cappuccino with oat milk", "preferences")[0] is False
    assert is_sensitive_data("My golden retriever is named Cooper", "pets")[0] is False
    assert is_sensitive_data("I work as a software engineer", "profession")[0] is False
    assert is_sensitive_data("My favorite hobby is landscape photography", "hobbies")[0] is False


def test_auto_approved_categories(db: Session, monkeypatch):
    """Test memory saved has status=approved if category in AUTO_APPROVED_MEMORY_CATEGORIES."""
    settings = get_settings()
    monkeypatch.setattr(settings, "AUTO_APPROVED_MEMORY_CATEGORIES", ["coffee_preference", "hobbies"])

    user = User(external_id="user_auto_approve")
    db.add(user)
    db.commit()

    manager = MemoryManager(provider=FakeProvider())

    # 1. In auto-approved list -> 'approved'
    mem1 = manager.save_candidate_memory(
        db, user.id, category="coffee_preference", content="Likes dark roast"
    )
    assert mem1 is not None
    assert mem1.status == "approved"

    # 2. Not in list -> 'pending'
    mem2 = manager.save_candidate_memory(
        db, user.id, category="travel_plans", content="Wants to visit Tokyo"
    )
    assert mem2 is not None
    assert mem2.status == "pending"


def test_batch_memory_extraction(db: Session):
    """Test candidate extraction in batch updates processing_status to extracted and saves memories."""
    user = User(external_id="batch_user")
    db.add(user)
    db.commit()

    conv = Conversation(external_id="batch_conv", user_id=user.id)
    db.add(conv)
    db.commit()

    now = datetime.now(timezone.utc)
    m1 = Message(
        conversation_id=conv.id,
        sender_id=user.external_id,
        direction="in",
        text="I love playing tennis on weekends",
        processing_status="stored",
        expires_at=now + timedelta(hours=6),
    )
    m2 = Message(
        conversation_id=conv.id,
        sender_id=user.external_id,
        direction="in",
        text="My golden retriever is named Charlie",
        processing_status="stored",
        expires_at=now + timedelta(hours=6),
    )
    db.add_all([m1, m2])
    db.commit()

    extraction_response = MemoryExtractionResponse(
        candidates=[
            CandidateMemory(category="hobbies", content="Plays tennis on weekends"),
            CandidateMemory(category="pets", content="Has a golden retriever named Charlie"),
        ]
    )
    fake_provider = FakeProvider(structured_response=extraction_response)
    manager = MemoryManager(provider=fake_provider)

    saved_count = manager.extract_candidate_memories_batch(db, batch_size=10)

    assert saved_count >= 2
    assert m1.processing_status == "extracted"
    assert m2.processing_status == "extracted"

    # Verify pending memories saved
    memories = db.execute(select(Memory).where(Memory.user_id == user.id)).scalars().all()
    assert len(memories) >= 2
    contents = [m.content for m in memories]
    assert any("tennis" in c.lower() for c in contents)
    assert any("charlie" in c.lower() for c in contents)


def test_conversation_summarizer_expiry_synchronization(db: Session):
    """Test summarizer derives summary from non-expired messages and synchronizes expiration."""
    user = User(external_id="sum_user")
    conv = Conversation(external_id="sum_conv", user_id=user.id)
    db.add_all([user, conv])
    db.commit()

    now = datetime.now(timezone.utc)
    target_expiry = now + timedelta(hours=3)

    m1 = Message(
        conversation_id=conv.id,
        sender_id=user.external_id,
        direction="in",
        text="Can I get pricing for the premium plan?",
        expires_at=target_expiry,
    )
    m2 = Message(
        conversation_id=conv.id,
        sender_id="bot",
        direction="out",
        text="Our premium plan is $29/mo.",
        expires_at=target_expiry,
    )
    db.add_all([m1, m2])
    db.commit()

    fake_provider = FakeProvider(text_response="User inquired about pricing for the premium plan.")
    summarizer = ConversationSummarizer(provider=fake_provider)
    summary = summarizer.generate_summary(db, conv.id)

    assert summary is not None
    assert summary.summary == "User inquired about pricing for the premium plan."
    summary_expiry = summary.expires_at
    if summary_expiry.tzinfo is None:
        summary_expiry = summary_expiry.replace(tzinfo=timezone.utc)
    assert summary_expiry == target_expiry


def test_retention_cleanup_policy(db: Session, client: TestClient):
    """Test cleanup_expired_data truly deletes expired messages and keeps unexpired ones."""
    conv = Conversation(external_id="cleanup_conv")
    db.add(conv)
    db.commit()

    now = datetime.now(timezone.utc)

    # 1. Expired message (expired 1 hour ago)
    expired_msg = Message(
        conversation_id=conv.id,
        sender_id="user_old",
        direction="in",
        text="Old expired message",
        expires_at=now - timedelta(hours=1),
    )
    # 2. Fresh message (expires in 5 hours)
    fresh_msg = Message(
        conversation_id=conv.id,
        sender_id="user_new",
        direction="in",
        text="Fresh message that must stay",
        expires_at=now + timedelta(hours=5),
    )
    # 3. Expired summary
    expired_summary = ConversationSummary(
        conversation_id=conv.id,
        summary="Old summary",
        expires_at=now - timedelta(minutes=10),
    )
    # 4. Fresh summary
    fresh_summary = ConversationSummary(
        conversation_id=conv.id,
        summary="Active summary",
        expires_at=now + timedelta(hours=2),
    )
    db.add_all([expired_msg, fresh_msg, expired_summary, fresh_summary])
    db.commit()

    # Retain IDs before cleanup commits
    expired_msg_id = expired_msg.id
    expired_summary_id = expired_summary.id
    fresh_msg_id = fresh_msg.id
    fresh_summary_id = fresh_summary.id

    # Trigger cleanup function
    counts = cleanup_expired_data(db)
    assert counts["deleted_messages"] == 1
    assert counts["deleted_summaries"] == 1

    # Verify expired items are deleted
    assert db.execute(select(Message).where(Message.id == expired_msg_id)).scalars().first() is None
    assert db.execute(select(ConversationSummary).where(ConversationSummary.id == expired_summary_id)).scalars().first() is None

    # Verify unexpired items are KEPT
    kept_msg = db.execute(select(Message).where(Message.id == fresh_msg_id)).scalars().first()
    assert kept_msg is not None
    assert kept_msg.text == "Fresh message that must stay"

    kept_summary = db.execute(select(ConversationSummary).where(ConversationSummary.id == fresh_summary_id)).scalars().first()
    assert kept_summary is not None
    assert kept_summary.summary == "Active summary"

    # Test POST /admin/cleanup API route
    settings = get_settings()
    resp = client.post("/admin/cleanup", headers={"X-Admin-Key": settings.ADMIN_API_KEY})
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
