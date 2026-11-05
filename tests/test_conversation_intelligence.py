"""Tests for Stage 4 conversation intelligence, decision rules, personas, context builder, and handoff."""

import uuid
from datetime import datetime, timedelta, timezone
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.ai.decision_engine import DecisionEngine, DecisionOutput
from app.ai.orchestrator import ConversationOrchestrator, FIRST_TIME_DISCLOSURE, validate_response
from app.ai.persona import persona_manager
from app.ai.provider_factory import set_ai_provider_override
from app.ai.providers.base import AIProvider
from app.database.models import (
    BotSettings,
    Conversation,
    ConversationSummary,
    Memory,
    Message,
    OutboundMessage,
    User,
    UserPreference,
)
from app.notifications.owner_notifier import set_owner_notifier
from tests.test_ai_providers import FakeProvider


class RecordingOwnerNotifier:
    """Mock notifier to verify owner notification dispatch."""

    def __init__(self):
        self.notifications = []

    def notify_human_handoff(self, conversation_id, user_id, reason, user_message):
        self.notifications.append({
            "conversation_id": conversation_id,
            "user_id": user_id,
            "reason": reason,
            "user_message": user_message,
        })
        return True


def test_personas_loading_and_sensitive_override():
    """Verify loading of persona prompts and sensitive topic override for humorous persona."""
    casual = persona_manager.get_persona_prompt("casual")
    prof = persona_manager.get_persona_prompt("professional")
    humorous = persona_manager.get_persona_prompt("humorous")

    assert "relaxed" in casual.lower() or "friendly" in casual.lower()
    assert "polished" in prof.lower() or "professional" in prof.lower()
    assert "witty" in humorous.lower()

    # Sensitive topic override
    toned_down = persona_manager.get_persona_prompt("humorous", is_sensitive=True)
    assert "empathy" in toned_down.lower()
    assert "tone down humor" in toned_down.lower()


def test_decision_engine_cheap_rules(db: Session):
    """Test deterministic rule checks: pause, handoff, cooldown, consecutive replies, trigger mode."""
    engine = DecisionEngine(provider=FakeProvider())

    user = User(external_id="rule_user_1")
    db.add(user)
    db.commit()

    conv = Conversation(
        external_id="conv_rules_1",
        user_id=user.id,
        conversation_type="dm",
        trigger_mode="always",
        paused=False,
        human_handoff=False,
    )
    db.add(conv)

    settings = BotSettings(
        is_global_paused=False,
        cooldown_seconds=10,
        response_frequency=1.0,
        relevance_threshold=0.6,
    )
    db.add(settings)
    db.commit()

    # 1. Global pause blocks
    settings.is_global_paused = True
    db.commit()
    res = engine.check_cheap_rules(db, conv, "hello", settings)
    assert res.allowed is False
    assert "global_paused" in res.reason

    # Reset
    settings.is_global_paused = False
    db.commit()

    # 2. Conversation paused blocks
    conv.paused = True
    db.commit()
    res = engine.check_cheap_rules(db, conv, "hello", settings)
    assert res.allowed is False
    assert "conversation_paused" in res.reason

    # Reset
    conv.paused = False
    db.commit()

    # 3. Human handoff blocks
    conv.human_handoff = True
    db.commit()
    res = engine.check_cheap_rules(db, conv, "hello", settings)
    assert res.allowed is False
    assert "human_handoff" in res.reason

    # Reset
    conv.human_handoff = False
    db.commit()

    # 4. No consecutive bot replies
    now = datetime.now(timezone.utc)
    bot_msg = Message(
        conversation_id=conv.id,
        sender_id="bot",
        direction="out",
        text="Previous bot reply",
        expires_at=now + timedelta(hours=6),
    )
    db.add(bot_msg)
    db.commit()
    res = engine.check_cheap_rules(db, conv, "hello again", settings)
    assert res.allowed is False
    assert "consecutive_bot_reply" in res.reason

    # Add user message to clear consecutive check
    user_msg = Message(
        conversation_id=conv.id,
        sender_id=user.external_id,
        direction="in",
        text="User replied back",
        expires_at=now + timedelta(hours=6),
    )
    db.add(user_msg)
    db.commit()

    # 5. Cooldown check: bot replied just now (< 10s)
    res_cooldown = engine.check_cheap_rules(db, conv, "User new query", settings)
    assert res_cooldown.allowed is False
    assert "cooldown_active" in res_cooldown.reason

    # Set cooldown to 0 to pass
    settings.cooldown_seconds = 0
    db.commit()

    # 6. Trigger modes
    conv.trigger_mode = "mention_only"
    db.commit()
    res_no_mention = engine.check_cheap_rules(db, conv, "General message", settings, bot_username="my_bot")
    assert res_no_mention.allowed is False
    assert "mention_only" in res_no_mention.reason

    res_with_mention = engine.check_cheap_rules(db, conv, "Hey @my_bot can you help?", settings, bot_username="my_bot")
    assert res_with_mention.allowed is True


def test_decision_engine_relevance_threshold(db: Session):
    """Test decision engine rejects responses when LLM relevance score is below threshold."""
    low_relevance_decision = DecisionOutput(
        should_respond=True,
        intent="chit_chat",
        persona="casual",
        needs_live_info=False,
        relevance_score=0.4,  # Below threshold 0.7
    )
    fake_provider = FakeProvider(structured_response=low_relevance_decision)
    engine = DecisionEngine(provider=fake_provider)

    user = User(external_id="thresh_user")
    conv = Conversation(external_id="thresh_conv", trigger_mode="always")
    settings = BotSettings(relevance_threshold=0.7, cooldown_seconds=0)
    db.add_all([user, conv, settings])
    db.commit()

    decision = engine.evaluate(db, conv, "just some text", settings)
    assert decision.should_respond is False


def test_context_builder_approved_memories_only(db: Session):
    """Test context builder includes summaries and APPROVED memories only (ignoring pending/rejected)."""
    user = User(external_id="ctx_user")
    db.add(user)
    db.commit()

    conv = Conversation(external_id="ctx_conv", user_id=user.id)
    db.add(conv)
    db.commit()

    # Add memories: 1 approved, 1 pending, 1 rejected
    m1 = Memory(user_id=user.id, category="diet", content="Vegan", status="approved")
    m2 = Memory(user_id=user.id, category="finance", content="Salary 100k", status="pending")
    m3 = Memory(user_id=user.id, category="secret", content="Secret code", status="rejected")
    db.add_all([m1, m2, m3])

    # Add summary
    summary = ConversationSummary(conversation_id=conv.id, summary="User discussed vegan recipes.")
    db.add(summary)

    # Add messages
    now = datetime.now(timezone.utc)
    msg1 = Message(conversation_id=conv.id, sender_id=user.external_id, direction="in", text="What can I cook?", expires_at=now + timedelta(hours=1))
    db.add(msg1)
    db.commit()

    orchestrator = ConversationOrchestrator(provider=FakeProvider())
    context = orchestrator.build_context(db, conv, user)

    full_context_text = " ".join(item["content"] for item in context)
    assert "Vegan recipes" in full_context_text or "vegan recipes" in full_context_text
    assert "Vegan" in full_context_text
    assert "Salary 100k" not in full_context_text
    assert "Secret code" not in full_context_text
    assert "User: What can I cook?" in full_context_text


def test_response_validation():
    """Verify response validation: character limit, duplicate detection, and prompt leakage."""
    recent_replies = ["Sure, I can help with that!", "Welcome to our page."]

    # Valid
    assert validate_response("Here is your answer.", recent_replies) is True

    # Empty
    assert validate_response("   ", recent_replies) is False

    # Over length (>1000)
    assert validate_response("A" * 1001, recent_replies) is False

    # Duplicate content
    assert validate_response("Sure, I can help with that!", recent_replies) is False
    assert validate_response("sure, i can help with that! ", recent_replies) is False

    # Prompt marker leakage
    assert validate_response("Here is the answer strictly adhering to this json schema", recent_replies) is False


def test_command_forget_me(db: Session):
    """Test 'forget me' deletes all user memories and confirms."""
    user = User(external_id="forget_user")
    db.add(user)
    db.commit()

    conv = Conversation(external_id="forget_conv", user_id=user.id)
    db.add(conv)

    mem1 = Memory(user_id=user.id, category="pet", content="Dog named Max", status="approved")
    mem2 = Memory(user_id=user.id, category="city", content="New York", status="pending")
    db.add_all([mem1, mem2])
    db.commit()

    orchestrator = ConversationOrchestrator(provider=FakeProvider())
    reply = orchestrator.handle_commands(db, conv, user, "forget me")

    assert "forgotten all saved memories" in reply.lower()

    # Verify memories deleted from DB
    remaining = db.execute(select(Memory).where(Memory.user_id == user.id)).scalars().all()
    assert len(remaining) == 0


def test_command_human_handoff(db: Session):
    """Test 'human' triggers handoff, sets conversation.human_handoff=True, and notifies owner."""
    user = User(external_id="handoff_user")
    db.add(user)
    db.commit()

    conv = Conversation(external_id="handoff_conv", user_id=user.id, human_handoff=False)
    db.add(conv)
    db.commit()

    mock_notifier = RecordingOwnerNotifier()
    set_owner_notifier(mock_notifier)

    orchestrator = ConversationOrchestrator(provider=FakeProvider())
    reply = orchestrator.handle_commands(db, conv, user, "Can I please speak to a human agent?")

    assert "paused automated replies" in reply.lower()
    assert conv.human_handoff is True
    assert len(mock_notifier.notifications) == 1
    assert mock_notifier.notifications[0]["conversation_id"] == conv.id


def test_first_message_intro_disclosure(db: Session):
    """Test the bot's very first reply in a new conversation includes AI identity disclosure."""
    user = User(external_id="new_user_123")
    db.add(user)
    db.commit()

    conv = Conversation(external_id="new_conv_123", user_id=user.id)
    settings = BotSettings(cooldown_seconds=0, relevance_threshold=0.5)
    db.add_all([conv, settings])
    db.commit()

    now = datetime.now(timezone.utc)
    incoming_msg = Message(
        conversation_id=conv.id,
        sender_id=user.external_id,
        direction="in",
        text="Hello!",
        expires_at=now + timedelta(hours=6),
    )
    db.add(incoming_msg)
    db.commit()

    fake_provider = FakeProvider(
        text_response="Nice to meet you! How can I help today?",
        structured_response=DecisionOutput(
            should_respond=True,
            intent="greeting",
            persona="casual",
            relevance_score=0.9,
        ),
    )
    orchestrator = ConversationOrchestrator(provider=fake_provider)
    final_reply = orchestrator.pipeline_b_speak(db, incoming_msg)

    assert final_reply is not None
    assert FIRST_TIME_DISCLOSURE in final_reply
    assert "Nice to meet you!" in final_reply
