"""Conversation Intelligence Orchestrator implementing Pipeline A (UNDERSTAND) and Pipeline B (SPEAK)."""

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from sqlalchemy import desc, select
from sqlalchemy.orm import Session
from app.ai.decision_engine import DecisionEngine, DecisionOutput
from app.ai.persona import persona_manager
from app.ai.provider_factory import get_ai_provider
from app.ai.providers.base import AIProvider
from app.core.config import get_settings
from app.core.logging import logger
from app.database.models import (
    BotSettings,
    Conversation,
    ConversationSummary,
    Memory,
    Message,
    User,
    UserPreference,
)
from app.database.repositories import (
    get_or_create_bot_settings,
    get_or_create_conversation,
    get_or_create_user,
    is_event_processed,
    record_processed_event,
)
from app.integrations.instagram.sender import InstagramSender
from app.notifications.owner_notifier import get_owner_notifier

FIRST_TIME_DISCLOSURE = (
    "👋 Hi! I'm an AI assistant. I can remember details you share to provide better answers. "
    "If you ever want me to forget them, just type 'forget me'. "
    "If you prefer speaking to a human, type 'human' at any time."
)


def validate_response(candidate_text: str, recent_bot_texts: List[str]) -> bool:
    """Validate candidate response against length, duplicate content, and safety rules.

    Args:
        candidate_text: Generated reply text.
        recent_bot_texts: List of recent bot messages in the same conversation.

    Returns:
        bool: True if response is safe and valid to send, False otherwise.
    """
    cleaned = candidate_text.strip()
    if not cleaned:
        logger.warning("Validation rejected: empty response")
        return False

    # 1. Instagram message character limit
    if len(cleaned) > 1000:
        logger.warning("Validation rejected: response exceeds 1000 chars (%d)", len(cleaned))
        return False

    # 2. Duplicate content check
    normalized_candidate = re.sub(r"\s+", " ", cleaned.lower())
    for prev in recent_bot_texts:
        normalized_prev = re.sub(r"\s+", " ", prev.strip().lower())
        if normalized_candidate == normalized_prev:
            logger.warning("Validation rejected: duplicate response detected")
            return False

    # 3. Basic safety check (ensure system prompt markers not leaked)
    leakage_markers = ["[persona guidelines]", "strictly adhering to this json schema", "system prompt"]
    if any(marker in cleaned.lower() for marker in leakage_markers):
        logger.warning("Validation rejected: prompt marker leakage detected")
        return False

    return True


class ConversationOrchestrator:
    """Manages the two decoupled pipelines: Pipeline A (UNDERSTAND) and Pipeline B (SPEAK)."""

    def __init__(
        self,
        decision_engine: Optional[DecisionEngine] = None,
        provider: Optional[AIProvider] = None,
        sender: Optional[InstagramSender] = None,
    ):
        self._provider = provider
        self.decision_engine = decision_engine or DecisionEngine(provider=self._provider)
        self.sender = sender or InstagramSender()

    @property
    def provider(self) -> AIProvider:
        return self._provider or get_ai_provider()

    def pipeline_a_understand(
        self,
        db: Session,
        sender_id: str,
        recipient_id: str,
        message_id: str,
        text: str,
    ) -> Optional[Message]:
        """Pipeline A: Store and track every incoming message. Pure rules, zero LLM calls."""
        if is_event_processed(db, message_id):
            logger.info("Message mid=%s already processed in Pipeline A", message_id)
            return None

        user = get_or_create_user(db, external_id=sender_id)
        conv_external_id = f"ig_{sender_id}_{recipient_id}"
        conversation = get_or_create_conversation(
            db,
            external_id=conv_external_id,
            user_id=user.id,
            conversation_type="dm",
        )

        settings = get_settings()
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=settings.MESSAGE_RETENTION_HOURS)

        message = Message(
            id=uuid.uuid4(),
            external_id=message_id,
            conversation_id=conversation.id,
            sender_id=sender_id,
            direction="in",
            text=text,
            processing_status="stored",
            expires_at=expires_at,
        )
        db.add(message)
        record_processed_event(db, external_id=message_id, event_type="instagram_message")

        db.commit()
        db.refresh(message)
        return message

    def build_context(
        self,
        db: Session,
        conversation: Conversation,
        user: User,
        max_messages: int = 15,
    ) -> List[Dict[str, str]]:
        """Construct LLM conversation context with APPROVED memories and conversation summaries."""
        # 1. Fetch recent messages
        recent_messages = (
            db.execute(
                select(Message)
                .where(Message.conversation_id == conversation.id)
                .order_by(desc(Message.created_at))
                .limit(max_messages)
            )
            .scalars()
            .all()
        )
        # Reverse to chronological order
        recent_messages = list(reversed(recent_messages))

        # 2. Fetch latest conversation summary
        latest_summary = (
            db.execute(
                select(ConversationSummary)
                .where(ConversationSummary.conversation_id == conversation.id)
                .order_by(desc(ConversationSummary.created_at))
            )
            .scalars()
            .first()
        )

        # 3. Fetch APPROVED memories only
        approved_memories = (
            db.execute(
                select(Memory)
                .where(Memory.user_id == user.id, Memory.status == "approved")
                .order_by(Memory.created_at)
            )
            .scalars()
            .all()
        )

        context_notes = []
        if latest_summary:
            context_notes.append(f"Previous Conversation Summary: {latest_summary.summary}")

        if approved_memories:
            memory_list = "\n".join(f"- [{m.category}] {m.content}" for m in approved_memories)
            context_notes.append(f"Verified facts you remember about the user:\n{memory_list}")

        chat_messages: List[Dict[str, str]] = []

        if context_notes:
            chat_messages.append({
                "role": "system",
                "content": "\n\n".join(context_notes),
            })

        for msg in recent_messages:
            role = "assistant" if msg.direction == "out" else "user"
            speaker_label = "Assistant" if role == "assistant" else "User"
            chat_messages.append({"role": role, "content": f"{speaker_label}: {msg.text}"})

        return chat_messages

    def handle_commands(
        self,
        db: Session,
        conversation: Conversation,
        user: User,
        text: str,
    ) -> Optional[str]:
        """Check for command keywords like 'forget me' or 'human'."""
        normalized = text.strip().lower()
        settings = get_settings()

        # 1. "forget me" command: deletes all user memories
        if normalized in ["forget me", "forget_me", "delete memories"]:
            user_memories = (
                db.execute(select(Memory).where(Memory.user_id == user.id))
                .scalars()
                .all()
            )
            for mem in user_memories:
                db.delete(mem)
            db.commit()
            logger.info("Executed 'forget me' command for user %s: deleted all memories", user.id)
            return "I've forgotten all saved memories and personal details about you."

        # 2. Human handoff command or keyword match
        is_handoff_command = normalized in ["human", "agent", "support", "help"]
        is_keyword_match = any(kw in normalized for kw in settings.HANDOFF_KEYWORDS)

        if is_handoff_command or is_keyword_match:
            conversation.human_handoff = True
            db.commit()

            notifier = get_owner_notifier()
            notifier.notify_human_handoff(
                conversation_id=conversation.id,
                user_id=user.id,
                reason="User triggered handoff command / keyword",
                user_message=text,
            )
            logger.info("Conversation %s switched to human_handoff=True", conversation.id)
            return "I've paused automated replies and notified our team. A human representative will assist you shortly."

        return None

    def pipeline_b_speak(
        self,
        db: Session,
        message: Message,
        bot_username: Optional[str] = None,
    ) -> Optional[str]:
        """Pipeline B: Evaluates whether to reply, chooses persona, builds context, generates and sends reply."""
        conversation = db.execute(
            select(Conversation).where(Conversation.id == message.conversation_id)
        ).scalars().first()
        if not conversation:
            return None

        user = db.execute(
            select(User).where(User.id == conversation.user_id)
        ).scalars().first()
        if not user:
            return None

        settings = get_settings()
        bot_settings = get_or_create_bot_settings(db)

        # 1. Check commands first
        command_reply = self.handle_commands(db, conversation, user, message.text)
        if command_reply:
            self._send_reply(db, conversation, user, command_reply)
            return command_reply

        # 2. Evaluate with Decision Engine
        user_preference = db.execute(
            select(UserPreference).where(UserPreference.user_id == user.id)
        ).scalars().first()

        decision = self.decision_engine.evaluate(
            db=db,
            conversation=conversation,
            incoming_text=message.text,
            bot_settings=bot_settings,
            bot_username=bot_username,
            user_preference=user_preference,
        )

        if not decision.should_respond:
            logger.info("Pipeline B decided not to speak for conversation %s", conversation.id)
            return None

        # 3. Determine if this is the bot's first reply in this conversation
        prior_bot_messages_count = (
            db.execute(
                select(Message).where(
                    Message.conversation_id == conversation.id,
                    Message.direction == "out",
                )
            )
            .scalars()
            .all()
        )
        is_first_reply = len(prior_bot_messages_count) == 0

        # 4. Build context
        context_messages = self.build_context(
            db=db,
            conversation=conversation,
            user=user,
            max_messages=settings.MAX_CONTEXT_MESSAGES,
        )

        # 5. Build persona system prompt
        system_directive = persona_manager.build_system_directive(
            persona_name=decision.persona,
            is_sensitive=False,
        )

        # 6. Generate reply with AIProvider
        candidate_reply = self.provider.generate_text(
            messages=context_messages,
            system_prompt=system_directive,
            temperature=0.7,
            max_tokens=250,
        )

        # 7. Validate response
        recent_bot_texts = [m.text for m in prior_bot_messages_count[-5:]]
        if not validate_response(candidate_reply, recent_bot_texts):
            logger.warning("Pipeline B: Response validation failed; message withheld.")
            return None

        # If first reply, prepend onboarding disclosure
        final_reply = candidate_reply
        if is_first_reply:
            final_reply = f"{FIRST_TIME_DISCLOSURE}\n\n{candidate_reply}"

        # 8. Send reply via sender and persist outbound message
        self._send_reply(db, conversation, user, final_reply)
        return final_reply

    def _send_reply(
        self,
        db: Session,
        conversation: Conversation,
        user: User,
        reply_text: str,
    ) -> None:
        """Send message via Instagram Send API and store an outbound Message record."""
        # Dispatch to Instagram
        self.sender.send_text_message(
            recipient_id=user.external_id,
            text=reply_text,
            db=db,
            conversation_id=conversation.id,
        )

        # Record outbound message in conversation history for retention & context
        settings = get_settings()
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=settings.MESSAGE_RETENTION_HOURS)

        outbound_msg = Message(
            id=uuid.uuid4(),
            conversation_id=conversation.id,
            sender_id=settings.INSTAGRAM_ACCOUNT_ID or "bot_assistant",
            direction="out",
            text=reply_text,
            processing_status="stored",
            expires_at=expires_at,
        )
        db.add(outbound_msg)
        db.commit()
