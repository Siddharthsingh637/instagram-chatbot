"""Conversation summarization adhering to ephemeral message retention policies."""

import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import desc, select
from sqlalchemy.orm import Session
from app.ai.provider_factory import get_ai_provider
from app.ai.providers.base import AIProvider
from app.core.logging import logger
from app.database.models import ConversationSummary, Message


class ConversationSummarizer:
    """Derives conversation summaries only from live, non-expired messages."""

    def __init__(self, provider: Optional[AIProvider] = None):
        self._provider = provider

    @property
    def provider(self) -> AIProvider:
        return self._provider or get_ai_provider()

    def generate_summary(self, db: Session, conversation_id: uuid.UUID) -> Optional[ConversationSummary]:
        """Summarize conversation from existing non-expired messages.

        Sets summary.expires_at to the latest message expires_at so summaries
        expire synchronously with temporary conversation context.
        """
        now = datetime.now(timezone.utc)

        # Retrieve messages that have not expired yet
        messages = (
            db.execute(
                select(Message)
                .where(
                    Message.conversation_id == conversation_id,
                    Message.expires_at > now,
                )
                .order_by(Message.created_at)
            )
            .scalars()
            .all()
        )

        if not messages or len(messages) < 2:
            return None

        # Build prompt from existing active messages
        dialogue = []
        for m in messages:
            speaker = "Assistant" if m.direction == "out" else "User"
            dialogue.append(f"{speaker}: {m.text}")

        prompt = (
            "Summarize the key topics and progress of this conversation succinctly in 1 to 3 sentences. "
            "Do NOT include sensitive information like passwords or financial data.\n\n"
            + "\n".join(dialogue)
        )

        summary_text = self.provider.generate_text(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are an expert conversation summarizer.",
            temperature=0.3,
            max_tokens=150,
        )

        # Synchronize summary expiration with the latest expiring message
        latest_expiration = max(m.expires_at for m in messages)
        if latest_expiration.tzinfo is None:
            latest_expiration = latest_expiration.replace(tzinfo=timezone.utc)
        last_message_id = messages[-1].id

        summary = ConversationSummary(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            summary=summary_text.strip(),
            last_message_id=last_message_id,
            expires_at=latest_expiration,
        )
        db.add(summary)
        db.commit()
        db.refresh(summary)

        logger.info(
            "Generated conversation summary id=%s for conversation=%s (expires_at=%s)",
            summary.id,
            conversation_id,
            summary.expires_at,
        )
        return summary
