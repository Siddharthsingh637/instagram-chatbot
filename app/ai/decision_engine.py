"""Decision engine: Cheap deterministic rule checks followed by structured LLM evaluation."""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.orm import Session
from app.ai.provider_factory import get_ai_provider
from app.ai.providers.base import AIProvider
from app.core.logging import logger
from app.database.models import BotSettings, Conversation, Message, UserPreference


class DecisionOutput(BaseModel):
    """Structured LLM decision output."""

    should_respond: bool = Field(
        ...,
        description="Whether the bot should generate and send an automated response.",
    )
    intent: str = Field(
        ...,
        description="Detected intent of the message (e.g. greeting, question, statement, handoff).",
    )
    persona: str = Field(
        default="casual",
        description="Recommended persona: casual, professional, or humorous.",
    )
    needs_live_info: bool = Field(
        default=False,
        description="Whether this query needs live search grounding.",
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence / relevance of responding to this message from 0.0 to 1.0.",
    )


class RuleCheckResult:
    """Outcome of cheap rule checks."""

    def __init__(self, allowed: bool, reason: str = ""):
        self.allowed = allowed
        self.reason = reason


class DecisionEngine:
    """Evaluates whether to reply using tiered deterministic rules and structured LLM analysis."""

    def __init__(self, provider: Optional[AIProvider] = None):
        self._provider = provider

    @property
    def provider(self) -> AIProvider:
        return self._provider or get_ai_provider()

    def check_cheap_rules(
        self,
        db: Session,
        conversation: Conversation,
        incoming_text: str,
        bot_settings: BotSettings,
        bot_username: Optional[str] = None,
    ) -> RuleCheckResult:
        """Execute deterministic, cheap rule checks without calling the LLM.

        Args:
            db: Database session.
            conversation: The target conversation.
            incoming_text: Plain text of incoming message.
            bot_settings: Current bot settings.
            bot_username: Bot's Instagram handle for mention checks.

        Returns:
            RuleCheckResult: allowed=True if message may proceed to LLM, else False with reason.
        """
        # 1. Global pause
        if bot_settings.is_global_paused:
            return RuleCheckResult(allowed=False, reason="bot_global_paused")

        # 2. Conversation paused
        if conversation.paused:
            return RuleCheckResult(allowed=False, reason="conversation_paused")

        # 3. Human handoff active
        if conversation.human_handoff:
            return RuleCheckResult(allowed=False, reason="human_handoff_active")

        # 4. No consecutive bot replies
        last_msg = (
            db.execute(
                select(Message)
                .where(Message.conversation_id == conversation.id)
                .order_by(desc(Message.created_at))
            )
            .scalars()
            .first()
        )
        if last_msg and last_msg.direction == "out":
            return RuleCheckResult(allowed=False, reason="consecutive_bot_reply")

        # 5. Cooldown check against last outbound bot reply
        last_outbound = (
            db.execute(
                select(Message)
                .where(
                    Message.conversation_id == conversation.id,
                    Message.direction == "out",
                )
                .order_by(desc(Message.created_at))
            )
            .scalars()
            .first()
        )
        if last_outbound:
            now = datetime.now(timezone.utc)
            created_at = last_outbound.created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            elapsed = (now - created_at).total_seconds()
            if elapsed < bot_settings.cooldown_seconds:
                return RuleCheckResult(
                    allowed=False,
                    reason=f"cooldown_active (elapsed {elapsed:.1f}s < {bot_settings.cooldown_seconds}s)",
                )

        # 6. Trigger mode check
        mode = conversation.trigger_mode or "always"
        normalized_text = incoming_text.lower().strip()

        is_mentioned = False
        if bot_username:
            clean_username = bot_username.lower().lstrip("@")
            is_mentioned = f"@{clean_username}" in normalized_text or clean_username in normalized_text

        is_question = normalized_text.endswith("?") or any(
            normalized_text.startswith(w) for w in ["who", "what", "where", "when", "why", "how", "can", "could", "is", "are", "do", "does"]
        )

        if mode == "mention_only":
            if not is_mentioned:
                return RuleCheckResult(allowed=False, reason="mention_only_not_mentioned")
        elif mode == "mention_plus_occasional":
            if not (is_mentioned or is_question):
                return RuleCheckResult(allowed=False, reason="mention_plus_occasional_not_triggered")

        return RuleCheckResult(allowed=True, reason="rules_passed")

    def evaluate(
        self,
        db: Session,
        conversation: Conversation,
        incoming_text: str,
        bot_settings: BotSettings,
        bot_username: Optional[str] = None,
        user_preference: Optional[UserPreference] = None,
    ) -> DecisionOutput:
        """Evaluate if the bot should speak.

        First checks cheap rules. If allowed, invokes structured LLM decision.
        """
        rule_result = self.check_cheap_rules(
            db=db,
            conversation=conversation,
            incoming_text=incoming_text,
            bot_settings=bot_settings,
            bot_username=bot_username,
        )

        if not rule_result.allowed:
            logger.info(
                "Decision engine blocked response via rule: %s (conversation=%s)",
                rule_result.reason,
                conversation.id,
            )
            return DecisionOutput(
                should_respond=False,
                intent="rule_blocked",
                persona=user_preference.preferred_persona if user_preference else "casual",
                needs_live_info=False,
                relevance_score=0.0,
            )

        # Build prompt for structured LLM decision
        preferred_persona = user_preference.preferred_persona if user_preference else "casual"
        system_prompt = (
            "You are a conversation intelligence decision module for an Instagram chatbot.\n"
            "Analyze the user's latest incoming message and decide whether the bot should reply.\n"
            f"User preferred persona: {preferred_persona}.\n"
            "Criteria:\n"
            "- should_respond: true if the message invites or benefits from a helpful reply; false if it's spam, pure gibberish, or inappropriate.\n"
            "- intent: classify the user's intent.\n"
            "- persona: choose 'casual', 'professional', or 'humorous' based on context.\n"
            "- needs_live_info: true if query requires up-to-the-minute external real-time data.\n"
            "- relevance_score: float from 0.0 to 1.0 indicating how relevant/valuable a reply is."
        )

        chat_messages = [{"role": "user", "content": incoming_text}]

        decision = self.provider.generate_structured(
            messages=chat_messages,
            schema=DecisionOutput,
            system_prompt=system_prompt,
            temperature=0.1,
        )

        # Respect relevance threshold
        if decision.relevance_score < bot_settings.relevance_threshold:
            logger.info(
                "Decision engine rejected reply: relevance %.2f < threshold %.2f",
                decision.relevance_score,
                bot_settings.relevance_threshold,
            )
            decision.should_respond = False

        return decision
