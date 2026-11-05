"""Memory Manager: Candidate memory extraction, sensitive data filtering, and lifecycle management."""

import re
import uuid
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.ai.provider_factory import get_ai_provider
from app.ai.providers.base import AIProvider
from app.core.config import get_settings
from app.core.logging import logger
from app.database.models import Conversation, Memory, Message, User

# Regex patterns detecting sensitive categories: financial, credentials, health, PII, addresses
SENSITIVE_PATTERNS = [
    # Passwords & Secrets
    re.compile(r"\b(password|passwd|pwd|pin\s*code|secret\s*key|api[_\s]*key|private\s*key)\b", re.IGNORECASE),
    # Government IDs & SSN
    re.compile(r"\b(ssn|social\s*security|passport\s*(?:no|number)?|driver'?s?\s*license|aadhaar|tax\s*id)\b", re.IGNORECASE),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # US SSN format
    # Financial data (Credit cards, bank accounts, CVV)
    re.compile(r"\b(credit\s*card|debit\s*card|cvv|cvc|bank\s*account|routing\s*number|salary|net\s*worth|crypto\s*wallet)\b", re.IGNORECASE),
    re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"),  # Visa/MC/Amex card format
    # Medical & Health conditions
    re.compile(r"\b(diagnosis|prescription|medication|illness|disease|bipolar|schizophrenia|hiv|cancer|medical\s*condition|depressed|suicid)\b", re.IGNORECASE),
    # Precise addresses (e.g. Street, Suite, Apt with numbers, ZIP)
    re.compile(r"\b\d{1,5}\s+[a-zA-Z0-9\.\s]+(?:street|st|avenue|ave|road|rd|blvd|lane|ln|drive|dr|apt|suite|ste)\b", re.IGNORECASE),
    re.compile(r"\b\d{5}(?:-\d{4})?\b"),  # US ZIP code format
]

DISALLOWED_CATEGORIES = {
    "health",
    "medical",
    "financial",
    "finance",
    "credentials",
    "password",
    "government_id",
    "address",
    "location_precise",
    "third_party",
}


def is_sensitive_data(content: str, category: str = "") -> Tuple[bool, str]:
    """Check if memory candidate contains disallowed sensitive or private personal information.

    Args:
        content: Memory content string.
        category: Memory category identifier.

    Returns:
        Tuple[bool, str]: (is_sensitive, reason_string)
    """
    cat_lower = category.lower().strip()
    if cat_lower in DISALLOWED_CATEGORIES:
        return True, f"Category '{category}' is in disallowed sensitive categories list"

    for pattern in SENSITIVE_PATTERNS:
        match = pattern.search(content)
        if match:
            return True, f"Matched sensitive pattern: '{match.group(0)}'"

    return False, ""


class CandidateMemory(BaseModel):
    """An individual candidate fact extracted from user messages."""
    category: str = Field(..., description="Category, e.g. preference, hobby, goal, pet, profession, name")
    content: str = Field(..., description="Fact stated by the user about themselves")


class MemoryExtractionResponse(BaseModel):
    """Structured LLM extraction response containing candidate user memories."""
    candidates: List[CandidateMemory] = Field(default_factory=list)


class MemoryManager:
    """Manages memory creation, candidate extraction batches, and policy enforcement."""

    def __init__(self, provider: Optional[AIProvider] = None):
        self._provider = provider

    @property
    def provider(self) -> AIProvider:
        return self._provider or get_ai_provider()

    def save_candidate_memory(
        self,
        db: Session,
        user_id: uuid.UUID,
        category: str,
        content: str,
        source_message_id: Optional[uuid.UUID] = None,
    ) -> Optional[Memory]:
        """Save a new memory candidate after running the sensitive data filter.

        Sets status='approved' if category is in AUTO_APPROVED_MEMORY_CATEGORIES, otherwise 'pending'.
        """
        is_sensitive, reason = is_sensitive_data(content, category)
        if is_sensitive:
            logger.warning(
                "Rejected sensitive candidate memory for user %s [category=%s]: %s",
                user_id,
                category,
                reason,
            )
            return None

        settings = get_settings()
        normalized_cat = category.strip().lower()
        auto_approved = normalized_cat in settings.AUTO_APPROVED_MEMORY_CATEGORIES

        memory = Memory(
            id=uuid.uuid4(),
            user_id=user_id,
            category=normalized_cat,
            content=content.strip(),
            status="approved" if auto_approved else "pending",
            source_message_id=source_message_id,
        )
        db.add(memory)
        db.commit()
        db.refresh(memory)

        logger.info(
            "Saved memory id=%s for user=%s [category=%s, status=%s]",
            memory.id,
            user_id,
            memory.category,
            memory.status,
        )
        return memory

    def extract_candidate_memories_batch(self, db: Session, batch_size: int = 10) -> int:
        """Asynchronously extract candidate memories from unprocessed stored user messages in batches.

        Args:
            db: Database session.
            batch_size: Number of messages to process in one batch.

        Returns:
            int: Number of candidate memories saved.
        """
        # Fetch unprocessed incoming messages
        messages = (
            db.execute(
                select(Message)
                .where(
                    Message.direction == "in",
                    Message.processing_status == "stored",
                )
                .limit(batch_size)
            )
            .scalars()
            .all()
        )

        if not messages:
            return 0

        saved_count = 0
        system_prompt = (
            "You are a personal memory extraction engine. Analyze user messages and extract clear, "
            "explicit facts the user states ABOUT THEMSELVES (e.g. name, preferences, hobbies, goals, pets).\n"
            "CRITICAL RULES:\n"
            "- Extract ONLY what the user clearly states about themselves.\n"
            "- NEVER extract health/medical details, passwords, credit cards, bank accounts, or government IDs.\n"
            "- NEVER extract third-party rumors or mentions of other people.\n"
            "- If no self-disclosed personal facts are present, return an empty candidates list."
        )

        for msg in messages:
            # Look up user_id from conversation
            conv = db.execute(
                select(Conversation).where(Conversation.id == msg.conversation_id)
            ).scalars().first()
            if not conv or not conv.user_id:
                msg.processing_status = "extracted"
                continue

            try:
                extraction = self.provider.generate_structured(
                    messages=[{"role": "user", "content": msg.text}],
                    schema=MemoryExtractionResponse,
                    system_prompt=system_prompt,
                    temperature=0.1,
                )

                for candidate in extraction.candidates:
                    saved = self.save_candidate_memory(
                        db=db,
                        user_id=conv.user_id,
                        category=candidate.category,
                        content=candidate.content,
                        source_message_id=msg.id,
                    )
                    if saved:
                        saved_count += 1

                msg.processing_status = "extracted"
            except Exception as exc:
                logger.error("Error extracting candidate memories for message %s: %s", msg.id, str(exc))
                # Mark as extracted to avoid infinite error loop on malformed input
                msg.processing_status = "extracted"

        db.commit()
        return saved_count
