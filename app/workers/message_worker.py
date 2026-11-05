"""Background worker coordinating Pipeline A (UNDERSTAND) and Pipeline B (SPEAK)."""

from typing import Any, Dict, Optional
from sqlalchemy.orm import Session
from app.ai.orchestrator import ConversationOrchestrator
from app.core.config import get_settings
from app.core.logging import logger
from app.database import connection as db_conn
from app.integrations.instagram.events import extract_text_messages

_default_orchestrator: Optional[ConversationOrchestrator] = None


def get_orchestrator() -> ConversationOrchestrator:
    """Return singleton ConversationOrchestrator instance."""
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = ConversationOrchestrator()
    return _default_orchestrator


def set_orchestrator(orchestrator: Optional[ConversationOrchestrator]) -> None:
    """Set custom orchestrator instance (e.g. for testing)."""
    global _default_orchestrator
    _default_orchestrator = orchestrator


def process_incoming_message(
    sender_id: str,
    recipient_id: str,
    message_id: str,
    text: str,
    db: Optional[Session] = None,
) -> Dict[str, Any]:
    """Execute Pipeline A (Store/Understand) and Pipeline B (Speak/Reply).

    Args:
        sender_id: External ID of the message sender.
        recipient_id: External ID of the message recipient (e.g. bot account).
        message_id: External message ID (e.g. Meta mid).
        text: Plain text content of the message.
        db: Optional database session. If None, a new session is managed.

    Returns:
        dict: Processing summary including message ID, status, and generated reply if any.
    """
    should_close_db = False
    if db is None:
        db = db_conn.SessionLocal()
        should_close_db = True

    try:
        settings = get_settings()
        orchestrator = get_orchestrator()

        # Pipeline A: UNDERSTAND (store, deduplicate, rules only, no LLM)
        message = orchestrator.pipeline_a_understand(
            db=db,
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_id=message_id,
            text=text,
        )

        if not message:
            return {"status": "duplicate", "message_id": message_id}

        # Pipeline B: SPEAK (rules + structured decision + context + reply)
        reply = orchestrator.pipeline_b_speak(
            db=db,
            message=message,
            bot_username=settings.INSTAGRAM_ACCOUNT_ID,
        )

        return {
            "status": "stored",
            "message_id": str(message.id),
            "external_id": message_id,
            "conversation_id": str(message.conversation_id),
            "replied": reply is not None,
            "reply": reply,
        }

    except Exception as exc:
        db.rollback()
        logger.error("Error processing incoming message mid=%s: %s", message_id, str(exc))
        raise
    finally:
        if should_close_db:
            db.close()


def process_background_webhook_payload(payload: Dict[str, Any]) -> None:
    """Entry point for FastAPI BackgroundTasks to process webhook payload safely without crashing."""
    settings = get_settings()
    bot_id = settings.INSTAGRAM_ACCOUNT_ID
    messages = extract_text_messages(payload, bot_account_id=bot_id)

    db = db_conn.SessionLocal()
    try:
        for msg in messages:
            try:
                process_incoming_message(
                    sender_id=msg.sender_id,
                    recipient_id=msg.recipient_id,
                    message_id=msg.message_id,
                    text=msg.text,
                    db=db,
                )
            except Exception as exc:
                logger.error("Failed to process message mid=%s: %s", msg.message_id, str(exc))
    finally:
        db.close()
