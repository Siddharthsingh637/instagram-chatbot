"""Message and summary retention cleanup policy implementation."""

from datetime import datetime, timezone
from typing import Dict
from sqlalchemy import delete
from sqlalchemy.orm import Session
from app.core.logging import logger
from app.database.models import ConversationSummary, Message


def cleanup_expired_data(db: Session) -> Dict[str, int]:
    """Delete all expired messages and expired summaries beyond their retention cutoff.

    Args:
        db: Database session.

    Returns:
        dict: Counts of deleted messages and summaries.
    """
    now = datetime.now(timezone.utc)

    # 1. Delete expired messages
    stmt_msgs = (
        delete(Message)
        .where(Message.expires_at <= now)
        .execution_options(synchronize_session=False)
    )
    res_msgs = db.execute(stmt_msgs)
    deleted_messages = res_msgs.rowcount if res_msgs.rowcount is not None else 0

    # 2. Delete expired summaries
    stmt_summaries = (
        delete(ConversationSummary)
        .where(
            ConversationSummary.expires_at.is_not(None),
            ConversationSummary.expires_at <= now,
        )
        .execution_options(synchronize_session=False)
    )
    res_summaries = db.execute(stmt_summaries)
    deleted_summaries = res_summaries.rowcount if res_summaries.rowcount is not None else 0

    db.commit()

    logger.info(
        "Retention cleanup executed at %s: deleted %d messages, %d summaries",
        now.isoformat(),
        deleted_messages,
        deleted_summaries,
    )

    return {
        "deleted_messages": deleted_messages,
        "deleted_summaries": deleted_summaries,
    }
