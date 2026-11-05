"""Default logging implementation of OwnerNotifier."""

import uuid
from typing import Optional
from app.core.logging import logger
from app.notifications.base import OwnerNotifier


class LoggingOwnerNotifier(OwnerNotifier):
    """Logs human handoff events to structured application logs."""

    def notify_human_handoff(
        self,
        conversation_id: uuid.UUID,
        user_id: Optional[uuid.UUID],
        reason: str,
        user_message: str,
    ) -> bool:
        logger.warning(
            "🚨 [HUMAN HANDOFF ALERT] Conversation %s (User %s) requested a human operator! "
            "Reason: '%s' | Triggering message: %r",
            conversation_id,
            user_id,
            reason,
            user_message,
        )
        return True


_notifier_instance: OwnerNotifier = LoggingOwnerNotifier()


def get_owner_notifier() -> OwnerNotifier:
    """Return the configured OwnerNotifier singleton."""
    return _notifier_instance


def set_owner_notifier(notifier: OwnerNotifier) -> None:
    """Set custom notifier (e.g. for testing or plugging in Telegram/Email)."""
    global _notifier_instance
    _notifier_instance = notifier
