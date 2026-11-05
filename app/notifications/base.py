"""Owner notification interface for human handoff alerts."""

from abc import ABC, abstractmethod
import uuid
from typing import Optional


class OwnerNotifier(ABC):
    """Abstract interface for notifying bot administrators or account owners."""

    @abstractmethod
    def notify_human_handoff(
        self,
        conversation_id: uuid.UUID,
        user_id: Optional[uuid.UUID],
        reason: str,
        user_message: str,
    ) -> bool:
        """Alert the owner that a human handoff event was triggered.

        Args:
            conversation_id: Internal conversation UUID.
            user_id: Internal user UUID if known.
            reason: Trigger reason (e.g. keyword match or explicit request).
            user_message: The message that triggered the handoff.

        Returns:
            bool: True if notification dispatch succeeded.
        """
        pass
