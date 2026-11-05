"""Instagram Send API client with retries, exponential backoff, and outbound logging."""

import time
import uuid
from datetime import datetime, timezone
from typing import Optional
import httpx
from sqlalchemy.orm import Session
from app.core.config import get_settings
from app.core.logging import logger
from app.database.models import OutboundMessage


class InstagramSender:
    """Sends text messages to Instagram users via Meta's Graph API."""

    def __init__(
        self,
        access_token: Optional[str] = None,
        graph_api_version: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
    ):
        settings = get_settings()
        self.access_token = access_token or settings.INSTAGRAM_ACCESS_TOKEN
        self.graph_api_version = graph_api_version or settings.GRAPH_API_VERSION
        self.http_client = http_client or httpx.Client(timeout=10.0)
        self.endpoint = f"https://graph.facebook.com/{self.graph_api_version}/me/messages"

    def send_text_message(
        self,
        recipient_id: str,
        text: str,
        db: Session,
        conversation_id: Optional[uuid.UUID] = None,
        max_retries: int = 3,
        initial_backoff: float = 0.5,
    ) -> OutboundMessage:
        """Send a plain text message to an Instagram recipient with retries and logging.

        Args:
            recipient_id: Instagram-scoped user ID (IGSID).
            text: Message body text.
            db: Database session to persist OutboundMessage.
            conversation_id: Optional internal conversation UUID.
            max_retries: Number of retry attempts on network/server errors.
            initial_backoff: Initial sleep duration in seconds for exponential backoff.

        Returns:
            OutboundMessage: The persisted record reflecting success or failure.
        """
        # Create initial outbound record
        outbound = OutboundMessage(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            recipient_id=recipient_id,
            text=text,
            status="pending",
        )
        db.add(outbound)
        db.commit()
        db.refresh(outbound)

        if not self.access_token:
            logger.warning("No INSTAGRAM_ACCESS_TOKEN configured; recording outbound as failed")
            outbound.status = "failed"
            outbound.error = "INSTAGRAM_ACCESS_TOKEN is not configured"
            db.commit()
            return outbound

        payload = {
            "recipient": {"id": recipient_id},
            "message": {"text": text},
        }
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

        last_error = ""
        backoff = initial_backoff

        for attempt in range(1, max_retries + 1):
            try:
                response = self.http_client.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                )

                if response.status_code == 200:
                    data = response.json()
                    provider_msg_id = data.get("message_id")
                    outbound.status = "sent"
                    outbound.provider_message_id = provider_msg_id
                    outbound.error = None
                    db.commit()
                    logger.info(
                        "Sent Instagram message to %s (attempt %d, mid: %s)",
                        recipient_id,
                        attempt,
                        provider_msg_id,
                    )
                    return outbound

                error_detail = response.text
                last_error = f"HTTP {response.status_code}: {error_detail}"
                logger.warning(
                    "Instagram Send API error (attempt %d/%d): %s",
                    attempt,
                    max_retries,
                    last_error,
                )

                # 4xx client errors (e.g. invalid recipient, permissions) should not retry
                if 400 <= response.status_code < 500:
                    break

            except Exception as exc:
                last_error = f"Network exception: {str(exc)}"
                logger.warning(
                    "Instagram Send network failure (attempt %d/%d): %s",
                    attempt,
                    max_retries,
                    last_error,
                )

            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2

        outbound.status = "failed"
        outbound.error = last_error
        db.commit()
        return outbound
