"""Pydantic schemas and parsers for Meta Instagram Webhook events."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Identity(BaseModel):
    """Sender or recipient identity object in Meta events."""
    id: str


class IncomingMessageData(BaseModel):
    """Message payload within a messaging event."""
    mid: str
    text: Optional[str] = None
    is_echo: Optional[bool] = False


class MessagingEvent(BaseModel):
    """Instagram messaging event wrapper."""
    sender: Identity
    recipient: Identity
    timestamp: Optional[int] = None
    message: Optional[IncomingMessageData] = None


class EntryItem(BaseModel):
    """Entry element within Meta webhook payload."""
    id: str
    time: Optional[int] = None
    messaging: Optional[List[MessagingEvent]] = Field(default_factory=list)


class WebhookPayload(BaseModel):
    """Top-level Instagram webhook payload received from Meta."""
    object: str
    entry: List[EntryItem] = Field(default_factory=list)


class ParsedMessage:
    """Normalized internal representation of an incoming Instagram DM."""

    def __init__(self, sender_id: str, recipient_id: str, message_id: str, text: str):
        self.sender_id = sender_id
        self.recipient_id = recipient_id
        self.message_id = message_id
        self.text = text


def extract_text_messages(payload: Dict[str, Any], bot_account_id: Optional[str] = None) -> List[ParsedMessage]:
    """Parse raw webhook JSON, filter out echoes, non-text, and bot-sent messages.

    Args:
        payload: Decoded JSON dictionary from Meta.
        bot_account_id: Optional Instagram Account ID of the bot to ignore self-sent messages.

    Returns:
        List[ParsedMessage]: Valid incoming text messages to process.
    """
    valid_messages: List[ParsedMessage] = []

    try:
        data = WebhookPayload.model_validate(payload)
    except Exception:
        # Ignore malformed or non-compliant webhook structures
        return valid_messages

    if data.object != "instagram":
        return valid_messages

    for entry in data.entry:
        if not entry.messaging:
            continue
        for event in entry.messaging:
            msg = event.message
            if not msg:
                # Event is not a message (e.g. read receipt, postback, delivery)
                continue

            # Ignore echoes (messages sent by the page/account itself)
            if msg.is_echo:
                continue

            # Ignore if sender matches the bot's configured account ID
            if bot_account_id and event.sender.id == bot_account_id:
                continue

            # Ignore empty or whitespace-only texts
            if not msg.text or not msg.text.strip():
                continue

            valid_messages.append(
                ParsedMessage(
                    sender_id=event.sender.id,
                    recipient_id=event.recipient.id,
                    message_id=msg.mid,
                    text=msg.text.strip(),
                )
            )

    return valid_messages
