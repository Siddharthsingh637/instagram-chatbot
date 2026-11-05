"""Development test endpoints for simulating incoming messages without Meta."""

import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.core.config import get_settings
from app.core.security import verify_admin_key
from app.database.connection import get_db
from app.workers.message_worker import process_incoming_message

router = APIRouter(prefix="/dev", tags=["Development"])


class SimulateMessageRequest(BaseModel):
    """Payload to simulate an incoming Instagram direct message."""
    sender_id: str = Field(..., description="Simulated user Instagram ID (IGSID)")
    text: str = Field(..., description="Message text content")
    recipient_id: Optional[str] = Field("bot_simulated_account", description="Bot Instagram ID")
    message_id: Optional[str] = Field(None, description="Custom message MID for dedup testing")


@router.post("/simulate-message")
def simulate_message(
    body: SimulateMessageRequest,
    db: Session = Depends(get_db),
    admin_key: str = Depends(verify_admin_key),
) -> dict:
    """Simulate an incoming Instagram DM for testing without Meta infrastructure.

    Only active when APP_ENV=development.
    Protected by ADMIN_API_KEY (X-Admin-Key header).
    """
    settings = get_settings()
    if settings.APP_ENV != "development":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Dev simulation endpoint is only available when APP_ENV=development",
        )

    mid = body.message_id or f"sim_mid_{uuid.uuid4().hex[:12]}"
    result = process_incoming_message(
        sender_id=body.sender_id,
        recipient_id=body.recipient_id or "bot_simulated_account",
        message_id=mid,
        text=body.text,
        db=db,
    )

    return result
