"""Instagram Webhook endpoints: Meta challenge verification and incoming event receiver."""

import json
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import PlainTextResponse
from app.core.config import get_settings
from app.core.logging import logger
from app.core.security import verify_meta_signature
from app.workers.message_worker import process_background_webhook_payload

router = APIRouter(prefix="/webhook", tags=["Webhook"])


@router.get("", response_class=PlainTextResponse)
def verify_meta_webhook(
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
) -> str:
    """Handle Meta's webhook verification challenge (GET /webhook).

    Args:
        hub_mode: Verification mode sent by Meta (e.g. 'subscribe').
        hub_verify_token: Token configured in Meta App Dashboard.
        hub_challenge: Challenge token from Meta to echo back.

    Returns:
        str: Raw challenge string if token matches.

    Raises:
        HTTPException: 403 Forbidden if verification token does not match.
    """
    settings = get_settings()

    if not settings.META_VERIFY_TOKEN:
        logger.error("META_VERIFY_TOKEN is not configured on server")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Webhook verification token not configured",
        )

    if hub_mode == "subscribe" and hub_verify_token == settings.META_VERIFY_TOKEN:
        logger.info("Meta webhook verification succeeded")
        return hub_challenge or ""

    logger.warning("Meta webhook verification failed: invalid token received")
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Verification token mismatch",
    )


@router.post("")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: Optional[str] = Header(None, alias="X-Hub-Signature-256"),
) -> dict:
    """Receive and verify incoming Instagram webhooks from Meta (POST /webhook).

    Verifies HMAC-SHA256 signature, immediately returns 200 OK, and schedules
    event processing as an asynchronous background task.
    """
    settings = get_settings()
    raw_body = await request.body()

    # Validate HMAC signature
    is_valid = verify_meta_signature(
        raw_body=raw_body,
        signature_header=x_hub_signature_256,
        secret=settings.META_APP_SECRET,
    )

    if not is_valid:
        logger.warning("Rejected webhook POST: invalid or missing X-Hub-Signature-256")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid request signature",
        )

    # Parse JSON payload
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception as exc:
        logger.error("Failed to decode webhook JSON: %s", str(exc))
        # Meta expects 200 to not retry broken payloads repeatedly
        return {"status": "ignored", "reason": "invalid_json"}

    # Process in background task to respond to Meta well within their 20s timeout
    background_tasks.add_task(process_background_webhook_payload, payload)

    return {"status": "ok"}
