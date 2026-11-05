"""Tests for InstagramSender and Meta Send API interactions."""

import uuid
import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.database.models import OutboundMessage
from app.integrations.instagram.sender import InstagramSender


def test_sender_success(db: Session):
    """Test successful message send records status=sent and provider_message_id."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert "/me/messages" in str(request.url)
        return httpx.Response(
            status_code=200,
            json={"recipient_id": "user_456", "message_id": "mid_fb_response_123"},
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    sender = InstagramSender(access_token="mock_token", http_client=mock_client)

    outbound = sender.send_text_message(
        recipient_id="user_456",
        text="Hello back!",
        db=db,
    )

    assert outbound.status == "sent"
    assert outbound.provider_message_id == "mid_fb_response_123"
    assert outbound.error is None

    # Check persistence
    persisted = db.execute(select(OutboundMessage).where(OutboundMessage.id == outbound.id)).scalars().first()
    assert persisted is not None
    assert persisted.status == "sent"


def test_sender_retry_on_server_error(db: Session):
    """Test sender retries with backoff on 500 server error and records failure if all fail."""
    attempts = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(status_code=500, text="Internal Server Error")

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    sender = InstagramSender(access_token="mock_token", http_client=mock_client)

    outbound = sender.send_text_message(
        recipient_id="user_456",
        text="Hello retry!",
        db=db,
        max_retries=3,
        initial_backoff=0.01,  # fast backoff for tests
    )

    assert attempts == 3
    assert outbound.status == "failed"
    assert "HTTP 500" in (outbound.error or "")


def test_sender_no_retry_on_client_error(db: Session):
    """Test sender does not retry on 4xx client errors (e.g. invalid recipient)."""
    attempts = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(status_code=400, text="Bad Request: Invalid OAuth access token")

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    sender = InstagramSender(access_token="mock_token", http_client=mock_client)

    outbound = sender.send_text_message(
        recipient_id="user_456",
        text="Hello no retry!",
        db=db,
        max_retries=3,
        initial_backoff=0.01,
    )

    assert attempts == 1  # No repeated attempts on 400
    assert outbound.status == "failed"
    assert "HTTP 400" in (outbound.error or "")


def test_sender_missing_token(db: Session):
    """Test sender gracefully records failure when access token is missing."""
    sender = InstagramSender(access_token="")
    outbound = sender.send_text_message(
        recipient_id="user_456",
        text="No token test",
        db=db,
    )
    assert outbound.status == "failed"
    assert "not configured" in (outbound.error or "")
