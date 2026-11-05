"""Tests for Instagram Webhook GET/POST and simulate-message dev endpoint."""

import hashlib
import hmac
import json
import uuid
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.core.config import get_settings
from app.database.models import Conversation, Message, ProcessedEvent, User


def create_meta_signature(body: bytes, secret: str) -> str:
    """Helper to generate X-Hub-Signature-256 header."""
    sig = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={sig}"


def test_meta_webhook_verification_success(client: TestClient):
    """GET /webhook returns challenge when hub.verify_token matches."""
    settings = get_settings()
    response = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": settings.META_VERIFY_TOKEN,
            "hub.challenge": "1158201444",
        },
    )
    assert response.status_code == 200
    assert response.text == "1158201444"


def test_meta_webhook_verification_failure(client: TestClient):
    """GET /webhook returns 403 when token does not match."""
    response = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong_token",
            "hub.challenge": "1158201444",
        },
    )
    assert response.status_code == 403


def test_webhook_post_signature_validation(client: TestClient):
    """POST /webhook validates X-Hub-Signature-256 header."""
    settings = get_settings()
    payload = {
        "object": "instagram",
        "entry": [
            {
                "id": "page_id_1",
                "messaging": [
                    {
                        "sender": {"id": "user_1"},
                        "recipient": {"id": "page_id_1"},
                        "message": {"mid": "mid_test_sig", "text": "Hello verification!"},
                    }
                ],
            }
        ],
    }
    raw_body = json.dumps(payload).encode("utf-8")

    # 1. Invalid signature
    resp_bad = client.post(
        "/webhook",
        content=raw_body,
        headers={"X-Hub-Signature-256": "sha256=invalidhex123"},
    )
    assert resp_bad.status_code == 403

    # 2. Missing signature
    resp_missing = client.post("/webhook", content=raw_body)
    assert resp_missing.status_code == 403

    # 3. Valid signature
    valid_sig = create_meta_signature(raw_body, settings.META_APP_SECRET)
    resp_ok = client.post(
        "/webhook",
        content=raw_body,
        headers={"X-Hub-Signature-256": valid_sig},
    )
    assert resp_ok.status_code == 200
    assert resp_ok.json() == {"status": "ok"}


def test_webhook_deduplication_and_storage(client: TestClient, db: Session):
    """Test webhook deduplication, user/conversation creation, and message storage."""
    settings = get_settings()
    mid = f"mid_dedup_{uuid.uuid4().hex[:8]}"
    sender_id = "ig_user_test_dedup"
    recipient_id = "ig_page_test_dedup"

    payload = {
        "object": "instagram",
        "entry": [
            {
                "id": recipient_id,
                "messaging": [
                    {
                        "sender": {"id": sender_id},
                        "recipient": {"id": recipient_id},
                        "message": {"mid": mid, "text": "Testing dedup and retention"},
                    }
                ],
            }
        ],
    }
    raw_body = json.dumps(payload).encode("utf-8")
    sig = create_meta_signature(raw_body, settings.META_APP_SECRET)

    # First delivery
    resp1 = client.post("/webhook", content=raw_body, headers={"X-Hub-Signature-256": sig})
    assert resp1.status_code == 200

    # Verify stored in DB
    user = db.execute(select(User).where(User.external_id == sender_id)).scalars().first()
    assert user is not None

    conv = db.execute(select(Conversation).where(Conversation.user_id == user.id)).scalars().first()
    assert conv is not None

    msgs = db.execute(select(Message).where(Message.external_id == mid)).scalars().all()
    assert len(msgs) == 1
    assert msgs[0].text == "Testing dedup and retention"
    assert msgs[0].direction == "in"
    assert msgs[0].processing_status == "stored"
    expires = msgs[0].expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    assert expires > datetime.now(timezone.utc)

    # Second duplicate delivery with same MID
    resp2 = client.post("/webhook", content=raw_body, headers={"X-Hub-Signature-256": sig})
    assert resp2.status_code == 200

    # Ensure message was not duplicated
    msgs_after = db.execute(select(Message).where(Message.external_id == mid)).scalars().all()
    assert len(msgs_after) == 1


def test_webhook_ignores_echoes_and_empty(client: TestClient, db: Session):
    """Test that echo messages and messages without text are ignored."""
    settings = get_settings()
    echo_mid = f"mid_echo_{uuid.uuid4().hex[:8]}"

    payload = {
        "object": "instagram",
        "entry": [
            {
                "id": "page_id_1",
                "messaging": [
                    {
                        "sender": {"id": "page_id_1"},
                        "recipient": {"id": "user_id_1"},
                        "message": {"mid": echo_mid, "text": "This is an echo", "is_echo": True},
                    },
                    {
                        "sender": {"id": "user_id_2"},
                        "recipient": {"id": "page_id_1"},
                        "message": {"mid": "mid_empty", "text": "   "},
                    },
                ],
            }
        ],
    }
    raw_body = json.dumps(payload).encode("utf-8")
    sig = create_meta_signature(raw_body, settings.META_APP_SECRET)

    resp = client.post("/webhook", content=raw_body, headers={"X-Hub-Signature-256": sig})
    assert resp.status_code == 200

    # Verify neither echo nor empty was stored
    echo_msg = db.execute(select(Message).where(Message.external_id == echo_mid)).scalars().first()
    assert echo_msg is None


def test_dev_simulate_message(client: TestClient, db: Session, monkeypatch):
    """Test POST /dev/simulate-message in development environment."""
    settings = get_settings()

    # 1. Blocked in test environment
    resp_blocked = client.post(
        "/dev/simulate-message",
        json={"sender_id": "sim_user_1", "text": "Hello"},
        headers={"X-Admin-Key": settings.ADMIN_API_KEY},
    )
    assert resp_blocked.status_code == 403

    # 2. Switch APP_ENV to development
    monkeypatch.setattr(settings, "APP_ENV", "development")

    # Missing/wrong admin key
    resp_unauth = client.post(
        "/dev/simulate-message",
        json={"sender_id": "sim_user_1", "text": "Hello"},
        headers={"X-Admin-Key": "wrong-key"},
    )
    assert resp_unauth.status_code == 401

    # Valid simulation
    sim_mid = f"sim_{uuid.uuid4().hex[:8]}"
    resp_ok = client.post(
        "/dev/simulate-message",
        json={"sender_id": "sim_user_1", "text": "Hello simulation", "message_id": sim_mid},
        headers={"X-Admin-Key": settings.ADMIN_API_KEY},
    )
    assert resp_ok.status_code == 200
    data = resp_ok.json()
    assert data["status"] == "stored"
    assert data["external_id"] == sim_mid

    # Verify stored in DB
    msg = db.execute(select(Message).where(Message.external_id == sim_mid)).scalars().first()
    assert msg is not None
    assert msg.text == "Hello simulation"
