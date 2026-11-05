"""Tests for Stage 6 Admin API endpoints, authentication, and moderation flows."""

import uuid
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.core.config import get_settings
from app.database.models import BotSettings, Conversation, Memory, Message, User


def test_admin_authentication_enforcement(client: TestClient):
    """Test that all admin endpoints reject missing or invalid X-Admin-Key headers."""
    # 1. Missing header -> 401
    resp_missing = client.get("/admin/settings")
    assert resp_missing.status_code == 401

    # 2. Invalid header -> 401
    resp_invalid = client.get("/admin/settings", headers={"X-Admin-Key": "wrong-secret"})
    assert resp_invalid.status_code == 401

    # 3. Valid header -> 200
    settings = get_settings()
    resp_valid = client.get("/admin/settings", headers={"X-Admin-Key": settings.ADMIN_API_KEY})
    assert resp_valid.status_code == 200


def test_admin_conversations_and_messages(client: TestClient, db: Session):
    """Test listing conversations and retrieving messages for a conversation."""
    settings = get_settings()
    headers = {"X-Admin-Key": settings.ADMIN_API_KEY}

    user = User(external_id="admin_user_1", username="admin_tester")
    db.add(user)
    db.commit()

    conv = Conversation(external_id="admin_conv_1", user_id=user.id, human_handoff=True)
    db.add(conv)
    db.commit()

    now = datetime.now(timezone.utc)
    msg = Message(
        conversation_id=conv.id,
        sender_id=user.external_id,
        direction="in",
        text="Hello admin!",
        expires_at=now,
    )
    db.add(msg)
    db.commit()

    # 1. GET /admin/conversations
    resp_convs = client.get("/admin/conversations", headers=headers)
    assert resp_convs.status_code == 200
    conv_list = resp_convs.json()
    assert len(conv_list) >= 1
    assert any(c["id"] == str(conv.id) for c in conv_list)

    # 2. GET /admin/conversations/{id}/messages
    resp_msgs = client.get(f"/admin/conversations/{conv.id}/messages", headers=headers)
    assert resp_msgs.status_code == 200
    msg_list = resp_msgs.json()
    assert len(msg_list) >= 1
    assert msg_list[0]["text"] == "Hello admin!"

    # 3. POST /admin/conversations/{id}/handoff/resume
    resp_resume = client.post(f"/admin/conversations/{conv.id}/handoff/resume", headers=headers)
    assert resp_resume.status_code == 200
    assert resp_resume.json()["human_handoff"] is False

    db.refresh(conv)
    assert conv.human_handoff is False


def test_admin_memory_moderation_lifecycle(client: TestClient, db: Session):
    """Test full memory lifecycle: pending list, approve, reject, edit, delete."""
    settings = get_settings()
    headers = {"X-Admin-Key": settings.ADMIN_API_KEY}

    user = User(external_id="mem_mod_user")
    db.add(user)
    db.commit()

    mem = Memory(
        user_id=user.id,
        category="interests",
        content="Interested in machine learning",
        status="pending",
    )
    db.add(mem)
    db.commit()

    # 1. GET /admin/memories/pending
    resp_pending = client.get("/admin/memories/pending", headers=headers)
    assert resp_pending.status_code == 200
    pending_list = resp_pending.json()
    assert any(m["id"] == str(mem.id) for m in pending_list)

    # 2. GET /admin/memories/{user_id}
    resp_user_mem = client.get(f"/admin/memories/{user.id}", headers=headers)
    assert resp_user_mem.status_code == 200
    assert len(resp_user_mem.json()) >= 1

    # 3. POST /admin/memories/{id}/approve
    resp_app = client.post(f"/admin/memories/{mem.id}/approve", headers=headers)
    assert resp_app.status_code == 200
    assert resp_app.json()["memory_status"] == "approved"
    db.refresh(mem)
    assert mem.status == "approved"

    # 4. POST /admin/memories/{id}/reject
    resp_rej = client.post(f"/admin/memories/{mem.id}/reject", headers=headers)
    assert resp_rej.status_code == 200
    assert resp_rej.json()["memory_status"] == "rejected"
    db.refresh(mem)
    assert mem.status == "rejected"

    # 5. PATCH /admin/memories/{id} (edit)
    resp_edit = client.patch(
        f"/admin/memories/{mem.id}",
        json={"content": "Loves artificial intelligence", "status": "approved"},
        headers=headers,
    )
    assert resp_edit.status_code == 200
    assert resp_edit.json()["content"] == "Loves artificial intelligence"
    assert resp_edit.json()["memory_status"] == "approved"
    db.refresh(mem)
    assert mem.content == "Loves artificial intelligence"

    # 6. DELETE /admin/memories/{id}
    resp_del = client.delete(f"/admin/memories/{mem.id}", headers=headers)
    assert resp_del.status_code == 200
    assert resp_del.json()["status"] == "success"
    db.expire_all()
    assert db.query(Memory).filter(Memory.id == mem.id).first() is None


def test_admin_settings_controls(client: TestClient, db: Session):
    """Test viewing, updating, pausing, and resuming global bot settings."""
    settings = get_settings()
    headers = {"X-Admin-Key": settings.ADMIN_API_KEY}

    # 1. GET /admin/settings
    resp_get = client.get("/admin/settings", headers=headers)
    assert resp_get.status_code == 200
    data = resp_get.json()
    assert "is_global_paused" in data

    # 2. PATCH /admin/settings
    resp_patch = client.patch(
        "/admin/settings",
        json={"cooldown_seconds": 25, "relevance_threshold": 0.8},
        headers=headers,
    )
    assert resp_patch.status_code == 200
    assert resp_patch.json()["cooldown_seconds"] == 25
    assert resp_patch.json()["relevance_threshold"] == 0.8

    # 3. POST /admin/settings/pause
    resp_pause = client.post("/admin/settings/pause", headers=headers)
    assert resp_pause.status_code == 200
    assert resp_pause.json()["is_global_paused"] is True

    # 4. POST /admin/settings/resume
    resp_resume = client.post("/admin/settings/resume", headers=headers)
    assert resp_resume.status_code == 200
    assert resp_resume.json()["is_global_paused"] is False
