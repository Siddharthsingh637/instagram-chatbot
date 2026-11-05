"""Unit tests for the health check endpoint."""

from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.database.connection import get_db
from app.main import app


def test_health_check_healthy(client: TestClient):
    """Test /health returns 200 and healthy status when DB is connected."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert data["app_env"] == "test"


def test_health_check_db_failure():
    """Test /health returns 503 and unhealthy status when DB connection fails."""
    mock_db = MagicMock()
    mock_db.execute.side_effect = Exception("DB connection timeout")

    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["database"] == "disconnected"
    finally:
        app.dependency_overrides.clear()
