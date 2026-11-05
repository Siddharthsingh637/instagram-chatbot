"""Pytest fixtures for testing Instagram Chatbot backend."""

import os
from typing import Generator
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Force test environment before loading settings
os.environ["APP_ENV"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["ADMIN_API_KEY"] = "test-admin-secret"
os.environ["META_APP_SECRET"] = "test-meta-secret"
os.environ["META_VERIFY_TOKEN"] = "test-verify-token"

from app.core.config import get_settings
import app.database.connection as conn_mod
from app.database.connection import Base, get_db
from app.main import app

# Create in-memory SQLite engine for tests with StaticPool so all connections share the DB
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Point connection module to test engine & session factory
conn_mod.engine = test_engine
conn_mod.SessionLocal = TestingSessionLocal


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create all tables in the test database once per session."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(autouse=True)
def clean_db():
    """Wipe database contents between tests to ensure complete test isolation."""
    yield
    with test_engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
        conn.commit()


@pytest.fixture(autouse=True)
def setup_fake_ai():
    """Automatically provide FakeProvider to all tests to guarantee no real API calls."""
    from app.ai.decision_engine import DecisionOutput
    from app.ai.provider_factory import set_ai_provider_override
    from tests.test_ai_providers import FakeProvider

    fake = FakeProvider(
        text_response="This is a test response from FakeProvider.",
        structured_response=DecisionOutput(
            should_respond=True,
            intent="greeting",
            persona="casual",
            needs_live_info=False,
            relevance_score=0.9,
        ),
    )
    set_ai_provider_override(fake)
    yield fake
    set_ai_provider_override(None)


@pytest.fixture
def db() -> Generator[Session, None, None]:
    """Provide a database session for each test."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db: Session) -> Generator[TestClient, None, None]:
    """Provide a TestClient with the database dependency overridden."""
    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
