"""Unit tests for configuration loading and validation."""

import pytest
from app.core.config import Settings


def test_settings_default_values():
    """Verify default configuration values."""
    settings = Settings(
        APP_ENV="development",
        DATABASE_URL="sqlite:///./test.db",
    )
    assert settings.APP_ENV == "development"
    assert settings.AI_PROVIDER == "groq"
    assert settings.MESSAGE_RETENTION_HOURS == 6
    assert settings.REPLY_COOLDOWN_SECONDS == 10
    assert settings.MAX_CONTEXT_MESSAGES == 15
    assert "human" in settings.HANDOFF_KEYWORDS
    assert "agent" in settings.HANDOFF_KEYWORDS


def test_handoff_keywords_parsing():
    """Verify parsing of comma-separated string for handoff keywords."""
    settings = Settings(
        APP_ENV="development",
        HANDOFF_KEYWORDS="representative, operator , live support",
    )
    assert settings.HANDOFF_KEYWORDS == ["representative", "operator", "live support"]


def test_production_validation_missing_credentials(monkeypatch):
    """Verify production mode raises ValueError when required credentials are missing."""
    # Ensure environment variables are clear for this test
    monkeypatch.delenv("META_APP_SECRET", raising=False)
    monkeypatch.delenv("META_VERIFY_TOKEN", raising=False)
    monkeypatch.delenv("INSTAGRAM_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError) as exc_info:
        Settings(
            APP_ENV="production",
            DATABASE_URL="sqlite:///./test.db",  # Invalid in production
            META_APP_SECRET=None,
            META_VERIFY_TOKEN=None,
            INSTAGRAM_ACCESS_TOKEN=None,
            ADMIN_API_KEY=None,
            GROQ_API_KEY=None,
        )

    error_message = str(exc_info.value)
    assert "Production startup failed" in error_message
    assert "DATABASE_URL" in error_message
    assert "META_APP_SECRET" in error_message
    assert "META_VERIFY_TOKEN" in error_message
    assert "INSTAGRAM_ACCESS_TOKEN" in error_message
    assert "ADMIN_API_KEY" in error_message
    assert "GROQ_API_KEY" in error_message


def test_production_validation_success():
    """Verify production mode succeeds when all required credentials are provided."""
    settings = Settings(
        APP_ENV="production",
        DATABASE_URL="postgresql+psycopg2://user:pass@localhost:5432/bot",
        META_APP_SECRET="prod-secret",
        META_VERIFY_TOKEN="prod-token",
        INSTAGRAM_ACCESS_TOKEN="prod-ig-token",
        ADMIN_API_KEY="prod-admin-key",
        AI_PROVIDER="groq",
        GROQ_API_KEY="prod-groq-key",
    )
    assert settings.APP_ENV == "production"
    assert settings.AI_PROVIDER == "groq"
