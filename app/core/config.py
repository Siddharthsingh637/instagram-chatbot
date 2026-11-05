"""Application configuration loaded from environment variables using Pydantic Settings."""

from functools import lru_cache
from typing import List, Literal, Optional
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with strong typing and environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Runtime Environment
    APP_ENV: Literal["development", "production", "test"] = "development"

    # Database
    DATABASE_URL: str = Field(
        default="sqlite:///./instagram_bot.db",
        description="Database connection URL. SQLite for local/test, PostgreSQL for Supabase/production.",
    )

    # AI Configuration
    AI_PROVIDER: Literal["groq", "openrouter", "gemini"] = "groq"
    GROQ_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    # Meta / Instagram Messaging API
    META_APP_SECRET: Optional[str] = None
    META_VERIFY_TOKEN: Optional[str] = None
    INSTAGRAM_ACCESS_TOKEN: Optional[str] = None
    INSTAGRAM_ACCOUNT_ID: Optional[str] = None
    GRAPH_API_VERSION: str = "v21.0"

    # Admin Security
    ADMIN_API_KEY: Optional[str] = None

    # Bot Policy & Tuning
    MESSAGE_RETENTION_HOURS: int = 6
    REPLY_COOLDOWN_SECONDS: int = 10
    MAX_CONTEXT_MESSAGES: int = 15
    HANDOFF_KEYWORDS: List[str] = Field(
        default_factory=lambda: ["human", "agent", "real person"],
        description="List of keywords triggering human handoff.",
    )
    AUTO_APPROVED_MEMORY_CATEGORIES: List[str] = Field(
        default_factory=list,
        description="Optional list of memory categories to automatically approve.",
    )

    @field_validator("HANDOFF_KEYWORDS", "AUTO_APPROVED_MEMORY_CATEGORIES", mode="before")
    @classmethod
    def parse_list_strings(cls, v: object) -> List[str]:
        """Convert comma-separated string to list if necessary."""
        if isinstance(v, str):
            return [item.strip().lower() for item in v.split(",") if item.strip()]
        if isinstance(v, list):
            return [str(item).strip().lower() for item in v if str(item).strip()]
        return []

    @model_validator(mode="after")
    def validate_production_requirements(self) -> "Settings":
        """Ensure all required production credentials and settings are present."""
        if self.APP_ENV == "production":
            missing = []
            if not self.DATABASE_URL or self.DATABASE_URL.startswith("sqlite"):
                missing.append("DATABASE_URL (must be a PostgreSQL connection URL in production)")
            if not self.META_APP_SECRET:
                missing.append("META_APP_SECRET")
            if not self.META_VERIFY_TOKEN:
                missing.append("META_VERIFY_TOKEN")
            if not self.INSTAGRAM_ACCESS_TOKEN:
                missing.append("INSTAGRAM_ACCESS_TOKEN")
            if not self.ADMIN_API_KEY:
                missing.append("ADMIN_API_KEY")

            if self.AI_PROVIDER == "groq" and not self.GROQ_API_KEY:
                missing.append("GROQ_API_KEY (required when AI_PROVIDER=groq)")
            elif self.AI_PROVIDER == "openrouter" and not self.OPENROUTER_API_KEY:
                missing.append("OPENROUTER_API_KEY (required when AI_PROVIDER=openrouter)")
            elif self.AI_PROVIDER == "gemini" and not self.GEMINI_API_KEY:
                missing.append("GEMINI_API_KEY (required when AI_PROVIDER=gemini)")

            if missing:
                raise ValueError(
                    f"Production startup failed. Missing required environment variables:\n- "
                    + "\n- ".join(missing)
                )

        return self


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    return Settings()
