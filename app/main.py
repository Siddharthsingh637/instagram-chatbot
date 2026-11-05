"""FastAPI application entrypoint for Instagram DM Chatbot."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from app.api.admin import router as admin_router
from app.api.dev import router as dev_router
from app.api.health import router as health_router
from app.api.webhook import router as webhook_router
from app.core.config import get_settings
from app.core.logging import logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for application startup and shutdown events."""
    setup_logging()
    settings = get_settings()
    logger.info("Starting Instagram Chatbot backend in [%s] mode", settings.APP_ENV)
    yield
    logger.info("Shutting down Instagram Chatbot backend")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Instagram AI Chatbot API",
        description="Autonomous AI-powered Instagram DM assistant with decoupled memory and intelligence pipelines.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Core routes
    app.include_router(health_router)
    app.include_router(webhook_router)
    app.include_router(dev_router)
    app.include_router(admin_router)

    return app


app = create_app()
