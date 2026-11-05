"""Health check endpoint checking application status and database connectivity."""

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.core.config import get_settings
from app.core.logging import logger
from app.database.connection import get_db

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check(response: Response, db: Session = Depends(get_db)) -> dict:
    """Check application status and database connectivity.

    Returns:
        dict: Health status payload including database connectivity.
    """
    settings = get_settings()
    db_status = "connected"

    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        logger.error("Health check database probe failed: %s", str(e))
        db_status = "disconnected"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    is_healthy = db_status == "connected"

    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "database": db_status,
        "app_env": settings.APP_ENV,
    }
