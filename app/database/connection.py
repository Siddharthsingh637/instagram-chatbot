"""Database connection, session management, and Base declarative class."""

import sqlite3
from typing import Generator
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from app.core.config import get_settings
from app.core.logging import logger


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy models."""
    pass


def _create_engine_instance() -> Engine:
    """Create and configure the SQLAlchemy engine based on configuration."""
    settings = get_settings()
    db_url = settings.DATABASE_URL

    connect_args = {}
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        engine = create_engine(db_url, connect_args=connect_args)

        # Enable SQLite foreign key enforcement
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection: sqlite3.Connection, connection_record: object) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine

    return create_engine(
        db_url,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=10,
        max_overflow=20,
    )


engine = _create_engine_instance()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session and guarantees closure."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connectivity() -> bool:
    """Check database connectivity by executing a lightweight query.

    Returns:
        bool: True if connection succeeded, False otherwise.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error("Database health check failed: %s", str(e))
        return False
