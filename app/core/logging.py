"""Structured logging configuration with token and secret masking."""

import logging
import re
import sys
from typing import Any

# Regex patterns to detect and mask sensitive tokens in log records
SENSITIVE_PATTERNS = [
    (re.compile(r"(bearer\s+)([a-zA-Z0-9_\-\.]{8,})", re.IGNORECASE), r"\1***MASKED***"),
    (re.compile(r"([a-zA-Z0-9_\-]{4})[a-zA-Z0-9_\-]{12,}([a-zA-Z0-9_\-]{4})"), r"\1***MASKED***\2"),
]


class SensitiveFilter(logging.Filter):
    """Filter that masks tokens and credentials from log output."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._mask(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self._mask(v) if isinstance(v, str) else v for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(self._mask(v) if isinstance(v, str) else v for v in record.args)
        return True

    @staticmethod
    def _mask(text: str) -> str:
        masked = text
        for pattern, replacement in SENSITIVE_PATTERNS:
            masked = pattern.sub(replacement, masked)
        return masked


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging for the application.

    Args:
        log_level: Logging level (e.g. INFO, DEBUG, WARNING).
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # Avoid adding duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.addFilter(SensitiveFilter())
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            handler.addFilter(SensitiveFilter())


logger = logging.getLogger("instagram_bot")
