"""Security utilities: HMAC-SHA256 signature verification, API key auth, and token masking."""

import hashlib
import hmac
from typing import Optional
from fastapi import Header, HTTPException, status
from app.core.config import get_settings


def verify_meta_signature(raw_body: bytes, signature_header: Optional[str], secret: Optional[str]) -> bool:
    """Verify Meta's X-Hub-Signature-256 header using constant-time HMAC-SHA256 comparison.

    Args:
        raw_body: The raw request body bytes.
        signature_header: The value of the X-Hub-Signature-256 header (format: sha256=...).
        secret: The Meta App Secret.

    Returns:
        bool: True if signature is valid, False otherwise.
    """
    if not signature_header or not secret:
        return False

    prefix = "sha256="
    if not signature_header.startswith(prefix):
        return False

    received_sig = signature_header[len(prefix):].strip()
    expected_sig = hmac.new(
        key=secret.encode("utf-8"),
        msg=raw_body,
        digestmod=hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(received_sig, expected_sig)


def verify_admin_key(x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key")) -> str:
    """FastAPI dependency to verify the ADMIN_API_KEY from X-Admin-Key header.

    Args:
        x_admin_key: Admin API key passed in header.

    Returns:
        str: The validated admin API key.

    Raises:
        HTTPException: 401 if header is missing or invalid.
    """
    settings = get_settings()
    configured_key = settings.ADMIN_API_KEY

    if not configured_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ADMIN_API_KEY is not configured on the server",
        )

    if not x_admin_key or not hmac.compare_digest(x_admin_key, configured_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-Key header",
        )

    return x_admin_key


def mask_secret(secret: Optional[str]) -> str:
    """Mask a secret string for safe logging.

    Args:
        secret: Sensitive string to mask.

    Returns:
        str: Partially masked string.
    """
    if not secret:
        return "<empty>"
    if len(secret) <= 8:
        return "***"
    return f"{secret[:3]}***{secret[-3:]}"
