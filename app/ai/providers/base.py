"""Abstract AI Provider interface and structured response extraction helpers."""

from abc import ABC, abstractmethod
import json
import re
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError
from app.core.logging import logger

T = TypeVar("T", bound=BaseModel)

JSON_FENCE_REGEX = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def extract_and_parse_json(text: str, schema: Type[T]) -> T:
    """Extract and validate JSON matching a Pydantic schema, stripping code fences if present.

    Args:
        text: Raw LLM output string containing JSON or markdown code fence.
        schema: Target Pydantic model class.

    Returns:
        T: Instantiated and validated Pydantic model.

    Raises:
        ValueError: If JSON cannot be parsed or validation fails.
    """
    cleaned = text.strip()

    # Match code fence if present
    match = JSON_FENCE_REGEX.search(cleaned)
    if match:
        cleaned = match.group(1).strip()
    else:
        # If no explicit fence, locate first '{' and last '}'
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("JSON decode failed on text: %r: %s", text, str(exc))
        raise ValueError(f"Invalid JSON returned by provider: {str(exc)}") from exc

    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        logger.warning("Pydantic validation failed for schema %s: %s", schema.__name__, str(exc))
        raise ValueError(f"Structured output schema mismatch: {str(exc)}") from exc


class AIProviderError(Exception):
    """Base exception for AI provider communication issues (timeouts, rate limits, server errors)."""
    pass


class AIProvider(ABC):
    """Abstract interface for AI model providers."""

    @abstractmethod
    def generate_text(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generate text completion from a sequence of chat messages.

        Args:
            messages: List of message dicts: [{"role": "user"|"assistant"|"system", "content": "..."}]
            system_prompt: Optional system prompt to prepend.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in generated completion.

        Returns:
            str: Generated completion text.
        """
        pass

    @abstractmethod
    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T:
        """Generate a structured response parsed and validated against a Pydantic schema.

        Retries once if initial output contains invalid JSON or fails schema validation.

        Args:
            messages: List of chat message dicts.
            schema: Target Pydantic model.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.

        Returns:
            T: Validated Pydantic model instance.
        """
        pass
