"""Gemini AI provider stub for upcoming live Google search grounding integration."""

from typing import Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel
from app.ai.providers.base import AIProvider, AIProviderError

T = TypeVar("T", bound=BaseModel)


class GeminiProvider(AIProvider):
    """Gemini provider stub, designed for live Google Search grounding integration in future stages."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model

    def generate_text(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Stub method for Gemini text generation."""
        if not self.api_key:
            raise AIProviderError("GEMINI_API_KEY is not configured")
        raise NotImplementedError("Gemini live search grounding provider will be fully activated in a future stage.")

    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T:
        """Stub method for Gemini structured generation."""
        if not self.api_key:
            raise AIProviderError("GEMINI_API_KEY is not configured")
        raise NotImplementedError("Gemini structured generation stub.")
