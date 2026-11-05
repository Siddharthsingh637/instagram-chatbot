"""AI Provider Factory with automatic fallback on timeout, rate limit, or server error."""

from typing import Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel
from app.ai.providers.base import AIProvider, AIProviderError
from app.ai.providers.gemini_provider import GeminiProvider
from app.ai.providers.groq_provider import GroqProvider
from app.ai.providers.openrouter_provider import OpenRouterProvider
from app.core.config import get_settings
from app.core.logging import logger

T = TypeVar("T", bound=BaseModel)


class FallbackAIClient(AIProvider):
    """Wraps a primary AIProvider with an ordered chain of fallback providers."""

    def __init__(self, providers: List[AIProvider]):
        if not providers:
            raise ValueError("FallbackAIClient requires at least one AIProvider")
        self.providers = providers

    def generate_text(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        last_error: Optional[Exception] = None

        for idx, provider in enumerate(self.providers):
            provider_name = provider.__class__.__name__
            try:
                return provider.generate_text(
                    messages=messages,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except AIProviderError as err:
                last_error = err
                logger.warning(
                    "AI provider [%s] (step %d/%d) failed with: %s. Attempting fallback...",
                    provider_name,
                    idx + 1,
                    len(self.providers),
                    str(err),
                )

        raise AIProviderError(f"All AI providers in fallback chain failed. Last error: {str(last_error)}")

    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T:
        last_error: Optional[Exception] = None

        for idx, provider in enumerate(self.providers):
            provider_name = provider.__class__.__name__
            try:
                return provider.generate_structured(
                    messages=messages,
                    schema=schema,
                    system_prompt=system_prompt,
                    temperature=temperature,
                )
            except AIProviderError as err:
                last_error = err
                logger.warning(
                    "AI provider [%s] structured generation (step %d/%d) failed with: %s. Attempting fallback...",
                    provider_name,
                    idx + 1,
                    len(self.providers),
                    str(err),
                )

        raise AIProviderError(f"All AI providers in fallback chain failed. Last error: {str(last_error)}")


_custom_provider_override: Optional[AIProvider] = None


def set_ai_provider_override(provider: Optional[AIProvider]) -> None:
    """Set a custom AI provider override (used for injecting FakeProvider in tests)."""
    global _custom_provider_override
    _custom_provider_override = provider


def get_ai_provider() -> AIProvider:
    """Instantiate and return the configured AI provider with fallback chain."""
    if _custom_provider_override is not None:
        return _custom_provider_override

    settings = get_settings()
    chain: List[AIProvider] = []

    # Build primary provider
    if settings.AI_PROVIDER == "groq":
        chain.append(GroqProvider(api_key=settings.GROQ_API_KEY or ""))
        if settings.OPENROUTER_API_KEY:
            chain.append(OpenRouterProvider(api_key=settings.OPENROUTER_API_KEY))
    elif settings.AI_PROVIDER == "openrouter":
        chain.append(OpenRouterProvider(api_key=settings.OPENROUTER_API_KEY or ""))
        if settings.GROQ_API_KEY:
            chain.append(GroqProvider(api_key=settings.GROQ_API_KEY))
    elif settings.AI_PROVIDER == "gemini":
        chain.append(GeminiProvider(api_key=settings.GEMINI_API_KEY or ""))
        if settings.GROQ_API_KEY:
            chain.append(GroqProvider(api_key=settings.GROQ_API_KEY))

    if not chain:
        # Fallback to stub provider if nothing is configured
        chain.append(GroqProvider(api_key=""))

    return FallbackAIClient(chain)
