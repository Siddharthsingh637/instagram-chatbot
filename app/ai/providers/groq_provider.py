"""Groq AI provider implementation."""

import json
from typing import Dict, List, Optional, Type, TypeVar
import httpx
from pydantic import BaseModel
from app.ai.providers.base import AIProvider, AIProviderError, extract_and_parse_json
from app.core.logging import logger

T = TypeVar("T", bound=BaseModel)


class GroqProvider(AIProvider):
    """Groq API provider using high-throughput LLaMA models."""

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        http_client: Optional[httpx.Client] = None,
        timeout: float = 15.0,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.http_client = http_client or httpx.Client(timeout=timeout)
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def _call_api(self, chat_messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        if not self.api_key:
            raise AIProviderError("GROQ_API_KEY is not configured")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = self.http_client.post(self.endpoint, json=payload, headers=headers)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            logger.warning("Groq API network/timeout error: %s", str(exc))
            raise AIProviderError(f"Groq network or timeout failure: {str(exc)}") from exc

        if response.status_code == 429:
            logger.warning("Groq API rate limit (429) hit: %s", response.text)
            raise AIProviderError(f"Groq rate limit exceeded (429): {response.text}")

        if response.status_code >= 500:
            logger.warning("Groq API server error (%d): %s", response.status_code, response.text)
            raise AIProviderError(f"Groq server error {response.status_code}: {response.text}")

        if response.status_code != 200:
            raise AIProviderError(f"Groq API error {response.status_code}: {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]

    def generate_text(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.extend(messages)

        return self._call_api(chat_messages, temperature=temperature, max_tokens=max_tokens)

    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T:
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        instruction = (
            f"You MUST respond ONLY with valid JSON strictly adhering to this JSON Schema:\n{schema_json}\n"
            "Do not output markdown explanations or extra text."
        )

        full_system = f"{system_prompt}\n\n{instruction}" if system_prompt else instruction
        chat_messages = [{"role": "system", "content": full_system}]
        chat_messages.extend(messages)

        raw = self._call_api(chat_messages, temperature=temperature, max_tokens=1000)

        # First attempt parse
        try:
            return extract_and_parse_json(raw, schema)
        except ValueError as err:
            logger.warning("First attempt to parse structured JSON failed: %s. Retrying once...", str(err))

            # Retry once with error correction prompt
            retry_messages = list(chat_messages)
            retry_messages.append({"role": "assistant", "content": raw})
            retry_messages.append({
                "role": "user",
                "content": f"Your response was not valid JSON conforming to the schema. Error: {str(err)}. Return ONLY valid JSON.",
            })

            second_raw = self._call_api(retry_messages, temperature=0.1, max_tokens=1000)
            return extract_and_parse_json(second_raw, schema)
