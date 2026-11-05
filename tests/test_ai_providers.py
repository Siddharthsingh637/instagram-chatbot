"""Tests for AI provider abstraction, JSON parsing, retry logic, and fallback orchestration."""

from typing import Dict, List, Optional, Type, TypeVar
import httpx
import pytest
from pydantic import BaseModel, Field
from app.ai.provider_factory import FallbackAIClient
from app.ai.providers.base import AIProvider, AIProviderError, extract_and_parse_json
from app.ai.providers.groq_provider import GroqProvider
from app.ai.providers.openrouter_provider import OpenRouterProvider

T = TypeVar("T", bound=BaseModel)


class SampleDecision(BaseModel):
    should_respond: bool
    intent: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class FakeProvider(AIProvider):
    """In-memory test double for AIProvider."""

    def __init__(
        self,
        text_response: str = "Hello from FakeProvider!",
        structured_response: Optional[BaseModel] = None,
        should_fail: bool = False,
        failure_error: Optional[AIProviderError] = None,
    ):
        self.text_response = text_response
        self.structured_response = structured_response
        self.should_fail = should_fail
        self.failure_error = failure_error or AIProviderError("Simulated provider failure")
        self.call_count = 0

    def generate_text(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        self.call_count += 1
        if self.should_fail:
            raise self.failure_error
        return self.text_response

    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T:
        self.call_count += 1
        if self.should_fail:
            raise self.failure_error
        if self.structured_response is not None:
            return self.structured_response  # type: ignore
        return schema.model_validate({"should_respond": True, "intent": "general", "relevance_score": 0.95})


def test_extract_and_parse_json():
    """Verify parsing valid JSON, markdown code fence stripping, and error cases."""
    # 1. Plain valid JSON
    raw_plain = '{"should_respond": true, "intent": "greeting", "relevance_score": 0.85}'
    parsed = extract_and_parse_json(raw_plain, SampleDecision)
    assert parsed.should_respond is True
    assert parsed.intent == "greeting"
    assert parsed.relevance_score == 0.85

    # 2. Markdown json code block
    raw_fence = """Here is the decision:
```json
{
  "should_respond": false,
  "intent": "statement",
  "relevance_score": 0.2
}
```
Hope this helps!"""
    parsed_fence = extract_and_parse_json(raw_fence, SampleDecision)
    assert parsed_fence.should_respond is False
    assert parsed_fence.intent == "statement"

    # 3. Invalid JSON syntax
    with pytest.raises(ValueError) as exc_info:
        extract_and_parse_json("not valid json", SampleDecision)
    assert "Invalid JSON" in str(exc_info.value)

    # 4. Schema validation failure
    with pytest.raises(ValueError) as exc_info2:
        extract_and_parse_json('{"should_respond": "yes", "relevance_score": 5.0}', SampleDecision)
    assert "schema mismatch" in str(exc_info2.value).lower()


def test_fallback_client_primary_success():
    """Test fallback client uses primary provider when it succeeds."""
    primary = FakeProvider(text_response="Primary response")
    secondary = FakeProvider(text_response="Secondary response")

    client = FallbackAIClient([primary, secondary])
    result = client.generate_text([{"role": "user", "content": "hi"}])

    assert result == "Primary response"
    assert primary.call_count == 1
    assert secondary.call_count == 0


def test_fallback_client_switches_on_failure():
    """Test fallback client seamlessly switches to secondary provider when primary fails."""
    primary = FakeProvider(should_fail=True)
    secondary = FakeProvider(text_response="Fallback response")

    client = FallbackAIClient([primary, secondary])
    result = client.generate_text([{"role": "user", "content": "hi"}])

    assert result == "Fallback response"
    assert primary.call_count == 1
    assert secondary.call_count == 1


def test_fallback_client_all_fail():
    """Test fallback client raises AIProviderError if all providers fail."""
    primary = FakeProvider(should_fail=True)
    secondary = FakeProvider(should_fail=True)

    client = FallbackAIClient([primary, secondary])
    with pytest.raises(AIProviderError) as exc_info:
        client.generate_text([{"role": "user", "content": "hi"}])
    assert "All AI providers in fallback chain failed" in str(exc_info.value)


def test_groq_provider_with_mock_http():
    """Test GroqProvider handles mock HTTP responses, rate limits, and structured generation."""
    # Test 200 OK
    def mock_ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": "Groq reply"}}]},
        )

    groq = GroqProvider(api_key="test_key", http_client=httpx.Client(transport=httpx.MockTransport(mock_ok)))
    assert groq.generate_text([{"role": "user", "content": "hello"}]) == "Groq reply"

    # Test 429 Rate Limit
    def mock_429(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=429, text="Rate limit reached")

    groq_429 = GroqProvider(api_key="test_key", http_client=httpx.Client(transport=httpx.MockTransport(mock_429)))
    with pytest.raises(AIProviderError) as exc_info:
        groq_429.generate_text([{"role": "user", "content": "hello"}])
    assert "429" in str(exc_info.value)


def test_structured_retry_on_invalid_json():
    """Test that GroqProvider retries once when the first response has invalid JSON."""
    calls = 0

    def mock_retry_flow(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            # First response is malformed
            return httpx.Response(status_code=200, json={"choices": [{"message": {"content": "Malformed JSON output"}}]})
        # Second response is valid
        return httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": '{"should_respond": true, "intent": "question", "relevance_score": 0.9}'}}]},
        )

    groq = GroqProvider(api_key="test_key", http_client=httpx.Client(transport=httpx.MockTransport(mock_retry_flow)))
    decision = groq.generate_structured([{"role": "user", "content": "Where are you located?"}], SampleDecision)

    assert calls == 2
    assert decision.should_respond is True
    assert decision.intent == "question"
    assert decision.relevance_score == 0.9
