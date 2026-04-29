from __future__ import annotations

import time
from typing import Any

from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    Anthropic,
    RateLimitError,
)

from src.tfmkg.domain.ports.llm import LLMMessage, LLMResult


class AnthropicMessagesClient:
    provider_name = "anthropic"

    def __init__(self, api_key: str, model: str, timeout_s: int):
        self.model_name = model
        self._client = Anthropic(
            api_key=api_key,
            timeout=timeout_s,
            max_retries=0,
        )

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 300,
    ) -> LLMResult:
        system_messages = [msg.content for msg in messages if msg.role == "system"]
        chat_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
            if msg.role in {"user", "assistant"}
        ]

        if not chat_messages:
            raise RuntimeError("Anthropic messages request failed: at least one user/assistant message is required.")

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        if system_messages:
            payload["system"] = "\n\n".join(system_messages)

        response = self._create_with_retry(payload)
        text = "".join(
            getattr(block, "text", "")
            for block in response.content
            if getattr(block, "type", "") == "text"
        ).strip()
        usage = getattr(response, "usage", None)
        return LLMResult(
            text=text,
            prompt_tokens=None if usage is None else getattr(usage, "input_tokens", None),
            completion_tokens=None if usage is None else getattr(usage, "output_tokens", None),
        )

    def _create_with_retry(self, payload: dict[str, Any]) -> Any:
        retries = 3
        for attempt in range(1, retries + 1):
            try:
                return self._client.messages.create(**payload)
            except (APITimeoutError, APIConnectionError, RateLimitError, APIStatusError) as exc:
                is_last = attempt == retries
                status_code = getattr(exc, "status_code", None)
                retryable = isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError))
                retryable = retryable or status_code is None or status_code >= 500
                if is_last or not retryable:
                    raise RuntimeError(f"Anthropic messages request failed: {exc}") from None
                time.sleep(0.3 * attempt)
        raise RuntimeError("Anthropic messages request failed after retries.")
