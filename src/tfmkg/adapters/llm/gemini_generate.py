from __future__ import annotations

import time
from typing import Any

import httpx

try:
    from google import genai
    from google.genai import errors, types
except ImportError:
    genai = None
    errors = None
    types = None

from src.tfmkg.domain.ports.llm import LLMMessage, LLMResult


class GeminiGenerateAdapter:
    provider_name = "gemini"

    def __init__(self, api_key: str, model: str, timeout_s: int):
        if genai is None:
            raise RuntimeError("google-genai is required when LLM_PROVIDER=gemini.")
        self.model_name = model
        retry_options = types.HttpRetryOptions(attempts=1)
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_s * 1000, retry_options=retry_options),
        )

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 300,
    ) -> LLMResult:
        system_messages = [msg.content for msg in messages if msg.role == "system"]
        contents = [
            {
                "role": "model" if msg.role == "assistant" else "user",
                "parts": [{"text": msg.content}],
            }
            for msg in messages
            if msg.role in {"user", "assistant"}
        ]

        if not contents:
            raise RuntimeError("Gemini generate request failed: at least one user/assistant message is required.")

        config: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if system_messages:
            config["system_instruction"] = "\n\n".join(system_messages)

        response = self._generate_with_retry({"model": self.model_name, "contents": contents, "config": config})
        usage = getattr(response, "usage_metadata", None)
        return LLMResult(
            text=(getattr(response, "text", "") or "").strip(),
            prompt_tokens=None if usage is None else getattr(usage, "prompt_token_count", None),
            completion_tokens=None if usage is None else getattr(usage, "candidates_token_count", None),
        )

    def _generate_with_retry(self, payload: dict[str, Any]) -> Any:
        api_errors = () if errors is None else (errors.APIError,)
        retries = 3
        for attempt in range(1, retries + 1):
            try:
                return self._client.models.generate_content(**payload)
            except api_errors + (httpx.TimeoutException, httpx.NetworkError) as exc:
                is_last = attempt == retries
                status_code = getattr(exc, "code", None)
                retryable = isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))
                retryable = retryable or status_code is None or status_code == 429 or status_code >= 500
                if is_last or not retryable:
                    raise RuntimeError(f"Gemini generate request failed: {exc}") from None
                time.sleep(0.3 * attempt)
        raise RuntimeError("Gemini generate request failed after retries.")
