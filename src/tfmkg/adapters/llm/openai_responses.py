from __future__ import annotations

import time

import httpx

from src.tfmkg.domain.ports.llm import LLMMessage, LLMResult


class OpenAIResponsesClient:
    provider_name = "openai"

    def __init__(self, api_key: str, base_url: str, model: str, timeout_s: int):
        self.model_name = model
        self._url = f"{base_url.rstrip('/')}/responses"
        self._timeout_s = timeout_s
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 300,
    ) -> LLMResult:
        input_items = [{"role": msg.role, "content": msg.content} for msg in messages]
        payload = {
            "model": self.model_name,
            "input": input_items,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        data = self._post_with_retry(payload)
        text = data.get("output_text", "").strip()
        usage = data.get("usage", {})
        return LLMResult(
            text=text,
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
        )

    def _post_with_retry(self, payload: dict) -> dict:
        retries = 3
        for attempt in range(1, retries + 1):
            try:
                with httpx.Client(timeout=self._timeout_s) as client:
                    response = client.post(self._url, headers=self._headers, json=payload)
                response.raise_for_status()
                return response.json()
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                is_last = attempt == retries
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                retryable = status_code is None or status_code >= 500
                if is_last or not retryable:
                    raise RuntimeError(f"OpenAI responses request failed: {exc}") from None
                time.sleep(0.3 * attempt)
        raise RuntimeError("OpenAI responses request failed after retries.")
