from __future__ import annotations

import time

import httpx

from src.tfmkg.domain.ports.llm import LLMMessage, LLMResult


class OllamaChatClient:
    provider_name = "ollama"

    def __init__(self, base_url: str, model: str, timeout_s: int, stream: bool = False):
        self.model_name = model
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self._timeout_s = timeout_s
        self._stream = stream

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 300,
    ) -> LLMResult:
        payload = {
            "model": self.model_name,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "stream": self._stream,
            "options": {"temperature": temperature, "num_predict": max_output_tokens},
        }
        data = self._post_with_retry(payload)
        text = data.get("message", {}).get("content", "").strip()
        return LLMResult(text=text)

    def _post_with_retry(self, payload: dict) -> dict:
        retries = 3
        for attempt in range(1, retries + 1):
            try:
                with httpx.Client(timeout=self._timeout_s) as client:
                    response = client.post(self._url, json=payload)
                response.raise_for_status()
                return response.json()
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                is_last = attempt == retries
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                retryable = status_code is None or status_code >= 500
                if is_last or not retryable:
                    raise RuntimeError(f"Ollama chat request failed: {exc}") from None
                time.sleep(0.3 * attempt)
        raise RuntimeError("Ollama chat request failed after retries.")
