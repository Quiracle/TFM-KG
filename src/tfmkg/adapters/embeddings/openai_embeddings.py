from __future__ import annotations

import time

import httpx


class OpenAIEmbeddingsClient:
    provider_name = "openai"

    def __init__(self, api_key: str, base_url: str, model: str, timeout_s: int):
        self.model_name = model
        self._url = f"{base_url.rstrip('/')}/embeddings"
        self._timeout_s = timeout_s
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model_name, "input": texts}
        data = self._post_with_retry(payload)
        return [item["embedding"] for item in data.get("data", [])]

    def embed_query(self, text: str) -> list[float]:
        embeddings = self.embed_texts([text])
        if not embeddings:
            raise RuntimeError("Embeddings provider returned no vectors.")
        return embeddings[0]

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
                    raise RuntimeError(f"OpenAI embeddings request failed: {exc}") from None
                time.sleep(0.3 * attempt)
        raise RuntimeError("OpenAI embeddings request failed after retries.")
