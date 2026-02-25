from __future__ import annotations

import time

import httpx


class OllamaEmbeddingsClient:
    provider_name = "ollama"

    def __init__(self, base_url: str, model: str, timeout_s: int):
        self.model_name = model
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model_name, "input": texts}
        try:
            data = self._post_with_retry(f"{self._base_url}/api/embed", payload)
            return data.get("embeddings", [])
        except RuntimeError as exc:
            if "404" not in str(exc):
                raise

        embeddings: list[list[float]] = []
        for text in texts:
            legacy_payload = {"model": self.model_name, "prompt": text}
            legacy_data = self._post_with_retry(f"{self._base_url}/api/embeddings", legacy_payload)
            vector = legacy_data.get("embedding")
            if vector is None:
                raise RuntimeError("Legacy Ollama embeddings response missing 'embedding'.")
            embeddings.append(vector)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        embeddings = self.embed_texts([text])
        if not embeddings:
            raise RuntimeError("Embeddings provider returned no vectors.")
        return embeddings[0]

    def _post_with_retry(self, url: str, payload: dict) -> dict:
        retries = 3
        for attempt in range(1, retries + 1):
            try:
                with httpx.Client(timeout=self._timeout_s) as client:
                    response = client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                is_last = attempt == retries
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                retryable = status_code is None or status_code >= 500
                if is_last or not retryable:
                    raise RuntimeError(f"Ollama embeddings request failed ({url}): {exc}") from None
                time.sleep(0.3 * attempt)
        raise RuntimeError("Ollama embeddings request failed after retries.")
