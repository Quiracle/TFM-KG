import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.api.dependencies import get_embedding_model, get_llm_client, get_vector_store
from apps.api.main import app


class _FakeVectorStore:
    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        return None

    def similarity_search(
        self, embedding: list[float], top_k: int, filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return []

    def index_stats(self) -> dict[str, Any]:
        return {
            "counts_by_source_type": [
                {"source_type": "doc_text", "count": 3},
                {"source_type": "kg_text", "count": 5},
            ],
            "counts_by_dataset_version": [
                {"dataset_version": "dev", "count": 8},
            ],
            "embedding_dim_inferred": 768,
        }


class _FakeEmbeddingModel:
    provider_name = "ollama"
    model_name = "embeddinggemma"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return []

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


class _FakeLLMClient:
    provider_name = "openai"
    model_name = "gpt-4.1-mini"

    def generate(self, messages: list[Any], **_: Any) -> Any:
        raise NotImplementedError


def test_index_stats_returns_counts_embedding_dim_and_providers() -> None:
    app.dependency_overrides[get_vector_store] = lambda: _FakeVectorStore()
    app.dependency_overrides[get_embedding_model] = lambda: _FakeEmbeddingModel()
    app.dependency_overrides[get_llm_client] = lambda: _FakeLLMClient()
    client = TestClient(app)

    response = client.get("/stats/index")

    assert response.status_code == 200
    payload = response.json()
    assert payload["counts_by_source_type"][0]["source_type"] == "doc_text"
    assert payload["counts_by_dataset_version"][0]["dataset_version"] == "dev"
    assert payload["embedding_dim"]["configured"] == 768
    assert payload["embedding_dim"]["inferred"] == 768
    assert payload["providers"] == {
        "llm_provider": "openai",
        "llm_model": "gpt-4.1-mini",
        "embeddings_provider": "ollama",
        "embed_model": "embeddinggemma",
    }
    app.dependency_overrides.clear()
