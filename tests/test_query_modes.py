import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.api.dependencies import (
    get_embedding_model,
    get_llm_client,
    get_telemetry_client,
    get_vector_store,
)
from apps.api.main import app


class _FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        return None

    def similarity_search(
        self, embedding: list[float], top_k: int, filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "embedding": embedding,
                "top_k": top_k,
                "filters": filters,
            }
        )
        return [
            {
                "chunk_id": "chunk-1",
                "source_type": "doc_text",
                "source_ref": "doc-1",
                "text": "Paris is the capital of France.",
            },
            {
                "chunk_id": "chunk-2",
                "source_type": "doc_text",
                "source_ref": "doc-2",
                "text": "France is in Europe.",
            },
        ]


class _FakeTelemetryClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def log_query_run(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class _FakeEmbeddingModel:
    provider_name = "ollama"
    model_name = "embeddinggemma"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] + [0.0] * 767


class _FakeLLMClient:
    provider_name = "ollama"
    model_name = "mistral:7b"

    def generate(self, messages: list[Any], **_: Any) -> Any:
        class _Result:
            text = "Evidence-based answer from retrieved facts."
            prompt_tokens = 12
            completion_tokens = 8

        return _Result()


def test_query_text_mode_uses_doc_text_filter_and_debug_ids() -> None:
    fake_store = _FakeVectorStore()
    fake_telemetry = _FakeTelemetryClient()
    fake_embeddings = _FakeEmbeddingModel()
    fake_llm = _FakeLLMClient()
    app.dependency_overrides[get_vector_store] = lambda: fake_store
    app.dependency_overrides[get_telemetry_client] = lambda: fake_telemetry
    app.dependency_overrides[get_embedding_model] = lambda: fake_embeddings
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"question": "What is this?", "mode": "text", "top_k": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "text"
    assert payload["top_k"] == 2
    assert len(payload["citations"]) == 2
    assert payload["citations"][0]["chunk_id"] == "chunk-1"
    assert payload["answer"] == "Evidence-based answer from retrieved facts."
    assert payload["debug"]["retrieved_chunk_ids"] == ["chunk-1", "chunk-2"]
    assert fake_store.calls[0]["filters"] == {"source_type": "doc_text"}
    assert fake_telemetry.calls[0]["retrieved_chunk_ids"] == ["chunk-1", "chunk-2"]
    assert fake_telemetry.calls[0]["abstained"] is False
    assert fake_telemetry.calls[0]["evidence_pack"]["prompt_meta"]["llm_model"] == "mistral:7b"
    app.dependency_overrides.clear()


def test_query_table_and_hybrid_modes_change_filters() -> None:
    fake_store = _FakeVectorStore()
    fake_telemetry = _FakeTelemetryClient()
    fake_embeddings = _FakeEmbeddingModel()
    fake_llm = _FakeLLMClient()
    app.dependency_overrides[get_vector_store] = lambda: fake_store
    app.dependency_overrides[get_telemetry_client] = lambda: fake_telemetry
    app.dependency_overrides[get_embedding_model] = lambda: fake_embeddings
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    client = TestClient(app)

    table_response = client.post(
        "/query",
        json={"question": "table q", "mode": "table", "top_k": 3},
    )
    hybrid_response = client.post(
        "/query",
        json={"question": "hybrid q", "mode": "hybrid", "top_k": 4},
    )

    assert table_response.status_code == 200
    assert hybrid_response.status_code == 200
    assert fake_store.calls[0]["filters"] == {"source_type": "table_row"}
    assert fake_store.calls[1]["filters"] == {"source_type": "kg_text"}
    assert len(fake_telemetry.calls) == 2
    app.dependency_overrides.clear()


def test_query_kg_mode_keeps_shape_without_vector_search() -> None:
    fake_store = _FakeVectorStore()
    fake_telemetry = _FakeTelemetryClient()
    fake_embeddings = _FakeEmbeddingModel()
    fake_llm = _FakeLLMClient()
    app.dependency_overrides[get_vector_store] = lambda: fake_store
    app.dependency_overrides[get_telemetry_client] = lambda: fake_telemetry
    app.dependency_overrides[get_embedding_model] = lambda: fake_embeddings
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"question": "kg q", "mode": "kg", "top_k": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "kg"
    assert payload["top_k"] == 5
    assert isinstance(payload["answer"], str)
    assert isinstance(payload["citations"], list)
    assert isinstance(payload["debug"], dict)
    assert "not implemented" in payload["answer"].lower()
    assert fake_store.calls == []
    assert fake_telemetry.calls[0]["retrieved_chunk_ids"] == []
    app.dependency_overrides.clear()


def test_query_abstains_when_no_evidence() -> None:
    class _EmptyVectorStore(_FakeVectorStore):
        def similarity_search(
            self, embedding: list[float], top_k: int, filters: dict[str, Any]
        ) -> list[dict[str, Any]]:
            return []

    fake_store = _EmptyVectorStore()
    fake_telemetry = _FakeTelemetryClient()
    fake_embeddings = _FakeEmbeddingModel()
    fake_llm = _FakeLLMClient()
    app.dependency_overrides[get_vector_store] = lambda: fake_store
    app.dependency_overrides[get_telemetry_client] = lambda: fake_telemetry
    app.dependency_overrides[get_embedding_model] = lambda: fake_embeddings
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"question": "unknown", "mode": "text", "top_k": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "I don't have enough information in the provided sources to answer that."
    assert payload["citations"] == []
    assert fake_telemetry.calls[0]["abstained"] is True
    app.dependency_overrides.clear()
