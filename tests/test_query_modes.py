import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.api.dependencies import (
    get_embedding_model,
    get_fuseki_client,
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
                "distance": 0.01,
            },
            {
                "chunk_id": "chunk-2",
                "source_type": "doc_text",
                "source_ref": "doc-2",
                "text": "France is in Europe.",
                "distance": 0.15,
            },
        ]

    def index_stats(self) -> dict[str, Any]:
        return {
            "counts_by_source_type": [{"source_type": "doc_text", "count": 2}],
            "counts_by_dataset_version": [{"dataset_version": "dev", "count": 2}],
            "embedding_dim_inferred": 768,
        }


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


class _FakeFusekiClient:
    def sparql(self, query: str) -> dict[str, Any]:
        if "FILTER(" in query:
            return {
                "results": {
                    "bindings": [
                        {
                            "s": {"type": "uri", "value": "https://example.org/Paris"},
                            "p": {"type": "uri", "value": "https://schema.org/name"},
                            "o": {"type": "literal", "value": "Paris"},
                        },
                        {
                            "s": {"type": "uri", "value": "https://example.org/Paris"},
                            "p": {"type": "uri", "value": "https://schema.org/country"},
                            "o": {"type": "uri", "value": "https://example.org/France"},
                        },
                    ]
                }
            }
        return {
            "results": {
                "bindings": [
                    {
                        "s": {"type": "uri", "value": "https://example.org/France"},
                        "p": {"type": "uri", "value": "https://schema.org/capital"},
                        "o": {"type": "uri", "value": "https://example.org/Paris"},
                    }
                ]
            }
        }


def test_query_text_mode_uses_doc_text_filter_and_debug_ids() -> None:
    fake_store = _FakeVectorStore()
    fake_telemetry = _FakeTelemetryClient()
    fake_embeddings = _FakeEmbeddingModel()
    fake_llm = _FakeLLMClient()
    app.dependency_overrides[get_vector_store] = lambda: fake_store
    app.dependency_overrides[get_telemetry_client] = lambda: fake_telemetry
    app.dependency_overrides[get_embedding_model] = lambda: fake_embeddings
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    app.dependency_overrides[get_fuseki_client] = lambda: _FakeFusekiClient()
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
    assert payload["abstained"] is False
    assert payload["citations"][0]["chunk_id"] == "chunk-1"
    assert payload["answer"] == "Evidence-based answer from retrieved facts."
    assert payload["debug"]["retrieved_chunk_ids"] == ["chunk-1", "chunk-2"]
    assert fake_store.calls[0]["filters"] == {"source_type": "doc_text", "dataset_version": "dev"}
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
    app.dependency_overrides[get_fuseki_client] = lambda: _FakeFusekiClient()
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
    assert fake_store.calls[0]["filters"] == {"source_type": "table_row", "dataset_version": "dev"}
    assert fake_store.calls[1]["filters"] == {"source_type": "kg_text", "dataset_version": "dev"}
    assert len(fake_telemetry.calls) == 2
    app.dependency_overrides.clear()


def test_query_kg_mode_uses_sparql_retrieval_without_vector_search() -> None:
    fake_store = _FakeVectorStore()
    fake_telemetry = _FakeTelemetryClient()
    fake_embeddings = _FakeEmbeddingModel()
    fake_llm = _FakeLLMClient()
    app.dependency_overrides[get_vector_store] = lambda: fake_store
    app.dependency_overrides[get_telemetry_client] = lambda: fake_telemetry
    app.dependency_overrides[get_embedding_model] = lambda: fake_embeddings
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    app.dependency_overrides[get_fuseki_client] = lambda: _FakeFusekiClient()
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
    assert payload["abstained"] is False
    assert len(payload["citations"]) > 0
    assert "kg-triple-" in payload["citations"][0]["chunk_id"]
    assert fake_store.calls == []
    assert fake_telemetry.calls[0]["retrieved_chunk_ids"][0].startswith("kg-triple-")
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
    app.dependency_overrides[get_fuseki_client] = lambda: _FakeFusekiClient()
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"question": "unknown", "mode": "text", "top_k": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "I don't have enough information in the provided sources to answer that."
    assert payload["abstained"] is True
    assert payload["citations"] == []
    assert fake_telemetry.calls[0]["abstained"] is True
    app.dependency_overrides.clear()


def test_query_debug_true_includes_hits_scores_and_evidence_text() -> None:
    fake_store = _FakeVectorStore()
    fake_telemetry = _FakeTelemetryClient()
    fake_embeddings = _FakeEmbeddingModel()
    fake_llm = _FakeLLMClient()
    app.dependency_overrides[get_vector_store] = lambda: fake_store
    app.dependency_overrides[get_telemetry_client] = lambda: fake_telemetry
    app.dependency_overrides[get_embedding_model] = lambda: fake_embeddings
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    app.dependency_overrides[get_fuseki_client] = lambda: _FakeFusekiClient()
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"question": "What is France?", "mode": "text", "top_k": 2, "debug": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["debug"]["dataset_version"] == "dev"
    assert payload["debug"]["providers"]["llm_provider"] == "ollama"
    assert payload["debug"]["providers"]["embeddings_provider"] == "ollama"
    assert payload["debug"]["retrieval_hits"][0]["chunk_id"] == "chunk-1"
    assert payload["debug"]["retrieval_hits"][0]["score"] == 0.01
    assert "Paris is the capital of France." in payload["debug"]["evidence_text"]
    app.dependency_overrides.clear()


def test_query_kg_mode_debug_includes_triples_and_evidence_pack() -> None:
    fake_store = _FakeVectorStore()
    fake_telemetry = _FakeTelemetryClient()
    fake_embeddings = _FakeEmbeddingModel()
    fake_llm = _FakeLLMClient()
    app.dependency_overrides[get_vector_store] = lambda: fake_store
    app.dependency_overrides[get_telemetry_client] = lambda: fake_telemetry
    app.dependency_overrides[get_embedding_model] = lambda: fake_embeddings
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    app.dependency_overrides[get_fuseki_client] = lambda: _FakeFusekiClient()
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"question": "What is the capital of France?", "mode": "kg", "top_k": 5, "debug": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "kg"
    assert "triples" in payload["debug"]
    assert len(payload["debug"]["triples"]) > 0
    assert "evidence_pack" in payload["debug"]
    assert len(payload["debug"]["evidence_pack"]["facts"]) > 0
    assert "kg_queries" in payload["debug"]
    app.dependency_overrides.clear()
