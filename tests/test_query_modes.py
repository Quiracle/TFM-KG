import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.api.dependencies import get_vector_store
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
        return [{"chunk_id": "chunk-1"}, {"chunk_id": "chunk-2"}]


def test_query_text_mode_uses_doc_text_filter_and_debug_ids() -> None:
    fake_store = _FakeVectorStore()
    app.dependency_overrides[get_vector_store] = lambda: fake_store
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"question": "What is this?", "mode": "text", "top_k": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "text"
    assert payload["top_k"] == 2
    assert payload["citations"] == []
    assert payload["debug"]["retrieved_chunk_ids"] == ["chunk-1", "chunk-2"]
    assert fake_store.calls[0]["filters"] == {"source_type": "doc_text"}
    app.dependency_overrides.clear()


def test_query_table_and_hybrid_modes_change_filters() -> None:
    fake_store = _FakeVectorStore()
    app.dependency_overrides[get_vector_store] = lambda: fake_store
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
    app.dependency_overrides.clear()


def test_query_kg_mode_keeps_shape_without_vector_search() -> None:
    fake_store = _FakeVectorStore()
    app.dependency_overrides[get_vector_store] = lambda: fake_store
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
    app.dependency_overrides.clear()
