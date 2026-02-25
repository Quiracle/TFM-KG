import os
import sys
import uuid
from pathlib import Path
from typing import Any

import psycopg
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.api.dependencies import get_embedding_model, get_llm_client, get_vector_store
from apps.api.main import app
from src.tfmkg.adapters.db.psycopg_client import normalize_psycopg_dsn


class _FakeVectorStore:
    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        return None

    def similarity_search(
        self, embedding: list[float], top_k: int, filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": "r-1",
                "source_type": "doc_text",
                "source_ref": "doc-1",
                "text": "Evidence one.",
            },
            {
                "chunk_id": "r-2",
                "source_type": "doc_text",
                "source_ref": "doc-2",
                "text": "Evidence two.",
            },
        ]


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

    def generate(self, messages: list[Any], **kwargs: Any) -> Any:
        class _Result:
            text = "Grounded answer."
            prompt_tokens = 20
            completion_tokens = 7

        return _Result()


def _database_url() -> str:
    return os.environ["DATABASE_URL"]


def test_query_writes_run_row_with_retrieved_ids() -> None:
    question = f"telemetry-{uuid.uuid4()}"
    dsn = normalize_psycopg_dsn(_database_url())
    app.dependency_overrides[get_vector_store] = lambda: _FakeVectorStore()
    app.dependency_overrides[get_embedding_model] = lambda: _FakeEmbeddingModel()
    app.dependency_overrides[get_llm_client] = lambda: _FakeLLMClient()
    client = TestClient(app)

    try:
        response = client.post(
            "/query",
            json={"question": question, "mode": "text", "top_k": 2},
        )

        assert response.status_code == 200

        with psycopg.connect(dsn) as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(
                    """
                    SELECT question, mode, top_k, retrieved_chunk_ids, dataset_version, abstained, evidence_pack
                    FROM runs
                    WHERE question = %s
                    ORDER BY ts DESC
                    LIMIT 1
                    """,
                    (question,),
                )
                row = cur.fetchone()

        assert row is not None
        assert row["question"] == question
        assert row["mode"] == "text"
        assert row["top_k"] == 2
        assert row["retrieved_chunk_ids"] == ["r-1", "r-2"]
        assert row["dataset_version"] == "dev"
        assert row["abstained"] is False
        assert row["evidence_pack"]["prompt_meta"]["llm_provider"] == "ollama"
        assert row["evidence_pack"]["prompt_meta"]["llm_model"] == "mistral:7b"
        assert row["evidence_pack"]["prompt_meta"]["embeddings_provider"] == "ollama"
    finally:
        app.dependency_overrides.clear()
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM runs WHERE question = %s", (question,))
            conn.commit()
