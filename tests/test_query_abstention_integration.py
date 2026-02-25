import os
import sys
import uuid
from pathlib import Path
from typing import Any

import psycopg
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.api.dependencies import get_embedding_model, get_fuseki_client, get_llm_client
from apps.api.main import app
from src.tfmkg.adapters.db.psycopg_client import normalize_psycopg_dsn


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
            text = "This should not be used when abstaining."
            prompt_tokens = 0
            completion_tokens = 0

        return _Result()


class _FakeFusekiClient:
    def sparql(self, query: str) -> dict:
        return {"results": {"bindings": []}}


def _database_url() -> str:
    return os.environ["DATABASE_URL"]


def test_query_abstains_for_empty_dataset_version_index() -> None:
    question = f"abstain-empty-{uuid.uuid4()}"
    dsn = normalize_psycopg_dsn(_database_url())
    missing_dataset = "__missing_dataset_for_abstention_test__"
    app.dependency_overrides[get_embedding_model] = lambda: _FakeEmbeddingModel()
    app.dependency_overrides[get_llm_client] = lambda: _FakeLLMClient()
    app.dependency_overrides[get_fuseki_client] = lambda: _FakeFusekiClient()
    client = TestClient(app)

    try:
        response = client.post(
            "/query",
            json={
                "question": question,
                "mode": "hybrid",
                "top_k": 5,
                "dataset_version": missing_dataset,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["abstained"] is True
        assert payload["answer"] == "I don't have enough information in the provided sources to answer that."
        assert payload["citations"] == []

        with psycopg.connect(dsn) as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(
                    """
                    SELECT abstained, retrieved_chunk_ids, dataset_version
                    FROM runs
                    WHERE question = %s
                    ORDER BY ts DESC
                    LIMIT 1
                    """,
                    (question,),
                )
                row = cur.fetchone()

        assert row is not None
        assert row["abstained"] is True
        assert row["retrieved_chunk_ids"] == []
        assert row["dataset_version"] == missing_dataset
    finally:
        app.dependency_overrides.clear()
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM runs WHERE question = %s", (question,))
            conn.commit()
