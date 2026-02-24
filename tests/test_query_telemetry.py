import os
import sys
import uuid
from pathlib import Path
from typing import Any

import psycopg
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.api.dependencies import get_vector_store
from apps.api.main import app
from src.tfmkg.adapters.db.psycopg_client import normalize_psycopg_dsn


class _FakeVectorStore:
    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        return None

    def similarity_search(
        self, embedding: list[float], top_k: int, filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return [{"chunk_id": "r-1"}, {"chunk_id": "r-2"}]


def _database_url() -> str:
    return os.environ["DATABASE_URL"]


def test_query_writes_run_row_with_retrieved_ids() -> None:
    question = f"telemetry-{uuid.uuid4()}"
    dsn = normalize_psycopg_dsn(_database_url())
    app.dependency_overrides[get_vector_store] = lambda: _FakeVectorStore()
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
                    SELECT question, mode, top_k, retrieved_chunk_ids, dataset_version
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
    finally:
        app.dependency_overrides.clear()
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM runs WHERE question = %s", (question,))
            conn.commit()
