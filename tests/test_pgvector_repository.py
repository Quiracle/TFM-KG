import os
import sys
import uuid
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tfmkg.adapters.db.psycopg_client import normalize_psycopg_dsn
from src.tfmkg.adapters.vectorstore.pgvector.repository import PgVectorRepository


def _database_url() -> str:
    return os.environ["DATABASE_URL"]


def _make_embedding(first_value: float) -> list[float]:
    return [first_value] + [0.0] * 767


def _fetch_chunk(chunk_id: str) -> tuple[str, dict, str] | None:
    query = """
        SELECT text, metadata, embedding::text
        FROM chunks
        WHERE chunk_id = %s
    """
    dsn = normalize_psycopg_dsn(_database_url())
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (chunk_id,))
            row = cur.fetchone()
    return row


def test_upsert_chunks_insert_and_update() -> None:
    repository = PgVectorRepository(_database_url())
    chunk_id = f"test-{uuid.uuid4()}"

    initial_chunk = {
        "chunk_id": chunk_id,
        "source_type": "doc_text",
        "source_ref": "doc-1",
        "dataset_version": "dev",
        "text": "initial text",
        "metadata": {"version": 1},
        "embedding": _make_embedding(1.0),
    }

    updated_chunk = {
        "chunk_id": chunk_id,
        "source_type": "doc_text",
        "source_ref": "doc-1",
        "dataset_version": "dev",
        "text": "updated text",
        "metadata": {"version": 2},
        "embedding": _make_embedding(2.0),
    }

    try:
        repository.upsert_chunks([initial_chunk])
        inserted = _fetch_chunk(chunk_id)

        assert inserted is not None
        assert inserted[0] == "initial text"
        assert inserted[1] == {"version": 1}
        assert inserted[2].startswith("[1")

        repository.upsert_chunks([updated_chunk])
        updated = _fetch_chunk(chunk_id)

        assert updated is not None
        assert updated[0] == "updated text"
        assert updated[1] == {"version": 2}
        assert updated[2].startswith("[2")
    finally:
        dsn = normalize_psycopg_dsn(_database_url())
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunks WHERE chunk_id = %s", (chunk_id,))
            conn.commit()
