from __future__ import annotations

import json
from typing import Any

import psycopg

from src.tfmkg.adapters.db.psycopg_client import normalize_psycopg_dsn
from src.tfmkg.domain.ports.vector_store import VectorStorePort


def _to_vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(value) for value in embedding) + "]"


class PgVectorRepository(VectorStorePort):
    def __init__(self, database_url: str):
        self._dsn = normalize_psycopg_dsn(database_url)

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return

        query = """
            INSERT INTO chunks (
                chunk_id,
                source_type,
                source_ref,
                dataset_version,
                text,
                metadata,
                embedding
            ) VALUES (
                %(chunk_id)s,
                %(source_type)s,
                %(source_ref)s,
                %(dataset_version)s,
                %(text)s,
                %(metadata)s::jsonb,
                %(embedding)s::vector
            )
            ON CONFLICT (chunk_id) DO UPDATE SET
                source_type = EXCLUDED.source_type,
                source_ref = EXCLUDED.source_ref,
                dataset_version = EXCLUDED.dataset_version,
                text = EXCLUDED.text,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding
        """

        rows = [
            {
                "chunk_id": chunk["chunk_id"],
                "source_type": chunk["source_type"],
                "source_ref": chunk["source_ref"],
                "dataset_version": chunk["dataset_version"],
                "text": chunk["text"],
                "metadata": json.dumps(chunk.get("metadata", {})),
                "embedding": _to_vector_literal(chunk["embedding"]),
            }
            for chunk in chunks
        ]

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.executemany(query, rows)
            conn.commit()

    def similarity_search(
        self, embedding: list[float], top_k: int, filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        where_parts: list[str] = []
        params: dict[str, Any] = {
            "query_embedding": _to_vector_literal(embedding),
            "top_k": top_k,
        }

        source_type = filters.get("source_type")
        if source_type:
            where_parts.append("source_type = %(source_type)s")
            params["source_type"] = source_type

        dataset_version = filters.get("dataset_version")
        if dataset_version:
            where_parts.append("dataset_version = %(dataset_version)s")
            params["dataset_version"] = dataset_version

        where_clause = ""
        if where_parts:
            where_clause = "WHERE " + " AND ".join(where_parts)

        query = f"""
            SELECT
                chunk_id,
                source_type,
                source_ref,
                dataset_version,
                text,
                metadata,
                embedding <=> %(query_embedding)s::vector AS distance
            FROM chunks
            {where_clause}
            ORDER BY embedding <=> %(query_embedding)s::vector ASC
            LIMIT %(top_k)s
        """

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

        return [dict(row) for row in rows]

    def index_stats(self) -> dict[str, Any]:
        by_source_type_query = """
            SELECT source_type, COUNT(*) AS count
            FROM chunks
            GROUP BY source_type
            ORDER BY source_type
        """
        by_dataset_version_query = """
            SELECT dataset_version, COUNT(*) AS count
            FROM chunks
            GROUP BY dataset_version
            ORDER BY dataset_version
        """
        inferred_dim_query = """
            SELECT vector_dims(embedding) AS embedding_dim
            FROM chunks
            WHERE embedding IS NOT NULL
            LIMIT 1
        """

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(by_source_type_query)
                by_source_type = [dict(row) for row in cur.fetchall()]

                cur.execute(by_dataset_version_query)
                by_dataset_version = [dict(row) for row in cur.fetchall()]

                cur.execute(inferred_dim_query)
                dim_row = cur.fetchone()

        return {
            "counts_by_source_type": by_source_type,
            "counts_by_dataset_version": by_dataset_version,
            "embedding_dim_inferred": None if dim_row is None else dim_row["embedding_dim"],
        }
