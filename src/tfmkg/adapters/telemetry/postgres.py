from __future__ import annotations

import json

import psycopg

from src.tfmkg.adapters.db.psycopg_client import normalize_psycopg_dsn


class PostgresTelemetryClient:
    def __init__(self, database_url: str):
        self._dsn = normalize_psycopg_dsn(database_url)

    def log_query_run(
        self,
        *,
        question: str,
        mode: str,
        top_k: int,
        retrieved_chunk_ids: list[str],
        evidence_pack: dict,
        answer: str,
        abstained: bool,
        latency_ms: int,
        dataset_version: str = "dev",
    ) -> None:
        query = """
            INSERT INTO runs (
                question,
                mode,
                top_k,
                retrieved_chunk_ids,
                evidence_pack,
                answer,
                abstained,
                latency_ms,
                dataset_version
            ) VALUES (
                %(question)s,
                %(mode)s,
                %(top_k)s,
                %(retrieved_chunk_ids)s::jsonb,
                %(evidence_pack)s::jsonb,
                %(answer)s,
                %(abstained)s,
                %(latency_ms)s,
                %(dataset_version)s
            )
        """
        params = {
            "question": question,
            "mode": mode,
            "top_k": top_k,
            "retrieved_chunk_ids": json.dumps(retrieved_chunk_ids),
            "evidence_pack": json.dumps(evidence_pack),
            "answer": answer,
            "abstained": abstained,
            "latency_ms": latency_ms,
            "dataset_version": dataset_version,
        }

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
            conn.commit()
