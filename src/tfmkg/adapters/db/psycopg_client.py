from __future__ import annotations

import psycopg


def normalize_psycopg_dsn(database_url: str) -> str:
    return database_url.replace("postgresql+psycopg://", "postgresql://", 1)


class PsycopgDBClient:
    def __init__(self, database_url: str):
        self._dsn = normalize_psycopg_dsn(database_url)

    def ping(self) -> None:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
