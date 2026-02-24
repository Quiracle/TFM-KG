from functools import lru_cache

from src.tfmkg.adapters.db import PsycopgDBClient
from src.tfmkg.adapters.telemetry import PostgresTelemetryClient
from src.tfmkg.adapters.triplestore import FusekiClient
from src.tfmkg.adapters.vectorstore.pgvector import PgVectorRepository
from src.tfmkg.domain.ports.vector_store import VectorStorePort
from src.tfmkg.core.config import settings


@lru_cache(maxsize=1)
def get_db_client() -> PsycopgDBClient:
    return PsycopgDBClient(settings.database_url)


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStorePort:
    return PgVectorRepository(settings.database_url)


@lru_cache(maxsize=1)
def get_fuseki_client() -> FusekiClient:
    return FusekiClient(settings.fuseki_url, settings.fuseki_dataset)


@lru_cache(maxsize=1)
def get_telemetry_client() -> PostgresTelemetryClient:
    return PostgresTelemetryClient(settings.database_url)
