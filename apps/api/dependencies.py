from functools import lru_cache

from src.tfmkg.adapters.embeddings import OllamaEmbeddingsClient, OpenAIEmbeddingsClient
from src.tfmkg.adapters.db import PsycopgDBClient
from src.tfmkg.adapters.llm import OllamaChatClient, OpenAIResponsesClient
from src.tfmkg.adapters.telemetry import PostgresTelemetryClient
from src.tfmkg.adapters.triplestore import FusekiClient
from src.tfmkg.adapters.vectorstore.pgvector import PgVectorRepository
from src.tfmkg.domain.ports.embeddings import EmbeddingModelPort
from src.tfmkg.domain.ports.llm import LLMClientPort
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


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModelPort:
    if settings.embeddings_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDINGS_PROVIDER=openai")
        return OpenAIEmbeddingsClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_embed_model,
            timeout_s=settings.ollama_timeout_s,
        )
    return OllamaEmbeddingsClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
        timeout_s=settings.ollama_timeout_s,
    )


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClientPort:
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return OpenAIResponsesClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_llm_model,
            timeout_s=settings.ollama_timeout_s,
        )
    return OllamaChatClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_llm_model,
        timeout_s=settings.ollama_timeout_s,
        stream=settings.ollama_stream,
    )
