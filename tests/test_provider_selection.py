import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.api import dependencies
from src.tfmkg.core.config import settings


def test_provider_selection_supports_mixed_mode() -> None:
    original = {
        "embeddings_provider": settings.embeddings_provider,
        "llm_provider": settings.llm_provider,
        "openai_api_key": settings.openai_api_key,
    }
    try:
        settings.embeddings_provider = "ollama"
        settings.llm_provider = "openai"
        settings.openai_api_key = "test-key"
        dependencies.get_embedding_model.cache_clear()
        dependencies.get_llm_client.cache_clear()

        embed_model = dependencies.get_embedding_model()
        llm_client = dependencies.get_llm_client()

        assert embed_model.provider_name == "ollama"
        assert llm_client.provider_name == "openai"
    finally:
        settings.embeddings_provider = original["embeddings_provider"]
        settings.llm_provider = original["llm_provider"]
        settings.openai_api_key = original["openai_api_key"]
        dependencies.get_embedding_model.cache_clear()
        dependencies.get_llm_client.cache_clear()


def test_openai_provider_requires_api_key() -> None:
    original = {
        "embeddings_provider": settings.embeddings_provider,
        "llm_provider": settings.llm_provider,
        "openai_api_key": settings.openai_api_key,
    }
    try:
        settings.embeddings_provider = "openai"
        settings.llm_provider = "openai"
        settings.openai_api_key = None
        dependencies.get_embedding_model.cache_clear()
        dependencies.get_llm_client.cache_clear()

        with pytest.raises(ValueError):
            dependencies.get_embedding_model()
        with pytest.raises(ValueError):
            dependencies.get_llm_client()
    finally:
        settings.embeddings_provider = original["embeddings_provider"]
        settings.llm_provider = original["llm_provider"]
        settings.openai_api_key = original["openai_api_key"]
        dependencies.get_embedding_model.cache_clear()
        dependencies.get_llm_client.cache_clear()
