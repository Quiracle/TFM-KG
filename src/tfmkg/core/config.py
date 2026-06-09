from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Connectivity
    database_url: str = "postgresql+psycopg://tfmkg:tfmkg@localhost:5432/tfmkg"
    fuseki_url: str = "http://localhost:3030"
    fuseki_dataset: str = "kg"

    # Providers
    embeddings_provider: str = "ollama"
    llm_provider: str = "ollama"
    judge_provider: str = "anthropic"
    judge_model: str = "claude-haiku-4-5-20251001"

    # Ollama
    ollama_base_url: str = "http://ollama:11434"
    ollama_embed_model: str = "embeddinggemma"
    ollama_llm_model: str = "mistral:7b"
    ollama_timeout_s: int = 120
    ollama_stream: bool = False

    # OpenAI
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_embed_model: str = "text-embedding-3-small"
    openai_llm_model: str = "gpt-4.1-mini"

    # Anthropic
    anthropic_api_key: str | None = None
    anthropic_llm_model: str = "claude-sonnet-4-5"

    # Gemini
    gemini_api_key: str | None = None
    gemini_llm_model: str = "gemini-2.5-flash"

    # Embedding schema contract
    embedding_dimension: int = 768

    # App
    log_level: str = "INFO"

settings = Settings()
