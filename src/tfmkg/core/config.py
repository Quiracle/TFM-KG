from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Connectivity
    database_url: str = "postgresql+psycopg://tfmkg:tfmkg@localhost:5432/tfmkg"
    fuseki_url: str = "http://localhost:3030"
    fuseki_dataset: str = "kg"

    # App
    log_level: str = "INFO"

settings = Settings()
