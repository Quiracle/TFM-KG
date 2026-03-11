from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPKGSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mcp_server_name: str = "tfmkg-kg-server"
    fuseki_url: str = "http://localhost:3030"
    fuseki_dataset: str = "kg"
    mcp_kg_timeout_ms: int = 10000
    mcp_kg_max_rows: int = 200

    @property
    def fuseki_query_url(self) -> str:
        return f"{self.fuseki_url.rstrip('/')}/{self.fuseki_dataset}/query"


settings = MCPKGSettings()
