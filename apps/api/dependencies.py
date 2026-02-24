from functools import lru_cache

from src.tfmkg.adapters.db import PsycopgDBClient
from src.tfmkg.core.config import settings


@lru_cache(maxsize=1)
def get_db_client() -> PsycopgDBClient:
    return PsycopgDBClient(settings.database_url)
