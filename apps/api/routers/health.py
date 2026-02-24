from fastapi import APIRouter, Depends, HTTPException

from apps.api.dependencies import get_db_client
from src.tfmkg.adapters.db import PsycopgDBClient

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/health/db")
def health_db(db_client: PsycopgDBClient = Depends(get_db_client)):
    try:
        db_client.ping()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Database health check failed: {exc}",
        ) from None
    return {"db": "ok"}
