from fastapi import APIRouter, Depends, HTTPException

from apps.api.dependencies import get_fuseki_client
from src.tfmkg.adapters.triplestore import FusekiClient

router = APIRouter(prefix="/kg", tags=["kg"])


@router.get("/ping")
def kg_ping(fuseki_client: FusekiClient = Depends(get_fuseki_client)):
    try:
        results_count = fuseki_client.ping()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"KG ping failed: {exc}",
        ) from None

    return {"status": "ok", "results_count": results_count}
