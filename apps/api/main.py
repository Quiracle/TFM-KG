from fastapi import FastAPI
from apps.api.routers.health import router as health_router
from apps.api.routers.kg import router as kg_router
from apps.api.routers.query import router as query_router

app = FastAPI(
    title="TFM KG + RAG API",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(kg_router)
app.include_router(query_router)
