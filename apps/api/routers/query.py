import time

from fastapi import APIRouter, Depends

from apps.api.dependencies import get_telemetry_client, get_vector_store
from apps.api.schemas.query import QueryRequest, QueryResponse
from src.tfmkg.adapters.telemetry import PostgresTelemetryClient
from src.tfmkg.domain.ports.vector_store import VectorStorePort

router = APIRouter(prefix="", tags=["query"])


def _stub_embedding(question: str) -> list[float]:
    # Placeholder embedding for milestone routing checks.
    seed = float((sum(ord(char) for char in question) % 1000) / 1000)
    return [seed] + [0.0] * 767


def _mode_to_source_type(mode: str) -> str | None:
    if mode == "text":
        return "doc_text"
    if mode == "table":
        return "table_row"
    if mode == "hybrid":
        return "kg_text"
    return None


@router.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    vector_store: VectorStorePort = Depends(get_vector_store),
    telemetry_client: PostgresTelemetryClient = Depends(get_telemetry_client),
) -> QueryResponse:
    start = time.perf_counter()

    source_type = _mode_to_source_type(req.mode)
    if req.mode == "kg":
        response = QueryResponse(
            answer="KG mode is not implemented yet.",
            mode=req.mode,
            top_k=req.top_k,
            citations=[],
            debug={"note": "kg mode not implemented yet"},
        )
        telemetry_client.log_query_run(
            question=req.question,
            mode=req.mode,
            top_k=req.top_k,
            retrieved_chunk_ids=[],
            evidence_pack={},
            answer=response.answer,
            latency_ms=int((time.perf_counter() - start) * 1000),
            dataset_version="dev",
        )
        return response

    retrieved = vector_store.similarity_search(
        embedding=_stub_embedding(req.question),
        top_k=req.top_k,
        filters={"source_type": source_type},
    )
    chunk_ids = [item["chunk_id"] for item in retrieved if "chunk_id" in item]

    response = QueryResponse(
        answer=f"(stub) Retrieved {len(chunk_ids)} chunk(s) for: {req.question}",
        mode=req.mode,
        top_k=req.top_k,
        citations=[],
        debug={"retrieved_chunk_ids": chunk_ids},
    )
    telemetry_client.log_query_run(
        question=req.question,
        mode=req.mode,
        top_k=req.top_k,
        retrieved_chunk_ids=chunk_ids,
        evidence_pack={},
        answer=response.answer,
        latency_ms=int((time.perf_counter() - start) * 1000),
        dataset_version="dev",
    )
    return response
