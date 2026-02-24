from fastapi import APIRouter, Depends

from apps.api.dependencies import get_vector_store
from apps.api.schemas.query import QueryRequest, QueryResponse
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
) -> QueryResponse:
    source_type = _mode_to_source_type(req.mode)
    if req.mode == "kg":
        return QueryResponse(
            answer="KG mode is not implemented yet.",
            mode=req.mode,
            top_k=req.top_k,
            citations=[],
            debug={"note": "kg mode not implemented yet"},
        )

    retrieved = vector_store.similarity_search(
        embedding=_stub_embedding(req.question),
        top_k=req.top_k,
        filters={"source_type": source_type},
    )
    chunk_ids = [item["chunk_id"] for item in retrieved if "chunk_id" in item]

    return QueryResponse(
        answer=f"(stub) Retrieved {len(chunk_ids)} chunk(s) for: {req.question}",
        mode=req.mode,
        top_k=req.top_k,
        citations=[],
        debug={"retrieved_chunk_ids": chunk_ids},
    )
