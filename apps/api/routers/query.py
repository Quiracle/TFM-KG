from fastapi import APIRouter
from apps.api.schemas.query import QueryRequest, QueryResponse

router = APIRouter(prefix="", tags=["query"])

@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    # Stub implementation: replace with real retrieval + LLM generation.
    # Keeping response shape stable helps you plug in RAG later without changing UI.
    return QueryResponse(
        answer=f"(stub) You asked: {req.question}",
        mode=req.mode,
        top_k=req.top_k,
        citations=[],
        debug={
            "note": "RAG pipeline not wired yet. Implement services/rag/rag_pipeline.py and call it here.",
        },
    )
