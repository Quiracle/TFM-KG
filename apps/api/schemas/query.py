from typing import Any, Literal
from pydantic import BaseModel, Field

RetrievalMode = Literal["kg", "text", "table", "hybrid"]

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    mode: RetrievalMode = Field("hybrid", description="Retrieval mode for benchmarking")
    top_k: int = Field(8, ge=1, le=50, description="How many items to retrieve")

class Citation(BaseModel):
    source_type: str
    source_ref: str
    chunk_id: str | None = None
    span: str | None = None

class QueryResponse(BaseModel):
    answer: str
    mode: RetrievalMode
    top_k: int
    citations: list[Citation] = []
    debug: dict[str, Any] = {}
