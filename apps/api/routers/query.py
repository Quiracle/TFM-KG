import time

from fastapi import APIRouter, Depends, HTTPException

from apps.api.dependencies import (
    get_embedding_model,
    get_llm_client,
    get_telemetry_client,
    get_vector_store,
)
from apps.api.schemas.query import Citation, QueryRequest, QueryResponse
from src.tfmkg.adapters.telemetry import PostgresTelemetryClient
from src.tfmkg.core.config import settings
from src.tfmkg.core.evidence import build_evidence_pack
from src.tfmkg.domain.ports.embeddings import EmbeddingModelPort
from src.tfmkg.domain.ports.llm import LLMClientPort, LLMMessage
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


def _abstain_answer() -> str:
    return "I don't have enough information in the provided sources to answer that."


def _build_evidence_prompt(question: str, evidence_facts: list[str]) -> list[LLMMessage]:
    evidence_lines = "\n".join(f"- {fact}" for fact in evidence_facts)
    return [
        LLMMessage(
            role="system",
            content=(
                "Answer only from the provided evidence facts. "
                "Do not invent facts. Keep the answer short and factual. "
                "If evidence is insufficient, answer exactly: "
                "\"I don't have enough information in the provided sources to answer that.\""
            ),
        ),
        LLMMessage(
            role="user",
            content=f"Question: {question}\n\nEvidence facts:\n{evidence_lines}",
        ),
    ]


@router.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    vector_store: VectorStorePort = Depends(get_vector_store),
    embedding_model: EmbeddingModelPort = Depends(get_embedding_model),
    llm_client: LLMClientPort = Depends(get_llm_client),
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
            abstained=False,
            latency_ms=int((time.perf_counter() - start) * 1000),
            dataset_version="dev",
        )
        return response

    try:
        query_embedding = embedding_model.embed_query(req.question)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Embedding provider error: {exc}") from None

    if len(query_embedding) != settings.embedding_dimension:
        raise HTTPException(
            status_code=500,
            detail=(
                "Embedding dimension mismatch: "
                f"got {len(query_embedding)}, expected {settings.embedding_dimension}."
            ),
        )

    retrieved = vector_store.similarity_search(
        embedding=query_embedding,
        top_k=req.top_k,
        filters={"source_type": source_type},
    )
    chunk_ids = [item["chunk_id"] for item in retrieved if "chunk_id" in item]
    evidence_pack = build_evidence_pack(retrieved)
    citations = [Citation.model_validate(item) for item in evidence_pack["citations"]]
    llm_prompt = _build_evidence_prompt(req.question, evidence_pack["facts"])

    llm_result = None
    abstained = len(evidence_pack["facts"]) == 0
    if abstained:
        answer = _abstain_answer()
    else:
        try:
            llm_result = llm_client.generate(messages=llm_prompt, temperature=0.2, max_output_tokens=300)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"LLM provider error: {exc}") from None
        answer = llm_result.text or _abstain_answer()
        if answer.strip() == _abstain_answer():
            abstained = True

    prompt_meta = {
        "llm_provider": llm_client.provider_name,
        "llm_model": llm_client.model_name,
        "embeddings_provider": embedding_model.provider_name,
        "embed_model": embedding_model.model_name,
        "prompt_tokens": None if llm_result is None else llm_result.prompt_tokens,
        "completion_tokens": None if llm_result is None else llm_result.completion_tokens,
        "abstained": abstained,
    }
    evidence_pack["prompt_meta"] = prompt_meta

    response = QueryResponse(
        answer=answer,
        mode=req.mode,
        top_k=req.top_k,
        citations=citations,
        debug={"retrieved_chunk_ids": chunk_ids, "prompt_meta": prompt_meta},
    )
    telemetry_client.log_query_run(
        question=req.question,
        mode=req.mode,
        top_k=req.top_k,
        retrieved_chunk_ids=chunk_ids,
        evidence_pack=evidence_pack,
        answer=response.answer,
        abstained=abstained,
        latency_ms=int((time.perf_counter() - start) * 1000),
        dataset_version="dev",
    )
    return response
