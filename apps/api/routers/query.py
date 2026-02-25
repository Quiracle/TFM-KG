import time
import re

from fastapi import APIRouter, Depends, HTTPException

from apps.api.dependencies import (
    get_embedding_model,
    get_fuseki_client,
    get_llm_client,
    get_telemetry_client,
    get_vector_store,
)
from apps.api.schemas.query import Citation, QueryRequest, QueryResponse
from src.tfmkg.adapters.telemetry import PostgresTelemetryClient
from src.tfmkg.adapters.triplestore import FusekiClient
from src.tfmkg.core.config import settings
from src.tfmkg.core.evidence import build_evidence_pack, should_abstain
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


def _build_evidence_text(evidence_facts: list[str]) -> str:
    return "\n".join(f"- {fact}" for fact in evidence_facts)


def _build_evidence_prompt(question: str, evidence_text: str) -> list[LLMMessage]:
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
            content=f"Question: {question}\n\nEvidence facts:\n{evidence_text}",
        ),
    ]


def _extract_keywords(question: str, limit: int = 3) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "this",
    }
    words = re.findall(r"[A-Za-z0-9_]+", question.lower())
    keywords: list[str] = []
    for word in words:
        if len(word) < 3 or word in stopwords:
            continue
        if word not in keywords:
            keywords.append(word)
        if len(keywords) >= limit:
            break
    return keywords


def _kg_search_query(keywords: list[str], limit: int) -> str:
    if not keywords:
        return f"SELECT ?s ?p ?o WHERE {{ ?s ?p ?o }} LIMIT {limit}"
    filters = " || ".join(f'CONTAINS(LCASE(STR(?o)), "{token}")' for token in keywords)
    return f"""
        SELECT ?s ?p ?o
        WHERE {{
            ?s ?p ?o .
            FILTER({filters})
        }}
        LIMIT {limit}
    """


def _kg_connected_query(seed_uris: list[str], limit: int) -> str:
    if not seed_uris:
        return f"SELECT ?s ?p ?o WHERE {{ ?s ?p ?o }} LIMIT {limit}"
    values = " ".join(f"<{uri}>" for uri in seed_uris)
    return f"""
        SELECT ?s ?p ?o
        WHERE {{
            {{
                VALUES ?seed {{ {values} }}
                ?seed ?p ?o .
                BIND(?seed AS ?s)
            }}
            UNION
            {{
                VALUES ?seed {{ {values} }}
                ?s ?p ?seed .
                BIND(?seed AS ?o)
            }}
        }}
        LIMIT {limit}
    """


def _kg_bindings_to_rows(bindings: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for idx, row in enumerate(bindings):
        subject = row.get("s", {}).get("value", "")
        predicate = row.get("p", {}).get("value", "")
        obj = row.get("o", {}).get("value", "")
        if not subject and not predicate and not obj:
            continue
        rows.append(
            {
                "chunk_id": f"kg-triple-{idx}",
                "source_type": "kg_triple",
                "source_ref": subject or predicate,
                "text": f"{subject} {predicate} {obj}".strip(),
                "score": None,
                "triple": {"s": subject, "p": predicate, "o": obj},
            }
        )
    return rows


def _fetch_kg_rows(fuseki_client: FusekiClient, question: str, top_k: int) -> tuple[list[dict], dict]:
    keywords = _extract_keywords(question)
    primary_query = _kg_search_query(keywords, max(top_k * 3, 25))
    primary_payload = fuseki_client.sparql(primary_query)
    primary_bindings = primary_payload.get("results", {}).get("bindings", [])

    seed_uris: list[str] = []
    for row in primary_bindings:
        value = row.get("s", {}).get("value")
        if value and value.startswith("http://"):
            seed_uris.append(value)
        if value and value.startswith("https://"):
            seed_uris.append(value)
        if len(seed_uris) >= top_k:
            break

    connected_query = _kg_connected_query(seed_uris, max(top_k * 4, 25))
    connected_payload = fuseki_client.sparql(connected_query)
    connected_bindings = connected_payload.get("results", {}).get("bindings", [])

    merged: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for row in primary_bindings + connected_bindings:
        triple_key = (
            row.get("s", {}).get("value", ""),
            row.get("p", {}).get("value", ""),
            row.get("o", {}).get("value", ""),
        )
        if triple_key in seen:
            continue
        seen.add(triple_key)
        merged.append(row)
        if len(merged) >= max(top_k * 5, 25):
            break

    return _kg_bindings_to_rows(merged), {
        "keywords": keywords,
        "primary_query": " ".join(primary_query.split()),
        "connected_query": " ".join(connected_query.split()),
        "primary_count": len(primary_bindings),
        "connected_count": len(connected_bindings),
    }


@router.get("/stats/index")
def index_stats(
    vector_store: VectorStorePort = Depends(get_vector_store),
    embedding_model: EmbeddingModelPort = Depends(get_embedding_model),
    llm_client: LLMClientPort = Depends(get_llm_client),
) -> dict:
    try:
        store_stats = vector_store.index_stats()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Index stats failed: {exc}") from None

    return {
        "counts_by_source_type": store_stats.get("counts_by_source_type", []),
        "counts_by_dataset_version": store_stats.get("counts_by_dataset_version", []),
        "embedding_dim": {
            "configured": settings.embedding_dimension,
            "inferred": store_stats.get("embedding_dim_inferred"),
        },
        "providers": {
            "llm_provider": llm_client.provider_name,
            "llm_model": llm_client.model_name,
            "embeddings_provider": embedding_model.provider_name,
            "embed_model": embedding_model.model_name,
        },
    }


@router.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    vector_store: VectorStorePort = Depends(get_vector_store),
    fuseki_client: FusekiClient = Depends(get_fuseki_client),
    embedding_model: EmbeddingModelPort = Depends(get_embedding_model),
    llm_client: LLMClientPort = Depends(get_llm_client),
    telemetry_client: PostgresTelemetryClient = Depends(get_telemetry_client),
) -> QueryResponse:
    start = time.perf_counter()
    dataset_version = req.dataset_version

    source_type = _mode_to_source_type(req.mode)
    kg_debug: dict | None = None
    if req.mode == "kg":
        try:
            retrieved, kg_debug = _fetch_kg_rows(fuseki_client, req.question, req.top_k)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"KG retrieval failed: {exc}") from None
    else:
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
            filters={"source_type": source_type, "dataset_version": dataset_version},
        )

    chunk_ids = [item["chunk_id"] for item in retrieved if "chunk_id" in item]
    evidence_pack = build_evidence_pack(retrieved)
    evidence_text = _build_evidence_text(evidence_pack["facts"])
    citations = [Citation.model_validate(item) for item in evidence_pack["citations"]]

    llm_result = None
    abstained, abstain_reason = should_abstain(
        question=req.question,
        retrieval_hits=retrieved,
        evidence_text=evidence_text,
    )
    if abstained:
        answer = _abstain_answer()
    else:
        llm_prompt = _build_evidence_prompt(req.question, evidence_text)
        try:
            llm_result = llm_client.generate(messages=llm_prompt, temperature=0.2, max_output_tokens=300)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"LLM provider error: {exc}") from None
        answer = llm_result.text or _abstain_answer()
        if answer.strip() == _abstain_answer() or not citations:
            abstained = True
            answer = _abstain_answer()
            citations = []
            if abstain_reason is None and not citations:
                abstain_reason = "missing_citations"

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

    retrieval_hits = [
        {
            "chunk_id": item.get("chunk_id"),
            "source_type": item.get("source_type"),
            "source_ref": item.get("source_ref"),
            "score": item.get("distance", item.get("score")),
        }
        for item in retrieved
    ]
    providers = {
        "llm_provider": llm_client.provider_name,
        "llm_model": llm_client.model_name,
        "embeddings_provider": embedding_model.provider_name,
        "embed_model": embedding_model.model_name,
    }
    debug_payload = {"retrieved_chunk_ids": chunk_ids, "prompt_meta": prompt_meta}
    if req.debug:
        debug_payload.update(
            {
                "retrieval_hits": retrieval_hits,
                "evidence_text": evidence_text,
                "dataset_version": dataset_version,
                "providers": providers,
                "abstain_reason": abstain_reason,
            }
        )
        if req.mode == "kg":
            debug_payload["triples"] = [item.get("triple", {}) for item in retrieved]
            debug_payload["kg_queries"] = kg_debug or {}
            debug_payload["evidence_pack"] = evidence_pack

    response = QueryResponse(
        answer=answer,
        mode=req.mode,
        top_k=req.top_k,
        abstained=abstained,
        citations=citations,
        debug=debug_payload,
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
        dataset_version=dataset_version,
    )
    return response
