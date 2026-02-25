from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.tfmkg.adapters.embeddings import OllamaEmbeddingsClient, OpenAIEmbeddingsClient
from src.tfmkg.adapters.vectorstore.pgvector import PgVectorRepository
from src.tfmkg.core.config import settings
from src.tfmkg.domain.ports.embeddings import EmbeddingModelPort

LOGGER = logging.getLogger("index_kg")


@dataclass(frozen=True)
class KGConfig:
    subjects_limit: int = 200
    triples_limit: int = 50
    batch_size: int = 16
    dataset_version: str = "dev"


def chunk_id_for_uri(uri: str) -> str:
    digest = hashlib.sha1(uri.encode("utf-8")).hexdigest()[:16]
    return f"kg:{digest}"


def fallback_label_from_uri(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rstrip("/").rsplit("/", 1)[-1]


def build_entity_card(uri: str, label: str, triples: list[tuple[str, str]]) -> str:
    sorted_triples = sorted(triples, key=lambda item: (item[0], item[1]))
    lines = [f"Entity URI: {uri}", f"Label: {label}", "Facts:"]
    for predicate, obj in sorted_triples:
        lines.append(f"- {predicate}: {obj}")
    return "\n".join(lines)


def _sparql_query(query: str) -> list[dict[str, Any]]:
    query_url = f"{settings.fuseki_url.rstrip('/')}/{settings.fuseki_dataset}/query"
    body = urlencode({"query": query}).encode("utf-8")
    request = Request(
        query_url,
        data=body,
        headers={
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urlopen(request, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("results", {}).get("bindings", [])


def _binding_value(binding: dict[str, Any], key: str) -> str:
    value = binding.get(key, {})
    if not isinstance(value, dict):
        return ""
    return str(value.get("value", ""))


def _get_embedding_model() -> EmbeddingModelPort:
    if settings.embeddings_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDINGS_PROVIDER=openai")
        return OpenAIEmbeddingsClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_embed_model,
            timeout_s=settings.ollama_timeout_s,
        )
    return OllamaEmbeddingsClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
        timeout_s=settings.ollama_timeout_s,
    )


def _fetch_subject_uris(limit: int) -> list[str]:
    query = f"""
        SELECT DISTINCT ?s
        WHERE {{
          ?s ?p ?o .
          FILTER(isIRI(?s))
        }}
        ORDER BY ?s
        LIMIT {limit}
    """
    rows = _sparql_query(query)
    return [_binding_value(row, "s") for row in rows if _binding_value(row, "s")]


def _fetch_label(uri: str) -> str:
    query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?label
        WHERE {{
          <{uri}> rdfs:label ?label .
          FILTER(lang(?label) = "" || langMatches(lang(?label), "en"))
        }}
        ORDER BY ?label
        LIMIT 1
    """
    rows = _sparql_query(query)
    if not rows:
        return fallback_label_from_uri(uri)
    label = _binding_value(rows[0], "label")
    return label or fallback_label_from_uri(uri)


def _fetch_triples(uri: str, limit: int) -> list[tuple[str, str]]:
    query = f"""
        SELECT ?p ?o
        WHERE {{
          <{uri}> ?p ?o .
        }}
        ORDER BY ?p ?o
        LIMIT {limit}
    """
    rows = _sparql_query(query)
    triples: list[tuple[str, str]] = []
    for row in rows:
        predicate = _binding_value(row, "p")
        obj = _binding_value(row, "o")
        if predicate and obj:
            triples.append((predicate, obj))
    return triples


def _build_chunks(subjects: list[str], cfg: KGConfig) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for index, uri in enumerate(subjects, start=1):
        label = _fetch_label(uri)
        triples = _fetch_triples(uri, cfg.triples_limit)
        text = build_entity_card(uri, label, triples)
        chunks.append(
            {
                "chunk_id": chunk_id_for_uri(uri),
                "source_type": "kg_text",
                "source_ref": uri,
                "dataset_version": cfg.dataset_version,
                "text": text,
                "metadata": {
                    "label": label,
                    "entity_uri": uri,
                    "triple_count": len(triples),
                },
            }
        )
        if index % 25 == 0:
            LOGGER.info("Prepared %s/%s entity cards", index, len(subjects))
    return chunks


def run(cfg: KGConfig) -> None:
    LOGGER.info(
        "Starting KG indexing | fuseki_dataset=%s | limit=%s | triples_limit=%s | batch_size=%s",
        settings.fuseki_dataset,
        cfg.subjects_limit,
        cfg.triples_limit,
        cfg.batch_size,
    )
    subjects = _fetch_subject_uris(cfg.subjects_limit)
    LOGGER.info("Found %s distinct subject URIs", len(subjects))
    if not subjects:
        LOGGER.warning("No subjects found. Nothing to index.")
        return

    chunks = _build_chunks(subjects, cfg)
    embedding_model = _get_embedding_model()
    vector_repo = PgVectorRepository(settings.database_url)
    total_upserted = 0

    for start in range(0, len(chunks), cfg.batch_size):
        batch = chunks[start : start + cfg.batch_size]
        texts = [item["text"] for item in batch]
        embeddings = embedding_model.embed_texts(texts)
        if len(embeddings) != len(batch):
            raise RuntimeError("Embedding provider returned unexpected number of vectors.")

        for chunk, embedding in zip(batch, embeddings):
            if len(embedding) != settings.embedding_dimension:
                raise RuntimeError(
                    f"Embedding dimension mismatch for {chunk['chunk_id']}: "
                    f"got {len(embedding)}, expected {settings.embedding_dimension}."
                )
            chunk["embedding"] = embedding
            chunk["metadata"]["embedding_provider"] = embedding_model.provider_name
            chunk["metadata"]["embedding_model"] = embedding_model.model_name

        vector_repo.upsert_chunks(batch)
        total_upserted += len(batch)
        LOGGER.info("Upserted %s/%s KG chunks", total_upserted, len(chunks))

    LOGGER.info("KG indexing completed. Upserted %s chunks.", total_upserted)


def parse_args() -> KGConfig:
    parser = argparse.ArgumentParser(description="Index KG entities from Fuseki into pgvector chunks.")
    parser.add_argument("--limit", type=int, default=200, help="Max distinct subject URIs to index.")
    parser.add_argument("--triples-per-entity", type=int, default=50, help="Triples per entity card.")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding/upsert batch size.")
    parser.add_argument("--dataset-version", type=str, default="dev", help="Dataset version for chunks.")
    args = parser.parse_args()
    return KGConfig(
        subjects_limit=args.limit,
        triples_limit=args.triples_per_entity,
        batch_size=args.batch_size,
        dataset_version=args.dataset_version,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
