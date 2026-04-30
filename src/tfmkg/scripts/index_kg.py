from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import psycopg

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.tfmkg.adapters.embeddings import OllamaEmbeddingsClient, OpenAIEmbeddingsClient
from src.tfmkg.adapters.db.psycopg_client import normalize_psycopg_dsn
from src.tfmkg.adapters.vectorstore.pgvector import PgVectorRepository
from src.tfmkg.core.config import settings
from src.tfmkg.domain.ports.embeddings import EmbeddingModelPort

LOGGER = logging.getLogger("index_kg")

AM_NS = "http://purl.org/collections/nl/am/"

SPARQL_PREFIXES = """
    PREFIX am:    <http://purl.org/collections/nl/am/>
    PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX skos:  <http://www.w3.org/2004/02/skos/core#>
    PREFIX ore:   <http://www.openarchives.org/ore/terms/>
"""

PROXY_SIMPLE_FIELDS: tuple[tuple[str, str, str, int], ...] = (
    ("title", "Title", "am:title", 12),
    ("object_number", "Object number", "am:objectNumber", 4),
    ("maker", "Maker", "", 8),
    ("priref", "Priref", "am:priref", 1),
    ("production_date_start", "Production date start", "am:productionDateStart", 2),
    ("production_date_end", "Production date end", "am:productionDateEnd", 2),
    ("production_period", "Production period", "am:productionPeriod", 4),
    ("production_place", "Production place", "am:productionPlace", 4),
    ("object_name", "Object type", "am:objectName", 8),
    ("object_category", "Object category", "am:objectCategory", 8),
    ("collection", "Collection", "am:collection", 8),
    ("material", "Material", "am:material", 10),
    ("technique", "Technique", "am:technique", 10),
    ("content_subject", "Subject", "am:contentSubject", 10),
    ("content_motif", "Motif", "am:contentMotifGeneral", 10),
    ("content_person", "Depicted person", "am:contentPersonName", 8),
    ("association_person", "Associated person", "am:associationPerson", 8),
    ("association_subject", "Associated subject", "am:associationSubject", 8),
    ("acquisition_method", "Acquisition method", "am:acquisitionMethod", 4),
    ("acquisition_date", "Acquisition date", "am:acquisitionDate", 4),
    ("credit_line", "Credit line", "am:creditLine", 4),
    ("title_count", "Distinct titles recorded", "", 1),
    ("maker_count", "Distinct makers recorded", "", 1),
    ("related_object_reference_count", "relatedObjectReference count", "", 1),
)

PERSON_SIMPLE_FIELDS: tuple[tuple[str, str, str, int], ...] = (
    ("name", "Name", "am:name", 8),
    ("priref", "Priref", "am:priref", 1),
    ("equivalent_name", "Equivalent name", "am:equivalentName", 12),
    ("birth_date_start", "Birth date start", "am:birthDateStart", 2),
    ("birth_place", "Birth place", "am:birthPlace", 4),
    ("death_date_start", "Death date start", "am:deathDateStart", 2),
    ("death_place", "Death place", "am:deathPlace", 4),
    ("nationality", "Nationality", "am:nationality", 6),
    ("occupation", "Occupation", "am:occupation", 8),
    ("biography", "Biography", "am:biography", 2),
    ("maker_works_count", "Number of proxies linked as maker", "", 1),
)

DIMENSION_KIND_ALIASES = {
    "hoogte": "height",
    "breedte": "width",
    "diepte": "depth",
    "diameter": "diameter",
    "lengte": "length",
}

SECTION_LIMITS = {
    "titles": 12,
    "makers": 8,
    "dimensions": 12,
    "locations": 6,
    "related_artworks": 10,
    "sample_works": 10,
}


@dataclass(frozen=True)
class KGConfig:
    subjects_limit: int = 200
    triples_limit: int = 50
    batch_size: int = 16
    dataset_version: str = "dev"
    skip_existing: bool = False
    embedding_retries: int = 5


def chunk_id_for_uri(uri: str) -> str:
    digest = hashlib.sha1(uri.encode("utf-8")).hexdigest()[:16]
    return f"kg:{digest}"


def fallback_label_from_uri(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rstrip("/").rsplit("/", 1)[-1]


def _clean_text(value: str, max_chars: int = 700) -> str:
    clean = " ".join(value.split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        clean = _clean_text(value)
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(clean)
    return deduped


def _values_for_field(facts: dict[str, list[str]], field: str, limit: int | None = None) -> list[str]:
    values = _dedupe(facts.get(field, []))
    if limit is None:
        return values
    return values[:limit]


def _join_values(values: list[str], max_items: int | None = None) -> str:
    selected = values if max_items is None else values[:max_items]
    return "; ".join(selected)


def _short_uri(value: str) -> str:
    if value.startswith(AM_NS):
        return "am:" + value.removeprefix(AM_NS)
    return fallback_label_from_uri(value)


def build_entity_card(
    uri: str,
    label: str,
    triples: list[tuple[str, str]],
    *,
    entity_type: str = "kg_entity",
    facts: dict[str, list[str]] | None = None,
    sections: dict[str, list[str]] | None = None,
) -> str:
    clean_label = _clean_text(label) or fallback_label_from_uri(uri)
    lines = [
        f"Entity type: {entity_type}",
        f"Entity URI: {uri}",
        f"Entity short URI: {_short_uri(uri)}",
        f"Label: {clean_label}",
    ]

    if facts:
        summary = _build_summary_line(entity_type, clean_label, facts)
        if summary:
            lines.append(f"Summary: {summary}")

        lines.append("Key facts:")
        for field_name, display_name, _, limit in _fields_for_entity_type(entity_type):
            values = _values_for_field(facts, field_name, limit)
            if values:
                lines.append(f"- {display_name}: {_join_values(values)}")

    if sections:
        for section_name, values in sections.items():
            clean_values = _dedupe(values)
            if not clean_values:
                continue
            lines.append(f"{section_name}:")
            for value in clean_values:
                lines.append(f"- {value}")

    if not facts and not sections:
        sorted_triples = sorted(triples, key=lambda item: (item[0], item[1]))
        lines.append("Facts:")
        for predicate, obj in sorted_triples:
            clean_obj = _clean_text(obj)
            if clean_obj:
                lines.append(f"- {predicate}: {clean_obj}")
    return "\n".join(lines)


def _fields_for_entity_type(entity_type: str) -> tuple[tuple[str, str, str, int], ...]:
    if entity_type == "person":
        return PERSON_SIMPLE_FIELDS
    if entity_type == "artwork_proxy":
        return PROXY_SIMPLE_FIELDS
    return ()


def _build_summary_line(entity_type: str, label: str, facts: dict[str, list[str]]) -> str:
    if entity_type == "artwork_proxy":
        titles = _values_for_field(facts, "title", 3)
        object_numbers = _values_for_field(facts, "object_number", 2)
        makers = _values_for_field(facts, "maker", 4)
        date_start = _values_for_field(facts, "production_date_start", 1)
        date_end = _values_for_field(facts, "production_date_end", 1)
        parts = [f"artwork title {titles[0] if titles else label}"]
        if object_numbers:
            parts.append(f"object number {object_numbers[0]}")
        if makers:
            parts.append(f"maker {_join_values(makers)}")
        if date_start and date_end:
            parts.append(f"production date {date_start[0]} to {date_end[0]}")
        elif date_start:
            parts.append(f"production date {date_start[0]}")
        return "; ".join(parts)

    if entity_type == "person":
        names = _values_for_field(facts, "name", 3)
        works_counts = _values_for_field(facts, "maker_works_count", 1)
        parts = [f"person or maker {names[0] if names else label}"]
        if works_counts:
            parts.append(f"number of proxies linked as maker {works_counts[0]}")
        return "; ".join(parts)

    return label


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


def _fetch_existing_kg_source_refs(dataset_version: str) -> set[str]:
    query = """
        SELECT source_ref
        FROM chunks
        WHERE source_type = 'kg_text'
          AND dataset_version = %s
    """
    dsn = normalize_psycopg_dsn(settings.database_url)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (dataset_version,))
            return {str(row[0]) for row in cur.fetchall() if row and row[0]}


def _embed_texts_with_retry(
    embedding_model: EmbeddingModelPort,
    texts: list[str],
    chunk_ids: list[str],
    *,
    max_attempts: int,
) -> list[list[float]]:
    if not texts:
        return []

    attempts = max(1, max_attempts)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            embeddings = embedding_model.embed_texts(texts)
            if len(embeddings) != len(texts):
                raise RuntimeError(
                    "Embedding provider returned unexpected number of vectors: "
                    f"got {len(embeddings)}, expected {len(texts)}."
                )
            return embeddings
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= attempts:
                break
            sleep_s = min(60.0, 2.0 * attempt)
            LOGGER.warning(
                "Embedding attempt %s/%s failed for %s texts (%s..%s): %s; retrying in %.1fs",
                attempt,
                attempts,
                len(texts),
                chunk_ids[0] if chunk_ids else "unknown",
                chunk_ids[-1] if chunk_ids else "unknown",
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)

    if len(texts) == 1:
        assert last_error is not None
        raise last_error

    split_at = len(texts) // 2
    LOGGER.warning(
        "Embedding batch of %s texts failed after %s attempts; splitting into %s + %s",
        len(texts),
        attempts,
        split_at,
        len(texts) - split_at,
    )
    return _embed_texts_with_retry(
        embedding_model,
        texts[:split_at],
        chunk_ids[:split_at],
        max_attempts=max_attempts,
    ) + _embed_texts_with_retry(
        embedding_model,
        texts[split_at:],
        chunk_ids[split_at:],
        max_attempts=max_attempts,
    )


def _limit_clause(limit: int) -> str:
    if limit <= 0:
        return ""
    return f"LIMIT {limit}"


def _values_clause(fields: tuple[tuple[str, str, str, int], ...]) -> str:
    values = [f"({predicate} \"{field_name}\")" for field_name, _, predicate, _ in fields if predicate]
    return "\n".join(values)


def _is_meaningful_value(field: str, value: str) -> bool:
    normalized = _clean_text(value).casefold()
    if not normalized:
        return False
    if field == "biography" and normalized in {"geen gegevens", "onbekend", "-"}:
        return False
    return True


def _entity_type_for_uri(uri: str) -> str:
    local = fallback_label_from_uri(uri)
    if local.startswith("proxy-"):
        return "artwork_proxy"
    if local.startswith("p-"):
        return "person"
    return "kg_entity"


def _values_uris(uris: list[str]) -> str:
    return " ".join(f"<{uri}>" for uri in uris)


def _empty_facts_for_uris(uris: list[str]) -> dict[str, dict[str, list[str]]]:
    return {uri: {} for uri in uris}


def _empty_sections_for_uris(uris: list[str]) -> dict[str, list[str]]:
    return {uri: [] for uri in uris}


def _fetch_subject_uris(limit: int) -> list[str]:
    query = f"""
        {SPARQL_PREFIXES}
        SELECT DISTINCT ?s ?rank
        WHERE {{
          {{
            ?s a ore:Proxy .
            BIND(1 AS ?rank)
          }}
          UNION
          {{
            ?s a am:Person .
            BIND(2 AS ?rank)
          }}
        }}
        ORDER BY ?rank ?s
        {_limit_clause(limit)}
    """
    rows = _sparql_query(query)
    return [_binding_value(row, "s") for row in rows if _binding_value(row, "s")]


def _fetch_label(uri: str) -> str:
    query = f"""
        {SPARQL_PREFIXES}
        SELECT ?label ?rank
        WHERE {{
          {{
            <{uri}> am:title ?label .
            BIND(1 AS ?rank)
          }}
          UNION
          {{
            <{uri}> am:name ?label .
            BIND(2 AS ?rank)
          }}
          UNION
          {{
            <{uri}> skos:prefLabel ?label .
            BIND(3 AS ?rank)
          }}
          UNION
          {{
            <{uri}> rdfs:label ?label .
            BIND(4 AS ?rank)
          }}
          FILTER(isLiteral(?label))
        }}
        ORDER BY ?rank LCASE(STR(?label))
        LIMIT 1
    """
    rows = _sparql_query(query)
    if not rows:
        return fallback_label_from_uri(uri)
    label = _binding_value(rows[0], "label")
    return label or fallback_label_from_uri(uri)


def _fetch_labels(uris: list[str]) -> dict[str, str]:
    if not uris:
        return {}

    query = f"""
        {SPARQL_PREFIXES}
        SELECT ?entity ?label ?rank
        WHERE {{
          VALUES ?entity {{ {_values_uris(uris)} }}
          {{
            ?entity am:title ?label .
            BIND(1 AS ?rank)
          }}
          UNION
          {{
            ?entity am:name ?label .
            BIND(2 AS ?rank)
          }}
          UNION
          {{
            ?entity skos:prefLabel ?label .
            BIND(3 AS ?rank)
          }}
          UNION
          {{
            ?entity rdfs:label ?label .
            BIND(4 AS ?rank)
          }}
          FILTER(isLiteral(?label))
        }}
        ORDER BY ?entity ?rank LCASE(STR(?label))
    """
    labels: dict[str, str] = {}
    for row in _sparql_query(query):
        entity = _binding_value(row, "entity")
        label = _binding_value(row, "label")
        if entity and label and entity not in labels:
            labels[entity] = label
    return {uri: labels.get(uri, fallback_label_from_uri(uri)) for uri in uris}


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


def _fetch_simple_facts_for_uris(
    uris: list[str],
    fields: tuple[tuple[str, str, str, int], ...],
) -> dict[str, dict[str, list[str]]]:
    facts_by_uri = _empty_facts_for_uris(uris)
    values_clause = _values_clause(fields)
    if not uris or not values_clause:
        return facts_by_uri

    query = f"""
        {SPARQL_PREFIXES}
        SELECT ?entity ?field ?value
        WHERE {{
          VALUES ?entity {{ {_values_uris(uris)} }}
          VALUES (?predicate ?field) {{
            {values_clause}
          }}
          ?entity ?predicate ?raw .
          OPTIONAL {{
            ?raw skos:prefLabel ?prefLabel .
            FILTER(LANG(?prefLabel) = "" || LANGMATCHES(LANG(?prefLabel), "nl") || LANGMATCHES(LANG(?prefLabel), "en"))
          }}
          OPTIONAL {{ ?raw am:name ?personName . }}
          OPTIONAL {{ ?raw rdfs:label ?rdfsLabel . }}
          OPTIONAL {{ ?raw am:title ?linkedTitle . }}
          BIND(COALESCE(?personName, ?prefLabel, ?rdfsLabel, ?linkedTitle, STR(?raw)) AS ?value)
        }}
        ORDER BY ?entity ?field LCASE(STR(?value))
    """
    for row in _sparql_query(query):
        entity = _binding_value(row, "entity")
        field = _binding_value(row, "field")
        value = _binding_value(row, "value")
        if entity in facts_by_uri and field and _is_meaningful_value(field, value):
            facts_by_uri[entity].setdefault(field, []).append(value)

    return {
        uri: {field: _dedupe(values) for field, values in facts.items()}
        for uri, facts in facts_by_uri.items()
    }


def _fetch_simple_facts(
    uri: str,
    fields: tuple[tuple[str, str, str, int], ...],
) -> dict[str, list[str]]:
    values_clause = _values_clause(fields)
    if not values_clause:
        return {}

    query = f"""
        {SPARQL_PREFIXES}
        SELECT ?field ?value
        WHERE {{
          VALUES (?predicate ?field) {{
            {values_clause}
          }}
          <{uri}> ?predicate ?raw .
          OPTIONAL {{
            ?raw skos:prefLabel ?prefLabel .
            FILTER(LANG(?prefLabel) = "" || LANGMATCHES(LANG(?prefLabel), "nl") || LANGMATCHES(LANG(?prefLabel), "en"))
          }}
          OPTIONAL {{ ?raw am:name ?personName . }}
          OPTIONAL {{ ?raw rdfs:label ?rdfsLabel . }}
          OPTIONAL {{ ?raw am:title ?linkedTitle . }}
          BIND(COALESCE(?personName, ?prefLabel, ?rdfsLabel, ?linkedTitle, STR(?raw)) AS ?value)
        }}
        ORDER BY ?field LCASE(STR(?value))
    """
    facts: dict[str, list[str]] = {}
    for row in _sparql_query(query):
        field = _binding_value(row, "field")
        value = _binding_value(row, "value")
        if field and _is_meaningful_value(field, value):
            facts.setdefault(field, []).append(value)
    return {field: _dedupe(values) for field, values in facts.items()}


def _count_from_query(query: str) -> int:
    rows = _sparql_query(query)
    if not rows:
        return 0
    raw_count = _binding_value(rows[0], "count")
    try:
        return int(raw_count)
    except ValueError:
        return 0


def _add_count_fact(
    facts: dict[str, list[str]],
    field: str,
    count: int,
    *,
    include_zero: bool = False,
) -> None:
    if count > 0 or include_zero:
        facts[field] = [str(count)]


def _fetch_proxy_counts(uri: str, facts: dict[str, list[str]]) -> None:
    _add_count_fact(facts, "title_count", len(_values_for_field(facts, "title")), include_zero=True)
    maker_count = _count_from_query(
        f"""
        {SPARQL_PREFIXES}
        SELECT (COUNT(DISTINCT ?person) AS ?count)
        WHERE {{
          <{uri}> am:maker ?makerNode .
          ?makerNode rdf:value ?person .
        }}
        """
    )
    related_count = _count_from_query(
        f"""
        {SPARQL_PREFIXES}
        SELECT (COUNT(DISTINCT ?related) AS ?count)
        WHERE {{
          <{uri}> am:relatedObjectReference ?related .
        }}
        """
    )
    _add_count_fact(facts, "maker_count", maker_count)
    _add_count_fact(facts, "related_object_reference_count", related_count)


def _fetch_proxy_makers(uri: str) -> list[str]:
    query = f"""
        {SPARQL_PREFIXES}
        SELECT ?person ?name ?qualifier
        WHERE {{
          <{uri}> am:maker ?makerNode .
          ?makerNode rdf:value ?person .
          OPTIONAL {{ ?person am:name ?name . }}
          OPTIONAL {{ ?makerNode am:creatorQualifier ?qualifier . }}
        }}
        ORDER BY LCASE(STR(?name)) ?person
    """
    makers: list[str] = []
    for row in _sparql_query(query):
        person_uri = _binding_value(row, "person")
        name = _binding_value(row, "name") or _short_uri(person_uri)
        qualifier = _binding_value(row, "qualifier")
        maker_text = name
        if qualifier:
            maker_text = f"{maker_text} ({qualifier})"
        if person_uri:
            maker_text = f"{maker_text} [{_short_uri(person_uri)}]"
        makers.append(maker_text)
    return _dedupe(makers)[: SECTION_LIMITS["makers"]]


def _fetch_proxy_makers_for_uris(uris: list[str]) -> dict[str, list[str]]:
    makers_by_uri = _empty_sections_for_uris(uris)
    if not uris:
        return makers_by_uri

    query = f"""
        {SPARQL_PREFIXES}
        SELECT ?proxy ?person ?name ?qualifier
        WHERE {{
          VALUES ?proxy {{ {_values_uris(uris)} }}
          ?proxy am:maker ?makerNode .
          ?makerNode rdf:value ?person .
          OPTIONAL {{ ?person am:name ?name . }}
          OPTIONAL {{ ?makerNode am:creatorQualifier ?qualifier . }}
        }}
        ORDER BY ?proxy LCASE(STR(?name)) ?person
    """
    for row in _sparql_query(query):
        proxy = _binding_value(row, "proxy")
        person_uri = _binding_value(row, "person")
        name = _binding_value(row, "name") or _short_uri(person_uri)
        qualifier = _binding_value(row, "qualifier")
        maker_text = name
        if qualifier:
            maker_text = f"{maker_text} ({qualifier})"
        if person_uri:
            maker_text = f"{maker_text} [{_short_uri(person_uri)}]"
        if proxy in makers_by_uri:
            makers_by_uri[proxy].append(maker_text)

    return {
        uri: _dedupe(makers)[: SECTION_LIMITS["makers"]]
        for uri, makers in makers_by_uri.items()
    }


def _dimension_kind(type_label: str, fallback_label: str) -> str:
    text = f"{type_label} {fallback_label}".casefold()
    for dutch_name, english_name in DIMENSION_KIND_ALIASES.items():
        if dutch_name in text:
            return english_name
    return type_label or fallback_label or "dimension"


def _fetch_proxy_dimensions(uri: str) -> list[str]:
    query = f"""
        {SPARQL_PREFIXES}
        SELECT ?label ?typeLabel ?value ?unit
        WHERE {{
          <{uri}> am:dimension ?dimension .
          OPTIONAL {{ ?dimension rdfs:label ?label . }}
          OPTIONAL {{
            ?dimension am:dimensionType ?typeUri .
            ?typeUri skos:prefLabel ?typeLabel .
          }}
          OPTIONAL {{ ?dimension am:dimensionValue ?value . }}
          OPTIONAL {{ ?dimension am:dimensionUnit ?unit . }}
        }}
        ORDER BY LCASE(STR(?typeLabel)) LCASE(STR(?label))
    """
    dimensions: list[str] = []
    for row in _sparql_query(query):
        label = _binding_value(row, "label")
        type_label = _binding_value(row, "typeLabel")
        value = _binding_value(row, "value")
        unit = _binding_value(row, "unit")
        kind = _dimension_kind(type_label, label)
        if value and unit:
            dimensions.append(f"Recorded {kind}: {value} {unit}; dimension label: {label or type_label}")
        elif label:
            dimensions.append(f"Dimension label: {label}")
    return _dedupe(dimensions)[: SECTION_LIMITS["dimensions"]]


def _fetch_proxy_dimensions_for_uris(uris: list[str]) -> dict[str, list[str]]:
    dimensions_by_uri = _empty_sections_for_uris(uris)
    if not uris:
        return dimensions_by_uri

    query = f"""
        {SPARQL_PREFIXES}
        SELECT ?proxy ?label ?typeLabel ?value ?unit
        WHERE {{
          VALUES ?proxy {{ {_values_uris(uris)} }}
          ?proxy am:dimension ?dimension .
          OPTIONAL {{ ?dimension rdfs:label ?label . }}
          OPTIONAL {{
            ?dimension am:dimensionType ?typeUri .
            ?typeUri skos:prefLabel ?typeLabel .
          }}
          OPTIONAL {{ ?dimension am:dimensionValue ?value . }}
          OPTIONAL {{ ?dimension am:dimensionUnit ?unit . }}
        }}
        ORDER BY ?proxy LCASE(STR(?typeLabel)) LCASE(STR(?label))
    """
    for row in _sparql_query(query):
        proxy = _binding_value(row, "proxy")
        label = _binding_value(row, "label")
        type_label = _binding_value(row, "typeLabel")
        value = _binding_value(row, "value")
        unit = _binding_value(row, "unit")
        kind = _dimension_kind(type_label, label)
        if proxy not in dimensions_by_uri:
            continue
        if value and unit:
            dimensions_by_uri[proxy].append(
                f"Recorded {kind}: {value} {unit}; dimension label: {label or type_label}"
            )
        elif label:
            dimensions_by_uri[proxy].append(f"Dimension label: {label}")

    return {
        uri: _dedupe(dimensions)[: SECTION_LIMITS["dimensions"]]
        for uri, dimensions in dimensions_by_uri.items()
    }


def _fetch_proxy_locations(uri: str) -> list[str]:
    query = f"""
        {SPARQL_PREFIXES}
        SELECT ?label ?locationLabel ?start ?end ?type
        WHERE {{
          <{uri}> am:locat ?locationNode .
          OPTIONAL {{ ?locationNode rdfs:label ?label . }}
          OPTIONAL {{
            ?locationNode am:currentLocation ?location .
            ?location skos:prefLabel ?locationLabel .
          }}
          OPTIONAL {{ ?locationNode am:currentLocationDateStart ?start . }}
          OPTIONAL {{ ?locationNode am:currentLocationDateEnd ?end . }}
          OPTIONAL {{ ?locationNode am:currentLocationType ?type . }}
        }}
        ORDER BY ASC(BOUND(?end)) DESC(STR(?start)) LCASE(STR(?label))
        LIMIT 12
    """
    locations: list[str] = []
    for row in _sparql_query(query):
        label = _binding_value(row, "label")
        location_label = _binding_value(row, "locationLabel")
        start = _binding_value(row, "start")
        end = _binding_value(row, "end")
        location_type = _binding_value(row, "type")
        if not any((label, location_label, start, end, location_type)):
            continue
        parts = []
        if label:
            parts.append(f"Location label: {label}")
        if location_label:
            parts.append(f"current location: {location_label}")
        if start:
            parts.append(f"start: {start}")
        if end:
            parts.append(f"end: {end}")
        else:
            parts.append("current record: true")
        if location_type:
            parts.append(f"type: {location_type}")
        locations.append("; ".join(parts))
    return _dedupe(locations)[: SECTION_LIMITS["locations"]]


def _location_text_from_row(row: dict[str, Any]) -> str:
    label = _binding_value(row, "label")
    location_label = _binding_value(row, "locationLabel")
    start = _binding_value(row, "start")
    end = _binding_value(row, "end")
    location_type = _binding_value(row, "type")
    if not any((label, location_label, start, end, location_type)):
        return ""
    parts = []
    if label:
        parts.append(f"Location label: {label}")
    if location_label:
        parts.append(f"current location: {location_label}")
    if start:
        parts.append(f"start: {start}")
    if end:
        parts.append(f"end: {end}")
    else:
        parts.append("current record: true")
    if location_type:
        parts.append(f"type: {location_type}")
    return "; ".join(parts)


def _fetch_proxy_locations_for_uris(uris: list[str]) -> dict[str, list[str]]:
    locations_by_uri = _empty_sections_for_uris(uris)
    if not uris:
        return locations_by_uri

    query = f"""
        {SPARQL_PREFIXES}
        SELECT ?proxy ?label ?locationLabel ?start ?end ?type
        WHERE {{
          VALUES ?proxy {{ {_values_uris(uris)} }}
          ?proxy am:locat ?locationNode .
          OPTIONAL {{ ?locationNode rdfs:label ?label . }}
          OPTIONAL {{
            ?locationNode am:currentLocation ?location .
            ?location skos:prefLabel ?locationLabel .
          }}
          OPTIONAL {{ ?locationNode am:currentLocationDateStart ?start . }}
          OPTIONAL {{ ?locationNode am:currentLocationDateEnd ?end . }}
          OPTIONAL {{ ?locationNode am:currentLocationType ?type . }}
        }}
        ORDER BY ?proxy ASC(BOUND(?end)) DESC(STR(?start)) LCASE(STR(?label))
    """
    for row in _sparql_query(query):
        proxy = _binding_value(row, "proxy")
        value = _location_text_from_row(row)
        if proxy in locations_by_uri and value:
            locations_by_uri[proxy].append(value)

    return {
        uri: _dedupe(locations)[: SECTION_LIMITS["locations"]]
        for uri, locations in locations_by_uri.items()
    }


def _fetch_proxy_related_artworks(uri: str) -> list[str]:
    query = f"""
        {SPARQL_PREFIXES}
        SELECT ?related ?title ?objectNumber
        WHERE {{
          <{uri}> am:relatedObjectReference ?related .
          OPTIONAL {{ ?related am:title ?title . }}
          OPTIONAL {{ ?related am:objectNumber ?objectNumber . }}
        }}
        ORDER BY LCASE(STR(?title)) ?related
        LIMIT 20
    """
    related_rows: list[str] = []
    for row in _sparql_query(query):
        related_uri = _binding_value(row, "related")
        title = _binding_value(row, "title") or _short_uri(related_uri)
        object_number = _binding_value(row, "objectNumber")
        value = f"relatedObjectReference title: {title}"
        if object_number:
            value = f"{value}; object number: {object_number}"
        if related_uri:
            value = f"{value}; URI: {_short_uri(related_uri)}"
        related_rows.append(value)
    return _dedupe(related_rows)[: SECTION_LIMITS["related_artworks"]]


def _fetch_proxy_related_artworks_for_uris(
    uris: list[str],
) -> tuple[dict[str, list[str]], dict[str, int]]:
    related_by_uri = _empty_sections_for_uris(uris)
    counts_by_uri = {uri: 0 for uri in uris}
    seen_related: dict[str, set[str]] = {uri: set() for uri in uris}
    if not uris:
        return related_by_uri, counts_by_uri

    query = f"""
        {SPARQL_PREFIXES}
        SELECT ?proxy ?related ?title ?objectNumber
        WHERE {{
          VALUES ?proxy {{ {_values_uris(uris)} }}
          ?proxy am:relatedObjectReference ?related .
          OPTIONAL {{ ?related am:title ?title . }}
          OPTIONAL {{ ?related am:objectNumber ?objectNumber . }}
        }}
        ORDER BY ?proxy LCASE(STR(?title)) ?related
    """
    for row in _sparql_query(query):
        proxy = _binding_value(row, "proxy")
        related_uri = _binding_value(row, "related")
        if proxy not in related_by_uri or not related_uri:
            continue

        if related_uri not in seen_related[proxy]:
            seen_related[proxy].add(related_uri)
            counts_by_uri[proxy] += 1

        title = _binding_value(row, "title") or _short_uri(related_uri)
        object_number = _binding_value(row, "objectNumber")
        value = f"relatedObjectReference title: {title}"
        if object_number:
            value = f"{value}; object number: {object_number}"
        value = f"{value}; URI: {_short_uri(related_uri)}"
        related_by_uri[proxy].append(value)

    return (
        {
            uri: _dedupe(related)[: SECTION_LIMITS["related_artworks"]]
            for uri, related in related_by_uri.items()
        },
        counts_by_uri,
    )


def _fetch_person_stats(uri: str, facts: dict[str, list[str]]) -> list[str]:
    works_count = _count_from_query(
        f"""
        {SPARQL_PREFIXES}
        SELECT (COUNT(DISTINCT ?proxy) AS ?count)
        WHERE {{
          ?proxy am:maker ?makerNode .
          ?makerNode rdf:value <{uri}> .
        }}
        """
    )
    _add_count_fact(facts, "maker_works_count", works_count)

    query = f"""
        {SPARQL_PREFIXES}
        SELECT ?proxy ?title ?objectNumber
        WHERE {{
          ?proxy am:maker ?makerNode .
          ?makerNode rdf:value <{uri}> .
          OPTIONAL {{ ?proxy am:title ?title . }}
          OPTIONAL {{ ?proxy am:objectNumber ?objectNumber . }}
        }}
        ORDER BY LCASE(STR(?title)) ?proxy
        LIMIT 20
    """
    sample_works: list[str] = []
    for row in _sparql_query(query):
        proxy_uri = _binding_value(row, "proxy")
        title = _binding_value(row, "title") or _short_uri(proxy_uri)
        object_number = _binding_value(row, "objectNumber")
        sample = f"Sample work as maker: {title}"
        if object_number:
            sample = f"{sample}; object number: {object_number}"
        if proxy_uri:
            sample = f"{sample}; URI: {_short_uri(proxy_uri)}"
        sample_works.append(sample)
    return _dedupe(sample_works)[: SECTION_LIMITS["sample_works"]]


def _fetch_person_stats_for_uris(uris: list[str]) -> tuple[dict[str, int], dict[str, list[str]]]:
    counts_by_uri = {uri: 0 for uri in uris}
    samples_by_uri = _empty_sections_for_uris(uris)
    if not uris:
        return counts_by_uri, samples_by_uri

    count_query = f"""
        {SPARQL_PREFIXES}
        SELECT ?person (COUNT(DISTINCT ?proxy) AS ?count)
        WHERE {{
          VALUES ?person {{ {_values_uris(uris)} }}
          ?proxy am:maker ?makerNode .
          ?makerNode rdf:value ?person .
        }}
        GROUP BY ?person
    """
    for row in _sparql_query(count_query):
        person = _binding_value(row, "person")
        raw_count = _binding_value(row, "count")
        if person not in counts_by_uri:
            continue
        try:
            counts_by_uri[person] = int(raw_count)
        except ValueError:
            counts_by_uri[person] = 0

    sample_query = f"""
        {SPARQL_PREFIXES}
        SELECT ?person ?proxy ?title ?objectNumber
        WHERE {{
          VALUES ?person {{ {_values_uris(uris)} }}
          ?proxy am:maker ?makerNode .
          ?makerNode rdf:value ?person .
          OPTIONAL {{ ?proxy am:title ?title . }}
          OPTIONAL {{ ?proxy am:objectNumber ?objectNumber . }}
        }}
        ORDER BY ?person LCASE(STR(?title)) ?proxy
    """
    for row in _sparql_query(sample_query):
        person = _binding_value(row, "person")
        proxy_uri = _binding_value(row, "proxy")
        if person not in samples_by_uri:
            continue
        title = _binding_value(row, "title") or _short_uri(proxy_uri)
        object_number = _binding_value(row, "objectNumber")
        sample = f"Sample work as maker: {title}"
        if object_number:
            sample = f"{sample}; object number: {object_number}"
        if proxy_uri:
            sample = f"{sample}; URI: {_short_uri(proxy_uri)}"
        samples_by_uri[person].append(sample)

    return (
        counts_by_uri,
        {
            uri: _dedupe(samples)[: SECTION_LIMITS["sample_works"]]
            for uri, samples in samples_by_uri.items()
        },
    )


def _build_proxy_card(uri: str, label: str) -> tuple[str, dict[str, Any]]:
    facts = _fetch_simple_facts(uri, PROXY_SIMPLE_FIELDS)
    makers = _fetch_proxy_makers(uri)
    if makers:
        facts["maker"] = makers
    _fetch_proxy_counts(uri, facts)

    dimensions = _fetch_proxy_dimensions(uri)
    locations = _fetch_proxy_locations(uri)
    related_artworks = _fetch_proxy_related_artworks(uri)

    sections: dict[str, list[str]] = {}
    if makers:
        sections["Makers"] = [f"Maker: {maker}" for maker in makers]
    if dimensions:
        sections["Dimensions"] = dimensions
    if locations:
        sections["Locations"] = locations
    if related_artworks:
        sections["Related artworks"] = related_artworks

    text = build_entity_card(
        uri,
        label,
        [],
        entity_type="artwork_proxy",
        facts=facts,
        sections=sections,
    )
    metadata = {
        "label": label,
        "entity_uri": uri,
        "entity_type": "artwork_proxy",
        "card_version": "kg-aware-v2",
        "fact_fields": sorted(facts.keys()),
        "section_names": sorted(sections.keys()),
    }
    return text, metadata


def _build_person_card(uri: str, label: str) -> tuple[str, dict[str, Any]]:
    facts = _fetch_simple_facts(uri, PERSON_SIMPLE_FIELDS)
    sample_works = _fetch_person_stats(uri, facts)

    sections: dict[str, list[str]] = {}
    if sample_works:
        sections["Maker work samples"] = sample_works

    text = build_entity_card(
        uri,
        label,
        [],
        entity_type="person",
        facts=facts,
        sections=sections,
    )
    metadata = {
        "label": label,
        "entity_uri": uri,
        "entity_type": "person",
        "card_version": "kg-aware-v2",
        "fact_fields": sorted(facts.keys()),
        "section_names": sorted(sections.keys()),
    }
    return text, metadata


def _build_fallback_card(uri: str, label: str, triples_limit: int) -> tuple[str, dict[str, Any]]:
    triples = _fetch_triples(uri, triples_limit)
    text = build_entity_card(uri, label, triples)
    metadata = {
        "label": label,
        "entity_uri": uri,
        "entity_type": "kg_entity",
        "card_version": "raw-triples-v1",
        "triple_count": len(triples),
    }
    return text, metadata


def _build_card_for_uri(uri: str, label: str, triples_limit: int) -> tuple[str, dict[str, Any]]:
    entity_type = _entity_type_for_uri(uri)
    if entity_type == "artwork_proxy":
        return _build_proxy_card(uri, label)
    if entity_type == "person":
        return _build_person_card(uri, label)
    return _build_fallback_card(uri, label, triples_limit)


def _metadata_for_card(
    *,
    label: str,
    uri: str,
    entity_type: str,
    facts: dict[str, list[str]],
    sections: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "label": label,
        "entity_uri": uri,
        "entity_type": entity_type,
        "card_version": "kg-aware-v2",
        "fact_fields": sorted(facts.keys()),
        "section_names": sorted(sections.keys()),
    }


def _build_proxy_card_from_prefetched(
    uri: str,
    label: str,
    facts: dict[str, list[str]],
    makers: list[str],
    dimensions: list[str],
    locations: list[str],
    related_artworks: list[str],
    related_count: int,
) -> tuple[str, dict[str, Any]]:
    facts = dict(facts)
    if makers:
        facts["maker"] = makers
        _add_count_fact(facts, "maker_count", len(makers))
    _add_count_fact(facts, "title_count", len(_values_for_field(facts, "title")), include_zero=True)
    _add_count_fact(facts, "related_object_reference_count", related_count)

    sections: dict[str, list[str]] = {}
    if makers:
        sections["Makers"] = [f"Maker: {maker}" for maker in makers]
    if dimensions:
        sections["Dimensions"] = dimensions
    if locations:
        sections["Locations"] = locations
    if related_artworks:
        sections["Related artworks"] = related_artworks

    text = build_entity_card(
        uri,
        label,
        [],
        entity_type="artwork_proxy",
        facts=facts,
        sections=sections,
    )
    metadata = _metadata_for_card(
        label=label,
        uri=uri,
        entity_type="artwork_proxy",
        facts=facts,
        sections=sections,
    )
    return text, metadata


def _build_person_card_from_prefetched(
    uri: str,
    label: str,
    facts: dict[str, list[str]],
    works_count: int,
    sample_works: list[str],
) -> tuple[str, dict[str, Any]]:
    facts = dict(facts)
    _add_count_fact(facts, "maker_works_count", works_count)

    sections: dict[str, list[str]] = {}
    if sample_works:
        sections["Maker work samples"] = sample_works

    text = build_entity_card(
        uri,
        label,
        [],
        entity_type="person",
        facts=facts,
        sections=sections,
    )
    metadata = _metadata_for_card(
        label=label,
        uri=uri,
        entity_type="person",
        facts=facts,
        sections=sections,
    )
    return text, metadata


def _build_chunks_for_batch(subjects: list[str], cfg: KGConfig) -> list[dict[str, Any]]:
    labels = _fetch_labels(subjects)
    proxy_uris = [uri for uri in subjects if _entity_type_for_uri(uri) == "artwork_proxy"]
    person_uris = [uri for uri in subjects if _entity_type_for_uri(uri) == "person"]

    proxy_facts = _fetch_simple_facts_for_uris(proxy_uris, PROXY_SIMPLE_FIELDS)
    proxy_makers = _fetch_proxy_makers_for_uris(proxy_uris)
    proxy_dimensions = _fetch_proxy_dimensions_for_uris(proxy_uris)
    proxy_locations = _fetch_proxy_locations_for_uris(proxy_uris)
    proxy_related, proxy_related_counts = _fetch_proxy_related_artworks_for_uris(proxy_uris)

    person_facts = _fetch_simple_facts_for_uris(person_uris, PERSON_SIMPLE_FIELDS)
    person_work_counts, person_sample_works = _fetch_person_stats_for_uris(person_uris)

    chunks: list[dict[str, Any]] = []
    for uri in subjects:
        label = labels.get(uri, fallback_label_from_uri(uri))
        entity_type = _entity_type_for_uri(uri)
        if entity_type == "artwork_proxy":
            text, metadata = _build_proxy_card_from_prefetched(
                uri=uri,
                label=label,
                facts=proxy_facts.get(uri, {}),
                makers=proxy_makers.get(uri, []),
                dimensions=proxy_dimensions.get(uri, []),
                locations=proxy_locations.get(uri, []),
                related_artworks=proxy_related.get(uri, []),
                related_count=proxy_related_counts.get(uri, 0),
            )
        elif entity_type == "person":
            text, metadata = _build_person_card_from_prefetched(
                uri=uri,
                label=label,
                facts=person_facts.get(uri, {}),
                works_count=person_work_counts.get(uri, 0),
                sample_works=person_sample_works.get(uri, []),
            )
        else:
            text, metadata = _build_fallback_card(uri, label, cfg.triples_limit)

        chunks.append(
            {
                "chunk_id": chunk_id_for_uri(uri),
                "source_type": "kg_text",
                "source_ref": uri,
                "dataset_version": cfg.dataset_version,
                "text": text,
                "metadata": metadata,
            }
        )

    return chunks


def _build_chunks(subjects: list[str], cfg: KGConfig) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for index, uri in enumerate(subjects, start=1):
        label = _fetch_label(uri)
        text, metadata = _build_card_for_uri(uri, label, cfg.triples_limit)
        chunks.append(
            {
                "chunk_id": chunk_id_for_uri(uri),
                "source_type": "kg_text",
                "source_ref": uri,
                "dataset_version": cfg.dataset_version,
                "text": text,
                "metadata": metadata,
            }
        )
        if index % 25 == 0:
            LOGGER.info("Prepared %s/%s entity cards", index, len(subjects))
    return chunks


def run(cfg: KGConfig) -> None:
    LOGGER.info(
        (
            "Starting KG indexing | fuseki_dataset=%s | limit=%s | triples_limit=%s | "
            "batch_size=%s | skip_existing=%s | embedding_retries=%s"
        ),
        settings.fuseki_dataset,
        cfg.subjects_limit,
        cfg.triples_limit,
        cfg.batch_size,
        cfg.skip_existing,
        cfg.embedding_retries,
    )
    subjects = _fetch_subject_uris(cfg.subjects_limit)
    LOGGER.info("Found %s distinct subject URIs", len(subjects))
    if not subjects:
        LOGGER.warning("No subjects found. Nothing to index.")
        return

    skipped_existing = 0
    if cfg.skip_existing:
        existing_source_refs = _fetch_existing_kg_source_refs(cfg.dataset_version)
        before_skip = len(subjects)
        subjects = [uri for uri in subjects if uri not in existing_source_refs]
        skipped_existing = before_skip - len(subjects)
        LOGGER.info(
            "Resume mode enabled. Skipping %s existing kg_text chunks for dataset_version=%s.",
            skipped_existing,
            cfg.dataset_version,
        )
        if not subjects:
            LOGGER.info("All KG subjects are already indexed. Nothing to do.")
            return

    embedding_model = _get_embedding_model()
    vector_repo = PgVectorRepository(settings.database_url)
    total_upserted = 0
    started_at = time.perf_counter()
    total_subjects = len(subjects)

    for start in range(0, total_subjects, cfg.batch_size):
        batch_subjects = subjects[start : start + cfg.batch_size]
        batch_number = (start // cfg.batch_size) + 1
        LOGGER.info(
            "Preparing KG batch %s | subjects=%s-%s/%s",
            batch_number,
            start + 1,
            min(start + len(batch_subjects), total_subjects),
            total_subjects,
        )
        batch = _build_chunks_for_batch(batch_subjects, cfg)
        texts = [item["text"] for item in batch]
        chunk_ids = [item["chunk_id"] for item in batch]
        embeddings = _embed_texts_with_retry(
            embedding_model,
            texts,
            chunk_ids,
            max_attempts=cfg.embedding_retries,
        )

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
        elapsed_s = time.perf_counter() - started_at
        rows_per_s = total_upserted / elapsed_s if elapsed_s > 0 else 0.0
        LOGGER.info(
            (
                "Upserted %s/%s KG chunks this run (%.2f%% of remaining) | "
                "skipped_existing=%s | last_batch=%s | elapsed=%.1fs | rate=%.2f chunks/s"
            ),
            total_upserted,
            total_subjects,
            (total_upserted / total_subjects) * 100,
            skipped_existing,
            len(batch),
            elapsed_s,
            rows_per_s,
        )

    LOGGER.info("KG indexing completed. Upserted %s chunks.", total_upserted)


def parse_args() -> KGConfig:
    parser = argparse.ArgumentParser(description="Index KG entities from Fuseki into pgvector chunks.")
    parser.add_argument("--limit", type=int, default=200, help="Max proxy/person URIs to index. Use 0 for all.")
    parser.add_argument("--triples-per-entity", type=int, default=50, help="Triples per entity card.")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding/upsert batch size.")
    parser.add_argument("--dataset-version", type=str, default="dev", help="Dataset version for chunks.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip kg_text chunks already present for the selected dataset version.",
    )
    parser.add_argument(
        "--embedding-retries",
        type=int,
        default=5,
        help="Embedding attempts before splitting a failing batch into smaller sub-batches.",
    )
    args = parser.parse_args()
    return KGConfig(
        subjects_limit=args.limit,
        triples_limit=args.triples_per_entity,
        batch_size=args.batch_size,
        dataset_version=args.dataset_version,
        skip_existing=args.skip_existing,
        embedding_retries=args.embedding_retries,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
