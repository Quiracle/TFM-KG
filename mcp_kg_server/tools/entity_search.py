from __future__ import annotations

from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from ..settings import settings
from .sparql_query import _execute_sparql_query


class EntitySearchResult(BaseModel):
    uri: str
    label: str
    score: float


class EntitySearchOutput(BaseModel):
    results: list[EntitySearchResult]


@dataclass
class _ScoredEntity:
    uri: str
    label: str
    score: float


def _escape_sparql_string(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _score_match(search_text: str, label: str) -> float:
    normalized_search = _normalize_text(search_text)
    normalized_label = _normalize_text(label)

    if normalized_label == normalized_search:
        return 1.0
    if normalized_label.startswith(normalized_search):
        return 0.8
    if normalized_search in normalized_label:
        return 0.5
    return 0.0


def _build_entity_search_query(text: str, limit: int, lang: str | None) -> str:
    escaped_text = _escape_sparql_string(text)

    lang_filter = ""
    if lang is not None:
        lang_value = lang.strip()
        if lang_value:
            escaped_lang = _escape_sparql_string(lang_value)
            lang_filter = (
                f'\n  FILTER(LANG(?label) = "" || LANGMATCHES(LANG(?label), "{escaped_lang}"))'
            )

    return (
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>\n"
        "SELECT ?uri ?label ?pred WHERE {\n"
        "  ?uri ?pred ?label .\n"
        "  VALUES ?pred {\n"
        "    rdfs:label\n"
        "    skos:prefLabel\n"
        "    <http://schema.org/name>\n"
        "    <https://schema.org/name>\n"
        "  }\n"
        "  FILTER(isLiteral(?label))\n"
        f'  FILTER(CONTAINS(LCASE(STR(?label)), LCASE("{escaped_text}"))){lang_filter}\n'
        "}\n"
        f"LIMIT {limit}"
    )


def register_entity_search_tool(server: FastMCP) -> None:
    @server.tool()
    def entity_search(
        text: str,
        limit: int = 10,
        lang: str | None = None,
    ) -> EntitySearchOutput:
        """
        Search entities by label-like predicates in the KG.
        """
        query_text = text.strip()
        if not query_text:
            raise ValueError("text must not be empty.")
        if limit <= 0:
            raise ValueError("limit must be a positive integer.")

        effective_limit = min(limit, settings.mcp_kg_max_rows)
        query = _build_entity_search_query(query_text, effective_limit, lang)
        payload = _execute_sparql_query(query, settings.mcp_kg_timeout_ms)

        bindings = payload.get("results", {}).get("bindings", [])
        by_key: dict[tuple[str, str], _ScoredEntity] = {}

        for binding in bindings:
            uri = binding.get("uri", {}).get("value")
            label = binding.get("label", {}).get("value")
            if not isinstance(uri, str) or not uri:
                continue
            if not isinstance(label, str) or not label:
                continue

            score = _score_match(query_text, label)
            if score <= 0.0:
                continue

            key = (uri, label)
            existing = by_key.get(key)
            if existing is None or score > existing.score:
                by_key[key] = _ScoredEntity(uri=uri, label=label, score=score)

        ranked = sorted(
            by_key.values(),
            key=lambda item: (-item.score, item.label.casefold(), item.uri),
        )

        return EntitySearchOutput(
            results=[
                EntitySearchResult(uri=item.uri, label=item.label, score=item.score)
                for item in ranked[:effective_limit]
            ]
        )
