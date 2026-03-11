from __future__ import annotations

import time

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from ..settings import settings
from .sparql_query import _execute_sparql_query

_CACHE_TTL_SECONDS = 300
_SUMMARY_CACHE: "SchemaSummaryOutput | None" = None
_SUMMARY_CACHE_EXPIRES_AT = 0.0


class ClassCount(BaseModel):
    class_uri: str
    count: int


class PredicateCount(BaseModel):
    predicate_uri: str
    count: int


class ExampleTriple(BaseModel):
    s: str
    p: str
    o: str


class SchemaSummaryOutput(BaseModel):
    top_classes: list[ClassCount]
    top_predicates: list[PredicateCount]
    example_triples: list[ExampleTriple]
    notes: str


def _parse_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def _get_cached_schema_summary() -> SchemaSummaryOutput | None:
    now = time.monotonic()
    if _SUMMARY_CACHE is not None and now < _SUMMARY_CACHE_EXPIRES_AT:
        return _SUMMARY_CACHE
    return None


def _set_cached_schema_summary(summary: SchemaSummaryOutput) -> None:
    global _SUMMARY_CACHE, _SUMMARY_CACHE_EXPIRES_AT
    _SUMMARY_CACHE = summary
    _SUMMARY_CACHE_EXPIRES_AT = time.monotonic() + _CACHE_TTL_SECONDS


def _reset_schema_summary_cache() -> None:
    global _SUMMARY_CACHE, _SUMMARY_CACHE_EXPIRES_AT
    _SUMMARY_CACHE = None
    _SUMMARY_CACHE_EXPIRES_AT = 0.0


def _build_top_classes_query(limit: int) -> str:
    return (
        "SELECT ?class_uri (COUNT(?s) AS ?count) WHERE {\n"
        "  ?s a ?class_uri .\n"
        "}\n"
        "GROUP BY ?class_uri\n"
        "ORDER BY DESC(?count)\n"
        f"LIMIT {limit}"
    )


def _build_top_predicates_query(limit: int) -> str:
    return (
        "SELECT ?predicate_uri (COUNT(*) AS ?count) WHERE {\n"
        "  ?s ?predicate_uri ?o .\n"
        "}\n"
        "GROUP BY ?predicate_uri\n"
        "ORDER BY DESC(?count)\n"
        f"LIMIT {limit}"
    )


def _build_example_triples_query(limit: int) -> str:
    return (
        "SELECT ?s ?p ?o WHERE {\n"
        "  ?s ?p ?o .\n"
        "}\n"
        f"LIMIT {limit}"
    )


def _parse_top_classes(payload: dict) -> list[ClassCount]:
    bindings = payload.get("results", {}).get("bindings", [])
    parsed: list[ClassCount] = []
    for binding in bindings:
        class_uri = binding.get("class_uri", {}).get("value")
        count_value = binding.get("count", {}).get("value")
        if not isinstance(class_uri, str) or not class_uri:
            continue
        if not isinstance(count_value, str):
            continue
        parsed.append(ClassCount(class_uri=class_uri, count=_parse_int(count_value)))
    return parsed


def _parse_top_predicates(payload: dict) -> list[PredicateCount]:
    bindings = payload.get("results", {}).get("bindings", [])
    parsed: list[PredicateCount] = []
    for binding in bindings:
        predicate_uri = binding.get("predicate_uri", {}).get("value")
        count_value = binding.get("count", {}).get("value")
        if not isinstance(predicate_uri, str) or not predicate_uri:
            continue
        if not isinstance(count_value, str):
            continue
        parsed.append(PredicateCount(predicate_uri=predicate_uri, count=_parse_int(count_value)))
    return parsed


def _parse_example_triples(payload: dict) -> list[ExampleTriple]:
    bindings = payload.get("results", {}).get("bindings", [])
    parsed: list[ExampleTriple] = []
    for binding in bindings:
        subj = binding.get("s", {}).get("value")
        pred = binding.get("p", {}).get("value")
        obj = binding.get("o", {}).get("value")
        if not isinstance(subj, str) or not subj:
            continue
        if not isinstance(pred, str) or not pred:
            continue
        if not isinstance(obj, str):
            continue
        parsed.append(ExampleTriple(s=subj, p=pred, o=obj))
    return parsed


def register_schema_summary_tool(server: FastMCP) -> None:
    @server.tool()
    def schema_summary() -> SchemaSummaryOutput:
        """
        Return a lightweight structural summary of the KG.
        """
        cached = _get_cached_schema_summary()
        if cached is not None:
            return cached

        top_n = min(10, settings.mcp_kg_max_rows)
        sample_n = min(5, settings.mcp_kg_max_rows)
        timeout_ms = settings.mcp_kg_timeout_ms

        top_classes = _parse_top_classes(
            _execute_sparql_query(_build_top_classes_query(top_n), timeout_ms)
        )
        top_predicates = _parse_top_predicates(
            _execute_sparql_query(_build_top_predicates_query(top_n), timeout_ms)
        )
        example_triples = _parse_example_triples(
            _execute_sparql_query(_build_example_triples_query(sample_n), timeout_ms)
        )

        summary = SchemaSummaryOutput(
            top_classes=top_classes,
            top_predicates=top_predicates,
            example_triples=example_triples,
            notes="Labels commonly use rdfs:label, skos:prefLabel, or schema:name.",
        )
        _set_cached_schema_summary(summary)
        return summary
