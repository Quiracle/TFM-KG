from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from ..settings import settings
from .sparql_query import _execute_sparql_query


class EntityFactTriple(BaseModel):
    s: str
    p: str
    o: str
    o_type: str
    o_lang: str | None = None


class EntityFactsOutput(BaseModel):
    triples: list[EntityFactTriple]
    row_count: int


def _validate_uri(uri: str) -> str:
    candidate = uri.strip()
    if not candidate:
        raise ValueError("uri must not be empty.")
    if any(char in candidate for char in ('<', '>', '"', " ", "\n", "\r", "\t")):
        raise ValueError("uri must be a valid IRI without spaces or angle brackets.")
    return candidate


def _build_outgoing_query(uri: str, limit: int) -> str:
    return (
        "SELECT ?p ?o WHERE {\n"
        f"  <{uri}> ?p ?o .\n"
        "}\n"
        f"LIMIT {limit}"
    )


def _build_incoming_query(uri: str, limit: int) -> str:
    return (
        "SELECT ?s ?p WHERE {\n"
        f"  ?s ?p <{uri}> .\n"
        "}\n"
        f"LIMIT {limit}"
    )


def _parse_outgoing_bindings(uri: str, payload: dict) -> list[EntityFactTriple]:
    bindings = payload.get("results", {}).get("bindings", [])
    triples: list[EntityFactTriple] = []
    for binding in bindings:
        predicate = binding.get("p", {}).get("value")
        obj = binding.get("o", {})
        obj_value = obj.get("value")
        obj_type = obj.get("type")
        if not isinstance(predicate, str) or not predicate:
            continue
        if not isinstance(obj_value, str):
            continue
        if not isinstance(obj_type, str):
            continue

        obj_lang = obj.get("xml:lang")
        triples.append(
            EntityFactTriple(
                s=uri,
                p=predicate,
                o=obj_value,
                o_type=obj_type,
                o_lang=obj_lang if isinstance(obj_lang, str) else None,
            )
        )
    return triples


def _parse_incoming_bindings(uri: str, payload: dict) -> list[EntityFactTriple]:
    bindings = payload.get("results", {}).get("bindings", [])
    triples: list[EntityFactTriple] = []
    for binding in bindings:
        subj = binding.get("s", {}).get("value")
        predicate = binding.get("p", {}).get("value")
        if not isinstance(subj, str) or not subj:
            continue
        if not isinstance(predicate, str) or not predicate:
            continue
        triples.append(
            EntityFactTriple(
                s=subj,
                p=predicate,
                o=uri,
                o_type="uri",
                o_lang=None,
            )
        )
    return triples


def register_entity_facts_tool(server: FastMCP) -> None:
    @server.tool()
    def entity_facts(
        uri: str,
        limit: int = 50,
        include_incoming: bool = False,
    ) -> EntityFactsOutput:
        """
        Return one-hop triples for an entity URI.
        """
        normalized_uri = _validate_uri(uri)
        if limit <= 0:
            raise ValueError("limit must be a positive integer.")

        effective_limit = min(limit, settings.mcp_kg_max_rows)
        timeout_ms = settings.mcp_kg_timeout_ms

        triples: list[EntityFactTriple] = []
        seen: set[tuple[str, str, str, str, str | None]] = set()

        outgoing_payload = _execute_sparql_query(
            _build_outgoing_query(normalized_uri, effective_limit),
            timeout_ms,
        )
        for triple in _parse_outgoing_bindings(normalized_uri, outgoing_payload):
            key = (triple.s, triple.p, triple.o, triple.o_type, triple.o_lang)
            if key not in seen:
                seen.add(key)
                triples.append(triple)
            if len(triples) >= effective_limit:
                return EntityFactsOutput(triples=triples, row_count=len(triples))

        if include_incoming and len(triples) < effective_limit:
            remaining = effective_limit - len(triples)
            incoming_payload = _execute_sparql_query(
                _build_incoming_query(normalized_uri, remaining),
                timeout_ms,
            )
            for triple in _parse_incoming_bindings(normalized_uri, incoming_payload):
                key = (triple.s, triple.p, triple.o, triple.o_type, triple.o_lang)
                if key not in seen:
                    seen.add(key)
                    triples.append(triple)
                if len(triples) >= effective_limit:
                    break

        return EntityFactsOutput(triples=triples, row_count=len(triples))
