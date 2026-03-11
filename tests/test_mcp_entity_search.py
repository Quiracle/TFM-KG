import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_kg_server.server import create_server
import mcp_kg_server.tools.entity_search as entity_search_tool


def _extract_structured_output(result):
    if isinstance(result, tuple):
        return result[1]
    return result


def test_entity_search_known_label_returns_uri_and_label(monkeypatch) -> None:
    payload = {
        "results": {
            "bindings": [
                {
                    "uri": {"type": "uri", "value": "https://example.org/paris"},
                    "label": {"type": "literal", "value": "Paris"},
                    "pred": {"type": "uri", "value": "http://www.w3.org/2000/01/rdf-schema#label"},
                },
                {
                    "uri": {"type": "uri", "value": "https://example.org/paris-france"},
                    "label": {"type": "literal", "value": "Paris, France"},
                    "pred": {"type": "uri", "value": "http://schema.org/name"},
                },
                {
                    "uri": {"type": "uri", "value": "https://example.org/description"},
                    "label": {"type": "literal", "value": "City of Paris"},
                    "pred": {"type": "uri", "value": "http://www.w3.org/2004/02/skos/core#prefLabel"},
                },
            ]
        }
    }

    monkeypatch.setattr(entity_search_tool, "_execute_sparql_query", lambda query, timeout_ms: payload)

    server = create_server()
    call_result = asyncio.run(server.call_tool("entity_search", {"text": "Paris"}))
    structured = _extract_structured_output(call_result)

    assert len(structured["results"]) == 3
    assert structured["results"][0]["uri"] == "https://example.org/paris"
    assert structured["results"][0]["label"] == "Paris"
    assert structured["results"][0]["score"] == 1.0


def test_entity_search_empty_results_are_graceful(monkeypatch) -> None:
    monkeypatch.setattr(
        entity_search_tool,
        "_execute_sparql_query",
        lambda query, timeout_ms: {"results": {"bindings": []}},
    )

    server = create_server()
    call_result = asyncio.run(server.call_tool("entity_search", {"text": "does-not-exist"}))
    structured = _extract_structured_output(call_result)

    assert structured["results"] == []


def test_entity_search_query_uses_expected_predicates_and_lang_filter(monkeypatch) -> None:
    captured = {}

    def _fake_execute(query, timeout_ms):
        captured["query"] = query
        captured["timeout_ms"] = timeout_ms
        return {"results": {"bindings": []}}

    monkeypatch.setattr(entity_search_tool, "_execute_sparql_query", _fake_execute)

    server = create_server()
    asyncio.run(
        server.call_tool(
            "entity_search",
            {"text": "Par", "limit": 5, "lang": "en"},
        )
    )

    assert "rdfs:label" in captured["query"]
    assert "skos:prefLabel" in captured["query"]
    assert "<http://schema.org/name>" in captured["query"]
    assert "<https://schema.org/name>" in captured["query"]
    assert "CONTAINS(LCASE(STR(?label)), LCASE(\"Par\"))" in captured["query"]
    assert "LANGMATCHES(LANG(?label), \"en\")" in captured["query"]
