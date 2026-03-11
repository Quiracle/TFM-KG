import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_kg_server.server import create_server
import mcp_kg_server.tools.entity_facts as entity_facts_tool


def _extract_structured_output(result):
    if isinstance(result, tuple):
        return result[1]
    return result


def test_entity_facts_returns_outgoing_triples(monkeypatch) -> None:
    uri = "https://example.org/paris"
    payload = {
        "results": {
            "bindings": [
                {
                    "p": {"type": "uri", "value": "https://schema.org/name"},
                    "o": {"type": "literal", "value": "Paris", "xml:lang": "en"},
                },
                {
                    "p": {"type": "uri", "value": "https://schema.org/country"},
                    "o": {"type": "uri", "value": "https://example.org/france"},
                },
            ]
        }
    }

    monkeypatch.setattr(entity_facts_tool, "_execute_sparql_query", lambda query, timeout_ms: payload)

    server = create_server()
    call_result = asyncio.run(server.call_tool("entity_facts", {"uri": uri}))
    structured = _extract_structured_output(call_result)

    assert structured["row_count"] == 2
    assert structured["triples"][0]["s"] == uri
    assert structured["triples"][0]["p"] == "https://schema.org/name"
    assert structured["triples"][0]["o"] == "Paris"
    assert structured["triples"][0]["o_type"] == "literal"
    assert structured["triples"][0]["o_lang"] == "en"


def test_entity_facts_include_incoming_adds_incoming_triples(monkeypatch) -> None:
    uri = "https://example.org/paris"
    calls: list[str] = []

    def _fake_execute(query: str, timeout_ms: int):
        calls.append(query)
        if "<https://example.org/paris> ?p ?o" in query:
            return {
                "results": {
                    "bindings": [
                        {
                            "p": {"type": "uri", "value": "https://schema.org/name"},
                            "o": {"type": "literal", "value": "Paris"},
                        }
                    ]
                }
            }
        return {
            "results": {
                "bindings": [
                    {
                        "s": {"type": "uri", "value": "https://example.org/france"},
                        "p": {"type": "uri", "value": "https://schema.org/capital"},
                    }
                ]
            }
        }

    monkeypatch.setattr(entity_facts_tool, "_execute_sparql_query", _fake_execute)

    server = create_server()
    call_result = asyncio.run(
        server.call_tool(
            "entity_facts",
            {"uri": uri, "limit": 10, "include_incoming": True},
        )
    )
    structured = _extract_structured_output(call_result)

    assert len(calls) == 2
    assert "<https://example.org/paris> ?p ?o" in calls[0]
    assert "?s ?p <https://example.org/paris>" in calls[1]
    assert structured["row_count"] == 2
    assert structured["triples"][1]["s"] == "https://example.org/france"
    assert structured["triples"][1]["o"] == uri
    assert structured["triples"][1]["o_type"] == "uri"
    assert structured["triples"][1]["o_lang"] is None


def test_entity_facts_handles_few_triples_gracefully(monkeypatch) -> None:
    monkeypatch.setattr(
        entity_facts_tool,
        "_execute_sparql_query",
        lambda query, timeout_ms: {"results": {"bindings": []}},
    )

    server = create_server()
    call_result = asyncio.run(
        server.call_tool(
            "entity_facts",
            {"uri": "https://example.org/missing", "limit": 50},
        )
    )
    structured = _extract_structured_output(call_result)

    assert structured["row_count"] == 0
    assert structured["triples"] == []
