import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_kg_server.server import create_server
import mcp_kg_server.tools.schema_summary as schema_summary_tool


def _extract_structured_output(result):
    if isinstance(result, tuple):
        return result[1]
    return result


def test_schema_summary_returns_non_empty_predicates_for_non_empty_kg(monkeypatch) -> None:
    schema_summary_tool._reset_schema_summary_cache()

    def _fake_execute(query: str, timeout_ms: int):
        if "GROUP BY ?class_uri" in query:
            return {
                "results": {
                    "bindings": [
                        {
                            "class_uri": {"type": "uri", "value": "https://example.org/City"},
                            "count": {"type": "literal", "value": "42"},
                        }
                    ]
                }
            }
        if "GROUP BY ?predicate_uri" in query:
            return {
                "results": {
                    "bindings": [
                        {
                            "predicate_uri": {"type": "uri", "value": "https://schema.org/name"},
                            "count": {"type": "literal", "value": "100"},
                        }
                    ]
                }
            }
        return {
            "results": {
                "bindings": [
                    {
                        "s": {"type": "uri", "value": "https://example.org/paris"},
                        "p": {"type": "uri", "value": "https://schema.org/name"},
                        "o": {"type": "literal", "value": "Paris"},
                    }
                ]
            }
        }

    monkeypatch.setattr(schema_summary_tool, "_execute_sparql_query", _fake_execute)

    server = create_server()
    call_result = asyncio.run(server.call_tool("schema_summary", {}))
    structured = _extract_structured_output(call_result)

    assert len(structured["top_predicates"]) == 1
    assert structured["top_predicates"][0]["predicate_uri"] == "https://schema.org/name"
    assert structured["top_predicates"][0]["count"] == 100
    assert structured["notes"]


def test_schema_summary_handles_empty_kg_gracefully(monkeypatch) -> None:
    schema_summary_tool._reset_schema_summary_cache()

    monkeypatch.setattr(
        schema_summary_tool,
        "_execute_sparql_query",
        lambda query, timeout_ms: {"results": {"bindings": []}},
    )

    server = create_server()
    call_result = asyncio.run(server.call_tool("schema_summary", {}))
    structured = _extract_structured_output(call_result)

    assert structured["top_classes"] == []
    assert structured["top_predicates"] == []
    assert structured["example_triples"] == []


def test_schema_summary_uses_in_memory_cache(monkeypatch) -> None:
    schema_summary_tool._reset_schema_summary_cache()
    call_count = {"value": 0}

    def _fake_execute(query: str, timeout_ms: int):
        call_count["value"] += 1
        if "GROUP BY ?class_uri" in query:
            return {"results": {"bindings": []}}
        if "GROUP BY ?predicate_uri" in query:
            return {"results": {"bindings": []}}
        return {"results": {"bindings": []}}

    monkeypatch.setattr(schema_summary_tool, "_execute_sparql_query", _fake_execute)

    server = create_server()
    first = _extract_structured_output(asyncio.run(server.call_tool("schema_summary", {})))
    second = _extract_structured_output(asyncio.run(server.call_tool("schema_summary", {})))

    assert first == second
    assert call_count["value"] == 3
