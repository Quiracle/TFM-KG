import asyncio
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs

import pytest
from mcp.server.fastmcp.exceptions import ToolError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_kg_server.server import create_server
import mcp_kg_server.tools.sparql_query as sparql_query_tool


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _extract_structured_output(result):
    if isinstance(result, tuple):
        return result[1]
    return result


def test_sparql_query_valid_select_returns_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    payload = {
        "head": {"vars": ["s"]},
        "results": {"bindings": [{"s": {"type": "uri", "value": "https://example.org/A"}}]},
    }

    def _fake_urlopen(request, timeout):  # noqa: ANN001
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse(payload)

    monkeypatch.setattr(sparql_query_tool, "urlopen", _fake_urlopen)

    server = create_server()
    call_result = asyncio.run(
        server.call_tool(
            "sparql_query",
            {"query": "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"},
        )
    )
    structured = _extract_structured_output(call_result)

    assert structured["row_count"] == 1
    assert structured["head"] == payload["head"]
    assert structured["results"] == payload["results"]
    assert structured["truncated"] is False
    assert isinstance(structured["latency_ms"], int)

    sent_query = parse_qs(captured["request"].data.decode("utf-8"))["query"][0]
    assert sent_query.strip().upper().endswith("LIMIT 1")
    assert captured["timeout"] == 10.0


def test_sparql_query_rejects_update_keywords_with_tool_error() -> None:
    server = create_server()

    with pytest.raises(ToolError, match="Read-only query required"):
        asyncio.run(
            server.call_tool(
                "sparql_query",
                {"query": "INSERT DATA { <x> <y> <z> }"},
            )
        )


def test_sparql_query_adds_limit_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    payload = {
        "head": {"vars": ["s"]},
        "results": {
            "bindings": [
                {"s": {"type": "uri", "value": "https://example.org/A"}},
                {"s": {"type": "uri", "value": "https://example.org/B"}},
            ]
        },
    }

    def _fake_urlopen(request, timeout):  # noqa: ANN001
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse(payload)

    monkeypatch.setattr(sparql_query_tool, "urlopen", _fake_urlopen)

    server = create_server()
    call_result = asyncio.run(
        server.call_tool(
            "sparql_query",
            {
                "query": "SELECT ?s WHERE { ?s ?p ?o }",
                "max_rows": 2,
                "timeout_ms": 2500,
            },
        )
    )
    structured = _extract_structured_output(call_result)

    sent_query = parse_qs(captured["request"].data.decode("utf-8"))["query"][0]

    assert "LIMIT 2" in sent_query.upper()
    assert captured["timeout"] == 2.5
    assert structured["row_count"] == 2
    assert structured["truncated"] is True
