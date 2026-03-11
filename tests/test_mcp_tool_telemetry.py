import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from mcp.server.fastmcp.exceptions import ToolError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_kg_server.server import create_server
from mcp_kg_server.settings import settings
import mcp_kg_server.tools.sparql_query as sparql_query_tool


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _new_test_log_path() -> Path:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"mcp_tool_calls_test_{uuid4().hex}.jsonl"


def test_mcp_tool_telemetry_logs_successful_tool_call(monkeypatch) -> None:
    log_path = _new_test_log_path()
    monkeypatch.setattr(settings, "mcp_tool_log_path", str(log_path))

    try:
        server = create_server()
        asyncio.run(server.call_tool("ping", {}))

        rows = _read_jsonl(log_path)
        assert len(rows) == 1
        assert rows[0]["tool_name"] == "ping"
        assert rows[0]["request"] == {}
        assert rows[0]["errors"] is None
        assert isinstance(rows[0]["latency_ms"], int)
    finally:
        log_path.unlink(missing_ok=True)


def test_mcp_tool_telemetry_logs_errors(monkeypatch) -> None:
    log_path = _new_test_log_path()
    monkeypatch.setattr(settings, "mcp_tool_log_path", str(log_path))

    try:
        server = create_server()
        with pytest.raises(ToolError):
            asyncio.run(
                server.call_tool(
                    "sparql_query",
                    {"query": "INSERT DATA { <x> <y> <z> }"},
                )
            )

        rows = _read_jsonl(log_path)
        assert len(rows) == 1
        assert rows[0]["tool_name"] == "sparql_query"
        assert rows[0]["errors"] is not None
        assert "Read-only query required" in rows[0]["errors"]["message"]
    finally:
        log_path.unlink(missing_ok=True)


def test_mcp_tool_telemetry_captures_row_count_and_truncated(monkeypatch) -> None:
    log_path = _new_test_log_path()
    monkeypatch.setattr(settings, "mcp_tool_log_path", str(log_path))
    monkeypatch.setattr(
        sparql_query_tool,
        "_execute_sparql_query",
        lambda query, timeout_ms: {
            "head": {"vars": ["s"]},
            "results": {
                "bindings": [
                    {"s": {"type": "uri", "value": "https://example.org/a"}},
                    {"s": {"type": "uri", "value": "https://example.org/b"}},
                ]
            },
        },
    )

    try:
        server = create_server()
        asyncio.run(
            server.call_tool(
                "sparql_query",
                {"query": "SELECT ?s WHERE { ?s ?p ?o }", "max_rows": 2},
            )
        )

        rows = _read_jsonl(log_path)
        assert len(rows) == 1
        assert rows[0]["tool_name"] == "sparql_query"
        assert rows[0]["row_count"] == 2
        assert rows[0]["truncated"] is True
    finally:
        log_path.unlink(missing_ok=True)
