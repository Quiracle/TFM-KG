import time
from typing import Any

from mcp.server.fastmcp import FastMCP

from .prompts import register_kg_query_assistant_prompt
from .settings import settings
from .telemetry import log_mcp_tool_call
from .tools import (
    register_entity_facts_tool,
    register_entity_search_tool,
    register_ping_tool,
    register_schema_summary_tool,
    register_sparql_query_tool,
)


class TelemetryFastMCP(FastMCP):
    async def call_tool(self, name: str, arguments: dict[str, Any]):
        started_at = time.perf_counter()
        result = None
        error = None
        try:
            result = await super().call_tool(name, arguments)
            return result
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            log_mcp_tool_call(
                tool_name=name,
                request=arguments,
                latency_ms=latency_ms,
                result=result,
                error=error,
            )


def create_server() -> FastMCP:
    mcp = TelemetryFastMCP(settings.mcp_server_name)
    register_ping_tool(mcp)
    register_sparql_query_tool(mcp)
    register_entity_search_tool(mcp)
    register_entity_facts_tool(mcp)
    register_schema_summary_tool(mcp)
    register_kg_query_assistant_prompt(mcp)
    return mcp


mcp_server = create_server()


def main() -> None:
    mcp_server.run(transport="stdio")


if __name__ == "__main__":
    main()
