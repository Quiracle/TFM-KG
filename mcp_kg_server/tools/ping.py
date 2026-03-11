from mcp.server.fastmcp import FastMCP


def register_ping_tool(server: FastMCP) -> None:
    @server.tool()
    def ping() -> dict[str, str]:
        """Simple MCP health check."""
        return {"status": "ok"}
