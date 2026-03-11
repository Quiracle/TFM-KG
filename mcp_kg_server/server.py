from mcp.server.fastmcp import FastMCP

from .settings import settings
from .tools import register_entity_search_tool, register_ping_tool, register_sparql_query_tool


def create_server() -> FastMCP:
    mcp = FastMCP(settings.mcp_server_name)
    register_ping_tool(mcp)
    register_sparql_query_tool(mcp)
    register_entity_search_tool(mcp)
    return mcp


mcp_server = create_server()


def main() -> None:
    mcp_server.run(transport="stdio")


if __name__ == "__main__":
    main()
