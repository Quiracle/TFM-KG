from mcp.server.fastmcp import FastMCP


def _format_kg_query_prompt(question: str, max_attempts: int) -> str:
    bounded_attempts = min(max(max_attempts, 1), 3)
    return (
        "You are an iterative KG query assistant. Use only read-only MCP tools.\n"
        f"Question: {question}\n\n"
        "Workflow:\n"
        "1. Run schema_summary first (once) unless schema is already known in this session.\n"
        "2. Resolve ambiguous names using entity_search and keep candidate URIs.\n"
        "3. Use sparql_query for structured retrieval over the selected URIs/predicates.\n"
        f"4. If results are empty or wrong shape, revise and retry up to {bounded_attempts} attempts total.\n"
        "5. Cite the exact URIs and predicates used in your final answer.\n"
        "6. If evidence is insufficient after retries, abstain clearly.\n\n"
        "Constraints:\n"
        "- Never use SPARQL UPDATE operations.\n"
        "- Keep queries bounded and tool-oriented.\n"
        "- Prefer precise URI-based filtering over broad scans."
    )


def register_kg_query_assistant_prompt(server: FastMCP) -> None:
    @server.prompt(name="kg_query_assistant")
    def kg_query_assistant(question: str, max_attempts: int = 3) -> str:
        """
        Prompt template for iterative, citation-first KG querying.
        """
        return _format_kg_query_prompt(question, max_attempts)
