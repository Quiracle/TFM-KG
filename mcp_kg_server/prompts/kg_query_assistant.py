from pathlib import Path

from mcp.server.fastmcp import FastMCP


_GUIDE_PATH = Path(__file__).resolve().parents[2] / "docs" / "amsterdam_museum_kg_guide.md"
_GUIDE_TEXT = _GUIDE_PATH.read_text(encoding="utf-8").strip()


def _format_kg_query_prompt(question: str, max_attempts: int) -> str:
    bounded_attempts = min(max(max_attempts, 1), 3)
    return (
        f"{_GUIDE_TEXT}\n\n"
        "Additional runtime instructions:\n"
        "- You are answering a live user question using only read-only MCP tools.\n"
        f"- User question: {question}\n"
        f"- Retry budget: at most {bounded_attempts} total structured retrieval attempts.\n"
        "- Start with `schema_summary` once per session only if you still need orientation.\n"
        "- Prefer `sparql_query` for precise retrieval; use `entity_search` only to disambiguate names.\n"
        "- Never use SPARQL UPDATE operations.\n"
        "- Keep queries bounded and avoid broad unfiltered scans.\n"
        "- Cite the exact URIs and predicates used in the final answer.\n"
        "- If evidence is still insufficient after the retry budget, say so clearly."
    )


def register_kg_query_assistant_prompt(server: FastMCP) -> None:
    @server.prompt(name="kg_query_assistant")
    def kg_query_assistant(question: str, max_attempts: int = 3) -> str:
        """
        Prompt template for iterative, citation-first KG querying.
        """
        return _format_kg_query_prompt(question, max_attempts)
