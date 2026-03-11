import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_kg_server.server import create_server


def _extract_text(messages) -> str:
    parts: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            content = message.get("content", {})
            if isinstance(content, dict):
                text = content.get("text")
                if isinstance(text, str):
                    parts.append(text)
        else:
            content = getattr(message, "content", None)
            text = getattr(content, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def test_kg_query_assistant_prompt_is_listed_with_arguments() -> None:
    server = create_server()
    prompts = asyncio.run(server.list_prompts())

    target = next((prompt for prompt in prompts if prompt.name == "kg_query_assistant"), None)
    assert target is not None

    args_by_name = {argument.name: argument for argument in (target.arguments or [])}
    assert "question" in args_by_name
    assert args_by_name["question"].required is True
    assert "max_attempts" in args_by_name
    assert args_by_name["max_attempts"].required is False


def test_kg_query_assistant_prompt_renders_tool_oriented_instructions() -> None:
    server = create_server()
    prompt = asyncio.run(
        server.get_prompt(
            "kg_query_assistant",
            {"question": "What is the capital of France?", "max_attempts": 2},
        )
    )

    text = _extract_text(prompt.messages)
    assert "schema_summary" in text
    assert "entity_search" in text
    assert "sparql_query" in text
    assert "2 attempts" in text
    assert "Cite the exact URIs and predicates" in text
    assert "Never use SPARQL UPDATE" in text
    assert "abstain" in text.lower()


def test_kg_query_assistant_prompt_uses_default_max_attempts() -> None:
    server = create_server()
    prompt = asyncio.run(
        server.get_prompt(
            "kg_query_assistant",
            {"question": "Who created the Eiffel Tower?"},
        )
    )

    text = _extract_text(prompt.messages)
    assert "3 attempts" in text
