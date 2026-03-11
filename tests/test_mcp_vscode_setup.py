from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_env_example_includes_m1_mcp_settings() -> None:
    env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")

    required_keys = (
        "FUSEKI_URL=",
        "FUSEKI_DATASET=",
        "MCP_KG_TIMEOUT_MS=",
        "MCP_KG_MAX_ROWS=",
    )
    for key in required_keys:
        assert key in env_example


def test_mcp_vscode_setup_doc_contains_required_configuration() -> None:
    setup_doc_path = REPO_ROOT / "docs" / "mcp-vscode-setup.md"
    assert setup_doc_path.exists()

    setup_doc = setup_doc_path.read_text(encoding="utf-8")

    required_tokens = (
        "MCP: Open Workspace Folder MCP Configuration",
        '"servers"',
        '"type": "stdio"',
        '"command":',
        '"args":',
        '"env":',
        "mcp_kg_server",
        "ping",
    )
    for token in required_tokens:
        assert token in setup_doc
