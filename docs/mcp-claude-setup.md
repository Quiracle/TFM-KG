# Claude MCP setup (KG Server)

This project exposes an MCP server over stdio via:


```bash
python -m mcp_kg_server
```

## 1) Prepare environment

1. Start the stack:

```bash
docker compose up --build -d
```

2. Ensure dependencies are installed locally (so `mcp` is available):

```bash
uv sync
```

3. Set MCP-related values in your `.env` (copy from `.env.example` if needed):
- `FUSEKI_URL`
- `FUSEKI_DATASET`
- `MCP_KG_TIMEOUT_MS`
- `MCP_KG_MAX_ROWS`

## 2) Register MCP server in Claude

1. Open folder %APPDATA%/Claude
2. Add this configuration to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "tfmkg-kg": {
      "command": "absolute/path/to/.venv/Scripts/python.exe",
      "args": ["-m", "mcp_kg_server"],
      "cwd": "absolute/path/to/repo",
      "env": {
        "PYTHONPATH": "absolute/path/to/repo",
        "FUSEKI_URL": "http://localhost:3030",
        "FUSEKI_DATASET": "kg",
        "MCP_KG_TIMEOUT_MS": "10000",
        "MCP_KG_MAX_ROWS": "200"
      }
    }
  }
}
```

Notes:
- On Linux/macOS, use `.venv/bin/python` for `command`.
- You can also use `python` if your selected interpreter already has the project dependencies installed.

## 3) Validate connection

1. Reload VS Code window after saving MCP config.
2. In the MCP tools UI, confirm server `tfmkg-kg` is connected.
3. Call tool `ping`.

Expected response:

```json
{"status":"ok"}
```
