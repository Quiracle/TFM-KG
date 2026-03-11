# IMPLEMENT.md — MCP KG Server (Fuseki) + Iterative Tool-Query QA

## Overview
We will implement an MCP server that exposes safe, read-only tools for querying the Knowledge Graph (Fuseki via SPARQL).
The client (Codex / VS Code) can call these tools iteratively to refine queries.

Key principles:
- Read-only queries only (no SPARQL UPDATE).
- Strong tool schemas (helps LLM use tools correctly).
- Result-size and timeout limits to prevent runaway queries.
- Telemetry logs all tool calls (query + row_count + latency).
- Server runs via stdio transport for VS Code MCP.

References (non-exhaustive):
- MCP Python SDK and server examples.
- MCP tools and prompts concepts.
- VS Code MCP server configuration docs.

---

## Shared verification commands

### Boot Fuseki + Postgres + API (existing stack)
- `docker compose up --build -d`

### Run MCP server locally (stdio)
- `python -m mcp_kg_server`

### Quick tool smoke test (manual)
Use VS Code MCP tools UI / Codex tool calls. (See M1 for VS Code config.)

---

## Milestone M0 — Add MCP server skeleton

### Goal
Create a new Python package `mcp_kg_server` with an MCP server running over stdio.

### Implementation
1) Add new top-level folder:
   - `mcp_kg_server/`
     - `__init__.py`
     - `server.py`
     - `settings.py`
     - `tools/`
     - `prompts/`
2) Add dependency:
   - add MCP Python SDK to `pyproject.toml` (e.g. `mcp` as per SDK docs)
3) Server must:
   - start successfully with `python -m mcp_kg_server`
   - register at least one trivial tool: `ping() -> {status:"ok"}`

### Acceptance criteria
- Running `python -m mcp_kg_server` starts without exceptions.
- VS Code can connect to it and list tools (at least `ping`).

---

## Milestone M1 — VS Code MCP configuration + docs

### Goal
Make it easy to connect VS Code to the MCP server.

### Implementation
1) Add `docs/mcp-vscode-setup.md` with:
   - how to register the server in VS Code MCP settings
   - example config snippet (command + args + env)
2) Add `.env.example` entries for Fuseki config used by MCP server:
   - `FUSEKI_URL`
   - `FUSEKI_DATASET`
   - `MCP_KG_TIMEOUT_MS`
   - `MCP_KG_MAX_ROWS`

### Acceptance criteria
- Following the docs, a developer can connect VS Code to the MCP server and call `ping`.

---

## Milestone M2 — Core tool: sparql_query (read-only, safe)

### Goal
Expose a `sparql_query` tool that can query Fuseki safely.

### Tool: `sparql_query`
**Input schema**
- `query: string` (required)
- `timeout_ms: int` (optional; default from env)
- `max_rows: int` (optional; default from env)

**Output schema**
- `row_count: int`
- `head: object` (SPARQL JSON head)
- `results: object` (SPARQL JSON results)
- `truncated: bool`
- `latency_ms: int`

### Safety requirements
Reject queries containing (case-insensitive):
- `INSERT`, `DELETE`, `LOAD`, `CLEAR`, `DROP`, `CREATE`, `MOVE`, `COPY`, `ADD`
Also reject multiple statements separated by `;` (simple heuristic ok).

Enforce:
- Always apply `LIMIT max_rows` if query lacks a LIMIT (simple parser/regex acceptable for MVP).
- Apply HTTP timeout.

### Acceptance criteria
- Calling `sparql_query` with a valid SELECT returns results.
- Calling with an UPDATE keyword returns an MCP tool error with helpful message.
- Query without LIMIT is capped by `max_rows`.

---

## Milestone M3 — Tool: entity_search (label-based lookup)

### Goal
Enable robust entity discovery without embeddings.

### Tool: `entity_search`
**Input**
- `text: string`
- `limit: int` default 10
- `lang: string | null` default null

**Output**
- `results: [{ uri, label, score }]`

### Implementation notes
- Use SPARQL to search `rdfs:label` (and optionally `skos:prefLabel`, `schema:name`) with case-insensitive contains.
- `score` can be heuristic: exact match > prefix > contains.

### Acceptance criteria
- Searching a known label returns its URI and label.
- Empty results are handled gracefully.

---

## Milestone M4 — Tool: entity_facts (1-hop facts for citations)

### Goal
Provide evidence packs directly from KG facts.

### Tool: `entity_facts`
**Input**
- `uri: string`
- `limit: int` default 50
- `include_incoming: bool` default false

**Output**
- `triples: [{ s, p, o, o_type, o_lang }]`
- `row_count: int`

### Implementation notes
- Query: `<uri> ?p ?o` with LIMIT.
- If include_incoming: `?s ?p <uri>` (separate query or UNION with cap).
- Return objects with datatype/lang info when possible.

### Acceptance criteria
- Given a URI from `entity_search`, `entity_facts` returns triples.
- Works if only a few triples exist.

---

## Milestone M5 — Tool: schema_summary (KG introspection)

### Goal
Give the model a quick understanding of the KG structure.

### Tool: `schema_summary`
**Output**
- `top_classes: [{ class_uri, count }]` (top N)
- `top_predicates: [{ predicate_uri, count }]` (top N)
- `example_triples: [{ s, p, o }]` (small sample)
- `notes: string` (e.g., “labels use rdfs:label”)

### Implementation notes
- Use lightweight SPARQL sampling queries (LIMITed).
- Cache result in-memory for ~5 minutes to reduce load.

### Acceptance criteria
- Tool returns non-empty predicates list on a non-empty KG.
- Works on empty KG (returns empty arrays, not errors).

---

## Milestone M6 — MCP prompts: “Iterative KG Query Assistant”

### Goal
Expose an MCP prompt template that teaches the LLM how to iterate.

### Prompt: `kg_query_assistant`
Provide an MCP prompt that instructs:
- Use `schema_summary` first (once) unless already known.
- Use `entity_search` to resolve ambiguous names to URIs.
- Use `sparql_query` for structured questions.
- If results are empty or wrong shape, revise query and try again (max 3 attempts).
- Always cite URIs/predicates used.
- If evidence is insufficient, abstain.

Arguments:
- `question: string`
- `max_attempts: int` default 3

### Acceptance criteria
- VS Code can list prompts and insert/use `kg_query_assistant`.
- The prompt is concise, tool-oriented, and enforces read-only + citations.

---

## Milestone M7 — Telemetry for MCP tool calls (separate from /query telemetry)

### Goal
Log each MCP tool call for evaluation and debugging.

### Implementation
- Add a lightweight log file (JSONL) by default:
  - `logs/mcp_tool_calls.jsonl`
- Each record:
  - timestamp, tool_name
  - request parameters (redact if needed)
  - row_count, truncated, latency_ms
  - errors

Optional:
- Also write to Postgres as `mcp_runs` table, but JSONL is acceptable for MVP.

### Acceptance criteria
- Each tool call creates a JSONL record.
- Errors are recorded with reason.

---

## Milestone M8 — “Answer mode” integration (optional but recommended)

### Goal
Add a new `/query` mode that leverages the same iterative workflow OR document a VS Code workflow using MCP.

Options:
A) Keep MCP for developer workflow only (VS Code / Codex uses MCP tools directly).
B) Add `/query` mode `kg_tool` that mimics the MCP iterative workflow internally (not via MCP) to make it reproducible for benchmarking.

For thesis benchmarking, B is recommended:
- it makes the behavior reproducible outside VS Code.

### Acceptance criteria
- Documented workflow exists to answer questions by iterative SPARQL tool use.
- If implementing `/query mode=kg_tool`, it must:
  - do 1–3 tool-like query attempts
  - abstain when evidence is insufficient
  - return citations (URIs/predicates) and debug info when requested
