# IMPLEMENT.md — Milestone plan (TFM KG + RAG)

This repo is a scaffold for a KG-backed RAG prototype:
- Fuseki = canonical KG (SPARQL)
- Postgres + pgvector = chunks + embeddings + telemetry
- FastAPI = API layer

Each milestone must be completed with small diffs and verified using the commands below.

---

## Shared verification commands

Boot:
- `docker compose up --build -d`

Smoke:
- `curl -s http://localhost:8000/health`

Logs (if failing):
- `docker compose logs -f api`
- `docker compose logs -f postgres`
- `docker compose logs -f fuseki`

Reset (destructive):
- `docker compose down -v`

---

## M0 — Scaffold boots reliably

### Goal
Compose boots all services. API exposes `/health`.

### Acceptance criteria
- `docker compose up --build -d` completes
- `curl -s http://localhost:8000/health` returns `{"status":"ok"}`

### Notes
No feature work; only fix boot/packaging issues.

---

## M1 — Postgres wiring + DB health endpoint

### Goal
Add a minimal Postgres connectivity check from the API.

### Implementation
- Create a small DB helper (psycopg) and dependency provider.
- Add endpoint: `GET /health/db` runs `SELECT 1`.

### Acceptance criteria
- `curl -s http://localhost:8000/health/db` returns `{ "db": "ok" }` (shape can include extra fields)
- Errors return a non-200 with a useful message (no stack trace leak by default)

---

## M2 — Vector store repository: upsert chunks

### Goal
Implement a concrete pgvector repository for the `chunks` table.

### Implementation
- Add `src/tfmkg/adapters/vectorstore/pgvector/repository.py`
- Implement:
  - `upsert_chunks(chunks)` inserting/updating by `chunk_id`
- Keep schema compatible with `migrations/001_init_pgvector.sql`

### Acceptance criteria
- A simple script or test inserts 1 chunk and can read it back.
- Upserting the same `chunk_id` updates `text/metadata/embedding`.

Recommended check (one of):
- Integration test via `pytest`
- A temporary admin endpoint (if you add one, remove it before later milestones)

---

## M3 — Similarity search (pgvector)

### Goal
Implement cosine similarity search over `chunks.embedding`.

### Implementation
- Implement `similarity_search(query_embedding, top_k, filters)`
- Support filters at least by:
  - `source_type`
  - `dataset_version`

### Acceptance criteria
- Insert at least 3 chunks with known embeddings
- Query returns closest chunk as rank #1
- `top_k` respected

---

## M4 — `/query` mode routing (no LLM yet)

### Goal
Make `/query` use retrieval modes for benchmarking:
- `text` -> vector search with `source_type=doc_text`
- `table` -> vector search with `source_type=table_row`
- `hybrid` -> vector search with `source_type=kg_text` (SPARQL expansion later)
- `kg` -> stub “not implemented yet” but keep response shape stable

### Acceptance criteria
- `/query` returns a stable JSON response with:
  - `answer`, `mode`, `top_k`, `citations` (maybe empty), `debug`
- `debug` includes retrieved `chunk_id`s when vector search is used
- Mode switch changes filtering behavior

---

## M5 — Fuseki adapter + KG ping endpoint

### Goal
Add a Fuseki client and a minimal endpoint to confirm SPARQL works.

### Implementation
- Add `src/tfmkg/adapters/triplestore/fuseki_client.py`
- Add endpoint: `GET /kg/ping`
  - Executes a safe query like `SELECT * WHERE { ?s ?p ?o } LIMIT 1`
  - Works even if dataset is empty (return empty result but status ok)

### Acceptance criteria
- `curl -s http://localhost:8000/kg/ping` returns a JSON payload showing:
  - `status: ok`
  - `results_count: <int>`

---

## M6 — Telemetry logging to Postgres (`runs` table)

### Goal
Log every `/query` call to the `runs` table for reproducible evaluation.

### Implementation
- Add a telemetry adapter inserting into `runs`
- Log fields:
  - question, mode, top_k
  - retrieved_chunk_ids (json)
  - evidence_pack (json) — can be minimal in early milestones
  - answer, abstained=false (for now)
  - latency_ms
  - dataset_version (use fixed "dev" for now)

### Acceptance criteria
- After calling `/query`, a row exists in `runs`
- Retrieved IDs are stored when retrieval was used

---

## M7 — Evidence pack + deterministic “answer from evidence” (still no LLM)

### Goal
Build an evidence pack and generate a deterministic placeholder answer using only evidence text.

### Implementation
- `EvidencePack` builder compacts retrieved chunks into:
  - short bullet facts
  - citations containing chunk_id + source_ref
- `/query` response includes `citations` list

### Acceptance criteria
- `/query` returns citations for retrieved chunks
- Answer is derived only from evidence text (no extra claims)

---

## M8 — LLM + Embeddings integration (Ollama default, OpenAI optional)

### Goal
Enable real RAG generation with:
- Local Ollama embeddings using **embeddinggemma**
- Local Ollama chat generation using **mistral:7b**
- Optional OpenAI backend for embeddings and/or chat via environment switch

This milestone must keep the `/query` response shape stable and remain evidence-grounded.

---

### Environment variables (must be supported)

#### Provider selection
- `EMBEDDINGS_PROVIDER`: `ollama` (default) | `openai`
- `LLM_PROVIDER`: `ollama` (default) | `openai`

#### Ollama
- `OLLAMA_BASE_URL`: default `http://ollama:11434` in Docker, `http://localhost:11434` outside
- `OLLAMA_EMBED_MODEL`: default `embeddinggemma`
- `OLLAMA_LLM_MODEL`: default `mistral:7b`
- `OLLAMA_TIMEOUT_S`: default `60`
- `OLLAMA_STREAM`: `false` default (streaming can be added later)

Notes:
- Use Ollama embeddings capability docs as reference for embeddings endpoint behavior and model selection. (Recommended embedding models include `embeddinggemma`.) 
- Vector length depends on the model; do not hardcode embedding dimension. Store the dimension in config/metadata, and validate it matches the DB schema or update schema accordingly.

#### OpenAI
- `OPENAI_API_KEY`: required if provider is `openai`
- `OPENAI_BASE_URL`: optional (defaults to OpenAI)
- `OPENAI_EMBED_MODEL`: default to a current embeddings model available in your OpenAI account
- `OPENAI_LLM_MODEL`: default to a current chat model available in your OpenAI account
- Prefer OpenAI **Responses API** for new integrations.

---

### Architecture / code placement

#### Ports (domain)
- `EmbeddingModelPort`:
  - `embed_texts(texts: list[str]) -> list[list[float]]`
  - `embed_query(text: str) -> list[float]`
- `LLMClientPort`:
  - `generate(messages: list[LLMMessage], **params) -> LLMResult`

#### Adapters (implementations)
Create:
- `src/tfmkg/adapters/embeddings/ollama_embeddings.py`
- `src/tfmkg/adapters/llm/ollama_chat.py`
- `src/tfmkg/adapters/embeddings/openai_embeddings.py`
- `src/tfmkg/adapters/llm/openai_responses.py`

All adapters must:
- be pure clients (no retrieval logic)
- implement retries/backoff for transient network errors
- return typed results with minimal post-processing

#### Wiring (dependency injection)
- `apps/api/deps.py` chooses providers based on `*_PROVIDER` env vars.
- Retrieval pipeline stays unchanged: it calls ports, not concrete adapters.

---

### Ollama API requirements

#### Embeddings
- Implement embeddings generation using Ollama’s embeddings capability.
- Provide a single function that accepts a batch of texts and returns vectors.
- Handle: empty inputs, long text (truncate strategy or chunk pre-step), timeouts.
- Log basic timing and model used (but never log raw user prompts unless explicitly configured).

#### Chat generation
- Implement chat generation using Ollama’s API (either the native Ollama endpoints or its OpenAI-compat mode, but pick ONE for MVP and keep it consistent).
- Must support:
  - system + user messages
  - temperature (default low, e.g., 0.2)
  - max tokens / output length (reasonable default)

Note:
- Ollama has an OpenAI-compatibility mode that supports the OpenAI Responses API (non-stateful flavor). If you use it, document limitations (no conversation state via previous_response_id).

---

### Prompting: evidence-grounded answers (must implement)

Create a strict “Answer only from evidence” prompt template:
- If evidence is insufficient, respond with an abstention:
  - `"I don’t have enough information in the provided sources to answer that."`
- Always include a “Citations” section in output OR return citations in the response object (preferred: return citations separately as structured data).

The prompt must:
- instruct the model to not invent facts
- require that each claim is supported by the evidence pack
- encourage short, factual answers

---

### Telemetry updates (extend M6 behavior)
When `LLM_PROVIDER` or `EMBEDDINGS_PROVIDER` is enabled:
- Log in `runs`:
  - `llm_provider`, `llm_model`
  - `embeddings_provider`, `embed_model`
  - `prompt_tokens`, `completion_tokens` if available (OpenAI typically provides; Ollama may not)
  - `abstained` decision
- Keep `evidence_pack` stored, to allow auditing.

---

### Acceptance criteria

1) Ollama end-to-end
- With Ollama running locally (or as a service in compose), `/query` with `mode=hybrid`:
  - retrieves evidence
  - generates an answer via `mistral:7b`
  - returns non-empty `citations` referring to retrieved chunks

2) Evidence grounding
- If retriever returns 0 hits, the system **must abstain** (no hallucinated answers).
- If retriever returns hits, the answer must quote/reflect evidence and provide citations.

3) OpenAI switch
- Setting:
  - `LLM_PROVIDER=openai` uses OpenAI generation
  - `EMBEDDINGS_PROVIDER=openai` uses OpenAI embeddings
- Mixed mode must work (e.g., Ollama embeddings + OpenAI generation).

4) Reliability
- Requests fail gracefully with clear errors when:
  - Ollama is unreachable
  - required models are not available/pulled
  - OpenAI key is missing/invalid

5) Reproducibility
- Telemetry logs provider + model names for every run.
- The response shape from `/query` remains stable.

---

### Verification steps

#### Local Ollama (outside Docker)
1) Ensure Ollama is running.
2) Pull models:
   - `ollama pull embeddinggemma`
   - `ollama pull mistral:7b`
3) Run compose services (API+DB+Fuseki), pointing to your local Ollama:
   - set `OLLAMA_BASE_URL=http://host.docker.internal:11434` (Docker Desktop) or run Ollama as a compose service.
4) Test:
   - `curl -s -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{"question":"What is X?","mode":"hybrid","top_k":5}'`

#### OpenAI
1) Export `OPENAI_API_KEY`
2) Set providers:
   - `LLM_PROVIDER=openai`
   - `EMBEDDINGS_PROVIDER=openai`
3) Repeat the `/query` test.

Document any platform-specific Docker networking notes in README.

## M9 — Retrieval debug output + index stats (observability)

### Goal
Make retrieval transparent and measurable so you can debug relevance, grounding, and dataset drift.

### Implementation
1) Add `GET /stats/index`:
   - returns counts grouped by `source_type`
   - returns counts grouped by `dataset_version`
   - returns `embedding_dim` currently configured (and/or inferred from one stored row)
   - returns current providers/models (LLM + embeddings)

2) Add debug output to `/query` (no behavior change when debug is off):
   - Extend `QueryRequest` with `debug: bool = false`
   - When `debug=true`, `QueryResponse.debug` must include:
     - `retrieval_hits`: list of `{chunk_id, source_type, source_ref, score}`
     - `evidence_text`: the exact evidence pack string passed to the LLM
     - `dataset_version` used
     - `providers`: `{llm_provider, llm_model, embeddings_provider, embed_model}`

### Acceptance criteria
- `curl -s http://localhost:8000/stats/index` returns JSON with the fields above.
- `/query` works the same as before when `debug=false`.
- `/query` with `debug=true` includes hit IDs + scores and evidence text.

## M10 — Grounding + abstention test suite (hallucination control)

### Goal
Prove the system abstains when evidence is missing or insufficient, and always returns citations when it answers.

### Implementation
1) Enforce strict policy in RAG pipeline:
   - If `retrieval_hits` is empty ⇒ answer must be the configured abstention string; `abstained=true`
   - If evidence pack exists but does not support answering (simple heuristic ok for now):
     - e.g., evidence pack length < N chars or no named entities overlap ⇒ abstain
   - When not abstaining, return non-empty `citations`

2) Add tests:
   - Unit tests for the abstention policy function
   - Integration test calling `/query` against an empty pgvector table (or filtered dataset_version) must abstain

3) Add `eval/question_sets/trap_questions.jsonl` with 10+ “not in KG” questions.

### Acceptance criteria
- With empty index, `/query` returns `abstained=true` and abstention message.
- With non-empty index, answer includes citations for retrieved chunks.
- `pytest` passes.

## M11 — Minimal SPARQL adapter + KG retrieval mode (KG lane)

### Goal
Enable KG-native retrieval for `mode="kg"` using Fuseki SPARQL (no embeddings).

### Implementation
1) Create Fuseki SPARQL adapter:
   - `sparql(query: str) -> dict` (JSON results)
   - configurable dataset URL from settings

2) Implement KG retriever:
   - For MVP: for a user question, do one of:
     - A) keyword-based label search (SPARQL `FILTER CONTAINS(LCASE(STR(?label)), ...)`)
     - B) fallback to `SELECT ?s ?p ?o LIMIT 25` if search yields nothing
   - Build EvidencePack from returned triples, include URIs.

3) Wire `/query`:
   - if `mode="kg"` use KG retriever and skip vector store.

### Acceptance criteria
- `curl -s http://localhost:8000/kg/ping` (or equivalent) returns ok.
- `/query` with `mode="kg"` returns an answer derived from SPARQL evidence (or abstains).
- Debug output for mode kg includes returned triples and evidence pack.

## M12 — Indexing pipeline: KG → kg_text chunks → pgvector embeddings

### Goal
Generate embeddings into Postgres from your existing KG test data.

### Implementation
1) Implement CLI indexer (recommended):
   - `python -m tfmkg.scripts.index_kg --limit 200 --dataset_version dev`
   - Steps:
     - SPARQL: list distinct subject URIs
     - For each URI: fetch up to N triples + optional label
     - Build deterministic “entity card” text
     - Batch embed with embedding provider
     - Upsert into `chunks` as `source_type="kg_text"`

2) Ensure chunk IDs are stable:
   - `chunk_id = hash(dataset_version + uri + text_template_version + text)`

3) Handle embedding dimension:
   - Validate embedding length matches DB vector dimension
   - If mismatch: fail with explicit error telling which migration to apply

### Acceptance criteria
- Running the indexer inserts >0 rows into `chunks`:
  - `SELECT COUNT(*) FROM chunks WHERE source_type='kg_text';`
- `/query mode=hybrid debug=true` shows retrieved `kg_text` hits.
- Re-running indexer is idempotent (row count stable, updated timestamps optional).

## M13 — Text and table lanes (comparative benchmark plumbing)

### Goal
Enable `mode="text"` and `mode="table"` lanes so you can run KG vs text vs table comparisons.

### Implementation
1) Create synthetic `doc_text` and `table_row` datasets from KG (MVP):
   - `doc_text`: a paragraph-style description per entity (can reuse card but different formatting)
   - `table_row`: a single-line CSV-like row per entity: `uri|label|date|creator|place|...` (fill unknown with blank)

2) Extend indexer to optionally create these:
   - `--include_doc_text`
   - `--include_table_rows`

3) Wire retrieval:
   - `mode="text"` filters vector store `source_type="doc_text"`
   - `mode="table"` filters vector store `source_type="table_row"`
   - `mode="hybrid"` stays `kg_text` (for now)

### Acceptance criteria
- After indexing with both flags:
  - counts exist for `doc_text` and `table_row`
- `/query` returns different hits depending on mode (confirmed via debug).

## M14 — Benchmark runner + reproducibility package (CSV export)

### Goal
Run a repeatable benchmark across modes and export results for analysis.

### Implementation
1) Create `eval/scripts/run_benchmark.py`:
   - Reads JSONL of `{id, question}` from `eval/question_sets/*.jsonl`
   - For each question, runs modes: `kg`, `text`, `table`, `hybrid`
   - Calls `/query` (HTTP) with `debug=false` by default
   - Writes `eval/out/results.csv` with:
     - question_id, question, mode
     - answer, abstained
     - citations_count
     - latency_ms (from telemetry or response)
     - dataset_version
     - providers/models

2) Add at least two question sets:
   - `kg_questions.jsonl` (answerable from KG)
   - `trap_questions.jsonl` (should abstain)

3) Document exact commands in README.

### Acceptance criteria
- `python eval/scripts/run_benchmark.py` produces a CSV with rows = questions * 4 modes.
- Results include abstentions for trap questions in all modes.
- Telemetry rows exist for each run (if you log query runs centrally).