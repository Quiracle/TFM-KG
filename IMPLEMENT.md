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

## M8 — Optional: real LLM integration (separate PR)

### Goal
Swap deterministic answer for LLM generation with a strict “use evidence only” prompt.

### Acceptance criteria
- If no evidence: model abstains (“not enough information”)
- Every non-trivial claim is supported by citations (at least chunk-level)
- Telemetry logs prompt params + abstain decision