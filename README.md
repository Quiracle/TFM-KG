# tfm-kg-rag (starter skeleton)

Starter monorepo for a KG + RAG prototype:
- Apache Jena Fuseki (SPARQL triplestore)
- Postgres + pgvector (vector store + telemetry)
- FastAPI (API)

## Quickstart
1) Copy env file:
```bash
cp .env.example .env
```

2) Run:
```bash
docker compose up --build
```

3) Verify:
- API health: http://localhost:8000/health
- Fuseki UI: http://localhost:3030
- Postgres: localhost:5432

## Notes
This is a scaffold: endpoints are wired, but retrieval/indexing are stubs to fill in.
