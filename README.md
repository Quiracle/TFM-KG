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

## KG Indexing (Test Dataset)
Use the indexing script to move KG entities from Fuseki into `chunks` as `source_type=kg_text`.

1) Start services:
```bash
docker compose up --build -d
```

2) (Optional) Load `data/example.ttl` into the Fuseki `kg` dataset.

3) Run indexing in the API container:
```bash
docker compose exec api python src/tfmkg/scripts/index_kg.py --limit 200 --triples-per-entity 50 --batch-size 16
```

4) Verify rows were written:
```bash
docker compose exec postgres psql -U tfmkg -d tfmkg -c "select source_type, count(*) from chunks group by source_type;"
```

The script builds deterministic KG-aware entity cards for hybrid retrieval. Artwork proxy cards resolve maker blank nodes, thesaurus labels, dimensions, locations, acquisition fields, and bounded related-artwork summaries; person cards include maker-work counts and small sample work lists. Dense or infrastructural relations are kept out of the card unless they support retrieval, then the cards are embedded in batches (Ollama `embeddinggemma` by default) and upserted by `chunk_id`.

If a long indexing run is interrupted after some `kg_text` chunks have already been written, resume without recomputing those rows:
```bash
docker compose exec api python src/tfmkg/scripts/index_kg.py --limit 0 --batch-size 16 --dataset-version dev --skip-existing --embedding-retries 5
```
