-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Chunks table: stores text + embeddings + metadata
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  source_type TEXT NOT NULL,          -- kg_text | doc_text | table_row
  source_ref TEXT NOT NULL,           -- URI / doc id / table id
  dataset_version TEXT NOT NULL,
  text TEXT NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  embedding vector(384)               -- adjust dimension to your embedding model
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_chunks_source_type ON chunks(source_type);
CREATE INDEX IF NOT EXISTS idx_chunks_dataset_version ON chunks(dataset_version);

-- Optional: IVF flat index (requires ANALYZE and enough rows to be meaningful)
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivfflat
--   ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Telemetry table: logs each query run for evaluation/benchmarking
CREATE TABLE IF NOT EXISTS runs (
  run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  question TEXT NOT NULL,
  mode TEXT NOT NULL,                 -- kg | text | table | hybrid
  top_k INT NOT NULL,
  retrieved_chunk_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
  evidence_pack JSONB NOT NULL DEFAULT '{}'::jsonb,
  answer TEXT NOT NULL,
  abstained BOOLEAN NOT NULL DEFAULT FALSE,
  latency_ms INT NOT NULL DEFAULT 0,
  dataset_version TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(ts);
CREATE INDEX IF NOT EXISTS idx_runs_mode ON runs(mode);

-- gen_random_uuid() lives in pgcrypto
CREATE EXTENSION IF NOT EXISTS pgcrypto;
