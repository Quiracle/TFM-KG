# AGENTS.md — Instructions for Codex (VS Code)

Codex: follow these rules whenever you work in this repository.

## Scope & workflow
- Work in small, reviewable diffs. Prefer 1 milestone per PR/commit.
- Do not refactor unrelated code. Avoid “cleanups” unless required for the task.
- If something is unclear, choose the simplest implementation that satisfies the milestone acceptance criteria.
- Keep module boundaries: domain ports/interfaces stay free of infrastructure details.

## Commands you must run (when applicable)
- Build & boot: `docker compose up --build -d`
- API smoke test: `curl -s http://localhost:8000/health`
- Tear down (if needed): `docker compose down -v`

If you add tests:
- Run them inside the API container:
  `docker compose exec api pytest -q`
(If pytest is not installed yet, add it explicitly and document in README.)

## Dependencies
- Do not introduce new services (no Qdrant/Redis/etc).
- Prefer Postgres + pgvector for embeddings and telemetry.
- If you add a Python dependency, add it to `pyproject.toml` and explain why in the PR summary.

## Data & database
- Do not change existing table names/columns unless a milestone requires it.
- Use the existing migrations approach (SQL under `migrations/`).
- Keep embedding dimension consistent with the schema unless explicitly updated everywhere.

## API behavior
- Keep response shapes stable for `/query` so the UI can depend on it.
- Always include `mode`, `top_k`, and (when available) `citations` and `debug`.

## Quality
- Prefer typed code and clear error messages.
- Add unit tests for pure logic (textifier/context builder).
- Add integration tests only when they’re cheap and stable (db upsert/search).