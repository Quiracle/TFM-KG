# RAGAS Evaluation Implementation Plan for TFM-KG

## Goal

Implement a reproducible evaluation pipeline that compares three systems on the same question set:

1. **LLM only**: Claude answers with no repository context.
2. **RAG**: existing `/query` endpoint using `mode="text"`.
3. **MCP**: Claude agent answers by using the existing MCP KG server tools.

The evaluation must:

- use **Claude API** for answer generation and MCP agent calls,
- use **a lightweight judge model** to score answers,
- use **RAGAS** for structured evaluation,
- preserve the current app architecture and API contracts,
- produce thesis-ready outputs: raw runs, scored runs, summary tables, and error slices.

---

## Repository Context You Must Respect

Before coding, inspect these files and derive your plan from them:

- `AGENTS.md`
- `README.md`
- `IMPLEMENT.md`
- `pyproject.toml`
- `apps/api/dependencies.py`
- `apps/api/routers/query.py`
- `apps/api/schemas/query.py`
- `src/tfmkg/core/evidence.py`
- `src/tfmkg/adapters/llm/*`
- `mcp_kg_server/**`
- `eval/question_sets/trap_questions.jsonl`

Important repo constraints:

- Work in **small, reviewable diffs**.
- Do **not** introduce new infrastructure services.
- Preserve `/query` response shape.
- Keep module boundaries clean.
- Prefer integration with existing adapters and settings instead of creating parallel stacks.

---

## Key Design Decisions

### 1) Keep the answering model fixed across systems

Use the same main answering model for all three systems so the comparison isolates the retrieval/tooling effect.

Recommended:

- **Answer model**: `claude-sonnet-4-6`
- **Judge model**: `claude-haiku-4-5-20251001`

### 2) Use one strict answer contract

All systems must follow the same answer policy.

Canonical abstain string:

```text
ABSTAIN
```

This is intentionally stricter than the current `/query` abstain message. Normalize all evaluation-facing outputs to this exact string.

### 3) Make MCP evaluation deterministic enough to benchmark

For MCP runs, do not evaluate free-form editor behavior. Implement a controlled evaluation runner that:

- starts from the question,
- exposes the MCP tools to Claude,
- allows tool-use loops up to a fixed max iteration count,
- captures every tool call,
- ends when Claude returns a final answer or the loop budget is exhausted.

Suggested limits:

- `max_iterations = 6`
- `max_tool_calls_per_turn = 4`
- `temperature = 0`

### 4) Use RAGAS in a practical way

Use RAGAS for:

- **binary correctness judge** via a custom discrete / aspect-style metric,
- **faithfulness** for RAG and MCP runs,
- **context precision** for RAG and MCP runs,
- **context recall** for RAG and MCP runs.

For answer correctness, prefer this staged approach:

- **Phase 1**: implement binary correctness with an LLM judge first.
- **Phase 2**: optionally add `AnswerCorrectness` if you are willing to add an embeddings dependency for semantic similarity.

This avoids blocking the project on embeddings setup while still using RAGAS effectively.

---

## What To Implement

## Milestone 0 — Repo reading and plan freeze

### Goal

Understand the current code paths and identify the minimum-change integration points.

### Tasks

- Read the files listed above.
- Confirm how `/query` currently behaves for `kg`, `text`, `table`, and `hybrid`.
- Confirm current LLM adapter structure and settings flow.
- Confirm how the MCP server is started and how tools are registered.
- Produce a short implementation note before coding:
  - affected files,
  - risks,
  - assumptions,
  - milestone order.

### Acceptance criteria

- You can explain, in repo-specific terms, how each of the 3 benchmarked systems will run.
- You identify where Anthropic support must be added.

---

## Milestone 1 — Add Anthropic LLM support

### Goal

Allow the project to use Claude through the same adapter abstraction used by the rest of the app.

### Tasks

Add Anthropic support to the existing LLM adapter layer.

Expected changes:

- `pyproject.toml`
- `src/tfmkg/adapters/llm/anthropic_messages.py` (or similar name)
- `src/tfmkg/adapters/llm/__init__.py`
- `src/tfmkg/core/config.py`
- `apps/api/dependencies.py`
- `.env.example`

### Requirements

- Support standard text generation through the existing `LLMClientPort`.
- Reuse the existing `LLMMessage` / `LLMResult` contract.
- Read provider/model/api key from settings.
- Preserve OpenAI and Ollama behavior.
- Include retries and clear runtime errors.

### Suggested settings

Add:

- `anthropic_api_key`
- `anthropic_base_url` (optional; default official API URL)
- `anthropic_llm_model`

### Acceptance criteria

- Setting `LLM_PROVIDER=anthropic` works for normal generation.
- Existing providers still work.
- No unrelated refactors.

---

## Milestone 2 — Add evaluation package structure

### Goal

Create a clean home for benchmark artifacts and scripts.

### Tasks

Create or extend:

```text
eval/
  datasets/
    core_eval_v1.jsonl
    trap_questions.jsonl
  prompts/
    answer_no_context_v1.txt
    answer_with_context_v1.txt
    mcp_agent_v1.txt
    judge_correctness_v1.txt
  runners/
    run_no_context.py
    run_rag_text.py
    run_mcp_agent.py
    common.py
  scoring/
    ragas_metrics.py
    judge_binary.py
    normalize.py
  reports/
    summarize_results.py
  outputs/
    raw/
    scored/
    reports/
```

### Acceptance criteria

- The package structure exists.
- Scripts are runnable from the repo root.
- Paths are stable and documented.

---

## Milestone 3 — Define the gold evaluation dataset schema

### Goal

Create the dataset that drives all benchmark runs taking into account the knowledge graph structure explained in docs/amsterdam_museum_kg_guide.md.

### JSONL schema

Each row should follow this schema:

```json
{
  "id": "q-001",
  "question": "Who created object X?",
  "reference_answer": "Object X was created by Jane Doe.",
  "reference_points": [
    "creator = Jane Doe"
  ],
  "expected_abstain": false,
  "question_type": "single_hop",
  "difficulty": "easy",
  "notes": "Accept J. Doe as equivalent.",
  "supporting_refs": ["uri:...", "chunk:..."]
}
```

Trap question rows should still include the same fields, with:

- `reference_answer = "ABSTAIN"`
- `expected_abstain = true`

### Dataset composition

Start with around **60–80 questions** total:

- single-hop factual questions,
- multi-hop questions,
- comparison / aggregation questions,
- unanswerable trap questions.

### Acceptance criteria

- Dataset schema is documented.
- At least 10 seed examples exist so the pipeline can be run immediately.
- Existing trap questions are normalized into the richer schema.

---

## Milestone 4 — Standardize prompts

### Goal

Make all systems answer under the same rules.

### Prompt A — no context

Use for the LLM-only baseline.

```text
You are answering factual questions.
Return a short factual answer.
If you are not sure, answer exactly: ABSTAIN
Do not invent facts.
```

### Prompt B — retrieved-context answerer

Use for RAG and any non-agent context-based answer generation.

```text
You are answering factual questions using only the provided evidence.
Return a short factual answer.
If the evidence is insufficient, answer exactly: ABSTAIN
Do not invent facts.
Do not use outside knowledge.
```

### Prompt C — MCP agent

Use for the Claude tool-using MCP runner.

```text
You are an evaluation agent answering questions about a knowledge graph.
You may use the provided tools to gather evidence.
Rules:
- Use tools only when needed.
- Prefer the smallest number of tool calls.
- If the answer cannot be supported, return exactly: ABSTAIN
- Do not invent facts.
- Final answer must be short and factual.
```

### Acceptance criteria

- Prompts live in files, not hardcoded strings only.
- A shared normalizer maps abstain variants to `ABSTAIN`.

---

## Milestone 5 — Implement the 3 runners

### Goal

Generate raw benchmark outputs for each system in a common format.

### Common raw output schema

Each run should emit JSONL rows like:

```json
{
  "question_id": "q-001",
  "system": "rag_text",
  "model": "claude-sonnet-4-6",
  "question": "...",
  "answer": "...",
  "normalized_answer": "...",
  "abstained": false,
  "latency_ms": 812,
  "retrieved_contexts": ["..."],
  "citations": [],
  "tool_trace": [],
  "meta": {}
}
```

### Runner 1 — `run_no_context.py`

Behavior:

- Load dataset rows.
- Send only the question and no-context prompt to Claude.
- Normalize output.
- Save raw JSONL.

### Runner 2 — `run_rag_text.py`

Behavior:

- Call the existing API `/query` with:
  - `mode="text"`
  - `debug=true`
- Capture:
  - answer,
  - abstained,
  - citations,
  - retrieved evidence from debug payload when available,
  - latency.
- Normalize output.
- Save raw JSONL.

### Runner 3 — `run_mcp_agent.py`

Behavior:

- Start or connect to the MCP KG server.
- Discover tool schemas.
- Run Claude in a tool loop:
  - send question + MCP prompt,
  - execute requested tools,
  - append tool results,
  - continue until final answer or max iterations.
- Record tool trace in detail.
- Normalize output.
- Save raw JSONL.

### Acceptance criteria

- Each runner can process a 5-question smoke dataset.
- All runners write the same raw output schema.
- Failures are logged clearly and do not silently corrupt outputs.

---

## Milestone 6 — Implement the lightweight correctness judge

### Goal

Score whether each model answer is correct or not using a cheap, strict judge.

### Judge model

Use:

- `claude-haiku-4-5-20251001`

### Judge prompt

Create `eval/prompts/judge_correctness_v1.txt`:

```text
You are grading answers for a factual QA benchmark.

Return JSON only:
{
  "verdict": 0,
  "reason": "",
  "missing_points": [],
  "extra_claims": []
}

Rules:
- verdict = 1 only if the answer is correct enough.
- Treat ABSTAIN as correct only when expected_abstain=true.
- Any unsupported extra factual claim should make verdict = 0.
- Minor wording differences are acceptable.
- Judge based on the reference answer and reference_points.

Question: {question}
Expected abstain: {expected_abstain}
Reference answer: {reference_answer}
Reference points: {reference_points}
Model answer: {model_answer}
```

### Tasks

- Implement judge script that reads raw outputs and dataset rows.
- Produce scored JSONL / CSV with:
  - `binary_correct`
  - `judge_reason`
  - `missing_points`
  - `extra_claims`

### Acceptance criteria

- Judge runs on outputs from all three systems.
- Invalid judge JSON is retried or surfaced cleanly.

---

## Milestone 7 — Integrate RAGAS metrics

### Goal

Add reusable RAGAS scoring for retrieval-grounded systems.

### Metrics to implement first

For **RAG** and **MCP**:

- `Faithfulness`
- `ContextPrecision`
- `ContextRecall`

For **all systems**:

- one custom binary correctness metric wrapper, or run the external judge and merge into the final report.

### Optional later metric

- `AnswerCorrectness` if embeddings are added.

### Implementation notes

- Use the current RAGAS API that matches the installed version.
- If using RAGAS v0.4+, prefer its experiment-oriented workflow.
- Keep the implementation minimal and documented.
- If `AnswerCorrectness` is added, explicitly wire an embedding model instead of assuming Anthropic provides one.

### Acceptance criteria

- RAGAS scoring works on a smoke dataset.
- Retrieval-based metrics only run when retrieved contexts are available.
- Missing contexts produce explicit skip reasons, not crashes.

---

## Milestone 8 — Judge alignment pass

### Goal

Make the lightweight judge stable enough for thesis use.

### Tasks

- Create a small manually labeled set of around 20–30 examples.
- Compare judge output vs manual label.
- Refine the judge prompt once or twice.
- Freeze prompt as `judge_correctness_v1`.

### Acceptance criteria

- Judge prompt is no longer changing between main experiment runs.
- Alignment notes are written to a short markdown file.

---

## Milestone 9 — Reporting scripts

### Goal

Generate thesis-ready summaries from the scored outputs.

### Required outputs

Produce:

1. **System comparison table**
   - correctness rate
   - abstain accuracy
   - hallucination rate
   - mean latency

2. **Retrieval diagnostics table** for RAG and MCP
   - faithfulness
   - context precision
   - context recall
   - tool calls per answer for MCP

3. **Error slices**
   - single-hop
   - multi-hop
   - trap questions
   - abstain cases

4. **Exportable CSV or markdown tables**

### Acceptance criteria

- A single command can generate a report from scored outputs.
- Report files are written under `eval/outputs/reports/`.

---

## Implementation Order

1. Anthropic adapter.
2. Eval folder structure.
3. Dataset schema and seed dataset.
4. Prompt files.
5. No-context runner.
6. RAG runner.
7. MCP runner.
8. Judge.
9. RAGAS metrics.
10. Reports.

This order ensures you can get usable benchmark data early.

---

## Non-Goals

Do **not** do these unless required later:

- no UI work,
- no new database services,
- no major `/query` refactor,
- no synthetic dataset generation pipeline,
- no attempt to benchmark free-form Codex/VS Code behavior.

---

## Suggested Commands

Use or adapt existing project conventions.

Examples:

```bash
docker compose up --build -d
curl -s http://localhost:8000/health
python eval/runners/run_no_context.py --dataset eval/datasets/core_eval_v1.jsonl
python eval/runners/run_rag_text.py --dataset eval/datasets/core_eval_v1.jsonl
python eval/runners/run_mcp_agent.py --dataset eval/datasets/core_eval_v1.jsonl
python eval/scoring/judge_binary.py --raw eval/outputs/raw/
python eval/scoring/ragas_metrics.py --raw eval/outputs/raw/
python eval/reports/summarize_results.py --scored eval/outputs/scored/
```

---

## Expected Final Deliverables

By the end of this implementation, the repository should contain:

- Anthropic provider support,
- a reusable evaluation dataset schema,
- prompt files,
- 3 benchmark runners,
- a lightweight correctness judge,
- RAGAS metrics integration,
- report generation scripts,
- smoke-test documentation.

---

## First Execution Target

Before building the full benchmark, get this narrow path working end-to-end:

- 5 questions only,
- all 3 runners,
- binary judge,
- one summary table.

Only then expand to the full dataset.

