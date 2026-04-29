# Dataset Schema (Milestone 3)

Each JSONL row must follow this structure:

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
  "supporting_refs": [
    "uri:...",
    "chunk:..."
  ]
}
```

## Field notes

- `id`: stable unique identifier for a question.
- `question`: natural language query shown to each system.
- `reference_answer`: canonical expected answer text.
- `reference_points`: minimal factual points required for correctness.
- `expected_abstain`: whether the correct output is `ABSTAIN`.
- `question_type`: benchmark slice label (for example `single_hop`, `multi_hop`, `comparison`, `aggregation`, `trap`).
- `difficulty`: benchmark difficulty label (`easy`, `medium`, `hard`).
- `notes`: evaluator guidance and accepted variants.
- `supporting_refs`: provenance hints such as URIs/chunk IDs when available.

## Trap question rule

Trap rows use:

- `reference_answer = "ABSTAIN"`
- `expected_abstain = true`

## Judge alignment set

`judge_alignment_v1.jsonl` extends the same core fields with manual labels for prompt-alignment checks:

- `model_answer`: candidate answer judged by the lightweight judge.
- `manual_verdict`: manually assigned binary label (`0` or `1`).
- `manual_reason`: short note explaining the manual label.
- `case_tag`: optional category for mismatch analysis.
