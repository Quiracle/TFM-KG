# Evaluation Package

This directory contains the benchmark pipeline structure and scripts.

## Stable paths

- Datasets: `eval/datasets/`
- Dataset schema reference: `eval/datasets/README.md`
- Prompts: `eval/prompts/`
- Runners: `eval/runners/`
- Scoring: `eval/scoring/`
  - Shared abstain normalizer: `eval/scoring/normalize.py`
  - Binary correctness judge: `eval/scoring/judge_binary.py`
  - Judge alignment pass: `eval/scoring/judge_alignment.py`
- Reports: `eval/reports/`
- Outputs:
  - raw runs: `eval/outputs/raw/`
  - scored runs: `eval/outputs/scored/`
  - summaries: `eval/outputs/reports/`

## Scaffold commands (from repo root)

```bash
python eval/runners/run_no_context.py --dataset eval/datasets/core_eval_v1.jsonl
python eval/runners/run_rag_text.py --dataset eval/datasets/core_eval_v1.jsonl
python eval/runners/run_mcp_agent.py --dataset eval/datasets/core_eval_v1.jsonl
python eval/scoring/judge_binary.py --raw eval/outputs/raw --dataset eval/datasets/core_eval_v1.jsonl
python eval/scoring/judge_alignment.py --alignment eval/datasets/judge_alignment_v1.jsonl
python eval/scoring/ragas_metrics.py --raw eval/outputs/raw --dataset eval/datasets/core_eval_v1.jsonl
python eval/reports/summarize_results.py --scored eval/outputs/scored
```

`judge_binary.py` now writes:

- `eval/outputs/scored/binary_judge_scored.jsonl`
- `eval/outputs/scored/binary_judge_scored.csv`

`ragas_metrics.py` writes:

- `eval/outputs/scored/ragas_metrics_scored.jsonl`
- `eval/outputs/scored/ragas_metrics_scored.csv`

Notes:

- `run_rag_text.py` calls `/query` with `mode="hybrid"` and forwards the selected prompt file as `system_prompt`.
- Retrieval metrics run only for `rag_text` and `mcp_agent` rows with non-empty `retrieved_contexts`.
- Rows without retrieval context are kept with explicit `ragas_skip_reason`.
- If present, binary judge fields are merged from `eval/outputs/scored/binary_judge_scored.jsonl`.

`judge_alignment.py` writes:

- `eval/outputs/scored/judge_alignment_scored.jsonl`
- `eval/outputs/scored/judge_alignment_scored.csv`
- `eval/outputs/reports/judge_alignment_report.md`

Frozen prompt notes:

- `eval/reports/judge_alignment_notes_v1.md`

`summarize_results.py` writes:

- `eval/outputs/reports/summary.md`
- `eval/outputs/reports/system_comparison.csv`
- `eval/outputs/reports/retrieval_diagnostics.csv`
- `eval/outputs/reports/error_slices.csv`

Other scripts may still be scaffolds depending on the current milestone.
