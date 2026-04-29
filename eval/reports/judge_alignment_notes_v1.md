# Judge Alignment Notes (v1)

Date frozen: 2026-04-20  
Prompt file: `eval/prompts/judge_correctness_v1.txt`

## Scope

- Manual alignment set: `eval/datasets/judge_alignment_v1.jsonl` (24 rows).
- Objective: compare judge verdicts against manual binary labels and inspect mismatch types.

## Procedure

1. Run alignment scoring:
   `python eval/scoring/judge_alignment.py --alignment eval/datasets/judge_alignment_v1.jsonl`
2. Review outputs:
   - `eval/outputs/scored/judge_alignment_scored.jsonl`
   - `eval/outputs/scored/judge_alignment_scored.csv`
   - `eval/outputs/reports/judge_alignment_report.md`

## Prompt refinement applied before freeze

- Clarified abstain handling for answerable questions.
- Clarified that mixed correct + unsupported claims must fail.
- Explicitly mapped `missing_points` and `extra_claims` usage.

## Latest alignment run status

- Container smoke run executed on 2026-04-20 with:
  `python eval/scoring/judge_alignment.py --alignment eval/datasets/judge_alignment_v1.jsonl --max-judge-attempts 1`
- Output report was generated successfully.
- Judge verdicts were unavailable in that run (24 judge errors), indicating Anthropic key/auth configuration must be fixed before final calibration statistics are recorded.

This prompt version is frozen as `judge_correctness_v1` for main experiment runs.
