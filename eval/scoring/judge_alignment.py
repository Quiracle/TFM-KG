from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.runners.common import (
    append_jsonl_row,
    ensure_output_dirs,
    init_jsonl_output,
    load_jsonl,
    load_text_file,
    resolve_repo_path,
)
from eval.scoring.judge_binary import _build_judge_client, _judge_answer_with_retry, _render_prompt
from src.tfmkg.core.config import settings

CSV_FIELDS = (
    "id",
    "case_tag",
    "manual_verdict",
    "judge_verdict",
    "agreement",
    "error_class",
    "judge_reason",
    "manual_reason",
    "judge_attempts",
    "judge_error",
    "question",
    "model_answer",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run judge alignment against a manually labeled set.")
    parser.add_argument(
        "--alignment",
        default="eval/datasets/judge_alignment_v1.jsonl",
        help="Path to manual alignment JSONL rows.",
    )
    parser.add_argument(
        "--prompt",
        default="eval/prompts/judge_correctness_v1.txt",
        help="Judge prompt template path.",
    )
    parser.add_argument(
        "--output-jsonl",
        "--output",
        default="eval/outputs/scored/judge_alignment_scored.jsonl",
        dest="output_jsonl",
        help="Path to write alignment row results JSONL.",
    )
    parser.add_argument(
        "--output-csv",
        default="eval/outputs/scored/judge_alignment_scored.csv",
        help="Path to write alignment row results CSV.",
    )
    parser.add_argument(
        "--report",
        default="eval/outputs/reports/judge_alignment_report.md",
        help="Path to write alignment summary markdown report.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Judge provider override. Defaults to JUDGE_PROVIDER from .env.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Judge model override. Defaults to JUDGE_MODEL from .env.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge sampling temperature.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=220,
        help="Max output tokens per judge call.",
    )
    parser.add_argument(
        "--max-judge-attempts",
        type=int,
        default=3,
        help="Retry attempts when judge output is invalid.",
    )
    return parser.parse_args()


def _normalize_manual_verdict(value: Any) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        raise ValueError("manual_verdict must be 0 or 1.") from None
    if normalized not in {0, 1}:
        raise ValueError("manual_verdict must be 0 or 1.")
    return normalized


def _validate_alignment_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required_fields = {
        "id",
        "question",
        "reference_answer",
        "reference_points",
        "expected_abstain",
        "model_answer",
        "manual_verdict",
    }
    validated: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise ValueError(f"Alignment row {idx} is missing fields: {missing}")

        normalized = dict(row)
        normalized["id"] = str(row.get("id", "")).strip() or f"alignment-{idx:03d}"
        normalized["question"] = str(row.get("question", "")).strip()
        normalized["reference_answer"] = str(row.get("reference_answer", "")).strip()
        normalized["model_answer"] = str(row.get("model_answer", "")).strip()
        normalized["manual_reason"] = str(row.get("manual_reason", "")).strip()
        normalized["case_tag"] = str(row.get("case_tag", "")).strip()
        normalized["expected_abstain"] = bool(row.get("expected_abstain", False))
        if not isinstance(row.get("reference_points"), list):
            raise ValueError(f"Alignment row {normalized['id']} must provide reference_points as a list.")
        normalized["manual_verdict"] = _normalize_manual_verdict(row.get("manual_verdict"))
        validated.append(normalized)
    return validated


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _write_alignment_report(path: Path, *, summary: dict[str, Any], mismatches: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Judge Alignment Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Prompt: `{summary['prompt_path']}`")
    lines.append(f"- Model: `{summary['model']}`")
    lines.append(f"- Alignment rows: {summary['total_rows']}")
    lines.append(f"- Judge returned verdict rows: {summary['scored_rows']}")
    lines.append(f"- Agreement count: {summary['agreement_count']}")
    lines.append(f"- Agreement rate (scored rows): {summary['agreement_rate_scored']:.3f}")
    lines.append(f"- False positives (judge=1, manual=0): {summary['false_positives']}")
    lines.append(f"- False negatives (judge=0, manual=1): {summary['false_negatives']}")
    lines.append(f"- Judge errors (no verdict): {summary['judge_errors']}")
    lines.append("")
    lines.append("## Mismatch Samples")
    lines.append("")
    if not mismatches:
        lines.append("No mismatches found.")
    else:
        for row in mismatches[:10]:
            lines.append(f"- `{row['id']}` (`{row.get('error_class', '')}`): {row.get('judge_reason', '')}")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    alignment_path = resolve_repo_path(args.alignment)
    prompt_path = resolve_repo_path(args.prompt)
    output_jsonl_path = resolve_repo_path(args.output_jsonl)
    output_csv_path = resolve_repo_path(args.output_csv)
    report_path = resolve_repo_path(args.report)

    if args.max_judge_attempts <= 0:
        raise SystemExit("--max-judge-attempts must be > 0")
    if args.max_output_tokens <= 0:
        raise SystemExit("--max-output-tokens must be > 0")
    if not alignment_path.exists():
        raise SystemExit(f"Alignment dataset not found: {alignment_path}")
    if not prompt_path.exists():
        raise SystemExit(f"Prompt not found: {prompt_path}")

    judge_provider = str(args.provider or settings.judge_provider).strip().lower()
    judge_model = str(args.model or settings.judge_model).strip()
    if not judge_model:
        raise SystemExit("JUDGE_MODEL is required.")

    alignment_rows = _validate_alignment_rows(load_jsonl(alignment_path))
    prompt_template = load_text_file(prompt_path)

    if len(alignment_rows) < 20 or len(alignment_rows) > 30:
        print(
            f"[judge_alignment] Warning: expected 20-30 rows, got {len(alignment_rows)}",
            file=sys.stderr,
        )

    ensure_output_dirs()
    init_jsonl_output(output_jsonl_path)

    judge_client = _build_judge_client(judge_provider, judge_model)

    results: list[dict[str, Any]] = []
    agreement_count = 0
    scored_rows = 0
    false_positives = 0
    false_negatives = 0
    judge_errors = 0

    for row in alignment_rows:
        prompt = _render_prompt(prompt_template, row, row["model_answer"])
        decision = _judge_answer_with_retry(
            client=judge_client,
            prompt=prompt,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            max_attempts=args.max_judge_attempts,
        )
        judge_verdict = decision.binary_correct
        manual_verdict = row["manual_verdict"]
        agreement = None if judge_verdict is None else judge_verdict == manual_verdict

        if judge_verdict is None:
            error_class = "judge_error"
            judge_errors += 1
        elif agreement:
            error_class = "match"
            agreement_count += 1
            scored_rows += 1
        else:
            scored_rows += 1
            if judge_verdict == 1 and manual_verdict == 0:
                error_class = "false_positive"
                false_positives += 1
            else:
                error_class = "false_negative"
                false_negatives += 1

        result_row = {
            "id": row["id"],
            "case_tag": row.get("case_tag", ""),
            "question": row["question"],
            "reference_answer": row["reference_answer"],
            "reference_points": row["reference_points"],
            "expected_abstain": row["expected_abstain"],
            "model_answer": row["model_answer"],
            "manual_verdict": manual_verdict,
            "manual_reason": row.get("manual_reason", ""),
            "judge_verdict": judge_verdict,
            "agreement": agreement,
            "error_class": error_class,
            "judge_reason": decision.judge_reason,
            "judge_attempts": decision.attempts,
            "judge_error": decision.judge_error,
        }
        results.append(result_row)
        append_jsonl_row(output_jsonl_path, result_row)

    _write_csv(output_csv_path, results)

    summary = {
        "total_rows": len(results),
        "scored_rows": scored_rows,
        "agreement_count": agreement_count,
        "agreement_rate_scored": (agreement_count / scored_rows) if scored_rows else 0.0,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "judge_errors": judge_errors,
        "prompt_path": str(prompt_path),
        "model": judge_model,
    }
    mismatches = [row for row in results if row.get("error_class") in {"false_positive", "false_negative"}]
    _write_alignment_report(report_path, summary=summary, mismatches=mismatches)

    print(f"[judge_alignment] Rows={len(results)} scored={scored_rows} agreement={agreement_count}")
    print(f"[judge_alignment] Judge provider: {judge_provider}")
    print(f"[judge_alignment] Judge model: {judge_model}")
    print(f"[judge_alignment] JSONL: {output_jsonl_path}")
    print(f"[judge_alignment] CSV: {output_csv_path}")
    print(f"[judge_alignment] Report: {report_path}")


if __name__ == "__main__":
    main()
