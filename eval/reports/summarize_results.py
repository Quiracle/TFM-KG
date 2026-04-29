from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.runners.common import load_jsonl, resolve_repo_path

SYSTEM_COMPARISON_HEADERS = (
    "system",
    "total_rows",
    "scored_rows",
    "correctness_rate",
    "abstain_accuracy",
    "hallucination_rate",
    "mean_latency_ms",
)

RETRIEVAL_HEADERS = (
    "system",
    "rows_with_context",
    "rows_with_ragas_scores",
    "faithfulness",
    "context_precision",
    "context_recall",
    "tool_calls_per_answer",
)

ERROR_SLICE_HEADERS = (
    "slice",
    "system",
    "total_cases",
    "judged_cases",
    "incorrect_cases",
    "error_rate",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation report tables from scored outputs.")
    parser.add_argument(
        "--scored",
        default="eval/outputs/scored",
        help="Directory containing scored outputs.",
    )
    parser.add_argument(
        "--output-md",
        "--output",
        default="eval/outputs/reports/summary.md",
        dest="output_md",
        help="Path to write report output.",
    )
    parser.add_argument(
        "--output-system-csv",
        default="eval/outputs/reports/system_comparison.csv",
        help="Path to write system comparison CSV.",
    )
    parser.add_argument(
        "--output-retrieval-csv",
        default="eval/outputs/reports/retrieval_diagnostics.csv",
        help="Path to write retrieval diagnostics CSV.",
    )
    parser.add_argument(
        "--output-slices-csv",
        default="eval/outputs/reports/error_slices.csv",
        help="Path to write error slices CSV.",
    )
    return parser.parse_args()


def _coerce_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def _coerce_binary_or_none(value: Any) -> int | None:
    try:
        num = int(value)
    except (TypeError, ValueError):
        return None
    if num not in {0, 1}:
        return None
    return num


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return []


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _format_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.3f}"


def _load_scored_inputs(scored_dir: Path) -> tuple[list[dict[str, Any]], bool, bool]:
    ragas_path = scored_dir / "ragas_metrics_scored.jsonl"
    judge_path = scored_dir / "binary_judge_scored.jsonl"

    has_ragas = ragas_path.exists()
    has_judge = judge_path.exists()

    if has_ragas:
        rows = load_jsonl(ragas_path)
    elif has_judge:
        rows = load_jsonl(judge_path)
    else:
        raise SystemExit(
            f"No scored inputs found. Expected at least one of: {ragas_path} or {judge_path}"
        )

    if has_judge:
        judge_rows = load_jsonl(judge_path)
        judge_index: dict[tuple[str, str], dict[str, Any]] = {}
        for row in judge_rows:
            key = (str(row.get("question_id", "")).strip(), str(row.get("system", "")).strip())
            if key[0] and key[1]:
                judge_index[key] = row

        for row in rows:
            key = (str(row.get("question_id", "")).strip(), str(row.get("system", "")).strip())
            judge_row = judge_index.get(key)
            if not judge_row:
                continue
            if row.get("binary_correct") is None:
                row["binary_correct"] = judge_row.get("binary_correct")
            if not row.get("judge_reason"):
                row["judge_reason"] = judge_row.get("judge_reason")
            if not row.get("missing_points"):
                row["missing_points"] = judge_row.get("missing_points", [])
            if not row.get("extra_claims"):
                row["extra_claims"] = judge_row.get("extra_claims", [])

    return rows, has_ragas, has_judge


def _group_by_system(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        system = str(row.get("system", "")).strip() or "unknown"
        grouped.setdefault(system, []).append(row)
    return grouped


def _build_system_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_by_system(rows)
    table: list[dict[str, Any]] = []

    for system in sorted(grouped):
        system_rows = grouped[system]
        binary = [_coerce_binary_or_none(row.get("binary_correct")) for row in system_rows]
        scored = [value for value in binary if value is not None]
        correctness_rate = _mean([float(value) for value in scored])

        abstain_eval_rows = [row for row in system_rows if bool(row.get("expected_abstain", False))]
        abstain_accuracy = _mean(
            [
                1.0 if bool(row.get("abstained", False)) else 0.0
                for row in abstain_eval_rows
            ]
        )

        hallucinations = []
        for row in system_rows:
            extra_claims = _coerce_string_list(row.get("extra_claims", []))
            expected_abstain = bool(row.get("expected_abstain", False))
            abstained = bool(row.get("abstained", False))
            hallucination = bool(extra_claims) or (expected_abstain and not abstained)
            hallucinations.append(1.0 if hallucination else 0.0)
        hallucination_rate = _mean(hallucinations)

        latencies = [
            float(_coerce_int(row.get("latency_ms", 0), default=0))
            for row in system_rows
        ]
        mean_latency_ms = _mean(latencies)

        table.append(
            {
                "system": system,
                "total_rows": len(system_rows),
                "scored_rows": len(scored),
                "correctness_rate": correctness_rate,
                "abstain_accuracy": abstain_accuracy,
                "hallucination_rate": hallucination_rate,
                "mean_latency_ms": mean_latency_ms,
            }
        )

    return table


def _build_retrieval_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_by_system(rows)
    table: list[dict[str, Any]] = []

    for system in ("rag_text", "mcp_agent"):
        system_rows = grouped.get(system, [])
        rows_with_context = [
            row for row in system_rows if len(_coerce_string_list(row.get("retrieved_contexts", []))) > 0
        ]
        faithfulness = [
            value
            for value in (_coerce_float_or_none(row.get("faithfulness")) for row in rows_with_context)
            if value is not None
        ]
        context_precision = [
            value
            for value in (_coerce_float_or_none(row.get("context_precision")) for row in rows_with_context)
            if value is not None
        ]
        context_recall = [
            value
            for value in (_coerce_float_or_none(row.get("context_recall")) for row in rows_with_context)
            if value is not None
        ]
        tool_calls_per_answer = _mean(
            [float(len(row.get("tool_trace", []) or [])) for row in system_rows]
        )
        rows_with_scores = 0
        for row in rows_with_context:
            if any(
                _coerce_float_or_none(row.get(metric_key)) is not None
                for metric_key in ("faithfulness", "context_precision", "context_recall")
            ):
                rows_with_scores += 1

        table.append(
            {
                "system": system,
                "rows_with_context": len(rows_with_context),
                "rows_with_ragas_scores": rows_with_scores,
                "faithfulness": _mean(faithfulness),
                "context_precision": _mean(context_precision),
                "context_recall": _mean(context_recall),
                "tool_calls_per_answer": tool_calls_per_answer,
            }
        )

    return table


def _build_error_slices(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slice_definitions: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("single_hop", lambda row: str(row.get("question_type", "")).strip() == "single_hop"),
        ("multi_hop", lambda row: str(row.get("question_type", "")).strip() == "multi_hop"),
        (
            "trap_questions",
            lambda row: bool(row.get("expected_abstain", False))
            or str(row.get("question_type", "")).strip() == "trap",
        ),
        ("abstain_cases", lambda row: bool(row.get("abstained", False))),
    ]

    grouped = _group_by_system(rows)
    systems = sorted(grouped.keys())
    systems_with_all = ["all"] + systems
    table: list[dict[str, Any]] = []

    for slice_name, predicate in slice_definitions:
        for system in systems_with_all:
            if system == "all":
                candidate_rows = rows
            else:
                candidate_rows = grouped.get(system, [])

            sliced = [row for row in candidate_rows if predicate(row)]
            judged = [
                _coerce_binary_or_none(row.get("binary_correct"))
                for row in sliced
            ]
            judged_values = [value for value in judged if value is not None]
            incorrect = sum(1 for value in judged_values if value == 0)
            error_rate = _mean([1.0 if value == 0 else 0.0 for value in judged_values])

            table.append(
                {
                    "slice": slice_name,
                    "system": system,
                    "total_cases": len(sliced),
                    "judged_cases": len(judged_values),
                    "incorrect_cases": incorrect,
                    "error_rate": error_rate,
                }
            )

    return table


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def _markdown_table(rows: list[dict[str, Any]], headers: tuple[str, ...]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    lines = [header_line, separator]

    for row in rows:
        values: list[str] = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(_format_float(value))
            elif value is None:
                values.append("NA")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_markdown_report(
    path: Path,
    *,
    source_rows: int,
    used_ragas: bool,
    used_judge: bool,
    system_table: list[dict[str, Any]],
    retrieval_table: list[dict[str, Any]],
    slices_table: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# Evaluation Summary")
    lines.append("")
    lines.append(f"- Source rows: {source_rows}")
    lines.append(f"- Used RAGAS scored file: {used_ragas}")
    lines.append(f"- Used judge scored file: {used_judge}")
    lines.append("")
    lines.append("## System Comparison")
    lines.append("")
    lines.append(_markdown_table(system_table, SYSTEM_COMPARISON_HEADERS))
    lines.append("")
    lines.append("## Retrieval Diagnostics")
    lines.append("")
    lines.append(_markdown_table(retrieval_table, RETRIEVAL_HEADERS))
    lines.append("")
    lines.append("## Error Slices")
    lines.append("")
    lines.append(_markdown_table(slices_table, ERROR_SLICE_HEADERS))
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    scored_dir = resolve_repo_path(args.scored)
    output_md_path = resolve_repo_path(args.output_md)
    output_system_csv = resolve_repo_path(args.output_system_csv)
    output_retrieval_csv = resolve_repo_path(args.output_retrieval_csv)
    output_slices_csv = resolve_repo_path(args.output_slices_csv)

    if not scored_dir.exists():
        raise SystemExit(f"Scored directory not found: {scored_dir}")
    if not scored_dir.is_dir():
        raise SystemExit(f"--scored must be a directory: {scored_dir}")

    rows, used_ragas, used_judge = _load_scored_inputs(scored_dir)
    system_table = _build_system_comparison(rows)
    retrieval_table = _build_retrieval_diagnostics(rows)
    slices_table = _build_error_slices(rows)

    _write_csv(output_system_csv, system_table, SYSTEM_COMPARISON_HEADERS)
    _write_csv(output_retrieval_csv, retrieval_table, RETRIEVAL_HEADERS)
    _write_csv(output_slices_csv, slices_table, ERROR_SLICE_HEADERS)
    _write_markdown_report(
        output_md_path,
        source_rows=len(rows),
        used_ragas=used_ragas,
        used_judge=used_judge,
        system_table=system_table,
        retrieval_table=retrieval_table,
        slices_table=slices_table,
    )

    print(f"[summarize_results] Source rows: {len(rows)}")
    print(f"[summarize_results] Used ragas file: {used_ragas}")
    print(f"[summarize_results] Used judge file: {used_judge}")
    print(f"[summarize_results] Markdown: {output_md_path}")
    print(f"[summarize_results] System CSV: {output_system_csv}")
    print(f"[summarize_results] Retrieval CSV: {output_retrieval_csv}")
    print(f"[summarize_results] Error slices CSV: {output_slices_csv}")


if __name__ == "__main__":
    main()
