from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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
    question_id_from_row,
    resolve_repo_path,
    to_jsonable,
)
from src.tfmkg.core.config import settings

DEFAULT_RAGAS_MODEL = "claude-haiku-4-5-20251001"
RETRIEVAL_SYSTEMS = {"rag_text", "mcp_agent"}

CSV_FIELDS = (
    "question_id",
    "system",
    "model",
    "question",
    "answer",
    "binary_correct",
    "judge_reason",
    "faithfulness",
    "context_precision",
    "context_recall",
    "ragas_skip_reason",
    "ragas_error",
    "question_type",
    "difficulty",
    "expected_abstain",
    "latency_ms",
    "abstained",
    "raw_source_file",
)


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return []


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _discover_raw_files(raw_path: Path) -> list[Path]:
    if raw_path.is_file():
        if raw_path.suffix != ".jsonl":
            raise SystemExit(f"--raw file must be a .jsonl file: {raw_path}")
        return [raw_path]

    if not raw_path.exists():
        raise SystemExit(f"Raw path not found: {raw_path}")
    if not raw_path.is_dir():
        raise SystemExit(f"--raw must point to a file or directory: {raw_path}")

    files = sorted(path for path in raw_path.glob("*.jsonl") if path.is_file())
    if not files:
        raise SystemExit(f"No JSONL raw files found under: {raw_path}")
    return files


def _build_dataset_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows, start=1):
        qid = question_id_from_row(row, idx)
        index[qid] = row
    return index


def _load_judge_index(judge_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not judge_path.exists():
        return {}
    rows = load_jsonl(judge_path)
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("question_id", "")).strip()
        system = str(row.get("system", "")).strip()
        if not qid or not system:
            continue
        result[(qid, system)] = row
    return result


def _evaluate_with_ragas(
    *,
    ragas_inputs: list[dict[str, Any]],
    model: str,
    temperature: float,
    max_output_tokens: int,
    batch_size: int | None,
    show_progress: bool,
) -> list[dict[str, Any]]:
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required to run RAGAS metrics.")

    try:
        from anthropic import Anthropic
        from datasets import Dataset
        from ragas import evaluate
        from ragas.llms import llm_factory
        from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "RAGAS dependencies are missing. Add/install ragas and datasets to run Milestone 7 scoring."
        ) from exc

    anthropic_client = Anthropic(
        api_key=settings.anthropic_api_key,
        timeout=settings.ollama_timeout_s,
        max_retries=0,
    )
    ragas_llm = llm_factory(
        model=model,
        provider="anthropic",
        client=anthropic_client,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )
    # RAGAS defaults include both `temperature` and `top_p`. Some Anthropic
    # models reject requests when both are provided, so strip the conflicting
    # `top_p` key from this instance before metric calls.
    original_map_provider_params = getattr(ragas_llm, "_map_provider_params", None)
    if callable(original_map_provider_params):
        def _map_provider_params_without_sampling_conflict() -> dict[str, Any]:
            provider_kwargs = original_map_provider_params()
            if not isinstance(provider_kwargs, dict):
                return {}
            cleaned_kwargs = {
                key: value for key, value in provider_kwargs.items() if value is not None
            }
            if "temperature" in cleaned_kwargs and "top_p" in cleaned_kwargs:
                cleaned_kwargs.pop("top_p", None)
            return cleaned_kwargs

        ragas_llm._map_provider_params = _map_provider_params_without_sampling_conflict

    metrics = [
        Faithfulness(llm=ragas_llm, name="faithfulness"),
        LLMContextPrecisionWithReference(llm=ragas_llm, name="context_precision"),
        LLMContextRecall(llm=ragas_llm, name="context_recall"),
    ]

    dataset = Dataset.from_list(ragas_inputs)
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        raise_exceptions=False,
        show_progress=show_progress,
        batch_size=batch_size,
    )
    dataframe = result.to_pandas()
    records = dataframe.to_dict(orient="records")
    return [to_jsonable(record) for record in records]


def _build_base_row(
    *,
    raw_row: dict[str, Any],
    dataset_row: dict[str, Any] | None,
    judge_row: dict[str, Any] | None,
    raw_source_file: str,
) -> dict[str, Any]:
    if dataset_row is None:
        expected_abstain = False
        question_type = ""
        difficulty = ""
        reference_answer = ""
        reference_points: list[str] = []
        notes = ""
    else:
        expected_abstain = bool(dataset_row.get("expected_abstain", False))
        question_type = str(dataset_row.get("question_type", "")).strip()
        difficulty = str(dataset_row.get("difficulty", "")).strip()
        reference_answer = str(dataset_row.get("reference_answer", "")).strip()
        reference_points = _coerce_string_list(dataset_row.get("reference_points", []))
        notes = str(dataset_row.get("notes", "")).strip()

    binary_correct = None if judge_row is None else judge_row.get("binary_correct")
    judge_reason = "" if judge_row is None else str(judge_row.get("judge_reason", "")).strip()
    missing_points = [] if judge_row is None else _coerce_string_list(judge_row.get("missing_points", []))
    extra_claims = [] if judge_row is None else _coerce_string_list(judge_row.get("extra_claims", []))

    return {
        "question_id": str(raw_row.get("question_id", "")).strip(),
        "system": str(raw_row.get("system", "")).strip(),
        "model": str(raw_row.get("model", "")).strip(),
        "question": str(raw_row.get("question", "")).strip(),
        "answer": str(raw_row.get("answer", "")).strip(),
        "normalized_answer": str(raw_row.get("normalized_answer", "")).strip(),
        "abstained": bool(raw_row.get("abstained", False)),
        "latency_ms": _coerce_int(raw_row.get("latency_ms", 0), default=0),
        "retrieved_contexts": _coerce_string_list(raw_row.get("retrieved_contexts", [])),
        "citations": to_jsonable(raw_row.get("citations", [])),
        "tool_trace": to_jsonable(raw_row.get("tool_trace", [])),
        "meta": to_jsonable(raw_row.get("meta", {})),
        "expected_abstain": expected_abstain,
        "question_type": question_type,
        "difficulty": difficulty,
        "reference_answer": reference_answer,
        "reference_points": reference_points,
        "notes": notes,
        "binary_correct": binary_correct,
        "judge_reason": judge_reason,
        "missing_points": missing_points,
        "extra_claims": extra_claims,
        "faithfulness": None,
        "context_precision": None,
        "context_recall": None,
        "ragas_skip_reason": "",
        "ragas_error": "",
        "raw_source_file": raw_source_file,
    }


def _write_scored_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "question_id": row.get("question_id", ""),
                    "system": row.get("system", ""),
                    "model": row.get("model", ""),
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "binary_correct": row.get("binary_correct", ""),
                    "judge_reason": row.get("judge_reason", ""),
                    "faithfulness": row.get("faithfulness", ""),
                    "context_precision": row.get("context_precision", ""),
                    "context_recall": row.get("context_recall", ""),
                    "ragas_skip_reason": row.get("ragas_skip_reason", ""),
                    "ragas_error": row.get("ragas_error", ""),
                    "question_type": row.get("question_type", ""),
                    "difficulty": row.get("difficulty", ""),
                    "expected_abstain": row.get("expected_abstain", False),
                    "latency_ms": row.get("latency_ms", 0),
                    "abstained": row.get("abstained", False),
                    "raw_source_file": row.get("raw_source_file", ""),
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS retrieval metrics for eval raw outputs.")
    parser.add_argument(
        "--raw",
        default="eval/outputs/raw",
        help="Path to raw runner output JSONL file or directory.",
    )
    parser.add_argument(
        "--dataset",
        default="eval/datasets/core_eval_v1.jsonl",
        help="Dataset JSONL used by the raw run.",
    )
    parser.add_argument(
        "--judge",
        default="eval/outputs/scored/binary_judge_scored.jsonl",
        help="Optional judge output JSONL to merge binary correctness fields.",
    )
    parser.add_argument(
        "--output-jsonl",
        "--output",
        default="eval/outputs/scored/ragas_metrics_scored.jsonl",
        dest="output_jsonl",
        help="Path to write scored output JSONL.",
    )
    parser.add_argument(
        "--output-csv",
        default="eval/outputs/scored/ragas_metrics_scored.csv",
        help="Path to write scored output CSV.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_RAGAS_MODEL,
        help="Model used by RAGAS LLM-based metrics.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature passed to RAGAS LLM factory.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=256,
        help="Max tokens per RAGAS metric LLM call.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Optional RAGAS evaluate batch size (0 means default).",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show RAGAS progress bars.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw_path = resolve_repo_path(args.raw)
    dataset_path = resolve_repo_path(args.dataset)
    judge_path = resolve_repo_path(args.judge)
    output_jsonl_path = resolve_repo_path(args.output_jsonl)
    output_csv_path = resolve_repo_path(args.output_csv)

    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    if args.max_output_tokens <= 0:
        raise SystemExit("--max-output-tokens must be > 0")
    if args.batch_size < 0:
        raise SystemExit("--batch-size must be >= 0")

    raw_files = _discover_raw_files(raw_path)
    dataset_rows = load_jsonl(dataset_path)
    dataset_index = _build_dataset_index(dataset_rows)
    judge_index = _load_judge_index(judge_path)
    judge_available = len(judge_index) > 0

    ensure_output_dirs()
    init_jsonl_output(output_jsonl_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ragas_metrics] Raw inputs: {[str(path) for path in raw_files]}")
    print(f"[ragas_metrics] Dataset rows: {len(dataset_rows)} from {dataset_path}")
    if judge_available:
        print(f"[ragas_metrics] Judge rows loaded from: {judge_path}")
    else:
        print(f"[ragas_metrics] Judge file missing or empty: {judge_path}")
    print(f"[ragas_metrics] Output JSONL: {output_jsonl_path}")
    print(f"[ragas_metrics] Output CSV: {output_csv_path}")

    scored_rows: list[dict[str, Any]] = []
    ragas_inputs: list[dict[str, Any]] = []
    ragas_target_indices: list[int] = []

    for raw_file in raw_files:
        for raw_row in load_jsonl(raw_file):
            question_id = str(raw_row.get("question_id", "")).strip()
            system = str(raw_row.get("system", "")).strip()
            dataset_row = dataset_index.get(question_id)
            judge_row = judge_index.get((question_id, system))
            row = _build_base_row(
                raw_row=raw_row,
                dataset_row=dataset_row,
                judge_row=judge_row,
                raw_source_file=raw_file.name,
            )

            if system not in RETRIEVAL_SYSTEMS:
                row["ragas_skip_reason"] = "non_retrieval_system"
            elif not row["retrieved_contexts"]:
                row["ragas_skip_reason"] = "missing_retrieved_contexts"
            elif dataset_row is None:
                row["ragas_skip_reason"] = "missing_dataset_row"
            else:
                ragas_target_indices.append(len(scored_rows))
                ragas_inputs.append(
                    {
                        "user_input": row["question"],
                        "response": row["normalized_answer"] or row["answer"],
                        "reference": row["reference_answer"],
                        "retrieved_contexts": row["retrieved_contexts"],
                    }
                )
            scored_rows.append(row)

    ragas_failures = 0
    if ragas_inputs:
        try:
            ragas_scores = _evaluate_with_ragas(
                ragas_inputs=ragas_inputs,
                model=args.model,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                batch_size=None if args.batch_size == 0 else args.batch_size,
                show_progress=args.show_progress,
            )
            if len(ragas_scores) != len(ragas_inputs):
                raise RuntimeError(
                    "RAGAS returned a different number of rows than the input batch."
                )

            for output_index, score_row in zip(ragas_target_indices, ragas_scores, strict=True):
                scored = scored_rows[output_index]
                faithfulness = _coerce_float_or_none(score_row.get("faithfulness"))
                context_precision = _coerce_float_or_none(score_row.get("context_precision"))
                context_recall = _coerce_float_or_none(score_row.get("context_recall"))

                scored["faithfulness"] = faithfulness
                scored["context_precision"] = context_precision
                scored["context_recall"] = context_recall

                if faithfulness is None and context_precision is None and context_recall is None:
                    scored["ragas_skip_reason"] = "ragas_no_scores"
                    scored["ragas_error"] = "RAGAS returned no numeric scores for this row."
                    ragas_failures += 1
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            ragas_failures = len(ragas_target_indices)
            for output_index in ragas_target_indices:
                scored_rows[output_index]["ragas_skip_reason"] = "ragas_evaluation_failed"
                scored_rows[output_index]["ragas_error"] = message
    else:
        print("[ragas_metrics] No retrieval rows eligible for RAGAS scoring.")

    merged_judge_rows = 0
    for row in scored_rows:
        if row.get("binary_correct") is not None:
            merged_judge_rows += 1
        append_jsonl_row(output_jsonl_path, row)

    _write_scored_csv(output_csv_path, scored_rows)

    print(
        f"[ragas_metrics] Completed rows={len(scored_rows)} "
        f"ragas_eligible={len(ragas_target_indices)} ragas_failures={ragas_failures} "
        f"judge_merged={merged_judge_rows}"
    )


if __name__ == "__main__":
    main()
