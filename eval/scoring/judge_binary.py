from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
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
    question_id_from_row,
    resolve_repo_path,
    to_jsonable,
    truncate_text,
)
from src.tfmkg.adapters.llm.anthropic_messages import AnthropicMessagesClient
from src.tfmkg.core.config import settings
from src.tfmkg.domain.ports.llm import LLMMessage

DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_SYSTEM_PROMPT = "You are a strict factual QA grader. Return JSON only."

CSV_FIELDS = (
    "question_id",
    "system",
    "model",
    "question",
    "answer",
    "binary_correct",
    "judge_reason",
    "missing_points",
    "extra_claims",
    "expected_abstain",
    "question_type",
    "difficulty",
    "latency_ms",
    "abstained",
    "judge_model",
    "judge_attempts",
    "judge_error",
    "raw_source_file",
)


@dataclass(frozen=True)
class JudgeDecision:
    binary_correct: int | None
    judge_reason: str
    missing_points: list[str]
    extra_claims: list[str]
    attempts: int
    judge_error: str | None = None


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value]
        return [item for item in items if item]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return []


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("Judge response did not include a JSON object.")
    return stripped[start : end + 1]


def _parse_judge_response(text: str) -> tuple[int, str, list[str], list[str]]:
    raw = _extract_json_object(text)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge response was not valid JSON: {exc}") from None

    if not isinstance(payload, dict):
        raise ValueError("Judge response JSON must be an object.")

    verdict_raw = payload.get("verdict")
    try:
        verdict = int(verdict_raw)
    except (TypeError, ValueError):
        raise ValueError("Judge response field 'verdict' must be 0 or 1.") from None
    if verdict not in {0, 1}:
        raise ValueError("Judge response field 'verdict' must be 0 or 1.")

    reason = str(payload.get("reason", "")).strip()
    missing_points = _coerce_string_list(payload.get("missing_points", []))
    extra_claims = _coerce_string_list(payload.get("extra_claims", []))
    return verdict, reason, missing_points, extra_claims


def _render_prompt(template: str, row: dict[str, Any], model_answer: str) -> str:
    expected_abstain = bool(row.get("expected_abstain", False))
    reference_points = _coerce_string_list(row.get("reference_points", []))
    replacements = {
        "question": str(row.get("question", "")).strip(),
        "expected_abstain": json.dumps(expected_abstain),
        "reference_answer": str(row.get("reference_answer", "")).strip(),
        "reference_points": json.dumps(reference_points, ensure_ascii=True),
        "model_answer": model_answer.strip(),
    }

    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


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


def _build_scored_row(
    *,
    raw_row: dict[str, Any],
    reference_row: dict[str, Any] | None,
    raw_source_file: str,
    decision: JudgeDecision,
    judge_model: str,
) -> dict[str, Any]:
    if reference_row is None:
        reference_points: list[str] = []
        supporting_refs: list[str] = []
        expected_abstain = False
        reference_answer = ""
        question_type = ""
        difficulty = ""
        notes = ""
    else:
        reference_points = _coerce_string_list(reference_row.get("reference_points", []))
        supporting_refs = _coerce_string_list(reference_row.get("supporting_refs", []))
        expected_abstain = bool(reference_row.get("expected_abstain", False))
        reference_answer = str(reference_row.get("reference_answer", "")).strip()
        question_type = str(reference_row.get("question_type", "")).strip()
        difficulty = str(reference_row.get("difficulty", "")).strip()
        notes = str(reference_row.get("notes", "")).strip()

    return {
        "question_id": str(raw_row.get("question_id", "")).strip(),
        "system": str(raw_row.get("system", "")).strip(),
        "model": str(raw_row.get("model", "")).strip(),
        "question": str(raw_row.get("question", "")).strip(),
        "answer": str(raw_row.get("answer", "")).strip(),
        "normalized_answer": str(raw_row.get("normalized_answer", "")).strip(),
        "abstained": bool(raw_row.get("abstained", False)),
        "latency_ms": _coerce_int(raw_row.get("latency_ms", 0), default=0),
        "retrieved_contexts": to_jsonable(raw_row.get("retrieved_contexts", [])),
        "citations": to_jsonable(raw_row.get("citations", [])),
        "tool_trace": to_jsonable(raw_row.get("tool_trace", [])),
        "meta": to_jsonable(raw_row.get("meta", {})),
        "reference_answer": reference_answer,
        "reference_points": reference_points,
        "expected_abstain": expected_abstain,
        "question_type": question_type,
        "difficulty": difficulty,
        "notes": notes,
        "supporting_refs": supporting_refs,
        "binary_correct": decision.binary_correct,
        "judge_reason": decision.judge_reason,
        "missing_points": decision.missing_points,
        "extra_claims": decision.extra_claims,
        "judge_model": judge_model,
        "judge_attempts": decision.attempts,
        "judge_error": decision.judge_error,
        "raw_source_file": raw_source_file,
    }


def _judge_answer_with_retry(
    *,
    client: AnthropicMessagesClient,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    max_attempts: int,
) -> JudgeDecision:
    last_error: str | None = None
    last_response_text = ""

    for attempt in range(1, max_attempts + 1):
        try:
            result = client.generate(
                [
                    LLMMessage(role="system", content=DEFAULT_SYSTEM_PROMPT),
                    LLMMessage(role="user", content=prompt),
                ],
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            last_response_text = result.text
            verdict, reason, missing_points, extra_claims = _parse_judge_response(last_response_text)
            return JudgeDecision(
                binary_correct=verdict,
                judge_reason=reason,
                missing_points=missing_points,
                extra_claims=extra_claims,
                attempts=attempt,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"

    detail = "Judge call failed"
    if last_error:
        detail = f"{detail}; {last_error}"
    if last_response_text.strip():
        detail = f"{detail}; last_response={truncate_text(last_response_text, max_chars=300)}"
    return JudgeDecision(
        binary_correct=None,
        judge_reason="Judge failed to produce valid JSON after retries.",
        missing_points=[],
        extra_claims=[],
        attempts=max_attempts,
        judge_error=detail,
    )


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
                    "missing_points": json.dumps(row.get("missing_points", []), ensure_ascii=True),
                    "extra_claims": json.dumps(row.get("extra_claims", []), ensure_ascii=True),
                    "expected_abstain": row.get("expected_abstain", False),
                    "question_type": row.get("question_type", ""),
                    "difficulty": row.get("difficulty", ""),
                    "latency_ms": row.get("latency_ms", 0),
                    "abstained": row.get("abstained", False),
                    "judge_model": row.get("judge_model", ""),
                    "judge_attempts": row.get("judge_attempts", 0),
                    "judge_error": row.get("judge_error", ""),
                    "raw_source_file": row.get("raw_source_file", ""),
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score raw eval outputs with a binary correctness judge.")
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
        "--prompt",
        default="eval/prompts/judge_correctness_v1.txt",
        help="Judge prompt template path.",
    )
    parser.add_argument(
        "--output-jsonl",
        "--output",
        default="eval/outputs/scored/binary_judge_scored.jsonl",
        dest="output_jsonl",
        help="Path to write scored JSONL output.",
    )
    parser.add_argument(
        "--output-csv",
        default="eval/outputs/scored/binary_judge_scored.csv",
        help="Path to write scored CSV output.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_JUDGE_MODEL,
        help="Anthropic model used as judge.",
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


def main() -> None:
    args = _parse_args()
    raw_path = resolve_repo_path(args.raw)
    dataset_path = resolve_repo_path(args.dataset)
    prompt_path = resolve_repo_path(args.prompt)
    output_jsonl_path = resolve_repo_path(args.output_jsonl)
    output_csv_path = resolve_repo_path(args.output_csv)

    if args.max_judge_attempts <= 0:
        raise SystemExit("--max-judge-attempts must be > 0")
    if args.max_output_tokens <= 0:
        raise SystemExit("--max-output-tokens must be > 0")
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    if not prompt_path.exists():
        raise SystemExit(f"Prompt not found: {prompt_path}")
    if not settings.anthropic_api_key:
        raise SystemExit("ANTHROPIC_API_KEY is required for judge_binary.")

    raw_files = _discover_raw_files(raw_path)
    dataset_rows = load_jsonl(dataset_path)
    prompt_template = load_text_file(prompt_path)
    dataset_index = _build_dataset_index(dataset_rows)

    ensure_output_dirs()
    init_jsonl_output(output_jsonl_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    judge_client = AnthropicMessagesClient(
        api_key=settings.anthropic_api_key,
        model=args.model,
        timeout_s=settings.ollama_timeout_s,
    )

    scored_rows: list[dict[str, Any]] = []
    total_rows = 0
    skipped_for_missing_dataset = 0
    judge_failures = 0

    print(f"[judge_binary] Raw inputs: {[str(path) for path in raw_files]}")
    print(f"[judge_binary] Dataset rows: {len(dataset_rows)} from {dataset_path}")
    print(f"[judge_binary] Prompt: {prompt_path}")
    print(f"[judge_binary] Judge model: {args.model}")

    for raw_file in raw_files:
        raw_rows = load_jsonl(raw_file)
        for raw_row in raw_rows:
            total_rows += 1
            question_id = str(raw_row.get("question_id", "")).strip()
            reference_row = dataset_index.get(question_id)

            if reference_row is None:
                skipped_for_missing_dataset += 1
                decision = JudgeDecision(
                    binary_correct=None,
                    judge_reason="Dataset row missing for question_id; judge skipped.",
                    missing_points=[],
                    extra_claims=[],
                    attempts=0,
                    judge_error=f"missing_dataset_row:{question_id}",
                )
            else:
                model_answer = str(raw_row.get("normalized_answer", "")).strip()
                if not model_answer:
                    model_answer = str(raw_row.get("answer", "")).strip()
                prompt = _render_prompt(prompt_template, reference_row, model_answer)
                decision = _judge_answer_with_retry(
                    client=judge_client,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    max_attempts=args.max_judge_attempts,
                )
                if decision.binary_correct is None:
                    judge_failures += 1

            scored_row = _build_scored_row(
                raw_row=raw_row,
                reference_row=reference_row,
                raw_source_file=raw_file.name,
                decision=decision,
                judge_model=args.model,
            )
            scored_rows.append(scored_row)
            append_jsonl_row(output_jsonl_path, scored_row)

    _write_scored_csv(output_csv_path, scored_rows)

    print(
        f"[judge_binary] Completed rows={total_rows} "
        f"missing_dataset_rows={skipped_for_missing_dataset} "
        f"judge_failures={judge_failures}"
    )
    print(f"[judge_binary] Scored JSONL: {output_jsonl_path}")
    print(f"[judge_binary] Scored CSV: {output_csv_path}")


if __name__ == "__main__":
    main()
