from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "eval"
DATASETS_DIR = EVAL_ROOT / "datasets"
PROMPTS_DIR = EVAL_ROOT / "prompts"
OUTPUTS_DIR = EVAL_ROOT / "outputs"
OUTPUTS_RAW_DIR = OUTPUTS_DIR / "raw"
OUTPUTS_SCORED_DIR = OUTPUTS_DIR / "scored"
OUTPUTS_REPORTS_DIR = OUTPUTS_DIR / "reports"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.scoring.normalize import ABSTAIN, normalize_answer_text

RAW_ROW_FIELDS = (
    "question_id",
    "system",
    "model",
    "question",
    "answer",
    "normalized_answer",
    "abstained",
    "latency_ms",
    "retrieved_contexts",
    "citations",
    "tool_trace",
    "meta",
)


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def ensure_output_dirs() -> None:
    OUTPUTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_SCORED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def touch_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def now_ms() -> int:
    return int(time.time() * 1000)


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump())
        except Exception:
            return str(value)
    return str(value)


def truncate_text(value: str, max_chars: int = 400) -> str:
    clean = " ".join(value.split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3] + "..."


def extract_retrieved_contexts_from_debug(debug: dict[str, Any]) -> list[str]:
    evidence_text = debug.get("evidence_text")
    contexts: list[str] = []
    if isinstance(evidence_text, str):
        for line in evidence_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("- "):
                stripped = stripped[2:].strip()
            contexts.append(truncate_text(stripped))

    if contexts:
        return contexts

    retrieval_hits = debug.get("retrieval_hits")
    if isinstance(retrieval_hits, list):
        fallback: list[str] = []
        for hit in retrieval_hits:
            if not isinstance(hit, dict):
                continue
            chunk_id = hit.get("chunk_id")
            source_ref = hit.get("source_ref")
            source_type = hit.get("source_type")
            score = hit.get("score")
            summary = (
                f"chunk_id={chunk_id} source_ref={source_ref} "
                f"source_type={source_type} score={score}"
            )
            fallback.append(truncate_text(summary))
        return fallback
    return []


def init_jsonl_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(to_jsonable(row), ensure_ascii=True) + "\n")


def select_dataset_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return rows
    return rows[:limit]


def question_id_from_row(row: dict[str, Any], idx: int) -> str:
    candidate = row.get("id")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return f"row-{idx:04d}"


def build_raw_row(
    *,
    question_id: str,
    system: str,
    model: str,
    question: str,
    answer: str,
    latency_ms: int,
    retrieved_contexts: list[str] | None = None,
    citations: list[Any] | None = None,
    tool_trace: list[dict[str, Any]] | None = None,
    meta: dict[str, Any] | None = None,
    forced_abstained: bool | None = None,
) -> dict[str, Any]:
    normalized_answer = normalize_answer_text(answer)
    abstained = normalized_answer == ABSTAIN
    if forced_abstained is not None:
        abstained = bool(forced_abstained or abstained)

    row = {
        "question_id": question_id,
        "system": system,
        "model": model,
        "question": question,
        "answer": answer,
        "normalized_answer": normalized_answer,
        "abstained": abstained,
        "latency_ms": max(0, int(latency_ms)),
        "retrieved_contexts": retrieved_contexts or [],
        "citations": citations or [],
        "tool_trace": tool_trace or [],
        "meta": meta or {},
    }
    return row


def build_error_row(
    *,
    question_id: str,
    system: str,
    model: str,
    question: str,
    error: Exception | str,
    latency_ms: int = 0,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message = str(error)
    detail = {"error": {"type": type(error).__name__ if isinstance(error, Exception) else "Error", "message": message}}
    if meta:
        detail.update(meta)
    return build_raw_row(
        question_id=question_id,
        system=system,
        model=model,
        question=question,
        answer=ABSTAIN,
        latency_ms=latency_ms,
        meta=detail,
    )


def log_runner_error(system: str, question_id: str, error: Exception | str) -> None:
    print(f"[{system}] question_id={question_id} error={error}", file=sys.stderr)


def extract_structured_tool_result(result: Any) -> Any:
    if isinstance(result, tuple):
        if len(result) >= 2:
            return result[1]
        if len(result) == 1:
            return result[0]
        return None
    return result
