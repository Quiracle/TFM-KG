import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.scoring import judge_binary


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")


def test_parse_judge_response_accepts_wrapped_json() -> None:
    verdict, reason, missing_points, extra_claims = judge_binary._parse_judge_response(
        'Here is the result:\n{"verdict":1,"reason":"ok","missing_points":[],"extra_claims":[]}'
    )
    assert verdict == 1
    assert reason == "ok"
    assert missing_points == []
    assert extra_claims == []


def test_judge_binary_retries_invalid_json_and_writes_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    raw_path = tmp_path / "raw" / "no_context_raw.jsonl"
    prompt_path = tmp_path / "judge_prompt.txt"
    output_jsonl = tmp_path / "scored.jsonl"
    output_csv = tmp_path / "scored.csv"

    _write_jsonl(
        dataset_path,
        [
            {
                "id": "q-1",
                "question": "Who painted Guernica?",
                "reference_answer": "Pablo Picasso painted Guernica.",
                "reference_points": ["creator = Pablo Picasso"],
                "expected_abstain": False,
                "question_type": "single_hop",
                "difficulty": "easy",
                "notes": "",
                "supporting_refs": [],
            }
        ],
    )
    _write_jsonl(
        raw_path,
        [
            {
                "question_id": "q-1",
                "system": "no_context",
                "model": "claude-sonnet-4-6",
                "question": "Who painted Guernica?",
                "answer": "Guernica was painted by Pablo Picasso.",
                "normalized_answer": "Guernica was painted by Pablo Picasso.",
                "abstained": False,
                "latency_ms": 123,
                "retrieved_contexts": [],
                "citations": [],
                "tool_trace": [],
                "meta": {},
            }
        ],
    )
    prompt_path.write_text(
        (
            "Question: {question}\n"
            "Expected abstain: {expected_abstain}\n"
            "Reference answer: {reference_answer}\n"
            "Reference points: {reference_points}\n"
            "Model answer: {model_answer}\n"
        ),
        encoding="utf-8",
    )

    class _FakeJudgeClient:
        calls = 0

        def __init__(self, *, api_key: str, model: str, timeout_s: int):
            self.api_key = api_key
            self.model = model
            self.timeout_s = timeout_s

        def generate(self, messages, *, temperature: float, max_output_tokens: int):
            _FakeJudgeClient.calls += 1
            if _FakeJudgeClient.calls == 1:
                return SimpleNamespace(text="not-json", prompt_tokens=None, completion_tokens=None)
            return SimpleNamespace(
                text='{"verdict":1,"reason":"Correct.","missing_points":[],"extra_claims":[]}',
                prompt_tokens=None,
                completion_tokens=None,
            )

    monkeypatch.setattr(judge_binary, "AnthropicMessagesClient", _FakeJudgeClient)
    original = {
        "anthropic_api_key": judge_binary.settings.anthropic_api_key,
        "judge_provider": judge_binary.settings.judge_provider,
        "judge_model": judge_binary.settings.judge_model,
    }
    judge_binary.settings.anthropic_api_key = "test-key"
    judge_binary.settings.judge_provider = "anthropic"
    judge_binary.settings.judge_model = "test-judge-model"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "judge_binary.py",
            "--raw",
            str(raw_path),
            "--dataset",
            str(dataset_path),
            "--prompt",
            str(prompt_path),
            "--output-jsonl",
            str(output_jsonl),
            "--output-csv",
            str(output_csv),
            "--max-judge-attempts",
            "3",
        ],
    )

    try:
        judge_binary.main()
    finally:
        judge_binary.settings.anthropic_api_key = original["anthropic_api_key"]
        judge_binary.settings.judge_provider = original["judge_provider"]
        judge_binary.settings.judge_model = original["judge_model"]

    assert _FakeJudgeClient.calls == 2
    scored_rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(scored_rows) == 1
    assert scored_rows[0]["binary_correct"] == 1
    assert scored_rows[0]["judge_reason"] == "Correct."
    assert scored_rows[0]["judge_model"] == "test-judge-model"
    assert scored_rows[0]["judge_attempts"] == 2
    assert output_csv.exists()
    assert "binary_correct" in output_csv.read_text(encoding="utf-8")


def test_judge_binary_surfaces_missing_dataset_row_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    raw_path = tmp_path / "raw" / "rag_text_raw.jsonl"
    prompt_path = tmp_path / "judge_prompt.txt"
    output_jsonl = tmp_path / "scored.jsonl"
    output_csv = tmp_path / "scored.csv"

    _write_jsonl(
        dataset_path,
        [
            {
                "id": "q-1",
                "question": "Who painted Guernica?",
                "reference_answer": "Pablo Picasso painted Guernica.",
                "reference_points": ["creator = Pablo Picasso"],
                "expected_abstain": False,
                "question_type": "single_hop",
                "difficulty": "easy",
                "notes": "",
                "supporting_refs": [],
            }
        ],
    )
    _write_jsonl(
        raw_path,
        [
            {
                "question_id": "q-missing",
                "system": "rag_text",
                "model": "api:/query",
                "question": "Who painted Guernica?",
                "answer": "ABSTAIN",
                "normalized_answer": "ABSTAIN",
                "abstained": True,
                "latency_ms": 45,
                "retrieved_contexts": [],
                "citations": [],
                "tool_trace": [],
                "meta": {},
            }
        ],
    )
    prompt_path.write_text("Question: {question}\nModel answer: {model_answer}\n", encoding="utf-8")

    class _UnusedClient:
        calls = 0

        def __init__(self, *, api_key: str, model: str, timeout_s: int):
            pass

        def generate(self, messages, *, temperature: float, max_output_tokens: int):
            _UnusedClient.calls += 1
            raise AssertionError("Judge client should not be called for missing dataset rows.")

    monkeypatch.setattr(judge_binary, "AnthropicMessagesClient", _UnusedClient)
    original = {
        "anthropic_api_key": judge_binary.settings.anthropic_api_key,
        "judge_provider": judge_binary.settings.judge_provider,
        "judge_model": judge_binary.settings.judge_model,
    }
    judge_binary.settings.anthropic_api_key = "test-key"
    judge_binary.settings.judge_provider = "anthropic"
    judge_binary.settings.judge_model = "test-judge-model"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "judge_binary.py",
            "--raw",
            str(raw_path),
            "--dataset",
            str(dataset_path),
            "--prompt",
            str(prompt_path),
            "--output-jsonl",
            str(output_jsonl),
            "--output-csv",
            str(output_csv),
        ],
    )

    try:
        judge_binary.main()
    finally:
        judge_binary.settings.anthropic_api_key = original["anthropic_api_key"]
        judge_binary.settings.judge_provider = original["judge_provider"]
        judge_binary.settings.judge_model = original["judge_model"]

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["binary_correct"] is None
    assert rows[0]["judge_attempts"] == 0
    assert rows[0]["judge_error"] == "missing_dataset_row:q-missing"
    assert _UnusedClient.calls == 0


def test_judge_binary_scores_rows_from_all_three_system_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    raw_dir = tmp_path / "raw"
    prompt_path = tmp_path / "judge_prompt.txt"
    output_jsonl = tmp_path / "scored.jsonl"
    output_csv = tmp_path / "scored.csv"

    _write_jsonl(
        dataset_path,
        [
            {
                "id": "q-1",
                "question": "Who painted Guernica?",
                "reference_answer": "Pablo Picasso painted Guernica.",
                "reference_points": ["creator = Pablo Picasso"],
                "expected_abstain": False,
                "question_type": "single_hop",
                "difficulty": "easy",
                "notes": "",
                "supporting_refs": [],
            }
        ],
    )

    shared_row = {
        "question_id": "q-1",
        "model": "claude-sonnet-4-6",
        "question": "Who painted Guernica?",
        "answer": "Pablo Picasso painted Guernica.",
        "normalized_answer": "Pablo Picasso painted Guernica.",
        "abstained": False,
        "latency_ms": 10,
        "retrieved_contexts": [],
        "citations": [],
        "tool_trace": [],
        "meta": {},
    }
    _write_jsonl(raw_dir / "no_context_raw.jsonl", [{**shared_row, "system": "no_context"}])
    _write_jsonl(raw_dir / "rag_text_raw.jsonl", [{**shared_row, "system": "rag_text"}])
    _write_jsonl(raw_dir / "mcp_agent_raw.jsonl", [{**shared_row, "system": "mcp_agent"}])

    prompt_path.write_text("Question: {question}\nModel answer: {model_answer}\n", encoding="utf-8")

    class _AlwaysValidClient:
        calls = 0

        def __init__(self, *, api_key: str, model: str, timeout_s: int):
            pass

        def generate(self, messages, *, temperature: float, max_output_tokens: int):
            _AlwaysValidClient.calls += 1
            return SimpleNamespace(
                text='{"verdict":1,"reason":"Correct.","missing_points":[],"extra_claims":[]}',
                prompt_tokens=None,
                completion_tokens=None,
            )

    monkeypatch.setattr(judge_binary, "AnthropicMessagesClient", _AlwaysValidClient)
    original = {
        "anthropic_api_key": judge_binary.settings.anthropic_api_key,
        "judge_provider": judge_binary.settings.judge_provider,
        "judge_model": judge_binary.settings.judge_model,
    }
    judge_binary.settings.anthropic_api_key = "test-key"
    judge_binary.settings.judge_provider = "anthropic"
    judge_binary.settings.judge_model = "test-judge-model"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "judge_binary.py",
            "--raw",
            str(raw_dir),
            "--dataset",
            str(dataset_path),
            "--prompt",
            str(prompt_path),
            "--output-jsonl",
            str(output_jsonl),
            "--output-csv",
            str(output_csv),
        ],
    )

    try:
        judge_binary.main()
    finally:
        judge_binary.settings.anthropic_api_key = original["anthropic_api_key"]
        judge_binary.settings.judge_provider = original["judge_provider"]
        judge_binary.settings.judge_model = original["judge_model"]

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 3
    assert {row["system"] for row in rows} == {"no_context", "rag_text", "mcp_agent"}
    assert all(row["binary_correct"] == 1 for row in rows)
    assert _AlwaysValidClient.calls == 3


def test_judge_binary_uses_configured_gemini_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    raw_path = tmp_path / "raw" / "rag_text_raw.jsonl"
    prompt_path = tmp_path / "judge_prompt.txt"
    output_jsonl = tmp_path / "scored.jsonl"
    output_csv = tmp_path / "scored.csv"

    _write_jsonl(
        dataset_path,
        [
            {
                "id": "q-1",
                "question": "Who painted Guernica?",
                "reference_answer": "Pablo Picasso painted Guernica.",
                "reference_points": ["creator = Pablo Picasso"],
                "expected_abstain": False,
                "question_type": "single_hop",
                "difficulty": "easy",
                "notes": "",
                "supporting_refs": [],
            }
        ],
    )
    _write_jsonl(
        raw_path,
        [
            {
                "question_id": "q-1",
                "system": "rag_text",
                "model": "gemini-2.5-flash",
                "question": "Who painted Guernica?",
                "answer": "Pablo Picasso painted Guernica.",
                "normalized_answer": "Pablo Picasso painted Guernica.",
                "abstained": False,
                "latency_ms": 10,
                "retrieved_contexts": ["creator = Pablo Picasso"],
                "citations": [],
                "tool_trace": [],
                "meta": {},
            }
        ],
    )
    prompt_path.write_text("Question: {question}\nModel answer: {model_answer}\n", encoding="utf-8")

    class _FakeGeminiJudgeClient:
        calls = 0
        init_args = {}

        def __init__(self, *, api_key: str, model: str, timeout_s: int):
            self.init_args = {"api_key": api_key, "model": model, "timeout_s": timeout_s}
            _FakeGeminiJudgeClient.init_args = self.init_args

        def generate(self, messages, *, temperature: float, max_output_tokens: int):
            _FakeGeminiJudgeClient.calls += 1
            return SimpleNamespace(
                text='{"verdict":1,"reason":"Correct.","missing_points":[],"extra_claims":[]}',
                prompt_tokens=None,
                completion_tokens=None,
            )

    monkeypatch.setattr(judge_binary, "GeminiGenerateAdapter", _FakeGeminiJudgeClient)
    original = {
        "gemini_api_key": judge_binary.settings.gemini_api_key,
        "judge_provider": judge_binary.settings.judge_provider,
        "judge_model": judge_binary.settings.judge_model,
    }
    judge_binary.settings.gemini_api_key = "test-gemini-key"
    judge_binary.settings.judge_provider = "gemini"
    judge_binary.settings.judge_model = "gemini-test-model"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "judge_binary.py",
            "--raw",
            str(raw_path),
            "--dataset",
            str(dataset_path),
            "--prompt",
            str(prompt_path),
            "--output-jsonl",
            str(output_jsonl),
            "--output-csv",
            str(output_csv),
        ],
    )

    try:
        judge_binary.main()
    finally:
        judge_binary.settings.gemini_api_key = original["gemini_api_key"]
        judge_binary.settings.judge_provider = original["judge_provider"]
        judge_binary.settings.judge_model = original["judge_model"]

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert _FakeGeminiJudgeClient.calls == 1
    assert _FakeGeminiJudgeClient.init_args["api_key"] == "test-gemini-key"
    assert _FakeGeminiJudgeClient.init_args["model"] == "gemini-test-model"
    assert rows[0]["judge_model"] == "gemini-test-model"
    assert rows[0]["binary_correct"] == 1
