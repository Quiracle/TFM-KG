import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.scoring import judge_alignment
from eval.scoring.judge_binary import JudgeDecision


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def test_judge_alignment_compares_manual_labels_and_writes_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    alignment_path = tmp_path / "judge_alignment.jsonl"
    prompt_path = tmp_path / "judge_prompt.txt"
    output_jsonl = tmp_path / "alignment_scored.jsonl"
    output_csv = tmp_path / "alignment_scored.csv"
    report_path = tmp_path / "alignment_report.md"

    _write_jsonl(
        alignment_path,
        [
            {
                "id": "ja-001",
                "question": "Who painted Guernica?",
                "reference_answer": "Pablo Picasso painted Guernica.",
                "reference_points": ["creator = Pablo Picasso"],
                "expected_abstain": False,
                "model_answer": "Pablo Picasso painted Guernica.",
                "manual_verdict": 1,
                "manual_reason": "Correct.",
                "case_tag": "exact_correct",
            },
            {
                "id": "ja-002",
                "question": "Who painted Guernica?",
                "reference_answer": "Pablo Picasso painted Guernica.",
                "reference_points": ["creator = Pablo Picasso"],
                "expected_abstain": False,
                "model_answer": "Salvador Dali painted Guernica.",
                "manual_verdict": 0,
                "manual_reason": "Wrong creator.",
                "case_tag": "wrong_entity",
            },
            {
                "id": "ja-003",
                "question": "What is the release date of Starship ZX-900 game console?",
                "reference_answer": "ABSTAIN",
                "reference_points": ["No reliable source evidence provided."],
                "expected_abstain": True,
                "model_answer": "ABSTAIN",
                "manual_verdict": 1,
                "manual_reason": "Expected abstention.",
                "case_tag": "abstain_correct",
            },
        ],
    )
    prompt_path.write_text("Question: {question}\nModel answer: {model_answer}\n", encoding="utf-8")

    class _FakeJudgeClient:
        def __init__(self, *, api_key: str, model: str, timeout_s: int):
            self.api_key = api_key
            self.model = model
            self.timeout_s = timeout_s

    decisions = iter(
        [
            JudgeDecision(binary_correct=1, judge_reason="Correct.", missing_points=[], extra_claims=[], attempts=1),
            JudgeDecision(binary_correct=1, judge_reason="Incorrectly passed.", missing_points=[], extra_claims=[], attempts=1),
            JudgeDecision(
                binary_correct=None,
                judge_reason="Judge failed.",
                missing_points=[],
                extra_claims=[],
                attempts=3,
                judge_error="timeout",
            ),
        ]
    )

    def _fake_judge(**kwargs):
        return next(decisions)

    monkeypatch.setattr(judge_alignment, "AnthropicMessagesClient", _FakeJudgeClient)
    monkeypatch.setattr(judge_alignment, "_judge_answer_with_retry", _fake_judge)
    original_key = judge_alignment.settings.anthropic_api_key
    judge_alignment.settings.anthropic_api_key = "test-key"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "judge_alignment.py",
            "--alignment",
            str(alignment_path),
            "--prompt",
            str(prompt_path),
            "--output-jsonl",
            str(output_jsonl),
            "--output-csv",
            str(output_csv),
            "--report",
            str(report_path),
        ],
    )

    try:
        judge_alignment.main()
    finally:
        judge_alignment.settings.anthropic_api_key = original_key

    rows = _read_jsonl(output_jsonl)
    assert len(rows) == 3
    assert rows[0]["error_class"] == "match"
    assert rows[1]["error_class"] == "false_positive"
    assert rows[2]["error_class"] == "judge_error"
    assert output_csv.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Agreement rate (scored rows)" in report_text
    assert "false_positive" in report_text


def test_validate_alignment_rows_rejects_invalid_manual_verdict() -> None:
    with pytest.raises(ValueError, match="manual_verdict must be 0 or 1"):
        judge_alignment._validate_alignment_rows(
            [
                {
                    "id": "ja-bad",
                    "question": "q",
                    "reference_answer": "a",
                    "reference_points": ["p"],
                    "expected_abstain": False,
                    "model_answer": "a",
                    "manual_verdict": 2,
                }
            ]
        )
