import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.scoring import ragas_metrics


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    result: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            result.append(json.loads(stripped))
    return result


def test_ragas_metrics_scores_only_eligible_retrieval_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    raw_dir = tmp_path / "raw"
    judge_path = tmp_path / "judge.jsonl"
    output_jsonl = tmp_path / "ragas_scored.jsonl"
    output_csv = tmp_path / "ragas_scored.csv"

    _write_jsonl(
        dataset_path,
        [
            {
                "id": "q-1",
                "question": "No context baseline question?",
                "reference_answer": "Answer 1",
                "reference_points": ["point-1"],
                "expected_abstain": False,
                "question_type": "single_hop",
                "difficulty": "easy",
                "notes": "",
                "supporting_refs": [],
            },
            {
                "id": "q-2",
                "question": "RAG question missing contexts?",
                "reference_answer": "Answer 2",
                "reference_points": ["point-2"],
                "expected_abstain": False,
                "question_type": "single_hop",
                "difficulty": "easy",
                "notes": "",
                "supporting_refs": [],
            },
            {
                "id": "q-3",
                "question": "MCP question with contexts?",
                "reference_answer": "Answer 3",
                "reference_points": ["point-3"],
                "expected_abstain": False,
                "question_type": "multi_hop",
                "difficulty": "medium",
                "notes": "",
                "supporting_refs": [],
            },
        ],
    )
    _write_jsonl(
        raw_dir / "no_context_raw.jsonl",
        [
            {
                "question_id": "q-1",
                "system": "no_context",
                "model": "claude-sonnet-4-6",
                "question": "No context baseline question?",
                "answer": "A",
                "normalized_answer": "A",
                "abstained": False,
                "latency_ms": 11,
                "retrieved_contexts": [],
                "citations": [],
                "tool_trace": [],
                "meta": {},
            }
        ],
    )
    _write_jsonl(
        raw_dir / "rag_text_raw.jsonl",
        [
            {
                "question_id": "q-2",
                "system": "rag_text",
                "model": "api:/query",
                "question": "RAG question missing contexts?",
                "answer": "B",
                "normalized_answer": "B",
                "abstained": False,
                "latency_ms": 22,
                "retrieved_contexts": [],
                "citations": [],
                "tool_trace": [],
                "meta": {},
            }
        ],
    )
    _write_jsonl(
        raw_dir / "mcp_agent_raw.jsonl",
        [
            {
                "question_id": "q-3",
                "system": "mcp_agent",
                "model": "claude-sonnet-4-6",
                "question": "MCP question with contexts?",
                "answer": "C",
                "normalized_answer": "C",
                "abstained": False,
                "latency_ms": 33,
                "retrieved_contexts": ["ctx-1", "ctx-2"],
                "citations": [],
                "tool_trace": [],
                "meta": {},
            }
        ],
    )
    _write_jsonl(
        judge_path,
        [
            {
                "question_id": "q-1",
                "system": "no_context",
                "binary_correct": 0,
                "judge_reason": "Incorrect.",
                "missing_points": [],
                "extra_claims": [],
            },
            {
                "question_id": "q-3",
                "system": "mcp_agent",
                "binary_correct": 1,
                "judge_reason": "Correct.",
                "missing_points": [],
                "extra_claims": [],
            },
        ],
    )

    calls: list[list[dict]] = []

    def _fake_evaluate_with_ragas(
        *,
        ragas_inputs: list[dict],
        provider: str,
        model: str,
        temperature: float,
        max_output_tokens: int,
        batch_size: int | None,
        show_progress: bool,
    ) -> list[dict]:
        calls.append(ragas_inputs)
        assert len(ragas_inputs) == 1
        assert ragas_inputs[0]["user_input"] == "MCP question with contexts?"
        return [{"faithfulness": 0.81, "context_precision": 0.66, "context_recall": 0.77}]

    monkeypatch.setattr(ragas_metrics, "_evaluate_with_ragas", _fake_evaluate_with_ragas)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ragas_metrics.py",
            "--raw",
            str(raw_dir),
            "--dataset",
            str(dataset_path),
            "--judge",
            str(judge_path),
            "--output-jsonl",
            str(output_jsonl),
            "--output-csv",
            str(output_csv),
        ],
    )

    ragas_metrics.main()

    rows = _read_jsonl(output_jsonl)
    assert len(rows) == 3
    by_system = {row["system"]: row for row in rows}

    assert by_system["no_context"]["ragas_skip_reason"] == "non_retrieval_system"
    assert by_system["no_context"]["faithfulness"] is None
    assert by_system["no_context"]["binary_correct"] == 0

    assert by_system["rag_text"]["ragas_skip_reason"] == "missing_retrieved_contexts"
    assert by_system["rag_text"]["context_precision"] is None

    assert by_system["mcp_agent"]["ragas_skip_reason"] == ""
    assert by_system["mcp_agent"]["faithfulness"] == 0.81
    assert by_system["mcp_agent"]["context_precision"] == 0.66
    assert by_system["mcp_agent"]["context_recall"] == 0.77
    assert by_system["mcp_agent"]["binary_correct"] == 1
    assert len(calls) == 1
    assert output_csv.exists()


def test_ragas_metrics_marks_missing_dataset_rows_without_crashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    raw_file = tmp_path / "raw" / "mcp_agent_raw.jsonl"
    output_jsonl = tmp_path / "ragas_scored.jsonl"
    output_csv = tmp_path / "ragas_scored.csv"

    _write_jsonl(
        dataset_path,
        [
            {
                "id": "q-1",
                "question": "Known question",
                "reference_answer": "Known answer",
                "reference_points": ["p"],
                "expected_abstain": False,
                "question_type": "single_hop",
                "difficulty": "easy",
                "notes": "",
                "supporting_refs": [],
            }
        ],
    )
    _write_jsonl(
        raw_file,
        [
            {
                "question_id": "q-missing",
                "system": "mcp_agent",
                "model": "claude-sonnet-4-6",
                "question": "Unknown question",
                "answer": "ABSTAIN",
                "normalized_answer": "ABSTAIN",
                "abstained": True,
                "latency_ms": 10,
                "retrieved_contexts": ["ctx"],
                "citations": [],
                "tool_trace": [],
                "meta": {},
            }
        ],
    )

    called = {"value": False}

    def _no_call(**kwargs):
        called["value"] = True
        return []

    monkeypatch.setattr(ragas_metrics, "_evaluate_with_ragas", _no_call)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ragas_metrics.py",
            "--raw",
            str(raw_file),
            "--dataset",
            str(dataset_path),
            "--output-jsonl",
            str(output_jsonl),
            "--output-csv",
            str(output_csv),
        ],
    )

    ragas_metrics.main()

    rows = _read_jsonl(output_jsonl)
    assert len(rows) == 1
    assert rows[0]["ragas_skip_reason"] == "missing_dataset_row"
    assert rows[0]["ragas_error"] == ""
    assert called["value"] is False


def test_ragas_metrics_surfaces_ragas_failures_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    raw_file = tmp_path / "raw" / "rag_text_raw.jsonl"
    output_jsonl = tmp_path / "ragas_scored.jsonl"
    output_csv = tmp_path / "ragas_scored.csv"

    _write_jsonl(
        dataset_path,
        [
            {
                "id": "q-1",
                "question": "RAG question",
                "reference_answer": "Reference",
                "reference_points": ["p"],
                "expected_abstain": False,
                "question_type": "single_hop",
                "difficulty": "easy",
                "notes": "",
                "supporting_refs": [],
            }
        ],
    )
    _write_jsonl(
        raw_file,
        [
            {
                "question_id": "q-1",
                "system": "rag_text",
                "model": "api:/query",
                "question": "RAG question",
                "answer": "Candidate",
                "normalized_answer": "Candidate",
                "abstained": False,
                "latency_ms": 44,
                "retrieved_contexts": ["ctx-a"],
                "citations": [],
                "tool_trace": [],
                "meta": {},
            }
        ],
    )

    def _raise_ragas_error(**kwargs):
        raise RuntimeError("simulated ragas failure")

    monkeypatch.setattr(ragas_metrics, "_evaluate_with_ragas", _raise_ragas_error)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ragas_metrics.py",
            "--raw",
            str(raw_file),
            "--dataset",
            str(dataset_path),
            "--output-jsonl",
            str(output_jsonl),
            "--output-csv",
            str(output_csv),
        ],
    )

    ragas_metrics.main()

    rows = _read_jsonl(output_jsonl)
    assert len(rows) == 1
    assert rows[0]["ragas_skip_reason"] == "ragas_evaluation_failed"
    assert "simulated ragas failure" in rows[0]["ragas_error"]
    assert rows[0]["faithfulness"] is None
    assert rows[0]["context_precision"] is None
    assert rows[0]["context_recall"] is None
