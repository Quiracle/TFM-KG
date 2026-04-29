import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.reports import summarize_results


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_summarize_results_generates_required_tables(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scored_dir = tmp_path / "scored"
    report_dir = tmp_path / "reports"

    ragas_rows = [
        {
            "question_id": "q-1",
            "system": "no_context",
            "model": "claude",
            "question": "q1",
            "answer": "a1",
            "binary_correct": 1,
            "expected_abstain": False,
            "abstained": False,
            "latency_ms": 100,
            "question_type": "single_hop",
            "extra_claims": [],
            "retrieved_contexts": [],
            "tool_trace": [],
        },
        {
            "question_id": "q-2",
            "system": "rag_text",
            "model": "api:/query",
            "question": "q2",
            "answer": "a2",
            "binary_correct": 0,
            "expected_abstain": True,
            "abstained": False,
            "latency_ms": 200,
            "question_type": "trap",
            "extra_claims": ["hallucinated claim"],
            "retrieved_contexts": ["ctx-a"],
            "faithfulness": 0.2,
            "context_precision": 0.3,
            "context_recall": 0.4,
            "tool_trace": [],
        },
        {
            "question_id": "q-3",
            "system": "rag_text",
            "model": "api:/query",
            "question": "q3",
            "answer": "a3",
            "binary_correct": 1,
            "expected_abstain": False,
            "abstained": False,
            "latency_ms": 150,
            "question_type": "multi_hop",
            "extra_claims": [],
            "retrieved_contexts": ["ctx-b"],
            "faithfulness": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6,
            "tool_trace": [],
        },
        {
            "question_id": "q-4",
            "system": "mcp_agent",
            "model": "claude",
            "question": "q4",
            "answer": "a4",
            "binary_correct": 0,
            "expected_abstain": False,
            "abstained": False,
            "latency_ms": 300,
            "question_type": "single_hop",
            "extra_claims": [],
            "retrieved_contexts": ["ctx-c"],
            "faithfulness": None,
            "context_precision": None,
            "context_recall": None,
            "tool_trace": [{"t": 1}, {"t": 2}],
        },
        {
            "question_id": "q-5",
            "system": "mcp_agent",
            "model": "claude",
            "question": "q5",
            "answer": "ABSTAIN",
            "binary_correct": 1,
            "expected_abstain": True,
            "abstained": True,
            "latency_ms": 120,
            "question_type": "trap",
            "extra_claims": [],
            "retrieved_contexts": [],
            "tool_trace": [{"t": 1}],
        },
    ]
    _write_jsonl(scored_dir / "ragas_metrics_scored.jsonl", ragas_rows)

    output_md = report_dir / "summary.md"
    output_system_csv = report_dir / "system_comparison.csv"
    output_retrieval_csv = report_dir / "retrieval_diagnostics.csv"
    output_slices_csv = report_dir / "error_slices.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_results.py",
            "--scored",
            str(scored_dir),
            "--output-md",
            str(output_md),
            "--output-system-csv",
            str(output_system_csv),
            "--output-retrieval-csv",
            str(output_retrieval_csv),
            "--output-slices-csv",
            str(output_slices_csv),
        ],
    )

    summarize_results.main()

    assert output_md.exists()
    assert output_system_csv.exists()
    assert output_retrieval_csv.exists()
    assert output_slices_csv.exists()

    system_rows = _read_csv(output_system_csv)
    by_system = {row["system"]: row for row in system_rows}
    assert by_system["no_context"]["correctness_rate"] == "1.0"
    assert by_system["rag_text"]["correctness_rate"] == "0.5"
    assert by_system["rag_text"]["hallucination_rate"] == "0.5"
    assert by_system["mcp_agent"]["abstain_accuracy"] == "1.0"

    retrieval_rows = _read_csv(output_retrieval_csv)
    retrieval_by_system = {row["system"]: row for row in retrieval_rows}
    assert retrieval_by_system["rag_text"]["rows_with_context"] == "2"
    assert retrieval_by_system["rag_text"]["rows_with_ragas_scores"] == "2"
    assert retrieval_by_system["mcp_agent"]["tool_calls_per_answer"] == "1.5"

    summary_text = output_md.read_text(encoding="utf-8")
    assert "## System Comparison" in summary_text
    assert "## Retrieval Diagnostics" in summary_text
    assert "## Error Slices" in summary_text


def test_summarize_results_merges_judge_when_ragas_missing_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scored_dir = tmp_path / "scored"
    report_dir = tmp_path / "reports"

    _write_jsonl(
        scored_dir / "ragas_metrics_scored.jsonl",
        [
            {
                "question_id": "q-1",
                "system": "rag_text",
                "question": "q1",
                "answer": "a1",
                "binary_correct": None,
                "expected_abstain": False,
                "abstained": False,
                "latency_ms": 50,
                "question_type": "single_hop",
                "retrieved_contexts": ["ctx"],
                "extra_claims": [],
                "tool_trace": [],
            }
        ],
    )
    _write_jsonl(
        scored_dir / "binary_judge_scored.jsonl",
        [
            {
                "question_id": "q-1",
                "system": "rag_text",
                "binary_correct": 1,
                "judge_reason": "Correct.",
                "missing_points": [],
                "extra_claims": [],
            }
        ],
    )

    output_system_csv = report_dir / "system_comparison.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_results.py",
            "--scored",
            str(scored_dir),
            "--output-system-csv",
            str(output_system_csv),
            "--output-retrieval-csv",
            str(report_dir / "retrieval.csv"),
            "--output-slices-csv",
            str(report_dir / "slices.csv"),
            "--output-md",
            str(report_dir / "summary.md"),
        ],
    )

    summarize_results.main()

    system_rows = _read_csv(output_system_csv)
    assert len(system_rows) == 1
    assert system_rows[0]["system"] == "rag_text"
    assert system_rows[0]["scored_rows"] == "1"
    assert system_rows[0]["correctness_rate"] == "1.0"
