import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.runners.common import (
    RAW_ROW_FIELDS,
    build_error_row,
    build_raw_row,
    extract_retrieved_contexts_from_debug,
)
from eval.scoring.normalize import ABSTAIN


def test_build_raw_row_uses_expected_schema_fields() -> None:
    row = build_raw_row(
        question_id="q-1",
        system="rag_text",
        model="claude-sonnet-4-5",
        question="Who created object X?",
        answer="ABSTAIN",
        latency_ms=123,
        retrieved_contexts=["ctx-a"],
        citations=[{"source_ref": "doc-1"}],
        tool_trace=[],
        meta={"k": "v"},
    )
    assert tuple(row.keys()) == RAW_ROW_FIELDS
    assert row["abstained"] is True
    assert row["normalized_answer"] == ABSTAIN


def test_build_raw_row_respects_forced_abstained() -> None:
    row = build_raw_row(
        question_id="q-2",
        system="rag_text",
        model="m",
        question="q",
        answer="Paris",
        latency_ms=10,
        forced_abstained=True,
    )
    assert row["abstained"] is True
    assert row["normalized_answer"] == "Paris"


def test_extract_retrieved_contexts_from_debug_prefers_evidence_text() -> None:
    contexts = extract_retrieved_contexts_from_debug(
        {
            "evidence_text": "- fact one\n- fact two",
            "retrieval_hits": [{"chunk_id": "c1"}],
        }
    )
    assert contexts == ["fact one", "fact two"]


def test_extract_retrieved_contexts_from_debug_uses_retrieval_hits_fallback() -> None:
    contexts = extract_retrieved_contexts_from_debug(
        {
            "retrieval_hits": [
                {
                    "chunk_id": "c1",
                    "source_ref": "doc-1",
                    "source_type": "doc_text",
                    "score": 0.1,
                }
            ]
        }
    )
    assert len(contexts) == 1
    assert "chunk_id=c1" in contexts[0]


def test_build_error_row_includes_error_detail() -> None:
    row = build_error_row(
        question_id="q-err",
        system="no_context",
        model="m",
        question="q",
        error=RuntimeError("boom"),
    )
    assert row["answer"] == ABSTAIN
    assert row["meta"]["error"]["message"] == "boom"
