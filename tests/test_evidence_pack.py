import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tfmkg.core.evidence import answer_from_evidence, build_evidence_pack, should_abstain


def test_build_evidence_pack_and_answer_from_retrieved_chunks() -> None:
    chunks = [
        {
            "chunk_id": "c1",
            "source_type": "doc_text",
            "source_ref": "doc-1",
            "text": "Cats are mammals.",
        },
        {
            "chunk_id": "c2",
            "source_type": "doc_text",
            "source_ref": "doc-2",
            "text": "Mammals are warm-blooded.",
        },
    ]

    evidence_pack = build_evidence_pack(chunks)
    answer = answer_from_evidence(evidence_pack["facts"])

    assert evidence_pack["citations"][0]["chunk_id"] == "c1"
    assert evidence_pack["citations"][1]["source_ref"] == "doc-2"
    assert "Cats are mammals." in answer
    assert "Mammals are warm-blooded." in answer


def test_answer_from_evidence_handles_empty_facts() -> None:
    assert answer_from_evidence([]) == "Not enough evidence found in retrieved chunks."


def test_should_abstain_when_no_retrieval_hits() -> None:
    abstain, reason = should_abstain(
        question="Who wrote this?",
        retrieval_hits=[],
        evidence_text="",
    )
    assert abstain is True
    assert reason == "no_retrieval_hits"


def test_should_abstain_when_evidence_is_too_short() -> None:
    abstain, reason = should_abstain(
        question="Who wrote this?",
        retrieval_hits=[{"chunk_id": "c1"}],
        evidence_text="Too short.",
        min_evidence_chars=20,
    )
    assert abstain is True
    assert reason == "insufficient_evidence_length"


def test_should_abstain_when_question_and_evidence_do_not_overlap() -> None:
    abstain, reason = should_abstain(
        question="Who discovered penicillin?",
        retrieval_hits=[{"chunk_id": "c1"}],
        evidence_text="Saturn has prominent rings and many moons in orbit.",
    )
    assert abstain is True
    assert reason == "no_question_evidence_overlap"


def test_should_not_abstain_when_evidence_is_sufficient_and_overlaps() -> None:
    abstain, reason = should_abstain(
        question="Who discovered penicillin?",
        retrieval_hits=[{"chunk_id": "c1"}],
        evidence_text="Penicillin was discovered by Alexander Fleming in 1928.",
    )
    assert abstain is False
    assert reason is None
