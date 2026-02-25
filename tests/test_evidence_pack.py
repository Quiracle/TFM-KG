import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tfmkg.core.evidence import answer_from_evidence, build_evidence_pack


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
