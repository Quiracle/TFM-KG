from __future__ import annotations

from typing import Any


def _to_fact(text: str) -> str:
    clean = " ".join(text.split())
    if len(clean) <= 160:
        return clean
    return clean[:157] + "..."


def build_evidence_pack(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    facts: list[str] = []
    citations: list[dict[str, Any]] = []

    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        if text:
            facts.append(_to_fact(text))

        citations.append(
            {
                "source_type": chunk.get("source_type", "unknown"),
                "source_ref": chunk.get("source_ref", ""),
                "chunk_id": chunk.get("chunk_id"),
            }
        )

    return {"facts": facts, "citations": citations}


def answer_from_evidence(facts: list[str]) -> str:
    if not facts:
        return "Not enough evidence found in retrieved chunks."
    return "Evidence-based answer: " + " | ".join(facts)
