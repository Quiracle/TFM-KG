from __future__ import annotations

import re
from typing import Any


def _to_fact(text: str) -> str:
    clean = " ".join(text.split())
    if len(clean) <= 400:
        return clean
    return clean[:397] + "..."


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


def _meaningful_tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
    words = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return {word for word in words if len(word) >= 3 and word not in stopwords}


def should_abstain(
    *,
    question: str,
    retrieval_hits: list[dict[str, Any]],
    evidence_text: str,
    min_evidence_chars: int = 40,
) -> tuple[bool, str | None]:
    if not retrieval_hits:
        return True, "no_retrieval_hits"

    if len(evidence_text.strip()) < min_evidence_chars:
        return True, "insufficient_evidence_length"

    q_tokens = _meaningful_tokens(question)
    e_tokens = _meaningful_tokens(evidence_text)
    if q_tokens and not (q_tokens & e_tokens):
        return True, "no_question_evidence_overlap"

    return False, None


def answer_from_evidence(facts: list[str]) -> str:
    if not facts:
        return "Not enough evidence found in retrieved chunks."
    return "Evidence-based answer: " + " | ".join(facts)
