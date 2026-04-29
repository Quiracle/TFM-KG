from __future__ import annotations


ABSTAIN = "ABSTAIN"


_ABSTAIN_EXACT = {
    "abstain",
    "i don't know",
    "i do not know",
    "not enough information",
    "insufficient information",
    "insufficient evidence",
    "cannot determine from the provided evidence",
}

_ABSTAIN_SUBSTRINGS = {
    "i don't have enough information in the provided sources to answer that",
    "i do not have enough information in the provided sources to answer that",
    "not enough information in the provided sources",
}


def _collapse_spaces(value: str) -> str:
    return " ".join(value.split())


def _strip_wrapping_quotes(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1].strip()
    return stripped


def normalize_answer_text(answer: str) -> str:
    text = _collapse_spaces(answer.strip())
    lowered = _strip_wrapping_quotes(text).casefold().rstrip(".!?")

    if lowered in _ABSTAIN_EXACT:
        return ABSTAIN
    if any(fragment in lowered for fragment in _ABSTAIN_SUBSTRINGS):
        return ABSTAIN
    return text
