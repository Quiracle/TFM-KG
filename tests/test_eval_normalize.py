import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.scoring.normalize import ABSTAIN, normalize_answer_text


def test_normalize_answer_text_maps_common_abstain_variants() -> None:
    assert normalize_answer_text("ABSTAIN") == ABSTAIN
    assert normalize_answer_text("I don't know.") == ABSTAIN
    assert normalize_answer_text("insufficient information") == ABSTAIN
    assert (
        normalize_answer_text("I don't have enough information in the provided sources to answer that.")
        == ABSTAIN
    )


def test_normalize_answer_text_leaves_non_abstain_answers() -> None:
    answer = "The creator is Claude Monet."
    assert normalize_answer_text(answer) == answer
