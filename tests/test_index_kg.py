import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tfmkg.scripts.index_kg import build_entity_card, chunk_id_for_uri, fallback_label_from_uri


def test_build_entity_card_is_deterministic() -> None:
    uri = "http://example.org/entity/Paris"
    label = "Paris"
    triples = [
        ("http://example.org/p/country", "France"),
        ("http://example.org/p/type", "City"),
    ]

    first = build_entity_card(uri, label, list(reversed(triples)))
    second = build_entity_card(uri, label, triples)

    assert first == second
    assert "Label: Paris" in first
    assert "- http://example.org/p/country: France" in first


def test_label_and_chunk_id_helpers() -> None:
    uri = "http://example.org/entity#Paris"
    assert fallback_label_from_uri(uri) == "Paris"
    assert chunk_id_for_uri(uri).startswith("kg:")
