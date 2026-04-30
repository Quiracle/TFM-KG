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


def test_build_entity_card_uses_meaningful_kg_fields() -> None:
    text = build_entity_card(
        "http://purl.org/collections/nl/am/proxy-2",
        "Monument voor Amsterdam",
        [],
        entity_type="artwork_proxy",
        facts={
            "title": ["Monument voor Amsterdam"],
            "object_number": ["A 54"],
            "maker": ["Sallieth, Mathias de [am:p-31674]"],
            "acquisition_method": ["schenking"],
            "related_object_reference_count": ["1"],
        },
        sections={
            "Dimensions": ["Recorded height: 39.1 cm; dimension label: hoogte a 39.1 cm"],
            "Related artworks": [
                "relatedObjectReference title: Oude en nieuwe brandspuiten; URI: am:proxy-2720"
            ],
        },
    )

    assert "Entity type: artwork_proxy" in text
    assert "Summary: artwork title Monument voor Amsterdam; object number A 54" in text
    assert "- Maker: Sallieth, Mathias de [am:p-31674]" in text
    assert "Recorded height: 39.1 cm" in text
    assert "relatedObjectReference title: Oude en nieuwe brandspuiten" in text


def test_label_and_chunk_id_helpers() -> None:
    uri = "http://example.org/entity#Paris"
    assert fallback_label_from_uri(uri) == "Paris"
    assert chunk_id_for_uri(uri).startswith("kg:")
