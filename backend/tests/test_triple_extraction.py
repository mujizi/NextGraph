from backend.app.triple_extraction.functions import clean_triplets


def test_clean_triplets_filters_invalid_rows_and_deduplicates() -> None:
    triplets = [
        {"subject": "Alice", "relation": "knows", "object": "Bob", "describe": "Alice knows Bob."},
        {"subject": " Alice ", "relation": "knows", "object": "Bob", "describe": ""},
        {"subject": "Alice", "relation": "", "object": "Bob", "describe": "invalid"},
        {"subject": "Solo", "relation": "is", "object": "Solo", "describe": "invalid"},
    ]

    assert clean_triplets(triplets) == [
        {"subject": "Alice", "relation": "knows", "object": "Bob", "describe": "Alice knows Bob."}
    ]


def test_clean_triplets_fills_missing_describe() -> None:
    assert clean_triplets(
        [{"subject": "Compressor", "relation": "causes", "object": "Overheat", "describe": "  "}]
    ) == [
        {
            "subject": "Compressor",
            "relation": "causes",
            "object": "Overheat",
            "describe": "Compressor causes Overheat",
        }
    ]
