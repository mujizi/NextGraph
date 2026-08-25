from backend.app.vectorization_pipeline.api import (
    _append_relation_id,
    _finalize_relation_ids,
)


def test_relation_ids_are_deduplicated_and_trimmed_to_latest_limit():
    entity_state = {
        "relation_ids": [f"r_{idx}" for idx in range(1000)],
        "relation_ids_set": {f"r_{idx}" for idx in range(1000)},
    }

    _append_relation_id(entity_state, "r_999")
    _append_relation_id(entity_state, "r_1000")
    _append_relation_id(entity_state, "r_1001")

    relation_ids, was_truncated = _finalize_relation_ids(
        entity_state,
        max_relations=1000,
        entity_name="Alice",
    )

    assert was_truncated is True
    assert len(relation_ids) == 1000
    assert relation_ids[0] == "r_2"
    assert relation_ids[-2:] == ["r_1000", "r_1001"]
    assert "r_1" not in relation_ids
