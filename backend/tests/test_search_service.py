from __future__ import annotations

from backend.app.core.config import Settings
from backend.app.search.local_global_hybrid.schemas import SearchMode, SearchRequest
from backend.app.search.local_global_hybrid.service import SearchService


class FakeEmbedder:
    dimension = 4

    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping

    def embed(self, text: str) -> list[float]:
        return self.mapping[text]


class FakeQueryEntityExtractor:
    def extract(self, query: str, *, user_id: str, kb_id: str, semantic_limit: int) -> list[str]:
        if query == "alice query" and user_id == "u1" and kb_id == "kb1":
            return ["Alice"]
        return []


class FakeRepository:
    def __init__(self):
        self.entity_rows = [
            {"id": "ent_alice", "name": "Alice", "relation_ids": ["rel_knows", "rel_parent"], "user_id": "u1", "kb_id": "kb1"},
            {"id": "ent_bob", "name": "Bob", "relation_ids": ["rel_knows", "rel_works"], "user_id": "u1", "kb_id": "kb1"},
            {"id": "ent_carol", "name": "Carol", "relation_ids": ["rel_parent", "rel_works"], "user_id": "u1", "kb_id": "kb1"},
        ]
        self.relation_rows = [
            {"id": "rel_knows", "subject_id": "ent_alice", "object_id": "ent_bob", "relation": "knows", "passage": "Alice knows Bob from school.", "passage_ids": ["p1"], "docment_id": "doc-a", "user_id": "u1", "kb_id": "kb1"},
            {"id": "rel_parent", "subject_id": "ent_carol", "object_id": "ent_alice", "relation": "mentored", "passage": "Carol mentored Alice during college.", "passage_ids": ["p2"], "docment_id": "doc-b", "user_id": "u1", "kb_id": "kb1"},
            {"id": "rel_works", "subject_id": "ent_bob", "object_id": "ent_carol", "relation": "works with", "passage": "Bob works with Carol at the lab.", "passage_ids": ["p3"], "docment_id": "doc-c", "user_id": "u1", "kb_id": "kb1"},
            {"id": "rel_ignored", "subject_id": "ent_bob", "object_id": "ent_alice", "relation": "ignores", "passage": "Bob ignores Alice.", "passage_ids": ["p4"], "docment_id": "doc-d", "user_id": "u2", "kb_id": "kb2"},
        ]
        self.passage_rows = [
            {"id": "p1", "passage": "Alice knows Bob from school.", "docment_id": "doc-a", "user_id": "u1", "kb_id": "kb1"},
            {"id": "p2", "passage": "Carol mentored Alice during college.", "docment_id": "doc-b", "user_id": "u1", "kb_id": "kb1"},
            {"id": "p3", "passage": "Bob works with Carol at the lab.", "docment_id": "doc-c", "user_id": "u1", "kb_id": "kb1"},
        ]

    def search_entities(self, query_vector, *, user_id: str, kb_id: str, limit: int):
        if user_id != "u1" or kb_id != "kb1":
            return []
        if query_vector == [0.9, 0.0, 0.0, 0.0]:
            return [dict(self.entity_rows[0], score=0.99)]
        if query_vector == [1.0, 0.0, 0.0, 0.0]:
            return [dict(self.entity_rows[0], score=0.91)]
        return []

    def search_relations(self, query_vector, *, user_id: str, kb_id: str, limit: int, relation_ids=None):
        if user_id != "u1" or kb_id != "kb1":
            return []
        if relation_ids:
            relation_ids = set(relation_ids)
            rows = [
                dict(self.relation_rows[0], score=0.86),
                dict(self.relation_rows[1], score=0.95),
                dict(self.relation_rows[2], score=0.72),
            ]
            return [row for row in rows if row["id"] in relation_ids][:limit]
        return [dict(self.relation_rows[0], score=0.88)]

    def get_relations_by_ids(self, relation_ids, *, user_id: str, kb_id: str):
        return [row for row in self.relation_rows if row["id"] in relation_ids and row["user_id"] == user_id and row["kb_id"] == kb_id]

    def get_entities_by_ids(self, entity_ids, *, user_id: str, kb_id: str):
        return [row for row in self.entity_rows if row["id"] in entity_ids and row["user_id"] == user_id and row["kb_id"] == kb_id]

    def get_passages_by_ids(self, passage_ids, *, user_id: str, kb_id: str):
        return [row for row in self.passage_rows if row["id"] in passage_ids and row["user_id"] == user_id and row["kb_id"] == kb_id]


def build_service() -> SearchService:
    settings = Settings(embedder_backend="azure_openai")
    return SearchService(
        settings=settings,
        repository=FakeRepository(),
        embedder=FakeEmbedder({"alice query": [1.0, 0.0, 0.0, 0.0], "Alice": [0.9, 0.0, 0.0, 0.0]}),
        query_entity_extractor=FakeQueryEntityExtractor(),
    )


def test_triple_search_expands_subgraph_from_seed_entities_and_relations():
    service = build_service()
    response = service.search(SearchRequest(query="alice query", user_id="u1", kb_id="kb1", mode=SearchMode.triple, expansion_degree=1))

    assert [hit.id for hit in response.results] == ["rel_parent", "rel_knows", "rel_works"]
    assert response.results[0].source_modes == ["triple"]
    assert response.results[0].subject_name == "Carol"
    assert response.results[0].object_name == "Alice"
    assert response.results[0].relation == "mentored"
    assert response.metadata["route"]["query_entities"] == ["Alice"]
    assert response.metadata["route"]["expanded_relation_count"] == 3
    assert response.metadata["route"]["expansion_history"][0]["operation"] == "init_merge"
    assert response.grounded_passages[0].id == "p2"


def test_hybrid_search_combines_semantic_and_triple_scores_and_grounding():
    service = build_service()
    response = service.search(SearchRequest(query="alice query", user_id="u1", kb_id="kb1", mode=SearchMode.hybrid))

    assert len(response.results) == 3
    hit = response.results[0]
    assert set(hit.source_modes) == {"semantic", "triple"}
    assert "semantic_score" in hit.score_breakdown
    assert "triple_score" in hit.score_breakdown
    assert hit.id == "rel_knows"
    assert response.grounded_passages[0].matched_relation_ids == ["rel_knows"]


def test_scope_filter_excludes_other_users():
    service = build_service()
    response = service.search(SearchRequest(query="alice query", user_id="u2", kb_id="kb2", mode=SearchMode.semantic))

    assert response.results == []
    assert response.entity_hits == []
    assert response.grounded_passages == []
