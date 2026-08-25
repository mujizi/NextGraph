from __future__ import annotations

from typing import Sequence

from backend.app.core.config import Settings
from backend.app.vector_database.providers.base import VectorStoreProvider
from backend.app.vector_database.providers.factory import get_vector_store


class GraphRepository:
    """Provider-backed access layer for entity/relation/passage retrieval."""

    def __init__(self, settings: Settings, provider: VectorStoreProvider | None = None):
        self.settings = settings
        self.provider = provider or get_vector_store(settings)

    def search_entities(self, query_vector: list[float], *, user_id: str, kb_id: str, limit: int) -> list[dict]:
        return self.provider.search_entities(query_vector, user_id=user_id, kb_id=kb_id, limit=limit)

    def search_relations(
        self,
        query_vector: list[float],
        *,
        user_id: str,
        kb_id: str,
        limit: int,
        relation_ids: Sequence[str] | None = None,
    ) -> list[dict]:
        return self.provider.search_relations(
            query_vector,
            user_id=user_id,
            kb_id=kb_id,
            limit=limit,
            relation_ids=relation_ids,
        )

    def get_entities_by_ids(self, entity_ids: Sequence[str], *, user_id: str, kb_id: str) -> list[dict]:
        return self.provider.get_entities_by_ids(entity_ids, user_id=user_id, kb_id=kb_id)

    def get_relations_by_ids(self, relation_ids: Sequence[str], *, user_id: str, kb_id: str) -> list[dict]:
        return self.provider.get_relations_by_ids(relation_ids, user_id=user_id, kb_id=kb_id)

    def get_passages_by_ids(self, passage_ids: Sequence[str], *, user_id: str, kb_id: str) -> list[dict]:
        return self.provider.get_passages_by_ids(passage_ids, user_id=user_id, kb_id=kb_id)


MilvusGraphRepository = GraphRepository
