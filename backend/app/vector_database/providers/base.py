from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class VectorStoreProvider(ABC):
    provider_name: str = "unknown"

    @abstractmethod
    def use_database(self, db_name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def batch_insert_graph_data(self, payload: dict[str, list[dict[str, Any]]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_entity_map_by_ids(self, kb_id: str, entity_ids: list[str]) -> dict[str, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def search_entities(self, query_vector: list[float], *, user_id: str, kb_id: str, limit: int) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_entities_by_names(
        self,
        names: Sequence[str],
        *,
        user_id: str,
        kb_id: str,
        limit_per_name: int = 5,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def search_relations(
        self,
        query_vector: list[float],
        *,
        user_id: str,
        kb_id: str,
        limit: int,
        relation_ids: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_entities_by_ids(self, entity_ids: Sequence[str], *, user_id: str, kb_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_relations_by_ids(self, relation_ids: Sequence[str], *, user_id: str, kb_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_passages_by_ids(self, passage_ids: Sequence[str], *, user_id: str, kb_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def discover_kb_ids(self, *, user_id: str, db_name: str, sample_limit: int = 20000) -> set[str]:
        raise NotImplementedError

    @abstractmethod
    def query_collection_preview(
        self,
        *,
        kb_id: str,
        user_id: str,
        db_name: str,
        limit: int,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def inspect_library(
        self,
        *,
        kb_id: str,
        user_id: str,
        db_name: str,
        limit: int,
    ) -> dict[str, Any]:
        raise NotImplementedError
