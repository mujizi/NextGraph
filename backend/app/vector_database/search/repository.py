from __future__ import annotations

import json
from typing import Any, Iterable, Optional, Sequence

from pymilvus import DataType, MilvusClient

from backend.app.core.config import Settings


class MilvusGraphRepository:
    """Milvus access layer for entity/relation/passage retrieval."""

    def __init__(self, settings: Settings, client: MilvusClient | None = None):
        self.settings = settings
        self.client = client or self._connect()

    def _connect(self) -> MilvusClient:
        client_kwargs: dict[str, Any] = {"uri": self.settings.milvus_uri}
        if self.settings.milvus_token:
            client_kwargs["token"] = self.settings.milvus_token

        bootstrap_client = MilvusClient(**client_kwargs)
        databases = set(bootstrap_client.list_databases())
        if self.settings.milvus_db not in databases:
            bootstrap_client.create_database(db_name=self.settings.milvus_db)

        return MilvusClient(db_name=self.settings.milvus_db, **client_kwargs)

    def _scope_filter(self, user_id: str, kb_id: str) -> str:
        return f"user_id == {json.dumps(user_id)} and kb_id == {json.dumps(kb_id)}"

    def _compose_filter(self, *parts: Optional[str]) -> str:
        return " and ".join(part for part in parts if part)

    def _ids_filter(self, ids: Sequence[str]) -> str:
        encoded = ", ".join(json.dumps(value) for value in ids)
        return f"id in [{encoded}]"

    def _map_search_result(self, raw: dict[str, Any]) -> dict[str, Any]:
        entity = dict(raw.get("entity") or {})
        entity.setdefault("id", raw.get("id"))
        entity["score"] = float(raw.get("distance", raw.get("score", 0.0)))
        return entity

    def search_entities(self, query_vector: list[float], *, user_id: str, kb_id: str, limit: int) -> list[dict[str, Any]]:
        if not self.client.has_collection(self.settings.entities_collection):
            return []

        results = self.client.search(
            collection_name=self.settings.entities_collection,
            data=[query_vector],
            limit=limit,
            filter=self._scope_filter(user_id, kb_id),
            output_fields=["id", "name", "relation_ids", "user_id", "kb_id"],
        )
        return [self._map_search_result(item) for item in (results[0] if results else [])]

    def search_relations(
        self,
        query_vector: list[float],
        *,
        user_id: str,
        kb_id: str,
        limit: int,
        relation_ids: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.client.has_collection(self.settings.relations_collection):
            return []

        filter_expr = self._scope_filter(user_id, kb_id)
        if relation_ids:
            filter_expr = self._compose_filter(filter_expr, self._ids_filter(relation_ids))

        results = self.client.search(
            collection_name=self.settings.relations_collection,
            data=[query_vector],
            limit=limit,
            filter=filter_expr,
            output_fields=[
                "id",
                "subject_id",
                "object_id",
                "relation",
                "passage",
                "passage_ids",
                "docment_id",
                "user_id",
                "kb_id",
            ],
        )
        return [self._map_search_result(item) for item in (results[0] if results else [])]

    def get_entities_by_ids(self, entity_ids: Sequence[str], *, user_id: str, kb_id: str) -> list[dict[str, Any]]:
        if not entity_ids or not self.client.has_collection(self.settings.entities_collection):
            return []

        results = self.client.query(
            collection_name=self.settings.entities_collection,
            filter=self._compose_filter(self._scope_filter(user_id, kb_id), self._ids_filter(entity_ids)),
            output_fields=["id", "name", "relation_ids", "user_id", "kb_id"],
        )
        return list(results)

    def get_relations_by_ids(self, relation_ids: Sequence[str], *, user_id: str, kb_id: str) -> list[dict[str, Any]]:
        if not relation_ids or not self.client.has_collection(self.settings.relations_collection):
            return []

        results = self.client.query(
            collection_name=self.settings.relations_collection,
            filter=self._compose_filter(self._scope_filter(user_id, kb_id), self._ids_filter(relation_ids)),
            output_fields=[
                "id",
                "subject_id",
                "object_id",
                "relation",
                "passage",
                "passage_ids",
                "docment_id",
                "user_id",
                "kb_id",
            ],
        )
        return list(results)

    def get_passages_by_ids(self, passage_ids: Sequence[str], *, user_id: str, kb_id: str) -> list[dict[str, Any]]:
        if not passage_ids or not self.client.has_collection(self.settings.passages_collection):
            return []

        results = self.client.query(
            collection_name=self.settings.passages_collection,
            filter=self._compose_filter(self._scope_filter(user_id, kb_id), self._ids_filter(passage_ids)),
            output_fields=["id", "passage", "docment_id", "user_id", "kb_id"],
        )
        return list(results)

    def create_demo_collections(self, *, dimension: int, drop_existing: bool = False) -> None:
        for collection_name in [self.settings.entities_collection, self.settings.relations_collection, self.settings.passages_collection]:
            if self.client.has_collection(collection_name):
                if drop_existing:
                    self.client.drop_collection(collection_name)
                else:
                    continue

            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
            if collection_name == self.settings.entities_collection:
                schema.add_field(field_name="name", datatype=DataType.VARCHAR, max_length=2048)
            elif collection_name == self.settings.relations_collection:
                schema.add_field(field_name="subject_id", datatype=DataType.VARCHAR, max_length=128)
                schema.add_field(field_name="object_id", datatype=DataType.VARCHAR, max_length=128)
                schema.add_field(field_name="relation", datatype=DataType.VARCHAR, max_length=4096)
                schema.add_field(field_name="passage", datatype=DataType.VARCHAR, max_length=8192)
                schema.add_field(field_name="docment_id", datatype=DataType.VARCHAR, max_length=256)
            else:
                schema.add_field(field_name="passage", datatype=DataType.VARCHAR, max_length=8192)
                schema.add_field(field_name="docment_id", datatype=DataType.VARCHAR, max_length=256)
            schema.add_field(field_name="user_id", datatype=DataType.VARCHAR, max_length=128)
            schema.add_field(field_name="kb_id", datatype=DataType.VARCHAR, max_length=128)
            schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dimension)

            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="IP")
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
                consistency_level="Bounded",
            )

    def insert_entities(self, rows: Sequence[dict[str, Any]]) -> None:
        if rows:
            self.client.insert(collection_name=self.settings.entities_collection, data=list(rows))

    def insert_relations(self, rows: Sequence[dict[str, Any]]) -> None:
        if rows:
            self.client.insert(collection_name=self.settings.relations_collection, data=list(rows))

    def insert_passages(self, rows: Sequence[dict[str, Any]]) -> None:
        if rows:
            self.client.insert(collection_name=self.settings.passages_collection, data=list(rows))

    def delete_by_ids(self, collection_name: str, ids: Iterable[str]) -> None:
        ids = list(ids)
        if ids and self.client.has_collection(collection_name):
            self.client.delete(collection_name=collection_name, filter=self._ids_filter(ids))
