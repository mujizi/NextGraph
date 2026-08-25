from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Optional, Sequence
from urllib.parse import urlparse

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusClient, connections, db, utility

from backend.app.core.config import Settings
from backend.app.vector_database.providers.base import VectorStoreProvider

logger = logging.getLogger(__name__)

RELATION_DESCRIBE_MAX_LENGTH = 4000
_ID_QUERY_BATCH_SIZE = 512
_WRITE_TARGET_BYTES = 48 * 1024 * 1024
_WRITE_MIN_BATCH_ROWS = 16


class MilvusVectorStoreProvider(VectorStoreProvider):
    provider_name = "milvus"

    def __init__(self, settings: Settings):
        self.settings = settings
        parsed = urlparse(settings.milvus_uri)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = str(parsed.port or 19530)
        self.uri = settings.milvus_uri
        self.token = settings.milvus_token
        self.db_name = settings.vector_db_database or settings.milvus_db
        self.client: MilvusClient | None = None
        self._collection_fields_cache: dict[str, set[str]] = {}
        self.use_database(self.db_name)

    def _connect_high_level_client(self) -> MilvusClient:
        client_kwargs: dict[str, Any] = {"uri": self.uri, "db_name": self.db_name}
        if self.token:
            client_kwargs["token"] = self.token
        return MilvusClient(**client_kwargs)

    def _collection_field_names(self, collection_name: str) -> set[str]:
        collection = Collection(collection_name)
        return {field.name for field in collection.schema.fields}

    def _sanitize_rows_for_collection(self, collection_name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return rows
        field_names = self._collection_field_names(collection_name)
        return [{key: value for key, value in row.items() if key in field_names} for row in rows]

    def _estimate_row_bytes(self, row: dict[str, Any]) -> int:
        total = 128
        for value in row.values():
            if isinstance(value, str):
                total += len(value.encode("utf-8"))
            elif isinstance(value, list):
                if value and all(isinstance(item, (int, float)) for item in value):
                    total += len(value) * 8
                else:
                    total += sum(len(str(item).encode("utf-8")) + 8 for item in value)
            elif isinstance(value, (int, float, bool)):
                total += 16
            elif value is not None:
                total += len(str(value).encode("utf-8"))
        return total

    def _iter_write_batches(self, rows: list[dict[str, Any]]):
        batch: list[dict[str, Any]] = []
        batch_bytes = 0
        for row in rows:
            row_bytes = self._estimate_row_bytes(row)
            if batch and len(batch) >= _WRITE_MIN_BATCH_ROWS and batch_bytes + row_bytes > _WRITE_TARGET_BYTES:
                yield batch
                batch = []
                batch_bytes = 0
            batch.append(row)
            batch_bytes += row_bytes
        if batch:
            yield batch

    def use_database(self, db_name: str) -> None:
        self.db_name = db_name or self.settings.vector_db_database or self.settings.milvus_db
        connections.disconnect(alias="default")
        try:
            connections.connect(alias="default", host=self.host, port=self.port)
            if self.db_name not in db.list_database():
                db.create_database(self.db_name)
        except Exception as exc:
            logger.warning("Milvus database 检查/创建失败，将继续使用当前连接: %s", exc)
        connections.disconnect(alias="default")
        connection_kwargs: dict[str, Any] = {"alias": "default", "host": self.host, "port": self.port, "db_name": self.db_name}
        if self.token:
            connection_kwargs["token"] = self.token
        connections.connect(**connection_kwargs)
        self.client = self._connect_high_level_client()
        self.create_collections_if_not_exists()
        self.init_indexes_and_load()

    def create_collections_if_not_exists(self) -> None:
        if not utility.has_collection(self.settings.entities_collection):
            schema = CollectionSchema(
                [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                    FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="kb_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=500),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
                    FieldSchema(name="relation_ids", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=1000, max_length=100),
                ],
                "实体表",
            )
            Collection(self.settings.entities_collection, schema)

        if not utility.has_collection(self.settings.relations_collection):
            schema = CollectionSchema(
                [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                    FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="kb_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="subject_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="object_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="relation", dtype=DataType.VARCHAR, max_length=1000),
                    FieldSchema(name="describe", dtype=DataType.VARCHAR, max_length=RELATION_DESCRIBE_MAX_LENGTH),
                    FieldSchema(name="passage", dtype=DataType.VARCHAR, max_length=16384),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
                    FieldSchema(name="passage_ids", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=100, max_length=100),
                ],
                "关系表",
            )
            Collection(self.settings.relations_collection, schema)

        if not utility.has_collection(self.settings.passages_collection):
            schema = CollectionSchema(
                [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                    FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="kb_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="docment_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="passage", dtype=DataType.VARCHAR, max_length=16384),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
                ],
                "段落表",
            )
            Collection(self.settings.passages_collection, schema)

    def init_indexes_and_load(self) -> None:
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 64, "efConstruction": 256},
        }
        for collection_name in [
            self.settings.entities_collection,
            self.settings.relations_collection,
            self.settings.passages_collection,
        ]:
            if not utility.has_collection(collection_name):
                continue
            collection = Collection(collection_name)
            if not collection.has_index():
                collection.create_index(field_name="embedding", index_params=index_params, index_name=f"{collection_name}_hnsw_index")
            collection.load()

    async def batch_insert_graph_data(self, payload: dict[str, list[dict[str, Any]]]) -> None:
        try:
            if payload["Table1_Entities"]:
                col_e = Collection(self.settings.entities_collection)
                entity_rows = self._sanitize_rows_for_collection(self.settings.entities_collection, payload["Table1_Entities"])
                for batch in self._iter_write_batches(entity_rows):
                    col_e.upsert(batch)
                col_e.flush()

            if payload["Table2_Relations"]:
                col_r = Collection(self.settings.relations_collection)
                relation_rows = self._sanitize_rows_for_collection(self.settings.relations_collection, payload["Table2_Relations"])
                for batch in self._iter_write_batches(relation_rows):
                    col_r.upsert(batch)
                col_r.flush()

            if payload["Table3_Passages"]:
                col_p = Collection(self.settings.passages_collection)
                passage_rows = self._sanitize_rows_for_collection(self.settings.passages_collection, payload["Table3_Passages"])
                for batch in self._iter_write_batches(passage_rows):
                    col_p.insert(batch)
                col_p.flush()
        except Exception as exc:
            logger.error("Milvus 写入出错: %s", exc)
            raise

    def get_entity_map_by_ids(self, kb_id: str, entity_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not entity_ids:
            return {}
        col = Collection(self.settings.entities_collection)
        results: dict[str, dict[str, Any]] = {}
        for start in range(0, len(entity_ids), _ID_QUERY_BATCH_SIZE):
            batch_ids = entity_ids[start : start + _ID_QUERY_BATCH_SIZE]
            ids_str = ", ".join([f"'{eid}'" for eid in batch_ids])
            expr = f"kb_id == '{kb_id}' and id in [{ids_str}]"
            rows = col.query(expr=expr, output_fields=["id", "name", "relation_ids", "embedding"])
            for item in rows:
                results[item["id"]] = item
        return results

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

    def _collection_fields(self, collection_name: str) -> set[str]:
        if collection_name not in self._collection_fields_cache:
            assert self.client is not None
            description = self.client.describe_collection(collection_name)
            fields = description.get("fields", [])
            self._collection_fields_cache[collection_name] = {str(field.get("name")) for field in fields if field.get("name")}
        return self._collection_fields_cache[collection_name]

    def _available_fields(self, collection_name: str, requested: Sequence[str]) -> list[str]:
        fields = self._collection_fields(collection_name)
        return [field for field in requested if field in fields]

    def _chunked_query(
        self,
        *,
        collection_name: str,
        ids: Sequence[str],
        user_id: str,
        kb_id: str,
        output_fields: Sequence[str],
    ) -> list[dict[str, Any]]:
        if not ids:
            return []
        assert self.client is not None
        deduped_ids = list(dict.fromkeys(ids))
        rows: list[dict[str, Any]] = []
        for start in range(0, len(deduped_ids), _ID_QUERY_BATCH_SIZE):
            batch_ids = deduped_ids[start : start + _ID_QUERY_BATCH_SIZE]
            batch_rows = self.client.query(
                collection_name=collection_name,
                filter=self._compose_filter(self._scope_filter(user_id, kb_id), self._ids_filter(batch_ids)),
                output_fields=self._available_fields(collection_name, output_fields),
            )
            rows.extend(list(batch_rows))
        return rows

    def search_entities(self, query_vector: list[float], *, user_id: str, kb_id: str, limit: int) -> list[dict[str, Any]]:
        assert self.client is not None
        if not self.client.has_collection(self.settings.entities_collection):
            return []
        results = self.client.search(
            collection_name=self.settings.entities_collection,
            data=[query_vector],
            limit=limit,
            filter=self._scope_filter(user_id, kb_id),
            output_fields=self._available_fields(self.settings.entities_collection, ["id", "name", "relation_ids", "user_id", "kb_id"]),
        )
        return [self._map_search_result(item) for item in (results[0] if results else [])]

    def get_entities_by_names(
        self,
        names: Sequence[str],
        *,
        user_id: str,
        kb_id: str,
        limit_per_name: int = 5,
    ) -> list[dict[str, Any]]:
        assert self.client is not None
        if not names or not self.client.has_collection(self.settings.entities_collection):
            return []

        output_fields = self._available_fields(
            self.settings.entities_collection,
            ["id", "name", "relation_ids", "user_id", "kb_id"],
        )
        rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        base_filter = self._scope_filter(user_id, kb_id)
        deduped_names = [name.strip() for name in dict.fromkeys(names) if str(name).strip()]

        for name in deduped_names:
            name_filter = f"name == {json.dumps(name)}"
            exact_rows = self.client.query(
                collection_name=self.settings.entities_collection,
                filter=self._compose_filter(base_filter, name_filter),
                output_fields=output_fields,
                limit=max(1, limit_per_name),
            )
            for row in exact_rows:
                entity_id = str(row.get("id") or "")
                if entity_id and entity_id in seen_ids:
                    continue
                if entity_id:
                    seen_ids.add(entity_id)
                rows.append(dict(row))
        return rows

    def search_relations(
        self,
        query_vector: list[float],
        *,
        user_id: str,
        kb_id: str,
        limit: int,
        relation_ids: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        assert self.client is not None
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
            output_fields=self._available_fields(
                self.settings.relations_collection,
                ["id", "subject_id", "object_id", "relation", "describe", "passage", "passage_ids", "docment_id", "user_id", "kb_id"],
            ),
        )
        return [self._map_search_result(item) for item in (results[0] if results else [])]

    def get_entities_by_ids(self, entity_ids: Sequence[str], *, user_id: str, kb_id: str) -> list[dict[str, Any]]:
        assert self.client is not None
        if not entity_ids or not self.client.has_collection(self.settings.entities_collection):
            return []
        return self._chunked_query(
            collection_name=self.settings.entities_collection,
            ids=entity_ids,
            user_id=user_id,
            kb_id=kb_id,
            output_fields=["id", "name", "relation_ids", "user_id", "kb_id"],
        )

    def get_relations_by_ids(self, relation_ids: Sequence[str], *, user_id: str, kb_id: str) -> list[dict[str, Any]]:
        assert self.client is not None
        if not relation_ids or not self.client.has_collection(self.settings.relations_collection):
            return []
        return self._chunked_query(
            collection_name=self.settings.relations_collection,
            ids=relation_ids,
            user_id=user_id,
            kb_id=kb_id,
            output_fields=["id", "subject_id", "object_id", "relation", "describe", "passage", "passage_ids", "docment_id", "user_id", "kb_id"],
        )

    def get_passages_by_ids(self, passage_ids: Sequence[str], *, user_id: str, kb_id: str) -> list[dict[str, Any]]:
        assert self.client is not None
        if not passage_ids or not self.client.has_collection(self.settings.passages_collection):
            return []
        return self._chunked_query(
            collection_name=self.settings.passages_collection,
            ids=passage_ids,
            user_id=user_id,
            kb_id=kb_id,
            output_fields=["id", "passage", "docment_id", "user_id", "kb_id"],
        )

    def query_collection_preview(self, *, kb_id: str, user_id: str, db_name: str, limit: int) -> dict[str, Any]:
        self.use_database(db_name)
        collections = {
            "entities": (self.settings.entities_collection, ["id", "name", "relation_ids", "user_id", "kb_id"]),
            "relations": (self.settings.relations_collection, ["id", "subject_id", "object_id", "relation", "describe", "passage", "user_id", "kb_id"]),
            "passages": (self.settings.passages_collection, ["id", "docment_id", "passage", "user_id", "kb_id"]),
        }
        expr = f'user_id == "{user_id}" and kb_id == "{kb_id}"'
        data: dict[str, Any] = {
            "kb_id": kb_id,
            "user_id": user_id,
            "milvus_db": db_name,
            "counts": {"entities": 0, "relations": 0, "passages": 0},
            "samples": {"entities": [], "relations": [], "passages": []},
            "top_entities": [],
            "graph_preview": {"nodes": [], "links": []},
        }
        entity_rows: list[dict[str, Any]] = []
        relation_rows: list[dict[str, Any]] = []
        for key, (collection_name, output_fields) in collections.items():
            collection = Collection(collection_name)
            collection.load()
            count_rows = collection.query(expr=expr, output_fields=["count(*)"])
            matched_count = int(count_rows[0].get("count(*)", 0)) if count_rows else 0
            rows = collection.query(expr=expr, output_fields=output_fields, limit=max(1, min(limit, 100)))
            data["counts"][key] = matched_count
            data["samples"][key] = rows
            if key == "entities":
                entity_rows = rows
            elif key == "relations":
                relation_rows = rows

        involved_entity_ids = set()
        for row in relation_rows:
            if row.get("subject_id"):
                involved_entity_ids.add(row["subject_id"])
            if row.get("object_id"):
                involved_entity_ids.add(row["object_id"])
        involved_entity_ids.update(row.get("id") for row in entity_rows if row.get("id"))
        all_involved_entities = self.get_entities_by_ids(sorted(involved_entity_ids), user_id=user_id, kb_id=kb_id) if involved_entity_ids else []
        entity_lookup = {row.get("id"): row.get("name") or row.get("id") for row in all_involved_entities}

        top_entities = sorted(
            (
                {"id": row.get("id"), "name": row.get("name") or row.get("id"), "relation_count": len(row.get("relation_ids") or [])}
                for row in entity_rows
            ),
            key=lambda item: item["relation_count"],
            reverse=True,
        )
        data["top_entities"] = top_entities[: min(8, len(top_entities))]
        graph_nodes: dict[str, dict[str, Any]] = {}
        graph_links: list[dict[str, Any]] = []
        for row in relation_rows:
            subject_id = row.get("subject_id")
            object_id = row.get("object_id")
            if not subject_id or not object_id:
                continue
            graph_nodes.setdefault(subject_id, {"id": subject_id, "name": entity_lookup.get(subject_id, subject_id), "kind": "entity"})
            graph_nodes.setdefault(object_id, {"id": object_id, "name": entity_lookup.get(object_id, object_id), "kind": "entity"})
            graph_links.append({"id": row.get("id"), "source": subject_id, "target": object_id, "label": row.get("relation") or "related_to", "kind": "relation"})
        data["graph_preview"] = {"nodes": list(graph_nodes.values()), "links": graph_links}
        return data

    def discover_kb_ids(self, *, user_id: str, db_name: str, sample_limit: int = 20000) -> set[str]:
        self.use_database(db_name)
        kb_ids: set[str] = set()
        batch_size = 500
        for collection_name in [self.settings.entities_collection, self.settings.relations_collection, self.settings.passages_collection]:
            try:
                collection = Collection(collection_name)
                collection.load()
            except Exception:
                continue
            scanned = 0
            offset = 0
            while scanned < sample_limit:
                rows = collection.query(expr='id != ""', output_fields=["id", "user_id", "kb_id"], limit=min(batch_size, sample_limit - scanned), offset=offset)
                if not rows:
                    break
                for row in rows:
                    if str(row.get("user_id") or "").strip() != user_id:
                        continue
                    kb_id = str(row.get("kb_id") or "").strip()
                    if kb_id:
                        kb_ids.add(kb_id)
                batch_count = len(rows)
                scanned += batch_count
                offset += batch_count
                if batch_count < min(batch_size, sample_limit - scanned + batch_count):
                    break
        return kb_ids

    def inspect_library(self, *, kb_id: str, user_id: str, db_name: str, limit: int) -> dict[str, Any]:
        self.use_database(db_name)
        collections = {
            "entities": (self.settings.entities_collection, ["id", "name", "user_id", "kb_id"]),
            "relations": (self.settings.relations_collection, ["id", "subject_id", "object_id", "relation", "describe", "passage", "user_id", "kb_id"]),
            "passages": (self.settings.passages_collection, ["id", "docment_id", "passage", "user_id", "kb_id"]),
        }
        data: dict[str, Any] = {"kb_id": kb_id, "user_id": user_id, "milvus_db": db_name, "collections": {}}
        expr = f'user_id == "{user_id}" and kb_id == "{kb_id}"'
        for key, (collection_name, output_fields) in collections.items():
            collection = Collection(collection_name)
            collection.load()
            count_rows = collection.query(expr=expr, output_fields=["count(*)"])
            matched_count = int(count_rows[0].get("count(*)", 0)) if count_rows else 0
            rows = collection.query(expr=expr, output_fields=output_fields, limit=max(1, min(limit, 100)))
            data["collections"][key] = {
                "collection": collection_name,
                "matched_count": matched_count,
                "sample_count": len(rows),
                "samples": rows,
            }
        return data
