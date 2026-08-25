from __future__ import annotations

import json
import logging
import os
from typing import Any, Sequence
from urllib.parse import urlparse

from backend.app.core.config import Settings
from backend.app.vector_database.providers.base import VectorStoreProvider

logger = logging.getLogger(__name__)

try:
    import tcvectordb
    from tcvectordb.model.document import Document, SearchParams
    from tcvectordb.model.enum import FieldType, IndexType, MetricType, ReadConsistency
    from tcvectordb.model.index import FilterIndex, HNSWParams, Index, VectorIndex
except ImportError:  # pragma: no cover - depends on optional package
    tcvectordb = None
    Document = None
    SearchParams = None
    FieldType = None
    IndexType = None
    MetricType = None
    ReadConsistency = None
    FilterIndex = None
    HNSWParams = None
    Index = None
    VectorIndex = None


class TencentVectorStoreProvider(VectorStoreProvider):
    provider_name = "tencent"
    vector_field_name = "vector"
    max_upsert_batch_size = 1000
    max_search_by_id_batch = 20
    max_query_by_id_batch = 1000

    def __init__(self, settings: Settings):
        self.settings = settings
        self.url = settings.tencent_vectordb_url
        self.username = settings.tencent_vectordb_username
        self.key = settings.tencent_vectordb_key
        self.db_name = settings.vector_db_database or settings.tencent_vectordb_database or "default"
        self.client = self._connect()
        self.use_database(self.db_name)

    def _require_sdk(self) -> None:
        if tcvectordb is None:
            raise RuntimeError("缺少 tcvectordb 依赖，请先安装 backend/requirements.txt 中的腾讯云向量数据库 SDK。")

    def _connect(self):
        self._require_sdk()
        parsed = urlparse(self.url)
        host = parsed.hostname
        if host:
            for env_name in ("NO_PROXY", "no_proxy"):
                current = os.environ.get(env_name, "")
                entries = [item.strip() for item in current.split(",") if item.strip()]
                if host not in entries:
                    entries.append(host)
                    os.environ[env_name] = ",".join(entries)
        return tcvectordb.RPCVectorDBClient(
            url=self.url,
            username=self.username,
            key=self.key,
            read_consistency=ReadConsistency.EVENTUAL_CONSISTENCY,
            timeout=30,
        )

    def use_database(self, db_name: str) -> None:
        self._require_sdk()
        self.db_name = db_name or self.settings.vector_db_database or self.settings.tencent_vectordb_database or "default"
        self._ensure_database(self.db_name)
        self._ensure_collection(self._entities_collection_name(), self._build_entities_index())
        self._ensure_collection(self._relations_collection_name(), self._build_relations_index())
        self._ensure_collection(self._passages_collection_name(), self._build_passages_index())

    def _collection_name_for_db(self, base_name: str) -> str:
        safe_db = "".join(ch if ch.isalnum() else "_" for ch in (self.db_name or "default")).strip("_") or "default"
        return f"{safe_db}_{base_name}"

    def _entities_collection_name(self) -> str:
        return self._collection_name_for_db(self.settings.entities_collection)

    def _relations_collection_name(self) -> str:
        return self._collection_name_for_db(self.settings.relations_collection)

    def _passages_collection_name(self) -> str:
        return self._collection_name_for_db(self.settings.passages_collection)

    def _ensure_database(self, db_name: str) -> None:
        try:
            self.client.create_database_if_not_exists(db_name)
        except Exception as exc:
            try:
                databases = self.client.list_databases()
            except Exception:
                databases = []
            existing = {str(getattr(item, "database_name", "") or getattr(item, "name", "") or item) for item in databases}
            if db_name not in existing:
                raise RuntimeError(f"Tencent VectorDB 数据库 {db_name} 创建失败: {exc}") from exc
            logger.info("Tencent VectorDB database %s already exists", db_name)

    def _ensure_collection(self, collection_name: str, index: Any) -> None:
        if self._collection_exists(collection_name):
            return
        try:
            self.client.create_collection_if_not_exists(
                database_name=self.db_name,
                collection_name=collection_name,
                shard=1,
                replicas=1,
                description=f"NextGraph {collection_name}",
                index=index,
            )
        except Exception as exc:
            message = str(exc)
            if "already exist" in message or "already exists" in message:
                logger.info("Tencent VectorDB collection %s/%s already exists", self.db_name, collection_name)
                return
            if self._collection_exists(collection_name):
                logger.info("Tencent VectorDB collection %s/%s already exists", self.db_name, collection_name)
                return
            raise RuntimeError(f"Tencent VectorDB 集合 {collection_name} 在数据库 {self.db_name} 中创建失败: {exc}") from exc
        if not self._collection_exists(collection_name):
            raise RuntimeError(f"Tencent VectorDB 集合 {collection_name} 在数据库 {self.db_name} 中不存在，初始化未完成。")

    def _collection_exists(self, collection_name: str) -> bool:
        try:
            if self.client.exists_collection(self.db_name, collection_name):
                return True
        except Exception as exc:
            logger.warning("Tencent VectorDB exists_collection check failed for %s/%s: %s", self.db_name, collection_name, exc)
        try:
            self.client.describe_collection(self.db_name, collection_name)
            return True
        except Exception:
            return False

    def _build_entities_index(self) -> Any:
        return Index(
            FilterIndex(name="id", field_type=FieldType.String, index_type=IndexType.PRIMARY_KEY),
            FilterIndex(name="user_id", field_type=FieldType.String, index_type=IndexType.FILTER),
            FilterIndex(name="kb_id", field_type=FieldType.String, index_type=IndexType.FILTER),
            FilterIndex(name="name", field_type=FieldType.String, index_type=IndexType.FILTER),
            FilterIndex(name="relation_ids", field_type=FieldType.Array, index_type=IndexType.FILTER),
            VectorIndex(
                name=self.vector_field_name,
                field_type=FieldType.Vector,
                dimension=1536,
                index_type=IndexType.HNSW,
                metric_type=MetricType.COSINE,
                params=HNSWParams(m=16, efconstruction=200),
            ),
        )

    def _build_relations_index(self) -> Any:
        return Index(
            FilterIndex(name="id", field_type=FieldType.String, index_type=IndexType.PRIMARY_KEY),
            FilterIndex(name="user_id", field_type=FieldType.String, index_type=IndexType.FILTER),
            FilterIndex(name="kb_id", field_type=FieldType.String, index_type=IndexType.FILTER),
            FilterIndex(name="subject_id", field_type=FieldType.String, index_type=IndexType.FILTER),
            FilterIndex(name="object_id", field_type=FieldType.String, index_type=IndexType.FILTER),
            FilterIndex(name="relation", field_type=FieldType.String, index_type=IndexType.FILTER),
            FilterIndex(name="describe", field_type=FieldType.String, index_type=IndexType.FILTER),
            FilterIndex(name="docment_id", field_type=FieldType.String, index_type=IndexType.FILTER),
            FilterIndex(name="passage_ids", field_type=FieldType.Array, index_type=IndexType.FILTER),
            VectorIndex(
                name=self.vector_field_name,
                field_type=FieldType.Vector,
                dimension=1536,
                index_type=IndexType.HNSW,
                metric_type=MetricType.COSINE,
                params=HNSWParams(m=16, efconstruction=200),
            ),
        )

    def _build_passages_index(self) -> Any:
        return Index(
            FilterIndex(name="id", field_type=FieldType.String, index_type=IndexType.PRIMARY_KEY),
            FilterIndex(name="user_id", field_type=FieldType.String, index_type=IndexType.FILTER),
            FilterIndex(name="kb_id", field_type=FieldType.String, index_type=IndexType.FILTER),
            FilterIndex(name="docment_id", field_type=FieldType.String, index_type=IndexType.FILTER),
            VectorIndex(
                name=self.vector_field_name,
                field_type=FieldType.Vector,
                dimension=1536,
                index_type=IndexType.HNSW,
                metric_type=MetricType.COSINE,
                params=HNSWParams(m=16, efconstruction=200),
            ),
        )

    def _scope_filter(self, user_id: str | None, kb_id: str | None) -> str:
        parts: list[str] = []
        if user_id:
            parts.append(f'user_id = "{user_id}"')
        if kb_id:
            parts.append(f'kb_id = "{kb_id}"')
        return " and ".join(parts)

    def _compose_filter(self, *parts: str | None) -> str:
        return " and ".join(part for part in parts if part)

    def _ids_filter(self, ids: Sequence[str]) -> str:
        encoded = ",".join(json.dumps(str(item)) for item in ids)
        return f"id in ({encoded})"

    def _single_id_filter(self, item_id: str) -> str:
        return f"id = {json.dumps(str(item_id))}"

    def _to_document(self, payload: dict[str, Any]):
        doc_payload = dict(payload)
        embedding = doc_payload.pop("embedding", None)
        if embedding is not None:
            doc_payload[self.vector_field_name] = embedding
        return Document(**doc_payload) if Document is not None else doc_payload

    def _normalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        vector = normalized.pop(self.vector_field_name, None)
        if vector is not None:
            normalized["embedding"] = vector
        return normalized

    def _iter_write_batches(self, rows: Sequence[dict[str, Any]]):
        for start in range(0, len(rows), self.max_upsert_batch_size):
            yield rows[start : start + self.max_upsert_batch_size]

    async def batch_insert_graph_data(self, payload: dict[str, list[dict[str, Any]]]) -> None:
        for collection_name, key in [
            (self._entities_collection_name(), "Table1_Entities"),
            (self._relations_collection_name(), "Table2_Relations"),
            (self._passages_collection_name(), "Table3_Passages"),
        ]:
            rows = payload.get(key) or []
            if not rows:
                continue
            for batch in self._iter_write_batches(rows):
                documents = [self._to_document(row) for row in batch]
                self.client.upsert(
                    database_name=self.db_name,
                    collection_name=collection_name,
                    documents=documents,
                    build_index=True,
                )

    def get_entity_map_by_ids(self, kb_id: str, entity_ids: list[str]) -> dict[str, dict[str, Any]]:
        rows = self.get_entities_by_ids(entity_ids, user_id="", kb_id=kb_id)
        return {str(row.get("id")): row for row in rows if row.get("id")}

    def search_entities(self, query_vector: list[float], *, user_id: str, kb_id: str, limit: int) -> list[dict[str, Any]]:
        results = self.client.search(
            database_name=self.db_name,
            collection_name=self._entities_collection_name(),
            vectors=[query_vector],
            limit=limit,
            filter=self._scope_filter(user_id, kb_id),
            params=SearchParams(ef=200),
            retrieve_vector=False,
            output_fields=["name", "relation_ids", "user_id", "kb_id"],
        )
        return [self._normalize_row(dict(item)) for item in (results[0] if results else [])]

    def get_entities_by_names(
        self,
        names: Sequence[str],
        *,
        user_id: str,
        kb_id: str,
        limit_per_name: int = 5,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        scope_filter = self._scope_filter(user_id, kb_id)
        deduped_names = [name.strip() for name in dict.fromkeys(names) if str(name).strip()]

        for name in deduped_names:
            escaped_name = name.replace('"', '\\"')
            name_filter = f'name = "{escaped_name}"'
            batch_rows = self.client.query(
                database_name=self.db_name,
                collection_name=self._entities_collection_name(),
                filter=self._compose_filter(scope_filter, name_filter) or None,
                limit=max(1, limit_per_name),
                output_fields=["name", "relation_ids", "user_id", "kb_id"],
                retrieve_vector=False,
            )
            for item in batch_rows:
                normalized = self._normalize_row(dict(item))
                entity_id = str(normalized.get("id") or "")
                if entity_id and entity_id in seen_ids:
                    continue
                if entity_id:
                    seen_ids.add(entity_id)
                rows.append(normalized)
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
        filter_expr = self._scope_filter(user_id, kb_id)
        if relation_ids:
            rows: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            relation_id_list = list(relation_ids)
            for start in range(0, len(relation_id_list), self.max_search_by_id_batch):
                batch_ids = relation_id_list[start : start + self.max_search_by_id_batch]
                batch_results = self.client.search_by_id(
                    database_name=self.db_name,
                    collection_name=self._relations_collection_name(),
                    document_ids=batch_ids,
                    limit=min(limit, len(batch_ids)),
                    filter=filter_expr or None,
                    params=SearchParams(ef=200),
                    retrieve_vector=False,
                    output_fields=["subject_id", "object_id", "relation", "describe", "passage", "passage_ids", "docment_id", "user_id", "kb_id"],
                )
                for item in (batch_results[0] if batch_results else []):
                    normalized = self._normalize_row(dict(item))
                    relation_id = str(normalized.get("id") or "")
                    if relation_id and relation_id in seen_ids:
                        continue
                    if relation_id:
                        seen_ids.add(relation_id)
                    rows.append(normalized)
            rows.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
            return rows[:limit]
        else:
            results = self.client.search(
                database_name=self.db_name,
                collection_name=self._relations_collection_name(),
                vectors=[query_vector],
                limit=limit,
                filter=filter_expr or None,
                params=SearchParams(ef=200),
                retrieve_vector=False,
                output_fields=["subject_id", "object_id", "relation", "describe", "passage", "passage_ids", "docment_id", "user_id", "kb_id"],
            )
            return [self._normalize_row(dict(item)) for item in (results[0] if results else [])]

    def _query_ids(self, collection_name: str, ids: Sequence[str], *, user_id: str, kb_id: str, output_fields: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        filter_expr = self._scope_filter(user_id, kb_id)
        deduped_ids = list(dict.fromkeys(ids))
        requested_fields = list(dict.fromkeys(["id", *output_fields]))
        fallback_fields = [field for field in requested_fields if field != "id"]
        rows: list[dict[str, Any]] = []
        for start in range(0, len(deduped_ids), self.max_query_by_id_batch):
            batch_ids = deduped_ids[start : start + self.max_query_by_id_batch]
            try:
                batch_rows = self.client.query(
                    database_name=self.db_name,
                    collection_name=collection_name,
                    document_ids=batch_ids,
                    filter=filter_expr or None,
                    limit=len(batch_ids),
                    output_fields=requested_fields,
                    retrieve_vector=False,
                )
            except Exception as exc:
                fallback_filter = self._compose_filter(filter_expr, self._ids_filter(batch_ids)) or None
                logger.warning(
                    "Tencent VectorDB document_ids query failed for %s/%s, falling back to filter query: %s",
                    self.db_name,
                    collection_name,
                    exc,
                )
                try:
                    batch_rows = self.client.query(
                        database_name=self.db_name,
                        collection_name=collection_name,
                        filter=fallback_filter,
                        limit=len(batch_ids),
                        output_fields=requested_fields,
                        retrieve_vector=False,
                    )
                    rows.extend(self._normalize_row(dict(item)) for item in batch_rows)
                    continue
                except Exception as fallback_exc:
                    logger.warning(
                        "Tencent VectorDB batch filter query failed for %s/%s, retrying one id at a time: %s",
                        self.db_name,
                        collection_name,
                        fallback_exc,
                    )
                    for item_id in batch_ids:
                        single_filter = self._compose_filter(filter_expr, self._single_id_filter(item_id)) or None
                        single_rows = self.client.query(
                            database_name=self.db_name,
                            collection_name=collection_name,
                            filter=single_filter,
                            limit=1,
                            output_fields=fallback_fields,
                            retrieve_vector=False,
                        )
                        for item in single_rows:
                            normalized = self._normalize_row(dict(item))
                            normalized.setdefault("id", str(item_id))
                            rows.append(normalized)
                    continue
            rows.extend(self._normalize_row(dict(item)) for item in batch_rows)
        return rows

    def get_entities_by_ids(self, entity_ids: Sequence[str], *, user_id: str, kb_id: str) -> list[dict[str, Any]]:
        return self._query_ids(
            self._entities_collection_name(),
            entity_ids,
            user_id=user_id,
            kb_id=kb_id,
            output_fields=["name", "relation_ids", "user_id", "kb_id", self.vector_field_name],
        )

    def get_relations_by_ids(self, relation_ids: Sequence[str], *, user_id: str, kb_id: str) -> list[dict[str, Any]]:
        return self._query_ids(
            self._relations_collection_name(),
            relation_ids,
            user_id=user_id,
            kb_id=kb_id,
            output_fields=["subject_id", "object_id", "relation", "describe", "passage", "passage_ids", "docment_id", "user_id", "kb_id"],
        )

    def get_passages_by_ids(self, passage_ids: Sequence[str], *, user_id: str, kb_id: str) -> list[dict[str, Any]]:
        return self._query_ids(
            self._passages_collection_name(),
            passage_ids,
            user_id=user_id,
            kb_id=kb_id,
            output_fields=["passage", "docment_id", "user_id", "kb_id"],
        )

    def query_collection_preview(self, *, kb_id: str, user_id: str, db_name: str, limit: int) -> dict[str, Any]:
        self.use_database(db_name)
        scope_filter = self._scope_filter(user_id, kb_id)
        data = {
            "kb_id": kb_id,
            "user_id": user_id,
            "milvus_db": db_name,
            "counts": {"entities": 0, "relations": 0, "passages": 0},
            "samples": {"entities": [], "relations": [], "passages": []},
            "top_entities": [],
            "graph_preview": {"nodes": [], "links": []},
        }
        entities = self.client.query(
            database_name=self.db_name,
            collection_name=self._entities_collection_name(),
            filter=scope_filter,
            limit=max(1, min(limit, 100)),
            output_fields=["name", "relation_ids", "user_id", "kb_id"],
            retrieve_vector=False,
        )
        relations = self.client.query(
            database_name=self.db_name,
            collection_name=self._relations_collection_name(),
            filter=scope_filter,
            limit=max(1, min(limit, 100)),
            output_fields=["subject_id", "object_id", "relation", "describe", "passage", "user_id", "kb_id"],
            retrieve_vector=False,
        )
        passages = self.client.query(
            database_name=self.db_name,
            collection_name=self._passages_collection_name(),
            filter=scope_filter,
            limit=max(1, min(limit, 100)),
            output_fields=["docment_id", "passage", "user_id", "kb_id"],
            retrieve_vector=False,
        )
        data["samples"]["entities"] = [dict(item) for item in entities]
        data["samples"]["relations"] = [dict(item) for item in relations]
        data["samples"]["passages"] = [dict(item) for item in passages]
        data["counts"]["entities"] = int(self.client.count(self.db_name, self._entities_collection_name(), filter=scope_filter))
        data["counts"]["relations"] = int(self.client.count(self.db_name, self._relations_collection_name(), filter=scope_filter))
        data["counts"]["passages"] = int(self.client.count(self.db_name, self._passages_collection_name(), filter=scope_filter))
        return data

    def discover_kb_ids(self, *, user_id: str, db_name: str, sample_limit: int = 20000) -> set[str]:
        self.use_database(db_name)
        kb_ids: set[str] = set()
        for collection_name in [self._entities_collection_name(), self._relations_collection_name(), self._passages_collection_name()]:
            rows = self.client.query(
                database_name=self.db_name,
                collection_name=collection_name,
                filter=f'user_id = "{user_id}"',
                limit=max(1, min(sample_limit, 1000)),
                output_fields=["kb_id", "user_id"],
                retrieve_vector=False,
            )
            for row in rows:
                kb_id = str(dict(row).get("kb_id") or "").strip()
                if kb_id:
                    kb_ids.add(kb_id)
        return kb_ids

    def inspect_library(self, *, kb_id: str, user_id: str, db_name: str, limit: int) -> dict[str, Any]:
        preview = self.query_collection_preview(kb_id=kb_id, user_id=user_id, db_name=db_name, limit=limit)
        return {
            "kb_id": kb_id,
            "user_id": user_id,
            "milvus_db": db_name,
            "collections": {
                "entities": {"collection": self.settings.entities_collection, "matched_count": preview["counts"]["entities"], "sample_count": len(preview["samples"]["entities"]), "samples": preview["samples"]["entities"]},
                "relations": {"collection": self.settings.relations_collection, "matched_count": preview["counts"]["relations"], "sample_count": len(preview["samples"]["relations"]), "samples": preview["samples"]["relations"]},
                "passages": {"collection": self.settings.passages_collection, "matched_count": preview["counts"]["passages"], "sample_count": len(preview["samples"]["passages"]), "samples": preview["samples"]["passages"]},
            },
        }
