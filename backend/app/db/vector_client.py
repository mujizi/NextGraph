import logging
import os
from typing import List, Dict, Any
from urllib.parse import urlparse

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, db, utility

logger = logging.getLogger(__name__)

RELATION_DESCRIBE_MAX_LENGTH = 4000
_ID_QUERY_BATCH_SIZE = 256
_WRITE_TARGET_BYTES = 48 * 1024 * 1024
_WRITE_MIN_BATCH_ROWS = 16

class MilvusClient:
    def __init__(self, host: str | None = None, port: str | None = None, db_name: str | None = None):
        uri = os.environ.get("NEXTGRAPH_MILVUS_URI", "http://10.1.80.16:19530")
        parsed = urlparse(uri)
        self.host = host or parsed.hostname or "10.1.80.16"
        self.port = port or str(parsed.port or 19530)
        self.db_name = db_name or os.environ.get("NEXTGRAPH_MILVUS_DB", "crx")
        # 建立连接
        try:
            self.use_database(self.db_name)
        except Exception as e:
            logger.error(f"Milvus 连接失败: {e}")

    def _collection_field_names(self, collection_name: str) -> set[str]:
        collection = Collection(collection_name)
        return {field.name for field in collection.schema.fields}

    def _sanitize_rows_for_collection(self, collection_name: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not rows:
            return rows
        field_names = self._collection_field_names(collection_name)
        return [{key: value for key, value in row.items() if key in field_names} for row in rows]

    def _estimate_row_bytes(self, row: Dict[str, Any]) -> int:
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

    def _iter_write_batches(self, rows: List[Dict[str, Any]]):
        batch: List[Dict[str, Any]] = []
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
        self.db_name = db_name or "crx"
        connections.disconnect(alias="default")
        try:
            connections.connect(alias="default", host=self.host, port=self.port)
            if self.db_name not in db.list_database():
                db.create_database(self.db_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Milvus database 检查/创建失败，将继续使用当前连接: %s", exc)
        connections.disconnect(alias="default")
        connections.connect(alias="default", host=self.host, port=self.port, db_name=self.db_name)
        logger.info("成功连接至 Milvus database: %s", self.db_name)
        self.create_collections_if_not_exists()
        self.init_indexes_and_load()

    def create_collections_if_not_exists(self):
        """根据你的表结构设计初始化三个集合"""
        
        # 1. Entities Collection
        if not utility.has_collection("Entities"):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="kb_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
                FieldSchema(name="relation_ids", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=1000, max_length=100)
            ]
            schema = CollectionSchema(fields, "实体表")
            Collection("Entities", schema)

        # 2. Relations Collection
        if not utility.has_collection("Relations"):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="kb_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="subject_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="object_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="relation", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="describe", dtype=DataType.VARCHAR, max_length=RELATION_DESCRIBE_MAX_LENGTH),
                FieldSchema(name="passage", dtype=DataType.VARCHAR, max_length=16384),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
                FieldSchema(name="passage_ids", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=100, max_length=100)
            ]
            schema = CollectionSchema(fields, "关系表")
            Collection("Relations", schema)
        else:
            self._warn_if_collection_missing_fields(
                "Relations",
                required_fields={"describe"},
                message=(
                    "Relations 集合缺少字段 %s。新的建库逻辑会生成 relation.describe 并将 embedding 切换到 describe 向量；"
                    "请重建该集合或执行 schema 迁移后再写入。"
                ),
            )

        # 3. Passage Collection
        if not utility.has_collection("Passage"):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="kb_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="docment_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="passage", dtype=DataType.VARCHAR, max_length=16384),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
            ]
            schema = CollectionSchema(fields, "段落表")
            Collection("Passage", schema)

    # ================= 新增的方法 =================
    def init_indexes_and_load(self):
        """为所有集合检查/创建 HNSW 索引，并加载到内存中"""
        collections_to_init = ["Entities", "Relations", "Passage"]
        
        # 统一的索引参数: HNSW, M=64, efConstruction=256
        index_params = {
            "metric_type": "COSINE", 
            "index_type": "HNSW",
            "params": {"M": 64, "efConstruction": 256}
        }

        for col_name in collections_to_init:
            if utility.has_collection(col_name):
                collection = Collection(col_name)
                
                # 检查是否已存在索引
                if not collection.has_index():
                    logger.info(f"正在为集合 {col_name} 创建 HNSW 索引 (M=64, ef=256)...")
                    collection.create_index(
                        field_name="embedding", 
                        index_params=index_params,
                        index_name=f"{col_name}_hnsw_index"
                    )
                    logger.info(f"集合 {col_name} 索引创建完成！")
                
                # 无论是否新建索引，都需要 Load 到内存才能搜索/后续操作不报错
                collection.load()
                logger.info(f"集合 {col_name} 已成功加载到内存。")
    # ============================================

    async def batch_insert_graph_data(self, payload: Dict[str, List[Dict[str, Any]]]):
        """将 Pipeline 产出的数据分发写入三个集合 (使用 upsert 确保更新存量数据)"""
        try:
            # 写入 Entities (表1)
            if payload["Table1_Entities"]:
                col_e = Collection("Entities")
                entity_rows = self._sanitize_rows_for_collection("Entities", payload["Table1_Entities"])
                for batch in self._iter_write_batches(entity_rows):
                    col_e.upsert(batch)
                col_e.flush()

            # 写入 Relations (表2)
            if payload["Table2_Relations"]:
                col_r = Collection("Relations")
                relation_rows = self._sanitize_rows_for_collection("Relations", payload["Table2_Relations"])
                for batch in self._iter_write_batches(relation_rows):
                    col_r.upsert(batch)
                col_r.flush()

            # 写入 Passage (表3)
            if payload["Table3_Passages"]:
                col_p = Collection("Passage")
                passage_rows = self._sanitize_rows_for_collection("Passage", payload["Table3_Passages"])
                for batch in self._iter_write_batches(passage_rows):
                    col_p.insert(batch)
                col_p.flush()
                
            logger.info("所有数据已成功存入 Milvus (已执行 Upsert)")
        except Exception as e:
            logger.error(f"Milvus 写入出错: {e}")
            raise e

    def get_entities_by_ids(self, kb_id: str, entity_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """根据 ID 批量获取实体及其关联的关系 ID"""
        if not entity_ids:
            return {}
        try:
            col = Collection("Entities")
            results: Dict[str, Dict[str, Any]] = {}
            for start in range(0, len(entity_ids), _ID_QUERY_BATCH_SIZE):
                batch_ids = entity_ids[start : start + _ID_QUERY_BATCH_SIZE]
                ids_str = ", ".join([f"'{eid}'" for eid in batch_ids])
                expr = f"kb_id == '{kb_id}' and id in [{ids_str}]"
                res = col.query(expr=expr, output_fields=["id", "name", "relation_ids", "embedding"])
                for item in res:
                    results[item["id"]] = item
            return results
        except Exception as e:
            logger.error(f"Milvus 查询实体失败: {e}")
            return {}

    def _warn_if_collection_missing_fields(self, collection_name: str, *, required_fields: set[str], message: str) -> None:
        try:
            collection = Collection(collection_name)
            schema_fields = {field.name for field in collection.schema.fields}
            missing_fields = sorted(required_fields - schema_fields)
            if missing_fields:
                logger.warning(message, ",".join(missing_fields))
        except Exception as exc:
            logger.warning("检查集合 %s schema 失败: %s", collection_name, exc)

# 实例化客户端供 Pipeline 调用
milvus_client = MilvusClient()
