import logging
import os
from typing import List, Dict, Any
from urllib.parse import urlparse

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, db, utility

logger = logging.getLogger(__name__)

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
                FieldSchema(name="passage", dtype=DataType.VARCHAR, max_length=16384),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
                FieldSchema(name="passage_ids", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=100, max_length=100)
            ]
            schema = CollectionSchema(fields, "关系表")
            Collection("Relations", schema)

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
                col_e.upsert(payload["Table1_Entities"])
                col_e.flush()

            # 写入 Relations (表2)
            if payload["Table2_Relations"]:
                col_r = Collection("Relations")
                col_r.upsert(payload["Table2_Relations"])
                col_r.flush()

            # 写入 Passage (表3)
            if payload["Table3_Passages"]:
                col_p = Collection("Passage") 
                col_p.insert(payload["Table3_Passages"])
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
            # 构造查询表达式
            ids_str = ", ".join([f"'{eid}'" for eid in entity_ids])
            expr = f"kb_id == '{kb_id}' and id in [{ids_str}]"
            res = col.query(expr=expr, output_fields=["id", "name", "relation_ids", "embedding"])
            return {item["id"]: item for item in res}
        except Exception as e:
            logger.error(f"Milvus 查询实体失败: {e}")
            return {}

# 实例化客户端供 Pipeline 调用
milvus_client = MilvusClient()
