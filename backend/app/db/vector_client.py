import logging
from typing import List, Dict, Any
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

logger = logging.getLogger(__name__)

class MilvusClient:
    def __init__(self, host="10.1.80.16", port="19530"):
        # 建立连接
        try:
            connections.connect(
                alias="default",
                host=host,
                port=port,
                db_name="crx"
            )
            logger.info("成功连接至 Milvus")
            
            # 1. 确保集合存在
            self.create_collections_if_not_exists()
            
            # 2. 确保索引存在并加载到内存 (新增这一步)
            self.init_indexes_and_load()
            
        except Exception as e:
            logger.error(f"Milvus 连接失败: {e}")

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
        """将 Pipeline 产出的数据分发写入三个集合"""
        try:
            # 写入 Entities (表1)
            if payload["Table1_Entities"]:
                col_e = Collection("Entities")
                col_e.insert(payload["Table1_Entities"])
                col_e.flush()

            # 写入 Relations (表2)
            if payload["Table2_Relations"]:
                col_r = Collection("Relations")
                col_r.insert(payload["Table2_Relations"])
                col_r.flush()

            # 写入 Passage (表3)
            if payload["Table3_Passages"]:
                col_p = Collection("Passage") 
                col_p.insert(payload["Table3_Passages"])
                col_p.flush()
                
            logger.info("所有数据已成功存入 Milvus")
        except Exception as e:
            logger.error(f"Milvus 写入出错: {e}")
            raise e

# 实例化客户端供 Pipeline 调用
milvus_client = MilvusClient()