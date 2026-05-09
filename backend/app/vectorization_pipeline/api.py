import json
import asyncio
import os
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import time
from openai import AsyncAzureOpenAI
import hashlib
from dotenv import load_dotenv
# 导入单步抽取函数
from app.triple_extraction.functions import _extract_single_chunk
from app.db.vector_client import milvus_client

load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter()

# 任务状态存储 (生产环境建议使用 Redis)
tasks_db: Dict[str, Dict[str, Any]] = {}

azure_aclient = AsyncAzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("NEXTGRAPH_AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get(
        "AZURE_OPENAI_ENDPOINT",
        "https://admin-1804-resource.cognitiveservices.azure.com/",
    ),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
)

class VectorizationRequest(BaseModel):
    user_id: str
    kb_id: str
    docment_id: str
    json_path: str
    max_concurrency: int = 10
    max_chunk_chars: int = 1500
    embedding_model: str = "text-embedding-3-small"

def generate_entity_id(kb_id: str, name: str) -> str:
    """使用 知识库ID + 实体名 生成唯一 Hash ID"""
    unique_string = f"{kb_id}_{name}"
    return "e_" + hashlib.md5(unique_string.encode('utf-8')).hexdigest()[:16]

def generate_relation_id(kb_id: str, subj: str, rel: str, obj: str) -> str:
    """使用 知识库ID + 三元组内容 生成唯一 Hash ID"""
    unique_string = f"{kb_id}_{subj}_{rel}_{obj}"
    return "r_" + hashlib.md5(unique_string.encode('utf-8')).hexdigest()[:16]

def merge_json_items(items: List[Dict], max_chars: int) -> List[str]:
    """将碎片化的 JSON items 合并，不切分超长块"""
    merged_chunks = []
    current_chunk = ""
    for item in items:
        text = item.get("text", "").strip()
        if not text: continue
        if len(current_chunk) + len(text) > max_chars and current_chunk:
            merged_chunks.append(current_chunk)
            current_chunk = text
        else:
            current_chunk = (current_chunk + "\n" + text) if current_chunk else text
    if current_chunk: merged_chunks.append(current_chunk)
    return merged_chunks

async def _get_embedding_with_retry(text: str, model: str, retries: int = 3) -> List[float]:
    """带重试机制的向量获取"""
    if not text or not text.strip():
        return []
    
    for i in range(retries):
        try:
            res = await azure_aclient.embeddings.create(
                input=text.replace("\n", " "), 
                model=model
            )
            return res.data[0].embedding
        except Exception as e:
            logger.warning("Embedding 尝试第 " + str(i+1) + " 次失败: " + str(e))
            if i < retries - 1:
                await asyncio.sleep(1 * (i + 1)) # 指数退避
            else:
                logger.error("Embedding 最终失败: " + str(text[:50]) + "...")
    return []

def update_task_status(task_id: str, step: str, msg: str, progress: float = 0, summary: dict = None):
    if task_id in tasks_db:
        tasks_db[task_id].update({
            "step": step,
            "msg": msg,
            "progress": progress,
            "updated_at": time.time()
        })
        if summary:
            tasks_db[task_id]["summary"] = summary

async def process_vectorization_task(task_id: str, request: VectorizationRequest):
    try:
        if not os.path.exists(request.json_path):
            update_task_status(task_id, "error", f"找不到文件: {request.json_path}")
            return

        # 1. 加载与合并
        update_task_status(task_id, "loading", "正在加载并合并文件...", 0.1)
        with open(request.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            items = data.get("items", [])
        chunks = merge_json_items(items, request.max_chunk_chars)
        total_chunks = len(chunks)
        msg = f"合并完成，共 {total_chunks} 个文本块"
        update_task_status(task_id, "init_done", msg, 0.2)

        # 2. 三元组抽取
        extracted_data = []
        sem = asyncio.Semaphore(request.max_concurrency)
        
        async def wrapped_extract(idx, chunk):
            async with sem:
                try:
                    triplets = await _extract_single_chunk(chunk)
                    return {"chunk_idx": idx, "chunk_text": chunk, "triplets": triplets}
                except Exception as e:
                    logger.error(f"Chunk {idx} 抽取失败: {e}")
                    return {"chunk_idx": idx, "chunk_text": chunk, "triplets": []}

        extract_tasks = [wrapped_extract(i, c) for i, c in enumerate(chunks)]
        count = 0
        for task in asyncio.as_completed(extract_tasks):
            result = await task
            extracted_data.append(result)
            count += 1
            progress = 0.2 + (count / total_chunks) * 0.3 # 占 30% 进度
            msg = f"[抽取进度] {count}/{total_chunks}"
            update_task_status(task_id, "extracting", msg, progress)

        # 3. 数据对齐
        update_task_status(task_id, "aligning", "正在对齐 ID...", 0.55)
        passages_payload, entities_map, relations_payload = [], {}, []

        # 定义字段长度上限 (与 Milvus Schema 保持一致)
        MAX_PASSAGE_LEN = 16000
        MAX_NAME_LEN = 500
        MAX_RELATION_LEN = 1000

        for idx, item in enumerate(extracted_data):
            c_id = f"{request.kb_id}_{request.docment_id}_c{str(idx+1).zfill(4)}"
            chunk_text = item["chunk_text"][:MAX_PASSAGE_LEN]
            
            passages_payload.append({
                "user_id": request.user_id, "kb_id": request.kb_id, "id": c_id,
                "passage": chunk_text, "docment_id": request.docment_id, "embedding": []
            })
            
            for trip in item["triplets"]:
                subj, rel, obj = trip["subject"], trip["relation"], trip["object"]
                if not subj or not obj: continue
                subj, rel, obj = subj[:MAX_NAME_LEN], rel[:MAX_RELATION_LEN], obj[:MAX_NAME_LEN]
                
                for name in [subj, obj]:
                    if name not in entities_map:
                        ent_id = generate_entity_id(request.kb_id, name)
                        entities_map[name] = {"id": ent_id, "name": name, "relation_ids": set(), "embedding": []}
                
                r_id = generate_relation_id(request.kb_id, subj, rel, obj)
                relations_payload.append({
                    "user_id": request.user_id, "kb_id": request.kb_id, "id": r_id,
                    "subject_id": entities_map[subj]["id"], "object_id": entities_map[obj]["id"],
                    "relation": rel, "passage": chunk_text, "passage_ids": [c_id], "embedding": []
                })
                entities_map[subj]["relation_ids"].add(r_id)
                entities_map[obj]["relation_ids"].add(r_id)

        entities_payload = list(entities_map.values())
        for e in entities_payload:
            e["relation_ids"] = list(e["relation_ids"])
            e.update({"user_id": request.user_id, "kb_id": request.kb_id})

        # 4. 向量化
        update_task_status(task_id, "vectorizing", "开始计算向量...", 0.6)
        v_sem = asyncio.Semaphore(request.max_concurrency)
        
        async def wrapped_embed(target: dict, text_key: str):
            async with v_sem:
                target["embedding"] = await _get_embedding_with_retry(target[text_key], request.embedding_model)

        embed_tasks = []
        for p in passages_payload: embed_tasks.append(wrapped_embed(p, "passage"))
        for e in entities_payload: embed_tasks.append(wrapped_embed(e, "name"))
        for r in relations_payload: embed_tasks.append(wrapped_embed(r, "relation"))
        
        v_count, total_v = 0, len(embed_tasks)
        for task in asyncio.as_completed(embed_tasks):
            await task
            v_count += 1
            if v_count % 10 == 0 or v_count == total_v:
                progress = 0.6 + (v_count / total_v) * 0.3 # 占 30% 进度
                msg = f"[向量化进度] {v_count}/{total_v}"
                update_task_status(task_id, "vectorizing", msg, progress)

        # 5. 持久化
        update_task_status(task_id, "storing", "正在写入数据库...", 0.95)
        valid_passages = [p for p in passages_payload if p["embedding"]]
        valid_entities = [e for e in entities_payload if e["embedding"]]
        valid_relations = [r for r in relations_payload if r["embedding"]]

        await milvus_client.batch_insert_graph_data({
            "Table1_Entities": valid_entities,
            "Table2_Relations": valid_relations,
            "Table3_Passages": valid_passages
        })
        
        summary = {
            "chunks": total_chunks,
            "entities": len(valid_entities),
            "relations": len(valid_relations),
            "failed_embeddings": total_v - (len(valid_passages) + len(valid_entities) + len(valid_relations))
        }
        update_task_status(task_id, "completed", "处理成功", 1.0, summary)

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        update_task_status(task_id, "error", f"任务失败: {str(e)}")

@router.post("/run")
async def run_vectorization_pipeline(request: VectorizationRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks_db[task_id] = {
        "id": task_id,
        "status": "pending",
        "step": "queued",
        "msg": "任务已入队",
        "progress": 0,
        "created_at": time.time()
    }
    background_tasks.add_task(process_vectorization_task, task_id, request)
    return {"task_id": task_id, "msg": "任务已启动"}

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="任务不存在")
    return tasks_db[task_id]

@router.get("/stream/{task_id}")
async def stream_task_progress(task_id: str):
    """通过 SSE 持续监控某个后台任务的进度"""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="任务不存在")

    async def event_generator():
        last_update = 0
        while True:
            task = tasks_db.get(task_id)
            if not task: break    
            # 只有在有更新时才发送，或者任务已结束
            if task["updated_at"] > last_update or task["step"] in ["completed", "error"]:
                yield f"data: {json.dumps(task, ensure_ascii=False)}\n\n"
                last_update = task["updated_at"]
            
            if task["step"] in ["completed", "error"]:
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
