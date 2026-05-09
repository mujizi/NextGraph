import asyncio
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# 初始化异步客户端 (生产环境请从配置/环境变量读取 API Key)

aclient = AsyncOpenAI(
    api_key="EMPTY", 
    base_url="http://127.0.0.1:8416/v1"
)
# ==========================================
# 定义大模型严格输出的数据结构
# ==========================================
class Triplet(BaseModel):
    subject: str = Field(description="主语实体名称")
    relation: str = Field(description="实体间的关系（动词、状态或描述）")
    object: str = Field(description="宾语实体名称")

class TripletExtractionResult(BaseModel):
    triplets: List[Triplet]

# ==========================================
# 核心大模型调用逻辑
# ==========================================
async def _extract_single_chunk(chunk_text: str, retries: int = 3) -> List[Dict[str, str]]:
    """调用大模型进行三元组抽取，带重试机制"""
    
    system_prompt = (
        "你是一个专业的知识图谱抽取专家。你的任务是从提供的文本中提取核心的实体-关系-实体三元组。\n\n"
        "### 抽取规则：\n"
        "1. **实体(Subject & Object)**：应当是具体的人、地点、物、概念或组织。避免使用代词（如“他”、“这”）。实体名称应保持简洁且一致。\n"
        "2. **关系(Relation)**：应当明确描述两个实体之间的关联（如“作者”、“位于”、“属于”、“创立了”）。关系应尽量简短且具有概括性。\n"
        "3. **去重与精炼**：不要提取琐碎或无意义的信息。如果同一个关系在文本中多次提及，只需提取一次。\n"
        "4. **完整性**：确保提取出的三元组在逻辑上是自洽的（即：[实体A] -> [关系] -> [实体B]）。\n"
        "5. **语言**：如果原文是中文，请使用中文提取。"
    )

    user_prompt = f"请分析以下文本，并提取出所有的核心三元组：\n\n文本内容：\n{chunk_text}"

    for i in range(retries):
        try:
            # 增加 timeout 参数以避免无限等待
            response = await aclient.beta.chat.completions.parse(
                model="/ai/qwen3.5_9b", 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=TripletExtractionResult,
                temperature=0.1,
                timeout=60.0 # 60秒超时
            )
            parsed_data = response.choices[0].message.parsed
            return [t.model_dump() for t in parsed_data.triplets]
        except Exception as e:
            if i < retries - 1:
                wait_time = (i + 1) * 2
                logger.warning(f"三元组抽取尝试第 {i+1} 次失败: {e}，{wait_time}s 后重试...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"三元组抽取最终失败: {e}")
    return []

# ==========================================
# 对外暴露的并发批次处理 API
# ==========================================
async def batch_extract_triplets_with_concurrency(
    chunks: List[str], 
    max_concurrency: int = 10
) -> List[Dict[str, Any]]:
    """
    带并发控制的批量抽三元组函数
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def _process(idx: int, chunk: str):
        async with semaphore:
            triplets = await _extract_single_chunk(chunk)
            return {"chunk_idx": idx, "chunk_text": chunk, "triplets": triplets}

    tasks = [_process(idx, chunk) for idx, chunk in enumerate(chunks)]
    return await asyncio.gather(*tasks)