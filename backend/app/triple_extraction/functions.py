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
    describe: str = Field(description="对该三元组关系的简短描述，便于做关系语义检索")

class TripletExtractionResult(BaseModel):
    triplets: List[Triplet]


def clean_triplets(triplets: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Normalize extracted triplets and drop obviously invalid duplicates."""

    cleaned: List[Dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for triplet in triplets:
        subject = str(triplet.get("subject", "")).strip()
        relation = str(triplet.get("relation", "")).strip()
        object_ = str(triplet.get("object", "")).strip()
        describe = str(triplet.get("describe", "")).strip()

        if not subject or not relation or not object_:
            continue
        if subject == object_:
            continue

        dedupe_key = (subject.casefold(), relation.casefold(), object_.casefold())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        cleaned.append(
            {
                "subject": subject,
                "relation": relation,
                "object": object_,
                "describe": describe or f"{subject} {relation} {object_}",
            }
        )

    return cleaned

# ==========================================
# 核心大模型调用逻辑
# ==========================================
async def _extract_single_chunk(chunk_text: str, retries: int = 3) -> List[Dict[str, str]]:
    """调用大模型进行三元组抽取，带重试机制"""

    system_prompt = (
        "你是一个专业的知识图谱与 Graph RAG 三元组抽取专家。"
        "你的任务是从提供的文本中提取核心的实体-关系-实体三元组，"
        "用于后续图数据库构建、语义检索和问答推理。\n\n"

        "### 抽取目标：\n"
        "1. 只抽取文本中明确表达或上下文强烈支持的核心事实关系。\n"
        "2. 优先抽取对问答、检索、推理有价值的信息，不追求数量。\n"
        "3. 不要基于常识补充原文没有表达的信息。\n"
        "4. 如果文本中没有有价值的关系事实，返回空列表。\n"
        "5. 每个文本块最多抽取 20 条最重要的三元组。\n\n"

        "### 字段含义：\n"
        "1. subject：关系的主语实体，应是短、稳定、可复用的节点名称。\n"
        "2. relation：subject 与 object 之间的关系，应简短、明确，适合作为图数据库中的边。\n"
        "3. object：关系的宾语实体，应是短、稳定、可复用的节点名称。\n"
        "4. describe：对该关系的自然语言说明，用于关系语义检索。"
        "describe 应保留原文中的关键条件、否定、可能性、因果、用途、限制或上下文信息。\n\n"

        "### 实体规则：\n"
        "1. subject 和 object 可以是人物、地点、组织、产品、物品、概念、指标、故障、状态、现象、操作、规则、风险、问题或短事件。\n"
        "2. 实体名称必须是短语，不要是完整句子。优先提取核心名词、名词短语或压缩后的短状态。\n"
        "3. 实体通常不超过 15 个汉字或 8 个英文单词。\n"
        "4. 避免使用代词，例如“他”、“它”、“这”、“该方法”。如果上下文能明确指代对象，应还原成具体实体；否则不要抽取。\n"
        "5. 不要把原因句、条件句、操作说明、完整现象描述、完整结论句直接当成实体。\n"
        "6. 对于长状态或长事件，应压缩成短实体。例如："
        "“温度控制未设置在合适位置”应压缩为“温度控制设置异常”；"
        "“用户无法登录系统”应压缩为“登录失败”；"
        "“设备长时间运行后温度升高”可压缩为“设备温度升高”。\n\n"

        "### 关系规则：\n"
        "1. relation 应简短且具有概括性，例如“属于”、“包含”、“位于”、“创立了”、“负责”、“使用”、“依赖”、“导致”、“可能导致”、“影响”、“解决”、“适用于”、“要求”、“禁止”、“表现为”。\n"
        "2. relation 不要写成长句。详细语义、条件和上下文应放入 describe。\n"
        "3. 如果原文表达因果关系，应优先抽取为“导致”、“可能导致”、“引发”、“影响”等关系。\n"
        "4. 如果原文表达解决方案，应优先抽取为“解决”、“缓解”、“用于处理”等关系。\n"
        "5. 如果原文表达组成结构，应优先抽取为“包含”、“由...组成”、“属于”等关系。\n"
        "6. 如果原文表达限制、禁止、要求、建议，应在 relation 或 describe 中保留这种语气。\n\n"

        "### 否定、条件和可能性规则：\n"
        "1. 不要丢失否定语义。例如“温度控制未设置在合适位置”不能抽成“温度控制 -> 设置在 -> 合适位置”。\n"
        "2. 如果原文有“未、不、不能、禁止、避免”等否定表达，必须在 relation 或 describe 中体现。\n"
        "3. 如果原文有“可能、容易、会、可、通常、建议、需要、必须、应当”等语气，必须在 describe 中体现。\n"
        "4. 如果关系只在特定条件下成立，describe 必须保留条件。\n\n"

        "### describe 规则：\n"
        "1. 每条三元组都必须生成 describe。\n"
        "2. describe 用 1 句自然语言概括这条关系在原文中的具体含义。\n"
        "3. describe 应包含 subject、relation、object 的语义，但不要只是机械拼接字段。\n"
        "4. describe 应优先保留原文中的关键词、限定条件、否定、因果、用途、时间或上下文。\n"
        "5. describe 长度通常控制在 10 到 60 个汉字或 8 到 40 个英文单词。\n\n"

        "### 示例：\n"
        "输入文本：如果温度控制未设置在合适位置，可能导致冷却不足。\n"
        "正确三元组：subject=温度控制设置异常, relation=可能导致, object=冷却不足, describe=当温度控制未设置在合适位置时，可能导致冷却不足。\n"
        "错误三元组：subject=温度控制, relation=设置在, object=合适位置, describe=温度控制设置在合适位置。\n\n"

        "输入文本：清理过滤网可以缓解空调制冷效果下降的问题。\n"
        "正确三元组：subject=清理过滤网, relation=缓解, object=制冷效果下降, describe=清理过滤网可以缓解空调制冷效果下降的问题。\n\n"

        "### 去重与质量控制：\n"
        "1. 不要提取琐碎、重复或无意义的信息。\n"
        "2. 如果同一个事实在文本中多次出现，只输出一次。\n"
        "3. 不要输出 subject、relation 或 object 为空的三元组。\n"
        "4. 不要输出 subject 和 object 完全相同的三元组。\n"
        "5. 如果一个关系缺少明确依据，不要输出。\n"
        "6. 如果原文是中文，请使用中文提取；如果原文是英文，请使用英文提取。\n\n"

        "### 输出要求：\n"
        "严格按照 TripletExtractionResult 的结构输出。"
        "每个 triplet 只能包含 subject、relation、object、describe 四个字段。"
    )

    user_prompt = f"请分析以下文本，并提取核心三元组：\n\n文本内容：\n{chunk_text}"

    for i in range(retries):
        try:
            response = await aclient.beta.chat.completions.parse(
                model="/ai/qwen3.5_9b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=TripletExtractionResult,
                temperature=0,
                timeout=60.0,
            )

            parsed_data = response.choices[0].message.parsed

            if not parsed_data or not parsed_data.triplets:
                return []

            triplets = [t.model_dump() for t in parsed_data.triplets]
            return clean_triplets(triplets)

        except Exception as e:
            if i < retries - 1:
                wait_time = (i + 1) * 2
                logger.warning(f"三元组抽取尝试第 {i + 1} 次失败: {e}，{wait_time}s 后重试...")
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
