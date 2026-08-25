import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
BACKEND_ROOT = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from backend.app.core.config import get_settings
from backend.app.search.local_global_hybrid.service import SearchService
from backend.app.search.local_global_hybrid.schemas import SearchRequest, SearchMode
from backend.app.vector_database.search.embedder import build_query_embedder
from backend.app.vector_database.search.repository import MilvusGraphRepository

async def main():
    settings = get_settings()
    settings.milvus_db = "crx_0529"
    
    repository = MilvusGraphRepository(settings)
    embedder = build_query_embedder(settings)
    service = SearchService(settings=settings, repository=repository, embedder=embedder)

    query = "人物小传怎么写"
    
    request = SearchRequest(
        query=query,
        user_id="admin_user",
        kb_id="0529",
        mode=SearchMode.hybrid,
        top_k=8,
        entity_top_k=10,
        relation_top_k=15,
        expansion_degree=1
    )

    print(f"🔍 正在检索问题: {query}\n" + "="*60)
    
    response = service.search(request)

    print("\n[ 模拟提示词上下文 (PROMPT CONTEXT) ]")
    print("以下是如果系统有'总结阶段'，大模型会看到的背景知识：\n")

    print("--- 1. 核心图谱关系 (Knowledge Graph Triples) ---")
    if response.results:
        for i, rel in enumerate(response.results):
            print(f"[{i+1}] {rel.subject_name} --({rel.relation})--> {rel.object_name}")
    else:
        print("未找到相关关系。")

    print("\n--- 2. 参考文本片段 (Grounded Passages) ---")
    if response.grounded_passages:
        for i, psg in enumerate(response.grounded_passages):
            # Clean up text for display
            clean_text = psg.passage.replace('\n', ' ').strip()
            print(f"证据 {i+1} (ID: {psg.id}):")
            print(f"   \"{clean_text[:400]}...\"\n")
    else:
        print("未找到相关文本片段。")

    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
