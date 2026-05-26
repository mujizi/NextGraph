
import asyncio
import json
import os
import sys
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
    # Force use the crx database as requested by user
    settings.milvus_db = "crx"
    
    repository = MilvusGraphRepository(settings)
    embedder = build_query_embedder(settings)
    service = SearchService(settings=settings, repository=repository, embedder=embedder)

    query = "坎贝尔是如何利用弗洛伊德和荣格的心理学理论来解释‘神话’与‘梦’的关系的？"
    request = SearchRequest(
        query=query,
        user_id="admin_user",
        kb_id="test_kb",
        mode=SearchMode.hybrid,
        top_k=5,
        entity_top_k=5,
        relation_top_k=5
    )

    print(f"--- [STAGE 0] Query: {query} ---")
    
    # 1. Entity Extraction
    print("\n--- [STAGE 1] Entity Extraction ---")
    try:
        entities = service.query_entity_extractor.extract(query, user_id=request.user_id, kb_id=request.kb_id, semantic_limit=request.entity_top_k)
        print(f"Extracted Entities: {entities}")
    except Exception as e:
        print(f"Extraction failed: {e}")
        entities = []

    # 2. Search Trace
    print("\n--- [STAGE 2] Search Trace (Internal Data) ---")
    try:
        trace = service.search_trace(request)
        
        print("\n[Seed Entities Found]:")
        for ent in trace['trace']['seed_entities']:
            print(f"- {ent['name']} (Score: {ent['score']:.4f})")
            
        print("\n[Expanded Relations]:")
        for rel in trace['trace']['result_relations']:
            print(f"- {rel['subject_name']} --[{rel['relation']}]--> {rel['object_name']} (Score: {rel['score']:.4f})")
            
        print("\n[Grounded Passages (Text Snippets)]:")
        for psg in trace['trace']['grounded_passages'][:2]:
            print(f"- ID: {psg['id']}\n  Content: {psg['passage'][:300]}...\n")
            
    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
