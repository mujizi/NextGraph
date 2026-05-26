
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
    settings.milvus_db = "crx_text0515" # Using the same as show_context.py
    
    repository = MilvusGraphRepository(settings)
    embedder = build_query_embedder(settings)
    service = SearchService(settings=settings, repository=repository, embedder=embedder)

    query = "冰箱温度设置后多久会固定"
    
    request = SearchRequest(
        query=query,
        user_id="admin_user",
        kb_id="0521",
        mode=SearchMode.hybrid,
        top_k=8,
        entity_top_k=10,
        relation_top_k=15,
        expansion_degree=1
    )

    print(f"DEBUG: Query: {query}")
    
    # Trace entities
    entities = service._extract_query_entities(request)
    print(f"DEBUG: Extracted Entities: {entities}")
    
    for entity in entities:
        rows = repository.search_entities(embedder.embed(entity), user_id=request.user_id, kb_id=request.kb_id, limit=5)
        print(f"\nDEBUG: Entity Search for '{entity}':")
        for row in rows:
            print(f"  - {row.get('name')} (Score: {row.get('score')}, ID: {row.get('id')})")

    # Trace semantic route relations
    query_vector = embedder.embed(query)
    semantic_relations = repository.search_relations(query_vector, user_id=request.user_id, kb_id=request.kb_id, limit=10)
    print(f"\nDEBUG: Semantic Route Top 10 Relations:")
    for row in semantic_relations:
        # We need to get entity names for display
        subject_id = row.get("subject_id")
        object_id = row.get("object_id")
        entities = repository.get_entities_by_ids([subject_id, object_id], user_id=request.user_id, kb_id=request.kb_id)
        entity_map = {e["id"]: e["name"] for e in entities}
        s_name = entity_map.get(subject_id, subject_id)
        o_name = entity_map.get(object_id, object_id)
        print(f"  - {s_name} --({row.get('relation')})--> {o_name} (Score: {row.get('score')})")

if __name__ == "__main__":
    asyncio.run(main())
