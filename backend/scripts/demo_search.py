from __future__ import annotations

import json

from backend.app.core.config import Settings
from backend.app.search.local_global_hybrid.schemas import SearchMode, SearchRequest
from backend.app.search.local_global_hybrid.service import SearchService
from backend.app.vector_database.search.embedder import build_query_embedder
from backend.app.vector_database.search.repository import MilvusGraphRepository

DEMO_USER_ID = "demo-user-001"
DEMO_KB_ID = "demo-kb-movie"


def seed_demo_data(service: SearchService) -> None:
    repository = service.repository
    embedder = service.embedder

    repository.create_demo_collections(dimension=embedder.dimension, drop_existing=True)

    entity_rows = [
        {"id": "ent_nolan", "name": "克里斯托弗·诺兰", "embedding": embedder.embed("克里斯托弗·诺兰"), "relation_ids": ["rel_directed_interstellar", "rel_directed_inception", "rel_collab_hans"], "user_id": DEMO_USER_ID, "kb_id": DEMO_KB_ID},
        {"id": "ent_interstellar", "name": "星际穿越", "embedding": embedder.embed("星际穿越"), "relation_ids": ["rel_directed_interstellar", "rel_music_interstellar", "rel_starring_interstellar", "rel_distributed_interstellar"], "user_id": DEMO_USER_ID, "kb_id": DEMO_KB_ID},
        {"id": "ent_hans", "name": "汉斯·季默", "embedding": embedder.embed("汉斯·季默"), "relation_ids": ["rel_music_interstellar", "rel_collab_hans"], "user_id": DEMO_USER_ID, "kb_id": DEMO_KB_ID},
        {"id": "ent_inception", "name": "盗梦空间", "embedding": embedder.embed("盗梦空间"), "relation_ids": ["rel_directed_inception"], "user_id": DEMO_USER_ID, "kb_id": DEMO_KB_ID},
        {"id": "ent_warner", "name": "华纳兄弟", "embedding": embedder.embed("华纳兄弟"), "relation_ids": ["rel_distributed_interstellar"], "user_id": DEMO_USER_ID, "kb_id": DEMO_KB_ID},
        {"id": "ent_mcconaughey", "name": "马修·麦康纳", "embedding": embedder.embed("马修·麦康纳"), "relation_ids": ["rel_starring_interstellar"], "user_id": DEMO_USER_ID, "kb_id": DEMO_KB_ID},
    ]

    relation_rows = [
        {"id": "rel_directed_interstellar", "subject_id": "ent_nolan", "object_id": "ent_interstellar", "relation": "导演", "passage": "《星际穿越》由克里斯托弗·诺兰执导。", "embedding": embedder.embed("克里斯托弗·诺兰 导演 星际穿越"), "passage_ids": ["passage-001"], "docment_id": "doc-001", "user_id": DEMO_USER_ID, "kb_id": DEMO_KB_ID},
        {"id": "rel_music_interstellar", "subject_id": "ent_hans", "object_id": "ent_interstellar", "relation": "配乐", "passage": "《星际穿越》的配乐由汉斯·季默创作。", "embedding": embedder.embed("汉斯·季默 为 星际穿越 配乐"), "passage_ids": ["passage-002"], "docment_id": "doc-001", "user_id": DEMO_USER_ID, "kb_id": DEMO_KB_ID},
        {"id": "rel_directed_inception", "subject_id": "ent_nolan", "object_id": "ent_inception", "relation": "导演", "passage": "克里斯托弗·诺兰还导演了《盗梦空间》。", "embedding": embedder.embed("克里斯托弗·诺兰 导演 盗梦空间"), "passage_ids": ["passage-003"], "docment_id": "doc-002", "user_id": DEMO_USER_ID, "kb_id": DEMO_KB_ID},
        {"id": "rel_distributed_interstellar", "subject_id": "ent_warner", "object_id": "ent_interstellar", "relation": "发行", "passage": "《星际穿越》由华纳兄弟发行。", "embedding": embedder.embed("华纳兄弟 发行 星际穿越"), "passage_ids": ["passage-004"], "docment_id": "doc-001", "user_id": DEMO_USER_ID, "kb_id": DEMO_KB_ID},
        {"id": "rel_starring_interstellar", "subject_id": "ent_mcconaughey", "object_id": "ent_interstellar", "relation": "主演", "passage": "马修·麦康纳主演了《星际穿越》。", "embedding": embedder.embed("马修·麦康纳 主演 星际穿越"), "passage_ids": ["passage-005"], "docment_id": "doc-001", "user_id": DEMO_USER_ID, "kb_id": DEMO_KB_ID},
        {"id": "rel_collab_hans", "subject_id": "ent_nolan", "object_id": "ent_hans", "relation": "合作", "passage": "克里斯托弗·诺兰曾与汉斯·季默多次合作。", "embedding": embedder.embed("克里斯托弗·诺兰 与 汉斯·季默 合作"), "passage_ids": ["passage-006"], "docment_id": "doc-003", "user_id": DEMO_USER_ID, "kb_id": DEMO_KB_ID},
    ]

    passage_rows = [
        {"id": "passage-001", "passage": "《星际穿越》由克里斯托弗·诺兰执导。", "embedding": embedder.embed("《星际穿越》由克里斯托弗·诺兰执导。"), "docment_id": "doc-001", "user_id": DEMO_USER_ID, "kb_id": DEMO_KB_ID},
        {"id": "passage-002", "passage": "《星际穿越》的配乐由汉斯·季默创作。", "embedding": embedder.embed("《星际穿越》的配乐由汉斯·季默创作。"), "docment_id": "doc-001", "user_id": DEMO_USER_ID, "kb_id": DEMO_KB_ID},
        {"id": "passage-003", "passage": "克里斯托弗·诺兰还导演了《盗梦空间》。", "embedding": embedder.embed("克里斯托弗·诺兰还导演了《盗梦空间》。"), "docment_id": "doc-002", "user_id": DEMO_USER_ID, "kb_id": DEMO_KB_ID},
        {"id": "passage-004", "passage": "《星际穿越》由华纳兄弟发行。", "embedding": embedder.embed("《星际穿越》由华纳兄弟发行。"), "docment_id": "doc-001", "user_id": DEMO_USER_ID, "kb_id": DEMO_KB_ID},
        {"id": "passage-005", "passage": "马修·麦康纳主演了《星际穿越》。", "embedding": embedder.embed("马修·麦康纳主演了《星际穿越》。"), "docment_id": "doc-001", "user_id": DEMO_USER_ID, "kb_id": DEMO_KB_ID},
        {"id": "passage-006", "passage": "克里斯托弗·诺兰曾与汉斯·季默多次合作。", "embedding": embedder.embed("克里斯托弗·诺兰曾与汉斯·季默多次合作。"), "docment_id": "doc-003", "user_id": DEMO_USER_ID, "kb_id": DEMO_KB_ID},
    ]

    repository.delete_by_ids(repository.settings.relations_collection, [row["id"] for row in relation_rows])
    repository.delete_by_ids(repository.settings.entities_collection, [row["id"] for row in entity_rows])
    repository.delete_by_ids(repository.settings.passages_collection, [row["id"] for row in passage_rows])
    repository.insert_entities(entity_rows)
    repository.insert_relations(relation_rows)
    repository.insert_passages(passage_rows)


def main() -> None:
    settings = Settings(
        entities_collection="entities_demo",
        relations_collection="relations_demo",
        passages_collection="passages_demo",
    )
    repository = MilvusGraphRepository(settings)
    service = SearchService(settings=settings, repository=repository, embedder=build_query_embedder(settings))
    seed_demo_data(service)

    query = "谁为星际穿越配乐？"
    for mode in [SearchMode.triple, SearchMode.semantic, SearchMode.hybrid]:
        response = service.search(SearchRequest(query=query, user_id=DEMO_USER_ID, kb_id=DEMO_KB_ID, mode=mode, top_k=3, entity_top_k=4, relation_top_k=4))
        print(f"\n=== {mode.value.upper()} ===")
        print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
