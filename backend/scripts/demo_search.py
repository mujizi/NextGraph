from __future__ import annotations

import json
from collections import defaultdict

from backend.app.core.config import Settings
from backend.app.search.local_global_hybrid.schemas import SearchMode, SearchRequest
from backend.app.search.local_global_hybrid.service import SearchService
from backend.app.vector_database.search.embedder import build_query_embedder
from backend.app.vector_database.search.repository import MilvusGraphRepository

DEMO_MOVIE_USER_ID = "demo-user-001"
DEMO_MOVIE_KB_ID = "demo-kb-movie"
DEMO_LAB_USER_ID = "demo-user-001"
DEMO_LAB_KB_ID = "demo-kb-lab"
DEMO_HISTORY_USER_ID = "demo-user-002"
DEMO_HISTORY_KB_ID = "demo-kb-history"


def _build_entity_row(embedder, *, user_id: str, kb_id: str, entity_id: str, name: str, relation_ids: list[str]):
    return {
        "id": entity_id,
        "name": name,
        "embedding": embedder.embed(name),
        "relation_ids": relation_ids,
        "user_id": user_id,
        "kb_id": kb_id,
    }


def _build_relation_row(
    embedder,
    *,
    user_id: str,
    kb_id: str,
    relation_id: str,
    subject_id: str,
    object_id: str,
    relation: str,
    passage: str,
    passage_id: str,
    docment_id: str,
    embedding_text: str,
):
    return {
        "id": relation_id,
        "subject_id": subject_id,
        "object_id": object_id,
        "relation": relation,
        "passage": passage,
        "embedding": embedder.embed(embedding_text),
        "passage_ids": [passage_id],
        "docment_id": docment_id,
        "user_id": user_id,
        "kb_id": kb_id,
    }


def _build_passage_row(embedder, *, user_id: str, kb_id: str, passage_id: str, passage: str, docment_id: str):
    return {
        "id": passage_id,
        "passage": passage,
        "embedding": embedder.embed(passage),
        "docment_id": docment_id,
        "user_id": user_id,
        "kb_id": kb_id,
    }


def seed_demo_data(service: SearchService) -> None:
    repository = service.repository
    embedder = service.embedder
    repository.create_demo_collections(dimension=embedder.dimension, drop_existing=True)

    relation_links: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    entities: list[dict] = []
    relations: list[dict] = []
    passages: list[dict] = []

    movie_relations = [
        ("rel_directed_interstellar", "ent_nolan", "ent_interstellar", "导演", "《星际穿越》由克里斯托弗·诺兰执导。", "passage-001", "doc-001", "克里斯托弗·诺兰 导演 星际穿越"),
        ("rel_music_interstellar", "ent_hans", "ent_interstellar", "配乐", "《星际穿越》的配乐由汉斯·季默创作。", "passage-002", "doc-001", "汉斯·季默 为 星际穿越 配乐"),
        ("rel_directed_inception", "ent_nolan", "ent_inception", "导演", "克里斯托弗·诺兰还导演了《盗梦空间》。", "passage-003", "doc-002", "克里斯托弗·诺兰 导演 盗梦空间"),
        ("rel_distributed_interstellar", "ent_warner", "ent_interstellar", "发行", "《星际穿越》由华纳兄弟发行。", "passage-004", "doc-001", "华纳兄弟 发行 星际穿越"),
        ("rel_starring_interstellar", "ent_mcconaughey", "ent_interstellar", "主演", "马修·麦康纳主演了《星际穿越》。", "passage-005", "doc-001", "马修·麦康纳 主演 星际穿越"),
        ("rel_collab_hans", "ent_nolan", "ent_hans", "合作", "克里斯托弗·诺兰曾与汉斯·季默多次合作。", "passage-006", "doc-003", "克里斯托弗·诺兰 与 汉斯·季默 合作"),
    ]
    movie_names = {
        "ent_nolan": "克里斯托弗·诺兰",
        "ent_interstellar": "星际穿越",
        "ent_hans": "汉斯·季默",
        "ent_inception": "盗梦空间",
        "ent_warner": "华纳兄弟",
        "ent_mcconaughey": "马修·麦康纳",
    }

    lab_relations = [
        ("rel_alice_knows_bob", "ent_alice", "ent_bob", "认识", "Alice knows Bob from the robotics lab.", "passage-101", "doc-lab-001", "Alice knows Bob"),
        ("rel_bob_works_carol", "ent_bob", "ent_carol", "合作", "Bob works with Carol on the vision pipeline.", "passage-102", "doc-lab-001", "Bob works with Carol"),
        ("rel_carol_mentors_alice", "ent_carol", "ent_alice", "指导", "Carol mentors Alice on graph retrieval experiments.", "passage-103", "doc-lab-002", "Carol mentors Alice"),
        ("rel_alice_builds_indexer", "ent_alice", "ent_indexer", "构建", "Alice builds the indexing service for the lab knowledge base.", "passage-104", "doc-lab-003", "Alice builds indexing service"),
    ]
    lab_names = {
        "ent_alice": "Alice",
        "ent_bob": "Bob",
        "ent_carol": "Carol",
        "ent_indexer": "Indexing Service",
    }

    history_relations = [
        ("rel_qin_unify_china", "ent_qin", "ent_china", "统一", "秦始皇完成了对中国的统一。", "passage-201", "doc-his-001", "秦始皇 统一 中国"),
        ("rel_china_capital_xianyang", "ent_china", "ent_xianyang", "都城", "秦朝时期重要都城是咸阳。", "passage-202", "doc-his-001", "秦朝 都城 咸阳"),
    ]
    history_names = {
        "ent_qin": "秦始皇",
        "ent_china": "中国",
        "ent_xianyang": "咸阳",
    }

    for rel in movie_relations:
        rid, sid, oid, relation, passage, pid, did, embed_text = rel
        relations.append(
            _build_relation_row(
                embedder,
                user_id=DEMO_MOVIE_USER_ID,
                kb_id=DEMO_MOVIE_KB_ID,
                relation_id=rid,
                subject_id=sid,
                object_id=oid,
                relation=relation,
                passage=passage,
                passage_id=pid,
                docment_id=did,
                embedding_text=embed_text,
            )
        )
        passages.append(_build_passage_row(embedder, user_id=DEMO_MOVIE_USER_ID, kb_id=DEMO_MOVIE_KB_ID, passage_id=pid, passage=passage, docment_id=did))
        relation_links[(DEMO_MOVIE_USER_ID, DEMO_MOVIE_KB_ID, sid)].append(rid)
        relation_links[(DEMO_MOVIE_USER_ID, DEMO_MOVIE_KB_ID, oid)].append(rid)

    for rel in lab_relations:
        rid, sid, oid, relation, passage, pid, did, embed_text = rel
        relations.append(
            _build_relation_row(
                embedder,
                user_id=DEMO_LAB_USER_ID,
                kb_id=DEMO_LAB_KB_ID,
                relation_id=rid,
                subject_id=sid,
                object_id=oid,
                relation=relation,
                passage=passage,
                passage_id=pid,
                docment_id=did,
                embedding_text=embed_text,
            )
        )
        passages.append(_build_passage_row(embedder, user_id=DEMO_LAB_USER_ID, kb_id=DEMO_LAB_KB_ID, passage_id=pid, passage=passage, docment_id=did))
        relation_links[(DEMO_LAB_USER_ID, DEMO_LAB_KB_ID, sid)].append(rid)
        relation_links[(DEMO_LAB_USER_ID, DEMO_LAB_KB_ID, oid)].append(rid)

    for rel in history_relations:
        rid, sid, oid, relation, passage, pid, did, embed_text = rel
        relations.append(
            _build_relation_row(
                embedder,
                user_id=DEMO_HISTORY_USER_ID,
                kb_id=DEMO_HISTORY_KB_ID,
                relation_id=rid,
                subject_id=sid,
                object_id=oid,
                relation=relation,
                passage=passage,
                passage_id=pid,
                docment_id=did,
                embedding_text=embed_text,
            )
        )
        passages.append(_build_passage_row(embedder, user_id=DEMO_HISTORY_USER_ID, kb_id=DEMO_HISTORY_KB_ID, passage_id=pid, passage=passage, docment_id=did))
        relation_links[(DEMO_HISTORY_USER_ID, DEMO_HISTORY_KB_ID, sid)].append(rid)
        relation_links[(DEMO_HISTORY_USER_ID, DEMO_HISTORY_KB_ID, oid)].append(rid)

    for entity_id, name in movie_names.items():
        entities.append(_build_entity_row(embedder, user_id=DEMO_MOVIE_USER_ID, kb_id=DEMO_MOVIE_KB_ID, entity_id=entity_id, name=name, relation_ids=relation_links[(DEMO_MOVIE_USER_ID, DEMO_MOVIE_KB_ID, entity_id)]))
    for entity_id, name in lab_names.items():
        entities.append(_build_entity_row(embedder, user_id=DEMO_LAB_USER_ID, kb_id=DEMO_LAB_KB_ID, entity_id=entity_id, name=name, relation_ids=relation_links[(DEMO_LAB_USER_ID, DEMO_LAB_KB_ID, entity_id)]))
    for entity_id, name in history_names.items():
        entities.append(_build_entity_row(embedder, user_id=DEMO_HISTORY_USER_ID, kb_id=DEMO_HISTORY_KB_ID, entity_id=entity_id, name=name, relation_ids=relation_links[(DEMO_HISTORY_USER_ID, DEMO_HISTORY_KB_ID, entity_id)]))

    repository.delete_by_ids(repository.settings.relations_collection, [row["id"] for row in relations])
    repository.delete_by_ids(repository.settings.entities_collection, [row["id"] for row in entities])
    repository.delete_by_ids(repository.settings.passages_collection, [row["id"] for row in passages])
    repository.insert_entities(entities)
    repository.insert_relations(relations)
    repository.insert_passages(passages)


def main() -> None:
    settings = Settings(
        entities_collection="entities_demo",
        relations_collection="relations_demo",
        passages_collection="passages_demo",
    )
    repository = MilvusGraphRepository(settings)
    service = SearchService(settings=settings, repository=repository, embedder=build_query_embedder(settings))
    seed_demo_data(service)

    scenarios = [
        {"label": "电影库-配乐问题", "query": "谁为星际穿越配乐？", "user_id": DEMO_MOVIE_USER_ID, "kb_id": DEMO_MOVIE_KB_ID},
        {"label": "电影库-导演问题", "query": "谁导演了盗梦空间？", "user_id": DEMO_MOVIE_USER_ID, "kb_id": DEMO_MOVIE_KB_ID},
        {"label": "实验室库-两跳关系", "query": "谁指导了 Alice？", "user_id": DEMO_LAB_USER_ID, "kb_id": DEMO_LAB_KB_ID},
        {"label": "实验室库-合作关系", "query": "Bob 和谁合作？", "user_id": DEMO_LAB_USER_ID, "kb_id": DEMO_LAB_KB_ID},
        {"label": "历史库-统一中国", "query": "谁统一了中国？", "user_id": DEMO_HISTORY_USER_ID, "kb_id": DEMO_HISTORY_KB_ID},
        {"label": "同用户不同知识库隔离", "query": "谁为星际穿越配乐？", "user_id": DEMO_LAB_USER_ID, "kb_id": DEMO_LAB_KB_ID},
    ]

    for scenario in scenarios:
        print(f"\n######## {scenario['label']} ########")
        for mode in [SearchMode.triple, SearchMode.semantic, SearchMode.hybrid]:
            response = service.search(
                SearchRequest(
                    query=scenario["query"],
                    user_id=scenario["user_id"],
                    kb_id=scenario["kb_id"],
                    mode=mode,
                    top_k=3,
                    entity_top_k=4,
                    relation_top_k=4,
                    expansion_degree=1,
                )
            )
            print(f"\n=== {mode.value.upper()} ===")
            print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
