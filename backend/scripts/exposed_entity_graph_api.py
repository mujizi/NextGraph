#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.config import Settings
from backend.app.search.local_global_hybrid.schemas import EntityNeighborhoodRequest
from backend.app.search.local_global_hybrid.service import SearchService
from backend.app.search.query_entity_extractor import AzureLLMQueryEntityExtractor
from backend.app.vector_database.search.embedder import build_query_embedder
from backend.app.vector_database.search.repository import MilvusGraphRepository

# =========================
# 这里统一改暴露服务参数
# host / port 默认读取 .env 中的 NEXTGRAPH_BACKEND_HOST / NEXTGRAPH_BACKEND_PORT，
# 没有配置时再回退到下面的默认值。
# =========================
DEFAULT_EXPOSED_HOST = "0.0.0.0"
DEFAULT_EXPOSED_PORT = 8711
EXPOSED_PATH = "/api/entity-graph/extract-and-expand"
ALLOW_ORIGINS = ["*"]

# =========================
# 这里统一改默认请求参数
# =========================
DEFAULT_GRAPH_PARAMS: dict[str, Any] = {
    "question": "主角和谁有冲突？",
    "user_id": "admin_user",
    "kb_id": "0616",
    "max_entities": 3,
    "name_match_limit": 3,
    "fallback_vector_limit": 3,
    "default_depth": 4,
    "default_relation_limit": 30,
    "shuffle_seed": 0,
}

class EntityExpansionOverride(BaseModel):
    entity_text: str = Field(..., min_length=1, description="针对某个抽取实体文本的覆盖配置。")
    depth: int | None = Field(default=None, ge=1, le=6, description="该实体单独的跳数。")
    relation_limit: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="该实体每一跳最多扩展多少条关系。",
    )
    center_entity_id: str | None = Field(
        default=None,
        description="可选，强制指定该实体使用哪个数据库实体 ID 作为中心点。",
    )


class EntityGraphRequest(BaseModel):
    question: str = Field(..., min_length=1, description="原始问题。")
    user_id: str = Field(..., min_length=1, description="用户 ID。")
    kb_id: str = Field(..., min_length=1, description="知识库 ID。")
    max_entities: int = Field(default=3, ge=1, le=10, description="最多抽取多少个实体词。")
    name_match_limit: int = Field(default=3, ge=1, le=10, description="每个抽取实体按名称精确匹配返回多少个数据库实体。")
    fallback_vector_limit: int = Field(default=3, ge=1, le=10, description="名称匹配不到时，向量兜底返回多少个实体。")
    default_depth: int = Field(default=4, ge=1, le=6, description="默认每个实体扩展多少跳。")
    default_relation_limit: int = Field(default=30, ge=1, le=50, description="默认每跳每个实体扩展多少条关系。")
    shuffle_seed: int = Field(default=0, ge=0, description="图扩展采样扰动种子。")
    entity_overrides: list[EntityExpansionOverride] = Field(
        default_factory=list,
        description="针对某些抽取实体的单独扩展参数覆盖。",
    )


class MatchedEntity(BaseModel):
    id: str
    name: str
    relation_ids: list[str] = Field(default_factory=list)
    match_mode: str
    source_entity_text: str


class ExtractedEntityGraph(BaseModel):
    entity_text: str
    depth: int
    relation_limit: int
    center_entity_id: str | None = None
    matched_entities: list[MatchedEntity] = Field(default_factory=list)
    graph: dict[str, Any] = Field(default_factory=lambda: {"nodes": [], "links": []})
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntityGraphResponse(BaseModel):
    protocol: str = "nextgraph.entity-graph.v1"
    question: str
    user_id: str
    kb_id: str
    extracted_entities: list[str] = Field(default_factory=list)
    entity_graphs: list[ExtractedEntityGraph] = Field(default_factory=list)
    merged_graph: dict[str, Any] = Field(default_factory=lambda: {"nodes": [], "links": []})
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_settings() -> Settings:
    return Settings(
        backend_host=os.environ.get("NEXTGRAPH_BACKEND_HOST", DEFAULT_EXPOSED_HOST),
        backend_port=int(os.environ.get("NEXTGRAPH_BACKEND_PORT", str(DEFAULT_EXPOSED_PORT))),
    )


def build_service(settings: Settings) -> SearchService:
    return SearchService(
        settings=settings,
        repository=MilvusGraphRepository(settings),
        embedder=build_query_embedder(settings),
        query_entity_extractor=AzureLLMQueryEntityExtractor(settings),
    )


settings = build_settings()
service = build_service(settings)
EXPOSED_HOST = settings.backend_host
EXPOSED_PORT = settings.backend_port

app = FastAPI(
    title="NextGraph Entity Graph API",
    version="0.1.0",
    description="Extract entities from a question, resolve them to DB entities, and return expanded graph structures.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _override_map(items: list[EntityExpansionOverride]) -> dict[str, EntityExpansionOverride]:
    return {item.entity_text.strip(): item for item in items if item.entity_text.strip()}


def _merge_graphs(graphs: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    links: dict[str, dict[str, Any]] = {}
    for graph in graphs:
        for node in graph.get("nodes", []):
            node_id = str(node.get("id") or "").strip()
            if not node_id:
                continue
            existing = nodes.get(node_id, {})
            merged = {**existing, **node}
            merged["is_seed"] = bool(existing.get("is_seed")) or bool(node.get("is_seed"))
            nodes[node_id] = merged
        for link in graph.get("links", []):
            link_id = str(link.get("id") or "").strip()
            if not link_id:
                continue
            existing = links.get(link_id, {})
            merged = {**existing, **link}
            existing_ids = set(existing.get("matched_entity_ids") or [])
            incoming_ids = set(link.get("matched_entity_ids") or [])
            merged["matched_entity_ids"] = sorted(existing_ids | incoming_ids)
            links[link_id] = merged
    return {"nodes": list(nodes.values()), "links": list(links.values())}


def _extract_entities(payload: EntityGraphRequest) -> list[str]:
    entities = service.query_entity_extractor.extract(
        payload.question,
        user_id=payload.user_id,
        kb_id=payload.kb_id,
        semantic_limit=payload.max_entities,
    )
    return [item for item in entities if item][: payload.max_entities]


def _match_entities_by_name_or_vector(
    *,
    entity_text: str,
    user_id: str,
    kb_id: str,
    name_match_limit: int,
    fallback_vector_limit: int,
) -> tuple[list[MatchedEntity], str]:
    match_started = time.perf_counter()
    name_match_started = time.perf_counter()
    name_rows = service.repository.provider.get_entities_by_names(
        [entity_text],
        user_id=user_id,
        kb_id=kb_id,
        limit_per_name=name_match_limit,
    )
    name_match_took_ms = round((time.perf_counter() - name_match_started) * 1000, 3)
    print(
        f"[exposed_entity_graph_api] name_match entity_text={entity_text!r} "
        f"hits={len(name_rows)} limit={name_match_limit} took_ms={name_match_took_ms}"
    )
    if name_rows:
        total_took_ms = round((time.perf_counter() - match_started) * 1000, 3)
        print(
            f"[exposed_entity_graph_api] match_done entity_text={entity_text!r} "
            f"strategy=name_exact total_took_ms={total_took_ms}"
        )
        return (
            [
                MatchedEntity(
                    id=str(row.get("id") or ""),
                    name=str(row.get("name") or row.get("id") or ""),
                    relation_ids=list(row.get("relation_ids") or []),
                    match_mode="name_exact",
                    source_entity_text=entity_text,
                )
                for row in name_rows
            ],
            "name_exact",
        )

    embed_started = time.perf_counter()
    query_vector = service.embedder.embed(entity_text)
    embed_took_ms = round((time.perf_counter() - embed_started) * 1000, 3)
    print(
        f"[exposed_entity_graph_api] vector_embed entity_text={entity_text!r} "
        f"took_ms={embed_took_ms}"
    )

    vector_search_started = time.perf_counter()
    vector_rows = service.repository.search_entities(
        query_vector,
        user_id=user_id,
        kb_id=kb_id,
        limit=fallback_vector_limit,
    )
    vector_search_took_ms = round((time.perf_counter() - vector_search_started) * 1000, 3)
    total_took_ms = round((time.perf_counter() - match_started) * 1000, 3)
    print(
        f"[exposed_entity_graph_api] vector_search entity_text={entity_text!r} "
        f"hits={len(vector_rows)} limit={fallback_vector_limit} took_ms={vector_search_took_ms}"
    )
    print(
        f"[exposed_entity_graph_api] match_done entity_text={entity_text!r} "
        f"strategy=vector_fallback total_took_ms={total_took_ms}"
    )
    return (
        [
            MatchedEntity(
                id=str(row.get("id") or ""),
                name=str(row.get("name") or row.get("id") or ""),
                relation_ids=list(row.get("relation_ids") or []),
                match_mode="vector_fallback",
                source_entity_text=entity_text,
            )
            for row in vector_rows
        ],
        "vector_fallback",
    )


def _build_entity_graph(
    *,
    entity_text: str,
    matches: list[MatchedEntity],
    match_strategy: str,
    payload: EntityGraphRequest,
    override: EntityExpansionOverride | None,
) -> ExtractedEntityGraph:
    build_started = time.perf_counter()
    depth = override.depth if override and override.depth is not None else payload.default_depth
    relation_limit = (
        override.relation_limit
        if override and override.relation_limit is not None
        else payload.default_relation_limit
    )
    center_entity_id = override.center_entity_id if override and override.center_entity_id else None
    seed_entity_ids = [item.id for item in matches if item.id]
    graph = {"nodes": [], "links": []}
    expansion_took_ms = 0.0

    if seed_entity_ids:
        expansion_started = time.perf_counter()
        expansion = service.entity_neighborhood(
            EntityNeighborhoodRequest(
                user_id=payload.user_id,
                kb_id=payload.kb_id,
                seed_entity_ids=seed_entity_ids,
                center_entity_id=center_entity_id,
                depth=depth,
                relation_limit=relation_limit,
                shuffle_seed=payload.shuffle_seed,
            )
        )
        expansion_took_ms = round((time.perf_counter() - expansion_started) * 1000, 3)
        graph = expansion.get("graph") or graph
        center_entity_id = str(expansion.get("center_entity_id") or center_entity_id or seed_entity_ids[0])

    return ExtractedEntityGraph(
        entity_text=entity_text,
        depth=depth,
        relation_limit=relation_limit,
        center_entity_id=center_entity_id,
        matched_entities=matches,
        graph=graph,
        metadata={
            "match_strategy": match_strategy,
            "matched_entity_count": len(matches),
            "seed_entity_ids": seed_entity_ids,
            "node_count": len(graph.get("nodes", [])),
            "link_count": len(graph.get("links", [])),
            "timing": {
                "expansion_took_ms": expansion_took_ms,
                "entity_total_took_ms": round((time.perf_counter() - build_started) * 1000, 3),
            },
        },
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "path": EXPOSED_PATH,
        "runtime": {
            "vector_store_provider": settings.vector_store_provider,
            "vector_db_database": settings.vector_db_database,
            "milvus_uri": settings.milvus_uri,
            "milvus_db": settings.milvus_db,
            "azure_openai_endpoint": settings.azure_openai_endpoint,
            "azure_openai_chat_deployment": settings.azure_openai_chat_deployment,
        },
        "defaults": DEFAULT_GRAPH_PARAMS,
        "protocol": "nextgraph.entity-graph.v1",
    }


@app.post(EXPOSED_PATH, response_model=EntityGraphResponse)
def extract_and_expand(payload: EntityGraphRequest) -> EntityGraphResponse:
    started = time.perf_counter()
    print(
        f"[exposed_entity_graph_api] start question={payload.question!r} kb_id={payload.kb_id} "
        f"max_entities={payload.max_entities} default_depth={payload.default_depth} "
        f"default_relation_limit={payload.default_relation_limit} shuffle_seed={payload.shuffle_seed}"
    )
    extract_started = time.perf_counter()
    extracted_entities = _extract_entities(payload)
    extract_took_ms = round((time.perf_counter() - extract_started) * 1000, 3)
    overrides = _override_map(payload.entity_overrides)
    entity_graphs: list[ExtractedEntityGraph] = []

    for entity_text in extracted_entities:
        match_started = time.perf_counter()
        matches, match_strategy = _match_entities_by_name_or_vector(
            entity_text=entity_text,
            user_id=payload.user_id,
            kb_id=payload.kb_id,
            name_match_limit=payload.name_match_limit,
            fallback_vector_limit=payload.fallback_vector_limit,
        )
        entity_graph = _build_entity_graph(
            entity_text=entity_text,
            matches=matches,
            match_strategy=match_strategy,
            payload=payload,
            override=overrides.get(entity_text),
        )
        entity_graph.metadata["timing"] = {
            **entity_graph.metadata.get("timing", {}),
            "match_took_ms": round((time.perf_counter() - match_started) * 1000, 3),
        }
        print(
            f"[exposed_entity_graph_api] entity entity_text={entity_text!r} "
            f"matched={len(matches)} center_entity_id={entity_graph.center_entity_id} "
            f"nodes={len(entity_graph.graph.get('nodes', []))} links={len(entity_graph.graph.get('links', []))} "
            f"match_took_ms={entity_graph.metadata['timing'].get('match_took_ms')} "
            f"expansion_took_ms={entity_graph.metadata['timing'].get('expansion_took_ms')} "
            f"entity_total_took_ms={entity_graph.metadata['timing'].get('entity_total_took_ms')}"
        )
        entity_graphs.append(entity_graph)

    merge_started = time.perf_counter()
    merged_graph = _merge_graphs([item.graph for item in entity_graphs])
    merge_took_ms = round((time.perf_counter() - merge_started) * 1000, 3)
    total_ms = round((time.perf_counter() - started) * 1000, 3)
    print(
        f"[exposed_entity_graph_api] done question={payload.question!r} kb_id={payload.kb_id} "
        f"extracted_entities={len(extracted_entities)} entity_graphs={len(entity_graphs)} "
        f"merged_nodes={len(merged_graph.get('nodes', []))} merged_links={len(merged_graph.get('links', []))} "
        f"extract_took_ms={extract_took_ms} merge_took_ms={merge_took_ms} total_took_ms={total_ms}"
    )
    return EntityGraphResponse(
        question=payload.question,
        user_id=payload.user_id,
        kb_id=payload.kb_id,
        extracted_entities=extracted_entities,
        entity_graphs=entity_graphs,
        merged_graph=merged_graph,
        metadata={
            "max_entities": payload.max_entities,
            "name_match_limit": payload.name_match_limit,
            "fallback_vector_limit": payload.fallback_vector_limit,
            "default_depth": payload.default_depth,
            "default_relation_limit": payload.default_relation_limit,
            "returned_entity_graph_count": len(entity_graphs),
            "merged_node_count": len(merged_graph.get("nodes", [])),
            "merged_link_count": len(merged_graph.get("links", [])),
            "timing": {
                "extract_took_ms": extract_took_ms,
                "merge_took_ms": merge_took_ms,
                "total_took_ms": total_ms,
            },
        },
    )


if __name__ == "__main__":
    print("Entity Graph API runtime config:")
    print(f"  host={EXPOSED_HOST}")
    print(f"  port={EXPOSED_PORT}")
    print(f"  path={EXPOSED_PATH}")
    print(f"  provider={settings.vector_store_provider}")
    print(f"  database={settings.vector_db_database}")
    print("启动命令建议:")
    print("  conda run -n nextgraph python backend/scripts/exposed_entity_graph_api.py")
    uvicorn.run(app, host=EXPOSED_HOST, port=EXPOSED_PORT)
