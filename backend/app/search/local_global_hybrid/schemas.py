from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    triple = "triple"
    semantic = "semantic"
    hybrid = "hybrid"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="检索问题或查询文本。")
    user_id: str = Field(..., min_length=1, description="用户 ID，用于权限/数据隔离。")
    kb_id: str = Field(..., min_length=1, description="知识库 ID，用于检索范围过滤。")
    mode: SearchMode = Field(default=SearchMode.hybrid, description="检索模式。")
    top_k: int = Field(default=10, ge=1, le=100, description="最终返回结果数量。")
    entity_top_k: int = Field(default=8, ge=1, le=100, description="实体向量召回数量。")
    relation_top_k: int = Field(default=8, ge=1, le=100, description="关系向量召回数量。")
    expansion_degree: int = Field(default=1, ge=0, le=4, description="子图扩展 hop 数。")


class EntityHit(BaseModel):
    id: str
    name: str
    score: float
    relation_ids: list[str] = Field(default_factory=list)


class RelationHit(BaseModel):
    id: str
    subject_id: str
    subject_name: str
    object_id: str
    object_name: str
    relation: str
    passage: str
    docment_id: str | None = None
    score: float
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    source_modes: list[str] = Field(default_factory=list)
    matched_entity_ids: list[str] = Field(default_factory=list)
    passage_ids: list[str] = Field(default_factory=list)


class GroundedPassage(BaseModel):
    id: str
    passage: str
    docment_id: str | None = None
    score: float
    matched_relation_ids: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    mode: SearchMode
    query: str
    user_id: str
    kb_id: str
    results: list[RelationHit] = Field(default_factory=list)
    entity_hits: list[EntityHit] = Field(default_factory=list)
    grounded_passages: list[GroundedPassage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
