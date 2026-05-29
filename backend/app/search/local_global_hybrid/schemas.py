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
    expansion_degree: int = Field(default=2, ge=0, le=4, description="子图扩展 hop 数。")


class EntityNeighborhoodRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="用户 ID，用于权限/数据隔离。")
    kb_id: str = Field(..., min_length=1, description="知识库 ID，用于检索范围过滤。")
    seed_entity_ids: list[str] = Field(default_factory=list, description="搜索阶段召回到的种子实体 ID。")
    center_entity_id: str | None = Field(default=None, description="实体阶段使用的中心实体 ID。")
    depth: int = Field(default=3, ge=1, le=4, description="实体阶段的局部扩展轮数。")
    relation_limit: int = Field(default=10, ge=1, le=30, description="最多抽取的关系数量。")
    shuffle_seed: int = Field(default=0, ge=0, description="用于随机换组的稳定扰动种子。")


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
    describe: str = ""
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


class RagAnswerRequest(SearchRequest):
    answer_top_k: int = Field(default=6, ge=1, le=20, description="用于生成答案的关系数量。")
    passage_top_k: int = Field(default=6, ge=1, le=20, description="用于生成答案的证据段落数量。")


class RagAnswerResponse(BaseModel):
    mode: SearchMode
    query: str
    retrieval_query: str
    user_id: str
    kb_id: str
    answer: str
    results: list[RelationHit] = Field(default_factory=list)
    grounded_passages: list[GroundedPassage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
