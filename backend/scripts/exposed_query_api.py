#!/usr/bin/env python3
from __future__ import annotations

import sys
import os
import time
import json
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.config import Settings
from backend.app.search.answer_generator import AzureOpenAIAnswerGenerator
from backend.app.search.local_global_hybrid.schemas import (
    ExternalQueryRequest,
    ExternalQueryResponse,
    RagAnswerRequest,
    SearchMode,
    SearchRequest,
)
from backend.app.search.local_global_hybrid.service import SearchService
from backend.app.search.query_fact_rewriter import AzureLLMQueryFactRewriter
from backend.app.vector_database.search.embedder import build_query_embedder
from backend.app.vector_database.search.repository import MilvusGraphRepository

# =========================
# 这里统一改暴露服务参数
# =========================
EXPOSED_HOST = "0.0.0.0"
EXPOSED_PORT = 8710
EXPOSED_PATH = "/api/search/external/query"
EXPOSED_STREAM_PATH = "/api/search/external/query/stream"
ALLOW_ORIGINS = ["*"]

# =========================
# 这里统一改查询默认参数
# 请求里不传时，会回落到这些值
# =========================
DEFAULT_QUERY_PARAMS: dict[str, Any] = {
    "question": "主角和谁有冲突？",
    "user_id": "admin_user",
    "kb_id": "0616",
    "mode": SearchMode.hybrid,
    "top_k": 6,
    "entity_top_k": 5,
    "relation_top_k": 6,
    "expansion_degree": 4,
    "include_answer": True,
    "answer_top_k": 5,
    "passage_top_k": 1,
}

# =========================
# 这里统一改后端检索配置
# 不写的参数继续沿用 backend/app/core/config.py 默认值
# =========================
SETTINGS_OVERRIDES: dict[str, Any] = {
    "backend_host": EXPOSED_HOST,
    "backend_port": EXPOSED_PORT,
    "vector_db_database": "bookk255",
    "vector_store_provider": "tencent",
    "tencent_vectordb_database": "bookk255",
    # "vector_store_provider": "milvus",
    # "vector_db_database": "crx",
    # "milvus_uri": "http://127.0.0.1:19530",
    # "milvus_db": "crx",
    # "tencent_vectordb_url": "http://127.0.0.1:80",
    # "azure_openai_api_key": "your-key",
    # "azure_openai_endpoint": "https://your-resource.openai.azure.com/",
    # "azure_openai_chat_deployment": "gpt-5.4-mini",
}


class ExternalQueryPayload(BaseModel):
    question: str | None = Field(default=None, description="外部调用方传入的问题文本。")
    user_id: str | None = Field(default=None, description="用户 ID。")
    kb_id: str | None = Field(default=None, description="知识库 ID。")
    mode: SearchMode | None = Field(default=None, description="检索模式。")
    top_k: int | None = Field(default=None, ge=1, le=100, description="最终返回结果数量。")
    entity_top_k: int | None = Field(default=None, ge=1, le=100, description="实体召回数量。")
    relation_top_k: int | None = Field(default=None, ge=1, le=100, description="关系召回数量。")
    expansion_degree: int | None = Field(default=None, ge=0, le=4, description="子图扩展 hop 数。")
    include_answer: bool | None = Field(default=None, description="是否同时返回模型答案。")
    answer_top_k: int | None = Field(default=None, ge=1, le=20, description="生成答案时使用的关系数量。")
    passage_top_k: int | None = Field(default=None, ge=1, le=20, description="生成答案时使用的证据段落数量。")


def build_settings() -> Settings:
    # Force this standalone script to use its local database/provider config
    # instead of inheriting conflicting values from the repo-level .env file.
    os.environ["NEXTGRAPH_VECTOR_STORE_PROVIDER"] = str(SETTINGS_OVERRIDES["vector_store_provider"])
    os.environ["NEXTGRAPH_VECTOR_DB_DATABASE"] = str(SETTINGS_OVERRIDES["vector_db_database"])
    os.environ["NEXTGRAPH_TENCENT_VECTORDB_DATABASE"] = str(SETTINGS_OVERRIDES["tencent_vectordb_database"])
    return Settings(**SETTINGS_OVERRIDES)


def build_service(settings: Settings) -> SearchService:
    return SearchService(
        settings=settings,
        repository=MilvusGraphRepository(settings),
        embedder=build_query_embedder(settings),
        query_fact_rewriter=AzureLLMQueryFactRewriter(settings),
        answer_generator=AzureOpenAIAnswerGenerator(settings),
    )


settings = build_settings()
service = build_service(settings)

app = FastAPI(
    title="NextGraph Exposed Query API",
    version="0.1.0",
    description="Standalone external query API for NextGraph.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "path": EXPOSED_PATH,
        "runtime": {
            "vector_store_provider": settings.vector_store_provider,
            "vector_db_database": settings.vector_db_database,
            "tencent_vectordb_url": settings.tencent_vectordb_url,
            "tencent_vectordb_database": settings.tencent_vectordb_database,
            "milvus_uri": settings.milvus_uri,
            "milvus_db": settings.milvus_db,
        },
        "defaults": {
            **DEFAULT_QUERY_PARAMS,
            "mode": str(DEFAULT_QUERY_PARAMS["mode"]),
        },
    }


def build_request(payload: ExternalQueryPayload) -> ExternalQueryRequest:
    merged = {
        **DEFAULT_QUERY_PARAMS,
        **payload.model_dump(exclude_none=True),
    }
    return ExternalQueryRequest.model_validate(merged)


def dump_event(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


@app.post(EXPOSED_PATH, response_model=ExternalQueryResponse)
def external_query(payload: ExternalQueryPayload) -> ExternalQueryResponse:
    started = time.perf_counter()
    request = build_request(payload)
    print(
        f"[exposed_query_api] start question={request.question!r} kb_id={request.kb_id} "
        f"mode={request.mode} top_k={request.top_k} entity_top_k={request.entity_top_k} "
        f"relation_top_k={request.relation_top_k} expansion_degree={request.expansion_degree} "
        f"include_answer={request.include_answer}"
    )

    if request.include_answer:
        answer_response = service.answer(
            RagAnswerRequest(
                query=request.question,
                user_id=request.user_id,
                kb_id=request.kb_id,
                mode=request.mode,
                top_k=request.top_k,
                entity_top_k=request.entity_top_k,
                relation_top_k=request.relation_top_k,
                expansion_degree=request.expansion_degree,
                answer_top_k=request.answer_top_k,
                passage_top_k=request.passage_top_k,
            )
        )
        total_ms = round((time.perf_counter() - started) * 1000, 3)
        print(
            f"[exposed_query_api] done answer question={request.question!r} kb_id={request.kb_id} "
            f"results={len(answer_response.results)} passages={len(answer_response.grounded_passages)} "
            f"search_took_ms={answer_response.metadata.get('took_ms')} external_query_took_ms={total_ms}"
        )
        return ExternalQueryResponse(
            question=request.question,
            user_id=request.user_id,
            kb_id=request.kb_id,
            mode=answer_response.mode,
            answer=answer_response.answer,
            retrieval_query=answer_response.retrieval_query,
            results=answer_response.results,
            grounded_passages=answer_response.grounded_passages,
            entity_hits=[],
            metadata={
                **answer_response.metadata,
                "external_query_took_ms": total_ms,
            },
        )

    search_response = service.search(
        SearchRequest(
            query=request.question,
            user_id=request.user_id,
            kb_id=request.kb_id,
            mode=request.mode,
            top_k=request.top_k,
            entity_top_k=request.entity_top_k,
            relation_top_k=request.relation_top_k,
            expansion_degree=request.expansion_degree,
        )
    )
    total_ms = round((time.perf_counter() - started) * 1000, 3)
    print(
        f"[exposed_query_api] done search question={request.question!r} kb_id={request.kb_id} "
        f"results={len(search_response.results)} entity_hits={len(search_response.entity_hits)} "
        f"passages={len(search_response.grounded_passages)} search_took_ms={search_response.metadata.get('took_ms')} "
        f"external_query_took_ms={total_ms}"
    )
    return ExternalQueryResponse(
        question=request.question,
        user_id=request.user_id,
        kb_id=request.kb_id,
        mode=search_response.mode,
        answer=None,
        retrieval_query=str(search_response.metadata.get("retrieval_query") or request.question),
        results=search_response.results,
        grounded_passages=search_response.grounded_passages,
        entity_hits=search_response.entity_hits,
        metadata={
            **search_response.metadata,
            "external_query_took_ms": total_ms,
        },
    )


@app.post(EXPOSED_STREAM_PATH)
def external_query_stream(payload: ExternalQueryPayload) -> StreamingResponse:
    request = build_request(payload)

    def event_stream():
        started = time.perf_counter()
        print(
            f"[exposed_query_api] stream start question={request.question!r} kb_id={request.kb_id} "
            f"mode={request.mode} top_k={request.top_k} include_answer={request.include_answer}"
        )
        try:
            if request.include_answer:
                prepared, answer_stream = service.stream_answer(
                    RagAnswerRequest(
                        query=request.question,
                        user_id=request.user_id,
                        kb_id=request.kb_id,
                        mode=request.mode,
                        top_k=request.top_k,
                        entity_top_k=request.entity_top_k,
                        relation_top_k=request.relation_top_k,
                        expansion_degree=request.expansion_degree,
                        answer_top_k=request.answer_top_k,
                        passage_top_k=request.passage_top_k,
                    )
                )
                retrieval_response = prepared["retrieval_response"]
                meta = {
                    "type": "meta",
                    "question": request.question,
                    "user_id": request.user_id,
                    "kb_id": request.kb_id,
                    "mode": str(retrieval_response.mode),
                    "retrieval_query": prepared["retrieval_query"],
                    "results": [item.model_dump() for item in retrieval_response.results],
                    "grounded_passages": [item.model_dump() for item in retrieval_response.grounded_passages],
                    "entity_hits": [],
                    "metadata": {
                        **retrieval_response.metadata,
                        "answer_top_k": request.answer_top_k,
                        "passage_top_k": request.passage_top_k,
                        "answer_context_score_threshold": service.settings.answer_context_score_threshold,
                        "answer_context_filtered_result_count": prepared["filtered_result_count"],
                        "answer_context_filtered_passage_count": prepared["filtered_passage_count"],
                        "context_preview": prepared["context"][:2000],
                    },
                }
                yield dump_event(meta)

                answer_parts: list[str] = []
                for delta in answer_stream:
                    answer_parts.append(delta)
                    yield dump_event({"type": "answer_delta", "delta": delta})

                answer_text = "".join(answer_parts).strip()
                total_ms = round((time.perf_counter() - started) * 1000, 3)
                yield dump_event(
                    {
                        "type": "done",
                        "answer": answer_text,
                        "metadata": {
                            **meta["metadata"],
                            "external_query_took_ms": total_ms,
                        },
                    }
                )
                print(
                    f"[exposed_query_api] stream done answer question={request.question!r} kb_id={request.kb_id} "
                    f"answer_context_results={len(prepared['selected_results'])} "
                    f"answer_context_passages={len(prepared['selected_passages'])} "
                    f"external_query_took_ms={total_ms}"
                )
                return

            search_response = service.search(
                SearchRequest(
                    query=request.question,
                    user_id=request.user_id,
                    kb_id=request.kb_id,
                    mode=request.mode,
                    top_k=request.top_k,
                    entity_top_k=request.entity_top_k,
                    relation_top_k=request.relation_top_k,
                    expansion_degree=request.expansion_degree,
                )
            )
            total_ms = round((time.perf_counter() - started) * 1000, 3)
            yield dump_event(
                {
                    "type": "meta",
                    "question": request.question,
                    "user_id": request.user_id,
                    "kb_id": request.kb_id,
                    "mode": str(search_response.mode),
                    "retrieval_query": str(search_response.metadata.get("retrieval_query") or request.question),
                    "results": [item.model_dump() for item in search_response.results],
                    "grounded_passages": [item.model_dump() for item in search_response.grounded_passages],
                    "entity_hits": [item.model_dump() for item in search_response.entity_hits],
                    "metadata": {
                        **search_response.metadata,
                        "external_query_took_ms": total_ms,
                    },
                }
            )
            yield dump_event({"type": "done", "answer": "", "metadata": {"external_query_took_ms": total_ms}})
        except Exception as exc:
            print(f"[exposed_query_api] stream failed question={request.question!r} kb_id={request.kb_id} error={exc}")
            yield dump_event({"type": "error", "message": str(exc)})

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


if __name__ == "__main__":
    print("Exposed Query API runtime config:")
    print(f"  host={EXPOSED_HOST}")
    print(f"  port={EXPOSED_PORT}")
    print(f"  path={EXPOSED_PATH}")
    print(f"  vector_store_provider={settings.vector_store_provider}")
    print(f"  vector_db_database={settings.vector_db_database}")
    print(f"  tencent_vectordb_url={settings.tencent_vectordb_url}")
    print(f"  tencent_vectordb_database={settings.tencent_vectordb_database}")
    print(f"  milvus_uri={settings.milvus_uri}")
    print(f"  milvus_db={settings.milvus_db}")
    uvicorn.run(app, host=EXPOSED_HOST, port=EXPOSED_PORT)
