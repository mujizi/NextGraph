from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from backend.app.search.local_global_hybrid.schemas import (
    EntityNeighborhoodRequest,
    ExternalQueryRequest,
    ExternalQueryResponse,
    RagAnswerRequest,
    RagAnswerResponse,
    SearchMode,
    SearchRequest,
    SearchResponse,
)
from backend.app.search.answer_generator import AzureOpenAIAnswerGenerator
from backend.app.search.query_fact_rewriter import AzureLLMQueryFactRewriter
from backend.app.search.local_global_hybrid.service import SearchService
from backend.app.vector_database.search.embedder import build_query_embedder
from backend.app.vector_database.search.repository import MilvusGraphRepository

router = APIRouter(prefix="/api/search", tags=["search"])


def get_search_service(request: Request) -> SearchService:
    service = getattr(request.app.state, "search_service", None)
    if service is None:
        settings = request.app.state.settings
        service = SearchService(
            settings=settings,
            repository=MilvusGraphRepository(settings),
            embedder=build_query_embedder(settings),
            query_fact_rewriter=AzureLLMQueryFactRewriter(settings),
            answer_generator=AzureOpenAIAnswerGenerator(settings),
        )
        request.app.state.search_service = service
    return service


@router.post("", response_model=SearchResponse)
def search(request: SearchRequest, service: SearchService = Depends(get_search_service)) -> SearchResponse:
    return service.search(request)


@router.post("/trace")
def search_trace(request: SearchRequest, service: SearchService = Depends(get_search_service)) -> dict:
    return service.search_trace(request)


@router.post("/entity-neighborhood")
def entity_neighborhood(
    request: EntityNeighborhoodRequest,
    service: SearchService = Depends(get_search_service),
) -> dict:
    return service.entity_neighborhood(request)


@router.post("/triple", response_model=SearchResponse)
def search_triple(
    request: SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    return service.search(request.model_copy(update={"mode": SearchMode.triple}))


@router.post("/semantic", response_model=SearchResponse)
def search_semantic(
    request: SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    return service.search(request.model_copy(update={"mode": SearchMode.semantic}))


@router.post("/hybrid", response_model=SearchResponse)
def search_hybrid(
    request: SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    return service.search(request.model_copy(update={"mode": SearchMode.hybrid}))


@router.post("/answer", response_model=RagAnswerResponse)
def search_answer(
    request: RagAnswerRequest,
    service: SearchService = Depends(get_search_service),
) -> RagAnswerResponse:
    return service.answer(request)


@router.post("/external/query", response_model=ExternalQueryResponse)
def external_query(
    request: ExternalQueryRequest,
    service: SearchService = Depends(get_search_service),
) -> ExternalQueryResponse:
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
            metadata=answer_response.metadata,
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
        metadata=search_response.metadata,
    )
