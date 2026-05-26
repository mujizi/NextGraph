from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from backend.app.search.local_global_hybrid.schemas import (
    EntityNeighborhoodRequest,
    SearchMode,
    SearchRequest,
    SearchResponse,
)
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
