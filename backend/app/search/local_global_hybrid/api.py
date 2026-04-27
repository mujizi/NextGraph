from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from backend.app.search.local_global_hybrid.schemas import SearchMode, SearchRequest, SearchResponse
from backend.app.search.local_global_hybrid.service import SearchService

router = APIRouter(prefix="/api/search", tags=["search"])


def get_search_service(request: Request) -> SearchService:
    return request.app.state.search_service


@router.post("", response_model=SearchResponse)
def search(request: SearchRequest, service: SearchService = Depends(get_search_service)) -> SearchResponse:
    return service.search(request)


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
