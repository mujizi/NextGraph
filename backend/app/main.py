from __future__ import annotations

from fastapi import FastAPI

from backend.app.core.config import Settings, get_settings
from backend.app.search.local_global_hybrid.api import router as search_router
from backend.app.vector_database.search.embedder import build_query_embedder
from backend.app.vector_database.search.repository import MilvusGraphRepository
from backend.app.search.local_global_hybrid.service import SearchService


def create_app(
    settings: Settings | None = None,
    search_service: SearchService | None = None,
) -> FastAPI:
    app = FastAPI(
        title="NextGraph Search API",
        version="0.1.0",
        description="Milvus-backed triple / semantic / hybrid retrieval for NextGraph.",
    )

    runtime_settings = settings or get_settings()
    app.state.settings = runtime_settings
    app.state.search_service = search_service or SearchService(
        settings=runtime_settings,
        repository=MilvusGraphRepository(runtime_settings),
        embedder=build_query_embedder(runtime_settings),
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(search_router)
    return app


app = create_app()
