from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import Settings, get_settings
from backend.app.parse_file import router as parse_router
from backend.app.search.local_global_hybrid.api import router as search_router
from backend.app.search.local_global_hybrid.service import SearchService


def create_app(
    settings: Settings | None = None,
    search_service: SearchService | None = None,
) -> FastAPI:
    app = FastAPI(
        title="NextGraph Backend",
        version="0.1.0",
        description="Document parsing and Milvus-backed graph retrieval for NextGraph.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    runtime_settings = settings or get_settings()
    app.state.settings = runtime_settings
    app.state.search_service = search_service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(parse_router)
    app.include_router(search_router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8004)
