from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.search.local_global_hybrid.schemas import RagAnswerRequest, RagAnswerResponse, SearchMode, SearchRequest
from backend.app.search.local_global_hybrid.service import SearchService


class StubService(SearchService):
    def __init__(self):
        pass

    def search(self, request: SearchRequest):
        from backend.app.search.local_global_hybrid.schemas import SearchResponse

        return SearchResponse(
            mode=request.mode,
            query=request.query,
            user_id=request.user_id,
            kb_id=request.kb_id,
            results=[],
            entity_hits=[],
            grounded_passages=[],
            metadata={"stub": True},
        )

    def answer(self, request: RagAnswerRequest):
        return RagAnswerResponse(
            mode=request.mode,
            query=request.query,
            retrieval_query="stub retrieval",
            user_id=request.user_id,
            kb_id=request.kb_id,
            answer="stub answer",
            results=[],
            grounded_passages=[],
            metadata={"stub": True},
        )


def test_search_endpoint_uses_injected_service():
    app = create_app(settings=Settings(embedder_backend="auto"), search_service=StubService())
    client = TestClient(app)

    response = client.post(
        "/api/search/hybrid",
        json={"query": "test", "user_id": "u1", "kb_id": "kb1", "mode": SearchMode.triple.value},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == SearchMode.hybrid.value
    assert data["metadata"]["stub"] is True
    assert data["grounded_passages"] == []


def test_answer_endpoint_uses_injected_service():
    app = create_app(settings=Settings(embedder_backend="auto"), search_service=StubService())
    client = TestClient(app)

    response = client.post(
        "/api/search/answer",
        json={"query": "test", "user_id": "u1", "kb_id": "kb1", "mode": SearchMode.hybrid.value},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["retrieval_query"] == "stub retrieval"
    assert data["answer"] == "stub answer"
    assert data["metadata"]["stub"] is True
