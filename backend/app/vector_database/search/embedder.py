from __future__ import annotations

import sys
from typing import Protocol

from openai import AzureOpenAI

from backend.app.core.config import Settings


class QueryEmbedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...


class AzureOpenAIEmbedder:
    """Azure OpenAI embedding backend for query vectors."""

    def __init__(self, settings: Settings):
        if not settings.azure_openai_api_key:
            raise ValueError("Azure OpenAI API key is missing. Set API_KEY or AZURE_OPENAI_API_KEY.")

        self._client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
            azure_deployment=settings.azure_openai_embedding_deployment,
        )
        self._deployment = settings.azure_openai_embedding_deployment
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed("dimension probe"))
        return self._dimension

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(input=[text], model=self._deployment)
        return list(response.data[0].embedding)


class VectorGraphRAGEmbedder:
    """Thin wrapper around vector-graph-rag's EmbeddingModel."""

    def __init__(self, settings: Settings):
        src_path = settings.vector_graph_rag_src
        if src_path and src_path not in sys.path:
            sys.path.insert(0, src_path)

        from vector_graph_rag.config import Settings as VGRAGSettings  # type: ignore
        from vector_graph_rag.storage.embeddings import EmbeddingModel  # type: ignore

        rag_settings = VGRAGSettings(
            embedding_model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
            openai_base_url=settings.openai_base_url,
        )
        self._model = EmbeddingModel(settings=rag_settings)
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed("dimension probe"))
        return self._dimension

    def embed(self, text: str) -> list[float]:
        return self._model.embed(text, text_type="query")


def build_query_embedder(settings: Settings) -> QueryEmbedder:
    if settings.embedder_backend in {"auto", "azure_openai"}:
        return AzureOpenAIEmbedder(settings)
    if settings.embedder_backend == "vector_graph_rag":
        return VectorGraphRAGEmbedder(settings)

    raise ValueError(f"Unsupported embedder backend: {settings.embedder_backend}")
