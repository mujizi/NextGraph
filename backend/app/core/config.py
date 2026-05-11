from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for NextGraph search services."""

    milvus_uri: str = Field(default="http://10.1.80.16:19530", description="Milvus HTTP endpoint.")
    milvus_db: str = Field(default="crx", description="Milvus database name.")
    milvus_token: Optional[str] = Field(default=None, description="Optional Milvus auth token.")

    entities_collection: str = Field(default="Entities", description="Entity collection name.")
    relations_collection: str = Field(default="Relations", description="Relation collection name.")
    passages_collection: str = Field(default="Passage", description="Passage collection name.")

    embedder_backend: Literal["auto", "azure_openai", "vector_graph_rag"] = Field(
        default="auto", description="How query embeddings should be produced."
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name when using vector_graph_rag backend.",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "NEXTGRAPH_OPENAI_API_KEY", "OPENAI_API_KEY", "VGRAG_OPENAI_API_KEY"
        ),
        description="OpenAI API key used by vector_graph_rag embedding backend.",
    )
    openai_base_url: Optional[str] = Field(default=None, description="Optional custom OpenAI base URL.")
    azure_openai_endpoint: str = Field(
        default="https://admin-1804-resource.cognitiveservices.azure.com/",
        description="Azure OpenAI endpoint for embedding requests.",
    )
    azure_openai_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "NEXTGRAPH_AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "API_KEY"
        ),
        description="Azure OpenAI API key.",
    )
    azure_openai_api_version: str = Field(
        default="2024-12-01-preview", description="Azure OpenAI API version."
    )
    azure_openai_embedding_deployment: str = Field(
        default="text-embedding-3-small",
        description="Azure OpenAI embedding deployment name.",
    )
    azure_openai_chat_deployment: str = Field(
        default="gpt-5.4-mini",
        description="Azure OpenAI chat deployment used for query entity extraction.",
    )
    azure_openai_chat_max_completion_tokens: int = Field(
        default=16384,
        ge=256,
        description="Max completion tokens for Azure OpenAI chat extraction calls.",
    )

    vector_graph_rag_src: str = Field(
        default="/opt/vector-graph-rag/src",
        description="Path to vector-graph-rag source tree.",
    )

    default_top_k: int = Field(default=10, ge=1, le=100)
    default_entity_top_k: int = Field(default=8, ge=1, le=100)
    default_relation_top_k: int = Field(default=8, ge=1, le=100)
    default_expansion_degree: int = Field(default=1, ge=0, le=4)
    relation_number_threshold: int = Field(default=1000, ge=1)
    entity_score_threshold: float = Field(default=0.2, ge=-1.0, le=1.0)
    relation_score_threshold: float = Field(default=0.3, ge=-1.0, le=1.0)
    hybrid_semantic_weight: float = Field(default=0.55, gt=0.0, lt=1.0)
    hybrid_triple_weight: float = Field(default=0.45, gt=0.0, lt=1.0)

    model_config = SettingsConfigDict(env_prefix="NEXTGRAPH_", env_file=".env", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
