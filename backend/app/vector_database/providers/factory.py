from __future__ import annotations

from functools import lru_cache

from backend.app.core.config import Settings, get_settings
from backend.app.vector_database.providers.base import VectorStoreProvider


def create_vector_store(settings: Settings) -> VectorStoreProvider:
    if settings.vector_store_provider == "tencent":
        from backend.app.vector_database.providers.tencent_provider import TencentVectorStoreProvider

        return TencentVectorStoreProvider(settings)
    from backend.app.vector_database.providers.milvus_provider import MilvusVectorStoreProvider

    return MilvusVectorStoreProvider(settings)


@lru_cache(maxsize=1)
def _get_default_vector_store() -> VectorStoreProvider:
    return create_vector_store(get_settings())


def get_vector_store(settings: Settings | None = None) -> VectorStoreProvider:
    if settings is None:
        return _get_default_vector_store()
    return create_vector_store(settings)
