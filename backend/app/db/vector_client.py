from __future__ import annotations

from backend.app.core.config import get_settings
from backend.app.vector_database.providers.factory import get_vector_store


class LegacyVectorClientAdapter:
    def __init__(self):
        self._provider = get_vector_store(get_settings())

    def use_database(self, db_name: str) -> None:
        self._provider.use_database(db_name)

    async def batch_insert_graph_data(self, payload):
        await self._provider.batch_insert_graph_data(payload)

    def get_entities_by_ids(self, kb_id: str, entity_ids: list[str]):
        return self._provider.get_entity_map_by_ids(kb_id, entity_ids)

    @property
    def provider(self):
        return self._provider


vector_store = get_vector_store(get_settings())
milvus_client = LegacyVectorClientAdapter()
