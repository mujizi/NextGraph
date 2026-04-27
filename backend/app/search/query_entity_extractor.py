from __future__ import annotations

import json
from typing import Protocol

from openai import AzureOpenAI

from backend.app.core.config import Settings


class QueryEntityExtractor(Protocol):
    def extract(
        self,
        query: str,
        *,
        user_id: str,
        kb_id: str,
        semantic_limit: int,
    ) -> list[str]: ...


class AzureLLMQueryEntityExtractor:
    """Extract query entities using Azure OpenAI chat completions."""

    SYSTEM_PROMPT = (
        "You extract named entities for knowledge-graph retrieval. "
        "Return strict JSON with key 'named_entities' whose value is an array of entity strings. "
        "Only include entities that are useful retrieval anchors. Do not explain."
    )

    def __init__(self, settings: Settings):
        if not settings.azure_openai_api_key:
            raise ValueError("Azure OpenAI API key is missing. Set API_KEY or AZURE_OPENAI_API_KEY.")

        self.settings = settings
        self.client = AzureOpenAI(
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
        )

    def extract(
        self,
        query: str,
        *,
        user_id: str,
        kb_id: str,
        semantic_limit: int,
    ) -> list[str]:
        response = self.client.chat.completions.create(
            model=self.settings.azure_openai_chat_deployment,
            max_completion_tokens=self.settings.azure_openai_chat_max_completion_tokens,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Extract the retrieval entities from this query. "
                        "Return JSON only.\n"
                        f"query: {query}\n"
                        f"max_entities: {semantic_limit}"
                    ),
                },
            ],
        )
        content = (response.choices[0].message.content or "{}").strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return []

        entities = data.get("named_entities", [])
        if not isinstance(entities, list):
            return []

        cleaned: list[str] = []
        for entity in entities:
            text = str(entity).strip()
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned[:semantic_limit]
