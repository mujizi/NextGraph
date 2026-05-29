from __future__ import annotations

from typing import Protocol

from openai import AzureOpenAI

from backend.app.core.config import Settings


class QueryFactRewriter(Protocol):
    def rewrite(
        self,
        query: str,
        *,
        user_id: str,
        kb_id: str,
    ) -> str: ...


class IdentityQueryFactRewriter:
    def rewrite(
        self,
        query: str,
        *,
        user_id: str,
        kb_id: str,
    ) -> str:
        return query.strip()


class AzureLLMQueryFactRewriter:
    SYSTEM_PROMPT = (
        "You rewrite user questions into short factual retrieval statements for graph-RAG relation retrieval. "
        "The rewritten text should help retrieve relation descriptions and evidence passages from a knowledge base. "
        "Keep the meaning faithful to the user question, include the key entities and relationship intent, "
        "and return only the rewritten text with no explanation."
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

    def rewrite(
        self,
        query: str,
        *,
        user_id: str,
        kb_id: str,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.settings.azure_openai_chat_deployment,
            max_completion_tokens=min(self.settings.azure_openai_chat_max_completion_tokens, 512),
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Rewrite the following user question into one retrieval-oriented factual statement.\n"
                        f"user_id: {user_id}\n"
                        f"kb_id: {kb_id}\n"
                        f"question: {query}"
                    ),
                },
            ],
        )
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten or query.strip()
