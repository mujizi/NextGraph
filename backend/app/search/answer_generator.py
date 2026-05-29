from __future__ import annotations

from typing import Protocol

from openai import AzureOpenAI

from backend.app.core.config import Settings


class AnswerGenerator(Protocol):
    def generate(
        self,
        *,
        query: str,
        retrieval_query: str,
        context: str,
    ) -> str: ...


class AzureOpenAIAnswerGenerator:
    SYSTEM_PROMPT = (
        "You are a grounded RAG answer assistant. "
        "Answer the user only with support from the provided knowledge-base materials. "
        "Prefer concrete, direct answers first, then briefly cite the supporting evidence. "
        "If the materials are insufficient, clearly say that the knowledge base does not provide enough evidence. "
        "Do not invent facts."
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

    def generate(
        self,
        *,
        query: str,
        retrieval_query: str,
        context: str,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.settings.azure_openai_chat_deployment,
            max_completion_tokens=self.settings.azure_openai_chat_max_completion_tokens,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"User question:\n{query}\n\n"
                        f"Retrieval statement:\n{retrieval_query}\n\n"
                        "Knowledge-base materials:\n"
                        f"{context}\n\n"
                        "Please answer in Chinese when the user question is Chinese."
                    ),
                },
            ],
        )
        return (response.choices[0].message.content or "").strip()
