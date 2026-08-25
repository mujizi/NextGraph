from __future__ import annotations

from collections.abc import Iterator
import time
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

    def stream_generate(
        self,
        *,
        query: str,
        retrieval_query: str,
        context: str,
    ) -> Iterator[str]: ...


class AzureOpenAIAnswerGenerator:
    SYSTEM_PROMPT = (
        "你是一名面向电影、剧本创作、文学写作、戏剧、叙事学、角色塑造、视听语言与创意写作研究的专业中文问答助手。"
        "当用户问题明显不属于电影、剧本创作、文学写作、戏剧、叙事、角色塑造、影像制作、创意写作或相关文化研究领域时，"
        "请礼貌说明该问题超出当前专业范围，并不要展开回答。"
        "请基于提供的内部参考材料回答用户问题，但不要向用户提到“知识库”“检索”“材料”“证据”“上下文”等系统实现细节。"
        "回答必须是一次性完整答复，不要请求用户继续补充，不要说“如果你愿意/如果你想/我可以继续”。"
        "优先直接给出结论，再展开定义、适用场景、关键要素和注意事项。"
        "当参考材料不足以形成严格定义时，也要用专业表达给出谨慎、可用的解释，说明“通常可理解为”或“在写作语境中一般指”。"
        "中文问题请使用中文回答，语气专业自然，内容充实，通常不少于 300 个汉字。"
        "不要编造具体出处、书名、页码或不存在的事实。"
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

    def _build_user_message(self, *, query: str, retrieval_query: str, context: str) -> str:
        return (
            f"User question:\n{query}\n\n"
            f"Retrieval statement:\n{retrieval_query}\n\n"
            "Internal reference notes, for your private use only. Do not mention them in the answer:\n"
            f"{context}\n\n"
            "Please answer in Chinese when the user question is Chinese."
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
                {"role": "user", "content": self._build_user_message(query=query, retrieval_query=retrieval_query, context=context)},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def stream_generate(
        self,
        *,
        query: str,
        retrieval_query: str,
        context: str,
    ) -> Iterator[str]:
        started = time.perf_counter()
        print(
            "[answer_generator] stream request "
            f"deployment={self.settings.azure_openai_chat_deployment} "
            f"max_completion_tokens={self.settings.azure_openai_chat_max_completion_tokens} "
            f"context_chars={len(context)}"
        )
        stream = self.client.chat.completions.create(
            model=self.settings.azure_openai_chat_deployment,
            max_completion_tokens=self.settings.azure_openai_chat_max_completion_tokens,
            stream=True,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_message(query=query, retrieval_query=retrieval_query, context=context)},
            ],
        )
        print(f"[answer_generator] stream opened took_ms={round((time.perf_counter() - started) * 1000, 3)}")
        first_delta_logged = False
        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta.content or ""
            if delta:
                if not first_delta_logged:
                    print(
                        "[answer_generator] stream first_delta "
                        f"took_ms={round((time.perf_counter() - started) * 1000, 3)}"
                    )
                    first_delta_logged = True
                yield delta
        print(f"[answer_generator] stream finished took_ms={round((time.perf_counter() - started) * 1000, 3)}")
