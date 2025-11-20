from __future__ import annotations

import time
from typing import Dict, List, Sequence

from openai import OpenAI

from .config import Settings
from .logger import get_logger
from .types import RetrievalResult

logger = get_logger(__name__)


class Generator:
    """负责调用大语言模型生成带引用的回答。"""

    def __init__(self, settings: Settings):
        if not settings.api_key:
            raise RuntimeError(
                "API_KEY is required to call the language model. "
                "Set API_KEY in your environment or .env file."
            )
        # 初始化 OpenAI 兼容客户端
        self.client = OpenAI(api_key=settings.api_key, base_url=settings.llm_base_url)
        self.model = settings.llm_model

    def _build_context_prompt(self, contexts: Sequence[RetrievalResult]) -> str:
        """将检索结果整理成提示词上下文。"""
        parts: List[str] = []
        for idx, result in enumerate(contexts, start=1):
            metadata = result.chunk.metadata
            source_name = metadata.get("source_name") or metadata.get("source") or ""
            parts.append(
                f"[{idx}] Source: {source_name}\n"
                f"Content: {result.chunk.text.strip()}\n"
            )
        return "\n".join(parts)

    def generate_answer(
        self, *, query: str, contexts: Sequence[RetrievalResult]
    ) -> Dict:
        """调用大模型生成答案，并整理引用数据。"""
        context_prompt = self._build_context_prompt(contexts)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert QA assistant. Only answer using the provided "
                    "context. Use numbered citations like [1], [2] matching the "
                    "context section. If information is missing, explicitly say you "
                    "cannot find it in the knowledge base."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Context sections:\n{context_prompt or 'No context available.'}\n\n"
                    "Answer in Chinese. Provide concise bullet points when suitable. "
                    "Always attach the relevant citation markers."
                ),
            },
        ]

        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            # DashScope 要求在非流式模式下显式关闭思维链
            extra_body={"enable_thinking": False},
        )
        latency = (time.perf_counter() - start) * 1000

        message = response.choices[0].message
        logger.info(
            "Generated answer with %s tokens in %.2f ms",
            response.usage and response.usage.total_tokens,
            latency,
        )

        citations: List[Dict] = []
        for idx, result in enumerate(contexts, start=1):
            metadata = result.chunk.metadata
            citations.append(
                {
                    "label": f"[{idx}]",
                    "source": metadata.get("source_name") or metadata.get("source"),
                    "doc_id": result.chunk.doc_id,
                    "chunk_id": result.chunk.chunk_id,
                    "excerpt": result.chunk.text.strip()[:200],
                }
            )

        return {
            "answer": message.content,
            "latency_ms": latency,
            "citations": citations,
            "raw_response": response.model_dump(),
        }
