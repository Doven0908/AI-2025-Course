from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .logger import get_logger
from .ragatouille_compat import ensure_ragatouille_dependencies
from .types import RetrievalResult

logger = get_logger(__name__)


@dataclass
class RerankerConfig:
    """重排器配置，用于在运行时切换实现。"""
    name: str = "none"
    model: str = "colbert-ir/colbertv2.0"


class BaseReranker:
    """重排器抽象基类。"""
    def rerank(
        self, query: str, candidates: Sequence[RetrievalResult], top_k: int
    ) -> List[RetrievalResult]:
        raise NotImplementedError


class NoOpReranker(BaseReranker):
    """默认重排器，直接返回原始结果。"""
    def rerank(
        self, query: str, candidates: Sequence[RetrievalResult], top_k: int
    ) -> List[RetrievalResult]:
        return list(candidates)[:top_k]


class ColBERTReranker(BaseReranker):
    """基于 ColBERTv2 的精排实现。"""

    def __init__(self, model_name: str):
        try:
            ensure_ragatouille_dependencies()
            from ragatouille import RAGPretrainedModel
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "Ragatouille is required for the ColBERT reranker. "
                "请先在当前环境中安装 `ragatouille`。"
            ) from exc

        logger.info("Loading ColBERTv2 reranker model %s", model_name)
        self._model = RAGPretrainedModel.from_pretrained(model_name)

    def rerank(
        self, query: str, candidates: Sequence[RetrievalResult], top_k: int
    ) -> List[RetrievalResult]:
        passages = [item.chunk.text for item in candidates]
        mapping = {idx: item for idx, item in enumerate(candidates)}
        results = self._model.rerank(query=query, documents=passages, k=top_k)

        # RAGatouille returns dicts with `content`, `score`, `rank`, `result_index`
        ranked: List[RetrievalResult] = []
        seen_texts = set()
        for result in results:
            score = float(result.get("score", 0.0))
            index = result.get("result_index")
            text = result.get("content") or result.get("text")
            if index is None:
                if not text:
                    continue
                try:
                    index = passages.index(text)
                except ValueError:
                    continue
            item = mapping.get(index)
            if item is None:
                continue
            if text and text in seen_texts:
                continue
            if text:
                seen_texts.add(text)
            item.chunk.metadata["score"] = score
            ranked.append(item)

        if len(ranked) < top_k:
            remaining = [item for item in candidates if item not in ranked]
            ranked.extend(remaining[: top_k - len(ranked)])

        return ranked[:top_k]


def build_reranker(config: RerankerConfig) -> BaseReranker:
    """根据配置生成对应的重排器实例。"""
    name = (config.name or "none").lower()
    if name in {"none", "disabled", "noop"}:
        return NoOpReranker()
    if name in {"colbert", "colbertv2"}:
        return ColBERTReranker(config.model)
    raise ValueError(f"Unsupported reranker: {config.name}")
