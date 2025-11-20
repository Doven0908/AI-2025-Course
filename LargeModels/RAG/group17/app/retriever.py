from __future__ import annotations

import time
from typing import List

from langchain_core.documents import Document

from .logger import get_logger
from .types import DocumentChunk, RetrievalResult
from .vector_store import VectorStoreManager

logger = get_logger(__name__)


class Retriever:
    """封装向量检索流程，并记录耗时信息。"""

    def __init__(self, vector_store: VectorStoreManager, *, top_k: int):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """执行向量检索并封装为 RetrievalResult 列表。"""
        start = time.perf_counter()
        documents_with_scores = self.vector_store.similarity_search_with_score(
            query, self.top_k
        )
        latency = (time.perf_counter() - start) * 1000
        logger.info(
            "Retrieved %s documents in %.2f ms", len(documents_with_scores), latency
        )

        results: List[RetrievalResult] = []
        for doc, score in documents_with_scores:
            metadata = doc.metadata or {}
            # 将得分写回元数据，方便后续分析
            metadata["score"] = float(score)
            chunk = DocumentChunk(
                doc_id=metadata.get("doc_id", ""),
                chunk_id=metadata.get("chunk_id", ""),
                text=doc.page_content,
                start_index=int(metadata.get("start_index", 0)),
                end_index=int(metadata.get("end_index", 0)),
                metadata=metadata,
            )
            results.append(RetrievalResult(chunk=chunk, score=score))
        return results
