from __future__ import annotations

import time
from typing import Dict, Optional

from .chunking import chunk_corpus
from .config import Settings, load_settings
from .data_ingestion import export_documents, load_documents
from .embedding_service import EmbeddingService
from .generator import Generator
from .logger import get_logger
from .metrics import MetricsCollector
from .reranker import RerankerConfig, build_reranker
from .retriever import Retriever
from .types import Answer
from .vector_store import VectorStoreManager

logger = get_logger(__name__)


class RAGPipeline:
    """统筹“预处理-检索-重排-生成”的端到端流程。"""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or load_settings()
        # 初始化向量服务、索引管理器与度量系统
        self.embedding_service = EmbeddingService(self.settings)
        self.vector_store = VectorStoreManager(
            index_path=self.settings.index_path,
            metadata_path=self.settings.store_doc_path,
            embedding_service=self.embedding_service,
        )
        self.metrics = MetricsCollector(self.settings.metrics_db_path)
        self._generator: Optional[Generator] = None
        self.reranker = build_reranker(
            RerankerConfig(
                name=self.settings.reranker_name,
                model=self.settings.reranker_model,
            )
        )

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            self._generator = Generator(self.settings)
        return self._generator

    def ingest(self) -> None:
        """读取文档并构建 FAISS 索引。"""

        documents = load_documents(self.settings.document_dir)
        if not documents:
            raise RuntimeError(
                f"No documents found in {self.settings.document_dir}. "
                "Populate the directory before running ingestion."
            )

        export_documents(documents, self.settings.storage_dir / "document_index.jsonl")

        chunks = chunk_corpus(
            documents,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )

        if not chunks:
            raise RuntimeError("No chunks produced. Check chunking parameters.")

        logger.info("Built %s chunks; creating vector index.", len(chunks))
        self.vector_store.build_index(chunks)
        logger.info("Ingestion pipeline completed.")

    def answer(self, query: str, *, top_k: Optional[int] = None) -> Answer:
        """执行一次检索增强问答。"""
        if not query.strip():
            raise ValueError("Query must not be empty.")

        start_total = time.perf_counter()
        # 先检索较大的候选集，再由重排器选出最终 Top-K
        retrieval_depth = max(self.settings.rerank_top_k, top_k or self.settings.top_k)
        retriever = Retriever(self.vector_store, top_k=retrieval_depth)

        start_retrieval = time.perf_counter()
        retrieval_results = retriever.retrieve(query)
        for idx, item in enumerate(retrieval_results, start=1):
            item.chunk.metadata["retrieval_rank"] = idx
        reranked_results = self.reranker.rerank(
            query, retrieval_results, top_k or self.settings.top_k
        )
        for idx, item in enumerate(reranked_results, start=1):
            item.chunk.metadata["rerank_rank"] = idx
        retrieval_ms = (time.perf_counter() - start_retrieval) * 1000

        start_generation = time.perf_counter()
        generation_payload = self.generator.generate_answer(
            query=query, contexts=reranked_results
        )
        generation_ms = (time.perf_counter() - start_generation) * 1000

        total_ms = (time.perf_counter() - start_total) * 1000

        context_payload = [
            {
                "doc_id": item.chunk.doc_id,
                "chunk_id": item.chunk.chunk_id,
                "text": item.chunk.text,
                "metadata": item.chunk.metadata,
            }
            for item in reranked_results
        ]

        answer = Answer(
            query=query,
            answer=generation_payload["answer"],
            citations=generation_payload["citations"],
            contexts=context_payload,
            latency_ms=total_ms,
        )

        self.metrics.record(
            query=query,
            latency_ms=total_ms,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            retrieved_k=len(reranked_results),
        )
        #retrieval_results = retriever.retrieve(query)
        #print(f"Retrieval Results: {[item.chunk.doc_id for item in retrieval_results]}")

        #reranked_results = self.reranker.rerank(query, retrieval_results, top_k or self.settings.top_k)
        #print(f"Reranked Results: {[item.chunk.doc_id for item in reranked_results]}")

        return answer

    def health(self) -> Dict[str, str]:
        """返回状态检查信息，便于 API 快速验证状态。"""

        return {
            "index_path": str(self.settings.index_path),
            "storage_dir": str(self.settings.storage_dir),
            "document_dir": str(self.settings.document_dir),
        }
