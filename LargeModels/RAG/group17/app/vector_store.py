from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .embedding_service import EmbeddingService
from .logger import get_logger
from .types import DocumentChunk

logger = get_logger(__name__)


class VectorStoreManager:
    """负责构建、保存与加载 FAISS 向量库，并提供检索接口。"""

    def __init__(
        self,
        *,
        index_path: Path,
        metadata_path: Path,
        embedding_service: EmbeddingService,
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_service = embedding_service
        self._vector_store: Optional[FAISS] = None

    def _ensure_vector_store(self) -> FAISS:
        """确保向量库已加载，如未初始化则从磁盘恢复。"""
        if self._vector_store is not None:
            return self._vector_store

        if self.index_path.exists() and self.metadata_path.exists():
            logger.info("Loading FAISS index from %s", self.index_path)
            self._vector_store = FAISS.load_local(
                str(self.index_path),
                self.embedding_service.as_langchain_embedding(),
                allow_dangerous_deserialization=True,
            )
        else:
            raise RuntimeError(
                "Vector store not initialised. Build the index before querying."
            )
        return self._vector_store

    def build_index(self, chunks: Iterable[DocumentChunk]) -> None:
        """将分块后的文本向量化并写入 FAISS 索引。"""
        docs: List[Document] = []

        for chunk in chunks:
            metadata = {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "source": chunk.metadata.get("source_path"),
                "source_name": chunk.metadata.get("source_name"),
                "start_index": chunk.start_index,
                "end_index": chunk.end_index,
            }
            docs.append(Document(page_content=chunk.text, metadata=metadata))

        logger.info("Embedding %s chunks for FAISS index", len(docs))
        adapter = self.embedding_service.as_langchain_embedding()

        index = FAISS.from_documents(
            documents=docs,
            embedding=adapter,
        )

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        index.save_local(str(self.index_path))

        with self.metadata_path.open("w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc.metadata, ensure_ascii=False) + "\n")

        self._vector_store = index
        logger.info("Persisted FAISS index with %s vectors", len(docs))

    def similarity_search(self, query: str, k: int) -> List[Document]:
        """执行相似度检索，返回文档列表。"""
        return self._ensure_vector_store().similarity_search(query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int
    ) -> List[tuple[Document, float]]:
        """执行检索并同时返回相似度分数。"""
        return self._ensure_vector_store().similarity_search_with_score(query, k=k)
