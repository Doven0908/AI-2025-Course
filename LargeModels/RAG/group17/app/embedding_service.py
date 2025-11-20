from __future__ import annotations
import hashlib
import pickle
import threading
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from langchain_core.embeddings import Embeddings
from .config import Settings
from .logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """FlagEmbedding 包装器，内置简单磁盘缓存以提升复用效率。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_lock = threading.Lock()
        self._cache_path = settings.embedding_cache_path
        self._cache: dict[str, np.ndarray] = {}
        self._cache_dirty = False
        self._cache_lock = threading.Lock()
        self._load_cache()

    def _load_model(self):
        """懒加载 embedding 模型，线程安全。"""
        if self._model is not None:
            return

        with self._model_lock:
            # 双重检查：其他线程可能已经加载完成
            if self._model is not None:
                return

            try:
                from FlagEmbedding import FlagAutoModel
            except ImportError as exc:
                raise RuntimeError(
                    "FlagEmbedding is required but not installed. "
                    "Please ensure requirements are installed."
                ) from exc

            logger.info(
                "Loading embedding model %s on %s",
                self.settings.embedding_model,
                self._device.upper(),
            )
            self._model = FlagAutoModel.from_finetuned(
                self.settings.embedding_model,
                query_instruction_for_retrieval=(
                    "Represent this sentence for searching relevant passages:"
                ),
                use_fp16=True,
            )
            if hasattr(self._model, "target_devices"):
                self._model.target_devices = [self._device]
            logger.info("Embedding model loaded successfully")

    def _load_cache(self) -> None:
        """从磁盘加载缓存。"""
        if self._cache_path.exists():
            try:
                with self._cache_path.open("rb") as f:
                    self._cache = pickle.load(f)
                logger.info("Loaded embedding cache with %s entries", len(self._cache))
            except Exception as exc:
                logger.warning("Failed to load embedding cache: %s", exc)
                self._cache = {}

    def _save_cache(self) -> None:
        """持久化缓存到磁盘。"""
        if not self._cache_dirty:
            return

        with self._cache_lock:  # ✅ 保护缓存写入
            if not self._cache_dirty:  # 双重检查
                return
            try:
                # 创建父目录
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self._cache_path.open("wb") as f:
                    pickle.dump(self._cache, f)
                logger.info("Persisted embedding cache (%s entries)", len(self._cache))
                self._cache_dirty = False
            except Exception as exc:
                logger.error("Failed to persist embedding cache: %s", exc)

    @staticmethod
    def _hash_text(text: str) -> str:
        """生成文本的哈希值作为缓存键。"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """调用底层模型完成批量编码。"""
        self._load_model()
        # BGE 系列模型的最大输入长度为 512 tokens，超出会导致内部
        # tokenizer 回退重试并最终触发 batch_size=0 的 edge case。
        # 将 max_length 固定为 512，减少 FlagEmbedding 内部重试。
        embeddings = self._model.encode(
            texts,
            batch_size=8,
            max_length=512,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        """批量编码文本，并自动读取/写入缓存。"""
        texts_list = list(texts)
        if not texts_list:  # ✅ 处理空列表
            return np.array([]).reshape(0, -1)

        embeddings: List[np.ndarray] = []
        missing_texts: List[str] = []
        missing_indices: List[int] = []

        # 读取缓存（使用锁保护）
        with self._cache_lock:
            for idx, text in enumerate(texts_list):
                key = self._hash_text(text)
                cached = self._cache.get(key)
                if cached is not None:
                    embeddings.append(cached)
                else:
                    embeddings.append(None)  # type: ignore
                    missing_texts.append(text)
                    missing_indices.append(idx)

        # 计算缺失的 embeddings
        if missing_texts:
            new_embeds = self._embed_batch(missing_texts)

            # 更新缓存（使用锁保护）
            with self._cache_lock:
                for offset, embed in enumerate(new_embeds):
                    target_idx = missing_indices[offset]
                    embeddings[target_idx] = embed
                    cache_key = self._hash_text(texts_list[target_idx])
                    self._cache[cache_key] = embed
                    self._cache_dirty = True

            self._save_cache()

        return np.vstack(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """针对单条查询返回向量。"""
        return self.embed_texts([query])[0]

    @property
    def model(self):
        """延迟加载属性，用于预热。"""
        self._load_model()
        return self._model

    def as_langchain_embedding(self) -> Embeddings:
        """暴露 LangChain 兼容接口，便于向量库调用。"""
        service = self

        class _Adapter(Embeddings):
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                vectors = service.embed_texts(texts)
                return vectors.tolist()

            def embed_query(self, text: str) -> List[float]:
                return service.embed_query(text).tolist()

        return _Adapter()
