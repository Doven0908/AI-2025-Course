from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _clean_api_key(raw: Optional[str]) -> Optional[str]:
    """清洗环境变量中的密钥，兼容带前缀的写法。"""
    if not raw:
        return None
    value = raw.strip().strip('"').strip("'")
    if value.lower().startswith("api_key="):
        value = value.split("=", 1)[1].strip()
    return value or None


@dataclass
class Settings:
    """应用配置对象，集中管理路径、模型与参数等信息。"""

    document_dir: Path
    storage_dir: Path
    index_path: Path
    store_doc_path: Path
    embedding_cache_path: Path
    metrics_db_path: Path
    api_key: Optional[str]
    llm_base_url: str
    llm_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    rerank_top_k: int
    reranker_name: str
    reranker_model: str
    eval_api_key: Optional[str]
    eval_llm_base_url: Optional[str]
    eval_llm_model: Optional[str]
    eval_embedding_model: Optional[str]


def load_settings() -> Settings:
    """从环境变量读取配置，并创建 Settings 实例。"""

    # 存储目录用于保存索引、缓存和指标数据库
    storage_dir = Path(os.getenv("STORAGE_DIR", "storage"))
    storage_dir.mkdir(parents=True, exist_ok=True)

    api_key = _clean_api_key(os.getenv("API_KEY"))

    eval_api_key = _clean_api_key(os.getenv("EVAL_API_KEY") or os.getenv("API_KEY"))
    if os.getenv("EVAL_LLM_BASE_URL"):
        eval_llm_base_url = os.getenv("EVAL_LLM_BASE_URL")
    elif os.getenv("EVAL_API_KEY"):
        eval_llm_base_url = os.getenv("EVAL_LLM_BASE_URL", "https://api.sydney-ai.com/v1")
    else:
        eval_llm_base_url = os.getenv("LLM_BASE_URL")

    eval_llm_model = os.getenv("EVAL_LLM_MODEL")
    if not eval_llm_model:
        eval_llm_model = (
            "deepseek-v3-0323"
            if os.getenv("EVAL_API_KEY")
            else os.getenv("LLM_MODEL", "qwen-flash")
        )

    # 换用DashScope兼容的embedding试一下
    eval_embedding_model = os.getenv("EVAL_EMBED_MODEL", 
                                     # "text-embedding-3-small",
                                    "text-embedding-v4"
                                    )

    return Settings(
        document_dir=Path(os.getenv("DOCUMENT_DIR", "Document")),
        storage_dir=storage_dir,
        index_path=storage_dir / os.getenv("INDEX_FILENAME", "vector_index.faiss"),
        store_doc_path=storage_dir / os.getenv("METADATA_FILENAME", "chunks.jsonl"),
        embedding_cache_path=storage_dir
        / os.getenv("EMBED_CACHE_FILENAME", "embedding_cache.pkl"),
        metrics_db_path=storage_dir / os.getenv("METRICS_DB_FILENAME", "metrics.db"),
        api_key=api_key,
        llm_base_url=os.getenv(
            "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        llm_model=os.getenv("LLM_MODEL", "qwen3-14b"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "400")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),
        top_k=int(os.getenv("TOP_K", "5")),
        rerank_top_k=int(os.getenv("RERANK_TOP_K", "12")),
        reranker_name=os.getenv("RERANKER", "none"),
        reranker_model=os.getenv("RERANKER_MODEL", "colbert-ir/colbertv2.0"),
        eval_api_key=eval_api_key,
        eval_llm_base_url=eval_llm_base_url,
        eval_llm_model=eval_llm_model,
        eval_embedding_model=eval_embedding_model,
    )
