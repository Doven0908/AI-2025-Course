from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Sequence


@dataclass
class RawDocument:
    """原始文档实体，记录读取到的全文内容与来源信息。"""
    doc_id: str
    text: str
    source_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """分块后的文档片段，用于向量化与检索。"""
    doc_id: str
    chunk_id: str
    text: str
    start_index: int
    end_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """检索结果，包含片段与相似度得分。"""
    chunk: DocumentChunk
    score: float


@dataclass
class Answer:
    """最终回答对象，携带回复、引用与上下文信息。"""
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    contexts: List[Dict[str, Any]]
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EvaluationSample:
    """离线评测样本，保存问题与标准答案。"""
    question: str
    ground_truths: Sequence[str]
