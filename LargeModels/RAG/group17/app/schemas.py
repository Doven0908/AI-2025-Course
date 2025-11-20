from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """问答请求体。"""
    query: str = Field(..., description="User question.")
    top_k: Optional[int] = Field(None, description="Override retrieval depth.")
    evaluate: bool = Field(False, description="Whether to run RAGAS evaluation.")
    ground_truths: Optional[List[str]] = Field(
        None,
        description="Optional reference answers used for RAGAS metrics.",
    )


class Citation(BaseModel):
    """引用信息结构，用于前端展示出处。"""
    label: str
    source: Optional[str]
    doc_id: str
    chunk_id: str
    excerpt: str


class EvaluationMetric(BaseModel):
    """RAGAS 单项指标得分。"""
    name: str
    score: Optional[float]
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None


class InlineEvaluation(BaseModel):
    """前端展示的 RAGAS 评估结果。"""
    metrics: List[EvaluationMetric] = Field(default_factory=list)
    used_ground_truths: bool = False
    ground_truth_source: str = "none"
    reference: Optional[str] = None
    references: List[str] = Field(default_factory=list)
    diagnosis: List[Dict[str, str]] = Field(default_factory=list)
    raw: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AskResponse(BaseModel):
    """问答响应，包含答案、引用与耗时。"""
    answer: str
    citations: List[Citation]
    contexts: List[Dict[str, Any]]
    latency_ms: float
    timestamp: datetime
    evaluation: Optional[InlineEvaluation] = None


class EvaluateRequest(BaseModel):
    """独立评估请求体。"""
    question: str
    answer: str
    contexts: List[str]
    ground_truths: Optional[List[str]] = None
    reference: Optional[str] = None


class EvaluateResponse(BaseModel):
    """独立评估响应。"""
    evaluation: InlineEvaluation


class FeedbackRequest(BaseModel):
    """用户反馈请求体。"""
    query: str
    helpful: bool
    comments: Optional[str] = None


class MetricsRecord(BaseModel):
    """单条查询指标。"""
    query: str
    latency_ms: float
    retrieval_ms: float
    generation_ms: float
    retrieved_k: int
    timestamp: datetime
    status: str


class MetricsResponse(BaseModel):
    """指标接口响应。"""
    aggregates: dict
    records: List[MetricsRecord]
