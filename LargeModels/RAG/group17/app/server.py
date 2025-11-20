from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import Settings, load_settings
from .logger import get_logger
from .pipeline import RAGPipeline
from .ragas_evaluator import RagasEvaluationManager
from .ragatouille_compat import ensure_ragatouille_dependencies
from .schemas import (
    AskRequest,
    AskResponse,
    EvaluateRequest,
    FeedbackRequest,
    InlineEvaluation,
    MetricsResponse,
)

logger = get_logger(__name__)
from fastapi.middleware.cors import CORSMiddleware


def create_app() -> FastAPI:
    """创建 FastAPI 应用并注册路由。"""
    settings = load_settings()
    pipelines: Dict[str, RAGPipeline] = {}
    pipeline_configs: Dict[str, Settings] = {}
    variant_metadata: Dict[str, dict] = {}

    pipeline_configs["faiss"] = settings
    variant_metadata["faiss"] = {
        "label": "FAISS 检索",
        "description": "使用向量相似度排序直接返回 Top-K 片段。",
        "available": True,
    }

    colbert_available = True
    colbert_error: str | None = None
    try:
        ensure_ragatouille_dependencies()
        from ragatouille import RAGPretrainedModel as _RAGCheck  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        colbert_available = False
        colbert_error = (
            "ColBERT 精排依赖 ragatouille，请先安装依赖后再尝试。"
        )
        logger.warning("ColBERT pipeline disabled: %s", exc)

    if colbert_available:
        try:
            colbert_settings = replace(
                settings,
                reranker_name="colbert",
                metrics_db_path=settings.metrics_db_path.with_name("metrics_colbert.db"),
            )
            pipeline_configs["colbert"] = colbert_settings
            variant_metadata["colbert"] = {
                "label": "ColBERTv2 精排",
                "description": "向量检索后使用 ColBERTv2 进行精排。",
                "available": True,
            }
        except Exception as exc:
            logger.error("Failed to prepare ColBERT settings: %s", exc)
            colbert_available = False
            colbert_error = str(exc)

    if not colbert_available:
        variant_metadata["colbert"] = {
            "label": "ColBERTv2 精排",
            "description": "向量检索后使用 ColBERTv2 进行精排。",
            "available": False,
            "error": colbert_error,
        }

    ragas_manager = RagasEvaluationManager(
        dataset_path=settings.storage_dir / "evaluation_dataset.json",
        settings=settings,
    )

    app = FastAPI(title="RAG QA Service", version="1.0.0")

    templates_dir = Path("templates")
    if templates_dir.exists():
        templates = Jinja2Templates(directory=str(templates_dir))
    else:
        templates = None

    static_dir = Path("static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    feedback_path = settings.storage_dir / "feedback.jsonl"

    def get_pipeline(variant: str) -> RAGPipeline:
        meta = variant_metadata.get(variant)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"未知的检索器: {variant}")
        if not meta.get("available", True):
            error_message = meta.get("error") or "该检索器当前不可用。"
            raise HTTPException(status_code=503, detail=error_message)

        pipeline = pipelines.get(variant)
        if pipeline is not None:
            return pipeline

        settings_obj = pipeline_configs.get(variant)
        if settings_obj is None:
            raise HTTPException(status_code=404, detail=f"未知的检索器: {variant}")

        try:
            pipeline = RAGPipeline(settings_obj)
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.exception("Failed to initialise pipeline '%s': %s", variant, exc)
            raise HTTPException(
                status_code=500,
                detail=f"初始化检索器 '{variant}' 失败: {exc}",
            ) from exc

        pipelines[variant] = pipeline
        return pipeline

    primary_pipeline = get_pipeline("faiss")

    # ✅ 预加载模型（修正版）
    logger.info("🔄 预加载模型中...")
    try:
        # 触发 embedding 模型加载
        _ = primary_pipeline.embedding_service.model

        # 触发 FAISS 索引加载（通过执行一次假查询）
        # 不要直接访问 .index 属性，而是调用方法
        try:
            primary_pipeline.vector_store.similarity_search_with_score("预热查询", k=1)
        except Exception:
            pass  # 索引可能为空，忽略错误

        logger.info("✅ 模型预加载完成")
    except Exception as e:
        logger.error(f"❌ 模型预加载失败: {e}")


    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        if not templates:
            return HTMLResponse(
                "<h1>RAG QA Service</h1><p>API is running. Use /docs for documentation.</p>"
            )
        return templates.TemplateResponse(
            "landing.html",
            {
                "request": request,
                "variants": variant_metadata,
            },
        )

    @app.get("/qa/{variant}", response_class=HTMLResponse, name="qa_page")
    async def qa_page(request: Request, variant: str):
        if not templates:
            raise HTTPException(status_code=503, detail="HTML templates not available")
        meta = variant_metadata.get(variant)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"未知的检索器: {variant}")
        if not meta.get("available", True):
            error_message = meta.get("error") or "该检索器当前不可用。"
            raise HTTPException(status_code=503, detail=error_message)
        return templates.TemplateResponse(
            "qa.html",
            {
                "request": request,
                "variant": variant,
                "variant_meta": variant_metadata.get(variant, {}),
                "variants": variant_metadata,
            },
        )

    @app.get("/ragas", response_class=HTMLResponse, name="ragas_page")
    async def ragas_page(request: Request):
        if not templates:
            raise HTTPException(status_code=503, detail="HTML templates not available")
        return templates.TemplateResponse(
            "ragas.html",
            {
                "request": request,
                "variants": variant_metadata,
            },
        )


    @app.post("/ask/{variant}", response_model=AskResponse, name="ask_variant")
    async def ask(variant: str, request: AskRequest):
        """问答接口,调用完整 RAG 流程。"""
        pipeline = get_pipeline(variant)
        try:
            # ✅ 将同步操作放到线程池执行，避免阻塞事件循环
            answer = await asyncio.to_thread(
                pipeline.answer,
                request.query,
                top_k=request.top_k
            )
            evaluation_payload = None
            if request.evaluate:
                contexts_texts = [ctx.get("text", "") for ctx in answer.contexts]
                ground_truths = request.ground_truths or []
                try:
                    raw_eval = await asyncio.to_thread(
                        ragas_manager.evaluate_inline,
                        question=request.query,
                        answer=answer.answer,
                        contexts=contexts_texts,
                        ground_truths=ground_truths if ground_truths else None,
                    )
                    evaluation_payload = {
                        "metrics": raw_eval.get("metrics", []),
                        "used_ground_truths": raw_eval.get("used_ground_truths", False),
                        "ground_truth_source": raw_eval.get("ground_truth_source", "none"),
                        "reference": raw_eval.get("reference"),
                        "references": raw_eval.get("references", []),
                        "diagnosis": raw_eval.get("diagnosis", []),
                    }
                except Exception as eval_exc:  # pragma: no cover - runtime guard
                    logger.warning("RAGAS evaluation failed: %s", eval_exc)
                    evaluation_payload = {
                        "metrics": [],
                        "used_ground_truths": bool(request.ground_truths),
                        "ground_truth_source": "error",
                        "error": str(eval_exc),
                        "diagnosis": [],
                    }
            return AskResponse(
                answer=answer.answer,
                citations=answer.citations,
                contexts=answer.contexts,
                latency_ms=answer.latency_ms,
                timestamp=answer.timestamp,
                evaluation=evaluation_payload,
            )
        except Exception as exc:
            logger.exception("Failed to answer query: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/feedback")
    async def feedback(request: FeedbackRequest):
        """记录用户反馈，写入本地 JSONL 文件。"""
        record = request.model_dump()
        record["timestamp"] = datetime.utcnow().isoformat()
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with feedback_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"status": "ok"}

    @app.get("/metrics/{variant}", response_model=MetricsResponse, name="metrics_variant")
    async def metrics(variant: str):
        """返回近期性能指标列表与聚合数据。"""
        pipeline = get_pipeline(variant)
        aggregates = pipeline.metrics.aggregates()
        records = [
            {
                "query": record.get("query", ""),
                "latency_ms": record.get("latency_ms", 0.0),
                "retrieval_ms": record.get("retrieval_ms", 0.0),
                "generation_ms": record.get("generation_ms", 0.0),
                "retrieved_k": record.get("retrieved_k", 0),
                "timestamp": record.get("timestamp", ""),
                "status": record.get("status", "unknown"),
            }
            for record in pipeline.metrics.recent()
        ]
        return MetricsResponse(aggregates=aggregates, records=records)

    @app.get("/health")
    async def health():
        """健康检查接口。"""
        # 默认返回主 pipeline 的健康信息
        return pipelines["faiss"].health()

    @app.get("/ragas/dataset")
    async def ragas_dataset():
        """返回当前评估数据集，方便前端展示。"""
        samples = await asyncio.to_thread(ragas_manager.load_dataset)
        return {"samples": samples}

    @app.post("/ragas/evaluate", response_model=InlineEvaluation)
    async def ragas_evaluate(request: EvaluateRequest):
        """独立执行一次 RAGAS 评估。"""
        try:
            result = await asyncio.to_thread(
                ragas_manager.evaluate_inline,
                question=request.question,
                answer=request.answer,
                contexts=request.contexts,
                ground_truths=request.ground_truths,
                reference=request.reference,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return InlineEvaluation(**result)

    @app.get("/ragas/{variant}/results")
    async def ragas_results(variant: str):
        """读取指定检索器的最新 RAGAS 评估结果。"""
        _ = get_pipeline(variant)
        cached = await asyncio.to_thread(ragas_manager.load_cached, variant)
        if cached is None:
            raise HTTPException(status_code=404, detail="尚未生成评估结果")
        return cached

    @app.post("/ragas/{variant}/run")
    async def ragas_run(variant: str):
        """触发一次 RAGAS 评估。"""
        pipeline = get_pipeline(variant)
        try:
            result = await asyncio.to_thread(ragas_manager.run, pipeline, variant=variant)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"status": "ok", "result": result.to_serializable()}

    return app


app = create_app()
