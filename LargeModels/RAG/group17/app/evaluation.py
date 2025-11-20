from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from .logger import get_logger
from .pipeline import RAGPipeline
from .retriever import Retriever

logger = get_logger(__name__)


@dataclass
class Sample:
    """评估数据集中单条样本。"""

    question: str
    ground_truths: List[str]


def _load_dataset(path: Path) -> List[Sample]:
    """根据文件格式加载评估数据集。"""
    samples: List[Sample] = []
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            samples.append(
                Sample(
                    question=record["question"],
                    ground_truths=list(record.get("ground_truths", [])),
                )
            )
    elif path.suffix.lower() in {".json"}:
        data = json.loads(path.read_text(encoding="utf-8"))
        for record in data:
            samples.append(
                Sample(
                    question=record["question"],
                    ground_truths=list(record.get("ground_truths", [])),
                )
            )
    else:
        raise ValueError(
            f"Unsupported dataset format: {path.suffix}. Use .json or .jsonl."
        )
    return samples


async def evaluate_dataset(
    *,
    pipeline: RAGPipeline,
    dataset_path: Path,
    limit: Optional[int] = None,
) -> Dict:
    """批量运行问答获得指标，并调用 RAGAS 打分。"""
    samples = _load_dataset(dataset_path)
    if limit:
        samples = samples[:limit]
    if not samples:
        raise ValueError("Dataset is empty.")

    ragas_rows = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    retrieval_hits = 0
    answer_hits = 0
    failure_cases: Dict[str, List[Dict]] = {
        "retrieval_error": [],
        "rerank_error": [],
        "generation_error": [],
    }

    for idx, sample in enumerate(samples, start=1):
        logger.info("Evaluating sample %s/%s", idx, len(samples))
        answer = await asyncio.to_thread(pipeline.answer, sample.question)
        contexts_info = answer.contexts
        contexts = [ctx["text"] for ctx in contexts_info]

        retriever = Retriever(
            pipeline.vector_store, top_k=pipeline.settings.rerank_top_k
        )
        raw_candidates = await asyncio.to_thread(retriever.retrieve, sample.question)
        raw_texts = [cand.chunk.text for cand in raw_candidates]

        ragas_rows["question"].append(sample.question)
        ragas_rows["answer"].append(answer.answer)
        ragas_rows["contexts"].append(contexts)
        ragas_rows["ground_truth"].append(sample.ground_truths)

        ground_truth_present = False
        first_hit_rank: Optional[int] = None
        for rank, context in enumerate(raw_texts, start=1):
            if any(gt and gt in context for gt in sample.ground_truths):
                ground_truth_present = True
                first_hit_rank = rank
                break

        if ground_truth_present:
            retrieval_hits += 1

        answer_contains_gt = any(
            gt and gt in answer.answer for gt in sample.ground_truths
        )
        if answer_contains_gt:
            answer_hits += 1

        final_contains_gt = any(
            any(gt and gt in ctx["text"] for gt in sample.ground_truths)
            for ctx in contexts_info
        )

        if not ground_truth_present:
            failure_cases["retrieval_error"].append(
                {
                    "question": sample.question,
                    "answer": answer.answer,
                    "ground_truths": sample.ground_truths,
                    "contexts": raw_texts,
                }
            )
        elif not final_contains_gt:
            failure_cases["rerank_error"].append(
                {
                    "question": sample.question,
                    "answer": answer.answer,
                    "ground_truths": sample.ground_truths,
                    "contexts": raw_texts,
                    "hit_rank": first_hit_rank,
                }
            )
        elif not answer_contains_gt:
            failure_cases["generation_error"].append(
                {
                    "question": sample.question,
                    "answer": answer.answer,
                    "ground_truths": sample.ground_truths,
                    "contexts": contexts,
                }
            )

    dataset = Dataset.from_dict(ragas_rows)
    ragas_result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    ragas_scores: Dict[str, float] = {}
    if hasattr(ragas_result, "to_pandas"):
        df = ragas_result.to_pandas()
        for column in df.columns:
            ragas_scores[column] = float(df[column].mean())
    elif hasattr(ragas_result, "to_dict"):
        ragas_scores = ragas_result.to_dict()
    else:
        ragas_scores = {"faithfulness": float("nan")}

    report = {
        "dataset_size": len(samples),
        "retrieval_recall": retrieval_hits / len(samples),
        "answer_hit_rate": answer_hits / len(samples),
        "ragas": ragas_scores,
        "failure_cases": failure_cases,
    }
    return report
