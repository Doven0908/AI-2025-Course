
import os
import time
from typing import Optional
import warnings
import openai
from pydantic import Field, BaseModel

from langchain_openai import ChatOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper


from llm_hallucination_evaluate.evaluator.base import Evaluator
from llm_hallucination_evaluate.solver.solve_result import SolveResult
from llm_hallucination_evaluate.evaluator.eval_result import EvalResult


from ragas.metrics import *



# YAML配置到Ragas指标的映射字典
METRIC_MAPPING = {
    # 基础准确率指标
    "accuracy": AnswerAccuracy(name="accuracy"),
    
    # 相关性指标
    "relevance": AnswerRelevancy(name="relevance"),
    
    # 正确性指标
    "correctness": AnswerCorrectness(name="correctness"),
    "factual_correctness": FactualCorrectness(name="factual_correctness", mode='f1'),
    
    # 忠实度指标
    "faithfulness": Faithfulness(name="faithfulness"),

    # Context相关指标
    "context_recall": LLMContextRecall(name="context_recall"),
    "context_precision": ContextPrecision(name="context_precision"),
    "context_relevance": ContextRelevance(name="context_relevance"),


    # 文本质量指标
    "conciseness": SimpleCriteriaScore(
        name="conciseness",
        definition="评估回答是否简洁明了，避免不必要的冗余信息"
    ),
    "coherence": SimpleCriteriaScore(
        name="coherence", 
        definition="评估回答的逻辑连贯性和结构合理性"
    ),
    "fluency": SimpleCriteriaScore(
        name="fluency",
        definition="评估回答的语言流畅度和语法正确性"
    ),
    
    # 整体质量
    "overall_quality": AspectCritic(
        name="overall_quality",
        definition="综合评估回答的整体质量，包括准确性、相关性和完整性"
    )
}

class RagasEvaluator(Evaluator):

    model: str = Field(..., description="用于 RAGAS 评估的语言模型名称")
    metrics: list[str] = Field(default_factory=list, description="评估指标列表")
    evaluator_llm: None = Field(None, description="RAGAS 评估使用的语言模型包装器")
    metric_instances: list = Field(default_factory=list, description="RAGAS 评估使用的指标实例列表")
    embeddings: None = Field(None, description="用于 RAGAS 评估的嵌入模型实例")

    def __init__(self, **data):
        super().__init__(**data)

        api_key = os.environ["OPENAI_API_KEY"]
        api_base = os.environ.get("OPENAI_API_BASE")

        llm = ChatOpenAI(model=self.model, api_key=api_key, base_url=api_base)

        self.evaluator_llm = LangchainLLMWrapper(llm)


        openai_client = openai.OpenAI(api_key=api_key, base_url=api_base)  
        self.embeddings = OpenAIEmbeddings(client=openai_client)

        self.add_metric_instances()


    def add_metric_instances(self):
        """根据配置的指标名称，初始化对应的指标实例列表"""
        self.metric_instances = []

        for metric_name in self.metrics:
            if metric_name in METRIC_MAPPING:
                self.metric_instances.append(METRIC_MAPPING[metric_name])
            else:
                warnings.warn(f"未知的 RAGAS 评估指标名称: {metric_name}")



    def single_evaluate(self, solve_result: SolveResult) -> EvalResult:
        """对单个求解结果进行 RAGAS 评估，返回评估结果"""

        dataset = EvaluationDataset.from_list([
            {
                "user_input": solve_result.dataitem.question,
                "reference": solve_result.dataitem.answer,   
                "response": solve_result.answer,
                "retrieved_contexts": solve_result.metadata.get('retrieved_contexts', None),
                "reference_contexts": [ctx.content for ctx in solve_result.dataitem.contexts],
            }
        ])

        start_time = time.time()

        result = evaluate(
            dataset=dataset,
            metrics=self.metric_instances,
            llm=self.evaluator_llm,
            embeddings=self.embeddings,
            raise_exceptions=False,
        )
        end_time = time.time()
        elapsed = end_time - start_time

        print(f"RAGAS 评估完成，耗时 {elapsed:.2f} 秒")

        if 'factual_correctness(mode=f1)' in result.scores[0]:
            result.scores[0]['factual_correctness'] = result.scores[0].pop('factual_correctness(mode=f1)')

        eval_result = EvalResult(
            evaluator_name=self.method,
            results=result.scores[0],
            solve_result=solve_result,
        )

        # import IPython; IPython.embed(); exit(1)

        return eval_result
