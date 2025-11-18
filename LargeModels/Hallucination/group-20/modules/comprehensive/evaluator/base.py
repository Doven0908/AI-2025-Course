
from pydantic import Field, BaseModel

from llm_hallucination_evaluate.utils.config import ExperimentConfig

class Evaluator(BaseModel):
    
    method: str = Field(..., description="评估方法名称")

    @classmethod
    def load_from_config(cls, config: ExperimentConfig) -> "Evaluator":
        
        assert len(config.evaluators) == 1, "当前仅支持单一评估器配置"
        assert config.evaluators[0].method == "ragas", "当前仅支持 RAGAS 评估方法"

        from llm_hallucination_evaluate.evaluator.ragas.ragas_evaluator import RagasEvaluator
        return RagasEvaluator(model=config.evaluators[0].model, metrics=config.evaluators[0].metrics, method=config.evaluators[0].method)

