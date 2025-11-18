
import json
import yaml

from pydantic import BaseModel, Field

from llm_hallucination_evaluate.solver.solve_result import SolveResult

class EvalResult(BaseModel):
    evaluator_name: str = Field(..., description="评估器名称")
    results: dict = Field(..., description="评估结果字典")
    solve_result: SolveResult = Field(..., description="对应的求解结果")

    def save(self, filepath: str):
        """将评估结果保存到指定文件路径"""
        yaml.dump(json.loads(self.model_dump_json()), open(filepath, 'w') )

    @staticmethod
    def read_from_yaml(filepath: str) -> "EvalResult":
        """从指定的YAML文件路径读取评估结果"""
        data = yaml.load(open(filepath, 'r'), Loader=yaml.FullLoader)
        return EvalResult.model_validate(data)
