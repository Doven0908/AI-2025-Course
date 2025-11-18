from typing import Callable
from pydantic import Field

from llm_hallucination_evaluate.dataset.base import DataItem
from llm_hallucination_evaluate.solver.base import Solver
from llm_hallucination_evaluate.solver.solve_result import SolveResult
from llm_hallucination_evaluate.utils.config import SolverConfig
from llm_hallucination_evaluate.solver.llm_model import get_model_func

class DirectAnswerSolver(Solver):
    """直接问答求解器"""


    @classmethod
    def from_(cls, solver_cfg: SolverConfig) -> "DirectAnswerSolver":
        """从配置中创建 DirectAnswerSolver 实例"""

        model_func = get_model_func(solver_cfg.type, solver_cfg.model)

        return cls(llm_qa_func=model_func, extra_args=solver_cfg.extra_args, method=solver_cfg.method, model_name=solver_cfg.model)



    def _solve(self, data_item: DataItem) -> SolveResult:

        """对单个数据项进行求解，返回求解结果"""

        # 调用LLM问答函数获取答案
        answer = self.llm_qa_func(
            data_item.question,
            **self.extra_args
        )

        # 构造SolveResult对象
        result = SolveResult(
            dataitem=data_item,
            answer=answer,
            metadata={}
        )

        return result
