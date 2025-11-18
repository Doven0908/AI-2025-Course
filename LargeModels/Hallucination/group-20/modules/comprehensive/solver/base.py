
import time
from typing import Callable, Optional
from pydantic import BaseModel, RootModel, Field

from llm_hallucination_evaluate.utils.config import MethodEnum, SolverConfig
from llm_hallucination_evaluate.dataset.base import DataItem
from llm_hallucination_evaluate.solver.solve_result import SolveResult

class Solver(BaseModel):
    
    method: MethodEnum = Field(..., description="求解方法")
    model_name: str = Field(..., description="使用的模型名称")
    llm_qa_func: Callable = Field(..., description="用于直接问答的LLM调用函数")
    extra_args: dict = Field(default_factory=dict, description="额外参数字典")

    @classmethod
    def load_from_config(cls, solver_cfg: SolverConfig) -> "Solver":
        """从配置中加载单个求解器"""
        if solver_cfg.method == MethodEnum.DIRECT_ANSWER:
            from llm_hallucination_evaluate.solver.direct_answer_solver import DirectAnswerSolver
            return DirectAnswerSolver.from_(solver_cfg)
        elif solver_cfg.method == MethodEnum.USE_RAG:
            from llm_hallucination_evaluate.solver.use_rag_solver import UseRagSolver
            return UseRagSolver.from_(solver_cfg)
        elif solver_cfg.method == MethodEnum.USE_COVE:
            from llm_hallucination_evaluate.solver.use_cove_solver import UseCoveSolver
            return UseCoveSolver.from_(solver_cfg)
        elif solver_cfg.method == MethodEnum.USE_SELF_RAG:
            from llm_hallucination_evaluate.solver.use_selfrag_solver import UseSelfragSolver
            return UseSelfragSolver.from_(solver_cfg)
        elif solver_cfg.method == MethodEnum.USE_SELFCHECKGPT:
            from llm_hallucination_evaluate.solver.use_selfcheckgpt_solver import UseSelfchekgptSolver
            return UseSelfchekgptSolver.from_(solver_cfg)
        else:
            raise ValueError(f"不支持的求解方法: {solver_cfg.method}")

    def _solve(self, data_item: DataItem) -> SolveResult:
        """对单个数据项进行求解，返回求解结果"""
        raise NotImplementedError("子类必须实现 _solve 方法")

    def solve(self, data_item: DataItem) -> SolveResult:

        start_time = time.time()
        result = self._solve(data_item)
        end_time = time.time()

        # 在结果中添加求解时间元数据
        result.metadata['solve_time'] = end_time - start_time
        result.metadata['solve_method'] = self.method
        result.metadata['model_name'] = self.model_name
        result.metadata['dataset'] = data_item.meta.get('dataset', 'unknown')

        return result



class MultiSolver(RootModel):
    root: list[Solver] = Field(..., description="多个求解器组成的列表")

    def __len__(self):
        return len(self.root)
    
    def __item__(self, index):
        return self.root[index]
    
    def __iter__(self):
        return iter(self.root)

    @classmethod
    def load_from_config(cls, config):
        """从配置中加载多个求解器"""

        print('\n加载求解器...')
        print('注意监控硬件资源使用情况...\n')
        solvers = []
        for solver_cfg in config.solvers:
            solver = Solver.load_from_config(solver_cfg)
            solvers.append(solver)
        return cls(root=solvers)

