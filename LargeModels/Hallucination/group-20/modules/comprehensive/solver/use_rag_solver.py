from typing import Callable
from pydantic import Field

from llm_hallucination_evaluate.dataset.base import DataItem
from llm_hallucination_evaluate.solver.base import Solver
from llm_hallucination_evaluate.solver.solve_result import SolveResult
from llm_hallucination_evaluate.utils.config import SolverConfig
from llm_hallucination_evaluate.solver.llm_model import get_model_func

class UseRagSolver(Solver):
    """使用rag求解器"""

    rag_docs_number: int = Field(..., description="检索文档数量")

    @classmethod
    def from_(cls, solver_cfg: SolverConfig) -> "UseRagSolver":
        """从配置中创建 UseRagSolver 实例"""

        model_func = get_model_func(solver_cfg.type, solver_cfg.model)
        assert solver_cfg.rag_docs_number is not None, "如果method为use-rag，rag_docs_number不能为空"

        return cls(llm_qa_func=model_func, extra_args=solver_cfg.extra_args, rag_docs_number=solver_cfg.rag_docs_number, method=solver_cfg.method, model_name=solver_cfg.model)



    def _solve(self, data_item: DataItem) -> SolveResult:

        """对单个数据项进行求解，返回求解结果"""

        rag_prompt = """Please answer the question using the following as a reference.
{}

question: {}
""".format('\n'.join(f'- {i+1} \n title: {context.title} \n content: {context.content}\n' for i, context in enumerate(data_item.contexts[:self.rag_docs_number])), data_item.question)

        # 调用LLM问答函数获取答案
        answer = self.llm_qa_func(
            rag_prompt,
            **self.extra_args
        )

        # 构造SolveResult对象
        result = SolveResult(
            dataitem=data_item,
            answer=answer,
            metadata={'retrieved_contexts': [context.content for context in data_item.contexts[:self.rag_docs_number]]}
        )

        return result
