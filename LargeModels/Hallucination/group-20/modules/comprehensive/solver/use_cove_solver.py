from typing import Callable
from pydantic import Field

from llm_hallucination_evaluate.dataset.base import DataItem
from llm_hallucination_evaluate.solver.base import Solver
from llm_hallucination_evaluate.solver.solve_result import SolveResult
from llm_hallucination_evaluate.utils.config import SolverConfig
from llm_hallucination_evaluate.solver.llm_model import get_model_func

class UseCoveSolver(Solver):
    """使用cove求解器
    
    在rag上完成cove，即 cove + rag 

    https://arxiv.org/abs/2309.11495
    """

    rag_docs_number: int = Field(..., description="检索文档数量")

    @classmethod
    def from_(cls, solver_cfg: SolverConfig) -> "UseCoveSolver":
        """从配置中创建 UseCoveSolver 实例"""

        model_func = get_model_func(solver_cfg.type, solver_cfg.model)
        assert solver_cfg.rag_docs_number is not None, "如果method为use-cove，rag_docs_number不能为空"

        return cls(llm_qa_func=model_func, extra_args=solver_cfg.extra_args, rag_docs_number=solver_cfg.rag_docs_number, method=solver_cfg.method, model_name=solver_cfg.model)



    def _solve(self, data_item: DataItem) -> SolveResult:

        """对单个数据项进行求解，返回求解结果"""

        ctxs_text = '\n'.join(f'- {i+1} \n title: {context.title} \n content: {context.content}\n' for i, context in enumerate(data_item.contexts[:self.rag_docs_number]))

        rag_prompt = """Please answer the question using the following as a reference.
{}

question: {}
""".format(ctxs_text, data_item.question)

        # 调用LLM问答函数获取答案
        print("调用LLM进行初步回答...")
        pre_answer = self.llm_qa_func(
            rag_prompt,
            **self.extra_args
        )

        question_generate_prompt = """Please raise some questions regarding the following text, as I need to use these questions to verify specific information within the text.
Text: {}
""".format(pre_answer)
        print("调用LLM生成验证问题...")
        verification_questions = self.llm_qa_func(
            question_generate_prompt,
            **self.extra_args
        )

        verification_questions_answer_prompt = """Please answer the following questions using the following as a reference.
Context:
{}
Questions:
{}
""".format(ctxs_text, verification_questions)
        
        print("调用LLM回答验证问题...")
        verification_answers = self.llm_qa_func(
            verification_questions_answer_prompt,
            **self.extra_args
        )

        final_answer_prompt = """Based on the following context, initial answer, verification questions, and their answers, please provide a final answer to the original question.
Context:
{}
Initial Answer:
{}
Verification Questions:
{}
Verification Answers:
{}
Original Question:
{}
""".format(
            ctxs_text,
            pre_answer,
            verification_questions,
            verification_answers,
            data_item.question
        )

        print("调用LLM生成最终答案...")
        final_answer = self.llm_qa_func(
            final_answer_prompt,
            **self.extra_args
        )


        # 构造SolveResult对象
        result = SolveResult(
            dataitem=data_item,
            answer=final_answer,
            metadata={
                'retrieved_contexts': [context.content for context in data_item.contexts[:self.rag_docs_number]],
                'pre_answer': pre_answer,
                'verification_questions': verification_questions,
                'verification_answers': verification_answers,
            }
        )

        return result
