import spacy
from typing import Callable
from pydantic import Field

from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt
from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt

from llm_hallucination_evaluate.dataset.base import DataItem
from llm_hallucination_evaluate.solver.base import Solver
from llm_hallucination_evaluate.solver.solve_result import SolveResult
from llm_hallucination_evaluate.utils.config import SolverConfig, TypeEnum
from llm_hallucination_evaluate.solver.llm_model import get_model_func


class UseSelfchekgptSolver(Solver):
    """使用selfcheckgpt求解器"""


    @classmethod
    def from_(cls, solver_cfg: SolverConfig) -> "UseSelfchekgptSolver":
        """从配置中创建 UseSelfchekgptSolver 实例"""

        model_func = get_model_func(TypeEnum.ONLINE, solver_cfg.model)

        return cls(llm_qa_func=model_func, extra_args=solver_cfg.extra_args, method=solver_cfg.method, model_name=solver_cfg.model)



    def _solve(self, data_item: DataItem) -> SolveResult:

        """对单个数据项进行求解，返回求解结果"""

        # 调用LLM问答函数获取答案

        print("正在使用LLM进行问答...")
        pre_answer = self.llm_qa_func(
            data_item.question,
            **self.extra_args
        )

        answer_samples = []

        for i in range(3):  # 采样多次以获得更多参考内容
            print(f"正在进行第 {i + 1} 次采样...")
            sample_answer = self.llm_qa_func(
                data_item.question,
                **self.extra_args
            )
            answer_samples.append(sample_answer)

        print("采样完成，开始进行SelfCheckGPT校验...")
        # 使用SelfCheckGPT进行校验
        selfcheck_prompt = SelfCheckAPIPrompt(client_type="openai", model=self.model_name)
        nlp = spacy.load("en_core_web_sm")
        sentences = [
            sent.text.strip() for sent in nlp(pre_answer).sents
        ]  # spacy sentence tokenization
        print("句子切分完成，开始评分...")
        sent_scores = selfcheck_prompt.predict(
            sentences=sentences,  # list of sentences
            sampled_passages=answer_samples,  # list of sampled passages
            verbose=True,  # whether to show a progress bar
        )
        print("评分完成。")
        print("句子得分:", sent_scores)

        verification_prompt = f"""I have a question, an answer, and a hallucination score for each sentence in that answer. A higher score indicates a greater degree of hallucination, meaning lower accuracy. I need you to verify the answer based on these scores and provide a revised response to the question.
Question: {data_item.question}
Answer: {pre_answer}

Hallucination Scores: 
{{}}

Please answer this question again.You should directly output the new answer without including any other content.
{data_item.question}
""".format('\n'.join([f'Sentence: {sentences[i]} | Score: {sent_scores[i]:.4f}' for i in range(len(sentences))]))

        print("正在根据评分结果进行最终回答生成...")
        final_answer = self.llm_qa_func(
            verification_prompt,
            **self.extra_args
        )


        # 构造SolveResult对象
        result = SolveResult(
            dataitem=data_item,
            answer=final_answer,
            metadata={
                'sentence_hallucination_scores': {sentences[i]: float(sent_scores[i]) for i in range(len(sentences))}
            }
        )

        return result
