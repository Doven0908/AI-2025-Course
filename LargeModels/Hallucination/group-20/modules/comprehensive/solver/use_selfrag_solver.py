# pyright: reportAttributeAccessIssue=false

from typing import Callable
from pydantic import Field
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams


from llm_hallucination_evaluate.dataset.base import DataItem
from llm_hallucination_evaluate.solver.base import Solver
from llm_hallucination_evaluate.solver.solve_result import SolveResult
from llm_hallucination_evaluate.utils.config import SolverConfig
from llm_hallucination_evaluate.solver.llm_model import get_model_func
from llm_hallucination_evaluate.solver.self_rag_run import run_step_generation_batch, call_model_beam_batch, args, TASK_INST, PROMPT_DICT, load_special_tokens, load_jsonlines, postprocess, fix_spacing


class UseSelfragSolver(Solver):
    """使用self-rag求解器"""

    rag_docs_number: int = Field(..., description="检索文档数量")
    llm_qa_func: None = Field(..., description="用于直接问答的LLM调用函数")

    @classmethod
    def from_(cls, solver_cfg: SolverConfig) -> "UseSelfragSolver":
        """从配置中创建 UseSelfragSolver 实例"""

        assert solver_cfg.rag_docs_number is not None, "如果method为use-self-rag，rag_docs_number不能为空"

        return cls(llm_qa_func=None, extra_args=solver_cfg.extra_args, method=solver_cfg.method, model_name=solver_cfg.model, rag_docs_number=solver_cfg.rag_docs_number)



    def _solve(self, data_item: DataItem) -> SolveResult:

        """对单个数据项进行求解，返回求解结果"""

        for k, v in self.extra_args.items():
            setattr(args, k, v)

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, padding_side="left")

        # 获取特殊token
        ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
            tokenizer, use_grounding=args.use_grounding, use_utility=args.use_utility)

        model = LLM(model=args.model_name,
            download_dir=args.download_dir, dtype=args.dtype, max_logprobs=50000)

        def generate(prompt, ctxs, max_new_tokens):
            processed_prompt = PROMPT_DICT["prompt_no_input"].format_map(
                {"instruction": prompt})
            return call_model_beam_batch(
                processed_prompt, 
                model=model, 
                max_new_tokens=max_new_tokens, 
                ctxs=ctxs, 
                query=prompt,
                rel_tokens=rel_tokens, 
                ret_tokens=ret_tokens, 
                grd_tokens=grd_tokens, 
                ut_tokens=ut_tokens,
                use_seqscore=args.use_seqscore, 
                threshold=args.threshold, 
                beam_width=args.beam_width, 
                max_depth=args.max_depth,
                w_rel=1.0, 
                w_sup=1.0, 
                w_use=0.5, 
                mode=args.mode, 
                ignore_cont=args.ignore_cont
            )


        prompt = data_item.question
        ctxs = [{'title': doc.title, 'text': doc.content} for doc in data_item.contexts][:self.rag_docs_number]
        instructions = TASK_INST[args.task]
        prev_gen = []
        prompt = instructions + "## Input:\n\n" + prompt
        
        final_pred, intermediate = generate(prompt, ctxs, args.max_new_tokens)

        # 处理输出结果
        final_output = ""
        docs = []
        prev_gen = []
        
        if "splitted_sentences" not in intermediate:
            answer = postprocess(final_pred)
        else:
            if len(postprocess(final_pred[0])) == 0:
                intermediate["splitted_sentences"][0], intermediate["ctxs"][0] = \
                    intermediate["splitted_sentences"][1], intermediate["ctxs"][1]
            
            for idx, (sent, doc) in enumerate(zip(intermediate["splitted_sentences"][0], intermediate["ctxs"][0])):
                if len(sent) == 0:
                    continue
                postprocessed_result = postprocess(sent)
                if postprocessed_result in prev_gen:
                    continue
                else:
                    prev_gen.append(postprocessed_result)
                final_output += postprocessed_result[:-1] + " [{}]".format(idx) + ". "
                docs.append(doc)
            
            if len(final_output) == 0:
                answer = fix_spacing(final_output)
            if len(final_output) > 0 and final_output[-1] == " ":
                final_output = final_output[:-1]
            
            answer = fix_spacing(final_output)
            answer = answer.replace(".[Continue to Use Evidence]", " [1]. ")
            answer = answer.replace(". [1] ", " [1]. ")
        


        # 构造SolveResult对象
        result = SolveResult(
            dataitem=data_item,
            answer=answer,
            metadata={
                'intermediate': intermediate["original_splitted_sentences"][0] if "original_splitted_sentences" in intermediate else None,
                'retrieved_contexts': [doc['text'] for doc in docs]
            }
        )

        return result
