import glob
from pydantic import BaseModel, Field
from tqdm import tqdm

from llm_hallucination_evaluate.evaluator.eval_result import EvalResult

class AnalyzeItem(BaseModel):
    metrics: dict[str, float | None] = Field(..., description="评估指标字典")
    dataset: str = Field(..., description="数据集名称")
    evaluator: str = Field(..., description="评估器名称")
    solve_method: str = Field(..., description="求解方法名称")
    solve_time: float = Field(..., description="求解时间（秒）")
    model_name: str = Field(..., description="使用的模型名称")

    @classmethod
    def from_eval_result(cls, eval_result: EvalResult) -> "AnalyzeItem":
        """
        从EvalResult对象创建AnalyzeItem实例
        Args:
            eval_result: 评估结果对象

        Returns:
            AnalyzeItem: 创建的分析项对象
        """
        return cls(
            metrics=eval_result.results,
            dataset=eval_result.solve_result.metadata["dataset"],
            evaluator=eval_result.evaluator_name,
            solve_method=eval_result.solve_result.metadata["solve_method"],
            solve_time=eval_result.solve_result.metadata["solve_time"],
            model_name=eval_result.solve_result.metadata["model_name"],
        )

class EvalResultAnalyzer(BaseModel):

    items: list[AnalyzeItem] = Field(default_factory=list)

    @classmethod
    def load_results(cls, results_dir: str) -> "EvalResultAnalyzer":
        """
        从指定目录加载所有评估结果文件
        
        Args:
            results_dir: 评估结果文件所在目录
            
        Returns:
            EvalResultAnalyzer: 包含所有评估结果的分析器对象
        """

        result_files = glob.glob(f"{results_dir}/*/*/*.yaml")
        items = []
        for file_path in tqdm(result_files, desc="加载评估结果文件..."):
            result = EvalResult.read_from_yaml(file_path)
            items.append(AnalyzeItem.from_eval_result(result))
        return cls(items=items)


if __name__ == "__main__":

    analyzer = EvalResultAnalyzer.load_results("./outputs/eval_results")

    import IPython; IPython.embed()


"""
这是一个用来展示、分析统计数据的文件，完成如下功能：

1、多维度评估
2、可视化展示
3、终端命令行简单交互系统

"""
