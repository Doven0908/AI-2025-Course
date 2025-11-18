import os
from pydantic import BaseModel, Field

from llm_hallucination_evaluate.utils.config import config
from llm_hallucination_evaluate.dataset.base import MutiDataSet
from llm_hallucination_evaluate.solver.base import MultiSolver
from llm_hallucination_evaluate.evaluator.base import Evaluator
from llm_hallucination_evaluate.utils.format import safe_name
from llm_hallucination_evaluate.utils.translate import translate_en_to_zh

def main():

    # 进行评估实验主函数

    # Step 1: 加载数据集、求解器和评估器
    datasets = MutiDataSet.load_from_config(config)
    solvers = MultiSolver.load_from_config(config)
    evaluator = Evaluator.load_from_config(config)


    # Step 2: 主循环，进行生成和评估
    for dataset_index, dataset in enumerate(datasets):

        print('\n')
        print(f"正在使用数据集 {dataset.name} [{dataset_index + 1} / {len(datasets)}] 对模型进行评估...")

        print(f"数据集包含 {len(dataset.items)} 条数据。")
        for item_index, item in enumerate(dataset.items) :

            print('\n')
            print('=' * 50)
            print(f"  正在处理第 {item_index + 1} 条数据 [{item_index + 1} / {len(dataset.items)}]...")

            print(f"    问题: {item.question}")
            print(f"    【翻译】: {translate_en_to_zh(item.question)}")
            if item.answer: 
                print(f"    参考答案: {item.answer}")
                print(f"    【翻译】: {translate_en_to_zh(item.answer)}")


            for solver in solvers:

                eval_res_save_file = f'{config.extra["output_dir"]}/eval_results/{dataset.name}/{safe_name(item.question)}/{safe_name(solver.method)}_with_{safe_name(solver.model_name)}.yaml'

                if os.path.exists(eval_res_save_file):
                    print(f"    评估结果文件已存在，跳过该模型的评估: {eval_res_save_file}")
                    continue
                os.makedirs(os.path.dirname(eval_res_save_file), exist_ok=True)

                # 2.1 运行求解器生成答案
                print(f"    使用 {solver.method} [{solver.model_name}] 回答问题...")
                result = solver.solve(item)

                print(f"    模型答案: {result.answer}")
                print(f"    【翻译】: {translate_en_to_zh(result.answer)}")
            
                # 2.2 使用评估器对答案进行评估

                print(f"    使用 {evaluator.method} 评估答案...")
                eval_results = evaluator.single_evaluate(result)

                print(f"    评估结果: {eval_results.results}")

                eval_results.save(eval_res_save_file)
                print(f"    评估结果已保存到: {eval_res_save_file}")



if __name__ == "__main__":
    main()



"""
利用 comprehensive 框架进行 LLM 幻觉评估实验的主程序。


使用说明：
1、修改配置文件 config.yaml ，指定数据集、求解器和评估器等参数。
2、确保环境变量中配置了所需的 API Key 和 base URL（如使用在线模型）。
3、运行 `python main.py` 开始评估实验。

"""

