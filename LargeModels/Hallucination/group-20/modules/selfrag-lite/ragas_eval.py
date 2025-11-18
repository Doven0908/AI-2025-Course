

import json
from langchain_openai import ChatOpenAI
from ragas.embeddings import OpenAIEmbeddings
import openai
import os
import time
import numpy as np


api_key = os.environ["OPENAI_API_KEY"]
api_base = os.environ.get("OPENAI_API_BASE")
llm = ChatOpenAI(model="gpt-4o", api_key=api_key, base_url=api_base)
openai_client = openai.OpenAI(api_key=api_key, base_url=api_base)
embeddings = OpenAIEmbeddings(client=openai_client)

EXP_FILE_NAME = './experiment_file.json'

data = json.load(open(EXP_FILE_NAME))

for mode in ['openai_answer', 'rag_answer', 'self_rag_answer']:

    print(f'正在对 {mode} 回答进行 RAGAS 评估...')
    dataset = []

    for item in data['data']:
        
        dataset.append(
            {
                "user_input":item['question'],
                "retrieved_contexts":[e['text'] for e in item['docs'][:6]],
                "response":item[mode],
                "reference":item['answer']
            }
        )

    from ragas import EvaluationDataset
    evaluation_dataset = EvaluationDataset.from_list(dataset)

    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper


    evaluator_llm = LangchainLLMWrapper(llm)
    from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness


    import time

    start_eval = time.time()
    result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=evaluator_llm)
    end_eval = time.time()
    elapsed = end_eval - start_eval
    print(f"RAGAS 评估完成，评估结果：{result}\n耗时 {elapsed:.2f} 秒")

    for i, item in enumerate(data['data']):
        item[f'{mode}_ragas_evaluation'] = result.scores[i]

json.dump(data, open(EXP_FILE_NAME, 'w'), indent=4)
print()
print()
