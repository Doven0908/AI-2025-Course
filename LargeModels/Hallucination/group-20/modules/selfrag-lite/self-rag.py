"""SELF-RAG（按需检索）交互示例。

该脚本展示如何使用本地模型 `selfrag/selfrag_llama2_7b` 与一个简单的检索器
进行检索增强生成的最小示例。

说明：
- 该文件为演示用途，保留交互入口以便调试和手动检查生成与检索行为。
- 运行前需保证 `vllm` 可用，且 `passage_retrieval.py` 中提供 `Retriever` 实现。
"""

from vllm import LLM, SamplingParams
import IPython

# Load a local model copy (non-blocking demo). Keep same default download dir as the repo.
model = LLM("selfrag/selfrag_llama2_7b", download_dir="/openbayes/home/model_cache", dtype="half")
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100, skip_special_tokens=False)


def format_prompt(input, paragraph=None):
  """Format a simple instruction-response prompt.

  Args:
    input (str): instruction or question text.
    paragraph (str|None): optional retrieved paragraph to include in the prompt.

  Returns:
    str: formatted prompt that the model expects.
  """
  prompt = "### Instruction:\n{0}\n\n### Response:\n".format(input)
  if paragraph is not None:
    prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
  return prompt


from passage_retrieval import Retriever
# Set up a demo retriever (indexing / search parameters are demo-only)
retriever = Retriever({})
retriever.setup_retriever_demo("facebook/contriever-msmarco", "enwiki_2020_intro_only/enwiki_2020_dec_intro_only.jsonl", "enwiki_2020_intro_only/enwiki_dec_2020_contriever_intro/*",  n_docs=5, save_or_load_index=False)

query_3 = "Explain the stick-slip phenomenon."
retrieved_documents = retriever.search_document_demo(query_3, 5)
prompts = [format_prompt(query_3, doc["title"] +"\n"+ doc["text"]) for doc in retrieved_documents]
# Generate predictions for retrieved contexts (demo mode)
preds = model.generate(prompts, sampling_params)
top_doc = retriever.search_document_demo(query_3, 1)[0]
print("参考文档: {0}\n模型预测: {1}".format(top_doc["title"] + "\n" + top_doc["text"], preds[0].outputs[0].text))


