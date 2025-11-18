import json
import numpy as np
import spacy
import torch

from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Mistral 7B became a gated repository so huggingface login is required to access it.
from huggingface_hub import login

login(token=HUGGINGFACE_TOKEN)

# We use Mistral 7B LLM to detect whether the response generated with Phi-2 LM is hallucinated or not using LLM Promting technique.
llm_model = "mistralai/Mistral-7B-Instruct-v0.2"
selfcheck_prompt = SelfCheckLLMPrompt(llm_model, device)

# Option2: API access (currently only support client_type="openai")
# from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt
# selfcheck_prompt = SelfCheckAPIPrompt(client_type="openai", model="gpt-3.5-turbo")




EXP_FILE_NAME = './experiment_file.json'

data = json.load(open(EXP_FILE_NAME))

for i, item in enumerate(data['data']):

    print(f'({i}/{len(data["data"])})问题: {item["question"]}\n')

    for mode in ['openai_answer', 'rag_answer', 'self_rag_answer']:

        print(f'正在对 {mode} 回答进行 RAGAS 评估...')

        
        nlp = spacy.load("en_core_web_sm")
        sentences = [
            sent.text.strip() for sent in nlp(item[mode]).sents
        ]  # spacy sentence tokenization
        sent_scores_prompt = selfcheck_prompt.predict(
            sentences=sentences,  # list of sentences
            sampled_passages=[item['openai_answer'], item['rag_answer'], item['self_rag_answer']],  # list of sampled passages
            verbose=True,  # whether to show a progress bar
        )

        # print(sent_scores_prompt)
        # print("Hallucination Score:", np.mean(sent_scores_prompt))
        print('回答包含 {} 条语句，幻觉得分为：{}\n平均幻觉得分为：{:.4f}'.format(len(sentences), sent_scores_prompt, np.mean(sent_scores_prompt)))

        item[f'{mode}_selfcheckgpt_sentence_hallucination_scores'] = list(sent_scores_prompt)

json.dump(data, open(EXP_FILE_NAME, 'w'), indent=4)
print()
print()
