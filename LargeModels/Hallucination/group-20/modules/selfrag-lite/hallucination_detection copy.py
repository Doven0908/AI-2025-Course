import numpy as np
import spacy
import torch
from transformers import pipeline

from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

pipe = pipeline("text-generation", device_map="auto")

prompt = """
Give me the professional journey of Ashish Vaswani in detail.
Answer:
"""

# As per the original paper the response is generated with greedy decoding
Response = pipe(prompt, do_sample=False, max_new_tokens=128, return_full_text=False)
Response


# The samples are generated for the same prompt with temperature as 1.
N = 20
Samples = pipe(
    [prompt] * N,
    temperature=1.0,
    do_sample=True,
    max_new_tokens=128,
    return_full_text=False,
)
print(Samples[0])

Response = Response[0]["generated_text"]
Samples = [sample[0]["generated_text"] for sample in Samples]

# Mistral 7B became a gated repository so huggingface login is required to access it.
from huggingface_hub import login

login(token=HUGGINGFACE_TOKEN)

# We use Mistral 7B LLM to detect whether the response generated with Phi-2 LM is hallucinated or not using LLM Promting technique.
llm_model = "mistralai/Mistral-7B-Instruct-v0.2"
selfcheck_prompt = SelfCheckLLMPrompt(llm_model, device)

# Option2: API access (currently only support client_type="openai")
# from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt
# selfcheck_prompt = SelfCheckAPIPrompt(client_type="openai", model="gpt-3.5-turbo")


nlp = spacy.load("en_core_web_sm")
sentences = [
    sent.text.strip() for sent in nlp(Response).sents
]  # spacy sentence tokenization
print(sentences)

sent_scores_prompt = selfcheck_prompt.predict(
    sentences=sentences,  # list of sentences
    sampled_passages=Samples,  # list of sampled passages
    verbose=True,  # whether to show a progress bar
)

print(sent_scores_prompt)
print("Hallucination Score:", np.mean(sent_scores_prompt))

import IPython
IPython.embed()
